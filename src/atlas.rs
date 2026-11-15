//! `--atlas FILE`: the static arrows of all neurons, built tile by tile.
//!
//! The arrows form one table C[later, earlier] = (contribution to p, contribution to q) per unit of
//! source activation. Read by rows it gives every neuron's inputs, read by columns every neuron's
//! outputs. The table is never materialized: for each target layer the rows (read vectors g', u' of all
//! its neurons) are multiplied with blocks of sources (token embeddings, W_down of every earlier layer,
//! PLE projections, biases) as GEMM, and every tile is reduced at once to the top k per row and per
//! column. Each neuron's token voice is computed in the same pass over the vocabulary.
//!
//! File format "APLA" (little endian), memory-mappable, built per layer and resumable:
//!   header (4096 bytes): magic, version, k, sizes, model description, per layer (ff, done, first record)
//!   one fixed-size record per neuron: repertoire + six lists of k entries
//!   (inputs by |p|, inputs by |q|, outputs by |p|, outputs by |q|, voice up, voice down).
use crate::blas;
use crate::fmt::{num, sci};
use crate::model::Model;
use crate::neuron::{adjoint, forward, path, pre_norm, Op, Read, ReadPoint, Repertoire, Src};
use crate::tensor::Mat;
use rayon::prelude::*;
use std::io::IsTerminal;
use std::path::Path;
use std::time::Instant;

const MAGIC: &[u8; 4] = b"APLA";
const VERSION: u32 = 1;
const HEADER: usize = 4096;
const LAYER_TABLE: usize = 512;
const ENTRY: usize = 16;
const REC_HEAD: usize = 80;
const N_LISTS: usize = 6;
const EMPTY: u8 = 0xFF;

/// One list entry: source (or target) and its arrow (p, q); for voice lists p is the logit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Entry {
    pub kind: u8,
    pub layer: u16,
    pub index: u32,
    pub p: f32,
    pub q: f32,
}

pub const K_TOKEN: u8 = 0;
pub const K_POSITION: u8 = 1;
pub const K_TYPE: u8 = 2;
pub const K_EMBNORM: u8 = 3;
pub const K_ATTNBIAS: u8 = 4;
pub const K_ATTNPOSTBIAS: u8 = 5;
pub const K_NEURON: u8 = 6;
pub const K_MLPBIAS: u8 = 7;
pub const K_MLPPOSTBIAS: u8 = 8;
pub const K_PLE: u8 = 9;
pub const K_VOICE: u8 = 10;

impl Entry {
    fn src(s: Src, p: f32, q: f32) -> Entry {
        let (kind, layer, index) = match s {
            Src::Token(t) => (K_TOKEN, 0, t),
            Src::Position(r) => (K_POSITION, 0, r),
            Src::TypeEmbed => (K_TYPE, 0, 0),
            Src::EmbedNormBias => (K_EMBNORM, 0, 0),
            Src::AttnBias(k) => (K_ATTNBIAS, k, 0),
            Src::AttnPostBias(k) => (K_ATTNPOSTBIAS, k, 0),
            Src::Neuron(k, i) => (K_NEURON, k, i),
            Src::MlpBias(k) => (K_MLPBIAS, k, 0),
            Src::MlpPostBias(k) => (K_MLPPOSTBIAS, k, 0),
            Src::PleDim(k, e) => (K_PLE, k, e),
            _ => (EMPTY, 0, 0),
        };
        Entry { kind, layer, index, p, q }
    }
    pub fn to_src(&self) -> Option<Src> {
        Some(match self.kind {
            K_TOKEN => Src::Token(self.index),
            K_POSITION => Src::Position(self.index),
            K_TYPE => Src::TypeEmbed,
            K_EMBNORM => Src::EmbedNormBias,
            K_ATTNBIAS => Src::AttnBias(self.layer),
            K_ATTNPOSTBIAS => Src::AttnPostBias(self.layer),
            K_NEURON => Src::Neuron(self.layer, self.index),
            K_MLPBIAS => Src::MlpBias(self.layer),
            K_MLPPOSTBIAS => Src::MlpPostBias(self.layer),
            K_PLE => Src::PleDim(self.layer, self.index),
            _ => return None,
        })
    }
    fn same(&self, o: &Entry) -> bool {
        self.kind == o.kind && self.layer == o.layer && self.index == o.index
    }
    fn write(&self, b: &mut [u8]) {
        b[0] = self.kind;
        b[1] = 0;
        b[2..4].copy_from_slice(&self.layer.to_le_bytes());
        b[4..8].copy_from_slice(&self.index.to_le_bytes());
        b[8..12].copy_from_slice(&self.p.to_le_bytes());
        b[12..16].copy_from_slice(&self.q.to_le_bytes());
    }
    fn read(b: &[u8]) -> Entry {
        Entry {
            kind: b[0],
            layer: u16::from_le_bytes([b[2], b[3]]),
            index: u32::from_le_bytes([b[4], b[5], b[6], b[7]]),
            p: f32::from_le_bytes([b[8], b[9], b[10], b[11]]),
            q: f32::from_le_bytes([b[12], b[13], b[14], b[15]]),
        }
    }
}

/// Fixed-capacity list sorted by key (descending), without duplicates.
#[derive(Clone)]
struct TopK {
    k: usize,
    key: Vec<f32>,
    ent: Vec<Entry>,
}

impl TopK {
    fn new(k: usize) -> Self {
        TopK { k, key: Vec::with_capacity(k), ent: Vec::with_capacity(k) }
    }
    #[inline]
    fn threshold(&self) -> f32 {
        if self.key.len() < self.k { f32::NEG_INFINITY } else { self.key[self.k - 1] }
    }
    fn insert(&mut self, key: f32, e: Entry) {
        if !(key > self.threshold()) {
            return;
        }
        if let Some(pos) = self.ent.iter().position(|x| x.same(&e)) {
            // the same source again (resumed build): keep one copy
            self.key.remove(pos);
            self.ent.remove(pos);
        }
        let at = self.key.partition_point(|x| *x >= key);
        self.key.insert(at, key);
        self.ent.insert(at, e);
        if self.key.len() > self.k {
            self.key.pop();
            self.ent.pop();
        }
    }
}

fn put_str(b: &mut [u8], s: &str) {
    let x = s.as_bytes();
    let n = x.len().min(b.len());
    b[..n].copy_from_slice(&x[..n]);
}

fn get_str(b: &[u8]) -> String {
    let n = b.iter().position(|c| *c == 0).unwrap_or(b.len());
    String::from_utf8_lossy(&b[..n]).into_owned()
}

fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

fn u64_at(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(b[o..o + 8].try_into().unwrap())
}

/// Read access to an atlas file.
pub struct Atlas {
    map: memmap2::Mmap,
    pub k: usize,
    pub n_layer: usize,
    pub rec: usize,
    ff: Vec<usize>,
    done: Vec<bool>,
    first: Vec<u64>,
    pub model: String,
}

/// One neuron's record.
pub struct Record {
    pub inputs_complete: bool,
    pub outputs_complete: bool,
    pub c: [f32; 2],
    pub gram: [f32; 3],
    pub a_max: f32,
    pub a_min: f32,
    pub lists: [Vec<Entry>; N_LISTS],
}

impl Atlas {
    pub fn open(p: &Path) -> Result<Atlas, String> {
        let f = std::fs::File::open(p).map_err(|e| format!("{}: {e} (build it with --atlas {} and no prompt)", p.display(), p.display()))?;
        let map = unsafe { memmap2::Mmap::map(&f) }.map_err(|e| format!("{}: {e}", p.display()))?;
        if map.len() < HEADER || &map[..4] != MAGIC {
            return Err(format!("{}: not an apollon atlas", p.display()));
        }
        if u32_at(&map, 4) != VERSION {
            return Err(format!("{}: atlas version {} (expected {VERSION})", p.display(), u32_at(&map, 4)));
        }
        let k = u32_at(&map, 12) as usize;
        let n_layer = u32_at(&map, 16) as usize;
        let rec = u32_at(&map, 32) as usize;
        let mut ff = Vec::new();
        let mut done = Vec::new();
        let mut first = Vec::new();
        for l in 0..n_layer {
            let o = LAYER_TABLE + 16 * l;
            ff.push(u32_at(&map, o) as usize);
            done.push(map[o + 4] == 1);
            first.push(u64_at(&map, o + 8));
        }
        let model = get_str(&map[80..208]);
        Ok(Atlas { map, k, n_layer, rec, ff, done, first, model })
    }

    pub fn layers_done(&self) -> Vec<bool> {
        self.done.clone()
    }

    pub fn record(&self, l: usize, i: usize) -> Option<Record> {
        if l >= self.n_layer || i >= self.ff[l] {
            return None;
        }
        let o = HEADER + (self.first[l] as usize + i) * self.rec;
        let b = &self.map[o..o + self.rec];
        if u32_at(b, 8) != 1 && !self.done[l] {
            // outputs may still be partly there; inputs are only valid when the layer is done
        }
        let f = |k: usize| f32::from_le_bytes([b[16 + 4 * k], b[17 + 4 * k], b[18 + 4 * k], b[19 + 4 * k]]);
        let list = |n: usize| -> Vec<Entry> {
            (0..self.k).map(|j| Entry::read(&b[REC_HEAD + (n * self.k + j) * ENTRY..][..ENTRY])).filter(|e| e.kind != EMPTY).collect()
        };
        Some(Record {
            inputs_complete: self.done[l],
            outputs_complete: self.done[l + 1..].iter().all(|x| *x),
            c: [f(0), f(1)],
            gram: [f(2), f(3), f(4)],
            a_max: f(5),
            a_min: f(6),
            lists: [list(0), list(1), list(2), list(3), list(4), list(5)],
        })
    }
}

// ====================================================================== build

pub struct BuildOpts {
    pub k: usize,
    pub layers: Option<(usize, usize)>,
}

/// read vectors of all neurons of layer l as rows [ff·(1 or 2)][d] (g' and u' interleaved)
fn read_rows(m: &Model, l: usize) -> (Vec<f32>, usize, bool) {
    let b = &m.blocks[l];
    let d = m.d;
    let ff = b.mlp.up.rows;
    let gated = b.mlp.gate.is_some();
    let nr = if gated { 2 } else { 1 };
    let norm = pre_norm(b);
    let ln = norm.is_some_and(|n| !n.rms);
    let mut r = vec![0f32; ff * nr * d];
    r.par_chunks_mut(nr * d).enumerate().for_each(|(i, out)| {
        let (gm, um) = match &b.mlp.gate {
            Some(g) => (g, Some(&b.mlp.up)),
            None => (&b.mlp.up, None),
        };
        gm.row(i, &mut out[..d]);
        if let Some(u) = um {
            u.row(i, &mut out[d..2 * d]);
        }
        for c in out.chunks_mut(d) {
            if let Some(n) = norm {
                for (x, g) in c.iter_mut().zip(&n.w) {
                    *x *= g;
                }
            }
            if ln {
                let mean = c.iter().map(|x| *x as f64).sum::<f64>() / d as f64;
                c.iter_mut().for_each(|x| *x = (*x as f64 - mean) as f32);
            }
        }
    });
    (r, nr, gated)
}

/// adjoint of a path applied to every row
fn rows_back(r: &[f32], d: usize, ops: &[Op]) -> Vec<f32> {
    let mut out = r.to_vec();
    out.par_chunks_mut(d).for_each(|row| adjoint(ops, row));
    out
}

/// Merge a tile C [nr·ff][nb] (rows: g', u' interleaved per target neuron) into the row lists
/// (inputs of the targets); sources j map to `src(j)`.
fn merge_rows(c: &[f32], nb: usize, nr: usize, ins: &mut [(TopK, TopK)], src: &(dyn Fn(usize) -> Src + Sync)) {
    ins.par_iter_mut().enumerate().for_each(|(i, (tp, tq))| {
        let gp = &c[(i * nr) * nb..(i * nr + 1) * nb];
        let gq = if nr == 2 { &c[(i * nr + 1) * nb..(i * nr + 2) * nb] } else { gp };
        for j in 0..nb {
            let (p, q) = (gp[j], if nr == 2 { gq[j] } else { 0.0 });
            if p.abs() > tp.threshold() {
                tp.insert(p.abs(), Entry::src(src(j), p, q));
            }
            if nr == 2 && q.abs() > tq.threshold() {
                tq.insert(q.abs(), Entry::src(src(j), p, q));
            }
        }
    });
}

fn write_list(b: &mut [u8], n: usize, k: usize, t: &TopK) {
    for j in 0..k {
        let e = t.ent.get(j).copied().unwrap_or(Entry { kind: EMPTY, layer: 0, index: 0, p: 0.0, q: 0.0 });
        e.write(&mut b[REC_HEAD + (n * k + j) * ENTRY..][..ENTRY]);
    }
}

fn read_list(b: &[u8], n: usize, k: usize) -> TopK {
    let mut t = TopK::new(k);
    for j in 0..k {
        let e = Entry::read(&b[REC_HEAD + (n * k + j) * ENTRY..][..ENTRY]);
        if e.kind != EMPTY {
            let key = if n == 3 { e.q.abs() } else { e.p.abs() };
            t.key.push(key);
            t.ent.push(e);
        }
    }
    t
}

/// dense copy of a whole matrix [rows][cols]
fn dense(w: &Mat) -> Vec<f32> {
    let mut v = vec![0f32; w.rows * w.cols];
    v.par_chunks_mut(w.cols).enumerate().for_each(|(r, o)| w.row(r, o));
    v
}

pub fn build(m: &Model, file: &Path, o: &BuildOpts) -> Result<(), String> {
    let t0 = Instant::now();
    let nl = m.n_layer();
    let d = m.d;
    let k = o.k.max(1);
    if nl > (HEADER - LAYER_TABLE) / 16 {
        return Err(format!("atlas: {nl} layers do not fit the header"));
    }
    let ffs: Vec<usize> = m.blocks.iter().map(|b| b.mlp.up.rows).collect();
    let rec = REC_HEAD + N_LISTS * k * ENTRY;
    let n_rec: usize = ffs.iter().sum();
    let size = HEADER + n_rec * rec;
    let exists = file.exists();
    let f = std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(false).open(file).map_err(|e| format!("{}: {e}", file.display()))?;
    if exists {
        let len = f.metadata().map_err(|e| e.to_string())?.len() as usize;
        if len != size {
            return Err(format!("{}: existing atlas has another size ({len} bytes, expected {size} for k = {k}); delete it or use the same --atlas-k", file.display()));
        }
    } else {
        f.set_len(size as u64).map_err(|e| format!("{}: {e}", file.display()))?;
    }
    let mut map = unsafe { memmap2::MmapOptions::new().map_mut(&f) }.map_err(|e| format!("{}: {e}", file.display()))?;
    let mut first = vec![0u64; nl];
    for l in 1..nl {
        first[l] = first[l - 1] + ffs[l - 1] as u64;
    }
    if exists {
        if &map[..4] != MAGIC || u32_at(&map, 4) != VERSION || u32_at(&map, 12) as usize != k || u32_at(&map, 16) as usize != nl || u32_at(&map, 20) as usize != d || get_str(&map[80..208]) != m.name {
            return Err(format!("{}: atlas belongs to another model or k; delete it to rebuild", file.display()));
        }
        eprintln!("atlas: resuming {} ({} of {nl} layers done)", file.display(), (0..nl).filter(|l| map[LAYER_TABLE + 16 * l + 4] == 1).count());
    } else {
        map[..4].copy_from_slice(MAGIC);
        map[4..8].copy_from_slice(&VERSION.to_le_bytes());
        map[8..12].copy_from_slice(&(HEADER as u32).to_le_bytes());
        map[12..16].copy_from_slice(&(k as u32).to_le_bytes());
        map[16..20].copy_from_slice(&(nl as u32).to_le_bytes());
        map[20..24].copy_from_slice(&(d as u32).to_le_bytes());
        map[24..28].copy_from_slice(&(m.embed.rows as u32).to_le_bytes());
        map[28] = m.blocks[0].mlp.gate.is_some() as u8;
        map[32..36].copy_from_slice(&(rec as u32).to_le_bytes());
        map[36..40].copy_from_slice(&(ENTRY as u32).to_le_bytes());
        map[40..48].copy_from_slice(&(n_rec as u64).to_le_bytes());
        put_str(&mut map[48..80], &m.arch);
        put_str(&mut map[80..208], &m.name);
        put_str(&mut map[208..336], &m.weight_types);
        for l in 0..nl {
            let ofs = LAYER_TABLE + 16 * l;
            map[ofs..ofs + 4].copy_from_slice(&(ffs[l] as u32).to_le_bytes());
            map[ofs + 4] = 0;
            map[ofs + 8..ofs + 16].copy_from_slice(&first[l].to_le_bytes());
        }
        // empty lists everywhere
        map[HEADER..].par_chunks_mut(rec).for_each(|r| {
            for j in 0..N_LISTS * k {
                r[REC_HEAD + j * ENTRY] = EMPTY;
            }
        });
        map.flush().map_err(|e| e.to_string())?;
    }
    let (lo, hi) = o.layers.unwrap_or((0, nl));
    let hi = hi.min(nl);
    let tty = std::io::stderr().is_terminal();
    let vocab = m.embed.rows;
    let vb = 2048usize;
    for l in lo..hi {
        if map[LAYER_TABLE + 16 * l + 4] == 1 {
            continue;
        }
        let tl = Instant::now();
        let ff = ffs[l];
        let (r, nr, gated) = read_rows(m, l);
        let mut ins: Vec<(TopK, TopK)> = vec![(TopK::new(k), TopK::new(k)); ff];
        let src_tok = |j: usize| Src::Token(j as u32);
        let rp = ReadPoint::MlpIn(l);
        // ---- voices of this layer's neurons: rows o_i through the path to the final norm (unit scale)
        let voice_rows: Option<Vec<f32>> = match (m.has_head, path(m, Src::Neuron(l as u16, 0), ReadPoint::Final, true)) {
            (true, Some(ops)) => {
                let down = dense(&m.blocks[l].mlp.down); // [d][ff]
                let mut o_rows = vec![0f32; ff * d];
                o_rows.par_chunks_mut(d).enumerate().for_each(|(i, row)| {
                    for r in 0..d {
                        row[r] = down[r * ff + i];
                    }
                    forward(&ops, row);
                    if let Some(n) = &m.final_norm {
                        let v = n.map_part(1.0, row);
                        row.copy_from_slice(&v);
                    }
                });
                Some(o_rows)
            }
            _ => None,
        };
        let mut vpos: Vec<TopK> = vec![TopK::new(k); if voice_rows.is_some() { ff } else { 0 }];
        let mut vneg: Vec<TopK> = vpos.clone();
        // ---- tokens (and the output matrix for the voices, which is the same matrix when tied)
        let tok_ops = path(m, Src::Token(0), rp, true);
        let r_tok = tok_ops.as_ref().map(|ops| rows_back(&r, d, ops));
        let lm = m.lm_head();
        let tied = std::ptr::eq(lm, &m.embed);
        let mut wb = vec![0f32; vb * d];
        let mut c = vec![0f32; ff * nr * vb];
        let mut cv = vec![0f32; ff * vb];
        let mut v0 = 0;
        let n_vocab_pass = if voice_rows.is_some() && !tied { 2 } else { 1 };
        for pass in 0..n_vocab_pass {
            let w = if pass == 0 { &m.embed } else { lm };
            let rows = w.rows;
            v0 = 0;
            while v0 < rows {
                let nb = vb.min(rows - v0);
                wb[..nb * d].par_chunks_mut(d).enumerate().for_each(|(i, o)| w.row(v0 + i, o));
                if pass == 0 {
                    if let Some(rt) = &r_tok {
                        blas::gemm_nt(rt, &wb[..nb * d], ff * nr, nb, d, &mut c[..ff * nr * nb]);
                        let base = v0;
                        merge_rows(&c[..ff * nr * nb], nb, nr, &mut ins, &move |j| src_tok(base + j));
                    }
                }
                if let Some(orows) = &voice_rows {
                    if pass == 1 || tied {
                        blas::gemm_nt(orows, &wb[..nb * d], ff, nb, d, &mut cv[..ff * nb]);
                        let base = v0;
                        vpos.par_iter_mut().zip(vneg.par_iter_mut()).enumerate().for_each(|(i, (tp, tn))| {
                            let row = &cv[i * nb..(i + 1) * nb];
                            for (j, &z) in row.iter().enumerate() {
                                if z > tp.threshold() {
                                    tp.insert(z, Entry { kind: K_VOICE, layer: 0, index: (base + j) as u32, p: z, q: 0.0 });
                                }
                                if -z > tn.threshold() {
                                    tn.insert(-z, Entry { kind: K_VOICE, layer: 0, index: (base + j) as u32, p: z, q: 0.0 });
                                }
                            }
                        });
                    }
                }
                v0 += nb;
                if tty {
                    eprint!("\ratlas: S{l} vocabulary {}/{}   ", v0, rows);
                }
            }
        }
        let _ = (vocab, v0);
        // ---- learned positions
        if let (Some((pe, _)), Some(ops)) = (&m.pos_embed, path(m, Src::Position(0), rp, true)) {
            let rb = rows_back(&r, d, &ops);
            let pw = dense(pe);
            let mut cp = vec![0f32; ff * nr * pe.rows];
            blas::gemm_nt(&rb, &pw, ff * nr, pe.rows, d, &mut cp);
            merge_rows(&cp, pe.rows, nr, &mut ins, &|j| Src::Position(j as u32));
        }
        // ---- biases (constant sources)
        let mut biases: Vec<(Src, Vec<f32>)> = Vec::new();
        if let Some(t) = &m.type_embed {
            biases.push((Src::TypeEmbed, t.clone()));
        }
        if let Some(b) = m.embed_norm.as_ref().and_then(|n| n.b.clone()) {
            biases.push((Src::EmbedNormBias, b));
        }
        for (kk, b) in m.blocks.iter().enumerate() {
            let k16 = kk as u16;
            for (s, v) in [
                (Src::AttnBias(k16), b.attn.bo.clone()),
                (Src::AttnPostBias(k16), b.n_attn_post.as_ref().and_then(|n| n.b.clone())),
                (Src::MlpBias(k16), b.mlp.bdown.clone()),
                (Src::MlpPostBias(k16), b.n_mlp_post.as_ref().and_then(|n| n.b.clone())),
            ] {
                if let Some(v) = v {
                    biases.push((s, v));
                }
            }
        }
        for (s, v) in &biases {
            if let Some(ops) = path(m, *s, rp, true) {
                let rb = rows_back(&r, d, &ops);
                let mut cb = vec![0f32; ff * nr];
                blas::gemm_nt(&rb, v, ff * nr, 1, d, &mut cb);
                let s = *s;
                merge_rows(&cb, 1, nr, &mut ins, &move |_| s);
            }
        }
        // ---- every earlier neuron: inputs of this layer (rows) and outputs of the earlier layer (columns)
        for kk in 0..nl {
            let Some(ops) = path(m, Src::Neuron(kk as u16, 0), rp, true) else { continue };
            let rb = rows_back(&r, d, &ops);
            let down = dense(&m.blocks[kk].mlp.down); // [d][ff_k]
            let ffk = ffs[kk];
            let cb = 2048usize.min(ffk);
            let mut blk = vec![0f32; d * cb];
            let mut ct = vec![0f32; ff * nr * cb];
            let rec0 = HEADER + first[kk] as usize * rec;
            let mut j0 = 0;
            while j0 < ffk {
                let nb = cb.min(ffk - j0);
                blk[..d * nb].par_chunks_mut(nb).enumerate().for_each(|(r_, o)| o.copy_from_slice(&down[r_ * ffk + j0..r_ * ffk + j0 + nb]));
                blas::gemm_nn(&rb, &blk[..d * nb], ff * nr, nb, d, &mut ct[..ff * nr * nb]);
                let base = j0;
                let k16 = kk as u16;
                merge_rows(&ct[..ff * nr * nb], nb, nr, &mut ins, &move |j| Src::Neuron(k16, (base + j) as u32));
                // columns: outputs of the earlier neurons j0..j0+nb into this layer
                let ctr = &ct[..ff * nr * nb];
                let l16 = l as u16;
                map[rec0 + j0 * rec..rec0 + (j0 + nb) * rec].par_chunks_mut(rec).enumerate().for_each(|(j, rb_)| {
                    let mut op = read_list(rb_, 2, k);
                    let mut oq = read_list(rb_, 3, k);
                    for i in 0..ff {
                        let p = ctr[(i * nr) * nb + j];
                        let q = if nr == 2 { ctr[(i * nr + 1) * nb + j] } else { 0.0 };
                        let e = Entry { kind: K_NEURON, layer: l16, index: i as u32, p, q };
                        if p.abs() > op.threshold() {
                            op.insert(p.abs(), e);
                        }
                        if nr == 2 && q.abs() > oq.threshold() {
                            oq.insert(q.abs(), e);
                        }
                    }
                    write_list(rb_, 2, k, &op);
                    write_list(rb_, 3, k, &oq);
                });
                j0 += nb;
            }
            if tty {
                eprint!("\ratlas: S{l} sources from S{kk}                ");
            }
        }
        // ---- PLE dimensions of earlier layers
        for kk in 0..nl {
            let (Some(pb), Some(ops)) = (&m.blocks[kk].ple, path(m, Src::PleDim(kk as u16, 0), rp, true)) else { continue };
            let rb = rows_back(&r, d, &ops);
            let pw = dense(&pb.proj); // [d][dp]
            let dp = pb.proj.cols;
            let mut cp = vec![0f32; ff * nr * dp];
            blas::gemm_nn(&rb, &pw, ff * nr, dp, d, &mut cp);
            let k16 = kk as u16;
            merge_rows(&cp, dp, nr, &mut ins, &move |j| Src::PleDim(k16, j as u32));
        }
        // ---- repertoire and write the records of this layer
        let rec_l = HEADER + first[l] as usize * rec;
        let radius = (d as f64).sqrt();
        let act = m.blocks[l].mlp.act;
        let blk = &m.blocks[l];
        let norm = pre_norm(blk);
        map[rec_l..rec_l + ff * rec].par_chunks_mut(rec).enumerate().for_each(|(i, b)| {
            let gq = &r[i * nr * d..(i + 1) * nr * d];
            let (g, u) = (&gq[..d], if nr == 2 { &gq[d..] } else { &gq[..0] });
            let d64 = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum::<f64>();
            let gram = if gated { [d64(g, g), d64(g, u), d64(u, u)] } else { [d64(g, g), 0.0, 0.0] };
            // offsets c = w·β + b
            let (cg, cu) = {
                let beta = norm.and_then(|n| n.b.as_ref());
                let mut wrow = vec![0f32; d];
                let mut off = |w: &Mat, bias: f64| -> f64 {
                    w.row(i, &mut wrow);
                    bias + beta.map_or(0.0, |bb| d64(&wrow, bb))
                };
                match &blk.mlp.gate {
                    Some(gm) => (off(gm, 0.0), off(&blk.mlp.up, blk.mlp.bup.as_ref().map_or(0.0, |x| x[i] as f64))),
                    None => (off(&blk.mlp.up, blk.mlp.bup.as_ref().map_or(0.0, |x| x[i] as f64)), 0.0),
                }
            };
            let rep = Repertoire::new([cg, cu], gram, radius, act, gated);
            b[0..4].copy_from_slice(&(l as u32).to_le_bytes());
            b[4..8].copy_from_slice(&(i as u32).to_le_bytes());
            b[8..12].copy_from_slice(&1u32.to_le_bytes());
            b[12..16].copy_from_slice(&(rep.rank as u32).to_le_bytes());
            let vals = [cg, cu, gram[0], gram[1], gram[2], rep.a_max, rep.a_min, rep.th_max, rep.th_min, rep.pt_max[0], rep.pt_max[1], rep.pt_min[0], rep.pt_min[1]];
            for (n, v) in vals.iter().enumerate() {
                b[16 + 4 * n..20 + 4 * n].copy_from_slice(&(*v as f32).to_le_bytes());
            }
            write_list(b, 0, k, &ins[i].0);
            write_list(b, 1, k, &ins[i].1);
            if let (Some(tp), Some(tn)) = (vpos.get(i), vneg.get(i)) {
                write_list(b, 4, k, tp);
                write_list(b, 5, k, tn);
            }
        });
        // ---- self-check: one neuron against the on-demand computation
        let probe_i = (l * 7919 + 13) % ff;
        let rd = Read::new(m, l, probe_i)?;
        let on_demand = crate::neuron::static_inputs(m, &rd);
        let mut best: Option<(Src, f64)> = None;
        for (_, v) in &on_demand.kinds {
            for a in v {
                if best.map_or(true, |b| a.p.abs() > b.1.abs()) {
                    best = Some((a.src, a.p));
                }
            }
        }
        let stored = ins[probe_i].0.ent.first().copied();
        // same source: compare the values; another source: only a tie is allowed (same magnitude)
        let dev = match (best, stored) {
            (Some((s, p)), Some(e)) if e.to_src() == Some(s) => ((e.p as f64 - p).abs() / p.abs().max(1e-30)) as f32,
            (Some((_, p)), Some(e)) => {
                let t = ((e.p.abs() as f64 - p.abs()).abs() / p.abs().max(1e-30)) as f32;
                if t < 1e-4 { t } else { f32::INFINITY }
            }
            _ => 0.0,
        };
        map[LAYER_TABLE + 16 * l + 4] = 1;
        map.flush().map_err(|e| e.to_string())?;
        if tty {
            eprint!("\r");
        }
        let left = (lo..hi).filter(|x| *x > l && map[LAYER_TABLE + 16 * x + 4] != 1).count();
        let secs = tl.elapsed().as_secs_f32();
        eprintln!(
            "atlas: S{l} done in {} s ({ff} neurons; check S{l} #{probe_i}: strongest input by |p| = on-demand result, dev. {}){}",
            num(secs, 1),
            if dev.is_finite() { sci(dev) } else { "MISMATCH".into() },
            if left > 0 { format!(" · ≈ {} s left", num(secs * left as f32 * 1.05, 0)) } else { String::new() }
        );
        if !dev.is_finite() || dev > 1e-3 {
            return Err(format!("atlas: self-check failed for S{l} #{probe_i} (strongest input differs from the on-demand computation)"));
        }
    }
    let done = (0..nl).filter(|l| map[LAYER_TABLE + 16 * l + 4] == 1).count();
    // self-check of the column side: once all later layers are done, the outputs of a neuron are complete
    if let Some(l0) = (0..nl.saturating_sub(1)).find(|&l| (l + 1..nl).all(|x| map[LAYER_TABLE + 16 * x + 4] == 1) && (l..nl).any(|x| (lo..hi).contains(&x))) {
        let i0 = (l0 * 104729 + 7) % ffs[l0];
        let outs = crate::neuron::static_outputs(m, l0, i0);
        let best = outs.neurons.iter().fold(None::<(Src, f64)>, |b, a| if b.map_or(true, |x| a.p.abs() > x.1.abs()) { Some((a.src, a.p)) } else { b });
        let o = HEADER + (first[l0] as usize + i0) * rec;
        let stored = read_list(&map[o..o + rec], 2, k).ent.first().copied();
        let dev = match (best, stored) {
            (Some((s, p)), Some(e)) if e.to_src() == Some(s) => ((e.p as f64 - p).abs() / p.abs().max(1e-30)) as f32,
            (Some((_, p)), Some(e)) => {
                let t = ((e.p.abs() as f64 - p.abs()).abs() / p.abs().max(1e-30)) as f32;
                if t < 1e-4 { t } else { f32::INFINITY }
            }
            _ => 0.0,
        };
        eprintln!("atlas: check S{l0} #{i0}: strongest output by |p| = on-demand result, dev. {}", if dev.is_finite() { sci(dev) } else { "MISMATCH".into() });
        if !dev.is_finite() || dev > 1e-3 {
            return Err(format!("atlas: self-check of the outputs failed for S{l0} #{i0}"));
        }
    }
    eprintln!("atlas: {} – {done} of {nl} layers done, k = {k}, {} MB, {} s", file.display(), size / (1 << 20), num(t0.elapsed().as_secs_f32(), 0));
    Ok(())
}
