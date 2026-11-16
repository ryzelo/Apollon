//! `--trace`: one MLP neuron at one prediction, decomposed exactly down to single scalar products.
//!
//! Level 0 (scalars): p − b = Σ_j W_gate[i,j]·x̂_j with the weights exactly as the forward pass unpacks
//! them. Level 1 (sources): the stream at the MLP input is the sum of everything written into it, so
//! with the frozen pre-norm scale s every source c contributes p_c = s·g'·x_c, and Σ_c p_c = p
//! (offset included). Pairs: with the frozen φ = act(p)/p, a = φ·Σ_c Σ_c' p_c·q_c' exactly.
//! Nothing is stored in advance: every product is recomputed from the recorded pass.
use crate::fmt::{chip, esc_html, factor, factor_html, num, pad, padl, pct, prompt_html, sci, tok, trunc, Style, CSS};
use crate::model::{top_probs, Checkpoint, Model, Res, RunOpts, Start, Trace, Variants};
use crate::neuron::{act_name, adjoint, col_arrows, dot64, forward_runtime, path, phi, pre_norm, read_back, repertoire, runtime_factor, Read, ReadPoint, Repertoire, Scale, Src};
use crate::readout::{columns, log_odds, softmax, voices};
use serde_json::json;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

/// One source's contribution to p and q of the traced neuron.
#[derive(Clone, Copy, Debug)]
pub struct Contrib {
    pub src: Src,
    pub p: f64,
    pub q: f64,
}

/// A self-check: deviation, tolerance, and what was compared.
pub struct Check {
    pub what: String,
    pub dev: f64,
    pub tol: f64,
}

impl Check {
    pub fn ok(&self) -> bool {
        self.dev.is_finite() && self.dev <= self.tol
    }
}

/// The exact runtime decomposition of one neuron at the focus.
pub struct Runtime {
    pub rd: Read,
    pub rep: Repertoire,
    /// stream at the MLP input, MLP input (pre-norm output), frozen pre-norm scale
    pub x: Vec<f32>,
    pub xhat: Vec<f32>,
    pub s: f64,
    /// p, q, a recomputed from x̂ in f64; a as recorded by the forward pass
    pub p: f64,
    pub q: f64,
    pub a: f64,
    pub a_rec: f64,
    pub phi: f64,
    /// exact partition of p and q: embedding parts, heads, biases, every earlier neuron, PLE dims, offset
    pub parts: Vec<Contrib>,
    /// the heads split by read position (Σ over a head's atoms = the head)
    pub atoms: Vec<Contrib>,
    /// number of earlier neurons with activation 0 (contribute exactly 0, not listed)
    pub n_silent: usize,
    pub checks: Vec<Check>,
}

impl Runtime {
    /// Contribution of a source to a: a_c = φ/2·(p_c·q + p·q_c) (gated), φ·p_c (not gated); Σ_c a_c = a.
    pub fn a_of(&self, c: &Contrib) -> f64 {
        if self.rd.gated { 0.5 * self.phi * (c.p * self.q + self.p * c.q) } else { self.phi * c.p }
    }
    pub fn level0(&self, gate: bool) -> Vec<(usize, f64)> {
        let w = if gate { &self.rd.wg } else { &self.rd.wu };
        w.iter().zip(&self.xhat).map(|(a, b)| *a as f64 * *b as f64).enumerate().collect()
    }
}

fn rel(dev: f64, scale: f64) -> f64 {
    dev / scale.max(1e-30)
}

/// Frozen runtime scale of a norm on a path, from the recorded pass.
fn scale_of(tr: &Trace) -> impl Fn(Scale) -> f32 + '_ {
    move |sc: Scale| match sc {
        Scale::AttnPost(k) => tr.focus_attn_post[k].unwrap_or(1.0),
        Scale::MlpPost(k) => tr.focus_mlp_post[k].unwrap_or(1.0),
        Scale::Ple(k) => tr.focus_ple_gated[k].as_ref().map_or(1.0, |x| x.1),
        Scale::Stream => 1.0,
    }
}

/// Raw vector of an embedding-type source at the focus.
fn raw_embed(m: &Model, ids: &[u32], focus: usize, src: Src) -> Vec<f32> {
    let mut v = vec![0f32; m.d];
    match src {
        Src::Token(t) => m.embed.row(t as usize, &mut v),
        Src::Position(r) => m.pos_embed.as_ref().unwrap().0.row(r as usize, &mut v),
        _ => {}
    }
    let _ = (ids, focus);
    v
}

/// The vector a source at position `pos` adds to the stream at read point `rp` (runtime; `tr` is a pass
/// whose focus is `pos`).
pub fn source_vector_at(m: &Model, tr: &Trace, ids: &[u32], pos: usize, rp: ReadPoint, src: Src) -> Option<Vec<f32>> {
    let d = m.d;
    let sc = scale_of(tr);
    let mapped = |src: Src, v: &[f32]| -> Option<Vec<f32>> {
        let ops = path(m, src, rp, false)?;
        let mut w = v.to_vec();
        forward_runtime(&ops, &mut w, &sc);
        Some(w)
    };
    match src {
        Src::Token(_) | Src::Position(_) => {
            let ops = path(m, src, rp, true)?;
            let mut v = raw_embed(m, ids, pos, src);
            forward_runtime(&ops, &mut v, &sc);
            Some(v)
        }
        Src::Head(k, h) => mapped(src, &tr.focus_heads[k as usize][h as usize * d..(h as usize + 1) * d]),
        Src::Atom(k, h, j) => {
            let nh = m.blocks[k as usize].attn.n_head;
            mapped(src, &tr.atom_attn[k as usize][(j as usize * nh + h as usize) * d..][..d])
        }
        Src::AttnBias(k) => mapped(src, tr.focus_attn_bias[k as usize].as_ref()?),
        Src::AttnPostBias(k) => mapped(src, m.blocks[k as usize].n_attn_post.as_ref()?.b.as_ref()?),
        Src::MlpBias(k) => mapped(src, tr.focus_mlp_bias[k as usize].as_ref()?),
        Src::MlpPostBias(k) => mapped(src, m.blocks[k as usize].n_mlp_post.as_ref()?.b.as_ref()?),
        Src::Neuron(k, i) => {
            let ops = path(m, src, rp, true)?;
            let a = tr.focus_act[k as usize][i as usize];
            let mut v: Vec<f32> = columns(&m.blocks[k as usize].mlp.down, &[i as usize]).iter().map(|x| x * a).collect();
            forward_runtime(&ops, &mut v, &sc);
            Some(v)
        }
        Src::PleDim(k, e) => {
            let ops = path(m, src, rp, true)?;
            let g = tr.focus_ple_gated[k as usize].as_ref()?.0[e as usize];
            let mut v: Vec<f32> = columns(&m.blocks[k as usize].ple.as_ref()?.proj, &[e as usize]).iter().map(|x| x * g).collect();
            forward_runtime(&ops, &mut v, &sc);
            Some(v)
        }
        Src::TypeEmbed | Src::EmbedNormBias | Src::Offset => None,
    }
}

/// The vector a source adds to the stream at the MLP input of layer l (runtime, focus).
pub fn source_vector(m: &Model, tr: &Trace, ids: &[u32], focus: usize, l: usize, src: Src) -> Option<Vec<f32>> {
    source_vector_at(m, tr, ids, focus, ReadPoint::MlpIn(l), src)
}

/// What reads the stream: read vectors at a read point and position; a source contributes
/// (scale·rg·x_c, scale·ru·x_c).
#[derive(Clone)]
pub struct Reader {
    pub pos: usize,
    pub rp: ReadPoint,
    pub rg: Vec<f32>,
    pub ru: Option<Vec<f32>>,
    pub scale: f64,
}

/// All sources at the reader's position, split exactly; `tr` is a pass whose focus is that position.
pub struct Sources {
    /// embedding parts, heads, biases, every earlier neuron, PLE dimensions
    pub parts: Vec<Contrib>,
    /// the heads split by the position they read
    pub atoms: Vec<Contrib>,
    pub n_silent: usize,
    /// Σ of all source vectors (for the vector check)
    pub xsum: Vec<f64>,
    pub atom_dev: f64,
    pub layer_dev: f64,
    pub ple_dev: f64,
}

pub fn enumerate_sources(m: &Model, tr: &Trace, ids: &[u32], r: &Reader) -> Result<Sources, String> {
    let d = m.d;
    let (pos, rp) = (r.pos, r.rp);
    let sc = scale_of(tr);
    let pq = |v: &[f32]| -> (f64, f64) { (r.scale * dot64(&r.rg, v), r.ru.as_ref().map_or(0.0, |u| r.scale * dot64(u, v))) };
    let mut parts: Vec<Contrib> = Vec::new();
    let mut atoms: Vec<Contrib> = Vec::new();
    let mut xsum = vec![0f64; d];
    let add_vec = |xs: &mut Vec<f64>, v: &[f32]| {
        for (a, b) in xs.iter_mut().zip(v) {
            *a += *b as f64;
        }
    };
    if m.type_embed.is_some() || m.embed_norm.is_some() {
        return Err("--trace: embedding norms / token types are only found in encoders, which are not supported".into());
    }
    // ---- embedding: token and (learned) position
    let mut emb_srcs = vec![Src::Token(ids[pos])];
    if let Some((_, off)) = &m.pos_embed {
        emb_srcs.push(Src::Position((pos + off) as u32));
    }
    for src in emb_srcs {
        let v = source_vector_at(m, tr, ids, pos, rp, src).ok_or("embedding path missing")?;
        add_vec(&mut xsum, &v);
        let (p, q) = pq(&v);
        parts.push(Contrib { src, p, q });
    }
    // ---- attention: heads (atoms inside), biases
    let mut atom_dev = 0f64;
    for k in 0..m.n_layer() {
        if path(m, Src::Head(k as u16, 0), rp, false).is_none() {
            continue;
        }
        let nh = m.blocks[k].attn.n_head;
        let at = &tr.atom_attn[k];
        let nf = if at.is_empty() { 0 } else { at.len() / (nh * d) };
        for h in 0..nh {
            let v = source_vector_at(m, tr, ids, pos, rp, Src::Head(k as u16, h as u16)).unwrap();
            add_vec(&mut xsum, &v);
            let (p, q) = pq(&v);
            parts.push(Contrib { src: Src::Head(k as u16, h as u16), p, q });
            let (mut sp, mut sq, mut sabs) = (0f64, 0f64, 0f64);
            for j in 0..nf {
                if tr.attn_focus[k][h * ids.len() + j] == 0.0 {
                    continue;
                }
                let src = Src::Atom(k as u16, h as u16, j as u32);
                let va = source_vector_at(m, tr, ids, pos, rp, src).unwrap();
                let (ap, aq) = pq(&va);
                sp += ap;
                sq += aq;
                sabs += ap.abs() + aq.abs();
                atoms.push(Contrib { src, p: ap, q: aq });
            }
            if nf > 0 {
                atom_dev = atom_dev.max(rel((sp - p).abs() + (sq - q).abs(), sabs.max(p.abs() + q.abs())));
            }
        }
        for src in [Src::AttnBias(k as u16), Src::AttnPostBias(k as u16)] {
            if let Some(v) = source_vector_at(m, tr, ids, pos, rp, src) {
                add_vec(&mut xsum, &v);
                let (p, q) = pq(&v);
                parts.push(Contrib { src, p, q });
            }
        }
    }
    // ---- every earlier neuron (all of them), MLP biases, PLE dimensions
    let mut n_silent = 0usize;
    let mut layer_dev = 0f64;
    let mut ple_dev = 0f64;
    let back = |ops: &[crate::neuron::Op]| -> (Vec<f32>, Option<Vec<f32>>) {
        let mut g = r.rg.clone();
        adjoint(ops, &mut g);
        let u = r.ru.as_ref().map(|u| {
            let mut u = u.clone();
            adjoint(ops, &mut u);
            u
        });
        (g, u)
    };
    for k in 0..m.n_layer() {
        let Some(ops) = path(m, Src::Neuron(k as u16, 0), rp, true) else { continue };
        let f = r.scale * runtime_factor(&ops, &sc);
        let (rg, ru) = back(&ops);
        let down = &m.blocks[k].mlp.down;
        let ag = col_arrows(down, &rg);
        let au = ru.as_ref().map(|u| col_arrows(down, u));
        let act = &tr.focus_act[k];
        let (mut sp, mut sq) = (0f64, 0f64);
        for (i2, &a) in act.iter().enumerate() {
            if a == 0.0 {
                n_silent += 1;
                continue;
            }
            let p = f * a as f64 * ag[i2] as f64;
            let q = au.as_ref().map_or(0.0, |u| f * a as f64 * u[i2] as f64);
            sp += p;
            sq += q;
            parts.push(Contrib { src: Src::Neuron(k as u16, i2 as u32), p, q });
        }
        // the layer's neurons together = the MLP output vector (vector path)
        let mut v = tr.focus_mlp[k].clone();
        if let Some(ops2) = path(m, Src::Neuron(k as u16, 0), rp, false) {
            forward_runtime(&ops2, &mut v, &sc);
        }
        add_vec(&mut xsum, &v);
        let (vp, vq) = pq(&v);
        layer_dev = layer_dev.max(rel((sp - vp).abs() + (sq - vq).abs(), (vp.abs() + vq.abs()).max(sp.abs() + sq.abs())));
        for src in [Src::MlpBias(k as u16), Src::MlpPostBias(k as u16)] {
            if let Some(v) = source_vector_at(m, tr, ids, pos, rp, src) {
                add_vec(&mut xsum, &v);
                let (p, q) = pq(&v);
                parts.push(Contrib { src, p, q });
            }
        }
        if let (Some(pb), Some((gin, _))) = (&m.blocks[k].ple, &tr.focus_ple_gated[k]) {
            if let Some(ops) = path(m, Src::PleDim(k as u16, 0), rp, true) {
                let f = r.scale * runtime_factor(&ops, &sc);
                let (rg, ru) = back(&ops);
                let ag = col_arrows(&pb.proj, &rg);
                let au = ru.as_ref().map(|u| col_arrows(&pb.proj, u));
                let (mut sp, mut sq) = (0f64, 0f64);
                for (e, &g) in gin.iter().enumerate() {
                    if g == 0.0 {
                        continue;
                    }
                    let p = f * g as f64 * ag[e] as f64;
                    let q = au.as_ref().map_or(0.0, |u| f * g as f64 * u[e] as f64);
                    sp += p;
                    sq += q;
                    parts.push(Contrib { src: Src::PleDim(k as u16, e as u32), p, q });
                }
                let mut v = tr.focus_ple[k].clone().unwrap_or_default();
                if let Some(ops2) = path(m, Src::PleDim(k as u16, 0), rp, false) {
                    forward_runtime(&ops2, &mut v, &sc);
                }
                add_vec(&mut xsum, &v);
                let (vp, vq) = pq(&v);
                ple_dev = ple_dev.max(rel((sp - vp).abs() + (sq - vq).abs(), (vp.abs() + vq.abs()).max(sp.abs() + sq.abs())));
            }
        }
    }
    Ok(Sources { parts, atoms, n_silent, xsum, atom_dev, layer_dev, ple_dev })
}

/// Exact runtime decomposition of neuron i of layer l at the focus. `tr` must come from a full pass with atoms
/// and that focus.
pub fn decompose_neuron(m: &Model, tr: &Trace, ids: &[u32], focus: usize, l: usize, i: usize) -> Result<Runtime, String> {
    let b = m.blocks.get(l).ok_or(format!("S{l}: the model has only {} layers", m.n_layer()))?;
    if b.res == Res::Post {
        return Err("--trace: runtime analysis is not supported for post-LN encoders (BERT/RoBERTa): every sublayer output is renormalized, so sources do not add up at the MLP input. Use --neuron for the static part.".into());
    }
    if !m.has_head {
        return Err("--trace needs a model with an output head".into());
    }
    let rd = Read::new(m, l, i)?;
    let rep = repertoire(&rd);
    let x = tr.focus_mlp_x[l].clone();
    let xhat = tr.focus_mlp_in[l].clone();
    let s = pre_norm(b).map_or(1.0, |n| n.scale(&x) as f64);
    let gated = rd.gated;
    let reader = Reader { pos: focus, rp: ReadPoint::MlpIn(l), rg: rd.g.clone(), ru: if gated { Some(rd.u.clone()) } else { None }, scale: s };
    let src = enumerate_sources(m, tr, ids, &reader)?;
    let mut parts = src.parts;
    let mut checks: Vec<Check> = Vec::new();
    checks.push(Check { what: "Σ atoms (head reads position j) = head".into(), dev: src.atom_dev, tol: 1e-4 });
    checks.push(Check { what: "Σ neurons of a layer (static arrow × scales × a) = layer's MLP output read by the neuron (vector path)".into(), dev: src.layer_dev, tol: 1e-4 });
    if m.blocks.iter().any(|b| b.ple.is_some()) {
        checks.push(Check { what: "Σ PLE dimensions of a layer = layer's PLE output read by the neuron".into(), dev: src.ple_dev, tol: 1e-4 });
    }
    parts.push(Contrib { src: Src::Offset, p: rd.cg, q: rd.cu });
    // ---- checks
    let xn = x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
    let xdev = src.xsum.iter().zip(&x).map(|(a, b)| (a - *b as f64).powi(2)).sum::<f64>().sqrt();
    checks.push(Check { what: "Σ source vectors = stream at the MLP input (recorded)".into(), dev: rel(xdev, xn), tol: 1e-5 });
    let p = dot64(&rd.wg, &xhat) + rd.bg;
    let q = if gated { dot64(&rd.wu, &xhat) + rd.bu } else { 0.0 };
    let (sum_p, sum_q) = parts.iter().fold((0f64, 0f64), |acc, c| (acc.0 + c.p, acc.1 + c.q));
    let abs_p: f64 = parts.iter().map(|c| c.p.abs()).sum();
    let abs_q: f64 = parts.iter().map(|c| c.q.abs()).sum();
    checks.push(Check { what: "Σ sources = p".into(), dev: rel((sum_p - p).abs(), abs_p.max(p.abs())), tol: 1e-4 });
    if gated {
        checks.push(Check { what: "Σ sources = q".into(), dev: rel((sum_q - q).abs(), abs_q.max(q.abs())), tol: 1e-4 });
    }
    let a = rd.value(p, q);
    let a_rec = tr.focus_act[l][i] as f64;
    let a_scale = a_rec.abs().max(1e-3 * rep.a_max.abs().max(rep.a_min.abs())).max(1e-12);
    checks.push(Check { what: "a = act(p)·q from x̂ = activation recorded in the forward pass".into(), dev: rel((a - a_rec).abs(), a_scale), tol: 1e-3 });
    let ph = phi(rd.act, p);
    let a_pairs = if gated { ph * sum_p * sum_q } else { ph * sum_p };
    checks.push(Check { what: "Σ pairs φ·p_c·q_c' = a".into(), dev: rel((a_pairs - a).abs(), a_scale), tol: 1e-3 });
    let r = rep.r_of(p, q);
    checks.push(Check { what: "(p, q) inside the ellipse: r ≤ 1".into(), dev: (r - 1.0).max(0.0), tol: 1e-6 });
    let span = (rep.a_max - rep.a_min).abs().max(1e-12);
    checks.push(Check { what: "a_min ≤ a ≤ a_max".into(), dev: ((rep.a_min - a).max(a - rep.a_max)).max(0.0) / span, tol: 1e-6 });
    Ok(Runtime { rd, rep, x, xhat, s, p, q, a, a_rec, phi: ph, parts, atoms: src.atoms, n_silent: src.n_silent, checks })
}

/// Static arrow × runtime scales × source activation = runtime contribution, for the strongest
/// sources of each kind: the static side goes through the adjoint path, the runtime side through the
/// vector path – two different computations.
pub fn static_vs_runtime(m: &Model, tr: &Trace, ids: &[u32], focus: usize, rt: &Runtime, n: usize) -> Check {
    let l = rt.rd.l;
    let rp = ReadPoint::MlpIn(l);
    let sc = scale_of(tr);
    let mut ranked: Vec<&Contrib> = rt.parts.iter().filter(|c| matches!(c.src, Src::Neuron(..) | Src::Token(_) | Src::Position(_) | Src::PleDim(..))).collect();
    ranked.sort_by(|a, b| (b.p.abs() + b.q.abs()).partial_cmp(&(a.p.abs() + a.q.abs())).unwrap_or(std::cmp::Ordering::Equal));
    let mut worst = 0f64;
    for c in ranked.into_iter().take(n) {
        let ops = path(m, c.src, rp, true).unwrap();
        // static: adjoint read vector · raw vector, per unit activation
        let raw: Vec<f32> = match c.src {
            Src::Neuron(k, i) => columns(&m.blocks[k as usize].mlp.down, &[i as usize]),
            Src::PleDim(k, e) => columns(&m.blocks[k as usize].ple.as_ref().unwrap().proj, &[e as usize]),
            _ => raw_embed(m, ids, focus, c.src),
        };
        let (rg, ru) = read_back(&rt.rd, &ops);
        let act = match c.src {
            Src::Neuron(k, i) => tr.focus_act[k as usize][i as usize] as f64,
            Src::PleDim(k, e) => tr.focus_ple_gated[k as usize].as_ref().unwrap().0[e as usize] as f64,
            _ => 1.0,
        };
        let f = rt.s * runtime_factor(&ops, &sc) * act;
        let sp = f * dot64(&rg, &raw);
        let sq = if rt.rd.gated { f * dot64(&ru, &raw) } else { 0.0 };
        // runtime: the vector the source adds, read by the neuron
        let v = source_vector(m, tr, ids, focus, l, c.src).unwrap();
        let vp = rt.s * dot64(&rt.rd.g, &v);
        let vq = if rt.rd.gated { rt.s * dot64(&rt.rd.u, &v) } else { 0.0 };
        worst = worst.max(rel((sp - vp).abs() + (sq - vq).abs(), vp.abs() + vq.abs()));
    }
    Check { what: format!("static arrow × runtime scales × source activation = runtime contribution ({n} strongest sources)"), dev: worst, tol: 1e-4 }
}

/// Scalar products of one source: W[i,j]·γ_j·s·x_c,j (LN: centered x_c); Σ_j = p_c.
pub fn level01(m: &Model, rt: &Runtime, v: &[f32], gate: bool) -> Vec<(usize, f64)> {
    let b = &m.blocks[rt.rd.l];
    let w = if gate { &rt.rd.wg } else { &rt.rd.wu };
    let norm = pre_norm(b);
    let mean = if rt.rd.ln { v.iter().map(|x| *x as f64).sum::<f64>() / v.len() as f64 } else { 0.0 };
    (0..v.len())
        .map(|j| {
            let gamma = norm.map_or(1.0, |n| n.w[j] as f64);
            (j, w[j] as f64 * gamma * rt.s * (v[j] as f64 - mean))
        })
        .collect()
}

pub fn top_abs(v: &[(usize, f64)], n: usize) -> (Vec<(usize, f64)>, f64) {
    let mut s = v.to_vec();
    s.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(std::cmp::Ordering::Equal));
    let rest: f64 = s.iter().skip(n).map(|x| x.1).sum();
    s.truncate(n);
    (s, rest)
}


// ====================================================================== reruns

/// Logits at the focus after rerunning from layer l with neuron activations replaced at the focus.
/// One variant per list of (neuron, value). Returns [clean row of the same pass, unchanged copy, variants…].
fn rerun_patches(m: &Model, ids: &[u32], focus: usize, l: usize, ck: &Checkpoint, variants: &[Vec<(usize, f32)>]) -> Vec<Vec<f32>> {
    let (t, d) = (ids.len(), m.d);
    let n = variants.len() + 1;
    let mut pos: Vec<usize> = (0..t).collect();
    let mut seg_start = vec![0usize; t];
    let mut seg_row0 = vec![0usize; t];
    let mut ids_all = ids.to_vec();
    for v in 0..n {
        pos.push(focus);
        seg_start.push(focus);
        seg_row0.push(t + v);
        ids_all.push(ids[focus]);
    }
    let rows = pos.len();
    let xl = &ck.x[l];
    let mut x = Vec::with_capacity(rows * d);
    for &q in &pos {
        x.extend_from_slice(&xl[q * d..(q + 1) * d]);
    }
    let kv: Vec<Option<(Vec<f32>, Vec<f32>)>> = ck
        .kv
        .iter()
        .enumerate()
        .map(|(s, e)| {
            if s >= l {
                return None;
            }
            e.as_ref().map(|(k, v)| {
                let kd = k.len() / t;
                let mut k2 = Vec::with_capacity(rows * kd);
                let mut v2 = Vec::with_capacity(rows * kd);
                for &q in &pos {
                    k2.extend_from_slice(&k[q * kd..(q + 1) * kd]);
                    v2.extend_from_slice(&v[q * kd..(q + 1) * kd]);
                }
                (k2, v2)
            })
        })
        .collect();
    let mut o = RunOpts {
        keep_final: true,
        start: Some(Start { layer: l, x, kv: Arc::new(kv) }),
        variants: Some(Arc::new(Variants { pos, seg_start, seg_row0 })),
        ..Default::default()
    };
    for (vi, v) in variants.iter().enumerate() {
        for &(i, val) in v {
            o.patch.push((l, t + 1 + vi, i, val));
        }
    }
    let tr = m.run(&ids_all, focus, &o);
    let mut xs = Vec::with_capacity((n + 1) * d);
    xs.extend_from_slice(&tr.x_final[focus * d..(focus + 1) * d]);
    for v in 0..n {
        xs.extend_from_slice(&tr.x_final[(t + v) * d..(t + v + 1) * d]);
    }
    let lg = m.final_logits(&xs, n + 1);
    let v = lg.len() / (n + 1);
    lg.chunks(v).map(|c| c.to_vec()).collect()
}

/// Effect on the output: log-odds of the target with − without, KL(real ‖ changed), new top token
#[derive(Clone, Copy)]
struct Eff {
    dlo: f64,
    kl: f64,
    top: (u32, f32),
}

fn eff(z: &[f32], z2: &[f32], y: u32) -> Eff {
    let (lz, _) = softmax(z);
    let (lz2, _) = softmax(z2);
    let kl: f64 = z.iter().zip(z2).map(|(a, b)| {
        let lp = *a as f64 - lz;
        lp.exp() * (lp - (*b as f64 - lz2))
    }).sum();
    Eff { dlo: log_odds(z, y as usize) - log_odds(z2, y as usize), kl: kl.max(0.0), top: top_probs(z2, 1)[0] }
}

/// Final-norm frame of the pass: (s_f, normed whole stream, logits, logZ, p)
struct Frame {
    s_f: f32,
    h_all: Vec<f32>,
    z: Vec<f32>,
    logz: f64,
    p: Vec<f32>,
}

fn frame(m: &Model, tr: &Trace) -> Frame {
    let nl = m.n_layer();
    let xf = &tr.focus_resid[nl];
    let (s_f, h_all) = match &m.final_norm {
        Some(n) => {
            let mut h = vec![0f32; m.d];
            let s = n.apply(xf, &mut h).0;
            (s, h)
        }
        None => (1.0, xf.clone()),
    };
    let z = tr.logits.clone();
    let (logz, p) = softmax(&z);
    Frame { s_f, h_all, z, logz, p }
}

/// Direct voices (exact, full softmax) of neurons at the focus: log-odds effect on the targets
fn direct_voices(m: &Model, tr: &Trace, fr: &Frame, neurons: &[(usize, usize)], ys: &[u32]) -> Vec<f32> {
    let sc = scale_of(tr);
    let mut h = fr.h_all.clone();
    for &(l, i) in neurons {
        let ops = path(m, Src::Neuron(l as u16, i as u32), ReadPoint::Final, true).unwrap();
        let a = tr.focus_act[l][i];
        let mut v: Vec<f32> = columns(&m.blocks[l].mlp.down, &[i]).iter().map(|x| x * a).collect();
        forward_runtime(&ops, &mut v, &sc);
        let hv = m.final_norm.as_ref().map_or(v.clone(), |n| n.map_part(fr.s_f, &v));
        h.extend(hv);
    }
    let (vs, _) = voices(m, &h, &fr.z, &fr.p, fr.logz, ys, 0);
    vs[1..].iter().map(|v| v.dlo[0]).collect()
}

/// Start neurons for --trace auto: all neurons at the focus ranked by a linearized direct log-odds
/// effect (selection only), the best candidates then by their exact direct voice.
fn auto_start(m: &Model, tr: &Trace, fr: &Frame, y0: u32, n: usize) -> Vec<(usize, usize, f32)> {
    let d = m.d;
    let lm = m.lm_head();
    let top = top_probs(&fr.z, 64);
    let py = fr.p[y0 as usize] as f64;
    let kap = |t: usize| m.logit_cap.map_or(1.0, |c| 1.0 - (fr.z[t] / c).powi(2));
    let mut e = vec![0f32; d];
    let mut buf = vec![0f32; d];
    lm.row(y0 as usize, &mut buf);
    e.iter_mut().zip(&buf).for_each(|(a, b)| *a += kap(y0 as usize) * b);
    let rest: f64 = top.iter().filter(|x| x.0 != y0).map(|x| x.1 as f64).sum::<f64>().max(1e-30);
    for &(t, pt) in &top {
        if t == y0 {
            continue;
        }
        let r = (pt as f64 / rest * (1.0 - py).max(1e-30) / (1.0 - py).max(1e-30)) as f32;
        lm.row(t as usize, &mut buf);
        let k = kap(t as usize);
        e.iter_mut().zip(&buf).for_each(|(a, b)| *a -= r * k * b);
    }
    let back = m.final_norm.as_ref().map_or(e.clone(), |nm| nm.back(fr.s_f, &e));
    let sc = scale_of(tr);
    let mut cand: Vec<(usize, usize, f32)> = Vec::new();
    for l in 0..m.n_layer() {
        let ops = path(m, Src::Neuron(l as u16, 0), ReadPoint::Final, true).unwrap();
        let mut r = back.clone();
        adjoint(&ops, &mut r);
        let f = runtime_factor(&ops, &sc) as f32;
        let w = col_arrows(&m.blocks[l].mlp.down, &r);
        for (i, &a) in tr.focus_act[l].iter().enumerate() {
            cand.push((l, i, a * f * w[i]));
        }
    }
    cand.sort_by(|a, b| b.2.abs().partial_cmp(&a.2.abs()).unwrap_or(std::cmp::Ordering::Equal));
    cand.truncate((4 * n).max(32));
    let ids: Vec<(usize, usize)> = cand.iter().map(|x| (x.0, x.1)).collect();
    let exact = direct_voices(m, tr, fr, &ids, &[y0]);
    let mut out: Vec<(usize, usize, f32)> = ids.iter().zip(&exact).map(|(&(l, i), &v)| (l, i, v)).collect();
    out.sort_by(|a, b| b.2.abs().partial_cmp(&a.2.abs()).unwrap_or(std::cmp::Ordering::Equal));
    out.truncate(n);
    out
}

// ====================================================================== removals

struct Removal {
    src: Src,
    a_c: f64,
    /// Δa = a − a' with: frozen scale and φ (linear), frozen scale only, exact recomputation
    da_lin: f64,
    da_frozen: f64,
    da_real: f64,
    a_new: f32,
    eff: Option<Eff>,
}

/// What this neuron's activation becomes without source c (only in this neuron): x' = x − x_c,
/// p' and q' exactly from Norm(x') (not with the frozen scale).
fn without_source(m: &Model, rt: &Runtime, v: &[f32]) -> f32 {
    let b = &m.blocks[rt.rd.l];
    let x2: Vec<f32> = rt.x.iter().zip(v).map(|(a, c)| a - c).collect();
    let xh = match pre_norm(b) {
        Some(n) => {
            let mut o = vec![0f32; x2.len()];
            n.apply(&x2, &mut o);
            o
        }
        None => x2,
    };
    let p = (dot64(&rt.rd.wg, &xh) + rt.rd.bg) as f32;
    let f = crate::model::act_fn(rt.rd.act);
    if rt.rd.gated {
        let q = (dot64(&rt.rd.wu, &xh) + rt.rd.bu) as f32;
        f(p) * q
    } else {
        f(p)
    }
}

// ====================================================================== content at a read position (phase 2)

/// What an atom carries – head h of layer k at the reader's position reading position j – split into the
/// sources at position j. With the attention weight, the norm scales at both positions and the value norm
/// frozen, the atom is linear in the stream x_j before the layer whose values the head reads (shared K/V:
/// the source layer). The reader's vectors are brought back through post-norm, W_O, the attention weight,
/// value norm, W_V and the pre-norm at j; the norm bias and value bias become a constant offset.
pub struct Content {
    pub kv_layer: usize,
    /// the atom's contribution to the reader (level-1 forward path)
    pub atom: (f64, f64),
    /// attention-norm bias and value bias through the same chain
    pub offset: (f64, f64),
    /// read vectors on the stream at position j (before kv_layer)
    pub reader: Reader,
    pub src: Sources,
    /// |Σ parts + offset − atom| relative; vector check at j; pass with focus j against the checkpoint
    pub dev: f64,
    pub xdev: f64,
    pub pass_dev: f64,
}

#[allow(clippy::too_many_arguments)]
pub fn content(m: &Model, tr_in: &Trace, tr_j: &Trace, ck: &Checkpoint, ids: &[u32], r: &Reader, k: usize, h: usize, j: usize) -> Result<Content, String> {
    use crate::neuron::Op;
    let d = m.d;
    let t = ids.len();
    let b = &m.blocks[k];
    let a = &b.attn;
    let (nh, hd) = (a.n_head, a.hd);
    // reader → attention output of layer k at the reader's position
    let ops = path(m, Src::Head(k as u16, h as u16), r.rp, false).ok_or("the head does not reach the reader")?;
    let start = |v: &[f32]| -> Vec<f32> {
        let mut g: Vec<f32> = v.iter().map(|x| (*x as f64 * r.scale) as f32).collect();
        adjoint(&ops, &mut g);
        if b.res == Res::Pre {
            if let Some(n) = &b.n_attn_post {
                let s_post = tr_in.focus_attn_post[k].unwrap_or(1.0);
                adjoint(&[Op::Norm(n, Scale::Stream)], &mut g);
                g.iter_mut().for_each(|x| *x *= s_post);
            }
        }
        g
    };
    let mut gs = vec![start(&r.rg)];
    if let Some(u) = &r.ru {
        gs.push(start(u));
    }
    // W_O of head h, the attention weight from the reader's position to j
    let w = tr_in.attn_focus[k][h * t + j];
    let rz: Vec<Vec<f32>> = gs.iter().map(|g| a.wo.t_matvec(g)[h * hd..(h + 1) * hd].iter().map(|x| x * w).collect()).collect();
    // value side: the layer whose K/V the head reads, at position j
    let kv = a.kv_from.unwrap_or(k);
    let bkv = &m.blocks[kv];
    let av = &bkv.attn;
    let xj = &ck.x[kv][j * d..(j + 1) * d];
    let n_attn = match (&bkv.n_attn, bkv.res) {
        (Some(n), Res::Pre | Res::Parallel) => Some(n),
        _ => None,
    };
    let (s_a, hhat) = match n_attn {
        Some(n) => {
            let mut o = vec![0f32; d];
            let s = n.apply(xj, &mut o).0;
            (s, o)
        }
        None => (1.0, xj.to_vec()),
    };
    let (wmat, row0, bias) = match &av.wqkv {
        Some(wq) => (wq, h * 3 * hd + 2 * hd, av.bqkv.as_ref()),
        None => {
            let g = h / (nh / a.n_kv.max(1)).max(1);
            let wv = if av.v_is_k { av.wk.as_ref() } else { av.wv.as_ref() }.ok_or("value matrix missing")?;
            (wv, g * hd, if av.v_is_k { av.bk.as_ref() } else { av.bv.as_ref() })
        }
    };
    let bias_v: Vec<f32> = (0..hd).map(|c| bias.map_or(0.0, |bb| bb[row0 + c])).collect();
    let vraw: Vec<f32> = wmat.matmul(&hhat, 1)[row0..row0 + hd].iter().zip(&bias_v).map(|(x, bb)| x + bb).collect();
    let rv: Vec<Vec<f32>> = match &av.vn {
        Some(n) => {
            let s_v = n.scale(&vraw);
            rz.iter().map(|g| g.iter().zip(&n.w).map(|(x, wn)| x * wn * s_v).collect()).collect()
        }
        None => rz,
    };
    // constant part: W_V·β of the attention norm plus the value bias
    let wbeta: Vec<f32> = match n_attn.and_then(|n| n.b.as_ref()) {
        Some(beta) => wmat.matmul(beta, 1)[row0..row0 + hd].to_vec(),
        None => vec![0f32; hd],
    };
    let cvec: Vec<f32> = wbeta.iter().zip(&bias_v).map(|(x, y)| x + y).collect();
    let offs: Vec<f64> = rv.iter().map(|g| dot64(g, &cvec)).collect();
    // back through W_V (the head's value rows) and the attention pre-norm at j (frozen scale)
    let rx: Vec<Vec<f32>> = rv
        .iter()
        .map(|g| {
            let mut u = vec![0f32; wmat.rows];
            u[row0..row0 + hd].copy_from_slice(g);
            let mut rh = wmat.t_matvec(&u);
            if let Some(n) = n_attn {
                adjoint(&[Op::Norm(n, Scale::Stream)], &mut rh);
                rh.iter_mut().for_each(|x| *x *= s_a);
            }
            rh
        })
        .collect();
    let reader = Reader { pos: j, rp: ReadPoint::AttnIn(kv), rg: rx[0].clone(), ru: rx.get(1).cloned(), scale: 1.0 };
    let src = enumerate_sources(m, tr_j, ids, &reader)?;
    // checks
    let va = source_vector_at(m, tr_in, ids, r.pos, r.rp, Src::Atom(k as u16, h as u16, j as u32)).ok_or("atom vector missing")?;
    let atom = (r.scale * dot64(&r.rg, &va), r.ru.as_ref().map_or(0.0, |u| r.scale * dot64(u, &va)));
    let offset = (offs[0], offs.get(1).copied().unwrap_or(0.0));
    let (sp, sq) = src.parts.iter().fold(offset, |acc, c| (acc.0 + c.p, acc.1 + c.q));
    let abs: f64 = src.parts.iter().map(|c| c.p.abs() + c.q.abs()).sum::<f64>() + offset.0.abs() + offset.1.abs();
    let dev = rel((sp - atom.0).abs() + (sq - atom.1).abs(), abs.max(atom.0.abs() + atom.1.abs()));
    let xn = xj.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
    let xdev = rel(src.xsum.iter().zip(xj).map(|(a, b)| (a - *b as f64).powi(2)).sum::<f64>().sqrt(), xn);
    let pass_dev = tr_j.focus_resid[kv].iter().zip(xj).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max) as f64;
    Ok(Content { kv_layer: kv, atom, offset, reader, src, dev, xdev, pass_dev })
}

// ====================================================================== backward zoom

/// a_c in units of one neuron's activation: φ/2·(p_c·q + p·q_c), not gated φ·p_c
#[derive(Clone, Copy)]
struct Units {
    p: f64,
    q: f64,
    phi: f64,
    gated: bool,
}

impl Units {
    fn a(&self, p: f64, q: f64) -> f64 {
        if self.gated { 0.5 * self.phi * (p * self.q + self.p * q) } else { self.phi * p }
    }
}

struct Node {
    pos: usize,
    l: usize,
    i: usize,
    a: f64,
    children: Vec<Child>,
}

enum Child {
    Neuron { a_c: f64, node: Node },
    Head { src: Src, a_c: f64, atoms: Vec<AtomView> },
    Other { src: Src, a_c: f64 },
}

struct AtomView {
    j: u32,
    a_c: f64,
    content: Option<ContentView>,
}

struct ContentView {
    kv: usize,
    offset_a: f64,
    dev: f64,
    items: Vec<Child>,
}

#[derive(Default)]
struct ZoomStats {
    contents: usize,
    dev: f64,
    xdev: f64,
    pass_dev: f64,
}

struct Zoom<'a> {
    m: &'a Model,
    ids: &'a [u32],
    ck: &'a Checkpoint,
    top: usize,
    traces: std::cell::RefCell<std::collections::HashMap<usize, std::rc::Rc<Trace>>>,
    runtimes: std::cell::RefCell<std::collections::HashMap<(usize, usize, usize), std::rc::Rc<Runtime>>>,
    stats: std::cell::RefCell<ZoomStats>,
}

impl<'a> Zoom<'a> {
    /// recording pass with focus pos (the values at pos are those of the main pass; checked)
    fn trace_at(&self, pos: usize) -> std::rc::Rc<Trace> {
        if let Some(t) = self.traces.borrow().get(&pos) {
            return t.clone();
        }
        let t = std::rc::Rc::new(self.m.run(self.ids, pos, &RunOpts { atoms: true, ..Default::default() }));
        self.traces.borrow_mut().insert(pos, t.clone());
        t
    }
    fn runtime(&self, pos: usize, l: usize, i: usize) -> Result<std::rc::Rc<Runtime>, String> {
        if let Some(r) = self.runtimes.borrow().get(&(pos, l, i)) {
            return Ok(r.clone());
        }
        let tr = self.trace_at(pos);
        let r = std::rc::Rc::new(decompose_neuron(self.m, &tr, self.ids, pos, l, i)?);
        self.runtimes.borrow_mut().insert((pos, l, i), r.clone());
        Ok(r)
    }
    fn leaf(&self, pos: usize, l: usize, i: usize) -> Node {
        Node { pos, l, i, a: self.trace_at(pos).focus_act[l][i] as f64, children: Vec::new() }
    }
    fn ranked(&self, parts: &[Contrib], u: &Units) -> Vec<(Src, f64)> {
        let mut r: Vec<(Src, f64)> = parts.iter().filter(|c| c.src != Src::Offset).map(|c| (c.src, u.a(c.p, c.q))).collect();
        r.sort_by(|x, y| y.1.abs().partial_cmp(&x.1.abs()).unwrap_or(std::cmp::Ordering::Equal));
        r.truncate(self.top);
        r
    }
    /// children for a list of sources at position pos, read by `reader` (a_c in units u)
    fn children(&self, pos: usize, src: &Sources, reader: &Reader, u: &Units, depth: usize) -> Result<Vec<Child>, String> {
        let mut out = Vec::new();
        for (s, a_c) in self.ranked(&src.parts, u) {
            match s {
                Src::Neuron(k, i) => {
                    let node = if depth > 1 { self.neuron(pos, k as usize, i as usize, depth - 1)? } else { self.leaf(pos, k as usize, i as usize) };
                    out.push(Child::Neuron { a_c, node });
                }
                Src::Head(k, h) => {
                    let mut at: Vec<(u32, f64)> = src
                        .atoms
                        .iter()
                        .filter_map(|c| match c.src {
                            Src::Atom(kk, hh, j) if kk == k && hh == h => Some((j, u.a(c.p, c.q))),
                            _ => None,
                        })
                        .collect();
                    at.sort_by(|x, y| y.1.abs().partial_cmp(&x.1.abs()).unwrap_or(std::cmp::Ordering::Equal));
                    at.truncate(self.top);
                    let mut atoms = Vec::new();
                    for (n, (j, a_j)) in at.into_iter().enumerate() {
                        let content = if depth > 1 && n < 2 { Some(self.content(reader, u, k as usize, h as usize, j as usize, depth - 1)?) } else { None };
                        atoms.push(AtomView { j, a_c: a_j, content });
                    }
                    out.push(Child::Head { src: s, a_c, atoms });
                }
                _ => out.push(Child::Other { src: s, a_c }),
            }
        }
        Ok(out)
    }
    fn neuron(&self, pos: usize, l: usize, i: usize, depth: usize) -> Result<Node, String> {
        let rt = self.runtime(pos, l, i)?;
        let u = Units { p: rt.p, q: rt.q, phi: rt.phi, gated: rt.rd.gated };
        let reader = Reader { pos, rp: ReadPoint::MlpIn(l), rg: rt.rd.g.clone(), ru: if rt.rd.gated { Some(rt.rd.u.clone()) } else { None }, scale: rt.s };
        let src = Sources { parts: rt.parts.clone(), atoms: rt.atoms.clone(), n_silent: rt.n_silent, xsum: Vec::new(), atom_dev: 0.0, layer_dev: 0.0, ple_dev: 0.0 };
        let children = self.children(pos, &src, &reader, &u, depth)?;
        Ok(Node { pos, l, i, a: rt.a, children })
    }
    fn content(&self, reader: &Reader, u: &Units, k: usize, h: usize, j: usize, depth: usize) -> Result<ContentView, String> {
        let tr_in = self.trace_at(reader.pos);
        let tr_j = self.trace_at(j);
        let c = content(self.m, &tr_in, &tr_j, self.ck, self.ids, reader, k, h, j)?;
        {
            let mut st = self.stats.borrow_mut();
            st.contents += 1;
            st.dev = st.dev.max(c.dev);
            st.xdev = st.xdev.max(c.xdev);
            st.pass_dev = st.pass_dev.max(c.pass_dev);
        }
        let items = self.children(j, &c.src, &c.reader, u, depth)?;
        Ok(ContentView { kv: c.kv_layer, offset_a: u.a(c.offset.0, c.offset.1), dev: c.dev, items })
    }
}

// ====================================================================== run

pub struct Opts {
    pub result_json: Option<PathBuf>,
    /// explicit neurons, or empty for "auto"
    pub neurons: Vec<(usize, usize)>,
    pub depth: usize,
    pub top: usize,
    pub scalars: usize,
    pub html: PathBuf,
    pub dump: Option<PathBuf>,
    pub atlas: Option<PathBuf>,
    /// level 0×1 on request for this source ("S14 head 1", "S14 head 1 @16", "S14 #868", "token", …)
    pub scalars_of: Option<String>,
}

/// "token", "position", "S14 head 1", "S14 head 1 @16", "S14 #868", "S3 PLE #5", "S4 attn bias", "S4 MLP bias"
fn parse_src(m: &Model, ids: &[u32], focus: usize, text: &str) -> Result<Src, String> {
    let err = || format!("--scalars-of '{text}': give a source like \"token\", \"S14 head 1\", \"S14 head 1 @16\", \"S14 #868\", \"S3 PLE #5\", \"S4 attn bias\", \"S4 MLP bias\"");
    let t = text.to_lowercase().replace('@', " @ ");
    let w: Vec<&str> = t.split_whitespace().collect();
    match w.as_slice() {
        ["token"] | ["embedding"] => return Ok(Src::Token(ids[focus])),
        ["position"] => return m.pos_embed.as_ref().map(|(_, off)| Src::Position((focus + off) as u32)).ok_or_else(|| "the model has no learned positions".to_string()),
        _ => {}
    }
    let l: u16 = w.first().and_then(|x| x.trim_start_matches(['s', 'l']).parse().ok()).ok_or_else(err)?;
    let num_of = |x: &str| x.trim_start_matches('#').parse::<u32>().map_err(|_| err());
    Ok(match &w[1..] {
        ["head", h] => Src::Head(l, num_of(h)? as u16),
        ["head", h, "@", j] | ["head", h, "reads", j] => Src::Atom(l, num_of(h)? as u16, num_of(j)?),
        [n] if n.starts_with('#') => Src::Neuron(l, num_of(n)?),
        ["ple", e] => Src::PleDim(l, num_of(e)?),
        ["attn" | "attention", "bias"] => Src::AttnBias(l),
        ["mlp", "bias"] => Src::MlpBias(l),
        _ => return Err(err()),
    })
}

/// Everything computed for one traced neuron.
struct Traced {
    rt: Runtime,
    direct: f32,
    total: Eff,
    removals: Vec<Removal>,
    atlas_rank: Vec<(Src, Option<usize>)>,
    tree: Node,
    /// level 0×1 for the strongest sources: (source, p_c, top products for p + rest, same for q)
    l01: Vec<L01>,
    /// level 0×1 for the source asked for with --scalars-of
    asked: Option<Result<L01, String>>,
}

struct L01 {
    src: Src,
    p: f64,
    q: f64,
    top_p: Vec<(usize, f64)>,
    rest_p: f64,
    top_q: Vec<(usize, f64)>,
    rest_q: f64,
}

pub fn run(m: &Model, ids: &[u32], focus: usize, target: Option<u32>, o: &Opts, st: &Style) -> Result<(), String> {
    if !m.has_head || m.blocks.iter().any(|b| b.res == Res::Post) {
        return Err("--trace: runtime analysis needs a decoder with an output head; post-LN encoders (BERT/RoBERTa) only have the static part (--neuron)".into());
    }
    let t0 = Instant::now();
    eprintln!("trace: recording pass (atoms, checkpoints) …");
    let mut tr0 = m.run(ids, focus, &RunOpts { atoms: true, checkpoint: true, ..Default::default() });
    let ck = tr0.checkpoint.take().ok_or("checkpoints missing")?;
    let tr = std::rc::Rc::new(tr0);
    let y0 = target.unwrap_or(top_probs(&tr.logits, 1)[0].0);
    let fr = frame(m, &tr);
    let atlas = match &o.atlas {
        Some(p) => {
            let at = crate::atlas::Atlas::open(p)?;
            if at.model != m.name {
                return Err(format!("{}: the atlas belongs to {}, not to {}", p.display(), at.model, m.name));
            }
            let done = at.layers_done();
            eprintln!("trace: atlas with k = {}, {} of {} layers built", at.k, done.iter().filter(|x| **x).count(), done.len());
            Some(at)
        }
        None => None,
    };
    let mut roots: Vec<(usize, usize)> = o.neurons.clone();
    let mut auto_info: Vec<(usize, usize, f32)> = Vec::new();
    if roots.is_empty() {
        eprintln!("trace auto: ranking all neurons at the focus by their direct voice …");
        auto_info = auto_start(m, &tr, &fr, y0, o.top);
        roots = auto_info.iter().map(|x| (x.0, x.1)).collect();
    }
    let zoomer = Zoom {
        m,
        ids,
        ck: &ck,
        top: o.top,
        traces: std::cell::RefCell::new(std::collections::HashMap::from([(focus, tr.clone())])),
        runtimes: std::cell::RefCell::new(std::collections::HashMap::new()),
        stats: std::cell::RefCell::new(ZoomStats::default()),
    };
    let mut traced: Vec<Traced> = Vec::new();
    for &(l, i) in &roots {
        eprintln!("trace S{l} #{i}: decomposition, reruns, backward zoom …");
        let mut rt = decompose_neuron(m, &tr, ids, focus, l, i)?;
        rt.checks.push(static_vs_runtime(m, &tr, ids, focus, &rt, 8));
        let direct = direct_voices(m, &tr, &fr, &[(l, i)], &[y0])[0];
        // removals: the neuron itself, and the strongest sources only from this neuron
        let mut ranked: Vec<Contrib> = rt.parts.iter().filter(|c| c.src != Src::Offset).cloned().collect();
        ranked.sort_by(|a, b| rt.a_of(b).abs().partial_cmp(&rt.a_of(a).abs()).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(o.top.max(6));
        let mut removals: Vec<Removal> = Vec::new();
        for c in &ranked {
            let v = source_vector(m, &tr, ids, focus, l, c.src).ok_or("source vector missing")?;
            let a2 = without_source(m, &rt, &v);
            let (pp, qq) = (rt.p - c.p, rt.q - c.q);
            let a_frozen = rt.rd.value(pp, qq);
            let a_lin = if rt.rd.gated { rt.phi * pp * qq } else { rt.phi * pp };
            removals.push(Removal { src: c.src, a_c: rt.a_of(c), da_lin: rt.a - a_lin, da_frozen: rt.a - a_frozen, da_real: rt.a - a2 as f64, a_new: a2, eff: None });
        }
        let a_rec = tr.focus_act[l][i];
        let mut variants: Vec<Vec<(usize, f32)>> = vec![vec![(i, a_rec)], vec![(i, 0.0)]];
        variants.extend(removals.iter().map(|r| vec![(i, r.a_new)]));
        let lg = rerun_patches(m, ids, focus, l, &ck, &variants);
        let (clean_row, ident, orig, zero) = (&lg[0], &lg[1], &lg[2], &lg[3]);
        let maxdev = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max) as f64;
        rt.checks.push(Check { what: "rerun with patch = original value = unchanged copy (|Δ logit|)".into(), dev: maxdev(orig, ident), tol: 1e-5 });
        rt.checks.push(Check { what: "unchanged copy in the rerun = clean pass (|Δ logit|)".into(), dev: maxdev(ident, &tr.logits).max(maxdev(clean_row, &tr.logits)), tol: 1e-3 });
        let total = eff(&tr.logits, zero, y0);
        for (r, z2) in removals.iter_mut().zip(&lg[4..]) {
            r.eff = Some(eff(&tr.logits, z2, y0));
        }
        // atlas: where the strongest sources stand in the static ranking, and the stored arrows
        let mut atlas_rank = Vec::new();
        if let Some(at) = &atlas {
            if let Some(rec) = at.record(l, i).filter(|r| r.inputs_complete) {
                let mut worst = 0f64;
                let mut n_found = 0usize;
                let sc = scale_of(&tr);
                for c in &ranked {
                    let pos = rec.lists[0].iter().position(|e| e.to_src() == Some(c.src));
                    atlas_rank.push((c.src, pos));
                    if let Some(pi) = pos {
                        n_found += 1;
                        let e = rec.lists[0][pi];
                        let ops = path(m, c.src, ReadPoint::MlpIn(l), true).unwrap();
                        let act = match c.src {
                            Src::Neuron(k, j) => tr.focus_act[k as usize][j as usize] as f64,
                            Src::PleDim(k, j) => tr.focus_ple_gated[k as usize].as_ref().map_or(0.0, |x| x.0[j as usize] as f64),
                            _ => 1.0,
                        };
                        let f = rt.s * runtime_factor(&ops, &sc) * act;
                        let (sp, sq) = (f * e.p as f64, f * e.q as f64);
                        worst = worst.max(rel((sp - c.p).abs() + (sq - c.q).abs(), c.p.abs() + c.q.abs()));
                    }
                }
                rt.checks.push(Check {
                    what: format!("atlas arrow × runtime scales × source activation = runtime contribution ({n_found} of the {} strongest sources are among the atlas top {}; only those are compared)", ranked.len(), at.k),
                    dev: worst,
                    tol: 1e-3,
                });
                let span = (rt.rep.a_max - rt.rep.a_min).abs().max(1e-12);
                let g = rt.rep.gram;
                let gdev = (0..3).map(|j| (rec.gram[j] as f64 - g[j]).abs()).fold(0f64, f64::max) / g[0].abs().max(g[2].abs()).max(1e-30);
                let cdev = ((rec.c[0] as f64 - rt.rep.c[0]).abs() + (rec.c[1] as f64 - rt.rep.c[1]).abs()) / (rt.rep.radius * g[0].max(g[2]).sqrt()).max(1e-30);
                let adev = ((rec.a_max as f64 - rt.rep.a_max).abs() + (rec.a_min as f64 - rt.rep.a_min).abs()) / span;
                rt.checks.push(Check { what: format!("atlas repertoire (center, Gram, a_max, a_min) = recomputed{}", if rec.outputs_complete { "" } else { "; the atlas outputs of this neuron are incomplete (later layers not built)" }), dev: gdev.max(cdev).max(adev), tol: 1e-4 });
            } else {
                eprintln!("trace: the atlas has no complete record for S{l} #{i} (layer not built yet)");
            }
        }
        *zoomer.stats.borrow_mut() = ZoomStats::default();
        let tree = zoomer.neuron(focus, l, i, o.depth.max(1))?;
        let zs = zoomer.stats.borrow();
        if zs.contents > 0 {
            rt.checks.push(Check { what: format!("zoom: Σ sources at the read position + value offset = atom ({} contents of read positions decomposed)", zs.contents), dev: zs.dev, tol: 1e-4 });
            rt.checks.push(Check { what: "zoom: Σ source vectors at the read position = stream from the checkpoint".into(), dev: zs.xdev, tol: 1e-5 });
            rt.checks.push(Check { what: "zoom: pass with focus at the read position = main pass at that position (|Δ|)".into(), dev: zs.pass_dev, tol: 0.0 });
        }
        drop(zs);
        let mut l01 = Vec::new();
        for c in &ranked {
            let v = source_vector(m, &tr, ids, focus, l, c.src).unwrap();
            let (top_p, rest_p) = top_abs(&level01(m, &rt, &v, true), o.scalars);
            let (top_q, rest_q) = if rt.rd.gated { top_abs(&level01(m, &rt, &v, false), o.scalars) } else { (Vec::new(), 0.0) };
            l01.push(L01 { src: c.src, p: c.p, q: c.q, top_p, rest_p, top_q, rest_q });
        }
        // level 0×1 check: Σ_j W[i,j]·γ_j·s·x_c,j = p_c
        let dev = l01.iter().map(|x| rel((x.top_p.iter().map(|y| y.1).sum::<f64>() + x.rest_p - x.p).abs(), x.p.abs().max(1e-12))).fold(0f64, f64::max);
        rt.checks.push(Check { what: "Σ scalar products of a source = its contribution p_c (level 0×1)".into(), dev, tol: 1e-4 });
        let asked = o.scalars_of.as_ref().map(|txt| -> Result<L01, String> {
            let src = parse_src(m, ids, focus, txt)?;
            let v = source_vector(m, &tr, ids, focus, l, src).ok_or(format!("{} does not reach the input of S{l} #{i}", src.label(m)))?;
            let p = rt.s * dot64(&rt.rd.g, &v);
            let q = if rt.rd.gated { rt.s * dot64(&rt.rd.u, &v) } else { 0.0 };
            let (top_p, rest_p) = top_abs(&level01(m, &rt, &v, true), o.scalars);
            let (top_q, rest_q) = if rt.rd.gated { top_abs(&level01(m, &rt, &v, false), o.scalars) } else { (Vec::new(), 0.0) };
            Ok(L01 { src, p, q, top_p, rest_p, top_q, rest_q })
        });
        traced.push(Traced { rt, direct, total, removals, atlas_rank, tree, l01, asked });
    }
    if let Some(path) = &o.result_json {
        use serde_json::json;
        let mut result = crate::export::result(m, ids, focus, y0, "trace", &top_probs(&tr.logits, 10));
        let components: Vec<_> = traced.iter().enumerate().map(|(index, t)| {
            let mut sources: Vec<_> = t.rt.parts.iter().collect();
            sources.sort_by(|a,b| t.rt.a_of(b).abs().total_cmp(&t.rt.a_of(a).abs()));
            let remaining = sources.iter().skip(o.top).map(|c| t.rt.a_of(c)).sum::<f64>();
            let mut children: Vec<_> = sources.iter().take(o.top).enumerate().map(|(j,c)| json!({
                "id":format!("trace-{index}-source-{j}"),"label":src_txt(m,ids,&c.src),
                "layer":null,"kind":"source","direct":null,"total":null,"children":[],
                "activation_contribution":t.rt.a_of(c),"p":c.p,"q":c.q,"source":src_json(&c.src)
            })).collect();
            if sources.len() > o.top {
                children.push(json!({"id":format!("trace-{index}-rest"),"label":"Remaining sources",
                    "layer":null,"kind":"remainder","direct":null,"total":null,"children":[],
                    "activation_contribution":remaining,"count":sources.len()-o.top}));
            }
            let rt = &t.rt;
            let src = |c: &Src| { let mut v = src_json(c); v["label"] = json!(src_txt(m, ids, c)); v };
            let scalars = |x: &L01| json!({"source":src(&x.src),"p":x.p,"q":x.q,
                "top_p":x.top_p.iter().map(|&(j, v)| json!({"dimension":j,"value":v})).collect::<Vec<_>>(),"rest_p":x.rest_p,
                "top_q":x.top_q.iter().map(|&(j, v)| json!({"dimension":j,"value":v})).collect::<Vec<_>>(),"rest_q":x.rest_q});
            let mut atoms: Vec<&Contrib> = rt.atoms.iter().collect();
            atoms.sort_by(|a, b| rt.a_of(b).abs().total_cmp(&rt.a_of(a).abs()));
            json!({"id":format!("trace-{index}"),"label":format!("S{} #{}",t.rt.rd.l,t.rt.rd.i),
                "layer":t.rt.rd.l,"unit":t.rt.rd.i,"kind":"neuron","direct":t.direct,
                "total":t.total.dlo,"kl":t.total.kl,"children":children,"activation":t.rt.a,
                "output_without":[{"id":t.total.top.0,"text":m.tok.piece(t.total.top.0),"probability":t.total.top.1}],
                "p":rt.p,"q":rt.q,"phi":rt.phi,"gated":rt.rd.gated,"silent_neurons":rt.n_silent,
                "activation_range":[rt.rep.a_min,rt.rep.a_max],
                "source_count":rt.parts.len(),
                "groups":groups(rt).iter().map(|g| json!({"group":g.0,"p":g.1,"q":g.2,"activation_contribution":g.3,"count":g.4})).collect::<Vec<_>>(),
                "read_positions":atoms.iter().take(o.top.max(12)).map(|c| { let mut v = src(&c.src); v["activation_contribution"] = json!(rt.a_of(c)); v }).collect::<Vec<_>>(),
                "tree":tree_json(m, ids, &t.tree, None),
                "removals":t.removals.iter().map(|r| json!({"source":src(&r.src),"activation_contribution":r.a_c,
                    "delta_linear":r.da_lin,"delta_frozen_scale":r.da_frozen,"delta_exact":r.da_real,"activation_without":r.a_new,
                    "total":r.eff.map(|e| e.dlo),"kl":r.eff.map(|e| e.kl),
                    "output_without":r.eff.map(|e| json!([{"id":e.top.0,"text":m.tok.piece(e.top.0),"probability":e.top.1}]))})).collect::<Vec<_>>(),
                "scalars":t.l01.iter().map(|x| scalars(x)).collect::<Vec<_>>(),
                "scalars_asked":t.asked.as_ref().and_then(|x| x.as_ref().ok()).map(|x| scalars(x)),
                "pairs":if rt.rd.gated { top_pairs(rt, 12).iter().map(|(a, b, v)| json!({"p_source":src(a),"q_source":src(b),"value":v})).collect::<Vec<_>>() } else { Vec::new() },
                "atlas_rank":t.atlas_rank.iter().map(|(c, r)| json!({"source":src(c),"rank":r})).collect::<Vec<_>>()})
        }).collect();
        result["components"] = json!(components);
        result["checks"] = json!(traced.iter().flat_map(|t| t.rt.checks.iter().map(|c| json!({
            "label":c.what,"deviation":c.dev,"tolerance":c.tol,"passed":c.ok()
        }))).collect::<Vec<_>>());
        result["report"] = json!(o.html.file_name().map(|x| x.to_string_lossy().to_string()));
        crate::export::write(path,&result)?;
    }
    for t in &traced {
        terminal(m, ids, focus, y0, t, o, st);
    }
    let html = html(m, ids, focus, y0, &traced, &auto_info, o);
    std::fs::write(&o.html, html).map_err(|e| format!("{}: {e}", o.html.display()))?;
    println!();
    println!("  {} {}", st.bold("Planes, sources → scalars, pairs, reruns:"), o.html.display());
    if let (Some(p), Some(t)) = (&o.dump, traced.first()) {
        let mut j = dump_json(m, ids, focus, &t.rt);
        // phase 2 for the reference test: the strongest atom's read position, split into its sources
        let rt = &t.rt;
        // the strongest atom from layer 1 on (there the read position carries heads and neurons, not only the embedding)
        let later: Vec<&Contrib> = rt.atoms.iter().filter(|c| matches!(c.src, Src::Atom(k, _, _) if k >= 1)).collect();
        let pool: Vec<&Contrib> = if later.is_empty() { rt.atoms.iter().collect() } else { later };
        if let Some(c) = pool.into_iter().max_by(|a, b| rt.a_of(a).abs().partial_cmp(&rt.a_of(b).abs()).unwrap_or(std::cmp::Ordering::Equal)) {
            if let Src::Atom(k, h, jj) = c.src {
                let reader = Reader { pos: focus, rp: ReadPoint::MlpIn(rt.rd.l), rg: rt.rd.g.clone(), ru: if rt.rd.gated { Some(rt.rd.u.clone()) } else { None }, scale: rt.s };
                let tj = zoomer.trace_at(jj as usize);
                let ct = content(m, &tr, &tj, &ck, ids, &reader, k as usize, h as usize, jj as usize)?;
                j["content"] = serde_json::json!({
                    "layer": k, "head": h, "pos": jj, "kv_layer": ct.kv_layer, "atom": [ct.atom.0, ct.atom.1], "offset": [ct.offset.0, ct.offset.1],
                    "sources": ct.src.parts.iter().map(|c| contrib_json(c)).collect::<Vec<_>>(),
                });
            }
        }
        std::fs::write(p, j.to_string()).map_err(|e| format!("{}: {e}", p.display()))?;
        println!("  {} {}", st.bold("Neuron dump (JSON):"), p.display());
    }
    eprintln!("trace: {} s", num(t0.elapsed().as_secs_f32(), 1));
    Ok(())
}

// ====================================================================== terminal

fn src_txt(m: &Model, ids: &[u32], c: &Src) -> String {
    match *c {
        Src::Atom(k, h, j) => format!("S{k} head {h} reads [{j}]{}", tok(m, ids[j as usize])),
        _ => c.label(m),
    }
}

fn check_line(c: &Check, st: &Style) -> String {
    let txt = format!("{} (dev. {}, tol. {})", c.what, sci(c.dev as f32), sci(c.tol as f32));
    if c.ok() { st.dim(&format!("✓ {txt}")) } else { st.red(&format!("✗ VIOLATED: {txt}")) }
}

fn groups(rt: &Runtime) -> Vec<(&'static str, f64, f64, f64, usize)> {
    let mut g: Vec<(&'static str, f64, f64, f64, usize)> = Vec::new();
    for c in &rt.parts {
        let name = c.src.group();
        let a = rt.a_of(c);
        match g.iter_mut().find(|x| x.0 == name) {
            Some(e) => {
                e.1 += c.p;
                e.2 += c.q;
                e.3 += a;
                e.4 += 1;
            }
            None => g.push((name, c.p, c.q, a, 1)),
        }
    }
    g
}

fn ranked_sources(rt: &Runtime) -> Vec<&Contrib> {
    let mut r: Vec<&Contrib> = rt.parts.iter().collect();
    r.sort_by(|a, b| rt.a_of(b).abs().partial_cmp(&rt.a_of(a).abs()).unwrap_or(std::cmp::Ordering::Equal));
    r
}

/// Top pairs φ·p_c·q_c' from the top-n sources by |p| times the top-n by |q| (these contain the
/// global top-n pairs, because the pair table is an outer product).
fn top_pairs(rt: &Runtime, n: usize) -> Vec<(Src, Src, f64)> {
    let mut bp: Vec<&Contrib> = rt.parts.iter().collect();
    bp.sort_by(|a, b| b.p.abs().partial_cmp(&a.p.abs()).unwrap_or(std::cmp::Ordering::Equal));
    let mut bq: Vec<&Contrib> = rt.parts.iter().collect();
    bq.sort_by(|a, b| b.q.abs().partial_cmp(&a.q.abs()).unwrap_or(std::cmp::Ordering::Equal));
    let mut out = Vec::new();
    for a in bp.iter().take(n) {
        for b in bq.iter().take(n) {
            out.push((a.src, b.src, rt.phi * a.p * b.q));
        }
    }
    out.sort_by(|a, b| b.2.abs().partial_cmp(&a.2.abs()).unwrap_or(std::cmp::Ordering::Equal));
    out.truncate(n);
    out
}

fn pos_txt(m: &Model, ids: &[u32], pos: usize) -> String {
    format!("[{pos}]{}", tok(m, ids[pos]))
}

fn print_children(m: &Model, ids: &[u32], here: usize, items: &[Child], indent: &str, st: &Style) {
    for (k, c) in items.iter().enumerate() {
        let last = k + 1 == items.len();
        let (branch, next) = if last { ("└ ", "   ") } else { ("├ ", "│  ") };
        let sub = format!("{indent}{next}");
        match c {
            Child::Neuron { a_c, node } => {
                let at = if node.pos != here { format!(" at {}", pos_txt(m, ids, node.pos)) } else { String::new() };
                println!("{indent}{branch}S{} #{}{at} – a_c {} (its own a = {})", node.l, node.i, num(*a_c as f32, 4), num(node.a as f32, 4));
                print_children(m, ids, node.pos, &node.children, &sub, st);
            }
            Child::Head { src, a_c, atoms } => {
                println!("{indent}{branch}{} – a_c {}, reads:", src.label(m), num(*a_c as f32, 4));
                for (n, av) in atoms.iter().enumerate() {
                    let last2 = n + 1 == atoms.len();
                    let (b2, n2) = if last2 { ("└ ", "   ") } else { ("├ ", "│  ") };
                    match &av.content {
                        Some(cv) => {
                            println!(
                                "{sub}{b2}{} – a_c {} → what {} carries before S{} {}",
                                pos_txt(m, ids, av.j as usize),
                                num(av.a_c as f32, 4),
                                pos_txt(m, ids, av.j as usize),
                                cv.kv,
                                st.dim(&format!("(+ value offset a_c {}; check dev. {})", num(cv.offset_a as f32, 4), sci(cv.dev as f32)))
                            );
                            print_children(m, ids, av.j as usize, &cv.items, &format!("{sub}{n2}"), st);
                        }
                        None => println!("{sub}{b2}{} – a_c {}", pos_txt(m, ids, av.j as usize), num(av.a_c as f32, 4)),
                    }
                }
            }
            Child::Other { src, a_c } => {
                let at = if matches!(src, Src::Token(_) | Src::Position(_)) && here != usize::MAX { format!(" at {}", pos_txt(m, ids, here)) } else { String::new() };
                println!("{indent}{branch}{}{at} – a_c {}", src_txt(m, ids, src), num(*a_c as f32, 4));
            }
        }
    }
}

fn terminal(m: &Model, ids: &[u32], focus: usize, y0: u32, t: &Traced, o: &Opts, st: &Style) {
    let rt = &t.rt;
    let rd = &rt.rd;
    println!();
    println!(
        "{}",
        st.bold(&format!("Trace S{} #{} at [{focus}]{} (target {}) – {}, {}", rd.l, rd.i, tok(m, ids[focus]), tok(m, y0), if rd.gated { "gated" } else { "not gated" }, act_name(rd.act)))
    );
    println!("  {}", st.dim(&format!("weights: {}; pre-norm {} with frozen scale s = {}", crate::inspect::weight_types(m, rd.l), if rd.ln { "LayerNorm" } else { "RMSNorm" }, num(rt.s as f32, 5))));
    let r = rt.rep.r_of(rt.p, rt.q);
    let rel_a = if rt.a >= 0.0 { rt.a / rt.rep.a_max } else { rt.a / rt.rep.a_min };
    println!(
        "  p = {}{} · a = {} (recorded {}) · φ = act(p)/p = {} · a/{} = {} · radius r = {}",
        num(rt.p as f32, 4),
        if rd.gated { format!(" · q = {}", num(rt.q as f32, 4)) } else { String::new() },
        num(rt.a as f32, 5),
        num(rt.a_rec as f32, 5),
        num(rt.phi as f32, 4),
        if rt.a >= 0.0 { "a_max" } else { "a_min" },
        num(rel_a as f32, 4),
        num(r as f32, 4)
    );
    for line in crate::inspect::repertoire_lines(rd, &rt.rep).iter().take(2) {
        println!("  {line}");
    }
    for c in &rt.checks {
        println!("  {}", check_line(c, st));
    }
    println!();
    println!("  {}", st.bold("Level 1 – sources by group (p_c = s·g'·x_c; contribution to a: a_c = φ/2·(p_c·q + p·q_c), Σ a_c = a)"));
    for (g, p, q, a, n) in groups(rt) {
        println!("     {} {:>7}   p {}{}   a {}", pad(g, 10), n, padl(&num(p as f32, 4), 10), if rd.gated { format!("   q {}", padl(&num(q as f32, 4), 10)) } else { String::new() }, padl(&num(a as f32, 5), 11));
    }
    if rt.n_silent > 0 {
        println!("     {}", st.dim(&format!("+ {} earlier neurons with activation exactly 0 (contribute 0)", rt.n_silent)));
    }
    println!("  {}", st.bold(&format!("Strongest sources (by |a_c|, of {}){}:", rt.parts.len(), if t.atlas_rank.is_empty() { "" } else { " – rank among the neuron's static inputs by |p| in the atlas" })));
    for c in ranked_sources(rt).iter().take(o.top.max(10)) {
        let rank = t.atlas_rank.iter().find(|x| x.0 == c.src).map(|x| x.1.map_or("not in the atlas top k".to_string(), |r| format!("static rank {}", r + 1)));
        println!(
            "     {} p {}{}   a {}  {}",
            pad(&trunc(&src_txt(m, ids, &c.src), 34), 34),
            padl(&num(c.p as f32, 4), 10),
            if rd.gated { format!("   q {}", padl(&num(c.q as f32, 4), 10)) } else { String::new() },
            padl(&num(rt.a_of(c) as f32, 5), 11),
            st.dim(&rank.unwrap_or_default())
        );
    }
    println!("  {}", st.bold(&format!("Level 0 – scalar products W[i,j]·x̂_j (top {} of {}):", o.scalars, rd.d)));
    for (gate, name) in [(true, "p − b"), (false, "q − b")] {
        if !gate && !rd.gated {
            continue;
        }
        let (top, rest) = top_abs(&rt.level0(gate), o.scalars);
        let total: f64 = top.iter().map(|x| x.1).sum::<f64>() + rest;
        println!("     {name} = {}: {} · rest {}", num(total as f32, 4), top.iter().map(|(j, v)| format!("[{j}] {}", num(*v as f32, 4))).collect::<Vec<_>>().join(" "), num(rest as f32, 4));
    }
    if let Some(x) = t.l01.first() {
        println!("  {}", st.bold(&format!("Level 0×1 – scalar products W[i,j]·γ_j·s·x_c,j of the strongest source {}:", src_txt(m, ids, &x.src))));
        println!("     p_c = {}: {} · rest {}", num(x.p as f32, 4), x.top_p.iter().map(|(j, v)| format!("[{j}] {}", num(*v as f32, 4))).collect::<Vec<_>>().join(" "), num(x.rest_p as f32, 4));
        if rd.gated {
            println!("     q_c = {}: {} · rest {}", num(x.q as f32, 4), x.top_q.iter().map(|(j, v)| format!("[{j}] {}", num(*v as f32, 4))).collect::<Vec<_>>().join(" "), num(x.rest_q as f32, 4));
        }
    }
    match &t.asked {
        Some(Ok(x)) => {
            println!("  {}", st.bold(&format!("Level 0×1 – requested source {} (Σ_j = its contribution):", src_txt(m, ids, &x.src))));
            println!("     p_c = {}: {} · rest {}", num(x.p as f32, 4), x.top_p.iter().map(|(j, v)| format!("[{j}] {}", num(*v as f32, 4))).collect::<Vec<_>>().join(" "), num(x.rest_p as f32, 4));
            if rd.gated {
                println!("     q_c = {}: {} · rest {}", num(x.q as f32, 4), x.top_q.iter().map(|(j, v)| format!("[{j}] {}", num(*v as f32, 4))).collect::<Vec<_>>().join(" "), num(x.rest_q as f32, 4));
            }
        }
        Some(Err(e)) => println!("  {}", st.yellow(&format!("Level 0×1 – requested source: {e}"))),
        None => {}
    }
    if rd.gated {
        println!("  {}", st.bold("Pairs – a = φ·Σ_c Σ_c' p_c·q_c' (strongest pairs):"));
        for (a, b, v) in top_pairs(rt, o.top.max(5)) {
            println!("     p from {} × q from {}: {}", trunc(&src_txt(m, ids, &a), 30), trunc(&src_txt(m, ids, &b), 30), num(v as f32, 5));
        }
    }
    println!();
    println!("  {}", st.bold(&format!("Removal effects (rerun from S{}, target {})", rd.l, tok(m, y0))));
    println!(
        "     neuron itself to 0: direct {} · total {} · indirect {} · KL {} · top without it {} {}",
        factor(t.direct),
        factor(t.total.dlo as f32),
        factor(t.total.dlo as f32 - t.direct),
        num(t.total.kl as f32, 3),
        tok(m, t.total.top.0),
        pct(t.total.top.1)
    );
    println!("     {}", st.dim("source removed only from this neuron: Δa = a − a' with frozen scale and φ (linear) · frozen scale only · exact (Norm(x − x_c)); effect = odds of the target with vs. without"));
    for r in &t.removals {
        let e = r.eff.unwrap();
        println!(
            "     {} a_c {}  Δa lin {} · frozen {} · exact {}  → odds {} {} · KL {}",
            pad(&trunc(&src_txt(m, ids, &r.src), 30), 30),
            padl(&num(r.a_c as f32, 4), 9),
            padl(&num(r.da_lin as f32, 4), 9),
            padl(&num(r.da_frozen as f32, 4), 9),
            padl(&num(r.da_real as f32, 4), 9),
            tok(m, y0),
            factor(e.dlo as f32),
            num(e.kl as f32, 4)
        );
    }
    println!();
    println!("  {}", st.bold(&format!("Backward zoom (depth {}, the {} strongest sources per neuron):", o.depth, o.top)));
    println!("  S{} #{} at {} (a = {})", t.tree.l, t.tree.i, pos_txt(m, ids, t.tree.pos), num(t.tree.a as f32, 4));
    print_children(m, ids, t.tree.pos, &t.tree.children, "  ", st);
}

// ====================================================================== HTML

fn children_html(m: &Model, ids: &[u32], here: usize, items: &[Child]) -> String {
    let pos_html = |p: usize| format!("[{p}]{}", chip(m, ids[p]));
    let mut h = String::from("<ul class=\"tree\">");
    for c in items {
        match c {
            Child::Neuron { a_c, node } => {
                let at = if node.pos != here { format!(" at {}", pos_html(node.pos)) } else { String::new() };
                let head = format!("S{} #{}{at} · a_c {} · its own a = {}", node.l, node.i, num(*a_c as f32, 4), num(node.a as f32, 4));
                if node.children.is_empty() {
                    h += &format!("<li>{head}</li>");
                } else {
                    h += &format!("<li><details><summary>{head}</summary>{}</details></li>", children_html(m, ids, node.pos, &node.children));
                }
            }
            Child::Head { src, a_c, atoms } => {
                h += &format!("<li>{} · a_c {}, reads:<ul class=\"tree\">", esc_html(&src.label(m)), num(*a_c as f32, 4));
                for av in atoms {
                    match &av.content {
                        Some(cv) => {
                            h += &format!(
                                "<li><details><summary>{} · a_c {} → what it carries before S{} <span class=\"mu\">(value offset a_c {}; check dev. {})</span></summary>{}</details></li>",
                                pos_html(av.j as usize),
                                num(av.a_c as f32, 4),
                                cv.kv,
                                num(cv.offset_a as f32, 4),
                                sci(cv.dev as f32),
                                children_html(m, ids, av.j as usize, &cv.items)
                            );
                        }
                        None => h += &format!("<li>{} · a_c {}</li>", pos_html(av.j as usize), num(av.a_c as f32, 4)),
                    }
                }
                h += "</ul></li>";
            }
            Child::Other { src, a_c } => {
                let at = if matches!(src, Src::Token(_) | Src::Position(_)) { format!(" at {}", pos_html(here)) } else { String::new() };
                h += &format!("<li>{}{at} · a_c {}</li>", esc_html(&src_txt(m, ids, src)), num(*a_c as f32, 4));
            }
        }
    }
    h + "</ul>"
}

fn html(m: &Model, ids: &[u32], focus: usize, y0: u32, traced: &[Traced], auto_info: &[(usize, usize, f32)], o: &Opts) -> String {
    let mut h = format!(
        r#"<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Neuron trace</title><style>{CSS}{}{TRACE_CSS}</style></head><body><main>
<h1>Neuron trace</h1><p class="meta">{} · focus token [{focus}] · target {}</p><p class="prompt">{}</p>"#,
        crate::inspect::PLANE_CSS,
        esc_html(&m.name),
        chip(m, y0),
        prompt_html(m, ids, focus)
    );
    if !auto_info.is_empty() {
        h += &format!(
            "<p>Start neurons (auto: strongest direct voice on {}): {}</p>",
            chip(m, y0),
            auto_info.iter().map(|(l, i, v)| format!("S{l} #{i} ({})", esc_html(&factor(*v)))).collect::<Vec<_>>().join(" · ")
        );
    }
    for t in traced {
        let rt = &t.rt;
        let rd = &rt.rd;
        h += &format!(
            "<h2>S{} #{} – {}, {}</h2><p class=\"meta\">{} · pre-norm {} · s = {}</p>",
            rd.l,
            rd.i,
            if rd.gated { "gated" } else { "not gated" },
            act_name(rd.act),
            esc_html(&crate::inspect::weight_types(m, rd.l)),
            if rd.ln { "LayerNorm" } else { "RMSNorm" },
            num(rt.s as f32, 5)
        );
        let r = rt.rep.r_of(rt.p, rt.q);
        h += &format!(
            "<p>p = {}{} · a = {} (recorded {}) · φ = {} · repertoire a ∈ [{}, {}] · radius r = {}</p>",
            num(rt.p as f32, 4),
            if rd.gated { format!(" · q = {}", num(rt.q as f32, 4)) } else { String::new() },
            num(rt.a as f32, 5),
            num(rt.a_rec as f32, 5),
            num(rt.phi as f32, 4),
            num(rt.rep.a_min as f32, 4),
            num(rt.rep.a_max as f32, 4),
            num(r as f32, 4)
        );
        // arrows tip to tail: the strongest sources by |p_c| + |q_c|, the rest as one arrow
        let mut by: Vec<&Contrib> = rt.parts.iter().filter(|c| c.src != Src::Offset).collect();
        by.sort_by(|a, b| (b.p.abs() + b.q.abs()).partial_cmp(&(a.p.abs() + a.q.abs())).unwrap_or(std::cmp::Ordering::Equal));
        let mut arrows: Vec<(String, f64, f64)> = by.iter().take(6).map(|c| (src_txt(m, ids, &c.src), c.p, c.q)).collect();
        let (sp, sq) = arrows.iter().fold((0.0, 0.0), |a, x| (a.0 + x.1, a.1 + x.2));
        arrows.push((format!("other {} sources", by.len().saturating_sub(6)), rt.p - rt.rd.cg - sp, rt.q - rt.rd.cu - sq));
        let pl = crate::inspect::Plane { rd, rep: &rt.rep, point: Some((rt.p, rt.q)), arrows: arrows.clone() };
        h += "<div class=\"planes\">";
        h += &crate::inspect::plane_svg(&pl, false);
        h += &crate::inspect::plane_svg(&pl, true);
        h += "</div>";
        h += &format!("<p class=\"mu\">Left: the whole repertoire, right: around the real point. Arrows from the offset c, tip to tail: {}</p>", crate::inspect::arrow_legend(&arrows));
        h += "<p class=\"check\">";
        for c in &rt.checks {
            h += &format!("{} {} (dev. {}, tol. {})<br>", if c.ok() { "✓" } else { "✗ VIOLATED:" }, esc_html(&c.what), sci(c.dev as f32), sci(c.tol as f32));
        }
        h += "</p>";
        // groups
        h += "<h4>Sources by group</h4><table class=\"tbl\"><tr><th>group</th><th>sources</th><th>p</th><th>q</th><th>a</th></tr>";
        for (g, p, q, a, n) in groups(rt) {
            h += &format!("<tr><td>{g}</td><td class=\"num\">{n}</td><td class=\"num\">{}</td><td class=\"num\">{}</td><td class=\"num\">{}</td></tr>", num(p as f32, 4), num(q as f32, 4), num(a as f32, 5));
        }
        h += "</table>";
        // sources -> scalars
        h += "<h4>Strongest sources → their scalar products</h4>";
        for x in &t.l01 {
            let a_c = rt.a_of(&Contrib { src: x.src, p: x.p, q: x.q });
            h += &format!(
                "<details class=\"info\"><summary>{} · p_c {} · q_c {} · a_c {}</summary><p class=\"mu\">p_c = Σ_j W[i,j]·γ_j·s·x_c,j: {} · rest {}</p>{}</details>",
                esc_html(&src_txt(m, ids, &x.src)),
                num(x.p as f32, 4),
                num(x.q as f32, 4),
                num(a_c as f32, 5),
                x.top_p.iter().map(|(j, v)| format!("[{j}] {}", num(*v as f32, 4))).collect::<Vec<_>>().join(" · "),
                num(x.rest_p as f32, 4),
                if rd.gated { format!("<p class=\"mu\">q_c: {} · rest {}</p>", x.top_q.iter().map(|(j, v)| format!("[{j}] {}", num(*v as f32, 4))).collect::<Vec<_>>().join(" · "), num(x.rest_q as f32, 4)) } else { String::new() }
            );
        }
        if let Some(Ok(x)) = &t.asked {
            h += &format!(
                "<details class=\"info\" open><summary>Requested source {} · p_c {} · q_c {}</summary><p class=\"mu\">p_c = Σ_j W[i,j]·γ_j·s·x_c,j: {} · rest {}</p>{}</details>",
                esc_html(&src_txt(m, ids, &x.src)),
                num(x.p as f32, 4),
                num(x.q as f32, 4),
                x.top_p.iter().map(|(j, v)| format!("[{j}] {}", num(*v as f32, 4))).collect::<Vec<_>>().join(" · "),
                num(x.rest_p as f32, 4),
                if rd.gated { format!("<p class=\"mu\">q_c: {} · rest {}</p>", x.top_q.iter().map(|(j, v)| format!("[{j}] {}", num(*v as f32, 4))).collect::<Vec<_>>().join(" · "), num(x.rest_q as f32, 4)) } else { String::new() }
            );
        }
        h += "<details class=\"info\"><summary>All sources by |a_c| (top 200)</summary><table class=\"tbl\"><tr><th>source</th><th>p_c</th><th>q_c</th><th>a_c</th></tr>";
        for c in ranked_sources(rt).iter().take(200) {
            h += &format!("<tr><td>{}</td><td class=\"num\">{}</td><td class=\"num\">{}</td><td class=\"num\">{}</td></tr>", esc_html(&src_txt(m, ids, &c.src)), num(c.p as f32, 4), num(c.q as f32, 4), num(rt.a_of(c) as f32, 5));
        }
        h += "</table></details>";
        // level 0
        h += "<details class=\"info\"><summary>Level 0 – all scalar products W[i,j]·x̂_j</summary>";
        for (gate, name) in [(true, "p − b"), (false, "q − b")] {
            if !gate && !rd.gated {
                continue;
            }
            let (top, rest) = top_abs(&rt.level0(gate), 64);
            h += &format!("<p><b>{name}</b> (top 64 of {}, rest {}): {}</p>", rd.d, num(rest as f32, 4), top.iter().map(|(j, v)| format!("[{j}] {}", num(*v as f32, 4))).collect::<Vec<_>>().join(" · "));
        }
        h += "</details>";
        if rd.gated {
            h += "<details class=\"info\"><summary>Pairs φ·p_c·q_c'</summary><table class=\"tbl\"><tr><th>p from</th><th>q from</th><th>φ·p_c·q_c'</th></tr>";
            for (a, b, v) in top_pairs(rt, 30) {
                h += &format!("<tr><td>{}</td><td>{}</td><td class=\"num\">{}</td></tr>", esc_html(&src_txt(m, ids, &a)), esc_html(&src_txt(m, ids, &b)), num(v as f32, 5));
            }
            h += "</table></details>";
        }
        // reruns
        h += &format!(
            "<h4>Removal effects</h4><p>Neuron to 0: direct {} · total {} · indirect {} · KL {} · top without it {} {}</p>",
            esc_html(&factor(t.direct)),
            esc_html(&factor(t.total.dlo as f32)),
            esc_html(&factor(t.total.dlo as f32 - t.direct)),
            num(t.total.kl as f32, 3),
            chip(m, t.total.top.0),
            esc_html(&pct(t.total.top.1))
        );
        h += "<table class=\"tbl\"><tr><th>source removed from this neuron only</th><th>a_c</th><th>Δa linear</th><th>Δa frozen scale</th><th>Δa exact</th><th>odds of the target with vs. without</th><th>KL</th><th>top</th></tr>";
        for r in &t.removals {
            let e = r.eff.unwrap();
            h += &format!(
                "<tr><td>{}</td><td class=\"num\">{}</td><td class=\"num\">{}</td><td class=\"num\">{}</td><td class=\"num\">{}</td><td class=\"num\">{}</td><td class=\"num\">{}</td><td>{} {}</td></tr>",
                esc_html(&src_txt(m, ids, &r.src)),
                num(r.a_c as f32, 4),
                num(r.da_lin as f32, 4),
                num(r.da_frozen as f32, 4),
                num(r.da_real as f32, 4),
                factor_html(e.dlo as f32),
                num(e.kl as f32, 4),
                chip(m, e.top.0),
                esc_html(&pct(e.top.1))
            );
        }
        h += "</table>";
        h += &format!(
            "<h4>Backward zoom</h4><p class=\"mu\">a_c in units of the activation of the neuron above; inside a head: what the read position carries, split into the sources at that position.</p><p>S{} #{} (a = {})</p>{}",
            t.tree.l,
            t.tree.i,
            num(t.tree.a as f32, 4),
            children_html(m, ids, t.tree.pos, &t.tree.children)
        );
    }
    h += &format!("<p class=\"mu foot\">apollon · trace · depth {} · top {}</p></main></body></html>\n", o.depth, o.top);
    h
}

const TRACE_CSS: &str = r#"
.tree{list-style:none;padding-left:18px;border-left:1px solid var(--line);margin:4px 0}
.tree li{margin:3px 0}
"#;

// ====================================================================== dump for the reference tests

pub fn src_json(c: &Src) -> serde_json::Value {
    use serde_json::json;
    match *c {
        Src::Token(t) => json!({"kind": "token", "index": t}),
        Src::Position(r) => json!({"kind": "position", "index": r}),
        Src::Head(k, h) => json!({"kind": "head", "layer": k, "head": h}),
        Src::Atom(k, h, j) => json!({"kind": "atom", "layer": k, "head": h, "pos": j}),
        Src::AttnBias(k) => json!({"kind": "attn_bias", "layer": k}),
        Src::AttnPostBias(k) => json!({"kind": "attn_post_bias", "layer": k}),
        Src::Neuron(k, i) => json!({"kind": "neuron", "layer": k, "index": i}),
        Src::MlpBias(k) => json!({"kind": "mlp_bias", "layer": k}),
        Src::MlpPostBias(k) => json!({"kind": "mlp_post_bias", "layer": k}),
        Src::PleDim(k, e) => json!({"kind": "ple", "layer": k, "index": e}),
        Src::Offset => json!({"kind": "offset"}),
        _ => json!({"kind": "other"}),
    }
}

/// The backward tree of one neuron: earlier neurons, heads with their read positions and what those
/// positions carried. Every node holds its contribution a_c to the activation of its parent.
fn tree_json(m: &Model, ids: &[u32], n: &Node, a_c: Option<f64>) -> serde_json::Value {
    json!({"kind":"neuron","label":format!("S{} #{}", n.l, n.i),"layer":n.l,"unit":n.i,"position":n.pos,
        "activation":n.a,"activation_contribution":a_c,"children":children_json(m, ids, n.pos, &n.children)})
}

fn children_json(m: &Model, ids: &[u32], here: usize, items: &[Child]) -> Vec<serde_json::Value> {
    items.iter().map(|c| match c {
        Child::Neuron { a_c, node } => tree_json(m, ids, node, Some(*a_c)),
        Child::Head { src, a_c, atoms } => {
            let mut v = src_json(src);
            v["label"] = json!(src.label(m));
            v["position"] = json!(here);
            v["activation_contribution"] = json!(a_c);
            v["children"] = json!(atoms.iter().map(|av| {
                let j = av.j as usize;
                let mut a = json!({"kind":"read","label":pos_txt(m, ids, j),"position":j,"activation_contribution":av.a_c,"children":[]});
                if let Some(cv) = &av.content {
                    a["before_layer"] = json!(cv.kv);
                    a["value_offset"] = json!(cv.offset_a);
                    a["children"] = json!(children_json(m, ids, j, &cv.items));
                }
                a
            }).collect::<Vec<_>>());
            v
        }
        Child::Other { src, a_c } => {
            let mut v = src_json(src);
            v["label"] = json!(src_txt(m, ids, src));
            v["position"] = json!(if here == usize::MAX { None } else { Some(here) });
            v["activation_contribution"] = json!(a_c);
            v["children"] = json!([]);
            v
        }
    }).collect()
}

fn contrib_json(c: &Contrib) -> serde_json::Value {
    let mut v = src_json(&c.src);
    v["p"] = serde_json::json!(c.p);
    v["q"] = serde_json::json!(c.q);
    v
}

fn dump_json(m: &Model, ids: &[u32], focus: usize, rt: &Runtime) -> serde_json::Value {
    use serde_json::json;
    let rep = &rt.rep;
    json!({
        "model": m.name, "ids": ids, "focus": focus, "layer": rt.rd.l, "index": rt.rd.i,
        "gated": rt.rd.gated, "act": act_name(rt.rd.act), "ln": rt.rd.ln, "d": rt.rd.d,
        "s": rt.s, "p": rt.p, "q": rt.q, "a": rt.a, "a_recorded": rt.a_rec, "phi": rt.phi,
        "bg": rt.rd.bg, "bu": rt.rd.bu, "cg": rt.rd.cg, "cu": rt.rd.cu,
        "xhat": rt.xhat, "x": rt.x,
        "scalars_p": rt.level0(true).iter().map(|x| x.1).collect::<Vec<_>>(),
        "scalars_q": if rt.rd.gated { rt.level0(false).iter().map(|x| x.1).collect::<Vec<_>>() } else { vec![] },
        "ellipse": {"c": rep.c, "gram": rep.gram, "radius": rep.radius, "rank": rep.rank, "a_max": rep.a_max, "a_min": rep.a_min, "th_max": rep.th_max, "th_min": rep.th_min, "pt_max": rep.pt_max, "pt_min": rep.pt_min, "z_max": rep.z_at(&rt.rd, rep.th_max)},
        "sources": rt.parts.iter().map(contrib_json).collect::<Vec<_>>(),
        "atoms": rt.atoms.iter().map(contrib_json).collect::<Vec<_>>(),
        "weights": crate::inspect::weight_types(m, rt.rd.l),
    })
}
