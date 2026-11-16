//! Probe: what exactly do the single units of one component do?
//!
//! For one component (the MLP of a layer, the per-layer embedding, the attention of a layer) every unit
//! inside it – neuron, PLE dimension, head – is changed by brute force and the model is rerun from that
//! layer. Many variants run in ONE pass: every variant is a copy of the positions from the changed spot
//! on, and sees, via attention, the clean prefix and itself (model::Variants). Measured is the output
//! (odds of the target token, KL, new top token) and, for the strongest units, everything in between:
//! logit lens after every layer, the voices (partial softmax) of every later attention/MLP/PLE, which
//! later neurons change and where heads look differently.
//!
//! Stages: 1. Where does the component act (whole component zeroed, per position)? 2. Brute force: every
//! unit zeroed on its own at the most important positions. 3. Interplay: the k strongest off together,
//! or only they on. 4. Dose: activation ×(−2 … 4). 5. Path through the later layers. 6. Contrast
//! prompt: insert the values from another prompt.
use crate::blas;
use crate::fmt::{chip, esc_html, factor, factor_html, num, pad, padl, pct, prompt_html, sci, tok, Style, CSS};
use crate::model::{top_probs, Ablation, Checkpoint, Model, Res, RowRecord, RunOpts, Start, Variants};
use crate::readout::{columns, log_odds, softmax, sum, top_insert, voices, Voice};
use crate::tensor::dot;
use rayon::prelude::*;
use std::collections::HashSet;
use std::io::IsTerminal;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

pub(crate) const NY: usize = 2; // target token, rival
const DOSE: [f32; 6] = [-2.0, -1.0, 0.0, 0.5, 2.0, 4.0];
const ALL: usize = usize::MAX; // unit at all positions at once
/// rows per variant pass
const BUDGET: usize = 768;

fn pos_label(q: usize) -> String {
    if q == ALL { "[all]".into() } else { format!("[{q}]") }
}

pub struct Opts {
    pub part: String,
    pub pos: Option<usize>,
    pub html: PathBuf,
    pub n_deep: usize,
    /// distributed mode, MLP: at most this many neurons (preselected by contribution size)
    pub max: usize,
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Mlp,
    Ple,
    Attn,
}

/// "S12 MLP", "S11 PLE", "S12 Attn" -> (layer, kind)
fn parse_part(s: &str) -> Result<(usize, Kind), String> {
    match crate::neuron::parse_part(s).map_err(|e| format!("--probe {e}"))? {
        crate::neuron::PartSpec::Mlp(l) => Ok((l, Kind::Mlp)),
        crate::neuron::PartSpec::Ple(l) => Ok((l, Kind::Ple)),
        crate::neuron::PartSpec::Attn(l) => Ok((l, Kind::Attn)),
        crate::neuron::PartSpec::Neuron(..) => Err(format!("--probe '{s}': a single neuron is examined with --trace \"{s}\" (or statically with --neuron)")),
    }
}

#[derive(Clone)]
enum Edit {
    Unit { p: usize, u: usize, f: f32 }, // unit u at position p times f
    Value { p: usize, u: usize, w: f32 }, // set the unit to a value (from the contrast prompt)
    Whole { p: usize },                   // whole component at position p to zero
}

#[derive(Clone)]
struct Variant {
    start: usize,
    edits: Vec<Edit>,
}

impl Variant {
    fn new(edits: Vec<Edit>) -> Self {
        let start = edits
            .iter()
            .map(|e| match e {
                Edit::Unit { p, .. } | Edit::Value { p, .. } | Edit::Whole { p } => *p,
            })
            .min()
            .unwrap_or(usize::MAX);
        Variant { start, edits }
    }
}

/// Output of one variant
#[derive(Clone, Copy, Default)]
pub(crate) struct Out {
    pub lo: [f32; NY], // log-odds
    pub top: [(u32, f32); 3],
    pub kl: f32, // KL(real ‖ variant)
}

/// Result of one variant: its output and that of the unchanged copy in the same pass
#[derive(Clone, Default)]
struct Outcome {
    out: Out,
    id: Out,
    deep: Option<Vec<DeepLayer>>,
}

impl Outcome {
    fn dlo(&self, i: usize) -> f32 {
        self.out.lo[i] - self.id.lo[i]
    }
}

/// Recording of a variant at the focus in a later layer
#[derive(Clone)]
struct DeepLayer {
    layer: usize,
    x: Vec<f32>,
    attn: Vec<f32>,
    mlp: Vec<f32>,
    ple: Vec<f32>,
    act: Vec<f32>,
    probs: Vec<f32>, // [H][T] by positions
}

// ------------------------------------------------------------------ readout (final norm -> output matrix -> softmax)

#[derive(Clone, Copy)]
struct Acc {
    m: f32,
    a: f64,
    ax: [f64; NY],
    ly: [f32; NY],
    top: [(f32, u32); 3],
    pl: f64,
}

/// For every row of xs (stream at the end): log-odds of the targets, top 3, KL to the real output p_ref.
pub(crate) fn read_out(m: &Model, xs: &[f32], ys: &[u32; NY], p_ref: &[f32], ref_plp: f64) -> Vec<Out> {
    read_out_fz(m, xs, ys, p_ref, ref_plp, None)
}

/// read_out with the scale of the final norm given for every row (frozen norms)
pub(crate) fn read_out_fz(m: &Model, xs: &[f32], ys: &[u32; NY], p_ref: &[f32], ref_plp: f64, fin: Option<f32>) -> Vec<Out> {
    let d = m.d;
    let n = xs.len() / d;
    if n == 0 {
        return Vec::new();
    }
    let h: Vec<f32> = match &m.final_norm {
        Some(nm) => {
            let mut h = vec![0f32; xs.len()];
            h.par_chunks_mut(d).zip(xs.par_chunks(d)).for_each(|(o, x)| {
                nm.apply_fz(x, o, fin);
            });
            h
        }
        None => xs.to_vec(),
    };
    let lm = m.lm_head();
    let v = lm.rows;
    let vb = ((16usize << 20) / n).clamp(256, 8192) / 64 * 64;
    let mut c = vec![0f32; n * vb];
    let mut st = vec![Acc { m: f32::NEG_INFINITY, a: 0.0, ax: [0.0; NY], ly: [0.0; NY], top: [(f32::NEG_INFINITY, 0); 3], pl: 0.0 }; n];
    let cap = m.logit_cap;
    let psum: f64 = p_ref.iter().map(|x| *x as f64).sum();
    let mut v0 = 0;
    while v0 < v {
        let nb = vb.min(v - v0);
        c[..n * nb].copy_from_slice(&lm.rows_times(v0, nb, &h, n));
        let pb = &p_ref[v0..v0 + nb];
        c[..n * nb].par_chunks_mut(nb).zip(st.par_iter_mut()).for_each_init(
            || (vec![0f32; vb], vec![0f32; vb]),
            |(t1, t2), (row, s)| {
                let (t1, t2) = (&mut t1[..nb], &mut t2[..nb]);
                if let Some(cp) = cap {
                    for (o, x) in t1.iter_mut().zip(row.iter()) {
                        *o = x / cp;
                    }
                    blas::tanh(t2, t1);
                    for (x, y) in row.iter_mut().zip(t2.iter()) {
                        *x = y * cp;
                    }
                }
                for (i, &y) in ys.iter().enumerate() {
                    let y = y as usize;
                    if y >= v0 && y < v0 + nb {
                        s.ly[i] = row[y - v0];
                    }
                }
                top_insert(&mut s.top, row, v0);
                let bm = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let mn = s.m.max(bm);
                for (o, x) in t1.iter_mut().zip(row.iter()) {
                    *o = x - mn;
                }
                blas::exp(t2, t1);
                let sa = sum(t2) as f64;
                let r = ((s.m - mn) as f64).exp();
                s.a = s.a * r + sa;
                for (i, &y) in ys.iter().enumerate() {
                    let y = y as usize;
                    let sx = if y >= v0 && y < v0 + nb {
                        let j = y - v0;
                        (sum(&t2[..j]) + sum(&t2[j + 1..])) as f64
                    } else {
                        sa
                    };
                    s.ax[i] = s.ax[i] * r + sx;
                }
                s.m = mn;
                s.pl += dot(row, pb) as f64;
            },
        );
        v0 += nb;
    }
    st.iter()
        .map(|s| {
            let lse = s.m as f64 + s.a.ln();
            let mut a = Out::default();
            for i in 0..NY {
                a.lo[i] = (s.ly[i] as f64 - (s.m as f64 + s.ax[i].max(1e-300).ln())) as f32;
            }
            for (o, (l, id)) in a.top.iter_mut().zip(s.top) {
                *o = (id, (l as f64 - lse).exp() as f32);
            }
            a.kl = (ref_plp - s.pl + lse * psum).max(0.0) as f32;
            a
        })
        .collect()
}

// ------------------------------------------------------------------ computing variants

struct Runner<'a> {
    m: &'a Model,
    ids: &'a [u32],
    focus: usize,
    l: usize,
    kind: Kind,
    ck: &'a Checkpoint,
    values: &'a [f32], // [T][nu] clean values of the units
    nu: usize,
    ys: [u32; NY],
    p_ref: &'a [f32],
    ref_plp: f64,
    n_passes: std::cell::Cell<usize>,
}

impl<'a> Runner<'a> {
    fn compute(&self, vars: &[Variant], deep: bool, info: &str) -> Vec<Outcome> {
        let (t, d, focus) = (self.ids.len(), self.m.d, self.focus);
        let mut out: Vec<Outcome> = Vec::with_capacity(vars.len());
        let tty = std::io::stderr().is_terminal();
        let t0 = Instant::now();
        let mut i = 0;
        let mut mark = 0.25f32; // without a terminal: progress in quarters
        while i < vars.len() {
            // batch: first the unchanged copy (reference), then as many variants as fit the budget
            let mut batch: Vec<&Variant> = Vec::new();
            let ident = Variant { start: focus, edits: Vec::new() };
            batch.push(&ident);
            let mut rows = t + 1;
            while i < vars.len() {
                let len = focus + 1 - vars[i].start.min(focus);
                if batch.len() > 1 && rows + len > BUDGET {
                    break;
                }
                rows += len;
                batch.push(&vars[i]);
                i += 1;
            }
            // build the rows
            let mut pos: Vec<usize> = (0..t).collect();
            let mut seg_start = vec![0usize; t];
            let mut seg_row0 = vec![0usize; t];
            let mut ids_all: Vec<u32> = self.ids.to_vec();
            let mut r0s = Vec::new();
            for v in &batch {
                let r0 = pos.len();
                r0s.push(r0);
                for q in v.start..=focus {
                    pos.push(q);
                    seg_start.push(v.start);
                    seg_row0.push(r0);
                    ids_all.push(self.ids[q]);
                }
            }
            let n_rows = pos.len();
            let xl = &self.ck.x[self.l];
            let mut x = Vec::with_capacity(n_rows * d);
            for &q in &pos {
                x.extend_from_slice(&xl[q * d..(q + 1) * d]);
            }
            let kv: Vec<Option<(Vec<f32>, Vec<f32>)>> = self
                .ck
                .kv
                .iter()
                .enumerate()
                .map(|(s, e)| {
                    if s >= self.l {
                        return None;
                    }
                    e.as_ref().map(|(k, v)| {
                        let kd = k.len() / t;
                        let mut k2 = Vec::with_capacity(n_rows * kd);
                        let mut v2 = Vec::with_capacity(n_rows * kd);
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
                start: Some(Start { layer: self.l, x, kv: Arc::new(kv) }),
                variants: Some(Arc::new(Variants { pos: pos.clone(), seg_start, seg_row0 })),
                ..Default::default()
            };
            let focus_rows: Vec<usize> = batch.iter().zip(&r0s).map(|(v, r0)| r0 + (focus - v.start.min(focus))).collect();
            if deep {
                o.record_rows = focus_rows.clone();
            }
            for (v, &r0) in batch.iter().zip(&r0s) {
                let row = |p: usize| r0 + (p - v.start);
                for e in &v.edits {
                    match *e {
                        Edit::Unit { p, u, f } => match self.kind {
                            Kind::Mlp => o.patch.push((self.l, row(p), u, f * self.values[p * self.nu + u])),
                            Kind::Ple => o.ple_scale.push((self.l, row(p), u, f)),
                            Kind::Attn => o.head_scale.push((self.l, u, row(p), f)),
                        },
                        Edit::Value { p, u, w } => match self.kind {
                            Kind::Mlp => o.patch.push((self.l, row(p), u, w)),
                            Kind::Ple => {
                                let w0 = self.values[p * self.nu + u];
                                o.ple_scale.push((self.l, row(p), u, if w0 != 0.0 { w / w0 } else { 1.0 }));
                            }
                            Kind::Attn => {}
                        },
                        Edit::Whole { p } => o.ablate.push(match self.kind {
                            Kind::Mlp => Ablation::MlpAt(self.l, row(p)),
                            Kind::Ple => Ablation::PleAt(self.l, row(p)),
                            Kind::Attn => Ablation::AttnAt(self.l, row(p)),
                        }),
                    }
                }
            }
            let tr = self.m.run(&ids_all, focus, &o);
            self.n_passes.set(self.n_passes.get() + 1);
            let mut xs = Vec::with_capacity(batch.len() * d);
            for &r in &focus_rows {
                xs.extend_from_slice(&tr.x_final[r * d..(r + 1) * d]);
            }
            let outs = read_out(self.m, &xs, &self.ys, self.p_ref, self.ref_plp);
            let deeps: Vec<Option<Vec<DeepLayer>>> = if deep {
                (0..batch.len()).map(|b| Some(deep_of(&tr.rows, b, &pos, r0s[b], batch[b].start.min(focus), t, d))).collect()
            } else {
                vec![None; batch.len()]
            };
            for (b, a) in outs.iter().enumerate().skip(1) {
                out.push(Outcome { out: *a, id: outs[0], deep: deeps[b].clone() });
            }
            if deep {
                // also return the unchanged copy (as the last element)
                out.push(Outcome { out: outs[0], id: outs[0], deep: deeps[0].clone() });
            }
            let el = t0.elapsed().as_secs_f32();
            let rest = el / i as f32 * (vars.len() - i) as f32;
            if tty {
                eprint!("\r  {info}: {i}/{} variants ({} s, ≈ {} s left)   ", vars.len(), num(el, 0), num(rest, 0));
            } else if vars.len() >= 200 && i as f32 / vars.len() as f32 >= mark {
                eprintln!("  {info}: {i}/{} variants ({} s, ≈ {} s left)", vars.len(), num(el, 0), num(rest, 0));
                while i as f32 / vars.len() as f32 >= mark {
                    mark += 0.25;
                }
            }
        }
        if tty {
            eprintln!();
        }
        out
    }
}

/// Recording of variant b (its focus row) per layer; attention weights mapped back to positions
fn deep_of(rows: &[RowRecord], b: usize, pos: &[usize], r0: usize, start: usize, t: usize, d: usize) -> Vec<DeepLayer> {
    let n_rows = pos.len();
    rows.iter()
        .map(|z| {
            let k = z.x.len() / d; // number of recorded rows
            let ff = z.act.len() / k;
            let h = z.probs.len() / (k * n_rows);
            let mut probs = vec![0f32; h * t];
            for hh in 0..h {
                let src = &z.probs[(b * h + hh) * n_rows..(b * h + hh + 1) * n_rows];
                for (j, &pj) in src.iter().enumerate() {
                    if pj == 0.0 {
                        continue;
                    }
                    // column j: clean row (j < t) or the variant's own row
                    let q = if j < t {
                        j
                    } else if j >= r0 && pos[j] >= start {
                        pos[j]
                    } else {
                        continue;
                    };
                    probs[hh * t + q] += pj;
                }
            }
            DeepLayer {
                layer: z.layer,
                x: z.x[b * d..(b + 1) * d].to_vec(),
                attn: z.attn[b * d..(b + 1) * d].to_vec(),
                mlp: z.mlp[b * d..(b + 1) * d].to_vec(),
                ple: z.ple[b * d..(b + 1) * d].to_vec(),
                act: z.act[b * ff..(b + 1) * ff].to_vec(),
                probs,
            }
        })
        .collect()
}

/// heads whose attention from the focus shifts the most (total variation): (layer offset, head, TV)
fn head_shifts(e: &Outcome, base: &Outcome, n_later: usize, t: usize) -> Vec<(usize, usize, f32)> {
    let mut kd: Vec<(usize, usize, f32)> = Vec::new();
    for s in 0..n_later {
        let a = &e.deep.as_ref().unwrap()[s].probs;
        let b0 = &base.deep.as_ref().unwrap()[s].probs;
        for hh in 0..a.len() / t {
            let tv: f32 = (0..t).map(|j| (a[hh * t + j] - b0[hh * t + j]).abs()).sum::<f32>() / 2.0;
            kd.push((s, hh, tv));
        }
    }
    kd.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    kd
}

/// later neurons whose change moves the lead of y0 over y1 the most: (neuron, contribution)
fn neuron_shifts(ti: &DeepLayer, t0: &DeepLayer, w: &[f32]) -> Vec<(usize, f32)> {
    let mut nv: Vec<(usize, f32)> = (0..w.len()).map(|n| (n, (ti.act[n] - t0.act[n]) * w[n])).collect();
    nv.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(std::cmp::Ordering::Equal));
    nv
}

/// strongest positions a head looks at
fn gaze(pr: &[f32], hh: usize, t: usize, n: usize) -> Vec<(usize, f32)> {
    let mut js: Vec<(usize, f32)> = (0..t).map(|j| (j, pr[hh * t + j])).collect();
    js.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    js.truncate(n);
    js
}

// ------------------------------------------------------------------ run

pub fn run(m: &Model, ids: &[u32], focus: usize, target: Option<u32>, contrast: Option<&[u32]>, o: &Opts, st: &Style) -> Result<(), String> {
    if !m.has_head || m.blocks.iter().any(|b| b.res == Res::Post || !b.attn.causal) {
        return Err("--probe needs a decoder model with an output head".into());
    }
    let (l, kind) = parse_part(&o.part)?;
    let nl = m.n_layer();
    if l >= nl {
        return Err(format!("S{l}: the model only has {nl} layers"));
    }
    if kind == Kind::Ple && m.blocks[l].ple.is_none() {
        return Err(format!("S{l} has no per-layer embedding"));
    }
    let t0 = Instant::now();
    let (t, d) = (ids.len(), m.d);
    eprintln!("probe {}: clean recording pass …", o.part);
    let mut tr = m.run(ids, focus, &RunOpts { record: true, checkpoint: true, ..Default::default() });
    let ck = tr.checkpoint.take().ok_or("checkpoints missing")?;
    let z = tr.logits.clone();
    let (logz, p) = softmax(&z);
    let ref_plp: f64 = z.iter().map(|x| { let lp = *x as f64 - logz; lp.exp() * lp }).sum();
    let top = top_probs(&z, 5);
    let y0 = target.unwrap_or(top[0].0);
    // contrast prompt
    let tr_c = match contrast {
        Some(g) => {
            if g.len() != t {
                return Err(format!("the contrast prompt has {} tokens, the prompt {} – inserting values needs the same count", g.len(), t));
            }
            eprintln!("contrast prompt: recording pass …");
            Some(m.run(g, focus, &RunOpts { record: true, ..Default::default() }))
        }
        None => None,
    };
    let y1 = match &tr_c {
        Some(tg) => {
            let tg_top = top_probs(&tg.logits, 2);
            if tg_top[0].0 != y0 { tg_top[0].0 } else { tg_top[1].0 }
        }
        None => top.iter().map(|x| x.0).find(|&i| i != y0).unwrap_or(0),
    };
    let ys = [y0, y1];
    let contrast_diff: String = contrast
        .map(|g| (0..t).filter(|&j| g[j] != ids[j]).map(|j| format!("[{j}]{}→{}", tok(m, ids[j]), tok(m, g[j]))).collect::<Vec<_>>().join(", "))
        .unwrap_or_default();
    let lo_clean = log_odds(&z, y0 as usize) as f32;

    // units and their clean values
    let b = &m.blocks[l];
    let (nu, values): (usize, Vec<f32>) = match kind {
        Kind::Mlp => (b.mlp.up.rows, tr.acts[l].clone()),
        Kind::Ple => (b.ple.as_ref().unwrap().gate.rows, tr.ple_gated.get(l).cloned().unwrap_or_default()),
        Kind::Attn => (b.attn.n_head, tr.head_norm[l].clone()),
    };
    if values.len() != t * nu {
        return Err("recording of the units is incomplete".into());
    }
    let values_c: Option<Vec<f32>> = tr_c.as_ref().map(|tg| match kind {
        Kind::Mlp => tg.acts[l].clone(),
        Kind::Ple => tg.ple_gated.get(l).cloned().unwrap_or_default(),
        Kind::Attn => Vec::new(),
    });
    let unit = |u: usize| -> String {
        match kind {
            Kind::Mlp => format!("#{u}"),
            Kind::Ple => format!("PLE #{u}"),
            Kind::Attn => format!("head {u}"),
        }
    };
    let runner = Runner { m, ids, focus, l, kind, ck: &ck, values: &values, nu, ys, p_ref: &p, ref_plp, n_passes: std::cell::Cell::new(0) };

    // ---- checks: unchanged copy = real pass; variant = single run
    let pr0 = runner.compute(&[Variant::new(vec![Edit::Whole { p: focus }])], false, "check");
    let dev_identity = (pr0[0].id.lo[0] - lo_clean).abs();

    // ---- 1. where does the component act? off per position · all off · all off except one
    let mut vars: Vec<Variant> = (0..=focus).map(|q| Variant::new(vec![Edit::Whole { p: q }])).collect();
    vars.push(Variant::new((0..=focus).map(|q| Edit::Whole { p: q }).collect()));
    for q in 0..=focus {
        vars.push(Variant::new((0..=focus).filter(|&r| r != q).map(|r| Edit::Whole { p: r }).collect()));
    }
    let by_pos_all = runner.compute(&vars, false, "stage 1 (positions)");
    let by_pos: Vec<Outcome> = by_pos_all[..=focus].to_vec();
    let all_off = by_pos_all[focus + 1].clone();
    let only_here: Vec<Outcome> = by_pos_all[focus + 2..].to_vec();
    let mut pos_rank: Vec<usize> = (0..=focus).collect();
    pos_rank.sort_by(|a, b| by_pos[*a].dlo(0).partial_cmp(&by_pos[*b].dlo(0)).unwrap_or(std::cmp::Ordering::Equal));
    let best = pos_rank[0];
    // distributed: no single position carries even half of the effect -> units at all positions at once
    let distributed = o.pos.is_none() && by_pos[best].dlo(0).abs() < 0.5 * all_off.dlo(0).abs();
    let q0 = o.pos.map(|q| q.min(focus)).unwrap_or(best);
    let qs: Vec<usize> = if distributed { (0..=focus).collect() } else { vec![q0] };
    let qmark = if distributed { ALL } else { q0 };
    let edits_for = |u: usize, f: f32| -> Vec<Edit> { qs.iter().map(|&p| Edit::Unit { p, u, f }).collect() };
    // cross-check of the variant computation against a single run
    let (single_ablation, compared) = if distributed {
        (match kind { Kind::Mlp => Ablation::Mlp(l), Kind::Ple => Ablation::Ple(l), Kind::Attn => Ablation::Attn(l) }, &all_off)
    } else {
        (match kind { Kind::Mlp => Ablation::MlpAt(l, q0), Kind::Ple => Ablation::PleAt(l, q0), Kind::Attn => Ablation::AttnAt(l, q0) }, &by_pos[q0])
    };
    let single = m.run(ids, focus, &RunOpts { ablate: vec![single_ablation], ..Default::default() });
    let dev_single = (log_odds(&single.logits, y0 as usize) as f32 - compared.out.lo[0]).abs();

    // ---- 2. brute force: every unit zeroed on its own (at q0 or at all positions at once)
    let all_active: Vec<usize> = (0..nu).filter(|&u| kind == Kind::Attn || qs.iter().any(|&p| values[p * nu + u] != 0.0)).collect();
    let mut cand = all_active.clone();
    let mut preselected: Option<usize> = None;
    if distributed && kind == Kind::Mlp && cand.len() > o.max {
        // preselect by size of the contribution over all positions: Σ_p |a_p| · ‖W_down[:, u]‖
        let cn = b.mlp.col_norms();
        let sc = |u: usize| -> f32 { qs.iter().map(|&p| values[p * nu + u].abs()).sum::<f32>() * cn[u] };
        cand.sort_by(|a, b| sc(*b).partial_cmp(&sc(*a)).unwrap_or(std::cmp::Ordering::Equal));
        preselected = Some(cand.len());
        cand.truncate(o.max);
    }
    let units: Vec<(usize, usize)> = cand.iter().map(|&u| (qmark, u)).collect();
    let vars: Vec<Variant> = units.iter().map(|&(_, u)| Variant::new(edits_for(u, 0.0))).collect();
    let t2 = Instant::now();
    let brute = runner.compute(&vars, false, "stage 2 (brute force)");
    let secs_brute = t2.elapsed().as_secs_f32();
    let mut rank: Vec<usize> = (0..units.len()).collect();
    rank.sort_by(|a, b| brute[*b].dlo(0).abs().partial_cmp(&brute[*a].dlo(0).abs()).unwrap_or(std::cmp::Ordering::Equal));

    // ---- 3. interplay: first those whose absence weakens the target the most
    let mut supports: Vec<usize> = (0..units.len()).collect();
    supports.sort_by(|a, b| brute[*a].dlo(0).partial_cmp(&brute[*b].dlo(0)).unwrap_or(std::cmp::Ordering::Equal));
    let n_act = supports.len();
    let mut ks: Vec<usize> = Vec::new();
    let mut k = 1;
    while k < n_act {
        ks.push(k);
        k *= 2;
    }
    ks.push(n_act);
    let mut vars: Vec<Variant> = Vec::new();
    for &k in &ks {
        vars.push(Variant::new(supports[..k].iter().flat_map(|&i| edits_for(units[i].1, 0.0)).collect()));
    }
    for &k in &ks {
        // only the k strongest on: all other active units off (also those not preselected)
        let keep: HashSet<usize> = supports[..k].iter().map(|&i| units[i].1).collect();
        let mut edits: Vec<Edit> = all_active.iter().filter(|u| !keep.contains(u)).flat_map(|&u| edits_for(u, 0.0)).collect();
        if edits.is_empty() {
            edits = edits_for(units[supports[0]].1, 1.0);
        }
        vars.push(Variant::new(edits));
    }
    let groups = runner.compute(&vars, false, "stage 3 (interplay)");
    let (grp_off, grp_only) = groups.split_at(ks.len());
    let flip = ks.iter().zip(grp_off).find(|(_, e)| e.out.top[0].0 != y0).map(|(k, e)| (*k, e.out.top[0]));

    // ---- 4. dose for the strongest units
    let n_deep = o.n_deep.min(rank.len());
    let deep_units: Vec<usize> = rank[..n_deep].to_vec();
    let mut vars = Vec::new();
    for &i in &deep_units {
        for &f in &DOSE {
            vars.push(Variant::new(edits_for(units[i].1, f)));
        }
    }
    let dose = runner.compute(&vars, false, "stage 4 (dose)");

    // ---- 5. path through the later layers (recorded at the focus)
    let mut vars: Vec<Variant> = deep_units.iter().map(|&i| Variant::new(edits_for(units[i].1, 0.0))).collect();
    vars.push(Variant::new(qs.iter().map(|&p| Edit::Whole { p }).collect()));
    let deep = runner.compute(&vars, true, "stage 5 (path)");

    // ---- 6. contrast prompt
    let contrast_res: Option<(Vec<Outcome>, Vec<String>)> = values_c.as_ref().filter(|w| !w.is_empty()).map(|wg| {
        let mut vars = Vec::new();
        let mut names = Vec::new();
        if !distributed {
            vars.push(Variant::new((0..nu).map(|u| Edit::Value { p: q0, u, w: wg[q0 * nu + u] }).collect()));
            names.push(format!("whole component at [{q0}]"));
        }
        vars.push(Variant::new((0..=focus).flat_map(|q| (0..nu).map(move |u| (q, u))).map(|(q, u)| Edit::Value { p: q, u, w: wg[q * nu + u] }).collect()));
        names.push("whole component at all positions".into());
        for &i in deep_units.iter() {
            let u = units[i].1;
            vars.push(Variant::new(qs.iter().map(|&p| Edit::Value { p, u, w: wg[p * nu + u] }).collect()));
            names.push(format!("{} {}", unit(u), pos_label(qmark)));
        }
        (runner.compute(&vars, false, "stage 6 (contrast prompt)"), names)
    });

    // ---- voices (partial softmax): units at the focus and all later components of the deep variants
    let mut mult = vec![1f32; nl + 1];
    for k in (0..nl).rev() {
        mult[k] = mult[k + 1] * m.blocks[k].scalar.unwrap_or(1.0);
    }
    let xf = tr.focus_resid[nl].clone();
    let fnorm = m.final_norm.as_ref();
    let s_f = fnorm.map_or(1.0, |n| n.scale(&xf));
    let fmap = |c: &[f32]| -> Vec<f32> { fnorm.map_or_else(|| c.to_vec(), |n| n.map_part(s_f, c)) };
    let h_all: Vec<f32> = match fnorm {
        Some(n) => {
            let mut h = vec![0f32; d];
            n.apply(&xf, &mut h);
            h
        }
        None => xf.clone(),
    };
    let voices_of = |vecs: &[Vec<f32>]| -> Vec<Voice> {
        let mut h = h_all.clone();
        for c in vecs {
            h.extend(fmap(c));
        }
        let (s, _) = voices(m, &h, &z, &p, logz, &ys, 0);
        s[1..].to_vec()
    };
    // units at the focus: their own contribution to the stream
    let unit_voice: Vec<Option<Voice>> = {
        let list: Vec<usize> = rank.iter().take(40).cloned().filter(|_| qs.contains(&focus)).collect();
        let mut vecs = Vec::new();
        match kind {
            Kind::Mlp => {
                let post = match (b.res == Res::Pre, &b.n_mlp_post, tr.focus_mlp_post[l]) {
                    (true, Some(n), Some(s)) => Some((n, s)),
                    _ => None,
                };
                let idx: Vec<usize> = list.iter().map(|&i| units[i].1).collect();
                let cols = columns(&b.mlp.down, &idx);
                for (kk, &u) in idx.iter().enumerate() {
                    let col = &cols[kk * d..(kk + 1) * d];
                    let mut vv = match post {
                        Some((n, s)) => n.map_part(s, col),
                        None => col.to_vec(),
                    };
                    let a = values[focus * nu + u] * mult[l];
                    vv.iter_mut().for_each(|x| *x *= a);
                    vecs.push(vv);
                }
            }
            Kind::Ple => {
                let pb = b.ple.as_ref().unwrap();
                let g = &values[focus * nu..(focus + 1) * nu];
                let pr = pb.proj.matmul(g, 1);
                let s = pb.norm.scale(&pr);
                let idx: Vec<usize> = list.iter().map(|&i| units[i].1).collect();
                let cols = columns(&pb.proj, &idx);
                for (kk, &u) in idx.iter().enumerate() {
                    let mut vv = pb.norm.map_part(s, &cols[kk * d..(kk + 1) * d]);
                    let a = g[u] * mult[l];
                    vv.iter_mut().for_each(|x| *x *= a);
                    vecs.push(vv);
                }
            }
            Kind::Attn => {
                for &i in &list {
                    let u = units[i].1;
                    vecs.push(tr.focus_heads[l][u * d..(u + 1) * d].iter().map(|x| x * mult[l]).collect());
                }
            }
        }
        let vs = if vecs.is_empty() { Vec::new() } else { voices_of(&vecs) };
        let mut out = vec![None; units.len()];
        for (kk, &i) in list.iter().enumerate() {
            out[i] = vs.get(kk).copied();
        }
        out
    };
    // components of the later layers for every deep variant (+ unchanged copy)
    let all_deep: Vec<&Outcome> = deep.iter().collect();
    let mut vecs = Vec::new();
    for e in &all_deep {
        for tl in e.deep.as_ref().unwrap() {
            let ml = mult[tl.layer];
            vecs.push(tl.attn.iter().map(|x| x * ml).collect::<Vec<f32>>());
            vecs.push(tl.mlp.iter().map(|x| x * ml).collect::<Vec<f32>>());
            vecs.push(tl.ple.iter().map(|x| x * ml).collect::<Vec<f32>>());
        }
    }
    eprintln!("voices of the later layers ({} partial softmaxes) …", vecs.len());
    let comp_voices = voices_of(&vecs);
    let n_later = nl - l;
    // logit lens after every layer
    let mut xs = Vec::new();
    for e in &all_deep {
        for tl in e.deep.as_ref().unwrap() {
            xs.extend_from_slice(&tl.x);
        }
    }
    let lens = read_out(m, &xs, &ys, &p, ref_plp);
    // later neurons: weight for the lead of y0 over y1 (direct, frozen readout)
    let lm = m.lm_head();
    let margin: Vec<f32> = {
        let mut a = vec![0f32; d];
        let mut bb = vec![0f32; d];
        lm.row(y0 as usize, &mut a);
        lm.row(y1 as usize, &mut bb);
        let fy = |y: u32, row: &[f32]| -> f32 {
            let raw = dot(row, &h_all);
            if m.logit_cap.is_some() && raw.abs() > 1e-6 { z[y as usize] / raw } else { 1.0 }
        };
        let (f0, f1) = (fy(y0, &a), fy(y1, &bb));
        let g: Vec<f32> = a.iter().zip(&bb).map(|(x, y)| f0 * x - f1 * y).collect();
        fnorm.map_or(g.clone(), |n| n.back(s_f, &g))
    };
    let weights: Vec<Vec<f32>> = (l..nl)
        .map(|k| {
            let bk = &m.blocks[k];
            let u = match (bk.res == Res::Pre, &bk.n_mlp_post, tr.focus_mlp_post[k]) {
                (true, Some(n), Some(s)) => n.back(s, &margin),
                _ => margin.clone(),
            };
            bk.mlp.down.t_matvec(&u).iter().map(|x| x * mult[k]).collect()
        })
        .collect();
    let secs = t0.elapsed().as_secs_f32();

    let r = Report {
        m,
        ids,
        o,
        focus,
        l,
        kind,
        ys,
        top: &top,
        unit: &unit,
        values: &values,
        nu,
        by_pos: &by_pos,
        pos_rank: &pos_rank,
        all_off: &all_off,
        only_here: &only_here,
        distributed,
        q0,
        qmark,
        units: &units,
        preselected,
        secs_brute,
        brute: &brute,
        rank: &rank,
        unit_voice: &unit_voice,
        ks: &ks,
        grp_off,
        grp_only,
        flip,
        deep_units: &deep_units,
        dose: &dose,
        all_deep: &all_deep,
        comp_voices: &comp_voices,
        n_later,
        lens: &lens,
        weights: &weights,
        contrast: contrast.is_some(),
        contrast_diff: &contrast_diff,
        contrast_res: contrast_res.as_ref(),
        dev: (dev_identity, dev_single),
        n_passes: runner.n_passes.get(),
        secs,
    };
    terminal(&r, st);
    std::fs::write(&o.html, html(&r)).map_err(|e| format!("{}: {e}", o.html.display()))?;
    println!("  {} {}", st.bold("All units, dose curves and paths:"), o.html.display());
    Ok(())
}

struct Report<'a> {
    m: &'a Model,
    ids: &'a [u32],
    o: &'a Opts,
    focus: usize,
    l: usize,
    kind: Kind,
    ys: [u32; NY],
    top: &'a [(u32, f32)],
    unit: &'a dyn Fn(usize) -> String,
    values: &'a [f32],
    nu: usize,
    by_pos: &'a [Outcome],
    pos_rank: &'a [usize],
    all_off: &'a Outcome,
    only_here: &'a [Outcome],
    distributed: bool,
    q0: usize,
    qmark: usize,
    units: &'a [(usize, usize)],
    preselected: Option<usize>,
    secs_brute: f32,
    brute: &'a [Outcome],
    rank: &'a [usize],
    unit_voice: &'a [Option<Voice>],
    ks: &'a [usize],
    grp_off: &'a [Outcome],
    grp_only: &'a [Outcome],
    flip: Option<(usize, (u32, f32))>,
    deep_units: &'a [usize],
    dose: &'a [Outcome],
    all_deep: &'a [&'a Outcome],
    comp_voices: &'a [Voice],
    n_later: usize,
    lens: &'a [Out],
    weights: &'a [Vec<f32>],
    contrast: bool,
    contrast_diff: &'a str,
    contrast_res: Option<&'a (Vec<Outcome>, Vec<String>)>,
    dev: (f32, f32),
    n_passes: usize,
    secs: f32,
}

impl Report<'_> {
    /// voice of component comp (0 attention, 1 MLP, 2 PLE) in layer offset s of deep variant vi
    fn comp(&self, vi: usize, s: usize, comp: usize) -> &Voice {
        &self.comp_voices[(vi * self.n_later + s) * 3 + comp]
    }
    /// index of the unchanged copy among the deep variants
    fn n_id(&self) -> usize {
        self.all_deep.len() - 1
    }
    fn value_at(&self, q: usize, u: usize) -> f32 {
        self.values[(if q == ALL { self.focus } else { q }) * self.nu + u]
    }
    fn deep_title(&self, vi: usize) -> String {
        if vi < self.deep_units.len() {
            let (q, u) = self.units[self.deep_units[vi]];
            format!("{} {} off", (self.unit)(u), pos_label(q))
        } else {
            format!("whole component {} off", pos_label(self.qmark))
        }
    }
    fn post_norm(&self) -> bool {
        let b = &self.m.blocks[self.l];
        match self.kind {
            Kind::Mlp => b.res == Res::Pre && b.n_mlp_post.is_some(),
            Kind::Ple => true,
            Kind::Attn => b.res == Res::Pre && b.n_attn_post.is_some(),
        }
    }
    fn where_txt(&self) -> String {
        if self.distributed { "at all positions at once".to_string() } else { format!("at [{}]{}", self.q0, tok(self.m, self.ids[self.q0])) }
    }
}

// ------------------------------------------------------------------ terminal

fn terminal(r: &Report, st: &Style) {
    let m = r.m;
    let (y0, y1) = (r.ys[0], r.ys[1]);
    let (ty0, ty1) = (tok(m, y0), tok(m, y1));
    let t = r.ids.len();
    let pos_txt = |q: usize| format!("[{q}]{}", tok(m, r.ids[q]));
    let res_txt = |e: &Outcome| -> String { format!("odds {ty0} {} · top {} {}", factor(e.dlo(0)), tok(m, e.out.top[0].0), pct(e.out.top[0].1)) };
    println!();
    println!("{}", st.bold(&format!("Probe {} at {} – what the units do", r.o.part, pos_txt(r.focus))));
    println!(
        "  output: {} · compared with {}{} · {} units per position",
        r.top.iter().take(3).map(|(i, q)| format!("{} {}", tok(m, *i), pct(*q))).collect::<Vec<_>>().join(" · "),
        ty1,
        if r.contrast { " (answer of the contrast prompt)" } else { " (second-best token)" },
        r.nu
    );
    if r.contrast {
        println!("  contrast prompt differs at: {}", r.contrast_diff);
    }
    println!("  {}", st.dim(&format!("checks: unchanged copy in the variant pass = real pass (dev. {} in log-odds) · variant = single run (dev. {})", sci(r.dev.0), sci(r.dev.1))));
    println!();
    println!("  {}", st.bold(&format!("1. Where does {} act? Whole component zeroed, at one position at a time:", r.o.part)));
    for &q in r.pos_rank.iter().take(6) {
        println!("     {} {}", pad(&pos_txt(q), 16), res_txt(&r.by_pos[q]));
    }
    println!("     {} {}", pad("all positions", 16), res_txt(r.all_off));
    let mut rescue: Vec<usize> = (0..=r.focus).collect();
    rescue.sort_by(|a, b| r.only_here[*b].dlo(0).partial_cmp(&r.only_here[*a].dlo(0)).unwrap_or(std::cmp::Ordering::Equal));
    println!(
        "     {} {}",
        st.dim("off everywhere except at one position – where it rescues the most on its own:"),
        rescue.iter().take(3).map(|&q| format!("{} ({})", pos_txt(q), factor(r.only_here[q].dlo(0) - r.all_off.dlo(0)))).collect::<Vec<_>>().join(" · ")
    );
    if r.distributed {
        println!("     {}", st.yellow(&format!("→ distributed: no single position carries even half – units are changed at all {} positions at once", r.focus + 1)));
    } else {
        println!("     {}", st.dim(&format!("→ local: probing at {}", pos_txt(r.q0))));
    }
    println!(
        "  {}",
        st.bold(&format!(
            "2. Brute force: {} units zeroed one by one {} ({} s){}",
            r.units.len(),
            r.where_txt(),
            num(r.secs_brute, 0),
            r.preselected.map_or(String::new(), |n| format!(" – preselected the {} largest of {n} active by Σ|a|·‖W_down‖ (--probe-max)", r.units.len()))
        ))
    );
    let count = |lim: f32| r.brute.iter().filter(|e| e.dlo(0).abs() >= lim).count();
    println!(
        "     change the odds of {ty0} by more than ×2: {} · ×10: {} · ×100: {} · flip the answer: {}",
        count(0.693),
        count(2.303),
        count(4.605),
        r.brute.iter().filter(|e| e.out.top[0].0 != y0).count()
    );
    for &i in r.rank.iter().take(12) {
        let (q, u) = r.units[i];
        let val = match r.kind {
            Kind::Attn => format!("‖·‖={}", num(r.value_at(q, u), 1)),
            _ => format!("a={}", num(r.value_at(q, u), 2)),
        };
        let own = r.unit_voice[i].map_or(String::new(), |s| format!(" · own voice {}", s.top[..3].iter().map(|(id, pr)| format!("{} {}", tok(m, *id), pct(*pr))).collect::<Vec<_>>().join(" ")));
        println!("     {} {} {} {}", pad(&format!("{} {}", (r.unit)(u), pos_label(q)), 16), padl(&val, 10), res_txt(&r.brute[i]), st.dim(&own));
    }
    println!("  {}", st.bold(&format!("3. Interplay {}: the k strongest supports off together / only they on", r.where_txt())));
    for (j, &k) in r.ks.iter().enumerate() {
        println!("     k = {:<6} off: {}   │ only they on: {}", k, res_txt(&r.grp_off[j]), res_txt(&r.grp_only[j]));
    }
    if r.post_norm() {
        println!(
            "     {}",
            st.yellow("Note: the output goes through a post-norm – switching off part of the units scales the rest up. Nonsense like \"z\" or \"ein\" then often only means: the rest was inflated.")
        );
    }
    match r.flip {
        Some((k, (id, pr))) => println!("     → the answer flips as soon as the {k} strongest are missing together: {} {}", tok(m, id), pct(pr)),
        None => println!("     → even all of them together do not flip the answer{}", if r.distributed { "" } else { " at this position" }),
    }
    println!("  {}", st.bold(&format!("4. Dose: activation times factor (1 = real); number = factor on the odds of {ty0}, → new top if it changes")));
    for (j, &i) in r.deep_units.iter().enumerate() {
        let (q, u) = r.units[i];
        let line: Vec<String> = DOSE
            .iter()
            .enumerate()
            .map(|(fi, f)| {
                let e = &r.dose[j * DOSE.len() + fi];
                let sp = if e.out.top[0].0 != y0 { format!(" → {} {}", tok(m, e.out.top[0].0), pct(e.out.top[0].1)) } else { String::new() };
                format!("×{}: {}{sp}", num(*f, 1), factor(e.dlo(0)))
            })
            .collect();
        println!("     {} {}", pad(&format!("{} {}", (r.unit)(u), pos_label(q)), 16), line.join(" │ "));
    }
    println!("  {}", st.bold("5. What runs differently afterwards (partial softmax of the later layers at the focus)"));
    let n_id = r.n_id();
    for (vi, e) in r.all_deep.iter().enumerate().take(n_id) {
        println!("   ▸ {} – {}", st.bold(&r.deep_title(vi)), res_txt(e));
        // logit lens: from which layer on is y0 ahead?
        let ahead = |vj: usize| -> Option<usize> { (0..r.n_later).find(|&s| (s..r.n_later).all(|s2| r.lens[vj * r.n_later + s2].top[0].0 == y0)).map(|s| r.l + s) };
        println!(
            "      logit lens: {ty0} ahead from {} (unchanged: from {})",
            ahead(vi).map_or("never".to_string(), |s| format!("S{s}")),
            ahead(n_id).map_or("never".to_string(), |s| format!("S{s}"))
        );
        // components whose direct voice changes the most
        let mut ch: Vec<(usize, usize, f32)> = Vec::new();
        for s in 0..r.n_later {
            for comp in 0..3 {
                ch.push((s, comp, r.comp(vi, s, comp).dlo[0] - r.comp(n_id, s, comp).dlo[0]));
            }
        }
        ch.sort_by(|a, b| b.2.abs().partial_cmp(&a.2.abs()).unwrap_or(std::cmp::Ordering::Equal));
        for &(s, comp, _) in ch.iter().take(4) {
            let (a0, a1) = (r.comp(n_id, s, comp), r.comp(vi, s, comp));
            let name = format!("S{} {}", r.l + s, ["attention", "MLP", "PLE"][comp]);
            println!(
                "      {} votes directly for {ty0} {} → {}   ({} → {})",
                pad(&name, 14),
                factor(a0.dlo[0]),
                factor(a1.dlo[0]),
                a0.top[..2].iter().map(|(id, pr)| format!("{} {}", tok(m, *id), pct(*pr))).collect::<Vec<_>>().join(" "),
                a1.top[..2].iter().map(|(id, pr)| format!("{} {}", tok(m, *id), pct(*pr))).collect::<Vec<_>>().join(" ")
            );
            if comp == 1 {
                let ti = &e.deep.as_ref().unwrap()[s];
                let t0_ = &r.all_deep[n_id].deep.as_ref().unwrap()[s];
                let txt: Vec<String> = neuron_shifts(ti, t0_, &r.weights[s])
                    .iter()
                    .take(3)
                    .map(|(n, c)| format!("#{n} a {}→{} ({} for {ty0} over {ty1})", num(t0_.act[*n], 2), num(ti.act[*n], 2), factor(*c)))
                    .collect();
                println!("      {}", st.dim(&format!("   neurons in it: {}", txt.join(" · "))));
            }
        }
        // heads that look elsewhere
        let base = r.all_deep[n_id];
        let look = |pr: &[f32], hh: usize| -> String { gaze(pr, hh, t, 2).iter().map(|(j, a)| format!("{} {}", pos_txt(*j), pct(*a))).collect::<Vec<_>>().join(" ") };
        for &(s, hh, tv) in head_shifts(e, base, r.n_later, t).iter().take(2).filter(|x| x.2 > 0.05) {
            let a = &e.deep.as_ref().unwrap()[s].probs;
            let b0 = &base.deep.as_ref().unwrap()[s].probs;
            println!("      S{} head {hh} looks elsewhere ({} shifted): {} → {}", r.l + s, pct(tv), look(b0, hh), look(a, hh));
        }
    }
    if let Some((eg, names)) = r.contrast_res {
        println!("  {}", st.bold(&format!("6. Values from the contrast prompt inserted (there the model answers {ty1})")));
        for (e, n) in eg.iter().zip(names) {
            println!("     {} odds {ty1} {} · {}", pad(n, 34), factor(e.dlo(1)), res_txt(e));
        }
    }
    println!();
    println!("  {}", st.dim(&format!("{} variant passes, total {} s. Odds = p/(1−p), factor vs. the unchanged copy in the same pass.", r.n_passes, num(r.secs, 0))));
}

// ------------------------------------------------------------------ HTML

fn html(r: &Report) -> String {
    let m = r.m;
    let (y0, y1) = (r.ys[0], r.ys[1]);
    let t = r.ids.len();
    let pos_html = |q: usize| format!("[{q}]{}", chip(m, r.ids[q]));
    let top_html = |e: &Outcome| format!("{} {}", chip(m, e.out.top[0].0), esc_html(&pct(e.out.top[0].1)));
    let mut h = String::with_capacity(1 << 20);
    h += &format!(
        r#"<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Probe {}</title><style>{CSS}{PROBE_CSS}</style></head><body><main>
<h1>Probe {} – what the units do</h1>
<p class="meta">{} · focus token [{}] · target {} vs. {}</p>
<p class="prompt">{}</p>
<p class="check">Checks: unchanged copy in the variant pass = real pass (dev. {} in log-odds) · variant = single run (dev. {}) · {} s</p>
"#,
        esc_html(&r.o.part),
        esc_html(&r.o.part),
        esc_html(&m.name),
        r.focus,
        chip(m, y0),
        chip(m, y1),
        prompt_html(m, r.ids, r.focus),
        esc_html(&sci(r.dev.0)),
        esc_html(&sci(r.dev.1)),
        num(r.secs, 0)
    );
    // 1. positions
    h += "<h2>1. Where does the component act?</h2><p class=\"mu\">Whole component zeroed, at one position at a time; factor on the odds p/(1−p) of the target token. \"only here on\": zeroed everywhere else – what this position rescues on its own.</p><table class=\"tbl\"><tr><th>position</th><th>odds</th><th>top afterwards</th><th>KL</th><th></th><th>only here on</th></tr>";
    let maxd = r.by_pos.iter().map(|e| e.dlo(0).abs()).fold(1e-6f32, f32::max);
    for q in 0..=r.focus {
        let e = &r.by_pos[q];
        h += &format!(
            "<tr><td>{}</td><td class=\"num\">{}</td><td>{}</td><td class=\"num\">{}</td><td><span class=\"mb\" style=\"display:inline-block;width:160px\"><span style=\"width:{:.0}%\"></span></span></td><td class=\"num\">{} {}</td></tr>",
            pos_html(q),
            factor_html(e.dlo(0)),
            top_html(e),
            num(e.out.kl, 2),
            100.0 * e.dlo(0).abs() / maxd,
            factor_html(r.only_here[q].dlo(0)),
            top_html(&r.only_here[q])
        );
    }
    h += &format!("<tr><td><b>all positions</b></td><td class=\"num\">{}</td><td>{}</td><td class=\"num\">{}</td><td></td><td></td></tr></table>", factor_html(r.all_off.dlo(0)), top_html(r.all_off), num(r.all_off.out.kl, 2));
    h += &format!(
        "<p><b>{}</b></p>",
        if r.distributed { "Distributed: no single position carries even half – the units are changed at all positions at once.".to_string() } else { format!("Local: probing at [{}].", r.q0) }
    );
    // 2. brute force
    h += &format!(
        "<h2>2. Brute force: every unit zeroed on its own</h2><p class=\"mu\">{} units {}; sorted by effect on the odds of {} (the 400 strongest). \"own voice\": partial softmax of what the unit itself writes at the focus token. Value: activation at the focus.</p>",
        r.units.len(),
        if r.distributed { "at all positions at once".to_string() } else { format!("at {}", pos_html(r.q0)) },
        chip(m, y0)
    );
    h += "<table class=\"tbl\"><tr><th>unit</th><th>pos.</th><th>value</th><th>odds target</th><th>odds rival</th><th>KL</th><th>top afterwards</th><th>own voice</th></tr>";
    for &i in r.rank.iter().take(400) {
        let (q, u) = r.units[i];
        let e = &r.brute[i];
        let own = r.unit_voice[i].map_or(String::new(), |s| s.top[..3].iter().map(|(id, pr)| format!("<span class=\"st\">{}<span class=\"pc\">{}</span></span>", chip(m, *id), esc_html(&pct(*pr)))).collect());
        h += &format!(
            "<tr><td>{}</td><td>{}</td><td class=\"num\">{}</td><td class=\"num\">{}</td><td class=\"num\">{}</td><td class=\"num\">{}</td><td>{}</td><td>{own}</td></tr>",
            esc_html(&(r.unit)(u)),
            pos_label(q),
            num(r.value_at(q, u), 2),
            factor_html(e.dlo(0)),
            factor_html(e.dlo(1)),
            num(e.out.kl, 3),
            top_html(e)
        );
    }
    h += "</table>";
    // 3. interplay
    h += "<h2>3. Interplay</h2><p class=\"mu\">The k strongest supports (whose absence weakens the target the most) off together – and conversely only they on, all others off.</p><table class=\"tbl\"><tr><th>k</th><th>k off: odds</th><th>top</th><th>only k on: odds</th><th>top</th></tr>";
    for (j, &k) in r.ks.iter().enumerate() {
        h += &format!(
            "<tr><td class=\"num\">{k}</td><td class=\"num\">{}</td><td>{}</td><td class=\"num\">{}</td><td>{}</td></tr>",
            factor_html(r.grp_off[j].dlo(0)),
            top_html(&r.grp_off[j]),
            factor_html(r.grp_only[j].dlo(0)),
            top_html(&r.grp_only[j])
        );
    }
    h += "</table>";
    // 4. dose
    h += &format!("<h2>4. Dose</h2><p class=\"mu\">Activation times factor (1 = real). Number: factor on the odds of {}; below, the new top token.</p><table class=\"tbl\"><tr><th>unit</th>", chip(m, y0));
    for f in DOSE {
        h += &format!("<th>×{}</th>", num(f, 1));
    }
    h += "</tr>";
    for (j, &i) in r.deep_units.iter().enumerate() {
        let (q, u) = r.units[i];
        h += &format!("<tr><td>{} {}</td>", esc_html(&(r.unit)(u)), pos_label(q));
        for fi in 0..DOSE.len() {
            let e = &r.dose[j * DOSE.len() + fi];
            h += &format!("<td class=\"num\">{}<br>{}</td>", factor_html(e.dlo(0)), top_html(e));
        }
        h += "</tr>";
    }
    h += "</table>";
    // 5. paths
    h += "<h2>5. What runs differently afterwards</h2><p class=\"mu\">For the strongest units: logit lens after every layer and the direct voice (partial softmax) of every later attention / MLP / PLE at the focus – unchanged vs. with the edit.</p>";
    let n_id = r.n_id();
    let base = r.all_deep[n_id];
    for (vi, e) in r.all_deep.iter().enumerate().take(n_id) {
        h += &format!(
            "<details class=\"info\"{}><summary>{} – odds {} {} · top {}</summary>",
            if vi == 0 { " open" } else { "" },
            esc_html(&r.deep_title(vi)),
            chip(m, y0),
            factor_html(e.dlo(0)),
            top_html(e)
        );
        h += "<table class=\"tbl\"><tr><th>layer</th><th>lens: top (unchanged → edit)</th><th>attention direct</th><th>MLP direct</th><th>PLE direct</th><th>MLP: strongest changed neurons</th></tr>";
        for s in 0..r.n_later {
            let li0 = &r.lens[n_id * r.n_later + s];
            let li1 = &r.lens[vi * r.n_later + s];
            let mut row = format!("<tr><td>S{}</td><td>{} {} → {} {}</td>", r.l + s, chip(m, li0.top[0].0), esc_html(&pct(li0.top[0].1)), chip(m, li1.top[0].0), esc_html(&pct(li1.top[0].1)));
            for comp in 0..3 {
                let (a0, a1) = (r.comp(n_id, s, comp), r.comp(vi, s, comp));
                let strong = (a1.dlo[0] - a0.dlo[0]).abs() > 0.693;
                row += &format!(
                    "<td class=\"num{}\">{} → {}<br><span class=\"pc\">{} → {}</span></td>",
                    if strong { " hl" } else { "" },
                    esc_html(&factor(a0.dlo[0])),
                    factor_html(a1.dlo[0]),
                    chip(m, a0.top[0].0),
                    chip(m, a1.top[0].0)
                );
            }
            let ti = &e.deep.as_ref().unwrap()[s];
            let t0_ = &base.deep.as_ref().unwrap()[s];
            row += &format!(
                "<td class=\"pc\">{}</td></tr>",
                neuron_shifts(ti, t0_, &r.weights[s])
                    .iter()
                    .take(2)
                    .filter(|x| x.1.abs() > 0.05)
                    .map(|(n, cc)| format!("#{n} {}→{} ({})", num(t0_.act[*n], 2), num(ti.act[*n], 2), esc_html(&factor(*cc))))
                    .collect::<Vec<_>>()
                    .join("<br>")
            );
            h += &row;
        }
        h += "</table>";
        // heads
        let look = |pr: &[f32], hh: usize| -> String { gaze(pr, hh, t, 3).iter().map(|(j, a)| format!("{} {}", pos_html(*j), esc_html(&pct(*a)))).collect::<Vec<_>>().join(" ") };
        h += "<h4>Heads that look elsewhere afterwards</h4><table class=\"tbl\"><tr><th>head</th><th>shifted</th><th>before</th><th>after</th></tr>";
        for &(s, hh, tv) in head_shifts(e, base, r.n_later, t).iter().take(6).filter(|x| x.2 > 0.02) {
            let a = &e.deep.as_ref().unwrap()[s].probs;
            let b0 = &base.deep.as_ref().unwrap()[s].probs;
            h += &format!("<tr><td>S{} head {hh}</td><td class=\"num\">{}</td><td>{}</td><td>{}</td></tr>", r.l + s, esc_html(&pct(tv)), look(b0, hh), look(a, hh));
        }
        h += "</table></details>";
    }
    if let Some((eg, names)) = r.contrast_res {
        h += &format!(
            "<h2>6. Values from the contrast prompt inserted</h2><p class=\"mu\">There the model answers {}.</p><table class=\"tbl\"><tr><th>inserted</th><th>odds {}</th><th>odds {}</th><th>top</th></tr>",
            chip(m, y1),
            chip(m, y1),
            chip(m, y0)
        );
        for (e, n) in eg.iter().zip(names) {
            h += &format!("<tr><td>{}</td><td class=\"num\">{}</td><td class=\"num\">{}</td><td>{}</td></tr>", esc_html(n), factor_html(e.dlo(1)), factor_html(e.dlo(0)), top_html(e));
        }
        h += "</table>";
    }
    h += "<p class=\"mu foot\">apollon · probe</p></main></body></html>\n";
    h
}

const PROBE_CSS: &str = r#"
table.tbl{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:4px 8px;display:block;overflow-x:auto;max-width:100%}
table.tbl td{vertical-align:top;border-top:1px solid var(--line)}
td.hl{background:var(--chip)}
"#;
