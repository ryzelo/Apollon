//! Decomposition of one forward pass into token voices – training-free, for this one prompt only.
//!
//! The stream at the end (focus token) is exactly the sum of its changes: embedding, every attention,
//! every MLP (Gemma 4: every per-layer embedding), each times the later layer scalars. Every change goes
//! through the final norm on its own – with the scale from EXACTLY this pass, the same for all – and
//! through the output matrix; the softmax of that is its voice. With a logit softcap (Gemma) every token
//! logit is scaled by the fixed factor softcap(z)/z from this pass (like the norm). So exactly:
//! sum of all voice logits (+ final-norm bias) = real logits, normalized product of all voices = real
//! output distribution. Heads, "head reads token j" and neurons are made readable the same way.
//!
//! Total effect: the change set to zero (at all positions or only at the focus), the model rerun from
//! its layer. Indirect = total − direct, on the log-odds scale; the share that only comes from the new
//! scale of the final norm is reported separately ("via norm").
use crate::blas;
use crate::fmt::{chip, esc_html, factor, factor_html, num, pad, padl, pct, pp, prompt_html, sci, tok, trunc, Style, CSS};
use crate::model::{top_probs, Ablation, Block, Model, Res, RunOpts, Start, Trace};
use crate::readout::{columns, log_odds, softmax, voices, Voice, K, NY};
use crate::tensor::{dot, norm};
use rayon::prelude::*;
use std::io::IsTerminal;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

/// tokens used to preselect sources and neurons
const NS: usize = 64;

pub struct Opts {
    pub result_json: Option<PathBuf>,
    pub focus_only: bool,
    pub html: PathBuf,
    /// sources shown per head and neurons per MLP
    pub k: usize,
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Total,
    Embedding,
    Attn,
    Mlp,
    Ple,
    FinalBias,
    Head(u16),
    AttnBias,
    PostNormBias,
    Source(u16, u32),
    OtherSources(u16, u32),
    Neuron(u32, f32),
    OtherNeurons(u32),
    MlpBias,
}

impl Kind {
    fn is_change(self) -> bool {
        matches!(self, Kind::Embedding | Kind::Attn | Kind::Mlp | Kind::Ple | Kind::FinalBias)
    }
}

#[derive(Clone, Copy)]
struct Part {
    kind: Kind,
    l: usize,
    change: usize, // index of the change this part belongs to
}

/// all parts as vectors in the stream (at the focus, already times the layer scalars)
struct Parts {
    parts: Vec<Part>,
    c: Vec<f32>,
    d: usize,
}

impl Parts {
    fn push(&mut self, kind: Kind, l: usize, change: usize, v: &[f32], f: f32) -> usize {
        self.parts.push(Part { kind, l, change });
        self.c.extend(v.iter().map(|x| x * f));
        self.parts.len() - 1
    }
    fn row(&self, i: usize) -> &[f32] {
        &self.c[i * self.d..(i + 1) * self.d]
    }
}

fn add(a: &mut [f32], b: &[f32]) {
    for (x, y) in a.iter_mut().zip(b) {
        *x += y;
    }
}

fn by_abs_desc(a: f32, b: f32) -> std::cmp::Ordering {
    b.abs().partial_cmp(&a.abs()).unwrap_or(std::cmp::Ordering::Equal)
}

// ------------------------------------------------------------------ preselection

/// Logits for the NS most likely tokens only; strength = p-weighted spread.
/// Only used to choose which sources/neurons are evaluated in full.
struct Preselect {
    ids: Vec<u32>,
    uh: Vec<f32>, // [ns][d] rows of the output matrix times the softcap factor (normed space)
    u: Vec<f32>,  // [ns][d] in stream space (mapped back through the final norm)
    w: Vec<f32>,
    ns: usize,
}

impl Preselect {
    fn strength(&self, lg: &[f32]) -> f32 {
        let mu: f32 = lg.iter().zip(&self.w).map(|(a, b)| a * b).sum();
        lg.iter().zip(&self.w).map(|(a, b)| b * (a - mu) * (a - mu)).sum::<f32>().sqrt()
    }
    fn logits(&self, x: &[f32], n: usize, d: usize) -> Vec<f32> {
        let mut out = vec![0f32; n * self.ns];
        if n > 0 {
            blas::gemm_nt(x, &self.u, n, self.ns, d, &mut out);
        }
        out
    }
}

// ------------------------------------------------------------------ total effect

/// (token, p without, p with)
type Shift = (u32, f32, f32);

#[derive(Clone)]
struct Effect {
    kl: f32,
    dlo: [f32; NY], // log-odds of y: with − without
    gains: Vec<Shift>,
    losses: Vec<Shift>,
    nd: f32, // weighted spread of the log change: direct
    ni: f32, // indirect
    top_without: Vec<(u32, f32)>,
}

fn effect(z: &[f32], logz: f64, z2: &[f32], l: &[f32], ys: &[u32]) -> Effect {
    let v = z.len();
    let mx = z2.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let logz2 = mx + z2.iter().map(|x| (*x as f64 - mx).exp()).sum::<f64>().ln();
    let acc = (0..v)
        .into_par_iter()
        .with_min_len(4096)
        .fold(
            || [0f64; 7],
            |mut a, t| {
                let lp = z[t] as f64 - logz;
                let lp2 = z2[t] as f64 - logz2;
                let (p, p2) = (lp.exp(), lp2.exp());
                let dt = lp - lp2;
                let dd = l[t] as f64;
                let di = dt - dd;
                let w = 0.5 * (p + p2);
                a[0] += p * dt;
                a[1] += w;
                a[2] += w * dd;
                a[3] += w * dd * dd;
                a[4] += w * di;
                a[5] += w * di * di;
                a
            },
        )
        .reduce(|| [0f64; 7], |mut a, b| {
            for i in 0..7 {
                a[i] += b[i];
            }
            a
        });
    let sd = |s: f64, q: f64| -> f32 { (q / acc[1] - (s / acc[1]).powi(2)).max(0.0).sqrt() as f32 };
    let mut dp: Vec<Shift> = (0..v)
        .filter_map(|t| {
            let p = (z[t] as f64 - logz).exp() as f32;
            let p2 = (z2[t] as f64 - logz2).exp() as f32;
            ((p - p2).abs() >= 5e-4).then_some((t as u32, p2, p))
        })
        .collect();
    dp.sort_by(|a, b| (b.2 - b.1).partial_cmp(&(a.2 - a.1)).unwrap_or(std::cmp::Ordering::Equal));
    let gains: Vec<_> = dp.iter().take(K).filter(|x| x.2 > x.1).cloned().collect();
    let losses: Vec<_> = dp.iter().rev().take(K).filter(|x| x.2 < x.1).cloned().collect();
    let mut dlo = [0f32; NY];
    for (i, &y) in ys.iter().enumerate() {
        dlo[i] = (log_odds(z, y as usize) - log_odds(z2, y as usize)) as f32;
    }
    Effect { kl: acc[0] as f32, dlo, gains, losses, nd: sd(acc[2], acc[3]), ni: sd(acc[4], acc[5]), top_without: top_probs(z2, K) }
}

// ------------------------------------------------------------------ run

struct Results {
    voices: Vec<Voice>,
    effects: Vec<Option<Effect>>, // per change (exact)
    children: Vec<Vec<usize>>,    // per change: its parts (indices into parts)
    fixed: Vec<bool>,             // total effect = direct voice (final-norm bias: nothing reads it)
    direct_dp: Vec<(Vec<Shift>, Vec<Shift>)>, // per row: raises / lowers directly
    norm_part: Vec<Option<[f32; NY]>>, // per change: part of the indirect effect that only comes from rescaling the final norm
}

/// Effect of change k (row ri) on the log-odds of target i: (direct, total, indirect).
fn effects(e: &Results, ri: usize, k: usize, i: usize) -> (f32, Option<f32>, Option<f32>) {
    let dd = e.voices[ri].dlo[i];
    if e.fixed[k] {
        return (dd, Some(dd), Some(0.0));
    }
    match &e.effects[k] {
        Some(w) => (dd, Some(w.dlo[i]), Some(w.dlo[i] - dd)),
        None => (dd, None, None),
    }
}

fn direct_share(e: &Results, k: usize) -> Option<f32> {
    if e.fixed[k] {
        return Some(1.0);
    }
    let w = e.effects[k].as_ref()?;
    (w.nd + w.ni > 0.0).then(|| w.nd / (w.nd + w.ni))
}

fn total_strength(e: &Results, k: usize) -> f32 {
    e.effects[k].as_ref().map_or(0.0, |w| w.kl)
}

fn name(m: &Model, ids: &[u32], t: &Part) -> String {
    match t.kind {
        Kind::Total => "whole stream".into(),
        Kind::Embedding => "embedding".into(),
        Kind::Attn => format!("S{} attention", t.l),
        Kind::Mlp => format!("S{} MLP", t.l),
        Kind::Ple => format!("S{} PLE", t.l),
        Kind::FinalBias => "final bias".into(),
        Kind::Head(h) => format!("head {h}"),
        Kind::AttnBias => "attn bias".into(),
        Kind::PostNormBias => "post-norm bias".into(),
        Kind::Source(_, j) => format!("reads [{j}]{}", tok(m, ids[j as usize])),
        Kind::OtherSources(_, n) => format!("other {n} sources"),
        Kind::Neuron(n, a) => format!("#{n} (a={})", num(a, 2)),
        Kind::OtherNeurons(n) => format!("other {n} neurons"),
        Kind::MlpBias => "MLP bias".into(),
    }
}

/// Bias of a post-norm (belongs to the change, but to no head/neuron)
fn post_bias(b: &Block, attn: bool) -> Option<&Vec<f32>> {
    if b.res != Res::Pre {
        return None;
    }
    (if attn { b.n_attn_post.as_ref() } else { b.n_mlp_post.as_ref() }).and_then(|n| n.b.as_ref())
}

pub fn run(m: &Model, ids: &[u32], focus: usize, target: Option<u32>, o: &Opts, st: &Style) -> Result<(), String> {
    if !m.has_head || m.blocks.iter().any(|b| b.res == Res::Post) {
        return Err("--decompose needs a decoder model with an output head (no post-LN)".into());
    }
    let t0 = Instant::now();
    let (d, nl) = (m.d, m.n_layer());
    eprintln!("recording pass (heads × sources, checkpoints) …");
    let mut tr = m.run(ids, focus, &RunOpts { atoms: true, checkpoint: true, ..Default::default() });
    let z = tr.logits.clone();
    let v = z.len();
    let (logz, p) = softmax(&z);
    let top = top_probs(&z, 10);
    let y0 = target.unwrap_or(top[0].0);
    let mut ys = vec![y0];
    for (i, _) in &top {
        if ys.len() < NY && *i != y0 {
            ys.push(*i);
        }
    }
    let mut mult = vec![1f32; nl + 1];
    for l in (0..nl).rev() {
        mult[l] = mult[l + 1] * m.blocks[l].scalar.unwrap_or(1.0);
    }
    let xf = tr.focus_resid[nl].clone();
    let fnorm = m.final_norm.as_ref();
    let s_f = fnorm.map_or(1.0, |n| n.scale(&xf));
    let fmap = |c: &[f32]| -> Vec<f32> { fnorm.map_or_else(|| c.to_vec(), |n| n.map_part(s_f, c)) };
    let fback = |g: &[f32]| -> Vec<f32> { fnorm.map_or_else(|| g.to_vec(), |n| n.back(s_f, g)) };
    let h_all: Vec<f32> = match fnorm {
        Some(n) => {
            let mut h = vec![0f32; d];
            n.apply(&xf, &mut h);
            h
        }
        None => xf.clone(),
    };
    let lm = m.lm_head();

    // preselection on the most likely tokens
    let pre = {
        let mut s_ids: Vec<u32> = top_probs(&z, NS).iter().map(|x| x.0).collect();
        if !s_ids.contains(&y0) {
            s_ids.push(y0);
        }
        let mut u = Vec::with_capacity(s_ids.len() * d);
        let mut uh = Vec::with_capacity(s_ids.len() * d);
        let mut w = Vec::new();
        let mut buf = vec![0f32; d];
        for &k in &s_ids {
            lm.row(k as usize, &mut buf);
            let raw = dot(&buf, &h_all);
            let f = if m.logit_cap.is_some() && raw.abs() > 1e-6 { z[k as usize] / raw } else { 1.0 };
            let g: Vec<f32> = buf.iter().map(|x| x * f).collect();
            u.extend(fback(&g));
            uh.extend_from_slice(&g);
            w.push(p[k as usize]);
        }
        // weights: half on the target token, half on the competition (by p) – so the lead counts
        // even when p(target) ≈ 1
        let ws: f32 = s_ids.iter().zip(&w).filter(|(t, _)| **t != y0).map(|(_, x)| *x).sum::<f32>().max(1e-30);
        for (t, x) in s_ids.iter().zip(w.iter_mut()) {
            *x = if *t == y0 { 0.5 } else { 0.5 * *x / ws };
        }
        let ns = s_ids.len();
        Preselect { ids: s_ids, uh, u, w, ns }
    };

    // ---- changes (rows 1..=na) and their parts
    let mut parts = Parts { parts: Vec::new(), c: Vec::new(), d };
    parts.push(Kind::Total, nl, 0, &xf, 1.0);
    let mut changes: Vec<usize> = Vec::new();
    let mut per_layer: Vec<[Option<usize>; 3]> = vec![[None; 3]; nl];
    changes.push(parts.push(Kind::Embedding, 0, 0, &tr.focus_resid[0], mult[0]));
    for l in 0..nl {
        let b = &m.blocks[l];
        let nh = b.attn.n_head;
        let mut a = vec![0f32; d];
        for h in 0..nh {
            add(&mut a, &tr.focus_heads[l][h * d..(h + 1) * d]);
        }
        if let Some(bv) = &tr.focus_attn_bias[l] {
            add(&mut a, bv);
        }
        if let Some(bb) = post_bias(b, true) {
            add(&mut a, bb);
        }
        let k = changes.len();
        per_layer[l][0] = Some(k);
        changes.push(parts.push(Kind::Attn, l, k, &a, mult[l]));
        let mut mv = tr.focus_mlp[l].clone();
        if let Some(bv) = &tr.focus_mlp_bias[l] {
            add(&mut mv, bv);
        }
        if let Some(bb) = post_bias(b, false) {
            add(&mut mv, bb);
        }
        let k = changes.len();
        per_layer[l][1] = Some(k);
        changes.push(parts.push(Kind::Mlp, l, k, &mv, mult[l]));
        if let Some(pv) = &tr.focus_ple[l] {
            let k = changes.len();
            per_layer[l][2] = Some(k);
            changes.push(parts.push(Kind::Ple, l, k, pv, mult[l]));
        }
    }
    let final_bias = fnorm.and_then(|n| n.b.as_ref());
    if final_bias.is_some() {
        let k = changes.len();
        changes.push(parts.push(Kind::FinalBias, nl, k, &vec![0f32; d], 1.0));
    }
    let na = changes.len();
    debug_assert!(changes.iter().enumerate().all(|(k, &r)| r == k + 1));
    // check: sum of the changes = stream at the end
    let mut sum = vec![0f64; d];
    for &r in &changes {
        for (a, x) in sum.iter_mut().zip(parts.row(r)) {
            *a += *x as f64;
        }
    }
    let stream_dev = (sum.iter().zip(&xf).map(|(a, b)| (a - *b as f64).powi(2)).sum::<f64>().sqrt() / norm(&xf).max(1e-30) as f64) as f32;

    // parts: heads -> sources, neurons
    let kq = o.k.max(1);
    let mut source_dev = 0f32;
    for l in 0..nl {
        let b = &m.blocks[l];
        let nh = b.attn.n_head;
        let ml = mult[l];
        let ia = per_layer[l][0].unwrap();
        for h in 0..nh {
            parts.push(Kind::Head(h as u16), l, ia, &tr.focus_heads[l][h * d..(h + 1) * d], ml);
        }
        if let Some(bv) = &tr.focus_attn_bias[l] {
            parts.push(Kind::AttnBias, l, ia, bv, ml);
        }
        if let Some(bb) = post_bias(b, true) {
            parts.push(Kind::PostNormBias, l, ia, bb, ml);
        }
        let at = &tr.atom_attn[l];
        if !at.is_empty() {
            let nf = at.len() / (nh * d);
            let lg = pre.logits(at, nf * nh, d);
            for h in 0..nh {
                let hv = &tr.focus_heads[l][h * d..(h + 1) * d];
                let mut s = vec![0f32; d];
                let mut js: Vec<(usize, f32)> = Vec::new();
                for j in 0..nf {
                    let q = &at[(j * nh + h) * d..][..d];
                    add(&mut s, q);
                    if q.iter().any(|x| *x != 0.0) {
                        js.push((j, pre.strength(&lg[(j * nh + h) * pre.ns..][..pre.ns])));
                    }
                }
                let nh_ = norm(hv);
                if nh_ > 0.0 {
                    let df: Vec<f32> = s.iter().zip(hv).map(|(a, b)| a - b).collect();
                    source_dev = source_dev.max(norm(&df) / nh_);
                }
                js.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let mut rest = hv.to_vec();
                for &(j, _) in js.iter().take(kq) {
                    let q = &at[(j * nh + h) * d..][..d];
                    for (r, x) in rest.iter_mut().zip(q) {
                        *r -= x;
                    }
                    parts.push(Kind::Source(h as u16, j as u32), l, ia, q, ml);
                }
                if js.len() > kq {
                    parts.push(Kind::OtherSources(h as u16, (js.len() - kq) as u32), l, ia, &rest, ml);
                }
            }
        }
        // neurons
        let im = per_layer[l][1].unwrap();
        let act = &tr.focus_act[l];
        let ff = act.len();
        let post = match (b.res == Res::Pre, &b.n_mlp_post, tr.focus_mlp_post[l]) {
            (true, Some(n), Some(s)) => Some((n, s)),
            _ => None,
        };
        let up: Vec<f32> = match post {
            Some((n, s)) => pre.u.chunks(d).flat_map(|r| n.back(s, r)).collect(),
            None => pre.u.clone(),
        };
        let g = b.mlp.down.t_matmul(&up, pre.ns); // [ns][ff]
        let mut nv: Vec<(usize, f32)> = (0..ff)
            .into_par_iter()
            .filter(|&i| act[i] != 0.0)
            .map(|i| {
                let lg: Vec<f32> = (0..pre.ns).map(|k| act[i] * ml * g[k * ff + i]).collect();
                (i, pre.strength(&lg))
            })
            .collect();
        nv.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let sel: Vec<usize> = nv.iter().take(kq).map(|x| x.0).collect();
        let cols = columns(&b.mlp.down, &sel);
        let mut rest = tr.focus_mlp[l].clone();
        for (k, &i) in sel.iter().enumerate() {
            let col = &cols[k * d..(k + 1) * d];
            let mut vv = match post {
                Some((n, s)) => n.map_part(s, col),
                None => col.to_vec(),
            };
            vv.iter_mut().for_each(|x| *x *= act[i]);
            for (r, x) in rest.iter_mut().zip(&vv) {
                *r -= x;
            }
            parts.push(Kind::Neuron(i as u32, act[i]), l, im, &vv, ml);
        }
        if nv.len() > sel.len() {
            parts.push(Kind::OtherNeurons((nv.len() - sel.len()) as u32), l, im, &rest, ml);
        }
        if let Some(bv) = &tr.focus_mlp_bias[l] {
            parts.push(Kind::MlpBias, l, im, bv, ml);
        }
        if let Some(bb) = post_bias(b, false) {
            parts.push(Kind::PostNormBias, l, im, bb, ml);
        }
    }
    let n = parts.parts.len();
    let mut hrows = vec![0f32; n * d];
    hrows.par_chunks_mut(d).enumerate().for_each(|(i, out)| match parts.parts[i].kind {
        Kind::Total => out.copy_from_slice(&h_all),
        Kind::FinalBias => out.copy_from_slice(final_bias.unwrap()),
        _ => out.copy_from_slice(&fmap(parts.row(i))),
    });
    eprintln!("{} voices over {} tokens (full softmax) …", n - 1, v);
    let t1 = Instant::now();
    let (voi, full) = voices(m, &hrows, &z, &p, logz, &ys, na);
    // direct, in the real output: which tokens the voice raises / lowers by how many percentage points
    // (p with vs. p without this voice, for the most likely tokens and the top without it)
    let mut ls = vec![0f32; n * pre.ns];
    blas::gemm_nt(&hrows, &pre.uh, n, pre.ns, d, &mut ls);
    let direct_dp: Vec<(Vec<Shift>, Vec<Shift>)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let s = &voi[i];
            let mut cand: Vec<Shift> = Vec::new();
            for (k, &t) in pre.ids.iter().enumerate() {
                let q = ((z[t as usize] - ls[i * pre.ns + k]) as f64 - s.lse_without).exp() as f32;
                cand.push((t, q, p[t as usize]));
            }
            for &(val, t) in &s.top_without {
                if val.is_finite() && !pre.ids.contains(&t) {
                    cand.push((t, (val as f64 - s.lse_without).exp() as f32, p[t as usize]));
                }
            }
            cand.sort_by(|a, b| (b.2 - b.1).partial_cmp(&(a.2 - a.1)).unwrap_or(std::cmp::Ordering::Equal));
            let raises: Vec<_> = cand.iter().take(3).filter(|x| x.2 - x.1 > 5e-4).cloned().collect();
            let lowers: Vec<_> = cand.iter().rev().take(3).filter(|x| x.1 - x.2 > 5e-4).cloned().collect();
            (raises, lowers)
        })
        .collect();
    let secs_voices = t1.elapsed().as_secs_f32();
    // check: sum of the voice logits = real logits, product of the voices = real distribution
    let mut zsum = vec![0f64; v];
    for k in 0..na {
        for (a, x) in zsum.iter_mut().zip(&full[k * v..(k + 1) * v]) {
            *a += *x as f64;
        }
    }
    let logit_dev = zsum.iter().zip(&z).map(|(a, b)| (a - *b as f64).abs()).fold(0f64, f64::max) as f32;
    let zmax = z.iter().map(|x| x.abs()).fold(0f32, f32::max);
    let kl_product = {
        let mx = zsum.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let lq = mx + zsum.iter().map(|x| (x - mx).exp()).sum::<f64>().ln();
        zsum.iter().zip(&z).map(|(q, zz)| { let lp = *zz as f64 - logz; lp.exp() * (lp - (q - lq)) }).sum::<f64>().max(0.0) as f32
    };

    // ---- normalization: stream at the end without the change, only the final norm (and softcap) recomputed.
    // (log-odds with − without) − direct = what the new scale of the final norm alone accounts for.
    let lo_p: Vec<f64> = ys.iter().map(|&y| log_odds(&z, y as usize)).collect();
    let mut norm_part: Vec<Option<[f32; NY]>> = vec![None; na];
    for chunk in (0..na).collect::<Vec<_>>().chunks(16) {
        let ks: Vec<usize> = chunk.iter().cloned().filter(|&k| parts.parts[changes[k]].kind != Kind::FinalBias).collect();
        if ks.is_empty() {
            continue;
        }
        let mut xs = Vec::with_capacity(ks.len() * d);
        for &k in &ks {
            xs.extend(xf.iter().zip(parts.row(changes[k])).map(|(a, b)| a - b));
        }
        let lg = m.final_logits(&xs, ks.len());
        for (r, &k) in ks.iter().enumerate() {
            let row = &lg[r * v..(r + 1) * v];
            let mut nt = [0f32; NY];
            for (i, &y) in ys.iter().enumerate() {
                nt[i] = (lo_p[i] - log_odds(row, y as usize)) as f32 - voi[changes[k]].dlo[i];
            }
            norm_part[k] = Some(nt);
        }
    }

    // ---- total effect: every change to zero, model rerun from its layer
    let ck = tr.checkpoint.take().ok_or("checkpoints missing")?;
    let mut effs: Vec<Option<Effect>> = vec![None; na];
    let kv = Arc::new(ck.kv.clone());
    let t2 = Instant::now();
    let lm_ = nl / 2;
    let tp = m.run(ids, focus, &RunOpts { start: Some(Start { layer: lm_, x: ck.x[lm_].clone(), kv: kv.clone() }), ..Default::default() });
    let rerun_dev = tp.logits.iter().zip(&z).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    let tty = std::io::stderr().is_terminal();
    let n_todo = changes.iter().filter(|&&r| parts.parts[r].kind != Kind::FinalBias).count();
    let mut n_reruns = 0usize;
    for (k, &ri) in changes.iter().enumerate() {
        let tl = parts.parts[ri];
        let l = tl.l;
        let only = o.focus_only;
        let (layer, x, ablate) = match tl.kind {
            Kind::Embedding => {
                let mut x = ck.x[0].clone();
                if only {
                    x[focus * d..(focus + 1) * d].iter_mut().for_each(|v| *v = 0.0);
                } else {
                    x.iter_mut().for_each(|v| *v = 0.0);
                }
                (0, x, vec![])
            }
            Kind::Attn => (l, ck.x[l].clone(), vec![if only { Ablation::AttnAt(l, focus) } else { Ablation::Attn(l) }]),
            Kind::Mlp => (l, ck.x[l].clone(), vec![if only { Ablation::MlpAt(l, focus) } else { Ablation::Mlp(l) }]),
            Kind::Ple => (l, ck.x[l].clone(), vec![if only { Ablation::PleAt(l, focus) } else { Ablation::Ple(l) }]),
            _ => continue,
        };
        let t2r = m.run(ids, focus, &RunOpts { ablate, start: Some(Start { layer, x, kv: kv.clone() }), ..Default::default() });
        effs[k] = Some(effect(&z, logz, &t2r.logits, &full[k * v..(k + 1) * v], &ys));
        n_reruns += 1;
        if tty {
            eprint!("\rtotal effect: rerun {n_reruns}/{n_todo} ({}) …   ", name(m, ids, &tl));
        }
    }
    if tty {
        eprintln!();
    }
    let secs_reruns = t2.elapsed().as_secs_f32();

    let mut children = vec![Vec::new(); na];
    for (i, tl) in parts.parts.iter().enumerate() {
        if tl.kind != Kind::Total && !tl.kind.is_change() {
            children[tl.change].push(i);
        }
    }
    let fixed: Vec<bool> = changes.iter().map(|&r| parts.parts[r].kind == Kind::FinalBias).collect();
    // check: no later block reads the last change -> total = direct + norm
    let last_dev = (0..na).rev().find(|&k| !fixed[k]).and_then(|k| {
        let w = effs[k].as_ref()?;
        let nt = norm_part[k]?;
        Some((name(m, ids, &parts.parts[changes[k]]), (w.dlo[0] - voi[changes[k]].dlo[0] - nt[0]).abs()))
    });
    let e = Results { voices: voi, effects: effs, children, fixed, direct_dp, norm_part };
    let secs = t0.elapsed().as_secs_f32();
    let checks = Checks { last_dev, stream_dev, logit_dev, zmax, kl_product, source_dev, rerun_dev, n_reruns, secs_reruns, secs_voices, secs, n_voices: n - 1 };
    let ctx = Ctx { m, ids, tr: &tr, focus, parts: &parts, changes: &changes, e: &e, ys: &ys, p: &p, o, hmax: (v as f32).log2(), checks: &checks };
    if let Some(path) = &o.result_json {
        use serde_json::json;
        let mut result = crate::export::result(m, ids, focus, y0, "decompose", &top);
        let node = |i: usize| {
            let part = &parts.parts[i];
            let kind = match part.kind {
                Kind::Embedding => "embedding", Kind::Attn => "attention", Kind::Mlp => "mlp", Kind::Ple => "ple",
                Kind::Head(_) => "head", Kind::Neuron(..) => "neuron", Kind::Source(..) => "source",
                Kind::OtherNeurons(_) | Kind::OtherSources(..) => "remainder", _ => "bias",
            };
            let total = if part.kind.is_change() { effects(&e, i, part.change, 0).1 } else { None };
            let unit = match part.kind { Kind::Neuron(n, _) => Some(n as usize), Kind::Head(h) => Some(h as usize), _ => None };
            let position = match part.kind { Kind::Source(_, pos) => Some(pos), _ => None };
            let mut item = json!({"id":format!("part-{i}"),"label":name(m,ids,part),"layer":part.l,"kind":kind,
                "direct":e.voices[i].dlo[0],"total":total,"children":[],"unit":unit,"position":position,
                "voice":crate::export::probabilities(m,&e.voices[i].top),
                "voice_entropy_bits":e.voices[i].h,"direct_kl":e.voices[i].kl});
            if part.kind.is_change() {
                let shift = |g: &[Shift]| g.iter().map(|&(id, without, with)| json!({
                    "id":id,"text":m.tok.piece(id),"without":without,"with":with})).collect::<Vec<_>>();
                let (gains, losses) = shifts(&e, part.change, i);
                item["raises"] = json!(shift(gains));
                item["lowers"] = json!(shift(losses));
                item["direct_share"] = json!(direct_share(&e, part.change));
                if let Some(w) = &e.effects[part.change] {
                    item["kl"] = json!(w.kl);
                    item["output_without"] = json!(crate::export::probabilities(m, &w.top_without));
                }
                item["indirect_norm"] = json!(e.norm_part[part.change].map(|v| v[0]));
            }
            if let Kind::Head(h) = part.kind {
                let (pf, tt) = (&tr.attn_focus[part.l], ids.len());
                item["attention"] = json!((0..tt).map(|j| pf[h as usize * tt + j]).collect::<Vec<_>>());
            }
            item
        };
        let components: Vec<_> = changes.iter().map(|&i| {
            let mut item = node(i);
            let indices = &e.children[parts.parts[i].change];
            let children: Vec<_> = indices.iter().filter(|&&j| !matches!(parts.parts[j].kind, Kind::Source(..) | Kind::OtherSources(..))).map(|&j| {
                let mut child = node(j);
                if let Kind::Head(head) = parts.parts[j].kind {
                    let sources: Vec<_> = indices.iter().filter(|&&k| matches!(parts.parts[k].kind, Kind::Source(h, _) | Kind::OtherSources(h, _) if h == head)).map(|&k| node(k)).collect();
                    child["children"] = json!(sources);
                }
                child
            }).collect();
            item["children"] = json!(children);
            item
        }).collect();
        result["components"] = json!(components);
        result["checks"] = json!([
            {"label":"Residual reconstruction", "deviation":stream_dev},
            {"label":"Logit reconstruction", "deviation":logit_dev},
            {"label":"Distribution reconstruction KL", "deviation":kl_product},
            {"label":"Attention source reconstruction", "deviation":source_dev},
            {"label":"Unchanged rerun", "deviation":rerun_dev}
        ]);
        result["intervention_scope"] = json!(if o.focus_only { "focus" } else { "all_positions" });
        result["report"] = json!(o.html.file_name().map(|x| x.to_string_lossy().to_string()));
        crate::export::write(path, &result)?;
    }
    terminal(&ctx, st);
    let html = html(&ctx);
    std::fs::write(&o.html, html).map_err(|er| format!("{}: {er}", o.html.display()))?;
    println!();
    println!("  {} {}", st.bold("Everything expandable (heads → source positions, neurons):"), o.html.display());
    Ok(())
}

struct Checks {
    last_dev: Option<(String, f32)>,
    stream_dev: f32,
    logit_dev: f32,
    zmax: f32,
    kl_product: f32,
    source_dev: f32,
    rerun_dev: f32,
    n_reruns: usize,
    secs_reruns: f32,
    secs_voices: f32,
    secs: f32,
    n_voices: usize,
}

struct Ctx<'a> {
    m: &'a Model,
    ids: &'a [u32],
    tr: &'a Trace,
    focus: usize,
    parts: &'a Parts,
    changes: &'a [usize],
    e: &'a Results,
    ys: &'a [u32],
    p: &'a [f32],
    o: &'a Opts,
    hmax: f32,
    checks: &'a Checks,
}

fn voice_txt(m: &Model, s: &Voice, n: usize) -> String {
    s.top[..n].iter().map(|(id, pr)| format!("{} {}", tok(m, *id), pct(*pr))).collect::<Vec<_>>().join(" · ")
}

fn shift_txt(m: &Model, g: &Shift) -> String {
    format!("{} {}", tok(m, g.0), pp(g.2 - g.1))
}

/// "raises " the" +9 pp, lowers " capital" −3 pp" (direct, in the real output)
fn dp_txt(m: &Model, dp: &(Vec<Shift>, Vec<Shift>)) -> String {
    let f = |v: &[Shift]| if v.is_empty() { "–".to_string() } else { v.iter().take(2).map(|g| shift_txt(m, g)).collect::<Vec<_>>().join(", ") };
    format!("raises {} · lowers {}", f(&dp.0), f(&dp.1))
}

/// Gains/losses of a change: from its total effect, or (final-norm bias) from its direct voice.
fn shifts<'a>(e: &'a Results, k: usize, ri: usize) -> (&'a [Shift], &'a [Shift]) {
    match &e.effects[k] {
        Some(w) => (&w.gains, &w.losses),
        None if e.fixed[k] => (&e.direct_dp[ri].0, &e.direct_dp[ri].1),
        None => (&[], &[]),
    }
}

/// where head h of layer l looks from the focus token, strongest first
fn head_targets(c: &Ctx, l: usize, h: usize) -> Vec<(usize, f32)> {
    let pf = &c.tr.attn_focus[l];
    let tt = c.ids.len();
    let mut js: Vec<(usize, f32)> = (0..tt).map(|j| (j, pf[h * tt + j])).filter(|x| x.1 > 0.0).collect();
    js.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    js
}

// ------------------------------------------------------------------ terminal

fn terminal(c: &Ctx, st: &Style) {
    let m = c.m;
    let e = c.e;
    let ck = c.checks;
    let y0 = c.ys[0];
    let ty = tok(m, y0);
    println!();
    println!("{}", st.bold(&format!("Decomposition: what every change of the stream at [{}]{} contributes – in token probabilities", c.focus, tok(m, c.ids[c.focus]))));
    println!("  output: {}", c.ys.iter().map(|&y| format!("{} {}", tok(m, y), pct(c.p[y as usize]))).collect::<Vec<_>>().join(" · "));
    for s in [
        "Every change of the residual stream (embedding, every attention, every MLP) goes through the final norm on its own – with",
        "the scale from exactly this pass, the same for all – and through the output matrix; its softmax = its direct voice.",
    ] {
        println!("  {}", st.dim(s));
    }
    if let Some(cp) = m.logit_cap {
        println!("  {}", st.dim(&format!("Softcap {}: every token logit scaled by the fixed factor softcap(z)/z from this pass (like the norm).", num(cp, 0))));
    }
    println!(
        "  checks: Σ changes = stream at the end (dev. {}) · Σ voice logits = real logits (max |Δ| {} at |z| ≤ {}) · normalized product of all voices = real distribution (KL {})",
        sci(ck.stream_dev),
        sci(ck.logit_dev),
        num(ck.zmax, 1),
        sci(ck.kl_product)
    );
    println!("          Σ sources = head (max dev. {}) · rerun from the middle without ablation = real pass (max |Δ logit| {})", sci(ck.source_dev), sci(ck.rerun_dev));
    if let Some((n, a)) = &ck.last_dev {
        println!("          no later block reads {n}: total = direct + norm (dev. {} in log-odds)", sci(*a));
    }
    let wo = if c.o.focus_only { format!("only at token [{}]", c.focus) } else { "at all positions".into() };
    println!("  total effect: every change set to zero {wo}, model rerun from its layer ({} runs, {} s)", ck.n_reruns, num(ck.secs_reruns, 1));
    println!();
    let ws = st.width.saturating_sub(108).clamp(16, 44);
    let head = format!(
        "  {} {} {} │ {} {} {} {} │ {} {} │ {} {}",
        pad("change", 13),
        pad("direct voice (top 3)", ws),
        padl("H/bit", 5),
        padl("direct", 8),
        padl("total", 9),
        padl("indirect", 9),
        padl("via norm", 9),
        padl("KL", 5),
        padl("dir.", 5),
        pad("gains through it", 18),
        pad("loses through it", 18)
    );
    println!("{}", st.dim(&format!("  {} factor on the odds p/(1−p) of {} vs. without the change; \"via norm\" = part of indirect that only comes from rescaling the final norm", " ".repeat(15 + ws + 6), ty)));
    println!("{}", st.bold(&head));
    for (k, &ri) in c.changes.iter().enumerate() {
        let tl = &c.parts.parts[ri];
        let s = &e.voices[ri];
        let (dd, gs, is) = effects(e, ri, k, 0);
        let gst = gs.map_or("–".to_string(), factor);
        let ist = is.map_or("–".to_string(), factor);
        let color = |txt: String, x: Option<f32>| -> String {
            match x {
                Some(x) if x > 0.693 => st.green(&txt),
                Some(x) if x < -0.693 => st.red(&txt),
                _ => txt,
            }
        };
        let (gains, losses) = shifts(e, k, ri);
        let gain = gains.first().map_or(String::new(), |g| shift_txt(m, g));
        let loss = losses.first().map_or(String::new(), |g| shift_txt(m, g));
        let kl = e.effects[k].as_ref().map_or(if e.fixed[k] { num(s.kl, 2) } else { String::new() }, |w| num(w.kl, 2));
        let share = direct_share(e, k).map_or("–".to_string(), |a| format!("{}%", num(100.0 * a, 0)));
        let vt = trunc(&voice_txt(m, s, 3), ws);
        let vlen = vt.chars().count();
        let vt = if s.h > c.hmax - 0.3 { st.dim(&vt) } else { vt };
        let nt = e.norm_part[k].map(|v| v[0]);
        println!(
            "  {} {} {} │ {} {} {} {} │ {} {} │ {} {}",
            pad(&name(m, c.ids, tl), 13),
            vt + &" ".repeat(ws.saturating_sub(vlen)),
            padl(&num(s.h, 1), 5),
            color(padl(&factor(dd), 8), Some(dd)),
            color(padl(&gst, 9), gs),
            color(padl(&ist, 9), is),
            st.dim(&padl(&nt.map_or("–".to_string(), factor), 9)),
            padl(&kl, 5),
            padl(&share, 5),
            pad(&trunc(&gain, 18), 18),
            pad(&trunc(&loss, 18), 18)
        );
    }
    // summary
    let ab: Vec<usize> = (0..c.changes.len()).filter(|&k| c.parts.parts[c.changes[k]].kind != Kind::FinalBias).collect();
    let mut by_total = ab.clone();
    by_total.sort_by(|a, b| total_strength(e, *b).partial_cmp(&total_strength(e, *a)).unwrap_or(std::cmp::Ordering::Equal));
    let mut by_direct = ab.clone();
    by_direct.sort_by(|a, b| by_abs_desc(e.voices[c.changes[*a]].dlo[0], e.voices[c.changes[*b]].dlo[0]));
    let list = |v: &[usize], f: &dyn Fn(usize) -> String| v.iter().map(|&k| format!("{} ({})", name(m, c.ids, &c.parts.parts[c.changes[k]]), f(k))).collect::<Vec<_>>().join(" · ");
    println!();
    println!("  {} {}", st.bold(&format!("Strongest direct (odds of {ty}):")), list(&by_direct[..6.min(by_direct.len())], &|k| factor(e.voices[c.changes[k]].dlo[0])));
    println!("  {} {}", st.bold("Strongest overall:"), list(&by_total[..6.min(by_total.len())], &|k| format!("KL {}", num(total_strength(e, k), 2))));
    let hidden: Vec<usize> = by_total.iter().cloned().filter(|&k| direct_share(e, k).is_some_and(|a| a < 0.25)).take(6).collect();
    if !hidden.is_empty() {
        println!(
            "  {} {}",
            st.bold("Hidden partial beliefs (act through later blocks, hardly directly):"),
            list(&hidden, &|k| format!("{} direct", direct_share(e, k).map_or("–".into(), |a| format!("{}%", num(100.0 * a, 0)))))
        );
    }
    // the 4 strongest changes expanded
    println!();
    println!("  {}", st.bold("The strongest changes expanded (everything else in the HTML file):"));
    for &k in by_total.iter().take(4) {
        let ri = c.changes[k];
        let tl = &c.parts.parts[ri];
        let (dd, gs, _) = effects(e, ri, k, 0);
        println!(
            "  ▸ {} – for {}: direct {}, total {}{}",
            st.bold(&name(m, c.ids, tl)),
            ty,
            factor(dd),
            gs.map_or("–".into(), factor),
            e.effects[k].as_ref().map_or(String::new(), |w| format!(" · without it: {}", w.top_without.iter().take(3).map(|(id, q)| format!("{} {}", tok(m, *id), pct(*q))).collect::<Vec<_>>().join(" · ")))
        );
        println!("      {}", st.dim(&format!("direct {}", dp_txt(m, &e.direct_dp[ri]))));
        if let Some(w) = &e.effects[k] {
            let g: Vec<String> = w.gains.iter().take(3).map(|g| shift_txt(m, g)).collect();
            let v: Vec<String> = w.losses.iter().take(3).map(|g| shift_txt(m, g)).collect();
            println!("      {}", st.dim(&format!("total raises {} · lowers {}", if g.is_empty() { "–".into() } else { g.join(", ") }, if v.is_empty() { "–".into() } else { v.join(", ") })));
        }
        let mut kids: Vec<usize> = e.children[k].iter().cloned().filter(|&i| !matches!(c.parts.parts[i].kind, Kind::Source(..) | Kind::OtherSources(..))).collect();
        kids.sort_by(|a, b| by_abs_desc(e.voices[*a].dlo[0], e.voices[*b].dlo[0]));
        for &i in kids.iter().take(3) {
            let t = &c.parts.parts[i];
            let s = &e.voices[i];
            let mut line = format!("      {} {}  H {} · direct {} · {}", pad(&name(m, c.ids, t), 18), trunc(&voice_txt(m, s, 3), 46), num(s.h, 1), factor(s.dlo[0]), dp_txt(m, &e.direct_dp[i]));
            if let Kind::Head(h) = t.kind {
                let js = head_targets(c, t.l, h as usize);
                line += &st.dim(&format!(" · looks at {}", js.iter().take(3).map(|(j, a)| format!("[{j}]{} {}", tok(m, c.ids[*j]), pct(*a))).collect::<Vec<_>>().join(" · ")));
            }
            println!("{line}");
            if let Kind::Head(h) = t.kind {
                let mut qs: Vec<usize> = e.children[k].iter().cloned().filter(|&q| matches!(c.parts.parts[q].kind, Kind::Source(hh, _) if hh == h)).collect();
                qs.sort_by(|a, b| by_abs_desc(e.voices[*a].dlo[0], e.voices[*b].dlo[0]));
                for &q in qs.iter().take(2) {
                    let s = &e.voices[q];
                    println!("        {} {}  H {} · direct {} · {}", st.dim(&pad(&name(m, c.ids, &c.parts.parts[q]), 22)), trunc(&voice_txt(m, s, 3), 42), num(s.h, 1), factor(s.dlo[0]), dp_txt(m, &e.direct_dp[q]));
                }
            }
        }
    }
    println!();
    println!(
        "  {}",
        st.dim(&format!(
            "{} voices in {} s, total {} s. Reruns start at the layer of the change; \"direct\" = by which factor the odds of {} are higher than without this voice alone.",
            ck.n_voices,
            num(ck.secs_voices, 1),
            num(ck.secs, 0),
            ty
        ))
    );
}

// ------------------------------------------------------------------ HTML

fn shift_html(m: &Model, g: &Shift, cls: &str) -> String {
    format!("<span class=\"st\">{}<span class=\"{cls}\">{}</span></span>", chip(m, g.0), esc_html(&pp(g.2 - g.1)))
}

fn dp_html(m: &Model, dp: &(Vec<Shift>, Vec<Shift>), n: usize, show_empty: bool) -> String {
    let mut o = String::new();
    for g in dp.0.iter().take(n) {
        o += &shift_html(m, g, "pos");
    }
    for g in dp.1.iter().take(n) {
        o += &shift_html(m, g, "neg");
    }
    if o.is_empty() {
        if !show_empty {
            return String::new();
        }
        o = "<span class=\"mu\">moves no candidate by 0.05 pp</span>".into();
    }
    format!("<span class=\"dp\">direct: {o}</span>")
}

fn voice_html(m: &Model, s: &Voice, n: usize) -> String {
    s.top[..n].iter().map(|(id, pr)| format!("<span class=\"st\">{}<span class=\"pc\">{}</span></span>", chip(m, *id), esc_html(&pct(*pr)))).collect::<Vec<_>>().join("")
}

fn opt_factor_html(x: Option<f32>) -> String {
    x.map_or("<span class=\"mu\">–</span>".into(), factor_html)
}

fn sharpness(s: &Voice, hmax: f32) -> String {
    let sh = (1.0 - s.h / hmax).clamp(0.0, 1.0);
    format!("<span class=\"num\" title=\"entropy {} bit of max. {} bit\">{}<span class=\"meter\"><span style=\"width:{:.0}%\"></span></span></span>", num(s.h, 2), num(hmax, 1), num(s.h, 1), 100.0 * sh)
}

/// one row for a part (head, source, neuron …)
fn part_row(c: &Ctx, i: usize, extra: &str) -> String {
    let s = &c.e.voices[i];
    format!(
        "<span class=\"nm\">{}</span><span class=\"vc\">{}{}</span>{}<span class=\"num\">{}</span><span class=\"num\">{}</span><span class=\"x\">{}</span>",
        esc_html(&name(c.m, c.ids, &c.parts.parts[i])),
        voice_html(c.m, s, 3),
        dp_html(c.m, &c.e.direct_dp[i], 2, false),
        sharpness(s, c.hmax),
        esc_html(&num(s.kl, 3)),
        factor_html(s.dlo[0]),
        extra
    )
}

fn html(c: &Ctx) -> String {
    let m = c.m;
    let e = c.e;
    let ck = c.checks;
    let y0 = c.ys[0];
    let mut h = String::with_capacity(1 << 20);
    h += &format!(
        r#"<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Decomposition of the pass</title><style>{CSS}{DECOMPOSE_CSS}</style></head><body><main>
<h1>Decomposition of the pass</h1>
<p class="meta">{} · {} layers · focus token [{}]</p>
<p class="prompt">{}</p>
<p>Output: {}</p>
"#,
        esc_html(&m.name),
        m.n_layer(),
        c.focus,
        prompt_html(m, c.ids, c.focus),
        c.ys.iter().map(|&y| format!("<span class=\"st\">{}<span class=\"pc\">{}</span></span>", chip(m, y), esc_html(&pct(c.p[y as usize])))).collect::<String>()
    );
    let wo = if c.o.focus_only { format!("only at token [{}]", c.focus) } else { "at all positions".into() };
    let total_txt = format!("exact: every change set to zero {wo}, model rerun from its layer ({} runs, {} s)", ck.n_reruns, num(ck.secs_reruns, 1));
    h += &format!(
        r#"<details class="info"><summary>How to read this</summary><div>
<p>At the end, the stream at the focus token is exactly the sum of its changes. Every change goes through the final norm on its own – with the scale from exactly this pass, the same for all – and through the output matrix{}. Its softmax is its <b>direct voice</b>. H is the entropy of the voice: {} bit means completely flat, i.e. the change hardly votes for tokens directly; the bar shows the sharpness.</p>
<p><b>direct</b>: by which factor the odds p/(1−p) of {} are higher than without this voice alone (for small p practically the factor on p; at p ≈ 1 this keeps the lead over the competition visible). Parts are sorted by this direct effect. <b>total</b> – {total_txt}. <b>indirect</b> = total ÷ direct: what acts through later blocks <i>and</i> the fact that the final norm rescales without the change. <b>via norm</b> is that second part alone (stream at the end without the change, only the final norm{} recomputed); the rest of indirect runs through later blocks. <b>KL</b>: how much the whole distribution changes (nats). <b>direct share</b>: spread of the direct vs. the indirect log change, weighted over the tokens that carry weight with or without the change.</p>
<p>Every row expands: attention into heads (with attention pattern) and heads into source positions (a<sub>j</sub>·W<sub>O</sub>v<sub>j</sub>), MLP into neurons (a<sub>i</sub>·W<sub>down</sub>[:,i]). Shown are the {} strongest sources per head and neurons per MLP (preselected over the {NS} most likely tokens), the rest as one row – together the parts give exactly the change.</p>
</div></details>
<p class="check">Checks: Σ changes = stream at the end (dev. {}) · Σ voice logits = real logits (max |Δ| {}) · normalized product of all voices = real distribution (KL {}) · Σ sources = head (dev. {}) · rerun without ablation = real pass (|Δ logit| {}){}</p>
"#,
        if m.logit_cap.is_some() { "; the softcap enters per token as the fixed factor softcap(z)/z from this pass" } else { "" },
        num(c.hmax, 1),
        esc_html(&tok(m, y0)),
        if m.logit_cap.is_some() { " and the softcap" } else { "" },
        c.o.k,
        esc_html(&sci(ck.stream_dev)),
        esc_html(&sci(ck.logit_dev)),
        esc_html(&sci(ck.kl_product)),
        esc_html(&sci(ck.source_dev)),
        esc_html(&sci(ck.rerun_dev)),
        ck.last_dev.as_ref().map_or(String::new(), |(n, a)| format!(" · no later block reads {}: total = direct + norm (dev. {})", esc_html(n), esc_html(&sci(*a))))
    );
    h += &format!(
        r#"<div class="tab"><div class="row hd"><span>change</span><span>direct voice</span><span class="num">H/bit</span><span class="num">direct</span><span class="num">total</span><span class="num">indirect</span><span class="num">via norm</span><span class="num">KL</span><span class="num">direct share</span><span>gains through it</span><span>loses through it</span></div>
<div class="row sub"><span></span><span></span><span></span><span class="num" style="grid-column:span 4">factor on the odds of {}</span><span></span><span></span><span></span><span></span></div>
"#,
        esc_html(&tok(m, y0))
    );
    let total_max = (0..c.changes.len()).map(|k| total_strength(e, k)).fold(1e-9f32, f32::max);
    for (k, &ri) in c.changes.iter().enumerate() {
        let tl = &c.parts.parts[ri];
        let s = &e.voices[ri];
        let (dd, gs, is) = effects(e, ri, k, 0);
        let (gains, losses) = shifts(e, k, ri);
        let gain: String = gains.iter().take(2).map(|g| shift_html(m, g, "pos")).collect();
        let loss: String = losses.iter().take(2).map(|g| shift_html(m, g, "neg")).collect();
        let bar = ((total_strength(e, k) / total_max).sqrt() * 100.0).clamp(0.0, 100.0);
        let share = direct_share(e, k);
        h += &format!(
            r#"<details class="change"><summary class="row"><span class="nm" style="--w:{bar:.0}%" title="bar: total effect">{}</span><span class="vc{}">{}{}</span>{}<span class="num">{}</span><span class="num">{}</span><span class="num">{}</span><span class="num">{}</span><span class="num">{}</span><span class="num">{}</span><span>{gain}</span><span>{loss}</span></summary><div class="expanded">"#,
            esc_html(&name(m, c.ids, tl)),
            if s.h > c.hmax - 0.3 { " flat" } else { "" },
            voice_html(m, s, 3),
            dp_html(m, &e.direct_dp[ri], 2, false),
            sharpness(s, c.hmax),
            factor_html(dd),
            opt_factor_html(gs),
            opt_factor_html(is),
            e.norm_part[k].map_or("<span class=\"mu\">–</span>".to_string(), |v| format!("<span class=\"mu\">{}</span>", esc_html(&factor(v[0])))),
            e.effects[k].as_ref().map_or(if e.fixed[k] { num(s.kl, 2) } else { "–".into() }, |w| num(w.kl, 2)),
            share.map_or("–".into(), |a| format!("<span class=\"share\"><span style=\"width:{:.0}%\"></span></span>{}%", 100.0 * a, num(100.0 * a, 0)))
        );
        // header of the expanded change
        h += "<div class=\"grid2\"><div><h4>Direct voice (top 5)</h4>";
        h += &voice_html(m, s, K);
        h += &format!("<p class=\"mu\">entropy {} bit · without this voice alone: KL {} nats</p>", num(s.h, 2), num(s.kl, 3));
        h += &format!("<p>{}</p></div>", dp_html(m, &e.direct_dp[ri], 3, true));
        h += "<div><h4>Effect on the candidates</h4><table class=\"tbl\"><tr><th></th><th>p</th><th>direct</th><th>total</th><th>indirect</th><th>via norm</th></tr><tr><td></td><td></td><td colspan=\"4\" class=\"mu\" style=\"font-size:11px\">factor on the odds p/(1−p)</td></tr>";
        for (i, &y) in c.ys.iter().enumerate() {
            let (dd, gs, is) = effects(e, ri, k, i);
            let nt = e.norm_part[k].map_or("–".to_string(), |v| factor(v[i]));
            h += &format!(
                "<tr><td>{}</td><td class=\"num\">{}</td><td class=\"num\">{}</td><td class=\"num\">{}</td><td class=\"num\">{}</td><td class=\"num mu\">{}</td></tr>",
                chip(m, y),
                esc_html(&pct(c.p[y as usize])),
                factor_html(dd),
                opt_factor_html(gs),
                opt_factor_html(is),
                esc_html(&nt)
            );
        }
        h += "</table></div>";
        if let Some(w) = &e.effects[k] {
            h += "<div><h4>Total effect: without → with</h4><table class=\"tbl\">";
            for (g, cls) in w.gains.iter().map(|g| (g, "pos")).chain(w.losses.iter().map(|g| (g, "neg"))) {
                h += &format!("<tr><td>{}</td><td class=\"num\">{} → {}</td><td class=\"num {cls}\">{}</td></tr>", chip(m, g.0), esc_html(&pct(g.1)), esc_html(&pct(g.2)), esc_html(&pp(g.2 - g.1)));
            }
            h += &format!("</table><p class=\"mu\">output without it: {}</p></div>", w.top_without.iter().map(|(id, q)| format!("{} {}", chip(m, *id), esc_html(&pct(*q)))).collect::<Vec<_>>().join(" · "));
        }
        h += "</div>";
        // parts
        let kids = &e.children[k];
        if !kids.is_empty() {
            let mut main_parts: Vec<usize> = kids.iter().cloned().filter(|&i| !matches!(c.parts.parts[i].kind, Kind::Source(..) | Kind::OtherSources(..))).collect();
            main_parts.sort_by(|a, b| by_abs_desc(e.voices[*a].dlo[0], e.voices[*b].dlo[0]));
            let title = match tl.kind {
                Kind::Attn => "Heads (expandable: attention pattern and source positions)",
                Kind::Mlp => "Neurons (strongest by direct effect)",
                _ => "Parts",
            };
            h += &format!(
                r#"<h4>{title}</h4><div class="parts"><div class="row2 hd"><span>part</span><span>direct voice</span><span class="num">H/bit</span><span class="num">KL dir.</span><span class="num">direct</span><span>{}</span></div>"#,
                if tl.kind == Kind::Attn { "looks at" } else { "" }
            );
            for &i in &main_parts {
                let t = &c.parts.parts[i];
                if let Kind::Head(hh) = t.kind {
                    let pf = &c.tr.attn_focus[t.l];
                    let tt = c.ids.len();
                    let short: String = head_targets(c, t.l, hh as usize).iter().take(2).map(|(j, a)| format!("<span class=\"st\">[{j}]{}<span class=\"pc\">{}</span></span>", chip(m, c.ids[*j]), esc_html(&pct(*a)))).collect();
                    h += &format!("<details class=\"head\"><summary class=\"row2\">{}</summary><div class=\"head-expanded\">", part_row(c, i, &short));
                    h += "<h5>Attention pattern (where the head looks from the focus token)</h5><div class=\"pattern\">";
                    let mut all: Vec<(usize, f32)> = (0..tt).map(|j| (j, pf[hh as usize * tt + j])).collect();
                    all.retain(|x| x.1 >= 0.005);
                    for (j, a) in all {
                        h += &format!("<div class=\"pz\"><span class=\"pi\">[{j}]</span>{}<span class=\"mb\"><span style=\"width:{:.1}%\"></span></span><span class=\"num\">{}</span></div>", chip(m, c.ids[j]), 100.0 * a, esc_html(&pct(a)));
                    }
                    h += "</div><h5>Source positions: what the head votes for because of token j</h5><div class=\"parts\">";
                    let mut qs: Vec<usize> = kids.iter().cloned().filter(|&q| matches!(c.parts.parts[q].kind, Kind::Source(x, _) | Kind::OtherSources(x, _) if x == hh)).collect();
                    qs.sort_by(|a, b| by_abs_desc(e.voices[*a].dlo[0], e.voices[*b].dlo[0]));
                    for q in qs {
                        let extra = match c.parts.parts[q].kind {
                            Kind::Source(_, j) => format!("<span class=\"mu\">attention {}</span>", esc_html(&pct(pf[hh as usize * tt + j as usize]))),
                            _ => String::new(),
                        };
                        h += &format!("<div class=\"row2\">{}</div>", part_row(c, q, &extra));
                    }
                    h += "</div></div></details>";
                } else {
                    h += &format!("<div class=\"row2\">{}</div>", part_row(c, i, ""));
                }
            }
            h += "</div>";
        }
        h += "</div></details>\n";
    }
    h += &format!(
        "</div><p class=\"mu foot\">{} voices over {} tokens in {} s · total {} s · apollon</p></main></body></html>\n",
        ck.n_voices,
        c.p.len(),
        num(ck.secs_voices, 1),
        num(ck.secs, 0)
    );
    h
}

const DECOMPOSE_CSS: &str = r#"
.tab{overflow-x:auto;border:1px solid var(--line);border-radius:8px;background:var(--card);margin-top:12px}
.row{display:grid;grid-template-columns:9.5rem minmax(16rem,2.4fr) 4rem 4.4rem 5rem 5rem 4.6rem 3.2rem 5.6rem minmax(8.5rem,1fr) minmax(8.5rem,1fr);gap:0 10px;align-items:center;padding:5px 12px;min-width:1260px}
.row.hd{font-weight:600;font-size:12px;color:var(--mu);border-bottom:1px solid var(--line)}
.row.sub{font-size:11px;color:var(--mu);padding-top:0}
details.change{border-top:1px solid var(--line)}
details.change>summary:hover,details.head>summary:hover{background:var(--chip)}
details.change[open]>summary{background:var(--chip)}
.nm{font-weight:600;white-space:nowrap}
.row .nm{background:linear-gradient(90deg,var(--kb) var(--w,0%),transparent var(--w,0%));border-radius:3px;padding:2px 6px;margin-left:-6px}
.vc{min-width:0}
.vc.flat>.st{opacity:.45}
.dp{display:block;font-size:12px;color:var(--mu);margin-top:1px}
.dp .tok{font-size:11.5px}
.meter{display:inline-block;width:28px;height:5px;background:var(--line);border-radius:3px;margin-left:5px;vertical-align:middle;overflow:hidden}
.meter span{display:block;height:100%;background:var(--acc)}
.share{display:inline-block;width:34px;height:5px;background:var(--line);border-radius:3px;margin-right:5px;vertical-align:middle;overflow:hidden}
.share span{display:block;height:100%;background:var(--pos)}
.expanded{padding:4px 16px 16px 24px;min-width:1260px;border-top:1px dashed var(--line)}
.grid2{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}
.parts{border:1px solid var(--line);border-radius:6px}
.row2{display:grid;grid-template-columns:13rem minmax(16rem,2fr) 4.2rem 4rem 4.6rem minmax(12rem,1.4fr);gap:0 12px;align-items:center;padding:3px 10px;border-top:1px solid var(--line)}
.parts>.row2:first-child,.parts>details:first-child>.row2{border-top:0}
.row2.hd{font-size:12px;color:var(--mu);font-weight:600}
.head-expanded{padding:4px 12px 12px 24px}
.pattern{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:2px 16px}
.pz{display:grid;grid-template-columns:2.6rem minmax(0,7rem) 1fr 3.4rem;gap:6px;align-items:center;font-size:12.5px}
.pi{color:var(--mu);font-variant-numeric:tabular-nums}
.x{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
"#;
