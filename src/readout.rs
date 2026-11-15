//! Reading vectors in the residual stream as token distributions, streamed over the whole
//! vocabulary in blocks (the output matrix is never held dequantized in memory).
use crate::blas;
use crate::model::Model;
use crate::tensor::{dot, Mat};
use rayon::prelude::*;

/// Top tokens kept per voice.
pub const K: usize = 5;
/// Maximum number of target tokens whose log-odds are tracked.
pub const NY: usize = 5;

/// Direct voice of one vector in the stream.
#[derive(Clone, Copy, Default)]
pub struct Voice {
    /// softmax of this vector's logits alone
    pub top: [(u32, f32); K],
    /// entropy in bits
    pub h: f32,
    /// KL(real output ‖ output without this voice), nats
    pub kl: f32,
    /// direct effect: log-odds ln(p/(1−p)) of target y with minus without this voice
    pub dlo: [f32; NY],
    /// ln Σ e^(z−l): normalizer of the output without this voice
    pub lse_without: f64,
    /// most likely tokens without this voice (value z−l)
    pub top_without: [(f32, u32); 3],
}

/// Running quantities of one row, accumulated block by block over the vocabulary.
#[derive(Clone, Copy)]
struct Acc {
    m: f32,
    a: f64, // Σ e^(l−m)
    b: f64, // Σ e^(l−m)·(l−m)
    m2: f32,
    a2: f64,        // Σ e^(z−l−m2)
    a2x: [f64; NY], // the same sum without token y_i (for the log-odds, no cancellation)
    pl: f64,        // Σ p·l
    top: [(f32, u32); K],
    ly: [f32; NY],
    oq: [(f32, u32); 3],
}

/// Fast sum in 16 lanes.
pub fn sum(v: &[f32]) -> f32 {
    let mut acc = [0f32; 16];
    let ch = v.chunks_exact(16);
    let rest: f32 = ch.remainder().iter().sum();
    for c in ch {
        for j in 0..16 {
            acc[j] += c[j];
        }
    }
    acc.iter().sum::<f32>() + rest
}

/// Insert the values of `row` (vocabulary offset v0) into a descending top-N list.
pub fn top_insert<const N: usize>(top: &mut [(f32, u32); N], row: &[f32], v0: usize) {
    let bm = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut thr = top[N - 1].0;
    if bm > thr {
        for (i, &x) in row.iter().enumerate() {
            if x > thr {
                let mut j = N - 1;
                while j > 0 && top[j - 1].0 < x {
                    top[j] = top[j - 1];
                    j -= 1;
                }
                top[j] = (x, (v0 + i) as u32);
                thr = top[N - 1].0;
            }
        }
    }
}

impl Acc {
    fn new() -> Self {
        Acc { m: f32::NEG_INFINITY, a: 0.0, b: 0.0, m2: f32::NEG_INFINITY, a2: 0.0, a2x: [0.0; NY], pl: 0.0, top: [(f32::NEG_INFINITY, 0); K], ly: [0.0; NY], oq: [(f32::NEG_INFINITY, 0); 3] }
    }
    #[allow(clippy::too_many_arguments)]
    fn update(&mut self, row: &[f32], v0: usize, z: &[f32], p: &[f32], ys: &[u32], t1: &mut [f32], t2: &mut [f32]) {
        let n = row.len();
        for (i, &y) in ys.iter().enumerate() {
            let y = y as usize;
            if y >= v0 && y < v0 + n {
                self.ly[i] = row[y - v0];
            }
        }
        let bm = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        top_insert(&mut self.top, row, v0);
        let mn = self.m.max(bm);
        for (o, x) in t1.iter_mut().zip(row) {
            *o = x - mn;
        }
        blas::exp(t2, t1);
        let sa = sum(t2) as f64;
        let sb = dot(t2, t1) as f64;
        if self.a > 0.0 {
            let dm = (self.m - mn) as f64;
            let r = dm.exp();
            self.b = r * (self.b + dm * self.a);
            self.a *= r;
        }
        self.a += sa;
        self.b += sb;
        self.m = mn;
        self.pl += dot(row, p) as f64;
        for i in 0..n {
            t1[i] = z[i] - row[i];
        }
        top_insert(&mut self.oq, t1, v0);
        let bm2 = t1.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mn2 = self.m2.max(bm2);
        for x in t1.iter_mut() {
            *x -= mn2;
        }
        blas::exp(t2, t1);
        let s2 = sum(t2) as f64;
        let r = ((self.m2 - mn2) as f64).exp();
        self.a2 = self.a2 * r + s2;
        for (i, &y) in ys.iter().enumerate() {
            let y = y as usize;
            let sx = if y >= v0 && y < v0 + n {
                let j = y - v0;
                (sum(&t2[..j]) + sum(&t2[j + 1..])) as f64
            } else {
                s2
            };
            self.a2x[i] = self.a2x[i] * r + sx;
        }
        self.m2 = mn2;
    }
    fn finish(&self, logz: f64, lo_p: &[f64], zy: &[f32]) -> Voice {
        let lz = self.m as f64 + self.a.ln();
        let lse2 = self.m2 as f64 + self.a2.ln();
        let mut s = Voice { h: ((self.a.ln() - self.b / self.a) / std::f64::consts::LN_2) as f32, kl: ((self.pl - logz + lse2) as f32).max(0.0), lse_without: lse2, top_without: self.oq, ..Default::default() };
        for (o, (l, id)) in s.top.iter_mut().zip(self.top) {
            *o = (id, (l as f64 - lz).exp() as f32);
        }
        for i in 0..lo_p.len() {
            // log-odds without this voice: (z_y − l_y) − ln Σ_{t≠y} e^(z_t − l_t)
            let lo_q = (zy[i] - self.ly[i]) as f64 - (self.m2 as f64 + self.a2x[i].max(1e-300).ln());
            s.dlo[i] = (lo_p[i] - lo_q) as f32;
        }
        s
    }
}

/// Send all rows h [n][d] (already mapped through the final norm) through the output matrix.
/// Row 0 is the whole stream (it supplies the softcap factors); rows 1..=n_full are also returned
/// as full logit rows. z, p, logz: real logits, probabilities and their log-normalizer.
pub fn voices(m: &Model, h: &[f32], z: &[f32], p: &[f32], logz: f64, ys: &[u32], n_full: usize) -> (Vec<Voice>, Vec<f32>) {
    let lo_p: Vec<f64> = ys.iter().map(|&y| log_odds(z, y as usize)).collect();
    let zy: Vec<f32> = ys.iter().map(|&y| z[y as usize]).collect();
    let d = m.d;
    let n = h.len() / d;
    let lm = m.lm_head();
    let v = lm.rows;
    let vb = ((16usize << 20) / n).clamp(256, 8192) / 64 * 64;
    let mut wb = vec![0f32; vb * d];
    let mut c = vec![0f32; n * vb];
    let mut acc = vec![Acc::new(); n];
    let mut full = vec![0f32; n_full * v];
    let mut f = vec![1f32; vb];
    let mut v0 = 0;
    while v0 < v {
        let nb = vb.min(v - v0);
        wb[..nb * d].par_chunks_mut(d).enumerate().for_each(|(i, r)| lm.row(v0 + i, r));
        blas::gemm_nt(h, &wb[..nb * d], n, nb, d, &mut c[..n * nb]);
        if m.logit_cap.is_some() {
            // softcap as a fixed per-token factor softcap(z)/z from this pass (row 0 = whole stream)
            for i in 0..nb {
                let r = c[i];
                f[i] = if r.abs() > 1e-6 { z[v0 + i] / r } else { 1.0 };
            }
            let fb = &f[..nb];
            c[..n * nb].par_chunks_mut(nb).for_each(|row| {
                for (x, ff) in row.iter_mut().zip(fb) {
                    *x *= ff;
                }
            });
        }
        let (zb, pb) = (&z[v0..v0 + nb], &p[v0..v0 + nb]);
        c[..n * nb].par_chunks(nb).zip(acc.par_iter_mut()).for_each_init(
            || (vec![0f32; vb], vec![0f32; vb]),
            |(t1, t2), (row, s)| s.update(row, v0, zb, pb, ys, &mut t1[..nb], &mut t2[..nb]),
        );
        for k in 0..n_full {
            full[k * v + v0..k * v + v0 + nb].copy_from_slice(&c[(k + 1) * nb..(k + 2) * nb]);
        }
        v0 += nb;
    }
    (acc.iter().map(|s| s.finish(logz, &lo_p, &zy)).collect(), full)
}

/// Log-odds ln(p_y / (1 − p_y)) = z_y − ln Σ_{t≠y} e^(z_t), without cancellation at p_y ≈ 1.
pub fn log_odds(z: &[f32], y: usize) -> f64 {
    let mx = z.iter().enumerate().filter(|(t, _)| *t != y).map(|(_, v)| *v).fold(f32::NEG_INFINITY, f32::max) as f64;
    let s: f64 = z.iter().enumerate().filter(|(t, _)| *t != y).map(|(_, v)| (*v as f64 - mx).exp()).sum();
    z[y] as f64 - (mx + s.ln())
}

/// Selected columns of a [rows][cols] matrix, as [idx][rows].
pub fn columns(w: &Mat, idx: &[usize]) -> Vec<f32> {
    let (rows, cols) = (w.rows, w.cols);
    let per: Vec<Vec<f32>> = (0..rows)
        .into_par_iter()
        .map_init(|| vec![0f32; cols], |buf, r| {
            w.row(r, buf);
            idx.iter().map(|&i| buf[i]).collect()
        })
        .collect();
    let mut out = vec![0f32; idx.len() * rows];
    for (r, v) in per.iter().enumerate() {
        for (k, x) in v.iter().enumerate() {
            out[k * rows + r] = *x;
        }
    }
    out
}

/// Real output distribution: (log-normalizer, probabilities).
pub fn softmax(z: &[f32]) -> (f64, Vec<f32>) {
    let mx = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let logz = mx + z.iter().map(|x| (*x as f64 - mx).exp()).sum::<f64>().ln();
    (logz, z.iter().map(|x| (*x as f64 - logz).exp() as f32).collect())
}
