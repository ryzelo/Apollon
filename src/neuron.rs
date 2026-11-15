//! One MLP neuron as a complete computing structure – the shared core of `--neuron`, `--trace` and
//! `--atlas`.
//!
//! Neuron i of layer l reads its whole input only through two numbers:
//!   p = g'·z + c_g,  q = u'·z + c_u,  a = act(p)·q (gated) or a = act(p) (not gated),
//! where z is the normalized stream (RMS: x/rms(x), LN: (x−mean)/σ, so ‖z‖ ≤ √d and for LN Σz = 0),
//! g' = w_gate ⊙ γ and u' = w_up ⊙ γ (centered for LN) are the effective read vectors and
//! c = w·β + b the offsets. The neuron's "large matrix" g'u'ᵀ has rank 1 and is never formed.
//!
//! * Repertoire: all reachable (p, q) form a filled ellipse c + √d·L·disk with L Lᵀ = G (Gram matrix
//!   of g', u'). act(p)·q is linear in q for fixed p, so its extremes lie on the boundary.
//! * Sources: every vector written into the stream (token embedding, position embedding, head, neuron,
//!   per-layer-embedding dimension, bias) reaches the MLP input along a fixed path of layer scalars
//!   and norms. With all norm scales held at 1 the path is linear, so every source has a static arrow
//!   (g'·v, u'·v) per unit of source activation. At runtime the frozen norm scales multiply in.
use crate::model::{Act, Block, Model, Norm, Res};
use crate::tensor::Mat;
use std::f64::consts::{PI, SQRT_2};

// ====================================================================== activation

pub fn act64(a: Act, x: f64) -> f64 {
    match a {
        Act::Silu => x / (1.0 + (-x).exp()),
        Act::Gelu => 0.5 * x * (1.0 + libm::erf(x / SQRT_2)),
        Act::GeluTanh => 0.5 * x * (1.0 + ((2.0 / PI).sqrt() * (x + 0.044715 * x * x * x)).tanh()),
        Act::Relu => x.max(0.0),
    }
}

/// Frozen factor φ = act(p)/p, so that act(p) = φ·p exactly; limit 0.5 at p → 0 (SiLU, GELU).
pub fn phi(a: Act, p: f64) -> f64 {
    if p.abs() < 1e-12 {
        return if a == Act::Relu { 0.0 } else { 0.5 };
    }
    act64(a, p) / p
}

pub fn act_name(a: Act) -> &'static str {
    match a {
        Act::Silu => "SiLU",
        Act::Gelu => "GELU",
        Act::GeluTanh => "GELU-tanh",
        Act::Relu => "ReLU",
    }
}

pub fn dot64(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum()
}

/// The norm whose output the MLP reads: the pre-FFN norm, for post-LN encoders the attention post-norm.
pub fn pre_norm(b: &Block) -> Option<&Norm> {
    match b.res {
        Res::Post => b.n_attn_post.as_ref(),
        _ => b.n_mlp.as_ref(),
    }
}

// ====================================================================== read vectors

/// What neuron i of layer l reads.
pub struct Read {
    pub l: usize,
    pub i: usize,
    pub d: usize,
    pub gated: bool,
    pub act: Act,
    /// pre-norm is a LayerNorm (centering)
    pub ln: bool,
    /// raw weight rows exactly as the forward pass unpacks them (gate and up; not gated: up only)
    pub wg: Vec<f32>,
    pub wu: Vec<f32>,
    /// effective read vectors in normalized space
    pub g: Vec<f32>,
    pub u: Vec<f32>,
    /// biases of p and q, offsets c = w·β + b
    pub bg: f64,
    pub bu: f64,
    pub cg: f64,
    pub cu: f64,
}

fn center(v: &mut [f32]) {
    let m = v.iter().map(|x| *x as f64).sum::<f64>() / v.len() as f64;
    v.iter_mut().for_each(|x| *x = (*x as f64 - m) as f32);
}

impl Read {
    pub fn new(m: &Model, l: usize, i: usize) -> Result<Read, String> {
        let b = m.blocks.get(l).ok_or(format!("S{l}: the model has only {} layers", m.n_layer()))?;
        let mlp = &b.mlp;
        let ff = mlp.up.rows;
        if i >= ff {
            return Err(format!("S{l} #{i}: the MLP has only {ff} neurons"));
        }
        let d = m.d;
        let norm = pre_norm(b);
        let row = |w: &Mat| {
            let mut v = vec![0f32; w.cols];
            w.row(i, &mut v);
            v
        };
        let gated = mlp.gate.is_some();
        let (wg, wu, bg, bu) = match &mlp.gate {
            Some(g) => (row(g), row(&mlp.up), 0.0, mlp.bup.as_ref().map_or(0.0, |b| b[i] as f64)),
            None => (row(&mlp.up), Vec::new(), mlp.bup.as_ref().map_or(0.0, |b| b[i] as f64), 0.0),
        };
        let ln = norm.is_some_and(|n| !n.rms);
        let eff = |w: &[f32]| -> Vec<f32> {
            let mut v: Vec<f32> = match norm {
                Some(n) => w.iter().zip(&n.w).map(|(a, g)| a * g).collect(),
                None => w.to_vec(),
            };
            if ln {
                center(&mut v);
            }
            v
        };
        let off = |w: &[f32], b: f64| -> f64 { b + norm.and_then(|n| n.b.as_ref()).map_or(0.0, |beta| dot64(w, beta)) };
        let g = eff(&wg);
        let u = if gated { eff(&wu) } else { Vec::new() };
        let cg = off(&wg, bg);
        let cu = if gated { off(&wu, bu) } else { 0.0 };
        Ok(Read { l, i, d, gated, act: mlp.act, ln, wg, wu, g, u, bg, bu, cg, cu })
    }

    pub fn gram(&self) -> [f64; 3] {
        if self.gated {
            [dot64(&self.g, &self.g), dot64(&self.g, &self.u), dot64(&self.u, &self.u)]
        } else {
            [dot64(&self.g, &self.g), 0.0, 0.0]
        }
    }

    /// a for given (p, q)
    pub fn value(&self, p: f64, q: f64) -> f64 {
        if self.gated { act64(self.act, p) * q } else { act64(self.act, p) }
    }
}

// ====================================================================== repertoire

#[derive(Clone, Debug)]
pub struct Repertoire {
    pub c: [f64; 2],
    pub gram: [f64; 3],
    /// lower Cholesky factor of the Gram matrix (L[1][1] = 0 when g' ∥ u')
    pub lmat: [[f64; 2]; 2],
    pub radius: f64,
    pub gated: bool,
    /// 2: ellipse, 1: segment (g' ∥ u', or not gated), 0: point
    pub rank: u8,
    pub a_max: f64,
    pub a_min: f64,
    pub th_max: f64,
    pub th_min: f64,
    pub pt_max: [f64; 2],
    pub pt_min: [f64; 2],
}

/// Pseudo-inverse of a symmetric positive semidefinite 2×2 matrix [a, b; b, c]
fn pinv2(g: [f64; 3]) -> [f64; 3] {
    let (a, b, c) = (g[0], g[1], g[2]);
    let tr = a + c;
    if tr <= 0.0 {
        return [0.0; 3];
    }
    let det = a * c - b * b;
    if det > 1e-12 * tr * tr {
        [c / det, -b / det, a / det]
    } else {
        // rank 1: G = λ v vᵀ, G⁺ = G / λ²
        [a / (tr * tr), b / (tr * tr), c / (tr * tr)]
    }
}

fn golden(f: &dyn Fn(f64) -> f64, mut lo: f64, mut hi: f64, maximize: bool) -> (f64, f64) {
    let r = (5f64.sqrt() - 1.0) / 2.0;
    let sg = if maximize { 1.0 } else { -1.0 };
    let mut x1 = hi - r * (hi - lo);
    let mut x2 = lo + r * (hi - lo);
    let (mut f1, mut f2) = (sg * f(x1), sg * f(x2));
    for _ in 0..80 {
        if f1 < f2 {
            lo = x1;
            x1 = x2;
            f1 = f2;
            x2 = lo + r * (hi - lo);
            f2 = sg * f(x2);
        } else {
            hi = x2;
            x2 = x1;
            f2 = f1;
            x1 = hi - r * (hi - lo);
            f1 = sg * f(x1);
        }
    }
    let x = 0.5 * (lo + hi);
    (x, f(x))
}

pub const GRID: usize = 8192;

impl Repertoire {
    /// Extremes of act(p)·q (or act(p)) over the filled ellipse c + radius·L·disk.
    pub fn new(c: [f64; 2], gram: [f64; 3], radius: f64, act: Act, gated: bool) -> Self {
        let gram = if gated { gram } else { [gram[0], 0.0, 0.0] };
        let (a, b, cc) = (gram[0], gram[1], gram[2]);
        let tr = a + cc;
        let mut lmat = [[0.0; 2]; 2];
        let rank;
        if tr <= 0.0 {
            rank = 0;
        } else if a > 1e-14 * tr {
            lmat[0][0] = a.sqrt();
            lmat[1][0] = b / lmat[0][0];
            let rest = cc - b * b / a;
            if rest > 1e-12 * tr {
                lmat[1][1] = rest.sqrt();
                rank = 2;
            } else {
                rank = 1;
            }
        } else {
            lmat[1][1] = cc.sqrt();
            rank = 1;
        }
        let mut r = Repertoire { c, gram, lmat, radius, gated, rank, a_max: 0.0, a_min: 0.0, th_max: 0.0, th_min: 0.0, pt_max: c, pt_min: c };
        let f = |th: f64| -> f64 {
            let pq = r.point(th);
            if gated { act64(act, pq[0]) * pq[1] } else { act64(act, pq[0]) }
        };
        // the boundary holds the extremes (gated); not gated, θ ∈ [0, π] already covers the whole interval
        let step = 2.0 * PI / GRID as f64;
        let (mut imax, mut imin) = (0usize, 0usize);
        let (mut vmax, mut vmin) = (f64::NEG_INFINITY, f64::INFINITY);
        for k in 0..GRID {
            let v = f(k as f64 * step);
            if v > vmax {
                vmax = v;
                imax = k;
            }
            if v < vmin {
                vmin = v;
                imin = k;
            }
        }
        let (tmax, fmax) = golden(&f, (imax as f64 - 1.0) * step, (imax as f64 + 1.0) * step, true);
        let (tmin, fmin) = golden(&f, (imin as f64 - 1.0) * step, (imin as f64 + 1.0) * step, false);
        let (th_max, a_max) = if fmax >= vmax { (tmax, fmax) } else { (imax as f64 * step, vmax) };
        let (th_min, a_min) = if fmin <= vmin { (tmin, fmin) } else { (imin as f64 * step, vmin) };
        r.a_max = a_max;
        r.a_min = a_min;
        r.th_max = th_max.rem_euclid(2.0 * PI);
        r.th_min = th_min.rem_euclid(2.0 * PI);
        r.pt_max = r.point(r.th_max);
        r.pt_min = r.point(r.th_min);
        r
    }

    /// boundary point at angle θ
    pub fn point(&self, th: f64) -> [f64; 2] {
        let (c, s) = (th.cos(), th.sin());
        let l = &self.lmat;
        [self.c[0] + self.radius * l[0][0] * c, self.c[1] + self.radius * (l[1][0] * c + l[1][1] * s)]
    }

    /// Radius of (p, q) inside the ellipse: 0 = center, 1 = boundary
    pub fn r_of(&self, p: f64, q: f64) -> f64 {
        let gp = pinv2(self.gram);
        let (dp, dq) = (p - self.c[0], if self.gated { q - self.c[1] } else { 0.0 });
        let v = dp * dp * gp[0] + 2.0 * dp * dq * gp[1] + dq * dq * gp[2];
        v.max(0.0).sqrt() / self.radius
    }

    /// Input direction z (‖z‖ = radius) that reaches the boundary point at angle θ: z = Mᵀ G⁺ L w·radius
    pub fn z_at(&self, rd: &Read, th: f64) -> Vec<f32> {
        let l = &self.lmat;
        let w = [th.cos(), th.sin()];
        let t = [l[0][0] * w[0], l[1][0] * w[0] + l[1][1] * w[1]];
        let gp = pinv2(self.gram);
        let y = [gp[0] * t[0] + gp[1] * t[1], gp[1] * t[0] + gp[2] * t[1]];
        (0..rd.d)
            .map(|j| (self.radius * (y[0] * rd.g[j] as f64 + if rd.gated { y[1] * rd.u[j] as f64 } else { 0.0 })) as f32)
            .collect()
    }
}

pub fn repertoire(rd: &Read) -> Repertoire {
    Repertoire::new([rd.cg, rd.cu], rd.gram(), (rd.d as f64).sqrt(), rd.act, rd.gated)
}

// ====================================================================== paths from sources to read points

/// Where something is read from the stream.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ReadPoint {
    /// the input of the pre-norm of the MLP of layer l
    MlpIn(usize),
    /// the input of the attention (pre-)norm of layer l
    AttnIn(usize),
    /// the input of the final norm
    Final,
}

/// Which frozen runtime scale a norm on a path uses.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Scale {
    AttnPost(usize),
    MlpPost(usize),
    Ple(usize),
    /// a norm inside the stream (post-LN encoders); no runtime scale is recorded for these
    Stream,
}

/// One linear step on a path. A norm counts with scale 1 (statically) or with its frozen
/// runtime scale; its bias is a separate constant source.
#[derive(Clone, Copy)]
pub enum Op<'a> {
    Mul(f32),
    Norm(&'a Norm, Scale),
}

/// A source: something written into the stream.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum Src {
    /// embedding row of a token (times the embedding scale)
    Token(u32),
    /// learned position embedding (row = position + offset)
    Position(u32),
    /// token-type embedding (BERT, constant)
    TypeEmbed,
    EmbedNormBias,
    /// what head h of layer k writes (runtime only)
    Head(u16, u16),
    /// head h of layer k reading position j (runtime only)
    Atom(u16, u16, u32),
    AttnBias(u16),
    AttnPostBias(u16),
    Neuron(u16, u32),
    MlpBias(u16),
    MlpPostBias(u16),
    PleDim(u16, u32),
    /// the target's own offset c = w·β + b
    Offset,
}

impl Src {
    pub fn label(&self, m: &Model) -> String {
        match *self {
            Src::Token(t) => format!("token {}", crate::fmt::tok(m, t)),
            Src::Position(p) => format!("position {}", p as usize - m.pos_embed.as_ref().map_or(0, |x| x.1)),
            Src::TypeEmbed => "token-type embedding".into(),
            Src::EmbedNormBias => "embedding-norm bias".into(),
            Src::Head(k, h) => format!("S{k} head {h}"),
            Src::Atom(k, h, j) => format!("S{k} head {h} reads [{j}]"),
            Src::AttnBias(k) => format!("S{k} attention bias"),
            Src::AttnPostBias(k) => format!("S{k} attention post-norm bias"),
            Src::Neuron(k, i) => format!("S{k} #{i}"),
            Src::MlpBias(k) => format!("S{k} MLP bias"),
            Src::MlpPostBias(k) => format!("S{k} MLP post-norm bias"),
            Src::PleDim(k, e) => format!("S{k} PLE #{e}"),
            Src::Offset => "offset (w·β + b)".into(),
        }
    }
    pub fn group(&self) -> &'static str {
        match self {
            Src::Token(_) | Src::Position(_) | Src::TypeEmbed | Src::EmbedNormBias => "embedding",
            Src::Head(..) | Src::Atom(..) => "heads",
            Src::Neuron(..) => "neurons",
            Src::PleDim(..) => "PLE",
            Src::AttnBias(_) | Src::AttnPostBias(_) | Src::MlpBias(_) | Src::MlpPostBias(_) => "biases",
            Src::Offset => "offset",
        }
    }
}

const NEVER: f64 = f64::INFINITY;

/// Time of a read point in the order of writes (attention 3k, MLP 3k+1, PLE 3k+2, layer scalar 3k+2.5;
/// post-LN norms 3k+0.25 and 3k+1.25).
fn read_time(m: &Model, rp: ReadPoint) -> f64 {
    match rp {
        ReadPoint::MlpIn(l) => match m.blocks[l].res {
            Res::Pre => 3.0 * l as f64 + 0.5,
            // parallel residual: the MLP reads the stream before the layer, i.e. after the previous scalar (3l − 0.5)
            Res::Parallel => 3.0 * l as f64 - 0.25,
            Res::Post => 3.0 * l as f64 + 0.2,
        },
        // the attention reads the stream before the layer: after the previous layer's scalar at 3(l−1) + 2.5
        ReadPoint::AttnIn(l) => 3.0 * l as f64 - 0.25,
        ReadPoint::Final => NEVER,
    }
}

/// Write time of a source and the local steps before it enters the stream (post-norms of decoder blocks).
fn write<'a>(m: &'a Model, src: Src) -> (f64, Vec<Op<'a>>) {
    let post = |k: usize, attn: bool| -> Vec<Op<'a>> {
        let b = &m.blocks[k];
        if b.res != Res::Pre {
            return vec![];
        }
        match if attn { b.n_attn_post.as_ref() } else { b.n_mlp_post.as_ref() } {
            Some(n) => vec![Op::Norm(n, if attn { Scale::AttnPost(k) } else { Scale::MlpPost(k) })],
            None => vec![],
        }
    };
    match src {
        Src::Token(_) => (-1.0, if m.embed_scale != 1.0 { vec![Op::Mul(m.embed_scale)] } else { vec![] }),
        Src::Position(_) | Src::TypeEmbed => (-1.0, vec![]),
        Src::EmbedNormBias => (-0.4, vec![]),
        Src::Head(k, _) | Src::Atom(k, _, _) | Src::AttnBias(k) => (3.0 * k as f64, post(k as usize, true)),
        Src::AttnPostBias(k) => (3.0 * k as f64 + 0.3, vec![]),
        Src::Neuron(k, _) | Src::MlpBias(k) => (3.0 * k as f64 + 1.0, post(k as usize, false)),
        Src::MlpPostBias(k) => (3.0 * k as f64 + 1.3, vec![]),
        Src::PleDim(k, _) => match m.blocks[k as usize].ple.as_ref() {
            Some(pb) => (3.0 * k as f64 + 2.0, vec![Op::Norm(&pb.norm, Scale::Ple(k as usize))]),
            None => (NEVER, vec![]),
        },
        Src::Offset => (NEVER, vec![]),
    }
}

/// Steps inside the stream (layer scalars; for post-LN encoders the norms) in time order.
fn stream_events(m: &Model) -> Vec<(f64, Op<'_>)> {
    let mut ev = Vec::new();
    if let Some(n) = &m.embed_norm {
        ev.push((-0.5, Op::Norm(n, Scale::Stream)));
    }
    for (k, b) in m.blocks.iter().enumerate() {
        if b.res == Res::Post {
            if let Some(n) = &b.n_attn_post {
                ev.push((3.0 * k as f64 + 0.25, Op::Norm(n, Scale::Stream)));
            }
            if let Some(n) = &b.n_mlp_post {
                ev.push((3.0 * k as f64 + 1.25, Op::Norm(n, Scale::Stream)));
            }
        }
        if let Some(s) = b.scalar {
            ev.push((3.0 * k as f64 + 2.5, Op::Mul(s)));
        }
    }
    ev
}

/// The linear path from a source to a read point (None if the source does not reach it).
/// `with_local`: include the source's own post-norm (false for runtime vectors that are already mapped).
pub fn path<'a>(m: &'a Model, src: Src, rp: ReadPoint, with_local: bool) -> Option<Vec<Op<'a>>> {
    let (tw, local) = write(m, src);
    let tr = read_time(m, rp);
    if tw >= tr {
        return None;
    }
    let mut ops = if with_local { local } else { vec![] };
    ops.extend(stream_events(m).into_iter().filter(|(t, _)| *t > tw && *t < tr).map(|x| x.1));
    Some(ops)
}

/// One step forward on a vector (statically: scale 1; at runtime: the given scale).
fn step(op: &Op, v: &mut [f32], scale: f32) {
    match op {
        Op::Mul(f) => v.iter_mut().for_each(|x| *x *= f),
        Op::Norm(n, _) => {
            let mapped = n.map_part(scale, v);
            v.copy_from_slice(&mapped);
        }
    }
}

/// Apply a path forward with norm scales 1.
pub fn forward(ops: &[Op], v: &mut [f32]) {
    for op in ops {
        step(op, v, 1.0);
    }
}

/// Apply a path forward with the frozen runtime scales.
pub fn forward_runtime(ops: &[Op], v: &mut [f32], scale: &dyn Fn(Scale) -> f32) {
    for op in ops {
        let s = match op {
            Op::Norm(_, sc) => scale(*sc),
            _ => 1.0,
        };
        step(op, v, s);
    }
}

/// Adjoint of a path (norm scales 1): turns a read vector at the read point into one at the source.
pub fn adjoint(ops: &[Op], g: &mut [f32]) {
    for op in ops.iter().rev() {
        match op {
            Op::Mul(f) => g.iter_mut().for_each(|x| *x *= f),
            Op::Norm(n, _) => {
                for (x, w) in g.iter_mut().zip(&n.w) {
                    *x *= w;
                }
                if !n.rms {
                    center(g);
                }
            }
        }
    }
}

/// Product of the frozen runtime scales on a path (each norm with fixed scale is linear, so its scale
/// factors out of the static arrow).
pub fn runtime_factor(ops: &[Op], scale: &dyn Fn(Scale) -> f32) -> f64 {
    ops.iter()
        .map(|op| match op {
            Op::Norm(_, sc) => scale(*sc) as f64,
            Op::Mul(_) => 1.0,
        })
        .product()
}

// ====================================================================== static arrows

/// Arrow of a source with raw vector v: (g'·path(v), u'·path(v)) per unit of source activation.
pub fn arrow(rd: &Read, ops: &[Op], v: &[f32]) -> (f64, f64) {
    let mut w = v.to_vec();
    forward(ops, &mut w);
    (dot64(&rd.g, &w), if rd.gated { dot64(&rd.u, &w) } else { 0.0 })
}

/// Read vectors brought back to the source along a path (adjoint): arrow = r·v for any raw v.
pub fn read_back(rd: &Read, ops: &[Op]) -> (Vec<f32>, Vec<f32>) {
    let mut g = rd.g.clone();
    adjoint(ops, &mut g);
    let u = if rd.gated {
        let mut u = rd.u.clone();
        adjoint(ops, &mut u);
        u
    } else {
        Vec::new()
    };
    (g, u)
}

/// Columns of a matrix all read with r at once: rᵀ W (one value per column) – for W_down and the PLE
/// projection, whose columns are the raw source vectors.
pub fn col_arrows(w: &Mat, r: &[f32]) -> Vec<f32> {
    w.t_matvec(r)
}

/// Rows of a matrix read with r: W r (one value per row) – for embedding tables.
pub fn row_arrows(w: &Mat, r: &[f32]) -> Vec<f32> {
    w.matmul(r, 1)
}


// ====================================================================== static inputs and outputs (on demand)

/// A static arrow: (contribution to p, contribution to q) per unit of source activation.
#[derive(Clone, Copy, Debug)]
pub struct Arrow {
    pub src: Src,
    pub p: f64,
    pub q: f64,
}

/// Top k by |p| and by |q|.
pub fn top2(v: &[Arrow], k: usize) -> (Vec<Arrow>, Vec<Arrow>) {
    let pick = |key: &dyn Fn(&Arrow) -> f64| -> Vec<Arrow> {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        let k = k.min(idx.len());
        if k == 0 {
            return Vec::new();
        }
        idx.select_nth_unstable_by(k - 1, |a, b| key(&v[*b]).partial_cmp(&key(&v[*a])).unwrap_or(std::cmp::Ordering::Equal));
        let mut out: Vec<Arrow> = idx[..k].iter().map(|&i| v[i]).collect();
        out.sort_by(|a, b| key(b).partial_cmp(&key(a)).unwrap_or(std::cmp::Ordering::Equal));
        out
    };
    (pick(&|a| a.p.abs()), pick(&|a| a.q.abs()))
}

/// All inputs of a neuron, by kind: (kind, number of sources, all arrows of that kind)
pub struct Inputs {
    pub kinds: Vec<(&'static str, Vec<Arrow>)>,
    /// layers whose heads write into the neuron's input: no static arrow (their output depends on what they read)
    pub head_layers: Vec<usize>,
}

pub fn static_inputs(m: &Model, rd: &Read) -> Inputs {
    let l = rd.l;
    let rp = ReadPoint::MlpIn(l);
    let gated = rd.gated;
    let rows = |w: &Mat, src_of: &dyn Fn(usize) -> Src, ops: &[Op]| -> Vec<Arrow> {
        let (rg, ru) = read_back(rd, ops);
        let ag = row_arrows(w, &rg);
        let au = if gated { row_arrows(w, &ru) } else { Vec::new() };
        (0..ag.len()).map(|r| Arrow { src: src_of(r), p: ag[r] as f64, q: if gated { au[r] as f64 } else { 0.0 } }).collect()
    };
    let cols = |w: &Mat, src_of: &dyn Fn(usize) -> Src, ops: &[Op]| -> Vec<Arrow> {
        let (rg, ru) = read_back(rd, ops);
        let ag = col_arrows(w, &rg);
        let au = if gated { col_arrows(w, &ru) } else { Vec::new() };
        (0..ag.len()).map(|c| Arrow { src: src_of(c), p: ag[c] as f64, q: if gated { au[c] as f64 } else { 0.0 } }).collect()
    };
    let mut kinds: Vec<(&'static str, Vec<Arrow>)> = Vec::new();
    if let Some(ops) = path(m, Src::Token(0), rp, true) {
        kinds.push(("tokens", rows(&m.embed, &|r| Src::Token(r as u32), &ops)));
    }
    if let (Some((pe, _)), Some(ops)) = (&m.pos_embed, path(m, Src::Position(0), rp, true)) {
        kinds.push(("positions", rows(pe, &|r| Src::Position(r as u32), &ops)));
    }
    let mut neurons = Vec::new();
    let mut ple = Vec::new();
    let mut biases = Vec::new();
    let mut bias = |src: Src, v: Option<&Vec<f32>>| {
        if let (Some(v), Some(ops)) = (v, path(m, src, rp, true)) {
            let (p, q) = arrow(rd, &ops, v);
            biases.push(Arrow { src, p, q });
        }
    };
    bias(Src::TypeEmbed, m.type_embed.as_ref());
    bias(Src::EmbedNormBias, m.embed_norm.as_ref().and_then(|n| n.b.as_ref()));
    let mut head_layers = Vec::new();
    for (k, b) in m.blocks.iter().enumerate() {
        if path(m, Src::Head(k as u16, 0), rp, true).is_some() {
            head_layers.push(k);
        }
        bias(Src::AttnBias(k as u16), b.attn.bo.as_ref());
        bias(Src::AttnPostBias(k as u16), b.n_attn_post.as_ref().and_then(|n| n.b.as_ref()));
        if let Some(ops) = path(m, Src::Neuron(k as u16, 0), rp, true) {
            neurons.extend(cols(&b.mlp.down, &|c| Src::Neuron(k as u16, c as u32), &ops));
        }
        bias(Src::MlpBias(k as u16), b.mlp.bdown.as_ref());
        bias(Src::MlpPostBias(k as u16), b.n_mlp_post.as_ref().and_then(|n| n.b.as_ref()));
        if let (Some(pb), Some(ops)) = (&b.ple, path(m, Src::PleDim(k as u16, 0), rp, true)) {
            ple.extend(cols(&pb.proj, &|c| Src::PleDim(k as u16, c as u32), &ops));
        }
    }
    kinds.push(("neurons", neurons));
    if !ple.is_empty() {
        kinds.push(("PLE dimensions", ple));
    }
    kinds.push(("biases", biases));
    Inputs { kinds, head_layers }
}

/// Where neuron (l, i) writes to.
pub struct Outputs {
    /// into the (p, q) plane of every later neuron (target as Src::Neuron)
    pub neurons: Vec<Arrow>,
    /// token voice through the final norm (unit scale) and the output matrix, per unit activation
    pub voice: Vec<(u32, f32)>,
    /// later heads: (layer, head, ‖W_Q o‖, ‖W_K o‖, ‖W_V o‖) – linear projections only (RoPE, Q/K/V norms ignored)
    pub heads: Vec<(usize, usize, f32, f32, f32)>,
}

pub fn static_outputs(m: &Model, l: usize, i: usize) -> Outputs {
    let col = crate::readout::columns(&m.blocks[l].mlp.down, &[i]);
    let src = Src::Neuron(l as u16, i as u32);
    let mut neurons = Vec::new();
    let mut heads = Vec::new();
    for (l2, b2) in m.blocks.iter().enumerate() {
        if let Some(ops) = path(m, src, ReadPoint::MlpIn(l2), true) {
            let mut v = col.clone();
            forward(&ops, &mut v);
            let vn = match pre_norm(b2) {
                Some(n) => n.map_part(1.0, &v),
                None => v,
            };
            let gated = b2.mlp.gate.is_some();
            let p = match &b2.mlp.gate {
                Some(g) => g.matmul(&vn, 1),
                None => b2.mlp.up.matmul(&vn, 1),
            };
            let q = if gated { b2.mlp.up.matmul(&vn, 1) } else { Vec::new() };
            neurons.extend((0..p.len()).map(|i2| Arrow { src: Src::Neuron(l2 as u16, i2 as u32), p: p[i2] as f64, q: if gated { q[i2] as f64 } else { 0.0 } }));
        }
        if let Some(ops) = path(m, src, ReadPoint::AttnIn(l2), true) {
            let mut v = col.clone();
            forward(&ops, &mut v);
            let vn = match (&b2.n_attn, b2.res) {
                (Some(n), Res::Pre | Res::Parallel) => n.map_part(1.0, &v),
                _ => v,
            };
            let a = &b2.attn;
            let hd = a.hd;
            let nrm = |x: &[f32]| x.iter().map(|y| (*y as f64).powi(2)).sum::<f64>().sqrt() as f32;
            if let Some(w) = &a.wqkv {
                let o = w.matmul(&vn, 1);
                for h in 0..a.n_head {
                    let s = &o[h * 3 * hd..(h + 1) * 3 * hd];
                    heads.push((l2, h, nrm(&s[..hd]), nrm(&s[hd..2 * hd]), nrm(&s[2 * hd..])));
                }
            } else {
                let qv = a.wq.as_ref().map(|w| w.matmul(&vn, 1)).unwrap_or_default();
                let kv = a.wk.as_ref().map(|w| w.matmul(&vn, 1));
                let vv = if a.v_is_k { kv.clone() } else { a.wv.as_ref().map(|w| w.matmul(&vn, 1)) };
                let rep = a.n_head / a.n_kv.max(1);
                for h in 0..a.n_head {
                    let g = h / rep.max(1);
                    let kn = kv.as_ref().map_or(f32::NAN, |k| nrm(&k[g * hd..(g + 1) * hd]));
                    let vn_ = vv.as_ref().map_or(f32::NAN, |k| nrm(&k[g * hd..(g + 1) * hd]));
                    heads.push((l2, h, nrm(&qv[h * hd..(h + 1) * hd]), kn, vn_));
                }
            }
        }
    }
    let voice = match (m.has_head, path(m, src, ReadPoint::Final, true)) {
        (true, Some(ops)) => {
            let mut v = col.clone();
            forward(&ops, &mut v);
            let vn = m.final_norm.as_ref().map_or(v.clone(), |n| n.map_part(1.0, &v));
            let lg = m.lm_head().matmul(&vn, 1);
            lg.into_iter().enumerate().map(|(t, x)| (t as u32, x)).collect()
        }
        _ => Vec::new(),
    };
    Outputs { neurons, voice, heads }
}

// ====================================================================== parts on the command line

/// A part named on the command line: "S12 MLP", "S11 PLE", "S12 Attn", "S12 #2328"
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum PartSpec {
    Mlp(usize),
    Ple(usize),
    Attn(usize),
    Neuron(usize, usize),
}

pub fn parse_part(s: &str) -> Result<PartSpec, String> {
    let err = || format!("'{s}': allowed are \"S12 MLP\", \"S11 PLE\", \"S12 Attn\", \"S12 #2328\"");
    let w: Vec<String> = s.split_whitespace().map(|x| x.to_lowercase()).collect();
    let l: usize = w.first().and_then(|x| x.trim_start_matches(['s', 'l']).parse().ok()).ok_or_else(err)?;
    let part = match w.get(1).map(|x| x.as_str()) {
        Some("mlp" | "ffn") => PartSpec::Mlp(l),
        Some("ple") => PartSpec::Ple(l),
        Some("attn" | "attention") => PartSpec::Attn(l),
        Some(x) if x.starts_with('#') => PartSpec::Neuron(l, x[1..].parse().map_err(|_| err())?),
        Some("neuron" | "n") => PartSpec::Neuron(l, w.get(2).map(|x| x.trim_start_matches('#')).and_then(|x| x.parse().ok()).ok_or_else(err)?),
        _ => return Err(err()),
    };
    let n_words = if matches!(w.get(1).map(|x| x.as_str()), Some("neuron" | "n")) { 3 } else { 2 };
    if w.len() > n_words {
        return Err(err());
    }
    Ok(part)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
        fn normal(&mut self) -> f64 {
            let (u, v) = (self.next().max(1e-300), self.next());
            (-2.0 * u.ln()).sqrt() * (2.0 * PI * v).cos()
        }
    }

    /// Sample z on/in the sphere of radius √d in d dimensions and check that no sample exceeds the
    /// extremes and that the extremes are approached.
    fn check(g: &[f64], u: &[f64], c: [f64; 2], act: Act, gated: bool, rng: &mut Rng) {
        let d = g.len();
        let gram = [g.iter().map(|x| x * x).sum(), g.iter().zip(u).map(|(x, y)| x * y).sum(), u.iter().map(|x| x * x).sum()];
        let rad = (d as f64).sqrt();
        let r = Repertoire::new(c, gram, rad, act, gated);
        let val = |p: f64, q: f64| if gated { act64(act, p) * q } else { act64(act, p) };
        let tol = 1e-9 * (1.0 + r.a_max.abs().max(r.a_min.abs()));
        let (mut smax, mut smin) = (f64::NEG_INFINITY, f64::INFINITY);
        for k in 0..40000 {
            let mut z: Vec<f64> = (0..d).map(|_| rng.normal()).collect();
            let n = z.iter().map(|x| x * x).sum::<f64>().sqrt();
            let radius = if k % 2 == 0 { rad } else { rad * rng.next().powf(1.0 / d as f64) };
            z.iter_mut().for_each(|x| *x *= radius / n);
            let p = c[0] + g.iter().zip(&z).map(|(a, b)| a * b).sum::<f64>();
            let q = c[1] + u.iter().zip(&z).map(|(a, b)| a * b).sum::<f64>();
            let a = val(p, q);
            assert!(a <= r.a_max + tol, "sample {a} above a_max {}", r.a_max);
            assert!(a >= r.a_min - tol, "sample {a} below a_min {}", r.a_min);
            assert!(r.r_of(p, q) <= 1.0 + 1e-9, "sample outside the ellipse: r = {}", r.r_of(p, q));
            smax = smax.max(a);
            smin = smin.min(a);
        }
        // the extremes are reached: the boundary points themselves give a_max / a_min
        assert!((val(r.pt_max[0], r.pt_max[1]) - r.a_max).abs() <= tol);
        assert!((val(r.pt_min[0], r.pt_min[1]) - r.a_min).abs() <= tol);
        // and random boundary samples come close (low dimension, so the boundary is hit often)
        let span = (r.a_max - r.a_min).max(1e-12);
        if d <= 3 {
            assert!(r.a_max - smax < 0.02 * span, "max not approached: {smax} vs {}", r.a_max);
            assert!(smin - r.a_min < 0.02 * span, "min not approached: {smin} vs {}", r.a_min);
        }
        // the input direction at θ_max reaches the boundary point exactly
        let rd = Read {
            l: 0,
            i: 0,
            d,
            gated,
            act,
            ln: false,
            wg: vec![],
            wu: vec![],
            g: g.iter().map(|x| *x as f32).collect(),
            u: if gated { u.iter().map(|x| *x as f32).collect() } else { vec![] },
            bg: 0.0,
            bu: 0.0,
            cg: c[0],
            cu: c[1],
        };
        let z = r.z_at(&rd, r.th_max);
        let p = c[0] + g.iter().zip(&z).map(|(a, b)| a * *b as f64).sum::<f64>();
        let zn = z.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        assert!((p - r.pt_max[0]).abs() < 1e-4 * (1.0 + p.abs()), "z* misses p: {p} vs {}", r.pt_max[0]);
        assert!(zn <= rad * (1.0 + 1e-5), "‖z*‖ = {zn} > √d");
        if gated {
            let q = c[1] + u.iter().zip(&z).map(|(a, b)| a * *b as f64).sum::<f64>();
            assert!((q - r.pt_max[1]).abs() < 1e-4 * (1.0 + q.abs()), "z* misses q: {q} vs {}", r.pt_max[1]);
        }
    }

    #[test]
    fn ellipse_extremes_random() {
        let mut rng = Rng(0x9E3779B97F4A7C15);
        for case in 0..40 {
            let d = [2, 3, 8, 32][case % 4];
            let s = 0.3 + 2.0 * rng.next();
            let g: Vec<f64> = (0..d).map(|_| s * rng.normal()).collect();
            let u: Vec<f64> = (0..d).map(|_| s * rng.normal()).collect();
            let c = [2.0 * rng.normal(), 2.0 * rng.normal()];
            let act = [Act::Silu, Act::Gelu, Act::GeluTanh, Act::Relu][case % 4];
            check(&g, &u, c, act, true, &mut rng);
            check(&g, &u, c, act, false, &mut rng);
        }
    }

    #[test]
    fn ellipse_degenerate() {
        let mut rng = Rng(12345);
        for d in [2, 3, 16] {
            let g: Vec<f64> = (0..d).map(|_| rng.normal()).collect();
            let par: Vec<f64> = g.iter().map(|x| -1.7 * x).collect();
            let zero = vec![0.0; d];
            for act in [Act::Silu, Act::GeluTanh] {
                check(&g, &par, [0.3, -0.2], act, true, &mut rng); // g' ∥ u': a segment
                check(&g, &zero, [0.3, 1.1], act, true, &mut rng); // u' = 0: q fixed
                check(&zero, &g, [-0.4, 0.5], act, true, &mut rng); // g' = 0: p fixed
                check(&zero, &zero, [0.7, 2.0], act, true, &mut rng); // a single point
            }
        }
    }
}
