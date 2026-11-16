//! Transformer as a construction kit: every layer is assembled from components (norms before/after,
//! attention variants, MLP variants, per-layer embeddings, scalars). One shared forward pass computes
//! all of them and records what the analyses need. The loaders at the end assemble the components
//! from configuration + weights (GGUF and Hugging Face / MLX).
use crate::gguf::{self, Gguf, Val};
use crate::quant::DType;
use crate::tensor::{dot, norm, Mat};
use crate::tokenizer::Tokenizer;
use rayon::prelude::*;
use serde_json::Value;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

// ====================================================================== components

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Act {
    Silu,
    Gelu,
    GeluTanh,
    Relu,
}

pub fn act_fn(a: Act) -> fn(f32) -> f32 {
    match a {
        Act::Silu => |x| x / (1.0 + (-x).exp()),
        Act::Gelu => |x| 0.5 * x * (1.0 + libm::erff(x / std::f32::consts::SQRT_2)),
        Act::GeluTanh => |x| 0.5 * x * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh()),
        Act::Relu => |x| x.max(0.0),
    }
}

fn parse_act(s: &str) -> Result<Act, String> {
    Ok(match s {
        "silu" | "swish" => Act::Silu,
        "gelu" => Act::Gelu,
        "gelu_new" | "gelu_fast" | "gelu_pytorch_tanh" | "gelu_tanh" => Act::GeluTanh,
        "relu" => Act::Relu,
        o => return Err(format!("activation '{o}' is not supported")),
    })
}

pub struct Norm {
    pub w: Vec<f32>,
    pub b: Option<Vec<f32>>,
    pub eps: f32,
    pub rms: bool,
}

impl Norm {
    fn rms(w: Vec<f32>, eps: f32) -> Self {
        Norm { w, b: None, eps, rms: true }
    }
    /// Normalize x into out; returns (1/rms or 1/sigma, mean)
    pub fn apply(&self, x: &[f32], out: &mut [f32]) -> (f32, f32) {
        let n = x.len() as f64;
        if self.rms {
            let ms = x.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / n;
            let s = (1.0 / (ms + self.eps as f64).sqrt()) as f32;
            for i in 0..x.len() {
                out[i] = x[i] * s * self.w[i];
            }
            (s, 0.0)
        } else {
            let mean = x.iter().map(|v| *v as f64).sum::<f64>() / n;
            let var = x.iter().map(|v| (*v as f64 - mean).powi(2)).sum::<f64>() / n;
            let s = (1.0 / (var + self.eps as f64).sqrt()) as f32;
            let m = mean as f32;
            for i in 0..x.len() {
                out[i] = (x[i] - m) * s * self.w[i] + self.b.as_ref().map_or(0.0, |b| b[i]);
            }
            (s, m)
        }
    }
    /// Like apply, with the scale s given (frozen); LayerNorm still subtracts the mean of x.
    /// With s = the scale apply would compute, the result is bit-identical to apply.
    pub fn apply_scale(&self, x: &[f32], out: &mut [f32], s: f32) {
        if self.rms {
            for i in 0..x.len() {
                out[i] = x[i] * s * self.w[i];
            }
        } else {
            let n = x.len() as f64;
            let m = (x.iter().map(|v| *v as f64).sum::<f64>() / n) as f32;
            for i in 0..x.len() {
                out[i] = (x[i] - m) * s * self.w[i] + self.b.as_ref().map_or(0.0, |b| b[i]);
            }
        }
    }
    /// apply with the scale computed (fz = None) or given; returns the scale used
    #[inline]
    pub fn apply_fz(&self, x: &[f32], out: &mut [f32], fz: Option<f32>) -> f32 {
        match fz {
            Some(s) => {
                self.apply_scale(x, out, s);
                s
            }
            None => self.apply(x, out).0,
        }
    }
    /// all rows [t][d]; with fz the scale of row i is fz[pos(i)]. Returns the rows and the scales used.
    pub fn rows_fz(&self, x: &[f32], d: usize, fz: Option<&[f32]>, pos: &(dyn Fn(usize) -> usize + Sync)) -> (Vec<f32>, Vec<f32>) {
        let t = x.len() / d;
        let mut out = vec![0f32; x.len()];
        let mut sc = vec![0f32; t];
        out.par_chunks_mut(d).zip(sc.par_iter_mut()).enumerate().for_each(|(i, (o, s))| {
            *s = self.apply_fz(&x[i * d..(i + 1) * d], o, fz.map(|f| f[pos(i)]));
        });
        (out, sc)
    }
    /// only the scale (1/rms or 1/sigma) of one row
    pub fn scale(&self, x: &[f32]) -> f32 {
        let n = x.len() as f64;
        let var = if self.rms {
            x.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / n
        } else {
            let mean = x.iter().map(|v| *v as f64).sum::<f64>() / n;
            x.iter().map(|v| (*v as f64 - mean).powi(2)).sum::<f64>() / n
        };
        (1.0 / (var + self.eps as f64).sqrt()) as f32
    }
    /// Transpose of the frozen norm (scale s fixed): gradient g at the output -> gradient at the input
    pub fn back(&self, s: f32, g: &[f32]) -> Vec<f32> {
        let mut v: Vec<f32> = g.iter().zip(&self.w).map(|(a, w)| a * w * s).collect();
        if !self.rms {
            let m = v.iter().sum::<f32>() / v.len() as f32;
            v.iter_mut().for_each(|x| *x -= m);
        }
        v
    }
    fn rows(&self, x: &[f32], d: usize) -> Vec<f32> {
        let mut out = vec![0f32; x.len()];
        for (xi, oi) in x.chunks(d).zip(out.chunks_mut(d)) {
            self.apply(xi, oi);
        }
        out
    }
    /// Linear map of a part c with the scale s held fixed (taken from the whole sum):
    /// RMS: c*s*w, LayerNorm: (c-mean(c))*s*w (the bias belongs to no part)
    pub fn map_part(&self, s: f32, c: &[f32]) -> Vec<f32> {
        let m = if self.rms { 0.0 } else { c.iter().sum::<f32>() / c.len() as f32 };
        c.iter().zip(&self.w).map(|(v, w)| (v - m) * s * w).collect()
    }
}

pub struct Rope {
    pub inv_freq: Vec<f32>, // length = rotated dims / 2 (entry 0 = not rotated)
    pub neox: bool,         // pairs (i, i+h/2) instead of (2i, 2i+1)
    pub pos_scale: f32,
}

impl Rope {
    fn apply(&self, v: &mut [f32], pos: usize) {
        let half = self.inv_freq.len();
        let p = pos as f32 / self.pos_scale;
        for i in 0..half {
            let (s, c) = (p * self.inv_freq[i]).sin_cos();
            let (i0, i1) = if self.neox { (i, i + half) } else { (2 * i, 2 * i + 1) };
            let (x0, x1) = (v[i0], v[i1]);
            v[i0] = x0 * c - x1 * s;
            v[i1] = x1 * c + x0 * s;
        }
    }
}

fn inv_freq(base: f32, dims: usize, denom: usize, div: Option<&[f32]>) -> Vec<f32> {
    (0..dims / 2)
        .map(|i| 1.0 / base.powf((2 * i) as f32 / denom as f32) / div.map_or(1.0, |v| v[i]))
        .collect()
}

pub struct Attn {
    pub n_head: usize,
    pub n_kv: usize,
    pub hd: usize,
    pub wq: Option<Mat>,
    pub wk: Option<Mat>,
    pub wv: Option<Mat>,
    pub wqkv: Option<Mat>, // GPT-NeoX: [q|k|v] per head
    pub bq: Option<Vec<f32>>,
    pub bk: Option<Vec<f32>>,
    pub bv: Option<Vec<f32>>,
    pub bqkv: Option<Vec<f32>>,
    pub qn: Option<Norm>,
    pub kn: Option<Norm>,
    pub vn: Option<Norm>,
    pub rope: Option<Rope>,
    pub scale: f32,
    pub softcap: Option<f32>,
    pub window: Option<usize>,
    pub causal: bool,
    pub kv_from: Option<usize>, // reuse keys/values of an earlier layer
    pub v_is_k: bool,
    pub wo: Mat,
    pub bo: Option<Vec<f32>>,
}

pub struct Mlp {
    pub gate: Option<Mat>,
    pub up: Mat,
    pub bup: Option<Vec<f32>>,
    pub down: Mat,
    pub bdown: Option<Vec<f32>>,
    pub act: Act,
    down_norms: OnceLock<Vec<f32>>,
}

/// Gemma 4: extra embedding per layer, gated into the stream
pub struct Ple {
    pub gate: Mat,
    pub proj: Mat,
    pub norm: Norm,
    pub act: Act,
}

#[derive(PartialEq, Clone, Copy)]
pub enum Res {
    Pre,      // x += f(norm(x))
    Parallel, // x += attn(n1(x)) + mlp(n2(x))
    Post,     // x = norm(x + f(x))  (BERT/RoBERTa)
}

pub struct Block {
    pub res: Res,
    pub n_attn: Option<Norm>,
    pub n_attn_post: Option<Norm>,
    pub n_mlp: Option<Norm>,
    pub n_mlp_post: Option<Norm>,
    pub attn: Attn,
    pub mlp: Mlp,
    pub ple: Option<Ple>,
    pub scalar: Option<f32>,
    pub label: String,
}

pub struct PleModel {
    embed: Mat,
    embed_scale: f32,
    proj: Mat,
    proj_scale: f32,
    norm: Norm,
    pub dp: usize,
}

impl PleModel {
    /// projection of the token vector into the per-layer inputs of all layers [L*dp][d]
    pub fn proj(&self) -> &Mat {
        &self.proj
    }
}

pub struct Model {
    pub name: String,
    pub format: String,
    pub arch: String,
    pub weight_types: String,
    pub d: usize,
    pub embed: Mat,
    pub embed_scale: f32,
    pub pos_embed: Option<(Mat, usize)>,
    pub type_embed: Option<Vec<f32>>,
    pub embed_norm: Option<Norm>,
    pub ple: Option<PleModel>,
    pub blocks: Vec<Block>,
    pub final_norm: Option<Norm>,
    lm_head: Option<Mat>,
    pub has_head: bool,
    pub logit_cap: Option<f32>,
    pub tok: Tokenizer,
}

// ====================================================================== recording

pub struct Trace {
    /// logits at the focus token
    pub logits: Vec<f32>,
    pub focus_resid: Vec<Vec<f32>>, // [L+1][d] stream at the focus after the embedding / each layer
    pub focus_heads: Vec<Vec<f32>>, // [L][H*d] what each head writes at the focus (after the post-norm)
    pub focus_attn_bias: Vec<Option<Vec<f32>>>,
    pub focus_mlp: Vec<Vec<f32>>, // [L][d] (without bias, after a post-norm if any)
    pub focus_mlp_bias: Vec<Option<Vec<f32>>>,
    pub focus_mlp_post: Vec<Option<f32>>, // scale of the MLP post-norm at the focus
    pub focus_ple: Vec<Option<Vec<f32>>>,
    pub focus_act: Vec<Vec<f32>>,   // [L][ff] MLP activations at the focus
    pub attn_focus: Vec<Vec<f32>>,  // [L][H*T] attention weights of the focus row
    pub head_norm: Vec<Vec<f32>>,   // [L][T*H] length of what each head writes
    /// stream that the MLP pre-norm reads at the focus [L][d] (Pre: after the attention, Parallel: before the layer)
    pub focus_mlp_x: Vec<Vec<f32>>,
    /// MLP input at the focus, i.e. the output of the pre-norm [L][d]
    pub focus_mlp_in: Vec<Vec<f32>>,
    /// scale of the attention post-norm at the focus
    pub focus_attn_post: Vec<Option<f32>>,
    /// gated input of the per-layer embedding at the focus [dp] and the scale of its norm
    pub focus_ple_gated: Vec<Option<(Vec<f32>, f32)>>,
    /// only with record: all MLP activations [L][T*ff]
    pub acts: Vec<Vec<f32>>,
    /// only with record: gated input of the per-layer embedding [L][T*dp] (empty without PLE)
    pub ple_gated: Vec<Vec<f32>>,
    /// only with atoms: contribution of every (source token, head) pair at the focus, after the post-norm: [L][nf*H*d]
    pub atom_attn: Vec<Vec<f32>>,
    /// only with keep_final: stream at the end, all rows
    pub x_final: Vec<f32>,
    /// only with checkpoint
    pub checkpoint: Option<Checkpoint>,
    /// only with record_rows
    pub rows: Vec<RowRecord>,
    /// only with dump: stream after the embedding and after each layer, all positions
    pub hidden_all: Option<Vec<Vec<f32>>>,
    /// only with dump: logits at all positions
    pub logits_all: Option<Vec<f32>>,
    /// only with record_inputs: per layer the inputs of every matrix product, all positions
    pub inputs: Vec<LayerInputs>,
    /// only with record_inputs: token vectors (input of the PLE input projection) [T][d]
    pub x_tok: Vec<f32>,
    /// only with record_norms
    pub norms: Option<NormScales>,
}

/// Ablation: set the output of a component to zero
#[derive(Clone, Debug, PartialEq)]
pub enum Ablation {
    /// at all positions
    Attn(usize),
    Mlp(usize),
    Ple(usize),
    /// only at one row: (layer, row)
    AttnAt(usize, usize),
    MlpAt(usize, usize),
    PleAt(usize, usize),
}

#[derive(Clone, Default)]
pub struct RunOpts {
    pub ablate: Vec<Ablation>,
    /// record all MLP activations and gated PLE inputs (all positions)
    pub record: bool,
    /// record per (source token, head) contributions at the focus
    pub atoms: bool,
    /// keep the stream before every layer (all positions) and shared K/V, for reruns from a layer
    pub checkpoint: bool,
    /// keep the stream at the end for all rows (tr.x_final)
    pub keep_final: bool,
    /// keep everything the reference tests compare (hidden_all, logits_all)
    pub dump: bool,
    /// set single neurons to a value: (layer, row, neuron, value)
    pub patch: Vec<(usize, usize, usize, f32)>,
    /// scale a head's output in one row: (layer, head, row, factor)
    pub head_scale: Vec<(usize, usize, usize, f32)>,
    /// scale the input of one per-layer-embedding dimension: (layer, row, dimension, factor)
    pub ple_scale: Vec<(usize, usize, usize, f32)>,
    /// start computing at this layer with the given stream (afterwards only the logits are valid)
    pub start: Option<Start>,
    /// variant batch (see Variants)
    pub variants: Option<Arc<Variants>>,
    /// record these rows in every layer (RowRecord)
    pub record_rows: Vec<usize>,
    /// switched-off scalar products, per row of the pass (see Cut)
    pub cut: Option<Arc<Cut>>,
    /// record the inputs of every matrix product at every position (tr.inputs, tr.x_tok)
    pub record_inputs: bool,
    /// record the scale of every norm at every position (tr.norms)
    pub record_norms: bool,
    /// every norm uses these scales (by position) instead of computing its own: frozen norms
    pub frozen_norms: Option<Arc<NormScales>>,
}

/// The scale (1/rms or 1/σ) of every norm at every position of one pass
#[derive(Clone, Default)]
pub struct NormScales {
    /// [T] embedding norm (if any)
    pub embed: Vec<f32>,
    /// [T][L] norm of the per-layer input projection (Gemma 4)
    pub ple_in: Vec<f32>,
    pub layers: Vec<LayerNorms>,
    /// [T] final norm
    pub fin: Vec<f32>,
}

#[derive(Clone, Default)]
pub struct LayerNorms {
    /// [T] attention pre-norm, [T][H] q norm, [T][KV] k and v norms, [T] attention post-norm
    pub attn: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub attn_post: Vec<f32>,
    /// [T] MLP pre-norm and post-norm, [T] norm of the per-layer embedding output
    pub mlp: Vec<f32>,
    pub mlp_post: Vec<f32>,
    pub ple: Vec<f32>,
}

// ====================================================================== cuts: switched-off scalar products

/// The matrix products of one layer whose single scalar products can be switched off.
/// QKᵀ and A·V (inside attention) are handled per head by HeadCut.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum Site {
    Q,
    K,
    V,
    O,
    Gate,
    Up,
    Down,
    /// Gemma 4: projection of the token vector into the per-layer input (units = dimensions of this layer)
    PleIn,
    PleGate,
    PleProj,
}

pub const N_SITES: usize = 10;

impl Site {
    /// Row-wise: a unit is one output y_o = Σ_c W[o,c]·x_c, its atoms run along c.
    /// Column-wise: a unit is one input x_c, its atoms W[r,c]·x_c run along the outputs r.
    pub fn row_wise(self) -> bool {
        !matches!(self, Site::O | Site::Down | Site::PleProj)
    }
    pub fn name(self) -> &'static str {
        match self {
            Site::Q => "Q",
            Site::K => "K",
            Site::V => "V",
            Site::O => "O",
            Site::Gate => "gate",
            Site::Up => "up",
            Site::Down => "down",
            Site::PleIn => "PLE-in",
            Site::PleGate => "PLE-gate",
            Site::PleProj => "PLE-proj",
        }
    }
}

/// Atoms of one unit (sorted indices along its atom axis)
#[derive(Clone, Debug)]
pub enum Atoms {
    /// these atoms are switched off, all others stay on
    Off(Arc<Vec<u32>>),
    /// only these atoms stay on
    On(Arc<Vec<u32>>),
}

impl Atoms {
    #[inline]
    pub fn is_off(&self, a: u32) -> bool {
        match self {
            Atoms::Off(l) => l.binary_search(&a).is_ok(),
            Atoms::On(l) => l.binary_search(&a).is_err(),
        }
    }
}

/// One unit with part of its atoms switched off; `w` holds the unit's weights along its atom axis
/// (row W[o,:] for a row-wise site, column W[:,c] for a column-wise one)
#[derive(Clone)]
pub struct PartialUnit {
    pub unit: u32,
    pub w: Arc<Vec<f32>>,
    pub atoms: Atoms,
    /// mean ablation, row-wise sites: Σ over the switched-off atoms of W[o,a]·x̄_a (a constant)
    pub mean_off: f32,
}

/// Switched-off scalar products of one matrix product at one row
#[derive(Clone, Default)]
pub struct MatCut {
    /// units switched off completely (every atom)
    pub off: Vec<u32>,
    pub partial: Vec<PartialUnit>,
    /// mean ablation, column-wise sites: Σ over the switched-off atoms of partial units of W[r,u]·x̄_u, laid out
    /// like one row of the output (a constant added to every row with this cut)
    pub mean_add: Option<Arc<Vec<f32>>>,
}

/// Switched-off products of QKᵀ (score q·k_j) or A·V (a_j·v_j) for one head at one query row; keys by position
#[derive(Clone, Default)]
pub struct KeyCut {
    pub off: Vec<u32>,
    /// key position, dimensions switched off
    pub partial: Vec<(u32, Atoms)>,
}

pub enum KeyMode<'a> {
    On,
    Off,
    Partial(&'a Atoms),
}

impl KeyCut {
    #[inline]
    pub fn mode(&self, pos: u32) -> KeyMode<'_> {
        if self.off.contains(&pos) {
            return KeyMode::Off;
        }
        match self.partial.iter().find(|(p, _)| *p == pos) {
            Some((_, a)) => KeyMode::Partial(a),
            None => KeyMode::On,
        }
    }
}

#[derive(Clone, Default)]
pub struct HeadCut {
    pub qk: KeyCut,
    pub av: KeyCut,
}

#[derive(Clone, Default)]
pub struct LayerCut {
    pub sites: [Option<Arc<MatCut>>; N_SITES],
    /// per head (empty = no head cut)
    pub heads: Vec<Option<Arc<HeadCut>>>,
}

/// Per row of the pass, per layer: which scalar products are switched off. Without `means` a
/// switched-off product is simply missing from its sum (zero ablation); with `means` it is replaced by
/// its mean over a reference distribution (mean ablation). Biases, norms, softmax, activations and RoPE
/// always run on what results.
pub struct Cut {
    pub rows: Vec<Vec<Option<Arc<LayerCut>>>>,
    pub means: Option<Arc<Means>>,
}

/// Mean ablation: what a switched-off atom stands for, per layer
#[derive(Clone, Default)]
pub struct LayerMeans {
    /// per site the mean input: row-wise sites x̄ along the atoms [n_in], column-wise x̄ per unit [n_units]
    pub x: [Vec<f32>; N_SITES],
    /// row-wise sites: W·x̄ per unit, i.e. what a unit switched off completely gives (before the bias)
    pub y: [Vec<f32>; N_SITES],
    /// [nh][hd] mean of q_e·k_e and of a_j·v_j[e] over all (query, key) pairs of the reference
    pub qk: Vec<f32>,
    pub av: Vec<f32>,
}

#[derive(Clone, Default)]
pub struct Means {
    pub layers: Vec<LayerMeans>,
}

impl Cut {
    #[inline]
    fn layer(&self, row: usize, l: usize) -> Option<&LayerCut> {
        self.rows.get(row)?.get(l)?.as_deref()
    }
    #[inline]
    fn site(&self, row: usize, l: usize, s: Site) -> Option<&MatCut> {
        self.layer(row, l)?.sites[s as usize].as_deref()
    }
    #[inline]
    fn head(&self, row: usize, l: usize, h: usize) -> Option<&HeadCut> {
        self.layer(row, l)?.heads.get(h)?.as_deref()
    }
    #[inline]
    fn lmeans(&self, l: usize) -> Option<&LayerMeans> {
        self.means.as_deref().map(|m| &m.layers[l])
    }
}

/// Row-wise cut on y [t][ny] (y = W·x, before the bias). Unit u is column map(u) of y. A partially cut
/// unit is summed directly over its atoms that stay on (never "full sum minus the switched-off ones",
/// which would lose precision when the switched-off products are the large ones).
#[allow(clippy::too_many_arguments)]
fn cut_rows(y: &mut [f32], x: &[f32], t: usize, ny: usize, cut: Option<&Cut>, l: usize, s: Site, map: impl Fn(usize) -> usize + Sync) {
    let Some(c) = cut else { return };
    let nx = x.len() / t;
    let ybar = c.lmeans(l).map(|m| &m.y[s as usize][..]);
    y[..t * ny].par_chunks_mut(ny).enumerate().for_each(|(i, yi)| {
        let Some(mc) = c.site(i, l, s) else { return };
        let xi = &x[i * nx..(i + 1) * nx];
        for &u in &mc.off {
            yi[map(u as usize)] = ybar.map_or(0.0, |y| y[u as usize]);
        }
        for p in &mc.partial {
            let o = map(p.unit as usize);
            // the atoms that stay on, summed directly; with mean ablation plus the constant of the others
            let on = match &p.atoms {
                Atoms::Off(list) => {
                    let mut acc = 0f32;
                    let mut k = 0;
                    for a in 0..nx {
                        if k < list.len() && list[k] as usize == a {
                            k += 1;
                            continue;
                        }
                        acc += p.w[a] * xi[a];
                    }
                    acc
                }
                Atoms::On(list) => list.iter().map(|&a| p.w[a as usize] * xi[a as usize]).sum::<f32>(),
            };
            yi[o] = if ybar.is_some() { on + p.mean_off } else { on };
        }
    });
}

/// Column-wise cut, part 1: the input x [t][nx] with every cut unit set to 0 – switched off completely
/// or partially (the atoms of a partial unit that stay on are added back one by one in part 2).
/// None: nothing to do.
fn cut_cols_in(x: &[f32], t: usize, cut: Option<&Cut>, l: usize, s: Site) -> Option<Vec<f32>> {
    let c = cut?;
    if !(0..t).any(|i| c.site(i, l, s).is_some()) {
        return None;
    }
    let nx = x.len() / t;
    let xbar = c.lmeans(l).map(|m| &m.x[s as usize][..]);
    let mut v = x.to_vec();
    for i in 0..t {
        if let Some(mc) = c.site(i, l, s) {
            for &u in &mc.off {
                v[i * nx + u as usize] = xbar.map_or(0.0, |xb| xb[u as usize]);
            }
            for p in &mc.partial {
                v[i * nx + p.unit as usize] = 0.0;
            }
        }
    }
    Some(v)
}

/// Column-wise cut, part 2 (after y = W·x' with x' from part 1): add the atoms W[r,u]·x_u that stay on
/// of every partially cut unit u. x is the original input; y has `ny` values per row, and idx(unit, r)
/// is the position of output r within the row; nout = number of outputs.
#[allow(clippy::too_many_arguments)]
fn cut_cols_out(y: &mut [f32], x: &[f32], t: usize, nout: usize, ny: usize, cut: Option<&Cut>, l: usize, s: Site, idx: impl Fn(usize, usize) -> usize + Sync) {
    let Some(c) = cut else { return };
    let nx = x.len() / t;
    y[..t * ny].par_chunks_mut(ny).enumerate().for_each(|(i, yi)| {
        let Some(mc) = c.site(i, l, s) else { return };
        // mean ablation: the switched-off atoms of partial units, a constant per row
        if let Some(ma) = &mc.mean_add {
            for (a, m) in yi.iter_mut().zip(ma.iter()) {
                *a += m;
            }
        }
        for p in &mc.partial {
            let u = p.unit as usize;
            let xu = x[i * nx + u];
            match &p.atoms {
                Atoms::On(list) => {
                    for &r in list.iter() {
                        yi[idx(u, r as usize)] += p.w[r as usize] * xu;
                    }
                }
                Atoms::Off(list) => {
                    let mut k = 0;
                    for r in 0..nout {
                        if k < list.len() && list[k] as usize == r {
                            k += 1;
                            continue;
                        }
                        yi[idx(u, r)] += p.w[r] * xu;
                    }
                }
            }
        }
    });
}

/// Inputs of every matrix product of one layer at every position (record_inputs)
#[derive(Clone, Default)]
pub struct LayerInputs {
    /// [T][d] input of Q/K/V (output of the attention pre-norm)
    pub h: Vec<f32>,
    /// [T][nh*hd] Q, [T][kvd] K and V as the projections give them (after the cut and the bias)
    pub q_proj: Vec<f32>,
    pub k_proj: Vec<f32>,
    pub v_proj: Vec<f32>,
    /// [T][nh*hd] q and [T][kvd] k after norm and RoPE, [T][kvd] v after its norm: what QKᵀ and A·V read
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    /// [T][nh][T] attention weights (key by position)
    pub probs: Vec<f32>,
    /// [T][nh*hd] input of W_O
    pub z: Vec<f32>,
    /// [T][d] MLP input (output of the pre-norm)
    pub h2: Vec<f32>,
    /// [T][ff] gate and up projections (before the activation), activation = input of W_down
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub act: Vec<f32>,
    /// [T][d] stream read by the PLE gate, [T][dp] input of the PLE projection
    pub ple_x: Vec<f32>,
    pub ple_g: Vec<f32>,
}

/// Many variants of one prompt in one pass: the first T rows are the real prompt (row = position);
/// every variant is a copy of the positions from its start up to the focus and sees, via attention,
/// the clean prefix before its start and itself – never other variants.
pub struct Variants {
    pub pos: Vec<usize>,       // per row: real position
    pub seg_start: Vec<usize>, // per row: first position of its segment (clean: 0)
    pub seg_row0: Vec<usize>,  // per row: row where its segment begins (clean: 0)
}

/// Recording of selected rows in one layer
#[derive(Clone)]
pub struct RowRecord {
    pub layer: usize,
    pub x: Vec<f32>,     // [nz][d] stream after the layer
    pub attn: Vec<f32>,  // [nz][d] what attention writes
    pub mlp: Vec<f32>,   // [nz][d] what the MLP writes
    pub ple: Vec<f32>,   // [nz][d] per-layer embedding (else 0)
    pub act: Vec<f32>,   // [nz][ff]
    pub probs: Vec<f32>, // [nz][H][T] attention weights
}

/// Intermediate state of a pass: stream before every layer [L][T*d], shared K/V
pub struct Checkpoint {
    pub x: Vec<Vec<f32>>,
    pub kv: Vec<Option<(Vec<f32>, Vec<f32>)>>,
}

/// Rerun from layer `layer` with stream `x` [T*d]; K/V of earlier layers from `kv`
#[derive(Clone)]
pub struct Start {
    pub layer: usize,
    pub x: Vec<f32>,
    pub kv: Arc<Vec<Option<(Vec<f32>, Vec<f32>)>>>,
}

impl RunOpts {
    fn has(&self, a: &Ablation) -> bool {
        self.ablate.contains(a)
    }
}

fn add_bias(m: &mut [f32], b: &Option<Vec<f32>>, w: usize) {
    if let Some(b) = b {
        for r in m.chunks_mut(w) {
            for (a, bb) in r.iter_mut().zip(b) {
                *a += bb;
            }
        }
    }
}

pub fn top_probs(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let z: f64 = logits.iter().map(|v| ((v - mx) as f64).exp()).sum();
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    let k = k.min(idx.len());
    idx.select_nth_unstable_by(k.saturating_sub(1), |a, b| logits[*b].partial_cmp(&logits[*a]).unwrap_or(std::cmp::Ordering::Equal));
    let mut top: Vec<(u32, f32)> = idx[..k].iter().map(|&i| (i as u32, (((logits[i] - mx) as f64).exp() / z) as f32)).collect();
    top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    top
}

struct AttnOut {
    pz: Vec<f32>,    // [nz][H][T] weights of the recorded rows
    heads: Vec<f32>, // [T][H][d] contribution of every head (without bias)
    probs_focus: Vec<f32>,
    src_heads: Vec<f32>, // [nf][H][d] only with atoms
    k: Vec<f32>,
    v: Vec<f32>,
    rec: Option<LayerInputs>, // only with record_inputs
    /// scales of the q, k and v norms per row (empty without such norms)
    nq: Vec<f32>,
    nk: Vec<f32>,
    nv: Vec<f32>,
}

// ====================================================================== forward pass

impl Model {
    pub fn lm_head(&self) -> &Mat {
        self.lm_head.as_ref().unwrap_or(&self.embed)
    }
    pub fn vocab_rows(&self) -> usize {
        self.lm_head().rows
    }
    pub fn n_layer(&self) -> usize {
        self.blocks.len()
    }

    /// final_logits of one row with the final norm's scale given (frozen norms)
    pub fn final_logits_scaled(&self, x: &[f32], s: f32) -> Vec<f32> {
        let mut normed = x.to_vec();
        if let Some(n) = &self.final_norm {
            n.apply_scale(x, &mut normed, s);
        }
        let mut lg = self.lm_head().matmul(&normed, 1);
        if let Some(c) = self.logit_cap {
            for v in lg.iter_mut() {
                *v = (*v / c).tanh() * c;
            }
        }
        lg
    }

    pub fn final_logits(&self, x: &[f32], t: usize) -> Vec<f32> {
        let normed = match &self.final_norm {
            Some(n) => n.rows(x, self.d),
            None => x.to_vec(),
        };
        let mut lg = self.lm_head().matmul(&normed, t);
        if let Some(c) = self.logit_cap {
            for v in lg.iter_mut() {
                *v = (*v / c).tanh() * c;
            }
        }
        lg
    }

    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self, l: usize, h: &[f32], t: usize, focus: usize, kv_store: &[Option<(Vec<f32>, Vec<f32>)>], atoms: bool, var: Option<&Variants>, rec_rows: &[usize],
        cut: Option<&Cut>, record: bool, fz: Option<&LayerNorms>,
    ) -> AttnOut {
        let a = &self.blocks[l].attn;
        let posi = |i: usize| var.map_or(i, |v| v.pos[i]);
        // visible rows for row i: two half-open ranges
        let ranges = |i: usize| -> [(usize, usize); 2] {
            match var {
                None => {
                    let hi = if a.causal { i } else { t - 1 };
                    let lo = match a.window {
                        Some(w) if a.causal => (i + 1).saturating_sub(w),
                        _ => 0,
                    };
                    [(lo, hi + 1), (0, 0)]
                }
                Some(v) => {
                    let p = v.pos[i];
                    let lo_pos = match a.window {
                        Some(w) => (p + 1).saturating_sub(w),
                        None => 0,
                    };
                    let (s0, r0) = (v.seg_start[i], v.seg_row0[i]);
                    let q0 = lo_pos.max(s0);
                    [(lo_pos.min(s0), s0), (r0 + (q0 - s0), i + 1)]
                }
            }
        };
        let (nh, hd) = (a.n_head, a.hd);
        let ad = nh * hd;
        let mut rec = record.then(LayerInputs::default);
        let (mut nk, mut nv): (Vec<f32>, Vec<f32>) = (Vec::new(), Vec::new());
        let (mut q, kv) = if let Some(w) = &a.wqkv {
            let mut qkv = w.matmul(h, t);
            // fused [q|k|v] per head: unit u of Q/K/V (head u/hd, dimension u%hd) -> its row in the fused matrix
            for (s, off) in [(Site::Q, 0), (Site::K, hd), (Site::V, 2 * hd)] {
                cut_rows(&mut qkv, h, t, 3 * ad, cut, l, s, |u| (u / hd) * 3 * hd + off + u % hd);
            }
            add_bias(&mut qkv, &a.bqkv, 3 * ad);
            let (mut q, mut k, mut v) = (vec![0f32; t * ad], vec![0f32; t * ad], vec![0f32; t * ad]);
            for i in 0..t {
                for hh in 0..nh {
                    let src = &qkv[i * 3 * ad + hh * 3 * hd..][..3 * hd];
                    q[i * ad + hh * hd..][..hd].copy_from_slice(&src[..hd]);
                    k[i * ad + hh * hd..][..hd].copy_from_slice(&src[hd..2 * hd]);
                    v[i * ad + hh * hd..][..hd].copy_from_slice(&src[2 * hd..]);
                }
            }
            (q, Some((k, v)))
        } else {
            let mut q = a.wq.as_ref().unwrap().matmul(h, t);
            cut_rows(&mut q, h, t, ad, cut, l, Site::Q, |u| u);
            add_bias(&mut q, &a.bq, ad);
            let kv = if a.kv_from.is_some() {
                None
            } else {
                let kd = a.n_kv * hd;
                let mut k = a.wk.as_ref().unwrap().matmul(h, t);
                cut_rows(&mut k, h, t, kd, cut, l, Site::K, |u| u);
                add_bias(&mut k, &a.bk, kd);
                let mut v = if a.v_is_k {
                    k.clone()
                } else {
                    let mut v = a.wv.as_ref().unwrap().matmul(h, t);
                    cut_rows(&mut v, h, t, kd, cut, l, Site::V, |u| u);
                    add_bias(&mut v, &a.bv, kd);
                    v
                };
                if let Some(r) = rec.as_mut() {
                    r.k_proj = k.clone();
                    r.v_proj = v.clone();
                }
                if let Some(n) = &a.vn {
                    let nkv = a.n_kv;
                    for (ci, c) in v.chunks_mut(hd).enumerate() {
                        let tmp = c.to_vec();
                        let (i, g) = (ci / nkv, ci % nkv);
                        nv.push(n.apply_fz(&tmp, c, fz.map(|f| f.v[posi(i) * nkv + g])));
                    }
                }
                Some((k, v))
            };
            (q, kv)
        };
        if let Some(r) = rec.as_mut() {
            r.h = h.to_vec();
            r.q_proj = q.clone();
            if let (Some(_), Some((k, v))) = (&a.wqkv, &kv) {
                r.k_proj = k.clone();
                r.v_proj = v.clone();
            }
        }
        let n_kv = if a.wqkv.is_some() { nh } else { a.n_kv };
        let kvd = n_kv * hd;
        let mut nq = vec![0f32; t * nh];
        q.par_chunks_mut(ad).zip(nq.par_chunks_mut(nh)).enumerate().for_each(|(i, (qr, nqr))| {
            for hh in 0..nh {
                let s = &mut qr[hh * hd..][..hd];
                if let Some(n) = &a.qn {
                    let tmp = s.to_vec();
                    nqr[hh] = n.apply_fz(&tmp, s, fz.map(|f| f.q[posi(i) * nh + hh]));
                }
                if let Some(r) = &a.rope {
                    r.apply(s, posi(i));
                }
            }
        });
        if a.qn.is_none() {
            nq = Vec::new();
        }
        let (k, v) = match kv {
            Some((mut k, v)) => {
                nk = vec![0f32; t * n_kv];
                k.par_chunks_mut(kvd).zip(nk.par_chunks_mut(n_kv)).enumerate().for_each(|(i, (kr, nkr))| {
                    for g in 0..n_kv {
                        let s = &mut kr[g * hd..][..hd];
                        if let Some(n) = &a.kn {
                            let tmp = s.to_vec();
                            nkr[g] = n.apply_fz(&tmp, s, fz.map(|f| f.k[posi(i) * n_kv + g]));
                        }
                        if let Some(r) = &a.rope {
                            r.apply(s, posi(i));
                        }
                    }
                });
                if a.kn.is_none() {
                    nk = Vec::new();
                }
                (k, v)
            }
            None => {
                let src = kv_store[a.kv_from.unwrap()].as_ref().expect("shared K/V missing");
                (src.0.clone(), src.1.clone())
            }
        };
        let rep = nh / n_kv;
        let nz = rec_rows.len();
        // every (head, row) on its own, in parallel: z_i [hd] and, where needed, the weights of row i [t]
        // (the arithmetic per row is the same as row by row)
        let need_p = |i: usize| i == focus || record || (nz > 0 && rec_rows.contains(&i));
        type RowOut = (Vec<f32>, Option<Vec<f32>>);
        let rows_out: Vec<RowOut> = (0..nh * t)
            .into_par_iter()
            .map_init(
                || vec![0f32; t],
                |s, idx| {
                    let (hh, i) = (idx / t, idx % t);
                    let g = hh / rep;
                    let mut zi = vec![0f32; hd];
                    let qi = &q[i * ad + hh * hd..][..hd];
                    let br = ranges(i);
                    let hc = cut.and_then(|c| c.head(i, l, hh));
                    let lm = cut.and_then(|c| c.lmeans(l));
                    let qkm = lm.map(|m| &m.qk[hh * hd..(hh + 1) * hd]);
                    let avm = lm.map(|m| &m.av[hh * hd..(hh + 1) * hd]);
                    let mut mx = f32::NEG_INFINITY;
                    for &(lo, hi) in &br {
                        for j in lo..hi {
                            let kj = &k[j * kvd + g * hd..][..hd];
                            let raw = match (hc.map(|c| c.qk.mode(posi(j) as u32)), qkm) {
                                (None | Some(KeyMode::On), _) => dot(qi, kj),
                                (Some(KeyMode::Off), None) => 0.0,
                                (Some(KeyMode::Partial(at)), None) => (0..hd).filter(|&e| !at.is_off(e as u32)).map(|e| qi[e] * kj[e]).sum(),
                                (Some(KeyMode::Off), Some(m)) => m.iter().sum(),
                                (Some(KeyMode::Partial(at)), Some(m)) => (0..hd).map(|e| if at.is_off(e as u32) { m[e] } else { qi[e] * kj[e] }).sum(),
                            };
                            let mut sc = raw * a.scale;
                            if let Some(c) = a.softcap {
                                sc = (sc / c).tanh() * c;
                            }
                            s[j] = sc;
                            mx = mx.max(sc);
                        }
                    }
                    let mut sum = 0f32;
                    for &(lo, hi) in &br {
                        for j in lo..hi {
                            s[j] = (s[j] - mx).exp();
                            sum += s[j];
                        }
                    }
                    for &(lo, hi) in &br {
                        for j in lo..hi {
                            s[j] /= sum;
                            let vj = &v[j * kvd + g * hd..][..hd];
                            match (hc.map(|c| c.av.mode(posi(j) as u32)), avm) {
                                (None | Some(KeyMode::On), _) => {
                                    for e in 0..hd {
                                        zi[e] += s[j] * vj[e];
                                    }
                                }
                                (Some(KeyMode::Off), None) => {}
                                (Some(KeyMode::Partial(at)), None) => {
                                    for e in 0..hd {
                                        if !at.is_off(e as u32) {
                                            zi[e] += s[j] * vj[e];
                                        }
                                    }
                                }
                                (Some(KeyMode::Off), Some(m)) => {
                                    for e in 0..hd {
                                        zi[e] += m[e];
                                    }
                                }
                                (Some(KeyMode::Partial(at)), Some(m)) => {
                                    for e in 0..hd {
                                        zi[e] += if at.is_off(e as u32) { m[e] } else { s[j] * vj[e] };
                                    }
                                }
                            }
                        }
                    }
                    let pr = need_p(i).then(|| {
                        let mut p = vec![0f32; t];
                        for &(lo, hi) in &br {
                            p[lo..hi].copy_from_slice(&s[lo..hi]);
                        }
                        p
                    });
                    (zi, pr)
                },
            )
            .collect();
        let mut pz_all = vec![0f32; nz * nh * t];
        for hh in 0..nh {
            for (zi, &r) in rec_rows.iter().enumerate() {
                if r < t {
                    pz_all[(zi * nh + hh) * t..(zi * nh + hh + 1) * t].copy_from_slice(rows_out[hh * t + r].1.as_ref().unwrap());
                }
            }
        }
        let mut zcat = vec![0f32; t * ad];
        let mut probs_focus = vec![0f32; nh * t];
        for hh in 0..nh {
            for i in 0..t {
                zcat[i * ad + hh * hd..][..hd].copy_from_slice(&rows_out[hh * t + i].0);
            }
            if focus < t {
                probs_focus[hh * t..(hh + 1) * t].copy_from_slice(rows_out[hh * t + focus].1.as_ref().unwrap());
            }
        }
        // O is column-wise: a unit is one input dimension (head u/hd, dimension u%hd)
        let zmask = cut_cols_in(&zcat, t, cut, l, Site::O);
        let mut heads = a.wo.matmul_split(zmask.as_deref().unwrap_or(&zcat), t, nh);
        let dm = a.wo.rows;
        cut_cols_out(&mut heads, &zcat, t, dm, nh * dm, cut, l, Site::O, |u, r| (u / hd) * dm + r);
        if let Some(r) = rec.as_mut() {
            r.q = q.clone();
            r.k = k.clone();
            r.v = v.clone();
            r.probs = vec![0f32; t * nh * t];
            for hh in 0..nh {
                for i in 0..t {
                    let p = rows_out[hh * t + i].1.as_ref().unwrap();
                    for j in 0..t {
                        r.probs[(i * nh + hh) * t + posi(j)] = p[j];
                    }
                }
            }
            r.z = zcat.clone();
        }
        if !atoms {
            return AttnOut { pz: pz_all, heads, probs_focus, src_heads: Vec::new(), k, v, rec, nq, nk, nv };
        }
        // per source token j and head k separately: W_O,k (a_kj v_kj)
        let nf = if a.causal { focus + 1 } else { t };
        let mut zf = vec![0f32; nf * ad];
        for j in 0..nf {
            for hh in 0..nh {
                let p = probs_focus[hh * t + j];
                let vj = &v[j * kvd + (hh / rep) * hd..][..hd];
                for (x, y) in zf[j * ad + hh * hd..][..hd].iter_mut().zip(vj) {
                    *x = p * y;
                }
            }
        }
        let src_heads = a.wo.matmul_split(&zf, nf, nh);
        AttnOut { pz: pz_all, heads, probs_focus, src_heads, k, v, rec, nq, nk, nv }
    }

    pub fn run(&self, ids: &[u32], focus: usize, o: &RunOpts) -> Trace {
        let (t, d) = (ids.len(), self.d);
        // ablations at single rows
        let (mut attn_at, mut mlp_at, mut ple_at) = (Vec::new(), Vec::new(), Vec::new());
        for a in &o.ablate {
            match a {
                Ablation::AttnAt(l, p) => attn_at.push((*l, *p)),
                Ablation::MlpAt(l, p) => mlp_at.push((*l, *p)),
                Ablation::PleAt(l, p) => ple_at.push((*l, *p)),
                _ => {}
            }
        }
        let var = o.variants.as_deref();
        if let Some(v) = var {
            assert_eq!(v.pos.len(), t, "variants: one position per row");
        }
        let posi = |i: usize| var.map_or(i, |v| v.pos[i]);
        let mut head_scale: HashMap<(usize, usize, usize), f32> = HashMap::new();
        for &(l, h, r, f) in &o.head_scale {
            *head_scale.entry((l, h, r)).or_insert(1.0) *= f;
        }
        let zero_rows = |v: &mut [f32], at: &[(usize, usize)], l: usize| {
            for &(a, p) in at {
                if a == l && p < t {
                    v[p * d..(p + 1) * d].iter_mut().for_each(|x| *x = 0.0);
                }
            }
        };
        // --- embedding
        let mut x = vec![0f32; t * d];
        for (i, &id) in ids.iter().enumerate() {
            let r = &mut x[i * d..(i + 1) * d];
            self.embed.row(id as usize, r);
            if self.embed_scale != 1.0 {
                for v in r.iter_mut() {
                    *v *= self.embed_scale;
                }
            }
        }
        let x_tok = x.clone();
        if let Some((pe, off)) = &self.pos_embed {
            let mut buf = vec![0f32; d];
            for i in 0..t {
                pe.row(posi(i) + off, &mut buf);
                for (a, b) in x[i * d..(i + 1) * d].iter_mut().zip(&buf) {
                    *a += b;
                }
            }
        }
        if let Some(te) = &self.type_embed {
            for r in x.chunks_mut(d) {
                for (a, b) in r.iter_mut().zip(te) {
                    *a += b;
                }
            }
        }
        let fz = o.frozen_norms.as_deref();
        let mut ns = o.record_norms.then(NormScales::default);
        if let Some(n) = &self.embed_norm {
            let (xn, sc) = n.rows_fz(&x, d, fz.map(|f| &f.embed[..]), &posi);
            x = xn;
            if let Some(r) = ns.as_mut() {
                r.embed = sc;
            }
        }
        // Gemma 4: input per layer = (norm(projection of the token vector) + own table) / sqrt(2)
        let nl = self.blocks.len();
        let cut = o.cut.as_deref();
        if let Some(c) = cut {
            assert_eq!(c.rows.len(), t, "cut: one entry per row");
        }
        let mut ple_in_sc: Vec<f32> = Vec::new();
        let per_layer: Option<Vec<f32>> = self.ple.as_ref().map(|p| {
            let w = nl * p.dp;
            let mut proj = p.proj.matmul(&x_tok, t);
            for l in 0..nl {
                cut_rows(&mut proj, &x_tok, t, w, cut, l, Site::PleIn, |u| l * p.dp + u);
            }
            for v in proj.iter_mut() {
                *v *= p.proj_scale;
            }
            let mut out = vec![0f32; t * w];
            let mut tok = vec![0f32; w];
            for i in 0..t {
                p.embed.row(ids[i] as usize, &mut tok);
                for l in 0..nl {
                    let seg = &proj[i * w + l * p.dp..][..p.dp];
                    let o2 = &mut out[i * w + l * p.dp..][..p.dp];
                    ple_in_sc.push(p.norm.apply_fz(seg, o2, fz.map(|f| f.ple_in[posi(i) * nl + l])));
                    for e in 0..p.dp {
                        o2[e] = (o2[e] + tok[l * p.dp + e] * p.embed_scale) * std::f32::consts::FRAC_1_SQRT_2;
                    }
                }
            }
            out
        });
        let mut tr = Trace {
            logits: vec![],
            focus_resid: vec![x[focus * d..(focus + 1) * d].to_vec()],
            focus_heads: vec![],
            focus_attn_bias: vec![],
            focus_mlp: vec![],
            focus_mlp_bias: vec![],
            focus_mlp_post: vec![],
            focus_ple: vec![],
            focus_act: vec![],
            attn_focus: vec![],
            head_norm: vec![],
            focus_mlp_x: vec![],
            focus_mlp_in: vec![],
            focus_attn_post: vec![],
            focus_ple_gated: vec![],
            acts: vec![],
            ple_gated: vec![],
            atom_attn: vec![],
            x_final: vec![],
            checkpoint: None,
            rows: vec![],
            hidden_all: if o.dump { Some(vec![x.clone()]) } else { None },
            logits_all: None,
            inputs: vec![],
            x_tok: if o.record_inputs { x_tok.clone() } else { vec![] },
            norms: None,
        };
        if let Some(r) = ns.as_mut() {
            r.ple_in = std::mem::take(&mut ple_in_sc);
        }
        let mut kv_store: Vec<Option<(Vec<f32>, Vec<f32>)>> = (0..nl).map(|_| None).collect();
        let shared: Vec<bool> = (0..nl).map(|l| self.blocks.iter().any(|b| b.attn.kv_from == Some(l))).collect();
        let l_start = o.start.as_ref().map_or(0, |s| s.layer);
        if let Some(s) = &o.start {
            x = s.x.clone();
            kv_store = (*s.kv).clone();
        }
        let mut x_ck: Vec<Vec<f32>> = Vec::new();
        for (l, b) in self.blocks.iter().enumerate() {
            if l < l_start {
                continue;
            }
            if o.checkpoint {
                x_ck.push(x.clone());
            }
            let x_old = x.clone();
            let mut ple_rows: Option<Vec<f32>> = None;
            let nh = b.attn.n_head;
            // ---------------- attention
            let fzl = fz.map(|f| &f.layers[l]);
            let mut ln = LayerNorms::default();
            let h = match (&b.n_attn, b.res) {
                (Some(n), Res::Pre | Res::Parallel) => {
                    let (hn, sc) = n.rows_fz(&x, d, fzl.map(|f| &f.attn[..]), &posi);
                    ln.attn = sc;
                    hn
                }
                _ => x.clone(),
            };
            let mut ao = self.attention(l, &h, t, focus, &kv_store, o.atoms, var, &o.record_rows, cut, o.record_inputs, fzl);
            ln.q = std::mem::take(&mut ao.nq);
            ln.k = std::mem::take(&mut ao.nk);
            ln.v = std::mem::take(&mut ao.nv);
            let mut rec = ao.rec.take();
            let attn_off = o.has(&Ablation::Attn(l)) || attn_at.contains(&(l, focus));
            if shared[l] {
                kv_store[l] = Some((ao.k.clone(), ao.v.clone()));
            }
            let mut attn = vec![0f32; t * d];
            for i in 0..t {
                for hh in 0..nh {
                    let f = if head_scale.is_empty() { 1.0 } else { head_scale.get(&(l, hh, i)).copied().unwrap_or(1.0) };
                    if f == 0.0 {
                        continue;
                    }
                    for (a, c) in attn[i * d..(i + 1) * d].iter_mut().zip(&ao.heads[(i * nh + hh) * d..][..d]) {
                        *a += f * c;
                    }
                }
            }
            add_bias(&mut attn, &b.attn.bo, d);
            if o.has(&Ablation::Attn(l)) {
                attn.iter_mut().for_each(|v| *v = 0.0);
            }
            zero_rows(&mut attn, &attn_at, l);
            // post-norm (Gemma) on the attention output: parts are scaled linearly along
            let post = if b.res == Res::Pre { b.n_attn_post.as_ref() } else { None };
            let mut written = attn.clone();
            let mut s_pos = vec![1f32; t];
            if let Some(n) = post {
                for i in 0..t {
                    s_pos[i] = n.apply_fz(&attn[i * d..(i + 1) * d], &mut written[i * d..(i + 1) * d], fzl.map(|f| f.attn_post[posi(i)]));
                }
                ln.attn_post = s_pos.clone();
            }
            zero_rows(&mut written, &attn_at, l);
            let map = |c: &[f32], i: usize| -> Vec<f32> { post.map_or_else(|| c.to_vec(), |n| n.map_part(s_pos[i], c)) };
            let mut head_norm = vec![0f32; t * nh];
            for i in 0..t {
                for hh in 0..nh {
                    let c = &ao.heads[(i * nh + hh) * d..][..d];
                    head_norm[i * nh + hh] = if post.is_some() { norm(&map(c, i)) } else { norm(c) };
                }
            }
            let fh: Vec<f32> = if attn_off { vec![0f32; nh * d] } else { (0..nh).flat_map(|hh| map(&ao.heads[(focus * nh + hh) * d..][..d], focus)).collect() };
            tr.focus_heads.push(fh);
            if o.atoms {
                let mut v = Vec::with_capacity(ao.src_heads.len());
                if !attn_off {
                    for c in ao.src_heads.chunks(d) {
                        v.extend(map(c, focus));
                    }
                }
                tr.atom_attn.push(v);
            }
            tr.focus_attn_bias.push(if attn_off { None } else { b.attn.bo.as_ref().map(|bo| map(bo, focus)) });
            tr.head_norm.push(head_norm);
            tr.attn_focus.push(ao.probs_focus);
            tr.focus_attn_post.push(post.map(|_| s_pos[focus]));
            // ---------------- residual + MLP
            let mut mlp_norm = |n: &Norm, xx: &[f32]| -> Vec<f32> {
                let (hn, sc) = n.rows_fz(xx, d, fzl.map(|f| &f.mlp[..]), &posi);
                ln.mlp = sc;
                hn
            };
            let h2 = match b.res {
                Res::Parallel => b.n_mlp.as_ref().map_or_else(|| x_old.clone(), |n| mlp_norm(n, &x_old)),
                Res::Pre => {
                    for (a, w) in x.iter_mut().zip(&written) {
                        *a += w;
                    }
                    b.n_mlp.as_ref().map_or_else(|| x.clone(), |n| mlp_norm(n, &x))
                }
                Res::Post => {
                    for (a, w) in x.iter_mut().zip(&attn) {
                        *a += w;
                    }
                    if let Some(n) = &b.n_attn_post {
                        x = n.rows(&x, d);
                    }
                    x.clone()
                }
            };
            tr.focus_mlp_x.push(match b.res {
                Res::Parallel => x_old[focus * d..(focus + 1) * d].to_vec(),
                _ => x[focus * d..(focus + 1) * d].to_vec(),
            });
            tr.focus_mlp_in.push(h2[focus * d..(focus + 1) * d].to_vec());
            let m = &b.mlp;
            let ff = m.up.rows;
            let mut act = m.up.matmul(&h2, t);
            cut_rows(&mut act, &h2, t, ff, cut, l, Site::Up, |u| u);
            add_bias(&mut act, &m.bup, ff);
            if let Some(r) = rec.as_mut() {
                r.h2 = h2.clone();
                r.up = act.clone();
            }
            let f = act_fn(m.act);
            if let Some(g) = &m.gate {
                let mut gg = g.matmul(&h2, t);
                cut_rows(&mut gg, &h2, t, ff, cut, l, Site::Gate, |u| u);
                if let Some(r) = rec.as_mut() {
                    r.gate = gg.clone();
                }
                act.par_chunks_mut(4096).zip(gg.par_chunks(4096)).for_each(|(a, g)| {
                    for (u, gv) in a.iter_mut().zip(g) {
                        *u *= f(*gv);
                    }
                });
            } else {
                act.par_chunks_mut(4096).for_each(|a| {
                    for u in a.iter_mut() {
                        *u = f(*u);
                    }
                });
            }
            let mlp_off = o.has(&Ablation::Mlp(l)) || mlp_at.contains(&(l, focus));
            if o.has(&Ablation::Mlp(l)) {
                act.iter_mut().for_each(|v| *v = 0.0);
            }
            for &(a, p) in &mlp_at {
                if a == l && p < t {
                    act[p * ff..(p + 1) * ff].iter_mut().for_each(|v| *v = 0.0);
                }
            }
            for &(pl, pos, n, val) in &o.patch {
                if pl == l && pos < t && n < ff {
                    act[pos * ff + n] = val;
                }
            }
            if o.record {
                tr.acts.push(act.clone());
            }
            // W_down is column-wise: a unit is one neuron's activation
            let act_in = cut_cols_in(&act, t, cut, l, Site::Down);
            let mut m0 = m.down.matmul(act_in.as_deref().unwrap_or(&act), t);
            cut_cols_out(&mut m0, &act, t, d, d, cut, l, Site::Down, |_, r| r);
            if let Some(r) = rec.as_mut() {
                r.act = act.clone();
            }
            let mut mo = m0.clone();
            if !o.has(&Ablation::Mlp(l)) {
                add_bias(&mut mo, &m.bdown, d);
            }
            zero_rows(&mut mo, &mlp_at, l);
            let mpost = if b.res == Res::Pre { b.n_mlp_post.as_ref() } else { None };
            let mut mw = mo.clone();
            let mut ms_focus = 1.0;
            if let Some(n) = mpost {
                let mut sc = vec![0f32; t];
                for i in 0..t {
                    let s = n.apply_fz(&mo[i * d..(i + 1) * d], &mut mw[i * d..(i + 1) * d], fzl.map(|f| f.mlp_post[posi(i)]));
                    sc[i] = s;
                    if i == focus {
                        ms_focus = s;
                    }
                }
                ln.mlp_post = sc;
            }
            zero_rows(&mut mw, &mlp_at, l);
            let mmap = |c: &[f32]| -> Vec<f32> { mpost.map_or_else(|| c.to_vec(), |n| n.map_part(ms_focus, c)) };
            tr.focus_mlp.push(mmap(&m0[focus * d..(focus + 1) * d]));
            tr.focus_mlp_bias.push(if mlp_off { None } else { m.bdown.as_ref().map(|bd| mmap(bd)) });
            tr.focus_mlp_post.push(mpost.map(|_| ms_focus));
            tr.focus_act.push(act[focus * ff..(focus + 1) * ff].to_vec());
            match b.res {
                Res::Parallel => {
                    for i in 0..t * d {
                        x[i] = x_old[i] + attn[i] + mo[i];
                    }
                }
                Res::Pre => {
                    for (a, w) in x.iter_mut().zip(&mw) {
                        *a += w;
                    }
                }
                Res::Post => {
                    for (a, w) in x.iter_mut().zip(&mo) {
                        *a += w;
                    }
                    if let Some(n) = &b.n_mlp_post {
                        x = n.rows(&x, d);
                    }
                }
            }
            // ---------------- Gemma 4: input per layer
            if let (Some(p), Some(pl), false) = (&b.ple, &per_layer, o.has(&Ablation::Ple(l))) {
                let dp = p.gate.rows;
                let mut g = p.gate.matmul(&x, t);
                cut_rows(&mut g, &x, t, dp, cut, l, Site::PleGate, |u| u);
                if let Some(r) = rec.as_mut() {
                    r.ple_x = x.clone();
                }
                let fa = act_fn(p.act);
                for i in 0..t {
                    for e in 0..dp {
                        g[i * dp + e] = fa(g[i * dp + e]) * pl[i * nl * dp + l * dp + e];
                    }
                }
                for &(pl_, r, e, f) in &o.ple_scale {
                    if pl_ == l && r < t && e < dp {
                        g[r * dp + e] *= f;
                    }
                }
                if o.record {
                    while tr.ple_gated.len() < l {
                        tr.ple_gated.push(Vec::new());
                    }
                    tr.ple_gated.push(g.clone());
                }
                let g_in = cut_cols_in(&g, t, cut, l, Site::PleProj);
                let mut pr = p.proj.matmul(g_in.as_deref().unwrap_or(&g), t);
                cut_cols_out(&mut pr, &g, t, d, d, cut, l, Site::PleProj, |_, r| r);
                if let Some(r) = rec.as_mut() {
                    r.ple_g = g.clone();
                }
                tr.focus_ple_gated.push(Some((g[focus * dp..(focus + 1) * dp].to_vec(), p.norm.scale(&pr[focus * d..(focus + 1) * d]))));
                let (mut pn, sc) = p.norm.rows_fz(&pr, d, fzl.map(|f| &f.ple[..]), &posi);
                ln.ple = sc;
                zero_rows(&mut pn, &ple_at, l);
                for (a, w) in x.iter_mut().zip(&pn) {
                    *a += w;
                }
                if !o.record_rows.is_empty() {
                    ple_rows = Some(pn.clone());
                }
                tr.focus_ple.push(Some(pn[focus * d..(focus + 1) * d].to_vec()));
            } else {
                tr.focus_ple.push(None);
                tr.focus_ple_gated.push(None);
            }
            if let Some(s) = b.scalar {
                for v in x.iter_mut() {
                    *v *= s;
                }
            }
            if let Some(r) = rec {
                tr.inputs.push(r);
            }
            if let Some(r) = ns.as_mut() {
                r.layers.push(ln);
            }
            if !o.record_rows.is_empty() {
                let nzr = o.record_rows.len();
                let mut z = RowRecord { layer: l, x: Vec::with_capacity(nzr * d), attn: Vec::with_capacity(nzr * d), mlp: Vec::with_capacity(nzr * d), ple: Vec::with_capacity(nzr * d), act: Vec::with_capacity(nzr * ff), probs: ao.pz.clone() };
                let aw: &[f32] = if b.res == Res::Pre { &written } else { &attn };
                let mwr: &[f32] = if b.res == Res::Pre { &mw } else { &mo };
                for &r in &o.record_rows {
                    z.x.extend_from_slice(&x[r * d..(r + 1) * d]);
                    z.attn.extend_from_slice(&aw[r * d..(r + 1) * d]);
                    z.mlp.extend_from_slice(&mwr[r * d..(r + 1) * d]);
                    match &ple_rows {
                        Some(pn) => z.ple.extend_from_slice(&pn[r * d..(r + 1) * d]),
                        None => z.ple.extend(std::iter::repeat(0f32).take(d)),
                    }
                    z.act.extend_from_slice(&act[r * ff..(r + 1) * ff]);
                }
                tr.rows.push(z);
            }
            tr.focus_resid.push(x[focus * d..(focus + 1) * d].to_vec());
            if let Some(hs) = tr.hidden_all.as_mut() {
                hs.push(x.clone());
            }
        }
        if o.keep_final {
            tr.x_final = x.clone();
        }
        if o.checkpoint {
            tr.checkpoint = Some(Checkpoint { x: x_ck, kv: kv_store.clone() });
        }
        if let (Some(r), Some(n)) = (ns.as_mut(), &self.final_norm) {
            r.fin = (0..t).map(|i| n.scale(&x[i * d..(i + 1) * d])).collect();
        }
        tr.norms = ns;
        if self.has_head {
            tr.logits = match (fz, &self.final_norm) {
                (Some(f), Some(_)) => self.final_logits_scaled(&x[focus * d..(focus + 1) * d], f.fin[posi(focus)]),
                _ => self.final_logits(&x[focus * d..(focus + 1) * d], 1),
            };
            if o.dump {
                tr.logits_all = Some(self.final_logits(&x, t));
            }
        }
        tr
    }

    /// Greedy continuation (each step a full pass)
    pub fn generate(&self, ids: &[u32], n: usize) -> Vec<u32> {
        let mut v = ids.to_vec();
        let mut out = Vec::new();
        for _ in 0..n {
            let tr = self.run(&v, v.len() - 1, &RunOpts::default());
            let next = top_probs(&tr.logits, 1)[0].0;
            if self.tok.is_special(next) {
                break;
            }
            v.push(next);
            out.push(next);
        }
        out
    }
}

// ====================================================================== weight sources

/// Uniform access to tensors from GGUF or safetensors (incl. MLX quantization)
trait Src {
    fn has(&self, n: &str) -> bool;
    fn mat(&self, n: &str) -> Result<Mat, String>;
    fn vec(&self, n: &str) -> Result<Vec<f32>, String> {
        Ok(self.mat(n)?.to_vec())
    }
    fn opt_vec(&self, n: &str) -> Result<Option<Vec<f32>>, String> {
        if self.has(n) { self.vec(n).map(Some) } else { Ok(None) }
    }
    fn opt_mat(&self, n: &str) -> Result<Option<Mat>, String> {
        if self.has(n) { self.mat(n).map(Some) } else { Ok(None) }
    }
}

impl Src for Gguf {
    fn has(&self, n: &str) -> bool {
        self.tensors.contains_key(n)
    }
    fn mat(&self, name: &str) -> Result<Mat, String> {
        let t = self.tensors.get(name).ok_or(format!("tensor {name} missing"))?;
        let dt = DType::from_ggml(t.typ).ok_or(format!("{name}: type {} cannot be decoded", crate::quant::ggml_name(t.typ)))?;
        let cols = t.ne[0];
        let rows: usize = t.ne[1..].iter().product::<usize>().max(1);
        Mat::new(rows, cols, dt, &self.data[t.off..], name)
    }
}

pub struct St {
    t: HashMap<String, (String, Vec<usize>, &'static [u8])>,
    q_bits: Option<(usize, usize)>, // (bits, group) from the MLX configuration
}

impl St {
    pub fn open(files: &[PathBuf], cfg: &Value) -> Result<Self, String> {
        let mut t = HashMap::new();
        for f in files {
            let data = gguf::map_file(f)?;
            let n = u64::from_le_bytes(data[..8].try_into().unwrap()) as usize;
            let hdr: Value = serde_json::from_slice(&data[8..8 + n]).map_err(|e| e.to_string())?;
            for (name, inf) in hdr.as_object().ok_or("safetensors: broken header")? {
                if name == "__metadata__" {
                    continue;
                }
                let dt = inf["dtype"].as_str().unwrap_or("").to_string();
                let shape: Vec<usize> = inf["shape"].as_array().map(|a| a.iter().filter_map(|v| v.as_u64().map(|x| x as usize)).collect()).unwrap_or_default();
                let off = inf["data_offsets"].as_array().ok_or("safetensors: offsets missing")?;
                let (b, e) = (off[0].as_u64().unwrap() as usize, off[1].as_u64().unwrap() as usize);
                t.insert(name.clone(), (dt, shape, &data[8 + n + b..8 + n + e]));
            }
        }
        let q = if cfg["quantization"].is_object() { &cfg["quantization"] } else { &cfg["quantization_config"] };
        let q_bits = q["bits"].as_u64().map(|b| (b as usize, q["group_size"].as_u64().unwrap_or(64) as usize));
        Ok(St { t, q_bits })
    }
    fn keys(&self) -> impl Iterator<Item = &String> {
        self.t.keys()
    }
}

impl Src for St {
    fn has(&self, n: &str) -> bool {
        self.t.contains_key(n)
    }
    fn mat(&self, n: &str) -> Result<Mat, String> {
        let (dt, shape, data) = self.t.get(n).ok_or(format!("tensor {n} missing"))?;
        if dt == "U32" {
            // MLX: X.weight (packed) + X.scales (+ X.biases)
            let base = n.strip_suffix(".weight").unwrap_or(n);
            let (sdt, sshape, sdata) = self.t.get(&format!("{base}.scales")).ok_or(format!("{n}: U32 without .scales (not MLX format)"))?;
            let sd = DType::from_st(sdt).ok_or(format!("{base}.scales: dtype {sdt}"))?;
            let bias = match self.t.get(&format!("{base}.biases")) {
                Some((bdt, _, bdata)) => Some((DType::from_st(bdt).ok_or(format!("{base}.biases: dtype {bdt}"))?, *bdata)),
                None => None,
            };
            let wpr = *shape.last().unwrap_or(&1);
            let rows = shape[..shape.len() - 1].iter().product::<usize>().max(1);
            let ng = *sshape.last().unwrap_or(&1);
            let gs = match self.q_bits {
                Some((bits, gs)) if wpr * 32 == ng * gs * bits => gs,
                _ => [32, 64, 128]
                    .into_iter()
                    .find(|g| (wpr * 32) % (ng * g) == 0 && matches!(wpr * 32 / (ng * g), 2 | 3 | 4 | 5 | 6 | 8))
                    .ok_or(format!("{n}: MLX group size unclear"))?,
            };
            return Mat::mlx(rows, wpr, data, (sd, *sdata), bias, ng, gs, n);
        }
        let d = DType::from_st(dt).ok_or(format!("{n}: dtype {dt} is not supported"))?;
        let cols = *shape.last().unwrap_or(&1);
        let rows = shape[..shape.len().saturating_sub(1)].iter().product::<usize>().max(1);
        Mat::new(rows, cols, d, data, n)
    }
}

fn type_summary(mats: &[&Mat]) -> String {
    let mut m: HashMap<String, usize> = HashMap::new();
    let mut total = 0;
    for x in mats {
        *m.entry(x.type_name()).or_default() += x.rows * x.cols;
        total += x.rows * x.cols;
    }
    let mut v: Vec<_> = m.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1));
    v.iter().map(|(k, n)| format!("{k} {:.0}%", *n as f64 * 100.0 / total.max(1) as f64)).collect::<Vec<_>>().join(", ")
}

impl Mlp {
    /// Length of every output column of W_down (computed once, then cached)
    pub fn col_norms(&self) -> &[f32] {
        self.down_norms.get_or_init(|| self.down.col_norms())
    }
}

fn mlp(gate: Option<Mat>, up: Mat, bup: Option<Vec<f32>>, down: Mat, bdown: Option<Vec<f32>>, act: Act) -> Mlp {
    Mlp { gate, up, bup, down, bdown, act, down_norms: OnceLock::new() }
}

fn attn_basic(n_head: usize, n_kv: usize, hd: usize, wo: Mat) -> Attn {
    Attn {
        n_head,
        n_kv,
        hd,
        wq: None,
        wk: None,
        wv: None,
        wqkv: None,
        bq: None,
        bk: None,
        bv: None,
        bqkv: None,
        qn: None,
        kn: None,
        vn: None,
        rope: None,
        scale: 1.0 / (hd as f32).sqrt(),
        softcap: None,
        window: None,
        causal: true,
        kv_from: None,
        v_is_k: false,
        wo,
        bo: None,
    }
}

fn finish(mut m: Model) -> Model {
    let mut mats: Vec<&Mat> = vec![&m.embed];
    for b in &m.blocks {
        for x in [&b.attn.wq, &b.attn.wk, &b.attn.wv, &b.attn.wqkv, &b.mlp.gate].into_iter().flatten() {
            mats.push(x);
        }
        mats.extend([&b.attn.wo, &b.mlp.up, &b.mlp.down]);
    }
    if let Some(h) = &m.lm_head {
        mats.push(h);
    }
    m.weight_types = type_summary(&mats);
    m
}

/// Last layer before `first` with the same type (source of shared keys/values)
fn kv_source(types: &[String], l: usize, first: usize) -> Option<usize> {
    (0..first).rev().find(|&k| types[k] == types[l])
}

// ====================================================================== GGUF

fn gu(g: &Gguf, k: &str) -> Option<usize> {
    match g.meta.get(k)? {
        Val::A(a) => a.first()?.u().map(|v| v as usize),
        v => v.u().map(|v| v as usize),
    }
}
fn gf(g: &Gguf, k: &str) -> Option<f32> {
    g.meta.get(k)?.f().map(|v| v as f32)
}

pub fn load_gguf(path: &std::path::Path, name: &str) -> Result<Model, String> {
    let g = gguf::open(path)?;
    let arch = g.meta.get("general.architecture").and_then(|v| v.s()).ok_or("GGUF without architecture")?.to_string();
    let k = |s: &str| format!("{arch}.{s}");
    let need = |s: &str| gu(&g, &k(s)).ok_or(format!("GGUF: {} missing", k(s)));
    if !matches!(arch.as_str(), "llama" | "mistral" | "qwen2" | "qwen3" | "gemma4") {
        return Err(format!("GGUF architecture '{arch}' is not supported yet (available: llama, mistral, qwen2, qwen3, gemma4)"));
    }
    let nl = need("block_count")?;
    let d = need("embedding_length")?;
    let n_head = need("attention.head_count")?;
    let eps = gf(&g, &k("attention.layer_norm_rms_epsilon")).unwrap_or(1e-5);
    let base = gf(&g, &k("rope.freq_base")).unwrap_or(10000.0);
    let tok = Tokenizer::from_gguf(&g.meta)?;
    let embed = g.mat("token_embd.weight")?;
    let lm_head = g.opt_mat("output.weight")?;
    let final_norm = Some(Norm::rms(g.vec("output_norm.weight")?, eps));
    let mut blocks: Vec<Block> = Vec::with_capacity(nl);
    let p = |i: usize, s: &str| format!("blk.{i}.{s}");
    let mut m = Model {
        name: name.into(), format: "GGUF".into(), arch: arch.clone(), weight_types: String::new(), d, embed, embed_scale: 1.0,
        pos_embed: None, type_embed: None, embed_norm: None, ple: None, blocks: Vec::new(), final_norm, lm_head, has_head: true, logit_cap: None, tok,
    };
    if arch == "gemma4" {
        // layer types are stored in the file; head size per layer from the shape of attn_q
        let swa: Vec<bool> = g.meta.get(&k("attention.sliding_window_pattern")).and_then(|v| v.a()).map(|a| a.iter().map(|v| v.b().unwrap_or(false)).collect()).ok_or("gemma4: sliding_window_pattern missing")?;
        let types: Vec<String> = swa.iter().map(|s| if *s { "local".into() } else { "global".into() }).collect();
        let window = need("attention.sliding_window")?;
        let n_kv = gu(&g, &k("attention.head_count_kv")).unwrap_or(n_head);
        let base_swa = gf(&g, &k("rope.freq_base_swa")).unwrap_or(10000.0);
        let shared = gu(&g, &k("attention.shared_kv_layers")).unwrap_or(0);
        let first_shared = nl - shared;
        let rf = g.opt_vec("rope_freqs.weight")?;
        let act = Act::GeluTanh;
        for i in 0..nl {
            let wq = g.mat(&p(i, "attn_q.weight"))?;
            let hd = wq.rows / n_head;
            let mut a = attn_basic(n_head, n_kv, hd, g.mat(&p(i, "attn_output.weight"))?);
            a.scale = 1.0;
            a.wq = Some(wq);
            a.qn = Some(Norm::rms(g.vec(&p(i, "attn_q_norm.weight"))?, eps));
            if shared > 0 && i >= first_shared {
                a.kv_from = kv_source(&types, i, first_shared);
                a.n_kv = blocks[a.kv_from.ok_or("shared K/V: no source layer")?].attn.n_kv;
            } else {
                let wk = g.mat(&p(i, "attn_k.weight"))?;
                a.n_kv = wk.rows / hd;
                a.wk = Some(wk);
                a.wv = g.opt_mat(&p(i, "attn_v.weight"))?;
                a.v_is_k = a.wv.is_none();
                a.kn = Some(Norm::rms(g.vec(&p(i, "attn_k_norm.weight"))?, eps));
                a.vn = Some(Norm::rms(vec![1.0; hd], eps));
            }
            let (b0, div) = if swa[i] { (base_swa, None) } else { (base, rf.as_deref()) };
            if let Some(dv) = div {
                if dv.len() != hd / 2 {
                    return Err(format!("gemma4: rope_freqs ({}) does not match head size {hd}", dv.len()));
                }
            }
            a.rope = Some(Rope { inv_freq: inv_freq(b0, hd, hd, div), neox: true, pos_scale: 1.0 });
            a.window = if swa[i] { Some(window) } else { None };
            blocks.push(Block {
                res: Res::Pre,
                n_attn: Some(Norm::rms(g.vec(&p(i, "attn_norm.weight"))?, eps)),
                n_attn_post: Some(Norm::rms(g.vec(&p(i, "post_attention_norm.weight"))?, eps)),
                n_mlp: Some(Norm::rms(g.vec(&p(i, "ffn_norm.weight"))?, eps)),
                n_mlp_post: Some(Norm::rms(g.vec(&p(i, "post_ffw_norm.weight"))?, eps)),
                attn: a,
                mlp: mlp(Some(g.mat(&p(i, "ffn_gate.weight"))?), g.mat(&p(i, "ffn_up.weight"))?, None, g.mat(&p(i, "ffn_down.weight"))?, None, act),
                ple: match g.opt_mat(&p(i, "inp_gate.weight"))? {
                    Some(gate) => Some(Ple { gate, proj: g.mat(&p(i, "proj.weight"))?, norm: Norm::rms(g.vec(&p(i, "post_norm.weight"))?, eps), act }),
                    None => None,
                },
                scalar: g.opt_vec(&p(i, "layer_output_scale.weight"))?.map(|v| v[0]),
                label: if swa[i] { format!("local {window}") } else { "global".into() },
            });
        }
        if let Some(pe) = g.opt_mat("per_layer_token_embd.weight")? {
            let dp = need("embedding_length_per_layer_input")?;
            m.ple = Some(PleModel {
                embed: pe,
                embed_scale: (dp as f32).sqrt(),
                proj: g.mat("per_layer_model_proj.weight")?,
                proj_scale: 1.0 / (d as f32).sqrt(),
                norm: Norm::rms(g.vec("per_layer_proj_norm.weight")?, eps),
                dp,
            });
        }
        m.embed_scale = (d as f32).sqrt();
        m.logit_cap = gf(&g, &k("final_logit_softcapping"));
    } else {
        let n_kv = gu(&g, &k("attention.head_count_kv")).unwrap_or(n_head);
        let hd = gu(&g, &k("attention.key_length")).unwrap_or(d / n_head);
        let rope_dims = gu(&g, &k("rope.dimension_count")).unwrap_or(hd);
        let pos_scale = match g.meta.get(&k("rope.scaling.type")).and_then(|v| v.s()).unwrap_or("none") {
            "none" => 1.0,
            "linear" => gf(&g, &k("rope.scaling.factor")).unwrap_or(1.0),
            o => return Err(format!("RoPE scaling '{o}' is not supported")),
        };
        let div = g.opt_vec("rope_freqs.weight")?;
        for i in 0..nl {
            let mut a = attn_basic(n_head, n_kv, hd, g.mat(&p(i, "attn_output.weight"))?);
            a.wq = Some(g.mat(&p(i, "attn_q.weight"))?);
            a.wk = Some(g.mat(&p(i, "attn_k.weight"))?);
            a.wv = Some(g.mat(&p(i, "attn_v.weight"))?);
            a.bq = g.opt_vec(&p(i, "attn_q.bias"))?;
            a.bk = g.opt_vec(&p(i, "attn_k.bias"))?;
            a.bv = g.opt_vec(&p(i, "attn_v.bias"))?;
            a.bo = g.opt_vec(&p(i, "attn_output.bias"))?;
            a.qn = g.opt_vec(&p(i, "attn_q_norm.weight"))?.map(|w| Norm::rms(w, eps));
            a.kn = g.opt_vec(&p(i, "attn_k_norm.weight"))?.map(|w| Norm::rms(w, eps));
            // llama.cpp: Llama weights are permuted for pairwise RoPE, Qwen uses the NeoX style
            a.rope = Some(Rope { inv_freq: inv_freq(base, rope_dims, rope_dims, div.as_deref()), neox: !matches!(arch.as_str(), "llama" | "mistral"), pos_scale });
            blocks.push(Block {
                res: Res::Pre,
                n_attn: Some(Norm::rms(g.vec(&p(i, "attn_norm.weight"))?, eps)),
                n_attn_post: None,
                n_mlp: Some(Norm::rms(g.vec(&p(i, "ffn_norm.weight"))?, eps)),
                n_mlp_post: None,
                attn: a,
                mlp: mlp(Some(g.mat(&p(i, "ffn_gate.weight"))?), g.mat(&p(i, "ffn_up.weight"))?, None, g.mat(&p(i, "ffn_down.weight"))?, None, Act::Silu),
                ple: None,
                scalar: None,
                label: String::new(),
            });
        }
    }
    m.blocks = blocks;
    Ok(finish(m))
}

// ====================================================================== Hugging Face / MLX

pub struct HfFiles {
    pub config: PathBuf,
    pub weights: Vec<PathBuf>,
    pub tokenizer_json: Option<PathBuf>,
    pub tokenizer_config: Option<PathBuf>,
    pub vocab_json: Option<PathBuf>,
    pub merges_txt: Option<PathBuf>,
    pub special_tokens: Option<PathBuf>,
}

fn read_json(p: &std::path::Path) -> Result<Value, String> {
    let s = std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?;
    serde_json::from_str(&s).map_err(|e| format!("{}: {e}", p.display()))
}

fn hf_tokenizer(f: &HfFiles, mt: &str) -> Result<Tokenizer, String> {
    let tc = f.tokenizer_config.as_ref().and_then(|p| read_json(p).ok());
    if let Some(p) = &f.tokenizer_json {
        return Tokenizer::from_json(&read_json(p)?, tc.as_ref());
    }
    if let (Some(v), Some(m)) = (&f.vocab_json, &f.merges_txt) {
        let mut sp: Vec<String> = Vec::new();
        if let Some(s) = f.special_tokens.as_ref().and_then(|p| read_json(p).ok()) {
            for (_, v) in s.as_object().into_iter().flatten() {
                if let Some(x) = v.as_str().or(v["content"].as_str()) {
                    sp.push(x.to_string());
                }
            }
        }
        let (pre, suf): (Vec<String>, Vec<String>) = if matches!(mt, "roberta" | "xlm-roberta" | "camembert") {
            (vec!["<s>".into()], vec!["</s>".into()])
        } else {
            (vec![], vec![])
        };
        let merges = std::fs::read_to_string(m).map_err(|e| e.to_string())?;
        return Tokenizer::from_vocab_merges(&read_json(v)?, &merges, &sp, &pre, &suf);
    }
    Err("no tokenizer found (tokenizer.json or vocab.json + merges.txt)".into())
}

pub fn load_hf(f: &HfFiles, name: &str) -> Result<Model, String> {
    let root = read_json(&f.config)?;
    let mt_root = root["model_type"].as_str().unwrap_or("?").to_string();
    // multimodal wrappers: the language part lives in text_config
    let c = if root["text_config"].is_object() { root["text_config"].clone() } else { root.clone() };
    let mt = c["model_type"].as_str().unwrap_or(&mt_root).to_string();
    let fam = mt.trim_end_matches("_text").to_string();
    let mut tok = hf_tokenizer(f, &fam)?;
    let st = St::open(&f.weights, &root)?;
    // find the weight prefix from a known name (model., language_model.model., ...)
    let find_prefix = |suffix: &str| st.keys().filter(|k| k.ends_with(suffix)).min_by_key(|k| k.len()).map(|k| k[..k.len() - suffix.len()].to_string());
    let gi = |k: &str| c[k].as_u64().map(|v| v as usize);
    let gfl = |k: &str| c[k].as_f64().map(|v| v as f32);
    let d = gi("hidden_size").or(gi("n_embd")).ok_or("config: hidden_size missing")?;
    let new_model = |blocks: Vec<Block>, embed: Mat, lm: Option<Mat>, fnorm: Option<Norm>| Model {
        name: name.into(), format: "safetensors".into(), arch: mt.clone(), weight_types: String::new(), d, embed, embed_scale: 1.0,
        pos_embed: None, type_embed: None, embed_norm: None, ple: None, blocks, final_norm: fnorm, lm_head: lm, has_head: true, logit_cap: None,
        tok: Tokenizer::placeholder(),
    };
    let mut model = match fam.as_str() {
        "llama" | "mistral" | "qwen2" | "qwen3" | "gemma3" | "gemma4" => {
            let pre = find_prefix("embed_tokens.weight").ok_or("embedding (embed_tokens) not found")?;
            let nl = gi("num_hidden_layers").ok_or("config: num_hidden_layers missing")?;
            let is_gemma = fam.starts_with("gemma");
            let eps = gfl("rms_norm_eps").unwrap_or(1e-6);
            // Gemma 3 stores norm weights as (w-1); Gemma 4 and Llama store them directly
            let shift = if fam == "gemma3" { 1.0 } else { 0.0 };
            let rn = |n: &str| -> Result<Norm, String> { Ok(Norm::rms(st.vec(n)?.into_iter().map(|v| v + shift).collect(), eps)) };
            let p = |i: usize, s: &str| format!("{pre}layers.{i}.{s}");
            let pattern = gi("sliding_window_pattern").unwrap_or(6);
            let types: Vec<String> = match c["layer_types"].as_array() {
                Some(a) => a.iter().map(|v| v.as_str().unwrap_or("full_attention").to_string()).collect(),
                None if is_gemma => (0..nl).map(|i| if (i + 1) % pattern != 0 { "sliding_attention".into() } else { "full_attention".into() }).collect(),
                None => vec!["full_attention".into(); nl],
            };
            let rp_root = &c["rope_parameters"];
            let rs = if c["rope_scaling"].is_object() { c["rope_scaling"].clone() } else { Value::Null };
            let shared = gi("num_kv_shared_layers").unwrap_or(0);
            let first_shared = nl - shared;
            let act = parse_act(c["hidden_activation"].as_str().or(c["hidden_act"].as_str()).unwrap_or(if is_gemma { "gelu_pytorch_tanh" } else { "silu" }))?;
            let mut blocks: Vec<Block> = Vec::new();
            for i in 0..nl {
                let ty = types.get(i).map(|s| s.as_str()).unwrap_or("full_attention");
                let wq = st.mat(&p(i, "self_attn.q_proj.weight"))?;
                let qn = if st.has(&p(i, "self_attn.q_norm.weight")) { Some(rn(&p(i, "self_attn.q_norm.weight"))?) } else { None };
                // head size: from the q norm (Gemma/Qwen3), else from the configuration
                let hd = qn
                    .as_ref()
                    .map(|n| n.w.len())
                    .or(if fam == "gemma4" && ty == "full_attention" { gi("global_head_dim") } else { gi("head_dim") })
                    .or(gi("num_attention_heads").map(|h| d / h))
                    .ok_or("head size unclear")?;
                let n_head = wq.rows / hd;
                let mut a = attn_basic(n_head, n_head, hd, st.mat(&p(i, "self_attn.o_proj.weight"))?);
                a.wq = Some(wq);
                a.bq = st.opt_vec(&p(i, "self_attn.q_proj.bias"))?;
                a.bo = st.opt_vec(&p(i, "self_attn.o_proj.bias"))?;
                a.qn = qn;
                if fam == "gemma4" && shared > 0 && i >= first_shared {
                    a.kv_from = kv_source(&types, i, first_shared);
                    a.n_kv = blocks[a.kv_from.ok_or("shared K/V: no source layer")?].attn.n_kv;
                } else {
                    let wk = st.mat(&p(i, "self_attn.k_proj.weight"))?;
                    a.n_kv = wk.rows / hd;
                    a.wk = Some(wk);
                    a.wv = st.opt_mat(&p(i, "self_attn.v_proj.weight"))?;
                    a.v_is_k = a.wv.is_none();
                    a.bk = st.opt_vec(&p(i, "self_attn.k_proj.bias"))?;
                    a.bv = st.opt_vec(&p(i, "self_attn.v_proj.bias"))?;
                    if st.has(&p(i, "self_attn.k_norm.weight")) {
                        a.kn = Some(rn(&p(i, "self_attn.k_norm.weight"))?);
                    }
                    if fam == "gemma4" {
                        a.vn = Some(Norm::rms(vec![1.0; hd], eps));
                    }
                }
                // RoPE per layer type (transformers 5: rope_parameters[type]; older: rope_theta/rope_scaling)
                let rp = if rp_root[ty].is_object() {
                    rp_root[ty].clone()
                } else if rp_root["rope_type"].is_string() || rp_root["rope_theta"].is_number() {
                    rp_root.clone()
                } else {
                    Value::Null
                };
                let theta = rp["rope_theta"]
                    .as_f64()
                    .map(|v| v as f32)
                    .or_else(|| if ty == "sliding_attention" && is_gemma { gfl("rope_local_base_freq").or(Some(10000.0)) } else { gfl("rope_theta") })
                    .unwrap_or(if is_gemma { 1_000_000.0 } else { 10000.0 });
                // in Gemma, the old-style rope_scaling only applies to global layers
                let scaling = if rp["rope_type"].is_string() { rp.clone() } else if !is_gemma || ty == "full_attention" { rs.clone() } else { Value::Null };
                let rtype = scaling["rope_type"].as_str().or(scaling["type"].as_str()).unwrap_or("default").to_string();
                let (invf, pos_scale) = match rtype.as_str() {
                    "default" => (inv_freq(theta, hd, hd, None), 1.0),
                    "linear" => (inv_freq(theta, hd, hd, None), scaling["factor"].as_f64().unwrap_or(1.0) as f32),
                    "llama3" => (llama3_inv_freq(theta, hd, &scaling), 1.0),
                    "proportional" => {
                        let prop = scaling["partial_rotary_factor"].as_f64().unwrap_or(1.0);
                        let rot = (prop * hd as f64 / 2.0) as usize;
                        let mut v = inv_freq(theta, hd, hd, None);
                        for x in v.iter_mut().skip(rot) {
                            *x = 0.0;
                        }
                        (v, 1.0)
                    }
                    o => return Err(format!("RoPE type '{o}' is not supported")),
                };
                a.rope = Some(Rope { inv_freq: invf, neox: true, pos_scale });
                if is_gemma {
                    a.scale = if fam == "gemma4" { 1.0 } else { 1.0 / gfl("query_pre_attn_scalar").unwrap_or(hd as f32).sqrt() };
                    // in transformers only Gemma 2 uses attn_logit_softcapping (Gemma 3/4 ignore the value)
                    if fam == "gemma2" {
                        a.softcap = gfl("attn_logit_softcapping");
                    }
                    if ty == "sliding_attention" {
                        a.window = gi("sliding_window");
                    }
                }
                let n_attn_post = if is_gemma { Some(rn(&p(i, "post_attention_layernorm.weight"))?) } else { None };
                let (n_mlp, n_mlp_post) = if is_gemma {
                    (Some(rn(&p(i, "pre_feedforward_layernorm.weight"))?), Some(rn(&p(i, "post_feedforward_layernorm.weight"))?))
                } else {
                    (Some(rn(&p(i, "post_attention_layernorm.weight"))?), None)
                };
                let ple = if st.has(&p(i, "per_layer_input_gate.weight")) {
                    Some(Ple {
                        gate: st.mat(&p(i, "per_layer_input_gate.weight"))?,
                        proj: st.mat(&p(i, "per_layer_projection.weight"))?,
                        norm: rn(&p(i, "post_per_layer_input_norm.weight"))?,
                        act,
                    })
                } else {
                    None
                };
                blocks.push(Block {
                    res: Res::Pre,
                    n_attn: Some(rn(&p(i, "input_layernorm.weight"))?),
                    n_attn_post,
                    n_mlp,
                    n_mlp_post,
                    attn: a,
                    mlp: mlp(Some(st.mat(&p(i, "mlp.gate_proj.weight"))?), st.mat(&p(i, "mlp.up_proj.weight"))?, None, st.mat(&p(i, "mlp.down_proj.weight"))?, None, act),
                    ple,
                    scalar: st.opt_vec(&p(i, "layer_scalar"))?.map(|v| v[0]),
                    label: match (is_gemma, ty) {
                        (true, "sliding_attention") => format!("local {}", gi("sliding_window").unwrap_or(0)),
                        (true, _) => "global".into(),
                        _ => String::new(),
                    },
                });
            }
            let embed = st.mat(&format!("{pre}embed_tokens.weight"))?;
            let lm_name = find_prefix("lm_head.weight").map(|p| format!("{p}lm_head.weight"));
            let tie = c["tie_word_embeddings"].as_bool().or(root["tie_word_embeddings"].as_bool()).unwrap_or(is_gemma || lm_name.is_none());
            let lm = match lm_name {
                Some(n) if !tie => Some(st.mat(&n)?),
                _ => None,
            };
            let mut m = new_model(blocks, embed, lm, Some(rn(&format!("{pre}norm.weight"))?));
            if is_gemma {
                // Gemma needs <bos> at the start; llama.cpp/Ollama always add it (GGUF add_bos_token),
                // the HF tokenizer of Gemma 4 only through the chat template
                tok.ensure_prefix("<bos>");
                m.embed_scale = (d as f32).sqrt();
                m.logit_cap = gfl("final_logit_softcapping");
            }
            if st.has(&format!("{pre}embed_tokens_per_layer.weight")) {
                let dp = gi("hidden_size_per_layer_input").ok_or("hidden_size_per_layer_input missing")?;
                m.ple = Some(PleModel {
                    embed: st.mat(&format!("{pre}embed_tokens_per_layer.weight"))?,
                    embed_scale: (dp as f32).sqrt(),
                    proj: st.mat(&format!("{pre}per_layer_model_projection.weight"))?,
                    proj_scale: 1.0 / (d as f32).sqrt(),
                    norm: rn(&format!("{pre}per_layer_projection_norm.weight"))?,
                    dp,
                });
            }
            m
        }
        "gpt_neox" => {
            let nl = gi("num_hidden_layers").ok_or("num_hidden_layers missing")?;
            let n_head = gi("num_attention_heads").ok_or("num_attention_heads missing")?;
            let hd = d / n_head;
            let pct = c["rotary_pct"].as_f64().or_else(|| c["partial_rotary_factor"].as_f64()).unwrap_or(1.0);
            let rot = (hd as f64 * pct) as usize;
            let base = c["rotary_emb_base"].as_f64().or_else(|| c["rope_theta"].as_f64()).or_else(|| c["rope_parameters"]["rope_theta"].as_f64()).unwrap_or(10000.0) as f32;
            let eps = gfl("layer_norm_eps").unwrap_or(1e-5);
            let act = parse_act(c["hidden_act"].as_str().unwrap_or("gelu"))?;
            let ln = |p: &str| -> Result<Norm, String> { Ok(Norm { w: st.vec(&format!("{p}.weight"))?, b: st.opt_vec(&format!("{p}.bias"))?, eps, rms: false }) };
            let parallel = c["use_parallel_residual"].as_bool().unwrap_or(true);
            let mut blocks = Vec::new();
            for i in 0..nl {
                let p = |s: &str| format!("gpt_neox.layers.{i}.{s}");
                let mut a = attn_basic(n_head, n_head, hd, st.mat(&p("attention.dense.weight"))?);
                a.wqkv = Some(st.mat(&p("attention.query_key_value.weight"))?);
                a.bqkv = st.opt_vec(&p("attention.query_key_value.bias"))?;
                a.bo = st.opt_vec(&p("attention.dense.bias"))?;
                a.rope = Some(Rope { inv_freq: inv_freq(base, rot, rot, None), neox: true, pos_scale: 1.0 });
                blocks.push(Block {
                    res: if parallel { Res::Parallel } else { Res::Pre },
                    n_attn: Some(ln(&p("input_layernorm"))?),
                    n_attn_post: None,
                    n_mlp: Some(ln(&p("post_attention_layernorm"))?),
                    n_mlp_post: None,
                    attn: a,
                    mlp: mlp(None, st.mat(&p("mlp.dense_h_to_4h.weight"))?, st.opt_vec(&p("mlp.dense_h_to_4h.bias"))?, st.mat(&p("mlp.dense_4h_to_h.weight"))?, st.opt_vec(&p("mlp.dense_4h_to_h.bias"))?, act),
                    ple: None,
                    scalar: None,
                    label: String::new(),
                });
            }
            new_model(blocks, st.mat("gpt_neox.embed_in.weight")?, Some(st.mat("embed_out.weight")?), Some(ln("gpt_neox.final_layer_norm")?))
        }
        "gpt_neo" => {
            let nl = gi("num_layers").or(gi("num_hidden_layers")).ok_or("num_layers missing")?;
            let n_head = gi("num_heads").or(gi("num_attention_heads")).ok_or("num_heads missing")?;
            let hd = d / n_head;
            let eps = gfl("layer_norm_epsilon").unwrap_or(1e-5);
            let act = parse_act(c["activation_function"].as_str().unwrap_or("gelu_new"))?;
            let window = gi("window_size").unwrap_or(256);
            // attention_types: [[["global","local"], n], ...] -> type per layer
            let mut types = Vec::new();
            for e in c["attention_types"].as_array().into_iter().flatten() {
                for _ in 0..e[1].as_u64().unwrap_or(1) {
                    for t in e[0].as_array().into_iter().flatten() {
                        types.push(t.as_str().unwrap_or("global").to_string());
                    }
                }
            }
            let ln = |p: &str| -> Result<Norm, String> { Ok(Norm { w: st.vec(&format!("{p}.weight"))?, b: st.opt_vec(&format!("{p}.bias"))?, eps, rms: false }) };
            let mut blocks = Vec::new();
            for i in 0..nl {
                let p = |s: &str| format!("transformer.h.{i}.{s}");
                let mut a = attn_basic(n_head, n_head, hd, st.mat(&p("attn.attention.out_proj.weight"))?);
                a.wq = Some(st.mat(&p("attn.attention.q_proj.weight"))?);
                a.wk = Some(st.mat(&p("attn.attention.k_proj.weight"))?);
                a.wv = Some(st.mat(&p("attn.attention.v_proj.weight"))?);
                a.bo = st.opt_vec(&p("attn.attention.out_proj.bias"))?;
                a.scale = 1.0; // GPT-Neo does not scale the scores
                let local = types.get(i).is_some_and(|t| t == "local");
                a.window = if local { Some(window) } else { None };
                blocks.push(Block {
                    res: Res::Pre,
                    n_attn: Some(ln(&p("ln_1"))?),
                    n_attn_post: None,
                    n_mlp: Some(ln(&p("ln_2"))?),
                    n_mlp_post: None,
                    attn: a,
                    mlp: mlp(None, st.mat(&p("mlp.c_fc.weight"))?, st.opt_vec(&p("mlp.c_fc.bias"))?, st.mat(&p("mlp.c_proj.weight"))?, st.opt_vec(&p("mlp.c_proj.bias"))?, act),
                    ple: None,
                    scalar: None,
                    label: if local { format!("local {window}") } else { "global".into() },
                });
            }
            let lm = if c["tie_word_embeddings"].as_bool() == Some(false) { Some(st.mat("lm_head.weight")?) } else { None };
            let mut m = new_model(blocks, st.mat("transformer.wte.weight")?, lm, Some(ln("transformer.ln_f")?));
            m.pos_embed = Some((st.mat("transformer.wpe.weight")?, 0));
            m
        }
        "roberta" | "bert" | "xlm-roberta" | "camembert" => {
            let pre = find_prefix("embeddings.word_embeddings.weight").ok_or("BERT embedding not found")?;
            let nl = gi("num_hidden_layers").ok_or("num_hidden_layers missing")?;
            let n_head = gi("num_attention_heads").ok_or("num_attention_heads missing")?;
            let hd = d / n_head;
            let eps = gfl("layer_norm_eps").unwrap_or(1e-12);
            let act = parse_act(c["hidden_act"].as_str().unwrap_or("gelu"))?;
            let ln = |p: &str| -> Result<Norm, String> { Ok(Norm { w: st.vec(&format!("{pre}{p}.weight"))?, b: st.opt_vec(&format!("{pre}{p}.bias"))?, eps, rms: false }) };
            let mut blocks = Vec::new();
            for i in 0..nl {
                let p = |s: &str| format!("{pre}encoder.layer.{i}.{s}");
                let mut a = attn_basic(n_head, n_head, hd, st.mat(&p("attention.output.dense.weight"))?);
                a.wq = Some(st.mat(&p("attention.self.query.weight"))?);
                a.wk = Some(st.mat(&p("attention.self.key.weight"))?);
                a.wv = Some(st.mat(&p("attention.self.value.weight"))?);
                a.bq = st.opt_vec(&p("attention.self.query.bias"))?;
                a.bk = st.opt_vec(&p("attention.self.key.bias"))?;
                a.bv = st.opt_vec(&p("attention.self.value.bias"))?;
                a.bo = st.opt_vec(&p("attention.output.dense.bias"))?;
                a.causal = false;
                let pl = format!("encoder.layer.{i}");
                blocks.push(Block {
                    res: Res::Post,
                    n_attn: None,
                    n_attn_post: Some(ln(&format!("{pl}.attention.output.LayerNorm"))?),
                    n_mlp: None,
                    n_mlp_post: Some(ln(&format!("{pl}.output.LayerNorm"))?),
                    attn: a,
                    mlp: mlp(None, st.mat(&p("intermediate.dense.weight"))?, st.opt_vec(&p("intermediate.dense.bias"))?, st.mat(&p("output.dense.weight"))?, st.opt_vec(&p("output.dense.bias"))?, act),
                    ple: None,
                    scalar: None,
                    label: String::new(),
                });
            }
            let mut m = new_model(blocks, st.mat(&format!("{pre}embeddings.word_embeddings.weight"))?, None, None);
            let off = if fam == "bert" { 0 } else { gi("pad_token_id").unwrap_or(1) + 1 };
            m.pos_embed = Some((st.mat(&format!("{pre}embeddings.position_embeddings.weight"))?, off));
            m.type_embed = match st.opt_mat(&format!("{pre}embeddings.token_type_embeddings.weight"))? {
                Some(te) => {
                    let mut r = vec![0f32; d];
                    te.row(0, &mut r);
                    Some(r)
                }
                None => None,
            };
            m.embed_norm = Some(ln("embeddings.LayerNorm")?);
            m.has_head = false;
            m
        }
        other => {
            return Err(format!(
                "model type '{other}' is not supported yet (available: llama, mistral, qwen2, qwen3, gemma3, gemma4, gpt_neox, gpt_neo, bert/roberta)"
            ))
        }
    };
    model.tok = tok;
    Ok(finish(model))
}

/// Llama 3 scaling of the RoPE frequencies (as in transformers)
fn llama3_inv_freq(base: f32, dims: usize, rs: &Value) -> Vec<f32> {
    let factor = rs["factor"].as_f64().unwrap_or(8.0) as f32;
    let lo = rs["low_freq_factor"].as_f64().unwrap_or(1.0) as f32;
    let hi = rs["high_freq_factor"].as_f64().unwrap_or(4.0) as f32;
    let old = rs["original_max_position_embeddings"].as_f64().unwrap_or(8192.0) as f32;
    let (lo_wl, hi_wl) = (old / lo, old / hi);
    inv_freq(base, dims, dims, None)
        .into_iter()
        .map(|f| {
            let wl = 2.0 * std::f32::consts::PI / f;
            let mut nf = if wl > lo_wl { f / factor } else { f };
            if !(wl < hi_wl) && !(wl > lo_wl) {
                let sm = (old / wl - lo) / (hi - lo);
                nf = (1.0 - sm) * nf / factor + sm * nf;
            }
            nf
        })
        .collect()
}
