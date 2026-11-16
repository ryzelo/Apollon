//! Circuit search down to single scalar products (--circuit).
//!
//! Every matrix product of the forward pass is a sum of scalar products: W[o,c]·x_c in the projections
//! (Q, K, V, O, gate, up, down and the products of the per-layer embedding) and q_e·k_e and a_j·v_j[e]
//! inside attention (QKᵀ and A·V). One such product at one position is an **atom**. The search switches
//! atoms off – a switched-off atom is simply missing from its sum – and keeps only what the prediction
//! needs, coarse to fine:
//!
//!   level 1  components at every position: attention (query side), K/V projections, MLP, PLE
//!   level 2  heads, K/V groups, neurons, PLE dimensions
//!   level 3  inside heads and K/V groups: q dimensions, keys of QKᵀ and of A·V, input columns of W_O,
//!            k/v dimensions, rows of the PLE input projection
//!   level 4  single atoms
//!
//! A piece stays only if switching it off – on top of everything already off – would break the
//! criterion: KL(full model ‖ circuit) ≤ ε and the same top token. Every decision is an exact rerun of
//! the model from the earliest layer it touches: no gradients, no linearization, no replacement model.
//! Magnitudes (|W·x|, activations, attention weights) only decide the order in which things are tried.
//! Many candidates run in one pass (model::Variants); every accepted step is rerun on its own and has
//! to pass the criterion again before it is kept. Pieces that provably have no effect any more (their
//! output is switched off, or their input is exactly zero) are switched off without a test, and the
//! output has to stay bit-identical.
use crate::fmt::{chip, esc_html, esc_piece, num, pct, prompt_html, sci, thousands, tok, Style, CSS};
use crate::model::{Atoms, Checkpoint, Cut, HeadCut, KeyCut, LayerCut, LayerInputs, LayerMeans, MatCut, Means, Model, NormScales, PartialUnit, PosMeans, RunOpts, Site, Start, Variants, N_SITES};
use crate::probe::{read_out_fz, Out, NY};
use crate::readout::{columns, softmax};
use crate::tensor::Mat;
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::IsTerminal;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

pub struct Opts {
    /// tolerance: KL(full ‖ circuit) in nats
    pub kl: f32,
    /// finest level (1 = components … 4 = atoms)
    pub depth: usize,
    /// stop refining after this many tests (what is unresolved then stays in the circuit)
    pub max_tests: usize,
    /// rows of the report lists
    pub show: usize,
    pub html: PathBuf,
    pub dump: Option<PathBuf>,
    /// test mode: no search, a random cut with this seed (written with --dump-circuit)
    pub random: Option<u64>,
    /// every norm keeps the scale of the full model (default); false: norms recompute their scale
    pub frozen: bool,
    /// mean ablation (default): a switched-off atom is replaced by its mean over `refs`; false: by 0
    pub mean: bool,
    /// reference sequences for the means (the analyzed sequence itself if empty)
    pub refs: Vec<Vec<u32>>,
}

/// rows per variant pass (besides the clean prompt)
const BUDGET: usize = 768;
/// share of ε that levels 1 … 4 may use cumulatively (the final phase may use all of it)
const FRAC: [f32; 4] = [0.25, 0.45, 0.6, 0.85];
/// a batched test has to stay below MARGIN·limit, the standalone rerun below the limit itself
const MARGIN: f32 = 0.998;
/// groups up to this size are tested element by element, larger ones are halved
const SMALL: usize = 8;
/// the same when blocks are resolved down to single elements (there most elements of a block that
/// failed as a whole are needed, so testing them one by one costs less than halving)
const SMALL_FINAL: usize = 32;
/// from this many pieces on, a level first removes the weakest pieces of every group in one sweep
const SWEEP_MIN: usize = 64;
const LEVELS: [&str; 4] = ["components", "heads · K/V groups · neurons · PLE dimensions", "q/k/v dimensions · keys · W_O columns · PLE inputs", "single scalar products"];

// ====================================================================== geometry of the model

struct LGeo {
    nh: usize,
    hd: usize,
    nkv: usize,
    ff: usize,
    gated: bool,
    up_bias: bool,
    own_kv: bool,
    v_site: bool,
    window: Option<usize>,
    dp: Option<usize>,
}

struct Geo {
    d: usize,
    t: usize,
    focus: usize,
    layers: Vec<LGeo>,
    /// layers whose attention reads the K/V computed in layer l
    readers: Vec<Vec<usize>>,
}

impl Geo {
    fn new(m: &Model, t: usize, focus: usize) -> Self {
        let layers: Vec<LGeo> = m
            .blocks
            .iter()
            .map(|b| {
                let a = &b.attn;
                LGeo {
                    nh: a.n_head,
                    hd: a.hd,
                    nkv: if a.wqkv.is_some() { a.n_head } else { a.n_kv },
                    ff: b.mlp.up.rows,
                    gated: b.mlp.gate.is_some(),
                    up_bias: b.mlp.bup.is_some(),
                    own_kv: a.kv_from.is_none(),
                    v_site: a.kv_from.is_none() && !a.v_is_k,
                    window: a.window,
                    dp: b.ple.as_ref().map(|p| p.gate.rows),
                }
            })
            .collect();
        let readers = (0..layers.len())
            .map(|l| (0..layers.len()).filter(|&l2| (l2 == l && layers[l].own_kv) || m.blocks[l2].attn.kv_from == Some(l)).collect())
            .collect();
        Geo { d: m.d, t, focus, layers, readers }
    }
    fn has(&self, l: usize, s: Site) -> bool {
        let g = &self.layers[l];
        match s {
            Site::Q | Site::O | Site::Up | Site::Down => true,
            Site::K => g.own_kv,
            Site::V => g.v_site,
            Site::Gate => g.gated,
            Site::PleIn | Site::PleGate | Site::PleProj => g.dp.is_some(),
        }
    }
    fn units(&self, l: usize, s: Site) -> usize {
        let g = &self.layers[l];
        match s {
            Site::Q | Site::O => g.nh * g.hd,
            Site::K | Site::V => g.nkv * g.hd,
            Site::Gate | Site::Up | Site::Down => g.ff,
            Site::PleIn | Site::PleGate | Site::PleProj => g.dp.unwrap_or(0),
        }
    }
    /// keys visible to the query at position p
    fn keys(&self, l: usize, p: usize) -> std::ops::Range<usize> {
        let lo = self.layers[l].window.map_or(0, |w| (p + 1).saturating_sub(w));
        lo..p + 1
    }
    /// all atoms of one cell (layer l, position p)
    fn cell_atoms(&self, l: usize, p: usize) -> u64 {
        let g = &self.layers[l];
        let mut n: u64 = ALL_SITES.iter().filter(|&&s| self.has(l, s)).map(|&s| (self.units(l, s) * self.d) as u64).sum();
        n += (2 * g.nh * self.keys(l, p).len() * g.hd) as u64;
        n
    }
}

const ALL_SITES: [Site; N_SITES] = [Site::Q, Site::K, Site::V, Site::O, Site::Gate, Site::Up, Site::Down, Site::PleIn, Site::PleGate, Site::PleProj];

// ====================================================================== pieces

type Ids = Arc<Vec<u32>>;

/// A piece of the computation at position p (where it is computed). The sets hold the children chosen.
#[derive(Clone, Debug)]
enum Piece {
    Attn { l: usize, p: usize },
    Kv { l: usize, p: usize },
    Mlp { l: usize, p: usize },
    Ple { l: usize, p: usize },
    Heads { l: usize, p: usize, h: Ids },
    KvGroups { l: usize, p: usize, g: Ids },
    Neurons { l: usize, p: usize, n: Ids },
    PleDims { l: usize, p: usize, e: Ids },
    Units { l: usize, p: usize, s: Site, u: Ids },
    Keys { l: usize, p: usize, h: usize, av: bool, j: Ids },
    Atoms { l: usize, p: usize, s: Site, u: usize, a: Ids },
    KeyDims { l: usize, p: usize, h: usize, j: usize, av: bool, e: Ids },
}

/// One elementary change of a cell
#[derive(Clone)]
enum Op {
    All(Site),
    Unit(Site, u32),
    Atoms(Site, u32, Ids),
    Key(usize, bool, u32),
    KeyDims(usize, bool, u32, Ids),
}

impl Piece {
    fn lp(&self) -> (usize, usize) {
        match self {
            Piece::Attn { l, p }
            | Piece::Kv { l, p }
            | Piece::Mlp { l, p }
            | Piece::Ple { l, p }
            | Piece::Heads { l, p, .. }
            | Piece::KvGroups { l, p, .. }
            | Piece::Neurons { l, p, .. }
            | Piece::PleDims { l, p, .. }
            | Piece::Units { l, p, .. }
            | Piece::Keys { l, p, .. }
            | Piece::Atoms { l, p, .. }
            | Piece::KeyDims { l, p, .. } => (*l, *p),
        }
    }
    /// the elementary changes of this piece (all in cell (l, p))
    fn ops(&self, g: &Geo, out: &mut Vec<Op>) {
        let (l, p) = self.lp();
        let lg = &g.layers[l];
        let head = |h: usize, out: &mut Vec<Op>| {
            for u in h * lg.hd..(h + 1) * lg.hd {
                out.push(Op::Unit(Site::Q, u as u32));
                out.push(Op::Unit(Site::O, u as u32));
            }
            for j in g.keys(l, p) {
                out.push(Op::Key(h, false, j as u32));
                out.push(Op::Key(h, true, j as u32));
            }
        };
        let group = |gr: usize, out: &mut Vec<Op>| {
            for u in gr * lg.hd..(gr + 1) * lg.hd {
                out.push(Op::Unit(Site::K, u as u32));
                if lg.v_site {
                    out.push(Op::Unit(Site::V, u as u32));
                }
            }
        };
        let neuron = |n: u32, out: &mut Vec<Op>| {
            if lg.gated {
                out.push(Op::Unit(Site::Gate, n));
            }
            out.push(Op::Unit(Site::Up, n));
            out.push(Op::Unit(Site::Down, n));
        };
        let dim = |e: u32, out: &mut Vec<Op>| {
            for s in [Site::PleIn, Site::PleGate, Site::PleProj] {
                out.push(Op::Unit(s, e));
            }
        };
        match self {
            Piece::Attn { .. } => (0..lg.nh).for_each(|h| head(h, out)),
            Piece::Kv { .. } => {
                out.push(Op::All(Site::K));
                if lg.v_site {
                    out.push(Op::All(Site::V));
                }
            }
            Piece::Mlp { .. } => {
                if lg.gated {
                    out.push(Op::All(Site::Gate));
                }
                out.push(Op::All(Site::Up));
                out.push(Op::All(Site::Down));
            }
            Piece::Ple { .. } => {
                for s in [Site::PleIn, Site::PleGate, Site::PleProj] {
                    out.push(Op::All(s));
                }
            }
            Piece::Heads { h, .. } => h.iter().for_each(|&h| head(h as usize, out)),
            Piece::KvGroups { g: gs, .. } => gs.iter().for_each(|&x| group(x as usize, out)),
            Piece::Neurons { n, .. } => n.iter().for_each(|&x| neuron(x, out)),
            Piece::PleDims { e, .. } => e.iter().for_each(|&x| dim(x, out)),
            Piece::Units { s, u, .. } => out.extend(u.iter().map(|&x| Op::Unit(*s, x))),
            Piece::Keys { h, av, j, .. } => out.extend(j.iter().map(|&x| Op::Key(*h, *av, x))),
            Piece::Atoms { s, u, a, .. } => out.push(Op::Atoms(*s, *u as u32, a.clone())),
            Piece::KeyDims { h, j, av, e, .. } => out.push(Op::KeyDims(*h, *av, *j as u32, e.clone())),
        }
    }
    /// number of elements of a set piece (1 for a whole component)
    fn len(&self) -> usize {
        match self {
            Piece::Attn { .. } | Piece::Kv { .. } | Piece::Mlp { .. } | Piece::Ple { .. } => 1,
            Piece::Heads { h: v, .. } | Piece::KvGroups { g: v, .. } | Piece::Neurons { n: v, .. } | Piece::PleDims { e: v, .. } | Piece::Units { u: v, .. } | Piece::Keys { j: v, .. } | Piece::Atoms { a: v, .. } | Piece::KeyDims { e: v, .. } => v.len(),
        }
    }
    /// a set piece split into one piece per element (atom sets stay as they are)
    fn singles(&self) -> Vec<Piece> {
        let one = |v: &Ids| -> Vec<Ids> { v.iter().map(|&x| Arc::new(vec![x])).collect() };
        match self {
            Piece::Heads { l, p, h } if h.len() > 1 => one(h).into_iter().map(|h| Piece::Heads { l: *l, p: *p, h }).collect(),
            Piece::KvGroups { l, p, g } if g.len() > 1 => one(g).into_iter().map(|g| Piece::KvGroups { l: *l, p: *p, g }).collect(),
            Piece::Neurons { l, p, n } if n.len() > 1 => one(n).into_iter().map(|n| Piece::Neurons { l: *l, p: *p, n }).collect(),
            Piece::PleDims { l, p, e } if e.len() > 1 => one(e).into_iter().map(|e| Piece::PleDims { l: *l, p: *p, e }).collect(),
            Piece::Units { l, p, s, u } if u.len() > 1 => one(u).into_iter().map(|u| Piece::Units { l: *l, p: *p, s: *s, u }).collect(),
            Piece::Keys { l, p, h, av, j } if j.len() > 1 => one(j).into_iter().map(|j| Piece::Keys { l: *l, p: *p, h: *h, av: *av, j }).collect(),
            _ => vec![self.clone()],
        }
    }
    /// number of atoms (for the report)
    fn n_atoms(&self, g: &Geo) -> u64 {
        let (l, p) = self.lp();
        let lg = &g.layers[l];
        let d = g.d as u64;
        let hd = lg.hd as u64;
        let nk = g.keys(l, p).len() as u64;
        let head = 2 * hd * d + 2 * nk * hd;
        let group = hd * d * if lg.v_site { 2 } else { 1 };
        let neuron = d * if lg.gated { 3 } else { 2 };
        match self {
            Piece::Attn { .. } => lg.nh as u64 * head,
            Piece::Kv { .. } => lg.nkv as u64 * group,
            Piece::Mlp { .. } => lg.ff as u64 * neuron,
            Piece::Ple { .. } => lg.dp.unwrap_or(0) as u64 * 3 * d,
            Piece::Heads { h, .. } => h.len() as u64 * head,
            Piece::KvGroups { g: gs, .. } => gs.len() as u64 * group,
            Piece::Neurons { n, .. } => n.len() as u64 * neuron,
            Piece::PleDims { e, .. } => e.len() as u64 * 3 * d,
            Piece::Units { u, .. } => u.len() as u64 * d,
            Piece::Keys { j, .. } => j.len() as u64 * hd,
            Piece::Atoms { a, .. } | Piece::KeyDims { e: a, .. } => a.len() as u64,
        }
    }
    fn name(&self, m: &Model) -> String {
        let (l, p) = self.lp();
        let at = format!("[{p}]");
        let list = |v: &Ids, pre: &str| -> String {
            if v.len() == 1 {
                format!("{pre}{}", v[0])
            } else {
                format!("{} ×{}", pre.trim_end_matches(['#', ' ']), v.len())
            }
        };
        let _ = m;
        match self {
            Piece::Attn { .. } => format!("S{l} attn {at}"),
            Piece::Kv { .. } => format!("S{l} K/V {at}"),
            Piece::Mlp { .. } => format!("S{l} MLP {at}"),
            Piece::Ple { .. } => format!("S{l} PLE {at}"),
            Piece::Heads { h, .. } => format!("S{l} {} {at}", list(h, "head ")),
            Piece::KvGroups { g, .. } => format!("S{l} {} {at}", list(g, "K/V group ")),
            Piece::Neurons { n, .. } => format!("S{l} {} {at}", list(n, "#")),
            Piece::PleDims { e, .. } => format!("S{l} {} {at}", list(e, "PLE dim ")),
            Piece::Units { s, u, .. } => format!("S{l} {} {at}", list(u, &format!("{} unit ", s.name()))),
            Piece::Keys { h, av, j, .. } => format!("S{l} head {h} {} {at}", list(j, if *av { "A·V key " } else { "QKᵀ key " })),
            Piece::Atoms { s, u, a, .. } => {
                if a.len() == 1 {
                    format!("S{l} {} {}", atom_name(*s, *u, a[0] as usize), at)
                } else {
                    format!("S{l} {} unit {u}: {} atoms {at}", s.name(), a.len())
                }
            }
            Piece::KeyDims { h, j, av, e, .. } => {
                let what = if *av { "A·V" } else { "QKᵀ" };
                if e.len() == 1 {
                    format!("S{l} head {h} {what} key [{j}] dim {} {at}", e[0])
                } else {
                    format!("S{l} head {h} {what} key [{j}]: {} dims {at}", e.len())
                }
            }
        }
    }
}

/// "W_gate[2328, 117]·x̂_117" and the like
fn atom_name(s: Site, u: usize, a: usize) -> String {
    match s {
        Site::Q => format!("W_Q[{u},{a}]·x̂_{a}"),
        Site::K => format!("W_K[{u},{a}]·x̂_{a}"),
        Site::V => format!("W_V[{u},{a}]·x̂_{a}"),
        Site::O => format!("W_O[{a},{u}]·z_{u}"),
        Site::Gate => format!("W_gate[{u},{a}]·x̂_{a} (#{u})"),
        Site::Up => format!("W_up[{u},{a}]·x̂_{a} (#{u})"),
        Site::Down => format!("W_down[{a},{u}]·a_{u} (#{u})"),
        Site::PleIn => format!("W_PLE-in[{u},{a}]·e_{a}"),
        Site::PleGate => format!("W_PLE-gate[{u},{a}]·x_{a}"),
        Site::PleProj => format!("W_PLE-proj[{a},{u}]·g_{u}"),
    }
}

// ====================================================================== state: what is switched off

/// Switched-off units and atoms of one matrix product (or of the keys of one head)
#[derive(Clone, Default)]
struct SetB {
    all: bool,
    off: BTreeSet<u32>,
    part: BTreeMap<u32, BTreeSet<u32>>,
}

impl SetB {
    fn is_empty(&self) -> bool {
        !self.all && self.off.is_empty() && self.part.is_empty()
    }
    fn unit_off(&self, u: u32) -> bool {
        self.all || self.off.contains(&u)
    }
    fn set_off(&mut self, u: u32) {
        if !self.all {
            self.off.insert(u);
            self.part.remove(&u);
        }
    }
    fn set_all(&mut self) {
        self.all = true;
        self.off.clear();
        self.part.clear();
    }
    fn add_atoms(&mut self, u: u32, a: &[u32], n: usize) {
        if self.unit_off(u) {
            return;
        }
        let e = self.part.entry(u).or_default();
        e.extend(a.iter().copied());
        if e.len() >= n {
            self.part.remove(&u);
            self.off.insert(u);
        }
    }
    /// atoms still on among n_units units with n atoms each
    fn kept(&self, n_units: usize, n: usize) -> u64 {
        if self.all {
            return 0;
        }
        let full = (n_units - self.off.len().min(n_units)) as u64 * n as u64;
        full - self.part.values().map(|s| s.len() as u64).sum::<u64>()
    }
    /// same, only over the given units (keys visible to a query)
    fn kept_in(&self, units: std::ops::Range<usize>, n: usize) -> u64 {
        if self.all {
            return 0;
        }
        units.map(|u| if self.off.contains(&(u as u32)) { 0 } else { (n - self.part.get(&(u as u32)).map_or(0, |s| s.len())) as u64 }).sum()
    }
}

#[derive(Clone, Default)]
struct HeadB {
    qk: SetB,
    av: SetB,
}

#[derive(Clone, Default)]
struct CellB {
    sites: [SetB; N_SITES],
    heads: Vec<HeadB>,
}

impl CellB {
    fn is_empty(&self) -> bool {
        self.sites.iter().all(|s| s.is_empty()) && self.heads.iter().all(|h| h.qk.is_empty() && h.av.is_empty())
    }
    fn apply(&mut self, op: &Op, g: &Geo, l: usize) {
        let d = g.d;
        let hd = g.layers[l].hd;
        if self.heads.is_empty() {
            self.heads = vec![HeadB::default(); g.layers[l].nh];
        }
        match op {
            Op::All(s) => self.sites[*s as usize].set_all(),
            Op::Unit(s, u) => self.sites[*s as usize].set_off(*u),
            Op::Atoms(s, u, a) => self.sites[*s as usize].add_atoms(*u, a, d),
            Op::Key(h, av, j) => {
                let hb = &mut self.heads[*h];
                if *av { hb.av.set_off(*j) } else { hb.qk.set_off(*j) }
            }
            Op::KeyDims(h, av, j, e) => {
                let hb = &mut self.heads[*h];
                if *av { hb.av.add_atoms(*j, e, hd) } else { hb.qk.add_atoms(*j, e, hd) }
            }
        }
    }
}

#[derive(Clone)]
struct State {
    cells: Vec<Vec<CellB>>, // [p][l]
}

// ====================================================================== building the cut for the forward pass

type Weights = RefCell<HashMap<(usize, Site, u32), Arc<Vec<f32>>>>;
type ColNorms = RefCell<HashMap<(usize, Site), Arc<Vec<f32>>>>;
/// per row of the pass, per layer
type RowCuts = Vec<Vec<Option<Arc<LayerCut>>>>;

struct Builder<'a> {
    m: &'a Model,
    /// mean ablation: the means (the constants of switched-off atoms of partial units)
    means: Option<Arc<Means>>,
    /// weights of partially cut units along their atom axis (rows or columns)
    w: Weights,
    /// lengths of the columns of W_O / PLE projections (proxies), per layer
    col_norms: ColNorms,
}

impl<'a> Builder<'a> {
    /// matrix of a site and the row of unit u in it (row-wise sites)
    fn mat(&self, l: usize, s: Site) -> (&'a Mat, Box<dyn Fn(usize) -> usize>) {
        let b = &self.m.blocks[l];
        let a = &b.attn;
        let hd = a.hd;
        let fused = |off: usize| -> Box<dyn Fn(usize) -> usize> { Box::new(move |u| (u / hd) * 3 * hd + off + u % hd) };
        let id: Box<dyn Fn(usize) -> usize> = Box::new(|u| u);
        match s {
            Site::Q => a.wq.as_ref().map_or_else(|| (a.wqkv.as_ref().unwrap(), fused(0)), |w| (w, id)),
            Site::K => a.wk.as_ref().map_or_else(|| (a.wqkv.as_ref().unwrap(), fused(hd)), |w| (w, id)),
            Site::V => a.wv.as_ref().map_or_else(|| (a.wqkv.as_ref().unwrap(), fused(2 * hd)), |w| (w, id)),
            Site::O => (&a.wo, id),
            Site::Gate => (b.mlp.gate.as_ref().unwrap(), id),
            Site::Up => (&b.mlp.up, id),
            Site::Down => (&b.mlp.down, id),
            Site::PleIn => {
                let p = self.m.ple.as_ref().unwrap();
                let dp = p.dp;
                (p.proj(), Box::new(move |u| l * dp + u))
            }
            Site::PleGate => (&b.ple.as_ref().unwrap().gate, id),
            Site::PleProj => (&b.ple.as_ref().unwrap().proj, id),
        }
    }
    /// fetch the columns of column-wise units in one pass over the matrix
    fn prefetch(&self, l: usize, s: Site, units: &[u32]) {
        if s.row_wise() {
            return;
        }
        let need: Vec<usize> = {
            let w = self.w.borrow();
            let mut v: Vec<usize> = units.iter().filter(|&&u| !w.contains_key(&(l, s, u))).map(|&u| u as usize).collect();
            v.sort_unstable();
            v.dedup();
            v
        };
        if need.is_empty() {
            return;
        }
        let (mat, _) = self.mat(l, s);
        let cols = columns(mat, &need);
        let rows = mat.rows;
        let mut w = self.w.borrow_mut();
        for (k, &u) in need.iter().enumerate() {
            w.insert((l, s, u as u32), Arc::new(cols[k * rows..(k + 1) * rows].to_vec()));
        }
    }
    fn weights(&self, l: usize, s: Site, u: u32) -> Arc<Vec<f32>> {
        if let Some(v) = self.w.borrow().get(&(l, s, u)) {
            return v.clone();
        }
        if !s.row_wise() {
            self.prefetch(l, s, &[u]);
            return self.w.borrow()[&(l, s, u)].clone();
        }
        let (mat, map) = self.mat(l, s);
        let mut buf = vec![0f32; mat.cols];
        mat.row(map(u as usize), &mut buf);
        let v = Arc::new(buf);
        self.w.borrow_mut().insert((l, s, u), v.clone());
        v
    }
    fn col_norms(&self, l: usize, s: Site) -> Arc<Vec<f32>> {
        if let Some(v) = self.col_norms.borrow().get(&(l, s)) {
            return v.clone();
        }
        let v = Arc::new(match s {
            Site::Down => self.m.blocks[l].mlp.col_norms().to_vec(),
            _ => self.mat(l, s).0.col_norms(),
        });
        self.col_norms.borrow_mut().insert((l, s), v.clone());
        v
    }
    fn site(&self, set: &SetB, l: usize, p: usize, s: Site, g: &Geo) -> Option<Arc<MatCut>> {
        if set.is_empty() {
            return None;
        }
        let n = g.d;
        let off: Vec<u32> = if set.all { (0..g.units(l, s) as u32).collect() } else { set.off.iter().copied().collect() };
        let xb = self.means.as_ref().map(|m| m.layers[l].pos[p].x(s));
        // column-wise: one row of the output (W_O split by heads: [head][d])
        let hd = g.layers[l].hd;
        let ny = if s == Site::O { g.layers[l].nh * g.d } else { g.d };
        let pos = |u: usize, r: usize| if s == Site::O { (u / hd) * g.d + r } else { r };
        let mut mean_add = (!s.row_wise() && xb.is_some() && !set.part.is_empty()).then(|| vec![0f32; ny]);
        let partial = if set.all {
            Vec::new()
        } else {
            set.part
                .iter()
                .map(|(&u, a)| {
                    let w = self.weights(l, s, u);
                    let mut mean_off = 0f32;
                    if let Some(xb) = xb {
                        if s.row_wise() {
                            mean_off = a.iter().map(|&k| w[k as usize] * xb[k as usize]).sum();
                        } else if let Some(ma) = mean_add.as_mut() {
                            for &r in a {
                                ma[pos(u as usize, r as usize)] += w[r as usize] * xb[u as usize];
                            }
                        }
                    }
                    PartialUnit { unit: u, w, atoms: atoms_of(a, n), mean_off }
                })
                .collect()
        };
        Some(Arc::new(MatCut { off, partial, mean_add: mean_add.map(Arc::new) }))
    }
    fn cell(&self, c: &CellB, l: usize, p: usize, g: &Geo) -> Option<Arc<LayerCut>> {
        if c.is_empty() {
            return None;
        }
        let mut lc = LayerCut::default();
        for s in ALL_SITES {
            lc.sites[s as usize] = self.site(&c.sites[s as usize], l, p, s, g);
        }
        let hd = g.layers[l].hd;
        lc.heads = c.heads.iter().map(|h| head_cut(h, hd)).collect();
        Some(Arc::new(lc))
    }
}

fn atoms_of(a: &BTreeSet<u32>, n: usize) -> Atoms {
    if a.len() * 2 <= n {
        Atoms::Off(Arc::new(a.iter().copied().collect()))
    } else {
        Atoms::On(Arc::new((0..n as u32).filter(|x| !a.contains(x)).collect()))
    }
}

fn key_cut(s: &SetB, hd: usize) -> KeyCut {
    KeyCut { off: s.off.iter().copied().collect(), partial: s.part.iter().map(|(&j, a)| (j, atoms_of(a, hd))).collect() }
}

fn head_cut(h: &HeadB, hd: usize) -> Option<Arc<HeadCut>> {
    if h.qk.is_empty() && h.av.is_empty() {
        return None;
    }
    Some(Arc::new(HeadCut { qk: key_cut(&h.qk, hd), av: key_cut(&h.av, hd) }))
}

// ====================================================================== the engine: exact reruns

struct Base {
    out: Out,
    ck: Checkpoint,
    logits: Vec<f32>,
}

#[derive(Default)]
struct Stats {
    passes: usize,
    tests: usize,
    commits: usize,
    refused: usize,
    /// largest |KL from the batch − KL of the standalone rerun| of an accepted step
    batch_dev: f32,
    /// largest |KL of the clean row in a batch − KL of the base run|
    clean_dev: f32,
    dead_atoms: u64,
    dead_identical: bool,
    secs: f32,
    stopped: bool,
}

struct Engine<'a> {
    m: &'a Model,
    ids: &'a [u32],
    g: Geo,
    b: Builder<'a>,
    ys: [u32; NY],
    top_ref: u32,
    p_ref: Vec<f32>,
    ref_plp: f64,
    state: State,
    built: RowCuts,
    base: Base,
    st: Stats,
    max_tests: usize,
    t0: Instant,
    tty: bool,
    /// frozen norms: the scales of the full model
    fz: Option<Arc<NormScales>>,
    /// mean ablation: the means of every atom
    means: Option<Arc<Means>>,
}

/// A candidate: pieces, with their elementary changes per cell
struct Cand {
    l0: usize,
    p0: usize,
    cells: BTreeMap<(usize, usize), Vec<Op>>, // (p, l) -> ops
}

impl<'a> Engine<'a> {
    /// the scale of the final norm at the focus (frozen norms)
    fn fin(&self) -> Option<f32> {
        self.fz.as_ref().and_then(|f| f.fin.get(self.g.focus).copied())
    }
    fn read(&self, xs: &[f32]) -> Vec<Out> {
        read_out_fz(self.m, xs, &self.ys, &self.p_ref, self.ref_plp, self.fin())
    }
    /// a standalone run with this cut (and the frozen norms, if any)
    fn opts(&self, rows: RowCuts) -> RunOpts {
        let pos = (0..rows.len() as u32).collect();
        RunOpts { cut: Some(Arc::new(Cut { rows, means: self.means.clone(), pos })), frozen_norms: self.fz.clone(), keep_final: true, ..Default::default() }
    }
    fn ok(&self, o: &Out, limit: f32) -> bool {
        o.kl <= limit && o.top[0].0 == self.top_ref
    }
    fn kl(&self) -> f32 {
        self.base.out.kl
    }
    fn cand(&self, pieces: &[Piece]) -> Cand {
        let mut cells: BTreeMap<(usize, usize), Vec<Op>> = BTreeMap::new();
        let (mut l0, mut p0) = (usize::MAX, usize::MAX);
        for pc in pieces {
            let (l, p) = pc.lp();
            l0 = l0.min(l);
            p0 = p0.min(p);
            pc.ops(&self.g, cells.entry((p, l)).or_default());
        }
        Cand { l0: l0.min(self.g.layers.len()), p0: p0.min(self.g.focus), cells }
    }
    /// the cut of cell (p, l) with extra ops on top of the state
    fn derive(&self, p: usize, l: usize, ops: &[Op]) -> Option<Arc<LayerCut>> {
        let cell = &self.state.cells[p][l];
        let mut sites: BTreeMap<usize, SetB> = BTreeMap::new();
        let mut heads: BTreeMap<usize, HeadB> = BTreeMap::new();
        let nh = self.g.layers[l].nh;
        let hd = self.g.layers[l].hd;
        for op in ops {
            match op {
                Op::All(s) | Op::Unit(s, _) | Op::Atoms(s, ..) => {
                    let sb = sites.entry(*s as usize).or_insert_with(|| cell.sites[*s as usize].clone());
                    match op {
                        Op::All(_) => sb.set_all(),
                        Op::Unit(_, u) => sb.set_off(*u),
                        Op::Atoms(_, u, a) => sb.add_atoms(*u, a, self.g.d),
                        _ => unreachable!(),
                    }
                }
                Op::Key(h, av, j) | Op::KeyDims(h, av, j, _) => {
                    let hb = heads.entry(*h).or_insert_with(|| cell.heads.get(*h).cloned().unwrap_or_default());
                    let set = if *av { &mut hb.av } else { &mut hb.qk };
                    match op {
                        Op::Key(..) => set.set_off(*j),
                        Op::KeyDims(.., e) => set.add_atoms(*j, e, hd),
                        _ => unreachable!(),
                    }
                }
            }
        }
        let mut lc: LayerCut = self.built[p][l].as_deref().cloned().unwrap_or_default();
        if lc.heads.is_empty() {
            lc.heads = vec![None; nh];
        }
        for (s, sb) in &sites {
            lc.sites[*s] = self.b.site(sb, l, p, ALL_SITES[*s], &self.g);
        }
        for (h, hb) in &heads {
            lc.heads[*h] = head_cut(hb, hd);
        }
        Some(Arc::new(lc))
    }

    /// Exact output of every candidate (each on top of the current state), many per pass
    fn eval(&mut self, pieces: &[Vec<Piece>], info: &str) -> Vec<Out> {
        let (t, d, focus) = (self.g.t, self.g.d, self.g.focus);
        let cands: Vec<Cand> = pieces.iter().map(|p| self.cand(p)).collect();
        let mut order: Vec<usize> = (0..cands.len()).collect();
        order.sort_by_key(|&k| (std::cmp::Reverse(cands[k].l0), cands[k].p0));
        let mut res = vec![Out::default(); cands.len()];
        let mut i = 0;
        let t_start = Instant::now();
        while i < order.len() {
            let mut batch: Vec<usize> = Vec::new();
            let mut rows = t;
            while i < order.len() {
                let c = &cands[order[i]];
                let len = focus + 1 - c.p0;
                if !batch.is_empty() && rows + len > t + BUDGET {
                    break;
                }
                rows += len;
                batch.push(order[i]);
                i += 1;
            }
            let l0 = batch.iter().map(|&k| cands[k].l0).min().unwrap();
            let mut pos: Vec<usize> = (0..t).collect();
            let mut seg_start = vec![0usize; t];
            let mut seg_row0 = vec![0usize; t];
            let mut ids_all: Vec<u32> = self.ids.to_vec();
            let mut cut_rows: RowCuts = self.built.clone();
            let mut focus_rows = vec![focus];
            for &k in &batch {
                let c = &cands[k];
                let r0 = pos.len();
                for q in c.p0..=focus {
                    pos.push(q);
                    seg_start.push(c.p0);
                    seg_row0.push(r0);
                    ids_all.push(self.ids[q]);
                    let mut layers = self.built[q].clone();
                    for ((p, l), ops) in c.cells.range((q, 0)..(q + 1, 0)) {
                        debug_assert_eq!(*p, q);
                        layers[*l] = self.derive(q, *l, ops);
                    }
                    cut_rows.push(layers);
                }
                focus_rows.push(r0 + (focus - c.p0));
            }
            let n_rows = pos.len();
            let xl = &self.base.ck.x[l0];
            let mut x = Vec::with_capacity(n_rows * d);
            for &q in &pos {
                x.extend_from_slice(&xl[q * d..(q + 1) * d]);
            }
            let kv: Vec<Option<(Vec<f32>, Vec<f32>)>> = self
                .base
                .ck
                .kv
                .iter()
                .enumerate()
                .map(|(s, e)| {
                    if s >= l0 {
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
            let pos_rows: Vec<u32> = pos.iter().map(|&q| q as u32).collect();
            let o = RunOpts {
                keep_final: true,
                start: Some(Start { layer: l0, x, kv: Arc::new(kv) }),
                variants: Some(Arc::new(Variants { pos: pos.clone(), seg_start, seg_row0 })),
                cut: Some(Arc::new(Cut { rows: cut_rows, means: self.means.clone(), pos: pos_rows })),
                frozen_norms: self.fz.clone(),
                ..Default::default()
            };
            let tr = self.m.run(&ids_all, focus, &o);
            let mut xs = Vec::with_capacity(focus_rows.len() * d);
            for &r in &focus_rows {
                xs.extend_from_slice(&tr.x_final[r * d..(r + 1) * d]);
            }
            let outs = self.read(&xs);
            self.st.clean_dev = self.st.clean_dev.max((outs[0].kl - self.base.out.kl).abs());
            for (b, &k) in batch.iter().enumerate() {
                res[k] = outs[b + 1];
            }
            self.st.passes += 1;
            self.st.tests += batch.len();
            if self.tty && cands.len() > batch.len() {
                eprint!("\r  {info}: {i}/{} tests ({} s)        ", cands.len(), num(t_start.elapsed().as_secs_f32(), 0));
            } else if !self.tty && self.t0.elapsed().as_secs_f32() > 20.0 {
                self.t0 = Instant::now();
                eprintln!("  {info}: {i}/{} tests in this call · {} tests, {} passes, {} commits so far · KL {}", cands.len(), thousands(self.st.tests), thousands(self.st.passes), self.st.commits, num(self.kl(), 4));
            }
        }
        if self.tty && cands.len() > 1 {
            eprint!("\r{}\r", " ".repeat(70));
        }
        res
    }

    /// rebuild the cut of the cells touched by `cand`
    fn rebuild(&mut self, cells: impl Iterator<Item = (usize, usize)>) {
        for (p, l) in cells {
            self.built[p][l] = self.b.cell(&self.state.cells[p][l], l, p, &self.g);
        }
    }

    fn run_base(&self, record: bool) -> (Base, Vec<LayerInputs>, Vec<f32>) {
        let o = RunOpts { checkpoint: true, record_inputs: record, ..self.opts(self.built.clone()) };
        let tr = self.m.run(self.ids, self.g.focus, &o);
        let d = self.g.d;
        let f = self.g.focus;
        let out = self.read(&tr.x_final[f * d..(f + 1) * d])[0];
        (Base { out, ck: tr.checkpoint.unwrap(), logits: tr.logits }, tr.inputs, tr.x_tok)
    }

    /// The base run after a change at layers ≥ l0 only: from the checkpoint before l0 (the layers
    /// before are unchanged, so this is bit-identical to a full run, just cheaper)
    fn run_base_from(&self, l0: usize) -> Base {
        let ck = &self.base.ck;
        if l0 == 0 || l0 >= ck.x.len() {
            return self.run_base(false).0;
        }
        let o = RunOpts { checkpoint: true, start: Some(Start { layer: l0, x: ck.x[l0].clone(), kv: Arc::new(ck.kv.clone()) }), ..self.opts(self.built.clone()) };
        let tr = self.m.run(self.ids, self.g.focus, &o);
        let d = self.g.d;
        let f = self.g.focus;
        let out = self.read(&tr.x_final[f * d..(f + 1) * d])[0];
        let mut nck = tr.checkpoint.unwrap();
        let mut x = ck.x[..l0].to_vec();
        x.append(&mut nck.x);
        nck.x = x;
        Base { out, ck: nck, logits: tr.logits }
    }

    /// Switch the pieces off for good, if a standalone rerun still passes the criterion
    fn commit(&mut self, pieces: &[Piece], limit: f32, kl_batch: Option<f32>) -> bool {
        let c = self.cand(pieces);
        #[allow(clippy::type_complexity)]
        let backup: Vec<((usize, usize), CellB, Option<Arc<LayerCut>>)> =
            c.cells.keys().map(|&(p, l)| ((p, l), self.state.cells[p][l].clone(), self.built[p][l].clone())).collect();
        for (&(p, l), ops) in &c.cells {
            for op in ops {
                self.state.cells[p][l].apply(op, &self.g, l);
            }
        }
        self.rebuild(c.cells.keys().copied());
        let base = self.run_base_from(c.l0);
        self.st.passes += 1;
        if self.ok(&base.out, limit) {
            if let Some(k) = kl_batch {
                self.st.batch_dev = self.st.batch_dev.max((k - base.out.kl).abs());
            }
            self.base = base;
            self.st.commits += 1;
            true
        } else {
            for ((p, l), cell, lc) in backup {
                self.state.cells[p][l] = cell;
                self.built[p][l] = lc;
            }
            self.st.refused += 1;
            false
        }
    }

    fn out_of_tests(&mut self) -> bool {
        if self.st.tests >= self.max_tests {
            self.st.stopped = true;
        }
        self.st.stopped
    }

    /// Greedy: blocks that may go individually, cheapest first, as long prefixes as possible per pass.
    /// Returns the blocks that could not go.
    fn accept(&mut self, groups: &[Group], mut rest: Vec<(Block, f32)>, limit: f32, info: &str) -> Vec<Block> {
        rest.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut rest: Vec<Block> = rest.into_iter().map(|x| x.0).collect();
        let mut refused: Vec<Block> = Vec::new();
        while !rest.is_empty() {
            if self.out_of_tests() {
                refused.append(&mut rest);
                break;
            }
            let sizes = prefix_sizes(rest.len());
            let cands: Vec<Vec<Piece>> = sizes.iter().map(|&s| rest[..s].iter().flat_map(|b| groups[b.g].make(&b.ids)).collect()).collect();
            let outs = self.eval(&cands, info);
            let best = sizes.iter().zip(&outs).filter(|(_, o)| self.ok(o, limit * MARGIN)).map(|(&s, o)| (s, o.kl)).max_by_key(|x| x.0);
            match best {
                Some((s, k)) => {
                    let pcs: Vec<Piece> = rest[..s].iter().flat_map(|b| groups[b.g].make(&b.ids)).collect();
                    if self.commit(&pcs, limit, Some(k)) {
                        rest.drain(..s);
                    } else {
                        refused.push(rest.remove(s - 1));
                    }
                }
                None => {
                    // even the cheapest one no longer fits on top: test the next chunk again on its own;
                    // if none of it fits any more, the rest is refused (the final recheck of every leaf
                    // alone catches anything that would still fit)
                    let n = rest.len().min(256);
                    let cands: Vec<Vec<Piece>> = rest[..n].iter().map(|b| groups[b.g].make(&b.ids)).collect();
                    let outs = self.eval(&cands, info);
                    let mut keep: Vec<(Block, f32)> = Vec::new();
                    for (b, o) in rest.drain(..n).zip(outs) {
                        if self.ok(&o, limit * MARGIN) {
                            keep.push((b, o.kl));
                        } else {
                            refused.push(b);
                        }
                    }
                    if keep.is_empty() {
                        refused.append(&mut rest);
                        break;
                    }
                    keep.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                    let mut front: Vec<Block> = keep.into_iter().map(|x| x.0).collect();
                    front.append(&mut rest);
                    rest = front;
                }
            }
        }
        refused
    }

    /// The sweep of a level. First the dose curve of every group: its weakest pieces up to a share τ of
    /// the group's total magnitude switched off, each group alone on top of the state, all in one batch.
    /// Then one joint choice per trade-off μ: every group takes the τ that maximizes atoms removed −
    /// μ·ΔKL. All joint choices run exactly; the one that removes the most atoms and passes is kept.
    /// Returns the number of pieces removed.
    fn sweep(&mut self, groups: &[Group], rem: &mut [Vec<u32>], limit: f32, info: &str) -> usize {
        const TAUS: [f64; 11] = [0.003, 0.01, 0.03, 0.06, 0.1, 0.16, 0.25, 0.35, 0.5, 0.7, 0.85];
        let base = self.kl();
        // prefix lengths per group (never the whole group: that is known to be needed)
        let ks: Vec<Vec<usize>> = groups
            .iter()
            .map(|g| {
                let mass: f64 = g.proxy.iter().map(|&x| x as f64).sum();
                TAUS.iter()
                    .map(|&tau| {
                        let mut acc = 0f64;
                        let mut k = 0;
                        while k + 1 < g.ids.len() {
                            acc += g.proxy[k] as f64;
                            if acc > tau * mass {
                                break;
                            }
                            k += 1;
                        }
                        k
                    })
                    .collect()
            })
            .collect();
        let apu: Vec<u64> = groups.iter().map(|g| g.make(&g.ids[..1]).iter().map(|p| p.n_atoms(&self.g)).sum::<u64>().max(1)).collect();
        let mut cands: Vec<Vec<Piece>> = Vec::new();
        let mut idx: Vec<(usize, usize)> = Vec::new();
        for (gi, kv) in ks.iter().enumerate() {
            let mut last = 0;
            for &k in kv {
                if k > last {
                    cands.push(groups[gi].make(&groups[gi].ids[..k]));
                    idx.push((gi, k));
                    last = k;
                }
            }
        }
        if cands.is_empty() {
            return 0;
        }
        let outs = self.eval(&cands, &format!("{info} dose curves"));
        drop(cands);
        // per group the options that pass alone: (pieces removed, ΔKL)
        let mut opts: Vec<Vec<(usize, f64)>> = vec![vec![(0, 0.0)]; groups.len()];
        for (&(gi, k), o) in idx.iter().zip(&outs) {
            if self.ok(o, limit * MARGIN) {
                opts[gi].push((k, (o.kl - base).max(0.0) as f64));
            }
        }
        let mut ratios: Vec<f64> = Vec::new();
        for (gi, v) in opts.iter().enumerate() {
            for &(k, c) in v {
                if k > 0 && c > 0.0 {
                    ratios.push(k as f64 * apu[gi] as f64 / c);
                }
            }
        }
        ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut mus: Vec<f64> = vec![0.0];
        for q in 0..28 {
            if let Some(&r) = ratios.get(((q as f64 + 0.5) / 28.0 * ratios.len() as f64) as usize) {
                mus.push(1.0 / r);
            }
        }
        mus.push(f64::INFINITY);
        let choose = |mu: f64| -> Vec<usize> {
            opts.iter()
                .enumerate()
                .map(|(gi, v)| {
                    let score = |&(k, c): &(usize, f64)| if mu.is_infinite() { if c <= 1e-7 { k as f64 } else { -1.0 } } else { k as f64 * apu[gi] as f64 - mu * c * 1.0 };
                    v.iter().max_by(|a, b| score(a).partial_cmp(&score(b)).unwrap()).map_or(0, |x| if score(x) >= 0.0 { x.0 } else { 0 })
                })
                .collect()
        };
        let mut choices: Vec<Vec<usize>> = Vec::new();
        for &mu in &mus {
            let c = choose(mu);
            if c.iter().any(|&k| k > 0) && !choices.contains(&c) {
                choices.push(c);
            }
        }
        if choices.is_empty() {
            return 0;
        }
        let mk = |ks: &[usize]| -> Vec<Piece> { groups.iter().zip(ks).filter(|(_, &k)| k > 0).flat_map(|(g, &k)| g.make(&g.ids[..k])).collect() };
        let joint: Vec<Vec<Piece>> = choices.iter().map(|c| mk(c)).collect();
        let outs = self.eval(&joint, &format!("{info} joint"));
        let gain = |c: &[usize]| -> u64 { c.iter().zip(&apu).map(|(&k, &a)| k as u64 * a).sum() };
        let mut best: Option<(usize, f32)> = None;
        for (i, o) in outs.iter().enumerate() {
            if self.ok(o, limit * MARGIN) && best.is_none_or(|(b, _)| gain(&choices[i]) > gain(&choices[b])) {
                best = Some((i, o.kl));
            }
        }
        let Some((bi, kl)) = best else { return 0 };
        let ks = &choices[bi];
        if self.commit(&joint[bi], limit, Some(kl)) {
            for (r, &k) in rem.iter_mut().zip(ks) {
                r.drain(..k);
            }
            ks.iter().sum()
        } else {
            0
        }
    }

    /// One level: switch off as much of the groups as the criterion allows. Returns the leaves that stay.
    fn refine(&mut self, groups: &[Group], limit: f32, whole_known: bool, exhaustive: bool, info: &str) -> (Vec<Piece>, usize) {
        let total: usize = groups.iter().map(|g| g.ids.len()).sum();
        let small = if exhaustive { SMALL_FINAL } else { SMALL };
        let mut rem: Vec<Vec<u32>> = groups.iter().map(|g| g.ids.clone()).collect();
        if whole_known && !exhaustive && total >= SWEEP_MIN {
            // the sweep may use 70% of what is left of the level's budget; the rest is for the halving
            let target = self.kl() + 0.7 * (limit - self.kl()).max(0.0);
            self.sweep(groups, &mut rem, target, info);
        }
        let mut open: Vec<Block> = Vec::new();
        for (gi, r) in rem.iter().enumerate() {
            if r.is_empty() {
                continue;
            }
            if r.len() <= small {
                open.extend(r.iter().map(|&x| Block { g: gi, ids: vec![x] }));
            } else if whole_known {
                let h = r.len() / 2;
                open.push(Block { g: gi, ids: r[..h].to_vec() });
                open.push(Block { g: gi, ids: r[h..].to_vec() });
            } else {
                open.push(Block { g: gi, ids: r.clone() });
            }
        }
        let mut kept: Vec<Piece> = Vec::new();
        let mut unresolved = 0;
        let mut round = 0;
        while !open.is_empty() {
            round += 1;
            if self.out_of_tests() {
                for b in &open {
                    unresolved += 1;
                    kept.extend(groups[b.g].make(&b.ids));
                }
                break;
            }
            let cands: Vec<Vec<Piece>> = open.iter().map(|b| groups[b.g].make(&b.ids)).collect();
            let outs = self.eval(&cands, info);
            let mut fine: Vec<(Block, f32)> = Vec::new();
            let mut stuck: Vec<Block> = Vec::new();
            for (b, o) in open.drain(..).zip(outs) {
                if self.ok(&o, limit * MARGIN) {
                    fine.push((b, o.kl));
                } else {
                    stuck.push(b);
                }
            }
            let tested = fine.len() + stuck.len();
            stuck.extend(self.accept(groups, fine, limit, info));
            // splitting further only pays while something still goes: stop when (almost) nothing did,
            // or when the budget is used up – what is left stays as whole blocks
            let removed = tested - stuck.len();
            let spent = self.kl() >= limit * 0.99;
            // (only once the blocks are small: that big halves cannot go is normal, not a sign that
            // nothing can go any more)
            let biggest = stuck.iter().map(|b| b.ids.len()).max().unwrap_or(0);
            let barren = !exhaustive && round >= 3 && tested >= 32 && biggest <= 64 && (removed as f64) < 0.05 * tested as f64;
            let spent = spent && !exhaustive;
            for b in stuck {
                if b.ids.len() == 1 {
                    kept.extend(groups[b.g].make(&b.ids));
                } else if self.out_of_tests() || spent || barren {
                    unresolved += 1;
                    kept.extend(groups[b.g].make(&b.ids));
                } else if b.ids.len() <= small {
                    open.extend(b.ids.iter().map(|&x| Block { g: b.g, ids: vec![x] }));
                } else {
                    let h = b.ids.len() / 2;
                    open.push(Block { g: b.g, ids: b.ids[..h].to_vec() });
                    open.push(Block { g: b.g, ids: b.ids[h..].to_vec() });
                }
            }
        }
        (kept, unresolved)
    }

    /// Every kept piece once more alone against the current state; whatever still fits goes (greedy).
    /// After a round, only the pieces that could go alone but were not taken are tested again (cheap
    /// rounds); when those find nothing, everything is tested again (a full round). It ends with a full
    /// round in which nothing can go – then every piece is individually necessary – or after `rounds`
    /// full rounds (one is enough after a level: the final phase runs until nothing changes).
    /// Returns how many went and, after a final full round that removed nothing, the individual outputs
    /// (aligned with `kept`).
    fn tighten(&mut self, kept: &mut Vec<Piece>, limit: f32, rounds: Option<usize>, info: &str) -> (usize, Vec<Out>) {
        let mut removed = 0;
        let mut full_rounds = 0;
        // None: test all; Some(flags): only the flagged pieces
        let mut subset: Option<Vec<bool>> = None;
        loop {
            if self.out_of_tests() || kept.is_empty() {
                return (removed, Vec::new());
            }
            if subset.is_none() {
                if rounds.is_some_and(|r| full_rounds >= r) {
                    return (removed, Vec::new());
                }
                full_rounds += 1;
            }
            let was_full = subset.is_none();
            let idx: Vec<usize> = match &subset {
                None => (0..kept.len()).collect(),
                Some(f) => (0..kept.len()).filter(|&i| f[i]).collect(),
            };
            let cands: Vec<Vec<Piece>> = idx.iter().map(|&i| vec![kept[i].clone()]).collect();
            let outs = self.eval(&cands, info);
            let fine: Vec<(Block, f32)> = idx.iter().zip(&outs).filter(|(_, o)| self.ok(o, limit * MARGIN)).map(|(&i, o)| (Block { g: 0, ids: vec![i as u32] }, o.kl)).collect();
            if fine.is_empty() {
                if subset.is_none() {
                    return (removed, outs);
                }
                subset = None;
                continue;
            }
            let mut was_fine = vec![false; kept.len()];
            for (b, _) in &fine {
                was_fine[b.ids[0] as usize] = true;
            }
            let gl = vec![Group { kind: GK::List(kept.clone()), ids: (0..kept.len() as u32).collect(), proxy: vec![0.0; kept.len()] }];
            let n0 = kept.len();
            let _ = self.accept(&gl, fine, limit, info);
            self.prune_dead();
            let keep: Vec<bool> = kept.iter().map(|pc| !self.piece_off(pc)).collect();
            let mut flags = Vec::with_capacity(n0);
            let mut i = 0;
            kept.retain(|_| {
                let k = keep[i];
                if k {
                    flags.push(was_fine[i]);
                }
                i += 1;
                k
            });
            removed += n0 - kept.len();
            if kept.len() == n0 {
                // candidates, but none could be taken: a full round cannot do better
                if was_full {
                    return (removed, Vec::new());
                }
                subset = None;
                continue;
            }
            subset = if flags.iter().any(|&f| f) { Some(flags) } else { None };
        }
    }

    // ------------------------------------------------------------------ pieces without any effect

    /// Ops that switch off what provably has no effect any more: a head whose W_O columns are all off
    /// (or whose A·V products are all off: its z is exactly 0), a neuron whose gate, up (without bias) or
    /// down is off, a PLE dimension whose gate or projection is off, K/V rows no product reads.
    fn dead_ops(&self) -> BTreeMap<(usize, usize), Vec<Op>> {
        let g = &self.g;
        // zero ablation: an input that is exactly 0 also makes pieces dead; with mean ablation only a
        // switched-off output does
        let zero = self.means.is_none();
        let mut ops: BTreeMap<(usize, usize), Vec<Op>> = BTreeMap::new();
        for p in 0..=g.focus {
            for (l, lg) in g.layers.iter().enumerate() {
                let c = &self.state.cells[p][l];
                if c.is_empty() {
                    continue;
                }
                let mut v: Vec<Op> = Vec::new();
                let keys = g.keys(l, p);
                for h in 0..lg.nh {
                    let hr = h * lg.hd..(h + 1) * lg.hd;
                    let o_off = hr.clone().all(|u| c.sites[Site::O as usize].unit_off(u as u32));
                    // (with mean ablation z is not 0 when all A·V products are off: then only W_O off counts)
                    let av_off = zero && c.heads.get(h).is_some_and(|hb| keys.clone().all(|j| hb.av.unit_off(j as u32)));
                    if o_off || av_off {
                        let hb = c.heads.get(h);
                        let done = |s: &SetB| keys.clone().all(|j| s.unit_off(j as u32));
                        let q_done = hr.clone().all(|u| c.sites[Site::Q as usize].unit_off(u as u32));
                        if !(o_off && q_done && hb.is_some_and(|hb| done(&hb.qk) && done(&hb.av))) {
                            for u in hr.clone() {
                                v.push(Op::Unit(Site::Q, u as u32));
                                v.push(Op::Unit(Site::O, u as u32));
                            }
                            for j in keys.clone() {
                                v.push(Op::Key(h, false, j as u32));
                                v.push(Op::Key(h, true, j as u32));
                            }
                        }
                    }
                }
                let (gs, us, ds) = (&c.sites[Site::Gate as usize], &c.sites[Site::Up as usize], &c.sites[Site::Down as usize]);
                if !(gs.is_empty() && us.is_empty() && ds.is_empty()) {
                    for n in 0..lg.ff as u32 {
                        let (go, uo, dn) = (gs.unit_off(n), us.unit_off(n), ds.unit_off(n));
                        let dead = if !zero { dn } else if lg.gated { go || (uo && !lg.up_bias) || dn } else { dn || (uo && !lg.up_bias) };
                        if dead && !((go || !lg.gated) && uo && dn) {
                            if lg.gated {
                                v.push(Op::Unit(Site::Gate, n));
                            }
                            if lg.gated || dn {
                                v.push(Op::Unit(Site::Up, n));
                            }
                            v.push(Op::Unit(Site::Down, n));
                        }
                    }
                }
                if let Some(dp) = lg.dp {
                    let (gs, ps, is) = (&c.sites[Site::PleGate as usize], &c.sites[Site::PleProj as usize], &c.sites[Site::PleIn as usize]);
                    let mut all_dead = true;
                    for e in 0..dp as u32 {
                        let dead = (zero && gs.unit_off(e)) || ps.unit_off(e);
                        all_dead &= dead;
                        if dead && !(gs.unit_off(e) && ps.unit_off(e)) {
                            v.push(Op::Unit(Site::PleGate, e));
                            v.push(Op::Unit(Site::PleProj, e));
                        }
                    }
                    // the PLE input of a layer is normalized over all its dimensions: dead only as a whole
                    if all_dead && !is.all {
                        v.push(Op::All(Site::PleIn));
                    }
                }
                if !v.is_empty() {
                    ops.entry((p, l)).or_default().extend(v);
                }
            }
        }
        // K/V rows at position p that no QKᵀ / A·V product reads any more
        for (l, lg) in g.layers.iter().enumerate() {
            if !lg.own_kv {
                continue;
            }
            for p in 0..=g.focus {
                let rep = lg.nh / lg.nkv;
                for gr in 0..lg.nkv {
                    let read = |av: bool| -> bool {
                        g.readers[l].iter().any(|&l2| {
                            (p..=g.focus).filter(|&i| g.keys(l2, i).contains(&p)).any(|i| {
                                let c = &self.state.cells[i][l2];
                                (gr * rep..(gr + 1) * rep).any(|h| c.heads.get(h).is_none_or(|hb| !(if av { &hb.av } else { &hb.qk }).unit_off(p as u32)))
                            })
                        })
                    };
                    let (qk_read, av_read) = (read(false), read(true));
                    let c = &self.state.cells[p][l];
                    let ks = &c.sites[Site::K as usize];
                    let vs = &c.sites[Site::V as usize];
                    let rows = gr * lg.hd..(gr + 1) * lg.hd;
                    let mut v = Vec::new();
                    let k_dead = if lg.v_site { !qk_read } else { !qk_read && !av_read };
                    if k_dead && !rows.clone().all(|u| ks.unit_off(u as u32)) {
                        v.extend(rows.clone().map(|u| Op::Unit(Site::K, u as u32)));
                    }
                    if lg.v_site && !av_read && !rows.clone().all(|u| vs.unit_off(u as u32)) {
                        v.extend(rows.clone().map(|u| Op::Unit(Site::V, u as u32)));
                    }
                    if !v.is_empty() {
                        ops.entry((p, l)).or_default().extend(v);
                    }
                }
            }
        }
        ops
    }

    /// Switch off everything without effect; the output has to stay bit-identical
    fn prune_dead(&mut self) {
        let before = self.atoms_kept();
        let logits0 = self.base.logits.clone();
        for _ in 0..4 {
            let ops = self.dead_ops();
            if ops.is_empty() {
                break;
            }
            for (&(p, l), v) in &ops {
                for op in v {
                    self.state.cells[p][l].apply(op, &self.g, l);
                }
            }
            self.rebuild(ops.keys().copied());
        }
        let n = before.saturating_sub(self.atoms_kept());
        if n == 0 {
            return;
        }
        let (base, _, _) = self.run_base(false);
        self.st.passes += 1;
        let same = base.logits.len() == logits0.len() && base.logits.iter().zip(&logits0).all(|(a, b)| a.to_bits() == b.to_bits());
        self.st.dead_identical &= same;
        self.st.dead_atoms += n;
        self.base = base;
    }

    fn atoms_kept(&self) -> u64 {
        let g = &self.g;
        let mut n = 0u64;
        for p in 0..=g.focus {
            for l in 0..g.layers.len() {
                n += self.cell_kept(p, l).iter().sum::<u64>();
            }
        }
        n
    }

    /// atoms still on in cell (p, l): per site, then QKᵀ, A·V
    fn cell_kept(&self, p: usize, l: usize) -> [u64; N_SITES + 2] {
        let g = &self.g;
        let lg = &g.layers[l];
        let c = &self.state.cells[p][l];
        let mut r = [0u64; N_SITES + 2];
        for s in ALL_SITES {
            if g.has(l, s) {
                r[s as usize] = c.sites[s as usize].kept(g.units(l, s), g.d);
            }
        }
        let keys = g.keys(l, p);
        for h in 0..lg.nh {
            match c.heads.get(h) {
                Some(hb) => {
                    r[N_SITES] += hb.qk.kept_in(keys.clone(), lg.hd);
                    r[N_SITES + 1] += hb.av.kept_in(keys.clone(), lg.hd);
                }
                None => {
                    r[N_SITES] += (keys.len() * lg.hd) as u64;
                    r[N_SITES + 1] += (keys.len() * lg.hd) as u64;
                }
            }
        }
        r
    }

    fn piece_off(&self, pc: &Piece) -> bool {
        let (l, p) = pc.lp();
        let c = &self.state.cells[p][l];
        let mut ops = Vec::new();
        pc.ops(&self.g, &mut ops);
        ops.iter().all(|op| match op {
            Op::All(s) => c.sites[*s as usize].all,
            Op::Unit(s, u) => c.sites[*s as usize].unit_off(*u),
            Op::Atoms(s, u, a) => {
                let sb = &c.sites[*s as usize];
                sb.unit_off(*u) || sb.part.get(u).is_some_and(|x| a.iter().all(|y| x.contains(y)))
            }
            Op::Key(h, av, j) => c.heads.get(*h).is_some_and(|hb| (if *av { &hb.av } else { &hb.qk }).unit_off(*j)),
            Op::KeyDims(h, av, j, e) => c.heads.get(*h).is_some_and(|hb| {
                let s = if *av { &hb.av } else { &hb.qk };
                s.unit_off(*j) || s.part.get(j).is_some_and(|x| e.iter().all(|y| x.contains(y)))
            }),
        })
    }
}

/// prefix lengths tested in one pass: 1, 2, 3, 4, 6, 8, 12, 16, … and the whole list
fn prefix_sizes(n: usize) -> Vec<usize> {
    let mut v = Vec::new();
    let mut s = 1usize;
    let mut k = 0;
    while s < n {
        v.push(s);
        s = if k % 2 == 0 { s + s.div_ceil(2) } else { s * 4 / 3 + 1 };
        k += 1;
    }
    v.push(n);
    v.dedup();
    v
}

// ====================================================================== groups: children of a kept piece, weakest first

#[derive(Clone)]
enum GK {
    List(Vec<Piece>),
    Heads { l: usize, p: usize },
    KvGroups { l: usize, p: usize },
    Neurons { l: usize, p: usize },
    PleDims { l: usize, p: usize },
    Units { l: usize, p: usize, s: Site },
    Keys { l: usize, p: usize, h: usize, av: bool },
    Atoms { l: usize, p: usize, s: Site, u: usize },
    KeyDims { l: usize, p: usize, h: usize, j: usize, av: bool },
}

#[derive(Clone)]
struct Group {
    kind: GK,
    /// ids of the children, weakest first
    ids: Vec<u32>,
    /// their magnitudes (only for the order and the sweep)
    proxy: Vec<f32>,
}

#[derive(Clone)]
struct Block {
    g: usize,
    ids: Vec<u32>,
}

impl Group {
    fn new(kind: GK, items: Vec<(u32, f32)>) -> Self {
        let mut items = items;
        items.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0)));
        Group { kind, ids: items.iter().map(|x| x.0).collect(), proxy: items.iter().map(|x| x.1).collect() }
    }
    fn make(&self, sel: &[u32]) -> Vec<Piece> {
        let ids = || {
            let mut v = sel.to_vec();
            v.sort_unstable();
            Arc::new(v)
        };
        match &self.kind {
            GK::List(v) => sel.iter().map(|&i| v[i as usize].clone()).collect(),
            GK::Heads { l, p } => vec![Piece::Heads { l: *l, p: *p, h: ids() }],
            GK::KvGroups { l, p } => vec![Piece::KvGroups { l: *l, p: *p, g: ids() }],
            GK::Neurons { l, p } => vec![Piece::Neurons { l: *l, p: *p, n: ids() }],
            GK::PleDims { l, p } => vec![Piece::PleDims { l: *l, p: *p, e: ids() }],
            GK::Units { l, p, s } => vec![Piece::Units { l: *l, p: *p, s: *s, u: ids() }],
            GK::Keys { l, p, h, av } => vec![Piece::Keys { l: *l, p: *p, h: *h, av: *av, j: ids() }],
            GK::Atoms { l, p, s, u } => vec![Piece::Atoms { l: *l, p: *p, s: *s, u: *u, a: ids() }],
            GK::KeyDims { l, p, h, j, av } => vec![Piece::KeyDims { l: *l, p: *p, h: *h, j: *j, av: *av, e: ids() }],
        }
    }
}

/// what a site reads at position p (row-wise sites: a vector; column-wise: the unit's input value)
fn site_input<'b>(inp: &'b LayerInputs, x_tok: &'b [f32], s: Site, p: usize, d: usize) -> &'b [f32] {
    let t = inp.h.len() / d;
    match s {
        Site::Q | Site::K | Site::V => &inp.h[p * d..(p + 1) * d],
        Site::Gate | Site::Up => &inp.h2[p * d..(p + 1) * d],
        Site::PleIn => &x_tok[p * d..(p + 1) * d],
        Site::PleGate => &inp.ple_x[p * d..(p + 1) * d],
        Site::O => {
            let n = inp.z.len() / t;
            &inp.z[p * n..(p + 1) * n]
        }
        Site::Down => {
            let n = inp.act.len() / t;
            &inp.act[p * n..(p + 1) * n]
        }
        Site::PleProj => {
            let n = inp.ple_g.len() / t;
            &inp.ple_g[p * n..(p + 1) * n]
        }
    }
}

impl<'a> Engine<'a> {
    /// The groups of the next level for the pieces kept so far. Neurons and the gate/projection of PLE
    /// dimensions have no level 3 of their own (switching off one of their units = switching off the
    /// whole neuron/dimension): they pass through to level 4.
    fn expand(&self, kept: &[Piece], level: usize, inp: &[LayerInputs], x_tok: &[f32]) -> (Vec<Group>, Vec<Piece>) {
        let g = &self.g;
        let d = g.d;
        let mut groups: Vec<Group> = Vec::new();
        let mut pass: Vec<Piece> = Vec::new();
        let one = |p: &Piece| -> Option<u32> {
            match p {
                Piece::Heads { h: v, .. } | Piece::KvGroups { g: v, .. } | Piece::Neurons { n: v, .. } | Piece::PleDims { e: v, .. } | Piece::Units { u: v, .. } | Piece::Keys { j: v, .. } if v.len() == 1 => Some(v[0]),
                _ => None,
            }
        };
        let mut ple_in: BTreeMap<(usize, usize), Vec<u32>> = BTreeMap::new();
        // blocks of neurons / PLE dimensions that stayed whole are split further at level 3 (they have no
        // level 3 of their own); other blocks are expanded element by element
        let mut rest: Vec<Piece> = Vec::new();
        for pc in kept {
            match (level, pc) {
                (3, Piece::Neurons { l, p, n }) if n.len() > 1 => {
                    let cn = self.b.col_norms(*l, Site::Down);
                    let a = site_input(&inp[*l], x_tok, Site::Down, *p, d);
                    groups.push(Group::new(GK::Neurons { l: *l, p: *p }, n.iter().map(|&x| (x, a[x as usize].abs() * cn[x as usize])).collect()));
                }
                (3, Piece::PleDims { l, p, e }) if e.len() > 1 => {
                    let cn = self.b.col_norms(*l, Site::PleProj);
                    let gv = site_input(&inp[*l], x_tok, Site::PleProj, *p, d);
                    groups.push(Group::new(GK::PleDims { l: *l, p: *p }, e.iter().map(|&x| (x, gv[x as usize].abs() * cn[x as usize])).collect()));
                }
                _ => rest.push(pc.clone()),
            }
        }
        let kept: Vec<Piece> = rest.iter().flat_map(|p| p.singles()).collect();
        for pc in &kept {
            let (l, p) = pc.lp();
            let lg = &g.layers[l];
            let li = &inp[l];
            match (level, pc) {
                (2, Piece::Attn { .. }) => groups.push(Group::new(GK::Heads { l, p }, (0..lg.nh as u32).map(|h| (h, 0.0)).collect())),
                (2, Piece::Kv { .. }) => groups.push(Group::new(GK::KvGroups { l, p }, (0..lg.nkv as u32).map(|x| (x, 0.0)).collect())),
                (2, Piece::Mlp { .. }) => {
                    let cn = self.b.col_norms(l, Site::Down);
                    let a = site_input(li, x_tok, Site::Down, p, d);
                    groups.push(Group::new(GK::Neurons { l, p }, (0..lg.ff).map(|n| (n as u32, a[n].abs() * cn[n])).collect()));
                }
                (2, Piece::Ple { .. }) => {
                    let cn = self.b.col_norms(l, Site::PleProj);
                    let gv = site_input(li, x_tok, Site::PleProj, p, d);
                    groups.push(Group::new(GK::PleDims { l, p }, (0..lg.dp.unwrap_or(0)).map(|e| (e as u32, gv[e].abs() * cn[e])).collect()));
                }
                (3, Piece::Heads { .. }) => {
                    let h = one(pc).unwrap() as usize;
                    let (nh, hd) = (lg.nh, lg.hd);
                    let ad = nh * hd;
                    let rows = h * hd..(h + 1) * hd;
                    let qp = &li.q_proj[p * ad..(p + 1) * ad];
                    groups.push(Group::new(GK::Units { l, p, s: Site::Q }, rows.clone().map(|u| (u as u32, qp[u].abs())).collect()));
                    let cn = self.b.col_norms(l, Site::O);
                    let z = &li.z[p * ad..(p + 1) * ad];
                    groups.push(Group::new(GK::Units { l, p, s: Site::O }, rows.clone().map(|u| (u as u32, z[u].abs() * cn[u])).collect()));
                    let t = g.t;
                    let pr = &li.probs[(p * nh + h) * t..(p * nh + h + 1) * t];
                    let gr = h / (nh / lg.nkv);
                    let kvd = lg.nkv * hd;
                    let vn = |j: usize| -> f32 { li.v[j * kvd + gr * hd..][..hd].iter().map(|x| x * x).sum::<f32>().sqrt() };
                    groups.push(Group::new(GK::Keys { l, p, h, av: false }, g.keys(l, p).map(|j| (j as u32, pr[j])).collect()));
                    groups.push(Group::new(GK::Keys { l, p, h, av: true }, g.keys(l, p).map(|j| (j as u32, pr[j] * vn(j))).collect()));
                }
                (3, Piece::KvGroups { .. }) => {
                    let gr = one(pc).unwrap() as usize;
                    let kvd = lg.nkv * lg.hd;
                    let rows = gr * lg.hd..(gr + 1) * lg.hd;
                    let kp = &li.k_proj[p * kvd..(p + 1) * kvd];
                    groups.push(Group::new(GK::Units { l, p, s: Site::K }, rows.clone().map(|u| (u as u32, kp[u].abs())).collect()));
                    if lg.v_site {
                        let vp = &li.v_proj[p * kvd..(p + 1) * kvd];
                        groups.push(Group::new(GK::Units { l, p, s: Site::V }, rows.clone().map(|u| (u as u32, vp[u].abs())).collect()));
                    }
                }
                (3, Piece::PleDims { .. }) => {
                    ple_in.entry((l, p)).or_default().push(one(pc).unwrap());
                    pass.push(pc.clone());
                }
                (3, Piece::Neurons { .. }) => pass.push(pc.clone()),
                (4, Piece::Units { s, .. }) => groups.push(self.atom_group(l, p, *s, one(pc).unwrap() as usize, li, x_tok)),
                (4, Piece::Neurons { .. }) => {
                    let n = one(pc).unwrap() as usize;
                    for s in [Site::Gate, Site::Up, Site::Down] {
                        if g.has(l, s) {
                            groups.push(self.atom_group(l, p, s, n, li, x_tok));
                        }
                    }
                }
                (4, Piece::PleDims { .. }) => {
                    let e = one(pc).unwrap() as usize;
                    for s in [Site::PleGate, Site::PleProj] {
                        groups.push(self.atom_group(l, p, s, e, li, x_tok));
                    }
                }
                (4, Piece::Keys { h, av, .. }) => {
                    let j = one(pc).unwrap() as usize;
                    let (nh, hd) = (lg.nh, lg.hd);
                    let gr = h / (nh / lg.nkv);
                    let (ad, kvd, t) = (nh * hd, lg.nkv * hd, g.t);
                    let items: Vec<(u32, f32)> = if *av {
                        let a = li.probs[(p * nh + h) * t + j];
                        (0..hd).map(|e| (e as u32, (a * li.v[j * kvd + gr * hd + e]).abs())).collect()
                    } else {
                        (0..hd).map(|e| (e as u32, (li.q[p * ad + h * hd + e] * li.k[j * kvd + gr * hd + e]).abs())).collect()
                    };
                    groups.push(Group::new(GK::KeyDims { l, p, h: *h, j, av: *av }, items));
                }
                _ => pass.push(pc.clone()),
            }
        }
        for ((l, p), es) in ple_in {
            groups.push(Group::new(GK::Units { l, p, s: Site::PleIn }, es.into_iter().map(|e| (e, 0.0)).collect()));
        }
        (groups, pass)
    }

    /// a block that stayed whole as a group of its elements (weakest first, where a magnitude is known)
    fn block_group(&self, b: &Piece, inp: &[LayerInputs], x_tok: &[f32]) -> Group {
        let sub = |g: Group, keep: &Ids| -> Group {
            let items: Vec<(u32, f32)> = g.ids.iter().zip(&g.proxy).filter(|(i, _)| keep.binary_search(i).is_ok()).map(|(i, p)| (*i, *p)).collect();
            Group::new(g.kind, items)
        };
        let flat = |kind: GK, v: &Ids| Group::new(kind, v.iter().map(|&x| (x, 0.0)).collect());
        match b {
            Piece::Atoms { l, p, s, u, a } => sub(self.atom_group(*l, *p, *s, *u, &inp[*l], x_tok), a),
            Piece::KeyDims { l, p, h, j, av, e } => {
                let lg = &self.g.layers[*l];
                let li = &inp[*l];
                let (nh, hd) = (lg.nh, lg.hd);
                let gr = h / (nh / lg.nkv);
                let (ad, kvd, t) = (nh * hd, lg.nkv * hd, self.g.t);
                let items: Vec<(u32, f32)> = e
                    .iter()
                    .map(|&x| {
                        let x = x as usize;
                        let v = if *av { li.probs[(p * nh + h) * t + j] * li.v[j * kvd + gr * hd + x] } else { li.q[p * ad + h * hd + x] * li.k[j * kvd + gr * hd + x] };
                        (x as u32, v.abs())
                    })
                    .collect();
                Group::new(GK::KeyDims { l: *l, p: *p, h: *h, j: *j, av: *av }, items)
            }
            Piece::Heads { l, p, h } => flat(GK::Heads { l: *l, p: *p }, h),
            Piece::KvGroups { l, p, g } => flat(GK::KvGroups { l: *l, p: *p }, g),
            Piece::Neurons { l, p, n } => flat(GK::Neurons { l: *l, p: *p }, n),
            Piece::PleDims { l, p, e } => flat(GK::PleDims { l: *l, p: *p }, e),
            Piece::Units { l, p, s, u } => flat(GK::Units { l: *l, p: *p, s: *s }, u),
            Piece::Keys { l, p, h, av, j } => flat(GK::Keys { l: *l, p: *p, h: *h, av: *av }, j),
            _ => Group { kind: GK::List(vec![b.clone()]), ids: vec![0], proxy: vec![0.0] },
        }
    }

    fn atom_group(&self, l: usize, p: usize, s: Site, u: usize, li: &LayerInputs, x_tok: &[f32]) -> Group {
        let d = self.g.d;
        let w = self.b.weights(l, s, u as u32);
        let x = site_input(li, x_tok, s, p, d);
        let items: Vec<(u32, f32)> = if s.row_wise() { (0..d).map(|a| (a as u32, (w[a] * x[a]).abs())).collect() } else { (0..d).map(|r| (r as u32, (w[r] * x[u]).abs())).collect() };
        Group::new(GK::Atoms { l, p, s, u }, items)
    }
}

// ====================================================================== mean ablation

/// The means for mean ablation, per layer and position of the analyzed sequence (T positions).
///
/// * With reference prompts (`--circuit-ref`, same format): position-wise – at position p the mean over
///   the references that have a position p, as is usual in circuit work (IOI and the like).
/// * Without, or where no reference reaches p: the mean over all positions of the references (the
///   analyzed sequence itself by default), except a leading special token.
/// * A leading special token (<bos>) is context-free – the same activations in every sequence – so at
///   position 0 its own activations are the mean: switching off atoms there changes nothing.
///
/// Means are taken of the inputs of every matrix product and of the factors of QKᵀ and A·V (q after norm
/// and RoPE, k, v, the attention weights per query/key position; for the weights the fallback is the
/// mean weight on <bos> or on any other key). W·x̄ is precomputed for the row-wise sites.
/// Returns the means and, per reference, the positions it contributed to the fallback.
fn compute_means(m: &Model, g: &Geo, b: &Builder, refs: &[Vec<u32>], ids: &[u32], positionwise: bool) -> (Means, Vec<Vec<usize>>) {
    let (nl, d, t) = (g.layers.len(), g.d, ids.len());
    let bos = m.tok.is_special(ids[0]);
    // per layer: one slot per position (0..t) and a global slot (t)
    #[derive(Clone, Default)]
    struct Acc {
        h: Vec<f64>,
        z: Vec<f64>,
        h2: Vec<f64>,
        act: Vec<f64>,
        ple_x: Vec<f64>,
        ple_g: Vec<f64>,
        tok: Vec<f64>,
        q: Vec<f64>,
        k: Vec<f64>,
        v: Vec<f64>,
        /// position slots [nh][t]; global slot [nh][2] = on <bos>, on any other key
        a: Vec<f64>,
        an: Vec<f64>,
        n: f64,
    }
    fn add(acc: &mut Vec<f64>, x: &[f32]) {
        if acc.is_empty() {
            acc.resize(x.len(), 0.0);
        }
        for (a, v) in acc.iter_mut().zip(x) {
            *a += *v as f64;
        }
    }
    let mut accs: Vec<Vec<Acc>> = vec![vec![Acc::default(); t + 1]; nl];
    let mut used = Vec::new();
    for r in refs {
        let rt = r.len();
        let tr = m.run(r, rt - 1, &RunOpts { record_inputs: true, ..Default::default() });
        let start = usize::from(rt > 1 && m.tok.is_special(r[0]));
        used.push((start..rt).collect::<Vec<_>>());
        for (l, li) in tr.inputs.iter().enumerate() {
            let lg = &g.layers[l];
            let (nh, hd) = (lg.nh, lg.hd);
            let (ad, kvd) = (nh * hd, lg.nkv * hd);
            for p in 0..rt {
                // the slot of position p (position-wise, and position 0 for <bos>) and the global slot
                let mut slots = Vec::new();
                if p < t && (positionwise || (p == 0 && bos && start == 1)) {
                    slots.push(p);
                }
                if p >= start {
                    slots.push(t);
                }
                for &sl in &slots {
                    let a = &mut accs[l][sl];
                    a.n += 1.0;
                    add(&mut a.h, &li.h[p * d..(p + 1) * d]);
                    add(&mut a.h2, &li.h2[p * d..(p + 1) * d]);
                    add(&mut a.z, site_input(li, &tr.x_tok, Site::O, p, d));
                    add(&mut a.act, site_input(li, &tr.x_tok, Site::Down, p, d));
                    if lg.dp.is_some() {
                        add(&mut a.ple_x, &li.ple_x[p * d..(p + 1) * d]);
                        add(&mut a.ple_g, site_input(li, &tr.x_tok, Site::PleProj, p, d));
                    }
                    add(&mut a.tok, &tr.x_tok[p * d..(p + 1) * d]);
                    add(&mut a.q, &li.q[p * ad..(p + 1) * ad]);
                    add(&mut a.k, &li.k[p * kvd..(p + 1) * kvd]);
                    add(&mut a.v, &li.v[p * kvd..(p + 1) * kvd]);
                    if sl < t {
                        if a.a.is_empty() {
                            a.a = vec![0.0; nh * t];
                            a.an = vec![0.0; nh * t];
                        }
                        for h in 0..nh {
                            for j in g.keys(l, p).filter(|&j| j < t) {
                                a.a[h * t + j] += li.probs[(p * nh + h) * rt + j] as f64;
                                a.an[h * t + j] += 1.0;
                            }
                        }
                    } else {
                        if a.a.is_empty() {
                            a.a = vec![0.0; nh * 2];
                            a.an = vec![0.0; nh * 2];
                        }
                        for h in 0..nh {
                            for j in g.keys(l, p) {
                                let c = usize::from(!(j == 0 && start == 1));
                                a.a[h * 2 + c] += li.probs[(p * nh + h) * rt + j] as f64;
                                a.an[h * 2 + c] += 1.0;
                            }
                        }
                    }
                }
            }
        }
    }
    let mean = |v: &[f64], n: f64| -> Vec<f32> { v.iter().map(|x| (x / n.max(1.0)) as f32).collect() };
    let mut layers = Vec::with_capacity(nl);
    for l in 0..nl {
        let nh = g.layers[l].nh;
        let glob = &accs[l][t];
        let mut pos = Vec::with_capacity(t);
        for p in 0..t {
            let a = &accs[l][p];
            let src = if a.n > 0.0 { a } else { glob };
            let mut pm = PosMeans {
                h: mean(&src.h, src.n),
                z: mean(&src.z, src.n),
                h2: mean(&src.h2, src.n),
                act: mean(&src.act, src.n),
                ple_x: mean(&src.ple_x, src.n),
                ple_g: mean(&src.ple_g, src.n),
                tok: mean(&src.tok, src.n),
                q: mean(&src.q, src.n),
                k: mean(&src.k, src.n),
                v: mean(&src.v, src.n),
                ..Default::default()
            };
            // attention weights: of this query position where known, else the global mean weight on
            // <bos> / on any other key
            pm.a = vec![0f32; nh * t];
            for h in 0..nh {
                for j in 0..t {
                    let known = a.n > 0.0 && a.an.get(h * t + j).is_some_and(|&c| c > 0.0);
                    pm.a[h * t + j] = if known {
                        (a.a[h * t + j] / a.an[h * t + j]) as f32
                    } else {
                        let c = usize::from(!(j == 0 && bos));
                        (glob.a.get(h * 2 + c).copied().unwrap_or(0.0) / glob.an.get(h * 2 + c).copied().unwrap_or(1.0).max(1.0)) as f32
                    };
                }
            }
            for s in ALL_SITES {
                if !g.has(l, s) || !s.row_wise() {
                    continue;
                }
                let (mat, map) = b.mat(l, s);
                let full = mat.matmul(pm.x(s), 1);
                pm.y[s as usize] = (0..g.units(l, s)).map(|u| full[map(u)]).collect();
            }
            pos.push(pm);
        }
        layers.push(LayerMeans { pos });
    }
    (Means { layers }, used)
}

// ====================================================================== flow

struct LevelLog {
    level: usize,
    groups: usize,
    pieces_in: usize,
    kept: usize,
    unresolved: usize,
    atoms: u64,
    kl: f32,
    tests: usize,
    passes: usize,
    secs: f32,
}

/// A leaf of the final circuit with the effect of switching it off alone
struct Leaf {
    piece: Piece,
    out: Out,
    value: Option<f32>,
}

pub fn run(m: &Model, ids: &[u32], focus: usize, target: Option<u32>, o: &Opts, st: &Style) -> Result<(), String> {
    if !m.has_head {
        return Err("--circuit needs a model with an output head".into());
    }
    if focus + 1 != ids.len() {
        return Err("--circuit analyzes the last token of the sequence".into());
    }
    if o.kl.is_nan() || o.kl <= 0.0 {
        return Err("--circuit-kl must be positive".into());
    }
    let t_all = Instant::now();
    let (t, d) = (ids.len(), m.d);
    let g = Geo::new(m, t, focus);
    let nl = g.layers.len();
    // the full model
    let full = m.run(ids, focus, &RunOpts { keep_final: true, record_norms: true, ..Default::default() });
    let fz = if o.frozen { full.norms.clone().map(Arc::new) } else { None };
    let (logz, p_ref) = softmax(&full.logits);
    let ref_plp: f64 = full.logits.iter().map(|x| { let lp = *x as f64 - logz; lp.exp() * lp }).sum();
    let mut order: Vec<usize> = (0..full.logits.len()).collect();
    order.sort_by(|a, b| full.logits[*b].partial_cmp(&full.logits[*a]).unwrap_or(std::cmp::Ordering::Equal));
    let top_ref = order[0] as u32;
    let y0 = target.unwrap_or(top_ref);
    let rival = order.iter().map(|&i| i as u32).find(|&i| i != y0).unwrap();
    let ys: [u32; NY] = [y0, rival];
    let full_out = read_out_fz(m, &full.x_final[focus * d..(focus + 1) * d], &ys, &p_ref, ref_plp, fz.as_ref().and_then(|f| f.fin.get(focus).copied()))[0];
    let mut b = Builder { m, means: None, w: RefCell::new(HashMap::new()), col_norms: RefCell::new(HashMap::new()) };
    let refs: Vec<Vec<u32>> = if o.refs.is_empty() { vec![ids.to_vec()] } else { o.refs.clone() };
    let (means, ref_pos) = if o.mean {
        let (mn, pos) = compute_means(m, &g, &b, &refs, ids, !o.refs.is_empty());
        (Some(Arc::new(mn)), pos)
    } else {
        (None, Vec::new())
    };
    b.means = means.clone();
    let state = State { cells: (0..t).map(|_| (0..nl).map(|_| CellB::default()).collect()).collect() };
    let built: RowCuts = (0..t).map(|_| vec![None; nl]).collect();
    let mut e = Engine {
        m,
        ids,
        g,
        b,
        ys,
        top_ref,
        p_ref,
        ref_plp,
        state,
        built,
        base: Base { out: full_out, ck: Checkpoint { x: vec![], kv: vec![] }, logits: vec![] },
        st: Stats { dead_identical: true, ..Default::default() },
        max_tests: o.max_tests,
        t0: Instant::now(),
        tty: std::io::stderr().is_terminal(),
        fz,
        means,
    };
    let (base, _, _) = e.run_base(false);
    // check: an empty cut changes nothing
    let empty_identical = base.logits.iter().zip(&full.logits).all(|(a, b)| a.to_bits() == b.to_bits());
    e.base = base;
    let total_atoms: u64 = (0..=focus).map(|p| (0..nl).map(|l| e.g.cell_atoms(l, p)).sum::<u64>()).sum();
    if let Some(seed) = o.random {
        random_cut(&mut e, seed);
        let (base, _, _) = e.run_base(false);
        e.base = base;
        let f = o.dump.as_ref().ok_or("--circuit-random needs --dump-circuit FILE")?;
        let r = Report { e: &e, refs: &refs, ref_pos: &ref_pos, o, ids, focus, full_out, circuit_out: e.base.out, comp_out: e.base.out, total_atoms, logs: &[], leaves: &[], empty_identical };
        std::fs::write(f, dump_json(&r, &full.logits)).map_err(|x| format!("{}: {x}", f.display()))?;
        println!("random cut (seed {seed}): {} of {} atoms on, KL {} · empty cut bit-identical: {}", big(e.atoms_kept()), big(total_atoms), num(e.base.out.kl, 4), yes(empty_identical));
        return Ok(());
    }
    eprintln!(
        "circuit search: {} atoms (scalar products) at {} positions × {} layers; criterion KL ≤ {} and top token {}; norms {}; {}",
        big(total_atoms),
        t,
        nl,
        num(o.kl, 3),
        tok(m, top_ref),
        if o.frozen { "frozen" } else { "live" },
        if o.mean { format!("mean ablation over {} reference positions", ref_pos.iter().map(|p| p.len()).sum::<usize>()) } else { "zero ablation".into() }
    );
    // level 1: every component at every position
    let mut comps: Vec<Piece> = Vec::new();
    for l in 0..nl {
        for p in 0..=focus {
            comps.push(Piece::Attn { l, p });
            if e.g.layers[l].own_kv {
                comps.push(Piece::Kv { l, p });
            }
            comps.push(Piece::Mlp { l, p });
            if e.g.layers[l].dp.is_some() {
                comps.push(Piece::Ple { l, p });
            }
        }
    }
    let n_comps = comps.len();
    let groups = vec![Group { kind: GK::List(comps), ids: (0..n_comps as u32).collect(), proxy: vec![0.0; n_comps] }];
    let mut logs: Vec<LevelLog> = Vec::new();
    let depth = o.depth.clamp(1, 4);
    let mut kept: Vec<Piece>;
    {
        let (t0, p0, s0) = (e.st.tests, e.st.passes, Instant::now());
        let (k, un) = e.refine(&groups, o.kl * FRAC[0], false, false, "level 1");
        e.prune_dead();
        kept = k.into_iter().filter(|pc| !e.piece_off(pc)).collect();
        e.tighten(&mut kept, o.kl * FRAC[0], Some(1), "level 1 recheck");
        logs.push(LevelLog { level: 1, groups: 1, pieces_in: n_comps, kept: kept.len(), unresolved: un, atoms: e.atoms_kept(), kl: e.kl(), tests: e.st.tests - t0, passes: e.st.passes - p0, secs: s0.elapsed().as_secs_f32() });
        report_level(logs.last().unwrap(), total_atoms, st);
    }
    for level in 2..=depth {
        let (t0, p0, s0) = (e.st.tests, e.st.passes, Instant::now());
        let (_, inp, x_tok) = e.run_base(true);
        e.st.passes += 1;
        let (groups, pass) = e.expand(&kept, level, &inp, &x_tok);
        drop(inp);
        let n_in: usize = groups.iter().map(|g| g.ids.len()).sum();
        // columns of column-wise sites needed at level 4: fetch per matrix in one pass
        if level == 4 {
            let mut need: BTreeMap<(usize, Site), Vec<u32>> = BTreeMap::new();
            for gr in &groups {
                if let GK::Atoms { l, s, u, .. } = gr.kind {
                    need.entry((l, s)).or_default().push(u as u32);
                }
            }
            for ((l, s), us) in need {
                e.b.prefetch(l, s, &us);
            }
        }
        let (k, un) = e.refine(&groups, o.kl * FRAC[level - 1], true, false, &format!("level {level}"));
        e.prune_dead();
        kept = k.into_iter().chain(pass).filter(|pc| !e.piece_off(pc)).collect();
        e.tighten(&mut kept, o.kl * FRAC[level - 1], Some(1), &format!("level {level} recheck"));
        logs.push(LevelLog { level, groups: groups.len(), pieces_in: n_in, kept: kept.len(), unresolved: un, atoms: e.atoms_kept(), kl: e.kl(), tests: e.st.tests - t0, passes: e.st.passes - p0, secs: s0.elapsed().as_secs_f32() });
        report_level(logs.last().unwrap(), total_atoms, st);
    }
    // final phase (full ε): blocks that stayed whole are split down to single elements, then every leaf
    // alone against the final circuit; whatever can still go, goes
    let mut leaves: Vec<Leaf>;
    {
        let (t0, p0, s0) = (e.st.tests, e.st.passes, Instant::now());
        let atoms0 = e.atoms_kept();
        let (singles, blocks): (Vec<Piece>, Vec<Piece>) = kept.drain(..).partition(|p| p.len() <= 1);
        if !blocks.is_empty() {
            let (_, inp, x_tok) = e.run_base(true);
            e.st.passes += 1;
            let groups: Vec<Group> = blocks.iter().map(|b| e.block_group(b, &inp, &x_tok)).collect();
            drop(inp);
            let mut need: BTreeMap<(usize, Site), Vec<u32>> = BTreeMap::new();
            for gr in &groups {
                if let GK::Atoms { l, s, u, .. } = gr.kind {
                    need.entry((l, s)).or_default().push(u as u32);
                }
            }
            for ((l, s), us) in need {
                e.b.prefetch(l, s, &us);
            }
            let (k, _) = e.refine(&groups, o.kl, true, true, "resolving blocks");
            e.prune_dead();
            kept = singles.into_iter().chain(k).filter(|pc| !e.piece_off(pc)).collect();
        } else {
            kept = singles;
        }
        let (_, mut outs) = e.tighten(&mut kept, o.kl, None, "minimality");
        let removed_final = atoms0.saturating_sub(e.atoms_kept()) as usize;
        if outs.len() != kept.len() {
            let cands: Vec<Vec<Piece>> = kept.iter().map(|p| vec![p.clone()]).collect();
            outs = e.eval(&cands, "minimality");
        }
        leaves = kept.iter().cloned().zip(outs).map(|(piece, out)| Leaf { piece, out, value: None }).collect();
        logs.push(LevelLog { level: 5, groups: 1, pieces_in: removed_final, kept: kept.len(), unresolved: 0, atoms: e.atoms_kept(), kl: e.kl(), tests: e.st.tests - t0, passes: e.st.passes - p0, secs: s0.elapsed().as_secs_f32() });
        report_level(logs.last().unwrap(), total_atoms, st);
    }
    // values of the atoms (for the report): one recording pass of the circuit
    let (_, inp, x_tok) = e.run_base(true);
    for lf in leaves.iter_mut() {
        lf.value = atom_value(&e, &lf.piece, &inp, &x_tok);
    }
    drop(inp);
    leaves.sort_by(|a, b| b.out.kl.partial_cmp(&a.out.kl).unwrap_or(std::cmp::Ordering::Equal));
    // verification: the circuit alone, and everything except the circuit
    let circuit_out = e.base.out;
    let comp = complement(&e);
    let comp_out = {
        let tr = m.run(ids, focus, &e.opts(comp));
        e.read(&tr.x_final[focus * d..(focus + 1) * d])[0]
    };
    e.st.secs = t_all.elapsed().as_secs_f32();
    let r = Report { e: &e, refs: &refs, ref_pos: &ref_pos, o, ids, focus, full_out, circuit_out, comp_out, total_atoms, logs: &logs, leaves: &leaves, empty_identical };
    terminal(&r, st);
    std::fs::write(&o.html, html(&r)).map_err(|x| format!("{}: {x}", o.html.display()))?;
    println!("{}", st.dim(&format!("HTML: {}", o.html.display())));
    if let Some(f) = &o.dump {
        std::fs::write(f, dump_json(&r, &full.logits)).map_err(|x| format!("{}: {x}", f.display()))?;
        println!("{}", st.dim(&format!("JSON: {}", f.display())));
    }
    Ok(())
}

/// A random cut over every kind of atom (test mode): whole sites, single units, single atoms, keys and
/// dimensions of QKᵀ and A·V. xorshift64*, so the same seed gives the same cut.
fn random_cut(e: &mut Engine, seed: u64) {
    let mut st = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut rnd = move || -> f64 {
        st ^= st >> 12;
        st ^= st << 25;
        st ^= st >> 27;
        (st.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
    };
    let g = &e.g;
    let mut all: BTreeMap<(usize, usize), Vec<Op>> = BTreeMap::new();
    for p in 0..=g.focus {
        for (l, lg) in g.layers.iter().enumerate() {
            let mut v = Vec::new();
            for s in ALL_SITES {
                if !g.has(l, s) {
                    continue;
                }
                let r = rnd();
                if r < 0.03 {
                    v.push(Op::All(s));
                    continue;
                }
                let (p_off, p_part) = if r < 0.35 { (0.15, 0.0) } else if r < 0.7 { (0.0, 0.2) } else if r < 0.85 { (0.1, 0.1) } else { (0.0, 0.0) };
                for u in 0..g.units(l, s) as u32 {
                    let x = rnd();
                    if x < p_off {
                        v.push(Op::Unit(s, u));
                    } else if x < p_off + p_part {
                        let a: Vec<u32> = (0..g.d as u32).filter(|_| rnd() < 0.5).collect();
                        if !a.is_empty() {
                            v.push(Op::Atoms(s, u, Arc::new(a)));
                        }
                    }
                }
            }
            for h in 0..lg.nh {
                for av in [false, true] {
                    for j in g.keys(l, p) {
                        let x = rnd();
                        if x < 0.12 {
                            v.push(Op::Key(h, av, j as u32));
                        } else if x < 0.3 {
                            let e2: Vec<u32> = (0..lg.hd as u32).filter(|_| rnd() < 0.5).collect();
                            if !e2.is_empty() {
                                v.push(Op::KeyDims(h, av, j as u32, Arc::new(e2)));
                            }
                        }
                    }
                }
            }
            if !v.is_empty() {
                all.insert((p, l), v);
            }
        }
    }
    for (&(p, l), v) in &all {
        for op in v {
            e.state.cells[p][l].apply(op, &e.g, l);
        }
    }
    // columns of the column-wise sites in one pass per matrix
    let mut need: BTreeMap<(usize, Site), Vec<u32>> = BTreeMap::new();
    for &(p, l) in all.keys() {
        for s in [Site::O, Site::Down, Site::PleProj] {
            need.entry((l, s)).or_default().extend(e.state.cells[p][l].sites[s as usize].part.keys());
        }
    }
    for ((l, s), us) in need {
        e.b.prefetch(l, s, &us);
    }
    let keys: Vec<(usize, usize)> = all.keys().copied().collect();
    e.rebuild(keys.into_iter());
}

/// value of a single atom in the circuit pass (w·x)
fn atom_value(e: &Engine, pc: &Piece, inp: &[LayerInputs], x_tok: &[f32]) -> Option<f32> {
    let d = e.g.d;
    match pc {
        Piece::Atoms { l, p, s, u, a } if a.len() == 1 => {
            let w = e.b.weights(*l, *s, *u as u32);
            let x = site_input(&inp[*l], x_tok, *s, *p, d);
            let a = a[0] as usize;
            Some(if s.row_wise() { w[a] * x[a] } else { w[a] * x[*u] })
        }
        Piece::KeyDims { l, p, h, j, av, e: dims } if dims.len() == 1 => {
            let lg = &e.g.layers[*l];
            let li = &inp[*l];
            let (nh, hd) = (lg.nh, lg.hd);
            let gr = h / (nh / lg.nkv);
            let (ad, kvd, t) = (nh * hd, lg.nkv * hd, e.g.t);
            let x = dims[0] as usize;
            Some(if *av { li.probs[(p * nh + h) * t + j] * li.v[j * kvd + gr * hd + x] } else { li.q[p * ad + h * hd + x] * li.k[j * kvd + gr * hd + x] })
        }
        _ => None,
    }
}

/// The cut that switches off exactly the atoms of the circuit and leaves everything else on
fn complement(e: &Engine) -> RowCuts {
    let g = &e.g;
    let mut rows: RowCuts = Vec::new();
    for p in 0..g.t {
        let mut layers = Vec::new();
        for (l, lg) in g.layers.iter().enumerate() {
            let c = &e.state.cells[p][l];
            let mut inv = CellB { heads: vec![HeadB::default(); lg.nh], ..Default::default() };
            for s in ALL_SITES {
                if !g.has(l, s) {
                    continue;
                }
                let sb = &c.sites[s as usize];
                if sb.all {
                    continue;
                }
                let iv = &mut inv.sites[s as usize];
                for u in 0..g.units(l, s) as u32 {
                    if sb.off.contains(&u) {
                        continue;
                    }
                    match sb.part.get(&u) {
                        Some(a) => {
                            let on: Vec<u32> = (0..g.d as u32).filter(|x| !a.contains(x)).collect();
                            iv.add_atoms(u, &on, g.d);
                        }
                        None => iv.set_off(u),
                    }
                }
            }
            let keys = g.keys(l, p);
            for h in 0..lg.nh {
                let hb = c.heads.get(h).cloned().unwrap_or_default();
                let HeadB { qk, av } = &mut inv.heads[h];
                for (src, dst) in [(&hb.qk, qk), (&hb.av, av)] {
                    for j in keys.clone().map(|j| j as u32) {
                        if src.off.contains(&j) {
                            continue;
                        }
                        match src.part.get(&j) {
                            Some(a) => {
                                let on: Vec<u32> = (0..lg.hd as u32).filter(|x| !a.contains(x)).collect();
                                dst.add_atoms(j, &on, lg.hd);
                            }
                            None => dst.set_off(j),
                        }
                    }
                }
            }
            layers.push(e.b.cell(&inv, l, p, g));
        }
        rows.push(layers);
    }
    rows
}

// ====================================================================== report

struct Report<'a, 'b> {
    e: &'a Engine<'b>,
    refs: &'a [Vec<u32>],
    ref_pos: &'a [Vec<usize>],
    o: &'a Opts,
    ids: &'a [u32],
    focus: usize,
    full_out: Out,
    circuit_out: Out,
    comp_out: Out,
    total_atoms: u64,
    logs: &'a [LevelLog],
    leaves: &'a [Leaf],
    empty_identical: bool,
}

/// 1234567 -> "1,234,567"; beyond 10⁹ as "1.2·10⁹"
fn big(n: u64) -> String {
    if n < 1_000_000_000 {
        thousands(n as usize)
    } else {
        let e = (n as f64).log10().floor() as i32;
        format!("{}·10{}", num((n as f64 / 10f64.powi(e)) as f32, 2), crate::fmt::superscript(e))
    }
}

fn share(a: u64, b: u64) -> String {
    let x = a as f64 / b.max(1) as f64 * 100.0;
    if x >= 0.1 {
        format!("{}%", num(x as f32, 1))
    } else {
        format!("{}%", crate::fmt::sci(x as f32))
    }
}

fn level_name(level: usize) -> String {
    if level <= 4 { format!("level {level} ({})", LEVELS[level - 1]) } else { "final (blocks split, every leaf alone against the final circuit)".into() }
}

fn report_level(lg: &LevelLog, total: u64, st: &Style) {
    let what = if lg.level <= 4 {
        format!("{} pieces in {} groups → {} kept", thousands(lg.pieces_in), thousands(lg.groups), thousands(lg.kept))
    } else {
        format!("blocks split to single elements, {} more atoms off → {} leaves", thousands(lg.pieces_in), thousands(lg.kept))
    };
    println!(
        "{}  {what} · atoms on {} ({}) · KL {} · {} tests in {} passes, {} s{}",
        st.bold(&level_name(lg.level)),
        big(lg.atoms),
        share(lg.atoms, total),
        num(lg.kl, 4),
        thousands(lg.tests),
        thousands(lg.passes),
        num(lg.secs, 0),
        if lg.unresolved > 0 { format!(" · {} blocks kept whole (splitting further removed nothing, or no budget/tests left)", lg.unresolved) } else { String::new() }
    );
}

fn out_txt(m: &Model, o: &Out, ys: &[u32; NY]) -> String {
    format!("top {} {} · log-odds {} {} · KL {}", tok(m, o.top[0].0), pct(o.top[0].1), tok(m, ys[0]), num(o.lo[0], 2), num(o.kl, 4))
}

/// per layer: atoms on by kind (attention: Q K V O QKᵀ A·V; MLP: gate up down; PLE)
#[allow(clippy::type_complexity)]
fn layer_rows(e: &Engine) -> Vec<(usize, [u64; 4], [u64; 4], Vec<String>)> {
    let g = &e.g;
    let mut out = Vec::new();
    for (l, lg) in g.layers.iter().enumerate() {
        let mut on = [0u64; 4]; // attn, kv, mlp, ple
        let mut tot = [0u64; 4];
        let mut units: Vec<String> = Vec::new();
        for p in 0..=g.focus {
            let k = e.cell_kept(p, l);
            on[0] += k[Site::Q as usize] + k[Site::O as usize] + k[N_SITES] + k[N_SITES + 1];
            on[1] += k[Site::K as usize] + k[Site::V as usize];
            on[2] += k[Site::Gate as usize] + k[Site::Up as usize] + k[Site::Down as usize];
            on[3] += k[Site::PleIn as usize] + k[Site::PleGate as usize] + k[Site::PleProj as usize];
            let nk = g.keys(l, p).len() as u64;
            tot[0] += (2 * lg.nh * lg.hd * g.d) as u64 + 2 * lg.nh as u64 * nk * lg.hd as u64;
            tot[1] += [Site::K, Site::V].iter().filter(|&&s| g.has(l, s)).map(|&s| (g.units(l, s) * g.d) as u64).sum::<u64>();
            tot[2] += [Site::Gate, Site::Up, Site::Down].iter().filter(|&&s| g.has(l, s)).map(|&s| (g.units(l, s) * g.d) as u64).sum::<u64>();
            tot[3] += [Site::PleIn, Site::PleGate, Site::PleProj].iter().filter(|&&s| g.has(l, s)).map(|&s| (g.units(l, s) * g.d) as u64).sum::<u64>();
            // which heads / neurons are still alive here
            let c = &e.state.cells[p][l];
            for h in 0..lg.nh {
                if (h * lg.hd..(h + 1) * lg.hd).any(|u| !c.sites[Site::O as usize].unit_off(u as u32)) {
                    units.push(format!("head {h} [{p}]"));
                }
            }
            let ds = &c.sites[Site::Down as usize];
            if !ds.all {
                let alive = lg.ff - ds.off.len();
                if alive > 0 && alive <= 12 {
                    for n in 0..lg.ff as u32 {
                        if !ds.unit_off(n) {
                            units.push(format!("#{n} [{p}]"));
                        }
                    }
                } else if alive > 12 {
                    units.push(format!("{alive} neurons [{p}]"));
                }
            }
        }
        out.push((l, on, tot, units));
    }
    out
}

fn terminal(r: &Report, st: &Style) {
    let e = r.e;
    let m = e.m;
    let ys = &e.ys;
    let kept = e.atoms_kept();
    println!();
    println!("{}", st.bold(&format!("Circuit for {} at token [{}]", tok(m, e.top_ref), r.focus)));
    println!("  criterion: KL(full ‖ circuit) ≤ {} nats and the same top token", num(r.o.kl, 3));
    println!("  {}", norms_txt(r.o.frozen));
    println!("  {}", ablation_txt(r));
    println!("  {} of {} atoms stay on ({}) · {} leaves", st.bold(&big(kept)), big(r.total_atoms), share(kept, r.total_atoms), thousands(r.leaves.len()));
    println!("  full model:            {}", out_txt(m, &r.full_out, ys));
    println!("  circuit only:          {}", out_txt(m, &r.circuit_out, ys));
    println!("  everything but it:     {}", out_txt(m, &r.comp_out, ys));
    println!();
    println!("{}", st.bold("Atoms on per layer (attention = Q, O, QKᵀ, A·V · K/V · MLP = gate, up, down · PLE)"));
    for (l, on, tot, units) in layer_rows(e) {
        if on.iter().all(|&x| x == 0) {
            continue;
        }
        let part = |i: usize, name: &str| if tot[i] > 0 && on[i] > 0 { format!("{name} {}", big(on[i])) } else { String::new() };
        let cols: Vec<String> = [part(0, "attn"), part(1, "K/V"), part(2, "MLP"), part(3, "PLE")].into_iter().filter(|s| !s.is_empty()).collect();
        let u = if units.len() > 8 { format!("{} … (+{})", units[..8].join(", "), units.len() - 8) } else { units.join(", ") };
        println!("  S{l:<3} {:<44} {}", cols.join(" · "), st.dim(&u));
    }
    println!();
    println!("{}", st.bold(&format!("The {} most necessary leaves (each switched off alone, everything else as in the circuit)", r.o.show.min(r.leaves.len()))));
    for lf in r.leaves.iter().take(r.o.show) {
        let v = lf.value.map_or(String::new(), |v| format!(" = {}", num(v, 4)));
        println!("  {:<58} → {} {} · KL {}", format!("{}{v}", lf.piece.name(m)), tok(m, lf.out.top[0].0), pct(lf.out.top[0].1), num(lf.out.kl, 3));
    }
    println!();
    let s = &e.st;
    println!(
        "{}",
        st.dim(&format!(
            "Checks: empty cut = normal pass (bit-identical: {}) · batch vs. standalone rerun of accepted steps |ΔKL| ≤ {} · clean rows in the batches vs. base |ΔKL| ≤ {} · {} atoms without effect switched off, output bit-identical: {} · {} commits, {} refused by the standalone rerun · {} tests in {} passes, {} s{}",
            yes(r.empty_identical),
            sci(s.batch_dev),
            sci(s.clean_dev),
            big(s.dead_atoms),
            yes(s.dead_identical),
            s.commits,
            s.refused,
            thousands(s.tests),
            thousands(s.passes),
            num(s.secs, 0),
            if s.stopped { " · stopped at the test budget: unresolved blocks stay whole in the circuit" } else { "" }
        ))
    );
}

fn ablation_txt(r: &Report) -> String {
    if r.o.mean {
        format!(
            "mean ablation: the activations in a switched-off atom are replaced by their means over {} (<bos> stays itself)",
            if r.o.refs.is_empty() { "all positions of the analyzed sequence (--circuit-ref gives reference prompts)".to_string() } else { format!("{} reference prompts, position by position", r.refs.len()) }
        )
    } else {
        "zero ablation: a switched-off atom is missing from its sum".into()
    }
}

fn norms_txt(frozen: bool) -> &'static str {
    if frozen {
        "norms frozen: every norm keeps the scale it has in the full model at that position (nothing left over is inflated)"
    } else {
        "norms live: every norm recomputes its scale from what is left (a post-norm inflates a small rest to full size)"
    }
}

fn yes(b: bool) -> &'static str {
    if b { "yes" } else { "NO" }
}

const CIRCUIT_CSS: &str = r#"
.grid{border-collapse:collapse;font-size:11px;font-variant-numeric:tabular-nums}
.grid td,.grid th{padding:1px 3px;text-align:right;border:1px solid var(--line)}
.grid th{color:var(--mu);font-weight:600}
.grid td.z{color:var(--line)}
.lvl td{padding:2px 10px 2px 0}
.leaf td{padding:1px 10px 1px 0;font-size:12.5px}
code{font-family:ui-monospace,Menlo,monospace;font-size:12.5px}
"#;

fn html(r: &Report) -> String {
    let e = r.e;
    let m = e.m;
    let g = &e.g;
    let ys = &e.ys;
    let kept = e.atoms_kept();
    let mut h = String::with_capacity(1 << 20);
    h += &format!(
        r#"<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Circuit search</title><style>{CSS}{CIRCUIT_CSS}</style></head><body><main>
<h1>Circuit down to single scalar products</h1>
<p class="meta">{} · {} layers · focus token [{}] · criterion KL ≤ {} nats and top token {}</p>
<p class="prompt">{}</p>
<p class="mu">{}<br>{}</p>
<p><b>{}</b> of {} atoms stay on ({}), {} leaves.</p>
<table class="tbl"><tr><th></th><th>top</th><th>log-odds {}</th><th>KL</th></tr>
<tr><td>full model</td><td>{} {}</td><td class="num">{}</td><td class="num">0</td></tr>
<tr><td>circuit only</td><td>{} {}</td><td class="num">{}</td><td class="num">{}</td></tr>
<tr><td>everything except the circuit</td><td>{} {}</td><td class="num">{}</td><td class="num">{}</td></tr></table>
"#,
        esc_html(&m.name),
        m.n_layer(),
        r.focus,
        num(r.o.kl, 3),
        chip(m, e.top_ref),
        prompt_html(m, r.ids, r.focus),
        esc_html(norms_txt(r.o.frozen)),
        esc_html(&ablation_txt(r)),
        big(kept),
        big(r.total_atoms),
        share(kept, r.total_atoms),
        thousands(r.leaves.len()),
        chip(m, ys[0]),
        chip(m, r.full_out.top[0].0),
        pct(r.full_out.top[0].1),
        num(r.full_out.lo[0], 2),
        chip(m, r.circuit_out.top[0].0),
        pct(r.circuit_out.top[0].1),
        num(r.circuit_out.lo[0], 2),
        num(r.circuit_out.kl, 4),
        chip(m, r.comp_out.top[0].0),
        pct(r.comp_out.top[0].1),
        num(r.comp_out.lo[0], 2),
        num(r.comp_out.kl, 3),
    );
    h += r#"<details class="info"><summary>How to read this</summary><div>
<p>An <b>atom</b> is one scalar product at one position: W[o,c]·x<sub>c</sub> in a projection (Q, K, V, O, gate, up, down, PLE), q<sub>e</sub>·k<sub>e</sub> in a score (QKᵀ) or a<sub>j</sub>·v<sub>j</sub>[e] in the mixing (A·V). Switching an atom off means it is missing from its sum; biases, norms, softmax, activations and RoPE always run on what is left.</p>
<p>The search goes coarse to fine: components at every position, then heads, K/V groups, neurons and PLE dimensions, then q/k/v dimensions, keys and W<sub>O</sub> columns, then single atoms. A piece stays only if switching it off on top of everything already off would break the criterion. Every decision is an exact rerun of the model; magnitudes only choose the order of the tests. Accepted steps are rerun on their own and must pass again. At the end every leaf is switched off alone against the final circuit: all leaves listed below are individually necessary.</p>
<p><b>Everything except the circuit</b> switches off exactly the atoms that the circuit keeps and leaves all others on.</p>
</div></details>
"#;
    h += "<h2>Levels</h2><table class=\"tbl lvl\"><tr><th>level</th><th>pieces</th><th>kept</th><th>atoms on</th><th>KL</th><th>tests</th><th>passes</th><th>s</th></tr>";
    for lg in r.logs {
        h += &format!(
            "<tr><td>{}</td><td class=\"num\">{}</td><td class=\"num\">{}</td><td class=\"num\">{} ({})</td><td class=\"num\">{}</td><td class=\"num\">{}</td><td class=\"num\">{}</td><td class=\"num\">{}</td></tr>",
            esc_html(&level_name(lg.level)),
            thousands(lg.pieces_in),
            thousands(lg.kept),
            big(lg.atoms),
            share(lg.atoms, r.total_atoms),
            num(lg.kl, 4),
            thousands(lg.tests),
            thousands(lg.passes),
            num(lg.secs, 0)
        );
    }
    h += "</table>";
    // grid: layers × positions
    h += "<h2>Atoms on per layer and position</h2><div style=\"overflow-x:auto\"><table class=\"grid\"><tr><th></th>";
    for p in 0..=r.focus {
        h += &format!("<th title=\"{}\">[{p}]<br>{}</th>", esc_html(&esc_piece(&m.tok.piece(r.ids[p]))), esc_html(&crate::fmt::trunc(&esc_piece(&m.tok.piece(r.ids[p])), 6)));
    }
    h += "</tr>";
    let mut mx = 1u64;
    let cells: Vec<Vec<u64>> = (0..g.layers.len()).map(|l| (0..=r.focus).map(|p| e.cell_kept(p, l).iter().sum::<u64>()).collect()).collect();
    for row in &cells {
        for &v in row {
            mx = mx.max(v);
        }
    }
    for (l, row) in cells.iter().enumerate() {
        h += &format!("<tr><th>S{l}</th>");
        for &v in row {
            if v == 0 {
                h += "<td class=\"z\">·</td>";
            } else {
                let a = ((v as f64).ln() / (mx as f64).ln().max(1e-9)).clamp(0.08, 1.0);
                h += &format!("<td style=\"background:rgba(47,95,179,{:.2})\" title=\"{}\">{}</td>", a * 0.85, thousands(v as usize), short(v));
            }
        }
        h += "</tr>";
    }
    h += "</table></div>";
    // per layer
    h += "<h2>Per layer</h2><table class=\"tbl\"><tr><th>layer</th><th>attention</th><th>K/V</th><th>MLP</th><th>PLE</th><th>alive</th></tr>";
    for (l, on, tot, units) in layer_rows(e) {
        let c = |i: usize| if tot[i] == 0 { "–".to_string() } else { format!("{} <span class=\"mu\">/ {}</span>", big(on[i]), big(tot[i])) };
        h += &format!("<tr><td>S{l}</td><td class=\"num\">{}</td><td class=\"num\">{}</td><td class=\"num\">{}</td><td class=\"num\">{}</td><td class=\"mu\">{}</td></tr>", c(0), c(1), c(2), c(3), esc_html(&units.join(", ")));
    }
    h += "</table>";
    // leaves
    h += &format!("<h2>All {} leaves, most necessary first</h2><p class=\"mu\">Each leaf switched off alone, everything else as in the circuit.</p><table class=\"tbl leaf\"><tr><th>leaf</th><th>value</th><th>top without it</th><th>log-odds {}</th><th>KL</th></tr>", thousands(r.leaves.len()), chip(m, ys[0]));
    for lf in r.leaves {
        h += &format!(
            "<tr><td><code>{}</code></td><td class=\"num\">{}</td><td>{} {}</td><td class=\"num\">{}</td><td class=\"num\">{}</td></tr>",
            esc_html(&lf.piece.name(m)),
            lf.value.map_or(String::new(), |v| num(v, 4)),
            chip(m, lf.out.top[0].0),
            pct(lf.out.top[0].1),
            num(lf.out.lo[0], 2),
            num(lf.out.kl, 4)
        );
    }
    h += "</table>";
    let s = &e.st;
    h += &format!(
        "<p class=\"check foot\">Checks: empty cut = normal pass (bit-identical: {}) · batch vs. standalone rerun of accepted steps |ΔKL| ≤ {} · clean rows in the batches vs. base |ΔKL| ≤ {} · {} atoms without effect switched off, output bit-identical: {} · {} commits, {} refused by the standalone rerun · {} tests in {} passes, {} s{}</p></main></body></html>",
        yes(r.empty_identical),
        sci(s.batch_dev),
        sci(s.clean_dev),
        big(s.dead_atoms),
        yes(s.dead_identical),
        s.commits,
        s.refused,
        thousands(s.tests),
        thousands(s.passes),
        num(s.secs, 0),
        if s.stopped { " · stopped at the test budget: unresolved blocks stay whole in the circuit" } else { "" }
    );
    h
}

fn short(v: u64) -> String {
    if v < 1000 {
        v.to_string()
    } else if v < 1_000_000 {
        format!("{}k", v / 1000)
    } else {
        format!("{}M", v / 1_000_000)
    }
}

/// Everything the reference test needs: tokens, the cut cell by cell, logits of the full model, the
/// circuit and its complement.
fn dump_json(r: &Report, full_logits: &[f32]) -> String {
    use serde_json::{json, Map, Value};
    let e = r.e;
    let g = &e.g;
    let set = |s: &SetB| -> Value {
        json!({
            "all": s.all,
            "off": s.off.iter().collect::<Vec<_>>(),
            "part": s.part.iter().map(|(u, a)| (u.to_string(), json!(a.iter().collect::<Vec<_>>()))).collect::<Map<String, Value>>(),
        })
    };
    let mut cells = Vec::new();
    for p in 0..g.t {
        for l in 0..g.layers.len() {
            let c = &e.state.cells[p][l];
            if c.is_empty() {
                continue;
            }
            let sites: Map<String, Value> = ALL_SITES.iter().filter(|&&s| !c.sites[s as usize].is_empty()).map(|&s| (s.name().to_string(), set(&c.sites[s as usize]))).collect();
            let heads: Map<String, Value> =
                c.heads.iter().enumerate().filter(|(_, hb)| !(hb.qk.is_empty() && hb.av.is_empty())).map(|(h, hb)| (h.to_string(), json!({"qk": set(&hb.qk), "av": set(&hb.av)}))).collect();
            cells.push(json!({"p": p, "l": l, "sites": sites, "heads": heads}));
        }
    }
    let circuit = e.m.run(r.ids, r.focus, &e.opts(e.built.clone())).logits;
    let comp = e.m.run(r.ids, r.focus, &e.opts(complement(e))).logits;
    json!({
        "tokens": r.ids,
        "focus": r.focus,
        "kl": r.o.kl,
        "frozen_norms": r.o.frozen,
        "ablation": if r.o.mean { "mean" } else { "zero" },
        "ref_tokens": r.refs,
        "ref_positions": r.ref_pos,
        "positionwise": !r.o.refs.is_empty(),
        "bos": e.m.tok.is_special(r.ids[0]),
        "means": e.means.as_ref().map(|mn| mn.layers.iter().map(|lm| lm.pos.iter().map(|pm| json!({
            "h": pm.h, "z": pm.z, "h2": pm.h2, "act": pm.act, "q": pm.q, "k": pm.k, "v": pm.v, "a": pm.a,
        })).collect::<Vec<_>>()).collect::<Vec<_>>()),
        "top": e.top_ref,
        "cells": cells,
        "logits_full": full_logits,
        "logits_circuit": circuit,
        "logits_complement": comp,
        "circuit_kl": r.circuit_out.kl,
        "atoms_on": e.atoms_kept(),
        "atoms_total": r.total_atoms,
    })
    .to_string()
}
