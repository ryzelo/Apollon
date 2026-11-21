//! The result folder of --circuit: the circuit as data, from coarse to fine, and the exact trace of every
//! scalar product back to the scalar products it reads.
//!
//!   DIR/README.md                       what the files hold
//!   DIR/circuit.json                    overview: outputs, paths of the self-expansion, components, units
//!   DIR/trace.json                      the trace backwards from the output, unit by unit
//!   DIR/layers/S07/layer.json           one layer: its components and units per position
//!   DIR/layers/S07/p09/attention/head2.json   one unit with every scalar product that is on in it
//!
//! Every atom (scalar product) that is on in the union of all paths is listed with its value in the pass of
//! the union, what it becomes when switched off, its path, its effect when switched off alone (against its
//! path and against the union) and its **sources**: the atoms whose values it is made of, each with the
//! exact amount it contributes. The decomposition is exact because, with frozen norms, everything between
//! two atoms is linear in the one factor that is traced:
//!
//! * an atom that reads the residual stream (W[u,c]·x̂_c in Q, K, V, gate, up, PLE gate) is the sum of
//!   what every earlier writer atom (W_O, W_down, W_PLE-proj at the same position) put into dimension c –
//!   times its post-norm, the layer scalars in between and the pre-norm of the reader – plus the
//!   embedding, the switched-off atoms (their means; 0 with zero ablation), the biases and, for
//!   LayerNorm, the mean subtraction;
//! * q_e·k_e: once over the Q atoms (k held fixed) and once over the K atoms (q held fixed), through the
//!   q/k norm and RoPE;
//! * a_j·v_j[e]: over the V atoms (the attention weight held fixed); the scores q·k_j of the same key are
//!   listed as its non-linear sources (softmax);
//! * W_O[o,u]·z_u: over the products a_j·v_j[e] of its head and dimension;
//! * W_down[o,n]·act_n: over the up atoms of neuron n (the gate held fixed); the gate atoms are its
//!   non-linear sources;
//! * the output: the target's logit (before a softcap) over the writer atoms at the focus.
//!
//! Every decomposition is checked: the sum of its parts has to give the recorded value. The largest
//! relative deviation over all of them is the "trace closure" reported in circuit.json.
use super::*;
use crate::model::{act_fn, LayerInputs, Norm, NormScales, Res, Rope};
use std::collections::{HashMap, VecDeque};
use std::path::Path as FsPath;

/// above this many atoms on, only the coarse data is written (units, components, paths)
const MAX_ATOMS: u64 = 250_000;
/// per atom, the strongest sources and readers listed by name (the rest summed up), so that the folder stays
/// small; the checks always use all of them
const MAX_EDGES: usize = 32;
/// unit files with more atoms are written compact instead of indented
const PRETTY_ATOMS: usize = 2000;

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
enum A {
    W { l: usize, p: usize, s: Site, u: u32, a: u32 },
    Qk { l: usize, p: usize, h: usize, j: usize, e: usize },
    Av { l: usize, p: usize, h: usize, j: usize, e: usize },
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
enum Unit {
    Head { l: usize, p: usize, h: usize },
    Kv { l: usize, p: usize, g: usize },
    Neuron { l: usize, p: usize, n: u32 },
    Ple { l: usize, p: usize, e: u32 },
}

impl Unit {
    fn lp(&self) -> (usize, usize) {
        match *self {
            Unit::Head { l, p, .. } | Unit::Kv { l, p, .. } | Unit::Neuron { l, p, .. } | Unit::Ple { l, p, .. } => (l, p),
        }
    }
    fn name(&self) -> String {
        match *self {
            Unit::Head { l, p, h } => format!("S{l} head {h} [{p}]"),
            Unit::Kv { l, p, g } => format!("S{l} K/V group {g} [{p}]"),
            Unit::Neuron { l, p, n } => format!("S{l} #{n} [{p}]"),
            Unit::Ple { l, p, e } => format!("S{l} PLE dim {e} [{p}]"),
        }
    }
    fn kind(&self) -> &'static str {
        match self {
            Unit::Head { .. } => "head",
            Unit::Kv { .. } => "kv group",
            Unit::Neuron { .. } => "neuron",
            Unit::Ple { .. } => "ple dimension",
        }
    }
    fn component(&self) -> &'static str {
        match self {
            Unit::Head { .. } => "attention",
            Unit::Kv { .. } => "kv",
            Unit::Neuron { .. } => "mlp",
            Unit::Ple { .. } => "ple",
        }
    }
    fn file(&self, wl: usize, wp: usize) -> String {
        let (l, p) = self.lp();
        let leaf = match *self {
            Unit::Head { h, .. } => format!("head{h}"),
            Unit::Kv { g, .. } => format!("group{g}"),
            Unit::Neuron { n, .. } => format!("neuron{n}"),
            Unit::Ple { e, .. } => format!("dim{e}"),
        };
        format!("layers/S{l:0wl$}/p{p:0wp$}/{}/{leaf}.json", self.component())
    }
}

/// where in a layer the residual stream is read
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Stage {
    Attn(usize),
    Mlp(usize),
    PleGate(usize),
    Final,
}

/// the residual stream at one stage and position, split into where it comes from
#[derive(Clone, Default)]
struct Snap {
    /// the switched-off atoms of writer sites (means; 0 with zero ablation)
    abl: Vec<f32>,
    bias: Vec<f32>,
    emb: Vec<f32>,
    /// the stream as the model has it there
    x: Vec<f32>,
}

/// one source of an atom: the atom it reads, how much of its value comes from there
struct Src {
    from: usize,
    /// contribution to the atom's value: k · (value of the source)
    c: f32,
    /// the exact coefficient k (everything between the two atoms, the other factors held at their value)
    k: f32,
    via: &'static str,
}

struct Info {
    a: A,
    value: f32,
    ablated: f32,
    sources: Vec<Src>,
    /// sources the value depends on non-linearly (softmax scores, gate pre-activations), with their value
    nonlinear: Vec<(usize, f32, &'static str)>,
    /// the parts that are not atoms: embedding, switched-off atoms, biases, norm terms
    rest: Vec<(&'static str, f32)>,
    /// for writers at the focus: contribution to the target's and the rival's logit
    to_output: Option<(f32, f32)>,
    closure: f32,
}

struct Ctx<'a, 'b> {
    e: &'a Engine<'b>,
    m: &'b Model,
    g: &'a Geo,
    inp: &'a [LayerInputs],
    x_tok: &'a [f32],
    norms: &'a NormScales,
    t: usize,
    d: usize,
    /// worst relative deviation of a check so far, and where
    worst: RefCell<(f32, String)>,
}

impl<'a, 'b> Ctx<'a, 'b> {
    /// relative deviation of a decomposition from the recorded value; the worst one is kept with its label
    fn check(&self, got: f32, want: f32, scale: f32, what: impl FnOnce() -> String) -> f32 {
        let dev = (got - want).abs() / scale.abs().max(want.abs()).max(1e-6);
        let mut w = self.worst.borrow_mut();
        if dev > w.0 {
            *w = (dev, what());
        }
        dev
    }
    fn pm(&self, l: usize, p: usize) -> Option<&PosMeans> {
        self.e.means.as_deref().map(|m| &m.layers[l].pos[p])
    }
    fn cell(&self, l: usize, p: usize) -> &CellB {
        &self.e.state.cells[p][l]
    }
    /// the layer whose K/V the attention of layer l reads
    fn kv_layer(&self, l: usize) -> usize {
        self.m.blocks[l].attn.kv_from.unwrap_or(l)
    }
    fn scalar(&self, l: usize) -> f32 {
        self.m.blocks[l].scalar.unwrap_or(1.0)
    }
    /// product of the layer scalars of layers lw … lr−1
    fn scal(&self, lw: usize, lr: usize) -> f32 {
        (lw..lr).map(|k| self.scalar(k)).product()
    }
    fn stage_layer(&self, s: Stage) -> usize {
        match s {
            Stage::Attn(l) | Stage::Mlp(l) | Stage::PleGate(l) => l,
            Stage::Final => self.g.layers.len(),
        }
    }
    /// can a writer (layer lw, kind 0 attention / 1 MLP / 2 PLE) be read at this stage? Returns the factor of
    /// the layer scalars in between.
    fn visible(&self, st: Stage, lw: usize, kind: u8) -> Option<f32> {
        let lr = self.stage_layer(st);
        let parallel = self.m.blocks.get(lr).is_some_and(|b| b.res == Res::Parallel);
        let ok = match st {
            Stage::Attn(l) => lw < l,
            Stage::Mlp(l) => lw < l || (!parallel && lw == l && kind == 0),
            Stage::PleGate(l) => lw < l || (lw == l && kind <= 1),
            Stage::Final => true,
        };
        ok.then(|| if lw < lr { self.scal(lw, lr) } else { 1.0 })
    }
    /// the post-norm of a writer on residual dimension o (1 without one)
    fn post(&self, lw: usize, kind: u8, p: usize, o: usize) -> f32 {
        let b = &self.m.blocks[lw];
        let ln = &self.norms.layers[lw];
        match kind {
            0 if b.res == Res::Pre => b.n_attn_post.as_ref().map_or(1.0, |n| n.w[o] * ln.attn_post[p]),
            1 if b.res == Res::Pre => b.n_mlp_post.as_ref().map_or(1.0, |n| n.w[o] * ln.mlp_post[p]),
            2 => b.ple.as_ref().map_or(1.0, |pl| pl.norm.w[o] * ln.ple[p]),
            _ => 1.0,
        }
    }
    /// the pre-norm of a stage at position p and its scale (None: the stage reads the raw stream)
    fn pre(&self, st: Stage, p: usize) -> Option<(&'b Norm, f32)> {
        match st {
            Stage::Attn(l) => self.m.blocks[l].n_attn.as_ref().map(|n| (n, self.norms.layers[l].attn[p])),
            Stage::Mlp(l) => self.m.blocks[l].n_mlp.as_ref().map(|n| (n, self.norms.layers[l].mlp[p])),
            Stage::PleGate(_) => None,
            Stage::Final => self.m.final_norm.as_ref().map(|n| (n, self.e.fin().unwrap_or_else(|| self.norms.fin[p]))),
        }
    }
    /// the stage at which a site reads the residual stream
    fn reads(&self, l: usize, s: Site) -> Option<Stage> {
        match s {
            Site::Q | Site::K | Site::V => Some(Stage::Attn(l)),
            Site::Gate | Site::Up => Some(Stage::Mlp(l)),
            Site::PleGate => Some(Stage::PleGate(l)),
            _ => None,
        }
    }
    fn bias(&self, l: usize, s: Site, u: usize) -> f32 {
        let a = &self.m.blocks[l].attn;
        let hd = a.hd;
        let fused = |off: usize| a.bqkv.as_ref().map_or(0.0, |b| b[(u / hd) * 3 * hd + off + u % hd]);
        match s {
            Site::Q => a.bq.as_ref().map_or_else(|| fused(0), |b| b[u]),
            Site::K => a.bk.as_ref().map_or_else(|| fused(hd), |b| b[u]),
            Site::V => a.bv.as_ref().map_or_else(|| fused(2 * hd), |b| b[u]),
            Site::Up => self.m.blocks[l].mlp.bup.as_ref().map_or(0.0, |b| b[u]),
            _ => 0.0,
        }
    }
    /// row-wise site: the switched-off atoms of unit u at (l, p), summed (with their means; 0 with zero ablation)
    fn row_ablated(&self, l: usize, p: usize, s: Site, u: u32) -> f32 {
        let Some(pm) = self.pm(l, p) else { return 0.0 };
        let sb = &self.cell(l, p).sites[s as usize];
        if sb.unit_off(u) {
            return pm.y[s as usize][u as usize];
        }
        match sb.part.get(&u) {
            Some(off) => {
                let w = self.e.b.weights(l, s, u);
                let xb = pm.x(s);
                off.iter().map(|&a| w[a as usize] * xb[a as usize]).sum()
            }
            None => 0.0,
        }
    }
    fn value(&self, a: &A) -> f32 {
        let d = self.d;
        match *a {
            A::W { l, p, s, u, a } => {
                let w = self.e.b.weights(l, s, u);
                let x = site_input(&self.inp[l], self.x_tok, s, p, d);
                if s.row_wise() { w[a as usize] * x[a as usize] } else { w[a as usize] * x[u as usize] }
            }
            A::Qk { l, p, h, j, e } => {
                let lg = &self.g.layers[l];
                let (ad, kvd, gr) = (lg.nh * lg.hd, lg.nkv * lg.hd, h / (lg.nh / lg.nkv));
                self.inp[l].q[p * ad + h * lg.hd + e] * self.inp[l].k[j * kvd + gr * lg.hd + e]
            }
            A::Av { l, p, h, j, e } => {
                let lg = &self.g.layers[l];
                let (kvd, gr) = (lg.nkv * lg.hd, h / (lg.nh / lg.nkv));
                self.inp[l].probs[(p * lg.nh + h) * self.t + j] * self.inp[l].v[j * kvd + gr * lg.hd + e]
            }
        }
    }
    /// what the atom becomes when it is switched off
    fn ablated(&self, a: &A) -> f32 {
        match *a {
            A::W { l, p, s, u, a } => self.pm(l, p).map_or(0.0, |pm| {
                let w = self.e.b.weights(l, s, u);
                let xb = pm.x(s);
                if s.row_wise() { w[a as usize] * xb[a as usize] } else { w[a as usize] * xb[u as usize] }
            }),
            A::Qk { l, p, h, j, e } => match (self.pm(l, p), self.pm(l, j)) {
                (Some(pq), Some(pk)) => {
                    let lg = &self.g.layers[l];
                    let gr = h / (lg.nh / lg.nkv);
                    pq.q[h * lg.hd + e] * pk.k[gr * lg.hd + e]
                }
                _ => 0.0,
            },
            A::Av { l, p, h, j, e } => match (self.pm(l, p), self.pm(l, j)) {
                (Some(pq), Some(pk)) => {
                    let lg = &self.g.layers[l];
                    let gr = h / (lg.nh / lg.nkv);
                    pq.a[h * pq.a.len() / lg.nh + j] * pk.v[gr * lg.hd + e]
                }
                _ => 0.0,
            },
        }
    }
}

impl A {
    fn unit(&self, g: &Geo) -> Unit {
        match *self {
            A::W { l, p, s, u, .. } => {
                let hd = g.layers[l].hd as u32;
                match s {
                    Site::Q | Site::O => Unit::Head { l, p, h: (u / hd) as usize },
                    Site::K | Site::V => Unit::Kv { l, p, g: (u / hd) as usize },
                    Site::Gate | Site::Up | Site::Down => Unit::Neuron { l, p, n: u },
                    Site::PleIn | Site::PleGate | Site::PleProj => Unit::Ple { l, p, e: u },
                }
            }
            A::Qk { l, p, h, .. } | A::Av { l, p, h, .. } => Unit::Head { l, p, h },
        }
    }
    fn name(&self) -> String {
        match *self {
            A::W { l, p, s, u, a } => {
                if s.row_wise() {
                    format!("S{l}[{p}] W_{}[{u},{a}]", s.name())
                } else {
                    format!("S{l}[{p}] W_{}[{a},{u}]", s.name())
                }
            }
            A::Qk { l, p, h, j, e } => format!("S{l}[{p}] h{h} QK key[{j}] e{e}"),
            A::Av { l, p, h, j, e } => format!("S{l}[{p}] h{h} AV key[{j}] e{e}"),
        }
    }
    fn fields(&self) -> Value {
        match *self {
            A::W { l, p, s, u, a } => {
                let (row, col) = if s.row_wise() { (u, a) } else { (a, u) };
                json!({"type": "matrix", "layer": l, "position": p, "matrix": s.name(), "row": row, "column": col,
                    "reads": if s.row_wise() { format!("input dimension {a}") } else { format!("input unit {u}") },
                    "writes": if s.row_wise() { format!("unit {u}") } else { format!("output dimension {a}") }})
            }
            A::Qk { l, p, h, j, e } => json!({"type": "score q·k", "layer": l, "position": p, "head": h, "key": j, "dim": e}),
            A::Av { l, p, h, j, e } => json!({"type": "mixing a·v", "layer": l, "position": p, "head": h, "key": j, "dim": e}),
        }
    }
}

/// act(x)/x, the factor that makes an activation linear at the point x (its limit at 0)
fn ratio(f: fn(f32) -> f32, x: f32) -> f32 {
    if x.abs() > 1e-12 {
        f(x) / x
    } else {
        f(1e-4) / 1e-4
    }
}

/// RoPE as a matrix: the coefficients R[e][e'] of row e (after the rotation) at position pos
fn rope_row(r: Option<&Rope>, pos: usize, e: usize) -> Vec<(usize, f32)> {
    let Some(r) = r else { return vec![(e, 1.0)] };
    let half = r.inv_freq.len();
    let pf = pos as f32 / r.pos_scale;
    let (i, first, partner) = if r.neox {
        if e < half {
            (e, true, e + half)
        } else if e < 2 * half {
            (e - half, false, e - half)
        } else {
            return vec![(e, 1.0)];
        }
    } else if e < 2 * half {
        (e / 2, e % 2 == 0, if e % 2 == 0 { e + 1 } else { e - 1 })
    } else {
        return vec![(e, 1.0)];
    };
    let (s, c) = (pf * r.inv_freq[i]).sin_cos();
    // v'[i0] = x0·c − x1·s, v'[i1] = x1·c + x0·s
    vec![(e, c), (partner, if first { -s } else { s })]
}

/// every atom that is on, as a piece of its own (for --circuit-from)
pub(super) fn atom_pieces(e: &Engine) -> Vec<Piece> {
    on_atoms(e)
        .into_iter()
        .map(|a| match a {
            A::W { l, p, s, u, a } => Piece::Atoms { l, p, s, u: u as usize, a: Arc::new(vec![a]) },
            A::Qk { l, p, h, j, e } => Piece::KeyDims { l, p, h, j, av: false, e: Arc::new(vec![e as u32]) },
            A::Av { l, p, h, j, e } => Piece::KeyDims { l, p, h, j, av: true, e: Arc::new(vec![e as u32]) },
        })
        .collect()
}

/// the atoms that are on in the engine's state
fn on_atoms(e: &Engine) -> Vec<A> {
    let g = &e.g;
    let mut v = Vec::new();
    for p in 0..=g.focus {
        for (l, lg) in g.layers.iter().enumerate() {
            let c = &e.state.cells[p][l];
            for s in ALL_SITES {
                if !g.has(l, s) {
                    continue;
                }
                let sb = &c.sites[s as usize];
                if sb.all {
                    continue;
                }
                for u in 0..g.units(l, s) as u32 {
                    if sb.off.contains(&u) {
                        continue;
                    }
                    let part = sb.part.get(&u);
                    for a in 0..g.d as u32 {
                        if part.is_none_or(|x| !x.contains(&a)) {
                            v.push(A::W { l, p, s, u, a });
                        }
                    }
                }
            }
            for h in 0..lg.nh {
                let hb = c.heads.get(h);
                for j in g.keys(l, p) {
                    for (av, set) in [(false, hb.map(|x| &x.qk)), (true, hb.map(|x| &x.av))] {
                        if set.is_some_and(|s| s.off.contains(&(j as u32))) {
                            continue;
                        }
                        let part = set.and_then(|s| s.part.get(&(j as u32)));
                        for x in 0..lg.hd {
                            if part.is_none_or(|y| !y.contains(&(x as u32))) {
                                v.push(if av { A::Av { l, p, h, j, e: x } } else { A::Qk { l, p, h, j, e: x } });
                            }
                        }
                    }
                }
            }
        }
    }
    v
}

/// The leaves as a lookup: the finest piece that holds an atom
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum LK {
    W(usize, usize, Site, u32, u32),
    Unit(usize, usize, Site, u32),
    KeyDim(usize, usize, usize, usize, bool, usize),
    Key(usize, usize, usize, bool, usize),
    Head(usize, usize, usize),
    Group(usize, usize, usize),
    Neuron(usize, usize, u32),
    Dim(usize, usize, u32),
    Comp(usize, usize, u8),
}

fn leaf_keys(pc: &Piece, g: &Geo) -> Vec<LK> {
    let _ = g;
    match pc {
        Piece::Atoms { l, p, s, u, a } => a.iter().map(|&x| LK::W(*l, *p, *s, *u as u32, x)).collect(),
        Piece::Units { l, p, s, u } => u.iter().map(|&x| LK::Unit(*l, *p, *s, x)).collect(),
        Piece::KeyDims { l, p, h, j, av, e } => e.iter().map(|&x| LK::KeyDim(*l, *p, *h, *j, *av, x as usize)).collect(),
        Piece::Keys { l, p, h, av, j } => j.iter().map(|&x| LK::Key(*l, *p, *h, *av, x as usize)).collect(),
        Piece::Heads { l, p, h } => h.iter().map(|&x| LK::Head(*l, *p, x as usize)).collect(),
        Piece::KvGroups { l, p, g } => g.iter().map(|&x| LK::Group(*l, *p, x as usize)).collect(),
        Piece::Neurons { l, p, n } => n.iter().map(|&x| LK::Neuron(*l, *p, x)).collect(),
        Piece::PleDims { l, p, e } => e.iter().map(|&x| LK::Dim(*l, *p, x)).collect(),
        Piece::Attn { l, p } => vec![LK::Comp(*l, *p, 0)],
        Piece::Kv { l, p } => vec![LK::Comp(*l, *p, 1)],
        Piece::Mlp { l, p } => vec![LK::Comp(*l, *p, 2)],
        Piece::Ple { l, p } => vec![LK::Comp(*l, *p, 3)],
    }
}

fn find_leaf(a: &A, g: &Geo, idx: &HashMap<LK, usize>) -> Option<usize> {
    let mut keys: Vec<LK> = Vec::new();
    match *a {
        A::W { l, p, s, u, a } => {
            let hd = g.layers[l].hd as u32;
            keys.push(LK::W(l, p, s, u, a));
            keys.push(LK::Unit(l, p, s, u));
            match s {
                Site::Q | Site::O => {
                    keys.push(LK::Head(l, p, (u / hd) as usize));
                    keys.push(LK::Comp(l, p, 0));
                }
                Site::K | Site::V => {
                    keys.push(LK::Group(l, p, (u / hd) as usize));
                    keys.push(LK::Comp(l, p, 1));
                }
                Site::Gate | Site::Up | Site::Down => {
                    keys.push(LK::Neuron(l, p, u));
                    keys.push(LK::Comp(l, p, 2));
                }
                _ => {
                    keys.push(LK::Dim(l, p, u));
                    keys.push(LK::Comp(l, p, 3));
                }
            }
        }
        A::Qk { l, p, h, j, e } | A::Av { l, p, h, j, e } => {
            let av = matches!(a, A::Av { .. });
            keys.extend([LK::KeyDim(l, p, h, j, av, e), LK::Key(l, p, h, av, j), LK::Head(l, p, h), LK::Comp(l, p, 0)]);
        }
    }
    keys.iter().find_map(|k| idx.get(k).copied())
}

fn out_json(m: &Model, o: &Out) -> Value {
    json!({"top": o.top.iter().map(|&(id, p)| json!({"id": id, "text": m.tok.piece(id), "probability": p})).collect::<Vec<_>>(),
        "target_log_odds": o.lo[0], "kl": o.kl})
}

fn write_json(dir: &FsPath, rel: &str, v: &Value) -> Result<(), String> {
    write_json_as(dir, rel, v, true)
}

/// Write many files at once: the folders first (each once), then the files in parallel. One file at a
/// time costs ~0.5 ms each on Windows (file creation, virus scan), which adds up over 10⁵ unit files.
fn write_all(dir: &FsPath, files: Vec<(String, String)>) -> Result<(), String> {
    use rayon::prelude::*;
    let parents: BTreeSet<std::path::PathBuf> = files.iter().filter_map(|(rel, _)| dir.join(rel).parent().map(|p| p.to_path_buf())).collect();
    for p in &parents {
        std::fs::create_dir_all(p).map_err(|x| format!("{}: {x}", p.display()))?;
    }
    files.into_par_iter().try_for_each(|(rel, text)| {
        let f = dir.join(&rel);
        std::fs::write(&f, text).map_err(|x| format!("{}: {x}", f.display()))
    })
}

fn json_text(v: &Value, pretty: bool) -> String {
    if pretty { serde_json::to_string_pretty(v) } else { serde_json::to_string(v) }.unwrap()
}

fn write_json_as(dir: &FsPath, rel: &str, v: &Value, pretty: bool) -> Result<(), String> {
    let f = dir.join(rel);
    if let Some(parent) = f.parent() {
        std::fs::create_dir_all(parent).map_err(|x| format!("{}: {x}", parent.display()))?;
    }
    let text = if pretty { serde_json::to_string_pretty(v) } else { serde_json::to_string(v) }.unwrap();
    std::fs::write(&f, text).map_err(|x| format!("{}: {x}", f.display()))
}

/// the first MAX_EDGES entries as they are, the others as one summary {count, sum}
fn capped(items: Vec<Value>, sums: &[f32]) -> (Value, Value) {
    if items.len() <= MAX_EDGES {
        return (json!(items), Value::Null);
    }
    let rest = &sums[MAX_EDGES..];
    (json!(items[..MAX_EDGES].to_vec()), json!({"count": rest.len(), "sum": rest.iter().sum::<f32>(), "sum_abs": rest.iter().map(|x| x.abs()).sum::<f32>()}))
}

/// Write the result folder. Returns the trace closure (largest relative deviation of any decomposition), or
/// None when there are too many atoms to list (then there is no trace, only units and components).
pub(super) fn write(r: &Report, dir: &FsPath, p_full: &[f32], plp_full: f64) -> Result<Option<(f32, Option<f32>)>, String> {
    let e = r.e;
    let m = e.m;
    let g = &e.g;
    let (t, d, focus, nl) = (g.t, g.d, g.focus, g.layers.len());
    let _ = (p_full, plp_full);
    std::fs::create_dir_all(dir).map_err(|x| format!("{}: {x}", dir.display()))?;
    // old unit files of an earlier run would mix in: the layers folder is written anew
    let _ = std::fs::remove_dir_all(dir.join("layers"));
    let n_on = e.atoms_kept();
    let wl = nl.saturating_sub(1).to_string().len().max(2);
    let wp = focus.to_string().len().max(2);
    // ------------------------------------------------------------ the pass of the union, fully recorded
    let o = RunOpts { record_inputs: true, record_norms: true, checkpoint: true, record_rows: (0..t).collect(), ..e.opts(e.built.clone()) };
    let tr = m.run(e.ids, focus, &o);
    let norms = tr.norms.as_ref().ok_or("no norm scales recorded")?;
    let ck = tr.checkpoint.as_ref().ok_or("no checkpoint recorded")?;
    let cx = Ctx { e, m, g, inp: &tr.inputs, x_tok: &tr.x_tok, norms, t, d, worst: RefCell::new((0.0, String::new())) };
    // leaves, units and components with their effects
    let mut leaf_idx: HashMap<LK, usize> = HashMap::new();
    for (i, lf) in r.leaves.iter().enumerate() {
        for k in leaf_keys(&lf.piece, g) {
            leaf_idx.insert(k, i);
        }
    }
    let unit_of_piece = |pc: &Piece| -> Option<Unit> {
        match pc {
            Piece::Heads { l, p, h } => Some(Unit::Head { l: *l, p: *p, h: h[0] as usize }),
            Piece::KvGroups { l, p, g } => Some(Unit::Kv { l: *l, p: *p, g: g[0] as usize }),
            Piece::Neurons { l, p, n } => Some(Unit::Neuron { l: *l, p: *p, n: n[0] }),
            Piece::PleDims { l, p, e } => Some(Unit::Ple { l: *l, p: *p, e: e[0] }),
            _ => None,
        }
    };
    let unit_fx: HashMap<Unit, Out> = r.fx.units.iter().filter_map(|(pc, o)| unit_of_piece(pc).map(|u| (u, *o))).collect();
    let comp_fx: HashMap<(usize, usize, &'static str), Out> = r
        .fx
        .comps
        .iter()
        .filter_map(|(pc, o)| {
            let (l, p) = pc.lp();
            let k = match pc {
                Piece::Attn { .. } => "attention",
                Piece::Kv { .. } => "kv",
                Piece::Mlp { .. } => "mlp",
                Piece::Ple { .. } => "ple",
                _ => return None,
            };
            Some(((l, p, k), *o))
        })
        .collect();
    // ------------------------------------------------------------ atoms
    let detailed = n_on <= MAX_ATOMS;
    let atoms: Vec<A> = if detailed { on_atoms(e) } else { Vec::new() };
    // columns of column-wise units in one pass per matrix
    let mut need: BTreeMap<(usize, Site), BTreeSet<u32>> = BTreeMap::new();
    for a in &atoms {
        if let A::W { l, s, u, .. } = *a {
            if !s.row_wise() {
                need.entry((l, s)).or_default().insert(u);
            }
        }
    }
    for ((l, s), us) in need {
        e.b.prefetch(l, s, &us.into_iter().collect::<Vec<_>>());
    }
    let mut infos: Vec<Info> = atoms.iter().map(|a| Info { a: *a, value: cx.value(a), ablated: cx.ablated(a), sources: Vec::new(), nonlinear: Vec::new(), rest: Vec::new(), to_output: None, closure: 0.0 }).collect();
    // lookups
    let mut writers: HashMap<(usize, usize), Vec<(usize, usize, u8)>> = HashMap::new(); // (p, o) -> (atom, layer, kind)
    let mut by_row: HashMap<(usize, usize, Site, u32), Vec<usize>> = HashMap::new(); // (l, p, site, unit)
    let mut av_by: HashMap<(usize, usize, usize, usize), Vec<usize>> = HashMap::new(); // (l, p, h, e)
    let mut qk_by: HashMap<(usize, usize, usize, usize), Vec<usize>> = HashMap::new(); // (l, p, h, j)
    for (i, a) in atoms.iter().enumerate() {
        match *a {
            A::W { l, p, s, u, a } => {
                let kind = match s {
                    Site::O => Some(0u8),
                    Site::Down => Some(1),
                    Site::PleProj => Some(2),
                    _ => None,
                };
                if let Some(k) = kind {
                    writers.entry((p, a as usize)).or_default().push((i, l, k));
                }
                by_row.entry((l, p, s, u)).or_default().push(i);
            }
            A::Av { l, p, h, e, .. } => av_by.entry((l, p, h, e)).or_default().push(i),
            A::Qk { l, p, h, j, .. } => qk_by.entry((l, p, h, j)).or_default().push(i),
        }
    }
    // ------------------------------------------------------------ the residual stream, stage by stage
    // writer sites: their switched-off atoms (means) and biases, per layer, kind and position, before the post-norm
    let mut w_abl: HashMap<(usize, u8), Vec<f32>> = HashMap::new(); // [t][d]
    let mut w_bias: HashMap<(usize, u8), Vec<f32>> = HashMap::new(); // [d]
    for l in 0..nl {
        let b = &m.blocks[l];
        for (kind, s) in [(0u8, Site::O), (1, Site::Down), (2, Site::PleProj)] {
            if !g.has(l, s) {
                continue;
            }
            match kind {
                0 => {
                    if let Some(bo) = &b.attn.bo {
                        w_bias.insert((l, 0), bo.clone());
                    }
                }
                1 => {
                    if let Some(bd) = &b.mlp.bdown {
                        w_bias.insert((l, 1), bd.clone());
                    }
                }
                _ => {}
            }
            let Some(means) = e.means.as_deref() else { continue };
            let n_in = g.units(l, s);
            // the input with every fully switched-off unit replaced by its mean, all others 0
            let mut xm = vec![0f32; t * n_in];
            let mut partial: Vec<(usize, u32, Vec<u32>)> = Vec::new();
            for p in 0..t {
                let sb = &e.state.cells[p][l].sites[s as usize];
                let xb = means.layers[l].pos[p].x(s);
                for u in 0..n_in as u32 {
                    if sb.unit_off(u) {
                        xm[p * n_in + u as usize] = xb[u as usize];
                    } else if let Some(a) = sb.part.get(&u) {
                        partial.push((p, u, a.iter().copied().collect()));
                    }
                }
            }
            let (mat, _) = e.b.mat(l, s);
            let mut y = if s == Site::O {
                // W_O split by heads, summed
                let nh = g.layers[l].nh;
                let ys = mat.matmul_split(&xm, t, nh);
                let mut y = vec![0f32; t * d];
                for p in 0..t {
                    for h in 0..nh {
                        for (a, b2) in y[p * d..(p + 1) * d].iter_mut().zip(&ys[(p * nh + h) * d..][..d]) {
                            *a += b2;
                        }
                    }
                }
                y
            } else {
                mat.matmul(&xm, t)
            };
            for (p, u, offs) in partial {
                let w = e.b.weights(l, s, u);
                let xb = means.layers[l].pos[p].x(s)[u as usize];
                for o2 in offs {
                    y[p * d + o2 as usize] += w[o2 as usize] * xb;
                }
            }
            w_abl.insert((l, kind), y);
        }
    }
    // the stages in order; running sums per position
    let mut snaps: HashMap<(usize, Stage), Snap> = HashMap::new();
    {
        let mut run_on = vec![vec![0f32; d]; t];
        let mut run_abl = vec![vec![0f32; d]; t];
        let mut run_bias = vec![vec![0f32; d]; t];
        let mut run_emb: Vec<Vec<f32>> = (0..t).map(|p| ck.x[0][p * d..(p + 1) * d].to_vec()).collect();
        // writer atoms per (layer, kind, position): (dim, value)
        let mut wat: HashMap<(usize, u8, usize), Vec<(usize, f32)>> = HashMap::new();
        for (i, a) in atoms.iter().enumerate() {
            if let A::W { l, p, s, a: o2, .. } = *a {
                let k = match s {
                    Site::O => 0u8,
                    Site::Down => 1,
                    Site::PleProj => 2,
                    _ => continue,
                };
                wat.entry((l, k, p)).or_default().push((o2 as usize, infos[i].value));
            }
        }
        let x_at = |st: Stage, p: usize| -> Vec<f32> {
            match st {
                Stage::Attn(l) => ck.x[l][p * d..(p + 1) * d].to_vec(),
                Stage::Mlp(l) => {
                    let mut x = ck.x[l][p * d..(p + 1) * d].to_vec();
                    if m.blocks[l].res != Res::Parallel {
                        for (a, b2) in x.iter_mut().zip(&tr.rows[l].attn[p * d..(p + 1) * d]) {
                            *a += b2;
                        }
                    }
                    x
                }
                Stage::PleGate(l) => {
                    let mut x = ck.x[l][p * d..(p + 1) * d].to_vec();
                    for (a, (b1, b2)) in x.iter_mut().zip(tr.rows[l].attn[p * d..(p + 1) * d].iter().zip(&tr.rows[l].mlp[p * d..(p + 1) * d])) {
                        *a += b1 + b2;
                    }
                    x
                }
                Stage::Final => tr.x_final[p * d..(p + 1) * d].to_vec(),
            }
        };
        let snap = |st: Stage, on: &[Vec<f32>], abl: &[Vec<f32>], bias: &[Vec<f32>], emb: &[Vec<f32>], snaps: &mut HashMap<(usize, Stage), Snap>| {
            for p in 0..t {
                let x = x_at(st, p);
                // (only meaningful when every atom is listed: otherwise the atoms' share is missing)
                for c in 0..d {
                    if !detailed {
                        break;
                    }
                    let rec = on[p][c] + abl[p][c] + bias[p][c] + emb[p][c];
                    let scale = on[p][c].abs() + abl[p][c].abs() + bias[p][c].abs() + emb[p][c].abs();
                    cx.check(rec, x[c], scale, || format!("residual at {st:?} [{p}] dim {c}"));
                }
                snaps.insert((p, st), Snap { abl: abl[p].clone(), bias: bias[p].clone(), emb: emb[p].clone(), x });
            }
        };
        for l in 0..nl {
            let parallel = m.blocks[l].res == Res::Parallel;
            snap(Stage::Attn(l), &run_on, &run_abl, &run_bias, &run_emb, &mut snaps);
            let add = |kind: u8, run_on: &mut Vec<Vec<f32>>, run_abl: &mut Vec<Vec<f32>>, run_bias: &mut Vec<Vec<f32>>| {
                for p in 0..t {
                    let post = |o2: usize| cx.post(l, kind, p, o2);
                    if let Some(v) = wat.get(&(l, kind, p)) {
                        for &(o2, val) in v {
                            run_on[p][o2] += post(o2) * val;
                        }
                    }
                    if let Some(y) = w_abl.get(&(l, kind)) {
                        for c in 0..d {
                            run_abl[p][c] += post(c) * y[p * d + c];
                        }
                    }
                    if let Some(bv) = w_bias.get(&(l, kind)) {
                        for c in 0..d {
                            run_bias[p][c] += post(c) * bv[c];
                        }
                    }
                }
            };
            if parallel {
                snap(Stage::Mlp(l), &run_on, &run_abl, &run_bias, &run_emb, &mut snaps);
                add(0, &mut run_on, &mut run_abl, &mut run_bias);
            } else {
                add(0, &mut run_on, &mut run_abl, &mut run_bias);
                snap(Stage::Mlp(l), &run_on, &run_abl, &run_bias, &run_emb, &mut snaps);
            }
            add(1, &mut run_on, &mut run_abl, &mut run_bias);
            if m.blocks[l].ple.is_some() {
                snap(Stage::PleGate(l), &run_on, &run_abl, &run_bias, &run_emb, &mut snaps);
                add(2, &mut run_on, &mut run_abl, &mut run_bias);
            }
            let sc = cx.scalar(l);
            if sc != 1.0 {
                for v in [&mut run_on, &mut run_abl, &mut run_bias, &mut run_emb] {
                    for row in v.iter_mut() {
                        for x in row.iter_mut() {
                            *x *= sc;
                        }
                    }
                }
            }
        }
        snap(Stage::Final, &run_on, &run_abl, &run_bias, &run_emb, &mut snaps);
    }
    // ------------------------------------------------------------ sources of every atom
    // the residual part: what the stage holds in dimension c, split by writer atom
    let resid_sources = |p: usize, st: Stage, c: usize, factor: f32, info: &mut Info| {
        let (pre, mu, nb) = match cx.pre(st, p) {
            Some((n, s)) => {
                let sn = &snaps[&(p, st)];
                let mu = if n.rms { 0.0 } else { sn.x.iter().map(|v| *v as f64).sum::<f64>() as f32 / d as f32 };
                (s * n.w[c], mu, n.b.as_ref().map_or(0.0, |b| b[c]))
            }
            None => (1.0, 0.0, 0.0),
        };
        let k = factor * pre;
        if let Some(ws) = writers.get(&(p, c)) {
            for &(wi, lw, kind) in ws {
                if let Some(sc) = cx.visible(st, lw, kind) {
                    let coef = k * cx.post(lw, kind, p, c) * sc;
                    info.sources.push(Src { from: wi, c: coef * infos_value(&atoms, wi, &cx), k: coef, via: "residual" });
                }
            }
        }
        let sn = &snaps[&(p, st)];
        info.rest.push(("embedding", k * sn.emb[c]));
        info.rest.push(("switched-off writers", k * sn.abl[c]));
        info.rest.push(("biases of writers", k * sn.bias[c]));
        if mu != 0.0 || nb != 0.0 {
            info.rest.push(("norm: mean subtraction and bias", factor * (-pre * mu + nb)));
        }
    };
    fn infos_value(atoms: &[A], i: usize, cx: &Ctx) -> f32 {
        cx.value(&atoms[i])
    }
    for i in 0..infos.len() {
        let a = infos[i].a;
        let mut info = Info { a, value: infos[i].value, ablated: infos[i].ablated, sources: Vec::new(), nonlinear: Vec::new(), rest: Vec::new(), to_output: None, closure: 0.0 };
        match a {
            A::W { l, p, s, u, a: c } if s.row_wise() => {
                let w = e.b.weights(l, s, u)[c as usize];
                match (s, cx.reads(l, s)) {
                    (Site::PleIn, _) => info.rest.push(("token vector (input of the per-layer embedding)", info.value)),
                    (_, Some(st)) => resid_sources(p, st, c as usize, w, &mut info),
                    _ => {}
                }
            }
            A::W { l, p, s: Site::O, u, a: o2 } => {
                let lg = &g.layers[l];
                let (h, x) = (u as usize / lg.hd, u as usize % lg.hd);
                let w = e.b.weights(l, Site::O, u)[o2 as usize];
                let mut on_sum = 0f32;
                if let Some(v) = av_by.get(&(l, p, h, x)) {
                    for &k in v {
                        let val = infos[k].value;
                        on_sum += val;
                        info.sources.push(Src { from: k, c: w * val, k: w, via: "mixing" });
                    }
                }
                // switched-off products of this head and dimension: their means (0 with zero ablation)
                let mut off_sum = 0f32;
                if let (Some(pq), Some(hb)) = (cx.pm(l, p), cx.cell(l, p).heads.get(h)) {
                    let gr = h / (lg.nh / lg.nkv);
                    for j in g.keys(l, p) {
                        if !hb.av.atom_on(j as u32, x as u32) {
                            off_sum += pq.a[h * pq.a.len() / lg.nh + j] * cx.pm(l, j).unwrap().v[gr * lg.hd + x];
                        }
                    }
                }
                let z = tr.inputs[l].z[p * lg.nh * lg.hd + u as usize];
                cx.check(on_sum + off_sum, z, on_sum.abs() + off_sum.abs(), || format!("z of S{l} head {h} dim {x} [{p}]"));
                info.rest.push(("switched-off mixing products", w * off_sum));
            }
            A::W { l, p, s: Site::Down, u: n, a: o2 } => {
                let lg = &g.layers[l];
                let li = &tr.inputs[l];
                let w = e.b.weights(l, Site::Down, n)[o2 as usize];
                let ff = lg.ff;
                let f = act_fn(m.blocks[l].mlp.act);
                let up_atoms = by_row.get(&(l, p, Site::Up, n)).cloned().unwrap_or_default();
                let up_on: f32 = up_atoms.iter().map(|&k| infos[k].value).sum();
                let up_abl = cx.row_ablated(l, p, Site::Up, n);
                let up_b = cx.bias(l, Site::Up, n as usize);
                cx.check(up_on + up_abl + up_b, li.up[p * ff + n as usize], up_on.abs() + up_abl.abs() + up_b.abs(), || format!("up of S{l} #{n} [{p}]"));
                let uv = li.up[p * ff + n as usize];
                if lg.gated {
                    // a = act(g)·u = φ·g·u with φ = act(g)/g held at its value: split symmetrically,
                    // half over the gate atoms (u fixed), half over the up atoms (g fixed) – the halves add
                    // up to φ·g·u exactly (as --trace splits a neuron)
                    let gv = li.gate[p * ff + n as usize];
                    let phi = ratio(f, gv);
                    let gate_atoms = by_row.get(&(l, p, Site::Gate, n)).cloned().unwrap_or_default();
                    let g_on: f32 = gate_atoms.iter().map(|&k| infos[k].value).sum();
                    let g_abl = cx.row_ablated(l, p, Site::Gate, n);
                    cx.check(g_on + g_abl, gv, g_on.abs() + g_abl.abs(), || format!("gate of S{l} #{n} [{p}]"));
                    let (ku, kg) = (w * phi * 0.5 * gv, w * phi * 0.5 * uv);
                    for &k in &up_atoms {
                        info.sources.push(Src { from: k, c: ku * infos[k].value, k: ku, via: "up (split with the gate)" });
                    }
                    for &k in &gate_atoms {
                        info.sources.push(Src { from: k, c: kg * infos[k].value, k: kg, via: "gate (split with up)" });
                    }
                    info.rest.push(("switched-off up atoms", ku * up_abl));
                    info.rest.push(("up bias", ku * up_b));
                    info.rest.push(("switched-off gate atoms", kg * g_abl));
                } else {
                    // a = act(u) = φ·u with φ = act(u)/u held at its value
                    let ku = w * ratio(f, uv);
                    for &k in &up_atoms {
                        info.sources.push(Src { from: k, c: ku * infos[k].value, k: ku, via: "up (activation ratio held fixed)" });
                    }
                    info.rest.push(("switched-off up atoms", ku * up_abl));
                    info.rest.push(("up bias", ku * up_b));
                }
            }
            A::W { l, p, s: Site::PleProj, u: x, a: o2 } => {
                // g_e = act(gate_e)·pl_e = (g_e/gate_e)·gate_e: linear in the gate with that ratio held fixed
                let lg = &g.layers[l];
                let dp = lg.dp.unwrap_or(0);
                let w = e.b.weights(l, Site::PleProj, x)[o2 as usize];
                let gate_atoms = by_row.get(&(l, p, Site::PleGate, x)).cloned().unwrap_or_default();
                let gsum: f32 = gate_atoms.iter().map(|&k| infos[k].value).sum::<f32>() + cx.row_ablated(l, p, Site::PleGate, x);
                let gv = tr.inputs[l].ple_g[p * dp + x as usize];
                if gsum.abs() > 1e-20 {
                    let kg = w * gv / gsum;
                    for &k in &gate_atoms {
                        info.sources.push(Src { from: k, c: kg * infos[k].value, k: kg, via: "PLE gate (ratio held fixed)" });
                    }
                    info.rest.push(("switched-off PLE gate atoms", kg * cx.row_ablated(l, p, Site::PleGate, x)));
                } else {
                    info.rest.push(("gated per-layer input (gate exactly 0)", info.value));
                }
            }
            A::Qk { l, p, h, j, e: x } => {
                let lg = &g.layers[l];
                let li = &tr.inputs[l];
                let lk = cx.kv_layer(l);
                let at = &m.blocks[l].attn;
                let atk = &m.blocks[lk].attn;
                let gr = h / (lg.nh / lg.nkv);
                let (ad, kvd) = (lg.nh * lg.hd, lg.nkv * lg.hd);
                let (q, k) = (li.q[p * ad + h * lg.hd + x], li.k[j * kvd + gr * lg.hd + x]);
                // q side (k held fixed): Q rows of this head through the q norm and RoPE
                let mut parts = [0f32; 3];
                for (x2, rc) in rope_row(at.rope.as_ref(), p, x) {
                    let nq = at.qn.as_ref().map_or(1.0, |n| n.w[x2] * cx.norms.layers[l].q[p * lg.nh + h]);
                    let row = (h * lg.hd + x2) as u32;
                    let coef = rc * nq * k;
                    for &ai in by_row.get(&(l, p, Site::Q, row)).map(|v| &v[..]).unwrap_or(&[]) {
                        info.sources.push(Src { from: ai, c: coef * infos[ai].value, k: coef, via: "q (k held fixed)" });
                        parts[0] += coef * infos[ai].value;
                    }
                    parts[1] += coef * cx.row_ablated(l, p, Site::Q, row);
                    parts[2] += coef * cx.bias(l, Site::Q, row as usize);
                }
                info.rest.push(("q: switched-off Q atoms", parts[1]));
                info.rest.push(("q: bias", parts[2]));
                cx.check(parts.iter().sum(), q * k, parts.iter().map(|v| v.abs()).sum(), || format!("q side of {}", a.name()));
                // k side (q held fixed): K rows of its group at the key position through the k norm and RoPE
                let mut kp = [0f32; 3];
                let nkvk = g.layers[lk].nkv;
                for (x2, rc) in rope_row(atk.rope.as_ref(), j, x) {
                    let nk = atk.kn.as_ref().map_or(1.0, |n| n.w[x2] * cx.norms.layers[lk].k[j * nkvk + gr]);
                    let row = (gr * lg.hd + x2) as u32;
                    let coef = rc * nk * q;
                    for &ai in by_row.get(&(lk, j, Site::K, row)).map(|v| &v[..]).unwrap_or(&[]) {
                        info.sources.push(Src { from: ai, c: coef * infos[ai].value, k: coef, via: "k (q held fixed)" });
                        kp[0] += coef * infos[ai].value;
                    }
                    kp[1] += coef * cx.row_ablated(lk, j, Site::K, row);
                    kp[2] += coef * cx.bias(lk, Site::K, row as usize);
                }
                info.rest.push(("k: switched-off K atoms", kp[1]));
                info.rest.push(("k: bias", kp[2]));
                cx.check(kp.iter().sum(), q * k, kp.iter().map(|v| v.abs()).sum(), || format!("k side of {}", a.name()));
            }
            A::Av { l, p, h, j, e: x } => {
                let lg = &g.layers[l];
                let li = &tr.inputs[l];
                let lk = cx.kv_layer(l);
                let atk = &m.blocks[lk].attn;
                let gr = h / (lg.nh / lg.nkv);
                let a_j = li.probs[(p * lg.nh + h) * t + j];
                let vs = if atk.v_is_k { Site::K } else { Site::V };
                let row = (gr * lg.hd + x) as u32;
                let nkvk = g.layers[lk].nkv;
                let nv = atk.vn.as_ref().map_or(1.0, |n| n.w[x] * cx.norms.layers[lk].v[j * nkvk + gr]);
                let coef = a_j * nv;
                let mut parts = [0f32; 3];
                for &ai in by_row.get(&(lk, j, vs, row)).map(|v| &v[..]).unwrap_or(&[]) {
                    info.sources.push(Src { from: ai, c: coef * infos[ai].value, k: coef, via: "v (weight held fixed)" });
                    parts[0] += coef * infos[ai].value;
                }
                parts[1] = coef * cx.row_ablated(lk, j, vs, row);
                parts[2] = coef * cx.bias(lk, vs, row as usize);
                info.rest.push(("v: switched-off V atoms", parts[1]));
                info.rest.push(("v: bias", parts[2]));
                cx.check(parts.iter().sum(), info.value, parts.iter().map(|v| v.abs()).sum(), || format!("v side of {}", a.name()));
                let sc = m.blocks[l].attn.scale;
                for &k in qk_by.get(&(l, p, h, j)).map(|v| &v[..]).unwrap_or(&[]) {
                    info.nonlinear.push((k, infos[k].value * sc, "score of this key (softmax)"));
                }
            }
            _ => {}
        }
        // closure of the main decomposition (q side for scores)
        if !info.sources.is_empty() || !info.rest.is_empty() {
            let main: f32 = info.sources.iter().filter(|s| !s.via.starts_with('k')).map(|s| s.c).sum::<f32>() + info.rest.iter().filter(|r| !r.0.starts_with("k:")).map(|r| r.1).sum::<f32>();
            let scale: f32 = info.sources.iter().map(|s| s.c.abs()).sum::<f32>() + info.rest.iter().map(|r| r.1.abs()).sum::<f32>();
            info.closure = cx.check(main, info.value, scale, || format!("decomposition of {}", a.name()));
        }
        infos[i] = info;
    }
    // ------------------------------------------------------------ the output: target and rival logit at the focus
    let lm = m.lm_head();
    let mut wu = [vec![0f32; d], vec![0f32; d]];
    lm.row(e.ys[0] as usize, &mut wu[0]);
    lm.row(e.ys[1] as usize, &mut wu[1]);
    let fin = &snaps[&(focus, Stage::Final)];
    let (fs, fnorm) = match cx.pre(Stage::Final, focus) {
        Some((n, s)) => (s, Some(n)),
        None => (1.0, None),
    };
    let h_fin: Vec<f32> = match fnorm {
        Some(n) => {
            let mut h = vec![0f32; d];
            n.apply_scale(&fin.x, &mut h, fs);
            h
        }
        None => fin.x.clone(),
    };
    let mut root = Vec::new();
    for yi in 0..if detailed { 2 } else { 0 } {
        let z: f32 = wu[yi].iter().zip(&h_fin).map(|(a, b)| a * b).sum();
        let mut parts: Vec<(&'static str, f32)> = Vec::new();
        let (mut emb, mut abl, mut bias) = (0f32, 0f32, 0f32);
        let mut norm_t = 0f32;
        let mu = match fnorm {
            Some(n) if !n.rms => fin.x.iter().map(|v| *v as f64).sum::<f64>() as f32 / d as f32,
            _ => 0.0,
        };
        let mut on_total = 0f32;
        for c in 0..d {
            let pre = fnorm.map_or(1.0, |n| fs * n.w[c]);
            emb += wu[yi][c] * pre * fin.emb[c];
            abl += wu[yi][c] * pre * fin.abl[c];
            bias += wu[yi][c] * pre * fin.bias[c];
            if let Some(n) = fnorm {
                norm_t += wu[yi][c] * (-pre * mu + n.b.as_ref().map_or(0.0, |b| b[c]));
            }
            if let Some(ws) = writers.get(&(focus, c)) {
                for &(wi, lw, kind) in ws {
                    let sc = cx.visible(Stage::Final, lw, kind).unwrap_or(1.0);
                    let contrib = wu[yi][c] * pre * cx.post(lw, kind, focus, c) * sc * infos[wi].value;
                    on_total += contrib;
                    let to = infos[wi].to_output.get_or_insert((0.0, 0.0));
                    if yi == 0 {
                        to.0 += contrib;
                    } else {
                        to.1 += contrib;
                    }
                }
            }
        }
        parts.extend([("atoms", on_total), ("embedding", emb), ("switched-off writers", abl), ("biases of writers", bias), ("norm: mean subtraction and bias", norm_t)]);
        let scale: f32 = parts.iter().map(|x| x.1.abs()).sum();
        let dev = cx.check(parts.iter().map(|x| x.1).sum(), z, scale, || format!("logit of {}", tok(m, e.ys[yi])));
        root.push(json!({"token": {"id": e.ys[yi], "text": m.tok.piece(e.ys[yi])}, "logit_before_softcap": z,
            "softcap": m.logit_cap, "parts": parts.iter().map(|(k, v)| json!({"part": k, "value": v})).collect::<Vec<_>>(), "closure": dev}));
    }
    // ------------------------------------------------------------ the flow: the output, split over every atom
    // With the norm scales, the attention weights and the activation ratios act(x)/x held at their values,
    // everything between the atoms is linear. Then every atom passes on an exact part of the output: its
    // value times its sensitivity S_A (how the output moves per unit of the atom's value along everything
    // downstream). The parts that are not atoms – embeddings, switched-off atoms, biases, norm terms – are the
    // roots; all roots together give the output exactly. Q·K products only steer where heads look (the
    // weights are held), so they and the Q/K atoms carry no flow; their causal role shows in the switch-off
    // tests.
    let flow = if detailed && !atoms.is_empty() { Some(flow_split(&atoms, &infos, &cx, e, m, g, &tr, fin, fs, fnorm, focus)) } else { None };
    let fnode_name = |n: &FNode| match n {
        FNode::Root(s) => s.clone(),
        FNode::U(u) => u.name(),
        FNode::Output => "output".to_string(),
    };
    // per unit: what flows out of it, what flows in (from which unit or root)
    let mut unit_out: HashMap<Unit, Vec<f32>> = HashMap::new();
    let mut unit_in: HashMap<Unit, Vec<(FNode, Vec<f32>)>> = HashMap::new();
    let mut unit_to: HashMap<Unit, Vec<(FNode, Vec<f32>)>> = HashMap::new();
    if let Some(fl) = &flow {
        let kk = fl.toks.len();
        for ((a, b), v) in &fl.edges {
            if let FNode::U(ua) = a {
                let x = unit_out.entry(*ua).or_insert_with(|| vec![0f32; kk]);
                x.iter_mut().zip(v).for_each(|(p2, q)| *p2 += q);
                unit_to.entry(*ua).or_default().push((b.clone(), v.clone()));
            }
            if let FNode::U(ub) = b {
                unit_in.entry(*ub).or_default().push((a.clone(), v.clone()));
            }
        }
    }
    let diff = |v: &[f32]| v[0] - v[1];
    let flow_json = |v: &[f32]| json!({"target_logit": v[0], "rival_logit": v[1], "difference": v[0] - v[1]});
    // ------------------------------------------------------------ units, edges between units
    let mut unit_atoms: BTreeMap<Unit, Vec<usize>> = BTreeMap::new();
    for (i, a) in atoms.iter().enumerate() {
        unit_atoms.entry(a.unit(g)).or_default().push(i);
    }
    let mut used_by: Vec<Vec<(usize, f32, &'static str)>> = vec![Vec::new(); infos.len()];
    for (i, inf) in infos.iter().enumerate() {
        for s in &inf.sources {
            used_by[s.from].push((i, s.c, s.via));
        }
    }
    let mut used_nl: Vec<Vec<(usize, &'static str)>> = vec![Vec::new(); infos.len()];
    for (i, inf) in infos.iter().enumerate() {
        for &(k, _, role) in &inf.nonlinear {
            used_nl[k].push((i, role));
        }
    }
    // unit graph: (from, to) -> (Σ contribution, Σ |contribution|, atom edges)
    #[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
    enum Node {
        U(Unit),
        Output,
    }
    let mut edges: HashMap<(Node, Node), (f64, f64, usize)> = HashMap::new();
    for inf in &infos {
        let to = Node::U(inf.a.unit(g));
        for s in &inf.sources {
            let from = Node::U(atoms[s.from].unit(g));
            if from == to {
                continue;
            }
            let x = edges.entry((from, to)).or_default();
            x.0 += s.c as f64;
            x.1 += s.c.abs() as f64;
            x.2 += 1;
        }
        if let Some((a, b2)) = inf.to_output {
            let x = edges.entry((Node::U(inf.a.unit(g)), Node::Output)).or_default();
            x.0 += (a - b2) as f64;
            x.1 += (a - b2).abs() as f64;
            x.2 += 1;
        }
    }
    // depth: steps back from the output (breadth first over the edges, reversed)
    let mut preds: HashMap<Node, Vec<Node>> = HashMap::new();
    for &(f, to) in edges.keys() {
        preds.entry(to).or_default().push(f);
    }
    let mut depth: HashMap<Node, usize> = HashMap::new();
    let mut queue = VecDeque::new();
    depth.insert(Node::Output, 0);
    queue.push_back(Node::Output);
    while let Some(n) = queue.pop_front() {
        let dn = depth[&n];
        for &pr in preds.get(&n).map(|v| &v[..]).unwrap_or(&[]) {
            if let std::collections::hash_map::Entry::Vacant(v) = depth.entry(pr) {
                v.insert(dn + 1);
                queue.push_back(pr);
            }
        }
    }
    let node_name = |n: &Node| match n {
        Node::U(u) => u.name(),
        Node::Output => "output".to_string(),
    };
    // ------------------------------------------------------------ unit files
    let leaf_of: Vec<Option<usize>> = atoms.iter().map(|a| find_leaf(a, g, &leaf_idx)).collect();
    let paths_of_unit = |ids: &[usize]| -> Vec<usize> {
        let s: BTreeSet<usize> = ids.iter().filter_map(|&i| leaf_of[i].map(|li| r.leaves[li].path)).collect();
        s.into_iter().collect()
    };
    let fx_json = |o: Option<&Out>| o.map_or(Value::Null, |o| out_json(m, o));
    let mut unit_rows: Vec<Value> = Vec::new();
    let mut unit_files: Vec<(String, String)> = Vec::new();
    let all_units: BTreeSet<Unit> = unit_atoms.keys().copied().chain(unit_fx.keys().copied()).collect();
    for u in &all_units {
        let ids = unit_atoms.get(u).cloned().unwrap_or_default();
        let file = u.file(wl, wp);
        let direct: f32 = ids.iter().filter_map(|&i| infos[i].to_output.map(|(a, b2)| a - b2)).sum();
        let reads: Vec<Value> = {
            let mut v: Vec<(&(Node, Node), &(f64, f64, usize))> = edges.iter().filter(|((_, to), _)| *to == Node::U(*u)).collect();
            v.sort_by(|a, b| b.1 .1.total_cmp(&a.1 .1));
            v.iter().map(|((f, _), (s, sa, n))| json!({"unit": node_name(f), "sum": s, "sum_abs": sa, "atom_edges": n})).collect()
        };
        let read_by: Vec<Value> = {
            let mut v: Vec<(&(Node, Node), &(f64, f64, usize))> = edges.iter().filter(|((f, _), _)| *f == Node::U(*u)).collect();
            v.sort_by(|a, b| b.1 .1.total_cmp(&a.1 .1));
            v.iter().map(|((_, to), (s, sa, n))| json!({"unit": node_name(to), "sum": s, "sum_abs": sa, "atom_edges": n})).collect()
        };
        let summary = json!({
            "unit": u.name(), "kind": u.kind(), "layer": u.lp().0, "position": u.lp().1,
            "token": m.tok.piece(e.ids[u.lp().1]), "atoms_on": ids.len(), "paths": paths_of_unit(&ids),
            "switched_off_alone_in_union": fx_json(unit_fx.get(u)),
            "direct_to_output_logit_difference": if ids.iter().any(|&i| infos[i].to_output.is_some()) { json!(direct) } else { Value::Null },
            "steps_to_output": depth.get(&Node::U(*u)),
            "reads_from_units": reads, "read_by_units": read_by, "file": file,
            "flow": unit_out.get(u).map(|v| flow_json(v)),
            "flow_from": unit_in.get(u).map(|v| {
                let mut v: Vec<&(FNode, Vec<f32>)> = v.iter().collect();
                v.sort_by(|a, b| diff(&b.1).abs().total_cmp(&diff(&a.1).abs()));
                v.iter().take(MAX_EDGES).map(|(n2, x)| json!({"from": fnode_name(n2), "flow": flow_json(x)})).collect::<Vec<_>>()
            }),
            "flow_to": unit_to.get(u).map(|v| {
                let mut v: Vec<&(FNode, Vec<f32>)> = v.iter().collect();
                v.sort_by(|a, b| diff(&b.1).abs().total_cmp(&diff(&a.1).abs()));
                v.iter().take(MAX_EDGES).map(|(n2, x)| json!({"to": fnode_name(n2), "flow": flow_json(x)})).collect::<Vec<_>>()
            }),
        });
        if detailed {
            let atoms_json: Vec<Value> = ids
                .iter()
                .map(|&i| {
                    let inf = &infos[i];
                    let lf = leaf_of[i].map(|li| &r.leaves[li]);
                    let mut src: Vec<&Src> = inf.sources.iter().collect();
                    src.sort_by(|a, b| b.c.abs().total_cmp(&a.c.abs()));
                    let mut ub = used_by[i].clone();
                    ub.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
                    let (src_l, src_rest) = capped(src.iter().map(|s| json!({"from": inf_name(&atoms, s.from), "contribution": s.c, "via": s.via})).collect(), &src.iter().map(|s| s.c).collect::<Vec<_>>());
                    let (ub_l, ub_rest) = capped(ub.iter().map(|(k, c, w)| json!({"atom": inf_name(&atoms, *k), "contribution": c, "via": w})).collect(), &ub.iter().map(|x| x.1).collect::<Vec<_>>());
                    json!({
                        "atom": inf.a.name(), "at": inf.a.fields(), "value": inf.value, "value_when_off": inf.ablated,
                        "path": lf.map(|l| l.path),
                        "leaf": lf.map(|l| json!({"name": l.piece.name(m), "atoms": l.piece.n_atoms(g),
                            "switched_off_alone_in_path": out_json(m, &l.in_path), "switched_off_alone_in_union": out_json(m, &l.out)})),
                        "sources": src_l, "sources_not_listed": src_rest,
                        "nonlinear_sources": inf.nonlinear.iter().map(|(k, v, w)| json!({"from": inf_name(&atoms, *k), "value": v, "role": w})).collect::<Vec<_>>(),
                        "rest": inf.rest.iter().filter(|r| r.1 != 0.0).map(|(k, v)| json!({"part": k, "value": v})).collect::<Vec<_>>(),
                        "closure": inf.closure,
                        "flow": flow.as_ref().map(|fl| { let kk = fl.toks.len(); flow_json(&fl.atom[i * kk..i * kk + 2]) }),
                        "to_output": inf.to_output.map(|(a, b2)| json!({"target_logit": a, "rival_logit": b2, "difference": a - b2})),
                        "used_by": ub_l, "used_by_not_listed": ub_rest,
                        "used_nonlinearly_by": used_nl[i].iter().map(|(k, w)| json!({"atom": inf_name(&atoms, *k), "role": w})).collect::<Vec<_>>(),
                    })
                })
                .collect();
            let mut v = summary.clone();
            v["atoms"] = json!(atoms_json);
            unit_files.push((file, json_text(&v, ids.len() <= PRETTY_ATOMS)));
        } else {
            unit_files.push((file, json_text(&summary, true)));
        }
        unit_rows.push(summary);
    }
    write_all(dir, unit_files)?;
    fn inf_name(atoms: &[A], i: usize) -> String {
        atoms[i].name()
    }
    // ------------------------------------------------------------ layer files
    let mut layers_json: Vec<Value> = Vec::new();
    for l in 0..nl {
        let units_l: Vec<&Value> = unit_rows.iter().filter(|u| u["layer"].as_u64() == Some(l as u64)).collect();
        // sorted by position and kind, so the file is the same in every run (comp_fx is a HashMap)
        let mut comps_here: Vec<_> = comp_fx.iter().filter(|((cl, _, _), _)| *cl == l).collect();
        comps_here.sort_by_key(|((_, p, k), _)| (*p, *k));
        let comps_l: Vec<Value> = comps_here
            .into_iter()
            .map(|((_, p, k), o)| json!({"component": k, "position": p, "token": m.tok.piece(e.ids[*p]), "switched_off_alone_in_union": out_json(m, o)}))
            .collect();
        if units_l.is_empty() && comps_l.is_empty() {
            continue;
        }
        let rel = format!("layers/S{l:0wl$}/layer.json");
        let positions: BTreeSet<u64> = units_l.iter().filter_map(|u| u["position"].as_u64()).collect();
        let v = json!({"layer": l, "components": comps_l, "positions": positions, "units": units_l.iter().map(|u| json!({
            "unit": u["unit"], "atoms_on": u["atoms_on"], "paths": u["paths"], "switched_off_alone_in_union": u["switched_off_alone_in_union"], "file": u["file"]})).collect::<Vec<_>>()});
        write_json(dir, &rel, &v)?;
        layers_json.push(json!({"layer": l, "file": rel, "units": units_l.len(), "atoms_on": units_l.iter().filter_map(|u| u["atoms_on"].as_u64()).sum::<u64>()}));
    }
    // ------------------------------------------------------------ trace.json: backwards from the output
    let mut order: Vec<(&Node, &usize)> = depth.iter().filter(|(n, _)| **n != Node::Output).collect();
    order.sort_by_key(|(n, dd)| (**dd, node_name(n)));
    let trace_units: Vec<Value> = order
        .iter()
        .map(|(n, dd)| {
            let mut ins: Vec<(&(Node, Node), &(f64, f64, usize))> = edges.iter().filter(|((_, to), _)| to == *n).collect();
            ins.sort_by(|a, b| b.1 .1.total_cmp(&a.1 .1));
            let mut outs: Vec<(&(Node, Node), &(f64, f64, usize))> = edges.iter().filter(|((f, _), _)| f == *n).collect();
            outs.sort_by(|a, b| b.1 .1.total_cmp(&a.1 .1));
            let file = match n {
                Node::U(u) => json!(u.file(wl, wp)),
                Node::Output => Value::Null,
            };
            json!({"unit": node_name(n), "steps_to_output": dd, "file": file,
                "feeds": outs.iter().map(|((_, to), (s, sa, k))| json!({"unit": node_name(to), "sum": s, "sum_abs": sa, "atom_edges": k})).collect::<Vec<_>>(),
                "reads": ins.iter().map(|((f, _), (s, sa, k))| json!({"unit": node_name(f), "sum": s, "sum_abs": sa, "atom_edges": k})).collect::<Vec<_>>()})
        })
        .collect();
    let unreached: Vec<String> = all_units.iter().filter(|u| !depth.contains_key(&Node::U(**u))).map(|u| u.name()).collect();
    let mut direct: Vec<(usize, f32)> = infos.iter().enumerate().filter_map(|(i, x)| x.to_output.map(|(a, b2)| (i, a - b2))).collect();
    direct.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
    let (closure, worst_at) = cx.worst.borrow().clone();
    let closure_v = if detailed { json!(closure) } else { Value::Null };
    write_json(
        dir,
        "trace.json",
        &json!({
            "what": "The circuit traced backwards from the output. steps_to_output = how many links a unit is away from the output; feeds/reads = summed exact contributions of the atom edges between two units (in the units of the receiving atoms; to the output: the target's minus the rival's logit before a softcap).",
            "output": root,
            "direct_atoms": direct.iter().take(200).map(|(i, c)| json!({"atom": atoms[*i].name(), "unit": atoms[*i].unit(g).name(), "logit_difference": c})).collect::<Vec<_>>(),
            "units": trace_units,
            "units_without_a_path_to_the_output": unreached,
            "closure": closure_v, "closure_worst_at": worst_at,
        }),
    )?;
    // ------------------------------------------------------------ flow.json and the matrix
    let mut flow_checks = Value::Null;
    let mut flow_dev: Option<f32> = None;
    if let Some(fl) = &flow {
        let kk = fl.toks.len();
        let total = fl.logits[0] - fl.logits[1];
        let share = |v: &[f32]| if total.abs() > 1e-9 { json!(diff(v) / total) } else { Value::Null };
        let tokj = |t: u32| json!({"id": t, "text": m.tok.piece(t)});
        // units by what flows out of them (onto target − rival)
        let mut units_f: Vec<(&Unit, &Vec<f32>)> = unit_out.iter().collect();
        units_f.sort_by(|a, b| diff(b.1).abs().total_cmp(&diff(a.1).abs()));
        let mut edges_f: Vec<(&(FNode, FNode), &Vec<f32>)> = fl.edges.iter().collect();
        edges_f.sort_by(|a, b| diff(b.1).abs().total_cmp(&diff(a.1).abs()));
        // roots by kind and position
        let mut roots: Vec<(&String, Vec<f32>)> = Vec::new();
        {
            let mut acc: BTreeMap<&String, Vec<f32>> = BTreeMap::new();
            for ((a, _), v) in &fl.edges {
                if let FNode::Root(r) = a {
                    let x = acc.entry(r).or_insert_with(|| vec![0f32; kk]);
                    x.iter_mut().zip(v).for_each(|(p2, q)| *p2 += q);
                }
            }
            roots.extend(acc);
            roots.sort_by(|a, b| diff(&b.1).abs().total_cmp(&diff(&a.1).abs()));
        }
        // the main path: from the output backwards, always along the incoming edge that carries the most
        let mut path: Vec<Value> = Vec::new();
        let mut node = FNode::Output;
        let mut seen: Vec<FNode> = Vec::new();
        for _ in 0..40 {
            let mut ins: Vec<(&FNode, &Vec<f32>)> = fl.edges.iter().filter(|((_, b), _)| *b == node).map(|((a, _), v)| (a, v)).collect();
            if ins.is_empty() {
                break;
            }
            ins.sort_by(|a, b| diff(b.1).total_cmp(&diff(a.1)));
            let into: f32 = ins.iter().map(|(_, v)| diff(v)).sum();
            path.push(json!({"into": fnode_name(&node), "flow_into_it": into, "share_of_total": if total.abs() > 1e-9 { json!(into / total) } else { Value::Null },
                "strongest_inputs": ins.iter().take(5).map(|(a, v)| json!({"from": fnode_name(a), "flow": diff(v), "share_of_total": share(v)})).collect::<Vec<_>>()}));
            seen.push(node.clone());
            let next = ins[0].0.clone();
            if matches!(next, FNode::Root(_)) || seen.contains(&next) {
                path.push(json!({"into": fnode_name(&next), "end": "a root: no atom before it"}));
                break;
            }
            node = next;
        }
        let worst = fl.dev_roots.max(fl.dev_units.0).max(fl.dev_matrix);
        flow_dev = Some(worst);
        flow_checks = json!({"roots_sum_to_the_logits": fl.dev_roots, "every_unit_passes_on_what_it_receives": fl.dev_units.0, "worst_unit": fl.dev_units.1,
            "matrix_rows_and_roots_sum_to_the_stream": if fl.matrix.is_some() { json!(fl.dev_matrix) } else { Value::Null }});
        if let Some(mx) = &fl.matrix {
            let bytes: Vec<u8> = mx.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(dir.join("flows_residual.f32"), bytes).map_err(|x| format!("{}: {x}", dir.display()))?;
            write_json_as(dir, "flows_residual.json", &json!({
                "what": "Row i = the flow of atom i into the stream at the focus at the end (before the final norm), float32 little-endian, rows × columns. The logit of any token t is W_U[t] · N(row) with N the final norm at its frozen scale (RMS: s·w⊙x; LayerNorm: s·w⊙(x − mean x)); the roots in flow.json (embeddings, switched-off atoms, biases) plus every writer's direct part give the stream exactly.",
                "rows": atoms.len(), "columns": d, "final_norm_scale": fs, "final_norm": fnorm.map(|nm| if nm.rms { "rms" } else { "layernorm" }),
                "atoms": atoms.iter().map(|a| a.name()).collect::<Vec<_>>()}), false)?;
        }
        let top_atoms: Vec<Value> = {
            let mut by: Vec<usize> = (0..atoms.len()).collect();
            by.sort_by(|&a, &b| (fl.atom[b * kk] - fl.atom[b * kk + 1]).abs().total_cmp(&(fl.atom[a * kk] - fl.atom[a * kk + 1]).abs()));
            by.iter().take(300).map(|&i| json!({"atom": atoms[i].name(), "unit": atoms[i].unit(g).name(), "flow": flow_json(&fl.atom[i * kk..i * kk + 2]), "share_of_total": share(&fl.atom[i * kk..i * kk + 2])})).collect()
        };
        write_json(dir, "flow.json", &json!({
            "what": "The output split over every atom. With the norm scales, the attention weights and the activation ratios act(x)/x held at their values, everything between atoms is linear, so every atom passes on an exact part of the output (value × sensitivity). The parts that are no atom are the roots; all roots together give each logit exactly. Units pass on exactly what they receive. Q·K products only steer where heads look and carry no flow (their effect shows in the switch-off tests). difference = onto the target's logit minus onto the rival's, before a softcap.",
            "tokens": fl.toks.iter().enumerate().map(|(ki, &t)| json!({"token": tokj(t), "logit_before_softcap": fl.logits[ki], "sum_of_the_roots": fl.roots_sum[ki]})).collect::<Vec<_>>(),
            "target_minus_rival": total,
            "main_path": path,
            "units": units_f.iter().map(|(u, v)| json!({"unit": u.name(), "file": u.file(wl, wp), "flow": flow_json(v), "share_of_total": share(v),
                "per_token": fl.toks.iter().zip(v.iter()).map(|(&t, x)| json!({"token": m.tok.piece(t), "flow": x})).collect::<Vec<_>>()})).collect::<Vec<_>>(),
            "edges": edges_f.iter().take(500).map(|((a, b), v)| json!({"from": fnode_name(a), "to": fnode_name(b), "flow": flow_json(v), "share_of_total": share(v)})).collect::<Vec<_>>(),
            "roots": roots.iter().map(|(r, v)| json!({"root": r, "flow": flow_json(v), "share_of_total": share(v)})).collect::<Vec<_>>(),
            "atoms": top_atoms,
            "voices": fl.voices.iter().map(|(i, up, down)| json!({"atom": atoms[*i].name(), "flow": flow_json(&fl.atom[i * kk..i * kk + 2]),
                "raises": up.iter().map(|(t, v)| json!({"token": m.tok.piece(*t), "logit": v})).collect::<Vec<_>>(),
                "lowers": down.iter().map(|(t, v)| json!({"token": m.tok.piece(*t), "logit": v})).collect::<Vec<_>>()})).collect::<Vec<_>>(),
            "matrix": if fl.matrix.is_some() { json!({"file": "flows_residual.f32", "index": "flows_residual.json", "rows": atoms.len(), "columns": d}) } else { Value::Null },
            "checks": flow_checks,
        }))?;
        // ---- the protocol: every step in the order the model computes it
        let sub_of = |a: &A| -> u8 {
            match *a {
                A::W { s, .. } => match s {
                    Site::Q | Site::K | Site::V => 0,
                    Site::O => 3,
                    Site::Gate | Site::Up => 4,
                    Site::Down => 5,
                    Site::PleIn | Site::PleGate => 6,
                    Site::PleProj => 7,
                },
                A::Qk { .. } => 1,
                A::Av { .. } => 2,
            }
        };
        let lp_of = |a: &A| -> (usize, usize) {
            match *a {
                A::W { l, p, .. } | A::Qk { l, p, .. } | A::Av { l, p, .. } => (l, p),
            }
        };
        let mut order_a: Vec<usize> = (0..atoms.len()).collect();
        order_a.sort_by_key(|&i| {
            let (l, p) = lp_of(&atoms[i]);
            (l, sub_of(&atoms[i]), p, i)
        });
        let mut lines = String::with_capacity(atoms.len() * 400);
        for (step, &i) in order_a.iter().enumerate() {
            let inf = &infos[i];
            let (l, p) = lp_of(&inf.a);
            let mut src: Vec<&Src> = inf.sources.iter().collect();
            src.sort_by(|a, b| b.c.abs().total_cmp(&a.c.abs()));
            let v = json!({"step": step + 1, "layer": l, "position": p, "token": m.tok.piece(e.ids[p]), "atom": inf.a.name(), "unit": inf.a.unit(g).name(),
                "value": inf.value, "value_when_off": inf.ablated, "flow": flow_json(&fl.atom[i * kk..i * kk + 2]),
                "reads": src.iter().take(3).map(|s2| json!({"from": atoms[s2.from].name(), "contribution": s2.c, "via": s2.via})).collect::<Vec<_>>(),
                "rest": inf.rest.iter().filter(|r| r.1 != 0.0).map(|(k, v)| json!({"part": k, "value": v})).collect::<Vec<_>>()});
            lines += &serde_json::to_string(&v).unwrap();
            lines.push('\n');
        }
        std::fs::write(dir.join("atoms_in_order.jsonl"), lines).map_err(|x| format!("{}: {x}", dir.display()))?;
        let unit_key = |u: &Unit| -> (usize, u8, usize) {
            match *u {
                Unit::Kv { l, p, .. } => (l, 0, p),
                Unit::Head { l, p, .. } => (l, 1, p),
                Unit::Neuron { l, p, .. } => (l, 2, p),
                Unit::Ple { l, p, .. } => (l, 3, p),
            }
        };
        let mut units_o: Vec<Unit> = unit_out.keys().copied().collect();
        units_o.sort_by_key(|u| (unit_key(u), *u));
        let q = |t: u32| format!("{:?}", m.tok.piece(t));
        let pct1 = |x: f32| format!("{:.1}%", x * 100.0);
        let tgt = q(e.ys[0]);
        let riv = q(e.ys[1]);
        let mut md = String::new();
        md += &format!("# How the model computes {tgt} – step by step\n\n");
        md += &format!("Model: {} · prediction at position {focus} · target {tgt}, rival {riv}\n\n", m.name);
        md += "Tokens: ";
        md += &e.ids.iter().enumerate().map(|(i, &t)| format!("[{i}] {:?}", m.tok.piece(t))).collect::<Vec<_>>().join(" ");
        md += &format!(
            "\n\nThe margin of {tgt} over {riv} (logits before a softcap): **{:+.3}** ({tgt} {:.3}, {riv} {:.3}). Every step below passes on an exact part of it (norm scales, attention weights and activation ratios held at their values; the parts add up to the margin exactly). A step is one head, K/V group, neuron or PLE dimension at one position, in the order the model computes them; Q·K products only steer where heads look and are not counted as flow.\n\nHow to read it: *passes on* and *what it adds to the output* describe the part of the final answer that travels through a step – flow is conserved, every step passes on exactly what it receives, so steps on the same route share the same signature. What a step itself decides is its *decision*: where a head looks (its attention weights, set by the Q·K products) and how strongly a neuron fires (its gate). Those are the non-linear switches the transport runs through.\n\n",
            total, fl.logits[0], fl.logits[1]
        );
        md += "## Where it starts\n\n";
        let mut emb: Vec<(&String, &Vec<f32>)> = roots.iter().filter(|(r, _)| r.starts_with("embedding")).map(|(r, v)| (*r, v)).collect();
        emb.sort_by(|a, b| diff(b.1).abs().total_cmp(&diff(a.1).abs()));
        let emb_total: f32 = emb.iter().map(|(_, v)| diff(v)).sum();
        for (r, v) in emb.iter().take(6) {
            md += &format!("- {r}: {:+.3} ({})\n", diff(v), pct1(diff(v) / total));
        }
        md += &format!(
            "- together the token embeddings: {:+.3} ({}); the rest, {:+.3} ({}), comes from what the switched-off parts give anyway (their means over the references, the biases): the computation every prompt of this kind shares\n\n",
            emb_total,
            pct1(emb_total / total),
            total - emb_total,
            pct1((total - emb_total) / total)
        );
        md += "## The steps\n\n";
        let mut small = 0usize;
        let mut why_dev = 0f32;
        let mut steps_json: Vec<Value> = Vec::new();
        for (k, u) in units_o.iter().enumerate() {
            let v = &unit_out[u];
            let (l, p) = u.lp();
            let n_at = unit_atoms.get(u).map_or(0, |x| x.len());
            let mut ins: Vec<(FNode, Vec<f32>)> = unit_in.get(u).cloned().unwrap_or_default();
            ins.sort_by(|a, b| diff(&b.1).abs().total_cmp(&diff(&a.1).abs()));
            let voice = fl.unit_voices.get(u);
            let off = unit_fx.get(u);
            steps_json.push(json!({"step": k + 1, "unit": u.name(), "layer": l, "position": p, "token": m.tok.piece(e.ids[p]), "multiplications": n_at,
                "passes_on": flow_json(v), "share_of_margin": share(v),
                "reads": ins.iter().take(6).map(|(n2, x)| json!({"from": fnode_name(n2), "flow": diff(x)})).collect::<Vec<_>>(),
                "raises": voice.map(|(up, _)| up.iter().map(|(t, x)| json!({"token": m.tok.piece(*t), "logit": x})).collect::<Vec<_>>()),
                "lowers": voice.map(|(_, dn)| dn.iter().map(|(t, x)| json!({"token": m.tok.piece(*t), "logit": x})).collect::<Vec<_>>()),
                "switched_off_alone": off.map(|o| out_json(m, o)), "file": u.file(wl, wp)}));
            let steps_len = steps_json.len();
            if (diff(v) / total).abs() < 0.005 {
                small += 1;
                continue;
            }
            md += &format!("### {}. Layer {l}, position {p} ({:?}): {} – {} multiplications\n", k + 1, m.tok.piece(e.ids[p]), u.name(), n_at);
            md += &format!("- reads: {}\n", ins.iter().take(4).map(|(n2, x)| format!("{} {:+.3}", fnode_name(n2), diff(x))).collect::<Vec<_>>().join(" · "));
            md += &format!("- passes on **{:+.3}** of the margin ({}): {tgt} {:+.3}, {riv} {:+.3}\n", diff(v), pct1(diff(v) / total), v[0], v[1]);
            if let Some((up, dn)) = voice {
                md += &format!(
                    "- what it adds to the output: raises {} · lowers {}\n",
                    up.iter().take(5).map(|(t, _)| q(*t)).collect::<Vec<_>>().join(", "),
                    dn.iter().take(5).map(|(t, _)| q(*t)).collect::<Vec<_>>().join(", ")
                );
            }
            // the decision of the step: where a head looks, how strongly a neuron fires
            let li = &tr.inputs[l];
            let decision: Option<String> = match *u {
                Unit::Head { h, .. } => {
                    let nh = g.layers[l].nh;
                    let pr = &li.probs[(p * nh + h) * t..(p * nh + h + 1) * t];
                    let mut ks: Vec<usize> = g.keys(l, p).collect();
                    ks.sort_by(|a, b| pr[*b].total_cmp(&pr[*a]));
                    Some(format!("looks at {}", ks.iter().take(4).map(|&j| format!("[{j}] {:?} {:.0}%", m.tok.piece(e.ids[j]), pr[j] * 100.0)).collect::<Vec<_>>().join(" · ")))
                }
                Unit::Neuron { n, .. } => {
                    let ff = g.layers[l].ff;
                    let (a_v, u_v) = (li.act[p * ff + n as usize], li.up[p * ff + n as usize]);
                    if g.layers[l].gated {
                        let gv = li.gate[p * ff + n as usize];
                        Some(format!("fires a = {a_v:+.3} (gate {gv:+.3} → act {:+.3}, up {u_v:+.3})", if u_v != 0.0 { a_v / u_v } else { 0.0 }))
                    } else {
                        Some(format!("fires a = {a_v:+.3} (pre-activation {u_v:+.3})"))
                    }
                }
                _ => None,
            };
            if let Some(dsc) = &decision {
                md += &format!("- decision: {dsc}\n");
                steps_json[steps_len - 1]["decision"] = json!(dsc);
            }
            // why: the decision split exactly over the multiplications that make it, grouped by their sources
            let src_name = |i: usize| -> String { atoms[i].unit(g).name() };
            let rest_name = |name: &str, pos: usize| -> String { format!("{name} [{pos}]") };
            let fmt_parts = |parts: &[(String, f32)], k: usize| -> String { parts.iter().take(k).map(|(n2, v)| format!("{n2} {v:+.3}")).collect::<Vec<_>>().join(" · ") };
            let sorted = |mut acc: HashMap<String, f32>| -> Vec<(String, f32)> {
                let mut v: Vec<(String, f32)> = acc.drain().collect();
                v.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
                v
            };
            match *u {
                Unit::Neuron { n, .. } => {
                    let lg = &g.layers[l];
                    let ff = lg.ff;
                    let f = act_fn(m.blocks[l].mlp.act);
                    let (a_v, uv) = (li.act[p * ff + n as usize], li.up[p * ff + n as usize]);
                    let ups = by_row.get(&(l, p, Site::Up, n)).cloned().unwrap_or_default();
                    let gates = by_row.get(&(l, p, Site::Gate, n)).cloned().unwrap_or_default();
                    // a = φ·g·u split symmetrically (gate atoms × φu/2, up atoms × φg/2) – or φ·u without a gate
                    let (kg, ku, rest_a) = if lg.gated {
                        let gv = li.gate[p * ff + n as usize];
                        let phi = ratio(f, gv);
                        let (kg, ku) = (phi * 0.5 * uv, phi * 0.5 * gv);
                        (kg, ku, kg * cx.row_ablated(l, p, Site::Gate, n) + ku * (cx.row_ablated(l, p, Site::Up, n) + cx.bias(l, Site::Up, n as usize)))
                    } else {
                        let ku = ratio(f, uv);
                        (0.0, ku, ku * (cx.row_ablated(l, p, Site::Up, n) + cx.bias(l, Site::Up, n as usize)))
                    };
                    let mut by_src: HashMap<String, f32> = HashMap::new();
                    let mut singles: Vec<(String, f32)> = Vec::new();
                    for (list, kk2) in [(&gates, kg), (&ups, ku)] {
                        for &t in list.iter() {
                            singles.push((atoms[t].name(), kk2 * infos[t].value));
                            for s2 in &infos[t].sources {
                                *by_src.entry(src_name(s2.from)).or_insert(0.0) += kk2 * s2.c;
                            }
                            for (name, r) in &infos[t].rest {
                                *by_src.entry(rest_name(name, p)).or_insert(0.0) += kk2 * r;
                            }
                        }
                    }
                    *by_src.entry(format!("switched-off gate/up atoms and bias of this neuron [{p}]")).or_insert(0.0) += rest_a;
                    let parts = sorted(by_src);
                    let tot: f32 = parts.iter().map(|x| x.1).sum();
                    why_dev = why_dev.max(cx.check(tot, a_v, parts.iter().map(|x| x.1.abs()).sum(), || format!("why-split of {}", u.name())));
                    singles.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
                    md += &format!("- why it fires: a = {a_v:+.3} = {} …\n", fmt_parts(&parts, 5));
                    md += &format!("  - strongest single multiplications: {}\n", fmt_parts(&singles, 4));
                    steps_json[steps_len - 1]["why"] = json!({"activation": a_v, "by_source": parts.iter().take(12).map(|(n2, v)| json!({"source": n2, "share": v})).collect::<Vec<_>>(),
                        "strongest_multiplications": singles.iter().take(12).map(|(n2, v)| json!({"atom": n2, "share": v})).collect::<Vec<_>>()});
                }
                Unit::Head { h, .. } => {
                    let lg = &g.layers[l];
                    let nh = lg.nh;
                    let sc = m.blocks[l].attn.scale;
                    let pr = &li.probs[(p * nh + h) * t..(p * nh + h + 1) * t];
                    let mut ks: Vec<usize> = g.keys(l, p).collect();
                    ks.sort_by(|a, b| pr[*b].total_cmp(&pr[*a]));
                    // check: the scores as split here, through scale, softcap and softmax, give the weights the model used
                    {
                        let gr = h / (nh / lg.nkv);
                        let hbc = cx.cell(l, p).heads.get(h);
                        let score_of = |j: usize| -> f32 {
                            let (qv, kv) = (&li.q[p * nh * lg.hd + h * lg.hd..][..lg.hd], &li.k[j * lg.nkv * lg.hd + gr * lg.hd..][..lg.hd]);
                            let raw: f32 = match (hbc, cx.pm(l, p), cx.pm(l, j)) {
                                (Some(hb), Some(pq), Some(pk)) => (0..lg.hd).map(|x| if hb.qk.atom_on(j as u32, x as u32) { qv[x] * kv[x] } else { pq.q[h * lg.hd + x] * pk.k[gr * lg.hd + x] }).sum(),
                                (Some(hb), None, _) => (0..lg.hd).filter(|&x| hb.qk.atom_on(j as u32, x as u32)).map(|x| qv[x] * kv[x]).sum(),
                                _ => qv.iter().zip(kv).map(|(a, b)| a * b).sum(),
                            };
                            let mut z = raw * sc;
                            if let Some(cap) = m.blocks[l].attn.softcap {
                                z = (z / cap).tanh() * cap;
                            }
                            z
                        };
                        let keys: Vec<usize> = g.keys(l, p).collect();
                        let zs: Vec<f32> = keys.iter().map(|&j| score_of(j)).collect();
                        let mx = zs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let den: f32 = zs.iter().map(|z| (z - mx).exp()).sum();
                        for (jj, &j) in keys.iter().enumerate() {
                            why_dev = why_dev.max(cx.check((zs[jj] - mx).exp() / den, pr[j], 1.0, || format!("attention weight of {} on [{j}]", u.name())));
                        }
                    }
                    let mut whys = Vec::new();
                    for &j in ks.iter().take(3) {
                        // the score of key j: its products q_e·k_e (on: real, off: their means), split over the
                        // sources of the key (q held fixed) and of the query (k held fixed)
                        let mut kside: HashMap<String, f32> = HashMap::new();
                        let mut qside: HashMap<String, f32> = HashMap::new();
                        let mut on_sum = 0f32;
                        for &qa in qk_by.get(&(l, p, h, j)).map(|v| &v[..]).unwrap_or(&[]) {
                            on_sum += infos[qa].value;
                            for s2 in &infos[qa].sources {
                                let side = if s2.via.starts_with('k') { &mut kside } else { &mut qside };
                                *side.entry(src_name(s2.from)).or_insert(0.0) += sc * s2.c;
                            }
                            for (name, r) in &infos[qa].rest {
                                let (side, pos) = if name.starts_with("k:") { (&mut kside, j) } else { (&mut qside, p) };
                                *side.entry(rest_name(name, pos)).or_insert(0.0) += sc * r;
                            }
                        }
                        let gr = h / (nh / lg.nkv);
                        let (qv, kv) = (&li.q[p * nh * lg.hd + h * lg.hd..][..lg.hd], &li.k[j * lg.nkv * lg.hd + gr * lg.hd..][..lg.hd]);
                        let raw: f32 = qv.iter().zip(kv).map(|(a, b)| a * b).sum();
                        // with every product on, raw is the score; the switched-off products count with their means
                        let off_part = match (cx.pm(l, p), cx.pm(l, j), cx.cell(l, p).heads.get(h)) {
                            (Some(pq), Some(pk), Some(hb)) => (0..lg.hd).filter(|&x| !hb.qk.atom_on(j as u32, x as u32)).map(|x| pq.q[h * lg.hd + x] * pk.k[gr * lg.hd + x]).sum::<f32>(),
                            (None, _, Some(hb)) => { let _ = hb; 0.0 }
                            _ => 0.0,
                        };
                        let hb_cut = cx.cell(l, p).heads.get(h).is_some_and(|hb| !hb.qk.is_empty());
                        let score = sc * if hb_cut { on_sum + off_part } else { raw };
                        let kparts = sorted(kside);
                        let qparts = sorted(qside);
                        md += &format!(
                            "- why it looks at [{j}] {:?} ({:.0}%): score {score:+.3} (switched-off products {:+.3}) · key side: {} · query side: {}\n",
                            m.tok.piece(e.ids[j]),
                            pr[j] * 100.0,
                            sc * off_part * f32::from(u8::from(hb_cut)),
                            fmt_parts(&kparts, 3),
                            fmt_parts(&qparts, 3)
                        );
                        whys.push(json!({"key": j, "token": m.tok.piece(e.ids[j]), "weight": pr[j], "score": score,
                            "key_side": kparts.iter().take(8).map(|(n2, v)| json!({"source": n2, "share": v})).collect::<Vec<_>>(),
                            "query_side": qparts.iter().take(8).map(|(n2, v)| json!({"source": n2, "share": v})).collect::<Vec<_>>()}));
                    }
                    steps_json[steps_len - 1]["why"] = json!(whys);
                }
                _ => {}
            }
            if let Some(o) = off {
                md += &format!("- switched off alone: {:?} {:.0}% (KL {:.3})\n", m.tok.piece(o.top[0].0), o.top[0].1 * 100.0, o.kl);
            }
            md += "\n";
        }
        if small > 0 {
            md += &format!("({small} more steps pass on less than 0.5% of the margin each; all of them are in steps.json, and every multiplication in atoms_in_order.jsonl.)\n\n");
        }
        md += &format!("## The end\n\n{tgt} {:.3} − {riv} {:.3} = {:+.3}: the sum of every root and every step's direct part, exact to {:.1e}. Every neuron's activation is the sum of its why-parts, and every head's weights follow from its split scores, exact to {:.1e}.\n", fl.logits[0], fl.logits[1], total, fl.dev_roots, why_dev);
        std::fs::write(dir.join("steps.md"), md).map_err(|x| format!("{}: {x}", dir.display()))?;
        write_json(dir, "steps.json", &json!({"what": "every unit of the circuit in the order the model computes it, with what it reads, what it passes on of the margin target − rival (exact), and what its contribution raises and lowers over the whole vocabulary", "margin": total, "steps": steps_json}))?;
    }
    // ------------------------------------------------------------ bias rows of the neurons that carry the most flow
    let mut bias_list: Vec<Value> = Vec::new();
    if let (Some(nb), true) = (r.o.bias, !unit_out.is_empty()) {
        let mut ns: Vec<(Unit, f32)> = unit_out.iter().filter(|(u, _)| matches!(u, Unit::Neuron { .. })).map(|(u, v)| (*u, diff(v))).collect();
        ns.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
        let mut seen: BTreeSet<(usize, u32)> = BTreeSet::new();
        let trb = crate::bias::record(m, e.ids, focus);
        let bdir = dir.join("bias");
        std::fs::create_dir_all(&bdir).map_err(|x| format!("{}: {x}", bdir.display()))?;
        for (u, fl_diff) in ns.iter() {
            if seen.len() >= nb {
                break;
            }
            if let Unit::Neuron { l, n, .. } = *u {
                if !seen.insert((l, n)) {
                    continue;
                }
                let v = crate::bias::analyze(m, e.ids, focus, &trb, l, n as usize, &bdir, r.o.bias_top)?;
                bias_list.push(json!({"neuron": v["neuron"], "folder": format!("bias/{}", v["folder"].as_str().unwrap_or("")), "flow_through_it": fl_diff,
                    "activation_range": v["repertoire"], "closure": v["checks"]["received_add_up_to_p_and_q"]}));
            }
        }
    }
    // ------------------------------------------------------------ circuit.json and README.md
    let s = &e.st;
    let comps_json: Vec<Value> = {
        let mut v: Vec<(&(usize, usize, &str), &Out)> = comp_fx.iter().collect();
        // ties (equal or NaN KL) by the key, so the order is the same in every run
        v.sort_by(|a, b| b.1.kl.total_cmp(&a.1.kl).then(a.0.cmp(b.0)));
        v.iter().map(|((l, p, k), o)| json!({"component": format!("S{l} {k} [{p}]"), "layer": l, "position": p, "kind": k, "switched_off_alone_in_union": out_json(m, o)})).collect()
    };
    let mut units_sorted = unit_rows.clone();
    units_sorted.sort_by(|a, b| {
        let ka = a["switched_off_alone_in_union"]["kl"].as_f64().unwrap_or(0.0);
        let kb = b["switched_off_alone_in_union"]["kl"].as_f64().unwrap_or(0.0);
        kb.total_cmp(&ka)
    });
    write_json(
        dir,
        "circuit.json",
        &json!({
            "schema": "apollon-circuit-folder/1",
            "model": m.name, "tokens": e.ids.iter().enumerate().map(|(i, &id)| json!({"position": i, "id": id, "text": m.tok.piece(id)})).collect::<Vec<_>>(),
            "focus": focus, "target": {"id": e.ys[0], "text": m.tok.piece(e.ys[0])}, "rival": {"id": e.ys[1], "text": m.tok.piece(e.ys[1])},
            "settings": {"criterion_kl": r.o.kl, "depth": r.o.depth, "ablation": if r.o.mean { "mean" } else { "zero" }, "ablation_text": ablation_txt(r),
                "norms": if r.o.frozen { "frozen" } else { "live" }, "reference_prompts": r.refs.len(), "max_paths": r.o.paths},
            "full_model": out_json(m, &r.full_out),
            "union_of_all_paths": {"atoms_on": n_on, "atoms_total": r.total_atoms, "alone": out_json(m, &r.circuit_out), "everything_else": out_json(m, &r.comp_out)},
            "paths": r.paths.iter().enumerate().map(|(i, pa)| json!({"path": i + 1, "atoms": pa.atoms, "leaves": pa.leaves,
                "searched_against": out_json(m, &pa.reference), "alone_vs_its_reference": out_json(m, &pa.alone), "alone_vs_full_model": out_json(m, &pa.alone_full),
                "union_without_it": r.fx.without_path.get(i).map(|o| out_json(m, o)), "kl_reference_to_baseline": pa.signal, "tests": pa.tests, "passes": pa.passes, "seconds": pa.secs,
                "levels": pa.logs.iter().map(|lg| json!({"level": level_name(lg.level), "kept": lg.kept, "atoms": lg.atoms, "kl": lg.kl, "tests": lg.tests})).collect::<Vec<_>>()})).collect::<Vec<_>>(),
            "expansion_stopped": r.stop,
            "components": comps_json,
            "units": units_sorted,
            "layers": layers_json,
            "atoms_listed": detailed, "atoms_listed_note": if detailed { Value::Null } else { json!(format!("{n_on} atoms are on (more than {MAX_ATOMS}): only units and components are written")) },
            "flow": flow_checks,
            "bias_rows": bias_list,
            "checks": {"trace_closure": closure_v, "trace_closure_worst_at": worst_at, "flow_worst_deviation": flow_dev, "empty_cut_bit_identical": r.empty_identical,
                "batch_vs_standalone_dkl": s.batch_dev, "clean_rows_dkl": s.clean_dev, "dead_atoms_output_bit_identical": s.dead_identical,
                "tests": s.tests, "passes": s.passes, "seconds": s.secs},
        }),
    )?;
    std::fs::write(dir.join("README.md"), README).map_err(|x| format!("{}: {x}", dir.display()))?;
    Ok(detailed.then_some((closure, flow_dev)))
}

const README: &str = "# apollon circuit folder

Written by `apollon --circuit` (self-expansion). Open the files from coarse to fine:

* `circuit.json` – the prompt, the outputs (full model, all paths together, everything except them), the
  paths of the self-expansion (path k = the minimal circuit of what is left without paths 1 … k−1), every
  component and every unit with its effect when switched off alone against the union, and the checks.
* `trace.json` – the circuit backwards from the output: for every unit how many links it is away from the
  output, which units it reads and which it feeds (summed exact contributions of the atom edges between
  them), and the atoms that write directly into the target's logit.
* `flow.json` – the output split over every atom (attention weights, activation ratios and norm scales held
  at their values, so everything between atoms is linear): the main path from the output back to the
  tokens, every unit and edge with the exact part of target − rival it carries, the roots (embeddings,
  switched-off atoms, biases) – all roots together give the logits exactly –, the strongest atoms with the
  tokens their flow raises and lowers.
* `steps.md` / `steps.json` – the protocol: every head, K/V group, neuron and PLE dimension of the circuit in
  the order the model computes it, with what it reads, the exact part of the margin target − rival it
  passes on, and the tokens its contribution raises and lowers over the whole vocabulary.
* `atoms_in_order.jsonl` – every multiplication in the order the model computes it (one JSON per line):
  value, value when off, flow, its strongest sources and the rest.
* `flows_residual.f32` / `.json` – the matrix: one row per atom, its flow into the stream at the end (float32);
  any token's part follows from the output matrix and the frozen final norm.
* `bias/S<l>_<i>/` (with --circuit-bias) – the bias rows of the neurons that carry the most flow: what one unit
  of each neuron's activation gives every later neuron, Q/K/V projection and token, what every source gives
  it, the range its activation allows, and at every position what it actually receives and gives (bias.json,
  full rows as float32).
* `layers/SLL/layer.json` – one layer: its components and units per position.
* `layers/SLL/pPP/<attention|kv|mlp|ple>/<unit>.json` – one head, K/V group, neuron or PLE dimension at one
  position with every scalar product that is on in it.

Every atom lists:

* `value` – its value in the pass of the union of all paths; `value_when_off` – what it becomes when switched
  off (its mean, or 0 with zero ablation);
* `path` and `leaf` – the path it belongs to and the effect of switching its leaf off alone, against the path
  and against the union;
* `sources` – the atoms its value is made of, each with its exact contribution (`via` names the factor that
  is traced: residual stream, q, k, v, mixing, up); the 32 largest by name, the others summed up in
  `sources_not_listed`;
* `nonlinear_sources` – what it depends on through a non-linearity (softmax scores, gate pre-activations);
* `rest` – the parts that are not atoms: embedding, switched-off atoms, biases, norm terms;
* `closure` – |value − (sources + rest)| relative to the size of the parts; `trace_closure` in circuit.json is
  the largest one over every decomposition (and over the residual stream at every stage);
* `to_output` – for writers at the focus: their contribution to the target's and the rival's logit;
* `flow` – what of the output passes through it: onto the target's and the rival's logit (see flow.json);
* `used_by` – the atoms that read it, with the contribution (the 32 largest; the others in
  `used_by_not_listed`); `used_nonlinearly_by` – the atoms that depend on it through a non-linearity.

Above 250,000 atoms on, only units and components are written (`atoms_listed` in circuit.json says so).
";

#[cfg(test)]
mod tests {
    use super::*;

    /// rope_row must reproduce Rope::apply (as a matrix) for both pair layouts and partial rotation
    #[test]
    fn rope_rows_match_the_rotation() {
        for neox in [true, false] {
            let r = Rope { inv_freq: vec![1.0, 0.37, 0.05], neox, pos_scale: 1.0 };
            let hd = 8; // 6 dims rotated, 2 not
            let v: Vec<f32> = (0..hd).map(|i| (i as f32 * 0.7).sin() + 0.3).collect();
            for pos in [0usize, 3, 11] {
                // reference: the rotation as the model applies it
                let half = r.inv_freq.len();
                let mut w = v.clone();
                for i in 0..half {
                    let (s, c) = (pos as f32 * r.inv_freq[i]).sin_cos();
                    let (i0, i1) = if neox { (i, i + half) } else { (2 * i, 2 * i + 1) };
                    let (x0, x1) = (w[i0], w[i1]);
                    w[i0] = x0 * c - x1 * s;
                    w[i1] = x1 * c + x0 * s;
                }
                for e in 0..hd {
                    let got: f32 = rope_row(Some(&r), pos, e).iter().map(|&(e2, c)| c * v[e2]).sum();
                    assert!((got - w[e]).abs() < 1e-6, "neox {neox} pos {pos} dim {e}: {got} vs {}", w[e]);
                }
            }
        }
    }
}

// ====================================================================== the flow of the output through every atom

#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
enum FNode {
    /// a part that is no atom: embedding, switched-off atoms, biases, norm terms (with the position it enters)
    Root(String),
    U(Unit),
    Output,
}

struct Flow {
    /// the tokens followed: target, rival, then the strongest of the union's output
    toks: Vec<u32>,
    /// their logits before a softcap, computed directly: W_U · N(x at the end)
    logits: Vec<f32>,
    /// per atom its flow onto each token's logit [n][K]
    atom: Vec<f32>,
    /// summed flow along every edge: root → unit, unit → unit, unit → output [K]
    edges: HashMap<(FNode, FNode), Vec<f32>>,
    /// all roots together, per token, and the largest relative deviation from the logits
    roots_sum: Vec<f32>,
    dev_roots: f32,
    /// every unit: flow in (from roots and other units) = flow out (to other units and the output)
    dev_units: (f32, String),
    /// the residual-space matrix [n][d]: the flow of every atom into the stream at the focus (before the final
    /// norm); all roots plus the atoms' direct parts give that stream exactly. None when too large.
    matrix: Option<Vec<f32>>,
    dev_matrix: f32,
    /// the strongest atoms (by flow onto target − rival) with the tokens their flow raises and lowers most
    voices: Vec<(usize, Vec<(u32, f32)>, Vec<(u32, f32)>)>,
    /// every unit: the tokens its outflow raises and lowers most over the whole vocabulary
    unit_voices: HashMap<Unit, (Vec<(u32, f32)>, Vec<(u32, f32)>)>,
}

/// the flow matrix is written when atoms × d stays below this many floats
const MAX_MATRIX: usize = 64 << 20;

#[allow(clippy::too_many_arguments)]
fn flow_split(atoms: &[A], infos: &[Info], cx: &Ctx, e: &Engine, m: &Model, g: &Geo, tr: &crate::model::Trace, fin: &Snap, fs: f32, fnorm: Option<&Norm>, focus: usize) -> Flow {
    let d = g.d;
    let n = atoms.len();
    let lm = m.lm_head();
    // the tokens followed
    let mut toks = vec![e.ys[0], e.ys[1]];
    if !tr.logits.is_empty() {
        let mut idx: Vec<usize> = (0..tr.logits.len()).collect();
        let kk = 24.min(idx.len());
        idx.select_nth_unstable_by(kk - 1, |a, b| tr.logits[*b].total_cmp(&tr.logits[*a]));
        let mut top = idx[..kk].to_vec();
        top.sort_by(|a, b| tr.logits[*b].total_cmp(&tr.logits[*a]));
        for i in top {
            if toks.len() >= 20 {
                break;
            }
            if !toks.contains(&(i as u32)) {
                toks.push(i as u32);
            }
        }
    }
    let kk = toks.len();
    // u_k: how token k's logit (before a softcap) moves with the stream at the focus, the final norm frozen
    let row_of = |t: u32| -> Vec<f32> {
        let mut r = vec![0f32; d];
        lm.row(t as usize, &mut r);
        r
    };
    let mut u = vec![0f32; kk * d];
    for (ki, &t) in toks.iter().enumerate() {
        let mut r = row_of(t);
        if let Some(nm) = fnorm {
            for c in 0..d {
                r[c] *= fs * nm.w[c];
            }
            if !nm.rms {
                let mu = r.iter().map(|x| *x as f64).sum::<f64>() as f32 / d as f32;
                r.iter_mut().for_each(|v| *v -= mu);
            }
        }
        u[ki * d..(ki + 1) * d].copy_from_slice(&r);
    }
    let mut hfin = fin.x.clone();
    if let Some(nm) = fnorm {
        nm.apply_scale(&fin.x, &mut hfin, fs);
    }
    let logits: Vec<f32> = toks.iter().map(|&t| row_of(t).iter().zip(&hfin).map(|(a, b)| a * b).sum()).collect();
    // later stages first: every source of an atom comes earlier
    let key = |a: &A| -> (usize, u8) {
        match *a {
            A::W { l, s, .. } => (
                l,
                match s {
                    Site::Q | Site::K | Site::V => 0,
                    Site::O => 3,
                    Site::Gate | Site::Up => 4,
                    Site::Down => 5,
                    Site::PleIn | Site::PleGate => 6,
                    Site::PleProj => 7,
                },
            ),
            A::Qk { l, .. } => (l, 1),
            A::Av { l, .. } => (l, 2),
        }
    };
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(key(&atoms[i])));
    // writers at the focus write straight into the stream at the end
    let direct = |i: usize| -> Option<(usize, f32)> {
        if let A::W { l, p, s, a: o2, .. } = atoms[i] {
            let kind = match s {
                Site::O => 0u8,
                Site::Down => 1,
                Site::PleProj => 2,
                _ => return None,
            };
            if p == focus {
                let sc = cx.visible(Stage::Final, l, kind).unwrap_or(1.0);
                return Some((o2 as usize, cx.post(l, kind, focus, o2 as usize) * sc));
            }
        }
        None
    };
    // the sensitivities, backwards: S_src += k · S_dst over every edge (Q·K products pass nothing on)
    let back = |dim: usize, init: &dyn Fn(usize, f32, &mut [f32])| -> Vec<f32> {
        let mut sv = vec![0f32; n * dim];
        for &i in &order {
            if let Some((o2, f)) = direct(i) {
                init(o2, f, &mut sv[i * dim..(i + 1) * dim]);
            }
            if matches!(atoms[i], A::Qk { .. }) {
                continue;
            }
            let sd: Vec<f32> = sv[i * dim..(i + 1) * dim].to_vec();
            if sd.iter().all(|x| *x == 0.0) {
                continue;
            }
            for s in &infos[i].sources {
                let dst = &mut sv[s.from * dim..(s.from + 1) * dim];
                for (a, b) in dst.iter_mut().zip(&sd) {
                    *a += s.k * b;
                }
            }
        }
        sv
    };
    let sk = back(kk, &|o2, f, row| {
        for ki in 0..kk {
            row[ki] += f * u[ki * d + o2];
        }
    });
    let pos_of = |a: &A| match *a {
        A::W { p, .. } | A::Qk { p, .. } | A::Av { p, .. } => p,
    };
    // flows, edges, roots
    let mut atom = vec![0f32; n * kk];
    let mut edges: HashMap<(FNode, FNode), Vec<f32>> = HashMap::new();
    let mut add = |a: FNode, b: FNode, f: f32, sv: &[f32]| {
        let x = edges.entry((a, b)).or_insert_with(|| vec![0f32; kk]);
        for (p, q) in x.iter_mut().zip(sv) {
            *p += f * q;
        }
    };
    let mut roots_sum = vec![0f32; kk];
    let mut roots_abs = vec![0f32; kk];
    for i in 0..n {
        if matches!(atoms[i], A::Qk { .. }) {
            continue;
        }
        let si = &sk[i * kk..(i + 1) * kk];
        for ki in 0..kk {
            atom[i * kk + ki] = infos[i].value * si[ki];
        }
        let ui = FNode::U(atoms[i].unit(g));
        if let Some((o2, f)) = direct(i) {
            let sv: Vec<f32> = (0..kk).map(|ki| infos[i].value * f * u[ki * d + o2]).collect();
            add(ui.clone(), FNode::Output, 1.0, &sv);
        }
        for s in &infos[i].sources {
            let us = FNode::U(atoms[s.from].unit(g));
            if us != ui {
                add(us, ui.clone(), s.k * infos[s.from].value, si);
            }
        }
        let pos = pos_of(&atoms[i]);
        for (name, r) in &infos[i].rest {
            if *r == 0.0 {
                continue;
            }
            add(FNode::Root(format!("{name} [{pos}]")), ui.clone(), *r, si);
            for ki in 0..kk {
                roots_sum[ki] += r * si[ki];
                roots_abs[ki] += (r * si[ki]).abs();
            }
        }
    }
    // the parts of the stream at the end that are no atom
    for (name, part) in [("embedding", &fin.emb), ("switched-off writers", &fin.abl), ("biases of writers", &fin.bias)] {
        let sv: Vec<f32> = (0..kk).map(|ki| (0..d).map(|c| u[ki * d + c] * part[c]).sum()).collect();
        if sv.iter().any(|x| *x != 0.0) {
            add(FNode::Root(format!("{name} [{focus}], straight into the stream at the end")), FNode::Output, 1.0, &sv);
        }
        for ki in 0..kk {
            roots_sum[ki] += sv[ki];
            roots_abs[ki] += sv[ki].abs();
        }
    }
    if let Some(b) = fnorm.and_then(|nm| nm.b.as_ref()) {
        let sv: Vec<f32> = toks.iter().map(|&t| row_of(t).iter().zip(b).map(|(a, c)| a * c).sum()).collect();
        add(FNode::Root("bias of the final norm".into()), FNode::Output, 1.0, &sv);
        for ki in 0..kk {
            roots_sum[ki] += sv[ki];
            roots_abs[ki] += sv[ki].abs();
        }
    }
    let dev_roots = (0..kk).map(|ki| (roots_sum[ki] - logits[ki]).abs() / roots_abs[ki].max(logits[ki].abs()).max(1e-6)).fold(0f32, f32::max);
    // every unit passes on exactly what it receives
    let mut flow_in: HashMap<Unit, Vec<f32>> = HashMap::new();
    let mut flow_out: HashMap<Unit, Vec<f32>> = HashMap::new();
    for ((a, b), v) in &edges {
        if let FNode::U(ub) = b {
            let x = flow_in.entry(*ub).or_insert_with(|| vec![0f32; kk]);
            x.iter_mut().zip(v).for_each(|(p, q)| *p += q);
        }
        if let FNode::U(ua) = a {
            let x = flow_out.entry(*ua).or_insert_with(|| vec![0f32; kk]);
            x.iter_mut().zip(v).for_each(|(p, q)| *p += q);
        }
    }
    let mut dev_units = (0f32, String::new());
    for (un, vin) in &flow_in {
        let vout = flow_out.get(un).cloned().unwrap_or_else(|| vec![0f32; kk]);
        for ki in 0..kk {
            let dev = (vin[ki] - vout[ki]).abs() / vin[ki].abs().max(vout[ki].abs()).max(1e-3 * logits[ki].abs()).max(1e-6);
            if dev > dev_units.0 {
                dev_units = (dev, format!("{} ({})", un.name(), m.tok.piece(toks[ki])));
            }
        }
    }
    // the residual-space matrix
    let mut unit_voices = HashMap::new();
    let (matrix, dev_matrix) = if n * d <= MAX_MATRIX {
        let mut sd = back(d, &|o2, f, row| row[o2] += f);
        // what flows out of every unit, as a vector in the stream at the end
        let mut uvec: BTreeMap<Unit, Vec<f32>> = BTreeMap::new();
        for i in 0..n {
            if matches!(atoms[i], A::Qk { .. }) {
                continue;
            }
            let ui = atoms[i].unit(g);
            if let Some((o2, f)) = direct(i) {
                uvec.entry(ui).or_insert_with(|| vec![0f32; d])[o2] += infos[i].value * f;
            }
            for s2 in &infos[i].sources {
                let us = atoms[s2.from].unit(g);
                if us != ui {
                    let w = s2.k * infos[s2.from].value;
                    let v = uvec.entry(us).or_insert_with(|| vec![0f32; d]);
                    for c in 0..d {
                        v[c] += w * sd[i * d + c];
                    }
                }
            }
        }
        if !uvec.is_empty() {
            let units: Vec<Unit> = uvec.keys().copied().collect();
            let mut x = vec![0f32; units.len() * d];
            for (r, un) in units.iter().enumerate() {
                let f = &uvec[un];
                let out = &mut x[r * d..(r + 1) * d];
                match fnorm {
                    Some(nm) => {
                        let mu = if nm.rms { 0.0 } else { f.iter().map(|v| *v as f64).sum::<f64>() as f32 / d as f32 };
                        for c in 0..d {
                            out[c] = (f[c] - mu) * fs * nm.w[c];
                        }
                    }
                    None => out.copy_from_slice(f),
                }
            }
            let lg = lm.matmul(&x, units.len());
            let vr = lm.rows;
            for (r, un) in units.iter().enumerate() {
                let row = &lg[r * vr..(r + 1) * vr];
                let mut idx: Vec<usize> = (0..vr).collect();
                idx.sort_by(|a, b| row[*b].total_cmp(&row[*a]));
                let up: Vec<(u32, f32)> = idx[..6].iter().map(|&t| (t as u32, row[t])).collect();
                let down: Vec<(u32, f32)> = idx[vr - 6..].iter().rev().map(|&t| (t as u32, row[t])).collect();
                unit_voices.insert(*un, (up, down));
            }
        }
        let mut rsum = vec![0f64; d];
        let mut rabs = vec![0f64; d];
        for i in 0..n {
            if matches!(atoms[i], A::Qk { .. }) {
                sd[i * d..(i + 1) * d].iter_mut().for_each(|x| *x = 0.0);
                continue;
            }
            for (name, r) in &infos[i].rest {
                let _ = name;
                for c in 0..d {
                    let v = (*r * sd[i * d + c]) as f64;
                    rsum[c] += v;
                    rabs[c] += v.abs();
                }
            }
            let val = infos[i].value;
            sd[i * d..(i + 1) * d].iter_mut().for_each(|x| *x *= val);
        }
        let mut dev = 0f32;
        for c in 0..d {
            let tot = rsum[c] + (fin.emb[c] + fin.abl[c] + fin.bias[c]) as f64;
            let sc = rabs[c] + (fin.emb[c].abs() + fin.abl[c].abs() + fin.bias[c].abs()) as f64;
            dev = dev.max(((tot - fin.x[c] as f64).abs() / sc.max(fin.x[c].abs() as f64).max(1e-6)) as f32);
        }
        (Some(sd), dev)
    } else {
        (None, 0.0)
    };
    // the strongest atoms: what their flow does to every token of the vocabulary
    let mut voices = Vec::new();
    if let Some(mx) = &matrix {
        let mut by: Vec<usize> = (0..n).filter(|&i| !matches!(atoms[i], A::Qk { .. })).collect();
        by.sort_by(|&a, &b| {
            let da = (atom[a * kk] - atom[a * kk + 1]).abs();
            let db = (atom[b * kk] - atom[b * kk + 1]).abs();
            db.total_cmp(&da)
        });
        by.truncate(40);
        let mut x = vec![0f32; by.len() * d];
        for (r, &i) in by.iter().enumerate() {
            let f = &mx[i * d..(i + 1) * d];
            let out = &mut x[r * d..(r + 1) * d];
            match fnorm {
                Some(nm) => {
                    let mu = if nm.rms { 0.0 } else { f.iter().map(|v| *v as f64).sum::<f64>() as f32 / d as f32 };
                    for c in 0..d {
                        out[c] = (f[c] - mu) * fs * nm.w[c];
                    }
                }
                None => out.copy_from_slice(f),
            }
        }
        if !by.is_empty() {
            let lg = lm.matmul(&x, by.len());
            let vr = lm.rows;
            for (r, &i) in by.iter().enumerate() {
                let row = &lg[r * vr..(r + 1) * vr];
                let mut idx: Vec<usize> = (0..vr).collect();
                idx.sort_by(|a, b| row[*b].total_cmp(&row[*a]));
                let up: Vec<(u32, f32)> = idx[..6].iter().map(|&t| (t as u32, row[t])).collect();
                let down: Vec<(u32, f32)> = idx[vr - 6..].iter().rev().map(|&t| (t as u32, row[t])).collect();
                voices.push((i, up, down));
            }
        }
    }
    Flow { toks, logits, atom, edges, roots_sum, dev_roots, dev_units, matrix, dev_matrix, voices, unit_voices }
}
