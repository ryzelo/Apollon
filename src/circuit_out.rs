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
    c: f32,
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
pub(super) fn write(r: &Report, dir: &FsPath, p_full: &[f32], plp_full: f64) -> Result<Option<f32>, String> {
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
                    info.sources.push(Src { from: wi, c: k * cx.post(lw, kind, p, c) * sc * infos_value(&atoms, wi, &cx), via: "residual" });
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
                        info.sources.push(Src { from: k, c: w * val, via: "mixing" });
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
                if lg.gated {
                    let gv = li.gate[p * ff + n as usize];
                    let fg = f(gv);
                    for &k in &up_atoms {
                        info.sources.push(Src { from: k, c: w * fg * infos[k].value, via: "up (gate held fixed)" });
                    }
                    info.rest.push(("switched-off up atoms", w * fg * up_abl));
                    info.rest.push(("up bias", w * fg * up_b));
                    let gate_atoms = by_row.get(&(l, p, Site::Gate, n)).cloned().unwrap_or_default();
                    let g_on: f32 = gate_atoms.iter().map(|&k| infos[k].value).sum();
                    let g_abl = cx.row_ablated(l, p, Site::Gate, n);
                    cx.check(g_on + g_abl, gv, g_on.abs() + g_abl.abs(), || format!("gate of S{l} #{n} [{p}]"));
                    for &k in &gate_atoms {
                        info.nonlinear.push((k, infos[k].value, "gate pre-activation"));
                    }
                } else {
                    for &k in &up_atoms {
                        info.nonlinear.push((k, infos[k].value, "pre-activation"));
                    }
                    info.rest.push(("activation of the neuron (not linear in its inputs)", info.value));
                }
            }
            A::W { l, p, s: Site::PleProj, u: x, .. } => {
                for &k in by_row.get(&(l, p, Site::PleGate, x)).map(|v| &v[..]).unwrap_or(&[]) {
                    info.nonlinear.push((k, infos[k].value, "PLE gate pre-activation"));
                }
                info.rest.push(("gated per-layer input (not linear in its inputs)", info.value));
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
                        info.sources.push(Src { from: ai, c: coef * infos[ai].value, via: "q (k held fixed)" });
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
                        info.sources.push(Src { from: ai, c: coef * infos[ai].value, via: "k (q held fixed)" });
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
                    info.sources.push(Src { from: ai, c: coef * infos[ai].value, via: "v (weight held fixed)" });
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
    let all_units: BTreeSet<Unit> = unit_atoms.keys().copied().chain(unit_fx.keys().copied()).collect();
    for u in &all_units {
        let ids = unit_atoms.get(u).cloned().unwrap_or_default();
        let file = u.file(wl, wp);
        let direct: f32 = ids.iter().filter_map(|&i| infos[i].to_output.map(|(a, b2)| a - b2)).sum();
        let reads: Vec<Value> = {
            let mut v: Vec<(&(Node, Node), &(f64, f64, usize))> = edges.iter().filter(|((_, to), _)| *to == Node::U(*u)).collect();
            v.sort_by(|a, b| b.1 .1.partial_cmp(&a.1 .1).unwrap_or(std::cmp::Ordering::Equal));
            v.iter().map(|((f, _), (s, sa, n))| json!({"unit": node_name(f), "sum": s, "sum_abs": sa, "atom_edges": n})).collect()
        };
        let read_by: Vec<Value> = {
            let mut v: Vec<(&(Node, Node), &(f64, f64, usize))> = edges.iter().filter(|((f, _), _)| *f == Node::U(*u)).collect();
            v.sort_by(|a, b| b.1 .1.partial_cmp(&a.1 .1).unwrap_or(std::cmp::Ordering::Equal));
            v.iter().map(|((_, to), (s, sa, n))| json!({"unit": node_name(to), "sum": s, "sum_abs": sa, "atom_edges": n})).collect()
        };
        let summary = json!({
            "unit": u.name(), "kind": u.kind(), "layer": u.lp().0, "position": u.lp().1,
            "token": m.tok.piece(e.ids[u.lp().1]), "atoms_on": ids.len(), "paths": paths_of_unit(&ids),
            "switched_off_alone_in_union": fx_json(unit_fx.get(u)),
            "direct_to_output_logit_difference": if ids.iter().any(|&i| infos[i].to_output.is_some()) { json!(direct) } else { Value::Null },
            "steps_to_output": depth.get(&Node::U(*u)),
            "reads_from_units": reads, "read_by_units": read_by, "file": file,
        });
        if detailed {
            let atoms_json: Vec<Value> = ids
                .iter()
                .map(|&i| {
                    let inf = &infos[i];
                    let lf = leaf_of[i].map(|li| &r.leaves[li]);
                    let mut src: Vec<&Src> = inf.sources.iter().collect();
                    src.sort_by(|a, b| b.c.abs().partial_cmp(&a.c.abs()).unwrap_or(std::cmp::Ordering::Equal));
                    let mut ub = used_by[i].clone();
                    ub.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(std::cmp::Ordering::Equal));
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
                        "to_output": inf.to_output.map(|(a, b2)| json!({"target_logit": a, "rival_logit": b2, "difference": a - b2})),
                        "used_by": ub_l, "used_by_not_listed": ub_rest,
                        "used_nonlinearly_by": used_nl[i].iter().map(|(k, w)| json!({"atom": inf_name(&atoms, *k), "role": w})).collect::<Vec<_>>(),
                    })
                })
                .collect();
            let mut v = summary.clone();
            v["atoms"] = json!(atoms_json);
            write_json_as(dir, &file, &v, ids.len() <= PRETTY_ATOMS)?;
        } else {
            write_json(dir, &file, &summary)?;
        }
        unit_rows.push(summary);
    }
    fn inf_name(atoms: &[A], i: usize) -> String {
        atoms[i].name()
    }
    // ------------------------------------------------------------ layer files
    let mut layers_json: Vec<Value> = Vec::new();
    for l in 0..nl {
        let units_l: Vec<&Value> = unit_rows.iter().filter(|u| u["layer"].as_u64() == Some(l as u64)).collect();
        let comps_l: Vec<Value> = comp_fx
            .iter()
            .filter(|((cl, _, _), _)| *cl == l)
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
            ins.sort_by(|a, b| b.1 .1.partial_cmp(&a.1 .1).unwrap_or(std::cmp::Ordering::Equal));
            let mut outs: Vec<(&(Node, Node), &(f64, f64, usize))> = edges.iter().filter(|((f, _), _)| f == *n).collect();
            outs.sort_by(|a, b| b.1 .1.partial_cmp(&a.1 .1).unwrap_or(std::cmp::Ordering::Equal));
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
    direct.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(std::cmp::Ordering::Equal));
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
    // ------------------------------------------------------------ circuit.json and README.md
    let s = &e.st;
    let comps_json: Vec<Value> = {
        let mut v: Vec<(&(usize, usize, &str), &Out)> = comp_fx.iter().collect();
        v.sort_by(|a, b| b.1.kl.partial_cmp(&a.1.kl).unwrap_or(std::cmp::Ordering::Equal));
        v.iter().map(|((l, p, k), o)| json!({"component": format!("S{l} {k} [{p}]"), "layer": l, "position": p, "kind": k, "switched_off_alone_in_union": out_json(m, o)})).collect()
    };
    let mut units_sorted = unit_rows.clone();
    units_sorted.sort_by(|a, b| {
        let ka = a["switched_off_alone_in_union"]["kl"].as_f64().unwrap_or(0.0);
        let kb = b["switched_off_alone_in_union"]["kl"].as_f64().unwrap_or(0.0);
        kb.partial_cmp(&ka).unwrap_or(std::cmp::Ordering::Equal)
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
            "checks": {"trace_closure": closure_v, "trace_closure_worst_at": worst_at, "empty_cut_bit_identical": r.empty_identical,
                "batch_vs_standalone_dkl": s.batch_dev, "clean_rows_dkl": s.clean_dev, "dead_atoms_output_bit_identical": s.dead_identical,
                "tests": s.tests, "passes": s.passes, "seconds": s.secs},
        }),
    )?;
    std::fs::write(dir.join("README.md"), README).map_err(|x| format!("{}: {x}", dir.display()))?;
    Ok(detailed.then_some(closure))
}

const README: &str = "# apollon circuit folder

Written by `apollon --circuit` (self-expansion). Open the files from coarse to fine:

* `circuit.json` – the prompt, the outputs (full model, all paths together, everything except them), the
  paths of the self-expansion (path k = the minimal circuit of what is left without paths 1 … k−1), every
  component and every unit with its effect when switched off alone against the union, and the checks.
* `trace.json` – the circuit backwards from the output: for every unit how many links it is away from the
  output, which units it reads and which it feeds (summed exact contributions of the atom edges between
  them), and the atoms that write directly into the target's logit.
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
