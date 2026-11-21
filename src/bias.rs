//! Bias rows of neurons (--bias, --circuit-bias).
//!
//! A neuron a = act(p)·q (not gated: a = act(p)) reads the stream through its pre-norm: p = s·(g'·x) + c_g,
//! q = s·(u'·x) + c_u, with s the norm's scale at that position. Everything written into the stream reaches
//! x along a linear path (post-norms and layer scalars, each with its scale), so every source contributes to p
//! and q exactly
//!
//!     Δp = s · f(path) · A_p(source) · (activation of the source)
//!
//! with A_p the static arrow – what one unit of the source's activation gives, at unit norm scales – and
//! f(path) the product of the runtime norm scales on the way. The same holds forwards: one unit of this
//! neuron's activation gives every later neuron, every later Q/K/V projection and every token's logit a fixed
//! amount. These fixed amounts are the neuron's **bias rows**:
//!
//! * the bias row (outputs): every later neuron (Δp, Δq per unit activation), every later Q/K/V projection
//!   (the full vectors), every token of the vocabulary (its logit);
//! * the bias column (inputs): every token embedding, every earlier neuron, PLE dimension and bias (Δp, Δq
//!   per unit of the source); heads have no static arrow (what a head writes depends on what it reads), at
//!   runtime they count with their actual output;
//! * the range: the neuron's activation lies in [a_min, a_max] (its repertoire), so the bias it can give a
//!   target lies between a_min and a_max times the arrow;
//! * the token cases: at every position of the prompt, the activations of all sources multiplied with
//!   their arrows – what the neuron actually receives and gives. The received parts add up to p and q as the
//!   forward pass computed them (checked at every position), so a = act(p)·q follows from them exactly.
use crate::fmt::{num, sci, tok, Style};
use crate::model::{Model, NormScales, Res, RunOpts, Trace};
use crate::neuron::{col_arrows, forward, read_back, repertoire, runtime_factor, static_inputs, static_outputs, Read, ReadPoint, Scale, Src};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

pub struct Opts {
    pub neurons: Vec<(usize, usize)>,
    pub dir: PathBuf,
    /// entries listed by name per list
    pub top: usize,
}

/// The pass the token cases are read from: the full model, every input and norm scale recorded.
pub fn record(m: &Model, ids: &[u32], focus: usize) -> Trace {
    m.run(ids, focus, &RunOpts { record_inputs: true, record_norms: true, keep_final: true, ..Default::default() })
}

pub fn run(m: &Model, ids: &[u32], focus: usize, o: &Opts, st: &Style) -> Result<(), String> {
    let tr = record(m, ids, focus);
    std::fs::create_dir_all(&o.dir).map_err(|x| format!("{}: {x}", o.dir.display()))?;
    let mut index = Vec::new();
    for &(l, i) in &o.neurons {
        let t0 = std::time::Instant::now();
        let v = analyze(m, ids, focus, &tr, l, i, &o.dir, o.top)?;
        print_summary(m, &v, st);
        println!("{}", st.dim(&format!("  {} s · {}", num(t0.elapsed().as_secs_f32(), 1), o.dir.join(v["folder"].as_str().unwrap_or("")).display())));
        index.push(json!({"neuron": v["neuron"], "folder": v["folder"], "closure": v["checks"]["received_add_up_to_p_and_q"]}));
    }
    write(&o.dir.join("index.json"), &json!({"what": "bias rows of neurons; one folder per neuron (bias.json and the full rows as float32)", "neurons": index}))?;
    Ok(())
}

fn write(f: &Path, v: &Value) -> Result<(), String> {
    std::fs::write(f, serde_json::to_string_pretty(v).unwrap()).map_err(|x| format!("{}: {x}", f.display()))
}

fn write_f32(f: &Path, v: &[f32]) -> Result<(), String> {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(f, bytes).map_err(|x| format!("{}: {x}", f.display()))
}

/// the norm scales of one position, as the paths ask for them
fn scale_at(ns: &NormScales, p: usize) -> impl Fn(Scale) -> f32 + '_ {
    move |s| match s {
        Scale::AttnPost(k) => ns.layers[k].attn_post.get(p).copied().unwrap_or(1.0),
        Scale::MlpPost(k) => ns.layers[k].mlp_post.get(p).copied().unwrap_or(1.0),
        Scale::Ple(k) => ns.layers[k].ple.get(p).copied().unwrap_or(1.0),
        Scale::Stream => 1.0,
    }
}

/// the scale of the MLP pre-norm of layer l at position p (1 without one)
fn mlp_scale(m: &Model, ns: &NormScales, l: usize, p: usize) -> f32 {
    if m.blocks[l].n_mlp.is_none() {
        return 1.0;
    }
    ns.layers[l].mlp.get(p).copied().unwrap_or(1.0)
}

fn src_name(m: &Model, s: &Src) -> String {
    match *s {
        Src::Neuron(k, c) => format!("S{k} #{c}"),
        Src::Head(k, h) => format!("S{k} head {h}"),
        Src::PleDim(k, e) => format!("S{k} PLE #{e}"),
        _ => s.label(m),
    }
}

/// The bias rows of neuron (l, i), its range, its token cases over the recorded pass; files into
/// dir/S<l>_<i>/. Returns the summary (also written as bias.json).
#[allow(clippy::too_many_arguments)]
pub fn analyze(m: &Model, ids: &[u32], focus: usize, tr: &Trace, l: usize, i: usize, dir: &Path, top: usize) -> Result<Value, String> {
    if m.blocks.iter().any(|b| b.res == Res::Post) {
        return Err("--bias: post-norm encoders are not supported".into());
    }
    let rd = Read::new(m, l, i)?;
    let rep = repertoire(&rd);
    let ns = tr.norms.as_ref().ok_or("--bias: no norm scales recorded")?;
    let t = ids.len();
    let folder = format!("S{l}_{i}");
    let nd = dir.join(&folder);
    std::fs::create_dir_all(&nd).map_err(|x| format!("{}: {x}", nd.display()))?;
    let ff = m.blocks[l].mlp.up.rows;
    // ------------------------------------------------------------ the bias column: what every source can give it
    let inputs = static_inputs(m, &rd);
    let mut col_files = Vec::new();
    for (kind, arrows) in &inputs.kinds {
        let mut v = Vec::with_capacity(arrows.len() * 2);
        for a in arrows {
            v.push(a.p as f32);
            v.push(a.q as f32);
        }
        let f = format!("column_{}.f32", kind.replace(' ', "_"));
        write_f32(&nd.join(&f), &v)?;
        col_files.push(json!({"kind": kind, "file": f, "entries": arrows.len(), "layout": "per source: Δp, Δq per unit of the source, at unit norm scales (float32)",
            "first": arrows.first().map(|a| src_name(m, &a.src)), "last": arrows.last().map(|a| src_name(m, &a.src))}));
    }
    let top_in = |kind: &str| -> Vec<Value> {
        let Some((_, arrows)) = inputs.kinds.iter().find(|(k, _)| *k == kind) else { return Vec::new() };
        let mut v: Vec<&crate::neuron::Arrow> = arrows.iter().collect();
        v.sort_by(|a, b| (b.p.abs() + b.q.abs()).total_cmp(&(a.p.abs() + a.q.abs())));
        v.iter().take(top).map(|a| json!({"source": src_name(m, &a.src), "dp": a.p, "dq": a.q})).collect()
    };
    // ------------------------------------------------------------ the bias row: what it can give every later part
    let outputs = static_outputs(m, l, i);
    let mut row = Vec::with_capacity(outputs.neurons.len() * 2);
    let mut row_layers: Vec<(usize, usize)> = Vec::new();
    for a in &outputs.neurons {
        row.push(a.p as f32);
        row.push(a.q as f32);
        if let Src::Neuron(k, _) = a.src {
            match row_layers.last_mut() {
                Some((lk, n)) if *lk == k as usize => *n += 1,
                _ => row_layers.push((k as usize, 1)),
            }
        }
    }
    write_f32(&nd.join("row_neurons.f32"), &row)?;
    // the full Q/K/V projections of every later layer (before q/k norms and RoPE)
    let col = crate::readout::columns(&m.blocks[l].mlp.down, &[i]);
    let mut qkv = Vec::new();
    let mut qkv_layers = Vec::new();
    for (l2, b2) in m.blocks.iter().enumerate() {
        let Some(ops) = crate::neuron::path(m, Src::Neuron(l as u16, i as u32), ReadPoint::AttnIn(l2), true) else { continue };
        let mut v = col.clone();
        forward(&ops, &mut v);
        let vn = match &b2.n_attn {
            Some(n) => n.map_part(1.0, &v),
            None => v,
        };
        let a = &b2.attn;
        let (q, k, vv) = match &a.wqkv {
            Some(w) => {
                let o = w.matmul(&vn, 1);
                let hd = a.hd;
                let mut q = Vec::new();
                let mut k = Vec::new();
                let mut v2 = Vec::new();
                for h in 0..a.n_head {
                    q.extend_from_slice(&o[h * 3 * hd..h * 3 * hd + hd]);
                    k.extend_from_slice(&o[h * 3 * hd + hd..h * 3 * hd + 2 * hd]);
                    v2.extend_from_slice(&o[h * 3 * hd + 2 * hd..(h + 1) * 3 * hd]);
                }
                (q, k, v2)
            }
            None => {
                let q = a.wq.as_ref().map(|w| w.matmul(&vn, 1)).unwrap_or_default();
                let k = a.wk.as_ref().map(|w| w.matmul(&vn, 1)).unwrap_or_default();
                let v2 = if a.v_is_k { k.clone() } else { a.wv.as_ref().map(|w| w.matmul(&vn, 1)).unwrap_or_default() };
                (q, k, v2)
            }
        };
        qkv_layers.push(json!({"layer": l2, "q": q.len(), "k": k.len(), "v": vv.len()}));
        qkv.extend(q);
        qkv.extend(k);
        qkv.extend(vv);
    }
    write_f32(&nd.join("row_qkv.f32"), &qkv)?;
    let voice: Vec<f32> = outputs.voice.iter().map(|x| x.1).collect();
    if !voice.is_empty() {
        write_f32(&nd.join("row_tokens.f32"), &voice)?;
    }
    // the targets it can move most, with the range its activation allows
    let range = |x: f64| -> [f64; 2] {
        let (a, b) = (rep.a_min * x, rep.a_max * x);
        [a.min(b), a.max(b)]
    };
    let mut outs: Vec<&crate::neuron::Arrow> = outputs.neurons.iter().collect();
    outs.sort_by(|a, b| (b.p.abs() + b.q.abs()).total_cmp(&(a.p.abs() + a.q.abs())));
    let top_out: Vec<Value> = outs.iter().take(top).map(|a| json!({"target": src_name(m, &a.src), "dp": a.p, "dq": a.q, "dp_range": range(a.p), "dq_range": range(a.q)})).collect();
    let mut vo: Vec<(u32, f32)> = outputs.voice.clone();
    vo.sort_by(|a, b| b.1.total_cmp(&a.1));
    let voice_top = |rev: bool| -> Vec<Value> {
        let it: Box<dyn Iterator<Item = &(u32, f32)>> = if rev { Box::new(vo.iter().rev()) } else { Box::new(vo.iter()) };
        it.take(top.min(20)).map(|(t2, x)| json!({"token": m.tok.piece(*t2), "logit": x, "range": range(*x as f64)})).collect()
    };
    // ------------------------------------------------------------ token cases: every position of the prompt
    let inp = &tr.inputs;
    let gated = rd.gated;
    let arrows_of = |kind: &str| inputs.kinds.iter().find(|(k, _)| *k == kind).map(|x| &x.1);
    let mut cases = Vec::new();
    let mut worst = (0f64, String::new());
    for p in 0..t {
        let sc = scale_at(ns, p);
        let s = mlp_scale(m, ns, l, p) as f64;
        let rp = ReadPoint::MlpIn(l);
        // (source, Δp, Δq) received at this position
        let mut recv: Vec<(String, &'static str, f64, f64)> = Vec::new();
        let fac = |src: Src| -> Option<f64> { crate::neuron::path(m, src, rp, true).map(|ops| s * runtime_factor(&ops, &sc)) };
        if let (Some(ar), Some(f)) = (arrows_of("tokens"), fac(Src::Token(0))) {
            let a = &ar[ids[p] as usize];
            recv.push((format!("token {}", tok(m, ids[p])), "embedding", f * a.p, f * a.q));
        }
        if let (Some(ar), Some((_, off)), Some(f)) = (arrows_of("positions"), m.pos_embed.as_ref(), fac(Src::Position(0))) {
            if let Some(a) = ar.get(p + off) {
                recv.push((format!("position {p}"), "embedding", f * a.p, f * a.q));
            }
        }
        if let Some(ar) = arrows_of("neurons") {
            let mut cur_layer = usize::MAX;
            let mut f = 0.0;
            for a in ar {
                if let Src::Neuron(k, c) = a.src {
                    let k = k as usize;
                    if k != cur_layer {
                        cur_layer = k;
                        f = fac(a.src).unwrap_or(0.0);
                    }
                    let ffk = m.blocks[k].mlp.up.rows;
                    let act = inp[k].act[p * ffk + c as usize] as f64;
                    if act != 0.0 {
                        recv.push((format!("S{k} #{c}"), "neurons", f * act * a.p, f * act * a.q));
                    }
                }
            }
        }
        if let Some(ar) = arrows_of("PLE dimensions") {
            for a in ar {
                if let (Src::PleDim(k, e), Some(f)) = (a.src, fac(a.src)) {
                    let k = k as usize;
                    let dp = m.blocks[k].ple.as_ref().map_or(0, |pb| pb.gate.rows);
                    let gv = inp[k].ple_g.get(p * dp + e as usize).copied().unwrap_or(0.0) as f64;
                    recv.push((format!("S{k} PLE #{e}"), "PLE", f * gv * a.p, f * gv * a.q));
                }
            }
        }
        if let Some(ar) = arrows_of("biases") {
            for a in ar {
                if let Some(f) = fac(a.src) {
                    recv.push((src_name(m, &a.src), "biases", f * a.p, f * a.q));
                }
            }
        }
        // heads: their actual output W_O·z at this position, read through the path
        for (k, b) in m.blocks.iter().enumerate() {
            let Some(ops) = crate::neuron::path(m, Src::Head(k as u16, 0), rp, true) else { continue };
            let f = s * runtime_factor(&ops, &sc);
            let (rg, ru) = read_back(&rd, &ops);
            let cg = col_arrows(&b.attn.wo, &rg);
            let cu = if gated { col_arrows(&b.attn.wo, &ru) } else { Vec::new() };
            let (nh, hd) = (b.attn.n_head, b.attn.hd);
            let z = &inp[k].z[p * nh * hd..(p + 1) * nh * hd];
            for h in 0..nh {
                let r = h * hd..(h + 1) * hd;
                let dp: f64 = r.clone().map(|c| cg[c] as f64 * z[c] as f64).sum();
                let dq: f64 = if gated { r.map(|c| cu[c] as f64 * z[c] as f64).sum() } else { 0.0 };
                recv.push((format!("S{k} head {h}"), "heads", f * dp, f * dq));
            }
        }
        recv.push(("offset (w·β + b)".into(), "offset", rd.cg, rd.cu));
        let (p_sum, q_sum): (f64, f64) = recv.iter().fold((0.0, 0.0), |acc, x| (acc.0 + x.2, acc.1 + x.3));
        let (p_abs, q_abs): (f64, f64) = recv.iter().fold((0.0, 0.0), |acc, x| (acc.0 + x.2.abs(), acc.1 + x.3.abs()));
        let li = &inp[l];
        let (p_rec, q_rec) = if gated { (li.gate[p * ff + i] as f64, li.up[p * ff + i] as f64) } else { (li.up[p * ff + i] as f64, 0.0) };
        let a_rec = li.act[p * ff + i] as f64;
        let dev_p = (p_sum - p_rec).abs() / p_abs.max(p_rec.abs()).max(1e-6);
        let dev_q = if gated { (q_sum - q_rec).abs() / q_abs.max(q_rec.abs()).max(1e-6) } else { 0.0 };
        for (dv, what) in [(dev_p, "p"), (dev_q, "q")] {
            if dv > worst.0 {
                worst = (dv, format!("{what} at [{p}]"));
            }
        }
        // by group, and the strongest single sources
        let mut groups: Vec<(&str, f64, f64)> = Vec::new();
        for x in &recv {
            match groups.iter_mut().find(|g| g.0 == x.1) {
                Some(g) => {
                    g.1 += x.2;
                    g.2 += x.3;
                }
                None => groups.push((x.1, x.2, x.3)),
            }
        }
        recv.sort_by(|a, b| (b.2.abs() + b.3.abs()).total_cmp(&(a.2.abs() + a.3.abs())));
        // what it gives at this position: its activation times the row, with the targets' scales
        let mut give: Vec<(String, f64, f64, f64)> = Vec::new();
        let mut cur = usize::MAX;
        let mut f2 = 0.0;
        for a in &outputs.neurons {
            if let Src::Neuron(k, c) = a.src {
                let k = k as usize;
                if k != cur {
                    cur = k;
                    f2 = crate::neuron::path(m, Src::Neuron(l as u16, i as u32), ReadPoint::MlpIn(k), true).map_or(0.0, |ops| mlp_scale(m, ns, k, p) as f64 * runtime_factor(&ops, &sc));
                }
                let (dp, dq) = (f2 * a_rec * a.p, f2 * a_rec * a.q);
                let ffk = m.blocks[k].mlp.up.rows;
                let target_p = if m.blocks[k].mlp.gate.is_some() { inp[k].gate[p * ffk + c as usize] } else { inp[k].up[p * ffk + c as usize] } as f64;
                give.push((format!("S{k} #{c}"), dp, dq, target_p));
            }
        }
        give.sort_by(|a, b| (b.1.abs() + b.2.abs()).total_cmp(&(a.1.abs() + a.2.abs())));
        cases.push(json!({
            "position": p, "token": m.tok.piece(ids[p]), "scale": s,
            "p": p_rec, "q": if gated { json!(q_rec) } else { Value::Null }, "activation": a_rec,
            "received_sum": {"p": p_sum, "q": if gated { json!(q_sum) } else { Value::Null }}, "deviation": {"p": dev_p, "q": dev_q},
            "received_by_group": groups.iter().map(|g| json!({"group": g.0, "dp": g.1, "dq": g.2})).collect::<Vec<_>>(),
            "received_strongest": recv.iter().take(top).map(|x| json!({"source": x.0, "group": x.1, "dp": x.2, "dq": x.3})).collect::<Vec<_>>(),
            "gives_strongest": give.iter().take(top).map(|x| json!({"target": x.0, "dp": x.1, "dq": x.2, "target_p": x.3})).collect::<Vec<_>>(),
        }));
    }
    // what it gives the output at the focus: its activation times its token row, through the final norm
    let at_focus = match crate::neuron::path(m, Src::Neuron(l as u16, i as u32), ReadPoint::Final, true) {
        Some(ops) if !voice.is_empty() => {
            let sc = scale_at(ns, focus);
            let f = ns.fin.get(focus).copied().unwrap_or(1.0) as f64 * runtime_factor(&ops, &sc) * inp[l].act[focus * ff + i] as f64;
            let mut v: Vec<(u32, f64)> = outputs.voice.iter().map(|(t2, x)| (*t2, f * *x as f64)).collect();
            v.sort_by(|a, b| b.1.total_cmp(&a.1));
            json!({"raises": v.iter().take(10).map(|(t2, x)| json!({"token": m.tok.piece(*t2), "logit": x})).collect::<Vec<_>>(),
                "lowers": v.iter().rev().take(10).map(|(t2, x)| json!({"token": m.tok.piece(*t2), "logit": x})).collect::<Vec<_>>()})
        }
        _ => Value::Null,
    };
    let summary = json!({
        "neuron": format!("S{l} #{i}"), "layer": l, "index": i, "folder": folder,
        "formula": if gated {
            "a = act(p)·q; p = s·Σ_sources f·A_p·(activation of the source) + c_g; q = s·Σ_sources f·A_q·(activation) + c_u (s: pre-norm scale, f: runtime scales on the path, A: static arrows of the bias column; heads count with their actual output)"
        } else {
            "a = act(p); p = s·Σ_sources f·A_p·(activation of the source) + c (s: pre-norm scale, f: runtime scales on the path, A: static arrows of the bias column; heads count with their actual output)"
        },
        "activation_function": crate::neuron::act_name(rd.act), "gated": gated,
        "offsets": {"c_g": rd.cg, "c_u": rd.cu},
        "repertoire": {"a_min": rep.a_min, "a_max": rep.a_max, "radius": rep.radius},
        "bias_row": {
            "what": "what one unit of this neuron's activation gives every later part, at unit norm scales; times the activation (and the runtime scales) it is what it actually gives; times a_min / a_max the range it can give",
            "neurons": {"file": "row_neurons.f32", "layout": "per target neuron: Δp, Δq (float32), layers in order", "layers": row_layers.iter().map(|(k, n)| json!({"layer": k, "neurons": n})).collect::<Vec<_>>(), "strongest": top_out},
            "qkv": {"file": "row_qkv.f32", "layout": "per later layer: the Q, K, V projection vectors (before q/k norms and RoPE), float32", "layers": qkv_layers},
            "tokens": {"file": if voice.is_empty() { Value::Null } else { json!("row_tokens.f32") }, "layout": "logit per token of the vocabulary (through the final norm at unit scale)", "raises": voice_top(false), "lowers": voice_top(true)},
        },
        "bias_column": {
            "what": "what one unit of every source gives this neuron's p and q, at unit norm scales (heads: runtime only)",
            "files": col_files,
            "strongest": {"tokens": top_in("tokens"), "neurons": top_in("neurons"), "biases": top_in("biases")},
            "head_layers": inputs.head_layers,
        },
        "token_cases": cases,
        "gives_the_output_at_the_focus": at_focus,
        "checks": {"received_add_up_to_p_and_q": worst.0, "worst_at": worst.1},
    });
    write(&nd.join("bias.json"), &summary)?;
    Ok(summary)
}

pub fn print_summary(m: &Model, v: &Value, st: &Style) {
    let _ = m;
    let f = |x: &Value| x.as_f64().unwrap_or(f64::NAN);
    println!();
    println!("{}", st.bold(&format!("Bias rows of {}", v["neuron"].as_str().unwrap_or("?"))));
    println!("  {}", v["formula"].as_str().unwrap_or(""));
    println!("  activation range [{}, {}]", num(f(&v["repertoire"]["a_min"]) as f32, 2), num(f(&v["repertoire"]["a_max"]) as f32, 2));
    let list = |x: &Value, key: &str| -> String {
        x.as_array()
            .map(|a| a.iter().take(5).map(|e| format!("{} ({}/{})", e[key].as_str().unwrap_or("?"), num(f(&e["dp"]) as f32, 3), num(f(&e["dq"]) as f32, 3))).collect::<Vec<_>>().join(" · "))
            .unwrap_or_default()
    };
    println!("  can give most to (Δp/Δq per unit): {}", list(&v["bias_row"]["neurons"]["strongest"], "target"));
    println!("  can get most from neurons (Δp/Δq per unit): {}", list(&v["bias_column"]["strongest"]["neurons"], "source"));
    if let Some(cases) = v["token_cases"].as_array() {
        for c in cases.iter().rev().take(1) {
            println!(
                "  at [{}] {:?}: p {} q {} → a {} · received by group: {}",
                c["position"],
                c["token"].as_str().unwrap_or(""),
                num(f(&c["p"]) as f32, 3),
                num(f(&c["q"]) as f32, 3),
                num(f(&c["activation"]) as f32, 3),
                c["received_by_group"].as_array().map(|g| g.iter().map(|x| format!("{} {}", x["group"].as_str().unwrap_or(""), num(f(&x["dp"]) as f32, 3))).collect::<Vec<_>>().join(" · ")).unwrap_or_default()
            );
        }
    }
    println!("{}", st.dim(&format!("  check: at every position the received parts add up to p and q – largest relative deviation {}", sci(f(&v["checks"]["received_add_up_to_p_and_q"]) as f32))));
}
