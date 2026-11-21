//! `--neuron "S12 #2328"`: static report for one neuron, computed on demand without a prompt –
//! repertoire, all inputs by source kind, all outputs. Also the (p, q)-plane drawing shared with `--trace`.
use crate::fmt::{chip, esc_html, num, pad, padl, sci, tok, trunc, Style, CSS};
use crate::model::Model;
use crate::neuron::{act64, act_name, arrow, path, read_back, repertoire, static_inputs, static_outputs, top2, Arrow, Read, ReadPoint, Repertoire, Src};
use std::path::Path;

pub struct Opts {
    pub k: usize,
}

fn cos(a: &[f32], b: &[f32]) -> f64 {
    let (ab, aa, bb) = a.iter().zip(b).fold((0f64, 0f64, 0f64), |s, (x, y)| (s.0 + *x as f64 * *y as f64, s.1 + (*x as f64).powi(2), s.2 + (*y as f64).powi(2)));
    ab / (aa * bb).sqrt().max(1e-300)
}

/// K/V part of a head line; layers that reuse K/V of an earlier layer have no K/V projection of their own
fn kv_txt(m: &Model, l2: usize, k: f32, v: f32) -> String {
    if k.is_nan() {
        format!("(K/V shared from S{})", m.blocks[l2].attn.kv_from.unwrap_or(0))
    } else {
        format!("‖K‖ {} ‖V‖ {}", num(k, 2), num(v, 2))
    }
}

pub fn src_name(m: &Model, s: &Src) -> String {
    s.label(m)
}

fn arrows_line(m: &Model, v: &[Arrow], by_q: bool) -> String {
    v.iter()
        .map(|a| format!("{} {}", trunc(&src_name(m, &a.src), 26), if by_q { num(a.q as f32, 3) } else { num(a.p as f32, 3) }))
        .collect::<Vec<_>>()
        .join(" · ")
}

/// Weight types of the neuron's matrices
pub fn weight_types(m: &Model, l: usize) -> String {
    let b = &m.blocks[l].mlp;
    format!("{}up {}, down {}", b.gate.as_ref().map_or(String::new(), |g| format!("gate {}, ", g.type_name())), b.up.type_name(), b.down.type_name())
}

pub struct Static {
    pub rd: Read,
    pub rep: Repertoire,
    pub inputs: crate::neuron::Inputs,
    pub outputs: crate::neuron::Outputs,
    pub check_in: f64,
    pub check_out: f64,
}

pub fn compute(m: &Model, l: usize, i: usize) -> Result<Static, String> {
    let rd = Read::new(m, l, i)?;
    let rep = repertoire(&rd);
    let inputs = static_inputs(m, &rd);
    let outputs = static_outputs(m, l, i);
    // checks: the adjoint path (all columns at once) against the forward path (one vector), for the
    // strongest neuron sources; and the outputs read by rows against the same table read by columns
    let mut check_in = 0f64;
    if let Some((_, ns)) = inputs.kinds.iter().find(|k| k.0 == "neurons") {
        let (tp, _) = top2(ns, 3);
        for a in tp {
            if let Src::Neuron(k, c) = a.src {
                let col = crate::readout::columns(&m.blocks[k as usize].mlp.down, &[c as usize]);
                let ops = path(m, a.src, ReadPoint::MlpIn(l), true).unwrap();
                let (p, q) = arrow(&rd, &ops, &col);
                check_in = check_in.max(((p - a.p).abs() + (q - a.q).abs()) / (p.abs() + q.abs()).max(1e-30));
            }
        }
    }
    let mut check_out = 0f64;
    let (tp, _) = top2(&outputs.neurons, 3);
    let col = crate::readout::columns(&m.blocks[l].mlp.down, &[i]);
    for a in tp {
        if let Src::Neuron(l2, i2) = a.src {
            let r2 = Read::new(m, l2 as usize, i2 as usize)?;
            let ops = path(m, Src::Neuron(l as u16, i as u32), ReadPoint::MlpIn(l2 as usize), true).unwrap();
            let (rg, ru) = read_back(&r2, &ops);
            let p = crate::neuron::dot64(&rg, &col);
            let q = if r2.gated { crate::neuron::dot64(&ru, &col) } else { 0.0 };
            check_out = check_out.max(((p - a.p).abs() + (q - a.q).abs()) / (p.abs() + q.abs()).max(1e-30));
        }
    }
    Ok(Static { rd, rep, inputs, outputs, check_in, check_out })
}

/// The static view as structured data: every list holds the strongest arrows by |p| and by |q|.
fn result_json(m: &Model, s: &Static, o: &Opts) -> serde_json::Value {
    use serde_json::json;
    let k = o.k.max(64);
    let arrows = |v: &[Arrow]| -> serde_json::Value {
        let item = |a: &Arrow| { let mut x = crate::trace::src_json(&a.src); x["label"] = json!(a.src.label(m)); x["p"] = json!(a.p); x["q"] = json!(a.q); x };
        let (tp, tq) = top2(v, k);
        json!({"count":v.len(),"by_p":tp.iter().map(item).collect::<Vec<_>>(),"by_q":tq.iter().map(item).collect::<Vec<_>>()})
    };
    let (rd, rep) = (&s.rd, &s.rep);
    let mut voice = s.outputs.voice.clone();
    voice.sort_by(|a, b| b.1.total_cmp(&a.1));
    let token = |&(t, x): &(u32, f32)| json!({"id":t,"text":m.tok.piece(t),"logit":x});
    let n = voice.len();
    let mut heads = s.outputs.heads.clone();
    heads.sort_by(|a, b| b.2.max(b.3).max(b.4).total_cmp(&a.2.max(a.3).max(a.4)));
    json!({"schema_version":2,"mode":"neuron","model":{"name":m.name,"layers":m.n_layer(),"architecture":m.arch},
        "tokens":[],"output":[],"components":[],"generated":[],"answer":"",
        "neuron":{"label":format!("S{} #{}", rd.l, rd.i),"layer":rd.l,"unit":rd.i,"gated":rd.gated,"activation_function":act_name(rd.act),
            "repertoire":{"center":rep.c,"gram":rep.gram,"radius":rep.radius,"rank":rep.rank,
                "activation_min":rep.a_min,"activation_max":rep.a_max,"point_min":rep.pt_min,"point_max":rep.pt_max},
            "inputs":s.inputs.kinds.iter().map(|(kind, v)| { let mut x = arrows(v); x["group"] = json!(kind); x }).collect::<Vec<_>>(),
            "runtime_only_head_layers":s.inputs.head_layers,
            "outputs":{"neurons":arrows(&s.outputs.neurons),
                "raises":voice.iter().take(k.min(n)).map(token).collect::<Vec<_>>(),
                "lowers":voice[n.saturating_sub(k)..].iter().rev().map(token).collect::<Vec<_>>(),
                "heads":heads.iter().take(k).map(|&(l2, h, q, kk, v)| json!({"layer":l2,"head":h,"query":q,"key":kk,"value":v})).collect::<Vec<_>>()}},
        "checks":[{"label":"Inputs via the adjoint path against the forward path","deviation":s.check_in},
            {"label":"Outputs read by rows against the same table read by columns","deviation":s.check_out}]})
}

pub fn run(m: &Model, l: usize, i: usize, o: &Opts, html: &Path, json: Option<&Path>, st: &Style) -> Result<(), String> {
    let t0 = std::time::Instant::now();
    eprintln!("neuron S{l} #{i}: static arrows over all sources and targets …");
    let s = compute(m, l, i)?;
    if let Some(path) = json {
        crate::export::write(path, &result_json(m, &s, o))?;
    }
    terminal(m, &s, o, st);
    std::fs::write(html, html_report(m, &s, o)).map_err(|e| format!("{}: {e}", html.display()))?;
    println!("  {} {}", st.bold("Report with the (p, q) plane:"), html.display());
    eprintln!("neuron: {} s", num(t0.elapsed().as_secs_f32(), 1));
    Ok(())
}

pub fn repertoire_lines(rd: &Read, rep: &Repertoire) -> Vec<String> {
    let mut v = Vec::new();
    let shape = match (rd.gated, rep.rank) {
        (false, _) => "interval (not gated: a = act(p))".to_string(),
        (true, 2) => "filled ellipse".into(),
        (true, 1) => "segment (g' ∥ u' or one of them 0)".into(),
        _ => "single point (g' = u' = 0)".into(),
    };
    v.push(format!(
        "repertoire ({shape}, radius √d = {}): a ∈ [{}, {}]",
        num(rep.radius as f32, 1),
        num(rep.a_min as f32, 4),
        num(rep.a_max as f32, 4)
    ));
    if !rd.gated {
        v.push(format!(
            "  p ∈ [{}, {}] · a_max at p = {} · a_min at p = {}",
            num((rep.c[0] - rep.radius * rep.lmat[0][0]) as f32, 3),
            num((rep.c[0] + rep.radius * rep.lmat[0][0]) as f32, 3),
            num(rep.pt_max[0] as f32, 3),
            num(rep.pt_min[0] as f32, 3)
        ));
    } else {
    v.push(format!(
        "  a_max at θ = {} (p = {}, q = {}) · a_min at θ = {} (p = {}, q = {}) · center c = ({}, {})",
        num(rep.th_max as f32, 4),
        num(rep.pt_max[0] as f32, 3),
        num(rep.pt_max[1] as f32, 3),
        num(rep.th_min as f32, 4),
        num(rep.pt_min[0] as f32, 3),
        num(rep.pt_min[1] as f32, 3),
        num(rep.c[0] as f32, 4),
        num(rep.c[1] as f32, 4)
    ));
    }
    let z = rep.z_at(rd, rep.th_max);
    let mut zi: Vec<(usize, f32)> = z.iter().cloned().enumerate().collect();
    zi.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
    let zn = z.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    v.push(format!(
        "  input direction z* for a_max (‖z*‖ = {}): largest coordinates {}",
        num(zn as f32, 1),
        zi.iter().take(8).map(|(j, x)| format!("[{j}] {}", num(*x, 2))).collect::<Vec<_>>().join(" ")
    ));
    v
}

fn terminal(m: &Model, s: &Static, o: &Opts, st: &Style) {
    let rd = &s.rd;
    let k = o.k.min(8);
    println!();
    println!(
        "{}",
        st.bold(&format!("Neuron S{} #{} – static (no prompt), {}, {}, pre-norm {}", rd.l, rd.i, if rd.gated { "gated" } else { "not gated" }, act_name(rd.act), if rd.ln { "LayerNorm" } else { "RMSNorm" }))
    );
    println!("  {}", st.dim(&format!("weights: {}", weight_types(m, rd.l))));
    let nrm = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    if rd.gated {
        println!(
            "  read vectors: ‖g'‖ = {} · ‖u'‖ = {} · cos(g', u') = {} · offsets c_g = {}, c_u = {}",
            num(nrm(&rd.g) as f32, 4),
            num(nrm(&rd.u) as f32, 4),
            num(cos(&rd.g, &rd.u) as f32, 3),
            num(rd.cg as f32, 4),
            num(rd.cu as f32, 4)
        );
    } else {
        println!("  read vector: ‖g'‖ = {} · offset c = {}", num(nrm(&rd.g) as f32, 4), num(rd.cg as f32, 4));
    }
    for line in repertoire_lines(rd, &s.rep) {
        println!("  {line}");
    }
    println!();
    println!("  {}", st.bold("Inputs – static arrows (contribution to p and q per unit of source activation, before the norm scales):"));
    for (kind, v) in &s.inputs.kinds {
        if v.is_empty() {
            continue;
        }
        let (tp, tq) = top2(v, k);
        println!("     {} ({}):", kind, crate::fmt::thousands(v.len()));
        println!("        by |p|: {}", arrows_line(m, &tp, false));
        if rd.gated {
            println!("        by |q|: {}", arrows_line(m, &tq, true));
        }
    }
    if !s.inputs.head_layers.is_empty() {
        println!(
            "     heads of layers S{}–S{}: {}",
            s.inputs.head_layers[0],
            s.inputs.head_layers.last().unwrap(),
            st.yellow("runtime only – what a head writes depends on what it reads, so it has no static arrow (see --trace)")
        );
    }
    println!();
    println!("  {}", st.bold("Outputs – where the neuron writes (per unit activation):"));
    if !s.outputs.neurons.is_empty() {
        let (tp, tq) = top2(&s.outputs.neurons, k);
        println!("     into later neurons ({}):", crate::fmt::thousands(s.outputs.neurons.len()));
        println!("        by |p|: {}", arrows_line(m, &tp, false));
        if s.outputs.neurons.iter().any(|a| a.q != 0.0) {
            println!("        by |q|: {}", arrows_line(m, &tq, true));
        }
    } else {
        println!("     into later neurons: none (last layer)");
    }
    if !s.outputs.voice.is_empty() {
        let mut v = s.outputs.voice.clone();
        v.sort_by(|a, b| b.1.total_cmp(&a.1));
        let n = v.len();
        println!(
            "     token voice (final norm at unit scale, output matrix{}): raises {}",
            if m.logit_cap.is_some() { "; the logit softcap is not applied" } else { "" },
            v.iter().take(k).map(|(t, x)| format!("{} {}", tok(m, *t), num(*x, 3))).collect::<Vec<_>>().join(" · ")
        );
        println!("        lowers {}", v[n.saturating_sub(k)..].iter().rev().map(|(t, x)| format!("{} {}", tok(m, *t), num(*x, 3))).collect::<Vec<_>>().join(" · "));
    }
    if !s.outputs.heads.is_empty() {
        let mut h = s.outputs.heads.clone();
        h.sort_by(|a, b| b.2.max(b.3.max(0.0)).max(b.4.max(0.0)).total_cmp(&a.2.max(a.3.max(0.0)).max(a.4.max(0.0))));
        println!(
            "     later heads ({}): {}",
            st.yellow("approximation – linear projections only, RoPE and Q/K/V norms ignored"),
            h.iter().take(k).map(|(l2, hh, q, kk, v)| format!("S{l2} head {hh} ‖Q‖ {} {}", num(*q, 2), kv_txt(m, *l2, *kk, *v))).collect::<Vec<_>>().join(" · ")
        );
    }
    println!();
    println!(
        "  {}",
        st.dim(&format!(
            "checks: inputs via the adjoint path = forward path (3 strongest neuron sources, dev. {}) · outputs read by rows = the same table read by columns (3 strongest targets, dev. {})",
            sci(s.check_in as f32),
            sci(s.check_out as f32)
        ))
    );
}

// ====================================================================== (p, q) plane as SVG

pub struct Plane<'a> {
    pub rd: &'a Read,
    pub rep: &'a Repertoire,
    /// the real point (p, q) of a pass
    pub point: Option<(f64, f64)>,
    /// arrows tip to tail from the center c; the last one should end at the point
    pub arrows: Vec<(String, f64, f64)>,
}

const COLORS: [&str; 8] = ["#2f5fb3", "#1d7a4c", "#b33a2e", "#8a5a00", "#6b3fa0", "#00808a", "#a0346b", "#556070"];

pub fn plane_svg(pl: &Plane, zoom: bool) -> String {
    let rd = pl.rd;
    let rep = pl.rep;
    let (w, h) = (360.0, 300.0);
    // path of the arrows and the bounding box
    let mut pts: Vec<(f64, f64)> = vec![(rep.c[0], rep.c[1])];
    let mut cur = (rep.c[0], rep.c[1]);
    for (_, dp, dq) in &pl.arrows {
        cur = (cur.0 + dp, cur.1 + dq);
        pts.push(cur);
    }
    if let Some(p) = pl.point {
        pts.push(p);
    }
    let boundary: Vec<(f64, f64)> = (0..=256).map(|k| rep.point(k as f64 * 2.0 * std::f64::consts::PI / 256.0)).map(|x| (x[0], x[1])).collect();
    let bbox = |v: &[(f64, f64)]| -> (f64, f64, f64, f64) {
        v.iter().fold((f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY), |b, p| (b.0.min(p.0), b.1.max(p.0), b.2.min(p.1), b.3.max(p.1)))
    };
    let (mut x0, mut x1, mut y0, mut y1) = if zoom { bbox(&pts) } else { bbox(&boundary.iter().chain(&pts).cloned().collect::<Vec<_>>()) };
    if !rd.gated {
        // not gated: horizontal p axis, vertical a = act(p)
        let (lo, hi) = (rep.c[0] - rep.radius * rep.lmat[0][0], rep.c[0] + rep.radius * rep.lmat[0][0]);
        if !zoom {
            x0 = x0.min(lo);
            x1 = x1.max(hi);
        }
        let n = 200;
        let vals: Vec<f64> = (0..=n).map(|k| act64(rd.act, x0 + (x1 - x0) * k as f64 / n as f64)).collect();
        y0 = vals.iter().cloned().fold(f64::INFINITY, f64::min);
        y1 = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    }
    let pad_ = |a: f64, b: f64| -> (f64, f64) {
        let s = (b - a).abs().max(1e-9 * (1.0 + a.abs()));
        (a - 0.12 * s, b + 0.12 * s)
    };
    let (x0, x1) = pad_(x0, x1);
    let (y0, y1) = pad_(y0, y1);
    let sx = |p: f64| (p - x0) / (x1 - x0) * w;
    let sy = |q: f64| h - (q - y0) / (y1 - y0) * h;
    let mut s = format!(r#"<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" class="plane" xmlns="http://www.w3.org/2000/svg"><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="context-stroke"/></marker></defs><rect width="{w}" height="{h}" class="bg"/>"#);
    if rd.gated {
        // color grid of act(p)·q inside the ellipse
        let n = 44;
        let mut cells = Vec::new();
        let mut amax = 1e-30f64;
        for a in 0..n {
            for b in 0..n {
                let p = x0 + (x1 - x0) * (a as f64 + 0.5) / n as f64;
                let q = y0 + (y1 - y0) * (b as f64 + 0.5) / n as f64;
                if rep.r_of(p, q) <= 1.0 {
                    let v = rd.value(p, q);
                    amax = amax.max(v.abs());
                    cells.push((a, b, v));
                }
            }
        }
        let (cw, ch) = (w / n as f64, h / n as f64);
        for (a, b, v) in cells {
            let op = (v.abs() / amax).sqrt() * 0.75;
            s += &format!(
                r#"<rect x="{:.1}" y="{:.1}" width="{:.2}" height="{:.2}" fill="{}" fill-opacity="{op:.3}"/>"#,
                a as f64 * cw,
                h - (b + 1) as f64 * ch,
                cw + 0.3,
                ch + 0.3,
                if v >= 0.0 { "var(--pos)" } else { "var(--neg)" }
            );
        }
        // zero line of q (a = 0 for q = 0) and of the gate
        s += &format!(r#"<polyline class="ell" points="{}"/>"#, boundary.iter().map(|p| format!("{:.1},{:.1}", sx(p.0), sy(p.1))).collect::<Vec<_>>().join(" "));
        for (pt, lab) in [(rep.pt_max, "max"), (rep.pt_min, "min")] {
            let (px, py) = (sx(pt[0]), sy(pt[1]));
            if (0.0..=w).contains(&px) && (0.0..=h).contains(&py) {
                s += &format!(r#"<circle cx="{px:.1}" cy="{py:.1}" r="4" class="mk"/><text x="{:.1}" y="{:.1}" class="lb">a_{lab}</text>"#, px + 6.0, py - 6.0);
            }
        }
    } else {
        let n = 200;
        let line: Vec<String> = (0..=n)
            .map(|k| {
                let p = x0 + (x1 - x0) * k as f64 / n as f64;
                format!("{:.1},{:.1}", sx(p), sy(act64(rd.act, p)))
            })
            .collect();
        s += &format!(r#"<polyline class="ell" points="{}"/>"#, line.join(" "));
        let (lo, hi) = (rep.c[0] - rep.radius * rep.lmat[0][0], rep.c[0] + rep.radius * rep.lmat[0][0]);
        s += &format!(r#"<rect x="{:.1}" y="{:.1}" width="{:.1}" height="4" fill="var(--acc)" fill-opacity="0.35"/>"#, sx(lo).max(0.0), h - 6.0, (sx(hi) - sx(lo)).min(w));
    }
    // axes through zero
    if (x0..x1).contains(&0.0) {
        s += &format!(r#"<line x1="{:.1}" y1="0" x2="{:.1}" y2="{h}" class="ax"/>"#, sx(0.0), sx(0.0));
    }
    if (y0..y1).contains(&0.0) {
        s += &format!(r#"<line x1="0" y1="{:.1}" x2="{w}" y2="{:.1}" class="ax"/>"#, sy(0.0), sy(0.0));
    }
    // arrows tip to tail
    let mut cur = (rep.c[0], rep.c[1]);
    let ypos = |q: f64, k: usize| if rd.gated { sy(q) } else { h - 14.0 - 6.0 * k as f64 };
    for (k, (_, dp, dq)) in pl.arrows.iter().enumerate() {
        let nxt = (cur.0 + dp, cur.1 + dq);
        s += &format!(
            r#"<line x1="{:.1}" y1="{:.1}" x2="{:.1}" y2="{:.1}" stroke="{}" stroke-width="2" marker-end="url(#ah)"/>"#,
            sx(cur.0),
            ypos(cur.1, k),
            sx(nxt.0),
            ypos(nxt.1, k),
            COLORS[k % COLORS.len()]
        );
        cur = nxt;
    }
    s += &format!(r#"<circle cx="{:.1}" cy="{:.1}" r="3" class="c0"/>"#, sx(rep.c[0]), if rd.gated { sy(rep.c[1]) } else { h - 14.0 });
    if let Some((p, q)) = pl.point {
        let a = rd.value(p, q);
        s += &format!(r#"<circle cx="{:.1}" cy="{:.1}" r="4.5" class="pt"/>"#, sx(p), if rd.gated { sy(q) } else { sy(a) });
    }
    s += &format!(r#"<text x="{}" y="{}" class="ax-l">p</text>"#, w - 12.0, h - 4.0);
    s += &format!(r#"<text x="4" y="12" class="ax-l">{}</text>"#, if rd.gated { "q" } else { "a" });
    s += &format!(r#"<text x="4" y="{}" class="ax-l">p {}…{} · {} {}…{}</text>"#, h - 4.0, num(x0 as f32, 2), num(x1 as f32, 2), if rd.gated { "q" } else { "a" }, num(y0 as f32, 2), num(y1 as f32, 2));
    s += "</svg>";
    s
}

pub fn arrow_legend(arrows: &[(String, f64, f64)]) -> String {
    arrows
        .iter()
        .enumerate()
        .map(|(k, (lab, dp, dq))| format!("<span class=\"lg\"><i style=\"background:{}\"></i>{} ({}, {})</span>", COLORS[k % COLORS.len()], esc_html(lab), num(*dp as f32, 3), num(*dq as f32, 3)))
        .collect()
}

pub const PLANE_CSS: &str = r#"
.plane{background:var(--card);border:1px solid var(--line);border-radius:6px;margin:4px 12px 4px 0}
.plane .bg{fill:var(--card)}
.plane .ell{fill:none;stroke:var(--fg);stroke-width:1.2;stroke-opacity:.6}
.plane .ax{stroke:var(--mu);stroke-width:.6;stroke-dasharray:3 3}
.plane .mk{fill:var(--fg)}
.plane .pt{fill:var(--acc);stroke:var(--card);stroke-width:1.5}
.plane .c0{fill:none;stroke:var(--fg);stroke-width:1.2}
.plane .lb,.plane .ax-l{font-size:10px;fill:var(--mu)}
.lg{display:inline-flex;align-items:center;gap:4px;margin:0 10px 2px 0;font-size:12px}
.lg i{display:inline-block;width:10px;height:3px}
.planes{display:flex;flex-wrap:wrap}
"#;

fn html_report(m: &Model, s: &Static, o: &Opts) -> String {
    let rd = &s.rd;
    let k = o.k;
    let mut h = format!(
        r#"<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Neuron S{} #{}</title><style>{CSS}{PLANE_CSS}</style></head><body><main>
<h1>Neuron S{} #{} – static</h1><p class="meta">{} · {} · {} · pre-norm {} · {}</p>"#,
        rd.l,
        rd.i,
        rd.l,
        rd.i,
        esc_html(&m.name),
        if rd.gated { "gated" } else { "not gated" },
        act_name(rd.act),
        if rd.ln { "LayerNorm" } else { "RMSNorm" },
        esc_html(&weight_types(m, rd.l))
    );
    h += "<h2>Repertoire</h2><p class=\"mu\">All (p, q) the neuron can reach form a filled ellipse (radius √d around the offset). Color: act(p)·q. Left: everything, right: around the offset.</p>";
    h += "<div class=\"planes\">";
    let pl = Plane { rd, rep: &s.rep, point: None, arrows: vec![] };
    h += &plane_svg(&pl, false);
    h += "</div>";
    for line in repertoire_lines(rd, &s.rep) {
        h += &format!("<p class=\"mu\">{}</p>", esc_html(&line));
    }
    let table = |title: &str, v: &[Arrow]| -> String {
        let (tp, tq) = top2(v, k);
        let mut t = format!("<details class=\"info\"><summary>{} ({})</summary><div class=\"planes\">", esc_html(title), crate::fmt::thousands(v.len()));
        for (lab, list) in [("by |p|", tp), ("by |q|", tq)] {
            if lab == "by |q|" && v.iter().all(|a| a.q == 0.0) {
                continue;
            }
            t += &format!("<table class=\"tbl\"><tr><th>{lab}</th><th>p</th><th>q</th></tr>");
            for a in list {
                let name = match a.src {
                    Src::Token(id) => chip(m, id),
                    _ => esc_html(&src_name(m, &a.src)),
                };
                t += &format!("<tr><td>{name}</td><td class=\"num\">{}</td><td class=\"num\">{}</td></tr>", num(a.p as f32, 4), num(a.q as f32, 4));
            }
            t += "</table>";
        }
        t + "</div></details>"
    };
    h += "<h2>Inputs</h2><p class=\"mu\">Static arrows: contribution to p and q per unit of source activation, before the norm scales.</p>";
    for (kind, v) in &s.inputs.kinds {
        if !v.is_empty() {
            h += &table(kind, v);
        }
    }
    if !s.inputs.head_layers.is_empty() {
        h += &format!("<p>Heads of layers S{}–S{}: runtime only – what a head writes depends on what it reads.</p>", s.inputs.head_layers[0], s.inputs.head_layers.last().unwrap());
    }
    h += "<h2>Outputs</h2>";
    if !s.outputs.neurons.is_empty() {
        h += &table("into later neurons", &s.outputs.neurons);
    }
    if !s.outputs.voice.is_empty() {
        let mut v = s.outputs.voice.clone();
        v.sort_by(|a, b| b.1.total_cmp(&a.1));
        let n = v.len();
        h += "<details class=\"info\"><summary>Token voice (final norm at unit scale, output matrix)</summary><div class=\"planes\">";
        for (lab, list) in [("raises", v[..k.min(n)].to_vec()), ("lowers", v[n.saturating_sub(k)..].iter().rev().cloned().collect::<Vec<_>>())] {
            h += &format!("<table class=\"tbl\"><tr><th>{lab}</th><th>logit</th></tr>");
            for (t, x) in list {
                h += &format!("<tr><td>{}</td><td class=\"num\">{}</td></tr>", chip(m, t), num(x, 4));
            }
            h += "</table>";
        }
        h += "</div></details>";
    }
    if !s.outputs.heads.is_empty() {
        let mut hs = s.outputs.heads.clone();
        hs.sort_by(|a, b| b.2.max(b.3).max(b.4).total_cmp(&a.2.max(a.3).max(a.4)));
        h += "<details class=\"info\"><summary>Later heads (approximation: linear projections, RoPE and Q/K/V norms ignored)</summary><table class=\"tbl\"><tr><th>head</th><th>‖W_Q o‖</th><th>‖W_K o‖</th><th>‖W_V o‖</th></tr>";
        for (l2, hh, q, kk, v) in hs.iter().take(4 * k) {
            let kvc = |x: f32| if x.is_nan() { "shared".to_string() } else { num(x, 3) };
            h += &format!("<tr><td>S{l2} head {hh}</td><td class=\"num\">{}</td><td class=\"num\">{}</td><td class=\"num\">{}</td></tr>", num(*q, 3), kvc(*kk), kvc(*v));
        }
        h += "</table></details>";
    }
    h += &format!(
        "<p class=\"check\">Checks: inputs via the adjoint path = forward path (dev. {}) · outputs read by rows = read by columns (dev. {})</p><p class=\"mu foot\">apollon · neuron</p></main></body></html>\n",
        sci(s.check_in as f32),
        sci(s.check_out as f32)
    );
    let _ = (pad, padl);
    h
}
