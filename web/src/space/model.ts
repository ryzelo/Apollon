// The model in space, for a decomposed prediction: layers run left to right, token positions front to back,
// and height is the signed effect on the target token. Components stand in lanes behind the last token.
import { Detail, Result, fmt, int, pct, tokenText } from "../api";
import { Family, Metric, Part, familyColor, familyOf, hash, restCount, valueOf } from "../scene";
import { Scene3, V3, arc, emptyScene, seeded, symTicks, symlog } from "./types";

const LANES: Family[] = ["attention", "mlp", "ple", "bias", "other"];
const REST_CAP = 1500;
const GRID = "#ffffff";

export function modelScene(result: Result, metric: Metric, hidden: Set<Family>, other?: Map<string, number | null>): Scene3 {
  const s = emptyScene();
  const parts = result.components.filter((p) => !hidden.has(familyOf(p)));
  if (!parts.length) return s;
  const L = result.model.layers, T = result.tokens.length;
  const fams = LANES.filter((f) => parts.some((p) => familyOf(p) === f));
  const dx = Math.min(0.6, 9.5 / (L + 1)), dz = Math.min(0.9, 7 / (T + fams.length));
  const depth = T + fams.length; // token rows, then one lane per component type
  const X = (col: number) => (col - L / 2) * dx; // column 0 is the input, layer l is column l + 1
  const Z = (row: number) => (row - (depth - 1) / 2) * dz;
  const val = (p: Part) => valueOf(p, metric) ?? p.direct ?? 0;
  const all: number[] = [];
  other?.forEach((v) => { if (v != null) all.push(Math.abs(v)); });
  parts.forEach((p) => { all.push(Math.abs(val(p))); (p.children || []).forEach((c) => all.push(Math.abs(val(c)))); });
  const max = Math.max(...all, 0.1);
  const H = 3.2, sy = H / symlog(max);
  const Y = (v: number) => symlog(v) * sy;
  const x0 = X(0) - dx, x1 = X(L) + dx, z0 = Z(0), z1 = Z(depth - 1);
  s.extent = [(x1 - x0) / 2 + 1, H, (z1 - z0) / 2 + 0.8];
  const focusZ = Z(result.focus);
  const laneZ = (f: Family) => (f === "embedding" ? focusZ : f === "attention" ? focusZ : Z(T + fams.indexOf(f)));
  const col = (p: Part) => (familyOf(p) === "embedding" || p.layer == null || p.layer < 0 ? 0 : p.layer + 1);

  /* ---------- floor: one line per token row and per layer ---------- */
  for (let j = 0; j < depth; j++) {
    const isFocus = j === result.focus;
    s.lines.push({ a: [x0, 0, Z(j)], b: [x1, 0, Z(j)], c: GRID, alpha: isFocus ? 0.3 : j < T ? 0.07 : 0.11, part: "" });
  }
  const layerStep = Math.max(1, Math.ceil((L + 1) / 48));
  for (let c = 0; c <= L; c += layerStep) s.lines.push({ a: [X(c), 0, z0], b: [X(c), 0, z1], c: GRID, alpha: 0.05, part: "" });
  const tokStep = Math.max(1, Math.ceil(T / 28));
  result.tokens.forEach((t, j) => {
    if (j % tokStep && j !== result.focus) return;
    s.labels.push({ p: [x0 - 0.15, 0, Z(j)], text: `${j} ${tokenText(t.text)}`, style: j === result.focus ? "tag" : "axis", align: "right" });
  });
  fams.forEach((f) => {
    if (f !== "attention") s.labels.push({ p: [x0 - 0.15, 0, laneZ(f)], text: f === "mlp" ? "MLP" : f === "ple" ? "Per-layer embedding" : f === "bias" ? "Bias" : "Remainder", style: "label", color: familyColor(f), align: "right" });
  });
  if (fams.includes("attention")) s.labels.push({ p: [x1 + 0.15, 0, focusZ], text: "Attention reads along this row", style: "label", color: familyColor("attention") });
  for (let c = 0; c <= L; c += Math.max(1, Math.ceil((L + 1) / 10))) {
    s.labels.push({ p: [X(c), 0, z1 + 0.35], text: c === 0 ? "In" : `S${c - 1}`, style: "axis", align: "center" });
  }
  // height axis at the back left corner
  s.lines.push({ a: [x0, -H, z0], b: [x0, H, z0], c: GRID, alpha: 0.25, part: "" });
  for (const t of symTicks(max)) {
    s.lines.push({ a: [x0, Y(t), z0], b: [x0 + 0.12, Y(t), z0], c: GRID, alpha: 0.4, part: "" });
    s.labels.push({ p: [x0 - 0.1, Y(t), z0], text: t === 0 ? "0" : fmt(t, t < 1 && t > -1 ? 1 : 0), style: "axis", align: "right" });
  }
  s.labels.push({ p: [x0, H + 0.25, z0], text: `Δ log-odds of ${tokenText(result.target?.text || "")}`, style: "axis", align: "center" });

  /* ---------- the target, at the end of the focus row ---------- */
  const target: V3 = [x1 + 0.1, 0, focusZ];
  s.points.push({ p: target, size: 9, c: "#f2f2f2", a: 1, part: "", tip: `Target ${tokenText(result.target?.text || "")}` });
  s.labels.push({ p: [x1 + 0.1, 0.32, focusZ], text: tokenText(result.target?.text || ""), style: "tag", align: "center" });

  let measured = 0, restDrawn = 0, restAll = 0, sources = 0;
  const tipOf = (p: Part, up?: Part) =>
    `${up ? `${up.label} · ` : ""}${p.label}${p.direct != null ? ` · direct ${fmt(p.direct)}` : ""}${p.total != null ? ` · total ${fmt(p.total)}` : ""}`;

  for (const p of parts) {
    const f = familyOf(p), color = familyColor(f);
    const x = X(col(p)), z = laneZ(f), v = val(p), k = Math.abs(v) / max;
    const at: V3 = [x, Y(v), z];
    // the component: a point at its height, on a stem from the floor
    s.lines.push({ a: [x, 0, z], b: at, c: color, alpha: 0.25 + 0.5 * k, part: p.id });
    s.points.push({ p: at, size: 5 + 9 * Math.sqrt(k), c: color, a: 0.95, part: p.id, tip: tipOf(p) });
    arc(s, at, target, [0, 0.25 + 0.5 * k, 0], color, 0.015 + 0.4 * k * k, p.id, 10);
    // the same component in the run compared with: a white point at its height there
    const ov = other?.get(p.id);
    if (ov != null) {
      const ghost: V3 = [x + dx * 0.18, Y(ov), z];
      s.points.push({ p: ghost, size: 4.5, c: "#f2f2f2", a: 0.9, part: p.id, tip: `${p.label} · other run ${fmt(ov)} · this run ${fmt(v)} · difference ${fmt(v - ov)}` });
      s.lines.push({ a: [x + dx * 0.18, Y(v), z], b: ghost, c: "#f2f2f2", alpha: 0.55, part: p.id });
    }

    const kids = p.children || [];
    const heads = kids.filter((c) => c.kind === "head");
    const rand = seeded(hash(p.id));
    kids.forEach((c) => {
      const cv = val(c), ck = Math.abs(cv) / max;
      if (c.kind === "remainder") {
        const n = restCount(c), draw = Math.min(REST_CAP, n);
        restAll += n; restDrawn += draw;
        const each = cv / n;
        for (let i = 0; i < draw; i++) {
          s.points.push({ p: [x + (rand() - 0.5) * dx * 0.86, Y(each) + (rand() - 0.5) * 0.05, z + (rand() - 0.5) * dz * 0.7], size: 1.4, c: color, a: 0.09, part: c.id, tip: `${p.label} · ${c.label} · together ${fmt(cv)}` });
        }
        return;
      }
      measured++;
      // heads keep their order inside the layer; other units spread by their index
      const hi = heads.indexOf(c);
      const off = hi >= 0 ? ((hi + 0.5) / heads.length - 0.5) : ((hash(c.id) % 1000) / 1000 - 0.5);
      const cx = x + off * dx * 0.8;
      const cp: V3 = [cx, Y(cv), z];
      s.lines.push({ a: [cx, 0, z], b: cp, c: color, alpha: 0.08 + 0.3 * ck, part: c.id });
      s.points.push({ p: cp, size: 2.6 + 6 * Math.sqrt(ck), c: color, a: 0.55 + 0.45 * ck, part: c.id, tip: tipOf(c, p) });
      // where the head looks: one floor line per position, as strong as its attention weight
      const attention = (c as Detail).attention;
      attention?.forEach((w, j) => {
        if (w < 0.04 || j === result.focus) return;
        s.lines.push({ a: [cx, 0, Z(j)], b: [cx, 0, focusZ], c: color, alpha: 0.22 * w, part: c.id });
      });
      // what it carries from each position it reads
      (c.children || []).forEach((src) => {
        if (src.position == null) return;
        sources++;
        const sv = val(src), sk = Math.abs(sv) / max;
        const sp: V3 = [cx, Y(sv), Z(src.position)];
        s.points.push({ p: sp, size: 2.2 + 5 * Math.sqrt(sk), c: color, a: 0.5 + 0.5 * sk, part: src.id, tip: `${p.label} · ${c.label} ${src.label} · direct ${fmt(src.direct)}${attention ? ` · attention ${pct(attention[src.position] ?? 0)}` : ""}` });
        arc(s, sp, cp, [0, 0.2, 0], color, 0.03 + 0.55 * sk, src.id, 8);
      });
    });
  }
  s.drawn = `${int(parts.length)} components, ${int(measured)} units and ${int(sources)} read positions measured on their own; ${int(restDrawn)} of ${int(restAll)} remainder units drawn dim.`;
  return s;
}
