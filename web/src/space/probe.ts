// A probed component in space: its units stand in a plane on the left, ranked front to back. From the
// units examined in detail, a path runs right through the later layers and shows what their removal changes there.
import { ProbeComponent, Result, fmt, int, tokenText } from "../api";
import { Metric, Part, familyColor, familyOf, valueOf } from "../scene";
import { Scene3, V3, emptyScene, symTicks, symlog } from "./types";

const GRID = "#ffffff";

export function probeScene(result: Result, metric: Metric): Scene3 {
  const s = emptyScene();
  const comps = result.components as ProbeComponent[];
  if (!comps.length) return s;
  const L = result.model.layers;
  const first = Math.min(...comps.map((c) => c.layer ?? 0));
  const cols = L - first; // the probed layer and every later one
  const dx = Math.min(0.7, 9 / (cols + 3));
  const X = (layer: number) => (layer - first - (cols - 1) / 2 + 1.2) * dx;
  const xUnits = X(first) - 2.6 * dx;
  const val = (p: Part) => valueOf(p, metric) ?? 0;
  // every height in the scene is a change of the target's log-odds: with the unit minus without it
  const heights: number[] = [0.1];
  comps.forEach((c) => {
    heights.push(Math.abs(val(c)));
    (c.children || []).forEach((u) => heights.push(Math.abs(val(u))));
    (c.downstream || []).forEach((d) => d.layers.forEach((l) => {
      heights.push(Math.abs(l.lens_base.target_log_odds - l.lens_variant.target_log_odds));
      l.components.forEach((k) => heights.push(Math.abs(k.direct_base - k.direct_variant)));
    }));
  });
  const max = Math.max(...heights);
  const H = 3.2, sy = H / symlog(max);
  const Y = (v: number) => symlog(v) * sy;
  const depthEach = 4.2, gap = 1.4;
  const totalDepth = comps.length * depthEach + (comps.length - 1) * gap;
  const x0 = xUnits - 0.5, x1 = X(L - 1) + 0.6;
  s.extent = [(x1 - x0) / 2 + 0.8, H, totalDepth / 2 + 0.8];
  let units = 0, lanes = 0;

  comps.forEach((comp, ci) => {
    const zBase = -totalDepth / 2 + ci * (depthEach + gap);
    const list = comp.children || [];
    const color = familyColor(familyOf(comp));
    const Z = (i: number) => zBase + (list.length > 1 ? (i / (list.length - 1)) * depthEach : depthEach / 2);
    const unitMax = Math.max(...list.map((u) => Math.abs(val(u))), 1e-6);
    // floor under this component
    for (let layer = first; layer < L; layer += Math.max(1, Math.ceil(cols / 40))) {
      s.lines.push({ a: [X(layer), 0, zBase], b: [X(layer), 0, zBase + depthEach], c: GRID, alpha: 0.05, part: "" });
    }
    s.lines.push({ a: [xUnits, 0, zBase], b: [xUnits, 0, zBase + depthEach], c: GRID, alpha: 0.25, part: "" });
    s.labels.push({ p: [xUnits, -0.3, zBase + depthEach + 0.3], text: `${int(list.length)} units of ${comp.label}`, style: "label", color, align: "center" });

    // the units: one point each, ranked from the most supporting at the back to the most suppressing in front
    list.forEach((u, i) => {
      units++;
      const v = val(u), k = Math.abs(v) / unitMax;
      const d = u as Part & { activation?: number };
      s.points.push({
        p: [xUnits, Y(v), Z(i)], size: 2.4 + 8 * Math.sqrt(k), c: color, a: 0.4 + 0.6 * k, part: u.id,
        tip: `${comp.label} · ${u.label}${u.position != null ? ` at [${u.position}]` : ""} · removed ${fmt(u.total)}${d.activation != null ? ` · value ${fmt(d.activation, 2)}` : ""}`,
      });
      if (k > 0.02) s.lines.push({ a: [xUnits, 0, Z(i)], b: [xUnits, Y(v), Z(i)], c: color, alpha: 0.06 + 0.3 * k, part: u.id });
    });

    // paths through the later layers
    (comp.downstream || []).forEach((path, pi) => {
      const ui = path.unit == null ? -1 : list.findIndex((u) => u.unit === path.unit && (u.position ?? null) === (path.position ?? null));
      const whole = ui < 0;
      const part = whole ? comp.id : list[ui].id;
      // the whole component runs behind its units; a unit's path starts at the unit
      const z = whole ? zBase + depthEach + 0.45 : Z(ui);
      const start: V3 = whole ? [xUnits, Y(val(comp)), z] : [xUnits, Y(val(list[ui])), z];
      if (whole) s.points.push({ p: start, size: 9, c: "#f2f2f2", a: 0.95, part, tip: `${comp.label} · whole component removed ${fmt(comp.total)}` });
      lanes++;
      s.lines.push({ a: [xUnits, 0, z], b: [x1, 0, z], c: GRID, alpha: 0.1, part });
      let prev = start;
      path.layers.forEach((l) => {
        const x = X(l.layer);
        // read after this layer: how much more the target gets with the unit than without it
        const lens = l.lens_base.target_log_odds - l.lens_variant.target_log_odds;
        const here: V3 = [x, Y(lens), z];
        s.lines.push({ a: prev, b: here, c: "#f2f2f2", alpha: 0.55, part });
        s.points.push({ p: here, size: 3, c: "#f2f2f2", a: 0.8, part, tip: `${path.label} · read after S${l.layer}: ${fmt(lens)} for ${tokenText(result.target?.text || "")} · top without it ${tokenText(l.lens_variant.top[0]?.text || "")}` });
        prev = here;
        // components of this layer that vote differently
        l.components.forEach((k, j) => {
          const dv = k.direct_base - k.direct_variant;
          if (Math.abs(dv) < 1e-4) return;
          const kc = familyColor(k.kind === "attention" ? "attention" : k.kind === "mlp" ? "mlp" : "ple");
          const kz = z + (j - 1) * 0.07;
          const share = Math.abs(dv) / max;
          s.lines.push({ a: [x, 0, kz], b: [x, Y(dv), kz], c: kc, alpha: 0.15 + 0.6 * share, part });
          s.points.push({ p: [x, Y(dv), kz], size: 2.4 + 6 * Math.sqrt(share), c: kc, a: 0.5 + 0.5 * share, part, tip: `${path.label} · S${l.layer} ${k.kind} votes ${fmt(k.direct_base)} with it, ${fmt(k.direct_variant)} without` });
        });
      });
      s.labels.push({ p: [x1 + 0.1, prev[1], z], text: whole ? `whole ${comp.label}` : list[ui].label, style: "axis" });
    });
  });

  for (let layer = first; layer < L; layer += Math.max(1, Math.ceil(cols / 10))) {
    s.labels.push({ p: [X(layer), 0, totalDepth / 2 + 0.75], text: `S${layer}`, style: "axis", align: "center" });
  }
  const zAxis = -totalDepth / 2;
  s.lines.push({ a: [x0, -H, zAxis], b: [x0, H, zAxis], c: GRID, alpha: 0.25, part: "" });
  for (const t of symTicks(max)) {
    s.lines.push({ a: [x0, Y(t), zAxis], b: [x0 + 0.12, Y(t), zAxis], c: GRID, alpha: 0.4, part: "" });
    s.labels.push({ p: [x0 - 0.1, Y(t), zAxis], text: t === 0 ? "0" : fmt(t, Math.abs(t) < 1 ? 1 : 0), style: "axis", align: "right" });
  }
  s.labels.push({ p: [x0, H + 0.25, zAxis], text: `Δ log-odds of ${tokenText(result.target?.text || "")}, with minus without`, style: "axis", align: "center" });
  s.drawn = `${int(units)} units, each removed on its own; ${int(lanes)} of them followed through the later layers.`;
  return s;
}
