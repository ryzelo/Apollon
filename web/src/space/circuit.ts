// The circuit in space: layers run left to right, token positions front to back. On the floor, every cell
// (layer, position) that keeps atoms glows as a cloud as dense as its share of kept atoms. Above it stand the
// leaves of the circuit at the height of what removing them alone costs the target, and every leaf is joined
// to the target at the end of the focus row.
import { Result, fmt, int, tokenText } from "../api";
import { Family, Part, familyColor, familyOf, hash, valueOf } from "../scene";
import { Scene3, V3, arc, emptyScene, seeded, symTicks, symlog } from "./types";

const GRID = "#ffffff";
const CLOUD_CAP = 6000;

export function circuitScene(result: Result, hidden: Set<Family>): Scene3 {
  const s = emptyScene();
  const c = result.circuit;
  const L = result.model.layers, T = result.tokens.length;
  const dx = Math.min(0.6, 9.5 / (L + 1)), dz = Math.min(0.9, 7 / Math.max(T, 1));
  const X = (layer: number) => (layer + 1 - L / 2) * dx;
  const Z = (pos: number) => (pos - (T - 1) / 2) * dz;
  const leaves: { part: Part; parent: Part }[] = result.components
    .filter((p) => !hidden.has(familyOf(p)))
    .flatMap((p) => (p.children || []).map((part) => ({ part, parent: p })));
  const max = Math.max(...leaves.map(({ part }) => Math.abs(valueOf(part, "total") ?? 0)), 0.1);
  const H = 3.2, sy = H / symlog(max);
  const Y = (v: number) => symlog(v) * sy;
  const x0 = X(-1) - dx, x1 = X(L - 1) + dx, z0 = Z(0), z1 = Z(T - 1);
  s.extent = [(x1 - x0) / 2 + 1, H, (z1 - z0) / 2 + 0.8];
  const focusZ = Z(result.focus);

  /* ---------- floor ---------- */
  for (let j = 0; j < T; j++) s.lines.push({ a: [x0, 0, Z(j)], b: [x1, 0, Z(j)], c: GRID, alpha: j === result.focus ? 0.3 : 0.07, part: "" });
  const layerStep = Math.max(1, Math.ceil(L / 48));
  for (let l = 0; l < L; l += layerStep) s.lines.push({ a: [X(l), 0, z0], b: [X(l), 0, z1], c: GRID, alpha: 0.05, part: "" });
  const tokStep = Math.max(1, Math.ceil(T / 28));
  result.tokens.forEach((t, j) => {
    if (j % tokStep && j !== result.focus) return;
    s.labels.push({ p: [x0 - 0.15, 0, Z(j)], text: `${j} ${tokenText(t.text)}`, style: j === result.focus ? "tag" : "axis", align: "right" });
  });
  for (let l = 0; l < L; l += Math.max(1, Math.ceil(L / 10))) s.labels.push({ p: [X(l), 0, z1 + 0.35], text: `S${l}`, style: "axis", align: "center" });
  s.lines.push({ a: [x0, -H, z0], b: [x0, H, z0], c: GRID, alpha: 0.25, part: "" });
  for (const t of symTicks(max)) {
    s.lines.push({ a: [x0, Y(t), z0], b: [x0 + 0.12, Y(t), z0], c: GRID, alpha: 0.4, part: "" });
    s.labels.push({ p: [x0 - 0.1, Y(t), z0], text: t === 0 ? "0" : fmt(t, t < 1 && t > -1 ? 1 : 0), style: "axis", align: "right" });
  }
  s.labels.push({ p: [x0, H + 0.25, z0], text: `Δ log-odds of ${tokenText(result.target?.text || "")} without the leaf`, style: "axis", align: "center" });

  /* ---------- kept atoms per cell: a cloud on the floor ---------- */
  let cloud = 0;
  if (c?.cells.length) {
    const fams: [Family, (x: { attention: number; kv: number; mlp: number; ple: number }) => number][] = [
      ["attention", (x) => x.attention + x.kv], ["mlp", (x) => x.mlp], ["ple", (x) => x.ple],
    ];
    const all = c.cells.flatMap((x) => fams.map(([, get]) => get(x)));
    const top = Math.log10(1 + Math.max(...all, 1));
    const budget = CLOUD_CAP / Math.max(1, c.cells.length * fams.length);
    for (const cell of c.cells) {
      fams.forEach(([f, get], fi) => {
        const n = get(cell);
        if (!n || hidden.has(f)) return;
        const k = Math.log10(1 + n) / top;
        const draw = Math.max(1, Math.round(k * Math.min(40, budget * 3)));
        const rand = seeded(hash(`${cell.layer}|${cell.position}|${f}`));
        const cx = X(cell.layer) + (fi - 1) * dx * 0.22, cz = Z(cell.position);
        for (let i = 0; i < draw; i++) {
          s.points.push({ p: [cx + (rand() - 0.5) * dx * 0.2, rand() * 0.06, cz + (rand() - 0.5) * dz * 0.5], size: 1.3, c: familyColor(f), a: 0.12 + 0.3 * k, part: "", tip: `S${cell.layer} at [${cell.position}] · ${int(n)} ${f === "attention" ? "attention" : f === "mlp" ? "MLP" : "PLE"} atoms kept` });
          cloud++;
        }
      });
    }
  }

  /* ---------- the target ---------- */
  const target: V3 = [x1 + 0.1, 0, focusZ];
  s.points.push({ p: target, size: 9, c: "#f2f2f2", a: 1, part: "", tip: `Target ${tokenText(result.target?.text || "")}` });
  s.labels.push({ p: [x1 + 0.1, 0.32, focusZ], text: tokenText(result.target?.text || ""), style: "tag", align: "center" });

  /* ---------- the leaves ---------- */
  for (const { part, parent } of leaves) {
    const color = familyColor(familyOf(part));
    const v = valueOf(part, "total") ?? 0, k = Math.abs(v) / max;
    const off = ((hash(part.id) % 1000) / 1000 - 0.5) * dx * 0.7;
    const at: V3 = [X(part.layer ?? parent.layer ?? 0) + off, Y(v), Z(part.position ?? result.focus)];
    s.lines.push({ a: [at[0], 0, at[2]], b: at, c: color, alpha: 0.1 + 0.4 * k, part: part.id });
    s.points.push({ p: at, size: 2.6 + 7 * Math.sqrt(k), c: color, a: 0.55 + 0.45 * k, part: part.id, tip: `${part.label} · Δ log-odds without it ${fmt(v)}` });
    if (k > 0.05) arc(s, at, target, [0, 0.2 + 0.4 * k, 0], color, 0.02 + 0.35 * k * k, part.id, 10);
  }
  s.drawn = `${int(leaves.length)} leaves of the circuit${c && c.leaves > c.leaves_listed ? ` (the ${int(c.leaves_listed)} most necessary of ${int(c.leaves)})` : ""}; ${int(cloud)} floor points for the kept atoms per layer and position.`;
  return s;
}
