// A traced neuron in space: the backward tree of what feeds its activation. Layers run left to right, token
// positions front to back; height is a source's share of the activation of the node it feeds.
import { Result, TraceComponent, TreeNode, fmt, int, tokenText } from "../api";
import { Family, familyColor } from "../scene";
import { Scene3, V3, arc, emptyScene, symTicks, symlog } from "./types";

const GRID = "#ffffff";
const familyOfNode = (k: string): Family =>
  k === "neuron" ? "mlp" : k === "head" || k === "atom" || k === "read" ? "attention" : k === "token" || k === "position" ? "embedding" : k === "ple" ? "ple" : /bias|offset/.test(k) ? "bias" : "other";

export function traceScene(result: Result): Scene3 {
  const s = emptyScene();
  const comps = (result.components as TraceComponent[]).filter((c) => c.tree);
  if (!comps.length) return s;
  const L = result.model.layers, T = result.tokens.length;
  const dx = Math.min(0.6, 9.5 / (L + 1)), dz = Math.min(0.9, 7 / Math.max(T, 2));
  const X = (col: number) => (col - L / 2) * dx;
  const Z = (j: number) => (j - (T - 1) / 2) * dz;
  const shares: number[] = [0.1];
  const walk = (n: TreeNode) => { if (n.activation_contribution != null) shares.push(Math.abs(n.activation_contribution)); n.children.forEach(walk); };
  comps.forEach((c) => walk(c.tree!));
  const max = Math.max(...shares);
  const H = 3.2, sy = H / symlog(max);
  const Y = (v: number) => symlog(v) * sy;
  const x0 = X(0) - dx, x1 = X(L) + dx;
  s.extent = [(x1 - x0) / 2 + 1, H, ((T - 1) * dz) / 2 + 0.9];

  // floor: token rows and layers
  for (let j = 0; j < T; j++) s.lines.push({ a: [x0, 0, Z(j)], b: [x1, 0, Z(j)], c: GRID, alpha: j === result.focus ? 0.3 : 0.07, part: "" });
  for (let c = 0; c <= L; c += Math.max(1, Math.ceil((L + 1) / 48))) s.lines.push({ a: [X(c), 0, Z(0)], b: [X(c), 0, Z(T - 1)], c: GRID, alpha: 0.05, part: "" });
  const tokStep = Math.max(1, Math.ceil(T / 28));
  result.tokens.forEach((t, j) => {
    if (j % tokStep && j !== result.focus) return;
    s.labels.push({ p: [x0 - 0.15, 0, Z(j)], text: `${j} ${tokenText(t.text)}`, style: j === result.focus ? "tag" : "axis", align: "right" });
  });
  for (let c = 0; c <= L; c += Math.max(1, Math.ceil((L + 1) / 10))) s.labels.push({ p: [X(c), 0, Z(T - 1) + 0.4], text: c === 0 ? "In" : `S${c - 1}`, style: "axis", align: "center" });
  s.lines.push({ a: [x0, -H, Z(0)], b: [x0, H, Z(0)], c: GRID, alpha: 0.25, part: "" });
  for (const t of symTicks(max)) {
    s.lines.push({ a: [x0, Y(t), Z(0)], b: [x0 + 0.12, Y(t), Z(0)], c: GRID, alpha: 0.4, part: "" });
    s.labels.push({ p: [x0 - 0.1, Y(t), Z(0)], text: t === 0 ? "0" : fmt(t, Math.abs(t) < 1 ? 1 : 0), style: "axis", align: "right" });
  }
  s.labels.push({ p: [x0, H + 0.25, Z(0)], text: "Share of the activation it feeds", style: "axis", align: "center" });

  let nodes = 0;
  comps.forEach((comp) => {
    const part = comp.id;
    /** Draws the children of a node that sits at `at`, in column `col` and at token position `pos`. */
    const draw = (kids: TreeNode[], at: V3, col: number, pos: number, depth: number) => {
      const local = Math.max(...kids.map((k) => Math.abs(k.activation_contribution ?? 0)), 1e-9);
      kids.forEach((k, i) => {
        nodes++;
        const a = k.activation_contribution ?? 0, share = Math.abs(a) / local;
        const fam = familyOfNode(k.kind), color = familyColor(fam);
        const kPos = k.position ?? pos;
        // a source without a layer of its own (a bias, the offset) stands just before the node it feeds
        const kCol = k.kind === "read" ? col : fam === "embedding" ? 0 : k.layer != null ? k.layer + 1 : col - 0.5;
        // siblings at the same place are nudged apart sideways so each stays a point of its own
        const p: V3 = [X(kCol) + (k.kind === "read" || k.layer == null ? 0 : ((i % 7) - 3) * dx * 0.05), Y(a), Z(kPos)];
        s.points.push({
          p, size: 3 + 7 * Math.sqrt(share) / Math.sqrt(depth), c: color, a: 0.5 + 0.5 * share, part,
          tip: `${k.label}${k.kind !== "read" && kPos !== pos ? ` at [${kPos}]` : ""} · share ${fmt(a, 4)}${k.activation != null ? ` · own activation ${fmt(k.activation, 3)}` : ""}`,
        });
        arc(s, p, at, [0, 0.15, 0], color, 0.08 + 0.55 * share, part, 8);
        if (k.children.length) draw(k.children, p, kCol, kPos, depth + 1);
      });
    };
    const root = comp.tree!;
    const at: V3 = [X(root.layer! + 1), 0, Z(root.position ?? result.focus)];
    s.points.push({ p: at, size: 11, c: "#f2f2f2", a: 1, part, tip: `${root.label} at [${root.position}] · activation ${fmt(root.activation, 3)} · removed ${fmt(comp.total)}` });
    s.labels.push({ p: [at[0], 0.3, at[2]], text: `${root.label} · a ${fmt(root.activation, 2)}`, style: "tag", align: "center" });
    draw(root.children, at, root.layer! + 1, root.position ?? result.focus, 1);
  });
  const all = comps.reduce((n, c) => n + (c.source_count || 0), 0);
  s.drawn = `${int(nodes)} sources drawn, the strongest at every step; the traced ${comps.length === 1 ? "neuron has" : "neurons have"} ${int(all)} sources in total.`;
  return s;
}
