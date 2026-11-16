// One neuron from its weights alone: what it reads on the left, what it writes into on the right.
// Layers run left to right; height and depth are the two coordinates (p, q) of the plane a neuron computes in.
import { Arrow, NeuronStatic, int } from "../api";
import { Family, familyColor } from "../scene";
import { Scene3, V3, emptyScene } from "./types";

const GRID = "#ffffff";
const fam = (k: string): Family =>
  k === "neuron" ? "mlp" : k === "head" || k === "atom" ? "attention" : k === "token" || k === "position" ? "embedding" : k === "ple" ? "ple" : /bias|offset/.test(k) ? "bias" : "other";

/** The strongest arrows by |p| and by |q| as one list without repeats. */
function union(list: { by_p: Arrow[]; by_q: Arrow[] }): Arrow[] {
  const seen = new Map<string, Arrow>();
  [...list.by_p, ...list.by_q].forEach((a) => { if (!seen.has(a.label)) seen.set(a.label, a); });
  return [...seen.values()];
}

export function neuronScene(n: NeuronStatic, layers: number): Scene3 {
  const s = emptyScene();
  const dx = Math.min(0.6, 9.5 / (layers + 1));
  const X = (col: number) => (col - layers / 2) * dx;
  const H = 3.2, D = 3;
  const x0 = X(0) - dx, x1 = X(layers) + dx;
  s.extent = [(x1 - x0) / 2 + 1, H, D + 0.6];
  const centre: V3 = [X(n.layer + 1), 0, 0];
  const colOf = (a: Arrow) => (fam(a.kind) === "embedding" ? 0 : a.layer != null ? a.layer + 1 : n.layer + 0.5);

  // the (p, q) plane at the neuron, and the floor along the layers
  s.lines.push({ a: [x0, 0, 0], b: [x1, 0, 0], c: GRID, alpha: 0.25, part: "" });
  s.lines.push({ a: [centre[0], -H, 0], b: [centre[0], H, 0], c: GRID, alpha: 0.3, part: "" });
  s.lines.push({ a: [centre[0], 0, -D], b: [centre[0], 0, D], c: GRID, alpha: 0.3, part: "" });
  for (let c = 0; c <= layers; c += Math.max(1, Math.ceil((layers + 1) / 48))) s.lines.push({ a: [X(c), 0, -D], b: [X(c), 0, D], c: GRID, alpha: 0.04, part: "" });
  for (let c = 0; c <= layers; c += Math.max(1, Math.ceil((layers + 1) / 10))) s.labels.push({ p: [X(c), 0, D + 0.35], text: c === 0 ? "In" : `S${c - 1}`, style: "axis", align: "center" });
  s.labels.push({ p: [centre[0], H + 0.2, 0], text: "p", style: "axis", align: "center" });
  s.labels.push({ p: [centre[0], 0, D + 0.15], text: n.gated ? "q (gate)" : "q · not gated", style: "axis", align: "center" });
  s.points.push({ p: centre, size: 12, c: "#f2f2f2", a: 1, part: "", tip: `${n.label} · reachable activation ${n.repertoire.activation_min.toFixed(3)} to ${n.repertoire.activation_max.toFixed(3)}` });
  s.labels.push({ p: [centre[0], 0.32, 0], text: n.label, style: "tag", align: "center" });

  const draw = (arrows: Arrow[], side: "in" | "out", what: string) => {
    const mp = Math.max(...arrows.map((a) => Math.abs(a.p)), 1e-30), mq = Math.max(...arrows.map((a) => Math.abs(a.q)), 1e-30);
    arrows.forEach((a) => {
      const color = familyColor(fam(a.kind));
      const share = Math.max(Math.abs(a.p) / mp, Math.abs(a.q) / mq);
      const p: V3 = [X(colOf(a)), (a.p / mp) * H, (a.q / mq) * D];
      s.points.push({ p, size: 2.6 + 7 * Math.sqrt(share), c: color, a: 0.45 + 0.55 * share, part: "", tip: `${what} ${a.label} · p ${a.p.toExponential(2)} · q ${a.q.toExponential(2)}` });
      s.lines.push({ a: side === "in" ? p : centre, b: side === "in" ? centre : p, c: color, alpha: 0.04 + 0.4 * share * share, part: "" });
    });
    return arrows.length;
  };
  let drawn = 0, total = 0;
  n.inputs.forEach((g) => { drawn += draw(union(g), "in", "Reads"); total += g.count; });
  drawn += draw(union(n.outputs.neurons), "out", "Writes into");
  total += n.outputs.neurons.count;
  s.labels.push({ p: [x0 - 0.1, 0, 0], text: "what it reads", style: "label", align: "right" });
  s.labels.push({ p: [x1 + 0.1, 0, 0], text: "what it writes into", style: "label" });
  s.drawn = `The ${int(drawn)} strongest of ${int(total)} static connections. Inputs and outputs are each scaled to their own largest arrow.`;
  return s;
}
