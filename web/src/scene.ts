// Particle scenes for the Apollon stage (Noir Particle design).
// Every scene is a flat list of lines, particles, nodes and HTML tags in CSS pixels.

export type Part = {
  id: string;
  label: string;
  layer: number | null;
  kind: string;
  direct: number | null;
  total: number | null;
  unit?: number | null;
  position?: number | null;
  activation_contribution?: number;
  source?: { kind: string; layer?: number };
  voice?: { id: number; text: string; probability: number }[];
  children?: Part[];
};
export type Metric = "direct" | "total";
export type Family = "embedding" | "attention" | "mlp" | "ple" | "bias" | "other";
export type View = "flow" | "units";

export const FAMILIES: { id: Family; label: string; color: string }[] = [
  { id: "embedding", label: "Embedding", color: "#f2f2f2" },
  { id: "attention", label: "Attention", color: "#3d9bff" },
  { id: "mlp", label: "MLP", color: "#ff9a3c" },
  { id: "ple", label: "Per-layer embedding", color: "#9b6bff" },
  { id: "bias", label: "Bias", color: "#ffd83d" },
  { id: "other", label: "Remainder", color: "#4d4d4d" },
];
export const familyColor = (f: Family) => FAMILIES.find((x) => x.id === f)!.color;

export function familyOf(p: Part): Family {
  const k = p.kind === "source" && p.source ? p.source.kind : p.kind;
  if (/^(embedding|token|position)$/.test(k)) return "embedding";
  if (/att|head|atom|source/.test(k)) return "attention";
  if (/mlp|neuron/.test(k)) return "mlp";
  if (/ple/.test(k)) return "ple";
  if (/bias|offset/.test(k)) return "bias";
  return "other";
}
export const valueOf = (p: Part, m: Metric): number | null =>
  p[m] ?? p.activation_contribution ?? null;
export const layerOf = (p: Part, parent?: Part): number =>
  p.layer ?? p.source?.layer ?? parent?.layer ?? 0;

/* ---------- colour ---------- */
const clamp01 = (x: number) => Math.min(1, Math.max(0, x));
export function turbo(t: number): string {
  t = clamp01(t);
  const r = 0.13572138 + t * (4.6153926 + t * (-42.66032258 + t * (132.13108234 + t * (-152.94239396 + t * 59.28637943))));
  const g = 0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (-14.18503333 + t * (4.27729857 + t * 2.82956604))));
  const b = 0.1066733 + t * (12.64194608 + t * (-60.58204836 + t * (110.36276771 + t * (-89.90310912 + t * 27.34824973))));
  return `rgb(${Math.round(clamp01(r) * 255)},${Math.round(clamp01(g) * 255)},${Math.round(clamp01(b) * 255)})`;
}

/* ---------- deterministic randomness ---------- */
export function hash(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
  return h >>> 0;
}
export function rng(seed: number) {
  let a = seed || 1;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* ---------- scene types ---------- */
export type Pt = { x: number; y: number; r: number; c: string; a: number; part: string; delay: number };
export type Ln = { x0: number; y0: number; x1: number; y1: number; c: string; a: number; w: number; part: string; delay: number; curve: boolean };
export type Node = { x: number; y: number; r: number; part: string };
export type Tag = { x: number; y: number; text: string; style: "tag" | "marker" | "label" | "axis"; color?: string; align?: "left" | "right" | "center" };
export type Box = { x: number; y: number; w: number; h: number; label: string; part: string; depth: number };
export type Scene = { points: Pt[]; lines: Ln[]; nodes: Node[]; tags: Tag[]; boxes: Box[]; grid: Ln[]; frame: { x: number; y: number; w: number; h: number }; drawn: number };

const empty = (w: number, h: number): Scene => ({ points: [], lines: [], nodes: [], tags: [], boxes: [], grid: [], frame: { x: 0, y: 0, w, h }, drawn: 0 });

export type Input = {
  parts: Part[]; // top-level components
  metric: Metric;
  layers: number;
  hidden: Set<Family>;
  output: { id: number; text: string; probability: number }[];
  target: string;
  mode: string;
};

/** The units a scene draws: top-level components, or their children when there are only one or two. */
export function units(parts: Part[]): { part: Part; parent?: Part }[] {
  if (parts.length > 2) return parts.map((part) => ({ part }));
  return parts.flatMap((parent) =>
    parent.children?.length ? parent.children.map((part) => ({ part, parent })) : [{ part: parent }],
  );
}

/* ====================================================================== */
/* FLOW — rows are component types, x is the layer, lines run to the target */
/* ====================================================================== */
export function flowScene(inp: Input, W: number, H: number): Scene {
  const s = empty(W, H);
  const items = units(inp.parts).filter(({ part }) => !inp.hidden.has(familyOf(part)));
  if (!items.length || W < 200) return s;
  const fams = FAMILIES.filter((f) => items.some(({ part }) => familyOf(part) === f.id));
  const left = Math.min(150, W * 0.16), right = Math.min(230, W * 0.24), top = 96, bottom = 170;
  const pw = W - left - right, ph = H - top - bottom;
  s.frame = { x: left - 24, y: top - 30, w: pw + right - 12, h: ph + 56 };
  const cols = inp.layers + 1;
  const colX = (p: Part, parent?: Part) => left + ((p.kind === "embedding" ? 0 : layerOf(p, parent) + 1) + 0.5) / cols * pw;
  const rowH = ph / fams.length;
  const rowY = (f: Family) => top + (fams.findIndex((x) => x.id === f) + 0.5) * rowH;
  const max = Math.max(...items.map(({ part }) => Math.abs(valueOf(part, inp.metric) ?? 0)), 1e-6);
  const hubX = left + pw + 40;
  const hubPlus = { x: hubX, y: top + ph * 0.3 }, hubMinus = { x: hubX, y: top + ph * 0.72 };

  // grid
  const step = Math.max(1, Math.ceil(cols / 60));
  for (let c = 0; c <= cols; c++) {
    const x = left + (c / cols) * pw;
    s.grid.push({ x0: x, y0: top - 12, x1: x, y1: top + ph + 8, c: "#151515", a: 1, w: 1, part: "", delay: 0, curve: false });
  }
  for (let c = 0; c < cols; c += Math.max(step, Math.ceil(cols / 12))) {
    const x = left + ((c + 0.5) / cols) * pw;
    s.tags.push({ x, y: top + ph + 22, text: c === 0 ? "In" : `S${c - 1}`, style: "axis", align: "center" });
  }
  fams.forEach((f) => {
    const y = rowY(f.id);
    s.grid.push({ x0: left - 8, y0: y, x1: left + pw, y1: y, c: "#1c1c1c", a: 1, w: 1, part: "", delay: 0, curve: false });
    s.tags.push({ x: left - 20, y, text: f.label, style: "label", color: f.color, align: "right" });
  });

  // chain mode (probe, trace): parts feed their component, the component feeds the target
  const chain = items.some((i) => i.parent);
  const parents = [...new Map(items.filter((i) => i.parent).map((i) => [i.parent!.id, i.parent!])).values()];
  const anchor = new Map<string, { x: number; y: number }>();
  parents.forEach((p, i) => {
    const x = Math.min(left + pw - 20, colX(p) + 0);
    const y = rowY(familyOf(p)) + (i - (parents.length - 1) / 2) * 26;
    anchor.set(p.id, { x, y });
    const v = valueOf(p, inp.metric) ?? 0;
    const hub = v >= 0 ? hubPlus : hubMinus;
    const k = Math.min(1, Math.abs(v) / Math.max(1e-6, ...parents.map((q) => Math.abs(valueOf(q, inp.metric) ?? 0))));
    s.lines.push({ x0: x, y0: y, x1: hub.x, y1: hub.y, c: "#f2f2f2", a: 0.35 + 0.5 * k, w: 1 + 1.5 * k, part: p.id, delay: 0.8, curve: true });
    s.nodes.push({ x, y, r: 6, part: p.id });
    const act = (p as Part & { activation?: number }).activation;
    s.tags.push({ x: x - 12, y: y - 18, align: "right", text: `${p.label}${act != null ? ` · activation ${act > 0 ? "+" : "−"}${Math.abs(act).toFixed(2)}` : ""} · ${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v).toFixed(3)}`, style: "tag" });
  });
  // probe: all parts share one layer, so spread them as a ranked column left of the component
  const fan = chain && items.every(({ part, parent }) => !parent || layerOf(part, parent) === layerOf(parent));
  const ranked = [...items].sort((a, b) => (valueOf(b.part, inp.metric) ?? 0) - (valueOf(a.part, inp.metric) ?? 0));

  // components + their parts
  items.forEach(({ part, parent }) => {
    const v = valueOf(part, inp.metric) ?? 0;
    const f = familyOf(part), color = familyColor(f);
    const k = Math.abs(v) / max;
    const r = rng(hash(part.id));
    let x = colX(part, parent) + (r() - 0.5) * Math.min(6, pw / cols / 3);
    let y = rowY(f) + (r() - 0.5) * rowH * 0.35;
    if (fan && parent) {
      const a = anchor.get(parent.id)!;
      const rank = ranked.findIndex((i) => i.part.id === part.id);
      x = Math.max(left + 40, a.x - Math.max(160, pw * 0.45));
      y = top + ((rank + 0.5) / ranked.length) * ph;
      s.tags.push({ x: x - 14, y, text: `${part.label}  ${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v).toFixed(3)}`, style: "axis", align: "right" });
    } else if (part.kind === "remainder" && parent) {
      x = Math.max(left + 10, colX(parent) - pw * 0.08);
    }
    const delay = (x - left) / pw;
    const to = parent ? anchor.get(parent.id)! : v >= 0 ? hubPlus : hubMinus;
    s.lines.push({ x0: x, y0: y, x1: to.x, y1: to.y, c: color, a: 0.1 + 0.6 * k, w: 0.6 + 1.8 * k, part: part.id, delay, curve: true });
    s.nodes.push({ x, y, r: 1.6 + 3.2 * Math.sqrt(k), part: part.id });
    // its measured parts: one dot each, feeding the component node
    const kids = (part.children || []).slice(0, 24);
    const kmax = Math.max(...kids.map((c) => Math.abs(valueOf(c, inp.metric) ?? 0)), 1e-6);
    kids.forEach((c, j) => {
      const cv = Math.abs(valueOf(c, inp.metric) ?? 0) / kmax;
      const cy = y + (j - (kids.length - 1) / 2) * Math.min(5, (rowH * 0.7) / Math.max(1, kids.length));
      const cx = x - 14 - r() * 18;
      s.lines.push({ x0: cx, y0: cy, x1: x, y1: y, c: color, a: 0.08 + 0.3 * cv, w: 0.5, part: c.id, delay, curve: true });
      s.points.push({ x: cx, y: cy, r: 1.2 + 1.4 * cv, c: color, a: 0.5 + 0.5 * cv, part: c.id, delay });
    });
  });

  // hubs and output matrix
  s.nodes.push({ x: hubPlus.x, y: hubPlus.y, r: 4, part: "" }, { x: hubMinus.x, y: hubMinus.y, r: 3, part: "" });
  s.tags.push({ x: hubPlus.x + 10, y: hubPlus.y - 18, text: `Supports ${inp.target}`, style: "tag" });
  s.tags.push({ x: hubMinus.x + 10, y: hubMinus.y - 18, text: "Suppresses", style: "tag" });
  const mx = hubX + 26, colsM = 10, gap = 5;
  let my = hubPlus.y + 12;
  inp.output.slice(0, 5).forEach((t, i) => {
    const n = Math.max(1, Math.round(t.probability * 120));
    const c = turbo(0.9 - i * 0.17);
    for (let j = 0; j < n; j++) {
      s.points.push({ x: mx + (j % colsM) * gap, y: my + Math.floor(j / colsM) * gap, r: 2.2, c, a: 0.95, part: `out-${t.id}`, delay: 1 });
    }
    s.tags.push({ x: mx + colsM * gap + 6, y: my + 2, text: `${t.text.replace(/ /g, "␣")} ${(t.probability * 100).toFixed(1)}%`, style: "axis", align: "left" });
    my += Math.max(1, Math.ceil(n / colsM)) * gap + 12;
  });
  s.drawn = items.length;
  return s;
}

/* ====================================================================== */
/* UNITS — particle treemap, area is the size of the effect               */
/* ====================================================================== */
type Rect = { x: number; y: number; w: number; h: number };
function squarify<T>(items: { v: number; item: T }[], rect: Rect): (Rect & { item: T })[] {
  const out: (Rect & { item: T })[] = [];
  const sum = items.reduce((a, b) => a + b.v, 0);
  if (!sum || rect.w <= 0 || rect.h <= 0) return out;
  const scale = (rect.w * rect.h) / sum;
  let rest = items.filter((i) => i.v > 0).sort((a, b) => b.v - a.v).map((i) => ({ ...i, a: i.v * scale }));
  let { x, y, w, h } = rect;
  const worst = (row: { a: number }[], side: number) => {
    const s = row.reduce((a, b) => a + b.a, 0);
    const mx = Math.max(...row.map((r) => r.a)), mn = Math.min(...row.map((r) => r.a));
    return Math.max((side * side * mx) / (s * s), (s * s) / (side * side * mn));
  };
  while (rest.length) {
    const side = Math.min(w, h);
    const row = [rest[0]];
    let i = 1;
    while (i < rest.length && worst([...row, rest[i]], side) <= worst(row, side)) row.push(rest[i++]);
    const s = row.reduce((a, b) => a + b.a, 0);
    if (w >= h) {
      const cw = s / h;
      let yy = y;
      for (const r of row) { const rh = r.a / cw; out.push({ x, y: yy, w: cw, h: rh, item: r.item }); yy += rh; }
      x += cw; w -= cw;
    } else {
      const rh = s / w;
      let xx = x;
      for (const r of row) { const rw = r.a / rh; out.push({ x: xx, y, w: rw, h: rh, item: r.item }); xx += rw; }
      y += rh; h -= rh;
    }
    rest = rest.slice(i);
  }
  return out;
}

/** How many units a remainder stands for ("other 2296 neurons"). */
export function restCount(p: Part): number {
  const n = (p as Part & { count?: number }).count ?? Number(/other (\d+)/.exec(p.label)?.[1]);
  return Number.isFinite(n) && n > 0 ? n : 1;
}
/** Most points drawn for the units of one remainder; a box says so when it holds more. */
export const REST_CAP = 3000;

/**
 * One point per unit of `owner`, laid out as a grid inside the box. Units measured on their own are bright and
 * sized by their share; the units of a remainder are small and dim. A part without parts is one point.
 */
function dots(s: Scene, r: Rect, color: string, alpha: number, owner: Part, m: Metric, delay: number) {
  if (r.w < 6 || r.h < 6) return;
  const list = owner.children?.length ? owner.children : [owner];
  const measured = list.filter((c) => c.kind !== "remainder");
  const rests = list.filter((c) => c.kind === "remainder");
  // a part's own parts carry no total effect; their direct contribution then sets the brightness
  const val = (c: Part) => Math.abs(valueOf(c, m) ?? c.direct ?? 0);
  const max = Math.max(...measured.map(val), 1e-6);
  const dim = rests.map((c) => Math.min(REST_CAP, restCount(c)));
  const n = measured.length + dim.reduce((x, y) => x + y, 0);
  const cols = Math.max(1, Math.min(n, Math.round(Math.sqrt((n * r.w) / r.h))));
  const rows = Math.ceil(n / cols);
  const cw = r.w / cols, ch = r.h / rows;
  const at = (i: number) => ({ x: r.x + ((i % cols) + 0.5) * cw, y: r.y + (Math.floor(i / cols) + 0.5) * ch });
  // measured units spread evenly through the grid, the remainder fills the cells between them
  const stride = n / Math.max(1, measured.length);
  const taken = new Set<number>();
  measured.forEach((c, i) => {
    const cell = Math.min(n - 1, Math.floor((i + 0.5) * stride));
    taken.add(cell);
    const share = val(c) / max;
    s.points.push({ ...at(cell), r: Math.max(2.4, Math.min(Math.min(cw, ch) * 0.8, 2.4 + 5 * Math.sqrt(share))), c: color, a: alpha * (0.45 + 0.55 * share), part: c.id, delay });
  });
  let cell = 0;
  rests.forEach((c, k) => {
    for (let j = 0; j < dim[k]; j++) {
      while (taken.has(cell)) cell++;
      s.points.push({ ...at(cell++), r: Math.max(1, Math.min(1.6, Math.min(cw, ch) * 0.5)), c: color, a: alpha * 0.28, part: c.id, delay });
    }
  });
}

export function unitScene(inp: Input, W: number, H: number): Scene {
  const s = empty(W, H);
  const items = units(inp.parts).filter(({ part }) => !inp.hidden.has(familyOf(part)));
  if (!items.length || W < 200) return s;
  const area = { x: 24, y: 64, w: W - 48 - (W > 900 ? 270 : 0), h: H - 64 - 150 };
  s.frame = { x: area.x - 10, y: area.y - 10, w: area.w + 20, h: area.h + 20 };
  const vals = items.map(({ part }) => ({ v: Math.abs(valueOf(part, inp.metric) ?? 0), item: part }));
  const sorted = [...vals].sort((a, b) => b.v - a.v);
  const top = sorted.slice(0, 40);
  const restV = sorted.slice(40).reduce((a, b) => a + b.v, 0);
  const rest: Part = { id: "__rest", label: `${sorted.length - top.length} more`, layer: null, kind: "remainder", direct: null, total: null };
  const cells = squarify([...top, ...(restV > 0 ? [{ v: restV, item: rest }] : [])], area);
  const max = Math.max(...top.map((t) => t.v), 1e-6);
  cells.forEach((c, i) => {
    const p = c.item;
    const v = valueOf(p, inp.metric) ?? 0;
    const color = p.id === "__rest" ? "#f2f2f2" : familyColor(familyOf(p));
    const pad = 2;
    const box = { x: c.x + pad, y: c.y + pad, w: c.w - 2 * pad, h: c.h - 2 * pad };
    s.boxes.push({ ...box, label: `${p.label}${p.id === "__rest" ? "" : `  ${v > 0 ? "+" : ""}${v.toFixed(3)}`}`, part: p.id, depth: 0 });
    const alpha = v < 0 ? 0.45 : 0.85;
    const delay = i / cells.length;
    const kids = (p.children || []).filter((k) => Math.abs(valueOf(k, inp.metric) ?? 0) > 0);
    const inner = { x: box.x + 6, y: box.y + 26, w: box.w - 12, h: box.h - 32 };
    if (kids.length > 1 && inner.w > 120 && inner.h > 80) {
      const sub = squarify(kids.slice(0, 12).map((k) => ({ v: Math.abs(valueOf(k, inp.metric) ?? 0), item: k })), inner);
      sub.forEach((sc) => {
        const sb = { x: sc.x + 2, y: sc.y + 2, w: sc.w - 4, h: sc.h - 4 };
        if (sb.w > 50 && sb.h > 34) {
          s.boxes.push({ ...sb, label: sc.item.label, part: sc.item.id, depth: 1 });
          dots(s, { x: sb.x, y: sb.y + 20, w: sb.w, h: sb.h - 20 }, color, alpha, sc.item, inp.metric, delay);
        } else dots(s, sb, color, alpha, sc.item, inp.metric, delay);
      });
    } else if (box.h > 30) dots(s, { x: box.x, y: box.y + 24, w: box.w, h: box.h - 24 }, color, alpha, p, inp.metric, delay);
    else dots(s, box, color, alpha, p, inp.metric, delay);
  });
  s.drawn = items.length;
  return s;
}

export function buildScene(view: View, inp: Input, W: number, H: number): Scene {
  return view === "flow" ? flowScene(inp, W, H) : unitScene(inp, W, H);
}

/* ====================================================================== */
/* drawing                                                                 */
/* ====================================================================== */
export function draw(ctx: CanvasRenderingContext2D, s: Scene, lit: Set<string> | null, t: number, hover: string) {
  const { width, height } = ctx.canvas;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.globalCompositeOperation = "source-over";
  ctx.globalAlpha = 1;
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, width, height);
  const dpr = window.devicePixelRatio || 1;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const fade = (delay: number) => clamp01((t - delay * 0.5) / 0.5);
  const dim = (part: string) => (lit && !lit.has(part) ? 0.08 : 1);

  for (const g of s.grid) {
    ctx.strokeStyle = g.c; ctx.lineWidth = 1; ctx.beginPath();
    ctx.moveTo(Math.round(g.x0) + 0.5, Math.round(g.y0) + 0.5); ctx.lineTo(Math.round(g.x1) + 0.5, Math.round(g.y1) + 0.5); ctx.stroke();
  }
  ctx.globalCompositeOperation = "lighter";
  for (const l of s.lines) {
    const a = l.a * dim(l.part) * fade(l.delay);
    if (a < 0.004) continue;
    ctx.globalAlpha = Math.min(1, a * (l.part === hover ? 2.5 : 1));
    ctx.strokeStyle = l.c; ctx.lineWidth = l.w;
    ctx.beginPath(); ctx.moveTo(l.x0, l.y0);
    if (l.curve) { const mx = (l.x0 + l.x1) / 2; ctx.bezierCurveTo(mx, l.y0, mx, l.y1, l.x1, l.y1); }
    else ctx.lineTo(l.x1, l.y1);
    ctx.stroke();
  }
  for (const p of s.points) {
    const a = p.a * dim(p.part) * fade(p.delay);
    if (a < 0.004) continue;
    ctx.globalAlpha = a; ctx.fillStyle = p.c;
    const r = p.part === hover ? p.r * 1.8 : p.r;
    ctx.fillRect(p.x - r / 2, p.y - r / 2, r, r);
  }
  ctx.globalCompositeOperation = "source-over";
  for (const n of s.nodes) {
    ctx.globalAlpha = (n.part ? dim(n.part) : 1) * fade(0);
    ctx.fillStyle = "#f2f2f2";
    ctx.beginPath(); ctx.arc(n.x, n.y, n.part === hover ? n.r + 2 : n.r, 0, Math.PI * 2); ctx.fill();
  }
  ctx.globalAlpha = 1;
}

/** Nearest component under the pointer. */
export function hit(s: Scene, x: number, y: number): string {
  let best = "", bd = 100;
  for (const n of s.nodes) { const d = (n.x - x) ** 2 + (n.y - y) ** 2; if (n.part && d < bd + n.r * n.r) { bd = d; best = n.part; } }
  if (best) return best;
  bd = 36;
  for (const p of s.points) { const d = (p.x - x) ** 2 + (p.y - y) ** 2; if (d < bd && !p.part.startsWith("out-")) { bd = d; best = p.part; } }
  if (best) return best;
  for (let i = s.boxes.length - 1; i >= 0; i--) { const b = s.boxes[i]; if (x >= b.x && x <= b.x + b.w && y >= b.y && y <= b.y + b.h) return b.part; }
  return "";
}
