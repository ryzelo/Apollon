// A 3D scene as plain data: points, line segments and labels in one coordinate frame.
// The builders in this folder turn a result into such a scene; stage/Space.tsx draws any of them.

export type V3 = [number, number, number];
/** One point is one measured unit, or one unit of a remainder (small and dim). */
export type P3 = { p: V3; size: number; c: string; a: number; part: string; tip?: string };
export type L3 = { a: V3; b: V3; c: string; alpha: number; part: string };
export type T3 = { p: V3; text: string; style: "axis" | "tag" | "label"; color?: string; align?: "left" | "right" | "center" };
export type Scene3 = {
  points: P3[];
  lines: L3[];
  labels: T3[];
  /** half extents of the scene around the origin, for the camera */
  extent: V3;
  /** what was drawn and what was left out, as a sentence for the legend */
  drawn: string;
};

export const emptyScene = (): Scene3 => ({ points: [], lines: [], labels: [], extent: [1, 1, 1], drawn: "" });

/** A signed value on a scale that keeps small effects visible: linear near zero, logarithmic beyond. */
export const SYM = 0.1;
export const symlog = (v: number) => Math.sign(v) * Math.log10(1 + Math.abs(v) / SYM);
/** Tick values for a symlog axis that reaches `max`. */
export function symTicks(max: number): number[] {
  const out = [0];
  for (const t of [0.1, 1, 10, 100, 1000]) if (t <= Math.max(max, 0.1) * 1.01) out.push(t, -t);
  return out;
}

/** A line bent upwards by `lift`, as short straight segments. */
export function arc(s: Scene3, a: V3, b: V3, lift: V3, c: string, alpha: number, part: string, n = 14) {
  let prev = a;
  for (let i = 1; i <= n; i++) {
    const t = i / n, k = 4 * t * (1 - t);
    const q: V3 = [a[0] + (b[0] - a[0]) * t + lift[0] * k, a[1] + (b[1] - a[1]) * t + lift[1] * k, a[2] + (b[2] - a[2]) * t + lift[2] * k];
    s.lines.push({ a: prev, b: q, c, alpha, part });
    prev = q;
  }
}

/** Deterministic pseudo-random numbers in [0, 1) from an integer seed, so a scene looks the same every time. */
export function seeded(seed: number) {
  let a = seed | 0 || 1;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
