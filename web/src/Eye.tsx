// The apollon logo: a shaded eye drawn from characters, on black, in white and grey only.
// It looks around and blinks while something loads; as the mark in the sidebar it only blinks.
import React, { useEffect, useRef } from "react";

const RAMP = " .,:;irsX253hMHGS#9B&@";
const COLS = 140, ROWS = 40, CW = 7, CH = 12;
const W = COLS * CW, H = ROWS * CH;
const clamp = (n: number, a = 0, b = 1) => Math.max(a, Math.min(b, n));
const gauss = (n: number) => Math.exp(-n * n);
const noise = (x: number, y: number) => { const n = Math.sin(x * 127.1 + y * 311.7) * 43758.5453; return n - Math.floor(n); };
const smooth = (t: number) => t * t * (3 - 2 * t);
const PERIOD = 5.2;

/** How far the lid is open at a time in the blink cycle: 0 closed, 1 open. */
function lid(phase: number) {
  if (phase < 0.12) return 0;
  if (phase < 0.67) return smooth((phase - 0.12) / 0.55);
  if (phase > 4.88) return 1 - smooth((phase - 4.88) / 0.32);
  return 1;
}

/** Brightness of the character cell (i, j), 0 to 1, for a lid opening and a gaze direction. */
export function eyeCell(i: number, j: number, open: number, look: number): number {
  const x = (i - 70) / 53, y = (j - 21) / 7.1;
  const shape = Math.pow(Math.max(0, 1 - x * x), 0.79);
  const seam = -0.19 * x + 0.35 * shape;
  const top = seam - 1.42 * shape * open;
  const bottom = seam + 0.19 * shape * open;
  const fade = gauss(Math.pow(x / 1.15, 4) + Math.pow((y + 0.02) / 2.75, 4));
  const brow = -1.67 - 0.38 * shape - 0.19 * x;
  const grain = noise(i, j) - 0.5;
  let v = (0.17 + 0.085 * gauss((y + 1.05) / 0.6) + 0.05 * gauss((y - 0.9) / 0.48) + grain * 0.073) * fade;
  const hairs = gauss((y - brow + 0.05 * Math.sin(i * 0.81)) / 0.23) * gauss(Math.pow((x + 0.05) / 1.02, 4));
  v += hairs * (0.32 + grain * 0.26);
  if (Math.abs(x) < 1) {
    const upper = top - y, lower = y - bottom;
    v += shape * (0.15 * gauss((upper - 0.2) / 0.17) - 0.1 * gauss((upper - 0.43) / 0.12) + 0.16 * gauss((lower - 0.15) / 0.14));
    if (y >= top && y <= bottom) {
      const dx = (x - look) / 0.345, dy = (y + 0.28) / 0.97;
      const r = Math.hypot(dx, dy), a = Math.atan2(dy, dx);
      v = (0.64 + 0.25 * clamp((y - top) / 0.65)) * (0.71 + 0.29 * shape) + grain * 0.037;
      if (r < 1) {
        const fibers = 0.11 * Math.sin(72 * a + 5 * r) + 0.09 * Math.sin(119 * a - 9 * r) + 0.055 * Math.sin(37 * a + 13 * r);
        const ring = gauss((r - 0.62) / 0.29);
        v = (0.23 + (fibers + 0.34) * ring + 0.06 * dy) * (0.63 + 0.37 * clamp((y - top) / 0.56));
        if (r < 0.4) v = 0.085 + grain * 0.025;
        v += 0.89 * gauss((dx + 0.3) / 0.19) * gauss((dy + 0.31) / 0.19);
        v += 0.34 * gauss((dx + 0.48) / 0.09) * gauss((dy + 0.13) / 0.1);
      }
      v += 0.1 * gauss((x + 0.92) / 0.078) * gauss((y + 0.12) / 0.32);
    }
    const lash = gauss((y - top) / (0.085 + 0.032 * (1 - open)));
    if (lash > 0.27) v = v * (1 - 0.84 * lash) + 0.05 * lash;
    if (open > 0.16 && y < top && x > -0.76) {
      const f = i - (top - y) * (6 + 10 * x);
      const strand = gauss(((f % 5) + 5) % 5 / 0.75);
      const len = 0.21 + 0.28 * clamp(x + 0.25) + 0.08 * noise(Math.floor(f / 5), 8);
      if (upper < len) v *= 1 - 0.78 * strand * (1 - upper / len);
    }
  }
  return clamp(v);
}

function paint(ctx: CanvasRenderingContext2D, open: number, look: number) {
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, W, H);
  ctx.font = '11.5px "IBM Plex Mono", "Courier New", monospace';
  ctx.textBaseline = "middle";
  for (let j = 0; j < ROWS; j++) for (let i = 0; i < COLS; i++) {
    const v = eyeCell(i, j, open, look);
    const idx = Math.round(v * (RAMP.length - 1));
    if (!idx) continue;
    // neutral greys up to the interface white #f2f2f2: colour belongs to the data, not to the mark
    const g = Math.round(70 + v * 172);
    ctx.fillStyle = `rgb(${g},${g},${g})`;
    ctx.fillText(RAMP[idx], i * CW, j * CH + CH / 2);
  }
}

/** The mark at sidebar size: every second cell as a dot whose size and brightness follow the shading. */
const MW = 360, MH = Math.round((MW * H) / W);
function paintMark(ctx: CanvasRenderingContext2D, open: number) {
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, MW, MH);
  const sx = MW / (COLS / 2), sy = MH / (ROWS / 2);
  ctx.fillStyle = "#f2f2f2";
  for (let j = 0; j < ROWS; j += 2) for (let i = 0; i < COLS; i += 2) {
    const raw = (eyeCell(i, j, open, 0) + eyeCell(i + 1, j, open, 0) + eyeCell(i, j + 1, open, 0) + eyeCell(i + 1, j + 1, open, 0)) / 4;
    // at this size the skin shading is dropped, so the eye itself stands out
    const v = clamp((raw - 0.24) / 0.5);
    if (v < 0.04) continue;
    const r = 0.6 + 1.9 * v;
    ctx.globalAlpha = 0.3 + 0.7 * v;
    ctx.fillRect((i / 2 + 0.5) * sx - r, (j / 2 + 0.5) * sy - r, 2 * r, 2 * r);
  }
  ctx.globalAlpha = 1;
}

export function Eye({ mode, className = "" }: {
  /** "loading": looks around and blinks. "mark": still, and blinks now and then. */
  mode: "loading" | "mark"; className?: string;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const ctx = ref.current?.getContext("2d");
    if (!ctx) return;
    const draw = (open: number, look: number) => (mode === "mark" ? paintMark(ctx, open) : paint(ctx, open, look));
    if (matchMedia("(prefers-reduced-motion: reduce)").matches) { draw(1, 0); return; }
    let raf = 0, start = 0, previous = -Infinity, lastOpen = -1;
    const tick = (now: number) => {
      raf = requestAnimationFrame(tick);
      start ||= now;
      if (now - previous < 40) return;
      const t = (now - start) / 1000 + 0.9;
      const open = lid(t % PERIOD);
      // the mark is redrawn only while the lid moves
      if (mode === "mark" && open === lastOpen) return;
      previous = now;
      lastOpen = open;
      draw(open, 0.085 * Math.sin(t * 1.25));
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [mode]);
  return <canvas ref={ref} width={mode === "mark" ? MW : W} height={mode === "mark" ? MH : H} className={`ap-eye ap-eye--${mode} ${className}`} role="img" aria-label="apollon" />;
}
