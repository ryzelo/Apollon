// The particle canvas: builds the scene for a view, draws it, and reports what the pointer is on.
import React, { useEffect, useMemo, useRef, useState } from "react";
import { Prob, flatten, fmt } from "../api";
import { Family, Metric, Part, Scene, View, buildScene, draw, hit, valueOf } from "../scene";

export function Canvas({ view, parts, metric, layers, hidden, output, target, mode, lit, onPick, onBrush }: {
  view: View; parts: Part[]; metric: Metric; layers: number; hidden: Set<Family>; mode: string;
  output: Prob[]; target: string; lit: Set<string> | null; onPick: (id: string) => void;
  /** the parts inside a rectangle the user dragged */
  onBrush: (ids: string[]) => void;
}) {
  const [box, setBox] = useState<{ x0: number; y0: number; x1: number; y1: number } | null>(null);
  const down = useRef<{ x: number; y: number } | null>(null);
  const dragged = useRef(false);
  const wrap = useRef<HTMLDivElement>(null), canvas = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const [hover, setHover] = useState<{ id: string; x: number; y: number } | null>(null);
  const progress = useRef(1);
  const scene = useMemo<Scene | null>(
    () => (size.w ? buildScene(view, { parts, metric, layers, hidden, output, target, mode }, size.w, size.h) : null),
    [view, parts, metric, layers, hidden, output, target, mode, size],
  );
  const byId = useMemo(() => flatten(parts), [parts]);
  useEffect(() => {
    const el = wrap.current!;
    const ro = new ResizeObserver(() => setSize({ w: el.clientWidth, h: el.clientHeight }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  useEffect(() => {
    const c = canvas.current;
    if (!c || !scene) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = Math.round(size.w * dpr); c.height = Math.round(size.h * dpr);
  }, [size, scene]);
  // fade in whenever the scene is rebuilt from new data or a new view
  useEffect(() => { progress.current = 0; }, [view, parts]);
  useEffect(() => {
    const c = canvas.current;
    if (!c || !scene) return;
    const ctx = c.getContext("2d")!;
    let raf = 0, start = performance.now() - progress.current * 900;
    const frame = (now: number) => {
      progress.current = Math.min(1, (now - start) / 900);
      draw(ctx, scene, lit, progress.current, hover?.id || "");
      if (progress.current < 1) raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  }, [scene, lit, hover?.id]);
  const hovered = hover ? byId.get(hover.id)?.part : undefined;
  return (
    <div className="ap-canvas" ref={wrap}
      onPointerDown={(e) => {
        const r = wrap.current!.getBoundingClientRect();
        down.current = { x: e.clientX - r.left, y: e.clientY - r.top };
        dragged.current = false;
        wrap.current!.setPointerCapture(e.pointerId);
      }}
      onPointerMove={(e) => {
        if (!scene) return;
        const r = wrap.current!.getBoundingClientRect();
        const x = e.clientX - r.left, y = e.clientY - r.top;
        const d = down.current;
        if (d && (dragged.current || Math.hypot(x - d.x, y - d.y) > 5)) {
          dragged.current = true;
          setHover(null);
          setBox({ x0: Math.min(d.x, x), y0: Math.min(d.y, y), x1: Math.max(d.x, x), y1: Math.max(d.y, y) });
          return;
        }
        const id = hit(scene, x, y);
        setHover(id ? { id, x, y } : null);
      }}
      onPointerUp={() => {
        down.current = null;
        if (!dragged.current || !box || !scene) return;
        const inside = (x: number, y: number) => x >= box.x0 && x <= box.x1 && y >= box.y0 && y <= box.y1;
        const ids = new Set<string>();
        scene.nodes.forEach((n) => { if (n.part && inside(n.x, n.y)) ids.add(n.part); });
        scene.points.forEach((p) => { if (p.part && !p.part.startsWith("out-") && inside(p.x, p.y)) ids.add(p.part); });
        scene.boxes.forEach((b) => { if (b.part !== "__rest" && inside(b.x + b.w / 2, b.y + b.h / 2)) ids.add(b.part); });
        setBox(null);
        onBrush([...ids]);
      }}
      onPointerLeave={() => setHover(null)}
      onClick={() => { if (!dragged.current) onPick(hover?.id || ""); }}
      style={{ cursor: hover ? "pointer" : "crosshair" }}
    >
      {box && <div className="np-brush" style={{ left: box.x0, top: box.y0, width: box.x1 - box.x0, height: box.y1 - box.y0, zIndex: 3 }} />}
      <canvas ref={canvas} aria-label="Component visualisation. Use the table view for keyboard access." />
      {scene && (
        <>
          <div className="np-crop" style={{ left: scene.frame.x, top: scene.frame.y, width: scene.frame.w, height: scene.frame.h }}><i /><i /><i /><i /></div>
          {scene.boxes.map((b) => (
            <div key={b.part + b.depth} className={`np-box ap-box ${lit && !lit.has(b.part) ? "is-dim" : ""}`} style={{ left: b.x, top: b.y, width: b.w, height: b.h }}>
              {b.w > 46 && b.h > 22 && <span className="label">{b.label}</span>}
            </div>
          ))}
          {scene.tags.map((t, i) => (
            <span key={i} className={`ap-tag ap-tag--${t.style} ap-align--${t.align || "left"}`} style={{ left: t.x, top: t.y, color: t.color }}>{t.text}</span>
          ))}
        </>
      )}
      {hover && hovered && (
        <span className="np-tag ap-hover" style={{ left: hover.x + 14, top: hover.y + 12 }}>
          {hovered.label}
          {hovered.direct != null && ` · direct ${fmt(hovered.direct)}`}
          {hovered.total != null && ` · total ${fmt(hovered.total)}`}
          {hovered.direct == null && hovered.total == null && ` · ${fmt(valueOf(hovered, metric))}`}
        </span>
      )}
    </div>
  );
}

