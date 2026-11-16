// The 3D stage: draws a Scene3 as additive particles and lines, lets the user orbit, and reports what the
// pointer is on. Labels are HTML placed over the canvas every frame.
import React, { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { Scene3 } from "../space/types";

const VERT = `
  attribute vec3 color; attribute float alpha; attribute float size; attribute float lit;
  uniform float uScale; uniform float uDim; uniform float uFade;
  varying vec3 vColor; varying float vAlpha;
  void main() {
    vColor = color;
    vAlpha = alpha * uFade * mix(1.0, mix(0.08, 1.0, lit), uDim);
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = max(1.5, size * uScale / -mv.z);
    gl_Position = projectionMatrix * mv;
  }`;
const FRAG = `
  varying vec3 vColor; varying float vAlpha;
  void main() {
    float d = length(gl_PointCoord - 0.5);
    float a = smoothstep(0.5, 0.15, d) * vAlpha;
    if (a < 0.01) discard;
    gl_FragColor = vec4(vColor * a, a);
  }`;

type Gl = {
  renderer: THREE.WebGLRenderer; scene: THREE.Scene; camera: THREE.PerspectiveCamera; controls: OrbitControls;
  points?: THREE.Points; lines?: THREE.LineSegments; material: THREE.ShaderMaterial; start: number;
};

export function Space({ scene, lit, onPick, viewKey }: {
  scene: Scene3; lit: Set<string> | null; onPick: (part: string) => void;
  /** changes when a different kind of scene is shown, so the camera starts over */
  viewKey: string;
}) {
  const wrap = useRef<HTMLDivElement>(null);
  const labels = useRef<HTMLDivElement>(null);
  const gl = useRef<Gl | null>(null);
  const data = useRef({ scene, lit });
  data.current = { scene, lit };
  const [hover, setHover] = useState<{ i: number; x: number; y: number } | null>(null);
  const hoverRef = useRef(-1);

  // renderer, camera and controls live as long as the component
  useEffect(() => {
    const el = wrap.current!;
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setClearColor(0x000000, 1);
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    el.prepend(renderer.domElement);
    const camera = new THREE.PerspectiveCamera(38, 1, 0.01, 200);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.3; // about 0.03 rad/s
    const stop = () => { controls.autoRotate = false; };
    renderer.domElement.addEventListener("pointerdown", stop);
    renderer.domElement.addEventListener("wheel", stop, { passive: true });
    const material = new THREE.ShaderMaterial({
      transparent: true, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending,
      uniforms: { uScale: { value: 1 }, uDim: { value: 0 }, uFade: { value: 0 } },
      vertexShader: VERT, fragmentShader: FRAG,
    });
    const g: Gl = { renderer, scene: new THREE.Scene(), camera, controls, material, start: performance.now() };
    gl.current = g;
    const resize = () => {
      const w = el.clientWidth, h = el.clientHeight;
      if (!w || !h) return;
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      // a point of size 1 covers about one CSS pixel at the distance of the scene's centre
      material.uniforms.uScale.value = renderer.getPixelRatio() * camera.position.length();
    };
    const ro = new ResizeObserver(resize);
    ro.observe(el);
    const v = new THREE.Vector3();
    let raf = 0;
    const frame = (now: number) => {
      raf = requestAnimationFrame(frame);
      controls.update();
      material.uniforms.uFade.value = Math.min(1, (now - g.start) / 800);
      material.uniforms.uScale.value = renderer.getPixelRatio() * camera.position.distanceTo(controls.target);
      renderer.render(g.scene, camera);
      // labels follow their anchors
      const box = labels.current, list = data.current.scene.labels;
      if (box) {
        const w = el.clientWidth, h = el.clientHeight;
        for (let i = 0; i < box.children.length && i < list.length; i++) {
          const node = box.children[i] as HTMLElement;
          v.set(...list[i].p).project(camera);
          const hidden = v.z > 1 || v.z < -1;
          node.style.display = hidden ? "none" : "";
          if (!hidden) node.style.transform = `translate(${((v.x + 1) / 2) * w}px, ${((1 - v.y) / 2) * h}px)`;
        }
      }
    };
    raf = requestAnimationFrame(frame);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      controls.dispose();
      g.points?.geometry.dispose();
      g.lines?.geometry.dispose();
      (g.lines?.material as THREE.Material | undefined)?.dispose();
      material.dispose();
      renderer.dispose();
      renderer.domElement.remove();
      gl.current = null;
    };
  }, []);

  // camera starts over for a different kind of scene
  useEffect(() => {
    const g = gl.current;
    if (!g) return;
    const [ex, ey, ez] = scene.extent;
    const r = Math.max(ex * 1.12, ey * 1.7, ez * 1.25);
    g.camera.position.set(r * 0.9, r * 0.7, r * 2.2);
    g.controls.target.set(0, -ey * 0.12, 0);
    g.controls.autoRotate = true;
    g.controls.update();
  }, [viewKey]);

  // geometry for the scene
  useEffect(() => {
    const g = gl.current;
    if (!g) return;
    if (g.points) { g.scene.remove(g.points); g.points.geometry.dispose(); }
    if (g.lines) { g.scene.remove(g.lines); g.lines.geometry.dispose(); (g.lines.material as THREE.Material).dispose(); }
    const n = scene.points.length;
    const pos = new Float32Array(n * 3), col = new Float32Array(n * 3), alpha = new Float32Array(n), size = new Float32Array(n), litA = new Float32Array(n).fill(1);
    const c = new THREE.Color();
    scene.points.forEach((p, i) => {
      pos.set(p.p, i * 3);
      c.set(p.c);
      col.set([c.r, c.g, c.b], i * 3);
      alpha[i] = p.a;
      size[i] = p.size;
    });
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
    geo.setAttribute("alpha", new THREE.BufferAttribute(alpha, 1));
    geo.setAttribute("size", new THREE.BufferAttribute(size, 1));
    geo.setAttribute("lit", new THREE.BufferAttribute(litA, 1));
    g.points = new THREE.Points(geo, g.material);
    g.points.frustumCulled = false;
    const m = scene.lines.length;
    const lpos = new Float32Array(m * 6), lcol = new Float32Array(m * 6);
    scene.lines.forEach((l, i) => { lpos.set(l.a, i * 6); lpos.set(l.b, i * 6 + 3); });
    const lgeo = new THREE.BufferGeometry();
    lgeo.setAttribute("position", new THREE.BufferAttribute(lpos, 3));
    lgeo.setAttribute("color", new THREE.BufferAttribute(lcol, 3));
    g.lines = new THREE.LineSegments(lgeo, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending }));
    g.lines.frustumCulled = false;
    g.scene.add(g.lines, g.points);
    g.start = performance.now();
    hoverRef.current = -1;
    setHover(null);
  }, [scene]);

  // what is lit: everything else dims to 8 %
  useEffect(() => {
    const g = gl.current;
    if (!g?.points || !g.lines) return;
    const litA = g.points.geometry.getAttribute("lit") as THREE.BufferAttribute;
    scene.points.forEach((p, i) => litA.setX(i, !lit || lit.has(p.part) ? 1 : 0));
    litA.needsUpdate = true;
    g.material.uniforms.uDim.value = lit ? 1 : 0;
    const lcol = g.lines.geometry.getAttribute("color") as THREE.BufferAttribute;
    const c = new THREE.Color();
    scene.lines.forEach((l, i) => {
      const a = l.alpha * (!lit || lit.has(l.part) ? 1 : 0.08);
      c.set(l.c);
      lcol.setXYZ(i * 2, c.r * a, c.g * a, c.b * a);
      lcol.setXYZ(i * 2 + 1, c.r * a, c.g * a, c.b * a);
    });
    lcol.needsUpdate = true;
  }, [scene, lit]);

  // the hovered point grows
  useEffect(() => {
    const g = gl.current;
    if (!g?.points) return;
    const size = g.points.geometry.getAttribute("size") as THREE.BufferAttribute;
    const prev = hoverRef.current, next = hover?.i ?? -1;
    if (prev === next) return;
    if (prev >= 0 && prev < scene.points.length) size.setX(prev, scene.points[prev].size);
    if (next >= 0) size.setX(next, scene.points[next].size * 1.8 + 3);
    size.needsUpdate = true;
    hoverRef.current = next;
  }, [hover, scene]);

  /** Nearest point under the pointer, in screen space. Measured points win over remainder points. */
  function pick(clientX: number, clientY: number) {
    const g = gl.current, el = wrap.current;
    if (!g || !el) return null;
    const rect = el.getBoundingClientRect();
    const x = clientX - rect.left, y = clientY - rect.top, w = rect.width, h = rect.height;
    const e = new THREE.Matrix4().multiplyMatrices(g.camera.projectionMatrix, g.camera.matrixWorldInverse).elements;
    let best = -1, bd = Infinity;
    const pts = data.current.scene.points;
    for (let i = 0; i < pts.length; i++) {
      const [px, py, pz] = pts[i].p;
      const cw = e[3] * px + e[7] * py + e[11] * pz + e[15];
      if (cw <= 0) continue;
      const sx = ((e[0] * px + e[4] * py + e[8] * pz + e[12]) / cw + 1) / 2 * w - x;
      const sy = (1 - (e[1] * px + e[5] * py + e[9] * pz + e[13]) / cw) / 2 * h - y;
      const reach = Math.max(7, pts[i].size * 0.8);
      const d = (sx * sx + sy * sy) / (reach * reach);
      if (d < 1 && d < bd) { bd = d; best = i; }
    }
    return best >= 0 ? { i: best, x, y } : null;
  }
  const down = useRef<{ x: number; y: number } | null>(null);
  const hovered = hover && hover.i < scene.points.length ? scene.points[hover.i] : null;
  return (
    <div className="ap-canvas ap-space" ref={wrap} style={{ cursor: hovered ? "pointer" : "grab" }}
      onPointerMove={(e) => { if (!e.buttons) setHover(pick(e.clientX, e.clientY)); }}
      onPointerLeave={() => setHover(null)}
      onPointerDown={(e) => { down.current = { x: e.clientX, y: e.clientY }; }}
      onClick={(e) => {
        // a drag orbits; only a click that stayed in place selects
        const d = down.current;
        if (d && Math.hypot(e.clientX - d.x, e.clientY - d.y) > 4) return;
        onPick(pick(e.clientX, e.clientY)?.i != null ? scene.points[pick(e.clientX, e.clientY)!.i].part : "");
      }}>
      <div className="np-crop ap-space-crop"><i /><i /><i /><i /></div>
      <div className="ap-space-labels" ref={labels}>
        {scene.labels.map((t, i) => (
          <span key={i} className="ap-space-anchor"><span className={`ap-tag ap-tag--${t.style} ap-align--${t.align || "left"}`} style={{ color: t.color }}>{t.text}</span></span>
        ))}
      </div>
      {hover && hovered?.tip && <span className="np-tag ap-hover" style={{ left: hover.x + 14, top: hover.y + 12 }}>{hovered.tip}</span>}
    </div>
  );
}
