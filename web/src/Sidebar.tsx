// Sidebar: views, the investigations with their follow-up runs, and the selection summary.
import React, { useMemo, useState } from "react";
import { Search, X } from "lucide-react";
import { Job, busy, failed, int, runLabel, shortModel } from "./api";
import { RunNode, StageView, investigations } from "./store";
import { Eye } from "./Eye";

export const VIEWS: { id: StageView; label: string; accent: string }[] = [
  { id: "space", label: "Space", accent: "#9b6bff" },
  { id: "flow", label: "Flow", accent: "#4fe37a" },
  { id: "units", label: "Units", accent: "#ff9a3c" },
  { id: "table", label: "Table", accent: "#3d9bff" },
];

/** A delete control that asks once before it acts: deleting a run removes its files for good. */
function Confirm({ label, ask, onConfirm, className = "" }: { label: React.ReactNode; ask: string; onConfirm: () => void; className?: string }) {
  const [armed, setArmed] = useState(false);
  return armed ? (
    <span className={`ap-confirm ${className}`} onMouseLeave={() => setArmed(false)}>
      <button className="np-link" onClick={(e) => { e.stopPropagation(); setArmed(false); onConfirm(); }}>{ask}</button>
    </span>
  ) : (
    <button className={`ap-confirm ${className}`} title="Delete" aria-label="Delete" onClick={(e) => { e.stopPropagation(); setArmed(true); }}>{label}</button>
  );
}

function Row({ node, current, onOpen, onDelete }: { node: RunNode; current: string; onOpen: (id: string) => void; onDelete: (ids: string[]) => void }) {
  const r = node.run;
  const root = node.depth === 0;
  return (
    <>
      <div className={`ap-run ${r.id === current ? "is-active" : ""} ${root ? "is-root" : ""}`} style={{ paddingLeft: Math.min(node.depth, 3) * 10 }}>
        <button className="ap-run-main" onClick={() => onOpen(r.id)}>
          <span className={`ap-status ap-status--${r.status}`} />
          <span className="ap-run-text">
            <span>{root ? r.request.prompt || runLabel(r.request) : runLabel(r.request)}</span>
            <small>
              {root ? `${runLabel(r.request)} · ${shortModel(r.request.model)}` : ""}
              {busy(r) ? `${root ? " · " : ""}${r.status}` : failed(r) ? `${root ? " · " : ""}${r.status}` : ""}
            </small>
          </span>
        </button>
        {!busy(r) && <Confirm className="ap-run-x" label={<X size={12} />} ask="Delete" onConfirm={() => onDelete([r.id])} />}
      </div>
      {node.children.map((c) => <Row key={c.run.id} node={c} current={current} onOpen={onOpen} onDelete={onDelete} />)}
    </>
  );
}

export function Sidebar({ runs, current, view, onView, onOpen, onNew, onDelete, online, searchRef, children }: {
  runs: Job[]; current: string; view: StageView; onView: (v: StageView) => void; onOpen: (id: string) => void;
  onNew: () => void; onDelete: (ids: string[]) => void; online: boolean; searchRef: React.RefObject<HTMLInputElement | null>; children?: React.ReactNode;
}) {
  const [query, setQuery] = useState("");
  const [showFailed, setShowFailed] = useState(false);
  const visible = useMemo(() => {
    const q = query.trim().toLowerCase();
    return q ? runs.filter((r) => `${r.request.prompt} ${r.request.kind} ${r.request.component || ""} ${r.request.model}`.toLowerCase().includes(q)) : runs;
  }, [runs, query]);
  const { roots, hidden } = useMemo(() => investigations(visible, showFailed), [visible, showFailed]);
  const failedRuns = useMemo(() => runs.filter(failed), [runs]);
  return (
    <aside className="np-sidebar">
      <div>
        <a className="ap-brand-eye" href="/" aria-label="apollon home"><Eye mode="mark" /></a>
        <a className="np-brand" href="/" tabIndex={-1} aria-hidden="true">apollon</a>
        <div className="np-spectrum ap-spectrum" />
      </div>
      <label className="np-search">
        <Search size={14} />
        <input ref={searchRef} placeholder="Search" value={query} onChange={(e) => setQuery(e.target.value)} aria-label="Search runs" />
        <kbd>⌘K</kbd>
      </label>
      <nav className="np-nav" aria-label="Views">
        {VIEWS.map((v) => (
          <a key={v.id} href={`#${v.id}`} className={view === v.id ? "is-active" : ""} onClick={(e) => { e.preventDefault(); onView(v.id); }}>{v.label}</a>
        ))}
        <a href="#new" className="np-add" onClick={(e) => { e.preventDefault(); onNew(); }}>+ New investigation</a>
      </nav>
      <div className="ap-runs">
        <span className="np-caption">Investigations · {int(roots.length)}</span>
        {roots.map((n) => <Row key={n.run.id} node={n} current={current} onOpen={onOpen} onDelete={onDelete} />)}
        {!runs.length && <p className="np-note">A prompt and its follow-up analyses appear here.</p>}
        {!!runs.length && !roots.length && query && <p className="np-note">No run matches “{query}”.</p>}
        {!!failedRuns.length && (
          <p className="ap-failed np-note">
            {showFailed ? `${int(failedRuns.length)} runs did not complete. ` : `${int(hidden.length)} runs that did not complete are left out. `}
            <button className="np-link" onClick={() => setShowFailed((s) => !s)}>{showFailed ? "Hide" : "Show"}</button>{" "}
            <Confirm label={<span className="np-link">Delete all</span>} ask={`Delete ${int(failedRuns.length)} runs for good`} onConfirm={() => onDelete(failedRuns.map((r) => r.id))} />
          </p>
        )}
      </div>
      <div className="np-stat-block">
        {children}
        <span className={`ap-engine ${online ? "is-on" : ""}`}><i />{online ? "Local engine" : "Engine unavailable"}</span>
      </div>
    </aside>
  );
}
