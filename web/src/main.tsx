import "@fontsource/inter/400.css";
import "@fontsource/inter/600.css";
import "@fontsource/ibm-plex-mono/400.css";
import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { Download, Info, Search, X } from "lucide-react";
import "./noir.css";
import "./style.css";
import { Detail, Job, Request, api, artifactUrl, busy, descendants, flatten, fmt, int, neuronSpec, probeSpec, runLabel, tokenText } from "./api";
import { FAMILIES, Family, Metric, Part, familyColor, familyOf, units, valueOf } from "./scene";
import { useWorkspace } from "./store";
import { Sidebar, VIEWS } from "./Sidebar";
import { Eye } from "./Eye";
import { Draft, Form } from "./panel/Form";
import { Details, PinView } from "./panel/Details";
import { Filter, Filters } from "./stage/Filters";
import { Canvas } from "./stage/Canvas";
import { PromptStrip } from "./stage/PromptStrip";
import { CHAIN_NOTES, NOTES, SPACE_NOTES } from "./stage/notes";
import { Space } from "./stage/Space";
import { Table } from "./stage/Table";
import { modelScene } from "./space/model";
import { circuitScene } from "./space/circuit";
import { probeScene } from "./space/probe";
import { traceScene } from "./space/trace";
import { neuronScene } from "./space/neuron";

function Empty({ title, text, children, eye }: { title: string; text: string; children?: React.ReactNode; eye?: boolean }) {
  return (
    <div className="ap-empty">
      {eye && <Eye mode="loading" />}
      <h2 className="ap-h2">{title}</h2>
      {text && <p className="np-note">{text}</p>}
      {children}
    </div>
  );
}

type Pin = { sig: string; label: string };
const NO_FILTER: Filter = { lo: -1, hi: Number.MAX_SAFE_INTEGER, min: 0 };
/** What identifies a part across runs: its layer, type, unit and position, not its id in one result. */
const sigOf = (p: Part, parent?: Part) =>
  p.layer == null && p.kind === "source" ? `source|${p.label}`
    : `${p.layer ?? parent?.layer ?? ""}|${p.kind}|${p.unit ?? ""}|${p.position ?? ""}|${parent?.kind === "head" ? parent.unit : ""}`;
const columnOf = (p: Part) => (familyOf(p) === "embedding" || p.layer == null || p.layer < 0 ? -1 : p.layer);
function loadPins(): Pin[] {
  try { const v = JSON.parse(localStorage.getItem("apollon-pins") || "[]"); return Array.isArray(v) ? v : []; } catch { return []; }
}

function App() {
  const ws = useWorkspace();
  const { job, runs, ui } = ws;
  const { view, metric } = ui;
  const [draft, setDraft] = useState<Draft | null>(null);
  const [query, setQuery] = useState("");
  const [hidden, setHidden] = useState<Set<Family>>(new Set());
  const [reading, setReading] = useState(true);
  const [artifacts, setArtifacts] = useState<string[]>([]);
  const runSearch = useRef<HTMLInputElement>(null);
  const [filter, setFilter] = useState<Filter>(NO_FILTER);
  const [brush, setBrush] = useState<Set<string> | null>(null);
  const [compare, setCompare] = useState("");
  const [otherJob, setOtherJob] = useState<Job | undefined>();
  const [pins, setPins] = useState<Pin[]>(loadPins);
  const [showFilters, setShowFilters] = useState(false);

  // a result the stage can show: it has a prompt context (static and file-only runs have none)
  const result = job?.result?.tokens?.length ? job.result : undefined;
  const allParts = result?.components || [];
  const index = useMemo(() => flatten(allParts), [allParts]);
  // the filter: a decomposition is narrowed by layer and size, a probe or trace by the size of its units
  const wide = allParts.length > 2;
  const size = (p: Part) => Math.abs(valueOf(p, metric) ?? p.direct ?? 0);
  const candidates = useMemo(() => (wide ? allParts : allParts.flatMap((c) => (c.children || []).filter((u) => u.kind !== "remainder"))), [allParts, wide]);
  const filterMax = useMemo(() => Math.max(...candidates.map(size), 0), [candidates, metric]);
  const filtering = filter.min > 0 || (wide && (filter.lo > -1 || filter.hi < (job?.result?.model?.layers ?? 0) - 1));
  const components = useMemo(() => {
    if (!filtering) return allParts;
    if (wide) return allParts.filter((p) => columnOf(p) >= filter.lo && columnOf(p) <= filter.hi && size(p) >= filter.min);
    return allParts.map((c) => ({ ...c, children: (c.children || []).filter((u) => u.kind === "remainder" || size(u) >= filter.min) }));
  }, [allParts, filtering, wide, filter, metric]);
  const shownCount = wide ? components.length : components.reduce((n, c) => n + (c.children || []).filter((u) => u.kind !== "remainder").length, 0);
  const viewResult = useMemo(() => (result && components !== allParts ? { ...result, components } : result), [result, components, allParts]);
  const part = index.get(ui.sel)?.part as Detail | undefined;
  const parentPart = index.get(ui.sel)?.parent;
  const accent = VIEWS.find((v) => v.id === view)!.accent;
  const mode = result?.mode || job?.request.kind || "";
  // a static neuron result has no prompt; it is shown in space only
  const neuron = job?.status === "completed" && job.result?.mode === "neuron" ? job.result.neuron : undefined;
  const layers = job?.result?.model?.layers ?? 0;

  // the run compared with: every part's value there, found by its signature
  const other = useMemo(() => {
    if (!otherJob?.result) return undefined;
    const there = new Map<string, number | null>();
    flatten(otherJob.result.components).forEach(({ part: p, parent }) => there.set(sigOf(p, parent), valueOf(p, metric) ?? p.direct ?? null));
    const here = new Map<string, number | null>();
    index.forEach(({ part: p, parent }, id) => { const s = sigOf(p, parent); if (there.has(s)) here.set(id, there.get(s)!); });
    return here;
  }, [otherJob, index, metric]);
  const pinViews = useMemo<PinView[]>(() => {
    const bySig = new Map<string, Part>();
    index.forEach(({ part: p, parent }) => bySig.set(sigOf(p, parent), p));
    return pins.map((pin) => {
      const p = bySig.get(pin.sig);
      return p ? { ...pin, id: p.id, value: valueOf(p, metric) ?? p.direct ?? null, other: other?.get(p.id), color: familyColor(familyOf(p)) } : pin;
    });
  }, [pins, index, metric, other]);
  const pinned = !!part && pins.some((p) => p.sig === sigOf(part, parentPart));
  function togglePin() {
    if (!part) return;
    const sig = sigOf(part, parentPart);
    setPins((old) => {
      const next = old.some((p) => p.sig === sig) ? old.filter((p) => p.sig !== sig) : [...old, { sig, label: `${parentPart ? `${parentPart.label} · ` : ""}${part.label}` }];
      localStorage.setItem("apollon-pins", JSON.stringify(next));
      return next;
    });
  }
  const unpin = (sig: string) => setPins((old) => { const next = old.filter((p) => p.sig !== sig); localStorage.setItem("apollon-pins", JSON.stringify(next)); return next; });
  /** Marks several parts at once: from a rectangle in the flow view, rows of the table, or the pins. */
  const mark = (ids: string[]) => { setBrush(ids.length ? new Set(ids) : null); if (ids.length) ws.select(""); };

  // what lights up: the selection with its parts, the marked parts, or the search matches
  const lit = useMemo(() => {
    if (part) {
      const s = descendants(part);
      if (parentPart) s.add(parentPart.id);
      return s;
    }
    if (brush) {
      const s = new Set<string>();
      brush.forEach((id) => { const p = index.get(id)?.part; if (p) descendants(p, s); });
      return s;
    }
    if (query.trim()) {
      const q = query.toLowerCase(), s = new Set<string>();
      index.forEach(({ part: p }) => { if (p.label.toLowerCase().includes(q)) descendants(p, s); });
      return s;
    }
    return null;
  }, [part, parentPart, brush, query, index]);
  // the sum counts a part once: a unit is left out when the component it belongs to is marked too
  const brushInfo = useMemo(() => (brush ? {
    count: brush.size,
    sum: [...brush].reduce((a, id) => {
      const e = index.get(id);
      return !e || (e.parent && brush.has(e.parent.id)) ? a : a + (valueOf(e.part, metric) ?? 0);
    }, 0),
  } : undefined), [brush, index, metric]);

  const drawnUnits = useMemo(() => units(components), [components]);
  const chain = drawnUnits.some((u) => u.parent);
  const available = useMemo(() => {
    const all = [...index.values()].map((x) => x.part);
    return { direct: all.some((p) => p.direct != null), total: all.some((p) => p.total != null) };
  }, [index]);
  const familyStats = useMemo(() => FAMILIES.map((f) => {
    const list = drawnUnits.filter(({ part: p }) => familyOf(p) === f.id);
    return { ...f, count: list.length, sum: list.reduce((a, { part: p }) => a + (valueOf(p, metric) ?? 0), 0) };
  }).filter((f) => f.count), [drawnUnits, metric]);

  const space = useMemo(() => {
    if (neuron) return neuronScene(neuron, layers);
    if (view !== "space" || !result || !components.length) return null;
    const r = viewResult!;
    return mode === "probe" ? probeScene(r, metric) : mode === "trace" ? traceScene(r) : mode === "circuit" ? circuitScene(r, hidden) : modelScene(r, metric, hidden, other);
  }, [neuron, layers, view, viewResult, components, mode, metric, hidden, other]);

  useEffect(() => { setQuery(""); setFilter(NO_FILTER); setBrush(null); setCompare(""); }, [ui.run]);
  // the filter's layer range ends at the model's last layer
  useEffect(() => { if (layers) setFilter((f) => (f.hi > layers - 1 ? { ...f, hi: layers - 1 } : f)); }, [layers, ui.run]);
  useEffect(() => {
    setOtherJob(undefined);
    if (!compare) return;
    let active = true;
    void api<Job>(`/api/runs/${compare}`).then((d) => { if (active) setOtherJob(d); }).catch(() => {});
    return () => { active = false; };
  }, [compare]);
  useEffect(() => { if (ws.loaded && !runs.length) setDraft((d) => d || { kind: "generate" }); }, [ws.loaded, runs.length]);
  useEffect(() => {
    setArtifacts([]);
    if (!job || job.status !== "completed") return;
    let active = true;
    api<string[]>(`/api/runs/${job.id}/artifacts`).then((files) => { if (active) setArtifacts(files); }).catch(() => {});
    return () => { active = false; };
  }, [job?.id, job?.status]);
  useEffect(() => {
    const other = metric === "direct" ? "total" : "direct";
    if (components.length && !available[metric] && available[other]) ws.setMetric(other);
  }, [available, metric, components.length]);
  // keyboard: the handler is replaced every render so it always sees the current selection
  const onKey = useRef<(e: KeyboardEvent) => void>(() => {});
  useEffect(() => {
    const key = (e: KeyboardEvent) => onKey.current(e);
    document.addEventListener("keydown", key);
    return () => document.removeEventListener("keydown", key);
  }, []);

  /** Starts a follow-up of the open run, or opens the run that already answered the same question. */
  function follow(extra: Partial<Request>) {
    if (!job) return;
    const r = job.request;
    const request: Request = {
      model: r.model, prompt: r.prompt, answer: r.answer, chat: r.chat, kind: r.kind,
      ...(r.target ? { target: r.target } : {}),
      options: r.options?.pos ? { pos: r.options.pos } : {},
      parent_id: job.id,
      ...extra,
    };
    if (extra.answer && extra.answer !== r.answer) delete request.target;
    if (request.component == null) delete request.component;
    const same = (o: Job) =>
      o.status === "completed" && o.request.kind === request.kind && o.request.model === request.model && o.request.prompt === request.prompt
      && o.request.answer === request.answer && !!o.request.chat === !!request.chat && (o.request.component || "") === (request.component || "")
      && (o.request.target || "") === (request.target || "") && JSON.stringify(o.request.options || {}) === JSON.stringify(request.options || {});
    const done = ["dump", "tokens"].includes(request.kind) ? undefined : runs.find(same);
    if (done) ws.open(done.id); else void ws.launch(request);
  }
  function exportResult() {
    if (!result) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(result, null, 2)], { type: "application/json" }));
    const a = document.createElement("a");
    a.href = url; a.download = `apollon-${job?.id}.json`; a.click();
    URL.revokeObjectURL(url);
  }
  const edit = () => job && setDraft({ ...job.request, parent_id: job.request.parent_id || undefined });

  /* ------------------------------------------------ rankings for the panel */
  const rankBase = chain ? drawnUnits.map((u) => u.part) : components;
  const ranked = useMemo(() => [...rankBase].sort((a, b) => Math.abs(valueOf(b, metric) ?? 0) - Math.abs(valueOf(a, metric) ?? 0)), [rankBase, metric]);
  const rest = drawnUnits.find((u) => u.part.kind === "remainder")?.part as Detail | undefined;
  const rankTitle = mode === "trace" ? `Strongest sources of ${components[0]?.label}` : mode === "probe" ? `Units of ${components.map((c) => c.label).join(", ")}` : mode === "circuit" ? "Blocks of the circuit" : "Strongest contributions";
  const rankNote = mode === "circuit"
    ? `Every layer’s attention, MLP and PLE steps that stay in the circuit. A value is the sum, over its leaves, of the target’s log-odds lost when that leaf alone is removed from the circuit. The ${Math.min(8, ranked.length)} largest of ${int(ranked.length)} blocks.`
    : mode === "trace"
    ? `Each source’s share of the neuron’s activation of ${fmt((components[0] as Detail | undefined)?.activation)}.${rest ? ` The ${int(rest.count || 0)} remaining sources add ${fmt(rest.activation_contribution)}.` : ""}`
    : mode === "probe"
      ? `Change of the target’s log-odds when one unit is removed and the model reruns. The ${Math.min(8, ranked.length)} largest of ${int(ranked.length)} units.`
      : `The ${Math.min(8, ranked.length)} largest of ${int(ranked.length)} components, by ${metric === "direct" ? "direct contribution" : "total effect"}. Choose one to light it on the stage.`;

  onKey.current = (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") { e.preventDefault(); runSearch.current?.focus(); return; }
    if (e.key === "Escape") { if (draft) setDraft(null); else if (part) ws.select(""); else setBrush(null); return; }
    const el = e.target as HTMLElement | null;
    if (draft || e.metaKey || e.ctrlKey || e.altKey || (el && /^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName))) return;
    if (!components.length) return;
    const base = parentPart && !chain ? parentPart : part; // layer steps move between whole components
    if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
      if (!base || chain) return;
      const lane = components.filter((p) => familyOf(p) === familyOf(base)).sort((a, b) => columnOf(a) - columnOf(b));
      const next = lane[lane.findIndex((p) => p.id === base.id) + (e.key === "ArrowRight" ? 1 : -1)];
      if (next) { e.preventDefault(); ws.select(next.id); }
    } else if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      const i = part ? ranked.findIndex((p) => p.id === part.id) : -1;
      const next = ranked[i < 0 ? 0 : i + (e.key === "ArrowDown" ? 1 : -1)];
      if (next) { e.preventDefault(); ws.select(next.id); }
    } else if (part && e.key === "Enter") {
      const probe = probeSpec(part), neuronName = neuronSpec(part, parentPart);
      if (probe) follow({ kind: "probe", component: probe }); else if (neuronName) follow({ kind: "trace", component: neuronName });
    } else if (part && e.key.toLowerCase() === "t") {
      const neuronName = neuronSpec(part, parentPart);
      if (neuronName) follow({ kind: "trace", component: neuronName });
    } else if (part && e.key.toLowerCase() === "p") togglePin();
  };

  /* ------------------------------------------------ stage */
  const report = artifacts.find((f) => f.endsWith(".html"));
  const tableRows = [...flatten(components).values()].filter(({ part: p }) => !query || p.label.toLowerCase().includes(query.toLowerCase())) as { part: Detail; parent?: Part }[];
  let stageBody: React.ReactNode;
  if (!ws.loaded && ws.online !== false && !job) stageBody = <Empty eye title="Connecting to the local engine" text="" />;
  else if (!job) stageBody = (
    <Empty eye title="A closer look at one prediction" text="Give apollon a local model and a prompt. It generates the answer, and each answer token can then be taken apart: which layers support it, what they depend on, and what the model says when a part is switched off.">
      <button className="np-btn np-btn--primary" onClick={() => setDraft({ kind: "generate" })}>New investigation</button>
    </Empty>
  );
  else if (busy(job)) stageBody = (
    <Empty eye title={job.status === "queued" ? "Waiting for the engine" : `${runLabel(job.request)} · computing`} text="The engine is calculating real model contributions. The view fills when the run completes.">
      <div className="np-progress ap-progress" />
      {job.progress && <p className="ap-mono">{job.progress}</p>}
      <button className="np-btn" onClick={() => void ws.cancel(job.id)}>Cancel run</button>
    </Empty>
  );
  else if (job.status === "failed" || job.status === "interrupted") stageBody = (
    <Empty title="The analysis could not complete" text="">
      <pre className="ap-errlog">{(job.error || "The engine returned an error.").slice(-1200)}</pre>
      <button className="np-btn" onClick={edit}>Edit and retry</button>
    </Empty>
  );
  else if (job.status === "cancelled") stageBody = (
    <Empty title="Run cancelled" text="Nothing was measured."><button className="np-btn" onClick={edit}>Edit and retry</button></Empty>
  );
  else if (neuron && space) stageBody = <Space scene={space} lit={null} viewKey={`neuron-${job.id}`} onPick={() => {}} />;
  else if (!result) stageBody = job.request.kind === "atlas" ? (
    <Empty title="Neuron atlas ready" text={`Built for layers ${job.request.options?.["atlas-layers"] || "all"} with ${job.request.options?.["atlas-k"] || "64"} entries per neuron. Trace a neuron of the same model to use it.`}>
      <button className="np-btn" onClick={() => setDraft({ kind: "trace", model: job.request.model, options: { "atlas-run": job.id } })}>Trace with this atlas</button>
    </Empty>
  ) : job.request.kind === "neuron" ? (
    <Empty title={`Static view of ${job.request.component || "a neuron"}`} text="What this neuron reads and writes, from its weights alone. The engine’s report holds the full tables.">
      {report && <a className="np-btn" href={artifactUrl(job.id, report)} target="_blank" rel="noopener">Open the report</a>}
    </Empty>
  ) : job.status === "completed" && !job.result ? (
    <Empty eye title="Loading the result" text="" />
  ) : (
    <Empty title={job.request.kind === "tokens" ? "Prompt token IDs exported" : "Forward pass exported"} text="The file is listed under Files in the panel." />
  );
  else if (!components.length) stageBody = (
    <Empty title={`The model answers “${(result.answer || "").trim().slice(0, 60)}”`} text="This run generated the answer and measured nothing yet. Choose an answer token above to see where its prediction comes from.">
      <button className="np-btn np-btn--primary" onClick={() => follow({ kind: "decompose", answer: 1 })}>Decompose the first token</button>
    </Empty>
  );
  else if (view === "table") stageBody = <Table rows={tableRows} selected={ui.sel} onSelect={ws.select} brush={brush} onBrush={mark} other={other} />;
  else if (view === "space") stageBody = space && space.points.length ? (
    <Space scene={space} lit={lit} viewKey={`${mode}-${job.id}`} onPick={(id) => ws.select(id && id !== ui.sel ? id : "")} />
  ) : (
    <Empty title="Nothing to place in space" text="This run was made before the engine stored what this view draws, or every component type is hidden. Run the analysis again, or choose the flow view.">
      <button className="np-btn" onClick={() => ws.setView("flow")}>Open the flow view</button>
    </Empty>
  );
  else stageBody = (
    <Canvas view={view} parts={components} metric={metric} layers={result.model.layers} hidden={hidden}
      output={result.output} target={tokenText(result.target?.text || "")} mode={mode} lit={lit}
      onPick={(id) => ws.select(id && id !== ui.sel ? id : "")} onBrush={mark} />
  );
  const note = neuron ? SPACE_NOTES.neuron
    : view === "space" ? SPACE_NOTES[mode] || SPACE_NOTES.decompose
      : view === "flow" && chain && CHAIN_NOTES[mode] ? CHAIN_NOTES[mode] : NOTES[view];
  const measured = !!result && components.length > 0;
  const showOverlays = (measured || !!neuron) && !busy(job);
  const selValue = part ? valueOf(part, metric) : null;

  return (
    <div className="np-app" style={{ ["--accent" as string]: accent }}>
      <Sidebar runs={runs} current={ui.run} view={view} onView={ws.setView} online={ws.online} searchRef={runSearch}
        onOpen={(id) => { ws.open(id); setDraft(null); }} onNew={() => setDraft({ kind: "generate" })} onDelete={(ids) => void ws.remove(ids)}>
        {part ? (
          <>
            <span className="np-stat">{fmt(selValue)}</span>
            <span>{part.label}</span>
            <span className="ap-quiet">{metric === "direct" ? "Direct contribution" : "Total effect"}</span>
            <button className="ap-clear" onClick={() => ws.select("")}>Clear</button>
          </>
        ) : brushInfo ? (
          <>
            <span className="np-stat">{int(brushInfo.count)}</span>
            <span>parts marked</span>
            <span className="ap-quiet">Together {fmt(brushInfo.sum)}</span>
            <button className="ap-clear" onClick={() => setBrush(null)}>Clear</button>
          </>
        ) : result && components.length ? (
          <>
            <span className="np-stat">{int(index.size)}</span>
            <span>parts measured</span>
            <span className="ap-quiet">{int(drawnUnits.length)} drawn as components</span>
          </>
        ) : null}
      </Sidebar>

      <section className="np-panel" style={{ ["--accent" as string]: accent }}>
        {ws.error && (
          <div className="np-error ap-error" role="alert">
            <span>{ws.online ? "The request failed. " : "Start the apollon server to connect. "}{ws.error.length > 220 ? ws.error.slice(0, 220) + "…" : ws.error}</span>
            <button className="np-link" onClick={() => { ws.setError(""); void ws.refresh(); }}>Retry</button>
          </div>
        )}
        {draft ? (
          <Form key={JSON.stringify(draft)} draft={draft} models={ws.models} runs={runs} online={ws.online} onClose={() => setDraft(null)}
            onLaunch={async (request) => { const ok = await ws.launch(request); if (ok) setDraft(null); return ok; }} />
        ) : job ? (
          <Details job={job} result={result} neuron={neuron} runs={runs} part={part} parent={parentPart} metric={metric} artifacts={artifacts}
            ranked={ranked} rankNote={rankNote} rankTitle={rankTitle} onSelect={ws.select} onFollow={follow} onNew={() => setDraft({ kind: "generate" })}
            compare={compare} onCompare={setCompare} other={other} brush={brushInfo} onClearBrush={() => setBrush(null)}
            pins={pinViews} pinned={pinned} onPin={togglePin} onUnpin={unpin} onLightPins={() => mark(pinViews.flatMap((p) => (p.id ? [p.id] : [])))} />
        ) : (
          <>
            <header className="ap-panel-head"><h1 className="np-title">A closer look at a model</h1></header>
            <p className="np-intro">apollon splits one forward pass into its parts, exactly and without training anything.</p>
            <p className="np-note">GGUF · safetensors · Ollama and Hugging Face caches.</p>
          </>
        )}
      </section>

      <main className="np-stage ap-stage">
        {result && !busy(job) && (
          <PromptStrip result={result} answer={job!.request.answer} analysed={job!.request.kind !== "generate"}
            weights={part?.attention} weightsOf={part?.attention ? `${parentPart?.label || ""} ${part.label}`.trim() : undefined}
            onDecompose={(answer) => follow({ kind: "decompose", answer, component: null })} />
        )}
        <div className="ap-stage-body">
          {stageBody}
          {showOverlays && (
            <>
              {measured && <div className="np-overlay np-overlay--tl">
                <div className="ap-findrow">
                <label className="np-search ap-find">
                  <Search size={14} />
                  <input placeholder="Find a component" value={query} onChange={(e) => setQuery(e.target.value)} aria-label="Find a component" />
                  {query && <button onClick={() => setQuery("")} aria-label="Clear search"><X size={13} /></button>}
                </label>
                <button className={`np-btn ap-filterbtn ${showFilters ? "is-active" : ""}`} onClick={() => setShowFilters((s) => !s)} aria-expanded={showFilters}>
                  {filtering ? `Filter · ${int(shownCount)} of ${int(candidates.length)}` : "Filter"}
                </button>
                </div>
                {showFilters && <Filters layers={layers} withLayers={wide} filter={{ ...filter, hi: Math.min(filter.hi, layers - 1) }} onFilter={setFilter} max={filterMax}
                  shown={shownCount} total={candidates.length} noun={wide ? "components" : "units"} />}
              </div>}
              <div className="np-overlay np-overlay--tr">
                {measured && <div className="np-seg ap-metric">
                  <button className={metric === "direct" ? "is-active" : ""} disabled={!available.direct} title={available.direct ? "Direct contribution" : "This analysis measures no direct contribution"} onClick={() => ws.setMetric("direct")}>Direct</button>
                  <button className={metric === "total" ? "is-active" : ""} disabled={!available.total} title={available.total ? "Total effect" : "This analysis measures no total effect"} onClick={() => ws.setMetric("total")}>Total</button>
                </div>}
                {measured && <button className="np-iconbtn" title="Export result" onClick={exportResult}><Download /></button>}
                <button className={`np-iconbtn ${reading ? "is-active" : ""}`} title="Reading this view" onClick={() => setReading((r) => !r)}><Info /></button>
              </div>
              {view !== "table" && (
                <div className="np-overlay np-overlay--note np-note ap-note"><b>{note.title}</b>{note.text}</div>
              )}
              <div className="np-overlay np-overlay--bl ap-legend-wrap">
                {space && (neuron || view === "space") && <span className="ap-drawn">{space.drawn}</span>}
                {measured && <span className="np-caption">By component type · {int(drawnUnits.length)} components, each point one unit</span>}
                {measured && <span className="np-caption">Target {tokenText(result!.target?.text || "")} · {metric === "direct" ? "direct" : "total"} Δ log-odds</span>}
                {measured && <div className="np-legend">
                  {familyStats.map((f) => (
                    <button key={f.id} className={`np-legend-row ${hidden.has(f.id) ? "is-off" : ""}`}
                      onClick={() => setHidden((old) => { const n = new Set(old); if (n.has(f.id)) n.delete(f.id); else n.add(f.id); return n; })}
                      title={hidden.has(f.id) ? "Show" : "Hide"}>
                      <span className="caret">▸</span>
                      <span className="np-dot" style={{ color: f.color }} />
                      <span>{f.label} · {f.count}</span>
                      <span className="count">{fmt(f.sum)}</span>
                    </button>
                  ))}
                </div>}
              </div>
              {reading && (
                <aside className="np-overlay np-overlay--br np-reading">
                  <header><span className="np-caption">Reading this view</span><button className="np-link" onClick={() => setReading(false)}>Hide</button></header>
                  <dl>{note.reading.map(([k, v]) => <React.Fragment key={k}><dt>{k}</dt><dd>{v}</dd></React.Fragment>)}</dl>
                </aside>
              )}
            </>
          )}
        </div>
      </main>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(<React.StrictMode><App /></React.StrictMode>);
