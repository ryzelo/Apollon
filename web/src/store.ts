// Workspace state: the runs on the server, the one being looked at, and what is selected in it.
// Run, view, metric and selection live in the URL hash, so a finding can be linked.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Job, ModelInfo, Request, api, busy, failed } from "./api";
import type { Metric, View } from "./scene";

export type StageView = "space" | View | "table";
type Ui = { run: string; view: StageView; metric: Metric; sel: string };

const VIEW_IDS: StageView[] = ["space", "flow", "units", "table"];
function readHash(): Partial<Ui> {
  const q = new URLSearchParams(location.hash.slice(1));
  const view = q.get("view") as StageView | null, metric = q.get("metric");
  return {
    ...(q.get("run") ? { run: q.get("run")! } : {}),
    ...(view && VIEW_IDS.includes(view) ? { view } : {}),
    ...(metric === "direct" || metric === "total" ? { metric } : {}),
    ...(q.get("sel") ? { sel: q.get("sel")! } : {}),
  };
}
function writeHash(ui: Ui) {
  const q = new URLSearchParams();
  if (ui.run) q.set("run", ui.run);
  q.set("view", ui.view);
  q.set("metric", ui.metric);
  if (ui.sel) q.set("sel", ui.sel);
  const next = `#${q}`;
  if (location.hash !== next) history.replaceState(null, "", next);
}

/** The run to open first: the newest one that measured something, else the newest that completed. */
function firstRun(runs: Job[]): string {
  const done = runs.filter((r) => r.status === "completed");
  return (done.find((r) => ["decompose", "probe", "trace", "circuit"].includes(r.request.kind)) || done.find((r) => r.request.kind === "generate") || done[0] || runs.find(busy))?.id || "";
}

export function useWorkspace() {
  const initial = useRef(readHash()).current;
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [runs, setRuns] = useState<Job[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [online, setOnline] = useState(false);
  const [error, setError] = useState("");
  const [ui, setUi] = useState<Ui>({ run: initial.run || "", view: initial.view || "space", metric: initial.metric || "direct", sel: initial.sel || "" });
  // the run whose view and metric were already chosen, so opening it again keeps the user's choice
  const settled = useRef(initial.view || initial.metric ? initial.run || "" : "");

  const job = useMemo(() => runs.find((r) => r.id === ui.run), [runs, ui.run]);
  useEffect(() => writeHash(ui), [ui]);
  useEffect(() => {
    const onHash = () => {
      const next = readHash();
      // a link that names a view or metric decides them, also for a run not opened before
      if (next.run && (next.view || next.metric)) settled.current = next.run;
      setUi((old) => ({ ...old, sel: "", ...next }));
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const loadModels = useCallback(async () => {
    try { setModels((await api<{ models: ModelInfo[] }>("/api/models")).models); }
    catch (e) { setError((e as Error).message); }
  }, []);
  const refresh = useCallback(async () => {
    try {
      const r = await api<{ runs: Job[] }>("/api/runs");
      // the list carries no results; keep the ones already fetched
      setRuns((old) => r.runs.map((j) => ({ ...j, result: old.find((o) => o.id === j.id)?.result, progress: old.find((o) => o.id === j.id)?.progress })));
      setOnline(true);
      setLoaded(true);
    } catch (e) {
      setOnline(false);
      setError((e as Error).message);
    }
  }, []);
  useEffect(() => {
    void loadModels();
    void refresh();
    const timer = setInterval(() => void refresh(), 3000);
    return () => clearInterval(timer);
  }, [loadModels, refresh]);

  // open a first run once the list is known, or when the linked run no longer exists
  useEffect(() => {
    if (!loaded) return;
    if (!ui.run || !runs.some((r) => r.id === ui.run)) {
      const next = firstRun(runs);
      if (next !== ui.run) setUi((old) => ({ ...old, run: next, sel: "" }));
    }
  }, [loaded, runs, ui.run]);

  // the open run's result, polled while it is in progress
  useEffect(() => {
    // wait for the run list: a detail fetched before it arrives would be dropped
    if (!ui.run || !loaded) return;
    let active = true;
    let timer = 0;
    const load = async () => {
      try {
        const detail = await api<Job>(`/api/runs/${ui.run}`);
        if (!active) return;
        setRuns((old) => old.map((j) => (j.id === ui.run ? detail : j)));
        if (busy(detail)) timer = window.setTimeout(() => void load(), 1500);
      } catch (e) {
        if (active) setError((e as Error).message);
      }
    };
    void load();
    return () => { active = false; clearTimeout(timer); };
  }, [ui.run, loaded]);

  // a freshly opened run starts in the view and metric that suit its analysis
  useEffect(() => {
    if (!job || busy(job) || settled.current === job.id) return;
    if (job.status === "completed" && !job.result) return;
    settled.current = job.id;
    const k = job.request.kind;
    setUi((old) => ({ ...old, metric: k === "probe" || k === "circuit" ? "total" : "direct" }));
  }, [job]);

  const open = useCallback((id: string) => setUi((old) => (old.run === id ? old : { ...old, run: id, sel: "" })), []);
  const select = useCallback((sel: string) => setUi((old) => ({ ...old, sel })), []);
  const setView = useCallback((view: StageView) => setUi((old) => ({ ...old, view })), []);
  const setMetric = useCallback((metric: Metric) => setUi((old) => ({ ...old, metric })), []);

  const launch = useCallback(async (request: Request): Promise<boolean> => {
    setError("");
    try {
      const added = await api<Job>("/api/runs", { method: "POST", body: JSON.stringify(request) });
      setRuns((r) => [added, ...r]);
      setUi((old) => ({ ...old, run: added.id, sel: "" }));
      return true;
    } catch (e) {
      setError((e as Error).message);
      return false;
    }
  }, []);
  const cancel = useCallback(async (id: string) => {
    try { await api(`/api/runs/${id}/cancel`, { method: "POST" }); await refresh(); }
    catch (e) { setError((e as Error).message); }
  }, [refresh]);
  /** Deletes finished runs and their files for good. */
  const remove = useCallback(async (ids: string[]) => {
    for (const id of ids) {
      try { await api(`/api/runs/${id}`, { method: "DELETE" }); }
      catch (e) { setError((e as Error).message); break; }
    }
    await refresh();
  }, [refresh]);

  return { models, runs, job, online, loaded, error, setError, ui, open, select, setView, setMetric, launch, cancel, remove, refresh, loadModels };
}
export type Workspace = ReturnType<typeof useWorkspace>;

/* ---------- investigations: runs as a tree along parent_id ---------- */
export type RunNode = { run: Job; children: RunNode[]; depth: number };
/** Roots are runs started on their own; follow-ups hang below the run they were started from. */
export function investigations(runs: Job[], showFailed: boolean): { roots: RunNode[]; hidden: Job[] } {
  const byId = new Map(runs.map((r) => [r.id, r]));
  const kids = new Map<string, Job[]>();
  const roots: Job[] = [];
  for (const r of runs) {
    const p = r.request.parent_id;
    // a run counts as a follow-up only when it continues its parent's model and prompt
    const parent = p && p !== r.id ? byId.get(p) : undefined;
    const follows = parent && parent.request.model === r.request.model && (parent.request.prompt === r.request.prompt || !r.request.prompt);
    if (follows) kids.set(p!, [...(kids.get(p!) || []), r]); else roots.push(r);
  }
  const hidden: Job[] = [];
  const build = (run: Job, depth: number, seen: Set<string>): RunNode | null => {
    if (seen.has(run.id)) return null;
    seen.add(run.id);
    const children = (kids.get(run.id) || []).slice().reverse().map((c) => build(c, depth + 1, seen)).filter((x): x is RunNode => !!x);
    if (failed(run) && !showFailed && !children.length) { hidden.push(run); return null; }
    return { run, children, depth };
  };
  const seen = new Set<string>();
  return { roots: roots.map((r) => build(r, 0, seen)).filter((x): x is RunNode => !!x), hidden };
}
