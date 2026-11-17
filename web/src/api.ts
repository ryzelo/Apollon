// Types of the local engine's API and the calls the workspace makes.
import type { Part } from "./scene";

export type Kind = "generate" | "decompose" | "probe" | "trace" | "circuit" | "neuron" | "atlas" | "tokens" | "dump";
export type Token = { id: number; text: string; index: number };
export type Prob = { id: number; text: string; probability: number };
/** A token whose probability a change moves: without the change, and with it. */
export type Shift = { id: number; text: string; without: number; with: number };
/** One intervention's output against the unchanged copy of the same pass. */
export type Outcome = { change: number; change_rival: number; kl: number; top: Prob[]; factor?: number; label?: string };

/** Fields a part can carry beyond what the scenes draw. Which ones exist depends on the analysis. */
export type Detail = Part & {
  activation?: number;
  kl?: number;
  count?: number;
  output_without?: Prob[];
  raises?: Shift[];
  lowers?: Shift[];
  indirect_norm?: number | null;
  direct_share?: number | null;
  attention?: number[];
  dose?: Outcome[];
  direct_voice?: number | null;
  all_off?: Outcome;
  contrast_patches?: Outcome[] | null;
  /** circuit leaf: the path it was found in, and the KL when it is switched off against that path alone */
  path?: number;
  kl_in_path?: number | null;
  /** decomposition: entropy of its voice and KL of the direct read-out */
  voice_entropy_bits?: number;
  direct_kl?: number;
};

/** A probed component: its units are the children; the rest is what the probe measured around them. */
export type LensRead = { target_log_odds: number; top: Prob[] };
export type ProbePath = {
  label: string; unit: number | null; position: number | null; outcome: Outcome;
  layers: {
    layer: number; lens_base: LensRead; lens_variant: LensRead;
    components: { kind: string; direct_base: number; direct_variant: number; voice_base: Prob[]; voice_variant: Prob[] }[];
    neurons: { unit: number; activation_base: number; activation_variant: number; margin_change: number }[];
    heads: { head: number; shift: number; attention_base: number[]; attention_variant: number[] }[];
  }[];
};
export type ProbeComponent = Detail & {
  positions?: { position: number; token: string; off_here: Outcome; only_here: Outcome }[];
  groups?: { k: number; off: Outcome; only: Outcome }[];
  flip?: { k: number; text: string; probability: number } | null;
  post_norm?: boolean;
  downstream?: ProbePath[];
  probed_position?: number | null;
  units_per_position?: number;
};

/** A node of a traced neuron's backward tree. */
export type TreeNode = {
  kind: string; label: string; layer?: number; unit?: number; head?: number; position?: number | null;
  activation?: number; activation_contribution?: number | null; children: TreeNode[];
};
/** A source of a traced neuron, as the engine names it. */
export type Source = { kind: string; label: string; layer?: number; head?: number; unit?: number; position?: number };
export type Scalars = { source: Source; p: number; q: number; rest_p: number; rest_q: number; top_p: { dimension: number; value: number }[]; top_q: { dimension: number; value: number }[] };
export type TraceComponent = Detail & {
  tree?: TreeNode; source_count?: number; silent_neurons?: number;
  p?: number; q?: number; phi?: number; gated?: boolean; activation_range?: [number, number];
  groups?: { group: string; p: number; q: number; activation_contribution: number; count: number }[];
  removals?: {
    source: Source; activation_contribution: number; delta_linear: number; delta_frozen_scale: number; delta_exact: number;
    activation_without: number; total: number | null; kl?: number | null; output_without?: Prob[];
  }[];
  scalars?: Scalars[];
  scalars_asked?: Scalars | null;
  pairs?: { p_source: Source; q_source: Source; value: number }[];
  atlas_rank?: { source: Source; rank: number | null }[];
  read_positions?: { label: string; layer: number; head: number; pos: number; activation_contribution: number }[];
};

/** A static connection of a neuron: a source or target with its two coordinates in the (p, q) plane. */
export type Arrow = { kind: string; label: string; layer?: number; index?: number; head?: number; p: number; q: number };
export type NeuronStatic = {
  label: string; layer: number; unit: number; gated: boolean; activation_function: string;
  repertoire: { activation_min: number; activation_max: number; radius: number; rank: number };
  inputs: { group: string; count: number; by_p: Arrow[]; by_q: Arrow[] }[];
  runtime_only_head_layers: number[];
  outputs: {
    neurons: { count: number; by_p: Arrow[]; by_q: Arrow[] };
    raises: { id: number; text: string; logit: number }[];
    lowers: { id: number; text: string; logit: number }[];
    heads: { layer: number; head: number; query: number; key: number; value: number }[];
  };
};

/** One output of the circuit search: its three most likely tokens, the target's log-odds and the KL to the full model.
 *  Values can be null when the engine produced no finite number. */
export type CircuitOut = { top: { id: number; text: string; probability: number | null }[]; target_log_odds: number | null; kl: number | null };
export type CircuitCell = { layer: number; position: number; attention: number; kv: number; mlp: number; ple: number };
export type CircuitInfo = {
  criterion_kl: number; depth: number;
  ablation: "mean" | "zero"; ablation_text: string; norms: "frozen" | "live"; norms_text: string;
  reference_sequences: number; reference_positions: number;
  atoms_kept: number; atoms_total: number; leaves: number; leaves_listed: number;
  full: CircuitOut; circuit: CircuitOut; complement: CircuitOut;
  levels: { level: number; name: string; groups: number; pieces_in: number; kept: number; unresolved: number; atoms: number; kl: number; tests: number; passes: number; seconds: number }[];
  layers: { layer: number; on: Record<string, number>; total: Record<string, number>; units: string[] }[];
  cells: CircuitCell[];
  tests: number; passes: number; commits: number; refused: number; seconds: number; stopped: boolean;
  /** self-expansion: one entry per route found, and why the search stopped */
  paths?: CircuitPath[];
  expansion_stop?: string;
};
export type CircuitPath = {
  path: number; atoms: number; leaves: number; tests: number; seconds: number;
  /** what it was searched against (the full model, or what the earlier paths leave), read against the full model */
  reference: CircuitOut;
  alone: CircuitOut; alone_vs_full: CircuitOut; union_without_it: CircuitOut | null;
  kl_reference_to_baseline: number;
};

export type Result = {
  schema_version: number;
  mode: string;
  model: { name: string; layers: number; architecture: string };
  tokens: Token[];
  generated?: Token[];
  focus: number;
  target: { id: number; text: string };
  rival?: { id: number; text: string; from_contrast_prompt: boolean };
  output: Prob[];
  components: Part[];
  checks: { label: string; deviation: number; tolerance?: number; passed?: boolean }[];
  answer?: string;
  /** only in a static neuron result */
  neuron?: NeuronStatic;
  /** only in a circuit result */
  circuit?: CircuitInfo;
};
export type Request = {
  model: string; prompt: string; kind: Kind; answer: number; target?: string | null; component?: string | null;
  chat?: boolean; parent_id?: string | null; options?: Record<string, string>;
};
export type Job = { id: string; status: string; request: Request; error?: string | null; result?: Result; progress?: string | null };
export type ModelInfo = { id: string; name: string; supported?: boolean; architecture?: string };

export const busy = (job?: Job) => !!job && ["queued", "running", "pending"].includes(job.status);
export const failed = (job: Job) => ["failed", "interrupted", "cancelled"].includes(job.status);

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, { ...init, headers: { "Content-Type": "application/json", ...init?.headers } });
  if (!response.ok) {
    const body = await response.text();
    let message = body;
    try { message = JSON.parse(body).error || body; } catch { /* plain text */ }
    throw new Error(message || `Request failed (${response.status})`);
  }
  return response.json();
}

export const artifactUrl = (id: string, file: string) => `/api/runs/${id}/artifacts/${encodeURIComponent(file)}`;

/* ---------- formatting ---------- */
export const fmt = (n: number | null | undefined, digits = 3) =>
  n == null ? "—" : `${n > 0 ? "+" : n < 0 ? "−" : ""}${Math.abs(n).toFixed(digits)}`;
export const int = (n: number) => n.toLocaleString("en-US");
export const pct = (p: number) => `${(p * 100).toFixed(p < 0.0995 ? 2 : 1)}%`;
export const tokenText = (t: string) => t.replace(/ /g, "␣").replace(/\n/g, "↵") || "∅";
export const shortModel = (name: string) => name.split(/[\\/]/).pop() || name;

/* ---------- part trees ---------- */
export function flatten(parts: Part[], parent?: Part, out = new Map<string, { part: Part; parent?: Part }>()) {
  for (const p of parts) { out.set(p.id, { part: p, parent }); flatten(p.children || [], p, out); }
  return out;
}
export function descendants(p: Part, out = new Set<string>()) {
  out.add(p.id);
  (p.children || []).forEach((c) => descendants(c, out));
  return out;
}

/** What the engine calls a part on the command line, when it can be probed or traced. */
export function probeSpec(p: Part): string | null {
  if (p.layer == null || p.layer < 0) return null;
  if (p.kind === "attention") return `S${p.layer} Attn`;
  if (p.kind === "mlp") return `S${p.layer} MLP`;
  if (p.kind === "ple") return `S${p.layer} PLE`;
  return null;
}
export const neuronSpec = (p: Part, parent?: Part): string | null => {
  const layer = p.layer ?? parent?.layer;
  return p.kind === "neuron" && p.unit != null && layer != null ? `S${layer} #${p.unit}` : null;
};

/** A short name for a run in the investigation list. */
export function runLabel(r: Request): string {
  switch (r.kind) {
    case "generate": return "Answer";
    case "decompose": return `Decompose token ${r.answer}`;
    case "probe": return `Probe ${r.component || ""}`;
    case "trace": return `Trace ${r.component === "auto" ? "strongest neurons" : r.component || ""}`;
    case "circuit": return `Circuit token ${r.answer}`;
    case "neuron": return `Static ${r.component || "neuron"}`;
    case "atlas": return "Neuron atlas";
    case "tokens": return "Token IDs";
    default: return "Forward pass";
  }
}
