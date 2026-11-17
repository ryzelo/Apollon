// The form that starts an investigation, or a follow-up whose parameters need editing.
import React, { useState } from "react";
import { X } from "lucide-react";
import { Job, Kind, ModelInfo, Request } from "../api";

const FORM_KINDS: { id: Kind; label: string; text: string }[] = [
  { id: "generate", label: "Answer", text: "Generate the answer first, then choose one of its tokens to decompose." },
  { id: "decompose", label: "Decompose", text: "Split the prediction of one answer token into the contributions of every layer." },
  { id: "probe", label: "Probe", text: "Switch the units of one component off one by one and rerun the model." },
  { id: "trace", label: "Trace", text: "Follow one neuron’s activation back to the sources that feed it." },
  { id: "circuit", label: "Circuit", text: "Search the smallest set of computation steps that still keeps the prediction, from whole blocks down to single multiplications. Every accepted step is checked by rerunning the model. This can take minutes." },
  { id: "neuron", label: "Static", text: "What one neuron reads and writes, from its weights alone. No prompt needed." },
  { id: "atlas", label: "Atlas", text: "Store every neuron’s strongest weight connections in one file, so a trace can rank its sources against them. No prompt needed. Gemma 3 1B: about 11 s per layer and 1.1 GB with 64 entries per neuron." },
];
const OPTIONS: Partial<Record<Kind, [string, string, string][]>> = {
  decompose: [["top", "Sources per component", "8"]],
  probe: [["probe-pos", "Intervention position", "Automatic"], ["probe-max", "Maximum distributed units", "1024"], ["depth", "Units to examine in detail", "6"], ["contrast", "Contrast prompt", "For example: What is 2+3?"]],
  trace: [["trace-depth", "Trace depth", "2"], ["trace-top", "Sources per node", "3"], ["scalars", "Scalar products shown", "10"], ["scalars-of", "Scalar products for source", "S14 head 1 @16"]],
  atlas: [["atlas-k", "Entries per neuron", "64"], ["atlas-layers", "Layers, from..to (end excluded)", "All, for example 0..8"]],
  circuit: [["circuit-kl", "Allowed change of the output (KL, nats)", "0.1"], ["circuit-depth", "Finest level, 1 = blocks … 4 = single multiplications", "4"], ["circuit-tests", "Test budget", "2000000"]],
};
/** Choices with a fixed list of values: [option, label, [value, text][], note]. The empty value is the engine's default. */
const CHOICES: Partial<Record<Kind, [string, string, [string, string][], string][]>> = {
  circuit: [
    ["circuit-ablation", "Removed steps are replaced by", [["", "Their mean (default)"], ["zero", "Zero"]], "The mean is taken over the reference prompts below, or over this prompt."],
    ["circuit-norms", "Normalization during interventions", [["", "Frozen at the full model (default)"], ["live", "Recomputed from what is left"]], "Recomputing lets a small rest be scaled up to full size."],
  ],
};

export type Draft = Partial<Request> & { kind: Kind };

export function Form({ draft, models, runs, online, onClose, onLaunch }: {
  draft: Draft; models: ModelInfo[]; runs: Job[]; online: boolean; onClose: () => void;
  onLaunch: (request: Request) => Promise<boolean>;
}) {
  const [model, setModel] = useState(draft.model || runs.find((r) => r.status === "completed")?.request.model || models.find((m) => m.supported !== false)?.id || "");
  const [prompt, setPrompt] = useState(draft.prompt || "");
  const [kind, setKind] = useState<Kind>(draft.kind);
  const [answer, setAnswer] = useState(draft.answer || 1);
  const [chat, setChat] = useState(!!draft.chat);
  const [component, setComponent] = useState(draft.component || "");
  const [target, setTarget] = useState(draft.target || "");
  const [options, setOptions] = useState<Record<string, string>>(draft.options || {});
  const [submitting, setSubmitting] = useState(false);
  const staticMode = kind === "neuron" || kind === "atlas";
  const known = models.find((m) => m.id === model);
  const send = async (request: Request) => {
    setSubmitting(true);
    await onLaunch(request);
    setSubmitting(false);
  };
  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    void send({
      model, options, prompt: staticMode ? "" : prompt, kind, answer, chat,
      ...(draft.parent_id ? { parent_id: draft.parent_id } : {}),
      ...(component && kind !== "generate" && kind !== "decompose" ? { component } : {}),
      ...(target && !staticMode ? { target } : {}),
    });
  };
  const needsPart = kind === "probe" || kind === "trace" || kind === "neuron";
  return (
    <form className="ap-form" onSubmit={submit}>
      <header className="ap-panel-head">
        <h1 className="np-title" style={{ ["--accent" as string]: "#f2f2f2" }}>{draft.parent_id ? "Follow-up analysis" : "New investigation"}</h1>
        <button type="button" className="np-iconbtn" onClick={onClose} aria-label="Close"><X /></button>
      </header>
      <p className="np-intro">Choose a local model and a prompt. The engine computes the real forward pass and splits it into its parts.</p>
      <div className="np-section">
        <label className="ap-label">Model
          <input className="np-input" autoFocus required list="local-models" value={model} onChange={(e) => setModel(e.target.value)} placeholder="Model ID or local model path" />
          <datalist id="local-models">{models.map((m) => <option key={m.id} value={m.id}>{m.supported === false ? `Not supported · ${m.architecture}` : m.architecture || m.name}</option>)}</datalist>
        </label>
        {!!models.length && !models.some((m) => m.supported !== false) && <p className="np-note">No supported local models were found. Enter the path of a GGUF file or a model folder.</p>}
        {known?.supported === false && <p className="np-error">apollon cannot run this model: {known.architecture}.</p>}
        {!staticMode && (
          <label className="ap-label">Prompt
            <textarea className="np-textarea" required rows={3} placeholder="What is 2+2?" value={prompt} onChange={(e) => setPrompt(e.target.value)} />
          </label>
        )}
        {!staticMode && (
          <label className="np-field">Use the model’s chat template
            <input className="np-switch" type="checkbox" checked={chat} onChange={(e) => setChat(e.target.checked)} />
          </label>
        )}
      </div>
      <div className="np-section">
        <span className="np-caption">Start with</span>
        <div className="np-seg np-seg--grid ap-kinds">
          {FORM_KINDS.map((k) => (
            <button type="button" key={k.id} className={kind === k.id ? "is-active" : ""} onClick={() => setKind(k.id)}>{k.label}</button>
          ))}
        </div>
        <p className="np-note">{FORM_KINDS.find((k) => k.id === kind)?.text}</p>
        {!staticMode && kind !== "generate" && (
          <label className="np-field">Answer token
            <input className="np-input ap-num" type="number" min={1} max={256} value={answer} onChange={(e) => setAnswer(Number(e.target.value))} required />
          </label>
        )}
        {needsPart && (
          <label className="ap-label">{kind !== "probe" ? "Neuron" : "Component"}
            <input className="np-input" required value={component} onChange={(e) => setComponent(e.target.value)} placeholder={kind !== "probe" ? "S12 #2328" : "S12 MLP"} />
            <span className="np-note">{kind === "probe" ? "Layer and component, for example S12 MLP or S12 Attn." : kind === "trace" ? "Layer and neuron index, for example S12 #2328, or auto for the strongest neurons." : "Layer and neuron index, for example S12 #2328."}</span>
          </label>
        )}
        {!staticMode && kind !== "generate" && (
          <label className="ap-label">Target token <span className="ap-quiet">optional</span>
            <input className="np-input" value={target} onChange={(e) => setTarget(e.target.value)} placeholder="Use the model’s predicted token" />
          </label>
        )}
      </div>
      <details className="np-disclosure np-section">
        <summary className="np-caption">Advanced options</summary>
        <div className="ap-options">
          {[...(!staticMode ? [["pos", "Prompt position", "Optional, negative counts from the end"]] : []), ["threads", "Worker threads", "Automatic"], ...(OPTIONS[kind] || [])].map(([key, label, hint]) => (
            <label key={key} className="ap-label">{label}
              <input className="np-input" value={options[key] || ""} placeholder={hint} onChange={(e) => setOptions((old) => ({ ...old, [key]: e.target.value }))} />
            </label>
          ))}
          {kind === "trace" && (
            <label className="ap-label">Neuron atlas
              <select className="np-select" value={options["atlas-run"] || ""} onChange={(e) => setOptions((old) => ({ ...old, "atlas-run": e.target.value }))}>
                <option value="">No atlas</option>
                {runs.filter((r) => r.status === "completed" && r.request.kind === "atlas" && r.request.model === model).map((r) => <option key={r.id} value={r.id}>Atlas {r.id.slice(-8)}</option>)}
              </select>
              <span className="np-note">An atlas stores each neuron’s strongest weight connections. <button type="button" className="np-link" onClick={() => setKind("atlas")}>Build one for this model</button></span>
            </label>
          )}
          {(CHOICES[kind] || []).map(([key, label, values, note]) => (
            <label key={key} className="ap-label">{label}
              <select className="np-select" value={options[key] || ""} onChange={(e) => setOptions((old) => ({ ...old, [key]: e.target.value }))}>
                {values.map(([v, text]) => <option key={v} value={v}>{text}</option>)}
              </select>
              <span className="np-note">{note}</span>
            </label>
          ))}
          {kind === "circuit" && options["circuit-ablation"] !== "zero" && (
            <label className="ap-label">Reference prompts <span className="ap-quiet">optional</span>
              <textarea className="np-textarea" rows={2} value={options["circuit-ref"] || ""} placeholder="3+5=; 7+1=; 4+4=" onChange={(e) => setOptions((old) => ({ ...old, "circuit-ref": e.target.value }))} />
              <span className="np-note">Separate several with “;”. Prompts in the same format, for example other additions, give the means removed steps are replaced by.</span>
            </label>
          )}
          {kind !== "atlas" && <label className="ap-label">Matrix products on
              <select className="np-select" value={options.gpu || ""} onChange={(e) => setOptions((old) => ({ ...old, gpu: e.target.value }))}>
                <option value="">Automatic (default)</option>
                <option value="metal">GPU · Metal (Apple Silicon)</option>
                <option value="cuda">GPU · CUDA (NVIDIA)</option>
                <option value="off">CPU only</option>
              </select>
              <span className="np-note">Automatic uses Metal on a Mac, CUDA when an NVIDIA driver is found, otherwise the CPU.</span>
            </label>}
          {kind === "decompose" && (
            <label className="np-field">Intervene only at the focus position
              <input className="np-switch" type="checkbox" checked={options["focus-only"] === "true"} onChange={(e) => setOptions((old) => ({ ...old, "focus-only": String(e.target.checked) }))} />
            </label>
          )}
        </div>
      </details>
      <div className="ap-actions">
        <button type="button" className="np-btn" onClick={onClose}>Cancel</button>
        <button className="np-btn np-btn--primary" disabled={submitting || !online || !model.trim()}>{submitting ? "Starting…" : kind === "generate" ? "Generate the answer" : kind === "atlas" ? "Build the atlas" : "Run analysis"}</button>
      </div>
    </form>
  );
}
