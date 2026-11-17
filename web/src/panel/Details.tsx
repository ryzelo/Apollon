// The panel for one run: what was asked, the selected part with its interventions, and the run's checks and files.
import React, { useEffect, useState } from "react";
import { CircuitInfo, CircuitOut, Detail, Job, NeuronStatic, Prob, Request, Result, api, artifactUrl, busy, descendants, fmt, int, neuronSpec, pct, probeSpec, runLabel, shortModel, tokenText } from "../api";
import { Metric, Part, familyColor, familyOf, valueOf } from "../scene";
import { NeuronHeads, ProbeMeasures, TraceMeasures } from "./Measures";

export function Lane({ label, value, max, color, active, before, onClick, mono, text }: {
  label: string; value: number | null; max: number; color: string; active?: boolean; before?: number | null; onClick?: () => void; mono?: boolean;
  /** the number as shown, when it is not a signed log-odds value */
  text?: string;
}) {
  const w = value == null ? 0 : (Math.abs(value) / max) * 50;
  return (
    <button className={`ap-lane ${active ? "is-active" : ""}`} onClick={onClick} disabled={!onClick}>
      <span className={`ap-lane-label ${mono ? "ap-monolabel" : ""}`}>{label}</span>
      <i className="ap-lane-track">
        <b style={{ width: `${w}%`, left: value != null && value < 0 ? `${50 - w}%` : "50%", background: color, opacity: value != null && value < 0 ? 0.55 : 1 }} />
      </i>
      {before !== undefined && <span className="ap-quiet np-num">{fmt(before)} →</span>}
      <span className="np-num">{text ?? fmt(value)}</span>
    </button>
  );
}

export function Probs({ items, n = 5, mark }: { items: Prob[]; n?: number; mark?: number }) {
  return (
    <>
      {items.slice(0, n).map((t) => (
        <div className={`ap-prob ${t.id === mark ? "is-target" : ""}`} key={t.id}>
          <span>{tokenText(t.text)}</span>
          <i><b style={{ width: `${Math.max(0.5, t.probability * 100)}%` }} /></i>
          <span className="np-num">{pct(t.probability)}</span>
        </div>
      ))}
    </>
  );
}

/** The real output next to the output of the intervention, so a changed answer is seen at once. */
function BeforeAfter({ real, without, label, target, base = "Real pass" }: { real: Prob[]; without: Prob[]; label: string; target: number; base?: string }) {
  const flips = real[0] && without[0] && real[0].id !== without[0].id;
  return (
    <div className="ap-voice">
      <span className="np-caption">Output · {base.toLowerCase()} and without {label}</span>
      <div className="ap-beforeafter">
        <div><span className="ap-quiet">{base}</span><Probs items={real} n={3} mark={target} /></div>
        <div><span className="ap-quiet">Without it</span><Probs items={without} n={3} mark={target} /></div>
      </div>
      <p className="np-note">
        {flips
          ? <>Without it the model answers <b className="ap-inline">{tokenText(without[0].text)}</b> instead of <b className="ap-inline">{tokenText(real[0].text)}</b>.</>
          : "The most likely token stays the same without it."}
      </p>
    </div>
  );
}

function Inspector({ part, parent, result, metric, busyRun, onSelect, onFollow, pinned, onPin }: {
  part: Detail; parent?: Part; result: Result; metric: Metric; busyRun: boolean;
  onSelect: (id: string) => void; onFollow: (extra: Partial<Request>) => void; pinned: boolean; onPin: () => void;
}) {
  const [contrast, setContrast] = useState("");
  const value = valueOf(part, metric);
  const probe = probeSpec(part), neuron = neuronSpec(part, parent);
  const without = part.output_without || part.all_off?.top;
  const tokens = result.tokens;
  const looks = part.attention
    ? part.attention.map((w, j) => ({ j, w })).sort((a, b) => b.w - a.w).slice(0, 6).filter((x) => x.w > 0.005)
    : [];
  const kidMax = Math.max(...(part.children || []).map((x) => Math.abs(valueOf(x, metric) ?? 0)), 1e-6);
  const circuit = result.mode === "circuit" ? result.circuit : undefined;
  return (
    <section className="np-section" style={{ ["--accent" as string]: familyColor(familyOf(part)) }}>
      <span className="np-caption">Selected part</span>
      <div className="ap-inspect">
        <div>
          <h2 className="ap-h2">{part.label}</h2>
          <p className="np-note">{part.kind}{part.layer != null && part.layer >= 0 ? ` in layer ${part.layer}` : ""}{part.position != null ? ` at position ${part.position}` : ""}. {circuit
            ? part.children?.length ? "Sum over its leaves of the target’s log-odds in the circuit minus without the leaf, each removed alone." : "Log-odds of the target in the circuit minus in the circuit without this leaf."
            : "Log-odds of the target with it minus without it."}</p>
        </div>
        <span className="np-stat">{fmt(value)}</span>
      </div>
      <div className="ap-actions">
        <button className={`np-btn ${pinned ? "is-active" : ""}`} onClick={onPin} title="Pinned parts stay listed in every run of this session">{pinned ? "Pinned" : "Pin"}</button>
        <span className="ap-keys">← → layer · ↑ ↓ rank · Enter probe · T trace · P pin · Esc clear</span>
      </div>
      <table className="np-table">
        <tbody>
          {part.direct != null && <tr><td>Direct contribution</td><td>{fmt(part.direct)}</td></tr>}
          {part.total != null && <tr><td>{circuit ? "Effect in the circuit, removed and rerun" : "Total effect, removed and rerun"}</td><td>{fmt(part.total)}</td></tr>}
          {part.total != null && part.direct != null && <tr><td>Indirect, through later layers</td><td>{fmt(part.total - part.direct)}</td></tr>}
          {part.indirect_norm != null && <tr><td>Of that, only the final norm rescaling</td><td>{fmt(part.indirect_norm)}</td></tr>}
          {part.direct_share != null && <tr><td>Direct share of its effect</td><td>{(part.direct_share * 100).toFixed(0)}%</td></tr>}
          {part.direct_kl != null && <tr><td>KL of its direct read-out</td><td>{part.direct_kl.toExponential(2)}</td></tr>}
          {part.voice_entropy_bits != null && <tr><td>Entropy of its voice</td><td>{part.voice_entropy_bits.toFixed(2)} bits</td></tr>}
          {part.direct_voice != null && <tr><td>Direct contribution at the focus</td><td>{fmt(part.direct_voice)}</td></tr>}
          {part.activation_contribution != null && <tr><td>Share of the activation</td><td>{fmt(part.activation_contribution)}</td></tr>}
          {part.activation != null && <tr><td>{circuit ? "Value of the product" : "Activation"}</td><td>{fmt(part.activation)}</td></tr>}
          {part.kl != null && <tr><td>{circuit ? "KL to the full model without it" : "KL divergence of the output without it"}</td><td>{part.kl.toExponential(2)}</td></tr>}
          {part.count != null && <tr><td>{circuit ? "Scalar products (atoms) in it" : "Sources in this remainder"}</td><td>{int(part.count)}</td></tr>}
          {circuit && part.path != null && (circuit.paths?.length ?? 0) > 1 && <tr><td>Found in path</td><td>{part.path}</td></tr>}
          {circuit && part.kl_in_path != null && (circuit.paths?.length ?? 0) > 1 && <tr><td>KL without it, within its path alone</td><td>{part.kl_in_path.toExponential(2)}</td></tr>}
        </tbody>
      </table>

      {(probe || neuron) && (
        <div className="ap-voice">
          <span className="np-caption">Intervene</span>
          <div className="ap-actions">
            {probe && <button className="np-btn np-btn--primary" disabled={busyRun} onClick={() => onFollow({ kind: "probe", component: probe })}>Switch its units off one by one</button>}
            {neuron && <button className="np-btn np-btn--primary" disabled={busyRun} onClick={() => onFollow({ kind: "trace", component: neuron })}>Trace why it is active</button>}
            {neuron && <button className="np-btn" disabled={busyRun} onClick={() => onFollow({ kind: "neuron", component: neuron, prompt: "" })}>Static view</button>}
          </div>
          {probe && (
            <form className="ap-contrast" onSubmit={(e) => { e.preventDefault(); if (contrast.trim()) onFollow({ kind: "probe", component: probe, options: { contrast: contrast.trim() } }); }}>
              <input className="np-input" value={contrast} onChange={(e) => setContrast(e.target.value)} placeholder="Contrast prompt, for example: What is 2+3?" aria-label="Contrast prompt" />
              <button className="np-btn" disabled={busyRun || !contrast.trim()}>Insert its values</button>
            </form>
          )}
          <p className="np-note">{probe ? "Each starts a new run below this one. A contrast prompt must have as many tokens as the prompt." : "Each starts a new run below this one."}</p>
        </div>
      )}

      {!!without?.length && (circuit
        ? <BeforeAfter real={probsOf(circuit.circuit)} without={without} label={part.label} target={result.target?.id} base="Circuit" />
        : <BeforeAfter real={result.output} without={without} label={part.label} target={result.target?.id} />)}

      {result.mode === "probe" && <ProbeMeasures part={part} result={result} />}
      {result.mode === "trace" && <TraceMeasures part={part} onSelect={onSelect} />}

      {(!!part.raises?.length || !!part.lowers?.length) && (
        <div className="ap-voice">
          <span className="np-caption">Tokens it moves · probability without → with</span>
          {[["Raises", part.raises || []] as const, ["Lowers", part.lowers || []] as const].map(([name, list]) => !!list.length && (
            <div className="ap-shifts" key={name}>
              <span className="ap-quiet">{name}</span>
              {list.slice(0, 4).map((s) => <span key={s.id} className="ap-shift"><b>{tokenText(s.text)}</b> {pct(s.without)} → {pct(s.with)}</span>)}
            </div>
          ))}
        </div>
      )}

      {!!looks.length && (
        <div className="ap-lanes">
          <span className="np-caption">Where it looks from the focus</span>
          {looks.map(({ j, w }) => <Lane key={j} mono label={`[${j}] ${tokenText(tokens[j]?.text ?? "")}`} value={w} max={1} text={pct(w)} color={familyColor("attention")} />)}
          <p className="np-note">Attention weights over the {int(part.attention!.length)} positions. The prompt above shows all of them.</p>
        </div>
      )}

      {!!part.dose?.length && (
        <div className="ap-voice">
          <span className="np-caption">Dose · activation times a factor</span>
          <table className="np-table ap-dose">
            <thead><tr><th>Factor</th><th>Top token</th>{result.rival && <th>Δ {tokenText(result.rival.text)}</th>}<th>Δ log-odds</th></tr></thead>
            <tbody>
              {part.dose.map((d) => (
                <tr key={d.factor} className={d.factor === 1 ? "ap-quiet" : ""}>
                  <td>×{d.factor}</td>
                  <td>{d.top[0] ? `${tokenText(d.top[0].text)} ${pct(d.top[0].probability)}` : "—"}</td>
                  {result.rival && <td>{fmt(d.change_rival)}</td>}
                  <td>{fmt(d.change)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="np-note">Change of the target’s log-odds against the real pass. ×0 switches the unit off, ×1 would be the real pass.</p>
        </div>
      )}

      {!!part.contrast_patches?.length && (
        <div className="ap-voice">
          <span className="np-caption">Values from the contrast prompt inserted</span>
          <table className="np-table ap-dose">
            <thead><tr><th>Inserted</th><th>Top token</th>{result.rival && <th>Δ {tokenText(result.rival.text)}</th>}<th>Δ log-odds</th></tr></thead>
            <tbody>
              {part.contrast_patches.map((d, i) => (
                <tr key={i}><td>{d.label}</td><td>{d.top[0] ? `${tokenText(d.top[0].text)} ${pct(d.top[0].probability)}` : "—"}</td>{result.rival && <td>{fmt(d.change_rival)}</td>}<td>{fmt(d.change)}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {!!part.voice?.length && (
        <div className="ap-voice">
          <span className="np-caption">Its voice, read out alone</span>
          <div className="ap-voice-tokens">{part.voice.slice(0, 6).map((v) => <span key={v.id}>{tokenText(v.text)}</span>)}</div>
        </div>
      )}
      {!!part.children?.length && (
        <div className="ap-lanes">
          <span className="np-caption">Its parts · {int(part.children.length)}</span>
          {part.children.slice(0, 12).map((c) => <Lane key={c.id} label={c.label} value={valueOf(c, metric)} max={kidMax} color={familyColor(familyOf(c))} onClick={() => onSelect(c.id)} />)}
          {part.children.length > 12 && <p className="np-note">The first 12 of {int(part.children.length)} are listed. The table view holds all of them.</p>}
        </div>
      )}
    </section>
  );
}

/** Probabilities of a circuit output, without the entries the engine could not compute. */
const probsOf = (o: CircuitOut): Prob[] => o.top.filter((t) => t.probability != null && Number.isFinite(t.probability)).map((t) => ({ ...t, probability: t.probability! }));
const num = (v: number | null | undefined, digits = 3) => (v == null || !Number.isFinite(v) ? "not finite" : v.toFixed(digits));
const big = (n: number) => (n >= 1e9 ? `${(n / 10 ** Math.floor(Math.log10(n))).toFixed(2)}·10^${Math.floor(Math.log10(n))}` : int(n));

/** The result of a circuit search: how much of the model is left, and what it predicts with and without the rest. */
function CircuitSummary({ c, target }: { c: CircuitInfo; target?: number }) {
  const share = c.atoms_kept / Math.max(1, c.atoms_total);
  const union = (c.paths?.length ?? 0) > 1;
  const rows: [string, string, CircuitOut][] = [
    ["Full model", "Every step switched on.", c.full],
    [union ? "Union of all paths only" : "Circuit only", "Only the kept steps; everything else is replaced.", c.circuit],
    [union ? "Everything but the paths" : "Everything but the circuit", "The kept steps replaced, the rest switched on.", c.complement],
  ];
  return (
    <section className="np-section" style={{ ["--accent" as string]: "#ff5fc8" }}>
      <span className="np-caption">Circuit</span>
      <div className="ap-inspect">
        <div>
          <h2 className="ap-h2">{big(c.atoms_kept)} of {big(c.atoms_total)} scalar products kept{union ? ` in ${c.paths!.length} paths` : ""}</h2>
          <p className="np-note">{share >= 0.001 ? `${(share * 100).toFixed(2)}%` : `${(share * 100).toExponential(1)}%`} of the forward pass, in {int(c.leaves)} leaves. Criterion: the same top token and KL to the full model at most {Number(c.criterion_kl.toPrecision(4))} nats.</p>
        </div>
        <span className="np-stat">{int(c.leaves)}</span>
      </div>
      {rows.map(([label, note, o]) => {
        const items = probsOf(o);
        return (
          <div className="ap-voice" key={label}>
            <span className="np-caption">{label}{items.length ? ` · KL ${num(o.kl, 4)} · target log-odds ${num(o.target_log_odds, 2)}` : ""}</span>
            {items.length ? <Probs items={items} n={3} mark={target} /> : <p className="np-error">The engine produced no finite output here.</p>}
            <p className="np-note">{note}</p>
          </div>
        );
      })}
      <table className="np-table">
        <tbody>
          <tr><td>Removed steps replaced by</td><td>{c.ablation === "mean" ? `mean over ${int(c.reference_positions)} positions in ${int(c.reference_sequences)} prompt${c.reference_sequences === 1 ? "" : "s"}` : "zero"}</td></tr>
          <tr><td>Normalization</td><td>{c.norms === "frozen" ? "frozen at the full model" : "recomputed"}</td></tr>
          <tr><td>Finest level</td><td>{c.depth}</td></tr>
          <tr><td>Tests · model passes</td><td>{int(c.tests)} · {int(c.passes)}</td></tr>
          <tr><td>Accepted · refused by the rerun</td><td>{int(c.commits)} · {int(c.refused)}</td></tr>
          <tr><td>Time</td><td>{Math.round(c.seconds)} s</td></tr>
        </tbody>
      </table>
      {!!c.paths?.length && <CircuitPaths c={c} target={target} />}
      {c.stopped && <p className="np-error">The test budget ran out. Blocks that were not resolved stay whole in the circuit.</p>}
      <div className="ap-voice">
        <span className="np-caption">Search levels</span>
        <table className="np-table ap-dose">
          <thead><tr><th>Level</th><th>Kept</th><th>Atoms on</th><th>KL</th></tr></thead>
          <tbody>
            {c.levels.map((l) => (
              <tr key={l.level} title={l.name}><td>{l.level <= 4 ? l.level : "final"}</td><td>{int(l.kept)}</td><td>{big(l.atoms)}</td><td>{num(l.kl, 4)}</td></tr>
            ))}
          </tbody>
        </table>
        <p className="np-note">From whole blocks (1) through heads and neurons (2) and dimensions and keys (3) to single multiplications (4). The final step splits remaining blocks and tests every leaf alone.</p>
      </div>
    </section>
  );
}

/** The routes the self-expansion found: each path alone, and the union without it. */
function CircuitPaths({ c, target }: { c: CircuitInfo; target?: number }) {
  const topOf = (o?: CircuitOut | null) => {
    const t = o ? probsOf(o)[0] : undefined;
    return t ? <span className={t.id === target ? "" : "ap-quiet"}>{tokenText(t.text)} {pct(t.probability)}</span> : "—";
  };
  return (
    <div className="ap-voice">
      <span className="np-caption">Routes · {c.paths!.length} path{c.paths!.length === 1 ? "" : "s"}</span>
      <table className="np-table ap-grid">
        <thead><tr><th className="ap-n">Path</th><th className="ap-n">Atoms</th><th className="ap-n">Alone</th><th className="ap-n">Union without it</th></tr></thead>
        <tbody>
          {c.paths!.map((p) => {
            // with one path, the union without it is everything but the circuit
            const without = p.union_without_it ?? (c.paths!.length === 1 ? c.complement : null);
            return (
            <tr key={p.path} title={`Searched against ${p.reference.top[0] ? tokenText(p.reference.top[0].text) : "—"}, KL ${num(p.reference.kl, 3)} to the full model and ${num(p.kl_reference_to_baseline, 3)} to everything off · ${int(p.leaves)} leaves · ${int(p.tests)} tests · ${Math.round(p.seconds)} s`}>
              <td className="ap-n">{p.path}</td>
              <td className="ap-n">{big(p.atoms)}</td>
              <td className="ap-n">{topOf(p.alone_vs_full)}<span className="ap-sub">KL {num(p.alone_vs_full.kl, 3)}</span></td>
              <td className="ap-n">{without ? <>{topOf(without)}<span className="ap-sub">KL {num(without.kl, 3)}</span></> : "—"}</td>
            </tr>
            );
          })}
        </tbody>
      </table>
      <p className="np-note">Path 1 is the minimal circuit of the full model. Each later path is the minimal circuit of what is left with the earlier paths switched off. KL is to the full model; a grey token is not the target. Hover a row for what it was searched against.</p>
      {c.expansion_stop && <p className="np-note">Stopped: {c.expansion_stop}.</p>}
    </div>
  );
}

/** What a neuron's weights say, without a prompt. */
function StaticNeuron({ n }: { n: NeuronStatic }) {
  const top = (list: { id: number; text: string; logit: number }[]) => (
    <div className="ap-voice-tokens">{list.slice(0, 10).map((t) => <span key={t.id} title={`logit per unit activation ${t.logit.toFixed(3)}`}>{tokenText(t.text)}</span>)}</div>
  );
  return (
    <section className="np-section" style={{ ["--accent" as string]: familyColor("mlp") }}>
      <span className="np-caption">Neuron {n.label}</span>
      <table className="np-table">
        <tbody>
          <tr><td>Activation function</td><td>{n.activation_function}{n.gated ? ", gated" : ""}</td></tr>
          <tr><td>Lowest activation it can reach</td><td>{fmt(n.repertoire.activation_min)}</td></tr>
          <tr><td>Highest activation it can reach</td><td>{fmt(n.repertoire.activation_max)}</td></tr>
          {n.inputs.map((g) => <tr key={g.group}><td>Static inputs · {g.group}</td><td>{int(g.count)}</td></tr>)}
          <tr><td>Later neurons it writes into</td><td>{int(n.outputs.neurons.count)}</td></tr>
        </tbody>
      </table>
      {!!n.outputs.raises.length && <div className="ap-voice"><span className="np-caption">Tokens it raises when active</span>{top(n.outputs.raises)}</div>}
      {!!n.outputs.lowers.length && <div className="ap-voice"><span className="np-caption">Tokens it lowers when active</span>{top(n.outputs.lowers)}</div>}
      <NeuronHeads n={n} />
      {!!n.runtime_only_head_layers.length && (
        <p className="np-note">Heads of layers {n.runtime_only_head_layers[0]} to {n.runtime_only_head_layers[n.runtime_only_head_layers.length - 1]} also write into it. What they write depends on the prompt, so they have no static connection; a trace shows them.</p>
      )}
    </section>
  );
}

/** A pinned part as it appears in the open run, and in the run compared with. */
export type PinView = { sig: string; label: string; id?: string; value?: number | null; other?: number | null; color?: string };

export function Details({ job, result, neuron, runs, part, parent, metric, artifacts, ranked, rankNote, rankTitle, onSelect, onFollow, onNew,
  compare, onCompare, other, brush, onClearBrush, pins, pinned, onPin, onUnpin, onLightPins }: {
  job: Job; result?: Result; neuron?: NeuronStatic; runs: Job[]; part?: Part; parent?: Part; metric: Metric; artifacts: string[];
  ranked: Part[]; rankNote: string; rankTitle: string;
  onSelect: (id: string) => void; onFollow: (extra: Partial<Request>) => void; onNew: () => void;
  /** the run compared with, and each part's value there */
  compare: string; onCompare: (id: string) => void; other?: Map<string, number | null>;
  brush?: { count: number; sum: number }; onClearBrush: () => void;
  pins: PinView[]; pinned: boolean; onPin: () => void; onUnpin: (sig: string) => void; onLightPins: () => void;
}) {
  const m: Metric = metric;
  const rankMax = Math.max(...ranked.map((p) => Math.abs(valueOf(p, m) ?? 0)), ...(other ? ranked.map((p) => Math.abs(other.get(p.id) ?? 0)) : []), 1e-6);
  const pinMax = Math.max(...pins.map((p) => Math.max(Math.abs(p.value ?? 0), Math.abs(p.other ?? 0))), 1e-6);
  const checks = (result || job.result)?.checks || [];
  const title = neuron ? "One neuron, from its weights" : result ? "Inside one prediction" : busy(job) ? "Analysis in progress" : job.status === "failed" ? "Analysis failed" : runLabel(job.request);
  const comparable = runs.filter((r) => r.status === "completed" && r.id !== job.id && r.request.kind === job.request.kind);
  return (
    <>
      <header className="ap-panel-head"><h1 className="np-title">{title}</h1></header>
      <div className="ap-meta np-caption">
        <span>{shortModel(result?.model.name || job.request.model)}</span>
        {result?.model.layers != null && <span>{result.model.layers} layers</span>}
        {result?.model.architecture && <span>{result.model.architecture}</span>}
        <span>{runLabel(job.request)}</span>
      </div>
      {result?.target && result.tokens.length > 0 && (
        <p className="np-note">
          The model predicts <b className="ap-inline">{tokenText(result.target.text)}</b> after position {result.focus}
          {result.output[0] ? ` with ${pct(result.output.find((t) => t.id === result.target.id)?.probability ?? result.output[0].probability)}` : ""}.
          {result.rival ? <> Measured against <b className="ap-inline">{tokenText(result.rival.text)}</b>{result.rival.from_contrast_prompt ? ", the prediction for the contrast prompt" : ""}.</> : ""}
          {result.components.length ? " Choose a part on the stage to see what it does and to intervene." : " Choose an answer token above the stage to decompose its prediction."}
        </p>
      )}
      {part && (
        <div className="np-selection">
          <span>Selected {part.label}: {fmt(valueOf(part, metric))} {metric === "direct" ? "direct" : "total"}, {int(descendants(part).size)} parts lit.</span>
          <button onClick={() => onSelect("")}>Clear</button>
        </div>
      )}
      {!part && brush && (
        <div className="np-selection">
          <span>Marked {int(brush.count)} parts: together {fmt(brush.sum)} {metric === "direct" ? "direct" : "total"}, counting each component once. They are lit in every view.</span>
          <button onClick={onClearBrush}>Clear</button>
        </div>
      )}
      {part && result && <Inspector key={part.id} part={part} parent={parent} result={result} metric={metric} busyRun={false} onSelect={onSelect} onFollow={onFollow} pinned={pinned} onPin={onPin} />}

      {!!pins.length && result && (
        <section className="np-section" style={{ ["--accent" as string]: "#ff5fc8" }}>
          <span className="np-caption">Pinned parts · {int(pins.length)}</span>
          <div className="ap-lanes">
            {pins.map((p) => (
              <div className="ap-pin" key={p.sig}>
                {p.id
                  ? <Lane label={p.label} value={p.value ?? null} max={pinMax} color={p.color || "#f2f2f2"} active={p.id === part?.id} before={other ? p.other ?? null : undefined} onClick={() => onSelect(p.id === part?.id ? "" : p.id!)} />
                  : <span className="ap-pin-missing">{p.label} · not measured in this run</span>}
                <button className="ap-pin-x" title="Unpin" aria-label={`Unpin ${p.label}`} onClick={() => onUnpin(p.sig)}>×</button>
              </div>
            ))}
          </div>
          <p className="np-note">Pinned parts are found again in every run by layer, type and unit{other ? "; the left number is the run compared with" : ""}. <button className="np-link" onClick={onLightPins}>Light them on the stage</button></p>
        </section>
      )}

      {neuron && <StaticNeuron n={neuron} />}
      {result?.circuit && <CircuitSummary c={result.circuit} target={result.target?.id} />}
      {!!result?.output?.length && (
        <section className="np-section" style={{ ["--accent" as string]: "#3d9bff" }}>
          <span className="np-caption">Output of the real pass</span>
          <p className="np-note">The five most likely next tokens. The target is marked.</p>
          <Probs items={result.output} mark={result.target?.id} />
        </section>
      )}

      {!!ranked.length && (
        <section className="np-section">
          <span className="np-caption">{other ? `Comparison · ${metric === "direct" ? "direct contribution" : "total effect"}` : rankTitle}</span>
          <p className="np-note">{rankNote}</p>
          <div className="ap-lanes">
            {ranked.slice(0, 8).map((p) => {
              return <Lane key={p.id} label={p.label} value={valueOf(p, m)} max={rankMax} color={familyColor(familyOf(p))} active={p.id === part?.id} before={other ? other.get(p.id) ?? null : undefined} onClick={() => onSelect(p.id === part?.id ? "" : p.id)} />;
            })}
          </div>
          {!!comparable.length && (
            <label className="np-field">Compare with
              <select className="np-select" value={compare} onChange={(e) => onCompare(e.target.value)}>
                <option value="">No other run</option>
                {comparable.map((r) => <option key={r.id} value={r.id}>{r.request.prompt.slice(0, 26)} · {runLabel(r.request)}</option>)}
              </select>
            </label>
          )}
          {other && <p className="np-note">The left number is the other run, matched by layer, type and unit. Each value refers to its own run’s target and context. In the space view the other run is a white point beside each component, and the table gains a column for it.</p>}
        </section>
      )}

      {!!checks.length && (
        <section className="np-section" style={{ ["--accent" as string]: "#2fe0d0" }}>
          <span className="np-caption">Numerical checks</span>
          <p className="np-note">The parts have to add up to the real pass. Every run checks this itself.</p>
          {checks.map((c, i) => (
            <div className={`ap-check ${c.passed === false ? "is-failed" : ""}`} key={i}>
              <span className="np-dot" style={{ color: c.passed === false ? "#ff4d5e" : "#4fe37a" }} />
              <span>{c.label}</span>
              <span className="ap-mono">{c.deviation.toExponential(1)}{c.tolerance != null ? ` / ${c.tolerance.toExponential(0)}` : ""}</span>
            </div>
          ))}
        </section>
      )}

      {job.status === "completed" && (
        <section className="np-section" style={{ ["--accent" as string]: "#f2f2f2" }}>
          <span className="np-caption">Files</span>
          <div className="ap-files">
            {artifacts.map((file) => (
              <a key={file} className="np-link" href={artifactUrl(job.id, file)} target={file.endsWith(".html") ? "_blank" : undefined} rel="noopener">
                {file.endsWith(".html") ? file.replace(/^report_/, "Open report ").replace(/\.html$/, "").replaceAll("_", " ") : file === "atlas.apla" ? "Download neuron atlas" : file === "tokens.json" ? "Download token IDs" : file === "neuron-trace.json" ? "Download trace data" : file === "circuit.json" ? "Download circuit data" : "Download forward pass"}
              </a>
            ))}
            {!!job.request.prompt && !["tokens", "dump"].includes(job.request.kind) && (
              <>
                <button className="np-link" onClick={() => onFollow({ kind: "dump", component: null })}>Export the forward pass</button>
                <button className="np-link" onClick={() => onFollow({ kind: "tokens", component: null })}>Export prompt token IDs</button>
              </>
            )}
          </div>
        </section>
      )}
      <div><button className="np-btn" onClick={onNew}>New investigation</button></div>
    </>
  );
}
