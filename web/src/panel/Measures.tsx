// What a probe or a trace measured around the selected part beyond its own numbers: per-position and group
// ablations, the paths into later layers, removals of sources, scalar products and pairs.
import React, { useState } from "react";
import { NeuronStatic, Outcome, ProbeComponent, ProbePath, Result, TraceComponent, fmt, int, pct, tokenText } from "../api";
import { Part } from "../scene";

const top = (o?: Outcome | null) => (o?.top[0] ? `${tokenText(o.top[0].text)} ${pct(o.top[0].probability)}` : "—");
/** A change of the target's log-odds with the top token beneath it. */
const Out = ({ o }: { o?: Outcome | null }) => <td className="ap-n">{fmt(o?.change)}<span className="ap-sub">{top(o)}</span></td>;
const sci = (v?: number | null) => (v == null || !Number.isFinite(v) ? "—" : v.toExponential(2));

/** A list cut to its first n rows, with a link that shows the rest. */
function useMore<T>(list: T[], n: number): [T[], React.ReactNode] {
  const [all, setAll] = useState(false);
  if (list.length <= n) return [list, null];
  return [all ? list : list.slice(0, n), (
    <p className="np-note">{all ? `All ${int(list.length)} are listed.` : `The first ${int(n)} of ${int(list.length)} are listed.`} <button className="np-link" onClick={() => setAll(!all)}>{all ? "Show fewer" : "Show all"}</button></p>
  )];
}

/** Rival column header, when the run measures against a second token. */
const rivalHead = (r?: Result["rival"]) => (r ? <th className="ap-n" title={`Change of the log-odds of ${tokenText(r.text)}`}>Δ {tokenText(r.text)}</th> : null);
const rivalCell = (r: Result["rival"] | undefined, o?: Outcome | null) => (r ? <td className="ap-n">{fmt(o?.change_rival)}</td> : null);

/* ---------------------------------------------------------------- probe */

export function ProbeMeasures({ part, result }: { part: ProbeComponent; result: Result }) {
  const rival = result.rival;
  const positions = [...(part.positions || [])].sort((a, b) =>
    Math.max(Math.abs(b.off_here.change), Math.abs(b.only_here.change)) - Math.max(Math.abs(a.off_here.change), Math.abs(a.only_here.change)));
  const [shownPositions, morePositions] = useMore(positions, 8);
  return (
    <>
      {(part.probed_position != null || part.units_per_position != null || part.post_norm) && (
        <table className="np-table">
          <tbody>
            {part.probed_position != null && <tr><td>Intervention position</td><td>[{part.probed_position}] {tokenText(result.tokens[part.probed_position]?.text ?? "")}</td></tr>}
            {part.units_per_position != null && <tr><td>Units per position</td><td>{int(part.units_per_position)}</td></tr>}
          </tbody>
        </table>
      )}
      {part.post_norm && <p className="np-note">Its output passes through a norm after it. Switching off some units scales the rest up, so an odd token after a removal can just mean the rest was inflated.</p>}

      {!!positions.length && (
        <div className="ap-voice">
          <span className="np-caption">Where it matters · per position</span>
          <table className="np-table ap-grid">
            <thead><tr><th>Position</th><th className="ap-n">Off here</th><th className="ap-n">Only here</th></tr></thead>
            <tbody>
              {shownPositions.map((p) => (
                <tr key={p.position}>
                  <td className="ap-monolabel">[{p.position}] {tokenText(p.token)}</td>
                  <Out o={p.off_here} /><Out o={p.only_here} />
                </tr>
              ))}
            </tbody>
          </table>
          {morePositions}
          <p className="np-note">Change of the target’s log-odds, with the top token beneath, when the whole component is switched off at this one position, or kept only at this position. Sorted by the larger of the two.</p>
        </div>
      )}

      {!!part.groups?.length && (
        <div className="ap-voice">
          <span className="np-caption">Its strongest units together</span>
          <table className="np-table ap-grid">
            <thead><tr><th className="ap-n">k</th><th className="ap-n">Off</th>{rivalHead(rival)}<th className="ap-n">Only on</th></tr></thead>
            <tbody>
              {part.groups.map((g) => (
                <tr key={g.k} className={part.flip?.k === g.k ? "is-flip" : ""}>
                  <td className="ap-n">{int(g.k)}</td>
                  <Out o={g.off} />{rivalCell(rival, g.off)}<Out o={g.only} />
                </tr>
              ))}
            </tbody>
          </table>
          <p className="np-note">
            The k units with the largest single effect, switched off together, or the only ones left on.{" "}
            {part.flip
              ? <>With the strongest <b className="ap-inline">{int(part.flip.k)}</b> off, the model answers <b className="ap-inline">{tokenText(part.flip.text)}</b> ({pct(part.flip.probability)}).</>
              : "No group of its strongest units changes the answer."}
          </p>
        </div>
      )}

      {!!part.downstream?.length && (
        <div className="ap-voice">
          <span className="np-caption">What changes in later layers · {int(part.downstream.length)} interventions</span>
          {part.downstream.map((d, i) => <Downstream key={i} d={d} rival={rival} />)}
          <p className="np-note">Each intervention is rerun from its layer. Per later layer: the target’s log-odds read out through the final norm and unembedding (logit lens), the component whose direct contribution moves most, and the neuron and head that change most.</p>
        </div>
      )}
    </>
  );
}

function Downstream({ d, rival }: { d: ProbePath; rival?: Result["rival"] }) {
  const rows = d.layers.map((l) => {
    const comp = [...l.components].sort((a, b) => Math.abs(b.direct_variant - b.direct_base) - Math.abs(a.direct_variant - a.direct_base))[0];
    const neuron = [...l.neurons].sort((a, b) => Math.abs(b.margin_change) - Math.abs(a.margin_change))[0];
    const head = [...l.heads].sort((a, b) => Math.abs(b.shift) - Math.abs(a.shift))[0];
    return { l, comp, neuron, head };
  });
  return (
    <details className="np-disclosure">
      <summary><span>{d.label} · Δ {fmt(d.outcome.change)} · {top(d.outcome)}{rival ? ` · Δ ${tokenText(rival.text)} ${fmt(d.outcome.change_rival)}` : ""}</span></summary>
      <table className="np-table ap-grid">
        <thead><tr><th className="ap-n">Layer</th><th className="ap-n">Lens</th><th>Moves most</th><th>Neuron</th><th>Head</th></tr></thead>
        <tbody>
          {rows.map(({ l, comp, neuron, head }) => (
            <tr key={l.layer}>
              <td className="ap-n">{l.layer}</td>
              <td className="ap-n" title={`${l.lens_base.target_log_odds.toFixed(3)} → ${l.lens_variant.target_log_odds.toFixed(3)}`}>{fmt(l.lens_variant.target_log_odds - l.lens_base.target_log_odds)}</td>
              <td>{comp ? `${comp.kind} ${fmt(comp.direct_variant - comp.direct_base)}` : "—"}</td>
              <td title={neuron ? `activation ${neuron.activation_base.toFixed(3)} → ${neuron.activation_variant.toFixed(3)}` : undefined}>{neuron ? `#${neuron.unit} ${fmt(neuron.margin_change)}` : "—"}</td>
              <td>{head ? `h${head.head} ${fmt(head.shift)}` : "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </details>
  );
}

/* ---------------------------------------------------------------- trace */

export function TraceMeasures({ part, onSelect }: { part: TraceComponent; onSelect: (id: string) => void }) {
  const [chosen, setChosen] = useState<number | null>(null);
  const byLabel = new Map((part.children || []).map((c: Part) => [c.label, c.id]));
  const rank = new Map((part.atlas_rank || []).map((a) => [a.source.label, a.rank]));
  const removals = part.removals || [];
  const [pairs, morePairs] = useMore(part.pairs || [], 8);
  const [reads, moreReads] = useMore(part.read_positions || [], 6);
  const groupMax = Math.max(...(part.groups || []).map((g) => Math.abs(g.activation_contribution)), 1e-6);
  const r = chosen != null ? removals[chosen] : undefined;
  const scalars = [...(part.scalars || []), ...(part.scalars_asked ? [part.scalars_asked] : [])];
  if (part.p == null && part.source_count == null && !removals.length) return null;
  return (
    <>
      <table className="np-table">
        <tbody>
          {part.p != null && <tr><td>{part.gated ? "Gate input p" : "Input p"}</td><td>{fmt(part.p)}</td></tr>}
          {part.gated && part.q != null && <tr><td>Up input q</td><td>{fmt(part.q)}</td></tr>}
          {part.phi != null && <tr><td>Activation factor φ(p)</td><td>{fmt(part.phi)}</td></tr>}
          {part.activation_range && <tr><td>Activation range over the repertoire</td><td>{fmt(part.activation_range[0], 1)} … {fmt(part.activation_range[1], 1)}</td></tr>}
          {part.source_count != null && <tr><td>Sources measured</td><td>{int(part.source_count)}</td></tr>}
          {part.silent_neurons != null && <tr><td>Earlier neurons silent here</td><td>{int(part.silent_neurons)}</td></tr>}
        </tbody>
      </table>

      {!!part.groups?.length && (
        <div className="ap-voice">
          <span className="np-caption">Where the activation comes from</span>
          <table className="np-table ap-grid">
            <thead><tr><th>Group</th><th className="ap-n">Sources</th><th className="ap-n">p</th>{part.gated && <th className="ap-n">q</th>}<th className="ap-n">Share</th></tr></thead>
            <tbody>
              {part.groups.map((g) => (
                <tr key={g.group}>
                  <td><i className="ap-bar" style={{ width: `${(Math.abs(g.activation_contribution) / groupMax) * 40}px`, opacity: g.activation_contribution < 0 ? 0.55 : 1 }} />{g.group}</td>
                  <td className="ap-n">{int(g.count)}</td><td className="ap-n">{fmt(g.p)}</td>{part.gated && <td className="ap-n">{fmt(g.q)}</td>}<td className="ap-n">{fmt(g.activation_contribution)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="np-note">The share is each group’s part of the activation; the groups add up to it.</p>
        </div>
      )}

      {!!removals.length && (
        <div className="ap-voice">
          <span className="np-caption">Sources removed and rerun</span>
          <table className="np-table ap-grid">
            <thead><tr><th>Source</th><th className="ap-n">Share</th><th className="ap-n">Δ activation</th><th className="ap-n">Δ log-odds</th>{!!rank.size && <th className="ap-n">Atlas rank</th>}</tr></thead>
            <tbody>
              {removals.map((x, i) => (
                <tr key={i} className={`ap-row ${chosen === i ? "is-active" : ""}`} onClick={() => setChosen(chosen === i ? null : i)}>
                  <td>{x.source.label}</td><td className="ap-n">{fmt(x.activation_contribution)}</td><td className="ap-n">{fmt(x.delta_exact)}</td><td className="ap-n">{fmt(x.total)}</td>
                  {!!rank.size && <td className="ap-n">{rank.get(x.source.label) ?? "not listed"}</td>}
                </tr>
              ))}
            </tbody>
          </table>
          {r ? (
            <div className="ap-voice">
              <table className="np-table">
                <tbody>
                  <tr><td>Activation without it</td><td>{fmt(r.activation_without)}</td></tr>
                  <tr><td>Δ activation, linear estimate</td><td>{fmt(r.delta_linear)}</td></tr>
                  <tr><td>Δ activation, norm scale frozen</td><td>{fmt(r.delta_frozen_scale)}</td></tr>
                  <tr><td>Δ activation, exact rerun</td><td>{fmt(r.delta_exact)}</td></tr>
                  {r.kl != null && <tr><td>KL of the output without it</td><td>{sci(r.kl)}</td></tr>}
                  {!!r.output_without?.length && <tr><td>Top token without it</td><td>{tokenText(r.output_without[0].text)} {pct(r.output_without[0].probability)}</td></tr>}
                </tbody>
              </table>
              {byLabel.has(r.source.label) && <p className="np-note"><button className="np-link" onClick={() => onSelect(byLabel.get(r.source.label)!)}>Select {r.source.label} on the stage</button></p>}
            </div>
          ) : <p className="np-note">Each source is removed alone and the model rerun. Δ activation is how far the neuron’s activation drops; Δ log-odds is what that does to the target. Choose a row for the estimates beside the exact value.</p>}
        </div>
      )}

      {!!reads.length && (
        <div className="ap-voice">
          <span className="np-caption">Positions the heads read</span>
          <table className="np-table ap-grid">
            <tbody>{reads.map((x, i) => <tr key={i}><td className="ap-monolabel">{x.label}</td><td className="ap-n">{fmt(x.activation_contribution)}</td></tr>)}</tbody>
          </table>
          {moreReads}
        </div>
      )}

      {!!scalars.length && (
        <div className="ap-voice">
          <span className="np-caption">Scalar products · strongest dimensions</span>
          {scalars.map((s, i) => (
            <details className="np-disclosure" key={i}>
              <summary><span>{s.source.label}{part.scalars_asked === s ? " · asked for" : ""} · p {fmt(s.p)}{part.gated ? ` · q ${fmt(s.q)}` : ""}</span></summary>
              <table className="np-table ap-grid">
                <thead><tr><th className="ap-n">Dim</th><th className="ap-n">into p</th>{part.gated && <><th className="ap-n">Dim</th><th className="ap-n">into q</th></>}</tr></thead>
                <tbody>
                  {s.top_p.map((d, j) => (
                    <tr key={j}>
                      <td className="ap-n">{d.dimension}</td><td className="ap-n">{fmt(d.value, 4)}</td>
                      {part.gated && <><td className="ap-n">{s.top_q[j]?.dimension ?? ""}</td><td className="ap-n">{s.top_q[j] ? fmt(s.top_q[j].value, 4) : ""}</td></>}
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="np-note">Single terms of the dot product between the source’s vector and the neuron’s input weights. The rest of the dimensions add {fmt(s.rest_p)} to p{part.gated ? ` and ${fmt(s.rest_q)} to q` : ""}.</p>
            </details>
          ))}
        </div>
      )}

      {!!pairs.length && (
        <div className="ap-voice">
          <span className="np-caption">Pairs · source into p times source into q</span>
          <table className="np-table ap-grid">
            <thead><tr><th>Into p</th><th>Into q</th><th className="ap-n">Product</th></tr></thead>
            <tbody>{pairs.map((x, i) => <tr key={i}><td>{x.p_source.label}</td><td>{x.q_source.label}</td><td className="ap-n">{fmt(x.value)}</td></tr>)}</tbody>
          </table>
          {morePairs}
          <p className="np-note">A gated neuron multiplies its two inputs, so its activation splits into products of one source on each side.</p>
        </div>
      )}
    </>
  );
}

/* ---------------------------------------------------------------- static neuron */

export function NeuronHeads({ n }: { n: NeuronStatic }) {
  const heads = [...n.outputs.heads].sort((a, b) => Math.max(b.query, b.key, b.value) - Math.max(a.query, a.key, a.value));
  const [shown, more] = useMore(heads, 8);
  if (!heads.length) return null;
  return (
    <div className="ap-voice">
      <span className="np-caption">Later heads it writes into · {int(heads.length)}</span>
      <table className="np-table ap-grid">
        <thead><tr><th>Head</th><th className="ap-n">Query</th><th className="ap-n">Key</th><th className="ap-n">Value</th></tr></thead>
        <tbody>{shown.map((h) => <tr key={`${h.layer}-${h.head}`}><td>S{h.layer} head {h.head}</td><td className="ap-n">{h.query.toFixed(3)}</td><td className="ap-n">{h.key.toFixed(3)}</td><td className="ap-n">{h.value.toFixed(3)}</td></tr>)}</tbody>
      </table>
      {more}
      <p className="np-note">Length of its output direction after each head’s query, key and value projection: ‖W_Q o‖, ‖W_K o‖, ‖W_V o‖. A linear estimate from the weights; RoPE and Q/K/V norms are ignored.</p>
    </div>
  );
}
