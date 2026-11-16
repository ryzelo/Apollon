// The prompt and the answer as tokens, above the stage. An answer token starts the decomposition of its
// prediction; the position the model predicts from is underlined; a selected head lights what it looks at.
import React from "react";
import { Result, Token, int, pct, tokenText } from "../api";

export function PromptStrip({ result, answer, analysed, weights, weightsOf, disabled, onDecompose }: {
  result: Result;
  /** 1-based answer token this run looks at */
  answer: number;
  /** the run measured that token's prediction (not only generated the answer) */
  analysed: boolean;
  /** attention of the selected head over the positions */
  weights?: number[];
  weightsOf?: string;
  disabled?: boolean;
  onDecompose: (answer: number) => void;
}) {
  const generated = result.generated || [];
  // the context holds the prompt and the answer tokens before the analysed one
  const nPrompt = Math.max(0, Math.min(result.tokens.length, result.tokens.length - (analysed ? answer - 1 : 0)));
  const prompt = result.tokens.slice(0, nPrompt);
  const max = weights ? Math.max(...weights, 1e-6) : 0;
  const bar = (position: number) =>
    weights && position < weights.length ? <i className="ap-tok-w" style={{ width: `${(weights[position] / max) * 100}%` }} /> : null;
  const tip = (t: Token, position: number | null, extra = "") =>
    `${position != null ? `Position ${position} · ` : ""}token ID ${t.id}${position != null && weights && position < weights.length ? ` · attention ${pct(weights[position])}` : ""}${extra}`;
  return (
    <div className="ap-strip">
      <div className="ap-strip-row">
        <span className="np-caption">Prompt · {int(prompt.length)}</span>
        <div className="ap-strip-tokens">
          {prompt.map((t, j) => (
            <span key={j} className={`ap-tok ${j === result.focus ? "is-focus" : ""}`} title={tip(t, j, j === result.focus ? " · the model predicts from here" : "")}>
              {tokenText(t.text)}{bar(j)}
            </span>
          ))}
        </div>
      </div>
      {!!generated.length && (
        <div className="ap-strip-row">
          <span className="np-caption">Answer · {int(generated.length)}</span>
          <div className="ap-strip-tokens">
            {generated.map((t, i) => {
              const position = nPrompt + i; // its place in the context, when it is part of it
              const inContext = analysed && i < answer - 1;
              return (
                <button key={i} disabled={disabled}
                  className={`ap-tok ap-tok--answer ${analysed && i + 1 === answer ? "is-active" : ""} ${inContext && position === result.focus ? "is-focus" : ""}`}
                  title={`${tip(t, inContext ? position : null)} · answer token ${i + 1}. Click to decompose its prediction.`}
                  onClick={() => onDecompose(i + 1)}>
                  {tokenText(t.text)}{inContext ? bar(position) : null}
                </button>
              );
            })}
          </div>
        </div>
      )}
      <p className="np-note ap-strip-note">
        {weights
          ? `Blue bars: where ${weightsOf} looks from the underlined position.`
          : analysed
            ? "The filled token is the prediction this run takes apart. Choose another answer token to decompose that one."
            : "Choose an answer token to decompose its prediction."}
      </p>
    </div>
  );
}
