# apollon – what the technique does and what we found

As of 2026-11-14 · models: Gemma 3 1B and Gemma 4 E4B (MLX 4-bit), Pythia-70m for testing.
The prompt is usually "Was ist 2+2?" ("What is 2+2?") in chat format; what we study is the answer token "4".

## The technique

### Decomposition into token voices (`--decompose`)
At the end, the residual stream is exactly the sum of its changes: the embedding, every attention, every MLP and,
in Gemma 4, every per-layer embedding (PLE). Each change is sent on its own through the final norm (with the scale
from exactly this pass) and the output matrix. The softmax of that is its **direct voice**. Then each change is set
to zero and the model is rerun from its layer; that is its **total effect**. **Indirect = total − direct** is what
runs through later layers. The part that only comes from rescaling the final norm is reported separately
("via norm"). All factors refer to the odds p/(1−p), because p of "4" is saturated (100%). Attention splits further
into heads and "head reads token j", and MLPs split into neurons.

Checks:
* the sum of the voices equals the real logits;
* the normalized product of all voices equals the real distribution (KL 0 for Gemma 3 and 4);
* a rerun with nothing removed is bit-identical;
* for the last change, total = direct + via norm (≤ 10⁻⁵).

### Probe (`--probe "S12 MLP"`)
A test bench for one component. Every unit inside it (neuron, PLE dimension, head) is changed by brute force, and the
model is rerun from that layer. Hundreds of variants run in one pass; through attention, each one sees only the
unchanged prompt and itself. The stages:
1. **Where** does the component act: at one position, or spread over all of them?
2. **Brute force:** every unit switched off on its own.
3. **Interplay:** the k strongest switched off together, or only they left on.
4. **Dose:** activation ×(−2 … 4).
5. **Path:** what changes afterwards in every later layer (partial softmax of every attention/MLP/PLE, single
   neurons, where heads look).
6. **Contrast prompt:** insert values from "Was ist 2+3?". Does the component carry the number?

Checks:
* the unchanged copy in the variant pass matches the real pass (2·10⁻⁶ in log-odds);
* a variant run on its own with the same edit gives the same result (≤ 7·10⁻⁵);
* the reference tests against transformers still pass (10/10).

## What we found

### In general
1. **Computing and saying the answer happen in different places.** In Gemma 4, only five changes are
   indispensable: S0 MLP, S12 MLP, S11 PLE, S0 attention and the embedding. None of them votes directly for "4"
   (direct share 0%). Only S37–S41 vote directly for "4", and they do so across languages: S39 MLP votes for "四"
   and "Four", S37 MLP for "четвер" and "fourth".
2. **The loud parts can be removed; the quiet ones cannot.** S41 MLP raises the odds of "4" directly by ×5·10⁶,
   but in total only by ×2.5; without it, the final norm and the PLE compensate. If instead S11 PLE is removed,
   Gemma 4 answers "5" (51%).
3. **Direct methods point at the wrong parts.** The logit lens and direct logit attribution highlight S37–S41.
   A gradient approximation also fails for Gemma 4 (r = −0.34 against exact). There are two reasons: "4" sits
   almost at the softcap (z = 29.3 of 30), and the important blocks act by switching later attention and gates.
   For Gemma 3, the direction of the approximation is right in 45 of 51 cases.
4. **The last layer often acts through the normalization.** Gemma 3's S25 MLP votes slightly against "4"
   directly (×0.42), but its total effect is ×117. All of that comes from the rescaling of the final norm (×280).

### Gemma 3 1B: a real number circuit
- Without S14 attention, S14 MLP, S16 MLP or S17 attention, the model answers "2". The direct voice of S17
  attention itself says "2" (45%).
- **S16 MLP** acts almost only at the last token. **8 of 6,912 neurons** carry the whole function. If they are
  all missing, the answer is "2"; if only they stay on, nothing changes (×1.00). Their voices are number words:
  #2328 "десять / cinq / five", #5370 "Fifth / sixth", #1085 "twice / two".
- The mechanism: without S16 MLP, head 3 in S17 no longer looks at the operand "2" (46%) but at `<bos>`.
  After that, neuron S24 #5561, which says "4", stops firing.
- With the values from "2+3", the odds of "5" rise by ×10⁷, but the answer stays "4". S16 carries part of the
  number, not all of it.

### Gemma 4 E4B: switches and infrastructure, not the number
- **S11 PLE** acts spread over all positions, most at "?" and "+". Two of its 256 dimensions are huge
  (#86 with a = −374, #243 with a = −113) and carry almost everything. Without the whole component, heads 1 and 4
  in S12 look at the wrong places (at "ist" in the question instead of in the answer). Afterwards S37/S39 no longer
  say "four", and S40 votes for "5". Values from "2+3", however, move "5" only by ×1.26. So S11 PLE does not carry
  the number; it makes sure that S12 finds the right places.
- **S12 MLP** is general infrastructure. No single neuron changes the odds by even ×2; even the 512 strongest
  together do not flip the answer, and values from "2+3" give ×1.75. Removed completely, it produces nonsense
  ("ein"): the model stops working.

### Earlier (backtracking mode of the predecessor tool, since removed): 167 + 186
Gemma 4 computes 353 correctly. Before the hundreds digit, "三百" (three hundred) shows up as an intermediate
step. The answer digits read the operand digits repeated in the answer (the hundreds place reads the tens 6 and
8). Seven parts at the answer position flip the hundreds digit from 3 to 2, i.e. they remove the carry – but only
in this problem. In 6 problems without a carry they have no effect, and in 4 other problems with a carry they
work only once (158 + 167 → 69%). There is no general carry circuit, rather a collection of rules of thumb.

## Limits
- Everything holds for **one prompt**. "2+2" is probably memorized.
- **Setting things to zero is a crude edit.** If the model then collapses completely (nonsense as the answer),
  that only shows "is needed", not "computes here". The telling cases are the ones where a *different number*
  comes out.
- **Post-norms (Gemma):** MLP and PLE outputs pass through their own norm. Switching off only part of the units
  inflates the rest. That is why S11 PLE without its two big dimensions gives nonsense ("z"), but removed
  completely gives a clean "5".
- **Preselection:** for S12 MLP (distributed), only the 1,024 largest of 10,240 neurons were tested individually.
- **Runtime** on an Apple M3 with 8 GB RAM: decomposition of Gemma 4 ≈ 6 min, probe of S12 MLP ≈ 8.5 min (it swaps).

## Reproduce
```
apollon -m mlx-community/gemma-4-e4b-it-4bit --chat --answer 6 --decompose "Was ist 2+2?"
apollon -m mlx-community/gemma-4-e4b-it-4bit --chat --answer 6 --probe "S11 PLE; S12 MLP" --contrast "Was ist 2+3?" "Was ist 2+2?"
apollon -m mlx-community/gemma-3-1b-it-4bit  --chat --answer 7 --probe "S16 MLP" --contrast "Was ist 2+3?" "Was ist 2+2?"
```
These commands print the terminal report and write the expandable HTML reports.
