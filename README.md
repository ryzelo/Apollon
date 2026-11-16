# apollon

**What does every part of a transformer contribute to one prediction?** apollon answers this for a single
prompt, exactly and without training anything: no features, no labels, no surrogate model, only the matrix
products of one forward pass. It is written in Rust without an ML framework and computes the forward pass
itself, so it can record, split up and rerun everything inside it.

* **`--decompose`** splits the pass into **token voices**. The residual stream at the end is exactly the sum of
  its changes (embedding, every attention, every MLP). Each change is read out on its own as a distribution
  over the vocabulary. Every change is then removed and the model rerun, which gives its **total effect**.
* **`--probe "S12 MLP"`** is a **test bench for one component**. Every unit inside it (neuron, per-layer-embedding
  dimension, head) is changed by brute force. apollon then follows what changes downstream, all the way to the
  output.
* **`--circuit`** finds the **circuit down to single scalar products**: every matrix product of the pass is
  a sum of products W[o,c]·x_c (or q_e·k_e, a_j·v_j[e] inside attention), and apollon switches them off,
  coarse to fine, until only those the prediction needs are left – each decision an exact rerun.
* **`--trace "S12 #2328"`**, **`--neuron`** and **`--atlas`** treat **one MLP neuron as a complete computing
  structure**. `--trace` follows one prediction down to single scalar products and back through the positions
  the heads read; `--neuron` reports without a prompt everything the neuron can do; `--atlas` stores that for all
  neurons.
* **`--serve`** opens a **local web workspace**: generate an answer, pick a token, and decompose, probe and
  trace from there, with 3D, flow, unit and table views.

The matrix products run on the GPU where there is one (Metal on macOS, CUDA on NVIDIA), else on the CPU.
Every analysis checks itself in every run: the parts have to add up to the real pass, and any violation is
reported. The forward pass is compared against Hugging Face transformers in the test suite.

## Install

You need a recent stable Rust toolchain (tested with 1.96; the `rayon` dependency needs at least 1.80).

```
git clone https://github.com/ryzelo/Apollon.git
cd Apollon
cargo build --release
./target/release/apollon --help
```

* **macOS (Apple Silicon)** is the tested platform. Large matrix products run on the GPU through Metal
  (see [GPU](#gpu-metal-and-cuda)), or through Apple's Accelerate on the CPU.
* **NVIDIA GPUs (Linux, Windows)**: CUDA is used automatically when a driver is installed. Nothing has to be
  set at build time – apollon loads the CUDA driver and NVRTC when it starts.
* **Other platforms** use plain-Rust fallbacks for the same operations. They are compiled and checked (forced
  on macOS), but have not been run on Linux or Windows hardware.
* `.cargo/config.toml` builds for the local CPU (`target-cpu=native`). Remove that line for a portable binary.

apollon brings no weights. It reads models you already have on disk:

* Ollama models (`~/.ollama/models` or `$OLLAMA_MODELS`)
* the Hugging Face cache (`~/.cache/huggingface/hub` or `$HF_HUB_CACHE`)
* any `.gguf` file
* any folder with `config.json` + `*.safetensors` + `tokenizer.json`

`apollon --list` shows which local models are supported.

## Quick start

```
apollon -m EleutherAI/pythia-70m "The capital of France is"            # only the greedy answer
apollon -m EleutherAI/pythia-70m --decompose "The capital of France is"
apollon -m mlx-community/gemma-3-1b-it-4bit --chat --answer 7 --decompose "Was ist 2+2?"
apollon -m mlx-community/gemma-3-1b-it-4bit --chat --answer 7 --probe "S16 MLP" --contrast "Was ist 2+3?" "Was ist 2+2?"
apollon -m mlx-community/gemma-3-1b-it-4bit --chat --answer 7 --trace "S16 #2328" "Was ist 2+2?"
apollon -m mlx-community/gemma-3-1b-it-4bit --neuron "S16 #2328"
apollon -m EleutherAI/pythia-14m --pos -1 --circuit "Monday, Tuesday, Wednesday,"
```

`-m` takes an Ollama name (`qwen2.5:3b`), a Hugging Face cache name (`EleutherAI/pythia-70m`), a `.gguf` file or
a model folder. `--chat` wraps the prompt in the model's chat format (Gemma 3/4, ChatML). The model's answer is
generated greedily first. `--answer 7` analyzes the prediction of the 7th answer token (`--answer 1,6` analyzes
several, and `--pos N` analyzes a prompt position instead). `--target " Paris"` picks the target token; by default
it is the predicted one. With no analysis flag, apollon only prints the answer. Every analysis prints a report in
the terminal and writes an expandable HTML report (`--html NAME` sets the base name).

Documentation:

* [FINDINGS.md](FINDINGS.md): what apollon showed on "Was ist 2+2?" ("What is 2+2?") in Gemma 3 1B and
  Gemma 4 E4B.
* [FUNCTIONS.md](FUNCTIONS.md): what every function in the code does, step by step, with its known limits.

## Decomposition into token voices (`--decompose`)

```
apollon -m mlx-community/gemma-4-e4b-it-4bit --chat --answer 6 --decompose "Was ist 2+2?"
```
At the end, the stream at the focus token is exactly the sum of its changes. These are the embedding, every
attention, every MLP and, in Gemma 4, every per-layer embedding (PLE). Each one is multiplied by the scalars of
the later layers.

1. **Direct voice.** Each change goes through the final norm on its own and then through the output matrix. The
   norm centers for LayerNorm and does not for RMSNorm, and it divides by the *stored* scale of this pass, the
   same for all changes. The softmax of the result is the change's voice. apollon shows its top tokens, its
   entropy (how sharp it is), and which candidates of the real output it raises or lowers by how many percentage
   points. The bias of the final norm gets its own row. With a logit softcap (Gemma), every token logit is scaled
   by the fixed factor softcap(z)/z from this pass.
2. **Checks.** The voice logits add up to the real logits, and the normalized product of all voices is the real
   distribution (Pythia-70m: KL 3·10⁻⁸, Gemma 3 1B and Gemma 4 E4B: KL 0).
3. **Parts.** Attention splits into heads (with their attention pattern), and heads split into source positions
   (a_j·W_O v_j). MLPs split into neurons (a_i·W_down[:, i]). apollon shows the strongest (`--top K`, default 8)
   and puts the rest in one row. They are preselected over the 64 most likely tokens.
4. **Total effect.** apollon sets the change to zero at all positions (`--focus-only`: only at the focus token)
   and reruns the model from its layer. The intermediate states come from the first pass; rerunning with nothing
   removed is bit-identical. It reports the winners and losers of the distribution, the KL, and
   **indirect = total − direct** on the log scale. Indirect contains two things: what runs through later blocks,
   and the final norm rescaling once the change is gone. The second part is shown on its own as **"via norm"**:
   the stream at the end minus the change, with only the final norm and the softcap recomputed. No later block
   reads the last change, so there total = direct + via norm must hold (Pythia-70m: 1·10⁻⁵, Gemma 3 1B: 4·10⁻⁶ in
   log-odds). **"Direct share"** compares the spread of the direct and the indirect log change. It is weighted over
   the tokens that carry weight with or without the change.

All factors refer to the **odds p/(1−p)** of the target token, not to p. For small p the two are the same. At
p ≈ 1 (saturated), the odds still show how much lead a change adds. The terminal shows the overview of all changes
in order and expands the four strongest. The HTML report (`<name>_decompose.html`; set the name with `--html NAME`)
lets you expand everything.

| Model | voices | reruns | time (Apple M3, 8 GB RAM) |
|---|---|---|---|
| Pythia-70m | 368 | 13 | 0.7 s |
| Gemma 3 1B | 1,327 | 53 | 19 s |
| Gemma 4 E4B | 3,865 | 127 | ≈ 6 min |

## Probe: what do the units of a component do? (`--probe`)

```
apollon -m mlx-community/gemma-3-1b-it-4bit --chat --answer 7 --probe "S16 MLP" --contrast "Was ist 2+3?" "Was ist 2+2?"
```
You pick a component (`"S12 MLP"`, `"S11 PLE"`, `"S12 Attn"`; several with `;`). apollon changes every unit inside
it by brute force (neuron, per-layer-embedding dimension, head) and reruns the model from that layer. Many variants
run in **one** pass. Each variant is a copy of the positions from the changed spot up to the focus. Through
attention, it sees the unchanged prefix and itself, never other variants. RoPE and windows use the real position,
and from 64 rows on, the matrix multiplications run through Accelerate. Two checks run every time: the unchanged
copy in the variant pass must equal the real pass, and one variant must equal the same edit run on its own.

1. **Where does the component act?** It is zeroed completely at one position at a time, and also everywhere
   except one position. If one position carries at least half of the effect, the probe runs there (**local**, or
   choose with `--probe-pos N`). Otherwise the component is **distributed**, and every unit is changed at all
   positions at once. For MLPs this uses the `--probe-max N` largest by Σ|a|·‖W_down[:, i]‖ (default 1024).
2. **Brute force.** Every active unit is zeroed on its own. apollon reports the odds of the target and of the rival,
   the KL, the new top token, and the unit's own voice (partial softmax of what it writes at the focus token).
3. **Interplay.** The k strongest supports are switched off together, and conversely only they stay on
   (k = 1, 2, 4, …, all). From which k does the answer flip, and do a few units carry the whole function?
4. **Dose.** The activation is multiplied by −2, −1, 0, 0.5, 2 and 4 for the strongest units (`--depth N`,
   default 6).
5. **Path.** For the same units: the logit lens after every later layer, and the voice of every later attention /
   MLP / PLE (unchanged → edited). It also shows which later neurons change and which heads look elsewhere
   afterwards.
6. **Contrast prompt** (`--contrast "Was ist 2+3?"`). Values from there are inserted: does the component carry the
   number? The answer of the main prompt is reused with the same substitution applied to it, also when the model
   inserts spaces ("2 + 2 =" → "2 + 3 ="). Both sequences must have the same number of tokens.

The result appears in the terminal and as HTML (`<name>_probe_S16_MLP.html`). Gemma 3 1B, S16 MLP: 13,824 variants
in 33 s. Gemma 4 E4B, S11 PLE: about 3 min; S12 MLP: about 8.5 min (with 8 GB RAM the machine swaps).

Caveat: Gemma's MLP and PLE outputs pass through a post-norm. Switching off part of the units therefore scales the
remaining ones up, and the probe warns about this.

## One neuron down to single scalar products (`--trace`, `--neuron`, `--atlas`)

```
apollon -m mlx-community/gemma-3-1b-it-4bit --chat --answer 7 --trace "S16 #2328" "Was ist 2+2?"
apollon -m mlx-community/gemma-3-1b-it-4bit --chat --answer 7 --trace auto "Was ist 2+2?"
apollon -m mlx-community/gemma-3-1b-it-4bit --neuron "S16 #2328"
apollon -m mlx-community/gemma-3-1b-it-4bit --atlas gemma3.apla [--atlas-k 64] [--atlas-layers 0..8]
```
Neuron i of layer l sees its whole input only through two numbers. The pre-norm turns the stream x into
z = x/rms(x) (LayerNorm: (x − mean)/σ), so ‖z‖ ≤ √d. Then:

* p = g'·z + c_g and q = u'·z + c_u, with the effective read vectors g' = w_gate ⊙ γ and u' = w_up ⊙ γ
  (mean-free for LayerNorm) and the offsets c = w·β + b.
* a = act(p)·q (gated: SwiGLU, GeGLU) or a = act(p) (not gated: GPT-NeoX, GPT-Neo).

No features, no labels and no trained substitute model are involved, only the matrix products of this one
pass. The neuron's "large matrix" g'u'ᵀ has rank 1 and is never formed.

**Repertoire (`--neuron`, no prompt).** All reachable (p, q) form a filled ellipse c + √d·L·disk, with
L Lᵀ = G the Gram matrix of g' and u'. For fixed p, act(p)·q is linear in q, so maximum and minimum lie on the
boundary. apollon searches the boundary with an 8,192-step angle grid, refines with a golden-section search,
and reports a_max, a_min, their angles and the input direction z* that reaches the maximum. If g' ∥ u', the
ellipse collapses to a segment. A not-gated neuron reaches an interval.

**Static arrows.** Every source reaches the MLP input along a fixed path of layer scalars and norms. The
sources are:

* token and position embeddings
* every earlier neuron (its W_down column, through its post-norm)
* every per-layer-embedding dimension
* every bias

With all norm scales held at 1 the path is linear. Each source therefore has an arrow (g'·v, u'·v) per unit of
source activation. Heads have no static arrow, because what a head writes depends on what it reads; the report
says so.

The outputs of a neuron are the same table read the other way: its arrow in the (p, q) plane of every later
neuron, its token voice through the final norm and the output matrix, and the length of its projection onto
W_Q, W_K, W_V of every later head. The head values are marked as an approximation, because RoPE and the Q/K/V
norms are ignored.

**Runtime decomposition (`--trace`, at the focus of the usual `--answer`/`--pos` flow):**

* *Level 0*: p − b = Σ_j W[i,j]·x̂_j, with the weights exactly as the forward pass unpacks them.
* *Level 1*: with the frozen pre-norm scale s, every source c contributes p_c = s·g'·x_c, and Σ_c p_c = p.
  The sources are the embedding, every head split into the positions it reads, every earlier neuron (all of
  them), every PLE dimension and the biases. The reference point is the stream at the MLP input. Its own
  layer's attention belongs to it, except in parallel-residual models.
* *Level 0×1*: for one source, its own scalar products W[i,j]·γ_j·s·x_c,j. They are shown for the strongest
  sources, or for any source requested with `--scalars-of "S14 head 1 @16"` (also `"S14 #868"`, `"token"`, …).
* *Pairs*: with the frozen φ = act(p)/p, a = φ·Σ_c Σ_c' p_c·q_c' holds exactly. The pair table is an outer
  product, so only the two lists are kept. Each source's share of a is a_c = φ/2·(p_c·q + p·q_c), and Σ a_c = a.
* *Position in the repertoire*: a/a_max (or a/a_min) and the radius r ∈ [0, 1].
* *Removal by rerun*:
  * the neuron set to 0, giving direct, total and indirect effect as in `--decompose`;
  * each strong source removed **only from this neuron**: p′ and q′ are recomputed exactly from Norm(x − x_c),
    not with the frozen scale, and the resulting a′ is set via `patch`. The frozen-linear and the exact Δa are
    reported side by side.
* *Backward zoom*: from the neuron to its strongest sources, recursively into earlier neurons (`--trace-depth`,
  `--trace-top`). Heads are split down to the position j they read, and from there into **what position j
  carries**.
  * With the attention weight, the norm scales at both positions and the value norm frozen, an atom is linear
    in the stream x_j before the layer whose values the head reads (with shared K/V, the source layer).
  * The reader's vectors go back through post-norm, W_O, the attention weight, value norm, W_V and the
    pre-norm at j. The norm bias and value bias become a constant offset.
  * The same source enumeration as at level 1 then splits x_j into the embedding at j, the heads and neurons at
    j, PLE and biases, recursively across positions. Neurons at j are themselves decomposed at j.
  * The parts at position j come from a recording pass with focus j. Its stream is checked to be bit-identical
    to the main pass.

Nothing is stored in advance. Every product is recomputed from the recorded pass.

**Atlas.** The arrows of all neurons form one table C[later, earlier]. Read by rows it gives the inputs, read by
columns the outputs. `--atlas` computes it tile by tile as GEMM (tokens, W_down of every earlier layer, PLE,
biases) and keeps only the top k per row and per column. The full table would be far too large: for Llama 3.2
1B over 100 GB, with several hundred trillion operations. The file (`APLA`, versioned header) has one fixed
record per neuron: ellipse, a_max/a_min, the top-k inputs by |p| and by |q|, the outputs the same way, and the
voice. It can be read through mmap, built in parts (`--atlas-layers`) and resumed. A resumed or repeated layer
gives a byte-identical file. A layer's outputs are complete once all later layers are built. With `--atlas`,
`--trace` shows where the runtime-strong sources rank among the static inputs.

**Self-checks in every run:**

* Σ scalar products = p − b;
* Σ source vectors = the recorded stream; Σ sources = p and q; Σ pairs = a; a = the activation recorded by the
  forward pass;
* static arrow × runtime scales × activation = runtime contribution (adjoint against vector path);
* r ≤ 1 and a_min ≤ a ≤ a_max – a violation is reported loudly;
* rerun with the original value = clean pass;
* in the zoom: Σ sources at the read position + offset = atom, Σ source vectors = the checkpoint stream, and the
  pass at the read position = the main pass.

The atlas checks one neuron per layer against the on-demand computation, and at the end the column side.
BERT/RoBERTa get the static part only, through the chain of LayerNorms; `--trace` refuses them with a message.

Gemma 3 1B (Apple M3, 8 GB RAM):

* `--neuron`: 0.7 s
* `--trace` of one neuron with reruns and zoom: 1.4 s; zoom depth 3 with 30 read positions decomposed: 12 s
* `--atlas`: about 11 s per layer; with k = 64 the file is 1.1 GB (Gemma 4 E4B: about 2.7 GB)

## Circuit down to single scalar products (`--circuit`)

```
apollon -m EleutherAI/pythia-14m --pos -1 --circuit "Monday, Tuesday, Wednesday,"
apollon -m mlx-community/gemma-3-1b-it-4bit --chat --answer 7 --circuit \
        --circuit-ref "Was ist 3+5?; Was ist 6+1?; Was ist 4+3?; Was ist 7+2?; Was ist 1+8?" "Was ist 2+2?"
```
Every matrix product of the forward pass is a sum of scalar products. One such product at one position is an
**atom**:

* W[o,c]·x_c in the projections Q, K, V, O, gate, up, down (and the three products of the per-layer embedding
  in Gemma 4);
* q_e·k_e in a score of QKᵀ and a_j·v_j[e] in the mixing A·V, per head, query position and key position.

Pythia-14m has 7.1 million atoms on a 6-token prompt, Gemma 3 1B 1.5·10¹⁰ on the 22-token chat prompt.
Biases, norms, RoPE, softmax and activations are not atoms; they always run on what results.
`--circuit` keeps only the atoms the prediction needs: **KL(full model ‖ circuit) ≤ ε** (`--circuit-kl`,
default 0.1 nats) and **the same top token**.

**What "switched off" means** (`--circuit-ablation`):

* **mean** (default, the usual choice in circuit research): the activations in a switched-off atom are
  replaced by their **means over a reference distribution**. Every atom is a product of two factors: in a
  projection W[o,c]·x_c becomes W[o,c]·x̄_c (a unit switched off completely gives W·x̄), in a score q_e·k_e
  becomes q̄_e·k̄_e, in a mixing a_j·v_j[e] becomes ā_j·v̄_j[e].
  * With reference prompts – `--circuit-ref "A; B; …"`, same format as the prompt, wrapped like it with
    `--chat` and completed with their own greedy answer – the means are taken **position by position**: at
    position p over the references at position p (as in IOI-style work). The circuit then is what carries
    the *difference* to the references – with other numbers, the numbers.
  * Without references (or where no reference reaches a position), the mean over all positions of the
    analyzed sequence is used.
  * A leading `<bos>` stays itself: it is the same in every sequence (and an attention sink), so its mean is
    its own activations.
* **zero**: a switched-off atom is missing from its sum. This takes the model further off its usual
  distribution: whatever is removed leaves a hole that later layers never see in normal use.

**The search goes coarse to fine.**

1. Components at every position: attention (the query side: Q, scores, mixing, O), the K/V projections, MLP,
   PLE.
2. Heads, K/V groups, neurons, PLE dimensions.
3. Inside heads and K/V groups: the q dimensions, every key of QKᵀ and of A·V, every input column of W_O, the
   k and v dimensions; the rows of the PLE input projection. (One unit of a neuron off is the whole neuron off,
   so neurons go straight to level 4.)
4. Single atoms.
5. Final phase: whatever stayed as a block is split down to single elements, then every leaf is switched off
   alone against the final circuit, until nothing more can go.

Within a level:

* **Dose curves and a joint choice.** For every group – the neurons of one MLP at one position, the atoms of
  one neuron's read, … – its weakest pieces up to a share τ of the group's magnitude are switched off, each
  group alone, all in one batch. Then, for about 30 trade-offs between atoms removed and KL, every group takes
  its best τ, and all of these joint choices run exactly. The one that removes the most atoms and passes is
  kept.
* **Halving** (adaptive group testing): what is left is tested in halves, the halves that fail are halved
  again, down to single pieces.
* **Greedy acceptance**: pieces that may go alone are accepted cheapest first, as long a prefix of the list as
  still fits, many prefixes per pass.
* **Recheck**: at the end of every level each kept piece is tested alone once more against the new state.
* Levels 1–4 may use 25, 45, 60 and 85% of ε cumulatively, the final phase all of it, so that the coarse
  levels do not use up what the fine ones need.

**What makes it exact:**

* Every decision is a rerun of the real model from the earliest layer it touches. Magnitudes (|W·x|,
  activations, attention weights) only decide the order of the tests: no gradients, no linearization, no
  surrogate or replacement model, no trained dictionary.
* Many candidates run in one pass, each as a copy of the positions from its change to the focus (as in
  `--probe`). Every accepted step is then rerun on its own and has to pass again; the largest
  |KL batch − KL standalone| is reported.
* What provably has no effect any more is switched off without a test: a head whose W_O columns are all off,
  a neuron whose down is off, K/V rows that no score or mixing reads, a PLE dimension whose projection is
  off – and with zero ablation also what only gets an exact 0 as input (a head whose A·V products are all
  off, a neuron whose gate or up is off). The output then has to stay **bit-identical**, and that is
  checked.
* At the end every leaf is individually necessary: switching off any single remaining atom alone breaks the
  criterion.
* An empty cut gives the normal pass bit for bit.
* `tests/compare_circuit.py` applies the same cut to transformers in float64 – every projection recomputed
  from its weights with the keep mask, an attention function that drops (or replaces by their means) the
  switched-off q_e·k_e and a_j·v_j[e] – for random cuts over every kind of atom, for the circuit and for its
  complement, with zero and mean ablation, frozen and live norms. For mean ablation it computes the means
  itself from transformers and compares them with apollon's.

**Norms: frozen by default.** With frozen norms (`--circuit-norms frozen`) every norm – pre-, post-, q/k/v,
PLE and final norm – keeps the scale it has in the full model at that position, so a part that stays acts
with its real size. With live norms (`--circuit-norms live`) every norm recomputes its scale from what is
left, and a post-norm (Gemma) blows up any small rest to full size. On Gemma 3 1B, "Was ist 2+2?", the live
search ends with 20 atoms of one neuron (S24 #5561, values around 0.0006): alone they give "4" at 92%, but
everything *except* them still gives "4" at 100% – the norm makes them sufficient, the model does not use them.
Both modes are exact reruns of a well-defined computation; the report states which one was used, and an empty
cut with frozen norms has to give the normal pass bit for bit.

**Output.** Atoms on per layer, with the heads and neurons still alive; all leaves, sorted by how much the
prediction breaks without each one alone, with the value of every single atom; the circuit alone and
**everything except the circuit** (exactly the circuit's atoms off, all others on). The HTML report
(`<name>_circuit.html`) adds the levels and a layer × position map. `--circuit-depth N` stops at level N
(1 = components … 4 = atoms), `--circuit-tests N` stops refining after N tests (what is unresolved then stays
whole, and the report says so), `--dump-circuit FILE` writes the cut cell by cell and the logits as JSON.

**Example** (Gemma 3 1B, "Was ist 2+2?" → "4", mean ablation over five prompts with other numbers): of
1.54·10¹⁰ atoms, 18,948 stay on. With only them the model still says "4" (91%); without them it says "2".
They follow the two operands "2" through early neurons into heads of layers 14, 17 and 18 at the answer
position; the most necessary single product is in the mixing of layer 18, head 0, reading the first operand.
45 minutes on the M3 GPU – see [FINDINGS.md](FINDINGS.md).

**Example** (Pythia-14m, "Monday, Tuesday, Wednesday," → " Wednesday", 24%, zero ablation): of 7,110,144 atoms,
1,989 stay on with frozen norms. With only them the model still says " Wednesday" (24%, KL 0.0998); with everything
except them it says " and" (KL 8.8). About 1.02 million exact tests in 8,750 passes, 6.5 minutes on an Apple M3 –
most of it in the final phase, which resolves 200,000 atoms in blocks down to single ones. With live norms
the circuit has 803 atoms; its most necessary single product is a score in layer 3, head 3 – query "," [5] ·
key " Wednesday" [4], dimension 25.

**Limits.**

* One prompt, one prediction.
* The circuit is *one* minimal set (no single leaf can go), found greedily – not necessarily the smallest. A
  different order of tests can find a different one.
* Mean ablation depends on the reference: the circuit explains the prediction *relative to* it. With
  similar prompts (other numbers, same format) the circuit is what carries the difference – the number –
  rather than what produces the format. A first version averaged over all positions, `<bos>` excluded: on
  Gemma 3 the search then mostly kept the attention sink at `<bos>` intact – a statement about the
  reference, not about arithmetic.
* A key whose score products are all off gets the score 0, not −∞: taking a key out of the softmax is not a
  matrix product, so it is not an atom.
* The runtime grows with the size of the circuit (most of it goes into resolving the last blocks down to
  single atoms) and with the number of sequential steps: every accepted step is a standalone rerun. Gemma 3
  1B: 45 minutes to single atoms on an 8 GB M3 with the GPU, under 2 minutes to heads and neurons
  (`--circuit-depth 2`). `--circuit-depth` and `--circuit-tests` bound it.

## GPU: Metal and CUDA

```
apollon -m … --gpu auto|metal|cuda|off …     (default auto; also $APOLLON_GPU)
apollon -m mlx-community/gemma-3-1b-it-4bit --selftest-gpu
```
The matrix products – every projection and the output matrix – run on the GPU; everything else (norms,
attention, the cuts of `--circuit`, the analyses) stays on the CPU.

* **The weights stay quantized as in the file.** A kernel unpacks a block of rows exactly as the CPU does –
  all 16 GGML formats (F32, F16, BF16, Q4_0 … Q8_0, Q2_K … Q6_K, IQ4_NL, IQ4_XS, MXFP4) and MLX affine 2–8 bit
  – and a second kernel multiplies. One kernel source serves both backends (`src/gpu/kernels.h`).
* **Metal (macOS):** the mapped model file is handed to the GPU without a copy (unified memory, one buffer per
  tensor); the product runs on Apple's tuned GEMM (Metal Performance Shaders), with apollon's own kernel as
  a fallback (`APOLLON_METAL_OWN_GEMM=1`).
* **CUDA (NVIDIA):** the driver API and NVRTC are loaded at runtime (`libcuda.so.1` / `nvcuda.dll`,
  `libnvrtc.so.*` / `nvrtc64_*.dll`); the kernels are compiled for the GPU found. Every weight tensor is copied
  to the GPU once, the first time it is used. Without a driver apollon simply runs on the CPU.
* **`--selftest-gpu`** checks the GPU against the CPU: every format with random blocks and every matrix of the
  first two layers and the output matrix of the given model – unpacked values bit for bit, products to float
  rounding – and times a product with 512 rows on both.

Apple M3, 8 GB (Gemma 3 1B, MLX 4-bit): all 16 formats and 7 MLX variants unpack bit-identically to the CPU;
products agree to ≤ 2.5·10⁻⁷ relative, the whole pass (residual after every layer) to 4·10⁻⁷, the logits to
6·10⁻⁶. A product of 512 rows with the 6,912 × 1,152 gate matrix takes 6.3 ms on the GPU against 15.6 ms on the
CPU. The CUDA path is compiled from the same kernel source and checked to translate to PTX (clang), but it has
not run on NVIDIA hardware yet: `apollon --gpu cuda --selftest-gpu -m …` is the first thing to run there.

## Local web workspace

Build the frontend once, then start the local server from the repository root:

```sh
cd web
npm ci
npm run build
cd ..
cargo run --release -- --serve
```

Open http://127.0.0.1:8765. Select a discovered model or enter a local model path,
write a prompt and generate an answer. The prompt and the answer appear as tokens
above the stage; choosing an answer token decomposes its prediction. Choosing a
component shows the real output next to the output without it, and starts a probe
of its units, a neuron trace or a contrast-prompt insertion with one click. Each
follow-up run is listed below the run it was started from. The space view, the flow
view, the unit view and the table share the selected component and metric, and the
address holds the open run, view, metric and selection, so a finding can be linked.

The space view is a 3D scene that can be turned and zoomed. For a decomposition it
places every part by layer, token position and signed effect on the target, with a
head's read positions as arcs. For a probe it shows every unit and, for the units
examined in detail, what their removal changes in each later layer. For a trace it
shows the backward tree across layers and positions, and for a static neuron its
strongest inputs and outputs in the (p, q) plane. One point is one unit; the units
of a remainder are drawn dim, and each scene states what it draws and leaves out.

Two sliders on the stage narrow every view to a range of layers and a least size of
effect. Dragging a rectangle in the flow view, or Shift and Ctrl in the table, marks
several parts, which then light up in all views. A second run can be compared: its
values appear beside each part in the panel, as a column in the table and as white
points in the space view. Pinned parts are found again in every run by layer, type
and unit. Keys: ← → step through the layers, ↑ ↓ through the ranking, Enter probes
the selected component, T traces a neuron, P pins, Esc clears.

The server listens only on loopback. One analysis runs at a time; queued and running
jobs can be cancelled. Runs and reports are stored in `output/web-runs/` and loaded
on restart. A job interrupted by shutdown is marked interrupted and can be rerun.
Model weights remain in the Rust worker. The browser receives structured results.
`DELETE /api/runs/{id}` removes a finished run and its files for good. While a run
is in progress, its detail carries the engine's last progress line. Set
`APOLLON_PORT` to serve on another local port.

Structured results (`--result-json`, schema version 2) hold what the HTML reports
show: for `--decompose` the tokens each change raises and lowers, the output without
it and every head's attention over the positions; for `--probe` the per-position
ablations, group ablations, dose curves, contrast patches and what changes in later
layers; for `--trace` the backward tree, removals, scalar products and pairs; for
`--neuron` the repertoire and the strongest static inputs and outputs.

On Windows with the MSVC Rust toolchain installed:

```powershell
cargo +stable-x86_64-pc-windows-msvc run --release --target x86_64-pc-windows-msvc -- --serve
```

For frontend development, run `npm run dev` in `web` alongside the Rust server.
The production frontend must be built before serving it with Rust.

The analysis form also supports static neuron reports. A trace can build and reuse
a neuron atlas under **Advanced options**, which also hold the mode-specific
parameters. Completed runs list their HTML reports and downloadable files in the
panel, where the prompt token IDs and the binary forward pass can be exported too.

## Supported – the construction kit

A weight file contains no computation recipe. apollon therefore assembles every layer from components that it
derives from the configuration and the weights:

* norms before and/or after attention and MLP (RMS/LayerNorm), and pre-, parallel- or post-LN residuals
* GQA/MQA, sliding windows, q/k/v norms and shared keys/values
* RoPE (normal, pairwise, NeoX, Llama 3, linear, proportional/partial) and learned positions
* score and logit softcapping
* gated and ungated MLPs (SiLU, GELU, GELU-tanh), per-layer embeddings and layer scalars
* bidirectional encoders without a head

| Family | GGUF (Ollama) | safetensors / MLX 4-bit |
|---|---|---|
| llama, mistral, qwen2, qwen3 | ✓ | ✓ |
| gemma4 | ✓ | ✓ |
| gemma3 | – | ✓ |
| gpt_neox (Pythia) | – | ✓ |
| gpt_neo (TinyStories) | – | ✓ |
| bert / roberta (encoder) | – | ✓ (forward pass and dump only, no prediction) |

* GGUF weights: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K–Q6_K, IQ4_NL, IQ4_XS, MXFP4; MLX affine
  2–8 bit. Everything stays in the memory-mapped file and is unpacked row by row while computing.
* Tokenizers: byte-level BPE (GPT-2, Qwen, Llama 3, GPT-NeoX, RoBERTa via vocab.json + merges.txt), and
  SentencePiece with scores (GGUF llama) or with merges (Gemma, Llama 2 `tokenizer.json`).
* In the Hugging Face cache, files are merged across all snapshots of a model.
* For Gemma, `<bos>` is prepended when the tokenizer does not do it itself (as llama.cpp/Ollama do).

Not described yet (clear error message): mixture-of-experts, Gemma 2, Phi, GPT-2, Falcon …

## Tested

| Test | Result |
|---|---|
| `tests/compare_hf.py`: transformers on tiny Llama, Pythia-14m/70m, random Qwen2/Qwen3/Llama 3 RoPE | tokens equal; residual per layer + logits: max. rel. dev. 7·10⁻⁸ … 8·10⁻⁵ |
| `tests/compare_hf_extra.py`: transformers on random Gemma 3/Gemma 4 with all special parts (prompt longer than the sliding window, shared K/V, PLE, scalars, softcapping), real TinyStories-33M (GPT-Neo), real unixcoder (RoBERTa) | max. rel. dev. 6·10⁻⁷ … 1.6·10⁻⁵ |
| `tests/compare_gguf_llama.py`: Llama as GGUF (q/k permuted, SentencePiece with scores) | tokens equal, logits 2.5·10⁻⁷ |
| `tests/compare_dequant_qwen_tok.py`: 16 weight formats against `gguf`; Qwen2.5 GGUF tokenizer against Qwen's `tokenizer.json` | bit-identical; 10/10 |
| `tests/compare_mlx.py`: MLX 4-bit against Apple's `mx.dequantize` (gemma-3-1b, gemma-4-e4b) | bit-identical |
| `tests/compare_tokenizer.py`: Gemma 4 (GGUF + MLX), Gemma 3, GPT-Neo, RoBERTa against the originals | 10/10 texts per model |
| `tests/compare_ollama.py`: the sorted top-5 probabilities against Ollama's logprobs (needs a running Ollama server) | qwen2.5:3b: ≤ 0.1 percentage points on five prompts, 2.8 on the chat prompt (Ollama computes with its own implementation and rounded activations) |
| `tests/compare_neuron.py`: `--trace … --dump-neuron` against transformers hooks + NumPy for random Llama, Qwen3, Gemma 3 and the real Pythia-14m: p, q, a, s, x̂, every scalar product, the contribution of the embedding, every head and every earlier neuron, biases, Gram matrix, extremes (400k-angle grid, 20k random inputs); for the strongest atom from layer 1 on, the chain to its read position, the value offset, and every source at that position | max. rel. dev. 10⁻⁹ … 1.4·10⁻⁶ |
| `tests/compare_circuit.py`: `--circuit --dump-circuit` against transformers in float64 – random Llama, Qwen3, Gemma 3 and the real Pythia-14m; four random cuts over every kind of atom (`--circuit-random`; zero and mean ablation, frozen and live norms, with and without `--circuit-ref`), the circuits of the search and their complements; the means against means computed from transformers | random models ≤ 6·10⁻⁷; Pythia-14m ≤ 8·10⁻⁵ (float32 rounding; passes if within max(10⁻⁴, 2 × the deviation of the same reference in float32)) |
| `--selftest-gpu` (Metal, Apple M3): every weight format with random blocks, every matrix of Gemma 3 1B (MLX 4-bit), Qwen2.5 3B (GGUF Q4_K/Q6_K), Gemma 4 E4B (GGUF), Pythia-14m/70m (F16) against the CPU | unpacked values bit-identical; products ≤ 2.5·10⁻⁷ |
| `cargo test`: ellipse extremes against random inputs on the sphere (2–32 dimensions, all activations, degenerate cases) | no sample above a_max or below a_min, extremes reached |

On Pythia-14m, float32 rounding matters: the model has very large activations. On the sentence "Numbers 12345 and
umlauts: äöüß €", apollon deviates by 2.2·10⁻⁴ from a float64 reference, and transformers in float32 by 1.1·10⁻⁴.
On the test sentences it stays below 10⁻⁴. The test inputs are deliberately multilingual (umlauts, emoji, CJK,
special tokens).

Running the tests:

```
cargo test --release                                   # unit tests (no models needed)
pip install --target /tmp/ref transformers gguf tokenizers safetensors torch numpy
PYTHONPATH=/tmp/ref python3 tests/compare_hf.py target/release/apollon /tmp/work
PYTHONPATH=/tmp/ref python3 tests/compare_neuron.py target/release/apollon /tmp/work
PYTHONPATH=/tmp/ref python3 tests/compare_circuit.py target/release/apollon /tmp/work
```

* Every script states in its docstring which models it needs. The small random models are generated on the
  fly; the real ones (Pythia, TinyStories, unixcoder, the Gemma tokenizers) must be in the Hugging Face cache.
* Cases whose model is missing are skipped.
* `compare_mlx.py` needs the `mlx` package (Apple Silicon).
* `--dump FILE`, `--tokens-only`, `--dump-neuron`, `--dump-circuit` and `--circuit-random` exist for these tests.

**Memory:** apollon itself needs little RAM, because the weights stay in the file. If Ollama holds the same large
model in memory at the same time, swap grows a lot on an 8 GB machine. Running `ollama stop <model>` first helps.

## License

MIT, see [LICENSE](LICENSE). Model weights are not part of this repository; their own licenses apply.
