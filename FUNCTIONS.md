# apollon – what the code actually does (as of 2026-11-16, including the circuit search)

All files in `src/` (≈ 15,500 lines of Rust, plus the GPU kernels in `src/gpu/kernels.h`) and `tests/` (8 Python scripts, plus Rust unit tests) were read.
There is no ML framework: apollon computes the forward pass itself. That is why it can record, split up and
rerun every intermediate quantity from any layer.

---

## 1. Flow of one call

```
main()
 ├─ parse_args()                      read the CLI
 ├─ load(spec) → resolve(spec)        find file / folder / Ollama name / HF cache name
 │    ├─ model::load_gguf()           GGUF (llama, mistral, qwen2, qwen3, gemma4)
 │    └─ model::load_hf()             safetensors incl. MLX 4-bit (also gemma3, gpt_neox, gpt_neo, bert/roberta)
 ├─ describe(m)                       architecture line on stderr
 ├─ --serve → server::serve()         the local web workspace (every analysis runs as a separate CLI process)
 ├─ gpu::init(--gpu)                  Metal / CUDA / CPU for the matrix products
 ├─ statics()                         no prompt needed: --neuron → inspect::run(), --atlas (without --trace) → atlas::build()
 └─ analyze()
      ├─ Tokenizer.encode(prompt)     (optionally chat_wrap first)
      ├─ --tokens-only  → token ids as JSON, done
      ├─ --dump FILE    → write_dump(), done   (for the reference tests)
      ├─ no analysis    → print generate(32 tokens), done
      └─ analyses()
           ├─ generate(max(--answer, 12) tokens, greedy)
           ├─ per chosen answer token k: sequence = prompt + answer[..k-1], focus = last token
           │    ├─ probe::run()      for every part from --probe
           │    ├─ trace::run()      with --trace (one neuron down to scalar products; reads --atlas if given)
           │    ├─ circuit::run()    with --circuit (circuit search down to single scalar products)
           │    └─ decompose::run()  with --decompose
           └─ with --contrast: second prompt, the same substitution applied to the answer too (2+2 → 2+3)
```

The analyses with a prompt (decompose, probe, trace, circuit) look at **one** prediction, the one at the focus token.
`--neuron` and `--atlas` need no prompt: they compute from the weights alone. All of them print a terminal
report and write an HTML file (the atlas writes a binary file).

---

## 2. Functions per file

### `main.rs` – CLI, finding models, flow (940 lines)
| Function | What it actually does |
|---|---|
| `parse_args` | Reads all options. Free words become the prompt. `--probe "A; B"`, `--trace` and `--neuron` are split at `;`. `--atlas-layers A..B` is a half-open range. Hidden: `--selftest-dequant TYPE IN OUT` unpacks raw GGML blocks (tests only). |
| `statics` | Analyses without a prompt: `--neuron` (static report per neuron) and, if no `--trace` is set, `--atlas` (build). |
| `home`, `ollama_root`, `hf_root` | Paths from `$HOME`, `$OLLAMA_MODELS` and `$HF_HUB_CACHE`. |
| `ollama_blob` | Reads the Ollama manifest `name:tag` and returns the model's blob path (a GGUF file). |
| `hf_files_in` | Collects config/weights/tokenizer across **all** snapshots of an HF repo. The weights come from the first snapshot that has any `.safetensors`. |
| `hf_snapshots` | Snapshot folders of an HF cache name, newest first. |
| `resolve` | In this order: file → GGUF, folder → HF, Ollama name, HF cache name. |
| `load` | `resolve`, then `load_gguf` or `load_hf`. |
| `list` | `--list`: all Ollama and HF models with ✓/✗ depending on the architecture. |
| `describe` | One-line architecture description (identical layers grouped, PLE, softcap, layer scalars). |
| `term_width` | Terminal width via `ioctl`, else `$COLUMNS`, else 140. |
| `chat_wrap` | Wraps the prompt in the chat format of Gemma 4 (`<\|turn>`), Gemma 3 (`<start_of_turn>`) or ChatML (`<\|im_start\|>`), decided by which special token exists. |
| `replace_loose` | Replaces `old` by `new` in the answer text and skips spaces ("2+2" also matches "2 + 2"). The first `ctx` characters are context only. |
| `substitution` | Finds the differing middle of prompt and contrast prompt (common start and end cut off, up to 2 characters of context kept in front). |
| `target_token` | `--target`: finds the token whose text matches exactly; an error if the text is not a single token. |
| `analyses` | Generates the answer greedily (at least 12 tokens) and calls probe/trace/decompose per `--answer k`. With `--pos` the prompt is only cut at that position, nothing is generated. |
| `print_answer` | Answer text plus numbered token list. |
| `write_dump` | Binary file `APLN`: header, token ids, residual after the embedding and after every layer (all positions), logits at all positions. |
| `analyze` | Tokenizes, checks the ids, sets the focus and branches (see the flow). |
| `main` | Entry: threads, color, loading, timing. |

### `model.rs` – transformer as a construction kit + loaders (2,284 lines)
**Components**
| Function | What it does |
|---|---|
| `act_fn`, `parse_act` | SiLU, GELU (erf), GELU-tanh, ReLU; reads the name from the config. |
| `Norm::rms` | Constructor for RMSNorm. |
| `Norm::apply` | RMSNorm or LayerNorm (computed in f64). Returns **the scale 1/rms or 1/σ** so that it can be frozen later. |
| `Norm::scale` | Only that scale for one row. |
| `Norm::back` | Transpose of the frozen norm: maps a direction in normed space back to residual space (centered for LayerNorm). |
| `Norm::rows` | The norm row by row over [T][d]. |
| `Norm::map_part` | **Core of the decompositions:** maps a part linearly with a fixed scale `s` (RMS: `c·s·w`, LN: `(c−mean c)·s·w`, without bias). The sum of all parts gives exactly the norm of the whole. |
| `Rope::apply`, `inv_freq`, `llama3_inv_freq` | RoPE pairwise or NeoX style, linearly scaled, Llama 3 scaling, "proportional" (Gemma 4: only part of the dimensions rotated). |
| `RunOpts::has` | Whether an ablation is set. |
| `add_bias` | Adds a bias to every row. |
| `top_probs` | The k most likely tokens with p. |

**Forward pass**
| Function | What it does |
|---|---|
| `Model::lm_head`, `vocab_rows`, `n_layer` | Output matrix (the embedding itself when tied), vocabulary size, number of layers. |
| `Model::final_logits` | Final norm → output matrix → logit softcap `tanh(z/c)·c`. |
| `Model::attention` | Q/K/V (also the fused QKV of NeoX), Q/K/V norms, RoPE, GQA, sliding window, score softcap, shared K/V from an earlier layer. Heads run in parallel. Returns each head's contribution `W_O,h·z_h` (via `matmul_split`), the attention weights of the focus and optionally **atoms**: head h reads source j, i.e. `W_O,h(a_hj·v_j)`. In **variant mode** every row sees the clean prefix before its segment plus its own segment, never other variants. |
| `Model::run` | The one shared forward pass: embedding (+ position/type embedding, embedding norm), Gemma 4 PLE inputs, then per layer attention → (post-norm) → residual → MLP → (post-norm) → PLE → layer scalar. Residual variants: pre, parallel (NeoX), post (BERT). **Edits:** ablation whole or per row (attn/MLP/PLE), `patch` (set a neuron to a value), `head_scale`, `ple_scale`, `start` (continue from layer l with a given stream), `variants` (batch of variants). **Recording:** everything at the focus (stream after every layer, heads, MLP output, activations, PLE, attention, bias parts, post-norm scale), optionally all activations, checkpoints (stream before every layer + K/V), selected rows per layer, dump. |
| Recording (added for the neuron analysis) | At the focus also: the stream the MLP pre-norm reads (`focus_mlp_x`; pre: after the attention, parallel: before the layer), the MLP input x̂ (`focus_mlp_in`), the scale of the attention post-norm (`focus_attn_post`), the gated PLE input with the scale of the PLE norm (`focus_ple_gated`). Pure additions: logits and dumps stay bit-identical. `act_fn`, `pos_embed`, `type_embed`, `embed_norm` are public. |
| Cuts (added for `--circuit`) | `RunOpts.cut`: per row of the pass and per layer a `LayerCut` – for every matrix product (`Site`: Q, K, V, O, gate, up, down, PLE-in, PLE-gate, PLE-proj) a `MatCut` (units switched off completely, `PartialUnit`s with the weights of the unit along its atom axis and the atoms off or on), and per head a `HeadCut` (keys of QKᵀ and of A·V switched off, or single dimensions of them). Row-wise sites (Q/K/V/gate/up/PLE-in/PLE-gate): a unit is one output y_o = Σ_c W[o,c]·x_c; column-wise sites (O/down/PLE-proj): a unit is one input x_c with the atoms W[r,c]·x_c along the outputs. `cut_rows` sets switched-off outputs to 0 and sums partial units **directly over the atoms that stay on**; `cut_cols_in` zeroes the input of every cut unit, `cut_cols_out` adds the atoms of partial units that stay on back one by one (never "full sum minus the switched-off ones": that loses precision when the switched-off products are the large ones). The cut acts before the bias; norms, RoPE, softmax and activations run on what is left. QKᵀ: a switched-off key gets the raw score 0 (the empty sum), then scale, softcap, softmax as usual; A·V: switched-off products are missing from z. In variant mode the keys are looked up by position. Fused QKV (NeoX): units are mapped to their row in the fused matrix. With an empty cut the pass is bit-identical to the old one (checked against a build without the cut code). |
| Mean ablation (added for `--circuit`) | `Cut.means` (`Means` → per layer `LayerMeans` → per position `PosMeans`: the mean inputs h, z, h2, act, PLE, token vector; W·x̄ for the row-wise sites; q̄, k̄, v̄ as attention reads them; ā per query/key position) and `Cut.pos` (the position of every row). QKᵀ off: q̄_e·k̄_e (k̄ of the key position), A·V off: ā·v̄_j[e]. With it a switched-off atom is replaced by its mean instead of 0: `cut_rows` gives an off unit W·x̄ and a partial unit Σ over its atoms that stay on plus `mean_off` (Σ of the switched-off atoms with x̄, a constant computed once when the cut is built); `cut_cols_in` puts x̄_u into the input of an off unit; `cut_cols_out` adds the atoms of partial units that stay on and once per row `mean_add` (the switched-off ones with x̄_u, laid out like a row of the output). The first version summed every atom per row with a lookup per atom; on Gemma 3 the final phase then took about 14 s per pass; QKᵀ and A·V add the mean products of the off dimensions. Without means the code paths are the old ones (zero mode bit-identical to the build before). |
| Frozen norms (added for `--circuit`) | `RunOpts.record_norms` fills `Trace.norms` (`NormScales`: the scale 1/rms or 1/σ of every norm at every position – embedding norm, PLE input norm per layer, per layer attention pre-norm, q/k/v norms per head/group, attention post-norm, MLP pre- and post-norm, PLE output norm, final norm). `RunOpts.frozen_norms` makes every norm use these scales (by the real position of the row) instead of its own: `Norm::apply_scale`, `apply_fz`, `rows_fz`; LayerNorm still subtracts the live mean. `final_logits_scaled`, `probe::read_out_fz` do the same for the final norm. With the scales of the same pass the result is bit-identical (checked in every `--circuit` run). |
| Recording (added for `--circuit`) | `RunOpts.record_inputs` fills `Trace.inputs` (per layer, all positions: the attention input, Q/K/V as projected and after norm + RoPE, all attention weights, z, the MLP input, gate/up/activation, the PLE inputs) and `Trace.x_tok`. Only for the proxies and the reported atom values. |
| `Model::generate` | Greedy continuation. **No KV cache:** every new token is a full pass over the whole sequence. Stops at a special token. |

**Weight sources and loaders**
| Function | What it does |
|---|---|
| `trait Src` (`has`, `mat`, `vec`, `opt_vec`, `opt_mat`) | Uniform access to tensors from GGUF or safetensors. |
| `impl Src for Gguf` | Tensor → `Mat` directly on the mapped file. |
| `St::open` | Reads the safetensors header, maps the files, takes the MLX quantization (bits/group) from the config. |
| `St::keys` | All tensor names. |
| `impl Src for St` | `U32` + `.scales` (+ `.biases`) is recognized as an MLX matrix. The group size comes from the config or is inferred (32/64/128). |
| `type_summary`, `finish` | Share of the weight types ("MLX-Q4/64 98%"). |
| `PleModel::proj` | The PLE input projection (all layers), for `--circuit`. |
| `Mlp::col_norms` | Lengths of the columns of W_down, computed once and cached. |
| `mlp`, `attn_basic` | Constructors with defaults. |
| `kv_source` | Gemma 4: for shared K/V the last earlier layer of the same type (local/global). |
| `gu`, `gf` | Read GGUF metadata as numbers. |
| `load_gguf` | Builds the model from GGUF. For Gemma 4 with everything it needs: Q/K norms, V norm without weight, attention scale 1, sliding window, shared K/V, `rope_freqs`, PLE, `layer_output_scale`, embedding ×√d, logit softcap. Pairwise RoPE for Llama/Mistral, NeoX style otherwise. |
| `read_json`, `hf_tokenizer` | Tokenizer from `tokenizer.json` or `vocab.json` + `merges.txt`. |
| `load_hf` | Llama/Mistral/Qwen2/3/Gemma 3/4 (finds the weight prefix, also `language_model.`), GPT-NeoX (parallel residual, partial RoPE), GPT-Neo (local/global, no score scaling), BERT/RoBERTa (post-LN, encoder without head). Gemma 3: norm weights +1. Gemma: `<bos>` is prepended. |

### `decompose.rs` – `--decompose` (1,129 lines)
| Function | What it does |
|---|---|
| `Kind::is_change` | Whether a part is a "change" (embedding, attn, MLP, PLE, final bias) or a sub-part (head, source, neuron, bias). |
| `Parts::push`, `Parts::row` | Collect part vectors at the focus, each already multiplied by the later layer scalars. |
| `add`, `by_abs_desc` | Vector addition; sorting by magnitude. |
| `Preselect::strength`, `Preselect::logits` | **Preselection:** logits only for the 64 most likely tokens, weighted half on the target and half on the competition by p; strength = weighted spread. Only decides which sources/neurons get their own row. |
| `effect` | Compares the real distribution with the one after the rerun: KL, Δ log-odds for up to 5 targets, winners/losers (≥ 0.05 pp), weighted spread of direct and indirect, top without the change. |
| `effects` | (direct, total, indirect = total − direct) of a change. |
| `direct_share` | Spread direct / (direct + indirect). |
| `total_strength` | KL of the total effect (the sort key). |
| `name` | Display name ("S12 MLP", "head 3", "reads [5]"+"", "#2328 (a=1.2)"). |
| `post_bias` | Bias of a post-norm (belongs to the change, but to no head/neuron). |
| **`run`** | The algorithm, see section 3. |
| `voice_txt`, `shift_txt`, `dp_txt`, `shifts`, `head_targets` | Text helpers: top tokens of a voice, ±pp, "raises/lowers", where a head looks. |
| `terminal` | Terminal report: checks, table of all changes (direct / total / indirect / via norm / KL / direct share / gains / losses), "strongest direct", "strongest overall", "hidden partial beliefs" (direct share < 25%), the 4 strongest changes expanded. |
| `shift_html`, `dp_html`, `voice_html`, `opt_factor_html`, `sharpness`, `part_row`, `html` | The same as an expandable HTML page (heads → source positions, neurons). |

### `probe.rs` – `--probe "S12 MLP"` (1,306 lines)
| Function | What it does |
|---|---|
| `pos_label` | "[7]" or "[all]". |
| `parse_part` | Uses `neuron::parse_part`; "S12 MLP" / "S11 PLE" / "S12 Attn" → (layer, kind); a single neuron is pointed to `--trace`/`--neuron`. |
| `Variant::new` | A variant is a list of edits; its start is the earliest affected position. |
| `Outcome::dlo` | Δ log-odds of the variant against the unchanged copy **from the same pass**. |
| `read_out` | For every row (stream at the end): the **real** final norm + softcap, then log-odds of target and rival, top 3 and KL to the real output, block by block over the vocabulary. |
| **`Runner::compute`** | Packs many variants into **one** pass (budget 768 rows). Rows 0..T are the clean prompt, then per variant a copy of its positions from `start` to the focus. The stream starts at the checkpoint before layer l; K/V of earlier layers are copied to the rows. Edits are translated: MLP → `patch`, PLE → `ple_scale`, attn → `head_scale`, whole → `Ablation::*At`. Returns per variant its output and that of the copy, optionally with the path through all later layers. |
| `deep_of` | Takes the variant's focus per layer from the recorded rows (stream, attn, MLP, PLE, activations) and maps attention weights back to positions. |
| `head_shifts` | Later heads whose attention from the focus shifts the most (total variation). |
| `neuron_shifts` | Later neurons: (Δ activation) × weight on the lead target − rival (frozen readout). |
| `gaze` | The strongest positions a head looks at. |
| **`run`** | The 6 stages, see section 3. |
| `Report::comp`, `n_id`, `value_at`, `deep_title`, `post_norm`, `where_txt` | Accessors for the report. |
| `terminal`, `html` | Report over all stages; the HTML holds all units, dose curves and paths. |

### `neuron.rs` – one MLP neuron as a computing structure (856 lines, core of --neuron/--trace/--atlas)
| Function | What it does |
|---|---|
| `act64`, `phi`, `act_name` | Activations in f64; φ = act(p)/p with the limit 0.5 at p → 0 (ReLU: 0). |
| `pre_norm` | The norm whose output the MLP reads: the pre-FFN norm, for post-LN (BERT) the attention post-norm. |
| `Read::new` | Effective read vectors g' = w_gate ⊙ γ, u' = w_up ⊙ γ (mean-free for LN), offsets c = w·β + b, raw weight rows exactly as the forward pass unpacks them. Not gated: g' from w_up, no q. |
| `Repertoire::new` | Ellipse c + √d·L·disk (L = Cholesky of the Gram matrix, robust for rank 1/0); act(p)·q on the boundary: 8,192-step angle grid + golden-section refinement. Not gated: the interval via θ ∈ [0, π]. |
| `Repertoire::r_of`, `z_at` | Radius of a point inside the ellipse (pseudo-inverse); input direction z* = √d·Mᵀ G⁺ L w (‖z*‖ = √d). |
| `path`, `ReadPoint`, `Src`, `Op` | Path of a source to a read point (MLP input, attention input, final norm), from the order of write events (attention 3k, MLP 3k+1, PLE 3k+2, layer scalar 3k+2.5; read points: MLP input 3l+0.5, attention input 3l−0.25). The source's own post-norm (Gemma) plus the layer scalars in between; for post-LN the chain of LayerNorms. |
| `forward`, `forward_runtime`, `adjoint`, `runtime_factor` | Path forward with norm scale 1 or with the frozen runtime scales; adjoint path (read vector back to the source); product of the runtime scales. |
| `arrow`, `read_back`, `col_arrows`, `row_arrows` | Static arrow (g'·v, u'·v); read vectors brought back to the source; all columns (W_down, PLE projection) or rows (embedding tables) at once. |
| `static_inputs`, `static_outputs`, `top2` | All inputs of a neuron by source kind (tokens, positions, earlier neurons, PLE, biases; heads only as "runtime only"); all outputs (later neurons, token voice, later heads as an approximation); top k by \|p\| and \|q\|. |
| `parse_part`, `PartSpec` | "S12 MLP" / "S11 PLE" / "S12 Attn" / "S12 #2328" (also "S12 neuron 2328"); shared by probe and trace. |
| Unit tests | Ellipse extremes against random samples on the sphere (2–32 dimensions, all activations), degenerate cases (g' ∥ u', u' = 0, g' = 0, both 0), z* hits the boundary point. |

### `trace.rs` – `--trace` (1,681 lines)
| Function | What it does |
|---|---|
| `source_vector_at`, `source_vector` | Vector of a source at any position and read point (runtime, with the frozen scales recorded at that position). |
| `Reader`, `enumerate_sources`, `Sources` | Read vectors at a read point and position; all sources there split exactly: embedding (token and learned position), every head of every reaching layer with its atoms, biases, **every** earlier neuron (adjoint via W_downᵀ), every PLE dimension. Group checks: Σ atoms = head, Σ neurons of a layer = vector path, Σ PLE dimensions = PLE output. |
| `decompose_neuron` | Runtime decomposition C at the focus (a reader with s·g', s·u' at the MLP input). Checks: Σ source vectors = recorded stream, Σ sources = p and q, a = recorded activation, Σ pairs = a, r ≤ 1, a_min ≤ a ≤ a_max. |
| `static_vs_runtime`, `level01`, `top_abs` | Check static arrow × scales × activation = runtime contribution (adjoint against forward); level 0×1. |
| `content` | **Phase 2:** what an atom carries (head h of layer k at the reader's position reading position j). The reader's vectors go back through post-norm, W_O, the attention weight, V norm, W_V and the pre-norm at j (all frozen) onto the stream before the layer whose values the head reads (shared K/V: the source layer); norm bias and value bias become an offset. Then `enumerate_sources` at j. Checks: Σ parts + offset = atom, Σ source vectors = checkpoint stream, pass with focus j = main pass at j. |
| `Zoom` (`trace_at`, `runtime`, `neuron`, `content`, `children`) | Backward zoom across positions: strongest sources by \|a_c\|, recursively into earlier neurons; heads → atoms → what the read position carries → its sources (neurons at j decomposed at j). Parts at another position come from a recording pass with that focus; passes and decompositions are cached. |
| `rerun_patches`, `eff` | Rerun from layer l via variants (one row each at the focus position, `patch` on the neuron); Δ log-odds, KL, new top. |
| `frame`, `direct_voices`, `auto_start` | Direct voice of neurons (full softmax, frozen final norm); for `--trace auto` a linearized preselection over all neurons at the focus first, then exact voices. |
| `without_source` | a′ without source c, only in this neuron: x − x_c exactly through the norm, p′/q′/a′ in f32 as in the forward pass. |
| `parse_src` | `--scalars-of`: "token", "position", "S14 head 1", "S14 head 1 @16", "S14 #868", "S3 PLE #5", "S4 attn bias", "S4 MLP bias". |
| `run`, `terminal`, `html`, `dump_json` | Flow, report, HTML with the (p,q) plane (whole and zoomed), sources → scalars, pairs, reruns, zoom tree; JSON for `tests/compare_neuron.py` (incl. z* and the phase-2 split of the strongest atom from layer 1 on). With `--atlas`: static rank of the strong sources, check stored arrow × scales × activation = runtime contribution and stored repertoire = recomputed. |

### `inspect.rs` – `--neuron` (512 lines)
| Function | What it does |
|---|---|
| `compute`, `run`, `terminal`, `html_report` | Static report: read vectors, repertoire, inputs by source kind, outputs. Checks: inputs adjoint = forward, outputs read by rows = read by columns. |
| `plane_svg`, `arrow_legend`, `PLANE_CSS` | (p,q) plane as inline SVG: ellipse, color grid of act(p)·q, a_max/a_min, arrows tip to tail; not gated: the curve act(p) with the interval. Also used by trace. |

### `atlas.rs` – `--atlas` (674 lines)
| Function | What it does |
|---|---|
| `build` | Per target layer: the read vectors of all its neurons as rows; GEMM against vocabulary blocks (token inputs and, in the same pass, every neuron's voice), learned positions, biases, W_down of every earlier layer in column blocks (`gemm_nn`), PLE projections. Every tile is reduced at once to the top k per row (inputs) and per column (outputs of earlier neurons, written directly into the mapped file). Repertoire per neuron. Progress on stderr, resumable via done flags in the header; entries are deduplicated, so a repeated layer gives the same file. Self-check per layer (strongest input = on-demand result) and at the end for the outputs. |
| `Atlas::open`, `record`, `Entry`, `TopK` | mmap read access; one record per neuron: repertoire + six lists (inputs by \|p\| / \|q\|, outputs by \|p\| / \|q\|, voice up / down). |

### `circuit.rs` – `--circuit` (circuit search down to single scalar products, 2,627 lines)
| Function | What it does |
|---|---|
| `Geo` | Shape of the model per layer: heads, head size, K/V groups, FFN width, gated or not, up bias, own K/V, V site (not with V = K), window, PLE width; `readers[l]`: the layers whose attention reads the K/V of layer l (shared K/V). `keys(l, p)`: the keys a query at p sees. `cell_atoms`: all atoms of one (layer, position). |
| `Piece` | A piece of the computation at one position: a component (`Attn` = query side of all heads: Q rows, scores, mixing, O columns; `Kv`; `Mlp`; `Ple`), a set of heads / K/V groups / neurons / PLE dimensions, a set of units of one site, a set of keys of QKᵀ or A·V, a set of atoms of one unit, a set of dimensions of one key. `ops` turns it into elementary changes of its cell (`Op`): a site all off, a unit off, atoms of a unit off, a key off, dimensions of a key off. `n_atoms`, `len`, `singles`, `name`. |
| `SetB`, `HeadB`, `CellB`, `State` | What is switched off, per position and layer: per site the units off (or all) and the atoms off per partial unit; per head the keys off and the dimensions off per key. A partial unit becomes a unit off once all its atoms are off. |
| `Builder` | Turns a `CellB` into the `LayerCut` of the forward pass. Fetches the weights of partial units along their atom axis: rows on demand, columns of column-wise sites in one pass over the matrix (`prefetch`). Lengths of the columns of W_O, W_down, PLE-proj for the proxies. Atoms are stored as the shorter list (off or on). |
| `Engine::eval` | Exact output of many candidates (each on top of the current state) in few passes: candidates sorted by start layer, packed up to 768 rows; every candidate is a copy of the positions from its earliest change to the focus (model::Variants), starting from the checkpoint of the base run at the earliest layer of the batch; the cut of every row is the base cut plus the candidate's changes in that cell (`derive` rebuilds only the touched sites and heads). Readout: full vocabulary, KL to the full model, top 3, log-odds of target and rival (`probe::read_out`). The clean row of every batch is compared with the base run. |
| `Engine::run_base_from` | The standalone run after a change at layers ≥ l0: from the checkpoint before l0 (bit-identical to a full run, since the layers before are unchanged – checked: same circuit, same logits as with full reruns). |
| `Engine::commit` | Applies pieces to the state and reruns the model standalone from the earliest layer it touches (`run_base_from`). Kept only if the standalone run passes the criterion, else restored ("refused"). Records |KL batch − KL standalone|. |
| `Engine::accept` | Greedy: blocks that may go alone, cheapest first; per pass up to ~2·log₂ n prefixes of the list, the longest passing prefix is committed. If even the cheapest no longer fits, the next 256 are retested alone; if none fits, the rest is refused. |
| `Engine::sweep` | Dose curves: per group its weakest pieces up to a share τ ∈ {0.3% … 85%} of the group's magnitude off, each group alone (one batch). Then for ~30 trade-offs μ every group takes the τ that maximizes atoms removed − μ·ΔKL (only options that passed alone), all joint choices run exactly, the one that removes the most atoms and passes is committed. |
| `Engine::refine` | One level: sweep (from level 2 on, at most 70% of what is left of the level budget), then halving: open blocks are tested alone, the passing ones go through `accept`, the failing ones are halved (weakest half / strongest half; blocks up to 8 elements – 32 in the final phase – element by element). Stops splitting when the budget is used up (KL ≥ 99% of the limit) or when a round from the third on removes < 5% of the blocks tested; those blocks stay whole. `exhaustive` (final phase) splits down to single elements regardless. |
| `Engine::tighten` | Every kept piece alone against the current state; whatever passes goes through `accept`. Then only the pieces that could go alone but were not taken are retested (cheap rounds); when they find nothing, a full round again. Ends with a full round in which nothing can go (every piece individually necessary), or after one full round (after a level, with the level budget). The first version repeated full rounds; on Gemma 3 each round of ~1,800 tests then removed about one piece. |
| `Engine::dead_ops`, `prune_dead` | Switch off what provably has no effect: a head whose W_O columns are all off, or – zero ablation only – whose A·V products are all off (z = 0 exactly), loses Q rows, scores and mixing; with mean ablation only rules where the output is switched off apply (a neuron with down off, a PLE dimension with projection off), since an input replaced by its mean is not 0; a neuron with gate off, up off (without bias) or down off loses the rest (not gated: down off → up, up off without bias → down); a PLE dimension with gate or projection off loses both, the PLE input of a layer only as a whole (it is normalized over all dimensions); K/V rows that no QKᵀ/A·V product of any reader layer reads (V = K: only if neither reads). Iterated; the base run afterwards must give bit-identical logits. |
| `Engine::expand`, `atom_group`, `block_group`, `site_input` | The groups of the next level with their proxies from a recording pass: heads (order unknown → tested alone), K/V groups, neurons by \|a\|·‖W_down[:,n]‖, PLE dimensions by \|g\|·‖proj[:,e]‖; per head q rows by \|q\|, O columns by \|z\|·‖W_O[:,c]‖, keys of QKᵀ by the attention weight, keys of A·V by a·‖v‖; K/V rows by \|k\|, \|v\|; atoms by \|W[o,c]·x_c\| (column-wise: \|W[r,c]\|·\|x_c\|), dimensions of a score by \|q_e·k_e\|, of a mixing by \|a·v_e\|. Neurons and the gate/projection of PLE dimensions skip level 3 (one of their units off = the whole neuron/dimension off). |
| `cell_kept`, `atoms_kept`, `piece_off` | Counting atoms that stay on; whether a piece is completely off. |
| `complement` | The cut that switches off exactly the circuit's atoms and leaves everything else on. |
| `compute_means` | Mean ablation: runs every reference sequence (recording the inputs). Per position of the analyzed sequence: with `--circuit-ref` the mean over the references at that position (position-wise), at position 0 with `<bos>` its own activations, otherwise the mean over the typical positions of the references (`typical_positions`: not the first, and no attention sink whose residual RMS exceeds 3× the sequence median at some pre-norm). For the inputs of every site, q (after norm and RoPE), k, v and the attention weights (fallback: the mean weight on key 0 / on any other typical key); W·x̄ per unit for the row-wise sites. |
| `random_cut` | Test mode `--circuit-random SEED`: a random cut over every kind of atom (sites all off, units, single atoms, keys, dimensions of keys) at every position and layer; xorshift64*. |
| `run` | Flow, see section 3. |
| `terminal`, `html`, `dump_json`, `layer_rows`, `report_level`, `atom_value` | Report: levels, atoms on per layer with the heads and neurons still alive, the leaves sorted by the KL without each one alone (with the value of single atoms), the circuit alone and everything except it, checks; HTML with a layer × position map; JSON with the cut cell by cell and the logits of the full model, the circuit and its complement (for `tests/compare_circuit.py`). |

### `gpu/` – Metal and CUDA backends for the matrix products
| Function | What it does |
|---|---|
| `kernels.h` | One kernel source for Metal (MSL) and CUDA (NVRTC), macros for the differences. `dequant`: one thread per GGML block / MLX group / 32 plain values, the formulas of quant.rs and tensor.rs line by line (f16 converted bit by bit, MXFP4's 2^(e−128) built bit by bit incl. subnormals). `gemm_nt`: tiled 64 × 64, k in steps of 16, each output summed over k in a fixed order. `gemm_sg` (Metal only): the same on the simdgroup matrix units. |
| `mod.rs`: `describe`, `DqP`, `GemmP`, `plan`, `row_block`, `gemm_plan` | What the kernels need for a matrix (format, byte ranges of weights / MLX scales / biases); the schedule of one product: blocks of up to 4,096 rows (at most 64 MB unpacked), per block one dequant and one GEMM per column range (`parts`, like `Mat::matmul_split`). |
| `register_region`, `locate` | Every memory-mapped file is registered (`gguf::map_file`); a slice of a weight is found in it. |
| `init`, `backend` | `--gpu auto` (Metal on macOS, else CUDA if a driver loads, else CPU), `metal`, `cuda`, `off`; `$APOLLON_GPU`. |
| `selftest` | `--selftest-gpu`: random blocks of all 16 formats and 7 MLX variants (bit widths, group sizes, scale types, with/without biases) with finite scale fields; unpacked values must equal the CPU bit for bit, products (t = 1, 5, 70 against an f64 reference) ≤ 10⁻⁵; the same for the output matrix and the first two layers of the model given; the time of a product with 64, 512, 2,048 rows. |
| `metal.rs` | Pipelines from the kernel source (fast math off). Weights: per tensor a buffer over the pages it lies in, without a copy (a whole file can exceed Metal's largest buffer, e.g. Gemma 4 E4B with 5.3 GB). Inputs / unpacked rows / results in shared scratch buffers that grow. GEMM through `MPSMatrixMultiplication` (cached per shape), the own `gemm_sg` with `APOLLON_METAL_OWN_GEMM=1`. Every product drains an autorelease pool. `APOLLON_GPU_TIMING=1` prints GPU and copy times. |
| `cuda.rs` | Driver API and NVRTC through `libloading` (several library names per platform); compiles the kernels with `--gpu-architecture=compute_XY` of the device; every weight tensor copied once; the same schedule with `gemm_nt`. On an error a product falls back to the CPU with a message. Not yet run on NVIDIA hardware; the kernel source translates to PTX with clang. |

### `server.rs` – `--serve` (local web workspace, 517 lines)
| Function | What it does |
|---|---|
| `serve` | axum server on 127.0.0.1 (`$APOLLON_PORT`, default 8765): the API below, and `web/dist` (the built frontend) for everything else. Loads earlier runs from `output/web-runs/`; a run interrupted by shutdown is marked interrupted. |
| `local_only`, `trusted` | Only the loopback host names are served; requests with a foreign `Origin` are refused. |
| `models`, `scan_models` | `GET /api/models`: the local models (a separate `--list` process, cached for a minute). |
| `create`, `worker`, `execute` | `POST /api/runs`: checks the request (model, prompt, kind ∈ generate / decompose / probe / trace / neuron / atlas / tokens / dump, answer 1–256, a component where needed), stores it, and runs it as a separate apollon process with `--result-json` (one at a time, a semaphore); the structured result and the HTML report land in the run's folder. |
| `cancel`, `remove`, `list`, `detail`, `progress` | Cancel a queued or running run (the process is killed), delete a finished one with its files, list runs, one run with its last progress line. |
| `artifacts`, `artifact`, `artifact_paths` | The files of a run (reports, exports) for download. |
| `save` | A run's `job.json`, written atomically (temporary file + rename). |

### `export.rs` – `--result-json`
| Function | What it does |
|---|---|
| `tokens`, `probabilities`, `result`, `write` | The common part of a structured result (schema version 2): model, tokens, focus, target, mode, output distribution; each analysis adds its own fields (decompose, probe, trace, neuron) before `write` stores the JSON. |

### `readout.rs` – reading vectors as token distributions (226 lines)
| Function | What it does |
|---|---|
| `sum` | Sum in 16 lanes. |
| `top_insert` | Inserts a block's values into a descending top-N list. |
| `Acc::new`, `Acc::update`, `Acc::finish` | Running over the vocabulary blocks: logsumexp, entropy, KL(real ‖ without this voice), log-odds **without** this voice (for every target, without cancellation at p ≈ 1), top 5 of the voice, top 3 without it. |
| **`voices`** | Sends all rows (already through the frozen final norm) block by block through the output matrix, never unpacking more than 256–8192 rows. With a softcap every token logit is multiplied by the fixed factor `z/raw` from row 0 (the whole stream). Optionally returns full logit rows for the changes. |
| `log_odds` | `z_y − ln Σ_{t≠y} e^{z_t}`, exact also at p ≈ 1. |
| `columns` | Selected columns of a matrix (e.g. W_down[:, i]). |
| `softmax` | (logZ, p) of the real logits. |

### `tensor.rs` – matrix directly on the mapped file (326 lines)
| Function | What it does |
|---|---|
| `scalar` | Single F32/F16/BF16 value (for MLX scales/biases). |
| `dot`, `norm` | Dot product in 16 lanes; Euclidean length. |
| `Mat::new`, `Mat::mlx` | Matrix from GGML blocks/F32/F16/BF16 or from MLX (`w = scale·q + bias`, 2–8 bit), sizes checked. |
| `Mat::type_name` | "Q4_K", "MLX-Q4/64" etc. |
| `Mat::row` | **Unpacks one row on demand.** Nothing is inflated in RAM for good. MLX 4-bit with whole-word groups has a fast path (8 values per u32, the same arithmetic: dumps bit-identical, about 30% faster passes on Gemma 3). |
| `Mat::to_vec` | Unpacks the whole matrix (only for vectors and norm weights). |
| `matmul`, `matmul_split`, `matmul_split_cpu`, `rows_times`, `view` | `x·Wᵀ`, optionally split by column ranges (per head): on the GPU if one is active, else on the CPU (macOS with ≥ 64 rows via `matmul_gemm` / Accelerate). `rows_times`: a block of rows (the output matrix in `read_out` / `voices`). `view`: the storage for the GPU. |
| `t_matvec`, `t_matmul` | `Wᵀu` for one or t vectors (back-projection, neuron weights). |
| `col_norms` | Lengths of all columns. |

### `quant.rs` – dequantization (397 lines)
`DType::from_ggml`, `from_st`, `block`, `name`; `ggml_name`; `f16`; `h`; `dequant` (dispatcher) and one function per format:
`q4_0, q4_1, q5_0, q5_1, q8_0, q2_k, q3_k, q4_k (with scale_min_k4), q5_k, q6_k, iq4_nl, iq4_xs, mxfp4`.
The formulas follow llama.cpp; tested against the `gguf` Python package.

### `gguf.rs` – GGUF parser (157 lines)
`map_file` (mmap, leaked as `'static`), `Rd::take/u32/u64/string/val` (reader), `open` (header, metadata, tensor infos, alignment), `Val::u/f/s/a/b` (metadata as number, string, array or bool).

### `tokenizer.rs` – tokenizer (518 lines)
| Function | What it does |
|---|---|
| `split_merge`, `bytes_to_unicode` | Split a merge line; the GPT-2 byte↔unicode table. |
| `base`, `placeholder` | Skeleton. |
| `ensure_prefix` | Prepend `<bos>` unless a prefix is already set. |
| `has_token`, `is_special`, `piece`, `vocab_size` | Queries; `piece` decodes a token into readable text (incomplete bytes as `\xNN`). |
| `add_merges`, `set_specials` | Merge ranks; special tokens sorted by length. |
| `from_gguf` | gpt2 BPE (regex depending on `pre`: qwen2/llama3/gpt2), SentencePiece with scores (llama), SentencePiece BPE with merges (gemma). |
| `from_json` | `tokenizer.json` (model type BPE only), byte-level or Metaspace, added tokens, pre-/post-processor. |
| `from_vocab_merges` | GPT-2/RoBERTa from `vocab.json` + `merges.txt`. |
| `encode` | Split off special tokens literally, the rest via `encode_plain`, then prefix and suffix. |
| `encode_plain` | Byte-level (regex pre-split) or SentencePiece (▁, byte fallback). |
| `bpe` | Repeatedly merge the pair with the lowest merge rank. |

### `blas.rs` – fast primitives (103 lines)
`gemm_nt`, `gemm_nn` (for the atlas), `gemm_nt_ld` (Accelerate `cblas_sgemm`, Rust with rayon otherwise), `exp`, `tanh` (Accelerate `vvexpf`/`vvtanhf`).

### `fmt.rs` – formatting (218 lines)
`Style::paint/bold/dim/yellow/green/red`, `num` (typographic minus), `pct`, `pp` (percentage points), `thousands`, `superscript`, `sci_exp`, `sci`, **`factor`** (e^Δ as "×3.42", "×9·10¹³", "÷250"), `esc_piece`, `tok`, `trunc`, `pad`, `padl`, `esc_html`, `chip`, `prompt_html`, `factor_html`, `CSS`.

### `tests/` – reference comparisons (Python) and unit tests
| Script | Checks |
|---|---|
| `compare_hf.py` | Tokens, residual after every layer and logits at all positions against transformers (random small Qwen2/Qwen3/Llama models + local models). |
| `compare_hf_extra.py` | Gemma 3/4 as small random models with all special parts; GPT-Neo (TinyStories), RoBERTa (unixcoder). |
| `compare_gguf_llama.py` | Converts a tiny Llama to GGUF (permuted q/k, SentencePiece) and compares with transformers. |
| `compare_dequant_qwen_tok.py` | All 16 block formats against `gguf.quants`; the Qwen GGUF tokenizer against the original. |
| `compare_tokenizer.py` | Tokenizers of further families against `tokenizers`/transformers. |
| `compare_mlx.py` | MLX 4-bit unpacking against `mx.dequantize` (needs the Python version the mlx wheel was built for). |
| `compare_ollama.py` | Top-5 probabilities against Ollama's logprobs. |
| `compare_circuit.py` | `--circuit --dump-circuit` against transformers in float64 (zero and mean ablation; for mean the means are computed from transformers alone – Linear inputs by hooks, q·k and a·v over the visible pairs from the replaced attention – and compared with apollon's): every projection recomputed per position with the keep mask over its scalar products (hooks), the eager attention replaced by one that drops the switched-off q_e·k_e and a_j·v_j[e]; norms, RoPE, softmax, activations from transformers. Random small Llama, Qwen3, Gemma 3 and the real Pythia-14m: three random cuts over every kind of atom (`--circuit-random`), the circuit of the search and its complement. Passes if apollon is within max(10⁻⁴, 2 × the deviation of the same reference in float32). |
| `compare_neuron.py` | `--trace … --dump-neuron` against transformers hooks + NumPy (random Llama, Qwen3, Gemma 3; real Pythia-14m): p, q, a, s, x̂, x, every scalar product, contributions of the embedding, every head and every earlier neuron, biases, Gram matrix, extremes (400k-angle grid, 20k random inputs); phase 2 for the strongest atom from layer 1 on: chain, value offset, every source at the read position. Deviations 10⁻⁹ … 1.4·10⁻⁶. |
| `cargo test` | Ellipse extremes against random inputs on the sphere, degenerate cases. |

The forward tests run via `--dump`, `--tokens-only` or `--selftest-dequant`. decompose and probe still have no
tests of their own (only their self-checks); the neuron decomposition has `compare_neuron.py` and the Rust unit
tests.

---

## 3. What the analyses compute, step by step

### `--decompose` (`decompose::run`)
1. **Recording pass** with `atoms` + `checkpoint`: everything at the focus, "head h reads source j", stream before every layer.
2. **Targets:** the predicted token (or `--target`) plus the next 4 of the top 10.
3. **Split the changes:** embedding, per layer attn (sum of heads + bias + post-norm bias), MLP, PLE, possibly a final bias, each times the product of the later layer scalars. **Check:** Σ changes = stream at the end.
4. **Sub-parts:** all heads; per head the k strongest sources (preselected over 64 tokens, the rest as "other N sources"); per MLP the k strongest neurons `a_i·W_down[:,i]` (through the post-norm with fixed scale, the rest as "other N neurons"). **Check:** Σ sources = head.
5. **Voices:** every part → `map_part` with the frozen scale of the final norm → output matrix → (softcap as a fixed factor) → softmax. Per voice: top 5, entropy, KL, direct factor on the odds, which candidates it raises or lowers by how many pp. **Checks:** Σ voice logits = real logits; normalized product of all voices = real distribution (KL).
6. **"via norm":** stream at the end minus the change, only the final norm and softcap recomputed, minus direct.
7. **Total effect:** per change (not per head/neuron) a rerun from its layer from the checkpoint, with the change set to zero (all positions, or only the focus with `--focus-only`). Indirect = total − direct. **Checks:** rerun from the middle without ablation = real pass; for the last change total = direct + via norm.
8. Terminal report and HTML.

### `--probe "S12 MLP"` (`probe::run`)
Units: MLP → neurons (activation), PLE → dimensions of the gated input, attn → heads.
0. Recording pass (`record` + `checkpoint`); with `--contrast` a second pass with the contrast prompt. The rival is its answer, otherwise the second-best token. **Check:** unchanged copy in the variant pass = real pass.
1. **Where:** the whole component off at each position, at all, at all but one. **Distributed** if no single position carries at least half of the total effect: then units are changed at all positions at once in the following stages, otherwise only at the strongest position. **Check:** the same ablation as a single normal run = result from the variant pass.
2. **Brute force:** every active unit set to 0 on its own. In the distributed MLP case the `--probe-max` (1024) neurons with the largest Σ|a|·‖W_down[:,i]‖ are preselected.
3. **Interplay:** the k strongest supports (k = 1, 2, 4, …, all) off together, or only those k on and all other active units off; reports from which k the answer flips.
4. **Dose:** the `--depth` (6) strongest units ×(−2, −1, 0, 0.5, 2, 4).
5. **Path:** the same units off (plus the whole component); recorded in every later layer: logit lens, the voices of attn/MLP/PLE (with the frozen norm scale of the **clean** pass), which later neurons shift the lead target − rival, which heads look elsewhere.
6. **Contrast:** values from the contrast prompt inserted (whole component; the deep units one by one).
Also the direct voice of the top 40 units at the focus.

### `--trace "S12 #2328"` / `--trace auto` (`trace::run`)
1. **Recording pass** with `atoms` + `checkpoint`. With `auto`: all neurons at the focus linearly preselected, the best by their exact direct voice, the `--trace-top` strongest as starts.
2. **Decomposition C** (`decompose_neuron`): level 0 (W[i,j]·x̂_j), level 1 (all sources, reference = stream at the MLP input, frozen pre-norm scale s), pairs (frozen φ), position in the repertoire.
3. **Checks** (above) plus static arrow × runtime scales × activation = runtime contribution for the 8 strongest sources.
4. **Level 0×1** for the strongest sources and for the one asked for with `--scalars-of`.
5. **Reruns** from layer l in one variant pass: original patch (= unchanged copy), neuron to 0 (total; direct from the voice; indirect = total − direct), per strong source a′ exactly from Norm(x − x_c).
6. **Backward zoom** to `--trace-depth`, across positions: heads → atoms → what the read position carries (phase 2) → its sources, recursively.
7. Report, HTML, optional JSON dump; with `--atlas` also the static rank and the atlas checks.

### `--circuit` (`circuit::run`)
1. **Full model** once (recording the norm scales): its distribution is the reference (KL), its top token must stay. With frozen norms (default) every later run uses these scales. Base run with an empty cut: must be bit-identical (with frozen norms this also checks the freezing).
2. **Level 1**: every component at every position (attention query side, K/V, MLP, PLE) as one list; halving over the list (budget 25% of ε), then dead pieces, then `tighten`.
3. **Levels 2–4** (budget 45%, 60%, 85% of ε): recording pass → `expand` → (columns prefetched for level 4) → `refine` (sweep, halving, accept) → dead pieces → `tighten`.
4. **Final phase** (full ε): blocks that stayed whole are split down to single elements (`refine` exhaustive), dead pieces, `tighten` until nothing changes. The last individual tests give, for every leaf, the output without it alone.
5. Recording pass of the circuit for the atom values; standalone run of the **complement**.
6. Terminal report, HTML, optional JSON.
Criterion everywhere: KL(full ‖ candidate) ≤ limit and the same top token. Batched tests must stay below 0.998·limit, the standalone rerun of a commit below the limit.

### `--neuron` / `--atlas`
`--neuron` computes on demand for one neuron: repertoire, `static_inputs` (adjoint, one matrix-vector product per source matrix), `static_outputs` (forward into every later layer). `--atlas` computes the same arrows for all neurons as GEMM and stores only the top k per neuron.

---

## 4. Things noticed while reading (not in the README)

1. **`--probe "Sx Attn"` with `--contrast`:** stage 6 is skipped silently. No contrast values are recorded for heads (`values_c` is empty), and `Edit::Value` does nothing for attn. The contrast then only affects the choice of the rival.
2. **PLE contrast with clean value 0:** inserting is done as a factor `w/w0`. With `w0 = 0` the factor becomes 1, so the contrast value is not inserted, without a message.
3. **`--probe-pos` help text:** it says "default: the most important positions". The code takes **one** position (the strongest) or, in the distributed case, all.
4. **Embedding ablation in `--decompose` (Gemma 4):** only the residual stream before layer 0 is zeroed. `Model::run` still computes the PLE inputs from the token ids and the unchanged embedding, so the total effect "embedding" does not include the path through the PLE.
5. **Head scaling before the post-norm:** in Gemma the attention output passes through its own norm. Setting a head to 0 lets that norm scale the remaining heads up. FINDINGS states the same limit for MLP/PLE; it holds for heads too.
6. **Granularity of the total effect:** it is measured only for whole changes (attn/MLP/PLE of a layer, embedding). Heads, sources and neurons have only their **direct** voice in `--decompose`, and only the preselected k get their own row.
7. **`generate` without KV cache**, and for analyses always at least 12 tokens, also with `--answer 1`. For large models that is 12 full passes before the analysis begins.
8. **No automated tests for decompose and probe:** the Python tests cover the forward pass, tokenizers and dequantization, and `compare_neuron.py` the neuron decomposition. Whether decompose and probe compute correctly is shown only by their self-checks.
9. **With `start`, `run` still computes the embedding** for all rows. The PLE inputs need it; otherwise it is wasted work (small for Gemma 4).
10. **Heads have no static arrow.** What a head writes depends on what it reads. `--neuron` reports their layers as "runtime only". In `--trace` the backward zoom follows a head through the position it reads into the sources at that position (phase 2).
11. **Frozen scale vs. exact:** removing a source from one neuron only can differ clearly from the linearization. Example Gemma 3 S16 #2328 on "2+2": for S16 head 0 the frozen computation predicts Δa = −0.37, exactly through the norm it is −0.63.
12. **Static rank ≠ runtime rank:** arrows are per unit of source activation. The runtime-strongest neuron sources are often not among the static top k, because their weight comes from a large activation.
13. **The repertoire is an outer bound:** with the norm's eps, ‖z‖ < √d, so the ellipse is an upper limit. In Gemma 3 1B it is huge (S16 #2328: a ∈ [−4,529, 4,705]); the real point lies at r = 0.017.
14. **Static token voice without softcap**, head outputs without RoPE and Q/K/V norms (marked as an approximation). Layers with shared K/V (Gemma 4) have no K/V projection of their own ("shared from S…").
15. **BERT/RoBERTa:** static only, through the chain of LayerNorms (post-LN); `--trace` refuses them with a message.
16. **Atlas size:** k = 64 gives 6,224 bytes per neuron, i.e. 1.1 GB for Gemma 3 1B and ≈ 2.7 GB for Gemma 4 E4B. A layer's outputs are complete only once all later layers are built; a partial atlas says so in the trace.
17. **Pythia:** the attention and MLP output biases are bit-identical in the weights, so the static report shows identical arrows; not a computation error.
18. **PLE dimensions** (Gemma 4) as sources: the arrow is per unit of the gated PLE input. That input itself depends on the stream and the token (the PLE token table), so it is a runtime quantity like a neuron activation.
19. **Read point of the attention input:** it has to lie after the previous layer's scalar (3(l−1)+2.5). An earlier version placed it exactly on that time and so dropped the scalar; the phase-2 self-checks caught it on Gemma 4 (the only family with layer scalars). Models without layer scalars were not affected.
20. **Gemma 4 in phase 2** (shared K/V, V norm, PLE at the read position) is covered by the self-checks; the Python reference covers Llama, Qwen3, Gemma 3 and Pythia.
21. **`--circuit`: what "minimal" means.** The result is 1-minimal at the resolution reached: switching off any single leaf alone breaks the criterion. It is one minimal set found greedily, not necessarily the smallest one; a different test order can find a different circuit. With `--circuit-tests` reached, unresolved blocks stay whole and the report says so.
22. **`--circuit`: the edit.** By default (mean ablation) a switched-off product is replaced by its mean over the reference positions; with `--circuit-ablation zero` it is missing from its sum ("does not happen"), as in `--decompose` and `--probe`. Biases stay. With `--circuit-norms live` the norms also rescale what is left (Gemma post-norms); by default they are frozen (item 26). Either way the circuit is a statement about this edit, not about the model on its usual distribution.
23. **`--circuit`: QKᵀ off is not masking.** A key whose score products are all off gets the raw score 0, not −∞: it stays in the softmax. Taking a key out of the softmax is not a matrix product and therefore not an atom.
24. **`--circuit`: budget per level.** Levels 1–4 may use 25 / 45 / 60 / 85% of ε cumulatively, the final phase all of it, the sweep of a level at most 70% of what is left of it. These shares only steer how much the coarse levels may take; the final criterion is always the full ε.
25. **`--circuit`: precision.** Partial units are summed directly over their atoms that stay on. The first version subtracted the switched-off atoms from the full sum; on Pythia-14m the complement (where the switched-off atoms are the large ones) then deviated 2.2·10⁻⁴ from the float64 reference, now 3.2·10⁻⁵.
26. **`--circuit`: frozen norms (default).** With live norms a post-norm (Gemma) divides whatever is left of its input by that rest's own RMS. On Gemma 3 1B "2+2" the live search ends with 20 atoms of one neuron (S24 #5561, values around 0.0006) that the post-FFN norm blows up to full size: alone they give "4" at 92%, but everything *except* them still gives "4" at 100% – the model does not use them, the norm makes them sufficient. With frozen norms every norm keeps the scale of the full model, so a part acts with its real size. `--circuit-norms live` keeps the other semantics.
