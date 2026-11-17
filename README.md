<p align="center">
  <img src="assets/eye.png" alt="The apollon logo: an eye drawn from characters" width="760">
</p>

<h1 align="center">apollon</h1>

<p align="center"><strong>Read what LLMs think. Now.</strong></p>

<p align="center">
  <a href="https://github.com/ryzelo/Apollon/actions/workflows/ci.yml"><img src="https://github.com/ryzelo/Apollon/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT license"></a>
  <img src="https://img.shields.io/badge/made%20with-Rust-orange.svg" alt="Rust">
</p>

apollon opens up a language model while it answers and shows you, for a single prediction, **which parts of the
model produced it and how much each part contributed**. It runs locally, on models you already have, and
trains nothing.

## Why apollon

When a model writes " Paris" after "The capital of France is", millions to billions of multiplications happen in
between. apollon computes the model itself, keeps every intermediate step, and answers:

* **Who decided?** Which layers, attention heads and neurons pushed towards " Paris", and which against it?
* **What if it were missing?** How does the answer change when one part is switched off?
* **What does this neuron do?** Which inputs make it fire, and what does it write into the model?
* **What is really needed?** What is the smallest set of computations that still gives the same answer?

**Exact, not estimated.** There is no second model trained on top (no sparse autoencoder, no probe classifier).
The parts always add up to the real forward pass, and apollon checks this in every run.

## Features

| | |
|---|---|
| `--decompose` | Splits the prediction into the contribution of the embedding and every attention and MLP (and per-layer embedding in Gemma 4), down to single heads, the positions they read, and single neurons. Shows what each part says on its own (direct effect) and what changes when it is removed and the model rerun (total effect). |
| `--probe` | A test bench for one component: switches off every neuron, head or embedding dimension in it, finds where it acts, how many units carry the function, how the answer reacts to stronger or weaker activations, and, with `--contrast`, whether it carries the information that differs between two prompts. |
| `--trace` | Takes one neuron apart at one prediction, down to single scalar products, and follows its inputs backwards through earlier neurons, heads and positions. `--trace auto` picks the most important neurons. |
| `--neuron` | Everything one neuron *can* do, without a prompt: its strongest and weakest possible activation, the input that triggers it, and its static inputs and outputs. |
| `--atlas` | The same for every neuron of a model, saved in one file that `--trace` can use. |
| `--circuit` | Finds a minimal circuit for a prediction: switches off everything the answer does not need, down to single scalar products, with mean or zero ablation and optional reference prompts. Then it expands itself: it searches the next route in what is left without the first, and so on, until the answer is gone without them. It writes a result folder you can open from coarse to fine – components, heads and neurons, then every scalar product with its value, its effect and the exact scalar products it is made of, traced back from the output. |
| `--serve` | A local web workspace for all of the above (see below). |

Every analysis checks itself in every run and reports any violation. Matrix products run on the GPU when there is
one (Metal on macOS, CUDA on NVIDIA), otherwise on the CPU; `--gpu auto|metal|cuda|off` chooses, and
`--selftest-gpu` compares the GPU against the CPU.

## Installation

Requires a recent stable [Rust](https://rustup.rs) toolchain (1.80 or newer).

```sh
git clone https://github.com/ryzelo/Apollon.git
cd Apollon
cargo build --release
```

The binary is `target/release/apollon`. It is built for your local CPU; for a portable binary, remove the
`target-cpu=native` line in `.cargo/config.toml`.

apollon brings no weights. It reads models you already have:

* Ollama models (`~/.ollama/models` or `$OLLAMA_MODELS`)
* the Hugging Face cache (`~/.cache/huggingface/hub` or `$HF_HUB_CACHE`)
* any `.gguf` file, or any folder with `config.json`, `*.safetensors` and `tokenizer.json`

`apollon --list` shows which of your local models are supported.

## Quick start

```sh
# which parts of the model produce " Paris"?
apollon -m EleutherAI/pythia-70m --decompose "The capital of France is"

# test bench for one MLP, against a contrast prompt ("Was ist 2+2?" = "What is 2+2?")
apollon -m mlx-community/gemma-3-1b-it-4bit --chat --answer 7 --probe "S16 MLP" --contrast "Was ist 2+3?" "Was ist 2+2?"

# one neuron down to single scalar products
apollon -m mlx-community/gemma-3-1b-it-4bit --chat --answer 7 --trace "S16 #2328" "Was ist 2+2?"

# the minimal circuit behind a prediction
apollon -m EleutherAI/pythia-14m --pos -1 --circuit "Monday, Tuesday, Wednesday,"
```

Every analysis prints a report in the terminal and writes an interactive HTML report. `apollon --help` lists all
options; the most important ones:

| Option | Meaning |
|---|---|
| `-m MODEL` | Ollama name (`qwen2.5:3b`), Hugging Face name (`EleutherAI/pythia-70m`), `.gguf` file or model folder |
| `--chat` | wrap the prompt in the model's chat format (Gemma 3/4, ChatML) |
| `--answer N` | analyze the N-th token of the model's answer (`--answer 1,6` for several) |
| `--pos N` | analyze the prediction at prompt position N instead (negative counts from the end) |
| `--target " Paris"` | the token to explain (default: the predicted one) |
| `--html NAME` | base name of the HTML reports |
| `--result-json FILE` | export the analysis as structured JSON |
| `--dump FILE` | write the tokens, the residual stream after every layer and the logits |

Without an analysis flag, apollon just generates the answer.

### Web workspace

```sh
cd web && npm ci && npm run build && cd ..
./target/release/apollon --serve
```

Then open http://127.0.0.1:8765, pick a model, write a prompt, and click on any answer token to take its
prediction apart. From there, one click probes a component, traces a neuron or inserts a contrast prompt. Every
result can be viewed as a 3D space, a flow, a unit view or a table; runs can be compared side by side, parts
pinned across runs, and every view shared as a link. Runs and reports are kept between sessions. The server
listens only on your own machine.

## Supported models

| Family | GGUF (Ollama) | safetensors / MLX 4-bit |
|---|:---:|:---:|
| Llama, Mistral, Qwen2, Qwen3 | ✓ | ✓ |
| Gemma 4 | ✓ | ✓ |
| Gemma 3 | – | ✓ |
| GPT-NeoX (Pythia), GPT-Neo | – | ✓ |
| BERT / RoBERTa (encoders) | – | forward pass only |

Quantized weights (Q4_0 … Q8_0, K-quants, IQ4, MXFP4, MLX 2–8 bit) are read directly from the file.
Mixture-of-experts, Gemma 2, Phi, GPT-2 and Falcon are not supported yet.

## Documentation

* **[User guide](docs/GUIDE.md)**: every analysis in detail, including the math, all flags, self-checks, limits,
  GPU backends and run times.
* **[Findings](docs/FINDINGS.md)**: what apollon showed about how Gemma 3 1B and Gemma 4 E4B answer "2+2".
* **[Code reference](docs/FUNCTIONS.md)**: what every function in the code does, step by step.

## Testing

```sh
cargo test --release
```

apollon's forward pass, tokenizers, dequantization, neuron traces and circuits are compared against Hugging
Face transformers, `gguf`, MLX and Ollama, with deviations between 10⁻⁹ and 10⁻⁵. These comparison scripts live
in [`tests/`](tests) and need the reference models locally; see [Tested](docs/GUIDE.md#tested) in the guide.

## Contributing

Issues and pull requests are welcome. Please run `cargo test --release` before opening a pull request, and
include the model and the command when you report a wrong result.

## License

[MIT](LICENSE) © 2026 Xerororo and Ryzelo. Model weights are not part of this repository; their own licenses
apply.
