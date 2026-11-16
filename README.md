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
| `--decompose` | Splits the prediction into the contribution of every attention, MLP and embedding, down to heads and neurons, with direct and total effect. |
| `--probe` | Switches off every unit of one component (neurons, heads) and follows the effect to the output. |
| `--trace` / `--neuron` / `--atlas` | Takes one neuron apart down to single scalar products, and maps what every neuron can do. |
| `--circuit` | Finds the minimal circuit the prediction needs, down to single scalar products. |
| `--serve` | A local web workspace with 3D, flow, unit and table views. |

Matrix products run on the GPU when there is one (Metal on macOS, CUDA on NVIDIA), otherwise on the CPU.

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

Every analysis prints a report in the terminal and writes an interactive HTML report.
`apollon --help` lists all options.

### Web workspace

```sh
cd web && npm ci && npm run build && cd ..
./target/release/apollon --serve
```

Then open http://127.0.0.1:8765, pick a model, write a prompt, and click on any answer token to take its
prediction apart. The server listens only on your own machine.

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
