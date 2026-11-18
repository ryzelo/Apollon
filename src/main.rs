//! apollon – what does every part of a transformer contribute to one prediction?
//!
//! Analyses of a single prompt, exact for this one forward pass:
//! --decompose  every change of the residual stream as a token voice, with its total effect by rerun
//! --probe      brute-force test bench for one component (neurons, PLE dimensions or heads)
//! --trace      one neuron down to single scalar products (--neuron, --atlas: static, no prompt)
//! --circuit    exact circuit search down to single scalar products
//! --serve      the local web workspace
mod atlas;
mod export;
mod server;
mod bias;
mod blas;
mod circuit;
mod decompose;
mod fmt;
mod gguf;
mod gpu;
mod inspect;
mod model;
mod neuron;
mod probe;
mod quant;
mod readout;
mod tensor;
mod tokenizer;
mod trace;

use fmt::{esc_piece, num, tok, Style};
use model::{Model, RunOpts};
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::time::Instant;

const HELP: &str = "apollon – what does every part of a transformer contribute to one prediction?

Usage:
  apollon -m MODEL [options] --decompose PROMPT
  apollon -m MODEL [options] --probe \"S12 MLP\" PROMPT
  apollon -m MODEL [options] --trace \"S12 #2328\" PROMPT   (or --trace auto)
  apollon -m MODEL [options] --circuit PROMPT          circuit down to single scalar products
  apollon -m MODEL --neuron \"S12 #2328\"  static report of one neuron, no prompt needed
  apollon -m MODEL --atlas FILE          build the static atlas of all neurons, no prompt needed
  apollon -m MODEL PROMPT                only generate the answer (greedy)
  apollon --list                         show the models found locally

MODEL can be:
  qwen2.5:3b                             Ollama name (~/.ollama/models or $OLLAMA_MODELS)
  EleutherAI/pythia-70m                  Hugging Face cache (~/.cache/huggingface/hub)
  /path/model.gguf | /path/hf-folder     file or folder (config.json + *.safetensors + tokenizer.json)

Analyses:
  --decompose      decompose the pass into token voices: every change of the stream (embedding, every
                   attention, every MLP, every per-layer embedding) goes through the final norm (scale of
                   this pass) and the output matrix on its own -> softmax = its direct voice; also heads,
                   source positions and neurons. Total effect: every change set to zero and the model
                   rerun from its layer; indirect = total - direct. Overview in the terminal, everything
                   expandable as HTML.
  --focus-only     with --decompose: zero changes only at the focus token instead of all positions
  --top K          with --decompose: sources per head / neurons per MLP shown (default 8)
  --probe PART     test bench for one component, e.g. \"S12 MLP\", \"S11 PLE\", \"S12 Attn\" (several with ;):
                   change every unit inside (neuron / PLE dimension / head) by brute force and rerun the
                   model from that layer – many variants in one pass. Measures the output and the partial
                   softmax of all later layers: where the component acts, which units matter, interplay,
                   dose, path through later layers.
  --probe-pos N    with --probe: probe only at position N (default: the most important positions)
  --depth N        with --probe: number of units examined in detail (dose, path; default 6)
  --contrast T     with --probe: insert values from prompt T, e.g. \"What is 2+3?\". The answer of the main
                   prompt is reused with the same substitution (2+2 -> 2+3); the token count must match.
  --probe-max N    with --probe, distributed mode, MLP: at most N neurons (preselected; default 1024)
  --trace NEURON   one MLP neuron at this prediction, exact down to single scalar products: p = Σ_j W[i,j]·x̂_j
                   (level 0); p = Σ_c p_c over all sources – embedding, every head and the position it reads,
                   every earlier neuron, PLE dimensions, biases (level 1, frozen pre-norm scale); a = φ·Σ p_c·q_c'
                   (pairs). Removal effects by rerun (the neuron to 0; a source removed only from this neuron,
                   recomputed exactly through the norm) and a backward zoom through the strongest sources – across
                   positions: what a head reads at position j is split into the sources at j.
                   NEURON is like \"S12 #2328\"; several with ;. \"auto\": the strongest neurons by direct voice.
  --trace-depth N  with --trace: depth of the backward zoom (default 2)
  --trace-top N    with --trace: start neurons for auto and sources per neuron in the zoom (default 3)
  --scalars N      with --trace: scalar products shown (default 10)
  --scalars-of SRC with --trace: level 0×1 for this source, e.g. \"S14 head 1\", \"S14 head 1 @16\", \"S14 #868\", token
  --dump-neuron F  with --trace: write the first neuron's decomposition as JSON (for tests/compare_neuron.py)
  --circuit        circuit search down to single scalar products: every matrix product is a sum of atoms
                   W[o,c]·x_c (Q, K, V, O, gate, up, down, PLE) or q_e·k_e and a_j·v_j[e] (QKᵀ, A·V), one per
                   position. Atoms are switched off – missing from their sum – coarse to fine (components per
                   position → heads/neurons → dimensions/keys → single atoms) until only what the prediction
                   needs is left: KL(full ‖ circuit) ≤ ε and the same top token. Every decision is an exact
                   rerun; at the end every leaf is individually necessary. Terminal report, HTML.
  --circuit-kl E   with --circuit: tolerance ε in nats (default 0.1)
  --circuit-depth N  with --circuit: finest level, 1 = components, 2 = heads/neurons, 3 = dimensions/keys,
                   4 = single atoms (default 4)
  --circuit-tests N  with --circuit: stop refining after N tests; unresolved blocks stay whole (default 2000000)
  --circuit-ablation A  with --circuit: mean (default) = a switched-off atom is replaced by its mean over
                   the reference positions (W[o,c]·x̄_c; for scores and mixing the mean of q_e·k_e and
                   a_j·v_j[e] over all query/key pairs); zero = it is missing from its sum
  --circuit-ref T  with --circuit and mean ablation: reference prompts for the means, several with ;
                   (wrapped like the prompt with --chat, plus their own greedy answer). Default: the
                   analyzed sequence itself
  --circuit-norms M  with --circuit: frozen (default) = every norm keeps the scale it has in the full model at
                   that position, so switching parts off cannot inflate what is left; live = norms recompute
                   their scale (a post-norm then blows up any small rest to full size)
  --bias NEURON    bias rows of a neuron (\"S18 #1267\"; several with ;): what one unit of its activation gives every
                   later neuron (Δp, Δq), every later Q/K/V projection and every token's logit; what every
                   token, earlier neuron, PLE dimension and bias gives it; the range its activation allows; and
                   at every position of the prompt what it actually receives and gives – the received parts add up
                   to its p and q exactly (checked). Full rows as float32 files in <html name>_bias/S<l>_<i>/
  --bias-out DIR   with --bias: the folder (default <html name>_bias)
  --bias-top N     entries listed by name per list (default 24)
  --circuit-show N with --circuit: leaves listed in the terminal (default 25)
  --circuit-paths N  with --circuit: self-expansion – after the minimal circuit, search the next one in what is
                   left without it, and so on, until the rest no longer gives the top token (every route is
                   found) or N paths (default 8; 1 = one minimal circuit). Every leaf, path, unit and
                   component is then switched off alone against the union of all paths
  --circuit-from F  with --circuit: no search – load the circuit from a --dump-circuit file (same prompt and
                   settings), test every atom alone against it, write the reports and the folder
  --circuit-bias N  with --circuit: also the bias rows (see --bias) of the N neurons of the circuit that carry
                   the most flow (all: every neuron of the circuit), into <folder>/bias/
  --circuit-out DIR  with --circuit: the result folder (default <html name>_circuit; none = no folder):
                   circuit.json, trace.json and layers/S…/p…/<attention|kv|mlp|ple>/<unit>.json with every
                   scalar product that is on, its value, its effect and its exact sources
  --dump-circuit F with --circuit: write the circuit and the logits as JSON (for tests/compare_circuit.py)
  --neuron NEURON  static report without a prompt: the repertoire (all activations the neuron can reach:
                   (p, q) lies in a filled ellipse), all inputs by source kind and all outputs (later neurons,
                   token voice, later heads)
  --atlas FILE     static arrows of all neurons, top k inputs and outputs per neuron, in a memory-mappable
                   file (format APLA); resumable. Without a prompt it is built; with --trace it is read.
  --atlas-k K      entries per list in the atlas (default 64)
  --atlas-layers A..B  build only the target layers A to B−1 (outputs of a layer are complete when all
                   later layers are built)

Options:
  --chat           wrap the prompt in the model's chat format (Gemma 3/4, ChatML)
  --answer LIST    which answer tokens to analyze (greedy, 1 = first), e.g. --answer 1,6 (default 1)
  --pos N          analyze the prediction at prompt token N instead (negative counts from the end)
  --target TOKEN   target token (default: the predicted token), e.g. --target \" Paris\"
  --html NAME      base name of the HTML reports (default apollon)
  --gpu MODE       matrix products on the GPU: auto (default: Metal on macOS, CUDA if an NVIDIA driver
                   is found, else CPU), metal, cuda, off. Also via $APOLLON_GPU. CUDA needs no build
                   option: the driver and NVRTC are loaded at runtime
  --color / --no-color
  --result-json F  export one runtime analysis as structured JSON
  --models-json    list discovered models as JSON
  --serve          start the local web application
  --threads N

For the reference tests:
  --dump FILE      write tokens, the residual after the embedding and after every layer, and the logits
                   at all positions (binary, see tests/)
  --tokens-only    print only the token ids as JSON

Architectures: llama, mistral, qwen2, qwen3, gemma4 (GGUF); also gemma3, gpt_neox, gpt_neo,
bert/roberta (safetensors, also MLX 4-bit). GGUF weights stay quantized in the file and are
dequantized row by row.";

#[derive(Default)]
struct Args {
    model: Option<String>,
    prompt: Vec<String>,
    pos: Option<i64>,
    target: Option<String>,
    top: Option<usize>,
    color: Option<bool>,
    threads: Option<usize>,
    list: bool,
    models_json: bool,
    serve: bool,
    result_json: Option<PathBuf>,
    dump: Option<String>,
    tokens_only: bool,
    chat: bool,
    answer: Vec<usize>,
    decompose: bool,
    focus_only: bool,
    html: Option<String>,
    probe: Vec<String>,
    probe_pos: Option<usize>,
    depth: Option<usize>,
    contrast: Option<String>,
    probe_max: Option<usize>,
    trace: Vec<String>,
    trace_depth: Option<usize>,
    trace_top: Option<usize>,
    scalars: Option<usize>,
    dump_neuron: Option<String>,
    scalars_of: Option<String>,
    neuron: Vec<String>,
    atlas: Option<String>,
    atlas_k: Option<usize>,
    atlas_layers: Option<(usize, usize)>,
    circuit: bool,
    circuit_kl: Option<f32>,
    circuit_depth: Option<usize>,
    circuit_tests: Option<usize>,
    circuit_show: Option<usize>,
    dump_circuit: Option<String>,
    circuit_random: Option<u64>,
    circuit_norms: Option<String>,
    circuit_ablation: Option<String>,
    circuit_ref: Vec<String>,
    circuit_paths: Option<usize>,
    circuit_out: Option<String>,
    circuit_from: Option<String>,
    circuit_bias: Option<String>,
    bias: Vec<String>,
    bias_out: Option<String>,
    bias_top: Option<usize>,
    gpu: Option<String>,
    selftest_gpu: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args::default();
    let mut it = std::env::args().skip(1);
    while let Some(x) = it.next() {
        let mut val = |n: &str| it.next().ok_or(format!("{n} needs a value"));
        let number = |s: String, n: &str| s.parse::<usize>().map_err(|_| format!("{n} expects a number"));
        match x.as_str() {
            "-h" | "--help" => {
                println!("{HELP}");
                std::process::exit(0);
            }
            "-m" | "--model" => a.model = Some(val(&x)?),
            "-p" | "--prompt" => a.prompt.push(val(&x)?),
            "--pos" => a.pos = Some(val(&x)?.parse().map_err(|_| "--pos expects a number")?),
            "--target" => a.target = Some(val(&x)?),
            "--top" => a.top = Some(number(val(&x)?, "--top")?),
            "--color" => a.color = Some(true),
            "--no-color" => a.color = Some(false),
            "--threads" => a.threads = Some(number(val(&x)?, "--threads")?),
            "--gpu" => a.gpu = Some(val(&x)?),
            // internal: GPU against CPU for every weight format and every matrix of the model
            "--selftest-gpu" => a.selftest_gpu = true,
            "--list" => a.list = true,
            "--models-json" => a.models_json = true,
            "--serve" => a.serve = true,
            "--result-json" => a.result_json = Some(PathBuf::from(val(&x)?)),
            "--dump" => a.dump = Some(val(&x)?),
            "--tokens-only" => a.tokens_only = true,
            "--chat" => a.chat = true,
            "--decompose" => a.decompose = true,
            "--focus-only" => a.focus_only = true,
            "--probe" => a.probe.extend(val(&x)?.split(';').map(|s| s.trim().to_string()).filter(|s| !s.is_empty())),
            "--probe-pos" => a.probe_pos = Some(number(val(&x)?, "--probe-pos")?),
            "--depth" => a.depth = Some(number(val(&x)?, "--depth")?),
            "--contrast" => a.contrast = Some(val(&x)?),
            "--probe-max" => a.probe_max = Some(number(val(&x)?, "--probe-max")?),
            "--trace" => a.trace.extend(val(&x)?.split(';').map(|s| s.trim().to_string()).filter(|s| !s.is_empty())),
            "--trace-depth" => a.trace_depth = Some(number(val(&x)?, "--trace-depth")?),
            "--trace-top" => a.trace_top = Some(number(val(&x)?, "--trace-top")?),
            "--scalars" => a.scalars = Some(number(val(&x)?, "--scalars")?),
            "--dump-neuron" => a.dump_neuron = Some(val(&x)?),
            "--scalars-of" => a.scalars_of = Some(val(&x)?),
            "--neuron" => a.neuron.extend(val(&x)?.split(';').map(|s| s.trim().to_string()).filter(|s| !s.is_empty())),
            "--atlas" => a.atlas = Some(val(&x)?),
            "--atlas-k" => a.atlas_k = Some(number(val(&x)?, "--atlas-k")?),
            "--atlas-layers" => {
                let v = val(&x)?;
                let (lo, hi) = v.split_once("..").ok_or("--atlas-layers expects a range like 0..8")?;
                a.atlas_layers = Some((number(lo.to_string(), "--atlas-layers")?, number(hi.to_string(), "--atlas-layers")?));
            }
            "--circuit" => a.circuit = true,
            "--circuit-kl" => a.circuit_kl = Some(val(&x)?.parse().map_err(|_| "--circuit-kl expects a number")?),
            "--circuit-depth" => a.circuit_depth = Some(number(val(&x)?, "--circuit-depth")?),
            "--circuit-tests" => a.circuit_tests = Some(number(val(&x)?, "--circuit-tests")?),
            "--circuit-show" => a.circuit_show = Some(number(val(&x)?, "--circuit-show")?),
            "--dump-circuit" => a.dump_circuit = Some(val(&x)?),
            "--circuit-ablation" => {
                let v = val(&x)?;
                if v != "zero" && v != "mean" {
                    return Err("--circuit-ablation expects zero or mean".into());
                }
                a.circuit_ablation = Some(v);
            }
            "--circuit-paths" => a.circuit_paths = Some(number(val(&x)?, "--circuit-paths")?),
            "--circuit-out" => a.circuit_out = Some(val(&x)?),
            "--circuit-from" => a.circuit_from = Some(val(&x)?),
            "--circuit-bias" => a.circuit_bias = Some(val(&x)?),
            "--bias" => a.bias.extend(val(&x)?.split(';').map(|s| s.trim().to_string()).filter(|s| !s.is_empty())),
            "--bias-out" => a.bias_out = Some(val(&x)?),
            "--bias-top" => a.bias_top = Some(number(val(&x)?, "--bias-top")?),
            "--circuit-ref" => a.circuit_ref.extend(val(&x)?.split(';').map(|s| s.trim().to_string()).filter(|s| !s.is_empty())),
            "--circuit-norms" => {
                let v = val(&x)?;
                if v != "frozen" && v != "live" {
                    return Err("--circuit-norms expects frozen or live".into());
                }
                a.circuit_norms = Some(v);
            }
            // internal, for tests/compare_circuit.py: a random cut instead of the search
            "--circuit-random" => a.circuit_random = Some(val(&x)?.parse().map_err(|_| "--circuit-random expects a number")?),
            "--html" => a.html = Some(val(&x)?.trim_end_matches(".html").to_string()),
            "--answer" => {
                a.answer = val(&x)?.split(',').map(|s| s.trim().parse::<usize>()).collect::<Result<_, _>>().map_err(|_| "--answer expects numbers, e.g. 1,6")?;
                if a.answer.contains(&0) {
                    return Err("--answer counts from 1".into());
                }
            }
            "--selftest-dequant" => {
                // internal, for the tests: TYPE-ID, input (raw blocks), output (f32)
                let (t, i, o) = (val(&x)?, it.next().ok_or("input missing")?, it.next().ok_or("output missing")?);
                let dt = quant::DType::from_ggml(t.parse().map_err(|_| "type id")?).ok_or("unknown type")?;
                let src = std::fs::read(&i).map_err(|e| e.to_string())?;
                let (be, bb) = dt.block();
                let mut dst = vec![0f32; src.len() / bb * be];
                quant::dequant(dt, &src, &mut dst);
                std::fs::write(&o, dst.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).map_err(|e| e.to_string())?;
                std::process::exit(0);
            }
            s if s.starts_with('-') && s.len() > 1 => return Err(format!("unknown option {s} (help: --help)")),
            _ => a.prompt.push(x),
        }
    }
    if a.result_json.is_some() {
        let modes = usize::from(a.decompose) + usize::from(a.circuit) + usize::from(!a.probe.is_empty()) + usize::from(!a.trace.is_empty()) + usize::from(!a.bias.is_empty());
        if modes > 1 || a.answer.len() > 1 || a.neuron.len() > 1 || (modes > 0 && !a.neuron.is_empty()) || a.dump.is_some() || a.tokens_only || (a.atlas.is_some() && a.trace.is_empty()) {
            return Err("--result-json accepts one runtime analysis and one answer position per invocation".into());
        }
    }
    Ok(a)
}

// ---------------------------------------------------------------------- finding models

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")).unwrap_or_else(|_| ".".into()))
}
fn ollama_root() -> PathBuf {
    std::env::var("OLLAMA_MODELS").map(PathBuf::from).unwrap_or_else(|_| home().join(".ollama/models"))
}
fn hf_root() -> PathBuf {
    std::env::var("HF_HUB_CACHE").map(PathBuf::from).unwrap_or_else(|_| home().join(".cache/huggingface/hub"))
}

enum Src {
    Gguf(PathBuf),
    Hf(model::HfFiles),
}

fn ollama_blob(name: &str) -> Option<PathBuf> {
    let (n, tag) = name.split_once(':').unwrap_or((name, "latest"));
    let root = ollama_root();
    let man = if n.contains('/') {
        root.join("manifests/registry.ollama.ai").join(n).join(tag)
    } else {
        root.join("manifests/registry.ollama.ai/library").join(n).join(tag)
    };
    let j: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(man).ok()?).ok()?;
    let layer = j["layers"].as_array()?.iter().find(|l| l["mediaType"] == "application/vnd.ollama.image.model")?;
    let blob = root.join("blobs").join(layer["digest"].as_str()?.replace(':', "-"));
    blob.exists().then_some(blob)
}

/// Collect the files of a model – in the HF cache across all snapshots
/// (weights and tokenizer sometimes live in different revisions).
fn hf_files_in(dirs: &[PathBuf]) -> Result<model::HfFiles, String> {
    let mut by_name: std::collections::HashMap<String, PathBuf> = std::collections::HashMap::new();
    let mut weights: Vec<PathBuf> = Vec::new();
    for d in dirs {
        let mut here: Vec<PathBuf> = Vec::new();
        for e in std::fs::read_dir(d).map_err(|e| format!("{}: {e}", d.display()))?.flatten() {
            let p = e.path();
            let n = e.file_name().to_string_lossy().to_string();
            if n.ends_with(".safetensors") {
                here.push(p);
            } else {
                by_name.entry(n).or_insert(p);
            }
        }
        if weights.is_empty() && !here.is_empty() {
            here.sort();
            weights = here;
        }
    }
    let get = |n: &str| by_name.get(n).cloned();
    let config = get("config.json").ok_or("config.json missing")?;
    if weights.is_empty() {
        return Err("no .safetensors weights (not downloaded, or only .bin)".into());
    }
    Ok(model::HfFiles {
        config,
        weights,
        tokenizer_json: get("tokenizer.json"),
        tokenizer_config: get("tokenizer_config.json"),
        vocab_json: get("vocab.json"),
        merges_txt: get("merges.txt"),
        special_tokens: get("special_tokens_map.json"),
    })
}

fn hf_snapshots(name: &str) -> Vec<PathBuf> {
    let snaps = hf_root().join(format!("models--{}", name.replace('/', "--"))).join("snapshots");
    let mut v: Vec<(std::time::SystemTime, PathBuf)> = std::fs::read_dir(snaps)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| (e.metadata().and_then(|m| m.modified()).unwrap_or(std::time::UNIX_EPOCH), e.path()))
        .collect();
    v.sort_by(|a, b| b.0.cmp(&a.0)); // newest first
    v.into_iter().map(|x| x.1).collect()
}

fn resolve(spec: &str) -> Result<Src, String> {
    let p = Path::new(spec);
    if p.is_file() {
        return Ok(Src::Gguf(p.to_path_buf()));
    }
    if p.is_dir() {
        return Ok(Src::Hf(hf_files_in(&[p.to_path_buf()])?));
    }
    if let Some(b) = ollama_blob(spec) {
        return Ok(Src::Gguf(b));
    }
    if spec.contains('/') {
        let snaps = hf_snapshots(spec);
        if !snaps.is_empty() {
            return hf_files_in(&snaps).map(Src::Hf).map_err(|e| format!("{spec}: {e}"));
        }
    }
    Err(format!("model '{spec}' not found (apollon --list shows what is available)"))
}

fn load(spec: &str) -> Result<Model, String> {
    match resolve(spec)? {
        Src::Gguf(p) => model::load_gguf(&p, spec),
        Src::Hf(f) => model::load_hf(&f, spec),
    }
}

fn list(as_json: bool) {
    let mut models = Vec::new();
    if !as_json { println!("Ollama ({}):", ollama_root().display()); }
    let man = ollama_root().join("manifests/registry.ollama.ai");
    let mut found = Vec::new();
    if let Ok(ns) = std::fs::read_dir(&man) {
        for n in ns.flatten() {
            for m in std::fs::read_dir(n.path()).into_iter().flatten().flatten() {
                for t in std::fs::read_dir(m.path()).into_iter().flatten().flatten() {
                    let ns_name = n.file_name().to_string_lossy().to_string();
                    let base = if ns_name == "library" { String::new() } else { format!("{ns_name}/") };
                    found.push(format!("{base}{}:{}", m.file_name().to_string_lossy(), t.file_name().to_string_lossy()));
                }
            }
        }
    }
    found.sort();
    for f in &found {
        let arch = ollama_blob(f).and_then(|b| gguf::open(&b).ok()).and_then(|g| g.meta.get("general.architecture").and_then(|v| v.s().map(String::from)));
        let ok = matches!(arch.as_deref(), Some("llama" | "qwen2" | "qwen3" | "mistral" | "gemma4"));
        models.push(serde_json::json!({"id":f,"name":f,"supported":ok,"architecture":arch}));
        if !as_json { println!("  {} {f}  ({})", if ok { "✓" } else { "✗" }, arch.unwrap_or("?".into())); }
    }
    if !as_json { println!("Hugging Face ({}):", hf_root().display()); }
    let mut hs: Vec<String> = std::fs::read_dir(hf_root())
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| e.file_name().to_string_lossy().strip_prefix("models--").map(|s| s.replacen("--", "/", 1)))
        .collect();
    hs.sort();
    for n in hs {
        let files = hf_files_in(&hf_snapshots(&n));
        let why = match &files {
            Err(e) => e.clone(),
            Ok(f) => std::fs::read_to_string(&f.config)
                .ok()
                .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
                .map(|c| {
                    let t = if c["text_config"].is_object() { &c["text_config"] } else { &c };
                    t["model_type"].as_str().or(c["model_type"].as_str()).unwrap_or("?").to_string()
                })
                .unwrap_or("?".into()),
        };
        let fam = why.trim_end_matches("_text");
        let ok = files.is_ok() && matches!(fam, "llama" | "mistral" | "qwen2" | "qwen3" | "gpt_neox" | "gpt_neo" | "gemma3" | "gemma4" | "roberta" | "bert" | "xlm-roberta");
        models.push(serde_json::json!({"id":n,"name":n,"supported":ok,"architecture":why}));
        if !as_json { println!("  {} {n}  ({why})", if ok { "✓" } else { "✗" }); }
    }
    if as_json { println!("{}", serde_json::json!({"models":models})); } else { println!("✓ = architecture is supported"); }
}

/// Short description of the architecture (identical layers grouped)
fn describe(m: &Model) -> String {
    let b = &m.blocks;
    let mut kinds: Vec<(String, usize)> = Vec::new();
    for x in b {
        let a = &x.attn;
        let k = format!(
            "{}{} heads×{} ({} KV{}), FFN {}",
            if x.label.is_empty() { String::new() } else { format!("{}: ", x.label) },
            a.n_head,
            a.hd,
            a.n_kv,
            if a.kv_from.is_some() { ", shared" } else { "" },
            x.mlp.up.rows
        );
        match kinds.iter_mut().find(|(s, _)| *s == k) {
            Some(e) => e.1 += 1,
            None => kinds.push((k, 1)),
        }
    }
    let parts: Vec<String> = kinds.iter().map(|(k, n)| format!("{n}× [{k}]")).collect();
    let mut extra = Vec::new();
    if m.ple.is_some() {
        extra.push("per-layer embedding".to_string());
    }
    if let Some(c) = m.logit_cap {
        extra.push(format!("logit softcap {c}"));
    }
    if b.iter().any(|x| x.scalar.is_some()) {
        extra.push("layer scalars".into());
    }
    if !m.has_head {
        extra.push("encoder".into());
    }
    format!(
        "{} · {} · {} ({}) · {} layers, d={}: {}{} · tokenizer: {}",
        m.name,
        m.arch,
        m.format,
        m.weight_types,
        b.len(),
        m.d,
        parts.join(" + "),
        if extra.is_empty() { String::new() } else { format!(" · {}", extra.join(", ")) },
        m.tok.info
    )
}

fn term_width() -> usize {
    #[cfg(unix)]
    unsafe {
        let mut ws: libc::winsize = std::mem::zeroed();
        if libc::ioctl(1, libc::TIOCGWINSZ, &mut ws) == 0 && ws.ws_col > 20 {
            return ws.ws_col as usize;
        }
    }
    std::env::var("COLUMNS").ok().and_then(|s| s.parse().ok()).unwrap_or(140)
}

fn chat_wrap(m: &Model, text: &str) -> Result<String, String> {
    if m.tok.has_token("<|turn>") {
        Ok(format!("<|turn>user\n{text}<turn|>\n<|turn>model\n"))
    } else if m.tok.has_token("<start_of_turn>") {
        Ok(format!("<start_of_turn>user\n{text}<end_of_turn>\n<start_of_turn>model\n"))
    } else if m.tok.has_token("<|im_start|>") {
        Ok(format!("<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"))
    } else {
        Err("--chat: no known chat format in this model's tokenizer".into())
    }
}

// ---------------------------------------------------------------------- contrast prompt

/// Replace every occurrence of `old` in `text` by `new` (the first `ctx` characters are context only);
/// spaces between the characters in the text are skipped ("2+2" also matches "2 + 2").
fn replace_loose(text: &str, old: &str, new: &str, ctx: usize) -> String {
    let t: Vec<char> = text.chars().collect();
    let pat: Vec<char> = old.chars().filter(|c| *c != ' ').collect();
    let new_c: Vec<char> = new.chars().collect();
    let new_part: String = new_c[ctx.min(new_c.len())..].iter().collect();
    if pat.is_empty() {
        return text.to_string();
    }
    let mut out = String::new();
    let mut i = 0;
    while i < t.len() {
        // try to match the pattern from i (skipping spaces in the text between pattern characters)
        let mut j = i;
        let mut k = 0;
        let mut start_old = None;
        while k < pat.len() && j < t.len() {
            if k > 0 && t[j] == ' ' {
                j += 1;
                continue;
            }
            if t[j] != pat[k] {
                break;
            }
            if k == ctx {
                start_old = Some(j);
            }
            j += 1;
            k += 1;
        }
        if k == pat.len() && t[i] == pat[0] {
            let sa = start_old.unwrap_or(j);
            out.extend(&t[i..sa]);
            out += &new_part;
            i = j;
        } else {
            out.push(t[i]);
            i += 1;
        }
    }
    out
}

/// The differing middle of prompt and contrast prompt, with up to 2 characters of common context in front:
/// (old, new, context length)
fn substitution(raw: &str, contrast: &str) -> (String, String, usize) {
    let (x, y): (Vec<char>, Vec<char>) = (raw.chars().collect(), contrast.chars().collect());
    let mut pa = 0;
    while pa < x.len() && pa < y.len() && x[pa] == y[pa] {
        pa += 1;
    }
    let mut sb = 0;
    while sb < x.len() - pa && sb < y.len() - pa && x[x.len() - 1 - sb] == y[y.len() - 1 - sb] {
        sb += 1;
    }
    let ctx: String = x[pa.saturating_sub(2)..pa].iter().collect();
    (ctx.clone() + &x[pa..x.len() - sb].iter().collect::<String>(), ctx + &y[pa..y.len() - sb].iter().collect::<String>(), pa.min(2))
}

fn target_token(m: &Model, a: &Args) -> Result<Option<u32>, String> {
    match &a.target {
        Some(s) => {
            let s = s.replace("\\n", "\n");
            Ok(Some((0..m.tok.vocab_size().min(m.vocab_rows()) as u32).find(|&i| m.tok.piece(i) == s).ok_or(format!("--target {s:?} is not a single token"))?))
        }
        None => Ok(None),
    }
}

/// Run the requested analyses at the focus (with --pos) or at the chosen answer tokens.
fn analyses(m: &Model, raw: &str, ids: &[u32], focus: usize, a: &Args, style: &Style) -> Result<(), String> {
    let target = target_token(m, a)?;
    // reference sequences for --circuit's mean ablation: wrapped like the prompt, plus their own greedy
    // answer (as the analyzed sequence has its answer so far)
    let mut circuit_refs: Vec<Vec<u32>> = Vec::new();
    if a.circuit && a.circuit_ablation.as_deref() != Some("zero") {
        for r in &a.circuit_ref {
            let mut seq = m.tok.encode(&if a.chat { chat_wrap(m, r)? } else { r.clone() });
            if a.pos.is_none() {
                let ans = m.generate(&seq, 12);
                seq.extend(ans);
            }
            if seq.is_empty() {
                return Err(format!("--circuit-ref {r:?} gives no tokens"));
            }
            circuit_refs.push(seq);
        }
    }
    let base = a.html.clone().unwrap_or_else(|| "apollon".into());
    let one = |seq: &[u32], f: usize, suffix: String, contrast: Option<Vec<u32>>| -> Result<(), String> {
        let mut probe_result: Option<serde_json::Value> = None;
        for (index, part) in a.probe.iter().enumerate() {
            let po = probe::Opts {
                part: part.clone(),
                result_json: a.result_json.clone(),
                pos: a.probe_pos,
                html: PathBuf::from(format!("{base}_probe_{}{suffix}.html", part.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect::<String>())),
                n_deep: a.depth.unwrap_or(6),
                max: a.probe_max.unwrap_or(1024),
            };
            probe::run(m, seq, f, target, contrast.as_deref(), &po, style)?;
            if let Some(path) = &a.result_json {
                let mut result: serde_json::Value = serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
                fn prefix_ids(part: &mut serde_json::Value, index: usize) {
                    if let Some(id) = part["id"].as_str() {
                        part["id"] = serde_json::json!(format!("probe-{index}-{id}"));
                    }
                    if let Some(children) = part["children"].as_array_mut() {
                        for child in children { prefix_ids(child, index); }
                    }
                }
                if let Some(parts) = result["components"].as_array_mut() {
                    for part in parts { prefix_ids(part, index); }
                }
                if let Some(combined) = &mut probe_result {
                    for key in ["components", "checks"] {
                        if let (Some(items), Some(more)) = (combined[key].as_array_mut(), result[key].as_array_mut()) {
                            items.append(more);
                        }
                    }
                } else {
                    probe_result = Some(result);
                }
            }
        }
        if let (Some(path), Some(result)) = (&a.result_json, probe_result) { export::write(path, &result)?; }
        if !a.trace.is_empty() {
            let mut neurons = Vec::new();
            for t in &a.trace {
                if t == "auto" {
                    continue;
                }
                match neuron::parse_part(t).map_err(|e| format!("--trace {e}"))? {
                    neuron::PartSpec::Neuron(l, i) => neurons.push((l, i)),
                    _ => return Err(format!("--trace '{t}': give a neuron like \"S12 #2328\" or auto")),
                }
            }
            let o = trace::Opts {
                neurons,
                result_json: a.result_json.clone(),
                depth: a.trace_depth.unwrap_or(2),
                top: a.trace_top.unwrap_or(3),
                scalars: a.scalars.unwrap_or(10),
                html: PathBuf::from(format!("{base}_trace{suffix}.html")),
                dump: a.dump_neuron.as_ref().map(PathBuf::from),
                atlas: a.atlas.as_ref().map(PathBuf::from),
                scalars_of: a.scalars_of.clone(),
            };
            trace::run(m, seq, f, target, &o, style)?;
        }
        if a.circuit {
            let o = circuit::Opts {
                kl: a.circuit_kl.unwrap_or(0.1),
                depth: a.circuit_depth.unwrap_or(4),
                max_tests: a.circuit_tests.unwrap_or(2_000_000),
                show: a.circuit_show.unwrap_or(25),
                html: PathBuf::from(format!("{base}_circuit{suffix}.html")),
                dump: a.dump_circuit.as_ref().map(PathBuf::from),
                result_json: a.result_json.clone(),
                random: a.circuit_random,
                frozen: a.circuit_norms.as_deref() != Some("live"),
                mean: a.circuit_ablation.as_deref() != Some("zero"),
                refs: circuit_refs.clone(),
                paths: a.circuit_paths.unwrap_or(8),
                out_dir: match a.circuit_out.as_deref() {
                    Some("none") => None,
                    Some(d) => Some(PathBuf::from(format!("{d}{suffix}"))),
                    None => Some(PathBuf::from(format!("{base}_circuit{suffix}"))),
                },
                from: a.circuit_from.as_ref().map(PathBuf::from),
                bias: match a.circuit_bias.as_deref() {
                    None => None,
                    Some("all") => Some(usize::MAX),
                    Some(n) => Some(n.parse().map_err(|_| "--circuit-bias expects a number or all")?),
                },
                bias_top: a.bias_top.unwrap_or(24),
            };
            circuit::run(m, seq, f, target, &o, style)?;
        }
        if !a.bias.is_empty() {
            let mut neurons = Vec::new();
            for t in &a.bias {
                match neuron::parse_part(t).map_err(|e| format!("--bias {e}"))? {
                    neuron::PartSpec::Neuron(l, i) => neurons.push((l, i)),
                    _ => return Err(format!("--bias '{t}': give a neuron like \"S12 #2328\"")),
                }
            }
            let o = bias::Opts { neurons, dir: PathBuf::from(a.bias_out.clone().map_or_else(|| format!("{base}_bias{suffix}"), |d| format!("{d}{suffix}"))), top: a.bias_top.unwrap_or(24) };
            bias::run(m, seq, f, &o, style)?;
        }
        if a.decompose {
            let o = decompose::Opts { result_json: a.result_json.clone(), focus_only: a.focus_only, html: PathBuf::from(format!("{base}_decompose{suffix}.html")), k: a.top.unwrap_or(8) };
            decompose::run(m, seq, f, target, &o, style)?;
        }
        Ok(())
    };
    // contrast prompt: apply the prompt's substitution to the (reused) answer as well,
    // so that both sequences line up position by position (2+2 is ** -> 2+3 is **)
    let wrap = |s: &str| if a.chat { chat_wrap(m, s) } else { Ok(s.to_string()) };
    let subst = a.contrast.as_ref().map(|g| substitution(raw, g));
    let contrast_seq = |answer: &[u32]| -> Result<Option<Vec<u32>>, String> {
        let (Some(g), Some((old, new, ctx))) = (&a.contrast, &subst) else { return Ok(None) };
        let atext: String = answer.iter().map(|i| m.tok.piece(*i)).collect();
        let atext = if old.is_empty() { atext } else { replace_loose(&atext, old, new, *ctx) };
        Ok(Some(m.tok.encode(&(wrap(g)? + &atext))))
    };
    if a.pos.is_some() {
        let g = contrast_seq(&[])?.map(|mut g| {
            g.truncate(focus + 1);
            g
        });
        if g.as_ref().is_some_and(|g| g.len() != focus + 1) {
            return Err("The contrast prompt does not reach the selected position".into());
        }
        return one(&ids[..=focus], focus, String::new(), g);
    }
    let ks = if a.answer.is_empty() { vec![1] } else { a.answer.clone() };
    let n_gen = (*ks.iter().max().unwrap()).max(12);
    eprintln!("generating the answer first (greedy, {n_gen} tokens) …");
    let ans = m.generate(ids, n_gen);
    print_answer(m, &ans, style);
    for &k in &ks {
        if k > ans.len() + 1 {
            if a.result_json.is_some() {
                return Err(format!("there is no answer token {k} (the answer ends after {} tokens)", ans.len()));
            }
            eprintln!("there is no answer token {k} (the answer ends after {} tokens)", ans.len());
            continue;
        }
        let mut seq = ids.to_vec();
        seq.extend_from_slice(&ans[..k - 1]);
        let f = seq.len() - 1;
        let suffix = if ks.len() > 1 { format!("_{k}") } else { String::new() };
        println!();
        println!("{}", style.bold(&format!("Answer token {k}{}", ans.get(k - 1).map_or(String::new(), |t| format!(" = {}", tok(m, *t))))));
        let g = contrast_seq(&ans[..k - 1])?;
        if let Some(g) = &g {
            if g.len() != seq.len() {
                return Err(format!(
                    "the contrast prompt gives {} instead of {} tokens ({}); inserting values needs both to be equally long",
                    g.len(),
                    seq.len(),
                    esc_piece(&g.iter().map(|i| m.tok.piece(*i)).collect::<String>())
                ));
            }
        }
        one(&seq, f, suffix, g)?;
        if let Some(path) = &a.result_json {
            let mut result: serde_json::Value = serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
            result["generated"] = serde_json::json!(export::tokens(m, &ans));
            result["answer"] = serde_json::json!(ans.iter().map(|id| m.tok.piece(*id)).collect::<String>());
            export::write(path, &result)?;
        }
    }
    Ok(())
}

fn print_answer(m: &Model, ans: &[u32], style: &Style) {
    println!("{} {}", style.bold("Model answer (greedy):"), esc_piece(&ans.iter().map(|i| m.tok.piece(*i)).collect::<String>()));
    println!("  {}", style.dim(&ans.iter().enumerate().map(|(i, t)| format!("{}:{}", i + 1, tok(m, *t))).collect::<Vec<_>>().join("  ")));
}

/// Test format: header + token ids + residual after the embedding/every layer + logits at all positions
fn write_dump(m: &Model, ids: &[u32], path: &str) -> Result<(), String> {
    let tr = m.run(ids, ids.len() - 1, &RunOpts { dump: true, ..Default::default() });
    let hs = tr.hidden_all.as_ref().unwrap();
    let empty = Vec::new();
    let lg = tr.logits_all.as_ref().unwrap_or(&empty);
    let mut out = Vec::new();
    out.extend_from_slice(b"APLN");
    for v in [ids.len() as u32, hs.len() as u32, m.d as u32, (lg.len() / ids.len()) as u32] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for id in ids {
        out.extend_from_slice(&id.to_le_bytes());
    }
    for h in hs.iter().chain(std::iter::once(lg)) {
        for v in h {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::write(path, out).map_err(|e| format!("{path}: {e}"))
}

/// Analyses without a prompt: the static neuron report (and building the atlas)
fn statics(m: &Model, a: &Args, style: &Style) -> Result<(), String> {
    let base = a.html.clone().unwrap_or_else(|| "apollon".into());
    for t in &a.neuron {
        match neuron::parse_part(t).map_err(|e| format!("--neuron {e}"))? {
            neuron::PartSpec::Neuron(l, i) => {
                let html = PathBuf::from(format!("{base}_neuron_S{l}_{i}.html"));
                inspect::run(m, l, i, &inspect::Opts { k: a.top.unwrap_or(8) }, &html, a.result_json.as_deref(), style)?;
            }
            _ => return Err(format!("--neuron '{t}': give a neuron like \"S12 #2328\"")),
        }
    }
    if let (Some(f), true) = (&a.atlas, a.trace.is_empty()) {
        atlas::build(m, std::path::Path::new(f), &atlas::BuildOpts { k: a.atlas_k.unwrap_or(64), layers: a.atlas_layers })?;
    }
    Ok(())
}

fn analyze(m: &Model, raw: &str, a: &Args, style: &Style) -> Result<(), String> {
    let text = if a.chat { chat_wrap(m, raw)? } else { raw.to_string() };
    let ids = m.tok.encode(&text);
    if ids.is_empty() {
        return Err("the prompt gives no tokens".into());
    }
    if a.tokens_only {
        println!("{}", serde_json::to_string(&ids).unwrap());
        return Ok(());
    }
    if let Some(&bad) = ids.iter().find(|&&i| i as usize >= m.embed.rows) {
        return Err(format!("token id {bad} lies outside the embedding ({})", m.embed.rows));
    }
    if let Some(p) = &a.dump {
        return write_dump(m, &ids, p);
    }
    if !m.has_head {
        return Err("this is an encoder without an output head: only --dump and --tokens-only are available".into());
    }
    let t = ids.len() as i64;
    let focus = match a.pos {
        None => t - 1,
        Some(p) if p < 0 => t + p,
        Some(p) => p,
    };
    if focus < 0 || focus >= t {
        return Err(format!("--pos {} lies outside 0..{}", a.pos.unwrap_or(0), t - 1));
    }
    if a.decompose || a.circuit || !a.probe.is_empty() || !a.trace.is_empty() || !a.bias.is_empty() {
        return analyses(m, raw, &ids, focus as usize, a, style);
    }
    let requested = a.answer.first().copied().unwrap_or(1);
    let ans = m.generate(&ids, 32.max(requested));
    if let Some(path) = &a.result_json {
        let mut seq = ids[..=focus as usize].to_vec();
        if a.pos.is_none() {
            if requested > ans.len() + 1 {
                return Err(format!("there is no answer token {requested} (the answer ends after {} tokens)", ans.len()));
            }
            seq.extend_from_slice(&ans[..requested-1]);
        }
        let at = seq.len()-1;
        let tr = m.run(&seq, at, &RunOpts::default());
        let output = model::top_probs(&tr.logits, 10);
        let target = target_token(m,a)?.unwrap_or(output[0].0);
        let mut result = export::result(m, &seq, at, target, "generate", &output);
        result["generated"] = serde_json::json!(export::tokens(m, &ans));
        result["answer"] = serde_json::json!(ans.iter().map(|id| m.tok.piece(*id)).collect::<String>());
        export::write(path, &result)?;
    }
    print_answer(m, &ans, style);
    eprintln!("(analyses: --decompose or --probe \"S12 MLP\"; see --help)");
    Ok(())
}

fn main() {
    let a = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(2);
        }
    };
    if a.serve {
        if let Err(e) = server::serve() { eprintln!("error: {e}"); std::process::exit(1); }
        return;
    }
    if a.list || a.models_json {
        list(a.models_json);
        return;
    }
    if let Some(n) = a.threads {
        rayon::ThreadPoolBuilder::new().num_threads(n).build_global().ok();
    }
    let Some(spec) = a.model.clone() else {
        eprintln!("{HELP}\n\nerror: no model given (-m). Available models: apollon --list");
        std::process::exit(2);
    };
    let static_only = !a.neuron.is_empty() || (a.atlas.is_some() && a.trace.is_empty()) || a.selftest_gpu;
    if a.prompt.is_empty() && !static_only {
        eprintln!("error: no prompt given (help: --help)");
        std::process::exit(2);
    }
    let gpu_choice = a.gpu.clone().or_else(|| std::env::var("APOLLON_GPU").ok()).unwrap_or_else(|| "auto".into());
    let backend = match gpu::init(&gpu_choice) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(2);
        }
    };
    let color = a.color.unwrap_or_else(|| std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none());
    let style = Style { color, width: term_width() };
    let t0 = Instant::now();
    let m = match load(&spec) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };
    if !a.tokens_only {
        eprintln!("{} · loaded in {} s · matrix products: {backend}", describe(&m), num(t0.elapsed().as_secs_f32(), 2));
    }
    if a.selftest_gpu {
        let mut mats: Vec<(String, &tensor::Mat)> = vec![("output".into(), m.lm_head())];
        for (l, b) in m.blocks.iter().enumerate().take(2) {
            let at = &b.attn;
            for (n, w) in [("q", &at.wq), ("k", &at.wk), ("v", &at.wv), ("qkv", &at.wqkv)] {
                if let Some(w) = w {
                    mats.push((format!("S{l} {n}"), w));
                }
            }
            mats.push((format!("S{l} o"), &at.wo));
            if let Some(g) = &b.mlp.gate {
                mats.push((format!("S{l} gate"), g));
            }
            mats.push((format!("S{l} up"), &b.mlp.up));
            mats.push((format!("S{l} down"), &b.mlp.down));
            if let Some(p) = &b.ple {
                mats.push((format!("S{l} PLE gate"), &p.gate));
                mats.push((format!("S{l} PLE proj"), &p.proj));
            }
        }
        let (rep, ok) = gpu::selftest(&mats);
        print!("{rep}");
        println!("{}", if ok { "GPU self-test: all OK" } else { "GPU self-test: FAILED" });
        std::process::exit(if ok { 0 } else { 1 });
    }
    if let Err(e) = statics(&m, &a, &style) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
    if a.prompt.is_empty() {
        return;
    }
    let text = a.prompt.join(" ");
    if let Err(e) = analyze(&m, &text, &a, &style) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
