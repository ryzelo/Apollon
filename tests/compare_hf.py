"""Exact comparison with Hugging Face transformers (the reference implementation).

Checks for every model: tokenization, residual stream after every layer and logits at all
positions. Small random models (Qwen2, Qwen3, Llama with GQA + Llama 3 RoPE) are created fresh
for this; local models from the HF cache are added.
  PYTHONPATH=<transformers> python3 tests/compare_hf.py target/release/apollon WORKDIR
"""
import glob, json, os, shutil, struct, subprocess, sys
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
BIN, WORK = sys.argv[1], sys.argv[2]
HF = os.path.expanduser("~/.cache/huggingface/hub")
snap = lambda n: sorted(glob.glob(f"{HF}/models--{n.replace('/', '--')}/snapshots/*/config.json"))[-1].rsplit("/", 1)[0]
# test inputs are deliberately multilingual (umlauts, emoji, CJK, special tokens)
TEXTS = ["Die Hauptstadt von Frankreich ist", "Hello world!  Two  spaces\nand a newline", "Zahlen 12345 und Ümläute: äöüß €",
         "def f(x):\n\treturn x**2  # Kommentar", "emoji 🙂 und 中文", "<|endoftext|>Spezial"]

def tiny(kind):
    from transformers import Qwen2Config, Qwen3Config, LlamaConfig, Qwen2ForCausalLM, Qwen3ForCausalLM, LlamaForCausalLM
    torch.manual_seed({"qwen2": 1, "qwen3": 2, "llama3": 3}[kind])
    common = dict(vocab_size=50304, hidden_size=64, intermediate_size=96, num_hidden_layers=3, num_attention_heads=4,
                  num_key_value_heads=2, max_position_embeddings=512)
    if kind == "qwen2":
        m = Qwen2ForCausalLM(Qwen2Config(**common, rope_theta=1e6, tie_word_embeddings=True))
    elif kind == "qwen3":
        m = Qwen3ForCausalLM(Qwen3Config(**common, head_dim=24, rope_theta=1e6, tie_word_embeddings=False))
    else:
        m = LlamaForCausalLM(LlamaConfig(**common, rope_theta=5e5, tie_word_embeddings=False, rope_scaling={
            "rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0, "original_max_position_embeddings": 64}))
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.08)          # everything random, biases and norm weights too
    d = os.path.join(WORK, f"tiny-{kind}")
    shutil.rmtree(d, ignore_errors=True)
    m.save_pretrained(d, safe_serialization=True)
    src = snap("EleutherAI/pythia-14m")
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        shutil.copy(os.path.join(src, f), d)
    return d

def dump(model_dir, text):
    out = os.path.join(WORK, "dump.bin")
    r = subprocess.run([BIN, "-m", model_dir, "--dump", out, text], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr)
    b = open(out, "rb").read()
    assert b[:4] == b"APLN"
    t, nh, d, v = struct.unpack("<4I", b[4:20])
    ids = list(struct.unpack(f"<{t}I", b[20:20 + 4 * t]))
    f = np.frombuffer(b[20 + 4 * t:], dtype="<f4")
    hid = f[: nh * t * d].reshape(nh, t, d)
    lg = f[nh * t * d:].reshape(t, v)
    return ids, hid, lg

def tokens(model_dir, text):
    r = subprocess.run([BIN, "-m", model_dir, "--tokens-only", text], capture_output=True, text=True)
    return json.loads(r.stdout)

ok = True
models = [snap("hf-internal-testing/tiny-random-LlamaForCausalLM"), snap("EleutherAI/pythia-14m"), snap("EleutherAI/pythia-70m"),
           tiny("qwen2"), tiny("qwen3"), tiny("llama3")]
for md in models:
    name = md.split("models--")[-1].split("/")[0] if "models--" in md else os.path.basename(md)
    # the random models carry the Pythia tokenizer; AutoTokenizer would replace its pre-tokenizer
    # for qwen2 -> use the tokenizer.json directly (package tokenizers) as the reference there
    if md.startswith(WORK):
        from tokenizers import Tokenizer as RawTok
        raw = RawTok.from_file(os.path.join(md, "tokenizer.json"))
        tk = lambda s: {"input_ids": raw.encode(s).ids}
    else:
        tk = AutoTokenizer.from_pretrained(md)
    hf = AutoModelForCausalLM.from_pretrained(md, dtype=torch.float32, attn_implementation="eager").eval()
    tok_ok = all(tokens(md, s) == tk(s)["input_ids"] for s in TEXTS)
    worst_h, worst_l = 0.0, 0.0
    for s in TEXTS[:3]:
        ids, hid, lg = dump(md, s)
        with torch.no_grad():
            o = hf(torch.tensor([ids]), output_hidden_states=True)
        hs = [h[0].numpy() for h in o.hidden_states]
        L = len(hs) - 1
        for l in range(L):   # in HF hs[L] is already final-normed -> checked via the logits
            scale = np.abs(hs[l]).max() + 1e-9
            worst_h = max(worst_h, float(np.abs(hid[l] - hs[l]).max() / scale))
        ref = o.logits[0].numpy()
        worst_l = max(worst_l, float(np.abs(lg - ref).max() / (np.abs(ref).max() + 1e-9)))
    good = tok_ok and worst_h < 1e-4 and worst_l < 1e-4
    ok &= good
    print(f"{'OK    ' if good else 'FAILED'} {name:45s} tokens equal: {tok_ok}  residual max. rel. dev. {worst_h:.2e}  logits {worst_l:.2e}")
sys.exit(0 if ok else 1)
