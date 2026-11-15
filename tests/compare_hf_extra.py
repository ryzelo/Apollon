"""Exact comparison of the further components with transformers:
Gemma 3 and Gemma 4 as small random models with all special parts (sliding window smaller than
the prompt, post-norms, shared K/V, per-layer embedding, proportional RoPE, softcapping,
layer scalars), plus real models: TinyStories (GPT-Neo) and unixcoder (RoBERTa encoder).
  PYTHONPATH=<transformers> python3 tests/compare_hf_extra.py BIN WORKDIR"""
import glob, json, os, shutil, struct, subprocess, sys
import numpy as np, torch
from transformers import AutoModel, AutoModelForCausalLM
BIN, WORK = sys.argv[1], sys.argv[2]
HF = os.path.expanduser("~/.cache/huggingface/hub")
def snaps(n): return sorted(glob.glob(f"{HF}/models--{n.replace('/', '--')}/snapshots/*/"), key=os.path.getmtime, reverse=True)
# test inputs are deliberately multilingual
TEXTS = ["Die Hauptstadt von Frankreich ist Paris, und die Hauptstadt von Deutschland ist Berlin. Und Italien?",
         "def f(x):\n\treturn x**2  # Kommentar über Ümläute",
         "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20"]

def tiny(kind):
    from transformers import Gemma3TextConfig, Gemma3ForCausalLM, Gemma4TextConfig, Gemma4ForCausalLM
    torch.manual_seed(7)
    if kind == "gemma3":
        cfg = Gemma3TextConfig(vocab_size=50304, hidden_size=64, intermediate_size=96, num_hidden_layers=7, num_attention_heads=4,
                               num_key_value_heads=2, head_dim=16, query_pre_attn_scalar=24, sliding_window=6, max_position_embeddings=512,
                               rope_theta=1e6, rope_local_base_freq=1e4, rope_scaling={"rope_type": "linear", "factor": 8.0},
                               final_logit_softcapping=30.0, attn_logit_softcapping=50.0)
        m = Gemma3ForCausalLM(cfg)
    else:
        cfg = Gemma4TextConfig(vocab_size=50304, vocab_size_per_layer_input=50304, hidden_size=64, intermediate_size=96, num_hidden_layers=8,
                               num_attention_heads=4, num_key_value_heads=2, head_dim=16, global_head_dim=32, sliding_window=6,
                               layer_types=["sliding_attention"] * 3 + ["full_attention"] + ["sliding_attention"] * 3 + ["full_attention"],
                               num_kv_shared_layers=4, hidden_size_per_layer_input=8, max_position_embeddings=512, final_logit_softcapping=30.0)
        m = Gemma4ForCausalLM(cfg)
    with torch.no_grad():
        for name, p in m.named_parameters():
            p.normal_(0, 0.08)
        for name, b in m.named_buffers():
            if name.endswith("layer_scalar"):
                b.uniform_(0.3, 1.2)   # non-trivial layer scalars
    d = os.path.join(WORK, f"tiny-{kind}")
    shutil.rmtree(d, ignore_errors=True)
    m.save_pretrained(d, safe_serialization=True)
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        shutil.copy(os.path.join(snaps("EleutherAI/pythia-14m")[0], f), d)
    return d, m

def dump(md, text):
    out = os.path.join(WORK, "dump.bin")
    r = subprocess.run([BIN, "-m", md, "--dump", out, text], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr)
    b = open(out, "rb").read()
    assert b[:4] == b"APLN"
    t, nh, d, v = struct.unpack("<4I", b[4:20])
    ids = list(struct.unpack(f"<{t}I", b[20:20 + 4 * t]))
    f = np.frombuffer(b[20 + 4 * t:], dtype="<f4")
    return ids, f[: nh * t * d].reshape(nh, t, d), (f[nh * t * d:].reshape(t, v) if v else None)

ok = True
cases = []
for kind in ("gemma3", "gemma4"):
    d, m = tiny(kind)
    cases.append((f"tiny-{kind} (random)", d, m.eval(), True))
ts = snaps("roneneldan/TinyStories-33M")
cases.append(("TinyStories-33M (GPT-Neo, real)", "roneneldan/TinyStories-33M", AutoModelForCausalLM.from_pretrained(next(s for s in ts if os.path.exists(s + "config.json")), dtype=torch.float32).eval(), True))
ux = next(s for s in snaps("microsoft/unixcoder-base") if os.path.exists(s + "config.json"))
cases.append(("unixcoder (RoBERTa encoder, real)", "microsoft/unixcoder-base", AutoModel.from_pretrained(ux, dtype=torch.float32, add_pooling_layer=False).eval(), False))
for name, md, hf, causal in cases:
    worst_h = worst_l = 0.0
    for s in TEXTS:
        ids, hid, lg = dump(md, s)
        with torch.no_grad():
            o = hf(torch.tensor([ids]), output_hidden_states=True)
        hs = [h[0].float().numpy() for h in o.hidden_states]
        n = len(hs) - 1 if causal else len(hs)    # decoder: in HF the last entry is already final-normed
        for l in range(n):
            worst_h = max(worst_h, float(np.abs(hid[l] - hs[l]).max() / (np.abs(hs[l]).max() + 1e-9)))
        if causal:
            ref = o.logits[0].numpy()
            worst_l = max(worst_l, float(np.abs(lg - ref).max() / (np.abs(ref).max() + 1e-9)))
    good = worst_h < 1e-4 and worst_l < 1e-4
    ok &= good
    print(f"{'OK    ' if good else 'FAILED'} {name:36s} {len(ids)}+ tokens  residual max. rel. dev. {worst_h:.2e}" + (f"  logits {worst_l:.2e}" if causal else "  (encoder: all layers)"))
sys.exit(0 if ok else 1)
