"""Reference check of the neuron decomposition (--trace … --dump-neuron) against transformers + NumPy.

For random small Llama, Qwen3 and Gemma 3 models (gated) and the real Pythia-14m (GPT-NeoX: parallel
residual, LayerNorm, biases, not gated) the neuron with the largest |a| in a middle layer is traced at
the last token. transformers runs with hooks; NumPy recomputes independently:
  * p, q, a, the pre-norm scale s, the MLP input x̂ and the stream x at the MLP input
  * every scalar product W[i,j]·x̂_j
  * the contribution of the embedding, of every head of every reachable layer and of every earlier
    neuron (with frozen post-norm scales), and the biases
  * the ellipse (Gram matrix) and its extremes by a dense angle grid and by random inputs on the sphere
  * phase 2: for the strongest atom (head h of layer k reading position j) the chain post-norm → W_O →
    attention weight → W_V → pre-norm at j, and the split of what position j carries into its sources
Weights are unquantized (float32), so apollon and the reference read identical numbers.
  PYTHONPATH=<transformers> python3 tests/compare_neuron.py BIN WORKDIR
"""
import glob, json, math, os, shutil, subprocess, sys
import numpy as np, torch
from transformers import AutoModelForCausalLM

BIN, WORK = sys.argv[1], sys.argv[2]
HF = os.path.expanduser("~/.cache/huggingface/hub")
snap = lambda n: sorted(glob.glob(f"{HF}/models--{n.replace('/', '--')}/snapshots/*/config.json"))[-1].rsplit("/", 1)[0]
TEXT = "The capital of France is Paris, and the capital of Italy is"


def save(m, name):
    d = os.path.join(WORK, f"neuron-{name}")
    shutil.rmtree(d, ignore_errors=True)
    m.save_pretrained(d, safe_serialization=True)
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        shutil.copy(os.path.join(snap("EleutherAI/pythia-14m"), f), d)
    return d


def tiny(kind):
    from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM, Gemma3TextConfig, Gemma3ForCausalLM
    torch.manual_seed({"llama": 11, "qwen3": 12, "gemma3": 13}[kind])
    common = dict(vocab_size=50304, hidden_size=64, intermediate_size=96, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)
    if kind == "llama":
        m = LlamaForCausalLM(LlamaConfig(**common, rope_theta=5e5, tie_word_embeddings=False))
    elif kind == "qwen3":
        m = Qwen3ForCausalLM(Qwen3Config(**common, head_dim=24, rope_theta=1e6, tie_word_embeddings=False))
    else:
        m = Gemma3ForCausalLM(Gemma3TextConfig(**common, head_dim=16, query_pre_attn_scalar=24, sliding_window=6, final_logit_softcapping=30.0))
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.08)
    return save(m, kind)


def rms_scale(x, eps):
    return 1.0 / math.sqrt(float(np.mean(x.astype(np.float64) ** 2)) + eps)


def ln_scale(x, eps):
    x = x.astype(np.float64)
    return 1.0 / math.sqrt(float(np.var(x)) + eps)


def gelu_tanh(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


ACTS = {"SiLU": lambda x: x / (1 + np.exp(-x)), "GELU": lambda x: 0.5 * x * (1 + np.vectorize(math.erf)(x / math.sqrt(2))), "GELU-tanh": gelu_tanh}


def run_case(name, md, kind):
    hf = AutoModelForCausalLM.from_pretrained(md, dtype=torch.float32, attn_implementation="eager").eval()
    ids = json.loads(subprocess.run([BIN, "-m", md, "--tokens-only", TEXT], capture_output=True, text=True).stdout)
    T = len(ids)
    f = T - 1
    neox = kind == "neox"
    layers = hf.gpt_neox.layers if neox else hf.model.layers
    L = len(layers)
    l = L // 2
    cap = {}

    def keep(key, which="out"):
        def hook(mod, inp, out):
            cap[key] = (inp[0] if which == "in" else out)[0].detach().double().numpy()
        return hook

    hs = []
    for k, ly in enumerate(layers):
        if neox:
            hs.append(ly.post_attention_layernorm.register_forward_hook(keep(("mlp_x", k), "in")))
            hs.append(ly.post_attention_layernorm.register_forward_hook(keep(("mlp_in", k))))
            hs.append(ly.mlp.dense_4h_to_h.register_forward_hook(keep(("act", k), "in")))
            hs.append(ly.attention.dense.register_forward_hook(keep(("z", k), "in")))
            hs.append(ly.mlp.dense_h_to_4h.register_forward_hook(keep(("up", k))))
        else:
            pre = ly.pre_feedforward_layernorm if kind == "gemma3" else ly.post_attention_layernorm
            hs.append(pre.register_forward_hook(keep(("mlp_x", k), "in")))
            hs.append(pre.register_forward_hook(keep(("mlp_in", k))))
            hs.append(ly.mlp.down_proj.register_forward_hook(keep(("act", k), "in")))
            hs.append(ly.mlp.gate_proj.register_forward_hook(keep(("gate", k))))
            hs.append(ly.mlp.up_proj.register_forward_hook(keep(("up", k))))
            hs.append(ly.self_attn.o_proj.register_forward_hook(keep(("z", k), "in")))
            hs.append(ly.self_attn.o_proj.register_forward_hook(keep(("attn_raw", k))))
            hs.append(ly.mlp.down_proj.register_forward_hook(keep(("mlp_raw", k))))
    with torch.no_grad():
        out = hf(torch.tensor([ids]), output_hidden_states=True, output_attentions=True)
    for h in hs:
        h.remove()
    emb = out.hidden_states[0][0].double().numpy()
    i = int(np.argmax(np.abs(cap[("act", l)][f])))
    dump = os.path.join(WORK, f"neuron-{name}.json")
    r = subprocess.run([BIN, "-m", md, "--pos", "-1", "--trace", f"S{l} #{i}", "--trace-top", "2", "--trace-depth", "1", "--dump-neuron", dump, "--html", os.path.join(WORK, f"neuron-{name}"), "--no-color", TEXT], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr)
    J = json.load(open(dump))
    W = lambda t: t.detach().double().numpy()
    dev = {}
    ly = layers[l]
    # ---- the neuron's own numbers
    if neox:
        nrm = ly.post_attention_layernorm
        gamma, beta, eps = W(nrm.weight), W(nrm.bias), nrm.eps
        wg, bg = W(ly.mlp.dense_h_to_4h.weight)[i], float(W(ly.mlp.dense_h_to_4h.bias)[i])
        s = ln_scale(cap[("mlp_x", l)][f], eps)
        p_ref = float(cap[("up", l)][f][i])
        a_ref = float(cap[("act", l)][f][i])
        gp = wg * gamma
        gp = gp - gp.mean()
        cg = float(wg @ beta) + bg
        up = None
    else:
        pre = ly.pre_feedforward_layernorm if kind == "gemma3" else ly.post_attention_layernorm
        gamma = W(pre.weight) + (1.0 if kind == "gemma3" else 0.0)
        eps = pre.eps if hasattr(pre, "eps") else pre.variance_epsilon
        s = rms_scale(cap[("mlp_x", l)][f], eps)
        wg, wu = W(ly.mlp.gate_proj.weight)[i], W(ly.mlp.up_proj.weight)[i]
        p_ref, q_ref = float(cap[("gate", l)][f][i]), float(cap[("up", l)][f][i])
        a_ref = float(cap[("act", l)][f][i])
        gp, up = wg * gamma, wu * gamma
        cg = 0.0
    rel = lambda a, b, sc=None: abs(a - b) / max(abs(b) if sc is None else sc, 1e-12)
    dev["p"] = rel(J["p"], p_ref)
    if up is not None:
        dev["q"] = rel(J["q"], q_ref)
    dev["a"] = rel(J["a"], a_ref)
    dev["s"] = rel(J["s"], s)
    xh = cap[("mlp_in", l)][f]
    dev["x̂"] = float(np.abs(np.array(J["xhat"]) - xh).max() / np.abs(xh).max())
    x = cap[("mlp_x", l)][f]
    dev["x"] = float(np.abs(np.array(J["x"]) - x).max() / np.abs(x).max())
    # ---- level 0: every scalar product
    sp = wg * xh
    dev["scalars"] = float(np.abs(np.array(J["scalars_p"]) - sp).max() / np.abs(sp).max())
    # ---- level 1: sources
    src = {}
    for c in J["sources"]:
        key = (c["kind"], c.get("layer"), c.get("head", c.get("index")))
        src[key] = (c["p"], c["q"])
    scale_sum = sum(abs(c["p"]) for c in J["sources"])
    worst = {"embedding": 0.0, "heads": 0.0, "neurons": 0.0, "biases": 0.0}
    read = lambda v: (s * float(gp @ v), s * float(up @ v) if up is not None else 0.0)

    def cmp(group, key, v):
        p, q = read(v)
        got = src.get(key, (0.0, 0.0))
        worst[group] = max(worst[group], (abs(got[0] - p) + abs(got[1] - q)) / max(scale_sum, 1e-12))

    cmp("embedding", ("token", None, ids[f]), emb[f])
    last = l if not neox else l - 1  # NeoX: the MLP reads the stream before the layer
    for k in range(0, last + 1):
        lk = layers[k]
        if neox:
            nh = hf.config.num_attention_heads
            hd = hf.config.hidden_size // nh
            Wo = W(lk.attention.dense.weight)
            z = cap[("z", k)][f]
            for h in range(nh):
                cmp("heads", ("head", k, h), Wo[:, h * hd:(h + 1) * hd] @ z[h * hd:(h + 1) * hd])
            cmp("biases", ("attn_bias", k, None), W(lk.attention.dense.bias))
        else:
            cfg = hf.config
            nh = cfg.num_attention_heads
            hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // nh
            Wo = W(lk.self_attn.o_proj.weight)
            z = cap[("z", k)][f]
            post = 1.0
            wpost = 1.0
            if kind == "gemma3":
                pn = lk.post_attention_layernorm
                post = rms_scale(cap[("attn_raw", k)][f], pn.eps)
                wpost = W(pn.weight) + 1.0
            for h in range(nh):
                cmp("heads", ("head", k, h), post * wpost * (Wo[:, h * hd:(h + 1) * hd] @ z[h * hd:(h + 1) * hd]))
    for k in range(0, l):
        lk = layers[k]
        a = cap[("act", k)][f]
        Wd = W(lk.mlp.dense_4h_to_h.weight) if neox else W(lk.mlp.down_proj.weight)
        post, wpost = 1.0, np.ones(Wd.shape[0])
        if kind == "gemma3":
            pn = lk.post_feedforward_layernorm
            post = rms_scale(cap[("mlp_raw", k)][f], pn.eps)
            wpost = W(pn.weight) + 1.0
        # all neurons of the layer at once: p_c = s · post · a · (g' ⊙ γ_post)·W_down[:, i']
        rp = s * post * a * ((gp * wpost) @ Wd)
        rq = s * post * a * ((up * wpost) @ Wd) if up is not None else np.zeros_like(rp)
        for i2 in range(len(a)):
            got = src.get(("neuron", k, i2), (0.0, 0.0))
            worst["neurons"] = max(worst["neurons"], (abs(got[0] - rp[i2]) + abs(got[1] - rq[i2])) / max(scale_sum, 1e-12))
        if neox:
            cmp("biases", ("mlp_bias", k, None), W(lk.mlp.dense_4h_to_h.bias))
    dev.update({f"sources: {g}": v for g, v in worst.items() if v > 0 or g != "biases"})
    total = sum(c["p"] for c in J["sources"])
    dev["Σ sources = p"] = rel(total, p_ref, scale_sum)
    # ---- ellipse
    G = np.array([[gp @ gp, gp @ up], [gp @ up, up @ up]]) if up is not None else np.array([[gp @ gp, 0], [0, 0]])
    Jg = J["ellipse"]["gram"]
    dev["Gram"] = float(np.abs(np.array([G[0, 0], G[0, 1], G[1, 1]]) - Jg).max() / np.abs(G).max())
    act = ACTS[J["act"]]
    R = math.sqrt(len(gp))
    c0 = np.array([cg, 0.0])
    th = np.linspace(0, 2 * math.pi, 400001)
    if up is not None:
        Lc = np.linalg.cholesky(G + 1e-300 * np.eye(2))
        P = c0[0] + R * Lc[0, 0] * np.cos(th)
        Q = c0[1] + R * (Lc[1, 0] * np.cos(th) + Lc[1, 1] * np.sin(th))
        vals = act(P) * Q
    else:
        P = c0[0] + R * math.sqrt(G[0, 0]) * np.cos(th)
        vals = act(P)
    amax, amin = float(vals.max()), float(vals.min())
    span = amax - amin
    dev["a_max (grid of 400k angles)"] = abs(J["ellipse"]["a_max"] - amax) / span
    dev["a_min"] = abs(J["ellipse"]["a_min"] - amin) / span
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(20000, len(gp)))
    Z *= R / np.linalg.norm(Z, axis=1, keepdims=True)
    if neox:
        Z -= Z.mean(axis=1, keepdims=True)  # LayerNorm: z is centered (‖z‖ ≤ √d stays true)
    ps = c0[0] + Z @ gp
    sv = act(ps) * (Z @ up) if up is not None else act(ps)
    above = float(max(0.0, (sv.max() - J["ellipse"]["a_max"]) / span))
    below = float(max(0.0, (J["ellipse"]["a_min"] - sv.min()) / span))
    dev["random inputs inside [a_min, a_max]"] = max(above, below)
    # ---- phase 2: what the strongest atom's read position carries
    C = J["content"]
    k, h, j = C["layer"], C["head"], C["pos"]
    lk = layers[k]
    hidden = [x[0].double().numpy() for x in out.hidden_states]
    nh = hf.config.num_attention_heads
    hd = (hf.config.hidden_size // nh) if neox else (getattr(hf.config, "head_dim", None) or hf.config.hidden_size // nh)
    chains = [s * gp] + ([s * up] if up is not None else [])
    if kind == "gemma3":
        pn = lk.post_attention_layernorm
        chains = [c * (W(pn.weight) + 1.0) * rms_scale(cap[("attn_raw", k)][f], pn.eps) for c in chains]
    A = float(out.attentions[k][0, h, f, j])
    Wo = W(lk.attention.dense.weight) if neox else W(lk.self_attn.o_proj.weight)
    rz = [Wo[:, h * hd:(h + 1) * hd].T @ c * A for c in chains]
    if neox:
        rows = slice(h * 3 * hd + 2 * hd, h * 3 * hd + 3 * hd)
        Wv, bv = W(lk.attention.query_key_value.weight)[rows], W(lk.attention.query_key_value.bias)[rows]
        ln = lk.input_layernorm
        sa = ln_scale(hidden[k][j], ln.eps)
        gam, beta = W(ln.weight), W(ln.bias)
    else:
        g = h // (nh // hf.config.num_key_value_heads)
        Wv, bv = W(lk.self_attn.v_proj.weight)[g * hd:(g + 1) * hd], np.zeros(hd)
        ln = lk.input_layernorm
        sa = rms_scale(hidden[k][j], ln.eps if hasattr(ln, "eps") else ln.variance_epsilon)
        gam, beta = W(ln.weight) + (1.0 if kind == "gemma3" else 0.0), np.zeros(len(gp))
    rx = []
    for r in rz:
        v = sa * gam * (Wv.T @ r)
        if neox:
            v = v - v.mean()
        rx.append(v)
    offs = [float(r @ (Wv @ beta + bv)) for r in rz]
    readj = lambda vec: (float(rx[0] @ vec), float(rx[1] @ vec) if len(rx) > 1 else 0.0)
    csrc = {}
    for c in C["sources"]:
        csrc[(c["kind"], c.get("layer"), c.get("head", c.get("index")))] = (c["p"], c["q"])
    cscale = sum(abs(c["p"]) + abs(c["q"]) for c in C["sources"]) + 1e-300
    cw = 0.0

    def ccmp(key, vec):
        nonlocal cw
        p, q = readj(vec)
        got = csrc.get(key, (0.0, 0.0))
        cw = max(cw, (abs(got[0] - p) + abs(got[1] - q)) / cscale)

    ccmp(("token", None, ids[j]), emb[j])
    for k2 in range(k):
        l2 = layers[k2]
        z = cap[("z", k2)][j]
        Wo2 = W(l2.attention.dense.weight) if neox else W(l2.self_attn.o_proj.weight)
        post2, wpost2 = 1.0, 1.0
        if kind == "gemma3":
            pn2 = l2.post_attention_layernorm
            post2, wpost2 = rms_scale(cap[("attn_raw", k2)][j], pn2.eps), W(pn2.weight) + 1.0
        for h2 in range(nh):
            ccmp(("head", k2, h2), post2 * wpost2 * (Wo2[:, h2 * hd:(h2 + 1) * hd] @ z[h2 * hd:(h2 + 1) * hd]))
        a2 = cap[("act", k2)][j]
        Wd2 = W(l2.mlp.dense_4h_to_h.weight) if neox else W(l2.mlp.down_proj.weight)
        mp, mw = 1.0, np.ones(Wd2.shape[0])
        if kind == "gemma3":
            pm = l2.post_feedforward_layernorm
            mp, mw = rms_scale(cap[("mlp_raw", k2)][j], pm.eps), W(pm.weight) + 1.0
        pr = mp * a2 * ((rx[0] * mw) @ Wd2)
        qr = mp * a2 * ((rx[1] * mw) @ Wd2) if len(rx) > 1 else np.zeros_like(pr)
        for i2 in range(len(a2)):
            got = csrc.get(("neuron", k2, i2), (0.0, 0.0))
            cw = max(cw, (abs(got[0] - pr[i2]) + abs(got[1] - qr[i2])) / cscale)
        if neox:
            ccmp(("attn_bias", k2, None), W(l2.attention.dense.bias))
            ccmp(("mlp_bias", k2, None), W(l2.mlp.dense_4h_to_h.bias))
    dev[f"phase 2: S{k} head {h} reads [{j}] – sources"] = cw
    dev["phase 2: value offset"] = (abs(C["offset"][0] - offs[0]) + (abs(C["offset"][1] - offs[1]) if len(offs) > 1 else 0.0)) / cscale
    tot = sum(c["p"] for c in C["sources"]) + C["offset"][0]
    dev["phase 2: Σ sources + offset = atom"] = abs(tot - C["atom"][0]) / cscale
    good = all(v < 1e-4 for v in dev.values())
    print(f"{'OK    ' if good else 'FAILED'} {name:14s} S{l} #{i} ({J['weights']}): " + " · ".join(f"{k} {v:.1e}" for k, v in dev.items()))
    return good


ok = True
cases = [("llama", tiny("llama"), "llama"), ("qwen3", tiny("qwen3"), "qwen3"), ("gemma3", tiny("gemma3"), "gemma3"), ("pythia-14m", snap("EleutherAI/pythia-14m"), "neox")]
for name, md, kind in cases:
    ok &= run_case(name, md, kind)
sys.exit(0 if ok else 1)
