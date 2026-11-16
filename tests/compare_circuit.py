"""Reference check of the switched-off scalar products (--circuit) against transformers.

apollon writes a cut – which atoms are off, cell by cell – and the logits it computes with that cut
(--dump-circuit). Here the same cut is applied to the Hugging Face model, independently of apollon's
code: every projection (Q, K, V, O, gate, up, down; GPT-NeoX: the fused QKV, dense, dense_h_to_4h,
dense_4h_to_h) is recomputed per position from its weights with a keep mask over its scalar products
W[o,c]·x_c, and the eager attention function is replaced by one that drops the switched-off products
q_e·k_e (scores) and a_j·v_j[e] (mixing). Norms, RoPE, softmax, activations and biases stay those of
transformers (in float64). Frozen norms (the default of --circuit): every norm module gets a hook that
multiplies its output by s_full/s_live (LayerNorm: without its bias), with s_full recorded from the full
model – all norm forms are proportional to their scale.

Mean ablation (the default of --circuit): a switched-off product is replaced by its mean over the
reference positions. The means are computed here from transformers alone (inputs of the Linear modules,
q_e·k_e and a_j·v_j[e] over the visible pairs) and compared with apollon's.

Cases per model:
  * random cuts over every kind of atom (--circuit-random SEED): whole sites, units, single atoms,
    keys and dimensions of QKᵀ and A·V, at every position and layer – zero and mean ablation, frozen
    and live norms, with and without --circuit-ref
  * the circuit found by the search (--circuit), frozen and live norms, and its complement (everything
    except the circuit off)
Models: random small Llama (GQA), Qwen3 (q/k norm), Gemma 3 (post-norms, sliding window, softcap,
embedding scale) and the real Pythia-14m (GPT-NeoX: fused QKV, parallel residual, LayerNorm, biases).
  PYTHONPATH=<transformers> python3 tests/compare_circuit.py BIN WORKDIR
"""
import glob, importlib, json, os, shutil, subprocess, sys
import numpy as np, torch
from transformers import AutoModelForCausalLM

BIN, WORK = sys.argv[1], sys.argv[2]
HF = os.path.expanduser("~/.cache/huggingface/hub")
snap = lambda n: sorted(glob.glob(f"{HF}/models--{n.replace('/', '--')}/snapshots/*/config.json"))[-1].rsplit("/", 1)[0]
TEXT = "Montag, Dienstag, Mittwoch, Donnerstag,"
REFS = ["Januar, Februar, März", "eins, zwei, drei, vier"]
SITES = ["Q", "K", "V", "O", "gate", "up", "down"]
LAST = {}
ROW_WISE = {"Q", "K", "V", "gate", "up"}


def save(m, name):
    d = os.path.join(WORK, f"circuit-{name}")
    shutil.rmtree(d, ignore_errors=True)
    m.save_pretrained(d, safe_serialization=True)
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        shutil.copy(os.path.join(snap("EleutherAI/pythia-14m"), f), d)
    return d


def tiny(kind):
    from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM, Gemma3TextConfig, Gemma3ForCausalLM
    torch.manual_seed({"llama": 21, "qwen3": 22, "gemma3": 23}[kind])
    common = dict(vocab_size=50304, hidden_size=64, intermediate_size=96, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)
    if kind == "llama":
        m = LlamaForCausalLM(LlamaConfig(**common, rope_theta=5e5, tie_word_embeddings=False))
    elif kind == "qwen3":
        m = Qwen3ForCausalLM(Qwen3Config(**common, head_dim=24, rope_theta=1e6, tie_word_embeddings=False))
    else:
        m = Gemma3ForCausalLM(Gemma3TextConfig(**common, head_dim=16, query_pre_attn_scalar=24, sliding_window=4, final_logit_softcapping=30.0))
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.08)
    return save(m, kind)


def keep_masks(J, units, d, hd, keys_of):
    """(p, l, site) -> keep mask [units][d]; (p, l, head, 'qk'|'av') -> keep mask [T][hd]"""
    T = len(J["tokens"])
    site, head = {}, {}
    for c in J["cells"]:
        p, l = c["p"], c["l"]
        for s, v in c["sites"].items():
            k = np.ones((units[(l, s)], d), dtype=bool)
            if v["all"]:
                k[:] = False
            for u in v["off"]:
                k[u] = False
            for u, a in v["part"].items():
                k[int(u), a] = False
            site[(p, l, s)] = k
        for h, hv in c["heads"].items():
            for which in ("qk", "av"):
                v = hv[which]
                k = np.ones((T, hd[l]), dtype=bool)
                for j in v["off"]:
                    k[j] = False
                for j, a in v["part"].items():
                    k[int(j), a] = False
                head[(p, l, int(h), which)] = k
    return site, head


def norm_modules(hf):
    return [(n, m) for n, m in hf.named_modules() if "norm" in type(m).__name__.lower() and not list(m.children())]


def scale_of(m, x):
    """1/rms or 1/σ over the last dimension, with the module's eps"""
    eps = getattr(m, "eps", None) or getattr(m, "variance_epsilon", None) or 1e-6
    x = x.double()
    if isinstance(m, torch.nn.LayerNorm):
        return 1.0 / torch.sqrt(x.var(dim=-1, unbiased=False) + eps)
    return 1.0 / torch.sqrt((x * x).mean(dim=-1) + eps)


def record_scales(hf, ids):
    rec, hooks = {}, []
    for n, m in norm_modules(hf):
        def hook(mod, inp, out, n=n):
            rec[n] = scale_of(mod, inp[0]).detach()
        hooks.append(m.register_forward_hook(hook))
    with torch.no_grad():
        hf(torch.tensor([ids]))
    for h in hooks:
        h.remove()
    return rec


def freeze_norms(hf, full):
    """every norm output times s_full/s_live (LayerNorm: its bias excluded); returns the hooks"""
    hooks = []
    for n, m in norm_modules(hf):
        def hook(mod, inp, out, n=n):
            ratio = (full[n] / scale_of(mod, inp[0])).unsqueeze(-1).to(out.dtype)
            b = mod.bias if isinstance(mod, torch.nn.LayerNorm) and mod.bias is not None else None
            return (out - b) * ratio + b if b is not None else out * ratio
        hooks.append(m.register_forward_hook(hook))
    return hooks


def eager_module(kind):
    return importlib.import_module({"llama": "transformers.models.llama.modeling_llama", "qwen3": "transformers.models.qwen3.modeling_qwen3",
                                    "gemma3": "transformers.models.gemma3.modeling_gemma3", "neox": "transformers.models.gpt_neox.modeling_gpt_neox"}[kind])


def linear_sites(ly, neox):
    """(module, sites whose input it is) per layer"""
    if neox:
        a = ly.attention
        return [(a.query_key_value, ["Q", "K", "V"]), (a.dense, ["O"]), (ly.mlp.dense_h_to_4h, ["up"]), (ly.mlp.dense_4h_to_h, ["down"])]
    a = ly.self_attn
    return [(a.q_proj, ["Q"]), (a.k_proj, ["K"]), (a.v_proj, ["V"]), (a.o_proj, ["O"]), (ly.mlp.gate_proj, ["gate"]), (ly.mlp.up_proj, ["up"]), (ly.mlp.down_proj, ["down"])]


def hf_means(hf, kind, J):
    """Mean ablation, computed here from transformers alone, with the rules apollon states: per position of
    the analyzed sequence the mean over the references at that position (position-wise, with
    --circuit-ref; and position 0 if it is <bos>), else the mean over the typical positions of the
    references (apollon's ref_positions: no position 0, no attention sink). Means of the input of every projection and of the factors of QKᵀ and
    A·V (q, k, v as the attention gets them, the attention weights per query/key position; their fallback
    is the mean weight on key 0 or on any other typical key)."""
    neox = kind == "neox"
    layers = hf.gpt_neox.layers if neox else hf.model.layers
    T = len(J["tokens"])
    pw, bos = J["positionwise"], J["bos"]
    acc = {}  # (layer, slot, name) -> [sum, count]; slot = position or "g"
    att = {}  # (layer, slot) -> (sum [H][T or 2], count)
    mod = eager_module(kind)
    orig = mod.eager_attention_forward
    cur = {}

    def put(l, slot, name, x):
        a = acc.setdefault((l, slot, name), [0.0, 0])
        a[0] = a[0] + x
        a[1] += 1

    def slots(p):
        out = []
        if p < T and (pw or (p == 0 and bos and cur["lead"])):
            out.append(p)
        if p in cur["used"]:
            out.append("g")
        return out

    def rec_eager(module, query, key, value, attention_mask, scaling=None, dropout=0.0, softcap=None, **kw):
        l = module.layer_idx
        if scaling is None:
            scaling = module.head_dim ** -0.5
        q = query[0].detach().double().numpy()
        k = key[0].detach().double().numpy()
        v = value[0].detach().double().numpy()
        n = q.shape[1]
        H = q.shape[0]
        mask = attention_mask[0, 0].detach().double().numpy()[:, :n] if attention_mask is not None else np.triu(np.full((n, n), -np.inf), 1)
        kk, vv = np.repeat(k, H // k.shape[0], axis=0), np.repeat(v, H // k.shape[0], axis=0)
        s = np.einsum("hie,hje->hij", q, kk) * scaling
        if softcap is not None:
            s = np.tanh(s / softcap) * softcap
        s = s + mask[None]
        s = s - s.max(axis=-1, keepdims=True)
        a = np.exp(s)
        a = a / a.sum(axis=-1, keepdims=True)
        for p in range(n):
            for sl in slots(p):
                put(l, sl, "q", q[:, p, :])
                put(l, sl, "k", k[:, p, :])
                put(l, sl, "v", v[:, p, :])
                vis = [j for j in range(n) if mask[p, j] == 0]
                if sl == "g":
                    w = att.setdefault((l, "g"), [np.zeros((H, 2)), np.zeros((H, 2))])
                    for j in vis:
                        if j != 0 and j not in cur["used"]:
                            continue
                        c = 0 if j == 0 else 1
                        w[0][:, c] += a[:, p, j]
                        w[1][:, c] += 1
                else:
                    w = att.setdefault((l, sl), [np.zeros((H, T)), np.zeros((H, T))])
                    for j in vis:
                        if j < T:
                            w[0][:, j] += a[:, p, j]
                            w[1][:, j] += 1
        z = np.einsum("hij,hje->hie", a, vv)
        return torch.tensor(z.transpose(1, 0, 2)[None], dtype=query.dtype), torch.tensor(a[None], dtype=query.dtype)

    hooks = []
    for l, ly in enumerate(layers):
        for lin, names in linear_sites(ly, neox):
            def hook(mod_, inp, out, l=l, names=names):
                x = inp[0][0].detach().double().numpy()
                for p in range(x.shape[0]):
                    for sl in slots(p):
                        for nm in names:
                            put(l, sl, nm, x[p])
            hooks.append(lin.register_forward_hook(hook))
    mod.eager_attention_forward = rec_eager
    try:
        with torch.no_grad():
            for toks, used in zip(J["ref_tokens"], J["ref_positions"]):
                cur["used"] = set(used)
                # a leading special token: the analyzed sequence starts with one, and so does this reference
                cur["lead"] = len(toks) > 1 and toks[0] == J["tokens"][0]
                hf(torch.tensor([toks]))
    finally:
        for h in hooks:
            h.remove()
        mod.eager_attention_forward = orig
    L = len(layers)
    H = hf.config.num_attention_heads

    def mean(l, p, name):
        a = acc.get((l, p, name))
        if a is None:
            a = acc[(l, "g", name)]
        return a[0] / a[1]

    x, qm, km, vm, am = {}, {}, {}, {}, {}
    for l in range(L):
        for p in range(T):
            for nm in SITES:
                if (l, "g", nm) in acc:
                    x[(l, p, nm)] = mean(l, p, nm)
            qm[(l, p)], km[(l, p)], vm[(l, p)] = mean(l, p, "q"), mean(l, p, "k"), mean(l, p, "v")
            row = np.zeros((H, T))
            w = att.get((l, p))
            g = att[(l, "g")]
            for h in range(H):
                for j in range(T):
                    if w is not None and w[1][h, j] > 0:
                        row[h, j] = w[0][h, j] / w[1][h, j]
                    else:
                        c = 0 if j == 0 else 1
                        row[h, j] = g[0][h, c] / max(g[1][h, c], 1)
            am[(l, p)] = row
    # how far apollon's own means are from these
    dev = 0.0
    rel = lambda got, ref: float(np.abs(np.array(got).reshape(-1) - np.asarray(ref).reshape(-1)).max() / max(np.abs(ref).max(), 1e-12))
    for l, lm in enumerate(J["means"]):
        for p, pm in enumerate(lm):
            for key, ref in (("h", x[(l, p, "Q")]), ("h2", x[(l, p, "up")]), ("z", x[(l, p, "O")]), ("act", x[(l, p, "down")]), ("q", qm[(l, p)]), ("k", km[(l, p)]), ("v", vm[(l, p)]), ("a", am[(l, p)])):
                d = rel(pm[key], ref)
                if d > 1e-4 and os.environ.get("CIRCUIT_DEBUG"):
                    print(f"    means {key} layer {l} position {p}: {d:.1e}")
                dev = max(dev, d)
    return {"x": x, "q": qm, "k": km, "v": vm, "a": am, "dev": dev}


def apply_cut(hf, kind, J):
    """Hooks + replaced eager attention for the cut in J; returns a function that removes them."""
    # the norm scales of the full model and the means, both before any hook is in place
    full_scales = record_scales(hf, J["tokens"]) if J.get("frozen_norms") else None
    M = hf_means(hf, kind, J) if J.get("ablation") == "mean" else None
    LAST["M"] = M
    cfg = hf.config
    neox = kind == "neox"
    layers = hf.gpt_neox.layers if neox else hf.model.layers
    nh = cfg.num_attention_heads
    nkv = nh if neox else cfg.num_key_value_heads
    hdim = cfg.hidden_size // nh if neox else (getattr(cfg, "head_dim", None) or cfg.hidden_size // nh)
    d, ff = cfg.hidden_size, cfg.intermediate_size
    L = len(layers)
    units = {}
    for l in range(L):
        units.update({(l, "Q"): nh * hdim, (l, "K"): nkv * hdim, (l, "V"): nkv * hdim, (l, "O"): nh * hdim, (l, "gate"): ff, (l, "up"): ff, (l, "down"): ff})
    site, head = keep_masks(J, units, d, {l: hdim for l in range(L)}, None)
    T = len(J["tokens"])
    hooks = []

    def recompute(lin, l, parts):
        """parts: list of (site, out_slice_of_unit) – rows of lin.weight belonging to each unit of the site"""
        W = lin.weight.detach().double().numpy()
        b = lin.bias.detach().double().numpy() if lin.bias is not None else None

        def hook(mod, inp, out):
            x = inp[0][0].detach().double().numpy()  # [T][in]
            y = out[0].detach().double().numpy().copy()
            for p in range(T):
                for s, row_of in parts:
                    k = site.get((p, l, s))
                    if k is None:
                        continue
                    xbar = M["x"][(l, p, s)] if M else None
                    if s in ROW_WISE:
                        for u in range(k.shape[0]):
                            r = row_of(u)
                            xs = x[p] * k[u] if xbar is None else np.where(k[u], x[p], xbar)
                            y[p, r] = (W[r] * xs).sum() + (b[r] if b is not None else 0.0)
                    else:  # column-wise: unit u = input column u, atoms along the outputs
                        X = x[p][None, :] * k.T if xbar is None else np.where(k.T, x[p][None, :], xbar[None, :])
                        y[p] = (W * X).sum(axis=1) + (b if b is not None else 0.0)
            return torch.tensor(y[None], dtype=out.dtype)
        return hook

    for l, ly in enumerate(layers):
        if neox:
            a = ly.attention
            hooks.append(a.query_key_value.register_forward_hook(recompute(a.query_key_value, l, [
                ("Q", lambda u: (u // hdim) * 3 * hdim + u % hdim),
                ("K", lambda u: (u // hdim) * 3 * hdim + hdim + u % hdim),
                ("V", lambda u: (u // hdim) * 3 * hdim + 2 * hdim + u % hdim)])))
            hooks.append(a.dense.register_forward_hook(recompute(a.dense, l, [("O", None)])))
            hooks.append(ly.mlp.dense_h_to_4h.register_forward_hook(recompute(ly.mlp.dense_h_to_4h, l, [("up", lambda u: u)])))
            hooks.append(ly.mlp.dense_4h_to_h.register_forward_hook(recompute(ly.mlp.dense_4h_to_h, l, [("down", None)])))
        else:
            a = ly.self_attn
            for s, lin in (("Q", a.q_proj), ("K", a.k_proj), ("V", a.v_proj)):
                hooks.append(lin.register_forward_hook(recompute(lin, l, [(s, lambda u: u)])))
            hooks.append(a.o_proj.register_forward_hook(recompute(a.o_proj, l, [("O", None)])))
            hooks.append(ly.mlp.gate_proj.register_forward_hook(recompute(ly.mlp.gate_proj, l, [("gate", lambda u: u)])))
            hooks.append(ly.mlp.up_proj.register_forward_hook(recompute(ly.mlp.up_proj, l, [("up", lambda u: u)])))
            hooks.append(ly.mlp.down_proj.register_forward_hook(recompute(ly.mlp.down_proj, l, [("down", None)])))

    mod = eager_module(kind)
    orig = mod.eager_attention_forward

    def eager(module, query, key, value, attention_mask, scaling=None, dropout=0.0, softcap=None, **kw):
        l = module.layer_idx
        if scaling is None:
            scaling = module.head_dim ** -0.5
        q = query[0].detach().double().numpy()  # [H][T][hd]
        k = key[0].detach().double().numpy()  # [KVH][T][hd]
        v = value[0].detach().double().numpy()
        H = q.shape[0]
        rep = H // k.shape[0]
        if attention_mask is not None:
            mask = attention_mask[0, 0].detach().double().numpy()[:, :T]
        else:  # no mask handed over: causal
            mask = np.triu(np.full((T, T), -np.inf), 1)
        z = np.zeros_like(q)
        A = np.zeros((H, T, T))
        for h in range(H):
            kh, vh = k[h // rep], v[h // rep]
            for p in range(T):
                kq = head.get((p, l, h, "qk"))
                prod = q[h, p][None, :] * kh  # [T][hd] – every product q_e·k_e
                if kq is not None:
                    prod = prod * kq if M is None else np.where(kq, prod, M["q"][(l, p)][h][None, :] * np.array([M["k"][(l, j)][h // rep] for j in range(T)]))
                s = prod.sum(axis=1) * scaling
                if softcap is not None:
                    s = np.tanh(s / softcap) * softcap
                s = s + mask[p]
                s = s - s.max()
                e = np.exp(s)
                a = e / e.sum()
                A[h, p] = a
                kv = head.get((p, l, h, "av"))
                contrib = a[:, None] * vh  # [T][hd] – every product a_j·v_j[e]
                if kv is not None:
                    contrib = contrib * kv if M is None else np.where(kv, contrib, np.array([M["a"][(l, p)][h, j] * M["v"][(l, j)][h // rep] for j in range(T)]))
                z[h, p] = contrib.sum(axis=0)
        out = torch.tensor(z.transpose(1, 0, 2)[None], dtype=query.dtype)
        return out, torch.tensor(A[None], dtype=query.dtype)

    mod.eager_attention_forward = eager
    if full_scales is not None:
        hooks.extend(freeze_norms(hf, full_scales))

    def undo():
        for x in hooks:
            x.remove()
        mod.eager_attention_forward = orig
    return undo


def reference(hf, kind, J):
    undo = apply_cut(hf, kind, J)
    try:
        with torch.no_grad():
            return hf(torch.tensor([J["tokens"]])).logits[0, J["focus"]].double().numpy()
    finally:
        undo()


def compare(hfs, kind, J, name, devs=None, label=None):
    """(deviation of apollon from the float64 reference, deviation of the float32 reference from it);
    with mean ablation also records how far apollon's means are from those computed here"""
    ref = reference(hfs[0], kind, J)
    m64 = LAST.get("M")
    r32 = reference(hfs[1], kind, J)
    m32 = LAST.get("M")
    rel = lambda a: float(np.abs(a - ref).max() / max(np.abs(ref).max(), 1e-12))
    if devs is not None and m64 is not None:
        # apollon's means against these, and how far the same means in float32 are from float64
        f32 = 0.0
        for key in ("x", "q", "k", "v", "a"):
            for kk, v64 in m64[key].items():
                v32 = np.asarray(m32[key][kk])
                f32 = max(f32, float(np.abs(v32 - v64).max() / max(np.abs(v64).max(), 1e-12)))
        devs[f"{label}: means"] = (m64["dev"], f32)
    return rel(np.array(J["logits_circuit"], dtype=np.float64)), rel(r32)


def run_case(name, md, kind):
    # the reference runs in float64; the same reference in float32 shows how sensitive each state is to
    # float32 rounding: apollon has to be within max(1e-4, 2 × that) of the float64 reference
    hf = AutoModelForCausalLM.from_pretrained(md, dtype=torch.float64, attn_implementation="eager").eval()
    hf32 = AutoModelForCausalLM.from_pretrained(md, dtype=torch.float32, attn_implementation="eager").eval()
    hfs = (hf, hf32)
    devs = {}
    # without any cut the replaced attention + hooks must reproduce transformers itself
    ids = json.loads(subprocess.run([BIN, "-m", md, "--tokens-only", TEXT], capture_output=True, text=True).stdout)
    with torch.no_grad():
        plain = hf(torch.tensor([ids])).logits[0, -1].double().numpy()
    empty = {"tokens": ids, "focus": len(ids) - 1, "cells": [], "logits_circuit": plain.tolist()}
    # (transformers computes the softmax in float32 even for a float64 model; the replaced attention does not)
    devs["no cut: hooks = transformers"] = compare(hfs, kind, empty, name)
    for seed, norms, abl, refs in ((1, "frozen", "mean", []), (2, "frozen", "zero", []), (3, "live", "mean", REFS), (4, "live", "zero", [])):
        dump = os.path.join(WORK, f"circuit-{name}-rand{seed}.json")
        extra = ["--circuit-ref", "; ".join(refs)] if refs else []
        r = subprocess.run([BIN, "-m", md, "--pos", "-1", "--circuit", "--circuit-norms", norms, "--circuit-ablation", abl, *extra, "--circuit-random", str(seed), "--dump-circuit", dump, "--html", os.path.join(WORK, f"circuit-{name}"), "--no-color", TEXT], capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(r.stderr)
        J = json.load(open(dump))
        on, tot = J["atoms_on"], J["atoms_total"]
        label = f"random cut {seed} ({100 * (1 - on / tot):.0f}% off, {norms} norms, {abl}{' with refs' if refs else ''})"
        devs[label] = compare(hfs, kind, J, name, devs, label)
    for norms, abl in (("frozen", "mean"), ("live", "zero")):
        search(hfs, kind, md, name, norms, abl, devs)
    good = all(a <= max(1e-4, 2 * f32) for a, f32 in devs.values())
    print(f"{'OK    ' if good else 'FAILED'} {name:12s} " + " · ".join(f"{k} {a:.1e} (float32 ref. {f32:.1e})" for k, (a, f32) in devs.items()))
    return good


def search(hfs, kind, md, name, norms, abl, devs):
    hf = hfs[0]
    dump = os.path.join(WORK, f"circuit-{name}-search-{norms}-{abl}.json")
    extra = ["--circuit-ref", "; ".join(REFS)] if abl == "mean" else []
    r = subprocess.run([BIN, "-m", md, "--pos", "-1", "--circuit", "--circuit-norms", norms, "--circuit-ablation", abl, *extra, "--circuit-kl", "0.5", "--dump-circuit", dump, "--html", os.path.join(WORK, f"circuit-{name}"), "--no-color", TEXT], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr)
    J = json.load(open(dump))
    label = f"search, {norms} norms, {abl}: circuit ({J['atoms_on']} of {J['atoms_total']} atoms on)"
    devs[label] = compare(hfs, kind, J, name, devs, label)
    # complement: every atom of the circuit off, everything else on
    C = dict(J)
    C["logits_circuit"] = J["logits_complement"]
    cells = []
    T = len(J["tokens"])
    L = len(hf.gpt_neox.layers if kind == "neox" else hf.model.layers)
    cfg = hf.config
    nh = cfg.num_attention_heads
    nkv = nh if kind == "neox" else cfg.num_key_value_heads
    hdim = cfg.hidden_size // nh if kind == "neox" else (getattr(cfg, "head_dim", None) or cfg.hidden_size // nh)
    d, ff = cfg.hidden_size, cfg.intermediate_size
    n_units = {"Q": nh * hdim, "K": nkv * hdim, "V": nkv * hdim, "O": nh * hdim, "gate": ff, "up": ff, "down": ff}
    window = getattr(cfg, "sliding_window", None)
    types = getattr(cfg, "layer_types", None)
    by = {(c["p"], c["l"]): c for c in J["cells"]}
    for p in range(T):
        for l in range(L):
            c = by.get((p, l), {"sites": {}, "heads": {}})
            sites = {}
            for s in SITES:
                if kind == "neox" and s == "gate":
                    continue
                v = c["sites"].get(s, {"all": False, "off": [], "part": {}})
                keep = np.ones((n_units[s], d), dtype=bool)
                if v["all"]:
                    keep[:] = False
                for u in v["off"]:
                    keep[u] = False
                for u, a in v["part"].items():
                    keep[int(u), a] = False
                off = [int(u) for u in range(n_units[s]) if keep[u].all()]
                part = {str(u): [int(x) for x in np.nonzero(keep[u])[0]] for u in range(n_units[s]) if keep[u].any() and not keep[u].all()}
                sites[s] = {"all": False, "off": off, "part": part}
            lo = 0
            if window and (types is None or types[l] == "sliding_attention"):
                lo = max(0, p + 1 - window)
            heads = {}
            for h in range(nh):
                hv = c["heads"].get(str(h), {"qk": {"off": [], "part": {}}, "av": {"off": [], "part": {}}})
                e = {}
                for which in ("qk", "av"):
                    v = hv[which]
                    keep = np.ones((T, hdim), dtype=bool)
                    for j in v["off"]:
                        keep[j] = False
                    for j, a in v["part"].items():
                        keep[int(j), a] = False
                    e[which] = {"off": [j for j in range(lo, p + 1) if keep[j].all()], "part": {str(j): [int(x) for x in np.nonzero(keep[j])[0]] for j in range(lo, p + 1) if keep[j].any() and not keep[j].all()}}
                heads[str(h)] = e
            if p <= J["focus"]:
                cells.append({"p": p, "l": l, "sites": sites, "heads": heads})
    C["cells"] = cells
    devs[f"search, {norms} norms, {abl}: everything except the circuit"] = compare(hfs, kind, C, name)


ok = True
cases = [("llama", tiny("llama"), "llama"), ("qwen3", tiny("qwen3"), "qwen3"), ("gemma3", tiny("gemma3"), "gemma3"), ("pythia-14m", snap("EleutherAI/pythia-14m"), "neox")]
for name, md, kind in cases:
    ok &= run_case(name, md, kind)
sys.exit(0 if ok else 1)
