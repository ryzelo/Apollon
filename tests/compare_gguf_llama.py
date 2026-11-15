"""Converts tiny-random-Llama (safetensors + tokenizer.model) to GGUF the way llama.cpp does
(q/k permuted for pairwise RoPE, SentencePiece vocabulary with scores) and compares
apollon on this GGUF with transformers: tokens + logits.
  PYTHONPATH=<transformers>:<gguf> python3 tests/compare_gguf_llama.py BIN WORKDIR"""
import glob, json, os, struct, subprocess, sys
import numpy as np, torch, gguf
from safetensors.numpy import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
BIN, WORK = sys.argv[1], sys.argv[2]
src = sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--hf-internal-testing--tiny-random-LlamaForCausalLM/snapshots/*/config.json")))[-1].rsplit("/", 1)[0]
cfg = json.load(open(f"{src}/config.json"))
W = load_file(f"{src}/model.safetensors")

def pb_pieces(buf):                       # minimal protobuf reader for the sentencepiece ModelProto
    def varint(b, i):
        r = s = 0
        while True:
            x = b[i]; i += 1; r |= (x & 0x7F) << s; s += 7
            if x < 0x80: return r, i
    i, out = 0, []
    while i < len(buf):
        key, i = varint(buf, i); f, wt = key >> 3, key & 7
        if wt == 2:
            n, i = varint(buf, i); payload = buf[i:i + n]; i += n
            if f == 1:
                piece, score, typ, j = "", 0.0, 1, 0
                while j < len(payload):
                    k2, j = varint(payload, j); f2, w2 = k2 >> 3, k2 & 7
                    if w2 == 2:
                        m, j = varint(payload, j); piece = payload[j:j + m].decode(); j += m
                    elif w2 == 5:
                        score = struct.unpack("<f", payload[j:j + 4])[0]; j += 4
                    elif w2 == 0:
                        typ, j = varint(payload, j)
                out.append((piece, score, typ))
        elif wt == 0: _, i = varint(buf, i)
        elif wt == 5: i += 4
        elif wt == 1: i += 8
    return out

pieces = pb_pieces(open(f"{src}/tokenizer.model", "rb").read())
nh, nkv, d = cfg["num_attention_heads"], cfg.get("num_key_value_heads") or cfg["num_attention_heads"], cfg["hidden_size"]
def permute(w, n_head):                    # as in convert_hf_to_gguf.py (LlamaModel.permute)
    return w.reshape(n_head, 2, w.shape[0] // n_head // 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)
path = os.path.join(WORK, "tiny-llama.gguf")
w = gguf.GGUFWriter(path, "llama")
w.add_block_count(cfg["num_hidden_layers"]); w.add_context_length(cfg["max_position_embeddings"])
w.add_embedding_length(d); w.add_feed_forward_length(cfg["intermediate_size"]); w.add_head_count(nh); w.add_head_count_kv(nkv)
w.add_layer_norm_rms_eps(cfg["rms_norm_eps"]); w.add_rope_freq_base(cfg.get("rope_theta", 10000.0)); w.add_rope_dimension_count(d // nh)
w.add_tokenizer_model("llama"); w.add_token_list([p[0] for p in pieces]); w.add_token_scores([p[1] for p in pieces])
w.add_token_types([p[2] for p in pieces]); w.add_bos_token_id(1); w.add_eos_token_id(2); w.add_add_bos_token(True)
names = {"model.embed_tokens.weight": "token_embd.weight", "model.norm.weight": "output_norm.weight", "lm_head.weight": "output.weight"}
for k, v in W.items():
    v = v.astype(np.float32)
    if k.startswith("model.layers."):
        i, rest = k.split(".")[2], ".".join(k.split(".")[3:])
        m = {"input_layernorm.weight": "attn_norm.weight", "post_attention_layernorm.weight": "ffn_norm.weight",
             "self_attn.q_proj.weight": "attn_q.weight", "self_attn.k_proj.weight": "attn_k.weight", "self_attn.v_proj.weight": "attn_v.weight",
             "self_attn.o_proj.weight": "attn_output.weight", "mlp.gate_proj.weight": "ffn_gate.weight", "mlp.up_proj.weight": "ffn_up.weight",
             "mlp.down_proj.weight": "ffn_down.weight"}[rest]
        if rest == "self_attn.q_proj.weight": v = permute(v, nh)
        if rest == "self_attn.k_proj.weight": v = permute(v, nkv)
        name = f"blk.{i}.{m}"
    else:
        name = names[k]
    w.add_tensor(name, v)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()

from tokenizers import Tokenizer as RawTok
raw = RawTok.from_file(f"{src}/tokenizer.json")
hf = AutoModelForCausalLM.from_pretrained(src, dtype=torch.float32).eval()
TEXTS = ["Die Hauptstadt von Frankreich ist", "Hello world!  Two  spaces\nand a newline", "Zahlen 12345 und Ümläute: äöüß €", "emoji 🙂 und 中文", " führendes Leerzeichen"]  # deliberately multilingual
ok = True
for s in TEXTS:
    mine = json.loads(subprocess.run([BIN, "-m", path, "--tokens-only", s], capture_output=True, text=True).stdout)
    ref = raw.encode(s).ids   # tokenizer.json semantics = llama.cpp (AutoTokenizer with legacy=False drops a leading space)
    out = os.path.join(WORK, "d.bin")
    subprocess.run([BIN, "-m", path, "--dump", out, s], capture_output=True, check=True)
    b = open(out, "rb").read(); assert b[:4] == b"APLN"; t, nhid, dd, v = struct.unpack("<4I", b[4:20])
    lg = np.frombuffer(b[20 + 4 * t:], dtype="<f4")[nhid * t * dd:].reshape(t, v)
    with torch.no_grad():
        r = hf(torch.tensor([ref])).logits[0].numpy()
    dev = float(np.abs(lg - r).max() / np.abs(r).max()) if mine == ref else float("nan")
    good = mine == ref and dev < 1e-4
    ok &= good
    print(f"{'OK    ' if good else 'FAILED'} {s!r:42} tokens equal: {mine == ref}  logits max. rel. dev. {dev:.2e}")
    if mine != ref: print("   apollon", mine, "\n   hf     ", ref)
sys.exit(0 if ok else 1)
