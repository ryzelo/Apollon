"""(1) Dequantization of all block formats against the gguf reference package (random blocks).
(2) Tokenizer from the Qwen2.5 GGUF against Qwen's original tokenizer.json.
  PYTHONPATH=<transformers>:<gguf> python3 tests/compare_dequant_qwen_tok.py BIN WORKDIR"""
import json, os, subprocess, sys
import numpy as np, gguf
from gguf.quants import dequantize
BIN, WORK = sys.argv[1], sys.argv[2]
ok = True
rng = np.random.default_rng(0)
for tid, name in [(2, "Q4_0"), (3, "Q4_1"), (6, "Q5_0"), (7, "Q5_1"), (8, "Q8_0"), (10, "Q2_K"), (11, "Q3_K"), (12, "Q4_K"),
                  (13, "Q5_K"), (14, "Q6_K"), (20, "IQ4_NL"), (23, "IQ4_XS"), (39, "MXFP4"), (1, "F16"), (30, "BF16"), (0, "F32")]:
    qt = gguf.GGMLQuantizationType(tid)
    be, bb = gguf.GGML_QUANT_SIZES[qt]
    raw = rng.integers(0, 256, size=(512 * bb,), dtype=np.uint8)
    i, o = os.path.join(WORK, "dq.in"), os.path.join(WORK, "dq.out")
    raw.tofile(i)
    subprocess.run([BIN, "--selftest-dequant", str(tid), i, o], check=True)
    mine = np.fromfile(o, dtype="<f4")
    with np.errstate(all="ignore"):
        ref = dequantize(raw.reshape(1, -1), qt).reshape(-1).astype(np.float32)
    same = np.array_equal(mine, ref, equal_nan=True)
    ok &= same
    print(f"{'OK    ' if same else 'FAILED'} dequantization {name:7s} {len(mine):6d} values bit-identical: {same}")

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
tj = hf_hub_download("Qwen/Qwen2.5-3B-Instruct", "tokenizer.json", cache_dir=os.path.join(WORK, "hfcache"))
qt = Tokenizer.from_file(tj)
# test inputs are deliberately multilingual
TEXTS = ["Die Hauptstadt von Frankreich ist", "Hello world!  Two  spaces\nand a newline\n\n\n", "Zahlen 12345678 und Ümläute: äöüß €",
         "def f(x):\n\treturn x**2  # Kommentar", "emoji 🙂🎉 und 中文日本語", "<|im_start|>user\nHi<|im_end|>\n", "I'm sure we'll SEE   it's   ok",
         "   führende Leerzeichen", "URL: https://example.com/a?b=c&d=e", "Tabs\t\tund\r\nWindows-Zeilen"]
probe = subprocess.run([BIN, "-m", "qwen2.5:3b", "--tokens-only", "x"], capture_output=True, text=True)
if probe.returncode:
    print("SKIP   Qwen tokenizer (the Ollama model qwen2.5:3b is not available)")
    TEXTS = []
for s in TEXTS:
    mine = json.loads(subprocess.run([BIN, "-m", "qwen2.5:3b", "--tokens-only", s], capture_output=True, text=True).stdout)
    ref = qt.encode(s).ids
    ok &= mine == ref
    print(f"{'OK    ' if mine == ref else 'FAILED'} Qwen tokenizer {s!r:48} {len(ref)} tokens")
    if mine != ref: print("   apollon  ", mine, "\n   reference", ref)
sys.exit(0 if ok else 1)
