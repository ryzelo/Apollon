"""Compares apollon with Ollama's logprobs for several prompts: the top-5 next-token probabilities
(sorted) must agree. Needs a running Ollama server (http://localhost:11434).
  python3 tests/compare_ollama.py BIN WORKDIR [MODEL]"""
import json, math, os, struct, subprocess, sys, urllib.request
import numpy as np
BIN, WORK = sys.argv[1], sys.argv[2]
MODEL = sys.argv[3] if len(sys.argv) > 3 else "qwen2.5:3b"
BASE = ["The capital of France is", "The quick brown fox jumps over the", "def fibonacci(n):\n    if n <", "1, 2, 3, 4, 5,", "Today I really don't feel like"]
CHAT = "<|im_start|>user\nWhat is your name?<|im_end|>\n<|im_start|>assistant\n" if "qwen" in MODEL else "<start_of_turn>user\nWhat is your name?<end_of_turn>\n<start_of_turn>model\n"
worst = 0
for p in BASE + [CHAT]:
    req = urllib.request.Request("http://localhost:11434/api/generate", data=json.dumps({"model": MODEL, "prompt": p, "raw": True, "stream": False, "logprobs": True, "top_logprobs": 5, "options": {"temperature": 0, "num_predict": 1}}).encode())
    ref = sorted((math.exp(t["logprob"]) for t in json.load(urllib.request.urlopen(req))["logprobs"][0]["top_logprobs"]), reverse=True)
    out = os.path.join(WORK, "ollama.bin")
    subprocess.run([BIN, "-m", MODEL, "--dump", out, p], capture_output=True, check=True)
    b = open(out, "rb").read(); assert b[:4] == b"APLN"; t, nh, d, v = struct.unpack("<4I", b[4:20])
    lg = np.frombuffer(b[20 + 4 * t:], dtype="<f4")[nh * t * d:].reshape(t, v)[-1].astype(np.float64)
    pr = np.exp(lg - lg.max()); pr /= pr.sum()
    mine = sorted(pr, reverse=True)[:len(ref)]
    dev = max(abs(a - b) for a, b in zip(ref, mine))
    worst = max(worst, dev)
    print(f"{p[:40]!r:44} ollama {' '.join(f'{x:.4f}' for x in ref)}  apollon {' '.join(f'{x:.4f}' for x in mine)}  diff {dev:.4f}")
print("max. deviation of the probabilities:", round(worst, 4))
