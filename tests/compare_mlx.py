"""MLX 4-bit unpacking against Apple's mlx (mx.dequantize): embedding rows of the prompt tokens.
apollon writes them to the dump as the residual after the embedding (x embedding scale).
  PYTHONPATH=<mlx> python3 tests/compare_mlx.py BIN WORKDIR MODEL"""
import glob, json, os, struct, subprocess, sys
import numpy as np, mlx.core as mx
BIN, WORK, MODEL = sys.argv[1:4]
snap = [s for s in sorted(glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{MODEL.replace('/', '--')}/snapshots/*/")))
        if glob.glob(s + "*.safetensors")][-1]
cfg = json.load(open(snap + "config.json"))
q = cfg["quantization"]
w = mx.load(glob.glob(snap + "*.safetensors")[0])
pre = [k for k in w if k.endswith("embed_tokens.weight")][0][: -len("weight")]
out = os.path.join(WORK, "mlx.bin")
r = subprocess.run([BIN, "-m", MODEL, "--dump", out, "Hello world, how are you today?"], capture_output=True, text=True)
b = open(out, "rb").read(); assert b[:4] == b"APLN"; t, nh, d, v = struct.unpack("<4I", b[4:20])
ids = list(struct.unpack(f"<{t}I", b[20:20 + 4 * t]))
mine = np.frombuffer(b[20 + 4 * t:], dtype="<f4")[: t * d].reshape(t, d)
idx = mx.array(ids)
# pass scales/biases as float32, otherwise mx.dequantize computes in bf16 and rounds on its own
ref = mx.dequantize(w[pre + "weight"][idx], w[pre + "scales"][idx].astype(mx.float32), w[pre + "biases"][idx].astype(mx.float32),
                    group_size=q["group_size"], bits=q["bits"])
ref = np.array(ref.astype(mx.float32)) * np.float32(np.sqrt(d))
err = float(np.abs(mine - ref).max() / np.abs(ref).max())
print(f"{'OK    ' if err < 1e-5 else 'FAILED'} {MODEL}: {t} embedding rows, max. rel. deviation from mx.dequantize {err:.2e}")
sys.exit(0 if err < 1e-5 else 1)
