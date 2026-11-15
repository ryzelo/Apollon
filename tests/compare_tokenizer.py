"""Tokenizers of the further model families against the originals (package tokenizers or transformers).
Cases whose model is not available locally are skipped. A Gemma 4 GGUF (e.g. from Ollama) is only
compared when given in the environment variable APOLLON_GEMMA4_GGUF (an Ollama name or a .gguf path).
  PYTHONPATH=<transformers> python3 tests/compare_tokenizer.py BIN"""
import glob, json, os, subprocess, sys
from tokenizers import Tokenizer
from transformers import AutoTokenizer
BIN = sys.argv[1]
HF = os.path.expanduser("~/.cache/huggingface/hub")
def snapfile(repo, f):
    found = sorted(glob.glob(f"{HF}/models--{repo.replace('/', '--')}/snapshots/*/{f}"))
    if not found:
        raise FileNotFoundError(f"{repo}/{f} is not in the Hugging Face cache")
    return found[-1]
# test inputs are deliberately multilingual
TEXTS = ["Die Hauptstadt von Frankreich ist", "Hello world!  Two  spaces\nand a newline\n\n\n", "Zahlen 12345678 und Ümläute: äöüß €",
         "def f(x):\n\treturn x**2  # Kommentar", "emoji 🙂🎉 und 中文日本語", "I'm sure we'll SEE   it's   ok",
         "   führende Leerzeichen", "URL: https://example.com/a?b=c&d=e", "Tabs\t\tund\r\nWindows-Zeilen", "<bos>Spezial<eos> mitten"]
BOS = 2  # <bos> in Gemma
def tok_json(repo, prefix=()):
    t = Tokenizer.from_file(snapfile(repo, "tokenizer.json"))
    return lambda s: list(prefix) + t.encode(s).ids

cases = [  # (model for apollon, reference function (built lazily), description)
    ("mlx-community/gemma-4-e4b-it-4bit", lambda: tok_json("mlx-community/gemma-4-e4b-it-4bit", [BOS]), "Gemma 4 MLX tokenizer.json (+<bos>)"),
    ("mlx-community/gemma-3-1b-it-4bit", lambda: tok_json("mlx-community/gemma-3-1b-it-4bit"), "Gemma 3 tokenizer.json"),
    ("roneneldan/TinyStories-33M", lambda: tok_json("roneneldan/TinyStories-33M"), "GPT-Neo tokenizer.json"),
    ("microsoft/unixcoder-base", lambda: (lambda t: lambda s: t(s)["input_ids"])(AutoTokenizer.from_pretrained(os.path.dirname(snapfile("microsoft/unixcoder-base", "vocab.json")) + "/")),
     "RoBERTa vocab.json+merges.txt vs. transformers"),
]
if os.environ.get("APOLLON_GEMMA4_GGUF"):
    cases.insert(0, (os.environ["APOLLON_GEMMA4_GGUF"], lambda: tok_json("google/gemma-4-E4B-it", [BOS]), "Gemma 4 GGUF vs. Google's tokenizer.json (+<bos> as llama.cpp)"))
ok = True
for model, make_ref, desc in cases:
    try:
        ref = make_ref()
    except FileNotFoundError as e:
        print(f"SKIP   {desc:62s} ({e})")
        continue
    bad = []
    for s in TEXTS:
        mine = json.loads(subprocess.run([BIN, "-m", model, "--tokens-only", s], capture_output=True, text=True).stdout or "null")
        if mine != ref(s):
            bad.append((s, mine, ref(s)))
    ok &= not bad
    print(f"{'OK    ' if not bad else 'FAILED'} {desc:62s} {len(TEXTS) - len(bad)}/{len(TEXTS)} texts equal")
    for s, a, b in bad[:3]:
        print(f"   {s!r}\n     apollon   {a}\n     reference {b}")
sys.exit(0 if ok else 1)
