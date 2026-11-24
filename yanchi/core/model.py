"""Model loading, chat template, greedy batched generation."""
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.paths import BASE_DIR, ROOT

MODEL_ID = "Qwen/Qwen3.5-0.8B"  # HF repo of the base model
BASE_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"  # pinned snapshot
DEFAULT_MODEL = str(BASE_DIR)
MAX_NEW_TOKENS = 512
STOP_TOKENS = ["<|im_end|>", "<|endoftext|>"]


# ------------------------------------------------------------------ loading
def resolve_model(model_path: str) -> str:
    """Local dir (relative to cwd, else to yanchi/, e.g. `models/sft-x`) or, failing that, an HF repo id."""
    p = Path(model_path)
    for cand in (p, ROOT / p):
        if cand.is_dir():
            return str(cand.resolve())
    if BASE_DIR.resolve() in (p.resolve(), (ROOT / p).resolve()):
        raise FileNotFoundError(f"{BASE_DIR} not found — run `python scripts/export_base.py` first")
    if p.is_absolute() or str(model_path).startswith("."):
        raise FileNotFoundError(f"model dir not found: {model_path}")
    return model_path  # HF repo id


def load_model(model_path: str = DEFAULT_MODEL):
    """Load a model dir saved with `save_pretrained` (base or fine-tuned), or an HF repo id."""
    model_path = resolve_model(model_path)
    tok = AutoTokenizer.from_pretrained(model_path)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda").eval()
    return model, tok


def weights_sha256(model_dir) -> str:
    """Hash of all *.safetensors in a model dir — identifies exactly which weights were evaluated."""
    h = hashlib.sha256()
    for f in sorted(Path(model_dir).glob("*.safetensors")):
        h.update(f.name.encode())
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                h.update(chunk)
    return h.hexdigest()


def model_info(model_path: str) -> dict:
    """What gets recorded in every result file about the evaluated model."""
    # causal-conv1d CUDA kernel vs transformers' torch fallback give different greedy outputs in bf16;
    # our runs use the fallback (package not installed) — record it so results from other machines are comparable.
    env = {"causal_conv1d_installed": importlib.util.find_spec("causal_conv1d") is not None}
    p = Path(resolve_model(model_path))
    if not p.is_dir():
        return {"model": model_path, "local": False, **env}
    prov = p / "provenance.json"
    return {"model": str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p), "local": True,
            "weights_sha256": weights_sha256(p), "provenance": json.load(open(prov)) if prov.exists() else None, **env}


# ------------------------------------------------------------------ prompting / generation
def render(tok, messages):
    """Official chat template, non-thinking mode."""
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def stop_token_ids(tok):
    """End-of-turn ids. Passed explicitly to generate() so behaviour never depends on a model dir's
    generation_config.json: the HF repo ships none, and the derived default stops only at <|endoftext|>
    (the base model emits `<|im_end|> \\n <|endoftext|>`, but an SFT model trained to end at <|im_end|> may not)."""
    return [tok.convert_tokens_to_ids(t) for t in STOP_TOKENS]


@torch.no_grad()
def generate_batch(model, tok, texts, max_new_tokens=MAX_NEW_TOKENS):
    """Greedy generation for a list of already-rendered prompts. Returns list of dicts."""
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    eos = stop_token_ids(tok)
    out = model.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens,
                         eos_token_id=eos, pad_token_id=tok.pad_token_id)
    new = out[:, enc.input_ids.shape[1]:].tolist()
    eos_ids = set(eos)
    res = []
    for ids in new:
        n = len(ids)  # strip right padding after the first stop token
        for i, t in enumerate(ids):
            if t in eos_ids:
                n = i + 1
                break
        ids = ids[:n]
        finished = bool(ids) and ids[-1] in eos_ids
        res.append({
            "response": tok.decode(ids, skip_special_tokens=True).strip(),
            "n_new_tokens": len(ids),
            "finish_reason": "stop" if finished else "length",
        })
    return res


class Timer:
    def __enter__(self):
        torch.cuda.synchronize()
        self.t = time.perf_counter()
        return self

    def __exit__(self, *a):
        torch.cuda.synchronize()
        self.s = time.perf_counter() - self.t
