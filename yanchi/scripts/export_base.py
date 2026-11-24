"""Export the pinned base model to models/qwen3.5-0.8b-base in the same format every fine-tuned model uses.

The HF repo is a multimodal checkpoint (Qwen3_5ForConditionalGeneration, with a vision tower). We only use the text
model, so we load it with AutoModelForCausalLM and `save_pretrained` it — exactly how an SFT / LoRA-merged model is
saved. Weights are bf16 in the repo, so this is lossless.

Usage: python scripts/export_base.py [--out models/qwen3.5-0.8b-base] [--force]
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make yanchi/core importable

from core.model import BASE_REVISION, MODEL_ID, stop_token_ids, weights_sha256
from core.paths import BASE_DIR
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(BASE_DIR))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        if not a.force:
            raise SystemExit(f"{out} exists (use --force to overwrite)")
        shutil.rmtree(out)

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=BASE_REVISION)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=BASE_REVISION, dtype=torch.bfloat16)
    assert all(p.dtype == torch.bfloat16 for p in model.parameters())
    # repo has no generation_config.json; derived default stops only at <|endoftext|>. Make <|im_end|> a stop
    # token too, so other tools (lm-eval generation, teammates' scripts) end turns correctly.
    model.generation_config.eos_token_id = stop_token_ids(tok)
    model.generation_config.pad_token_id = tok.pad_token_id
    model.save_pretrained(out)
    tok.save_pretrained(out)  # includes chat_template.jinja

    prov = {"method": "base", "source": MODEL_ID, "revision": BASE_REVISION, "class": type(model).__name__,
            "dtype": "bfloat16", "n_params": sum(p.numel() for p in model.parameters()),
            "transformers": transformers.__version__, "weights_sha256": weights_sha256(out)}
    json.dump(prov, open(out / "provenance.json", "w"), indent=2)
    print(json.dumps(prov, indent=2))
    print("files:", sorted(f.name for f in out.iterdir()))


if __name__ == "__main__":
    main()
