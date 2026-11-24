"""Merge LoRA adapters from a train_lora.py run into evaluable model dirs (save_pretrained format).

    python scripts/merge_lora.py --run runs/lora_r16_lr2e-4                 # every adapter_* in the run
    python scripts/merge_lora.py --run runs/lora_r16_lr2e-4 --tags 1000ex 4000ex
(paths relative to yanchi/)

-> models/lora-<run name>-<tag>/  with weights (bf16), tokenizer, generation_config (stop at <|im_end|> and
<|endoftext|>: the teammate's generate() passes no eos, so this file decides when a fine-tuned model stops)
and provenance.json. These dirs are what every eval script takes as --model.
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.model import STOP_TOKENS, model_info, resolve_model
from core.paths import MODELS


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="train_lora.py out_dir, relative to yanchi/")
    ap.add_argument("--tags", nargs="*", default=None, help="e.g. 1000ex 4000ex (default: all adapter_* dirs)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    run = ROOT / a.run
    metrics = json.load(open(run / "metrics.json"))
    targs = metrics["args"]
    base_path = resolve_model(metrics["base_model"])
    base_info = model_info(base_path)
    adapters = sorted(run.glob("adapter_*")) if a.tags is None else [run / f"adapter_{t}" for t in a.tags]
    tok = AutoTokenizer.from_pretrained(base_path)
    stop_ids = [tok.convert_tokens_to_ids(t) for t in STOP_TOKENS]

    for ad in adapters:
        tag = ad.name.removeprefix("adapter_")
        out = MODELS / f"lora-{run.name}-{tag}"
        if out.exists() and not a.overwrite:
            print(f"[skip] {out} exists (--overwrite to redo)")
            continue
        # fresh base per adapter: merge_and_unload modifies weights in place
        base = AutoModelForCausalLM.from_pretrained(base_path, dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(base, ad).merge_and_unload()  # delta (fp32 adapters) added to bf16 weights
        model.generation_config.eos_token_id = stop_ids
        model.generation_config.pad_token_id = tok.pad_token_id
        if out.exists():
            shutil.rmtree(out)
        model.save_pretrained(out)
        tok.save_pretrained(out)
        state = json.load(open(ad / "train_state.json"))
        prov = {
            "method": "lora",
            "base_model": base_info["model"],
            "base_weights_sha256": base_info.get("weights_sha256"),
            "base_provenance": base_info.get("provenance"),
            "train_file": targs["train_file"],
            "train_file_sha256": sha256_file(targs["train_file"]),
            "data_order": f"random.seed({targs['seed']}); random.shuffle(data) per epoch "
                          "(same as shenghe_shao/train_soft_prompt.py); first examples_seen rows",
            "run_dir": str(run.relative_to(ROOT)),
            "adapter": str(ad.relative_to(ROOT)),
            "budget": state,  # examples_seen, tokens_seen, train_flops, train_time_sec, step, peak mem
            "flops_convention": metrics["flops_convention"],
            "trainable_params": metrics["trainable_params"],
            "hyperparams": {k: targs[k] for k in ("r", "alpha", "dropout", "targets", "lr", "weight_decay",
                                                  "warmup_steps", "batch_size", "grad_accum", "max_len",
                                                  "epochs", "seed", "grad_checkpointing")},
            "gpu": metrics["gpu"],
            "merged": "merge_and_unload into bf16 base weights",
        }
        json.dump(prov, open(out / "provenance.json", "w"), indent=2)
        print(f"[merged] {ad.relative_to(ROOT)} -> {out.relative_to(ROOT)}")
        del model, base


if __name__ == "__main__":
    main()
