"""Fit the additive safety direction on the shared WildJailbreak train split."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import torch

from shenghe_shao.soft_prompt import format_prompt

from .dom import BENIGN, HARMFUL, LAYER_NUMBER, block_hook, hidden_states, load_model, sha256


@torch.inference_mode()
def fit(model, tokenizer, rows, batch_size):
    tokenizer.padding_side = "left"
    device = model.get_input_embeddings().weight.device
    sums = {}
    counts = Counter()
    last_token_states = []

    def capture(_module, _inputs, output):
        # Left padding puts the last real chat-template token at column -1.
        last_token_states.append(hidden_states(output)[:, -1].to(torch.float32).clone())

    with block_hook(model, capture):
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            prompts = [format_prompt(tokenizer, row["prompt"]) for row in batch]
            encoded = tokenizer(
                prompts, return_tensors="pt", padding=True, add_special_tokens=False
            ).to(device)
            model.model(**encoded, use_cache=False)
            (activations,) = last_token_states
            last_token_states.clear()
            for row, activation in zip(batch, activations, strict=True):
                kind = row["data_type"]
                if kind not in sums:
                    sums[kind] = torch.zeros_like(activation)
                sums[kind].add_(activation)
                counts[kind] += 1

    def mean(kinds):
        kinds = sorted(kinds)
        return sum(sums[kind] for kind in kinds if kind in sums) / sum(
            counts[kind] for kind in kinds
        )

    delta = mean(HARMFUL) - mean(BENIGN)
    if not torch.isfinite(delta).all() or delta.norm() == 0:
        raise ValueError("Difference of means is zero or nonfinite")
    return delta.cpu(), dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch_size must be positive")

    with args.train.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    kinds = {row["data_type"] for row in rows}
    if kinds - (HARMFUL | BENIGN) or not (kinds & HARMFUL and kinds & BENIGN):
        parser.error("--train must contain known WildJailbreak types from both classes")
    input_sha256 = sha256(args.train)

    model, tokenizer = load_model(args.model, device=args.device)
    synchronizer = {"cuda": torch.cuda.synchronize, "mps": torch.mps.synchronize}
    synchronize = synchronizer.get(args.device)
    if synchronize is not None:
        synchronize()
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    delta, counts = fit(model, tokenizer, rows, args.batch_size)
    if synchronize is not None:
        synchronize()

    metadata = {
        "model": args.model,
        "train_sha256": input_sha256,
        "layer_number": LAYER_NUMBER,
        "token_position": "last_chat_prompt_token",
        "counts": counts,
        "batch_size": args.batch_size,
        "delta_l2": delta.norm().item(),
        "extraction_seconds": time.perf_counter() - start,
        "device": args.device,
        "dtype": str(model.get_input_embeddings().weight.dtype),
        "peak_allocated_cuda_bytes": (
            torch.cuda.max_memory_allocated() if args.device == "cuda" else None
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"delta": delta, **metadata}, args.output)
    print(json.dumps({**metadata, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
