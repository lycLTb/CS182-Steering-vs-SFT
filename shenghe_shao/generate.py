"""Generate responses for a prompt file, with or without a trained soft prompt.
Output is jsonl (input row + "response"), ready for the DeepSeek judge.

    # base model (no intervention) baseline
    python generate.py --model $WORK/cs182_assets/models/Qwen3.5-0.8B \
        --input $WORK/cs182_assets/splits/dev.jsonl --output $WORK/outputs/base_dev.jsonl

    # with soft prompt
    python generate.py --model ... --soft_prompt $WORK/runs/sp_n20_lr5e-3/soft_prompt_final.pt \
        --input ... --output $WORK/outputs/sp_dev.jsonl
"""
import argparse
import json
import os
import time

import torch

from soft_prompt import SoftPrompt, generate, load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--soft_prompt", default=None)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--prompt_key", default="prompt")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    model, tok = load_model(args.model)
    soft = None
    if args.soft_prompt:
        soft, meta = SoftPrompt.load(args.soft_prompt)
        soft = soft.cuda()
        print("loaded soft prompt:", meta)

    with open(args.input, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    method = "soft_prompt" if soft is not None else "base"
    t0 = time.perf_counter()
    with open(args.output, "w", encoding="utf-8") as out:
        for i in range(0, len(rows), args.batch_size):
            batch = rows[i: i + args.batch_size]
            responses = generate(model, tok, soft, [r[args.prompt_key] for r in batch], args.max_new_tokens)
            for r, resp in zip(batch, responses):
                out.write(json.dumps({**r, "method": method, "response": resp}, ensure_ascii=False) + "\n")
            out.flush()
            print(f"{min(i + args.batch_size, len(rows))}/{len(rows)}  ({time.perf_counter() - t0:.0f}s)")
    print("done ->", args.output)


if __name__ == "__main__":
    main()
