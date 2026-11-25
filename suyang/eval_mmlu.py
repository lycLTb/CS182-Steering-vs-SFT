"""Score 0-shot MMLU with baseline or DoM next-token logits."""
import argparse
from contextlib import nullcontext
import glob
import json
import os
from collections import defaultdict

import pandas as pd
import torch

from shenghe_shao.soft_prompt import build_train_batch, format_prompt

from .dom import DoM, load_model

LETTERS = ["A", "B", "C", "D"]


def load_mmlu(mmlu_dir):
    files = sorted(glob.glob(os.path.join(mmlu_dir, "all", "test-*.parquet")))
    if not files:
        raise FileNotFoundError(f"no all/test-*.parquet under {mmlu_dir}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def make_question(row):
    opts = "\n".join(f"{l}. {c}" for l, c in zip(LETTERS, row["choices"]))
    return (f"The following is a multiple choice question about {row['subject'].replace('_', ' ')}.\n\n"
            f"{row['question']}\n{opts}\n\nAnswer with only the letter (A, B, C, or D).")


@torch.no_grad()
def last_token_logits(model, tok, prompts, device):
    """Return logits at each row's last real prompt token."""
    batch = [{"prompt": p, "response": ""} for p in prompts]
    # build_train_batch appends the end-of-turn token to the (empty) response; strip it again
    input_ids, mask, _ = build_train_batch(tok, batch, max_len=4096, device=device)
    lengths = mask.sum(1) - 1  # drop the trailing <|im_end|>
    mask = (torch.arange(mask.size(1), device=device)[None] < lengths[:, None]).long()
    emb = model.get_input_embeddings()(input_ids)
    logits = model(inputs_embeds=emb, attention_mask=mask).logits
    idx = lengths - 1
    return logits[torch.arange(logits.size(0), device=device), idx].float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mmlu_dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dom")
    ap.add_argument("--dom_strength", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="random subset for quick checks (0 = all 14k)")
    ap.add_argument("--device", choices=("cuda", "mps", "cpu"), default="cuda")
    args = ap.parse_args()

    dom = DoM.load(args.dom, args.dom_strength) if args.dom else None
    model, tok = load_model(args.model, device=args.device)
    tok.padding_side = "right"

    df = load_mmlu(args.mmlu_dir)
    if args.limit:
        df = df.sample(n=args.limit, random_state=0).reset_index(drop=True)
    letter_ids = [tok(l, add_special_tokens=False).input_ids[0] for l in LETTERS]

    # sanity check: the scored position should be right after the assistant header
    print("---- example prompt ----\n" + format_prompt(tok, make_question(df.iloc[0])) + "<<score here>>")

    correct, per_subject = 0, defaultdict(lambda: [0, 0])
    intervention = dom.apply(model) if dom is not None else nullcontext()
    with intervention:
        for i in range(0, len(df), args.batch_size):
            rows = df.iloc[i: i + args.batch_size]
            prompts = [make_question(row) for _, row in rows.iterrows()]
            logits = last_token_logits(model, tok, prompts, args.device)
            preds = logits[:, letter_ids].argmax(1).tolist()
            for (_, r), p in zip(rows.iterrows(), preds):
                ok = int(p == int(r["answer"]))
                correct += ok
                per_subject[r["subject"]][0] += ok
                per_subject[r["subject"]][1] += 1
            if (i // args.batch_size) % 50 == 0:
                done = min(i + args.batch_size, len(df))
                print(f"{done}/{len(df)}  running acc = {correct / done:.4f}")

    result = {
        "method": "base",
        "n": len(df),
        "accuracy": correct / len(df),
        "per_subject": {s: c / n for s, (c, n) in sorted(per_subject.items())},
    }
    if dom is not None:
        result.update(method="dom_l11", config_id=dom.config_id)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"MMLU accuracy = {result['accuracy']:.4f}  (n={len(df)})  -> {args.output}")


if __name__ == "__main__":
    main()
