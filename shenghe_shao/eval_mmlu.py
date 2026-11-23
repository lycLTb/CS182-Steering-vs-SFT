"""0-shot MMLU accuracy, with or without a soft prompt.

The question is chat-formatted (same template as training); we compare the
model's next-token logits for "A"/"B"/"C"/"D" right after the assistant header.
No judge needed.

    CUDA_VISIBLE_DEVICES=1 python eval_mmlu.py --model $WORK/cs182_assets/models/Qwen3.5-0.8B \
        --mmlu_dir $WORK/cs182_assets/data/mmlu --output $WORK/outputs/mmlu_base.json
    # add --soft_prompt .../soft_prompt_final.pt to evaluate a trained soft prompt
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import pandas as pd
import torch

from soft_prompt import SoftPrompt, build_train_batch, format_prompt, load_model

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
def last_token_logits(model, tok, soft, prompts, device):
    """Right-padded batch: [soft][prompt][pad]; return logits at each row's last real token."""
    batch = [{"prompt": p, "response": ""} for p in prompts]
    # build_train_batch appends the end-of-turn token to the (empty) response; strip it again
    input_ids, mask, _ = build_train_batch(tok, batch, max_len=4096, device=device)
    lengths = mask.sum(1) - 1  # drop the trailing <|im_end|>
    mask = (torch.arange(mask.size(1), device=device)[None] < lengths[:, None]).long()
    emb = model.get_input_embeddings()(input_ids)
    n_soft = 0
    if soft is not None:
        n_soft = soft.n_tokens
        sp = soft.embedding.to(device=emb.device, dtype=emb.dtype).unsqueeze(0).expand(emb.size(0), -1, -1)
        emb = torch.cat([sp, emb], 1)
        mask = torch.cat([mask.new_ones(mask.size(0), n_soft), mask], 1)
    logits = model(inputs_embeds=emb, attention_mask=mask).logits
    idx = n_soft + lengths - 1
    return logits[torch.arange(logits.size(0), device=device), idx].float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mmlu_dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--soft_prompt", default=None)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="random subset for quick checks (0 = all 14k)")
    args = ap.parse_args()

    model, tok = load_model(args.model)
    tok.padding_side = "right"
    soft = None
    if args.soft_prompt:
        soft, meta = SoftPrompt.load(args.soft_prompt)
        soft = soft.cuda()
        print("loaded soft prompt:", meta)

    df = load_mmlu(args.mmlu_dir)
    if args.limit:
        df = df.sample(n=args.limit, random_state=0).reset_index(drop=True)
    letter_ids = [tok(l, add_special_tokens=False).input_ids[0] for l in LETTERS]

    # sanity check: the scored position should be right after the assistant header
    print("---- example prompt ----\n" + format_prompt(tok, make_question(df.iloc[0])) + "<<score here>>")

    correct, per_subject = 0, defaultdict(lambda: [0, 0])
    for i in range(0, len(df), args.batch_size):
        rows = df.iloc[i: i + args.batch_size]
        logits = last_token_logits(model, tok, soft, [make_question(r) for _, r in rows.iterrows()], "cuda")
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
        "method": "soft_prompt" if soft is not None else "base",
        "soft_prompt": args.soft_prompt,
        "n": len(df),
        "accuracy": correct / len(df),
        "per_subject": {s: c / n for s, (c, n) in sorted(per_subject.items())},
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"MMLU accuracy = {result['accuracy']:.4f}  (n={len(df)})  -> {args.output}")


if __name__ == "__main__":
    main()
