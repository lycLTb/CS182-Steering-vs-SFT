"""Train a soft prompt on a frozen model with the supervised (response-only) objective.

    CUDA_VISIBLE_DEVICES=0 python train_soft_prompt.py \
        --model $WORK/cs182_assets/models/Qwen3.5-0.8B \
        --train_file $WORK/cs182_assets/splits/train.jsonl \
        --out_dir $WORK/runs/sp_n20_lr5e-3 \
        --time_budget_min 40 --save_at_min 10,20,40

`--save_at_min` saves a checkpoint at each budget level within one run (the LR is
constant after warmup, so a checkpoint at t equals a run stopped at t).
Writes: soft_prompt_<min>min.pt, soft_prompt_final.pt, train_log.jsonl, metrics.json
"""
import argparse
import json
import os
import random
import time

import torch

from soft_prompt import DEFAULT_INIT_TEXT, SoftPrompt, build_train_batch, forward_with_soft_prompt, load_model


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_tokens", type=int, default=20)
    ap.add_argument("--init_text", default=DEFAULT_INIT_TEXT)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--warmup_steps", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--max_len", type=int, default=1536)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=0, help="0 = no limit")
    ap.add_argument("--time_budget_min", type=float, default=0, help="0 = no limit")
    ap.add_argument("--save_at_min", default="", help="e.g. 10,20,40")
    ap.add_argument("--save_at_examples", default="", help="data budgets, e.g. 1000,2000,4000")
    ap.add_argument("--grad_checkpointing", action="store_true", help="trade speed for much lower memory")
    ap.add_argument("--untimed_warmup_batches", type=int, default=3,
                    help="fwd+bwd passes before the clock starts (kernel JIT/autotune); no update")
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda"

    model, tok = load_model(args.model, device)
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()  # HF only checkpoints in train mode; weights stay frozen (no dropout in Qwen)
    data = load_jsonl(args.train_file)
    soft = SoftPrompt.from_text(model, tok, args.n_tokens, args.init_text).to(device)
    n_params = soft.embedding.numel()
    print(f"train examples: {len(data)} | trainable params: {n_params}")

    # sanity check: show exactly what the model is trained on
    ids, _, labels = build_train_batch(tok, data[:1], args.max_len, device)
    print("---- example input ----\n" + tok.decode(ids[0]))
    print("---- supervised part ----\n" + tok.decode(ids[0][labels[0] != -100]) + "\n----")

    opt = torch.optim.AdamW([soft.embedding], lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, args.warmup_steps)))
    save_at = sorted(float(x) for x in args.save_at_min.split(",") if x.strip())
    save_at_ex = sorted(int(x) for x in args.save_at_examples.split(",") if x.strip())
    budget_s = args.time_budget_min * 60

    def save(tag, elapsed, step, seen, tokens):
        path = os.path.join(args.out_dir, f"soft_prompt_{tag}.pt")
        soft.save(path, model=args.model, n_tokens=args.n_tokens, step=step,
                  train_time_sec=elapsed, examples_seen=seen, tokens_seen=tokens)
        print(f"[save] {path}")

    for k in range(args.untimed_warmup_batches):
        batch = data[k * args.batch_size: (k + 1) * args.batch_size]
        input_ids, mask, labels = build_train_batch(tok, batch, args.max_len, device)
        forward_with_soft_prompt(model, soft, input_ids, mask, labels).loss.backward()
        soft.embedding.grad = None
    print(f"untimed warmup: {args.untimed_warmup_batches} batches done")

    log = open(os.path.join(args.out_dir, "train_log.jsonl"), "w")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    step, seen, tokens, micro, done = 0, 0, 0, 0, False

    for epoch in range(args.epochs):
        random.shuffle(data)
        for i in range(0, len(data), args.batch_size):
            batch = data[i: i + args.batch_size]
            input_ids, mask, labels = build_train_batch(tok, batch, args.max_len, device)
            loss = forward_with_soft_prompt(model, soft, input_ids, mask, labels).loss
            (loss / args.grad_accum).backward()
            micro += 1
            seen += len(batch)
            tokens += int(mask.sum())
            if micro % args.grad_accum:
                continue

            torch.nn.utils.clip_grad_norm_([soft.embedding], 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            elapsed = time.perf_counter() - t0  # loss.item() below syncs; fine at log granularity

            if step % args.log_every == 0:
                rec = {"step": step, "epoch": epoch, "loss": loss.item(), "lr": sched.get_last_lr()[0],
                       "elapsed_sec": round(elapsed, 1), "examples_seen": seen, "tokens_seen": tokens}
                print(rec)
                log.write(json.dumps(rec) + "\n")
                log.flush()

            while save_at and elapsed >= save_at[0] * 60:
                save(f"{save_at.pop(0):g}min", elapsed, step, seen, tokens)
            while save_at_ex and seen >= save_at_ex[0]:
                save(f"{save_at_ex.pop(0)}ex", elapsed, step, seen, tokens)
            if (args.max_steps and step >= args.max_steps) or (budget_s and elapsed >= budget_s):
                done = True
                break
        if done:
            break

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    save("final", elapsed, step, seen, tokens)
    metrics = {
        "method": "soft_prompt",
        "train_time_sec": round(elapsed, 1),
        "peak_mem_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "trainable_params": n_params,
        "steps": step,
        "examples_seen": seen,
        "tokens_seen": tokens,
        "gpu": torch.cuda.get_device_name(0),
        "args": vars(args),
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
