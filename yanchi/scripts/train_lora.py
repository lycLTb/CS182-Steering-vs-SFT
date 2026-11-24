"""LoRA SFT on WildJailbreak train.jsonl -- same data, format, order and budget protocol as the
teammate's soft prompt (shenghe_shao/train_soft_prompt.py), only the trainable part differs.

    python scripts/train_lora.py --out_dir runs/lora_r16_lr2e-4
    python scripts/train_lora.py --out_dir runs/smoke --max_steps 5 --save_at_examples 16   # smoke test
(--out_dir is relative to yanchi/)

Shared with the soft-prompt run (so data exposure is matched exactly):
  * train file = <repo>/train.jsonl, `random.seed(seed); random.shuffle(data)` per epoch, 1 epoch
  * batch construction = shenghe_shao/soft_prompt.build_train_batch (chat template, thinking off,
    loss on response + <|im_end|>, prompt truncated to max_len-64, right padding)
  * LR: linear warmup then constant -> the adapter saved after N examples equals a run stopped at N,
    so one run gives every budget level (--save_at_examples)

Writes to out_dir: adapter_<N>ex/ (+ adapter_final/), each with train_state.json; train_log.jsonl; metrics.json.
Merge into evaluable model dirs with scripts/merge_lora.py.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # .../yanchi
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "shenghe_shao"))  # teammate's soft_prompt.py (batching / loading)

import torch
from peft import LoraConfig, get_peft_model

from core.flops import FlopCounter
from core.sft_loss import causal_lm_loss
from core.model import DEFAULT_MODEL, resolve_model
from soft_prompt import build_train_batch, load_model

REPO = ROOT.parent
# full-attention q/k/v/o (6 layers) + Gated DeltaNet in/out projections (18 layers); no MLP, gates, lm_head
DEFAULT_TARGETS = "q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,out_proj"


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--train_file", default=str(REPO / "train.jsonl"))
    ap.add_argument("--out_dir", required=True, help="relative to yanchi/, e.g. runs/lora_r16_lr2e-4")
    # LoRA
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--targets", default=DEFAULT_TARGETS, help="comma-separated module names")
    # optimisation (defaults = teammate's soft-prompt protocol except lr)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--warmup_steps", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--max_len", type=int, default=1536)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=0, help="0 = no limit")
    ap.add_argument("--save_at_examples", default="1000,2000,4000", help="data budgets")
    ap.add_argument("--grad_checkpointing", action=argparse.BooleanOptionalAction, default=True,
                    help="on by default: without it bs 8 x ~1.5k tokens OOMs on 12 GB")
    ap.add_argument("--untimed_warmup_batches", type=int, default=3,
                    help="fwd+bwd passes before the clock starts (kernel JIT/autotune); no update")
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)  # NB: no other python-random calls before the shuffle -> same order as soft prompt
    torch.manual_seed(args.seed)  # LoRA A init
    device = "cuda"

    model_path = resolve_model(args.model)
    model, tok = load_model(model_path, device)  # bf16, frozen, eval mode
    n_base = sum(p.numel() for p in model.parameters())
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    cfg = LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout, bias="none",
                     target_modules=[t.strip() for t in args.targets.split(",") if t.strip()],
                     task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)  # adapters are created in fp32 (autocast_adapter_dtype), base stays bf16
    if args.grad_checkpointing or args.dropout > 0:
        model.train()  # HF only checkpoints in train mode; Qwen has no dropout of its own
    params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    print(f"base params: {n_base} | trainable (LoRA): {n_train} | dtypes: {sorted({str(p.dtype) for p in params})}")
    flops = FlopCounter(model.config, n_base=n_base, n_train=n_train)
    # = HF's labels= loss, but logits only on supervised positions, chunked (full logits OOM on 12 GB)
    loss_fn = lambda ids, m, lab: causal_lm_loss(model.get_base_model(), ids, m, lab)

    data = load_jsonl(args.train_file)
    print(f"train examples: {len(data)} from {args.train_file}")

    # sanity check: show exactly what the model is trained on
    ids, _, labels = build_train_batch(tok, data[:1], args.max_len, device)
    print("---- example input ----\n" + tok.decode(ids[0]))
    print("---- supervised part ----\n" + tok.decode(ids[0][labels[0] != -100]) + "\n----")

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, args.warmup_steps)))
    save_at_ex = sorted(int(x) for x in args.save_at_examples.split(",") if x.strip())

    def save(tag, elapsed, step, seen, tokens):
        path = out_dir / f"adapter_{tag}"
        model.save_pretrained(path)
        state = {"tag": tag, "step": step, "examples_seen": seen, "tokens_seen": tokens,
                 "train_flops": flops.total, "train_time_sec": round(elapsed, 1),
                 "peak_mem_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3)}
        json.dump(state, open(path / "train_state.json", "w"), indent=2)
        print(f"[save] {path}  {state}")

    for k in range(args.untimed_warmup_batches):
        batch = data[k * args.batch_size: (k + 1) * args.batch_size]
        input_ids, mask, labels = build_train_batch(tok, batch, args.max_len, device)
        loss_fn(input_ids, mask, labels).backward()
        opt.zero_grad(set_to_none=True)
    print(f"untimed warmup: {args.untimed_warmup_batches} batches done")

    log = open(out_dir / "train_log.jsonl", "w")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    step, seen, tokens, micro, done = 0, 0, 0, 0, False

    for epoch in range(args.epochs):
        random.shuffle(data)
        for i in range(0, len(data), args.batch_size):
            batch = data[i: i + args.batch_size]
            input_ids, mask, labels = build_train_batch(tok, batch, args.max_len, device)
            loss = loss_fn(input_ids, mask, labels)
            (loss / args.grad_accum).backward()
            micro += 1
            seen += len(batch)
            lengths = mask.sum(1).tolist()
            tokens += sum(lengths)
            flops.add(lengths)
            if micro % args.grad_accum:
                continue

            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            elapsed = time.perf_counter() - t0

            if step % args.log_every == 0:
                rec = {"step": step, "epoch": epoch, "loss": loss.item(), "lr": sched.get_last_lr()[0],
                       "elapsed_sec": round(elapsed, 1), "examples_seen": seen, "tokens_seen": tokens,
                       "train_flops": flops.total}
                print(rec)
                log.write(json.dumps(rec) + "\n")
                log.flush()

            while save_at_ex and seen >= save_at_ex[0]:
                save(f"{save_at_ex.pop(0)}ex", elapsed, step, seen, tokens)
            if args.max_steps and step >= args.max_steps:
                done = True
                break
        if done:
            break

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    save("final", elapsed, step, seen, tokens)
    metrics = {
        "method": "lora",
        "base_model": model_path,
        "train_time_sec": round(elapsed, 1),
        "peak_mem_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "trainable_params": n_train,
        "steps": step,
        "examples_seen": seen,
        "tokens_seen": tokens,
        "train_flops": flops.total,
        "flops_convention": flops.describe(),
        "gpu": torch.cuda.get_device_name(0),
        "args": vars(args),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
