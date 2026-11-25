#!/usr/bin/env python3
"""Learned steering training on the v2 WildJailbreak split (4000 training examples).

Same objective, optimizer, clipping and truncation as train_methods.py:
response-only cross-entropy, AdamW (weight decay 0), gradient clip 1.0, batch 1,
prompt <= 256 / response <= 128 tokens, vector added at the output of
model.model.layers[layer] at every position.

v2 differences from steering_train.py:
- data: data/wildjailbreak_v2/train.jsonl (1000 of each of the 4 data types);
- learning rate fixed at 0.01 (the earlier search's choice); no search here;
- the dev set has no reference responses, so no validation NLL is computed;
  checkpoint and alpha are chosen later by judging dev generations;
- checkpoints at 16, 32, ..., 8192 steps (4000 steps = 1 epoch).

Outputs under steering_v2/: checkpoints/learned_L{layer}_s{seed}_step{N}/
(vector.pt + metadata.json) and logs/train_L{layer}_s{seed}.jsonl.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

from run_benchmarks import Engine, Monitor, ROOT, save_json, metadata
from train_methods import encode, loss
from steering_flops import ModelCosts

RUNS = ROOT / "steering_v2"
DATA = ROOT / "data" / "wildjailbreak_v2" / "train.jsonl"
STEPS = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
LEARNING_RATE = 0.01
RUNNER_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def run_name(layer, seed, step, smoke=False):
    return f"{'smoke_' if smoke else ''}learned_L{layer}_s{seed}_step{step}"


def load_rows():
    rows = [json.loads(line) for line in DATA.read_text().splitlines() if line.strip()]
    # train_methods.encode reads adversarial-or-vanilla as the prompt and completion as the target.
    return [{"adversarial": r["prompt"], "vanilla": "", "completion": r["response"],
             "data_type": r["data_type"]} for r in rows]


def sizes(x):
    return int(x["input_ids"].shape[1]), int(x["response_tokens"])


def dom_direction(torch, layer):
    path = ROOT / "checkpoints" / "dom_large" / "vector.pt"
    meta = ROOT / "checkpoints" / "dom_large" / "metadata.json"
    if layer != 11 or not path.exists() or json.loads(meta.read_text()).get("layer", 11) != layer:
        return None
    return torch.load(path, weights_only=True, map_location="cpu").float()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=182, help="training data order")
    p.add_argument("--layer", type=int, default=11)
    p.add_argument("--max-steps", type=int, default=STEPS[-1])
    p.add_argument("--device", default="mps", choices=["mps", "cpu"])
    p.add_argument("--smoke", action="store_true", help="16 steps, checkpoints prefixed smoke_")
    args = p.parse_args()
    if args.smoke:
        args.max_steps = 16
    final = RUNS / "checkpoints" / run_name(args.layer, args.seed, args.max_steps, args.smoke) / "metadata.json"
    if final.exists() and json.loads(final.read_text()).get("training_runner_sha256") == RUNNER_SHA:
        print(json.dumps({"event": "skip", "reason": "already trained with this runner", "path": str(final)}))
        return
    args.variant, args.dtype, args.max_new_tokens = "baseline", "bfloat16", 1024

    rows = load_rows()
    engine = Engine(args)
    torch = engine.torch
    engine.model.config.use_cache = False
    costs = ModelCosts.from_model(engine.model)
    training = [encode(engine, r) for r in rows]
    vector = torch.nn.Parameter(torch.zeros(engine.model.config.hidden_size, device=engine.device,
                                            dtype=torch.float32))
    engine.install_vector(vector, args.layer)
    dom = dom_direction(torch, args.layer)
    opt = torch.optim.AdamW([vector], lr=LEARNING_RATE, weight_decay=0, foreach=False)

    rng = random.Random(args.seed)
    log_path = RUNS / "logs" / f"{'smoke_' if args.smoke else ''}train_L{args.layer}_s{args.seed}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w")
    n = len(training)
    order, step, elapsed, flops, tokens, response_tokens, losses = [], 0, 0.0, 0, 0, 0, []
    model_meta = metadata(args)
    with Monitor(engine.device) as mem:
        while step < args.max_steps:
            if step % n == 0:
                order = list(range(n))
                rng.shuffle(order)
            x = training[order[step % n]]
            T, R = sizes(x)
            engine.sync()
            start = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            objective = loss(engine, x)
            if not torch.isfinite(objective).item():
                raise RuntimeError("Nonfinite training loss")
            objective.backward()
            g = vector.grad
            if g is None or not torch.isfinite(g).all().item() or g.abs().max().item() == 0:
                raise RuntimeError("Missing finite nonzero steering gradient")
            torch.nn.utils.clip_grad_norm_([vector], 1.0)
            opt.step()
            engine.sync()
            elapsed += time.perf_counter() - start
            step += 1
            flops += costs.learned_step(args.layer, T, R)
            tokens += T
            response_tokens += R
            losses.append(objective.item())
            log.write(json.dumps({"step": step, "example": order[(step - 1) % n], "loss": losses[-1],
                                  "tokens": T, "response_tokens": R, "cumulative_flops": flops}) + "\n")
            log.flush()
            if step in STEPS or step == args.max_steps:
                v = vector.detach().float().cpu().clone()
                window = losses[-256:]
                info = {"method": "learned", "layer": args.layer, "scale": 1.0, "seed": args.seed,
                        "data": "wildjailbreak_v2/train.jsonl (4000)", "step": step, "epochs": step / n,
                        "data_exposure": step, "cumulative_training_flops": flops, "tokens_seen": tokens,
                        "response_tokens_seen": response_tokens, "training_seconds": elapsed,
                        "seconds_per_step": elapsed / step, "learning_rate": LEARNING_RATE,
                        "learning_rate_source": "steering_runs/lr_search_L11.json (search on the earlier split)",
                        "mean_train_loss_last_256": sum(window) / len(window),
                        "validation_response_nll": None,
                        "vector_norm": v.norm().item(),
                        "cosine_to_dom_large": None if dom is None else
                            torch.nn.functional.cosine_similarity(v, dom, dim=0).item(),
                        "trainable_parameters": vector.numel(), "optimizer": "AdamW", "batch_size": 1,
                        "maximum_prompt_tokens": 256, "maximum_response_tokens": 128,
                        "data_order": f"random.Random({args.seed}), reshuffled every {n} steps",
                        "flops_costs": costs.describe(), "training_runner_sha256": RUNNER_SHA,
                        **mem.result(), "model": model_meta}
                folder = RUNS / "checkpoints" / run_name(args.layer, args.seed, step, args.smoke)
                folder.mkdir(parents=True, exist_ok=True)
                torch.save(v, folder / "vector.pt")
                save_json(folder / "metadata.json", info)
                print(json.dumps({"event": "checkpoint", "name": folder.name, "flops": flops,
                                  "mean_train_loss_last_256": info["mean_train_loss_last_256"],
                                  "cosine_to_dom_large": info["cosine_to_dom_large"],
                                  "seconds": round(elapsed, 1)}), flush=True)
    log.close()


if __name__ == "__main__":
    main()
