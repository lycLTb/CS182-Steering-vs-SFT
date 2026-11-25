#!/usr/bin/env python3
"""Evaluate steering checkpoints on the v2 split. Generations go to the external
DeepSeek judge; MMLU is scored locally in FP32 as in the baseline.

  --split curve  every checkpoint: HarmBench 200 (standard), WildJailbreak test subset 400,
                 XSTest 450, MMLU 500; 512 new tokens.      -> steering_v2/results/
  --split dev    choosing checkpoint and alpha: 400 dev prompts; 512 new tokens.
                                                                 -> steering_v2/dev/results/
  --split test   once, on the selected setting: full 2210-prompt jailbreak test;
                 1024 new tokens, as in the existing baseline run (results/baseline/
                 wildjailbreak.jsonl is the baseline for this split).
                                                                 -> steering_v2/test/results/

  python steering_eval_v2.py --split dev --baseline
  python steering_eval_v2.py --split curve --checkpoint learned_L11_s182_step4096
  python steering_eval_v2.py --split dev --checkpoint selected --alpha 2
  python steering_eval_v2.py --split test --checkpoint selected --alpha selected

"selected" reads steering_v2/selection.json (written by steering_select_v2.py after
the dev generations are judged). Interrupted runs resume where they stopped.
"""
import argparse
import gc
import json

import run_benchmarks as rb

RUNS = rb.ROOT / "steering_v2"
# root, benchmarks, max new tokens, generation batch size. curve and dev use batch 16 for
# throughput (the same for every v2 variant, baseline included); test keeps batch 4 and
# 1024 tokens to match the existing full baseline run.
SPLITS = {"curve": (RUNS, ["harmbench", "wildjailbreak", "xstest", "mmlu"], 512, 16),
          "dev": (RUNS / "dev", ["wildjailbreak"], 512, 16),
          "test": (RUNS / "test", ["wildjailbreak"], 1024, 4)}
GENERATION = ("harmbench", "wildjailbreak", "xstest")


def selection():
    path = RUNS / "selection.json"
    if not path.exists():
        raise SystemExit("No steering_v2/selection.json yet: judge the dev generations, then run steering_select_v2.py")
    return json.loads(path.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", required=True, choices=sorted(SPLITS))
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument("--baseline", action="store_true")
    target.add_argument("--checkpoint", help="name under steering_v2/checkpoints, or 'selected'")
    p.add_argument("--alpha", default="1", help="inference-time strength, or 'selected'")
    p.add_argument("--benchmarks", nargs="+", help="override the split's benchmark list")
    p.add_argument("--batch-size", type=int, help="override the split's batch size")
    p.add_argument("--limit", type=int, default=0, help="smoke test: at most this many rows per benchmark")
    p.add_argument("--device", default="mps", choices=["mps", "cpu"])
    args = p.parse_args()
    root, benchmarks, max_new_tokens, batch_size = SPLITS[args.split]
    batch_size = args.batch_size or batch_size
    benchmarks = args.benchmarks or benchmarks
    if not (RUNS / "data" / "subsets_manifest.json").exists():
        raise SystemExit("Run steering_subsets_v2.py first")

    checkpoint, alpha, info = None, 0.0, {}
    if not args.baseline:
        checkpoint = selection()["checkpoint"] if args.checkpoint == "selected" else args.checkpoint
        alpha = float(selection()["alpha"]) if args.alpha == "selected" else float(args.alpha)
        info = json.loads((RUNS / "checkpoints" / checkpoint / "metadata.json").read_text())
        name = checkpoint + ("" if alpha == 1 else f"_a{alpha:g}")
    else:
        name = "baseline_curve" if args.split == "curve" else "baseline"

    groups = [([b for b in benchmarks if b in GENERATION], "bfloat16"),
              ([b for b in benchmarks if b == "mmlu"], "float32")]
    original_root = rb.ROOT
    for group, dtype in groups:
        if not group:
            continue
        engine_args = argparse.Namespace(variant="baseline", device=args.device, dtype=dtype, seed=182,
                                         batch_size=batch_size, benchmarks=group,
                                         max_new_tokens=max_new_tokens, limit=args.limit)
        engine = rb.Engine(engine_args)  # loaded unsteered from the real snapshot
        if checkpoint:
            vector = engine.torch.load(RUNS / "checkpoints" / checkpoint / "vector.pt",
                                       weights_only=True, map_location=args.device)
            engine.install_vector(vector, info["layer"], alpha)
        engine_args.variant = name
        rb.ROOT = root  # data/ and results/ now resolve inside this split's root
        try:
            rb.evaluate(engine, engine_args)
        finally:
            rb.ROOT = original_root
        del engine
        gc.collect()
        if args.device == "mps":
            import torch
            torch.mps.empty_cache()

    rb.save_json(root / "results" / name / "steering.json", {
        "name": name, "split": args.split, "checkpoint": checkpoint, "alpha": alpha,
        "layer": info.get("layer"), "seed": info.get("seed"), "step": info.get("step", 0),
        "cumulative_training_flops": info.get("cumulative_training_flops", 0),
        "max_new_tokens": max_new_tokens, "batch_size": batch_size, "partial_limit": args.limit or None})
    print(json.dumps({"event": "done", "split": args.split, "name": name}))


if __name__ == "__main__":
    main()
