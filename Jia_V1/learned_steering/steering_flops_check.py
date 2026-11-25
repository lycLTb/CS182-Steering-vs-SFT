#!/usr/bin/env python3
"""Check the analytic FLOPs formula against torch's FlopCounterMode.

Runs a few real training examples on CPU in FP32 (FlopCounterMode counts
operators, not hardware, so CPU gives the same count as MPS). Writes
steering_runs/flops_check.json. A ratio within 0.90-1.10 means the analytic
axis is safe to use; otherwise use measured_over_analytic as a correction.
"""
import argparse
import json
import statistics

from run_benchmarks import Engine, ROOT, save_json
from train_methods import prepare_split, encode, loss
from steering_flops import ModelCosts

RUNS = ROOT / "steering_runs"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layer", type=int, default=11)
    p.add_argument("--examples", type=int, default=3)
    args = p.parse_args()
    engine_args = argparse.Namespace(variant="baseline", device="cpu", dtype="float32", seed=182,
                                     max_new_tokens=1024)
    engine = Engine(engine_args)
    torch = engine.torch
    from torch.utils.flop_counter import FlopCounterMode
    costs = ModelCosts.from_model(engine.model)
    vector = torch.nn.Parameter(torch.zeros(engine.model.config.hidden_size, dtype=torch.float32))
    engine.install_vector(vector, args.layer)
    split = prepare_split()
    rows = []
    for row in split["train"][:args.examples]:
        x = encode(engine, row)
        T, R = int(x["input_ids"].shape[1]), int(x["response_tokens"])
        with FlopCounterMode(display=False) as forward_counter, torch.no_grad():
            loss(engine, x)
        with FlopCounterMode(display=False) as step_counter:
            objective = loss(engine, x)
            objective.backward()
        vector.grad = None
        measured_forward = forward_counter.get_total_flops()
        measured_step = step_counter.get_total_flops()
        analytic_forward = costs.forward(T, R)
        analytic_step = costs.learned_step(args.layer, T, R)
        rows.append({"tokens": T, "response_tokens": R,
                     "measured_forward": measured_forward, "analytic_forward": analytic_forward,
                     "measured_step": measured_step, "analytic_step": analytic_step,
                     "forward_ratio": measured_forward / analytic_forward,
                     "step_ratio": measured_step / analytic_step})
        print(json.dumps(rows[-1]), flush=True)
    ratio = statistics.mean(r["step_ratio"] for r in rows)
    result = {"layer": args.layer, "examples": rows, "measured_over_analytic": ratio,
              "within_10_percent": 0.9 <= ratio <= 1.1, "costs": costs.describe()}
    save_json(RUNS / "flops_check.json", result)
    print(json.dumps({"measured_over_analytic": ratio, "within_10_percent": result["within_10_percent"]}))


if __name__ == "__main__":
    main()
