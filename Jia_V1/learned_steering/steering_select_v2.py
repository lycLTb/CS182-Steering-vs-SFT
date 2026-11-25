#!/usr/bin/env python3
"""Record the dev-selected checkpoint and alpha in steering_v2/selection.json.

Run after the dev generations are judged:
  python steering_select_v2.py --checkpoint learned_L11_s182_step4096 --alpha 1 --reason "best dev score"
  python steering_select_v2.py --alpha 2 --reason "alpha sweep"   # keep checkpoint, set alpha
Later phases (e2 alpha sweep, full test, e3/e4 steps) read this file.
"""
import argparse
import json
import time

from run_benchmarks import ROOT, save_json

PATH = ROOT / "steering_v2" / "selection.json"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint")
    p.add_argument("--alpha", type=float)
    p.add_argument("--reason", required=True)
    args = p.parse_args()
    current = json.loads(PATH.read_text()) if PATH.exists() else {"alpha": 1.0, "history": []}
    if args.checkpoint:
        if not (ROOT / "steering_v2" / "checkpoints" / args.checkpoint / "vector.pt").exists():
            raise SystemExit(f"No checkpoint {args.checkpoint}")
        current["checkpoint"] = args.checkpoint
    if args.alpha is not None:
        current["alpha"] = args.alpha
    if "checkpoint" not in current:
        raise SystemExit("Give --checkpoint the first time")
    current["history"].append({"time": time.strftime("%F %T"), "checkpoint": current["checkpoint"],
                               "alpha": current["alpha"], "reason": args.reason})
    save_json(PATH, current)
    print(json.dumps({k: current[k] for k in ("checkpoint", "alpha")}))


if __name__ == "__main__":
    main()
