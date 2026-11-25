#!/bin/bash
# Learned steering on the v2 WildJailbreak split (4000 train / 400 dev / 2210 test).
#   setup  subsets + 16-step smoke test on curve and dev (~10 min)
#   e1     train 8192 steps; curve evals coarse-to-fine; dev runs (baseline, 1024/4096/8192)
#   -- judge dev, then: python steering_select_v2.py --checkpoint ... --reason ...
#   e2     alpha 0.5 / 2 / 4 on the selected checkpoint, dev split
#   -- judge dev, then: python steering_select_v2.py --alpha ... --reason ...
#   test   full 2210-prompt jailbreak test on the selected checkpoint and alpha
#   e3     seeds 0 and 1; curve evals at steps 4096 and 8192
#   e4     layers 5 and 17; curve evals at steps 4096 and 8192
#   status
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-../.venv/bin/python}"
mkdir -p steering_v2/logs

run() { echo "[$(date '+%F %T')] $*"; "$PY" "$@"; }
curve() {  # curve <layer> <seed> <steps...>
  local layer=$1 seed=$2; shift 2
  for s in "$@"; do run steering_eval_v2.py --split curve --checkpoint "learned_L${layer}_s${seed}_step${s}"; done
}

case "${1:-}" in
  setup)
    run steering_subsets_v2.py
    run steering_train_v2.py --smoke
    run steering_eval_v2.py --split curve --checkpoint smoke_learned_L11_s182_step16 --limit 4
    run steering_eval_v2.py --split dev --checkpoint smoke_learned_L11_s182_step16 --limit 4
    ;;
  e1)
    run steering_train_v2.py --seed 182 --layer 11
    # Coarse-to-fine: every finished point is usable on its own (first 5 = preliminary curve).
    run steering_eval_v2.py --split curve --baseline --benchmarks harmbench wildjailbreak xstest
    curve 11 182 4096 256 1024 64 8192 2048 512 128 32 16
    run steering_eval_v2.py --split dev --baseline
    for s in 1024 4096 8192; do run steering_eval_v2.py --split dev --checkpoint "learned_L11_s182_step${s}"; done
    ;;
  e2)
    for a in 0.5 2 4; do run steering_eval_v2.py --split dev --checkpoint selected --alpha "$a"; done
    ;;
  test)
    run steering_eval_v2.py --split test --checkpoint selected --alpha selected
    ;;
  e3)
    for seed in 0 1; do run steering_train_v2.py --seed "$seed" --layer 11; done
    for seed in 0 1; do curve 11 "$seed" 4096 8192; done
    ;;
  e4)
    for layer in 5 17; do run steering_train_v2.py --seed 182 --layer "$layer"; done
    for layer in 5 17; do curve "$layer" 182 4096 8192; done
    ;;
  status)
    "$PY" - <<'EOF'
import json, pathlib
v2 = pathlib.Path("steering_v2")
for m in sorted((v2 / "checkpoints").glob("*/metadata.json")):
    i = json.loads(m.read_text())
    print(f"ckpt  {m.parent.name:34s} flops={i['cumulative_training_flops']:.3e} loss256={i['mean_train_loss_last_256']:.3f} cos_dom={i['cosine_to_dom_large']}")
for split in ("results", "dev/results", "test/results"):
    for d in sorted((v2 / split).glob("*")):
        done = {f.stem: sum(1 for _ in f.open()) for f in d.glob("*.jsonl")}
        print(f"{split:12s} {d.name:34s} {done}")
sel = v2 / "selection.json"
print("selection:", json.loads(sel.read_text()) if sel.exists() else "not set")
EOF
    ;;
  *)
    sed -n '2,12p' "$0"; exit 1 ;;
esac
