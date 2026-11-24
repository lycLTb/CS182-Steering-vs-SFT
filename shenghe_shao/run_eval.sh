#!/usr/bin/env bash
# Evaluate base model + every soft prompt checkpoint in a run directory.
#   export DEEPSEEK_API_KEY=sk-...
#   bash run_eval.sh <run_dir> [gpu_id] [mmlu_limit] [dev|test]
#
#   dev  (default): dev.jsonl only -- use this while tuning hyperparameters
#   test          : test_harmbench / test_jailbreak / test_xstest -- final numbers, run once
#
# e.g. bash run_eval.sh $WORK/runs/sp_n20_lr5e-3 1 1000 dev 2>&1 | tee $WORK/logs/eval.log
# Finished steps are skipped on re-run. If a generate step was interrupted midway,
# delete that partial *.jsonl before re-running.
set -euo pipefail

RUN=${1:?usage: bash run_eval.sh <run_dir> [gpu] [mmlu_limit] [dev|test]}
GPU=${2:-1}
MMLU_LIMIT=${3:-1000}            # 0 = full MMLU (14k questions)
MODE=${4:-dev}
M=$WORK/cs182_assets/models/Qwen3.5-0.8B
S=$WORK/cs182_assets/splits
OUT=$WORK/outputs/$(basename "$RUN")
mkdir -p "$OUT"
: "${DEEPSEEK_API_KEY:?export DEEPSEEK_API_KEY first}"
export CUDA_VISIBLE_DEVICES=$GPU

case $MODE in
  dev)  SETS="dev" ;;
  test) SETS="test_harmbench test_jailbreak test_xstest" ;;
  *) echo "mode must be dev or test"; exit 1 ;;
esac

# name -> soft prompt flag ("" = base model)
declare -A SP=( [base]="" )
for ck in "$RUN"/soft_prompt_*ex.pt; do
  n=$(basename "$ck" .pt); SP[${n#soft_prompt_}]="--soft_prompt $ck"
done

for name in "${!SP[@]}"; do
  for set in $SETS; do
    echo "================ $name / $set ================"
    f="$OUT/${name}_${set}"
    [ -f "$f.jsonl" ] || \
      python generate.py --model "$M" ${SP[$name]} --input "$S/$set.jsonl" --output "$f.jsonl"
    [ -f "$f.summary.json" ] || \
      python judge.py --input "$f.jsonl" --judge_model deepseek-flash
  done
  # file name records the question count (n0 = full MMLU) so dev subsets are never reused for test
  [ -f "$OUT/mmlu_${name}_n${MMLU_LIMIT}.json" ] || \
    python eval_mmlu.py --model "$M" ${SP[$name]} --mmlu_dir "$WORK/cs182_assets/data/mmlu" \
      --output "$OUT/mmlu_${name}_n${MMLU_LIMIT}.json" --limit "$MMLU_LIMIT"
  if [ "$MODE" = test ]; then
    [ -f "$OUT/mtb_${name}.jsonl" ] || \
      python mt_bench.py gen --model "$M" ${SP[$name]} --data_dir "$WORK/cs182_assets/data/mt_bench" \
        --output "$OUT/mtb_${name}.jsonl"
    [ -f "$OUT/mtb_${name}.summary.json" ] || \
      python mt_bench.py judge --output "$OUT/mtb_${name}.jsonl"
  fi
done

echo "================ RESULTS ($MODE) ================"
python - "$OUT" "$MMLU_LIMIT" $SETS <<'EOF'
import json, sys, os
out, mmlu_n, sets = sys.argv[1], sys.argv[2], sys.argv[3:]
names = sorted({f.split("_")[0] for f in os.listdir(out)
                 if f.endswith(".summary.json") and not f.startswith("mtb_")})
cols = []
for s in sets:
    cols += [(s, "ASR"), (s, "over_refusal")]
print(f"{'model':<10}" + "".join(f"{s.replace('test_', '')[:9] + ':' + k[:6]:>18}" for s, k in cols) + f"{'MMLU':>8}{'MTBench':>9}")
for n in names:
    row = f"{n:<10}"
    for s, k in cols:
        p = f"{out}/{n}_{s}.summary.json"
        v = json.load(open(p)).get(k) if os.path.exists(p) else None
        row += f"{v:>18.3f}" if v is not None else f"{'-':>18}"
    m = f"{out}/mmlu_{n}_n{mmlu_n}.json"
    row += f"{json.load(open(m))['accuracy']:>8.3f}" if os.path.exists(m) else f"{'-':>8}"
    t = f"{out}/mtb_{n}.summary.json"
    row += f"{json.load(open(t))['overall']:>9.2f}" if os.path.exists(t) else f"{'-':>9}"
    print(row)
EOF
