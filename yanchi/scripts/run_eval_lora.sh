#!/usr/bin/env bash
# Evaluate the base model + every merged LoRA checkpoint of a run with the TEAMMATE's eval scripts
# (shenghe_shao/{generate,judge,eval_mmlu,mt_bench}.py), so LoRA and soft prompt numbers come from the
# same code path (same batching, judge prompt, MMLU protocol). Adapted from shenghe_shao/run_eval.sh.
#
#   bash scripts/run_eval_lora.sh <run_name> [gpu] [mmlu_limit] [dev|test]
#
#   <run_name>      basename of the train_lora.py out_dir; evaluates models/lora-<run_name>-*ex
#                   (merge them first: python scripts/merge_lora.py --run runs/<run_name>)
#   dev  (default): dev.jsonl only -- use this while tuning hyperparameters
#   test          : test_harmbench / test_jailbreak / test_xstest + MT-Bench -- final numbers, run once
#
# Data = the shared jsonl files in the repo root. Outputs -> yanchi/outputs/lora/<run_name>/ (git-ignored);
# base-model results are computed once in yanchi/outputs/lora/_base/ and linked into every run dir.
# Env overrides: MMLU_DIR (dir with all/test-*.parquet; default = HF cache of cais/mmlu), MTB_DIR.
# Finished steps are skipped on re-run. If a generate step was interrupted midway, delete that partial *.jsonl.
set -euo pipefail

RUN=${1:?usage: bash scripts/run_eval_lora.sh <run_name> [gpu] [mmlu_limit] [dev|test]}
GPU=${2:-0}
MMLU_LIMIT=${3:-1000}            # 0 = full MMLU test (14k questions); teammate's default 1000
MODE=${4:-dev}

Y=$(cd "$(dirname "$0")/.." && pwd)        # yanchi/
REPO=$(dirname "$Y")
T=$REPO/shenghe_shao                       # teammate's scripts
PY=${PY:-python}
OUT=$Y/outputs/lora/$RUN
BASE_OUT=$Y/outputs/lora/_base
mkdir -p "$OUT" "$BASE_OUT"
export CUDA_VISIBLE_DEVICES=$GPU

if [ -z "${DEEPSEEK_API_KEY:-}" ] && [ -f "$Y/.deepseek_api_key" ]; then
  DEEPSEEK_API_KEY=$(tr -d '[:space:]' < "$Y/.deepseek_api_key"); export DEEPSEEK_API_KEY
fi
: "${DEEPSEEK_API_KEY:?export DEEPSEEK_API_KEY or put it in yanchi/.deepseek_api_key}"

MMLU_DIR=${MMLU_DIR:-$(ls -d "$HOME"/.cache/huggingface/hub/datasets--cais--mmlu/snapshots/*/ 2>/dev/null | head -1)}
[ -d "$MMLU_DIR/all" ] || { echo "MMLU_DIR not found (need <dir>/all/test-*.parquet)"; exit 1; }

# MT-Bench: teammate's loader globs every *.jsonl in the dir and needs `reference` to be a list on every row
MTB_DIR=${MTB_DIR:-$Y/outputs/lora/_mt_bench_data}
if [ "$MODE" = test ] && [ ! -f "$MTB_DIR/question.jsonl" ]; then
  mkdir -p "$MTB_DIR"
  $PY - "$Y/data/mtbench_questions.jsonl" "$MTB_DIR/question.jsonl" <<'EOF'
import json, sys
with open(sys.argv[2], "w") as f:
    for l in open(sys.argv[1]):
        q = json.loads(l)
        q["reference"] = q.get("reference") or []
        f.write(json.dumps(q, ensure_ascii=False) + "\n")
EOF
fi

complete() {  # $1 output jsonl, $2 expected line count; deletes a partial file (teammate's scripts overwrite, no resume)
  [ -f "$1" ] || return 1
  local n; n=$(grep -c . "$1" || true)
  [ "$n" -eq "$2" ] && return 0
  echo "[partial] $1 has $n/$2 rows -> regenerate"; rm -f "$1" "${1%.jsonl}.judged.jsonl" "${1%.jsonl}.summary.json"; return 1
}

input_file() {  # set name -> shared jsonl in the repo root (test_jailbreak is committed as "test_jailbreak..jsonl")
  for f in "$REPO/$1.jsonl" "$REPO/$1..jsonl"; do [ -f "$f" ] && { echo "$f"; return; }; done
  echo "missing $REPO/$1.jsonl" >&2; exit 1
}

case $MODE in
  dev)  SETS="dev" ;;
  test) SETS="test_harmbench test_jailbreak test_xstest" ;;
  *) echo "mode must be dev or test"; exit 1 ;;
esac

# name -> model dir; name is the budget tag (no "_": the results table splits file names on it)
declare -A M=( [base]="$Y/models/qwen3.5-0.8b-base" )
for d in "$Y"/models/lora-"$RUN"-[0-9]*ex; do
  [ -d "$d" ] || continue
  M[${d##*-}]="$d"
done
[ ${#M[@]} -gt 1 ] || { echo "no models/lora-$RUN-*ex — run scripts/merge_lora.py first"; exit 1; }

for name in "${!M[@]}"; do
  if [ "$name" = base ]; then D=$BASE_OUT; else D=$OUT; fi
  for set in $SETS; do
    echo "================ $name / $set ================"
    f="$D/${name}_${set}"
    complete "$f.jsonl" "$(grep -c . "$(input_file "$set")")" || \
      $PY "$T/generate.py" --model "${M[$name]}" --input "$(input_file "$set")" --output "$f.jsonl"
    [ -f "$f.summary.json" ] || \
      $PY "$T/judge.py" --input "$f.jsonl" --judge_model deepseek-flash
  done
  [ -f "$D/mmlu_${name}.json" ] || \
    $PY "$T/eval_mmlu.py" --model "${M[$name]}" --mmlu_dir "$MMLU_DIR" \
      --output "$D/mmlu_${name}.json" --limit "$MMLU_LIMIT"
  if [ "$MODE" = test ]; then
    complete "$D/mtb_${name}.jsonl" 80 || \
      $PY "$T/mt_bench.py" gen --model "${M[$name]}" --data_dir "$MTB_DIR" --output "$D/mtb_${name}.jsonl"
    [ -f "$D/mtb_${name}.summary.json" ] || \
      $PY "$T/mt_bench.py" judge --output "$D/mtb_${name}.jsonl"
  fi
done

# base results live in _base/, link them in so the table shows them next to the LoRA checkpoints
for f in "$BASE_OUT"/base_*.summary.json "$BASE_OUT"/mmlu_base.json "$BASE_OUT"/mtb_base.summary.json; do
  [ -f "$f" ] && ln -sf "$f" "$OUT/"
done

echo "================ RESULTS ($RUN, $MODE) ================"
$PY - "$OUT" $SETS <<'EOF'
import json, sys, os
out, sets = sys.argv[1], sys.argv[2:]
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
    m = f"{out}/mmlu_{n}.json"
    row += f"{json.load(open(m))['accuracy']:>8.3f}" if os.path.exists(m) else f"{'-':>8}"
    t = f"{out}/mtb_{n}.summary.json"
    row += f"{json.load(open(t))['overall']:>9.2f}" if os.path.exists(t) else f"{'-':>9}"
    print(row)
EOF
