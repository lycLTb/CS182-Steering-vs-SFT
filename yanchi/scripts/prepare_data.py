"""Download benchmarks and build fixed-seed evaluation subsets (see doc/experiment_setup.md).

Outputs (all under ./data):
  harmbench_test.jsonl        159 standard text behaviors (test)
  harmbench_val.jsonl         41 standard text behaviors (val, for tuning only)
  wj_eval_adv_harmful.jsonl   2000 adversarial-harmful prompts (full eval split)
  wj_eval_adv_benign.jsonl    210 adversarial-benign prompts (full)
  xstest.jsonl                450 prompts (250 safe + 200 unsafe)
  mtbench_questions.jsonl     80 questions x 2 turns
  mtbench_reference.jsonl     GPT-4 reference answers (math/reasoning/coding)
  mtbench_judge_prompts.jsonl official judge prompts
  mmlu_subset_ids.json        50 test questions per subject (seeded)
  ids/*.json                  ID lists for every subset
"""
import csv
import sys
import io
import json
import random
from collections import defaultdict
from pathlib import Path

import requests
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make yanchi/core importable
from core.paths import DATA  # noqa: E402

SEED = 182
OUT = DATA
IDS = OUT / "ids"
OUT.mkdir(exist_ok=True)
IDS.mkdir(exist_ok=True)

HB_BASE = "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets"
XSTEST_URL = "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv"
MTB_BASE = "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data"


def fetch(url: str) -> str:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.text


def write_jsonl(path: Path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{path.name}: {len(rows)}")


def write_ids(name: str, ids):
    with open(IDS / f"{name}.json", "w") as f:
        json.dump(list(ids), f, indent=0)


def harmbench():
    for split in ["test", "val"]:
        text = fetch(f"{HB_BASE}/harmbench_behaviors_text_{split}.csv")
        rows = [r for r in csv.DictReader(io.StringIO(text)) if r["FunctionalCategory"] == "standard"]
        rows = [
            {"id": r["BehaviorID"], "prompt": r["Behavior"], "semantic_category": r["SemanticCategory"]}
            for r in rows
        ]
        write_jsonl(OUT / f"harmbench_{split}.jsonl", rows)
        write_ids(f"harmbench_{split}", [r["id"] for r in rows])


def wildjailbreak():
    ds = load_dataset("allenai/wildjailbreak", "eval", delimiter="\t", keep_default_na=False)["train"]
    harmful, benign = [], []
    for i, r in enumerate(ds):
        row = {"id": f"wj_eval_{i}", "prompt": r["adversarial"], "label": r["label"], "data_type": r["data_type"]}
        (harmful if r["data_type"] == "adversarial_harmful" else benign).append(row)
    print(f"wildjailbreak eval: {len(harmful)} adv-harmful, {len(benign)} adv-benign")
    harmful_sub = harmful  # full eval split (2000); previously a seeded 500 subset
    write_jsonl(OUT / "wj_eval_adv_harmful.jsonl", harmful_sub)
    write_jsonl(OUT / "wj_eval_adv_benign.jsonl", benign)
    write_ids("wj_eval_adv_harmful", [r["id"] for r in harmful_sub])
    write_ids("wj_eval_adv_benign", [r["id"] for r in benign])
    # also cache the train split (used later for steering / LoRA)
    tr = load_dataset("allenai/wildjailbreak", "train", delimiter="\t", keep_default_na=False)["train"]
    print(f"wildjailbreak train cached: {len(tr)} rows")


def xstest():
    text = fetch(XSTEST_URL)
    rows = []
    for r in csv.DictReader(io.StringIO(text)):
        rows.append({"id": f"xstest_{r['id']}", "prompt": r["prompt"], "type": r["type"], "label": r["label"],
                     "focus": r.get("focus", ""), "note": r.get("note", "")})
    n_safe = sum(r["label"] == "safe" for r in rows)
    print(f"xstest: {n_safe} safe, {len(rows) - n_safe} unsafe")
    write_jsonl(OUT / "xstest.jsonl", rows)
    write_ids("xstest", [r["id"] for r in rows])


def mtbench():
    for src, dst in [
        ("mt_bench/question.jsonl", "mtbench_questions.jsonl"),
        ("mt_bench/reference_answer/gpt-4.jsonl", "mtbench_reference.jsonl"),
        ("judge_prompts.jsonl", "mtbench_judge_prompts.jsonl"),
    ]:
        text = fetch(f"{MTB_BASE}/{src}")
        (OUT / dst).write_text(text)
        print(f"{dst}: {len(text.strip().splitlines())}")


def mmlu():
    ds = load_dataset("cais/mmlu", "all")
    _ = load_dataset("cais/mmlu", "all", split="dev")  # few-shot exemplars
    by_subj = defaultdict(list)
    for i, r in enumerate(ds["test"]):
        by_subj[r["subject"]].append(i)
    rng = random.Random(SEED)
    subset = {s: sorted(rng.sample(idx, min(50, len(idx)))) for s, idx in sorted(by_subj.items())}
    total = sum(len(v) for v in subset.values())
    print(f"mmlu: {len(subset)} subjects, {total} test questions in subset")
    # indices are positions within each subject's test split (lm-eval per-subject task order)
    per_subj_pos = {}
    for s, idx in subset.items():
        pos_map = {g: p for p, g in enumerate(by_subj[s])}
        per_subj_pos[s] = [pos_map[g] for g in idx]
    with open(OUT / "mmlu_subset_ids.json", "w") as f:
        json.dump({"seed": SEED, "n_per_subject": 50, "global_test_idx": subset, "per_subject_pos": per_subj_pos}, f)
    # also pre-cache lm-eval's per-subject copy of MMLU
    for s in subset:
        try:
            load_dataset("cais/mmlu", s)
        except Exception as e:  # noqa: BLE001
            print("warn: could not cache", s, e)
            break


if __name__ == "__main__":

    steps = sys.argv[1:] or ["harmbench", "xstest", "mtbench", "wildjailbreak", "mmlu"]
    for s in steps:
        print(f"== {s}")
        globals()[s]()
