"""MMLU 5-shot loglik via lm-eval-harness on the full *validation* split (1531 Qs; few-shot from dev).

Usage: python scripts/run_mmlu.py --out outputs/baseline [--n_per_subject K]   (K>0: first K val Qs/subject, smoke test)
"""
import argparse
import os

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make yanchi/core importable

from core.model import DEFAULT_MODEL, model_info, resolve_model
from core.paths import DATA, ROOT
import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager, get_task_dict

SPLIT = "validation"


def build_tasks():
    subjects = sorted(json.load(open(DATA / "mmlu_subset_ids.json"))["per_subject_pos"])  # 57 subjects
    td = get_task_dict([f"mmlu_{s}" for s in subjects], TaskManager())
    for t in td.values():
        t.config.test_split = SPLIT  # default mmlu tasks evaluate on `test`
    return td


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, help="model dir (default: exported base)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_per_subject", type=int, default=0, help="0 = full validation split")
    ap.add_argument("--bs", type=int, default=2)  # full-seq logits x 248k vocab (fp32 in lm-eval): bs>=4 OOMs on 12GB
    a = ap.parse_args()

    td = build_tasks()
    samples = None
    if a.n_per_subject:
        samples = {n: list(range(min(a.n_per_subject, len(t.eval_docs)))) for n, t in td.items()}
    n_per = {n: (len(samples[n]) if samples else len(t.eval_docs)) for n, t in td.items()}

    lm = HFLM(pretrained=resolve_model(a.model), dtype="bfloat16", batch_size=a.bs, device="cuda")
    res = lm_eval.simple_evaluate(model=lm, tasks=list(td.values()), num_fewshot=5, samples=samples,
                                  log_samples=False, random_seed=0, numpy_random_seed=0, torch_random_seed=0,
                                  fewshot_random_seed=1234)
    per = {t: r["acc,none"] for t, r in res["results"].items() if t in n_per}
    n = sum(n_per.values())
    micro = sum(per[t] * n_per[t] for t in per) / n
    summary = {**model_info(a.model), "split": SPLIT, "fewshot_split": "dev", "num_fewshot": 5, "n_questions": n,
               "acc_micro": micro, "acc_macro": sum(per.values()) / len(per), "per_subject": per,
               "n_per_subject": n_per, "chat_template": False}
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    tag = f"n{a.n_per_subject}" if a.n_per_subject else "full"
    json.dump(summary, open(out / f"mmlu_{SPLIT}_{tag}.json", "w"), indent=2)
    print(f"MMLU {SPLIT} acc (micro) = {micro:.4f} over {n} questions")


if __name__ == "__main__":
    main()
