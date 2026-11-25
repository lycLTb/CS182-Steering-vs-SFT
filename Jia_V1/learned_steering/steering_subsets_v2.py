#!/usr/bin/env python3
"""Evaluation data for the v2 WildJailbreak split (4000 train / 400 dev / 2210 test).

Builds three evaluation roots, each in the file layout run_benchmarks.evaluate reads:
  steering_v2/        curve, used at every checkpoint:
                      HarmBench: the 200 prompts of data/eval_v2/harmbench.jsonl (the 200
                        "standard" behaviors; written as their original harmbench.csv rows);
                      XSTest: data/eval_v2/xstest.jsonl, all 450 (250 safe + 200 unsafe),
                        identical to data/xstest.csv;
                      WildJailbreak: fixed subset of the TEST file, 360 adversarial_harmful
                        + 40 adversarial_benign (seed 182);
                      MMLU: the existing 500-question subset (copied from steering_runs/data).
  steering_v2/dev/    all 400 dev prompts. Used only to choose the checkpoint and alpha.
  steering_v2/test/   the full 2210-prompt test file. Run once, on the selected setting.

The baseline's curve result on MMLU is copied from steering_runs/results/baseline_curve
(same file, same model); HarmBench, XSTest and WildJailbreak are regenerated.
Rerunning only verifies hashes; subsets are never redrawn.
"""
import csv
import hashlib
import json
import random
import shutil

from run_benchmarks import ROOT, save_json

SOURCE = ROOT / "data" / "wildjailbreak_v2"
OLD = ROOT / "steering_runs"
V2 = ROOT / "steering_v2"
SEED = 182
REUSED = ("mmlu_test.jsonl", "mmlu_dev.jsonl", "manifest.json")
EVAL = ROOT / "data" / "eval_v2"
FIELDS = ["adversarial", "label", "data_type", "source_row"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name, folder=SOURCE):
    return [json.loads(line) for line in (folder / name).read_text().splitlines() if line.strip()]


def write_tsv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        for i, r in rows:
            writer.writerow({"adversarial": r["prompt"], "label": r.get("label", ""),
                             "data_type": r["data_type"], "source_row": i})


def main():
    manifest_path = V2 / "data" / "subsets_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for rel, digest in manifest["files"].items():
            assert sha(V2 / rel) == digest, f"{rel} changed since it was created"
        print(json.dumps({"status": "verified", "files": len(manifest["files"])}))
        return
    old_manifest = json.loads((OLD / "data" / "subsets_manifest.json").read_text())
    for name, info in old_manifest["files"].items():
        if name in REUSED:
            assert sha(OLD / "data" / name) == info["sha256"], f"old subset {name} changed"

    test = read("test.jsonl")
    dev = read("dev.jsonl")
    assert len(read("train.jsonl")) == 4000 and len(dev) == 400 and len(test) == 2210

    (V2 / "data").mkdir(parents=True, exist_ok=True)
    for name in REUSED:
        shutil.copyfile(OLD / "data" / name, V2 / "data" / name)
    # HarmBench: the given 200 prompts, written as their original csv rows (BehaviorID kept).
    with (ROOT / "data" / "harmbench.csv").open() as f:
        reader = csv.DictReader(f)
        hb_fields, by_behavior = reader.fieldnames, {r["Behavior"].strip(): r for r in reader}
    hb = [by_behavior[r["prompt"].strip()] for r in read("harmbench.jsonl", EVAL)]
    assert len(hb) == 200 and not any(r.get("ContextString") for r in hb)
    with (V2 / "data" / "harmbench.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=hb_fields)
        writer.writeheader()
        writer.writerows(hb)
    # XSTest: all 450, identical to the original file (checked prompt by prompt).
    with (ROOT / "data" / "xstest.csv").open() as f:
        xs = [r["prompt"] for r in csv.DictReader(f)]
    assert xs == [r["prompt"] for r in read("xstest.jsonl", EVAL)]
    shutil.copyfile(ROOT / "data" / "xstest.csv", V2 / "data" / "xstest.csv")

    rng = random.Random(SEED)
    harmful = [i for i, r in enumerate(test) if r["data_type"] == "adversarial_harmful"]
    benign = [i for i, r in enumerate(test) if r["data_type"] == "adversarial_benign"]
    chosen = sorted(rng.sample(harmful, 360) + rng.sample(benign, 40))
    write_tsv(V2 / "data" / "wildjailbreak_eval.tsv", [(i, test[i]) for i in chosen])

    for root, rows in (("dev", list(enumerate(dev))), ("test", list(enumerate(test)))):
        write_tsv(V2 / root / "data" / "wildjailbreak_eval.tsv", rows)
        shutil.copyfile(OLD / "data" / "manifest.json", V2 / root / "data" / "manifest.json")

    # The baseline's MMLU curve answers are reused (same subset); the rest is regenerated.
    src, dst = OLD / "results" / "baseline_curve", V2 / "results" / "baseline_curve"
    dst.mkdir(parents=True, exist_ok=True)
    for bench in ("mmlu",):
        for suffix in (".jsonl", "_metadata.json", "_summary.json"):
            shutil.copyfile(src / f"{bench}{suffix}", dst / f"{bench}{suffix}")

    files = {str(p.relative_to(V2)): sha(p) for p in sorted(V2.glob("**/data/*")) if p.name != "subsets_manifest.json"}
    save_json(manifest_path, {
        "seed": SEED, "files": files, "wildjailbreak_curve_source_rows": chosen,
        "sources": {**{n: sha(SOURCE / n) for n in ("train.jsonl", "dev.jsonl", "test.jsonl")},
                    **{"eval_v2/" + n: sha(EVAL / n) for n in ("harmbench.jsonl", "xstest.jsonl")}},
        "note": "curve: HarmBench 200 standard (eval_v2/harmbench.jsonl), XSTest 450, WildJailbreak "
                "360 harmful + 40 benign from test.jsonl, MMLU 500 copied from steering_runs/data; "
                "dev and test roots hold the full files"})
    print(json.dumps({"status": "created", "files": len(files)}))


if __name__ == "__main__":
    main()
