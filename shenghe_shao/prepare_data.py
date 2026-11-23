"""Build fixed train / dev / test splits from WildJailbreak.

The same splits should be used by every method (diff-of-means, learned steering,
LoRA, soft prompt) so data exposure is matched -- share the output jsonl files.

    python prepare_data.py --wjb_dir $WORK/cs182_assets/data/wildjailbreak \
                           --out_dir $WORK/cs182_assets/splits

Outputs:
    train.jsonl           {prompt, response, data_type}   n_per_type x 4 types
    dev.jsonl             {prompt, data_type}             held out from train, for tuning
    test_jailbreak.jsonl  {prompt, data_type, label}      WildJailbreak eval set (unseen jailbreaks)
"""
import argparse
import glob
import json
import os

import pandas as pd

TYPES = ["vanilla_harmful", "vanilla_benign", "adversarial_harmful", "adversarial_benign"]


def find(root, name):
    hits = glob.glob(os.path.join(root, "**", name), recursive=True)
    if not hits:
        raise FileNotFoundError(f"{name} not found under {root}")
    return hits[0]


def read_tsv(path):
    return pd.read_csv(path, sep="\t", keep_default_na=False)


def write_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows):>6} rows -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wjb_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_per_type", type=int, default=1000)
    ap.add_argument("--n_dev_per_type", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    train = read_tsv(find(args.wjb_dir, "train.tsv"))
    print("train.tsv columns:", list(train.columns), "rows:", len(train))
    print(train["data_type"].value_counts().to_string())
    # adversarial rows carry the jailbreak in `adversarial`; vanilla rows only have `vanilla`
    train["prompt"] = [a if a.strip() else v for a, v in zip(train["adversarial"], train["vanilla"])]

    train_rows, dev_rows = [], []
    for t in TYPES:
        sub = train[train["data_type"] == t].sample(frac=1.0, random_state=args.seed)
        need = args.n_per_type + args.n_dev_per_type
        if len(sub) < need:
            raise ValueError(f"{t}: only {len(sub)} rows, need {need}")
        for _, r in sub.iloc[: args.n_per_type].iterrows():
            train_rows.append({"prompt": r["prompt"], "response": r["completion"], "data_type": t})
        for _, r in sub.iloc[args.n_per_type: need].iterrows():
            dev_rows.append({"prompt": r["prompt"], "data_type": t})

    import random
    random.Random(args.seed).shuffle(train_rows)
    write_jsonl(train_rows, os.path.join(args.out_dir, "train.jsonl"))
    write_jsonl(dev_rows, os.path.join(args.out_dir, "dev.jsonl"))

    ev = read_tsv(find(args.wjb_dir, "eval.tsv"))
    print("eval.tsv columns:", list(ev.columns), "rows:", len(ev))
    test_rows = [
        {"prompt": r["adversarial"], "data_type": r["data_type"], "label": int(r["label"])}
        for _, r in ev.iterrows()
    ]
    write_jsonl(test_rows, os.path.join(args.out_dir, "test_jailbreak.jsonl"))


if __name__ == "__main__":
    main()
