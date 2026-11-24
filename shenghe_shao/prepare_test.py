"""Convert HarmBench and XSTest into the same jsonl format as dev.jsonl.

    python prepare_test.py --data_dir $WORK/cs182_assets/data --out_dir $WORK/cs182_assets/splits

Outputs (next to test_jailbreak.jsonl from prepare_data.py):
    test_harmbench.jsonl  {prompt, data_type="harmbench_harmful", category}   HarmBench standard behaviors
    test_xstest.jsonl     {prompt, data_type="xstest_benign"|"xstest_harmful", type}
        xstest_benign  = safe prompts that look harmful -> refusal here is over-refusal
        xstest_harmful = the unsafe contrast prompts    -> the model should refuse
"""
import argparse
import glob
import json
import os

import pandas as pd


def read_parquets(root, sub=""):
    files = sorted(glob.glob(os.path.join(root, sub, "**", "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no parquet files under {os.path.join(root, sub)}")
    print("reading:", [os.path.relpath(f, root) for f in files])
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print("columns:", list(df.columns), "rows:", len(df))
    return df


def write_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows):>5} rows -> {path}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    # HarmBench: use the "standard" subset (plain harmful behaviors, no extra context)
    hb_root = os.path.join(args.data_dir, "harmbench")
    sub = "standard" if os.path.isdir(os.path.join(hb_root, "standard")) else ""
    hb = read_parquets(hb_root, sub)
    write_jsonl(
        [{"prompt": r["prompt"], "data_type": "harmbench_harmful", "category": r.get("category", "")}
         for _, r in hb.iterrows()],
        os.path.join(args.out_dir, "test_harmbench.jsonl"),
    )

    # XSTest: label is "safe" / "unsafe"
    xs = read_parquets(os.path.join(args.data_dir, "xstest"))
    print("label counts:\n" + xs["label"].value_counts().to_string())
    write_jsonl(
        [{"prompt": r["prompt"],
          "data_type": "xstest_benign" if str(r["label"]).lower() == "safe" else "xstest_harmful",
          "type": r.get("type", "")}
         for _, r in xs.iterrows()],
        os.path.join(args.out_dir, "test_xstest.jsonl"),
    )


if __name__ == "__main__":
    main()
