"""Download the model and datasets used in the project.

Run this ON THE SERVER (not locally):
    pip install -U huggingface_hub
    hf auth login            # needed for the gated WildJailbreak dataset
    python download_assets.py --out /path/to/big/disk/cs182_assets

Large files are kept outside the git repo on purpose.
"""
import argparse
import os

from huggingface_hub import snapshot_download

MODEL = "Qwen/Qwen3.5-0.8B"

DATASETS = {
    "wildjailbreak": "allenai/wildjailbreak",  # gated: accept terms on the HF page first
    "harmbench": "walledai/HarmBench",
    "xstest": "walledai/XSTest",
    "mmlu": "cais/mmlu",
    "mt_bench": "HuggingFaceH4/mt_bench_prompts",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="directory on a large disk")
    parser.add_argument("--skip-model", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if not args.skip_model:
        path = snapshot_download(MODEL, local_dir=os.path.join(args.out, "models", MODEL.split("/")[-1]))
        print(f"[ok] model -> {path}")

    for name, repo in DATASETS.items():
        try:
            path = snapshot_download(repo, repo_type="dataset", local_dir=os.path.join(args.out, "data", name))
            print(f"[ok] {name} -> {path}")
        except Exception as e:  # keep going if one dataset fails (e.g. gated access not granted)
            print(f"[FAIL] {name} ({repo}): {e}")


if __name__ == "__main__":
    main()
