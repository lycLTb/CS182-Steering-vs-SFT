#!/usr/bin/env python3
"""Fetch benchmark inputs from their publishers, recording immutable revisions."""
import csv
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import urllib.error

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "CS182-local-research/1.0"})
    with urllib.request.urlopen(req, timeout=60) as response:
        return response.read()


def prepare():
    DATA.mkdir(parents=True, exist_ok=True)
    manifest_path = DATA / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    sources = {
        "harmbench.csv": ("centerforaisafety/HarmBench", "data/behavior_datasets/harmbench_behaviors_text_all.csv"),
        "xstest.csv": ("paul-rottger/xstest", "xstest_prompts.csv"),
        "mt_bench.jsonl": ("lm-sys/FastChat", "fastchat/llm_judge/data/mt_bench/question.jsonl"),
        "judge_prompts.jsonl": ("lm-sys/FastChat", "fastchat/llm_judge/data/judge_prompts.jsonl"),
        "mt_reference.jsonl": ("lm-sys/FastChat", "fastchat/llm_judge/data/mt_bench/reference_answer/gpt-4.jsonl"),
    }
    revisions = {}
    for name, (repo, path) in sources.items():
        target = DATA / name
        if target.exists() and name in manifest:
            assert hashlib.sha256(target.read_bytes()).hexdigest() == manifest[name]["sha256"]
            continue
        if repo not in revisions:
            info = json.loads(fetch(f"https://api.github.com/repos/{repo}/commits/HEAD"))
            revisions[repo] = info["sha"]
        revision = revisions[repo]
        url = f"https://raw.githubusercontent.com/{repo}/{revision}/{path}"
        blob = fetch(url)
        target.write_bytes(blob)
        manifest[name] = {"source": url, "revision": revision, "sha256": hashlib.sha256(blob).hexdigest()}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print("Downloaded", name, len(blob), flush=True)
    # The dataset publisher maintains an aggregate 'all' split, avoiding 57 requests.
    from huggingface_hub import HfApi, hf_hub_download, get_token
    api = HfApi()
    revision = manifest.get("mmlu", {}).get("revision") or api.dataset_info("cais/mmlu").sha
    import pyarrow.parquet as pq
    for split in ("test", "dev", "validation"):
        name = f"mmlu_{split}.jsonl"
        target = DATA / name
        if target.exists():
            assert hashlib.sha256(target.read_bytes()).hexdigest() == manifest[name]["sha256"]
            continue
        file = f"all/{split}-00000-of-00001.parquet"
        local = hf_hub_download("cais/mmlu", file, repo_type="dataset", revision=revision,
                                cache_dir=str(DATA / "hf-cache"))
        rows = pq.read_table(local).to_pylist()
        target.write_text("".join(json.dumps(r) + "\n" for r in rows))
        manifest[name] = {"source": "https://huggingface.co/datasets/cais/mmlu", "revision": revision,
                          "rows": len(rows), "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
        manifest["mmlu"] = {"revision": revision}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print("Downloaded", name, len(rows), "rows", flush=True)
    # Do not substitute a mirror or bypass the publisher's access agreement.
    for split in ("train", "eval"):
        name = f"wildjailbreak_{split}.tsv"
        target = DATA / name
        if target.exists():
            continue
        if not get_token():
            manifest[name] = {"status": "blocked", "reason": "Authenticate with Hugging Face after accepting dataset access"}
            continue
        try:
            rev = manifest.get("wildjailbreak", {}).get("revision") or api.dataset_info("allenai/wildjailbreak").sha
            local = hf_hub_download("allenai/wildjailbreak", f"{split}/{split}.tsv", repo_type="dataset",
                                    revision=rev, cache_dir=str(DATA / "hf-cache"))
            import shutil
            shutil.copyfile(local, target)
            manifest[name] = {"source": "https://huggingface.co/datasets/allenai/wildjailbreak", "revision": rev,
                              "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
            manifest["wildjailbreak"] = {"revision": rev}
        except Exception as exc:
            manifest[name] = {"status": "blocked", "reason": f"{type(exc).__name__}: {exc}"}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    for name in sources:
        p = DATA / name
        rows = list(csv.DictReader(io.StringIO(p.read_text()))) if p.suffix == ".csv" else [json.loads(x) for x in p.read_text().splitlines()]
        print(name, "rows:", len(rows), "fields:", list(rows[0]), flush=True)


if __name__ == "__main__":
    prepare()
