import json
from pathlib import Path


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def done_ids(path):
    """IDs already present in an output JSONL (for resumable runs)."""
    p = Path(path)
    return {r["id"] for r in read_jsonl(p)} if p.exists() else set()


def append_jsonl(path, rows):
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
