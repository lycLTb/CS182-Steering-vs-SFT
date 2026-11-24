"""Generate responses for the safety benchmarks + MT-Bench, save JSONL (judged offline later).

Usage:
  python scripts/generate.py --bench harmbench_test xstest --limit 16 --out outputs/smoke
  python scripts/generate.py --bench all --out outputs/baseline
  python scripts/generate.py --model models/<name> --bench all --out outputs/<name>
(--out is relative to yanchi/)

Resumable: rows whose id already exists in the output file are skipped (WSL crashes...).
Prompts are sorted by length and batched (left padding) for throughput.
"""
import argparse
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make yanchi/core importable

from core.io import append_jsonl, done_ids, read_jsonl
from core.model import (DEFAULT_MODEL, MAX_NEW_TOKENS, STOP_TOKENS, Timer, generate_batch, load_model,
                        model_info, render)
from core.paths import DATA, ROOT
import torch
import transformers

SINGLE_TURN = {
    "harmbench_test": "harmbench_test.jsonl",
    "harmbench_val": "harmbench_val.jsonl",
    "wj_eval_adv_harmful": "wj_eval_adv_harmful.jsonl",
    "wj_eval_adv_benign": "wj_eval_adv_benign.jsonl",
    "xstest": "xstest.jsonl",
}
DEFAULT = ["harmbench_test", "wj_eval_adv_harmful", "wj_eval_adv_benign", "xstest", "mtbench"]


def run_single_turn(model, tok, name, out_dir, bs, limit, max_new_tokens):
    rows = read_jsonl(DATA / SINGLE_TURN[name])
    if limit:
        rows = rows[:limit]
    out = out_dir / f"{name}.jsonl"
    skip = done_ids(out)
    todo = [r for r in rows if r["id"] not in skip]
    print(f"[{name}] {len(rows)} prompts, {len(skip)} done, {len(todo)} to go", flush=True)
    for r in todo:
        r["_text"] = render(tok, [{"role": "user", "content": r["prompt"]}])
        r["_len"] = len(tok(r["_text"], add_special_tokens=False).input_ids)
    todo.sort(key=lambda r: r["_len"], reverse=True)  # longest first -> OOM shows up early
    tot_t, tot_tok = 0.0, 0
    for i in range(0, len(todo), bs):
        batch = todo[i:i + bs]
        with Timer() as t:
            gens = generate_batch(model, tok, [r["_text"] for r in batch], max_new_tokens)
        tot_t += t.s
        tot_tok += sum(g["n_new_tokens"] for g in gens)
        append_jsonl(out, [{**{k: v for k, v in r.items() if not k.startswith("_")},
                            "prompt_tokens": r["_len"], **g} for r, g in zip(batch, gens)])
        print(f"  [{name}] {min(i + bs, len(todo))}/{len(todo)}  batch {t.s:.1f}s  "
              f"maxlen {batch[0]['_len']}", flush=True)
    return {"n": len(todo), "seconds": tot_t, "new_tokens": tot_tok}


def run_mtbench(model, tok, out_dir, bs, limit, max_new_tokens):
    qs = read_jsonl(DATA / "mtbench_questions.jsonl")
    if limit:
        qs = qs[:limit]
    out = out_dir / "mtbench.jsonl"
    skip = done_ids(out)
    todo = [{"id": f"mtbench_{q['question_id']}", **q} for q in qs if f"mtbench_{q['question_id']}" not in skip]
    print(f"[mtbench] {len(qs)} questions, {len(skip)} done, {len(todo)} to go", flush=True)
    tot_t, tot_tok = 0.0, 0
    for i in range(0, len(todo), bs):
        batch = todo[i:i + bs]
        # turn 1
        msgs = [[{"role": "user", "content": q["turns"][0]}] for q in batch]
        with Timer() as t1:
            g1 = generate_batch(model, tok, [render(tok, m) for m in msgs], max_new_tokens)
        # turn 2 conditioned on our own turn-1 answer
        for m, g, q in zip(msgs, g1, batch):
            m += [{"role": "assistant", "content": g["response"]}, {"role": "user", "content": q["turns"][1]}]
        with Timer() as t2:
            g2 = generate_batch(model, tok, [render(tok, m) for m in msgs], max_new_tokens)
        tot_t += t1.s + t2.s
        tot_tok += sum(g["n_new_tokens"] for g in g1 + g2)
        append_jsonl(out, [{"id": q["id"], "question_id": q["question_id"], "category": q["category"],
                            "turns": q["turns"], "responses": [a["response"], b["response"]],
                            "n_new_tokens": [a["n_new_tokens"], b["n_new_tokens"]],
                            "finish_reason": [a["finish_reason"], b["finish_reason"]]}
                           for q, a, b in zip(batch, g1, g2)])
        print(f"  [mtbench] {min(i + bs, len(todo))}/{len(todo)}  {t1.s + t2.s:.1f}s", flush=True)
    return {"n": len(todo), "seconds": tot_t, "new_tokens": tot_tok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", nargs="+", default=["all"])
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="model dir saved with save_pretrained (default: exported base), e.g. models/sft-xxx")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="first N prompts per benchmark (smoke tests)")
    ap.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    a = ap.parse_args()
    benches = DEFAULT if a.bench == ["all"] else a.bench

    out_dir = ROOT / a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    model, tok = load_model(a.model)
    meta = {**model_info(a.model), "dtype": "bfloat16", "stop_tokens": STOP_TOKENS, "decoding": "greedy", "max_new_tokens": a.max_new_tokens,
            "batch_size": a.bs, "thinking": False,
            "torch": torch.__version__, "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0), "host": platform.node(), "timing": {}}
    for b in benches:
        if b == "mtbench":
            meta["timing"][b] = run_mtbench(model, tok, out_dir, a.bs, a.limit, a.max_new_tokens)
        else:
            meta["timing"][b] = run_single_turn(model, tok, b, out_dir, a.bs, a.limit, a.max_new_tokens)
    meta["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
    with open(out_dir / f"meta_{'_'.join(benches)}.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps({k: v for k, v in meta.items() if k in ("timing", "peak_mem_gb")}, indent=2))


if __name__ == "__main__":
    main()
