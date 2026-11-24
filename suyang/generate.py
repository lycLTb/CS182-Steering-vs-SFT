"""Generate baseline or DoM responses as judge-ready JSONL."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
import time

from shenghe_shao.soft_prompt import generate

from .dom import DoM, load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dom")
    ap.add_argument("--dom_strength", type=float, default=1.0)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--prompt_key", default="prompt")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", choices=("cuda", "mps", "cpu"), default="cuda")
    args = ap.parse_args()

    dom = DoM.load(args.dom, args.dom_strength) if args.dom else None
    model, tok = load_model(args.model, device=args.device)

    with open(args.input, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    method = "dom_l11" if dom is not None else "base"
    t0 = time.perf_counter()
    intervention = dom.apply(model) if dom is not None else nullcontext()
    with intervention, open(args.output, "w", encoding="utf-8") as out:
        for i in range(0, len(rows), args.batch_size):
            batch = rows[i: i + args.batch_size]
            prompts = [row[args.prompt_key] for row in batch]
            responses = generate(
                model, tok, None, prompts, args.max_new_tokens, device=args.device
            )
            for r, resp in zip(batch, responses):
                result = {**r, "method": method, "response": resp}
                if dom is not None:
                    prompt_id = hashlib.sha256(r[args.prompt_key].encode()).hexdigest()
                    result.setdefault("prompt_id", prompt_id)
                    result["config_id"] = dom.config_id
                out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
            print(f"{min(i + args.batch_size, len(rows))}/{len(rows)}  ({time.perf_counter() - t0:.0f}s)")
    print("done ->", args.output)


if __name__ == "__main__":
    main()
