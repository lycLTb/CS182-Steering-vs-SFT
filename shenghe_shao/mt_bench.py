"""MT-Bench: 80 two-turn questions, each answer graded 1-10 by DeepSeek
(single-answer grading, following the original MT-Bench judge prompts).

    # 1) generate answers (GPU)
    CUDA_VISIBLE_DEVICES=1 python mt_bench.py gen --model $WORK/cs182_assets/models/Qwen3.5-0.8B \
        --data_dir $WORK/cs182_assets/data/mt_bench --output $WORK/outputs/mtb_base.jsonl \
        [--soft_prompt .../soft_prompt_4000ex.pt]
    # 2) grade (API)
    python mt_bench.py judge --output $WORK/outputs/mtb_base.jsonl
Writes <output> (answers), <output>.judged.jsonl and <output>.summary.json
"""
import argparse
import glob
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

JUDGE_T1 = """[Instruction]
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. {ref_note}Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".
{ref}
[Question]
{q1}

[The Start of Assistant's Answer]
{a1}
[The End of Assistant's Answer]"""

JUDGE_T2 = """Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. {ref_note}You evaluation should focus on the assistant's answer to the second user question. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".
{ref}
<|The Start of Assistant A's Conversation with User|>

### User:
{q1}

### Assistant A:
{a1}

### User:
{q2}

### Assistant A:
{a2}

<|The End of Assistant A's Conversation with User|>"""


def load_questions(data_dir):
    """Accepts question.jsonl (FastChat format) or parquet; normalises to
    {id, category, turns: [q1, q2], reference: [r1, r2] or None}."""
    files = glob.glob(os.path.join(data_dir, "**", "*.jsonl"), recursive=True)
    if files:
        rows = [json.loads(l) for f in files for l in open(f, encoding="utf-8") if l.strip()]
        df = pd.DataFrame(rows)
    else:
        files = glob.glob(os.path.join(data_dir, "**", "*.parquet"), recursive=True)
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print("MT-Bench files:", [os.path.relpath(f, data_dir) for f in files], "columns:", list(df.columns))
    turns_col = "turns" if "turns" in df.columns else "prompt"
    id_col = next(c for c in ("question_id", "prompt_id", "id") if c in df.columns)
    ref_col = next((c for c in ("reference", "references") if c in df.columns), None)
    qs = []
    for _, r in df.iterrows():
        ref = list(r[ref_col]) if ref_col is not None and r[ref_col] is not None and len(r[ref_col]) else None
        qs.append({"id": str(r[id_col]), "category": r.get("category", ""), "turns": list(r[turns_col]), "reference": ref})
    print(f"{len(qs)} questions")
    return qs


def cmd_gen(args):
    from soft_prompt import SoftPrompt, generate, load_model
    model, tok = load_model(args.model)
    soft = None
    if args.soft_prompt:
        soft, _ = SoftPrompt.load(args.soft_prompt)
        soft = soft.cuda()
    qs = load_questions(args.data_dir)
    # MT-Bench is run WITHOUT a system prompt, same as every other eval in this project
    convs = [[{"role": "user", "content": q["turns"][0]}] for q in qs]
    a1 = []
    for i in range(0, len(qs), args.batch_size):
        a1 += generate(model, tok, soft, convs[i: i + args.batch_size], args.max_new_tokens)
        print(f"turn 1: {len(a1)}/{len(qs)}")
    convs = [c + [{"role": "assistant", "content": a}, {"role": "user", "content": q["turns"][1]}]
             for c, a, q in zip(convs, a1, qs)]
    a2 = []
    for i in range(0, len(qs), args.batch_size):
        a2 += generate(model, tok, soft, convs[i: i + args.batch_size], args.max_new_tokens)
        print(f"turn 2: {len(a2)}/{len(qs)}")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for q, x, y in zip(qs, a1, a2):
            f.write(json.dumps({**q, "answers": [x, y], "soft_prompt": args.soft_prompt}, ensure_ascii=False) + "\n")
    print("done ->", args.output)


def judge_prompts(r):
    ref = r.get("reference")
    out = []
    for turn in (0, 1):
        if ref:
            note = "Your evaluation should consider correctness and helpfulness. You will be given a reference answer. Identify and correct any mistakes. "
            block = f"\n[The Start of Reference Answer]\n{ref[turn] if turn < len(ref) else ref[-1]}\n[The End of Reference Answer]\n"
        else:
            note, block = "", ""
        t = JUDGE_T1 if turn == 0 else JUDGE_T2
        out.append(t.format(ref_note=note, ref=block, q1=r["turns"][0], a1=r["answers"][0],
                            q2=r["turns"][1], a2=r["answers"][1]))
    return out


def cmd_judge(args):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url=args.base_url)
    rows = [json.loads(l) for l in open(args.output, encoding="utf-8") if l.strip()]

    def grade(prompt):
        for attempt in range(5):
            try:
                txt = client.chat.completions.create(
                    model=args.judge_model, messages=[{"role": "user", "content": prompt}],
                    temperature=0, max_tokens=1024, extra_body={"thinking": {"type": "disabled"}},
                ).choices[0].message.content
                m = re.search(r"\[\[(\d+\.?\d*)\]\]", txt)
                return (float(m.group(1)) if m else None), txt
            except Exception as e:
                err = repr(e)
                time.sleep(2 ** attempt)
        return None, f"ERROR {err}"

    jobs = [(i, t, p) for i, r in enumerate(rows) for t, p in enumerate(judge_prompts(r))]
    with ThreadPoolExecutor(args.workers) as ex:
        results = list(ex.map(lambda j: grade(j[2]), jobs))
    for (i, t, _), (score, txt) in zip(jobs, results):
        rows[i].setdefault("scores", [None, None])[t] = score
        rows[i].setdefault("judge_raw", [None, None])[t] = txt

    base = args.output.replace(".jsonl", "")
    with open(base + ".judged.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    s1 = [r["scores"][0] for r in rows if r["scores"][0] is not None]
    s2 = [r["scores"][1] for r in rows if r["scores"][1] is not None]
    summary = {"turn1": sum(s1) / len(s1), "turn2": sum(s2) / len(s2),
               "overall": (sum(s1) + sum(s2)) / (len(s1) + len(s2)),
               "n_failed": 2 * len(rows) - len(s1) - len(s2)}
    json.dump(summary, open(base + ".summary.json", "w"), indent=2)
    print("MT-Bench:", summary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["gen", "judge"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--model")
    ap.add_argument("--data_dir")
    ap.add_argument("--soft_prompt", default=None)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--judge_model", default="deepseek-flash")
    ap.add_argument("--base_url", default="https://api.deepseek.com")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    cmd_gen(args) if args.stage == "gen" else cmd_judge(args)


if __name__ == "__main__":
    main()
