"""Small-batch test of the DeepSeek judge: does the API's safety filter block harmful judge inputs?

Sends ~30 judge requests built from outputs/baseline (deliberately over-sampling responses that look *harmful*,
since those are what a provider filter would block) and reports, per benchmark:
  ok / output_blocked (finish_reason=content_filter) / input_blocked (HTTP 400 "Content Exists Risk") /
  other API errors / unparseable judge outputs.

Usage:
  python scripts/judge_smoketest.py --dry_run   # build + print prompts only, no API calls
  python scripts/judge_smoketest.py             # needs env DEEPSEEK_API_KEY or yanchi/.deepseek_api_key
"""
import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make yanchi/core importable
from core import deepseek  # noqa: E402
from core import judge as jp  # noqa: E402
from core.io import read_jsonl  # noqa: E402
from core.paths import DATA, OUTPUTS  # noqa: E402

# ------------------------------------------------------------------ config
# API key / base URL / model: core/deepseek.py
THINKING = {"type": "disabled"}  # thinking is ON by default; temperature is ignored in thinking mode
TEMPERATURE = 0.0
MAX_TOKENS = {"harmbench": 8, "xstest": 16, "mtbench": 2048}  # HarmBench cls: 1 token; XSTest: 16; FastChat: 2048

RESP_DIR = OUTPUTS / "baseline"
OUT_DIR = OUTPUTS / "judge_smoke"
SEED = 0


# ------------------------------------------------------------------ build the test set
def build_requests(n_harm_comply=6, n_harm_refuse=2):
    rng = random.Random(SEED)
    reqs = []

    def pick(rows, k):
        return rng.sample(rows, min(k, len(rows)))

    # harmfulness judge: HarmBench + WildJailbreak adversarial-harmful (behaviour = the prompt given to the model)
    for bench in ["harmbench_test", "wj_eval_adv_harmful"]:
        rows = read_jsonl(RESP_DIR / f"{bench}.jsonl")
        comply = [r for r in rows if not jp.keyword_refusal(r["response"])]
        refuse = [r for r in rows if jp.keyword_refusal(r["response"])]
        for r in pick(comply, n_harm_comply) + pick(refuse, n_harm_refuse):
            reqs.append({"bench": bench, "judge": "harmbench", "id": r["id"], "kw_refusal": jp.keyword_refusal(r["response"]),
                         "messages": jp.harmbench_messages(r["prompt"], r["response"])})

    # refusal judge: WildJailbreak adversarial-benign + XSTest safe / unsafe
    benign = read_jsonl(RESP_DIR / "wj_eval_adv_benign.jsonl")
    xs = read_jsonl(RESP_DIR / "xstest.jsonl")
    for bench, rows in [("wj_eval_adv_benign", pick(benign, 3)),
                        ("xstest_safe", pick([r for r in xs if r["label"] == "safe"], 3)),
                        ("xstest_unsafe", pick([r for r in xs if r["label"] == "unsafe"], 3))]:
        for r in rows:
            reqs.append({"bench": bench, "judge": "xstest", "id": r["id"], "kw_refusal": jp.keyword_refusal(r["response"]),
                         "messages": jp.xstest_messages(r["prompt"], r["response"])})

    # quality judge: MT-Bench (one ref-category question + two general; one turn-2)
    qs = {f"mtbench_{q['question_id']}": q for q in read_jsonl(DATA / "mtbench_questions.jsonl")}
    mt = {r["id"]: r for r in read_jsonl(RESP_DIR / "mtbench.jsonl")}
    ref_ids = [i for i in mt if qs[i]["category"] in jp.NEED_REF_CATS]
    gen_ids = [i for i in mt if qs[i]["category"] not in jp.NEED_REF_CATS]
    picks = [(i, 1) for i in pick(ref_ids, 1) + pick(gen_ids, 2)] + [(pick(ref_ids, 1)[0], 2)]
    for i, turn in picks:
        msgs, name = jp.mtbench_messages(qs[i], mt[i]["responses"], turn)
        reqs.append({"bench": "mtbench", "judge": "mtbench", "id": f"{i}_t{turn}", "template": name,
                     "messages": msgs})
    return reqs


# ------------------------------------------------------------------ API
def call(client, req, retries=4):
    import openai
    rec = {k: v for k, v in req.items() if k != "messages"}
    for attempt in range(retries + 1):
        t = time.time()
        try:
            r = client.chat.completions.create(
                model=deepseek.MODEL, messages=req["messages"], temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS[req["judge"]], extra_body={"thinking": THINKING})
            c = r.choices[0]
            rec.update(status="output_blocked" if c.finish_reason == "content_filter" else "ok",
                       finish_reason=c.finish_reason, content=c.message.content, served_model=r.model,
                       system_fingerprint=getattr(r, "system_fingerprint", None),
                       usage=r.usage.model_dump() if r.usage else None)
            break
        except openai.APIStatusError as e:
            msg = str(getattr(e, "message", e))
            rec.update(http_status=e.status_code, error=msg[:500])
            if e.status_code == 400 and "risk" in msg.lower():
                rec["status"] = "input_blocked"
                break
            if e.status_code in (429, 500, 503, 504) and attempt < retries:
                time.sleep(2 ** attempt * 2)
                continue
            rec["status"] = f"http_{e.status_code}"
            break
        except openai.APIConnectionError as e:
            rec.update(error=str(e)[:500], status="connection_error")
            if attempt < retries:
                time.sleep(2 ** attempt * 2)
                continue
            break
    rec["latency_s"] = round(time.time() - t, 2)
    rec["attempts"] = attempt + 1
    parse = {"harmbench": jp.parse_harmbench, "xstest": jp.parse_xstest, "mtbench": jp.parse_mtbench}[req["judge"]]
    rec["parsed"] = parse(rec.get("content")) if rec.get("status") == "ok" else None
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    reqs = build_requests()
    print(f"{len(reqs)} judge requests:", dict(Counter(r["bench"] for r in reqs)))
    if a.dry_run:
        for j in ["harmbench", "xstest", "mtbench"]:
            r = next(x for x in reqs if x["judge"] == j)
            print(f"\n{'=' * 30} {j} example ({r['id']}) {'=' * 30}")
            for m in r["messages"]:
                print(f"--- [{m['role']}]\n{m['content'][:1500]}{' …' if len(m['content']) > 1500 else ''}")
        return

    client = deepseek.client(timeout=120)

    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        recs = list(ex.map(lambda r: call(client, r), reqs))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"raw_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    with open(out, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---------------- report
    print(f"\ndone in {time.time() - t0:.0f}s -> {out}")
    print(f"served model(s): {dict(Counter(r.get('served_model') for r in recs))} | "
          f"fingerprints: {dict(Counter(r.get('system_fingerprint') for r in recs))}")
    print(f"\n{'bench':20s} {'n':>3s} {'ok':>4s} {'out_blk':>8s} {'in_blk':>7s} {'err':>4s} {'unparsed':>9s}")
    by = defaultdict(list)
    for r in recs:
        by[r["bench"]].append(r)
    for b, rs in by.items():
        st = Counter(r.get("status") for r in rs)
        err = sum(v for k, v in st.items() if k not in ("ok", "output_blocked", "input_blocked"))
        unp = sum(r["status"] == "ok" and r["parsed"] is None for r in rs)
        print(f"{b:20s} {len(rs):3d} {st['ok']:4d} {st['output_blocked']:8d} {st['input_blocked']:7d} {err:4d} {unp:9d}")

    print("\nper-request (harmful-looking responses are where a filter would bite):")
    for r in recs:
        kw = "" if r["judge"] == "mtbench" else ("kw:refuse" if r["kw_refusal"] else "kw:comply")
        out_txt = (r.get("content") or r.get("error") or "").replace("\n", " ")[:70]
        print(f"  {r['bench']:19s} {r['id'][:28]:28s} {kw:9s} {r.get('status'):15s} parsed={r['parsed']!s:16s} "
              f"{r['latency_s']:5.1f}s  {out_txt!r}")
    tok = Counter()
    for r in recs:
        for k, v in (r.get("usage") or {}).items():
            if isinstance(v, int):
                tok[k] += v
    print("\ntoken usage:", dict(tok))


if __name__ == "__main__":
    main()
