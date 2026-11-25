"""Learned-steering figures in the team's shared style (base point + PFLOPs axis, examples on top).

    ../.venv/bin/python plot_report.py steering_v2 figures

Reads judged results from steering_v2/ and writes PNGs to the output folder; the per-checkpoint
numbers behind the figures go to steering_v2/report_metrics.json (local only).
"""
import json, math, sys, collections
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

V2 = Path(sys.argv[1])          # .../benchmarks/steering_v2
OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE = "#2a78d6", "#eb6834"
BG, GRID, AX, INK, MUTED = "#fcfcfa", "#e3e3e3", "#bdbdb5", "#4a4a4a", "#8a8a8a"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": AX,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG})

STEPS = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
BASE_X = 3.0e12                 # where the unsteered point sits on the log axis


def wilson(k, n):
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def read_judged(p):
    best = {}
    for line in p.read_text().splitlines():
        r = json.loads(line); k = (r["id"], r["judge"])
        if r.get("label") is not None or k not in best or best[k].get("label") is None:
            best[k] = r
    return list(best.values())


def rate(rows, hit, keep):
    rs = [r for r in rows if keep(r) and r["label"] is not None]
    return (sum(hit(r) for r in rs), len(rs)) if rs else None


def flops(step):
    if step == 0:
        return 0
    return json.loads((V2 / f"checkpoints/learned_L11_s182_step{step}/metadata.json").read_text())


# ---------- collect metrics per checkpoint ----------
SUMMARY = {r['variant']: r for r in json.loads((V2 / 'judged_summary.json').read_text()) if r['split'] == 'curve'}
M = {}
for step in [0] + STEPS:
    d = V2 / "results" / ("baseline_curve" if step == 0 else f"learned_L11_s182_step{step}")
    if not d.exists():
        continue
    m = {}
    srow = SUMMARY.get(d.name, {})
    if srow.get("harmbench_asr_n") == 200:
        m["hb_asr"] = (srow["harmbench_asr_k"], 200)
    if (d / "wildjailbreak_judged.jsonl").exists():
        w = read_judged(d / "wildjailbreak_judged.jsonl")
        if len(w) == 400:
            m["wj_asr"] = rate(w, lambda r: r["label"] == "yes", lambda r: r["judge"] == "harm")
            m["wj_ref_full"] = rate(w, lambda r: r["label"] == "2_full_refusal", lambda r: r["judge"] == "refusal")
            m["wj_ref_any"] = rate(w, lambda r: r["label"] in ("2_full_refusal", "3_partial_refusal"), lambda r: r["judge"] == "refusal")
    if (d / "xstest_judged.jsonl").exists():
        x = read_judged(d / "xstest_judged.jsonl")
        if len(x) == 450:
            m["xs_ref_full"] = rate(x, lambda r: r["label"] == "2_full_refusal", lambda r: r["xstest_label"] == "safe")
            m["xs_ref_any"] = rate(x, lambda r: r["label"] in ("2_full_refusal", "3_partial_refusal"), lambda r: r["xstest_label"] == "safe")
            m["xs_unsafe_comp"] = rate(x, lambda r: r["label"] == "1_full_compliance", lambda r: r["xstest_label"] == "unsafe")
    if srow.get("mmlu_accuracy_n") == 500:
        m["mmlu"] = (srow["mmlu_accuracy_k"], 500)
    elif (d / "mmlu_summary.json").exists():
        ms = json.loads((d / "mmlu_summary.json").read_text())
        if ms.get("completed") == 500:
            m["mmlu"] = (round(ms["accuracy"] * 500), 500)
    meta = flops(step)
    m["flops"] = 0 if step == 0 else meta["cumulative_training_flops"]
    m["tokens"] = 0 if step == 0 else meta["tokens_seen"]
    m["seconds"] = 0 if step == 0 else meta["training_seconds"]
    M[step] = m
json.dump(M, open(V2 / "report_metrics.json", "w"), indent=1)


def label(step):
    return "base" if step == 0 else (f"{step // 1024}k" if step >= 1024 else str(step))


def xpos(step):
    return BASE_X if step == 0 else M[step]["flops"]


def setup_axis(ax, title, ylabel, steps_shown):
    ax.set_xscale("log")
    ax.set_xlim(2.0e12, 6e15)
    ticks = [BASE_X, 1e13, 1e14, 1e15]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["0", "0.01", "0.1", "1"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("Training compute (PFLOPs, log scale)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold", pad=24)
    # examples-seen labels along the top, like the team figures
    for st in steps_shown:
        if st in (32, 128, 512, 2048):
            continue
        ax.annotate(label(st), (xpos(st), 1.0), xycoords=("data", "axes fraction"),
                    xytext=(0, 5), textcoords="offset points", ha="center", va="bottom",
                    fontsize=8.5, color=MUTED)
    # axis-break marks between the base point and the log range
    for xx in (4.6e12,):
        ax.annotate("//", (xx, 0), xycoords=("data", "axes fraction"), ha="center", va="center",
                    fontsize=10, color=AX, annotation_clip=False)


def series(ax, key, color, marker, name, scale=100):
    steps = [s for s in sorted(M) if M[s].get(key)]
    xs, ys, lo, hi = [], [], [], []
    for s in steps:
        k, n = M[s][key]
        p, a, b = wilson(k, n)
        xs.append(xpos(s)); ys.append(p * scale); lo.append((p - a) * scale); hi.append((b - p) * scale)
    # dashed link from the unsteered point into the log range
    if steps and steps[0] == 0:
        ax.plot(xs[:2], ys[:2], color=color, lw=1.6, ls=(0, (3, 2)), zorder=2)
        ax.plot(xs[1:], ys[1:], color=color, lw=2.6, zorder=2)
    else:
        ax.plot(xs, ys, color=color, lw=2.6, zorder=2)
    ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=color, elinewidth=1.4, capsize=4, capthick=1.4, zorder=3)
    ax.plot(xs, ys, ls="none", marker=marker, ms=8, color=color, mec="white", mew=1.3, zorder=4)
    HANDLES.setdefault(id(ax), []).append(matplotlib.lines.Line2D([], [], color=color, lw=2.6, marker=marker, ms=8,
                                                                  mec="white", mew=1.3, label=name))
    return steps


HANDLES = {}


def legend(ax, loc):
    ax.legend(handles=HANDLES.get(id(ax)), loc=loc, frameon=False, fontsize=10.5, labelcolor=INK, handlelength=2.2)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)


all_steps = sorted(M)

# 1. Safety vs compute
fig, ax = plt.subplots(figsize=(6.4, 4.6))
setup_axis(ax, "Safety vs. training compute", "Attack success rate (%)  ↓", all_steps)
series(ax, "hb_asr", BLUE, "o", "HarmBench")
series(ax, "wj_asr", ORANGE, "s", "WildJailbreak (unseen jailbreaks)")
ax.set_ylim(bottom=0)
legend(ax, "upper right")
save(fig, "ls_safety_vs_flops")

# 2. Over-refusal vs compute (full refusals)
fig, ax = plt.subplots(figsize=(6.4, 4.6))
setup_axis(ax, "Over-refusal vs. training compute", "Over-refusal rate (%)  ↓", all_steps)
series(ax, "wj_ref_full", BLUE, "o", "WildJailbreak adversarial-benign")
series(ax, "xs_ref_full", ORANGE, "s", "XSTest safe")
ax.set_ylim(0, 100)
legend(ax, "upper left")
save(fig, "ls_overrefusal_vs_flops")

# 3. Capability
fig, ax = plt.subplots(figsize=(6.4, 4.6))
setup_axis(ax, "MMLU (5-shot, 500-question subset)", "Accuracy (%)  ↑", all_steps)
series(ax, "mmlu", BLUE, "o", "MMLU")
ax.set_ylim(38, 56)
save(fig, "ls_capability_vs_flops")

# 4. Trade-off
KEY = [0, 16, 64, 256, 512, 1024, 2048, 4096, 8192]
OFF = {("wj", 0): (6, 4), ("wj", 16): (6, -12), ("wj", 64): (6, 4), ("wj", 256): (6, 4), ("wj", 1024): (-22, -12),
       ("wj", 4096): (6, 4), ("wj", 8192): (6, -12), ("wj", 512): (6, 4), ("wj", 2048): (6, 4),
       ("xs", 0): (7, -4), ("xs", 16): (-8, 6), ("xs", 64): (6, 4), ("xs", 256): (6, 4), ("xs", 1024): (6, 2),
       ("xs", 4096): (6, 4), ("xs", 8192): (6, 4), ("xs", 2048): (6, 4)}
fig, ax = plt.subplots(figsize=(6.0, 5.0))
for tag, key_x, key_y, color, marker, name in (("wj", "wj_ref_full", "wj_asr", BLUE, "o", "WildJailbreak eval"),
                                               ("xs", "xs_ref_full", "xs_unsafe_comp", ORANGE, "s", "XSTest")):
    steps = [s for s in KEY if M[s].get(key_x) and M[s].get(key_y)]
    xs = [100 * M[s][key_x][0] / M[s][key_x][1] for s in steps]
    ys = [100 * M[s][key_y][0] / M[s][key_y][1] for s in steps]
    ax.plot(xs, ys, ls="none", marker=marker, ms=8, color=color, mec="white", mew=1.3, zorder=3)
    HANDLES.setdefault(id(ax), []).append(matplotlib.lines.Line2D([], [], color=color, lw=0, marker=marker, ms=8,
                                                                  mec="white", mew=1.3, label=name))
    for s, x, y in zip(steps, xs, ys):
        ax.annotate(label(s), (x, y), xytext=OFF[(tag, s)], textcoords="offset points", fontsize=8.5, color=INK)
ax.set_xlim(-2, 85); ax.set_ylim(0, 25)
ax.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_xlabel("Over-refusal on benign prompts (%)  ↓")
ax.set_ylabel("Attack success rate (%)  ↓")
ax.set_title("Safety / over-refusal trade-off (lower-left is better)", loc="left", fontweight="bold", fontsize=11.5)
legend(ax, "upper right")
save(fig, "ls_tradeoff")

# 5. Training loss
log = [json.loads(l) for l in (V2 / "logs/train_L11_s182.jsonl").read_text().splitlines()]
steps = np.array([r["step"] for r in log]); loss = np.array([r["loss"] for r in log])
fig, ax = plt.subplots(figsize=(7.0, 4.2))
blk = 32
bx = steps[blk - 1::blk]; by = loss[: len(bx) * blk].reshape(-1, blk).mean(1)
ax.plot(bx, by, color=ORANGE, alpha=0.30, lw=1.1)
win = 8
ma = np.array([by[max(0, i - win + 1): i + 1].mean() for i in range(len(by))])
ax.plot(bx, ma, color=BLUE, lw=2.6)
HANDLES[id(ax)] = [matplotlib.lines.Line2D([], [], color=BLUE, lw=2.6, label=f"moving average ({win} logs = {win * blk} steps)"),
                   matplotlib.lines.Line2D([], [], color=ORANGE, alpha=0.4, lw=1.1, label=f"raw log ({blk}-step mean)")]
ax.axvline(4000, color=AX, lw=1.2, ls=(0, (4, 3)))
ax.annotate("1 epoch", (4000, 1.0), xycoords=("data", "axes fraction"), xytext=(4, -14),
            textcoords="offset points", fontsize=9, color=MUTED)
ax.set_xlim(0, 8300); ax.set_ylim(1.38, 1.85)
ax.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_xlabel("Training examples seen")
ax.set_ylabel("Training loss (response tokens)")
ax.set_title("Learned steering training loss (layer 11, seed 182)", loc="left", fontweight="bold")
legend(ax, "lower left")
save(fig, "ls_train_loss")
print(json.dumps({s: {k: v for k, v in m.items() if k not in ("tokens", "seconds")} for s, m in M.items()}))
