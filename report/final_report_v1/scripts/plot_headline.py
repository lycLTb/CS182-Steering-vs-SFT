"""Headline figure: all methods vs. training compute (test sets).

Run from the repo root:
    python report/final_report_v1/scripts/plot_headline.py

Panels: (a) HarmBench attack success rate, (b) over-refusal on XSTest safe prompts, (c) MMLU accuracy.
Each method starts from its own base-model point. DoM is a two-piece step: base until its one-time
extraction cost, the steered value afterwards. Error bars are 95% Wilson intervals.
A method is drawn in a panel only if it has data for that metric.
"""
import csv
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(ROOT, "report", "final_report_v1", "figures", "headline_vs_flops.png")

# Fixed method -> color/marker mapping (same palette as the individual reports). LoRA slot reserved.
METHODS = {
    "DoM steering":     dict(color="#2a78d6", marker="o"),
    "Learned steering": dict(color="#eb6834", marker="s"),
    "Soft prompt":      dict(color="#1baf7a", marker="^"),
    "LoRA":             dict(color="#eda100", marker="D"),
}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

X_BASE = 0.002   # where the base model (0 FLOPs) is drawn on the log axis
X_MAX = 4.5      # right end of the DoM step
DOM_PFLOPS = 9.68e12 / 1e15  # one-time extraction cost (DoM report)

# panel key -> (csv column, csv n column, DoM summary file, DoM field)
PANELS = [
    ("(a) Attack success rate on HarmBench ↓", "Attack success rate (%)",
     "HarmBench ASR", "HarmBench ASR n", "test_harmbench_{}.summary.json", "ASR"),
    ("(b) Over-refusal on XSTest safe prompts ↓", "Refusal rate (%)",
     "XSTest-safe over-refusal", "XSTest-safe over-refusal n", "test_xstest_{}.summary.json", "over_refusal"),
    ("(c) MMLU accuracy ↑", "Accuracy (%)",
     "MMLU accuracy", "MMLU accuracy n", None, None),
]


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
        "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "legend.frameon": False, "legend.labelcolor": INK2,
        "lines.linewidth": 2, "lines.markersize": 7, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def wilson(p, n, z=1.96):
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def series_from_csv(rows, col, ncol):
    """[(pflops, p, n)] for rows that have this metric; base row gets x = 0."""
    out = []
    for r in rows:
        if r.get(col):
            out.append((float(r["train_PFLOPs"]), float(r[col]), int(r[ncol])))
    return out


def load_dom(fname, field):
    d = os.path.join(ROOT, "suyang", "budget_10tflops_results1")
    vals = []
    for tag in ("base", "dom"):
        with open(os.path.join(d, fname.format(tag))) as f:
            s = json.load(f)
        vals.append((s[field], s["n"]))
    return vals  # [(p_base, n), (p_dom, n)]


def draw_curve(ax, pts, m):
    """pts: [(pflops, p, n)] with the base at pflops == 0."""
    xs = [X_BASE if x == 0 else x for x, _, _ in pts]
    ys = [100 * p for _, p, _ in pts]
    lo_hi = [wilson(p, n) for _, p, n in pts]
    err = [[y - 100 * lo for y, (lo, _) in zip(ys, lo_hi)], [100 * hi - y for y, (_, hi) in zip(ys, lo_hi)]]
    # base -> first trained point is not to scale on a log axis: draw it dashed
    ax.plot(xs[:2], ys[:2], color=m["color"], ls="--", lw=1.5, zorder=2)
    ax.plot(xs[1:], ys[1:], color=m["color"], zorder=2)
    ax.errorbar(xs, ys, yerr=err, fmt=m["marker"], color=m["color"], mec="white", mew=1.2,
                capsize=3, elinewidth=1.2, zorder=3)


def draw_dom_step(ax, base, dom, m):
    (pb, nb), (pd, nd) = base, dom
    yb, yd = 100 * pb, 100 * pd
    ax.plot([X_BASE, DOM_PFLOPS, DOM_PFLOPS, X_MAX], [yb, yb, yd, yd],
            color=m["color"], ls="--", lw=1.5, zorder=2)
    for x, p, n in ((X_BASE, pb, nb), (DOM_PFLOPS, pd, nd)):
        lo, hi = wilson(p, n)
        ax.errorbar([x], [100 * p], yerr=[[100 * (p - lo)], [100 * (hi - p)]], fmt=m["marker"],
                    color=m["color"], mec="white", mew=1.2, capsize=3, elinewidth=1.2, zorder=3)


def format_xaxis(ax):
    ax.set_xscale("log")
    ax.set_xlim(X_BASE / 1.6, X_MAX * 1.6)
    ax.set_xticks([X_BASE, 0.01, 0.1, 1])
    ax.set_xticklabels(["0", "0.01", "0.1", "1"])
    ax.minorticks_off()
    ax.set_xlabel("Training compute (PFLOPs, log scale)")
    # break mark between the base slot and the log-scaled part
    xb = math.sqrt(X_BASE * 0.006)
    ax.text(xb, 0, "//", transform=ax.get_xaxis_transform(), ha="center", va="center",
            color=MUTED, fontsize=9, clip_on=False)


def main():
    style()
    sp = read_csv(os.path.join(ROOT, "shenghe_shao", "figures", "results_test.csv"))
    ls = read_csv(os.path.join(ROOT, "Jia_V1", "learned_steering", "figures", "results_learned_steering.csv"))

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (title, ylabel, col, ncol, dom_file, dom_field) in zip(axes, PANELS):
        if dom_file:
            base, dom = load_dom(dom_file, dom_field)
            draw_dom_step(ax, base, dom, METHODS["DoM steering"])
        draw_curve(ax, series_from_csv(ls, col, ncol), METHODS["Learned steering"])
        draw_curve(ax, series_from_csv(sp, col, ncol), METHODS["Soft prompt"])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        format_xaxis(ax)
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylim(bottom=0)

    handles = [plt.Line2D([], [], color=METHODS[k]["color"], marker=METHODS[k]["marker"], mec="white",
                          ls="--" if k == "DoM steering" else "-", label=k)
               for k in ("DoM steering", "Learned steering", "Soft prompt")]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
