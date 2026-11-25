"""Figures for the LoRA section of the report (dev results).

    python scripts/plot_report.py            # -> figures/lora_train_loss.png, figures/lora_dev_vs_flops.png

Inputs (all produced by train_lora.py / run_eval_lora.sh):
  runs/<run>/train_log.jsonl                 training loss every 10 steps (one batch of 8)
  runs/<run>/adapter_<N>ex/train_state.json  training FLOPs at each checkpoint
  outputs/lora/<run>/<N>ex_dev.judged.jsonl  per-prompt judge labels on dev (4 types x 100)
  outputs/lora/<run>/mmlu_<N>ex.json         MMLU accuracy (1000 questions)
Error bars: 95% Wilson score intervals (binomial proportions; robust near 0%, where several ASR values sit).
"""
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUNS, OUT, FIG = ROOT / "runs", ROOT / "outputs" / "lora", ROOT / "figures"

# reference categorical palette (fixed order, validated) + neutral inks; same style as the soft-prompt figures
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

# target-module configurations at lr 2e-4: label -> (eval dir, [training runs holding the adapters])
CONFIGS = {
    "softmax-only (1.1M)": ("lora_softmax_r16_lr2e-4", ["lora_softmax_r16_lr2e-4"]),
    "attn+GDN (4.9M)": ("lora_r16_lr2e-4", ["lora_r16_lr2e-4"]),
    "all-linear (10.2M)": ("lora_all_r16_lr2e-4", ["lora_all_r16_lr2e-4_upto512", "lora_all_r16_lr2e-4"]),
}


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
        "axes.titlecolor": INK, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 9, "legend.frameon": False, "legend.labelcolor": INK2,
        "lines.linewidth": 2, "lines.markersize": 6, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, p - (c - h), (c + h) - p  # point, lower error, upper error


def read_jsonl(p):
    return [json.loads(l) for l in open(p)]


def dev_metrics(judged, mmlu):
    rows = read_jsonl(judged)
    harm = [r for r in rows if r["data_type"].endswith("_harmful")]
    advb = [r for r in rows if r["data_type"] == "adversarial_benign"]
    acc = json.load(open(mmlu))
    n_mmlu = acc.get("n", 1000)
    return {
        "asr": wilson(sum(r["judge_harmful"] for r in harm), len(harm)),
        "advb": wilson(sum(r["judge_refusal"] for r in advb), len(advb)),
        "mmlu": wilson(round(acc["accuracy"] * n_mmlu), n_mmlu),
    }


def checkpoints(eval_dir, runs):
    """[(examples, flops, metrics)] for every budget that has both an adapter and dev results."""
    pts = {}
    for run in runs:
        for st in (RUNS / run).glob("adapter_*ex/train_state.json"):
            s = json.load(open(st))
            tag = s["tag"]
            judged, mmlu = OUT / eval_dir / f"{tag}_dev.judged.jsonl", OUT / eval_dir / f"mmlu_{tag}.json"
            if judged.exists() and mmlu.exists() and len(read_jsonl(judged)) == 400:
                pts[s["examples_seen"]] = (s["examples_seen"], s["train_flops"], dev_metrics(judged, mmlu))
    return [pts[k] for k in sorted(pts)]


def smooth(y, w):
    return [sum(y[max(0, i - w + 1): i + 1]) / len(y[max(0, i - w + 1): i + 1]) for i in range(len(y))]


def loss_curve(run):
    log = read_jsonl(RUNS / run / "train_log.jsonl")
    return [r["examples_seen"] for r in log], [r["loss"] for r in log]


def fig_loss():
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for i, (lab, (_, runs)) in enumerate(CONFIGS.items()):
        x, y = loss_curve(runs[-1])
        ax.plot(x, smooth(y, 5), color=SERIES[i], lw=1.8, label=lab)
    ax.set_title("LoRA target modules, lr=2e-4")
    ax.set_xlabel("Training examples seen")
    ax.set_xlim(0, 4000)
    ax.legend(fontsize=8)
    ax.set_ylabel("Training loss")
    fig.savefig(FIG / "lora_train_loss.png")
    plt.close(fig)


def fig_metrics():
    base_dir = OUT / "_base"
    base = dev_metrics(base_dir / "base_dev.judged.jsonl", base_dir / "mmlu_base.json")
    panels = [("asr", "(a) Attack success rate ↓", "Attack success rate (%)"),
              ("advb", "(b) Over-refusal ↓", "Refusal on adversarial benign (%)"),
              ("mmlu", "(c) Capability ↑", "MMLU accuracy (%)")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    data = {lab: checkpoints(ev, runs) for lab, (ev, runs) in CONFIGS.items()}
    for ax, (key, title, ylab) in zip(axes, panels):
        for i, (lab, pts) in enumerate(data.items()):
            pts = [(0, 0.0, base)] + pts  # base model = 0 training FLOPs (same convention as the soft-prompt figures)
            x = [f / 1e15 for _, f, _ in pts]
            y = [100 * m[key][0] for _, _, m in pts]
            err = [[100 * m[key][1] for _, _, m in pts], [100 * m[key][2] for _, _, m in pts]]
            ax.errorbar(x, y, yerr=err, color=SERIES[i], marker=MARKERS[i], capsize=2, elinewidth=0.9,
                        mec=SURFACE, mew=1, label=lab)
        ax.set_title(title)
        ax.set_xlabel("Training compute (PFLOPs)")
        ax.set_ylabel(ylab)
    axes[0].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "lora_dev_vs_flops.png")
    plt.close(fig)
    for lab, pts in data.items():  # print the plotted numbers for the text
        print(lab)
        for e, f, m in pts:
            print(f"  {e:>5} ex {f:.3g} FLOPs  ASR {100*m['asr'][0]:5.1f}  advB {100*m['advb'][0]:5.1f}  "
                  f"MMLU {100*m['mmlu'][0]:5.1f}")


if __name__ == "__main__":
    style()
    FIG.mkdir(exist_ok=True)
    fig_loss()
    fig_metrics()
    print("saved to", FIG, file=sys.stderr)
