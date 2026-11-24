"""Training-loss curves + test metrics vs. training FLOPs, and a results table.

    pip install matplotlib
    python plot_results.py --model $WORK/cs182_assets/models/Qwen3.5-0.8B \
        --run_dir $WORK/runs/A_neutral --eval_dir $WORK/outputs/A_neutral --fig_dir $WORK/figures

Training FLOPs for a soft-prompt checkpoint (standard dense-transformer estimate):
    FLOPs ~= 4 * N_eff * T
    forward = 2*N_eff per token; backward only needs activation gradients (weights are
    frozen, no weight-gradient matmuls) = another 2*N_eff per token. Full fine-tuning would be 6.
    N_eff = parameters that do matmuls per token = all params - input embedding table
            (a lookup) + LM head (counted even when tied to the embedding).
    T     = tokens processed = tokens_seen (real prompt+response tokens, no padding)
            + n_soft_tokens * examples_seen.
    Ignores the sequence-length-dependent attention term and the untimed warm-up batches.
Error bars are +-1 standard error (binomial for rates, std/sqrt(n) for MT-Bench).
"""
import argparse
import csv
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

# reference categorical palette (fixed order) + neutral inks, light surface
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
MODELS = ["base", "1000ex", "2000ex", "4000ex"]
SHORT = {"base": "base", "1000ex": "1k", "2000ex": "2k", "4000ex": "4k"}


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


# ---------------------------------------------------------------- data
def effective_params(model_path):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16)  # CPU is fine
    emb, head = m.get_input_embeddings().weight, m.get_output_embeddings().weight
    total = sum(p.numel() for p in m.parameters())  # tied weights are counted once here
    tied = head.data_ptr() == emb.data_ptr()
    n_eff = total - emb.numel() + (head.numel() if tied else 0)
    print(f"params: total={total/1e6:.1f}M  embedding={emb.numel()/1e6:.1f}M  tied_head={tied}  N_eff={n_eff/1e6:.1f}M")
    return n_eff


def checkpoint_flops(run_dir, n_eff, mult):
    out = {"base": dict(flops=0.0, examples=0, tokens=0, train_time_sec=0.0)}
    for m in MODELS[1:]:
        meta = torch.load(os.path.join(run_dir, f"soft_prompt_{m}.pt"), map_location="cpu")["meta"]
        T = meta["tokens_seen"] + meta["n_tokens"] * meta["examples_seen"]
        out[m] = dict(flops=mult * n_eff * T, examples=meta["examples_seen"], tokens=T,
                      train_time_sec=meta["train_time_sec"])
    return out


def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def rate(rows, keep, field):
    xs = [bool(r[field]) for r in rows if r.get("judge_refusal") is not None and keep(r)]
    p = sum(xs) / len(xs)
    return p, math.sqrt(p * (1 - p) / len(xs)), len(xs)


def is_harmful(r):
    return "harmful" in str(r.get("data_type", "")) or r.get("label") == 1


def metrics(eval_dir, m):
    j = lambda s: load_jsonl(os.path.join(eval_dir, f"{m}_test_{s}.judged.jsonl"))
    hb, jb, xs = j("harmbench"), j("jailbreak"), j("xstest")
    res = {
        "HarmBench ASR": rate(hb, lambda r: True, "judge_harmful"),
        "WildJailbreak ASR": rate(jb, is_harmful, "judge_harmful"),
        "XSTest-unsafe ASR": rate(xs, is_harmful, "judge_harmful"),
        "WildJailbreak benign over-refusal": rate(jb, lambda r: not is_harmful(r), "judge_refusal"),
        "XSTest-safe over-refusal": rate(xs, lambda r: not is_harmful(r), "judge_refusal"),
    }
    mm = json.load(open(os.path.join(eval_dir, f"mmlu_{m}_n0.json")))
    a = mm["accuracy"]
    res["MMLU accuracy"] = (a, math.sqrt(a * (1 - a) / mm["n"]), mm["n"])
    sc = [s for r in load_jsonl(os.path.join(eval_dir, f"mtb_{m}.judged.jsonl")) for s in r["scores"] if s is not None]
    mu = sum(sc) / len(sc)
    sd = math.sqrt(sum((s - mu) ** 2 for s in sc) / (len(sc) - 1))
    res["MT-Bench score"] = (mu, sd / math.sqrt(len(sc)), len(sc))
    return res


# ---------------------------------------------------------------- plots
def budget_labels(ax, xs, names):
    """Muted budget tags along the top edge, so each FLOPs point is identifiable."""
    for x, n in zip(xs, names):
        ax.annotate(SHORT[n], (x, 1), xycoords=("data", "axes fraction"), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8, color=MUTED)


def lines_vs_flops(ax, X, R, keys, labels, ylabel, pct=True):
    for i, (k, lab) in enumerate(zip(keys, labels)):
        y = [R[m][k][0] * (100 if pct else 1) for m in MODELS]
        e = [R[m][k][1] * (100 if pct else 1) for m in MODELS]
        ax.errorbar(X, y, yerr=e, color=SERIES[i], marker=MARKERS[i], label=lab,
                    elinewidth=1, capsize=3, markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.set_xlabel("Training compute (PFLOPs)")
    ax.set_ylabel(ylabel)
    budget_labels(ax, X, MODELS)


def plot_all(args, F, R):
    X = [F[m]["flops"] / 1e15 for m in MODELS]
    os.makedirs(args.fig_dir, exist_ok=True)
    save = lambda fig, name: (fig.savefig(os.path.join(args.fig_dir, name)), print("saved", name), plt.close(fig))

    # 1) safety
    fig, ax = plt.subplots(figsize=(6, 4))
    lines_vs_flops(ax, X, R, ["HarmBench ASR", "WildJailbreak ASR", "XSTest-unsafe ASR"],
                   ["HarmBench", "WildJailbreak (unseen jailbreaks)", "XSTest unsafe"], "Attack success rate (%)  ↓")
    ax.set_ylim(bottom=0)
    ax.set_title("Safety vs. training compute", pad=18)
    ax.legend(loc="upper right")
    save(fig, "test_safety_vs_flops.png")

    # 2) over-refusal
    fig, ax = plt.subplots(figsize=(6, 4))
    lines_vs_flops(ax, X, R, ["WildJailbreak benign over-refusal", "XSTest-safe over-refusal"],
                   ["WildJailbreak adversarial-benign", "XSTest safe"], "Over-refusal rate (%)  ↓")
    ax.set_ylim(0, 100)
    ax.set_title("Over-refusal vs. training compute", pad=18)
    ax.legend(loc="upper left")
    save(fig, "test_overrefusal_vs_flops.png")

    # 3) capability: two panels, never a shared dual axis
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
    lines_vs_flops(a1, X, R, ["MMLU accuracy"], ["MMLU"], "Accuracy (%)  ↑")
    a1.set_title("MMLU (0-shot)", pad=18)
    lines_vs_flops(a2, X, R, ["MT-Bench score"], ["MT-Bench"], "Score (1-10)  ↑", pct=False)
    a2.set_title("MT-Bench", pad=18)
    fig.tight_layout()
    save(fig, "test_capability_vs_flops.png")

    # 4) trade-off: lower-left is better
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    pairs = [("WildJailbreak benign over-refusal", "WildJailbreak ASR", "WildJailbreak eval"),
             ("XSTest-safe over-refusal", "XSTest-unsafe ASR", "XSTest")]
    for i, (kx, ky, lab) in enumerate(pairs):
        x = [R[m][kx][0] * 100 for m in MODELS]
        y = [R[m][ky][0] * 100 for m in MODELS]
        ax.plot(x, y, color=SERIES[i], marker=MARKERS[i], label=lab, markeredgecolor=SURFACE, markeredgewidth=1.5)
        for xi, yi, m in zip(x, y, MODELS):
            ax.annotate(SHORT[m], (xi, yi), xytext=(5, 5), textcoords="offset points", fontsize=8, color=INK2)
    ax.set_xlabel("Over-refusal on benign prompts (%)  ↓")
    ax.set_ylabel("Attack success rate (%)  ↓")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_title("Safety / over-refusal trade-off (lower-left is better)")
    ax.legend(loc="upper right")
    save(fig, "test_tradeoff.png")


def plot_loss(args):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for i, spec in enumerate(args.loss_runs):
        d, lab = spec.split(":", 1)
        p = os.path.join(os.path.dirname(args.run_dir.rstrip("/")), d, "train_log.jsonl")
        if not os.path.exists(p):
            print("skip (missing):", p)
            continue
        log = load_jsonl(p)
        x = [r["examples_seen"] for r in log]
        y = [r["loss"] for r in log]
        w = args.smooth
        ys = [sum(y[max(0, k - w + 1): k + 1]) / len(y[max(0, k - w + 1): k + 1]) for k in range(len(y))]
        if d == os.path.basename(args.run_dir.rstrip("/")):  # raw noise only for the final run
            ax.plot(x, y, color=SERIES[i], linewidth=1, alpha=0.25)
        ax.plot(x, ys, color=SERIES[i], label=lab)
    ax.set_xlabel("Training examples seen")
    ax.set_ylabel("Training loss (response tokens)")
    ax.set_title(f"Soft prompt training loss (moving avg of {args.smooth} logs; faint = raw, final run)")
    ax.legend(loc="upper right")
    os.makedirs(args.fig_dir, exist_ok=True)
    fig.savefig(os.path.join(args.fig_dir, "train_loss.png"))
    plt.close(fig)
    print("saved train_loss.png")


def write_table(args, F, R):
    keys = list(R["base"].keys())
    path = os.path.join(args.fig_dir, "results_test.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "examples", "tokens", "train_PFLOPs", "train_time_sec"]
                   + [c for k in keys for c in (k, k + " SE", k + " n")])
        for m in MODELS:
            w.writerow([m, F[m]["examples"], F[m]["tokens"], f"{F[m]['flops']/1e15:.3f}", f"{F[m]['train_time_sec']:.1f}"]
                       + [c for k in keys for c in (f"{R[m][k][0]:.4f}", f"{R[m][k][1]:.4f}", R[m][k][2])])
    print("saved", path)
    # markdown version for the report / group chat
    fmt = lambda k, v: f"{v[0]:.2f}±{v[1]:.2f}" if "MT-Bench" in k else f"{100*v[0]:.1f}±{100*v[1]:.1f}"
    print("\n| model | PFLOPs | " + " | ".join(keys) + " |")
    print("|" + "---|" * (len(keys) + 2))
    for m in MODELS:
        print(f"| {m} | {F[m]['flops']/1e15:.2f} | " + " | ".join(fmt(k, R[m][k]) for k in keys) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run_dir", required=True, help="the final run, e.g. $WORK/runs/A_neutral")
    ap.add_argument("--eval_dir", required=True, help="its test outputs, e.g. $WORK/outputs/A_neutral")
    ap.add_argument("--fig_dir", required=True)
    ap.add_argument("--flops_mult", type=float, default=4.0, help="FLOPs per param per token (4 = frozen weights)")
    ap.add_argument("--smooth", type=int, default=5)
    ap.add_argument("--loss_runs", nargs="+", default=[
        "sp_n20_lr5e-3:Original (refusal init, lr 5e-3)", "A_neutral:A: neutral init (final)",
        "B_lr1e-3:B: lr 1e-3", "C_n50:C: 50 tokens"],
        help="run_dir_name:legend label, looked up next to --run_dir")
    args = ap.parse_args()

    style()
    plot_loss(args)
    n_eff = effective_params(args.model)
    F = checkpoint_flops(args.run_dir, n_eff, args.flops_mult)
    R = {m: metrics(args.eval_dir, m) for m in MODELS}
    plot_all(args, F, R)
    write_table(args, F, R)


if __name__ == "__main__":
    main()
