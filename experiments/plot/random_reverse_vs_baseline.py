"""Per-metric raincloud comparison: Baseline-320k vs Reverse-token vs Random-pairing.

One (4, 3) PDF per metric, mirroring the data_size.py convention.
  - per-sample metrics (token_f1, bleu, rouge_*, bertscore): violin + box + jittered points
  - loss / perplexity: clean summary bars (no per-sample values available)

Filenames:  random_reverse_vs_baseline_<metric>.pdf

Sources:
  Baseline:  results/.../transfer-m3b-size320000/val_einv_metrics.json    (full 80K val)
  Random:    results/.../transfer-m3b-size320000-random-2b200/test1000_einv_metrics.json
  Reverse:   results/.../transfer-m3b-size320000-reverse/val1000_einv_metrics.json
"""
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from utils import p_args
from utils.plot.fonts import set_fonts

set_fonts()

PROJ = Path(os.environ.get("PROJ", "."))
RESULTS = PROJ / "results" / "mistral7B" / "l10t16p16"

COLORS = {
    "Baseline 320k": "#7B8EC6",   # dusty periwinkle
    "Reverse":       "#B783D8",   # soft mauve purple
    "Random":        "#7FB8B2",   # muted teal
}
LIGHT_COLORS = {
    "Baseline 320k": "#DCE3F7",
    "Reverse":       "#EEE0F7",
    "Random":        "#DCEFEB",
}
EDGE = "#4A4A4A"
GRID = "#D9D9D9"
TEXT = "#333333"

plt.rcParams.update({
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.linewidth": 0.85,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def metric_samples_from_json(path):
    with open(path) as f:
        d = json.load(f)
    sc = d["scores"]

    def arr(vals):
        return np.asarray(vals, dtype=float)

    samples = {
        "loss":      None,
        "ppl":       None,
        "bertscore": arr([s["bertscore_f1"] for s in sc]),
        "token_f1":  arr([s["token_f1"] for s in sc]),
        "bleu":      arr([s["bleu"] for s in sc]),
        "rouge_1":   arr([s["rouge1"][0] for s in sc]),
        "rouge_2":   arr([s["rouge2"][0] for s in sc]),
        "rouge_l":   arr([s["rougeL"][0] for s in sc]),
    }
    summary = {
        "loss": (float(d["loss"]), None, None, None, None),
        "ppl":  (float(d["perplexity"]), None, None, None, None),
    }
    for k, a in samples.items():
        if a is None:
            continue
        std = float(a.std(ddof=1))
        var = float(a.var(ddof=1))
        sem = std / np.sqrt(len(a))
        summary[k] = (float(a.mean()), std, var, sem, len(a))
    return summary, samples


baseline, baseline_samples = metric_samples_from_json(
    RESULTS / "transfer-m3b-size320000" / "val_einv_metrics.json"
)
random_, random_samples = metric_samples_from_json(
    RESULTS / "transfer-m3b-size320000-random-2b200" / "test1000_einv_metrics.json"
)
reverse, reverse_samples = metric_samples_from_json(
    RESULTS / "transfer-m3b-size320000-reverse" / "val1000_einv_metrics.json"
)

conditions = [
    ("Baseline 320k", baseline, baseline_samples),
    ("Reverse",       reverse,  reverse_samples),
    ("Random",        random_,  random_samples),
]

metric_order = ["loss", "ppl", "token_f1", "bleu",
                "rouge_1", "rouge_2", "rouge_l", "bertscore"]
titles = {
    "loss": "Loss", "ppl": "Perplexity",
    "token_f1": "Token F1", "bleu": "BLEU",
    "rouge_1": "ROUGE-1", "rouge_2": "ROUGE-2", "rouge_l": "ROUGE-L",
    "bertscore": "BERTScore",
}
lower_is_better = {"loss", "ppl"}


def setup_ax(ax):
    """Inherit project font sizes from set_fonts() (title=12, label=11, ticks=10)."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(EDGE)
    ax.spines["bottom"].set_color(EDGE)
    ax.tick_params(axis="x", colors=TEXT, length=0)
    ax.tick_params(axis="y", colors=TEXT, width=0.8)
    ax.grid(True, axis="y", alpha=0.20)
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)


def downsample_for_scatter(a, max_points=200, rng=None):
    if rng is None:
        rng = np.random.default_rng(7)
    if len(a) <= max_points:
        return a
    idx = rng.choice(len(a), size=max_points, replace=False)
    return a[idx]


def save_summary_metric(mkey, title):
    """Loss/PPL: simple summary bar with value annotation."""
    fig, ax = plt.subplots(figsize=(4, 3))
    setup_ax(ax)

    labels = [c[0] for c in conditions]
    x = np.arange(len(labels))
    vals = [c[1][mkey][0] for c in conditions]

    fmt = "{:.3f}" if mkey == "loss" else "{:.2f}"
    for i, (label, val) in enumerate(zip(labels, vals)):
        ax.bar(i, val, width=0.56,
               color=LIGHT_COLORS[label],
               edgecolor=COLORS[label],
               linewidth=1.3, alpha=0.95)
        ax.scatter(i, val, s=24, color=COLORS[label],
                   edgecolor="white", linewidth=0.8, zorder=3)
        ax.annotate(fmt.format(val), (i, val),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=9)

    ax.set_ylim(0, max(vals) * 1.20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    suffix = " (↓)"
    ax.set_title(title + suffix, pad=6)
    ax.set_ylabel("Loss" if mkey == "loss" else "PPL")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    fig.tight_layout()
    path = fig_dir / f"random_reverse_vs_baseline_{mkey}.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def save_raincloud_metric(mkey, title, rng):
    """Per-sample metric: violin + box + jittered scatter."""
    fig, ax = plt.subplots(figsize=(4, 3))
    setup_ax(ax)

    labels = [c[0] for c in conditions]
    x_positions = np.arange(len(labels))
    all_values = [c[2][mkey] for c in conditions]

    # Violin
    vp = ax.violinplot(all_values, positions=x_positions, widths=0.72,
                      showmeans=False, showmedians=False, showextrema=False)
    for body, label in zip(vp["bodies"], labels):
        body.set_facecolor(LIGHT_COLORS[label])
        body.set_edgecolor(COLORS[label])
        body.set_alpha(0.82)
        body.set_linewidth(1.0)

    for i, (label, arr) in enumerate(zip(labels, all_values)):
        # jittered scatter
        vals = downsample_for_scatter(arr, rng=rng)
        jitter = rng.normal(0, 0.05, size=len(vals))
        ax.scatter(np.full_like(vals, i, dtype=float) + jitter, vals,
                   s=5, color=COLORS[label], alpha=0.30,
                   linewidths=0, zorder=2)

        # whisker bounds
        q1, med, q3 = np.percentile(arr, [25, 50, 75])
        iqr = q3 - q1
        low = float(np.min(arr[arr >= q1 - 1.5 * iqr]))
        high = float(np.max(arr[arr <= q3 + 1.5 * iqr]))

        # box
        box_w = 0.20
        ax.add_patch(plt.Rectangle((i - box_w / 2, q1), box_w, q3 - q1,
                                   facecolor="white", edgecolor=EDGE,
                                   linewidth=0.85, zorder=4))
        ax.plot([i - box_w / 2, i + box_w / 2], [med, med],
                color=EDGE, linewidth=1.0, zorder=5)
        ax.plot([i, i], [low, q1],   color=EDGE, linewidth=0.7, zorder=4)
        ax.plot([i, i], [q3, high],  color=EDGE, linewidth=0.7, zorder=4)
        ax.plot([i - box_w / 3, i + box_w / 3], [low, low],
                color=EDGE, linewidth=0.7, zorder=4)
        ax.plot([i - box_w / 3, i + box_w / 3], [high, high],
                color=EDGE, linewidth=0.7, zorder=4)

        # mean annotation above whisker
        mu, std, var, sem, n = conditions[i][1][mkey]
        ax.annotate(f"μ={mu:.3f}", (i, high),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=9)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.set_title(title + " (↑)", pad=6)
    ax.set_ylabel("Score")

    ymin = min(np.min(a) for a in all_values)
    ymax = max(np.max(a) for a in all_values)
    pad = 0.10 * (ymax - ymin + 1e-9)
    ax.set_ylim(ymin - pad, ymax + 1.65 * pad)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    fig.tight_layout()
    path = fig_dir / f"random_reverse_vs_baseline_{mkey}.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


fig_dir = PROJ / "results" / "figs" / "mistral7B" / "l10t16p16"
fig_dir.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(7)
saved = []
for mkey in metric_order:
    title = titles[mkey]
    if mkey in lower_is_better:
        saved.append(save_summary_metric(mkey, title))
    else:
        saved.append(save_raincloud_metric(mkey, title, rng))

print(f"Saved {len(saved)} PDFs to {fig_dir}/:")
for p in saved:
    print(" ", p.name)

# table
print("\n" + " " * 12 + "  ".join(f"{c[0]:>42s}" for c in conditions))
for m in metric_order:
    cells = []
    for c in conditions:
        mu, std, var, sem, n = c[1][m]
        if std is None:
            cells.append(f"{mu:>42.4f}")
        else:
            cells.append(f"{mu:>6.4f} ± {std:.4f} | Var={var:.4g} | SEM={sem:.4f} | n={n}")
    print(f"{titles[m]:>10s}  " + "  ".join(cells))
