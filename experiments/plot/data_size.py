import os
import matplotlib.pyplot as plt
import numpy as np
from utils import p_args
from utils.plot.fonts import set_fonts

set_fonts()

fig_dir = p_args.path.fig_dir
os.makedirs(fig_dir, exist_ok=True)

# =========================
# Data (5-size sweep on Llama-3.2-3B with Mistral-7B layer-10, 16 tokens)
# =========================
train_sizes = np.array([10, 50, 100, 320, 640], dtype=float)
train_labels = ["10K", "50K", "100K", "320K", "640K"]

metrics = {
    "loss":      [2.485, 1.785, 1.641, 1.329, 1.235],
    "ppl":       [12.00, 5.96, 5.16, 3.78, 3.44],
    "token_f1":  [0.262, 0.389, 0.415, 0.477, 0.505],
    "bleu":      [0.098, 0.179, 0.202, 0.259, 0.287],
    "rouge_1":   [0.258, 0.394, 0.421, 0.489, 0.517],
    "rouge_2":   [0.091, 0.196, 0.220, 0.287, 0.314],
    "rouge_l":   [0.239, 0.377, 0.406, 0.475, 0.503],
    "bertscore": [0.636, 0.727, 0.745, 0.782, 0.798],
}

colors = {
    "loss":      "#5B8DB8",
    "ppl":       "#7AA6C7",
    "token_f1":  "#4FAF7A",
    "bleu":      "#8A78C9",
    "rouge_1":   "#74CFCB",
    "rouge_2":   "#99DDD8",
    "rouge_l":   "#6EC6C4",
    "bertscore": "#B287B7",
}

titles = {
    "loss":      "Loss",
    "ppl":       "Perplexity",
    "token_f1":  "Token F1",
    "bleu":      "BLEU",
    "rouge_1":   "ROUGE-1",
    "rouge_2":   "ROUGE-2",
    "rouge_l":   "ROUGE-L",
    "bertscore": "BERTScore",
}


def save_metric(name, yvals, color, title):
    y = np.array(yvals, dtype=float)

    fig, ax = plt.subplots(figsize=(4, 3))

    ax.plot(
        train_sizes, y,
        marker='o', linewidth=1.9, markersize=4.4,
        color=color,
    )
    ax.fill_between(
        train_sizes, y, np.min(y),
        color=color, alpha=0.12,
    )

    ax.set_xscale("log")
    ax.set_xticks(train_sizes)
    ax.set_xticklabels(train_labels, fontsize=8)
    ax.tick_params(axis='y', labelsize=8)

    ax.set_title(title, fontsize=10.5, pad=4)
    ax.set_xlabel("Train Size", fontsize=8.5)

    if name == "loss":
        ax.set_ylabel("Loss", fontsize=8.5)
        fmt = "{:.3f}"
    elif name == "ppl":
        ax.set_ylabel("PPL", fontsize=8.5)
        fmt = "{:.2f}"
    else:
        ax.set_ylabel("Score", fontsize=8.5)
        fmt = "{:.3f}"

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.20)
    ax.grid(False, axis="x")

    # Annotate 10K, 320K, 640K
    for xk, yk in [(10, y[0]), (320, y[3]), (640, y[4])]:
        ax.annotate(
            fmt.format(yk), (xk, yk),
            xytext=(0, 6), textcoords="offset points",
            ha="center", fontsize=7.5,
        )

    fig.tight_layout()
    path = os.path.join(fig_dir, f"data_size_{name}.pdf")
    fig.savefig(path)
    plt.close(fig)
    return path


order = ["loss", "ppl", "token_f1", "bleu", "rouge_1", "rouge_2", "rouge_l", "bertscore"]
saved = [save_metric(k, metrics[k], colors[k], titles[k]) for k in order]

print(f"Saved {len(saved)} PDFs to {fig_dir}/:")
for p in saved:
    print(" ", os.path.basename(p))
