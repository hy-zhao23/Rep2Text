import os
import importlib.util
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_fonts_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                           "..", "..", "utils", "plot", "fonts.py"))
_spec = importlib.util.spec_from_file_location("_plot_fonts", _fonts_path)
_fonts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fonts)
_fonts.set_fonts()

# =========================================================
# Data
# =========================================================
raters = ["beauty", "hf78", "tttttt", "lexy", "wow"]

data = {
    "Structure": np.array([0.462, 0.804, 0.747, 0.730, 0.812]),
    "Entity":    np.array([0.215, 0.842, 0.574, 0.389, 0.432]),
    "Topic":     np.array([0.510, 0.850, 0.686, 0.520, 0.766]),
}

gpt = {
    "Structure": 0.524,
    "Entity":    0.571,
    "Topic":     0.717,
}

metrics = ["Structure", "Entity", "Topic"]

# =========================================================
# Style / colors
# =========================================================
BAR_COLOR = "#B7D0E6"
POINT_COLOR = "#8FB9E3"
MEAN_COLOR = "#0B5CAB"
GPT_COLOR = "#F57C00"
GRID_COLOR = "#D9D9D9"
TEXT_COLOR = "#222222"
BRACKET_COLOR = "#8C8C8C"

SAVE_PREFIX = "results/human_eval/plots/semantic_scores_refined"
ANNOTATE_RATERS = False
USE_SAMPLE_SD = True

# Match conventions from utils/plot/fonts.py:
#   axes.labelsize=11, axes.titlesize=12, xtick=10, ytick=10, legend=8
FS_TITLE = 14        # suptitle
FS_METRIC = 12       # left-side metric name (bold)
FS_INLINE_GPT = 10   # GPT value label
FS_INLINE_HUM = 10   # Human value label
FS_DELTA_LBL = 11    # right-side "GPT - Human"
FS_DELTA_VAL = 11    # right-side delta value
FS_TICK = 10
FS_AXLABEL = 11
FS_LEGEND = 10

# =========================================================
# Figure
# =========================================================
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(4, 3.5), sharex=True)

x_min, x_max = 0.2, 0.9
xticks = np.arange(0.2, 1.0, 0.1)

label_y = {
    "Structure": {"gpt": 0.50, "human": -0.40, "bracket": 0.33},
    "Entity":    {"gpt": 0.48, "human": -0.40, "bracket": 0.31},
    "Topic":     {"gpt": 0.48, "human": -0.40, "bracket": 0.31},
}

for ax, metric in zip(axes, metrics):
    vals = data[metric]
    hm = vals.mean()
    hs = vals.std(ddof=1 if USE_SAMPLE_SD else 0)
    gs = gpt[metric]
    delta = gs - hm

    low = max(0, hm - hs)
    high = min(1, hm + hs)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.85, 0.85)
    for x in xticks:
        ax.axvline(x, color=GRID_COLOR, lw=1.0, ls="--", zorder=0)

    ax.plot([low, high], [0, 0], color=BAR_COLOR, lw=17,
            solid_capstyle="round", alpha=0.95, zorder=1)

    jitter = np.linspace(-0.18, 0.18, len(vals))
    ax.scatter(vals, jitter, s=72, color=POINT_COLOR, alpha=0.95,
               edgecolor="white", linewidth=0.9, zorder=3)

    if ANNOTATE_RATERS:
        for x, y, r in zip(vals, jitter, raters):
            ax.text(x, y + 0.06, r, ha="center", va="bottom",
                    fontsize=FS_LEGEND, color=TEXT_COLOR)

    ax.scatter(hm, 0, s=290, color=MEAN_COLOR, edgecolor="white",
               linewidth=1.3, zorder=5)
    ax.plot([hm, hm], [-0.28, -0.06], color=MEAN_COLOR, lw=1.7, zorder=4)

    ax.scatter(gs, 0, s=235, marker="D", color=GPT_COLOR,
               edgecolor="white", linewidth=1.1, zorder=6)

    yb = label_y[metric]["bracket"]
    left_x = min(gs, hm)
    right_x = max(gs, hm)
    ax.plot([gs, gs], [0.09, yb], color=BRACKET_COLOR, lw=1.2, ls="--", zorder=2)
    ax.plot([hm, hm], [0.09, yb], color=BRACKET_COLOR, lw=1.2, ls="--", zorder=2)
    ax.plot([left_x, right_x], [yb, yb], color=BRACKET_COLOR, lw=1.2, ls="--", zorder=2)

    ax.text(-0.05, 0, metric, transform=ax.transAxes, ha="right", va="center",
            fontsize=FS_METRIC, fontweight="bold", color=TEXT_COLOR)

    ax.text(gs, label_y[metric]["gpt"], f"GPT {gs:.3f}", ha="center", va="bottom",
            fontsize=FS_INLINE_GPT, fontweight="bold", color=GPT_COLOR)
    ax.text(hm, label_y[metric]["human"], f"Human {hm:.3f}\u00b1{hs:.3f}",
            ha="center", va="top", fontsize=FS_INLINE_HUM, color=MEAN_COLOR)

    ax.text(1.045, 0.10, "GPT \u2212 Human", transform=ax.transAxes,
            ha="left", va="center", fontsize=FS_DELTA_LBL, fontweight="bold",
            color=TEXT_COLOR)
    ax.text(1.045, -0.17, f"{delta:+.3f}", transform=ax.transAxes,
            ha="left", va="center", fontsize=FS_DELTA_VAL,
            color=GPT_COLOR if delta > 0 else TEXT_COLOR)

    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.spines["bottom"].set_color(TEXT_COLOR)

axes[-1].set_xticks(xticks)
axes[-1].tick_params(axis="x", labelsize=FS_TICK, width=1.2, length=6, colors=TEXT_COLOR)
axes[-1].set_xlabel("Semantic score", fontsize=FS_AXLABEL, labelpad=10, color=TEXT_COLOR)

fig.suptitle("Human vs. GPT-4.1-mini Semantic Scores",
             fontsize=FS_TITLE, fontweight="bold", y=0.98, color=TEXT_COLOR)

legend_elements = [
    Line2D([0], [0], color=BAR_COLOR, lw=12, solid_capstyle="round",
           label="Human mean \u00b1 SD"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=POINT_COLOR,
           markeredgecolor="white", markersize=9, label="Individual raters"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=MEAN_COLOR,
           markeredgecolor="white", markersize=12, label="Human mean"),
    Line2D([0], [0], marker="D", color="w", markerfacecolor=GPT_COLOR,
           markeredgecolor="white", markersize=10, label="GPT-4.1-mini"),
]

fig.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, 0.91),
           ncol=4, frameon=False, fontsize=FS_LEGEND, handlelength=1.4,
           columnspacing=1.0, handletextpad=0.5)

plt.subplots_adjust(left=0.10, right=0.90, top=0.80, hspace=0.28, bottom=0.14)

plt.savefig(f"{SAVE_PREFIX}.pdf", bbox_inches="tight")
print(f"Saved {SAVE_PREFIX}.pdf")
