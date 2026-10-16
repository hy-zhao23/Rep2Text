"""Plot loss & perplexity curves for the 6 training runs.

vec2text (HF Trainer) — parse trainer_state.json
  - mistral7B vec base
  - mistral7B corrector
  - llama8B  vec base
  - llama8B  corrector

transfer (custom training, [Train] stdout lines) — parse .log
  - mistral7B -> Qwen2.5-14B (32-token)
  - llama8B   -> Qwen2.5-14B (32-token)
"""
import json
import os
import re
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.ticker import MaxNLocator

# --- Calibri (same setup as utils/plot/fonts.py, inlined to avoid utils.__init__ side-effects) ---
_calibri_dir = os.path.join(os.path.expanduser("~"), ".fonts/Calibri")
if os.path.isdir(_calibri_dir):
    for _fname in os.listdir(_calibri_dir):
        if _fname.lower().endswith((".ttf", ".otf")):
            font_manager.fontManager.addfont(os.path.join(_calibri_dir, _fname))

plt.rcParams.update({
    "font.family":      "Calibri",
    "font.sans-serif":  ["Calibri"],
    "font.weight":      "normal",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.labelsize":   11,
    "axes.titlesize":   12,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  8,
})

PROJ = os.environ.get("PROJ", ".")
OUT_DIR = os.path.join(PROJ, "results", "figs", "training_dynamics")
os.makedirs(OUT_DIR, exist_ok=True)

# --- color palette: soft pastel style ---
COLORS = {
    # Ours: periwinkle + sage green
    "ours_mistral": "#7B8EC6",
    "ours_llama":   "#8FAF9D",

    # mistral7B vec2text: lavender blue + dusty blue
    "mistral_base": "#9B8CE3",
    "mistral_corr": "#8FB6D8",

    # llama8B vec2text: soft purple + muted teal
    "llama_base":   "#B783D8",
    "llama_corr":   "#7FB8B2",
}

# --- config: paths & display order ---
RUNS = [
    {
        "label": "Mistral7B Vec. Base",
        "kind":  "trainer_state",
        "path":  f"{PROJ}/experiments/vec2text/test/results/mistral7b_l10t32_20260303_154531/trainer_state.json",
        "color": COLORS["mistral_base"],
        "ls":    "-",
    },
    {
        "label": "Mistral7B Corrector",
        "kind":  "trainer_state",
        "path":  f"{PROJ}/experiments/vec2text/test/results/corrector_mistral7b_l10t32_20260506_011259/trainer_state.json",
        "color": COLORS["mistral_corr"],
        "ls":    "-",
    },
    {
        "label": "Llama8B Vec. Base",
        "kind":  "trainer_state",
        "path":  f"{PROJ}/experiments/vec2text/test/results/llama8b_l10t32_20260505_011646/trainer_state.json",
        "color": COLORS["llama_base"],
        "ls":    "-",
    },
    {
        "label": "Llama8B Corrector",
        "kind":  "trainer_state",
        "path":  f"{PROJ}/experiments/vec2text/test/results/corrector_llama8b_l10t32_20260505_215740/trainer_state.json",
        "color": COLORS["llama_corr"],
        "ls":    "-",
    },
    {
        "label": "Ours (Mistral7B)",
        "kind":  "train_log",
        "path":  f"{PROJ}/logs/trans-q14b-32tok_849434.log",
        "color": COLORS["ours_mistral"],
        "ls":    "-",
    },
    {
        "label": "Ours (Llama8B)",
        "kind":  "train_log",
        "path":  f"{PROJ}/logs/trans-l8b-q14b-32tok_1019047.log",
        "color": COLORS["ours_llama"],
        "ls":    "-",
    },
]


# ---------- parsers ----------
def parse_trainer_state(path):
    """Return (epoch, loss, ppl). ppl = train_perplexity if present, else exp(loss)."""
    with open(path) as f:
        data = json.load(f)
    epoch, loss, ppl = [], [], []
    for e in data["log_history"]:
        if "loss" not in e or "epoch" not in e:
            continue
        epoch.append(e["epoch"])
        loss.append(e["loss"])
        if "train_perplexity" in e:
            ppl.append(e["train_perplexity"])
        else:
            ppl.append(math.exp(min(e["loss"], 50)))
    return np.array(epoch), np.array(loss), np.array(ppl)


_TRAIN_LINE = re.compile(
    r"\[Train\]\s+Ep\s+(\d+)\s+Step\s+(\d+):\s+loss=([0-9.]+),\s+recent=([0-9.eE+\-]+),\s+ppl=([0-9.eE+\-]+)"
)


def parse_train_log(path):
    """Return (progress, loss, ppl). progress is monotonically increasing global step index."""
    epochs, steps, losses, ppls = [], [], [], []
    with open(path, "r", errors="replace") as f:
        for line in f:
            m = _TRAIN_LINE.search(line)
            if not m:
                continue
            ep, st, ls, _rec, pp = m.groups()
            epochs.append(int(ep))
            steps.append(int(st))
            losses.append(float(ls))
            ppls.append(float(pp))
    epochs = np.asarray(epochs)
    steps = np.asarray(steps)
    losses = np.asarray(losses)
    ppls = np.asarray(ppls)
    # Build a continuous epoch axis: epoch + step / steps_per_epoch
    if len(steps) == 0:
        return np.array([]), np.array([]), np.array([])
    spe = {}
    for e in np.unique(epochs):
        spe[e] = steps[epochs == e].max()
    cont = np.array(
        [e + (s / spe[e]) for e, s in zip(epochs, steps)],
        dtype=float,
    )
    order = np.argsort(cont)
    return cont[order], losses[order], ppls[order]


# ---------- load all runs ----------
def rolling_mean_std(y, w):
    """Centered rolling mean and std over a window of size w (clipped at edges)."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n == 0:
        return y, np.zeros_like(y)
    w = max(3, min(w, n))
    half = w // 2
    mean = np.empty(n)
    std = np.empty(n)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        seg = y[lo:hi]
        mean[i] = seg.mean()
        std[i] = seg.std()
    return mean, std


curves = []
for r in RUNS:
    if r["kind"] == "trainer_state":
        x, l, p = parse_trainer_state(r["path"])
    else:
        x, l, p = parse_train_log(r["path"])
    print(f"{r['label']:<45s} n={len(x):>6d}  ep=[{x.min():.2f},{x.max():.2f}]  "
          f"loss=[{l.min():.3f},{l.max():.3f}]  ppl=[{p.min():.2f},{p.max():.2f}]")
    curves.append((r, x, l, p))


# ---------- plot helpers ----------
GROUPS = [
    # (display_title, member_labels, filename_slug)
    ("Ours (→ Qwen2.5-14B)",
     ["Ours (Mistral7B)", "Ours (Llama8B)"],
     "Ours_to_Qwen2.5-14B"),
    ("Mistral7B Vec2text",
     ["Mistral7B Vec. Base", "Mistral7B Corrector"],
     "mistral7B_vec2text"),
    ("Llama8B Vec2text",
     ["Llama8B Vec. Base",  "Llama8B Corrector"],
     "llama8B_vec2text"),
]

curves_by_label = {r["label"]: (r, x, l, p) for (r, x, l, p) in curves}


def plot_group(title, labels, fslug, metric, ylabel, log_y, prefix):
    fig, ax = plt.subplots(figsize=(3, 2.5))
    for lbl in labels:
        r, x, l, p = curves_by_label[lbl]
        y = l if metric == "loss" else p
        w = max(11, len(y) // 100) | 1  # adaptive odd window
        mean, std = rolling_mean_std(y, w=w)
        if log_y:
            lo = np.clip(mean - std, 1e-6, None)
            hi = mean + std
        else:
            lo, hi = mean - std, mean + std
        ax.fill_between(x, lo, hi, color=r["color"], alpha=0.18, linewidth=0)
        ax.plot(x, mean, color=r["color"], linestyle=r["ls"],
                linewidth=1.6, label=r["label"], alpha=0.95)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", frameon=True, fontsize=8)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f"{prefix}_{fslug}.pdf")
    fig.savefig(out)
    print(f"saved -> {out}")
    plt.close(fig)


for title, labels, fslug in GROUPS:
    plot_group(title, labels, fslug, "loss", "Training loss",       log_y=False, prefix="loss")
    plot_group(title, labels, fslug, "ppl",  "Training perplexity", log_y=True,  prefix="ppl")
