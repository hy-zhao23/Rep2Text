import numpy as np
import matplotlib.pyplot as plt
from utils.core.files import read_json
from utils import p_args
import os
from utils.plot.fonts import set_fonts

set_fonts()

lexical_metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
semantic_metrics = ['Structure', 'Entity', 'Topic']
bert_metric = 'BERTScore F1'
bert_label = 'BERT-F1'

expansions = [0.5, 1, 2, 4, 8]
expansion_paths = [
    'l10t16p16/exp05/val_einv_metrics_summary.json',
    'l10t16p16/exp1/val_einv_metrics_summary.json',
    'l10t16p16/exp2/val_einv_metrics_summary.json',
    'l10t16p16/exp4/val_einv_metrics_summary.json',
    'l10t16p16/exp8/val_einv_metrics_summary.json',
]

n_vecs = [1, 2, 4, 8, 16, 32]
nvec_paths = [
    'l10t16p1/p1/val_einv_metrics_summary.json',
    'l10t16p2/p2/val_einv_metrics_summary.json',
    'l10t16p4/p4/val_einv_metrics_summary.json',
    'l10t16p8/p8/val_einv_metrics_summary.json',
    'l10t16p16/exp05/val_einv_metrics_summary.json',
    'l10t16p32/p32/val_einv_metrics_summary.json',
]


def aggregate(paths):
    lex = {m: [] for m in lexical_metrics}
    sem = {m: [] for m in semantic_metrics}
    bert = []
    for p in paths:
        data = read_json(os.path.join(os.path.dirname(p_args.path.result_dir), p))
        for m in lexical_metrics:
            lex[m].append(np.mean(data[m]) if m in data else np.nan)
        for m in semantic_metrics:
            sem[m].append(np.mean(data[m]) / 5.0 if m in data else np.nan)
        bert.append(np.mean(data[bert_metric]) if bert_metric in data else np.nan)
    return lex, sem, bert


exp_lex, exp_sem, exp_bert = aggregate(expansion_paths)
nvec_lex, nvec_sem, nvec_bert = aggregate(nvec_paths)

colors_lexical = ['#fdae61', '#f46d43', '#d73027']
markers_lexical = ['v', 'p', 'h']
colors_semantic = ['#74add1', '#4575b4', '#313695']
markers_semantic = ['o', 's', '^']
bert_color = '#5e3c99'

SUBPLOT_FIGSIZE = (3.5, 2.7)


def plot_lexical(ax, x_positions, lex, xticklabels, xlabel, title):
    for i, m in enumerate(lexical_metrics):
        ax.plot(x_positions, lex[m],
                color=colors_lexical[i], marker=markers_lexical[i],
                linestyle='-', linewidth=2, markersize=7,
                label=m, markeredgewidth=0, alpha=0.9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Score (0-1)')
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(xticklabels)
    ax.grid(axis='y', alpha=0.3, linestyle='--', zorder=1)
    ax.grid(axis='x', alpha=0.2, linestyle='--', zorder=1)
    ax.set_title(title)
    ax.legend(loc='best', frameon=True, framealpha=0.9, edgecolor='gray')


def plot_semantic(ax, x_positions, sem, bert, xticklabels, xlabel, title):
    for i, m in enumerate(semantic_metrics):
        ax.plot(x_positions, sem[m],
                color=colors_semantic[i], marker=markers_semantic[i],
                linestyle='-', linewidth=2, markersize=7,
                label=m, markeredgewidth=0, alpha=0.9)
    ax.plot(x_positions, bert,
            color=bert_color, marker='D', linestyle='--',
            linewidth=2, markersize=7, label=bert_label,
            markeredgewidth=0, alpha=0.9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Score (0-1)')
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(xticklabels)
    ax.grid(axis='y', alpha=0.3, linestyle='--', zorder=1)
    ax.grid(axis='x', alpha=0.2, linestyle='--', zorder=1)
    ax.set_title(title)
    ax.legend(loc='best', frameon=True, framealpha=0.9, edgecolor='gray')


exp_x = np.arange(len(expansions))
exp_labels = [f'{e}x' for e in expansions]
nvec_x = np.arange(len(n_vecs))
nvec_labels = [f'{k}' for k in n_vecs]

fig_dir = os.path.dirname(p_args.path.fig_dir)
os.makedirs(fig_dir, exist_ok=True)

panels = [
    ('expansion_lexical.pdf', plot_lexical,
     (exp_x, exp_lex, exp_labels,
      r'Adapter Hidden Expansion Factor $f$', 'Lexical Overlap')),
    ('expansion_semantic.pdf', plot_semantic,
     (exp_x, exp_sem, exp_bert, exp_labels,
      r'Adapter Hidden Expansion Factor $f$', 'Sem./Struct. Preservation')),
    ('nvec_lexical.pdf', plot_lexical,
     (nvec_x, nvec_lex, nvec_labels,
      r'Number of Projection Vectors $k$', 'Lexical Overlap')),
    ('nvec_semantic.pdf', plot_semantic,
     (nvec_x, nvec_sem, nvec_bert, nvec_labels,
      r'Number of Projection Vectors $k$', 'Sem./Struct. Preservation')),
]

for filename, plot_fn, args in panels:
    fig, ax = plt.subplots(figsize=SUBPLOT_FIGSIZE)
    plot_fn(ax, *args)
    plt.tight_layout()
    fig_path = os.path.join(fig_dir, filename)
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved {fig_path}")
