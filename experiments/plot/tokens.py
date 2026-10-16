import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.core.files import read_json
from utils import p_args
import os
from utils.plot.fonts import set_fonts

# ============================================================
# CONFIGURATION FOR TOKEN LENGTH EVALUATION
# ============================================================
set_fonts()
# Metrics to evaluate
lexical_metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']           # Lexical Overlap
semantic_metrics = ['Structure', 'Entity', 'Topic']            # Semantic/Structural Preservation (LLM-as-a-judge)
bert_metric = 'BERTScore F1'                                   # Drawn separately (dashed)
bert_label = 'BERT-F1'                                         # Short label for plotting

# Token lengths to evaluate (multipliers of base length)
token_lengths = [8, 16, 32, 64]  # These represent the actual token length multipliers

# Paths to data files for different token lengths
token_length_paths = [
    'l10t8p8/t8/val_einv_metrics_summary.json',      # 8x tokens
    'l10t16p16/exp05/val_einv_metrics_summary.json',   # 16x tokens
    'l10t32p32/t32/val_einv_metrics_summary.json',   # 32x tokens
    'l10t64p64/t64/val_einv_metrics_summary.json'    # 64x tokens
]

# Load data for all token lengths
token_data = []
for path in token_length_paths:
    full_path = os.path.join(os.path.dirname(p_args.path.result_dir), path)
    data = read_json(full_path)
    token_data.append(data)

print(f"Loaded data for {len(token_data)} token length configurations")
print(f"Token lengths: {token_lengths}")
print(f"Available metrics: {list(token_data[0].keys())}")

# ============================================================
# COMPUTE MEAN VALUES FOR EACH TOKEN LENGTH AND METRIC
# ============================================================
mean_lexical = {metric: [] for metric in lexical_metrics}
mean_semantic = {metric: [] for metric in semantic_metrics}
mean_bert = []

for i, data in enumerate(token_data):
    print(f"\nProcessing token length {token_lengths[i]}x:")

    for metric in lexical_metrics:
        if metric in data:
            values = np.array(data[metric])
            mean_val = np.mean(values)
            mean_lexical[metric].append(mean_val)
            print(f"  {metric}: {mean_val:.4f}")
        else:
            print(f"  Warning: {metric} not found in data")
            mean_lexical[metric].append(np.nan)

    for metric in semantic_metrics:
        if metric in data:
            values = np.array(data[metric]) / 5.0
            mean_val = np.mean(values)
            mean_semantic[metric].append(mean_val)
            print(f"  {metric}: {mean_val:.4f}")
        else:
            print(f"  Warning: {metric} not found in data")
            mean_semantic[metric].append(np.nan)

    if bert_metric in data:
        values = np.array(data[bert_metric])
        mean_val = np.mean(values)
        mean_bert.append(mean_val)
        print(f"  {bert_metric}: {mean_val:.4f}")
    else:
        print(f"  Warning: {bert_metric} not found in data")
        mean_bert.append(np.nan)

# ============================================================
# PLOT 1: LEXICAL OVERLAP (ROUGE-1/2/L)
# ============================================================
x_positions = np.arange(len(token_lengths))

fig_dir = os.path.dirname(p_args.path.fig_dir)
os.makedirs(fig_dir, exist_ok=True)

colors_lexical = ['#fdae61', '#f46d43', '#d73027']
markers_lexical = ['v', 'p', 'h']

fig, ax = plt.subplots(figsize=(3, 2.8))
for i, metric in enumerate(lexical_metrics):
    ax.plot(x_positions, mean_lexical[metric],
            color=colors_lexical[i],
            marker=markers_lexical[i],
            linestyle='-',
            linewidth=2.5,
            markersize=8,
            label=metric,
            markeredgewidth=0.5,
            markeredgecolor='white',
            alpha=0.9,
            zorder=3)

ax.set_xlabel('Number of Tokens', fontsize=14)
ax.set_ylabel('Score (0-1)', fontsize=14)
ax.set_ylim(0, 1.05)
ax.set_xticks(x_positions)
ax.set_xticklabels([f'{length}' for length in token_lengths])
ax.grid(axis='y', alpha=0.3, linestyle='--', zorder=1)
ax.grid(axis='x', alpha=0.2, linestyle='--', zorder=1)
ax.set_title('Lexical Overlap', fontsize=16, fontweight='bold')
ax.legend(loc='best', frameon=True, framealpha=0.9, edgecolor='gray', fontsize=11)

plt.subplots_adjust(left=0.18, right=0.97, top=0.88, bottom=0.18)
fig_path = os.path.join(fig_dir, 'token_lexical.pdf')
plt.savefig(fig_path, dpi=300)
plt.close()
print(f"\n✓ Lexical plot saved to {fig_path}")

# ============================================================
# PLOT 2: SEMANTIC/STRUCTURAL PRESERVATION (Structure/Entity/Topic + BERTScore F1 dashed)
# ============================================================
colors_semantic = ['#74add1', '#4575b4', '#313695']
markers_semantic = ['o', 's', '^']

fig, ax = plt.subplots(figsize=(3, 2.8))
for i, metric in enumerate(semantic_metrics):
    ax.plot(x_positions, mean_semantic[metric],
            color=colors_semantic[i],
            marker=markers_semantic[i],
            linestyle='-',
            linewidth=2.5,
            markersize=8,
            label=metric,
            markeredgewidth=0.5,
            markeredgecolor='white',
            alpha=0.9,
            zorder=3)

# BERTScore F1 — dashed, distinct from LLM-as-a-judge
ax.plot(x_positions, mean_bert,
        color='#5e3c99',
        marker='D',
        linestyle='--',
        linewidth=2.5,
        markersize=8,
        label=bert_label,
        markeredgewidth=0.5,
        markeredgecolor='white',
        alpha=0.9,
        zorder=3)

ax.set_xlabel('Number of Tokens', fontsize=14)
ax.set_ylabel('Score (0-1)', fontsize=14)
ax.set_ylim(0, 1.05)
ax.set_xticks(x_positions)
ax.set_xticklabels([f'{length}' for length in token_lengths])
ax.grid(axis='y', alpha=0.3, linestyle='--', zorder=1)
ax.grid(axis='x', alpha=0.2, linestyle='--', zorder=1)
ax.set_title('Sem./Struct. Preservation', fontsize=16, fontweight='bold')
ax.legend(loc='best', frameon=True, framealpha=0.9, edgecolor='gray', fontsize=11)

plt.subplots_adjust(left=0.18, right=0.97, top=0.88, bottom=0.18)
fig_path = os.path.join(fig_dir, 'token_semantic.pdf')
plt.savefig(fig_path, dpi=300)
plt.close()
print(f"✓ Semantic plot saved to {fig_path}")