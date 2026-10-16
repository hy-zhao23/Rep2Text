import numpy as np
import matplotlib.pyplot as plt
from utils.core.files import read_json
from utils import p_args
import os
from utils.plot.fonts import set_fonts

# ============================================================
# YOUR CONFIGURATION
# ============================================================
set_fonts()

# Metrics to plot
lexical_metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']           # Lexical Overlap
semantic_metrics = ['Structure', 'Entity', 'Topic']            # Semantic/Structural Preservation (LLM-as-a-judge)
bert_metric = 'BERTScore F1'                                   # Drawn separately (dashed)
bert_label = 'BERT-F1'                                         # Short label for plotting
layers = [5, 10, 15, 20, 25, 30]

# Paths to data files
paths = [
    'l5t16p16/l5/val_einv_metrics_summary.json',
    'l10t16p16/exp05/val_einv_metrics_summary.json',
    'l15t16p16/l15/val_einv_metrics_summary.json',
    'l20t16p16/l20/val_einv_metrics_summary.json',
    'l25t16p16/l25/val_einv_metrics_summary.json',
    'l30t16p16/l30/val_einv_metrics_summary.json'
]

# Load data for all layers
layer_data = []
for path in paths:
    data = read_json(os.path.join(os.path.dirname(p_args.path.result_dir), path))
    layer_data.append(data)

print(f"Loaded data for {len(layer_data)} layers")
print(f"Available metrics: {list(layer_data[0].keys())}")

# ============================================================
# COMPUTE MEAN VALUES FOR EACH LAYER AND METRIC
# ============================================================
mean_lexical = {metric: [] for metric in lexical_metrics}
mean_semantic = {metric: [] for metric in semantic_metrics}
mean_bert = []

for data in layer_data:
    for metric in lexical_metrics:
        if metric in data:
            mean_lexical[metric].append(np.mean(data[metric]))
        else:
            print(f"Warning: {metric} not found in data")
            mean_lexical[metric].append(np.nan)

    for metric in semantic_metrics:
        if metric in data:
            mean_lexical_val = np.mean(data[metric]) / 5.0
            mean_semantic[metric].append(mean_lexical_val)
        else:
            print(f"Warning: {metric} not found in data")
            mean_semantic[metric].append(np.nan)

    if bert_metric in data:
        mean_bert.append(np.mean(data[bert_metric]))
    else:
        print(f"Warning: {bert_metric} not found in data")
        mean_bert.append(np.nan)

# ============================================================
# PLOT 1: LEXICAL OVERLAP (ROUGE-1/2/L)
# ============================================================
fig_dir = os.path.dirname(p_args.path.fig_dir)
os.makedirs(fig_dir, exist_ok=True)

colors_lexical = ['#fdae61', '#f46d43', '#d73027']
markers_lexical = ['v', 'p', 'h']

fig, ax = plt.subplots(figsize=(3, 2.8))
for i, metric in enumerate(lexical_metrics):
    ax.plot(layers, mean_lexical[metric],
            color=colors_lexical[i],
            marker=markers_lexical[i],
            linestyle='-',
            linewidth=2,
            markersize=7,
            label=metric,
            markeredgewidth=0,
            alpha=0.9)

ax.set_xlabel('ℓ-th Layers', fontsize=14)
ax.set_ylabel('Score (0-1)', fontsize=14)
ax.set_ylim(0, 1.05)
ax.set_xticks(layers)
ax.set_xticklabels([f'L{layer}' for layer in layers])
ax.grid(axis='y', alpha=0.3, linestyle='--', zorder=1)
ax.grid(axis='x', alpha=0.2, linestyle='--', zorder=1)
ax.set_title('Lexical Overlap', fontsize=16, fontweight='bold')
ax.legend(loc='best', frameon=True, framealpha=0.9, edgecolor='gray', fontsize=11)

plt.subplots_adjust(left=0.18, right=0.97, top=0.88, bottom=0.18)
fig_path = os.path.join(fig_dir, 'layerwise_lexical.pdf')
plt.savefig(fig_path, dpi=300)
plt.close()
print(f"✓ Lexical plot saved to {fig_path}")

# ============================================================
# PLOT 2: SEMANTIC/STRUCTURAL PRESERVATION (Structure/Entity/Topic + BERTScore F1 dashed)
# ============================================================
colors_semantic = ['#74add1', '#4575b4', '#313695']
markers_semantic = ['o', 's', '^']

fig, ax = plt.subplots(figsize=(3, 2.8))
for i, metric in enumerate(semantic_metrics):
    ax.plot(layers, mean_semantic[metric],
            color=colors_semantic[i],
            marker=markers_semantic[i],
            linestyle='-',
            linewidth=2,
            markersize=7,
            label=metric,
            markeredgewidth=0,
            alpha=0.9)

# BERTScore F1 — dashed, distinct from LLM-as-a-judge
ax.plot(layers, mean_bert,
        color='#5e3c99',
        marker='D',
        linestyle='--',
        linewidth=2,
        markersize=7,
        label=bert_label,
        markeredgewidth=0,
        alpha=0.9)

ax.set_xlabel('ℓ-th Layers', fontsize=14)
ax.set_ylabel('Score (0-1)', fontsize=14)
ax.set_ylim(0, 1.05)
ax.set_xticks(layers)
ax.set_xticklabels([f'L{layer}' for layer in layers])
ax.grid(axis='y', alpha=0.3, linestyle='--', zorder=1)
ax.grid(axis='x', alpha=0.2, linestyle='--', zorder=1)
ax.set_title('Sem./Struct. Preservation', fontsize=16, fontweight='bold')
ax.legend(loc='best', frameon=True, framealpha=0.9, edgecolor='gray', fontsize=11)

plt.subplots_adjust(left=0.18, right=0.97, top=0.88, bottom=0.18)
fig_path = os.path.join(fig_dir, 'layerwise_semantic.pdf')
plt.savefig(fig_path, dpi=300)
plt.close()
print(f"✓ Semantic plot saved to {fig_path}")