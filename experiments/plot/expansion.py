import numpy as np
import pandas as pd
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
continuous_metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
discrete_metrics = ['Structure', 'Entity', 'Topic', 'BERTScore F1']
expansions = [0.5, 1, 2, 4, 8]

# Paths to data files
paths = [
    'l10t16p16/exp05/val_einv_metrics_summary.json',
    'l10t16p16/exp1/val_einv_metrics_summary.json',
    'l10t16p16/exp2/val_einv_metrics_summary.json',
    'l10t16p16/exp4/val_einv_metrics_summary.json',
    'l10t16p16/exp8/val_einv_metrics_summary.json'
]

# Load data for all expansions
expansion_data = []
for path in paths:
    data = read_json(os.path.join(os.path.dirname(p_args.path.result_dir), path))
    expansion_data.append(data)

print(f"Loaded data for {len(expansion_data)} expansions")
print(f"Available metrics: {list(expansion_data[0].keys())}")

# ============================================================
# COMPUTE MEAN VALUES FOR EACH LAYER AND METRIC
# ============================================================
# Prepare data structure: {metric: [mean_l5, mean_l10, ...]}
mean_continuous = {metric: [] for metric in continuous_metrics}
mean_discrete = {metric: [] for metric in discrete_metrics}

for data in expansion_data:
    # Continuous metrics
    for metric in continuous_metrics:
        if metric in data:
            mean_val = np.mean(data[metric])
            mean_continuous[metric].append(mean_val)
        else:
            print(f"Warning: {metric} not found in data")
            mean_continuous[metric].append(np.nan)
    
    # Discrete metrics
    for metric in discrete_metrics:
        if metric in data:
            mean_val = np.mean(data[metric])
            if metric == 'Structure' or metric == 'Entity' or metric == 'Topic':
                mean_val = mean_val / 5.0
            mean_discrete[metric].append(mean_val)
        else:
            print(f"Warning: {metric} not found in data")
            mean_discrete[metric].append(np.nan)

# ============================================================
# CREATE SINGLE-AXIS PLOT (SAME SCALE)
# ============================================================
fig, ax = plt.subplots(figsize=(4, 3.2))

# Create evenly spaced x positions
x_positions = np.arange(len(expansions))

# Define colors and markers for continuous metrics - ORANGE
colors_left = ['#fdae61', '#f46d43', '#d73027']
markers_left = ['v', 'p', 'h']
linestyles_left = ['-', '-', '-']

# Plot continuous metrics
for i, metric in enumerate(continuous_metrics):
    ax.plot(x_positions, mean_continuous[metric], 
            color=colors_left[i], 
            marker=markers_left[i], 
            linestyle=linestyles_left[i],
            linewidth=2, 
            markersize=7,
            label=metric,
            markeredgewidth=0,
            markeredgecolor='white',
            alpha=0.9)

# Define colors and markers for discrete metrics - BLUE
colors_right = ['#74add1', '#4575b4', '#313695', '#abd9e9']
markers_right = ['o', 's', '^', 'D']
linestyles_right = ['-.', '-.', '-.', '-.']

# Plot discrete metrics on same axis
for i, metric in enumerate(discrete_metrics):
    ax.plot(x_positions, mean_discrete[metric], 
            color=colors_right[i], 
            marker=markers_right[i], 
            linestyle=linestyles_right[i],
            linewidth=2, 
            markersize=7,
            label=metric,
            markeredgewidth=0,
            alpha=0.9)

# Customize axis
ax.set_xlabel('Adapter Hidden Expansion Factor $f$')
ax.set_ylabel('Score (0-1)')
ax.set_ylim(0, 1.05)
ax.set_xticks(x_positions)
ax.set_xticklabels([f'{expansion}x' for expansion in expansions])
ax.grid(axis='y', alpha=0.3, linestyle='--', zorder=1)
ax.grid(axis='x', alpha=0.2, linestyle='--', zorder=1)

# Title
ax.set_title('Performance on Adapter Hidden Expansion Factors $f$')

# Get all legend elements
lines, labels = ax.get_legend_handles_labels()

# Create legend
legend = fig.legend(
    lines,
    labels,
    loc='lower left',
    bbox_to_anchor=(1, 0.35),
    frameon=True,
    ncol=1,
    framealpha=0.9,
    edgecolor='gray',
    markerscale=0.7,
    title='Metrics',
    fontsize=9
)

# Adjust layout to make room for the legend
plt.tight_layout(rect=[0, 0, 0.9, 1])

# Add legend back to the axes
ax.add_artist(legend)

# Save figure
fig_dir = os.path.dirname(p_args.path.fig_dir)
os.makedirs(fig_dir, exist_ok=True)
fig_path = os.path.join(fig_dir, 'expansion.pdf')
plt.tight_layout()
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Dual y-axis plot saved to {fig_path}")