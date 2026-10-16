import numpy as np
from utils.core.files import read_json
import os
from utils.plot.fonts import set_fonts

# ============================================================
# YOUR CONFIGURATION
# ============================================================
set_fonts()

# Metrics to plot
continuous_metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BERTScore F1']
discrete_metrics = ['Structure', 'Entity', 'Topic']
transfer = ['g7b', 'm7b', 'l8b', 'l8b3b', 'm3b', 'l3b']

# Paths to data files
paths = [
    'gemma7B/l10t16p16/g7b/val_einv_metrics_summary.json',
    'mistral7B/l10t16p16/m7b/val_einv_metrics_summary.json',
    'llama8B/l10t16p16/exp05/val_einv_metrics_summary.json',
    'llama3B/l10t16p16/l8b/val_einv_metrics_summary.json',
    'mistral7B/l10t16p16/m3b/val_einv_metrics_summary.json',
    'llama3B/l10t16p16/l3b/val_einv_metrics_summary.json'
]

# Load data for all expansions
transfer_data = []
for path in paths:
    data = read_json(os.path.join(os.getenv("PROJ"), "results" ,path))
    transfer_data.append(data)

print(f"Loaded data for {len(transfer_data)} models.")
print(f"Available metrics: {list(transfer_data[0].keys())}")

# ============================================================
# COMPUTE MEAN VALUES FOR EACH LAYER AND METRIC
# ============================================================
# Prepare data structure: {metric: [mean_l5, mean_l10, ...]}

for i, data in enumerate(transfer_data):
    # Continuous metrics
    print(f"Processing data for model {transfer[i]}")
    for metric in continuous_metrics:
        if metric in data:
            mean_val = np.mean(data[metric])
            std_val = np.std(data[metric])
            print(f"{metric}: Mean = {mean_val:.2f}, Std = {std_val:.2f}")
        else:
            print(f"Warning: {metric} not found in data")
    
    # Discrete metrics
    for metric in discrete_metrics:
        if metric in data:
            data[metric] = np.array(data[metric]) / 5.0
            mean_val = np.mean(data[metric])
            std_val = np.std(data[metric])
            print(f"{metric}: Mean = {mean_val:.2f}, Std = {std_val:.2f}")
        else:
            print(f"Warning: {metric} not found in data")