import os
from utils.core.files import read_json

file = os.path.join(os.getenv("PROJ"), "results", "ood.json")
data = read_json(file)

# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd
# sns.set_style("whitegrid")

# # Set font family globally
# plt.rcParams['font.family'] = 'EB Garamond'
# plt.rcParams['font.weight'] = 'semibold'
# plt.rcParams['axes.labelweight'] = 'semibold'
# plt.rcParams['axes.titleweight'] = 'bold'
# plt.rcParams['axes.labelsize'] = 11        # x and y labels
# plt.rcParams['axes.titlesize'] = 12        # title
# plt.rcParams['xtick.labelsize'] = 10        # x-axis tick labels
# plt.rcParams['ytick.labelsize'] = 10        # y-axis tick labels
# plt.rcParams['legend.fontsize'] = 8 

# # Prepare the data as a long-form DataFrame
# df = pd.DataFrame({
#     "ROUGE-1": [d.get("ROUGE-1", 0) for d in data],
#     "ROUGE-2": [d.get("ROUGE-2", 0) for d in data],
#     "ROUGE-L": [d.get("ROUGE-L", 0) for d in data],
#     "BERTScore": [d.get("BERTScore F1", 0) for d in data],
# })

# # Better styling
# fig, axes = plt.subplots(1, 4, figsize=(9, 2.5))
# axes = axes.flatten()

# metrics = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore"]
# colors = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3"]
# thresholds = [0.48, 0.28, 0.47, 0.78]  # Adjust these thresholds as needed

# for idx, (ax, metric, color, threshold) in enumerate(zip(axes, metrics, colors, thresholds)):
#     ax.hist(df[metric], bins=20, color=color, alpha=0.7, edgecolor="black", linewidth=0.8)
#     ax.set_xlabel("Score")
#     ax.set_ylabel("Frequency")
#     ax.set_title(metric)
#     ax.set_xlim(0, 1)
#     ax.grid(axis="y", alpha=0.3)
    
#     # Add threshold line
#     ax.axvline(threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold: {threshold}")
    
#     # Calculate percentage above threshold
#     pct_above = (df[metric] >= threshold).sum() / len(df[metric]) * 100
#     ax.text(0.98, 0.97, f"{pct_above:.1f}% above\nthreshold", 
#             transform=ax.transAxes, fontsize=9, verticalalignment="top", 
#             horizontalalignment="right", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
#     ax.legend(loc="upper left")

# plt.suptitle("Distribution of Evaluation Metrics", fontsize=12, fontweight="bold")
# plt.tight_layout()

# save_path = os.path.join(os.getenv("PROJ"), "results", "figs", "ood_rouge.pdf")
# plt.savefig(save_path, dpi=300, bbox_inches='tight')
# plt.close()
# print(f"✓ OOD evaluation metrics plot saved to {save_path}")


import numpy as np
Metrics = ['Structure', 'Entity', 'Topic'] 
name = {'Structure': 'structure_score', 'Entity': 'entity_score', 'Topic': 'topic_score'} 
print("GPT Evaluation metrics:")
structure_scores = []
entity_scores = []
topic_scores = []

for d in data:
    structure_scores.append(d['structure_score']/5.0)
    entity_scores.append(d['entity_score']/5.0)
    topic_scores.append(d['topic_score']/5.0)

print(f"Structure: {np.mean(structure_scores):.2f} ± {np.std(structure_scores):.2f}")
print(f"Entity: {np.mean(entity_scores):.2f} ± {np.std(entity_scores):.2f}")
print(f"Topic: {np.mean(topic_scores):.2f} ± {np.std(topic_scores):.2f}")


file = os.path.join(os.getenv("PROJ"), "results", "human.csv")
import pandas as pd

df = pd.read_csv(file)
print("Human metrics:")
structure_scores = []
entity_scores = []
topic_scores = []
for i in range(len(df)):
    structure_scores.append(df.iloc[i]['Structure']/5.0)
    entity_scores.append(df.iloc[i]['Entity']/5.0)
    topic_scores.append(df.iloc[i]['Topic']/5.0)

print(f"Structure: {np.mean(structure_scores):.2f} ± {np.std(structure_scores):.2f}")
print(f"Entity: {np.mean(entity_scores):.2f} ± {np.std(entity_scores):.2f}")
print(f"Topic: {np.mean(topic_scores):.2f} ± {np.std(topic_scores):.2f}")