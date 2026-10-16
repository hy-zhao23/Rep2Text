# ============================================================
# CREATE OPTION B FIGURE — Style matching target (Image 2)
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy import stats
from matplotlib.patches import Patch, Polygon, FancyBboxPatch, Circle
from matplotlib.lines import Line2D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from utils.core.files import read_json
from utils.plot.fonts import set_fonts
from utils import p_args

set_fonts()

pre_data = read_json(os.path.join(os.getenv("PROJ"), "results/llama8B/l10t16p16/exp4/val_einv_metrics_summary.json"))
ft_data = read_json(os.path.join(os.getenv("PROJ"), "results/llama8B/l10t16p16/exp4/val_einv_lora_metrics_summary.json"))

# -----------------------------
# Colors
# -----------------------------
colors = {
    'Pretrain': '#CFE4C3',
    'Finetune': '#9FCFE3',
    'Mean': '#2c2c2c',
    'Line': '#8a8a8a',
    'panel_bg': '#f0f5fa',
    'panel_border': '#b0c4d8',
    'badge_bg': '#5b7f5b',
}

# Discrete score colors (0-5): user palette
score_colors = ['#e6e5e6', '#f1e3d1', '#e8e4c0', '#cee3c6', '#97cdc7', '#6caace']
score_edge   = ['#999999', '#c8a060', '#b8b060', '#7a9a6a', '#4a8a4a', '#2a6ea0']

continuous_metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BERTScore F1']
discrete_metrics = ['Structure', 'Entity', 'Topic']
classes = ['Pretrain', 'Finetune']


# ============================================================
# Helper: draw rounded rectangle border around an axes
# ============================================================
def add_panel_border(fig, ax, pad=0.012, radius=0.02):
    bbox = ax.get_position()
    rect = FancyBboxPatch(
        (bbox.x0 - pad, bbox.y0 - pad),
        bbox.width + 2 * pad,
        bbox.height + 2 * pad,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor='none',
        edgecolor=colors['panel_border'],
        linewidth=1.2,
        transform=fig.transFigure,
        zorder=-10,
        clip_on=False,
    )
    fig.add_artist(rect)


# ============================================================
# Helper: draw badge label (circled letter)
# ============================================================
def add_badge(fig, ax, letter, x_offset=-0.04, y_offset=1.06):
    ax.text(
        x_offset, y_offset, letter,
        transform=ax.transAxes,
        fontsize=13, fontweight='bold', color='white',
        ha='center', va='center',
        bbox=dict(
            boxstyle='circle,pad=0.25',
            facecolor=colors['badge_bg'],
            edgecolor='none'
        ),
        zorder=10
    )


# ============================================================
# Panel (a): Violin + box — each metric has Pretrain/Finetune
#            side by side with x-tick labels per violin
# ============================================================
def draw_violin_box(ax, pre_data, ft_data, metrics):
    # Each metric group: positions for Pretrain, Finetune
    group_gap = 3.6
    within_gap = 1.4

    metric_centers = []
    for i, metric in enumerate(metrics):
        center = i * group_gap
        pos_pre = center - within_gap / 2
        pos_ft = center + within_gap / 2
        metric_centers.append(center)

        for j, (class_name, pos) in enumerate(zip(classes, [pos_pre, pos_ft])):
            class_data = np.array(pre_data[metric] if class_name == 'Pretrain' else ft_data[metric])

            # Violin by KDE
            if len(np.unique(class_data)) > 1:
                kde = stats.gaussian_kde(class_data)
                y_min, y_max = class_data.min(), class_data.max()
                pad_v = 0.05 * (y_max - y_min + 1e-6)
                y_range = np.linspace(y_min - pad_v, y_max + pad_v, 150)
                y_range = np.clip(y_range, 0, 1)

                density = kde(y_range)
                density = density / density.max() * 0.38

                x_right = pos + density
                x_left = pos - density

                vertices = (
                    [(x, y) for x, y in zip(x_right, y_range)] +
                    [(x, y) for x, y in zip(reversed(x_left), reversed(y_range))]
                )

                poly = Polygon(
                    vertices,
                    facecolor=colors[class_name],
                    edgecolor='none',
                    alpha=0.3,
                    linewidth=0,
                    zorder=2
                )
                ax.add_patch(poly)

            # Boxplot
            ax.boxplot(
                [class_data],
                positions=[pos],
                widths=0.28,
                patch_artist=True,
                showfliers=False,
                showmeans=True,
                whis=[5, 95],
                medianprops=dict(color=colors['Mean'], linewidth=0.8),
                meanprops=dict(
                    marker='*',
                    markerfacecolor=colors['Mean'],
                    markeredgecolor=colors['Mean'],
                    markersize=6,
                    markeredgewidth=0.7,
                    zorder=6
                ),
                boxprops=dict(
                    facecolor=colors[class_name],
                    alpha=0.95,
                    edgecolor=colors['Mean'],
                    linewidth=0.7,
                    zorder=5
                ),
                whiskerprops=dict(color=colors['Mean'], linewidth=0.7, zorder=5),
                capprops=dict(color=colors['Mean'], linewidth=0.7, zorder=5),
                vert=True,
                manage_ticks=False
            )


    ax.set_ylabel('Score', fontsize=10)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlim(-1.2, (len(metrics) - 1) * group_gap + 1.2)

    ax.set_xticks(metric_centers)
    ax.set_xticklabels(metrics, rotation=0, fontsize=10, fontweight='bold')
    ax.tick_params(axis='y', labelsize=9)

    ax.grid(axis='y', alpha=0.25, linestyle='--', linewidth=0.6, zorder=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend inside panel (lower left)
    legend_elements = [
        Patch(facecolor=colors['Pretrain'], alpha=0.8, edgecolor='#4f884f', label='Pretrain'),
        Patch(facecolor=colors['Finetune'], alpha=0.8, edgecolor='#2479a8', label='Finetune'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor=colors['Mean'],
               markeredgecolor=colors['Mean'], markersize=8, label='Mean', linestyle='None')
    ]
    ax.legend(
        handles=legend_elements,
        loc='lower right',
        fontsize=9,
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        edgecolor='#cccccc',
        ncol=1,
        handlelength=1.2,
    )


# ============================================================
# Panel (b): 100% stacked bars for discrete metrics
# ============================================================
def draw_discrete_stacked(ax, pre_data, ft_data, metrics):
    y_base = np.arange(len(metrics))[::-1] * 2.4
    bar_h = 0.65

    for i, metric in enumerate(metrics):
        y = y_base[i]

        pre_counts = np.array([np.sum(np.array(pre_data[metric]) == s) for s in range(6)], dtype=float)
        ft_counts = np.array([np.sum(np.array(ft_data[metric]) == s) for s in range(6)], dtype=float)

        pre_pct = pre_counts / pre_counts.sum() * 100
        ft_pct = ft_counts / ft_counts.sum() * 100

        # Metric title
        ax.text(
            50, y + 0.85,
            f'{metric} Score',
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold'
        )

        # Pretrain bar
        left = 0
        for s in range(6):
            ax.barh(
                y + bar_h / 2,
                pre_pct[s],
                left=left,
                height=bar_h,
                color=score_colors[s],
                edgecolor=score_edge[s],
                linewidth=0.5
            )
            if pre_pct[s] >= 3:
                ax.text(left + pre_pct[s] / 2, y + bar_h / 2, f'{pre_pct[s]:.0f}%',
                        ha='center', va='center', fontsize=8, fontweight='bold')
            left += pre_pct[s]

        # Finetune bar
        left = 0
        for s in range(6):
            ax.barh(
                y - bar_h / 2,
                ft_pct[s],
                left=left,
                height=bar_h,
                color=score_colors[s],
                edgecolor=score_edge[s],
                linewidth=0.5
            )
            if ft_pct[s] >= 3:
                ax.text(left + ft_pct[s] / 2, y - bar_h / 2, f'{ft_pct[s]:.0f}%',
                        ha='center', va='center', fontsize=8, fontweight='bold')
            left += ft_pct[s]

        # Left labels
        ax.text(-3, y + bar_h / 2, 'Pretrain', ha='right', va='center',
                fontsize=9.5, color='#4f884f', fontweight='bold')
        ax.text(-3, y - bar_h / 2, 'Finetune', ha='right', va='center',
                fontsize=9.5, color='#2479a8', fontweight='bold')

    ax.set_xlim(-18, 100)
    ax.set_ylim(-1.0, y_base[0] + 3.4)
    ax.set_xlabel('Percentage of Samples', fontsize=10)
    ax.set_yticks([])
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_xticklabels([f'{x}%' for x in np.arange(0, 101, 20)], fontsize=9)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Score-level legend (larger, colored with new palette)
    legend_handles = [
        Patch(facecolor=score_colors[s], edgecolor=score_edge[s], linewidth=0.8, label=str(s))
        for s in range(6)
    ]
    ax.legend(
        handles=legend_handles,
        title='Score',
        ncol=6,
        fontsize=9.5,
        title_fontsize=10,
        frameon=False,
        loc='upper center',
        bbox_to_anchor=(0.52, 1.04),
        handlelength=1.2,
        columnspacing=1.0
    )


# ============================================================
# Panel (c): Mean dumbbell — show actual values, not deltas
#            Continuous on left (0-1), Discrete on right (0-5)
# ============================================================
def draw_mean_summary(ax, pre_data, ft_data, continuous_metrics, discrete_metrics):
    all_metrics = continuous_metrics + discrete_metrics
    n_cont = len(continuous_metrics)
    n_disc = len(discrete_metrics)
    n_all = len(all_metrics)

    x = np.arange(n_all)

    pre_means_raw = []
    ft_means_raw = []
    pre_stds_raw = []
    ft_stds_raw = []

    for m in continuous_metrics:
        pre_means_raw.append(np.mean(pre_data[m]))
        ft_means_raw.append(np.mean(ft_data[m]))
        pre_stds_raw.append(np.std(pre_data[m]))
        ft_stds_raw.append(np.std(ft_data[m]))

    for m in discrete_metrics:
        pre_means_raw.append(np.mean(pre_data[m]))
        ft_means_raw.append(np.mean(ft_data[m]))
        pre_stds_raw.append(np.std(pre_data[m]))
        ft_stds_raw.append(np.std(ft_data[m]))

    pre_means_raw = np.array(pre_means_raw)
    ft_means_raw = np.array(ft_means_raw)
    pre_stds_raw = np.array(pre_stds_raw)
    ft_stds_raw = np.array(ft_stds_raw)

    # Discrete /5 to map to same 0-1 axis as continuous
    pre_plot = pre_means_raw.copy()
    ft_plot = ft_means_raw.copy()
    pre_err = pre_stds_raw.copy()
    ft_err = ft_stds_raw.copy()
    for i in range(n_cont, n_all):
        pre_plot[i] = pre_means_raw[i] / 5.0
        ft_plot[i] = ft_means_raw[i] / 5.0
        pre_err[i]  = pre_stds_raw[i]  / 5.0
        ft_err[i]   = ft_stds_raw[i]   / 5.0

    # Horizontal offset so dumbbell line has slope, not vertical
    dx = 0.28
    x_pre = x - dx
    x_ft  = x + dx

    # Connecting line segments
    for i in range(n_all):
        ax.plot(
            [x_pre[i], x_ft[i]],
            [pre_plot[i], ft_plot[i]],
            color=colors['Line'],
            linewidth=1.4,
            zorder=1,
        )

    # Scatter dots
    ax.scatter(x_pre, pre_plot, s=70, color=colors['Pretrain'],
               edgecolor='#4f884f', linewidth=1.1, zorder=3)
    ax.scatter(x_ft, ft_plot, s=70, color=colors['Finetune'],
               edgecolor='#2479a8', linewidth=1.1, zorder=3)

    # Mean value labels (raw scale: 0-1 for continuous, 0-5 for discrete)
    for i in range(n_all):
        ax.text(
            x_pre[i], pre_plot[i] - 0.07,
            f'{pre_means_raw[i]:.2f}',
            ha='center', va='top',
            fontsize=11, color='#4f884f', fontweight='bold',
            alpha=1.0, zorder=10,
        )
        ax.text(
            x_ft[i], ft_plot[i] - 0.07,
            f'{ft_means_raw[i]:.2f}',
            ha='center', va='top',
            fontsize=11, color='#2479a8', fontweight='bold',
            alpha=1.0, zorder=10,
        )

        # Improvement Δ label above the line midpoint
        delta = ft_means_raw[i] - pre_means_raw[i]
        sign = '+' if delta >= 0 else ''
        mid_x = x[i]
        top_y = max(pre_plot[i], ft_plot[i])
        ax.text(
            mid_x, top_y + 0.06,
            f'Δ {sign}{delta:.2f}',
            ha='center', va='bottom',
            fontsize=10, color='#444444', fontweight='bold',
            alpha=1.0, zorder=10,
        )

    # Divider line
    ax.axvline(n_cont - 0.5, color='#999999', linewidth=0.8, linestyle='--', alpha=0.7)

    # Per-metric icons below x-axis (DejaVu Sans for symbol coverage)
    metric_icons = ['◇', '◇', '◇', '◇', '▦', '◉', '◐']
    for i, icon in enumerate(metric_icons):
        ax.text(
            x[i], -0.22, icon,
            ha='center', va='top',
            fontsize=13,
            color='#555555',
            fontfamily='DejaVu Sans',
            transform=ax.get_xaxis_transform(),
        )

    # Category labels (further down to leave room for x-tick + icon)
    ax.text(
        (n_cont - 1) / 2, -0.40, 'Continuous Metrics',
        ha='center', va='top',
        fontsize=12, color='#4f884f', fontweight='bold',
        transform=ax.get_xaxis_transform(),
    )
    ax.text(
        n_cont + (n_disc - 1) / 2, -0.40, 'Discrete Metrics (0–5)',
        ha='center', va='top',
        fontsize=12, color='#2479a8', fontweight='bold',
        transform=ax.get_xaxis_transform(),
    )

    ax.set_ylabel('Score (Mean)', fontsize=12)
    ax.set_ylim(-0.02, 1.18)
    ax.set_xticks(x)
    ax.set_xticklabels(all_metrics, rotation=0, fontsize=12, fontweight='bold')
    ax.tick_params(axis='y', labelsize=11)

    ax.grid(axis='y', alpha=0.25, linestyle='--', linewidth=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['Pretrain'],
               markeredgecolor='#4f884f', markersize=10, label='Pretrain', linestyle='None'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['Finetune'],
               markeredgecolor='#2479a8', markersize=10, label='Finetune', linestyle='None'),
        Line2D([0], [0], color=colors['Line'], linewidth=1.4, label='Improvement'),
    ]
    ax.legend(
        handles=legend_elements,
        loc='lower right',
        fontsize=11,
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        edgecolor='#cccccc',
        ncol=1,
    )


# ============================================================
# Build three separate figures
# ============================================================
os.makedirs(p_args.path.fig_dir, exist_ok=True)


def render_panel(draw_fn, fname, figsize, draw_args, border=True):
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    draw_fn(ax, *draw_args)
    if border:
        fig.canvas.draw()
        add_panel_border(fig, ax, pad=0.015)
    base = os.path.join(p_args.path.fig_dir, os.path.splitext(fname)[0])
    for ext in ('.pdf', '.svg'):
        out = base + ext
        fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved {out}")
    plt.close(fig)


render_panel(
    draw_violin_box,
    'pre-ft-a-violin.pdf',
    figsize=(4, 2.5),
    draw_args=(pre_data, ft_data, continuous_metrics),
    border=False,
)

render_panel(
    draw_discrete_stacked,
    'pre-ft-b-discrete.pdf',
    figsize=(4, 2.5),
    draw_args=(pre_data, ft_data, discrete_metrics),
    border=False,
)

render_panel(
    draw_mean_summary,
    'pre-ft-c-mean.pdf',
    figsize=(7, 2),
    draw_args=(pre_data, ft_data, continuous_metrics, discrete_metrics),
    border=False,
)