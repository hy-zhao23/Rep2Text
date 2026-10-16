import numpy as np
import os
from utils.core.files import read_json
from utils import p_args

judge_metrics = ['Structure', 'Entity', 'Topic']

schemes = {
    'Adapter Expansion Factor': (
        ['0.5x', '1x', '2x', '4x', '8x'],
        [
            'l10t16p16/exp05/val_einv_metrics_summary.json',
            'l10t16p16/exp1/val_einv_metrics_summary.json',
            'l10t16p16/exp2/val_einv_metrics_summary.json',
            'l10t16p16/exp4/val_einv_metrics_summary.json',
            'l10t16p16/exp8/val_einv_metrics_summary.json',
        ],
    ),
    'Number of Tokens': (
        ['8', '16', '32', '64'],
        [
            'l10t8p8/t8/val_einv_metrics_summary.json',
            'l10t16p16/exp05/val_einv_metrics_summary.json',
            'l10t32p32/t32/val_einv_metrics_summary.json',
            'l10t64p64/t64/val_einv_metrics_summary.json',
        ],
    ),
    'Layer ℓ': (
        ['L5', 'L10', 'L15', 'L20', 'L25', 'L30'],
        [
            'l5t16p16/l5/val_einv_metrics_summary.json',
            'l10t16p16/exp05/val_einv_metrics_summary.json',
            'l15t16p16/l15/val_einv_metrics_summary.json',
            'l20t16p16/l20/val_einv_metrics_summary.json',
            'l25t16p16/l25/val_einv_metrics_summary.json',
            'l30t16p16/l30/val_einv_metrics_summary.json',
        ],
    ),
}

base = os.path.dirname(p_args.path.result_dir)


def per_config(path):
    data = read_json(os.path.join(base, path))
    return {m: float(np.mean(data[m]) / 5.0) for m in judge_metrics}


for name, (labels, paths) in schemes.items():
    print(f"\n=== {name} (LLM-as-a-judge, normalized 0-1) ===")
    rows = [per_config(p) for p in paths]
    header = f"{'Metric':<11}" + "".join(f"{l:>9}" for l in labels) + f"{'Mean':>9}{'Std':>9}{'Range':>9}"
    print(header)
    print("-" * len(header))
    for m in judge_metrics:
        series = np.array([r[m] for r in rows])
        cells = "".join(f"{v:>9.4f}" for v in series)
        print(f"{m:<11}{cells}{series.mean():>9.4f}{series.std(ddof=0):>9.4f}{series.max() - series.min():>9.4f}")
