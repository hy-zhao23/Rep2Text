import numpy as np
import os
from utils.core.files import read_json
from utils import p_args

lexical_metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
semantic_metrics = ['Structure', 'Entity', 'Topic']
bert_metric = 'BERTScore F1'

expansions = [0.5, 1, 2, 4, 8]
expansion_paths = [
    'l10t16p16/exp05/val_einv_metrics_summary.json',
    'l10t16p16/exp1/val_einv_metrics_summary.json',
    'l10t16p16/exp2/val_einv_metrics_summary.json',
    'l10t16p16/exp4/val_einv_metrics_summary.json',
    'l10t16p16/exp8/val_einv_metrics_summary.json',
]

token_lengths = [8, 16, 32, 64]
token_paths = [
    'l10t8p8/t8/val_einv_metrics_summary.json',
    'l10t16p16/exp05/val_einv_metrics_summary.json',
    'l10t32p32/t32/val_einv_metrics_summary.json',
    'l10t64p64/t64/val_einv_metrics_summary.json',
]

base = os.path.dirname(p_args.path.result_dir)


def per_config_mean(path):
    data = read_json(os.path.join(base, path))
    out = {}
    for m in lexical_metrics:
        out[m] = float(np.mean(data[m]))
    for m in semantic_metrics:
        out[m] = float(np.mean(data[m]) / 5.0)
    out[bert_metric] = float(np.mean(data[bert_metric]))
    return out


def summarize(name, values, paths):
    print(f"\n=== {name} ===")
    rows = [per_config_mean(p) for p in paths]
    metrics = lexical_metrics + semantic_metrics + [bert_metric]
    header = f"{'Metric':<14}" + "".join(f"{str(v):>9}" for v in values) + f"{'Mean':>9}{'Std':>9}{'Range':>9}"
    print(header)
    print("-" * len(header))
    for m in metrics:
        series = np.array([r[m] for r in rows])
        mean = series.mean()
        std = series.std(ddof=0)
        rng = series.max() - series.min()
        cells = "".join(f"{v:>9.4f}" for v in series)
        print(f"{m:<14}{cells}{mean:>9.4f}{std:>9.4f}{rng:>9.4f}")


summarize("Adapter Expansion Factor", [f"{e}x" for e in expansions], expansion_paths)
summarize("Number of Tokens", token_lengths, token_paths)
