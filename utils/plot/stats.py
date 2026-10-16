import os
from typing import List
from utils.core.files import read_json
from utils import p_args
import numpy as np


def get_metrics():
    return {
        "loss": "loss",
        "perplexity": "perplexity",
        "rouge": ["rouge1", "rouge2", "rougeL"],
        # "bleu": ["bleu", "bp", "precision_1", "precision_2", "precision_3"],
        "bertscore": ["precision", "recall", "f1"]
    }

def get_metric_files(label: str) -> List[str]:
    p_dir = os.getenv("PROJ", ".")
    file = os.path.join(p_dir, "labels.json")
    
    files = {}
    _labels = read_json(file)[label]
    for k, v in _labels.items():
        files[k] = os.path.join(p_dir, "results", f"{p_args.path.model_name}", v)

    return files

def get_metrics_stats(label):
    files = get_metric_files(label)

    stats = {}
    for k, f in files.items():
        data = read_json(f)
        # Assume data is a list of metric dicts
        all_metrics = get_metrics()
        for metric, _sub_metrics in all_metrics.items():
            if metric not in stats.keys():
                stats[metric] = {}
            if isinstance(_sub_metrics, list):
                for _n in _sub_metrics:
                    if _n not in stats[metric].keys():
                        stats[metric][_n] = {}
                    vals = np.array(data[_n])
                    stats[metric][_n][k] = vals 
            else:
                stats[metric][k] = data[metric]
                
    return stats
