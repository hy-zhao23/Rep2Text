#!/usr/bin/env python
# coding: utf-8

from utils.core.path import get_metrics_path, get_gpt_scores_path, get_metrics_sum_path
import argparse
from utils.project_utils import set_global_seed
from utils.core.files import read_json, write_json
import math

args = argparse.ArgumentParser()
args.add_argument("--model_type", type=str, default="einv", required=False)
args = args.parse_args()

set_global_seed(42)

file = get_metrics_path(split="val", model_type=args.model_type)
data = read_json(file)

def get_gpt_scores():

    file = get_gpt_scores_path(split="val", model_type=args.model_type)
    gpt_scores = read_json(file)

    _structure_scores = []
    _entity_scores = []
    _topic_scores = []
    _rouge1_scores = []
    _rouge2_scores = []
    _rougeL_scores = []
    _bertscore_f1_scores = []

    for score in gpt_scores:
        structure = score.get('structure_score')
        entity = score.get('entity_score')
        topic = score.get('topic_score')
        rouge1 = score.get('rouge1')
        rouge2 = score.get('rouge2')
        rougeL = score.get('rougeL')
        bertscore_f1 = score.get('bertscore_f1')
        if structure is not None:
            _structure_scores.append(structure)
        if entity is not None:
            _entity_scores.append(entity)
        if topic is not None:
            _topic_scores.append(topic)
        if rouge1 is not None:
            _rouge1_scores.append(rouge1)
        if rouge2 is not None:
            _rouge2_scores.append(rouge2)
        if rougeL is not None:
            _rougeL_scores.append(rougeL)
        if bertscore_f1 is not None:
            _bertscore_f1_scores.append(bertscore_f1)

    gpt_scores = {
        'structure': _structure_scores,
        'entity': _entity_scores,
        'topic': _topic_scores,
        'rouge1': _rouge1_scores,
        'rouge2': _rouge2_scores,
        'rougeL': _rougeL_scores,
        'bertscore_f1': _bertscore_f1_scores
    }

    return gpt_scores
gpt_scores = get_gpt_scores()

metrics = {}
_metrics = {
    'loss': 'loss', 
    'perlexity': 'perplexity', 
    'rouge1': 'ROUGE-1', 
    'rouge2': 'ROUGE-2', 
    'rougeL': 'ROUGE-L', 
    'bertscore_f1': 'BERTScore F1', 
    'token_f1': 'Token F1',
    'bleu': 'BLEU',
    'structure': 'Structure', 
    'entity': 'Entity', 
    'topic': 'Topic'
}

metrics['loss'] = data['loss']
metrics['perplexity'] = data['perplexity']

def _mean_std(values):
    if not values:
        return {"mean": None, "std": None, "count": 0}
    flat = []
    for v in values:
        if isinstance(v, list):
            flat.extend(v)
        else:
            flat.append(v)
    if not flat:
        return {"mean": None, "std": None, "count": 0}
    mean = sum(flat) / len(flat)
    var = sum((v - mean) ** 2 for v in flat) / len(flat)
    return {"mean": mean, "std": math.sqrt(var), "count": len(flat)}

for metric in _metrics.keys():
    if metric in data.keys():
        metric_name = _metrics[metric]
        if isinstance(data[metric], list):
            metrics[metric_name] = _mean_std(data[metric])
    elif metric in gpt_scores.keys():
        metric_name = _metrics[metric]
        _scores = []
        for score in gpt_scores[metric]:
            if score is not None:
                _scores.append(score)
        metrics[metric_name] = _mean_std(_scores)

write_json(metrics, get_metrics_sum_path(split="val", model_type=args.model_type))