'''
This file defines all the scores for the evaluation.
'''

from bert_score import score
from utils.core.gpu_distribute import gh

#===============================================
# BERT SCORE
#===============================================

def compute_bertscore(gt_text, greedy_text, model_type='microsoft/deberta-v3-large'):
    # Normalize to list inputs for bert_score
    if isinstance(gt_text, str):
        gt_text = [gt_text]
    if isinstance(greedy_text, str):
        greedy_text = [greedy_text]
    P, R, F1 = score(
        gt_text,
        greedy_text,
        lang='en',
        model_type=model_type,
        verbose=False,
        device=gh.device,
    )
    return {'precision': P.tolist()[0], 'recall': R.tolist()[0], 'f1': F1.tolist()[0]}


def compute_bertscore_batch(
    gt_texts,
    greedy_texts,
    model_type='microsoft/deberta-v3-large',
):
    if not gt_texts:
        return [], [], []
    P, R, F1 = score(
        gt_texts,
        greedy_texts,
        lang='en',
        model_type=model_type,
        verbose=False,
        device=gh.device,
    )
    return P.tolist(), R.tolist(), F1.tolist()


#===============================================
# TOKEN F1
#===============================================
from collections import Counter

def compute_token_f1(gt_ids, greedy_ids):
    """Compute token-level F1 based on token ID overlap (SQuAD-style)."""
    gt_counts = Counter(gt_ids)
    pred_counts = Counter(greedy_ids)
    common = sum((gt_counts & pred_counts).values())
    if common == 0:
        return 0.0
    precision = common / len(greedy_ids)
    recall = common / len(gt_ids)
    return 2 * precision * recall / (precision + recall)


#===============================================
# BLEU SCORE
#===============================================
import sacrebleu as _sacrebleu

def compute_bleu(gt_text, greedy_text):
    """Sentence-level BLEU using sacrebleu (0-100 scale, normalised to 0-1)."""
    return _sacrebleu.sentence_bleu(greedy_text, [gt_text]).score / 100.0


#===============================================
# ROUGE SCORE
#===============================================
from rouge_score import rouge_scorer

def compute_rouge_score(gt_text, greedy_text, metrics=None):
    if metrics is None:
        metrics = ['rouge1', 'rouge2', 'rougeL']
    # Initialize scorer with the metrics you want
    scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)
    rouge_scores = {metric: [] for metric in metrics}

    scores = scorer.score(gt_text, greedy_text)
    for metric in metrics:
        rouge_scores[metric].append(scores[metric].fmeasure)

    return rouge_scores
