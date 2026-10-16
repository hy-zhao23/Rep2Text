import numpy as np
from typing import Dict, List, Any 
from .scores import compute_bertscore, compute_bertscore_batch, compute_rouge_score, compute_token_f1, compute_bleu
from utils import logger, gh
from utils.core.files import write_json

class EvaluationMetrics:
    """
    Comprehensive evaluation metrics calculator for language models.
    Supports loss, perplexity, ROUGE, BERTScore, and token-level accuracy.
    """
    
    def __init__(self, 
                get_rouge: bool = True,
                get_bertscore: bool = True,
                get_bleu: bool = True,
                get_meteor: bool = True,
                get_token_accuracy: bool = True,
                rouge_metrics: List[str] = ['rouge1', 'rouge2', 'rougeL'],
                bertscore_model: str = 'microsoft/deberta-v3-large',
                bertscore_batch_size: int = 256,
                bleu_max_ngram: int = 3,
                topk_values: List[int] = [1, 5, 10]):
        """
        Initialize the metrics calculator.
        
        Args:
            get_rouge: Whether to compute ROUGE scores
            get_bertscore: Whether to compute BERTScore
            get_bleu: Whether to compute BLEU scores
            get_meteor: Whether to compute METEOR scores
            get_token_accuracy: Whether to compute token-level accuracy
            rouge_metrics: List of ROUGE metrics to compute ['rouge1', 'rouge2', 'rougeL']
            bertscore_model: BERTScore model to use
            bleu_weights: Weights for BLEU n-grams [(1,0,0,0), (0.5,0.5,0,0), etc.]
            topk_values: List of k values for top-k accuracy [1, 5, 10]
        """
        self.get_rouge = get_rouge
        self.get_bertscore = get_bertscore
        self.get_bleu = get_bleu
        self.get_meteor = get_meteor
        self.get_token_accuracy = get_token_accuracy
        
        self.rouge_metrics = rouge_metrics or ['rouge1', 'rouge2', 'rougeL']
        self.bertscore_model = bertscore_model
        self.bertscore_batch_size = bertscore_batch_size
        self.bleu_weights = [(1,0,0,0), (0.5,0.5,0,0), (0.33,0.33,0.33,0), (0.25,0.25,0.25,0.25)]
        self.bleu_max_ngram = bleu_max_ngram
        self.topk_values = topk_values or [1, 5, 10]
        
        self.reset()
    
    def reset(self):
        """Reset all accumulated data."""
        self.batch_losses = []
        self.batch_decoded = []
    
    def add_batch_results(self, batch_loss: float, batch_decoded: List[Dict] = None):
        self.batch_losses.append(batch_loss)
        if batch_decoded is not None:
            self.batch_decoded.extend(batch_decoded)
    
    def compute_loss_metrics(self) -> Dict[str, float]:
        """Compute loss-based metrics."""
        if not self.batch_losses:
            return {}
        
        avg_loss = np.mean(self.batch_losses)
        perplexity = np.exp(np.clip(avg_loss, a_min=None, a_max=20.0))

        logger.info(f"Avg Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")

        return {
            'loss': float(avg_loss),
            'perplexity': float(perplexity)
        }
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Compute all enabled metrics."""
        all_metrics = {}
        scores = []

        if self.batch_decoded:
            gt_texts = [item["gt_text"] for item in self.batch_decoded]
            greedy_texts = [item["greedy_text"] for item in self.batch_decoded]

            bert_f1_scores = []
            if self.get_bertscore:
                bs = max(int(self.bertscore_batch_size), 1)
                for i in range(0, len(gt_texts), bs):
                    _, _, f1 = compute_bertscore_batch(
                        gt_texts[i:i + bs],
                        greedy_texts[i:i + bs],
                        model_type=self.bertscore_model,
                    )
                    bert_f1_scores.extend(f1)

            for idx, item in enumerate(self.batch_decoded):
                rouge_score = compute_rouge_score(item["gt_text"], item["greedy_text"])
                item.update({
                    "rouge1": rouge_score["rouge1"],
                    "rouge2": rouge_score["rouge2"],
                    "rougeL": rouge_score["rougeL"],
                })
                if self.get_bertscore and idx < len(bert_f1_scores):
                    item["bertscore_f1"] = bert_f1_scores[idx]
                if "gt_ids" in item and "greedy_ids" in item:
                    item["token_f1"] = compute_token_f1(item["gt_ids"], item["greedy_ids"])
                item["bleu"] = compute_bleu(item["gt_text"], item["greedy_text"])
                scores.append(item)
        
        # Loss-based metrics
        all_metrics.update(self.compute_loss_metrics())
        all_metrics["scores"] = scores

        # Aggregate token_f1 as a list for summary.py compatibility
        token_f1_values = [item["token_f1"] for item in scores if "token_f1" in item]
        if token_f1_values:
            all_metrics["token_f1"] = token_f1_values

        # Aggregate bleu as a list for summary.py compatibility
        bleu_values = [item["bleu"] for item in scores if "bleu" in item]
        if bleu_values:
            all_metrics["bleu"] = bleu_values

        return all_metrics
    
    def save_metrics(self, file):
        """Save computed metrics to a file."""
        if not gh.is_main:
            logger.error("! Empty metrics, nothing to save.")
            return
        write_json(self.get_metrics_summary(), file)