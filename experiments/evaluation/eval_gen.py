"""
Example usage of the new evaluation system with EvaluationMetrics class.

This demonstrates how to use the refactored evaluation system that supports
multiple metrics including ROUGE, BLEU, METEOR, BERTScore, and token accuracy.
"""

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from utils.eval import run_eval
from utils.models import load_pretrained_model
from utils import logger, gh
import argparse
from utils.data.builder import create_dataloader


def _subsample(data_loader, n, seed):
    """Cut the eval set down to n randomly chosen samples.

    Without this, how many samples the text metrics (BLEU/ROUGE/BERTScore/token_f1)
    actually cover was decided by the logging cadence: evaluation.py only decodes
    every log_every-th batch, so the count drifted with batch_size and val size
    across experiments (4000 / 4032 / 4096 / 4336 / 80000 all appear in results/).
    Sampling an explicit n here makes the eval size a stated choice, and every
    metric -- loss included -- is then computed over the same n samples.
    """
    dataset = data_loader.dataset
    total = len(dataset)
    if n is None or n >= total:
        logger.info("📏 Eval set: using all %d samples", total)
        return data_loader

    idx = np.random.default_rng(seed).choice(total, size=n, replace=False)
    idx.sort()  # keep file/rank locality; the split indices are already shuffled
    logger.info("📏 Eval set: %d of %d samples (random, seed=%d)", n, total, seed)
    return DataLoader(
        Subset(dataset, idx.tolist()),
        batch_size=data_loader.batch_size,
        shuffle=False,
        collate_fn=data_loader.collate_fn,
        num_workers=data_loader.num_workers,
        pin_memory=data_loader.pin_memory,
    )

def main():
    parser = argparse.ArgumentParser(description="Evaluate a pretrained model with custom metrics.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--model_type", type=str, default="einv", required=False, help="einv or einv_lora")
    parser.add_argument("--eval", type=str, default="val", required=False, help="Dataset split to evaluate on")
    parser.add_argument("--max_eval_samples", type=int, default=1000,
                        help="Randomly sample this many eval examples (None/0 = whole split)")
    parser.add_argument("--eval_seed", type=int, default=42,
                        help="Seed for the eval subsample, so every run scores the same examples")
    parser.add_argument("--device_map", type=str, default=None,
                        help='Decoder placement. Default is the local GPU. Pass "auto" to spread '
                             'layers over every visible GPU — 70B+ does not fit on one 80G card.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    multi_gpu = args.device_map == "auto"
    tokenizer, model, context_len = load_pretrained_model(
        model_type=args.model_type,
        resume_from=True, # Set False for training
        device_map=args.device_map or gh.device,
        # configure_kwargs only honors "auto" when device says cuda; otherwise it
        # would silently fall through and leave the model on CPU.
        device="cuda" if multi_gpu else "cpu",
    )

    data_type = "pretrain"
    if args.model_type == "einv_lora":
        data_type = "finetune"

    data_loader = create_dataloader(config, tokenizer, model.config.hidden_size, context_len, args.eval, data_type=data_type)

    n = args.max_eval_samples or None
    data_loader = _subsample(data_loader, n, args.eval_seed)
    capped = n is not None

    logger.info(f"Starting evaluation on {args.eval} set for model {args.model_type}...")
    logger.info(f"Dataset size: {len(data_loader)} batches.")

    # decode_all: the set is already the size we asked for, so every batch is
    # scored rather than every log_every-th one.
    run_eval(model, config, data_loader, tokenizer, model_type=args.model_type, split=args.eval,
             save_results=True, decode_all=capped)


if __name__ == "__main__":
    main()