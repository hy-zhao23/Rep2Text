"""Run run_eval on the first N samples of a split with a custom batch size.

Forces log_every=1 so every batch is greedy-decoded (otherwise text metrics
on a tiny subset would be near-empty). Saves under a custom split label so
output files don't collide with full-split eval (e.g., test1000 vs test).
"""
import argparse
import yaml
import torch
from torch.utils.data import DataLoader, Subset

from utils.eval import run_eval
from utils.models import load_pretrained_model
from utils import logger, gh
from utils.data.builder import create_dataloader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model_type", default="einv")
    ap.add_argument("--eval", default="test", help="split to load: train/val/test")
    ap.add_argument("--n_samples", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--save_split_label", default=None,
                    help="filename prefix for outputs; defaults to '<split><N>'")
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    config["logging"]["log_every"] = 1  # decode every batch on small subset

    tokenizer, model, context_len = load_pretrained_model(
        model_type=args.model_type,
        resume_from=True,
        device_map=gh.device,
    )

    data_type = "finetune" if args.model_type == "einv_lora" else "pretrain"
    base_loader = create_dataloader(
        config, tokenizer, model.config.hidden_size, context_len,
        args.eval, data_type=data_type,
    )

    n_total = len(base_loader.dataset)
    n = min(args.n_samples, n_total)
    if n < args.n_samples:
        logger.warning(f"requested {args.n_samples} > available {n_total}; using {n}")

    sub_ds = Subset(base_loader.dataset, range(n))
    loader = DataLoader(
        sub_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=base_loader.num_workers,
        pin_memory=base_loader.pin_memory,
        collate_fn=base_loader.collate_fn,
        drop_last=False,
    )

    save_split = args.save_split_label or f"{args.eval}{n}"
    logger.info(f"Eval subset: {n} samples / split={args.eval} / "
                f"batch_size={args.batch_size} / save_label={save_split}")
    logger.info(f"Dataloader batches: {len(loader)}")

    run_eval(
        model, config, loader, tokenizer,
        model_type=args.model_type, split=save_split, save_results=True,
    )


if __name__ == "__main__":
    main()
