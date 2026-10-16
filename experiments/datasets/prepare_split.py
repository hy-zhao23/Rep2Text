"""Generate the train/val/test split indices ahead of a distributed run.

data_splitter builds these lazily on first use, guarded by `if gh.is_main` and a
barrier. That means rank 0 spends minutes scanning the embedding cache, shuffling
2M indices and writing ~30 MB of pkl while every other rank sits spinning inside a
NCCL collective -- a one-sided stall that has repeatedly wedged the communicator
and killed the job on the watchdog timeout 30 minutes later.

Running this single-process before `accelerate launch` removes the skew entirely:
by the time the ranks start, the pkl files exist and every rank takes the same
"already generated" path. It is a no-op when they are already there.

Usage:  EINV_CONFIG=tokens/setup128_3b.yaml python experiments/datasets/prepare_split.py
"""

import argparse

from utils import p_args, logger
from utils.core.path import pt_data_idx_path, ft_data_idx_path
from utils.data.data_splitter import split_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-generate split indices for a distributed run")
    parser.add_argument(
        "--data_type",
        default="pretrain",
        choices=["pretrain", "finetune"],
        help="Which split to materialise (finetune.py uses 'finetune')",
    )
    args = parser.parse_args()

    logger.info(
        "🧩 Preparing %s split | emb_dir=%s | max_len=%s",
        args.data_type,
        p_args.path.embedding_dir,
        p_args.data.max_len,
    )

    train_idx, val_idx, test_idx = split_data(data_type=args.data_type)
    n = lambda d: sum(len(v) for v in d.values())
    logger.info(
        "✅ Split ready: train=%d val=%d test=%d", n(train_idx), n(val_idx), n(test_idx)
    )
    logger.info("   %s", pt_data_idx_path)
    logger.info("   %s", ft_data_idx_path)


if __name__ == "__main__":
    main()
