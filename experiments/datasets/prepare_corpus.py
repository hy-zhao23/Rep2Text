"""Build datasets/all_texts.pkl in a single process, before any multi-rank job.

tokenize_sequences.py used to do this inline on rank 0 while the other 29 ranks
sat at a barrier — 29 idle nodes billed for the whole download, and HF's file
locks did not actually serialize it: the lock name embeds the absolute path, and
/path/... vs /real/path/... hash to different lock files, so two ranks
both ran download_and_prepare and one deleted the other's .incomplete dir.

Running it standalone removes both problems. Idempotent — re-running with the
pkl already present is a no-op.
"""
import gc
import sys
from pathlib import Path

from utils import p_args, logger
from utils.core import write_pkl
from utils.data.preprocess import download_wikipedia


def combine_texts(dataset) -> list:
    all_texts = []
    # train then test, matching the original ordering in tokenize_sequences.py —
    # the token sampling is seeded, so a different order means a different 2M
    # subset and the scaling curve stops being comparable.
    for split in ("train", "test"):
        if split in dataset:
            all_texts.extend(dataset[split]["text"])
            logger.info(f"  {split}: {len(dataset[split])} rows -> running total {len(all_texts)}")
    return all_texts


def main() -> int:
    out = Path(p_args.path.data_dir) / "all_texts.pkl"
    if out.exists():
        logger.info(f"✅ {out} already exists ({out.stat().st_size / 1e9:.1f} GB), nothing to do.")
        return 0

    logger.info("🔄 Loading corpus (cached locally; no download expected)...")
    dataset = download_wikipedia()
    logger.info(f"✅ Corpus loaded: { {k: len(v) for k, v in dataset.items()} }")

    all_texts = combine_texts(dataset)
    del dataset
    gc.collect()

    logger.info(f"🔄 Writing {len(all_texts)} texts to {out} ...")
    write_pkl(all_texts, out)
    logger.info(f"✅ Wrote {out} ({out.stat().st_size / 1e9:.1f} GB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
