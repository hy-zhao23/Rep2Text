#!/usr/bin/env python
"""Generate train_index pickles for the 3B data-size sweep.

Strategy: flatten the baseline 640K train indices in shard order, shuffle
once with a fixed seed, take the first N for each sweep size. Smaller N is
a strict subset of larger N (nested), so the comparison only varies in
"how much data the model saw" — the val/test sets are identical across
sizes (we copy baseline val/test pickles to each size's filename).

Files written (per size N):
  pretrain_index_{N}_l16_o0.pkl       -- copy of baseline (pool definition)
  finetune_index_{N}_l16_o0.pkl       -- copy of baseline
  pretrain_train_index_{N}_l16_o0.pkl -- first N from shuffled flat list
  pretrain_val_index_{N}_l16_o0.pkl   -- copy of baseline (shared val)
  pretrain_test_index_{N}_l16_o0.pkl  -- copy of baseline (shared test)
"""
import pickle
import random
import shutil
from pathlib import Path

DATASETS = Path(__file__).resolve().parents[2] / "datasets" / "wikipedia"
BASELINE_TRAIN_SIZE = 2000000   # current m3b config -> 640K train pool
LABEL = "l16_o0"                # max_len=16, overlap_len=0
SIZES = [10000, 50000, 100000, 320000]
SEED = 42

src = {
    "pretrain":       DATASETS / f"pretrain_index_{BASELINE_TRAIN_SIZE}_{LABEL}.pkl",
    "finetune":       DATASETS / f"finetune_index_{BASELINE_TRAIN_SIZE}_{LABEL}.pkl",
    "pretrain_train": DATASETS / f"pretrain_train_index_{BASELINE_TRAIN_SIZE}_{LABEL}.pkl",
    "pretrain_val":   DATASETS / f"pretrain_val_index_{BASELINE_TRAIN_SIZE}_{LABEL}.pkl",
    "pretrain_test":  DATASETS / f"pretrain_test_index_{BASELINE_TRAIN_SIZE}_{LABEL}.pkl",
}
for k, p in src.items():
    assert p.exists(), f"Missing baseline file: {p}"

with open(src["pretrain_train"], "rb") as f:
    baseline_train = pickle.load(f)

# Flatten in deterministic shard order, then shuffle once.
flat = [(shard, idx) for shard in sorted(baseline_train) for idx in baseline_train[shard]]
print(f"baseline train total: {len(flat):,} samples across {len(baseline_train)} shards")

random.seed(SEED)
random.shuffle(flat)

for N in SIZES:
    assert N <= len(flat), f"N={N} > baseline train size {len(flat)}"
    subset = flat[:N]
    grouped = {}
    for shard, idx in subset:
        grouped.setdefault(shard, []).append(idx)

    # Sort indices within each shard for deterministic ordering
    for shard in grouped:
        grouped[shard].sort()

    out_train = DATASETS / f"pretrain_train_index_{N}_{LABEL}.pkl"
    with open(out_train, "wb") as f:
        pickle.dump(grouped, f)

    # Shared val/test/pretrain/finetune across all sizes (copy baseline)
    shutil.copyfile(src["pretrain"],     DATASETS / f"pretrain_index_{N}_{LABEL}.pkl")
    shutil.copyfile(src["finetune"],     DATASETS / f"finetune_index_{N}_{LABEL}.pkl")
    shutil.copyfile(src["pretrain_val"], DATASETS / f"pretrain_val_index_{N}_{LABEL}.pkl")
    shutil.copyfile(src["pretrain_test"],DATASETS / f"pretrain_test_index_{N}_{LABEL}.pkl")

    n_shards = len(grouped)
    n_total = sum(len(v) for v in grouped.values())
    print(f"  N={N:>6}: train={n_total} across {n_shards} shards  -> {out_train.name}")

print("done.")
