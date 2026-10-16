#!/usr/bin/env python
"""Merge sharded outputs from a SLURM-array run of ood_clinical_summary.py.

Reads `ood_clinical_summary_{pairs,texts}.rank{R}of{W}.{json,npz,csv}` and
writes the canonical (no-suffix) files. Sorts everything by `idx` so the
output is identical to a single-process run.
"""
import csv
import os
import sys
from pathlib import Path

import numpy as np

PROJ_ROOT = Path(os.getenv("PROJ") or Path(__file__).resolve().parents[2])
if str(PROJ_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJ_ROOT))

os.environ.setdefault("EINV_CONFIG", "transfer/l8b-to-q14b-32token.yaml")

from utils import p_args, logger
from utils.core.files import read_json, write_json


def main():
	world_size = int(os.getenv("OOD_WORLD_SIZE", "0"))
	if world_size <= 1:
		sys.exit("Set OOD_WORLD_SIZE to the array size used during generation (>=2).")

	out_dir = p_args.path.result_dir
	missing = []
	for r in range(world_size):
		for ext in ("pairs.json", "texts.json", "pairs.npz", "texts.csv"):
			f = out_dir / f"ood_clinical_summary_{ext.split('.')[0]}.rank{r}of{world_size}.{ext.split('.')[1]}"
			if not f.exists():
				missing.append(str(f))
	if missing:
		sys.exit("Missing shard files:\n  " + "\n  ".join(missing))

	# ── Merge JSON pairs / texts ─────────────────────────────────────────
	all_pairs, all_texts = [], []
	for r in range(world_size):
		all_pairs.extend(read_json(str(out_dir / f"ood_clinical_summary_pairs.rank{r}of{world_size}.json")))
		all_texts.extend(read_json(str(out_dir / f"ood_clinical_summary_texts.rank{r}of{world_size}.json")))
	all_pairs.sort(key=lambda x: x["idx"])
	all_texts.sort(key=lambda x: x["idx"])

	# Sanity check: idx coverage
	idx_set = {row["idx"] for row in all_pairs}
	expected = set(range(len(all_pairs)))
	if idx_set != expected:
		dup = len(all_pairs) - len(idx_set)
		gaps = sorted(expected - idx_set)[:10]
		logger.warning(f"idx mismatch: {dup} duplicates, first missing: {gaps}")

	pairs_json = out_dir / "ood_clinical_summary_pairs.json"
	texts_json = out_dir / "ood_clinical_summary_texts.json"
	write_json(all_pairs, str(pairs_json))
	write_json(all_texts, str(texts_json))

	# ── Merge NPZ ────────────────────────────────────────────────────────
	all_indices, all_dec_ids, all_embs = [], [], []
	for r in range(world_size):
		z = np.load(out_dir / f"ood_clinical_summary_pairs.rank{r}of{world_size}.npz", allow_pickle=True)
		all_indices.append(z["shard_indices"])
		all_dec_ids.append(z["dec_input_ids"])
		all_embs.append(z["emb_last_token"])
	indices = np.concatenate(all_indices)
	dec_ids = np.concatenate(all_dec_ids)
	embs = np.concatenate(all_embs, axis=0)
	order = np.argsort(indices, kind="stable")
	pairs_npz = out_dir / "ood_clinical_summary_pairs.npz"
	np.savez(pairs_npz, dec_input_ids=dec_ids[order], emb_last_token=embs[order])

	# ── Merge CSV ────────────────────────────────────────────────────────
	rows = [{"idx": r["idx"], "summary": r["summary"], "gt_text": r["gt_text"], "gen_text": r["gen_text"]} for r in all_texts]
	texts_csv = out_dir / "ood_clinical_summary_texts.csv"
	with open(texts_csv, "w", newline="", encoding="utf-8") as f:
		w = csv.writer(f)
		w.writerow(["idx", "summary", "gt_text", "gen_text"])
		for r in rows:
			w.writerow([r["idx"], r["summary"], r["gt_text"], r["gen_text"]])

	logger.info(f"✅ Merged {world_size} shards → {len(all_pairs)} samples")
	logger.info(f"   {pairs_json}")
	logger.info(f"   {texts_json}")
	logger.info(f"   {pairs_npz}")
	logger.info(f"   {texts_csv}")


if __name__ == "__main__":
	main()
