"""Filter clinical_notes_summary.json to 1000 entries whose EMB-model token
length is <= max_len (32), then save the result.

Output: datasets/wikipedia/clinical_notes/clinical_notes_filtered.json
        Each record preserves all original fields and adds:
          - emb_token_len : int   (token count under the emb tokenizer)
"""

import os
import sys
import json
import random
import numpy as np
from pathlib import Path

os.environ.setdefault("PROJ", str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("EINV_CONFIG", "transfer/q14b-32token.yaml")

PROJ_ROOT = Path(os.getenv("PROJ") or Path(__file__).resolve().parents[2])
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from transformers import AutoTokenizer
from utils import p_args, logger
from utils.core.files import read_json, write_json

FILTER_SEED = 42
TARGET = 1000


def main():
    emb_id  = p_args.model.emb_id or p_args.model.dec_id
    max_len = p_args.data.max_len

    summary_path  = p_args.path.data_dir / "clinical_notes" / "clinical_notes_summary.json"
    filtered_path = p_args.path.data_dir / "clinical_notes" / "clinical_notes_filtered.json"

    # ── Load source summaries ─────────────────────────────────────────────
    rows = read_json(str(summary_path))
    if rows is None:
        raise FileNotFoundError(f"Not found or empty: {summary_path}")

    valid = [r for r in rows if isinstance(r.get("summary"), str) and r["summary"].strip()]
    logger.info(f"Source records: total={len(rows)}  valid_summary={len(valid)}")

    # ── Tokenize with EMB model ───────────────────────────────────────────
    tok = AutoTokenizer.from_pretrained(
        emb_id,
        cache_dir=str(p_args.path.weight_dir),
        token=p_args.settings.hf_token,
        use_fast=True,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    summaries = [r["summary"].strip() for r in valid]
    emb_lens  = [len(tok(s, add_special_tokens=False)["input_ids"]) for s in summaries]

    # ── Token-length stats (before filter) ───────────────────────────────
    arr = np.array(emb_lens)
    within = int((arr <= max_len).sum())
    logger.info(
        f"[EMB token stats | {emb_id.split('/')[-1]}] "
        f"n={len(arr)}  min={arr.min()}  max={arr.max()}  "
        f"mean={arr.mean():.1f}  median={float(np.median(arr)):.1f}  "
        f"<={max_len}: {within}/{len(arr)} ({100*within/len(arr):.1f}%)"
    )

    # ── Filter: keep only entries with emb token len <= max_len ──────────
    candidates = [
        {**row, "emb_token_len": l}
        for row, l in zip(valid, emb_lens)
        if l <= max_len
    ]
    logger.info(f"Candidates after filter: {len(candidates)}")

    if len(candidates) < TARGET:
        logger.warning(
            f"Only {len(candidates)} candidates meet emb<={max_len}. "
            f"Consider regenerating clinical_notes_summary.json with more source notes."
        )

    # ── Random sample TARGET entries ──────────────────────────────────────
    rng = random.Random(FILTER_SEED)
    selected = rng.sample(candidates, min(TARGET, len(candidates)))

    sel_lens = np.array([r["emb_token_len"] for r in selected])
    logger.info(
        f"[Selected stats] n={len(selected)}  "
        f"min={sel_lens.min()}  max={sel_lens.max()}  "
        f"mean={sel_lens.mean():.1f}  median={float(np.median(sel_lens)):.1f}"
    )

    # ── Save ──────────────────────────────────────────────────────────────
    write_json(selected, str(filtered_path))

    print("=== Filter Done ===")
    print(f"emb_model        = {emb_id}")
    print(f"max_len          = {max_len}")
    print(f"source_total     = {len(rows)}")
    print(f"valid_summary    = {len(valid)}")
    print(f"candidates       = {len(candidates)}")
    print(f"selected         = {len(selected)}")
    print(f"token min/max    = {sel_lens.min()} / {sel_lens.max()}")
    print(f"token mean       = {sel_lens.mean():.1f}")
    print(f"token median     = {float(np.median(sel_lens)):.1f}")
    print(f"output           = {filtered_path}")


if __name__ == "__main__":
    main()
