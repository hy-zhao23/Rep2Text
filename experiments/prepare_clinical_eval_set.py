#!/usr/bin/env python3
"""
Filter clinical_notes_summary.json to entries whose **T5 token length** ≤ 32,
take the first N (default 1000), and save to a fixed eval file.

Usage
-----
    python test/experiments/prepare_clinical_eval_set.py          # defaults
    python test/experiments/prepare_clinical_eval_set.py --n 500 --max_t5_len 40

Output
------
    test/data/clinical_notes_eval1k.json
    (list of {"summary": ..., "t5_len": ..., "mistral_len": ...})
"""

import argparse
import json
import os
import sys
from pathlib import Path

import sentencepiece as spm
from tqdm import tqdm

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent           # test/experiments/
VEC2TEXT_DIR = SCRIPT_DIR.parents[1]                    # scripts/vec2text/
PROJ         = VEC2TEXT_DIR.parents[1]                  # embedding/

DATA_PATH = (
    PROJ / "datasets" / "wikipedia" / "clinical_notes" / "clinical_notes_summary.json"
)
SPIECE_MODEL = (
    VEC2TEXT_DIR / "test" / "results"
    / "mistral7b_l10t32_20260303_154531" / "spiece.model"
)
CLINICAL_DIR = PROJ / "datasets" / "wikipedia" / "clinical_notes"
OUT_FILE     = CLINICAL_DIR / "clinical_notes_eval1k.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare fixed clinical eval set")
    parser.add_argument("--n",           type=int, default=1000,
                        help="Number of samples to keep (default 1000)")
    parser.add_argument("--max_t5_len",  type=int, default=32,
                        help="Max T5 token length threshold (default 32)")
    parser.add_argument("--field",       default="summary",
                        choices=["note", "summary"],
                        help="Field to read from JSON (default: summary)")
    parser.add_argument("--out",         default=str(OUT_FILE),
                        help=f"Output path (default: {OUT_FILE})")
    args = parser.parse_args()

    # ── 1. load data ──────────────────────────────────────────────────────────
    print(f"Loading {DATA_PATH} ...")
    with open(DATA_PATH) as f:
        rows = json.load(f)

    texts = [
        r[args.field].strip()
        for r in rows
        if isinstance(r.get(args.field), str) and r[args.field].strip()
    ]
    print(f"  {len(texts)} valid '{args.field}' entries found")

    # ── 2. compute T5 token lengths via sentencepiece ─────────────────────────
    print(f"Computing T5 token lengths (spiece model: {SPIECE_MODEL}) ...")
    sp = spm.SentencePieceProcessor()
    sp.Load(str(SPIECE_MODEL))

    t5_lens = [len(sp.Encode(t)) for t in tqdm(texts, desc="  T5-tokenising", ncols=80)]

    # ── 3. filter ─────────────────────────────────────────────────────────────
    passed = [(t, l) for t, l in zip(texts, t5_lens) if l <= args.max_t5_len]
    print(f"  {len(passed)} / {len(texts)} pass T5 tok ≤ {args.max_t5_len}")

    if len(passed) < args.n:
        print(
            f"WARNING: only {len(passed)} samples pass the filter "
            f"(requested {args.n}); saving all of them."
        )
    selected = passed[: args.n]
    print(f"  Keeping {len(selected)} samples")

    # ── 4. save ───────────────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)  # datasets dir already exists

    records = [
        {"summary": text, "t5_len": t5_len}
        for text, t5_len in selected
    ]
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(records)} records → {out_path}")
    print(f"  T5 len: min={min(r['t5_len'] for r in records)}"
          f"  max={max(r['t5_len'] for r in records)}"
          f"  mean={sum(r['t5_len'] for r in records)/len(records):.1f}")


if __name__ == "__main__":
    main()
