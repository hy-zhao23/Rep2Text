#!/usr/bin/env python
"""
Score the OOD clinical summary texts with all metrics:
  - ROUGE-1, ROUGE-2, ROUGE-L
  - BERTScore F1
  - GPT: structure, entity, topic  (domain="clinical")

Input:  {result_dir}/ood_clinical_summary_texts.csv  (idx, summary, gt_text, gen_text)
Output: {result_dir}/ood_clinical_summary_scores.json  (per-sample scores)
        {result_dir}/ood_clinical_summary_scores.csv   (wide-format CSV)

Usage:
    export EINV_CONFIG=transfer/q14b-32token.yaml
    python experiments/test/score_clinical_summary.py
"""

import os
import csv
import sys
import math
from pathlib import Path
from tqdm import tqdm

os.environ.setdefault("PROJ", str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("EINV_CONFIG", "transfer/q14b-32token.yaml")

PROJ_ROOT = Path(os.getenv("PROJ") or Path(__file__).resolve().parents[2])
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from utils import p_args, logger
from utils.core.files import read_json, write_json

# Two-phase mode: SKIP_GPT runs ROUGE+BERTScore (needs GPU), ONLY_GPT runs
# only the GPT API calls (CPU-only). Default = both in one job.
SKIP_GPT = os.getenv("OOD_SKIP_GPT", "0") == "1"
ONLY_GPT = os.getenv("OOD_ONLY_GPT", "0") == "1"
assert not (SKIP_GPT and ONLY_GPT), "OOD_SKIP_GPT and OOD_ONLY_GPT are mutually exclusive"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
result_dir  = p_args.path.result_dir          # results/mistral7B/l10t32p32
texts_csv   = result_dir / "ood_clinical_summary_texts.csv"
local_json  = result_dir / "ood_clinical_summary_local_scores.json"  # phase-1 intermediate
scores_json = result_dir / "ood_clinical_summary_scores.json"
scores_csv  = result_dir / "ood_clinical_summary_scores.csv"

TRAIN_TAG = os.getenv("TRAIN_TAG", "N/A")

print(f"\n{'='*60}")
print(f"  OOD Clinical Summary — Scoring")
print(f"{'='*60}")
print(f"  config     : {os.getenv('EINV_CONFIG', 'N/A')}")
print(f"  train_tag  : {TRAIN_TAG}")
print(f"  emb_model  : {p_args.model.emb_id}")
print(f"  dec_model  : {p_args.model.dec_id}")
print(f"  input      : {texts_csv}")
print(f"  out json   : {scores_json}")
print(f"  out csv    : {scores_csv}")
print()

# ---------------------------------------------------------------------------
# Phase 1 (GPU) — ROUGE + BERTScore  → write local_scores.json
# Phase 2 (CPU) — load local_scores + run GPT API → write final scores.{json,csv}
# Default (no env flag): both phases in one job
# ---------------------------------------------------------------------------

if not ONLY_GPT:
    from utils.eval.scores import compute_rouge_score, compute_bertscore

    print("[1/2] Phase 1: ROUGE + BERTScore (loading texts) ...")
    rows = []
    with open(texts_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    print(f"      {len(rows)} samples from {texts_csv.name}")

    results = []
    for row in tqdm(rows, desc="  ROUGE+BERT", ncols=80):
        gt_text  = row["gt_text"]
        gen_text = row["gen_text"]
        sample = {"idx": row["idx"], "gt_text": gt_text, "gen_text": gen_text}

        rouge = compute_rouge_score(gt_text, gen_text)
        sample["rouge1"]  = rouge["rouge1"][0]
        sample["rouge2"]  = rouge["rouge2"][0]
        sample["rougeL"]  = rouge["rougeL"][0]

        bert = compute_bertscore(gt_text, gen_text)
        sample["bertscore_f1"] = bert["f1"]

        results.append(sample)

    write_json(results, str(local_json))
    print(f"      💾 phase-1 → {local_json}")

    if SKIP_GPT:
        print("\n  OOD_SKIP_GPT=1: stopping after phase 1.")
        sys.exit(0)
else:
    print("[1/2] Phase 1: skipped (OOD_ONLY_GPT=1) — loading prior local_scores.json ...")
    results = read_json(str(local_json))
    if results is None:
        sys.exit(f"ERROR: {local_json} not found. Run phase 1 first (OOD_SKIP_GPT=1).")
    print(f"      {len(results)} samples loaded from {local_json.name}")

# ---------------------------------------------------------------------------
# Phase 2 — GPT scoring (CPU, API-bound)
# ---------------------------------------------------------------------------
from utils.eval import GPTScorer

print(f"\n[2/2] Phase 2: GPT scoring on {len(results)} samples ...")
for sample in tqdm(results, desc="  GPT", ncols=80):
    scorer = GPTScorer(sample["gt_text"], sample["gen_text"], domain="clinical")
    scorer.gpt_rating()
    sample["structure_score"] = scorer.structure_score
    sample["entity_score"]    = scorer.entity_score
    sample["topic_score"]     = scorer.topic_score

# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------
print(f"\nSaving results ...")

write_json(results, str(scores_json))
print(f"      JSON  → {scores_json}")

_fields = [
    "idx", "gt_text", "gen_text",
    "rouge1", "rouge2", "rougeL",
    "bertscore_f1",
    "structure_score", "entity_score", "topic_score",
]
with open(scores_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=_fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(results)
print(f"      CSV   → {scores_csv}")

# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def _mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return float("nan"), float("nan"), 0
    mean = sum(vals) / len(vals)
    std  = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
    return mean, std, len(vals)

_metric_labels = [
    ("rouge1",          "ROUGE-1"),
    ("rouge2",          "ROUGE-2"),
    ("rougeL",          "ROUGE-L"),
    ("bertscore_f1",    "BERTScore F1"),
    ("structure_score", "Structure (GPT)"),
    ("entity_score",    "Entity    (GPT)"),
    ("topic_score",     "Topic     (GPT)"),
]

print(f"\n{'='*60}")
print(f"  RESULTS  (n={len(results)}, split='ood_clinical_summary')")
print(f"{'='*60}")
for key, label in _metric_labels:
    mean, std, n = _mean_std([r.get(key) for r in results])
    print(f"  {label:<20s}: {mean:.4f}  ± {std:.4f}  (n={n})")

print(f"\n  [Sample outputs]")
for i in range(min(5, len(results))):
    gt  = results[i]["gt_text"][:90].replace("\n", " ")
    gen = results[i]["gen_text"][:90].replace("\n", " ")
    print(f"\n  [{i}] GT  : {gt}")
    print(f"       Gen : {gen}")
    print(f"       R-1={results[i]['rouge1']:.3f}  R-L={results[i]['rougeL']:.3f}"
          f"  BERT={results[i]['bertscore_f1']:.3f}"
          f"  Struct={results[i]['structure_score']}  Ent={results[i]['entity_score']}  Topic={results[i]['topic_score']}")
print()
