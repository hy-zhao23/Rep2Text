"""Compute non-GPT metrics (token-F1, BLEU, ROUGE-1/2/L) for the m3b reverse eval.

Inputs:
  results/llama3B/l10t16p16/full_sentence/m3b_reverse/orig_decoded.json
  results/llama3B/l10t16p16/full_sentence/m3b_reverse/reversed_decoded.json
  results/llama3B/l10t16p16/full_sentence/m3b_reverse/comparison.json

Two splits:
  - orig:        pred(orig_emb)     vs orig gold
  - reversed:    pred(rev_emb)      vs reversed gold     (in-distribution-direction match)
  - rev_vs_orig: pred(rev_emb)      vs orig gold         (semantic-recovery check, ignoring direction)

Output: results/llama3B/l10t16p16/full_sentence/m3b_reverse/metrics_summary.json
"""
import json
import os
from pathlib import Path
import numpy as np

os.environ.setdefault("EINV_CONFIG", "transfer/m3b.yaml")
os.environ.setdefault("TRAIN_TAG", "transfer-m3b")

from utils.eval.scores import compute_rouge_score, compute_bleu, compute_token_f1  # noqa: E402

PROJ = Path(__file__).resolve().parents[2]
OUT_DIR = PROJ / "results/llama3B/l10t16p16/full_sentence/m3b_reverse"


def aggregate(items, key):
    vals = [it[key] for it in items if key in it]
    if not vals:
        return None
    # rouge values come as length-1 lists from compute_rouge_score
    flat = [v[0] if isinstance(v, list) else v for v in vals]
    return {
        "mean": float(np.mean(flat)),
        "median": float(np.median(flat)),
        "n": len(flat),
    }


def score_split(name, decoded, override_gt_text=None, override_gt_ids=None):
    """decoded: list of {gt_text, greedy_text, gt_ids, greedy_ids}.
       override_gt_text/ids: same length list to replace gt for re-comparison."""
    out = []
    for i, item in enumerate(decoded):
        gt_text = override_gt_text[i] if override_gt_text is not None else item["gt_text"]
        gt_ids = override_gt_ids[i] if override_gt_ids is not None else item.get("gt_ids")
        pr_text = item["greedy_text"]
        pr_ids = item.get("greedy_ids")
        rec = {"idx": i, "gt_text": gt_text, "pred_text": pr_text}
        rec.update(compute_rouge_score(gt_text, pr_text))
        rec["bleu"] = compute_bleu(gt_text, pr_text)
        if gt_ids is not None and pr_ids is not None:
            rec["token_f1"] = compute_token_f1(gt_ids, pr_ids)
        out.append(rec)
    summary = {
        "split": name,
        "n": len(out),
        "rouge1": aggregate(out, "rouge1"),
        "rouge2": aggregate(out, "rouge2"),
        "rougeL": aggregate(out, "rougeL"),
        "bleu": aggregate(out, "bleu"),
        "token_f1": aggregate(out, "token_f1"),
    }
    return summary, out


def main():
    with open(OUT_DIR / "orig_decoded.json") as f:
        orig = json.load(f)
    with open(OUT_DIR / "reversed_decoded.json") as f:
        rev = json.load(f)

    # For rev_vs_orig: compare reversed-emb prediction against the ORIGINAL gold.
    # gt_text/ids of the orig split are exactly the original gold.
    orig_gt_texts = [it["gt_text"] for it in orig]
    orig_gt_ids = [it["gt_ids"] for it in orig]

    splits = []
    s_orig, _ = score_split("orig (pred_orig_emb vs gold_orig)", orig)
    splits.append(s_orig)
    s_rev, _ = score_split("reversed (pred_rev_emb vs gold_reversed)", rev)
    splits.append(s_rev)
    s_rvso, _ = score_split(
        "rev_vs_orig (pred_rev_emb vs gold_orig)",
        rev,
        override_gt_text=orig_gt_texts,
        override_gt_ids=orig_gt_ids,
    )
    splits.append(s_rvso)

    summary = {"splits": splits}
    out_path = OUT_DIR / "metrics_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
