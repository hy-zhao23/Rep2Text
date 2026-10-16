"""
Sample 250 pairs for human sanity evaluation.

Clinical (aligned): same 250 gt indices from both einv and vec2text.
Wiki (unaligned):   einv and v2t each independently sample 250 from their own
                    eval sets (different gt texts).

Output layout under results/human_eval/:
  clinical_sample250/
    meta.json
    metrics.json              (per-sample metrics, both models)
    batches/batch_01..10.json (gt + einv_pred + v2t_pred, 25 each)

  wiki_sample250/
    einv/
      meta.json
      metrics.json
      batches/batch_01..10.json
    v2t/
      meta.json
      metrics.json
      batches/batch_01..10.json
"""
import json
import random
import argparse
import os

# ---------- clinical paths ----------
CLIN_EINV_PATH   = "results/mistral7B/l10t32p32/ood_clinical_summary_texts.json"
CLIN_EINV_SCORES = "results/mistral7B/l10t32p32/ood_clinical_summary_scores.json"
CLIN_V2T_PATH    = "experiments/vec2text/test/results/mistral7b_l10t32_20260303_154531/eval/results_clinical_summary.json"
CLIN_OUT_DIR     = "results/human_eval/clinical_sample250"

# ---------- wiki paths ----------
WIKI_EINV_PATH   = "results/mistral7B/l10t32p32/transfer-q14b-32token-lr15/val_einv_decoded.json"
WIKI_V2T_PATH    = "experiments/vec2text/test/results/mistral7b_l10t32_20260303_154531/eval/results_wikipedia_val.json"
WIKI_OUT_DIR     = "results/human_eval/wiki_sample250"

EINV_MODEL = "mistral7B/l10t32p32/transfer-q14b-32token-lr15"
V2T_MODEL  = "mistral7b_l10t32 (vec2text, 50 steps)"


def save_batches(out_dir, meta, text_records, metric_records, batch_size):
    os.makedirs(out_dir, exist_ok=True)
    # meta.json
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    # metrics.json
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"meta": meta, "samples": metric_records}, f, indent=2, ensure_ascii=False)
    # batches
    batch_dir = os.path.join(out_dir, "batches")
    os.makedirs(batch_dir, exist_ok=True)
    batches = [text_records[i:i+batch_size] for i in range(0, len(text_records), batch_size)]
    for i, batch in enumerate(batches):
        with open(os.path.join(batch_dir, f"batch_{i+1:02d}.json"), "w") as f:
            json.dump({"meta": {**meta, "batch": i+1}, "samples": batch},
                      f, indent=2, ensure_ascii=False)
    print(f"  Saved meta + metrics + {len(batches)} batches → {out_dir}/")


def sample_clinical(seed, n, batch_size):
    print("\n[Clinical - aligned]")
    with open(CLIN_EINV_PATH) as f:
        einv_data = json.load(f)
    with open(CLIN_EINV_SCORES) as f:
        einv_scores = json.load(f)
    with open(CLIN_V2T_PATH) as f:
        v2t_full = json.load(f)

    einv_by_idx    = {int(d["idx"]): d for d in einv_data}
    einv_score_idx = {int(d["idx"]): d for d in einv_scores}
    v2t_by_idx     = {int(d["index"]): d for d in v2t_full["samples"]}

    common_ids = sorted(set(einv_by_idx.keys()) & set(v2t_by_idx.keys()))
    print(f"  Common indices: {len(common_ids)}")

    random.seed(seed)
    sampled_ids = sorted(random.sample(common_ids, n))

    text_records, metric_records = [], []
    for idx in sampled_ids:
        e  = einv_by_idx[idx]
        es = einv_score_idx.get(idx, {})
        v  = v2t_by_idx[idx]
        text_records.append({
            "index":            idx,
            "clinical_summary": e.get("summary", ""),
            "gt_text":          v["ground_truth"],
            "einv_pred":        e["gen_text"],
            "v2t_pred":         v["prediction"],
        })
        metric_records.append({
            "index": idx,
            "einv": {
                "rouge1":          es.get("rouge1", ""),
                "rouge2":          es.get("rouge2", ""),
                "rougeL":          es.get("rougeL", ""),
                "bertscore_f1":    es.get("bertscore_f1", ""),
                "token_f1":        es.get("token_f1", ""),
                "bleu":            es.get("bleu", ""),
                "structure_score": es.get("structure_score", ""),
                "entity_score":    es.get("entity_score", ""),
                "topic_score":     es.get("topic_score", ""),
            },
            "v2t": {
                "token_f1": v["token_f1"],
                "bleu":     v["bleu"],
            },
        })

    meta = {
        "domain": "clinical", "aligned": True,
        "seed": seed, "n_samples": n, "batch_size": batch_size,
        "einv_model": EINV_MODEL, "v2t_model": V2T_MODEL,
        "einv_source": CLIN_EINV_PATH, "einv_scores": CLIN_EINV_SCORES,
        "v2t_source":  CLIN_V2T_PATH,
    }
    save_batches(CLIN_OUT_DIR, meta, text_records, metric_records, batch_size)


def sample_wiki(seed, n, batch_size):
    print("\n[Wiki - unaligned, einv]")
    with open(WIKI_EINV_PATH) as f:
        einv_data = json.load(f)   # list, fields: gt_text, greedy_text, rouge1, ...

    random.seed(seed)
    einv_sampled = random.sample(range(len(einv_data)), n)

    text_records, metric_records = [], []
    for rank, pos in enumerate(sorted(einv_sampled)):
        e = einv_data[pos]
        rouge1 = e.get("rouge1", "")
        if isinstance(rouge1, str) and rouge1.startswith("["):
            rouge1 = json.loads(rouge1)[0]
        rouge2 = e.get("rouge2", "")
        if isinstance(rouge2, str) and rouge2.startswith("["):
            rouge2 = json.loads(rouge2)[0]
        rougeL = e.get("rougeL", "")
        if isinstance(rougeL, str) and rougeL.startswith("["):
            rougeL = json.loads(rougeL)[0]
        text_records.append({
            "index":      pos,
            "gt_text":    e["gt_text"],
            "einv_pred":  e["greedy_text"],
        })
        metric_records.append({
            "index": pos,
            "rouge1":       rouge1,
            "rouge2":       rouge2,
            "rougeL":       rougeL,
            "bertscore_f1": e.get("bertscore_f1", ""),
            "token_f1":     e.get("token_f1", ""),
        })

    meta = {
        "domain": "wikipedia", "model": "einv", "aligned": False,
        "seed": seed, "n_samples": n, "batch_size": batch_size,
        "einv_model": EINV_MODEL, "source": WIKI_EINV_PATH,
    }
    save_batches(os.path.join(WIKI_OUT_DIR, "einv"), meta, text_records, metric_records, batch_size)

    print("[Wiki - unaligned, v2t]")
    with open(WIKI_V2T_PATH) as f:
        v2t_full = json.load(f)
    v2t_data = v2t_full["samples"]

    random.seed(seed + 1)   # different seed so samples differ
    v2t_sampled = random.sample(range(len(v2t_data)), n)

    text_records, metric_records = [], []
    for pos in sorted(v2t_sampled):
        v = v2t_data[pos]
        text_records.append({
            "index":    int(v["index"]),
            "gt_text":  v["ground_truth"],
            "v2t_pred": v["prediction"],
        })
        metric_records.append({
            "index":       int(v["index"]),
            "token_f1":    v.get("token_f1", ""),
            "rouge1":      v.get("rouge1", ""),
            "rouge2":      v.get("rouge2", ""),
            "rougeL":      v.get("rougeL", ""),
            "bertscore_f1": v.get("bertscore_f1", ""),
        })

    meta = {
        "domain": "wikipedia", "model": "v2t", "aligned": False,
        "seed": seed + 1, "n_samples": n, "batch_size": batch_size,
        "v2t_model": V2T_MODEL, "source": WIKI_V2T_PATH,
    }
    save_batches(os.path.join(WIKI_OUT_DIR, "v2t"), meta, text_records, metric_records, batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--n",          type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=25)
    args = parser.parse_args()
    sample_clinical(args.seed, args.n, args.batch_size)
    sample_wiki(args.seed, args.n, args.batch_size)
