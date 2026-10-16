#!/usr/bin/env python3
"""
Consolidate human-eval wiki 250-item results for:
  - einv          (250 items, its own GT set, different from v2t)
  - v2t base      (250 items; full metrics drawn from the 1000-item eval file)
  - v2t corrector (same 250 GT items as v2t base; metrics from corrector eval file)

Because einv and v2t were sampled with different seeds they do NOT share GT texts.
This script therefore produces TWO parallel blocks:
  • v2t_items      – 250 entries, each with gt / v2t_base / v2t_corrector
  • einv_items     – 250 entries, each with gt / einv
and a unified summary table so the three models can be compared numerically.

Outputs
-------
  results/human_eval/wiki_sample250/wiki_250_consolidated.json
  results/human_eval/wiki_sample250/wiki_250_summary.json
  (also prints a formatted table to stdout)

Usage
-----
  cd "$PROJ"  # repo root
  python experiments/evaluation/wiki_human_eval_summary.py
"""

import json, math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import sacrebleu as _sacrebleu

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, '..', '..')

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
V2T_BASE_SRC    = os.path.join(ROOT, 'experiments/vec2text/test/results',
                  'mistral7b_l10t32_20260303_154531/eval/results_wikipedia_val.json')
CORR_SRC        = os.path.join(ROOT, 'experiments/vec2text/test/results',
                  'corrector_mistral7b_l10t32_20260307_145238/eval',
                  'results_wikipedia_val_corrector50steps.json')

V2T_HUMAN_DIR   = os.path.join(ROOT, 'results/human_eval/wiki_sample250/v2t')
EINV_HUMAN_DIR  = os.path.join(ROOT, 'results/human_eval/wiki_sample250/einv')

OUT_DIR         = os.path.join(ROOT, 'results/human_eval/wiki_sample250')
OUT_FULL        = os.path.join(OUT_DIR, 'wiki_250_consolidated.json')
OUT_SUMMARY     = os.path.join(OUT_DIR, 'wiki_250_summary.json')


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _flat(v):
    """Flatten a list or scalar metric value to a single float."""
    if isinstance(v, list):
        return sum(v) / len(v) if v else float('nan')
    return float(v)


def _bleu(ref: str, hyp: str) -> float:
    """Sentence-level BLEU in 0–100 scale (sacrebleu)."""
    return _sacrebleu.sentence_bleu(hyp, [ref]).score


def _mean_std(values):
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return {"mean": None, "std": None, "n": 0}
    n    = len(vals)
    mean = sum(vals) / n
    var  = sum((x - mean) ** 2 for x in vals) / n
    return {"mean": round(mean, 6), "std": round(math.sqrt(var), 6), "n": n}


def load_batches(batch_dir):
    """Load all batch_XX.json files sorted, return list of sample dicts."""
    files = sorted(f for f in os.listdir(batch_dir) if f.endswith('.json'))
    out = []
    for fn in files:
        data = json.load(open(os.path.join(batch_dir, fn)))
        out.extend(data['samples'])
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Load full 1000-item v2t and corrector sources (index → sample)
# ──────────────────────────────────────────────────────────────────────────────
print("Loading source files …")
v2t_base_src  = json.load(open(V2T_BASE_SRC))
v2t_by_idx    = {s['index']: s for s in v2t_base_src['samples']}

corr_src      = json.load(open(CORR_SRC))
corr_by_idx   = {s['index']: s for s in corr_src['samples']}

METRIC_KEYS = ['bleu', 'token_f1', 'rouge1', 'rouge2', 'rougeL',
               'bertscore_f1', 'structure_score', 'entity_score', 'topic_score']


def extract_v2t_metrics(sample: dict) -> dict:
    return {k: sample[k] for k in METRIC_KEYS if k in sample}


# ──────────────────────────────────────────────────────────────────────────────
# Build v2t 250-item aligned block
# ──────────────────────────────────────────────────────────────────────────────
print("Building v2t-aligned 250 items …")
v2t_batches  = load_batches(os.path.join(V2T_HUMAN_DIR, 'batches'))
v2t_meta     = json.load(open(os.path.join(V2T_HUMAN_DIR, 'meta.json')))

v2t_items = []
for s in sorted(v2t_batches, key=lambda x: x['index']):
    idx = s['index']
    gt  = s['gt_text']

    base = v2t_by_idx.get(idx)
    corr = corr_by_idx.get(idx)

    base_metrics = extract_v2t_metrics(base) if base else {}
    corr_metrics = extract_v2t_metrics(corr) if corr else {}
    entry = {
        "index":    idx,
        "gt_text":  gt,
        "v2t_base": dict(pred=base['prediction'] if base else None, **base_metrics),
        "v2t_corrector": dict(pred=corr['prediction'] if corr else None, **corr_metrics),
    }
    v2t_items.append(entry)

print(f"  v2t aligned items: {len(v2t_items)}")


# ──────────────────────────────────────────────────────────────────────────────
# Build einv 250-item block (compute BLEU from raw text pairs)
# ──────────────────────────────────────────────────────────────────────────────
print("Building einv 250 items + computing BLEU …")
einv_batches  = load_batches(os.path.join(EINV_HUMAN_DIR, 'batches'))
einv_metrics  = json.load(open(os.path.join(EINV_HUMAN_DIR, 'metrics.json')))
einv_meta     = json.load(open(os.path.join(EINV_HUMAN_DIR, 'meta.json')))
einv_met_by_idx = {s['index']: s for s in einv_metrics['samples']}

einv_items = []
for s in sorted(einv_batches, key=lambda x: x['index']):
    idx      = s['index']
    gt_text  = s['gt_text']
    pred     = s['einv_pred']
    met      = einv_met_by_idx.get(idx, {})

    bleu_val = _bleu(gt_text, pred)

    entry = {
        "index":   idx,
        "gt_text": gt_text,
        "einv": {
            "pred":         pred,
            "bleu":         bleu_val,
            "token_f1":     _flat(met.get('token_f1')),
            "rouge1":       _flat(met.get('rouge1')),
            "rouge2":       _flat(met.get('rouge2')),
            "rougeL":       _flat(met.get('rougeL')),
            "bertscore_f1": _flat(met.get('bertscore_f1')),
            # structure/entity/topic not available for einv
        },
    }
    einv_items.append(entry)

print(f"  einv items: {len(einv_items)}")


# ──────────────────────────────────────────────────────────────────────────────
# Aggregate summary
# ──────────────────────────────────────────────────────────────────────────────
print("Computing summary statistics …")

def aggregate_model(items, model_key):
    per_metric = {k: [] for k in METRIC_KEYS}
    for item in items:
        m = item[model_key]
        for k in METRIC_KEYS:
            v = m.get(k)
            if v is not None:
                per_metric[k].append(float(v))
    return {k: _mean_std(vs) for k, vs in per_metric.items() if vs}

summary = {
    "v2t_base":      aggregate_model(v2t_items, 'v2t_base'),
    "v2t_corrector": aggregate_model(v2t_items, 'v2t_corrector'),
    "einv":          aggregate_model(einv_items, 'einv'),
}


# ──────────────────────────────────────────────────────────────────────────────
# Save consolidated JSON
# ──────────────────────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)

consolidated = {
    "meta": {
        "note":       ("v2t_base and v2t_corrector share the same 250 GT texts. "
                       "einv uses a different 250 GT texts (sampled with a different seed). "
                       "All metrics are per-sample; BLEU is in 0–100 scale."),
        "v2t_base_model":      v2t_meta.get('v2t_model', ''),
        "v2t_corrector_model": corr_src.get('meta', {}).get('model', 'corrector_mistral7b_l10t32'),
        "einv_model":          einv_meta.get('einv_model', ''),
        "n_v2t":     len(v2t_items),
        "n_einv":    len(einv_items),
    },
    "v2t_items":  v2t_items,
    "einv_items": einv_items,
    "summary":    summary,
}

with open(OUT_FULL, 'w', encoding='utf-8') as f:
    json.dump(consolidated, f, indent=2, ensure_ascii=False)
print(f"\nSaved consolidated JSON → {OUT_FULL}")

summary_only = {"meta": consolidated["meta"], "summary": summary}
with open(OUT_SUMMARY, 'w', encoding='utf-8') as f:
    json.dump(summary_only, f, indent=2, ensure_ascii=False)
print(f"Saved summary JSON     → {OUT_SUMMARY}")


# ──────────────────────────────────────────────────────────────────────────────
# Pretty-print summary table
# ──────────────────────────────────────────────────────────────────────────────
METRIC_LABEL = {
    'bleu':           'BLEU (0-100)',
    'token_f1':       'Token F1',
    'rouge1':         'ROUGE-1',
    'rouge2':         'ROUGE-2',
    'rougeL':         'ROUGE-L',
    'bertscore_f1':   'BERTScore F1',
    'structure_score':'Structure',
    'entity_score':   'Entity',
    'topic_score':    'Topic',
}

COL_W = 22

def fmt_cell(d):
    if d is None or d.get('n', 0) == 0 or d.get('mean') is None:
        return 'N/A'.center(COL_W)
    m, s, n = d['mean'], d['std'], d['n']
    return f"{m:.4f} ± {s:.4f} (n={n})".center(COL_W)

header_models = ['v2t_base', 'v2t_corrector', 'einv']
col0 = 16  # metric name column width

print()
print("=" * (col0 + 3 * COL_W + 4))
print("  WIKI-250 HUMAN EVAL SUMMARY")
print(f"  v2t GT set (n=250, seed=43):  v2t_base vs v2t_corrector")
print(f"  einv GT set (n=250, seed=42): einv (different GT texts)")
print("=" * (col0 + 3 * COL_W + 4))

# header row
print(f"{'Metric':<{col0}}", end="")
for m in header_models:
    print(f"  {m.center(COL_W)}", end="")
print()
print("-" * (col0 + 3 * COL_W + 4))

for mk in METRIC_KEYS:
    label = METRIC_LABEL.get(mk, mk)
    print(f"{label:<{col0}}", end="")
    for model in header_models:
        d = summary[model].get(mk)
        print(f"  {fmt_cell(d)}", end="")
    print()

print("=" * (col0 + 3 * COL_W + 4))
print()
print("Note: BLEU is sentence-level sacrebleu (0-100).")
print("      Structure/Entity/Topic scores are GPT-4.1-mini ratings (0-1).")
print("      These GPT scores are NOT available for einv.")
