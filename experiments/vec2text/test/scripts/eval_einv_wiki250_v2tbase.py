#!/usr/bin/env python3
"""
Run v2t base inference on the 250 einv-sampled Wikipedia GT texts,
compute all metrics for both einv and v2t_base, and save a consolidated
output ready for human scoring.

Output format (per sample):
  {
    "index":        <int>,
    "gt_text":      <str>,
    "einv_pred":    <str>,
    "v2t_base_pred":<str>,
    "einv_metrics": { bleu, token_f1, rouge1, rouge2, rougeL, bertscore_f1,
                      structure_score, entity_score, topic_score },
    "v2t_base_metrics": { ... same keys ... }
  }

Saved to: results/human_eval/wiki_sample250/einv/v2t_base_comparison.json
          results/human_eval/wiki_sample250/einv/v2t_base_comparison_summary.json

Usage (GPU node required for Mistral-7B + BERTScore):
    cd "$PROJ"/scripts/vec2text
    python3 test/scripts/eval_einv_wiki250_v2tbase.py [--skip_gpt] [--skip_bertscore]
"""

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import transformers
from tqdm import tqdm

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent        # test/scripts/
VEC2TEXT_DIR = SCRIPT_DIR.parents[1]                  # scripts/vec2text/
PROJ         = VEC2TEXT_DIR.parents[1]                # embedding/

if str(VEC2TEXT_DIR) not in sys.path:
    sys.path.insert(0, str(VEC2TEXT_DIR))

os.environ.setdefault("HF_HOME", str(PROJ / "tmp" / "huggingface"))

EMBEDDER_MODEL   = "mistralai/Mistral-7B-v0.1"
CHECKPOINT       = str(VEC2TEXT_DIR / "test" / "results" / "mistral7b_l10t32_20260303_154531")
EINV_BATCH_DIR   = PROJ / "results" / "human_eval" / "wiki_sample250" / "einv" / "batches"
EINV_METRICS_FILE= PROJ / "results" / "human_eval" / "wiki_sample250" / "einv" / "metrics.json"
OUT_DIR          = PROJ / "results" / "human_eval" / "wiki_sample250" / "einv"
MAX_TOKENS       = 32
DEFAULT_EMBEDDINGS_FROM_LAYER_N = 11


# ── Helpers ───────────────────────────────────────────────────────────────────

def _token_f1(pred: str, ref: str) -> float:
    p, r = pred.lower().split(), ref.lower().split()
    if not p or not r: return 0.0
    common = set(p) & set(r)
    if not common: return 0.0
    prec = len(common) / len(p)
    rec  = len(common) / len(r)
    return 2 * prec * rec / (prec + rec)


def _bleu(ref: str, hyp: str) -> float:
    from sacrebleu.metrics import BLEU
    return BLEU(effective_order=True).sentence_score(hyp, [ref]).score


def _rouge(ref: str, hyp: str) -> dict:
    from rouge_score import rouge_scorer as rs
    scorer = rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    s = scorer.score(ref, hyp)
    return {k: s[k].fmeasure for k in ("rouge1", "rouge2", "rougeL")}


def _bertscore_batch(refs, hyps, model_type, device, batch_size=64):
    from bert_score import score as bscore
    P, R, F1 = bscore(hyps, refs, lang="en", model_type=model_type,
                      verbose=False, device=device, batch_size=batch_size)
    return F1.tolist()


def _gpt_score(gt, gen, client, gpt_model, domain="wikipedia"):
    """Call GPT for structure/entity/topic scores. Returns dict or None on error."""
    import openai, time

    prompts = {
        "structure": f"""
You are given two sentences:
[GT]: {gt}
[GEN]: {gen}

Evaluate the **Structural Frame Similarity**. Focus on whether [GEN] preserves the core grammatical structure and sentence skeleton of [GT].

Scoring: 5=identical, 4=nearly identical, 3=moderately similar, 2=somewhat similar, 1=minimally similar, 0=completely different.
Answer: [ANS] structure: [score]/5
""",
        "entity": f"""
You are given two sentences:
[GT]: {gt}
[GEN]: {gen}

Evaluate the **Entity/Role Consistency and Plausibility**. Are entities in [GEN] appropriate replacements for those in [GT] (same type/category/domain)?

Scoring: 5=same entities, 4=highly appropriate substitutions, 3=same category with notable differences, 2=different category but related, 1=incompatible or irrelevant, 0=completely wrong.
Answer: [ANS] entity: [score]/5
""",
        "topic": f"""
You are given two sentences:
[GT]: {gt}
[GEN]: {gen}

Evaluate the **Topic/Domain Coherence**. Does [GEN] stay in the same general topic area as [GT]?

Scoring: 5=identical topic, 4=closely related, 3=same broad domain, 2=loosely related, 1=tangentially related, 0=completely unrelated.
Answer: [ANS] topic: [score]/5
""",
    }

    scores = {}
    for key, prompt in prompts.items():
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=gpt_model,
                    messages=[{"role": "user", "content": prompt.strip()}],
                    temperature=0,
                    max_tokens=50,
                )
                text = resp.choices[0].message.content
                m = re.search(r'(\d+(?:\.\d+)?)\s*/\s*5', text)
                scores[key] = float(m.group(1)) / 5.0 if m else None
                break
            except Exception as e:
                if attempt == 2:
                    scores[key] = None
                time.sleep(1)
    return scores


def _mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals: return {"mean": None, "std": None, "n": 0}
    n = len(vals)
    m = sum(vals) / n
    v = sum((x - m) ** 2 for x in vals) / n
    return {"mean": round(m, 6), "std": round(math.sqrt(v), 6), "n": n}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",       default=CHECKPOINT)
    parser.add_argument("--batch_size",       type=int, default=8)
    parser.add_argument("--max_tokens",       type=int, default=MAX_TOKENS)
    parser.add_argument(
        "--embeddings_from_layer_n",
        type=int,
        default=DEFAULT_EMBEDDINGS_FROM_LAYER_N,
        help=(
            "HF hidden_states layer index to embed from. Use 11 to match "
            "TransformerLens blocks.10.hook_resid_post / l10t32."
        ),
    )
    parser.add_argument("--bertscore_model",  default="microsoft/deberta-v3-large")
    parser.add_argument("--bertscore_batch",  type=int, default=32)
    parser.add_argument("--gpt_model",        default="gpt-4.1-mini")
    parser.add_argument("--skip_gpt",         action="store_true")
    parser.add_argument("--skip_bertscore",   action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"  Einv-250 Wiki → V2T Base inference + full metrics")
    print(f"{'='*60}")
    print(f"  checkpoint       : {args.checkpoint}")
    print(f"  device           : {device}")
    print(f"  emb layer        : hidden_states[{args.embeddings_from_layer_n}]")
    print(f"  skip_bertscore   : {args.skip_bertscore}")
    print(f"  skip_gpt         : {args.skip_gpt}")
    print()

    # ── 1. Load einv 250 batch data ───────────────────────────────────────────
    print("[1/6] Loading einv 250 batch data ...")
    einv_rows = []  # {index, gt_text, einv_pred}
    for fn in sorted(os.listdir(EINV_BATCH_DIR)):
        if not fn.endswith(".json"): continue
        data = json.load(open(EINV_BATCH_DIR / fn))
        einv_rows.extend(data["samples"])
    einv_rows.sort(key=lambda x: x["index"])

    # Load per-sample einv metrics (rouge, bertscore, token_f1)
    einv_met_raw = json.load(open(EINV_METRICS_FILE))
    einv_met_by_idx = {s["index"]: s for s in einv_met_raw["samples"]}

    gt_texts   = [s["gt_text"]   for s in einv_rows]
    einv_preds = [s["einv_pred"] for s in einv_rows]
    indices    = [s["index"]     for s in einv_rows]
    print(f"    Loaded {len(einv_rows)} samples")

    # ── 2. Embed GT texts with Mistral-7B ─────────────────────────────────────
    print(f"\n[2/6] Embedding {len(gt_texts)} GT texts with Mistral-7B ...")
    src_tokenizer = transformers.AutoTokenizer.from_pretrained(
        EMBEDDER_MODEL, cache_dir=os.environ["HF_HOME"], use_fast=True
    )
    if src_tokenizer.pad_token is None:
        src_tokenizer.pad_token = src_tokenizer.eos_token

    emb_model = transformers.AutoModelForCausalLM.from_pretrained(
        EMBEDDER_MODEL,
        cache_dir=os.environ["HF_HOME"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    emb_model.eval()

    all_embs = []
    for i in tqdm(range(0, len(gt_texts), args.batch_size), desc="  Embedding"):
        batch = gt_texts[i: i + args.batch_size]
        enc = src_tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=args.max_tokens, add_special_tokens=False,
        ).to(device)
        with torch.no_grad():
            out = emb_model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                output_hidden_states=True, return_dict=True,
            )
        hidden   = out.hidden_states[args.embeddings_from_layer_n]
        mask = enc["attention_mask"].to(hidden.device)
        seq_lens = mask.shape[1] - 1 - mask.flip(dims=(1,)).argmax(dim=1)
        rows = torch.arange(len(batch), device=hidden.device)
        embs = hidden[rows, seq_lens].float().cpu()
        all_embs.append(embs)

    frozen_embeddings = torch.cat(all_embs, dim=0)
    print(f"    Embeddings: {frozen_embeddings.shape}")

    del emb_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── 3. Load v2t base model ────────────────────────────────────────────────
    print(f"\n[3/6] Loading v2t base model from {args.checkpoint} ...")
    t5_tokenizer = transformers.AutoTokenizer.from_pretrained(args.checkpoint)
    from vec2text.models.config import InversionConfig
    from vec2text.models import InversionModel
    config    = InversionConfig.from_pretrained(args.checkpoint)
    inv_model = InversionModel.from_pretrained(
        args.checkpoint, config=config, ignore_mismatched_sizes=False,
    ).to(device)
    inv_model.eval()
    print(f"    max_seq_length={config.max_seq_length}")

    # ── 4. Generate v2t base predictions ─────────────────────────────────────
    print(f"\n[4/6] Generating v2t base predictions ...")
    gen_kwargs = {
        "early_stopping": False, "num_beams": 1,
        "do_sample": False, "no_repeat_ngram_size": 0,
        "max_length": config.max_seq_length + 8,
    }
    v2t_preds = []
    for i in tqdm(range(0, len(gt_texts), args.batch_size), desc="  Decoding"):
        batch_emb = frozen_embeddings[i: i + args.batch_size].to(device)
        with torch.no_grad():
            out_ids = inv_model.generate(
                inputs={"frozen_embeddings": batch_emb},
                generation_kwargs=gen_kwargs,
            )
        v2t_preds.extend(t5_tokenizer.batch_decode(out_ids, skip_special_tokens=True))
    print(f"    Generated {len(v2t_preds)} predictions")

    del inv_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── 5. Compute base metrics (BLEU, token_f1, ROUGE) ───────────────────────
    print(f"\n[5/6] Computing base metrics (BLEU, Token F1, ROUGE) ...")

    einv_bleu, v2t_bleu = [], []
    einv_f1,   v2t_f1   = [], []
    einv_rouge, v2t_rouge = [], []

    for gt, ep, vp in zip(gt_texts, einv_preds, v2t_preds):
        einv_bleu.append(_bleu(gt, ep))
        v2t_bleu.append(_bleu(gt, vp))
        einv_f1.append(_token_f1(ep, gt))
        v2t_f1.append(_token_f1(vp, gt))
        einv_rouge.append(_rouge(gt, ep))
        v2t_rouge.append(_rouge(gt, vp))

    # BERTScore
    einv_bscore, v2t_bscore = [None]*len(gt_texts), [None]*len(gt_texts)
    if not args.skip_bertscore:
        print(f"\n     Computing BERTScore ({args.bertscore_model}) ...")
        einv_bscore = _bertscore_batch(gt_texts, einv_preds, args.bertscore_model,
                                       device, batch_size=args.bertscore_batch)
        v2t_bscore  = _bertscore_batch(gt_texts, v2t_preds,  args.bertscore_model,
                                       device, batch_size=args.bertscore_batch)

    # GPT scores
    einv_gpt = [{"structure_score": None, "entity_score": None, "topic_score": None}] * len(gt_texts)
    v2t_gpt  = [{"structure_score": None, "entity_score": None, "topic_score": None}] * len(gt_texts)
    if not args.skip_gpt:
        import openai as _openai
        client = _openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        print(f"\n     Computing GPT scores ({args.gpt_model}, n={len(gt_texts)*2}) ...")
        einv_gpt, v2t_gpt = [], []
        for gt, ep, vp in tqdm(zip(gt_texts, einv_preds, v2t_preds),
                                total=len(gt_texts), desc="  GPT (einv+v2t)"):
            gs_e = _gpt_score(gt, ep, client, args.gpt_model)
            gs_v = _gpt_score(gt, vp, client, args.gpt_model)
            einv_gpt.append({
                "structure_score": gs_e.get("structure"),
                "entity_score":    gs_e.get("entity"),
                "topic_score":     gs_e.get("topic"),
            })
            v2t_gpt.append({
                "structure_score": gs_v.get("structure"),
                "entity_score":    gs_v.get("entity"),
                "topic_score":     gs_v.get("topic"),
            })

    # ── 6. Build per-sample consolidated records ──────────────────────────────
    print(f"\n[6/6] Building consolidated JSON ...")
    samples = []
    for i, (idx, gt, ep, vp) in enumerate(zip(indices, gt_texts, einv_preds, v2t_preds)):
        # einv existing bertscore from metrics.json (if bertscore not recomputed)
        em   = einv_met_by_idx.get(idx, {})
        e_bs = einv_bscore[i] if not args.skip_bertscore else float(em.get("bertscore_f1", 0) or 0)

        samples.append({
            "index":         idx,
            "gt_text":       gt,
            "einv_pred":     ep,
            "v2t_base_pred": vp,
            "einv_metrics": {
                "bleu":            round(einv_bleu[i], 6),
                "token_f1":        round(einv_f1[i],   6),
                "rouge1":          round(einv_rouge[i]["rouge1"], 6),
                "rouge2":          round(einv_rouge[i]["rouge2"], 6),
                "rougeL":          round(einv_rouge[i]["rougeL"], 6),
                "bertscore_f1":    round(e_bs, 6) if e_bs is not None else None,
                "structure_score": einv_gpt[i]["structure_score"] if not args.skip_gpt else None,
                "entity_score":    einv_gpt[i]["entity_score"]    if not args.skip_gpt else None,
                "topic_score":     einv_gpt[i]["topic_score"]     if not args.skip_gpt else None,
            },
            "v2t_base_metrics": {
                "bleu":            round(v2t_bleu[i], 6),
                "token_f1":        round(v2t_f1[i],   6),
                "rouge1":          round(v2t_rouge[i]["rouge1"], 6),
                "rouge2":          round(v2t_rouge[i]["rouge2"], 6),
                "rougeL":          round(v2t_rouge[i]["rougeL"], 6),
                "bertscore_f1":    round(v2t_bscore[i], 6) if v2t_bscore[i] is not None else None,
                "structure_score": v2t_gpt[i]["structure_score"] if not args.skip_gpt else None,
                "entity_score":    v2t_gpt[i]["entity_score"]    if not args.skip_gpt else None,
                "topic_score":     v2t_gpt[i]["topic_score"]     if not args.skip_gpt else None,
            },
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    MKEYS = ["bleu", "token_f1", "rouge1", "rouge2", "rougeL",
             "bertscore_f1", "structure_score", "entity_score", "topic_score"]

    def agg(samples, model_key):
        out = {}
        for mk in MKEYS:
            vals = [s[model_key][mk] for s in samples if s[model_key][mk] is not None]
            out[mk] = _mean_std(vals)
        return out

    summary = {
        "einv":     agg(samples, "einv_metrics"),
        "v2t_base": agg(samples, "v2t_base_metrics"),
    }

    # ── Save ──────────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_full    = OUT_DIR / "v2t_base_comparison.json"
    out_summary = OUT_DIR / "v2t_base_comparison_summary.json"

    full_data = {
        "meta": {
            "description": ("250 einv-sampled Wikipedia GT texts. "
                            "Both einv and v2t_base predictions evaluated against the same GT. "
                            "BLEU in 0-100, other scores in 0-1."),
            "n": len(samples),
            "checkpoint": args.checkpoint,
            "bertscore_model": args.bertscore_model,
            "gpt_model": args.gpt_model,
            "skip_gpt": args.skip_gpt,
            "skip_bertscore": args.skip_bertscore,
        },
        "summary": summary,
        "samples": samples,
    }

    with open(out_full, "w", encoding="utf-8") as f:
        json.dump(full_data, f, indent=2, ensure_ascii=False)
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump({"meta": full_data["meta"], "summary": summary}, f, indent=2, ensure_ascii=False)

    print(f"\n  Saved → {out_full}")
    print(f"  Saved → {out_summary}")

    # ── Pretty summary table ──────────────────────────────────────────────────
    MLABELS = {
        "bleu":            "BLEU (0-100)",
        "token_f1":        "Token F1",
        "rouge1":          "ROUGE-1",
        "rouge2":          "ROUGE-2",
        "rougeL":          "ROUGE-L",
        "bertscore_f1":    "BERTScore F1",
        "structure_score": "Structure",
        "entity_score":    "Entity",
        "topic_score":     "Topic",
    }
    CW = 30
    print(f"\n{'='*74}")
    print(f"  WIKI-250 (einv GT set)  —  einv vs v2t_base")
    print(f"{'='*74}")
    print(f"{'Metric':<18}", end="")
    for m in ("einv", "v2t_base"):
        print(f"  {m.center(CW)}", end="")
    print()
    print("-" * 74)
    for mk in MKEYS:
        lbl = MLABELS.get(mk, mk)
        print(f"{lbl:<18}", end="")
        for model in ("einv", "v2t_base"):
            d = summary[model].get(mk, {})
            if d.get("mean") is None or d.get("n", 0) == 0:
                cell = "N/A"
            else:
                cell = f"{d['mean']:.4f} ± {d['std']:.4f} (n={d['n']})"
            print(f"  {cell.center(CW)}", end="")
        print()
    print("=" * 74)
    print()


if __name__ == "__main__":
    main()
