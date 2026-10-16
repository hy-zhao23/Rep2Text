"""Aggregate metrics for all three methods on the full_sentence subset.

For each method, computes per-sample Token F1, BLEU, ROUGE-1/2/L, BERTScore F1
and saves a results.json + summary.json under each method's folder.
No LLM-judge.
"""
import json
import sys
from pathlib import Path
import numpy as np
from collections import Counter

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ))

ROOT = PROJ / "results/mistral7B/l10t32p32/full_sentence"
METHODS = ["ours", "vec2text_base", "vec2text_corrector50"]


def token_f1(gt_tokens, pr_tokens):
    gc = Counter(gt_tokens); pc = Counter(pr_tokens)
    common = sum((gc & pc).values())
    if common == 0: return 0.0
    p = common / len(pr_tokens); r = common / len(gt_tokens)
    return 2 * p * r / (p + r)


def main():
    from rouge_score import rouge_scorer
    import sacrebleu
    from bert_score import BERTScorer

    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    bert = BERTScorer(model_type="microsoft/deberta-v3-large", num_layers=24, lang="en", rescale_with_baseline=False)

    summary = {}
    for method in METHODS:
        mdir = ROOT / method
        # Our format vs vec2text format differ
        if method == "ours":
            dec_path = mdir / "val_einv_decoded.json"
            if not dec_path.exists():
                print(f"[skip] {dec_path} not found")
                continue
            with open(dec_path) as f:
                items = json.load(f)
            refs = [it["gt_text"] for it in items]
            preds = [it["greedy_text"] for it in items]
            gt_ids_list = [it.get("gt_ids", []) for it in items]
            pr_ids_list = [it.get("greedy_ids", []) for it in items]
        else:
            pred_path = mdir / "predictions.json"
            if not pred_path.exists():
                print(f"[skip] {pred_path} not found")
                continue
            with open(pred_path) as f:
                items = json.load(f)
            refs = [it["ground_truth"] for it in items]
            preds = [it["prediction"] for it in items]
            # tokenize on whitespace for token-F1 (mistral tokenizer would be ideal but consistent with vec2text approach)
            gt_ids_list = [r.lower().split() for r in refs]
            pr_ids_list = [p.lower().split() for p in preds]

        # Token F1 (id-based for ours, word-based for vec2text — to compare apples-to-apples,
        # also compute word-F1 for ours)
        word_f1 = [token_f1(r.lower().split(), p.lower().split()) for r, p in zip(refs, preds)]
        id_f1 = [token_f1(g, pr) for g, pr in zip(gt_ids_list, pr_ids_list)]

        # BLEU
        bleu_scores = [sacrebleu.sentence_bleu(p, [r]).score for p, r in zip(preds, refs)]

        # ROUGE
        r1, r2, rL = [], [], []
        for r, p in zip(refs, preds):
            s = rouge.score(r, p)
            r1.append(s["rouge1"].fmeasure)
            r2.append(s["rouge2"].fmeasure)
            rL.append(s["rougeL"].fmeasure)

        # BERTScore F1
        _, _, bf = bert.score(preds, refs)
        bf = bf.tolist()

        per_sample = []
        for i in range(len(refs)):
            per_sample.append({
                "index": i,
                "gt": refs[i],
                "pred": preds[i],
                "token_f1_word": word_f1[i],
                "token_f1_id":   id_f1[i],
                "bleu":          bleu_scores[i],
                "rouge1":        r1[i],
                "rouge2":        r2[i],
                "rougeL":        rL[i],
                "bertscore_f1":  bf[i],
            })

        out_path = mdir / "metrics.json"
        with open(out_path, "w") as f:
            json.dump(per_sample, f, indent=2, ensure_ascii=False)
        print(f"  saved {out_path}")

        s = {
            "n": len(refs),
            "token_f1_word_mean": float(np.mean(word_f1)),
            "token_f1_id_mean":   float(np.mean(id_f1)),
            "bleu_mean":          float(np.mean(bleu_scores)),
            "rouge1_mean":        float(np.mean(r1)),
            "rouge2_mean":        float(np.mean(r2)),
            "rougeL_mean":        float(np.mean(rL)),
            "bertscore_f1_mean":  float(np.mean(bf)),
        }
        summary[method] = s
        print(f"  {method}: TF1(word)={s['token_f1_word_mean']:.3f}  BLEU={s['bleu_mean']:.2f}  R-L={s['rougeL_mean']:.3f}  BERT-F1={s['bertscore_f1_mean']:.3f}")

    out_summary = ROOT / "summary.json"
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {out_summary}")


if __name__ == "__main__":
    main()
