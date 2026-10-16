"""GPT-rate the 500 sentence-aligned 16-token 3B inversion outputs.

Reads:  results/llama3B/l10t16p16/full_sentence/ours/val_einv_decoded.json
Writes: results/llama3B/l10t16p16/full_sentence/ours/val_einv_gpt_scores.json
"""
import json
import os
from pathlib import Path
from tqdm import tqdm

# Match the eval pipeline's config so utils import order is consistent
os.environ["EINV_CONFIG"] = "models/setup3B.yaml"
os.environ.setdefault("TRAIN_TAG", "l3b")

from utils import logger  # noqa: E402
from utils.eval import GPTScorer  # noqa: E402
from utils.eval.scores import compute_rouge_score, compute_bertscore  # noqa: E402
from utils.project_utils import set_global_seed  # noqa: E402

set_global_seed()

PROJ = Path(__file__).resolve().parents[2]
DEC_PATH = PROJ / "results/llama3B/l10t16p16/full_sentence/ours/val_einv_decoded.json"
OUT_PATH = PROJ / "results/llama3B/l10t16p16/full_sentence/ours/val_einv_gpt_scores.json"


def main():
    with open(DEC_PATH) as f:
        decoded = json.load(f)
    logger.info(f"Loaded {len(decoded)} decoded samples from {DEC_PATH}")

    gpt_scores = []
    for item in tqdm(decoded, desc="GPT scoring"):
        gt = item["gt_text"]
        gen = item["greedy_text"]

        scorer = GPTScorer(gt, gen)
        scorer.gpt_rating()

        sample = {
            "gt_text": gt,
            "greedy_text": gen,
            "structure_score": scorer.structure_score / 5.0,
            "entity_score": scorer.entity_score / 5.0,
            "topic_score": scorer.topic_score / 5.0,
        }

        rouge_score = compute_rouge_score(gt, gen)
        sample["rouge1"] = rouge_score["rouge1"]
        sample["rouge2"] = rouge_score["rouge2"]
        sample["rougeL"] = rouge_score["rougeL"]

        bert = compute_bertscore([gt], [gen])
        sample["bertscore_f1"] = bert["f1"]

        gpt_scores.append(sample)

        # Periodic save in case of API failures mid-run
        if len(gpt_scores) % 50 == 0:
            with open(OUT_PATH, "w") as f:
                json.dump(gpt_scores, f, indent=2, ensure_ascii=False)

    with open(OUT_PATH, "w") as f:
        json.dump(gpt_scores, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(gpt_scores)} GPT scores to {OUT_PATH}")


if __name__ == "__main__":
    main()
