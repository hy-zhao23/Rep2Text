from utils.eval import GPTScorer
from utils import logger
from utils.core.path import get_decoded_path, get_gpt_scores_path
from utils.project_utils import set_global_seed
from utils.core.files import read_json, write_json
from utils.eval.scores import compute_rouge_score, compute_bertscore

import random
from tqdm import tqdm
import argparse
set_global_seed()

def get_gt_gen_pairs(model_type="einv", split="val", num_samples=1000):
    decoded_file = get_decoded_path(split=split, model_type=model_type)
    decoded_data = read_json(decoded_file)
    gt_gen_pairs = []
    for item in decoded_data:
        gt_gen_pairs.append((item["gt_text"], item["greedy_text"]))

    logger.info(f"Got {len(gt_gen_pairs)} GT-GEN pairs from {decoded_file}")

    if num_samples is None or num_samples <= 0:
        return gt_gen_pairs

    sample_size = min(num_samples, len(gt_gen_pairs))
    logger.info(f"Sampling {sample_size} GT-GEN pairs")
    if sample_size < len(gt_gen_pairs):
        gt_gen_pairs = random.sample(gt_gen_pairs, sample_size)

    return gt_gen_pairs

def main():
    parser = argparse.ArgumentParser(description="Evaluate GPT scores")
    parser.add_argument("--model_type", type=str, required=True, help="einv or einv_lora")
    parser.add_argument("--split", type=str, default="val", required=False, help="Dataset split to evaluate on")
    parser.add_argument("--num_samples", type=int, default=1000, required=False, help="Number of samples to score")
    args = parser.parse_args()

    gt_gen_pairs = get_gt_gen_pairs(args.model_type, split=args.split, num_samples=args.num_samples)
    file = get_gpt_scores_path(split=args.split, model_type=args.model_type)
    
    gpt_scores = []

    for gt, gen in tqdm(gt_gen_pairs, desc="Evaluating GPT scores"):
        scorer = GPTScorer(gt, gen) 
        scorer.gpt_rating()
        _sample = {
            "gt_text": gt,
            "greedy_text": gen,
            "structure_score": scorer.structure_score/5.0,
            "entity_score": scorer.entity_score/5.0,
            "topic_score": scorer.topic_score/5.0
        }

        rouge_score = compute_rouge_score(gt, gen)
        _sample["rouge1"] = rouge_score["rouge1"]
        _sample["rouge2"] = rouge_score["rouge2"]
        _sample["rougeL"] = rouge_score["rougeL"]

        bertscore_score = compute_bertscore([gt], [gen])
        _sample["bertscore_f1"] = bertscore_score["f1"]


        gpt_scores.append(_sample)

    write_json(gpt_scores, file)
    logger.info(f"Saved {len(gpt_scores)} GPT scores to {file}")


if __name__ == "__main__":
    main()