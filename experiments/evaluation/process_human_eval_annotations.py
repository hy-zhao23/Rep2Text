import argparse
import json
import math
import os
import re
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path

import openai


GPT_DIMENSIONS = ("structure", "entity", "topic")
COMPARISON_MODELS = ("einv", "v2t_base")
COMPARISON_METRIC_KEYS = (
    "bleu",
    "token_f1",
    "rouge1",
    "rouge2",
    "rougeL",
    "bertscore_f1",
    "structure_score",
    "entity_score",
    "topic_score",
)
HUMAN_SCORE_KEYS = {
    "structural_frame_similarity": "structure_score",
    "entity_preservation": "entity_score",
    "topic_consistency": "topic_score",
}


def build_prompts(gt_text, pred_text):
    return {
        "structure": f"""
You are given two sentences:
[GT]: {gt_text}

[GEN]: {pred_text}

Evaluate the Structural Frame Similarity between these sentences.

Focus on whether [GEN] preserves the core grammatical structure and sentence skeleton of [GT]:
- Same basic clause structure
- Same number and type of major phrases in similar positions
- Similar syntactic relationships between components
- Consistent verb tense/aspect when applicable

Scoring Guidelines:
- 5: Identical structure with almost exactly the same words in the same order
- 4: Nearly identical structure with minor substitutions or small modifier changes
- 3: Moderately similar structure with notable changes
- 2: Somewhat similar structure with significant differences
- 1: Minimally similar structure
- 0: Completely different structure

Answer in exactly this format:
[ANS] structure: [score]/5
""".strip(),
        "entity": f"""
You are given two sentences:
[GT]: {gt_text}
[GEN]: {pred_text}

Evaluate the Entity/Role Consistency and Plausibility between these sentences.

Assess whether entities in [GEN] are appropriate replacements for those in [GT], considering entity type matching and contextual plausibility.

Scoring Guidelines:
- 5: Perfect or near-perfect entity matches
- 4: Strong contextual matches with minor differences
- 3: Plausible matches with notable differences
- 2: Weak matches with significant mismatches
- 1: Poor matches; mostly wrong type but not fully nonsensical
- 0: Nonsensical or completely wrong entity replacements

Answer in exactly this format:
[ANS] entity: [score]/5
""".strip(),
        "topic": f"""
You are given two sentences:
[GT]: {gt_text}
[GEN]: {pred_text}

Evaluate the Topical Relevance between these sentences.

Assess whether [GEN] maintains the same overall subject matter, theme, or knowledge domain as [GT].

Scoring Guidelines:
- 5: Same specific topic
- 4: Same general topic with minor shifts
- 3: Related topics within a broader category
- 2: Loosely related topics
- 1: Minimally related topics
- 0: Unrelated topics

Answer in exactly this format:
[ANS] topic: [score]/5
""".strip(),
    }


def score_dimension(client, model, prompt, max_retries, sleep_seconds):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=50,
            )
            content = response.choices[0].message.content or ""
            match = re.search(r"(\d)\s*/\s*5", content)
            if match:
                return int(match.group(1)) / 5.0
        except Exception:
            pass

        if attempt < max_retries - 1:
            time.sleep(sleep_seconds)

    return None


def mean_std(values):
    clean = [value for value in values if value is not None]
    if not clean:
        return {"mean": None, "std": None, "n": 0}
    mean = sum(clean) / len(clean)
    variance = sum((value - mean) ** 2 for value in clean) / len(clean)
    return {
        "mean": round(mean, 6),
        "std": round(math.sqrt(variance), 6),
        "n": len(clean),
    }


def normalize_human_scores(answer):
    normalized = {}
    for model_key in COMPARISON_MODELS:
        raw = answer.get(model_key, {})
        normalized[model_key] = {
            metric_key: (raw.get(human_key) / 5.0 if raw.get(human_key) is not None else None)
            for human_key, metric_key in HUMAN_SCORE_KEYS.items()
        }
    return normalized


def build_model_summary(samples, model_key):
    metric_key = f"{model_key}_metrics"
    return {
        metric_name: mean_std([sample[metric_key][metric_name] for sample in samples])
        for metric_name in COMPARISON_METRIC_KEYS
    }


def build_human_summary(samples):
    summary = {}
    for model_key in COMPARISON_MODELS:
        summary[model_key] = {
            metric_name: mean_std(
                [sample["human_scores_normalized"][model_key][metric_name] for sample in samples]
            )
            for metric_name in HUMAN_SCORE_KEYS.values()
        }
    return summary


def count_null_fields(samples):
    counter = Counter()
    for sample in samples:
        for model_key in ("einv_metrics", "v2t_base_metrics"):
            for dim in GPT_DIMENSIONS:
                metric_name = f"{dim}_score"
                if sample[model_key][metric_name] is None:
                    counter[f"{model_key}.{metric_name}"] += 1
    return dict(counter)


def resolve_annotator_paths(args):
    paths = []
    for item in args.annotators:
        path = Path(item)
        if any(char in item for char in "*?[]"):
            paths.extend(sorted(Path().glob(item)))
        else:
            paths.append(path)

    unique_paths = []
    seen = set()
    for path in paths:
        resolved = str(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_paths.append(path)

    if not unique_paths:
        raise RuntimeError("No annotator files matched the given inputs.")

    return unique_paths


def fill_missing_scores(sample, client, model, max_retries, sleep_seconds, score_cache):
    filled_fields = []
    remaining_null_fields = []

    prompt_cache = {
        "einv": build_prompts(sample["gt_text"], sample["einv_pred"]),
        "v2t_base": build_prompts(sample["gt_text"], sample["v2t_base_pred"]),
    }

    for model_name, metric_container in (("einv", "einv_metrics"), ("v2t_base", "v2t_base_metrics")):
        pred_text = sample["einv_pred"] if model_name == "einv" else sample["v2t_base_pred"]
        for dim in GPT_DIMENSIONS:
            metric_name = f"{dim}_score"
            if sample[metric_container][metric_name] is not None:
                continue

            cache_key = (sample["gt_text"], pred_text, dim, model)
            if cache_key not in score_cache:
                score_cache[cache_key] = score_dimension(
                    client=client,
                    model=model,
                    prompt=prompt_cache[model_name][dim],
                    max_retries=max_retries,
                    sleep_seconds=sleep_seconds,
                )

            score = score_cache[cache_key]
            if score is None:
                remaining_null_fields.append(f"{metric_container}.{metric_name}")
            else:
                sample[metric_container][metric_name] = score
                filled_fields.append(f"{metric_container}.{metric_name}")

    return filled_fields, remaining_null_fields


def process_annotator_file(
    annotator_path,
    comparison_data,
    comparison_path,
    output_dir,
    client,
    model,
    max_retries,
    sleep_seconds,
    score_cache,
):
    annotator_data = json.loads(annotator_path.read_text(encoding="utf-8"))
    answers = annotator_data["answers"]
    samples_by_index = {entry["index"]: entry for entry in comparison_data["samples"]}
    ordered_indices = [entry["global_index"] for entry in answers]
    missing_indices = [index for index in ordered_indices if index not in samples_by_index]
    if missing_indices:
        raise RuntimeError(f"Missing indices for {annotator_path}: {missing_indices}")

    output_samples = []
    filled_counter = Counter()
    remaining_counter = Counter()

    for answer in answers:
        index = answer["global_index"]
        sample = deepcopy(samples_by_index[index])
        filled_fields, remaining_fields = fill_missing_scores(
            sample=sample,
            client=client,
            model=model,
            max_retries=max_retries,
            sleep_seconds=sleep_seconds,
            score_cache=score_cache,
        )

        for field in filled_fields:
            filled_counter[field] += 1
        for field in remaining_fields:
            remaining_counter[field] += 1

        sample["annotation"] = answer
        sample["human_scores_normalized"] = normalize_human_scores(answer)
        sample["rescore_info"] = {
            "filled_fields": filled_fields,
            "remaining_null_fields": remaining_fields,
        }
        output_samples.append(sample)

    output_data = {
        "meta": {
            "source_annotator_file": str(annotator_path),
            "annotator_id": annotator_data.get("annotator_id"),
            "task_name": annotator_data.get("task_name"),
            "submitted_at_client": annotator_data.get("submitted_at_client"),
            "source_comparison": str(comparison_path),
            "subset_n": len(output_samples),
            "gpt_model_for_rescore": model,
            "max_retries_per_missing_dimension": max_retries,
            "filled_null_counts": dict(filled_counter),
            "remaining_null_counts": dict(remaining_counter),
            "original_comparison_meta": comparison_data.get("meta", {}),
        },
        "comparison_summary": {
            "einv": build_model_summary(output_samples, "einv"),
            "v2t_base": build_model_summary(output_samples, "v2t_base"),
        },
        "human_summary": build_human_summary(output_samples),
        "null_counts_after_rescore": count_null_fields(output_samples),
        "samples": output_samples,
    }

    output_path = output_dir / f"{annotator_path.stem}_rescored_subset.json"
    output_path.write_text(json.dumps(output_data, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "annotator_file": str(annotator_path),
        "annotator_id": annotator_data.get("annotator_id", annotator_path.stem),
        "output_file": str(output_path),
        "subset_n": len(output_samples),
        "filled_null_counts": dict(filled_counter),
        "remaining_null_counts": dict(remaining_counter),
        "null_counts_after_rescore": count_null_fields(output_samples),
    }


def main(args):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required to rescore missing GPT fields.")

    annotator_paths = resolve_annotator_paths(args)
    comparison_path = Path(args.comparison)
    comparison_data = json.loads(comparison_path.read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = openai.OpenAI(api_key=api_key)
    score_cache = {}
    manifest = []

    for position, annotator_path in enumerate(annotator_paths, start=1):
        print(f"[{position}/{len(annotator_paths)}] Processing {annotator_path} ...")
        manifest.append(
            process_annotator_file(
                annotator_path=annotator_path,
                comparison_data=comparison_data,
                comparison_path=comparison_path,
                output_dir=output_dir,
                client=client,
                model=args.model,
                max_retries=args.max_retries,
                sleep_seconds=args.sleep_seconds,
                score_cache=score_cache,
            )
        )

    manifest_data = {
        "meta": {
            "comparison": args.comparison,
            "gpt_model_for_rescore": args.model,
            "max_retries_per_missing_dimension": args.max_retries,
            "annotator_file_count": len(manifest),
        },
        "annotators": manifest,
    }
    manifest_path = output_dir / "annotator_rescore_manifest.json"
    manifest_path.write_text(json.dumps(manifest_data, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse one or more annotator JSON files, extract their comparison samples, and fill missing GPT metric nulls."
    )
    parser.add_argument(
        "--annotators",
        nargs="+",
        default=["results/human_eval/hf78.json"],
        help="Annotator JSON files or glob patterns such as results/human_eval/hf*.json",
    )
    parser.add_argument(
        "--comparison",
        default="results/human_eval/wiki_sample250/v2t_base_comparison.json",
        help="Path to the full comparison JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/human_eval/wiki_sample250/annotator_rescored",
        help="Directory for per-annotator outputs and the manifest.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="OpenAI model used to fill missing GPT metrics.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=10,
        help="Retry count per missing dimension.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=1.0,
        help="Sleep between failed attempts.",
    )
    main(parser.parse_args())