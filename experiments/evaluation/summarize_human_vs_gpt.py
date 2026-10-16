import argparse
import csv
import json
import math
from pathlib import Path


DIMENSIONS = (
    ("structural_frame_similarity", "structure_score"),
    ("entity_preservation", "entity_score"),
    ("topic_consistency", "topic_score"),
)
MODELS = ("einv", "v2t_base")


def mean_std(values):
    if not values:
        return {"mean": None, "std": None, "n": 0}
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "mean": mean,
        "std": math.sqrt(variance),
        "n": len(values),
    }


def load_rescored_files(rescored_dir):
    files = sorted(rescored_dir.glob("*_rescored_subset.json"))
    if not files:
        raise RuntimeError(f"No rescored subset files found in {rescored_dir}")

    data = []
    for file_path in files:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        data.append((file_path, payload))
    return data


def common_indices(rescored_payloads):
    sets = []
    for _, payload in rescored_payloads:
        indices = {sample["index"] for sample in payload["samples"]}
        sets.append(indices)
    intersection = set.intersection(*sets)
    return sorted(intersection)


def build_summary(rescored_payloads, indices):
    summary_rows = []

    # Human: aggregate across all annotators on common indices.
    # GPT: aggregate once on common indices (deduplicated by index).
    first_samples_by_index = {
        sample["index"]: sample
        for sample in rescored_payloads[0][1]["samples"]
        if sample["index"] in indices
    }

    for model in MODELS:
        metric_key = f"{model}_metrics"

        for human_key, gpt_key in DIMENSIONS:
            annotator_means = []
            for _, payload in rescored_payloads:
                annotator_values = []
                for sample in payload["samples"]:
                    if sample["index"] not in indices:
                        continue
                    value = sample["annotation"][model][human_key]
                    if value is not None:
                        annotator_values.append(value / 5.0)
                if annotator_values:
                    annotator_means.append(sum(annotator_values) / len(annotator_values))

            gpt_values = []
            for index in indices:
                value = first_samples_by_index[index][metric_key][gpt_key]
                if value is not None:
                    gpt_values.append(value)

            human_stats = mean_std(annotator_means)
            gpt_stats = mean_std(gpt_values)

            summary_rows.append(
                {
                    "model": model,
                    "dimension": gpt_key,
                    "human_mean": human_stats["mean"],
                    "human_std": human_stats["std"],
                    "human_n": human_stats["n"],
                    "gpt_mean": gpt_stats["mean"],
                    "gpt_std": gpt_stats["std"],
                    "gpt_n": gpt_stats["n"],
                    "mean_gap_human_minus_gpt": (
                        (human_stats["mean"] - gpt_stats["mean"])
                        if (human_stats["mean"] is not None and gpt_stats["mean"] is not None)
                        else None
                    ),
                }
            )

    return summary_rows


def write_outputs(output_prefix, summary_rows, metadata):
    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")

    payload = {
        "meta": metadata,
        "summary": summary_rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "model",
                "dimension",
                "human_mean",
                "human_std",
                "human_n",
                "gpt_mean",
                "gpt_std",
                "gpt_n",
                "mean_gap_human_minus_gpt",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    return json_path, csv_path


def main():
    parser = argparse.ArgumentParser(
        description="Summarize human-vs-GPT scores for multiple annotators on common samples."
    )
    parser.add_argument(
        "--rescored-dir",
        default="results/human_eval/wiki_sample250/annotator_rescored",
        help="Directory containing *_rescored_subset.json files.",
    )
    parser.add_argument(
        "--output-prefix",
        default="results/human_eval/wiki_sample250/annotator_rescored/human_vs_gpt_5annotators_summary",
        help="Output prefix for JSON/CSV.",
    )
    args = parser.parse_args()

    rescored_dir = Path(args.rescored_dir)
    output_prefix = Path(args.output_prefix)

    rescored_payloads = load_rescored_files(rescored_dir)
    indices = common_indices(rescored_payloads)
    if len(indices) != 200:
        print(f"Warning: common sample count is {len(indices)} (expected 200)")

    summary_rows = build_summary(rescored_payloads, indices)

    metadata = {
        "rescored_dir": str(rescored_dir),
        "annotator_count": len(rescored_payloads),
        "annotator_files": [str(file_path) for file_path, _ in rescored_payloads],
        "common_sample_count": len(indices),
        "common_indices": indices,
        "scale": "all means/std are normalized to [0,1]",
    }

    json_path, csv_path = write_outputs(output_prefix, summary_rows, metadata)
    print(f"Saved JSON summary: {json_path}")
    print(f"Saved CSV summary : {csv_path}")


if __name__ == "__main__":
    main()