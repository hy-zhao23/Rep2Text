from datasets import load_dataset
import os
import json
import random
from openai import OpenAI
from typing import Any, cast
from utils.core import write_json
from utils import p_args

SAMPLE_SEED = 42
SAMPLE_SIZE = 2500


def get_clinical_notes(already_sampled: set[int] | None = None):
    """Return (new_indices, new_notes) sampled from indices NOT yet in already_sampled.

    Using a fixed seed on the *remaining* pool means each new call appends a
    deterministic, non-overlapping batch — no previously-processed note is
    ever re-summarized.
    """
    ds = load_dataset("Sadaftb/clinical-nlp-patient-notes", split="train")
    notes = list(cast(Any, ds["pn_history"]))

    already_sampled = already_sampled or set()
    remaining = [i for i in range(len(notes)) if i not in already_sampled]

    n_new = min(SAMPLE_SIZE, len(remaining))
    rng = random.Random(SAMPLE_SEED + len(already_sampled))  # shift seed per batch
    new_indices = rng.sample(remaining, n_new)
    new_notes = [notes[i] for i in new_indices]
    return new_indices, new_notes


def parse_summary(summary):
    if summary.startswith("[ans]"):
        return summary[5:].strip()
    return None


def gpt_summary(note, _model, client=None):

    prompt = f"""You're given a piece of clinical note in the following context. Summarize the clinical note into ONE fluent sentence (<=25 tokens). Must include: patient name, date of birth, gender, and key symptoms. Start with \"[ans]\". Output only the summary.\n\n{note}"""
    i = 0

    while i < 10:
        response = client.chat.completions.create(
            model=_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes clinical notes."},
                {"role": "user", "content": f"{prompt}"}
            ],
            temperature=0.7,
            max_tokens=150
        )

        summary = parse_summary(response.choices[0].message.content or "")
        if summary is not None:
            return summary
        i += 1
    return None


def main():
    out_dir = p_args.path.data_dir / "clinical_notes"
    os.makedirs(out_dir, exist_ok=True)

    summary_path = out_dir / "clinical_notes_summary.json"
    index_path   = out_dir / "clinical_notes_indices.json"

    # ── Load already-completed entries ────────────────────────────────────
    summaries: list[dict] = []
    done_indices: set[int] = set()
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summaries = json.load(f)
        for row in summaries:
            if isinstance(row.get("dataset_idx"), int):
                done_indices.add(row["dataset_idx"])
        print(f"Resuming: {len(done_indices)} entries already summarized.")
    else:
        print("No existing summary file — starting fresh.")

    # ── Sample only from indices NOT yet processed ─────────────────────
    new_indices, new_notes = get_clinical_notes(already_sampled=done_indices)
    print(f"New indices to summarize: {len(new_indices)}")

    # Persist the full accumulated index list
    all_indices = sorted(done_indices | set(new_indices))
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(all_indices, f)

    # ── Summarize only new notes, append incrementally ─────────────────
    _model = os.getenv("GPT_ID", "gpt-4.1-mini")
    api_key = os.getenv("OPENAI_API_KEY") or p_args.settings.openai_key
    client = OpenAI(api_key=api_key) if api_key else OpenAI()
    for pos, (idx, note) in enumerate(zip(new_indices, new_notes)):
        summary = gpt_summary(note, client=client, _model=_model)
        row = {"dataset_idx": idx, "note": note, "summary": summary}
        summaries.append(row)
        print(f"[+{pos}] idx={idx} — {summary}")

        # Incremental save so progress survives job interruption
        write_json(summaries, str(summary_path))

    write_json(summaries, str(summary_path))
    print(f"\nDone. {len(summaries)} total entries written to {summary_path}")
    print(f"Index file: {index_path}")


if __name__ == "__main__":
    main()



