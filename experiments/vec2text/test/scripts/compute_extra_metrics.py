#!/usr/bin/env python3
"""
Compute additional metrics on an existing results JSON file:
  - ROUGE-1, ROUGE-2, ROUGE-L   (rouge_score)
  - BERTScore F1                 (bert_score, microsoft/deberta-v3-large)
  - Structure score  0-5         (GPT via utils.eval.GPTScorer)
  - Entity score     0-5         (GPT via utils.eval.GPTScorer)
  - Topic score      0-5         (GPT via utils.eval.GPTScorer)

GPT scores are stored normalized to [0, 1] (div 5), matching the embedding
project convention in scripts/evaluation/eval_gpt.py.

A summary JSON is written alongside the main output file, e.g.:
  results_wikipedia_val.json  ->  results_wikipedia_val_summary.json

Use --domain clinical to apply clinical-specific entity/topic prompts.

Usage
-----
    python test/scripts/compute_extra_metrics.py \
        --input  test/results/.../results_wikipedia_val.json \
        [--output  test/results/.../results_wikipedia_val_full.json] \
        [--openai_key sk-...] \
        [--gpt_model gpt-4.1-mini] \
        [--bertscore_model microsoft/deberta-v3-large] \
        [--bertscore_batch 64] \
        [--max_samples 1000] \
        [--domain wikipedia|clinical] \
        [--skip_gpt] \
        [--skip_bertscore]
"""

import argparse
import json
import math
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import re
import traceback

import numpy as np
import openai
import torch
from rouge_score import rouge_scorer
from tqdm import tqdm


# -- Self-contained GPTScorer (no dependency on embedding project settings) --

class GPTScorer:
    DOMAINS = ("wikipedia", "clinical")

    def __init__(self, gt_sen, gen_sen, domain="wikipedia"):
        if domain not in self.DOMAINS:
            raise ValueError(f"domain must be one of {self.DOMAINS}, got '{domain}'")
        self.gt_sen = gt_sen
        self.gen_sen = gen_sen
        self.domain = domain
        self.client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.structure_score = None
        self.entity_score = None
        self.topic_score = None

    @property
    def structure_prompt(self):
        return f"""
            You are given two sentences:
            [GT]: {self.gt_sen}

            [GEN]: {self.gen_sen}

            Evaluate the **Structural Frame Similarity** between these sentences.

            Focus on whether [GEN] preserves the core grammatical structure and sentence skeleton of [GT]:
                - Same basic clause structure (e.g., "X was Y of Z" pattern maintained)
                - Same number and type of major phrases (noun phrases, verb phrases in similar positions)
                - Similar syntactic relationships between components
                - Consistent verb tense/aspect (past vs present, perfect vs simple)

            **Scoring Guidelines:**
                - **5**: Identical structure - with almost exactly the same words in the same order
                - **4**: Identical or nearly identical structure - same sentence pattern with only entity substitutions, minor variations like reordering of parallel elements or addition/removal of one modifier
                - **3**: Moderately similar - core structure maintained but with notable changes (e.g., active to passive, clause reordering)
                - **2**: Somewhat similar - recognizable structural elements but significant differences
                - **1**: Minimally similar - only basic sentence type matches (e.g., both declarative)
                - **0**: Completely different structures

            Answer in the following format:
            [ANS] structure: [score]/5
        """

    @property
    def entity_prompt(self):
        if self.domain == "clinical":
            return self._entity_prompt_clinical
        return self._entity_prompt_wikipedia

    @property
    def _entity_prompt_wikipedia(self):
        return f"""
            You are given two sentences:
            [GT]: {self.gt_sen}
            [GEN]: {self.gen_sen}

            Evaluate the **Entity/Role Consistency and Plausibility** between these sentences.

            Assess whether entities in [GEN] are appropriate replacements for those in [GT], considering:

            **Entity Type Matching Rules:**
                - **Person**: Same role category (athlete↔athlete, scholar↔scholar), similar fame domain, appropriate gender/nationality for context
                - **Location**: Same geographic level (city↔city, country↔country), reasonable proximity or same region
                - **Date/Time**: Numerical proximity (±5 years = high similarity, ±10 years = moderate, ±20 years = low)
                - **Organization**: Same category (university↔university, company↔company, sports team↔sports team)
                - **Event**: Same type (sports↔sports, war↔war, festival↔festival), similar scale/importance
                - **Product/Work**: Same category (game↔game, book↔book, film↔film), similar medium or genre
                - **Concept**: Same domain (math term↔math term, scientific concept↔related concept)

            **Scoring Guidelines:**
                - **5**: Perfect or near-perfect entity matches - all entities are equivalent or highly similar in type and role
                - **4**: Strong matches - entities are contextually appropriate with minor differences (e.g., similar universities, nearby years)
                - **3**: Moderate matches - entities are plausible but with notable differences (e.g., different sports, ±15 years, different genre)
                - **2**: Weak matches - some entities match type but with significant mismatches (e.g., athlete→politician, city→country)
                - **1**: Poor matches - most entities are wrong type but not completely nonsensical
                - **0**: Nonsensical replacements - entities make no sense in context or are completely wrong types

            **Note**: Minor variations within the same category should NOT be penalized heavily. "Asbury College" → "Southern Methodist University" is a strong match (both universities).

            Answer in the following format:
            [ANS] entity: [score]/5
        """

    @property
    def _entity_prompt_clinical(self):
        return f"""
            You are given two clinical text snippets:
            [GT]: {self.gt_sen}
            [GEN]: {self.gen_sen}

            Evaluate the **Entity/Role Consistency and Plausibility** between these snippets.

            Assess whether entities in [GEN] are appropriate replacements for those in [GT], considering:

            **Clinical Entity Type Matching Rules:**
                - **Patient Demographics**: Age (±5 years = high similarity, ±10 years = moderate), same biological sex, similar DOB precision
                - **Symptoms / Chief Complaint**: Same symptom category (chest pain↔chest tightness, SOB↔dyspnea, nausea↔vomiting); penalise crossing body systems (e.g., chest↔abdominal)
                - **Diagnosis / Condition**: Same condition class (respiratory infection↔respiratory infection, cardiac↔cardiac, neurological↔neurological); similar severity (acute↔acute, chronic↔chronic)
                - **Medications**: Same drug class or mechanism (antibiotic↔antibiotic, SSRI↔SSRI, diuretic↔diuretic); penalise entirely different classes
                - **Vital Signs / Lab Values**: Numerical proximity (BP ±10 mmHg, O2 sat ±3%, glucose ±30 mg/dL); same parameter type
                - **Anatomical Location**: Same body region or system (left lower lobe↔right lower lobe, abdomen↔abdomen); penalise crossing systems
                - **Dates / Admission Timing**: Numerical proximity (±30 days = high, ±1 year = moderate, >5 years = low)
                - **Medical Procedures**: Same procedure category (surgery↔surgery, imaging↔imaging, IV therapy↔IV therapy)

            **Scoring Guidelines:**
                - **5**: All entities match in type and clinical role — equivalent or near-equivalent replacements
                - **4**: Strong matches — same categories with minor differences (e.g., slightly different age, similar drug class)
                - **3**: Moderate matches — entities are plausible but with notable mismatches (e.g., different body system, different drug class)
                - **2**: Weak matches — some entities align but significant clinical mismatches exist (e.g., symptom↔diagnosis, medication↔lab value)
                - **1**: Poor matches — most entity types are wrong, though not nonsensical in a medical context
                - **0**: Nonsensical — entities are medically incoherent or completely unrelated to clinical text

            Answer in the following format:
            [ANS] entity: [score]/5
        """

    @property
    def topic_prompt(self):
        if self.domain == "clinical":
            return self._topic_prompt_clinical
        return self._topic_prompt_wikipedia

    @property
    def _topic_prompt_wikipedia(self):
        return f"""
            You are given two sentences:
            [GT]: {self.gt_sen}
            [GEN]: {self.gen_sen}

            Evaluate the **Topical Relevance** between these sentences.

            Assess whether [GEN] maintains the same overall subject matter, theme, or knowledge domain as [GT].

            **Topic Categories to Consider:**
                - Biography/People (life events, achievements, roles)
                - Geography/Locations (places, regions, spatial relationships)
                - Organizations/Institutions (companies, universities, government bodies)
                - Historical/Contemporary Events (wars, elections, ceremonies)
                - Arts & Entertainment (films, music, literature, games)
                - Sports & Athletics (competitions, teams, athletes)
                - Science & Technology (concepts, discoveries, technical topics)
                - Nature & Biology (species, phenomena, ecosystems)
                - Products & Brands (commercial items, creative works)

            **Scoring Guidelines:**
                - **5**: Same specific topic - discussing the exact same subject or very closely related subtopic
                - **4**: Same general topic - within the same domain with minor shifts
                - **3**: Related topics - share a broader category but different focus
                - **2**: Loosely related - some thematic connection but substantially different topics
                - **1**: Minimally related - only the most general connection
                - **0**: Unrelated - completely different topics with no meaningful connection

            Answer in the following format:
            [ANS] topic: [score]/5
        """

    @property
    def _topic_prompt_clinical(self):
        return f"""
            You are given two clinical text snippets:
            [GT]: {self.gt_sen}
            [GEN]: {self.gen_sen}

            Evaluate the **Topical Relevance** between these snippets.

            Assess whether [GEN] maintains the same clinical subject matter, disease domain, or care context as [GT].

            **Scoring Guidelines:**
                - **5**: Same specific clinical topic — same disease, condition, or care scenario
                - **4**: Same clinical domain with minor shifts
                - **3**: Related clinical domains — share a broader category but different focus
                - **2**: Loosely related — some medical connection but substantially different clinical contexts
                - **1**: Minimally related — both are clinical text but discuss completely different physiological systems
                - **0**: Unrelated — no meaningful clinical or topical connection

            Answer in the following format:
            [ANS] topic: [score]/5
        """

    def gpt_rating(self, max_retries=10):
        self.structure_score = self._gpt_rating(self.structure_prompt, max_retries)
        self.entity_score    = self._gpt_rating(self.entity_prompt,    max_retries)
        self.topic_score     = self._gpt_rating(self.topic_prompt,     max_retries)

    def _gpt_rating(self, prompt, max_retries):
        _model = os.environ.get("GPT_ID", "gpt-4.1-mini")
        for _ in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=_model,
                )
                result = resp.choices[0].message.content
                score = self._extract_score(result)
                if score is not None:
                    return score
            except Exception as e:
                print(f"[GPTScorer] error: {e}", file=sys.stderr)
                traceback.print_exc()
        print("[GPTScorer] Failed to get valid score after retries", file=sys.stderr)
        return None

    def _extract_score(self, result):
        parts = result.split("[ANS]")
        if len(parts) < 2:
            return None
        match = re.search(r'(\d)\s*/\s*5', parts[-1].strip())
        return int(match.group(1)) if match else None


# -- ROUGE ---------------------------------------------------------------

def compute_rouge_batch(preds, refs, metrics=("rouge1", "rouge2", "rougeL")):
    scorer = rouge_scorer.RougeScorer(list(metrics), use_stemmer=True)
    results = {m: [] for m in metrics}
    for pred, ref in zip(preds, refs):
        scores = scorer.score(ref, pred)
        for m in metrics:
            results[m].append(scores[m].fmeasure)
    return results  # dict[str, list[float]], values in [0, 1]


# -- BERTScore -----------------------------------------------------------

def compute_bertscore_batch(preds, refs, model_type, batch_size, device):
    from bert_score import score as bscore
    P, R, F1 = bscore(
        preds, refs,
        lang="en",
        model_type=model_type,
        batch_size=batch_size,
        verbose=False,
        device=device,
    )
    return F1.tolist()  # list[float] in [0, 1]


# -- GPT scores via GPTScorer --------------------------------------------

def compute_gpt_scores_batch(preds, refs, domain, gpt_model,
                             samples=None, out_path=None,
                             save_every=10, workers=1):
    """
    Returns three lists: structure, entity, topic (each float in [0,1] or None).
    Scores are divided by 5 to match the embedding project eval_gpt.py convention.

    Resumable: if `samples` already have structure_score/entity_score/topic_score,
    those rows are reused (skipped). Every `save_every` newly-scored rows, the
    updated samples (with current ROUGE/BERTScore/GPT) are flushed to `out_path`.

    `workers > 1` uses a ThreadPoolExecutor to score multiple samples concurrently.
    """
    os.environ["GPT_ID"] = gpt_model
    n = len(preds)

    structure = [None] * n
    entity    = [None] * n
    topic     = [None] * n

    # seed from existing samples (resume)
    resumed = 0
    pending = []
    if samples is not None:
        for i, s in enumerate(samples):
            if (s.get("structure_score") is not None or
                s.get("entity_score")    is not None or
                s.get("topic_score")     is not None):
                structure[i] = s.get("structure_score")
                entity[i]    = s.get("entity_score")
                topic[i]     = s.get("topic_score")
                resumed += 1
            else:
                pending.append(i)
    else:
        pending = list(range(n))
    if resumed:
        print(f"  [resume] reusing {resumed}/{n} existing GPT scores")

    def _score_one(idx):
        scorer = GPTScorer(refs[idx], preds[idx], domain=domain)
        scorer.gpt_rating()
        s = scorer.structure_score / 5.0 if scorer.structure_score is not None else None
        e = scorer.entity_score    / 5.0 if scorer.entity_score    is not None else None
        t = scorer.topic_score     / 5.0 if scorer.topic_score     is not None else None
        return idx, s, e, t

    save_lock = threading.Lock()
    new_since_save = 0

    def _maybe_flush(progress):
        nonlocal new_since_save
        if out_path is None or samples is None:
            return
        if new_since_save < save_every:
            return
        tmp = str(out_path) + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"samples": samples, "_partial_gpt_progress": progress, "_total": n}, f, indent=2, ensure_ascii=False)
        os.replace(tmp, out_path)
        new_since_save = 0

    pbar = tqdm(total=len(pending), desc=f"  GPT scoring (workers={workers})", ncols=80)
    if workers <= 1:
        for idx in pending:
            _, s, e, t = _score_one(idx)
            structure[idx], entity[idx], topic[idx] = s, e, t
            if samples is not None:
                samples[idx]["structure_score"] = s
                samples[idx]["entity_score"]    = e
                samples[idx]["topic_score"]     = t
            new_since_save += 1
            _maybe_flush(idx + 1)
            pbar.update(1)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_score_one, idx): idx for idx in pending}
            max_done_idx = -1
            for fut in as_completed(futures):
                idx, s, e, t = fut.result()
                with save_lock:
                    structure[idx], entity[idx], topic[idx] = s, e, t
                    if samples is not None:
                        samples[idx]["structure_score"] = s
                        samples[idx]["entity_score"]    = e
                        samples[idx]["topic_score"]     = t
                    new_since_save += 1
                    if idx > max_done_idx:
                        max_done_idx = idx
                    _maybe_flush(max_done_idx + 1)
                pbar.update(1)
    pbar.close()
    return structure, entity, topic


# -- summary helpers (embedding project style) ---------------------------

def _mean_std(values):
    """Compute mean/std/count over a flat or nested list, skipping None."""
    flat = []
    for v in values:
        if v is None:
            continue
        if isinstance(v, list):
            flat.extend(v)
        else:
            flat.append(v)
    if not flat:
        return {"mean": None, "std": None, "count": 0}
    mean = sum(flat) / len(flat)
    var  = sum((x - mean) ** 2 for x in flat) / len(flat)
    return {"mean": mean, "std": math.sqrt(var), "count": len(flat)}


def build_summary(rouge, bs_f1, structure_scores, entity_scores, topic_scores,
                  existing_metrics, skip_bertscore, skip_gpt, samples=None):
    """Return an embedding-style summary dict."""
    summary = {}

    # pass-through existing scalar metrics (bleu, token_f1, exact_match, ...)
    # For token_f1 and bleu, compute std from per-sample values if available.
    per_sample_token_f1 = (
        [s["token_f1"] * 100 for s in samples if "token_f1" in s]
        if samples else []
    )
    per_sample_bleu = (
        [s["bleu"] for s in samples if "bleu" in s]
        if samples else []
    )

    for k, v in existing_metrics.items():
        if k == "token_f1" and per_sample_token_f1:
            summary[k] = _mean_std(per_sample_token_f1)
        elif k == "bleu" and per_sample_bleu:
            summary[k] = _mean_std(per_sample_bleu)
        elif isinstance(v, (int, float)):
            summary[k] = {"mean": v, "std": None, "count": None}
        elif isinstance(v, dict):
            summary[k] = v

    summary["ROUGE-1"]      = _mean_std(rouge["rouge1"])
    summary["ROUGE-2"]      = _mean_std(rouge["rouge2"])
    summary["ROUGE-L"]      = _mean_std(rouge["rougeL"])

    if not skip_bertscore:
        summary["BERTScore F1"] = _mean_std(bs_f1)

    if not skip_gpt:
        summary["Structure"] = _mean_std(structure_scores)
        summary["Entity"]    = _mean_std(entity_scores)
        summary["Topic"]     = _mean_std(topic_scores)

    return summary


# -- main ----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute extra metrics on results JSON")
    parser.add_argument("--input",  required=True, help="Path to results JSON file")
    parser.add_argument("--output", default=None,  help="Output path (default: overwrite input)")
    parser.add_argument("--openai_key",       default=None)
    parser.add_argument("--gpt_model",        default="gpt-4.1-mini")
    parser.add_argument("--bertscore_model",  default="microsoft/deberta-v3-large")
    parser.add_argument("--bertscore_batch",  type=int, default=64)
    parser.add_argument("--max_samples",      type=int, default=0, help="0 = all")
    parser.add_argument("--domain",           default="wikipedia",
                        choices=["wikipedia", "clinical"],
                        help="GPT prompt style: 'wikipedia' (default) or 'clinical'")
    parser.add_argument("--skip_gpt",         action="store_true")
    parser.add_argument("--skip_bertscore",   action="store_true")
    parser.add_argument("--gpt_workers",      type=int, default=1,
                        help="Concurrent threads for GPT scoring (default 1)")
    args = parser.parse_args()

    # set OpenAI key before GPTScorer is used
    api_key = args.openai_key or os.environ.get("OPENAI_API_KEY")
    if not api_key and not args.skip_gpt:
        raise ValueError("Provide --openai_key or set OPENAI_API_KEY env var")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    out_path     = args.output or args.input
    summary_path = str(Path(out_path).with_suffix("")) + "_summary.json"

    # load results
    with open(args.input) as f:
        data = json.load(f)

    samples = data["samples"]
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    preds = [s["prediction"]   for s in samples]
    refs  = [s["ground_truth"] for s in samples]
    n     = len(samples)

    print(f"\n  Samples   : {n}")
    print(f"  Output    : {out_path}")
    print(f"  Summary   : {summary_path}")
    print(f"  Domain    : {args.domain}")
    print(f"  BERTScore : {'skip' if args.skip_bertscore else args.bertscore_model}")
    print(f"  GPT model : {'skip' if args.skip_gpt else args.gpt_model}")

    # ROUGE
    print("\n[1/3] Computing ROUGE ...")
    rouge = compute_rouge_batch(preds, refs)
    for i, s in enumerate(samples):
        s["rouge1"] = rouge["rouge1"][i]
        s["rouge2"] = rouge["rouge2"][i]
        s["rougeL"] = rouge["rougeL"][i]
    print(f"  ROUGE-1 mean: {np.mean(rouge['rouge1'])*100:.2f}%")
    print(f"  ROUGE-2 mean: {np.mean(rouge['rouge2'])*100:.2f}%")
    print(f"  ROUGE-L mean: {np.mean(rouge['rougeL'])*100:.2f}%")

    # BERTScore
    bs_f1 = [None] * n
    if not args.skip_bertscore:
        print("\n[2/3] Computing BERTScore ...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        bs_f1 = compute_bertscore_batch(
            preds, refs,
            model_type=args.bertscore_model,
            batch_size=args.bertscore_batch,
            device=device,
        )
        for i, s in enumerate(samples):
            s["bertscore_f1"] = bs_f1[i]
        print(f"  BERTScore F1 mean: {np.mean(bs_f1)*100:.2f}%")
    else:
        print("\n[2/3] BERTScore skipped.")

    # GPT scores
    structure_scores = [None] * n
    entity_scores    = [None] * n
    topic_scores     = [None] * n

    if not args.skip_gpt:
        print("\n[3/3] Computing GPT scores ...")
        structure_scores, entity_scores, topic_scores = compute_gpt_scores_batch(
            preds, refs, domain=args.domain, gpt_model=args.gpt_model,
            samples=samples, out_path=out_path, save_every=10,
            workers=args.gpt_workers,
        )
        for i, s in enumerate(samples):
            s["structure_score"] = structure_scores[i]  # normalized [0, 1]
            s["entity_score"]    = entity_scores[i]
            s["topic_score"]     = topic_scores[i]

        valid_s = [v for v in structure_scores if v is not None]
        valid_e = [v for v in entity_scores    if v is not None]
        valid_t = [v for v in topic_scores     if v is not None]
        print(f"  Structure mean : {np.mean(valid_s):.3f}  (n={len(valid_s)})")
        print(f"  Entity    mean : {np.mean(valid_e):.3f}  (n={len(valid_e)})")
        print(f"  Topic     mean : {np.mean(valid_t):.3f}  (n={len(valid_t)})")
    else:
        print("\n[3/3] GPT scoring skipped.")

    # save main results
    data["samples"] = samples
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved results to : {out_path}")

    # build & save summary (embedding project style)
    summary = build_summary(
        rouge, bs_f1, structure_scores, entity_scores, topic_scores,
        existing_metrics=data.get("metrics", {}),
        skip_bertscore=args.skip_bertscore,
        skip_gpt=args.skip_gpt,
        samples=samples,
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Saved summary to : {summary_path}")

    # print summary table
    print("\n  -- Summary ------------------------------------------")
    for name, stats in summary.items():
        if stats is None or stats.get("mean") is None:
            continue
        mean = stats["mean"]
        std  = stats.get("std")
        cnt  = stats.get("count", "")
        std_str = f" +/- {std:.4f}" if std is not None else ""
        print(f"  {name:<18}: {mean:.4f}{std_str}  (n={cnt})")
    print()


if __name__ == "__main__":
    main()
