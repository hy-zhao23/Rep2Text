"""Build a sentence-aligned 16-token eval set from unused Wikipedia articles.

For each article in the held-out wiki TEST split:
  1. Drop the title (the chunk before the first ``\n\n``).
  2. Sentence-split the body.
  3. Find the longest consecutive prefix of sentences that ends in .!? and
     tokenizes (llama-3.1-8B, no special tokens) to <= MAX_TOK tokens.
  4. Require >= MIN_TOK tokens so we don't keep trivially short fragments.

Collects N_CAND such candidates, then random-samples N_SAMPLE for inversion.

Outputs:
  results/llama8B/l10t16p16/full_sentence/sentences.json       (N_CAND)
  results/llama8B/l10t16p16/full_sentence/sentences_500.json   (N_SAMPLE)
"""
import json
import os
import re
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc
from transformers import AutoTokenizer
from tqdm import tqdm
import nltk
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import sent_tokenize

PROJ = Path(__file__).resolve().parents[2]
ARROW_DIR = PROJ / "datasets/wikipedia/ArmelR___the-pile-splitted/Wikipedia (en)/0.0.0/970296c12832ff2131ea13a04c35d8949cd7ee1e"
OUT_DIR = PROJ / "results/llama8B/l10t16p16/full_sentence"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMB_MODEL = "meta-llama/Llama-3.1-8B"
MAX_TOK = 16
MIN_TOK = 6
N_CAND = 1000
N_SAMPLE = 500
SEED = 0

def split_sentences(body: str):
    body = body.strip()
    if not body:
        return []
    # nltk's punkt handles abbreviations like Mr., U.S., C. correctly.
    return [s.strip() for s in sent_tokenize(body) if s.strip()]


def truncate_to_full_sentences(body: str, tokenizer, max_tok: int, min_tok: int):
    """Return the longest sentence-aligned prefix that fits in [min_tok, max_tok]."""
    sents = split_sentences(body)
    if not sents:
        return None
    chunk = ""
    last_ok = None
    for s in sents:
        candidate = (chunk + " " + s).strip() if chunk else s
        # Require sentence-final punctuation
        if not re.search(r"[.!?][\"'\)\]]?$", candidate):
            break
        n_tok = len(tokenizer.encode(candidate, add_special_tokens=False))
        if n_tok > max_tok:
            break
        chunk = candidate
        last_ok = (chunk, n_tok)
    if last_ok is None:
        return None
    text, n_tok = last_ok
    if n_tok < min_tok:
        return None
    return text, n_tok


def iter_test_arrow():
    files = sorted(ARROW_DIR.glob("the-pile-splitted-test-*.arrow"))
    for path in files:
        with pa.memory_map(str(path), "r") as src:
            reader = pa.ipc.open_stream(src)
            tbl = reader.read_all()
        texts = tbl.column("text").to_pylist()
        for local_idx, text in enumerate(texts):
            yield path.name, local_idx, text


def main():
    print(f"Loading tokenizer: {EMB_MODEL}")
    tok = AutoTokenizer.from_pretrained(
        EMB_MODEL, use_fast=True, token=os.environ.get("HF_API_KEY")
    )

    candidates = []
    seen_texts = set()
    pbar = tqdm(total=N_CAND, desc="Collecting candidates")
    for orig_idx, (shard, local_idx, text) in enumerate(iter_test_arrow()):
        if not isinstance(text, str) or "\n\n" not in text:
            continue
        # Drop title -> body is everything after first \n\n
        body = text.split("\n\n", 1)[1].strip()
        # Take just the first paragraph to avoid mixing in section headers
        body = body.split("\n\n", 1)[0].strip()
        if len(body) < 20:
            continue
        result = truncate_to_full_sentences(body, tok, MAX_TOK, MIN_TOK)
        if result is None:
            continue
        truncated, n_tok = result
        # Dedup on the truncated text
        key = truncated.lower()
        if key in seen_texts:
            continue
        seen_texts.add(key)
        candidates.append({
            "orig_idx": orig_idx,
            "shard": shard,
            "local_idx": local_idx,
            "n_tokens": n_tok,
            "gt_orig": body[:400],
            "gt_truncated": truncated,
        })
        pbar.update(1)
        if len(candidates) >= N_CAND:
            break
    pbar.close()

    if len(candidates) < N_CAND:
        print(f"WARNING: only collected {len(candidates)}/{N_CAND} candidates")

    out_all = OUT_DIR / "sentences.json"
    with open(out_all, "w") as f:
        json.dump(candidates, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(candidates)} candidates to {out_all}")

    # Random sample N_SAMPLE
    rng = random.Random(SEED)
    sampled = rng.sample(candidates, min(N_SAMPLE, len(candidates)))
    out_500 = OUT_DIR / f"sentences_{N_SAMPLE}.json"
    with open(out_500, "w") as f:
        json.dump(sampled, f, indent=2, ensure_ascii=False)
    print(f"Sampled {len(sampled)} -> {out_500}")

    # Quick stats
    lens = [c["n_tokens"] for c in candidates]
    print(f"Token length: min={min(lens)} max={max(lens)} mean={sum(lens)/len(lens):.1f}")


if __name__ == "__main__":
    main()
