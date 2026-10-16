"""Collect 500 sentence-aligned Wikipedia samples with token length >= 14
(close to the 16-token cap). Reuses the same logic as build_16token_eval.py.
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
OUT_DIR = PROJ / "datasets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMB_MODEL = "meta-llama/Llama-3.1-8B"
MAX_TOK = 16
MIN_TOK = 14  # only "long" sentences close to 16 tokens
N_TARGET = 500
SEED = 0


def split_sentences(body: str):
    body = body.strip()
    if not body:
        return []
    return [s.strip() for s in sent_tokenize(body) if s.strip()]


def truncate_to_full_sentences(body, tokenizer, max_tok, min_tok):
    sents = split_sentences(body)
    if not sents:
        return None
    chunk = ""
    last_ok = None
    for s in sents:
        candidate = (chunk + " " + s).strip() if chunk else s
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
    seen = set()
    pbar = tqdm(total=N_TARGET, desc="Collecting >=14-tok sentences")
    for orig_idx, (shard, local_idx, text) in enumerate(iter_test_arrow()):
        if not isinstance(text, str) or "\n\n" not in text:
            continue
        body = text.split("\n\n", 1)[1].strip()
        body = body.split("\n\n", 1)[0].strip()
        if len(body) < 20:
            continue
        result = truncate_to_full_sentences(body, tok, MAX_TOK, MIN_TOK)
        if result is None:
            continue
        truncated, n_tok = result
        key = truncated.lower()
        if key in seen:
            continue
        seen.add(key)
        candidates.append({
            "orig_idx": orig_idx,
            "shard": shard,
            "local_idx": local_idx,
            "n_tokens": n_tok,
            "gt_truncated": truncated,
        })
        pbar.update(1)
        if len(candidates) >= N_TARGET:
            break
    pbar.close()

    print(f"Collected {len(candidates)} candidates")
    out_json = OUT_DIR / "full_sentence_3b_long_500.json"
    with open(out_json, "w") as f:
        json.dump(candidates, f, indent=2, ensure_ascii=False)
    out_txt = OUT_DIR / "full_sentence_3b_long_500.txt"
    with open(out_txt, "w") as f:
        for r in sorted(candidates, key=lambda x: -x["n_tokens"]):
            f.write(f"[{r['n_tokens']:>2}] {r['gt_truncated']}\n")
    print(f"Saved -> {out_json}")
    print(f"Saved -> {out_txt}")
    lens = [c["n_tokens"] for c in candidates]
    from collections import Counter
    print("len dist:", sorted(Counter(lens).items()))


if __name__ == "__main__":
    main()
