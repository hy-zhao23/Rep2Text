"""CPU-only step: build REVERSED-text token shards for the Mistral 2M training pool.

For each (rank, batch) shard in tmp/embeddings/mistral7B/l10t16/:
  - load forward tokens (N, 16)
  - per row: decode -> word-split -> reverse -> re-tokenize with Mistral
            (max_length=16, no special tokens, right pad)
  - save reversed tokens (N, 16) under
    tmp/embeddings_reverse/mistral7B/l10t16/tokens_r{R}_b{B}_l16_o0.npz

The forward / encode step is intentionally NOT done here. Run it later on GPU
to fill in embeddings_r*_b*_l16_o0.npz with matching row order.

Sample order is preserved so existing pretrain_*_index_*.pkl files map onto
the new shards directly.

Usage:
  python -m experiments.datasets.reverse_mistral_tokens \
      [--shard-start S] [--shard-end E] [--workers W]

Resumable per shard (skips if output exists).
"""
import argparse
import os
import re
from pathlib import Path
from multiprocessing import Pool

import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

PROJ = Path(__file__).resolve().parents[2]
SRC_DIR = PROJ / "tmp/embeddings/mistral7B/l10t16"
DST_DIR = PROJ / "tmp/embeddings_reverse/mistral7B/l10t16"
DST_DIR.mkdir(parents=True, exist_ok=True)

EMB_MODEL = "mistralai/Mistral-7B-v0.1"
MAX_TOKENS = 16

# Set per-process tokenizer once
_TOKENIZER = None
_PAD_ID = None


def _init_worker(model_name, hf_home):
    global _TOKENIZER, _PAD_ID
    os.environ.setdefault("HF_HOME", hf_home)
    tok = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, cache_dir=hf_home,
        token=os.environ.get("HF_API_KEY"),
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    _TOKENIZER = tok
    _PAD_ID = tok.pad_token_id


def _process_row(token_row):
    """Decode forward tokens -> reverse words -> re-tokenize to fixed 16-token row."""
    ids = [int(t) for t in token_row if int(t) != _PAD_ID]
    text = _TOKENIZER.decode(ids, skip_special_tokens=True).strip()
    rev_text = " ".join(text.split()[::-1])
    enc = _TOKENIZER(
        rev_text,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=MAX_TOKENS,
        add_special_tokens=False,
    )
    return enc["input_ids"][0].astype(np.int32)


def list_shards():
    pat = re.compile(r"tokens_r(\d+)_b(\d+)_l16_o0\.npz")
    out = []
    for p in sorted(SRC_DIR.iterdir()):
        m = pat.match(p.name)
        if not m:
            continue
        out.append((int(m[1]), int(m[2])))
    out.sort()
    return out


def process_shard(rank, batch_id, workers):
    src_tok_path = SRC_DIR / f"tokens_r{rank}_b{batch_id}_l16_o0.npz"
    dst_tok_path = DST_DIR / f"tokens_r{rank}_b{batch_id}_l16_o0.npz"
    if dst_tok_path.exists():
        return f"skip r{rank}_b{batch_id} (exists)"

    src_tokens = np.load(src_tok_path)["tokens"]
    n = src_tokens.shape[0]

    out = np.empty((n, MAX_TOKENS), dtype=np.int32)

    rows_iter = (src_tokens[i] for i in range(n))

    if workers <= 1:
        # serial path
        global _TOKENIZER, _PAD_ID
        _init_worker(EMB_MODEL, os.environ.get("HF_HOME", str(PROJ / "tmp/huggingface")))
        for i, row in enumerate(tqdm(rows_iter, total=n, desc=f"r{rank}_b{batch_id}", leave=False)):
            out[i] = _process_row(row)
    else:
        with Pool(
            workers,
            initializer=_init_worker,
            initargs=(EMB_MODEL, os.environ.get("HF_HOME", str(PROJ / "tmp/huggingface"))),
        ) as pool:
            for i, row in enumerate(tqdm(
                pool.imap(_process_row, rows_iter, chunksize=512),
                total=n, desc=f"r{rank}_b{batch_id}", leave=False,
            )):
                out[i] = row

    tmp_path = dst_tok_path.with_suffix(".tmp.npz")
    np.savez(tmp_path, tokens=out)
    tmp_path.replace(dst_tok_path)
    return f"done r{rank}_b{batch_id} (n={n})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-start", type=int, default=0)
    parser.add_argument("--shard-end", type=int, default=-1, help="exclusive; -1 = all")
    parser.add_argument("--workers", type=int, default=8, help="processes per shard for retokenization")
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", str(PROJ / "tmp/huggingface"))

    shards = list_shards()
    if args.shard_end == -1:
        args.shard_end = len(shards)
    shards = shards[args.shard_start:args.shard_end]
    print(f"Processing {len(shards)} shard(s) with workers={args.workers}")
    print(f"src={SRC_DIR}\ndst={DST_DIR}")

    for (r, b) in shards:
        msg = process_shard(r, b, args.workers)
        print(msg, flush=True)


if __name__ == "__main__":
    main()
