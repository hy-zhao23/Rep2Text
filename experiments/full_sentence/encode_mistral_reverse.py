"""Encode 500 long-sentence texts with Mistral-7B layer 10 in TWO variants:

  1. ORIGINAL  -> tmp/embeddings/full_sentence/mistral7b_l16/
  2. REVERSED  -> tmp/embeddings/full_sentence/mistral7b_reverse_l16/

Reversal: word-split on whitespace, reverse the word list, rejoin with spaces,
then re-tokenize with the Mistral tokenizer at max_length=16 (no special tokens).

Stored last-token hidden state corresponds to the m3b transfer setup
(mistral encoder -> Llama-3.2-3B decoder, max_len=16) so embeddings can be
inverted by the 640k m3b baseline at weights/mistral7B/l10t16p16/transfer-m3b/.
"""
import json
import os
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

PROJ = Path(__file__).resolve().parents[2]
EMB_MODEL = "mistralai/Mistral-7B-v0.1"
LAYER = 10
MAX_TOKENS = 16
BATCH = 8

SENT_PATH = PROJ / "datasets/full_sentence_3b_long_500.json"
OUT_BASE = PROJ / "tmp/embeddings/full_sentence"
ORIG_DIR = OUT_BASE / "mistral7b_l16"
REV_DIR = OUT_BASE / "mistral7b_reverse_l16"
ORIG_DIR.mkdir(parents=True, exist_ok=True)
REV_DIR.mkdir(parents=True, exist_ok=True)


def reverse_words(text: str) -> str:
    return " ".join(text.split()[::-1])


def encode(texts, tok, model, device, hidden_dim, desc):
    embs = np.zeros((len(texts), hidden_dim), dtype=np.float32)
    toks = np.full((len(texts), MAX_TOKENS), tok.pad_token_id, dtype=np.int32)
    for i in tqdm(range(0, len(texts), BATCH), desc=desc):
        batch = texts[i:i + BATCH]
        enc = tok(
            batch,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MAX_TOKENS,
            add_special_tokens=False,
        ).to(device)
        with torch.no_grad():
            out = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                output_hidden_states=True,
                return_dict=True,
            )
        hidden = out.hidden_states[LAYER + 1]
        mask = enc["attention_mask"]
        seq_lens = mask.shape[1] - 1 - mask.flip(dims=(1,)).argmax(dim=1)
        last = hidden[torch.arange(len(batch), device=device), seq_lens]
        embs[i:i + len(batch)] = last.float().cpu().numpy()
        toks[i:i + len(batch)] = enc["input_ids"].cpu().numpy().astype(np.int32)
    return embs, toks


def save(out_dir, embs, toks):
    emb_path = out_dir / f"embeddings_r0_b0_l{MAX_TOKENS}_o0.npz"
    tok_path = out_dir / f"tokens_r0_b0_l{MAX_TOKENS}_o0.npz"
    np.savez(emb_path, embeddings=embs)
    np.savez(tok_path, tokens=toks)
    print(f"  embeddings -> {emb_path}  shape={embs.shape}")
    print(f"  tokens     -> {tok_path}  shape={toks.shape}")


def main():
    os.environ.setdefault("HF_HOME", str(PROJ / "tmp/huggingface"))

    with open(SENT_PATH) as f:
        sentences = json.load(f)
    orig_texts = [s["gt_truncated"] for s in sentences]
    rev_texts = [reverse_words(t) for t in orig_texts]
    print(f"Loaded {len(orig_texts)} sentences from {SENT_PATH}")
    print(f"orig[0]: {orig_texts[0]}")
    print(f"rev[0] : {rev_texts[0]}")

    tok = AutoTokenizer.from_pretrained(
        EMB_MODEL, use_fast=True, cache_dir=os.environ["HF_HOME"],
        token=os.environ.get("HF_API_KEY"),
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"Loading {EMB_MODEL} ...")
    model = AutoModelForCausalLM.from_pretrained(
        EMB_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        cache_dir=os.environ["HF_HOME"],
        token=os.environ.get("HF_API_KEY"),
    )
    model.eval()
    device = next(model.parameters()).device
    hidden_dim = model.config.hidden_size
    print(f"hidden_dim={hidden_dim}")

    print("[1/2] Encoding ORIGINAL ...")
    embs_o, toks_o = encode(orig_texts, tok, model, device, hidden_dim, "orig")
    save(ORIG_DIR, embs_o, toks_o)

    print("[2/2] Encoding REVERSED ...")
    embs_r, toks_r = encode(rev_texts, tok, model, device, hidden_dim, "reversed")
    save(REV_DIR, embs_r, toks_r)

    pair_path = REV_DIR / "reversed_texts.json"
    with open(pair_path, "w") as f:
        json.dump(
            [{"orig": o, "reversed": r} for o, r in zip(orig_texts, rev_texts)],
            f, indent=2, ensure_ascii=False,
        )
    print(f"Saved text pairs -> {pair_path}")


if __name__ == "__main__":
    main()
