"""Run the m3b 640k baseline (mistral-7B encoder -> Llama-3.2-3B decoder) on
both ORIGINAL and REVERSED 16-token full-sentence embeddings.

Adapter:  weights/mistral7B/l10t16p16/transfer-m3b/projector.bin
Reads NPZ from:
  tmp/embeddings/full_sentence/mistral7b_l16/         (ORIG)
  tmp/embeddings/full_sentence/mistral7b_reverse_l16/ (REVERSED)

Saves decoded outputs + per-sample comparison + simple token-EM/word-overlap
to results/llama3B/l10t16p16/full_sentence/m3b_reverse/.
"""
import json
import os
from pathlib import Path
import random
from functools import partial

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

os.environ["EINV_CONFIG"] = "transfer/m3b.yaml"
os.environ["TRAIN_TAG"] = "transfer-m3b"

from utils import p_args, gh, logger  # noqa: E402,F401
from utils.models import load_pretrained_model  # noqa: E402
from utils.models.constants import IGNORE_INDEX, DEFAULT_EM_TOKEN  # noqa: E402
from utils.eval.test_utils import greedy_decode_batch  # noqa: E402
from utils.data.builder import custom_collate_fn  # noqa: E402

PROJ = Path(__file__).resolve().parents[2]
ORIG_NPZ = PROJ / "tmp/embeddings/full_sentence/mistral7b_l16"
REV_NPZ = PROJ / "tmp/embeddings/full_sentence/mistral7b_reverse_l16"
SENT_PATH = PROJ / "datasets/full_sentence_3b_long_500.json"
OUT_DIR = PROJ / "results/llama3B/l10t16p16/full_sentence/m3b_reverse"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_TOKENS = 16


class FullSentenceDataset(Dataset):
    def __init__(self, embs, src_tokens, src_tokenizer, dec_tokenizer, sys_prompt, prompts, max_len=200):
        self.embs = embs
        self.src_tokens = src_tokens
        self.src_tokenizer = src_tokenizer
        self.dec_tokenizer = dec_tokenizer
        self.sys_prompt = sys_prompt
        self.prompts = prompts
        self.max_len = max_len

    def __len__(self):
        return len(self.embs)

    def _prefix_str(self, user_prompt):
        em_tokens = DEFAULT_EM_TOKEN * p_args.proj.n_vecs
        bos = self.dec_tokenizer.bos_token if self.dec_tokenizer.bos_token is not None else ""
        return f"{bos}{em_tokens}\n{self.sys_prompt}\n{user_prompt}\nAnswer:"

    def __getitem__(self, idx):
        emb = torch.tensor(self.embs[idx], dtype=torch.bfloat16)
        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)

        src_ids = [int(t) for t in self.src_tokens[idx] if int(t) != self.src_tokenizer.pad_token_id]
        text = self.src_tokenizer.decode(src_ids, skip_special_tokens=True).strip()
        ans_ids = self.dec_tokenizer.encode(text, add_special_tokens=False)

        question = random.choice(self.prompts)
        prefix_ids = self.dec_tokenizer.encode(self._prefix_str(question), add_special_tokens=False)

        input_ids = list(prefix_ids) + list(ans_ids)
        ans_start = len(prefix_ids)
        if self.dec_tokenizer.eos_token_id is not None:
            input_ids.append(self.dec_tokenizer.eos_token_id)
        if len(input_ids) > self.max_len:
            input_ids = input_ids[:self.max_len]
            ans_start = min(ans_start, len(input_ids))

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        if input_ids.numel() > 1 and ans_start >= 0:
            labels[ans_start - 1: -1] = input_ids[ans_start:]

        return {"embedding": emb, "input_ids": input_ids, "labels": labels}


def run_split(name, npz_dir, src_tok, tokenizer, model, sys_prompt, prompts, embedding_dim, gold_texts):
    emb_path = npz_dir / f"embeddings_r0_b0_l{MAX_TOKENS}_o0.npz"
    tok_path = npz_dir / f"tokens_r0_b0_l{MAX_TOKENS}_o0.npz"
    embs = np.load(emb_path)["embeddings"]
    src_tokens = np.load(tok_path)["tokens"]
    logger.info(f"[{name}] Loaded {len(embs)} samples from {npz_dir}")

    dataset = FullSentenceDataset(embs, src_tokens, src_tok, tokenizer, sys_prompt, prompts,
                                  max_len=p_args.data.max_seq_len)
    collate_fn = partial(custom_collate_fn, embedding_dim=embedding_dim)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_fn)

    base_model = model.unwrap_module() if hasattr(model, "unwrap_module") else model
    model.eval()
    input_device = next(base_model.parameters()).device

    decoded_all = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Inverting [{name}]"):
            batch = {k: v.to(input_device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            decoded = greedy_decode_batch(model, batch, tokenizer, log=False)
            # decoded is list of dicts with keys like 'pred','target' (or similar) per item
            decoded_all.extend(decoded)

    out_path = OUT_DIR / f"{name}_decoded.json"
    with open(out_path, "w") as f:
        json.dump(decoded_all, f, indent=2, ensure_ascii=False)
    logger.info(f"[{name}] Saved -> {out_path}")
    return decoded_all


def main():
    random.seed(0)
    logger.info("Loading m3b transfer model (mistral->3B, 640k baseline) ...")
    tokenizer, model, _ = load_pretrained_model(
        model_type="einv", resume_from=True, device_map="cuda",
    )

    from transformers import AutoTokenizer
    src_tok = AutoTokenizer.from_pretrained(p_args.model.emb_id, use_fast=True)
    if src_tok.pad_token is None:
        src_tok.pad_token = src_tok.eos_token

    with open(p_args.path.prompt_dir / "prompts.json") as f:
        prompts = json.load(f)
    with open(p_args.path.prompt_dir / "sys_prompt.txt") as f:
        sys_prompt = f.read()

    with open(SENT_PATH) as f:
        gold = json.load(f)
    gold_texts = [g["gt_truncated"] for g in gold]

    # discover hidden_dim from one of the npz
    embedding_dim = np.load(ORIG_NPZ / f"embeddings_r0_b0_l{MAX_TOKENS}_o0.npz")["embeddings"].shape[1]

    orig_dec = run_split("orig", ORIG_NPZ, src_tok, tokenizer, model,
                         sys_prompt, prompts, embedding_dim, gold_texts)
    rev_dec = run_split("reversed", REV_NPZ, src_tok, tokenizer, model,
                        sys_prompt, prompts, embedding_dim, gold_texts)

    # build a combined comparison file
    pair = []
    for i, (o, r) in enumerate(zip(orig_dec, rev_dec)):
        pair.append({
            "idx": i,
            "gold_orig": gold_texts[i],
            "gold_reversed": " ".join(gold_texts[i].split()[::-1]),
            "pred_from_orig_emb": o,
            "pred_from_reversed_emb": r,
        })
    out_pair = OUT_DIR / "comparison.json"
    with open(out_pair, "w") as f:
        json.dump(pair, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved comparison -> {out_pair}")


if __name__ == "__main__":
    main()
