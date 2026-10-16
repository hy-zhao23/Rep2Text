"""Run the trained 16-token Llama-3.1-8B inverter on the 500 full-sentence embeddings.

Reads NPZ from tmp/embeddings/full_sentence/llama8b/, builds an in-memory
dataset (mirroring eval_ours.py for the 32-token transfer setup), runs greedy
decoding with the l10t16p16/exp05 adapter, and saves results to
results/llama8B/l10t16p16/full_sentence/ours/.
"""
import json
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 16-token llama8B (l10t16p16) inverter; tokens/setup16.yaml has the new schema
# (dec_id/emb_id directly under models:) that settings.py requires.
os.environ["EINV_CONFIG"] = "tokens/setup16.yaml"
os.environ.setdefault("TRAIN_TAG", "exp05")

from utils import p_args, gh, logger  # noqa: E402,F401
from utils.models import load_pretrained_model  # noqa: E402
from utils.models.constants import IGNORE_INDEX, DEFAULT_EM_TOKEN  # noqa: E402
from utils.eval.test_utils import greedy_decode_batch  # noqa: E402
from utils.eval.metrics import EvaluationMetrics  # noqa: E402
from utils.data.builder import custom_collate_fn  # noqa: E402
from functools import partial  # noqa: E402
import random  # noqa: E402

PROJ = Path(__file__).resolve().parents[2]
NPZ_DIR = PROJ / "tmp/embeddings/full_sentence/llama8b"
SENT_PATH = PROJ / "results/llama8B/l10t16p16/full_sentence/sentences_500.json"
OUT_DIR = PROJ / "results/llama8B/l10t16p16/full_sentence/ours"
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


def main():
    random.seed(0)
    logger.info("Loading model ...")
    tokenizer, model, _ = load_pretrained_model(
        model_type="einv", resume_from=True, device_map="cuda",
    )

    from transformers import AutoTokenizer
    src_tok = AutoTokenizer.from_pretrained(p_args.model.emb_id, use_fast=True)
    if src_tok.pad_token is None:
        src_tok.pad_token = src_tok.eos_token

    emb_path = NPZ_DIR / f"embeddings_r0_b0_l{MAX_TOKENS}_o0.npz"
    tok_path = NPZ_DIR / f"tokens_r0_b0_l{MAX_TOKENS}_o0.npz"
    embs = np.load(emb_path)["embeddings"]
    src_tokens = np.load(tok_path)["tokens"]
    logger.info(f"Loaded {len(embs)} samples from {NPZ_DIR}")

    with open(p_args.path.prompt_dir / "prompts.json") as f:
        prompts = json.load(f)
    with open(p_args.path.prompt_dir / "sys_prompt.txt") as f:
        sys_prompt = f.read()

    dataset = FullSentenceDataset(embs, src_tokens, src_tok, tokenizer, sys_prompt, prompts,
                                  max_len=p_args.data.max_seq_len)
    embedding_dim = embs.shape[1]
    collate_fn = partial(custom_collate_fn, embedding_dim=embedding_dim)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_fn)

    metrics_cal = EvaluationMetrics()
    base_model = model.unwrap_module() if hasattr(model, "unwrap_module") else model

    model.eval()
    input_device = next(base_model.parameters()).device
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inverting (16-token ours)"):
            batch = {k: v.to(input_device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            decoded = greedy_decode_batch(model, batch, tokenizer, log=False)
            metrics_cal.add_batch_results(batch_loss=0.0, batch_decoded=decoded)

    out_dec = OUT_DIR / "val_einv_decoded.json"
    with open(out_dec, "w") as f:
        json.dump(metrics_cal.batch_decoded, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved decoded to {out_dec}")

    out_met = OUT_DIR / "val_einv_metrics_summary.json"
    metrics_cal.save_metrics(out_met)
    logger.info(f"Saved metrics summary to {out_met}")


if __name__ == "__main__":
    main()
