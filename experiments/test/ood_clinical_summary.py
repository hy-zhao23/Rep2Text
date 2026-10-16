#!/usr/bin/env python

import os
import csv
import sys
from pathlib import Path

import numpy as np
import torch as th
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

os.environ.setdefault("PROJ", str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("EINV_CONFIG", "transfer/q14b-32token.yaml")  # 14B, 32-token setting

PROJ_ROOT = Path(os.getenv("PROJ") or Path(__file__).resolve().parents[2])
if str(PROJ_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJ_ROOT))

from utils import p_args, logger
from utils.core.files import read_json, write_json
from utils.core.path import _get_adapter_dir
from utils.models.builder import _check_projector_dims, load_projector_weights
from utils.models.model import EInvConfig, EInvLlamaForCausalLM
from utils.models.constants import DEFAULT_EM_TOKEN


@th.no_grad()
def _get_last_token_embeddings(input_ids: th.Tensor, attention_mask: th.Tensor, model) -> np.ndarray:
	"""Extract last *real* token hidden state at the configured layer.
	Training tokens are all exactly max_len with no padding (sliding window),
	so [:, -1, :] is correct there. Here sequences may be shorter and right-padded,
	so we index each row by its last non-pad position."""
	layer_name = f"blocks.{p_args.model.embedding_layer}.hook_resid_post"
	device = next(model.parameters()).device

	_, cache = model.run_with_cache(
		input_ids.to(device),
		attention_mask=attention_mask.to(device),
		stop_at_layer=p_args.model.embedding_layer + 1,
		names_filter=lambda name: name == layer_name,
	)
	hidden = cache[layer_name]  # (B, T, D)

	# Last real token, independent of padding side.
	attention_mask = attention_mask.to(hidden.device)
	last_real = (
		attention_mask.shape[1]
		- 1
		- attention_mask.flip(dims=(1,)).argmax(dim=1)
	)
	embeddings = hidden[
		th.arange(hidden.size(0), device=hidden.device), last_real, :
	].float().cpu().numpy()
	return embeddings


def _is_valid_summary(x):
	return isinstance(x, str) and len(x.strip()) > 0


def _load_clinical_summaries():
	# Load the pre-filtered file (emb token len <= max_len, 1000 entries).
	# Run experiments/datasets/filter_clinical_notes.py first to generate it.
	filtered_path = p_args.path.data_dir / "clinical_notes" / "clinical_notes_filtered.json"
	rows = read_json(str(filtered_path))
	if rows is None:
		raise FileNotFoundError(
			f"Filtered summary file not found: {filtered_path}\n"
			"Run:  python experiments/datasets/filter_clinical_notes.py"
		)

	summaries = [str(r["summary"]).strip() for r in rows if _is_valid_summary(r.get("summary"))]
	return filtered_path, rows, summaries


def _build_tokenizer(model_id):
	tokenizer = AutoTokenizer.from_pretrained(
		model_id,
		cache_dir=p_args.path.weight_dir,
		token=p_args.settings.hf_token,
		use_fast=True,
		padding_side="right",
	)
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token
	return tokenizer


def _load_trained_model(tokenizer):
	"""Load EInv model = base Qwen decoder + projector.bin.

	The adapter dir contains only `projector.bin` (no accelerate save_state),
	so we build the base model from `dec_id` (using its native vocab_size)
	and let `initialize_einv_tokenizer` add the EM special token + resize
	embeddings, then inject the trained projector weights.
	"""
	device = "cuda" if th.cuda.is_available() else "cpu"

	config = EInvConfig.from_pretrained(p_args.model.dec_id)

	model = EInvLlamaForCausalLM.from_pretrained(
		p_args.model.dec_id,
		config=config,
		low_cpu_mem_usage=True,
		torch_dtype=th.bfloat16,
		cache_dir=p_args.path.weight_dir,
	)
	_check_projector_dims(config, model)
	# Adds EM token to tokenizer + resizes embedding to len(tokenizer)
	model.initialize_einv_tokenizer(tokenizer)

	proj_path = os.path.join(_get_adapter_dir(), "projector.bin")
	if not load_projector_weights(model, proj_path):
		raise RuntimeError(f"Failed to load projector from {proj_path}")

	model = model.to(device)
	model.eval()
	return model, device


def main():
	emb_id = p_args.model.emb_id or p_args.model.dec_id
	dec_id = p_args.model.dec_id

	summary_path, all_rows, summaries = _load_clinical_summaries()
	if len(summaries) == 0:
		raise ValueError(f"No valid summaries found in {summary_path}")
	available_valid_summaries = len(summaries)

	max_samples = int(os.getenv("OOD_MAX_SAMPLES", "0"))
	if max_samples > 0:
		summaries = summaries[:max_samples]

	emb_tokenizer = _build_tokenizer(emb_id)
	dec_tokenizer = _build_tokenizer(dec_id)
	max_len = p_args.data.max_len

	# ── Confirm token lengths on the already-filtered data ───────────────
	emb_raw_lens = [len(emb_tokenizer(s, add_special_tokens=False)["input_ids"]) for s in summaries]
	dec_raw_lens = [len(dec_tokenizer(s, add_special_tokens=False)["input_ids"]) for s in summaries]

	def _stats(lens, label):
		arr = np.array(lens)
		within = int((arr <= max_len).sum())
		logger.info(
			f"[Token Stats | {label}] n={len(arr)}  "
			f"min={arr.min()}  max={arr.max()}  "
			f"mean={arr.mean():.1f}  median={float(np.median(arr)):.1f}  "
			f"<={max_len}: {within}/{len(arr)} ({100*within/len(arr):.1f}%)"
		)
		return {"min": int(arr.min()), "max": int(arr.max()),
				"mean": round(float(arr.mean()), 2), "median": float(np.median(arr)),
				f"within_{max_len}": within, "total": len(arr)}

	emb_token_stats = _stats(emb_raw_lens, f"emb({emb_id.split('/')[-1]})")
	dec_token_stats = _stats(dec_raw_lens, f"dec({dec_id.split('/')[-1]})")
	# ─────────────────────────────────────────────────────────────────────

	# ── Load prompt templates (same files used during training) ──────────
	sys_prompt_path = p_args.path.prompt_dir / "sys_prompt.txt"
	prompts_path = p_args.path.prompt_dir / "prompts.json"
	with open(sys_prompt_path, "r") as f:
		sys_prompt = f.read().strip()
	with open(prompts_path, "r") as f:
		import json as _json
		user_prompts = _json.load(f)
	# Use first prompt deterministically for OOD eval
	user_prompt = user_prompts[0]
	# ─────────────────────────────────────────────────────────────────────

	emb_tokens = emb_tokenizer(
		summaries,
		add_special_tokens=False,
		padding=True,
		truncation=True,
		max_length=max_len,
		return_tensors="pt",
	)
	# dec_tokens: only used to store gt_ids for scoring output
	dec_tokens = dec_tokenizer(
		summaries,
		add_special_tokens=False,
		padding=False,
		truncation=True,
		max_length=max_len,
	)

	emb_input_ids = emb_tokens["input_ids"]
	emb_attention_mask = emb_tokens["attention_mask"]

	# ── Embedding cache (skip encoder load if already cached) ────────────
	emb_cache_path = p_args.path.result_dir / "ood_clinical_summary_emb_cache.npz"
	if emb_cache_path.exists():
		logger.info(f"📦 Loading cached embeddings from {emb_cache_path}")
		_cache = np.load(emb_cache_path)
		emb_last_token = _cache["emb_last_token"]
		assert emb_last_token.shape[0] == len(summaries), (
			f"Cache size mismatch: {emb_last_token.shape[0]} vs {len(summaries)} summaries. "
			f"Delete {emb_cache_path} to regenerate."
		)
	else:
		logger.info(f"🔄 Loading encoder {emb_id} to compute embeddings...")
		emb_model = HookedTransformer.from_pretrained(
			emb_id,
			device="cuda" if th.cuda.is_available() else "cpu",
			n_devices=1,
			cache_dir=p_args.path.weight_dir,
			dtype=th.bfloat16,
		)
		emb_last_token = _get_last_token_embeddings(emb_input_ids, emb_attention_mask, emb_model)

		# Persist cache so a re-run (or decoder OOM retry) skips the encoder
		emb_cache_path.parent.mkdir(parents=True, exist_ok=True)
		np.savez(emb_cache_path, emb_last_token=emb_last_token)
		logger.info(f"💾 Saved embedding cache to {emb_cache_path}")

		# Free encoder GPU memory before loading the larger decoder
		# (encoder+decoder together ~44GB doesn't fit on a 40GB A100)
		del emb_model
		if th.cuda.is_available():
			th.cuda.empty_cache()

	pair_count = min(len(summaries), int(emb_last_token.shape[0]))

	# ── Optional sharding for parallel SLURM array runs ──────────────────
	# Set OOD_WORLD_SIZE>1 in env to split work; each rank handles
	# round-robin slice [rank, rank+W, rank+2W, ...]. Outputs get a
	# `.rank{R}of{W}` suffix; run _merge_clinical_shards.py to combine.
	world_size = int(os.getenv("OOD_WORLD_SIZE", "1"))
	rank = int(os.getenv("OOD_RANK", "0"))
	assert 0 <= rank < world_size, f"OOD_RANK={rank} out of [0,{world_size})"
	my_indices = list(range(rank, pair_count, world_size))
	logger.info(
		f"Shard rank {rank}/{world_size}: handling {len(my_indices)} of {pair_count} samples"
	)
	# ─────────────────────────────────────────────────────────────────────

	gen_model, device = _load_trained_model(dec_tokenizer)

	# ── Build generation prefix (matches training dataloader format) ─────
	bos = dec_tokenizer.bos_token or ""
	em_tokens_str = DEFAULT_EM_TOKEN * p_args.proj.n_vecs
	prefix_str = f"{bos}{em_tokens_str}\n{sys_prompt}\n{user_prompt}\nAnswer:"
	prefix_ids = dec_tokenizer.encode(prefix_str, add_special_tokens=False)
	prefix_len = len(prefix_ids)
	prefix_tensor = th.tensor([prefix_ids], dtype=th.long).to(device)
	prefix_attn = th.ones_like(prefix_tensor)
	logger.info(f"Generation prefix length: {prefix_len} tokens")
	# ─────────────────────────────────────────────────────────────────────

	import torch.nn.functional as F

	numeric_output = []
	text_output = []
	for idx in my_indices:
		# L2-normalize embedding, matching training dataloader behaviour
		emb_raw = th.tensor(emb_last_token[idx], dtype=th.bfloat16)
		emb = F.normalize(emb_raw, p=2, dim=-1).unsqueeze(0).to(device)  # (1, D)

		with th.no_grad():
			generated = gen_model.generate(
				input_ids=prefix_tensor,
				hidden_embeds=emb,
				attention_mask=prefix_attn,
				do_sample=False,
				num_beams=1,
				max_new_tokens=p_args.data.gen_max_len or 32,
				pad_token_id=dec_tokenizer.eos_token_id,
				use_cache=False,
			)

		gen_ids = generated[0, prefix_len:].tolist()
		gt_ids = dec_tokens["input_ids"][idx]  # list, no padding

		gt_text = summaries[idx]  # ground truth is the original summary
		gen_text = dec_tokenizer.decode(gen_ids, skip_special_tokens=True)

		logger.info(f"[{idx}] gt_text: {gt_text}")
		logger.info(f"[{idx}] gen_text: {gen_text}")

		numeric_output.append(
			{
				"idx": idx,
				"summary": summaries[idx],
				"dec_input_ids": gt_ids,
				"emb_last_token": emb_last_token[idx].tolist(),
			}
		)

		text_output.append(
			{
				"idx": idx,
				"summary": summaries[idx],
				"gt_ids": gt_ids,
				"gen_ids": gen_ids,
				"gt_text": gt_text,
				"gen_text": gen_text,
			}
		)

	suffix = f".rank{rank}of{world_size}" if world_size > 1 else ""
	result_pairs_json = p_args.path.result_dir / f"ood_clinical_summary_pairs{suffix}.json"
	result_pairs_npz = p_args.path.result_dir / f"ood_clinical_summary_pairs{suffix}.npz"
	result_text_json = p_args.path.result_dir / f"ood_clinical_summary_texts{suffix}.json"
	result_text_csv = p_args.path.result_dir / f"ood_clinical_summary_texts{suffix}.csv"

	write_json(numeric_output, str(result_pairs_json))
	write_json(text_output, str(result_text_json))

	shard_emb = np.stack([emb_last_token[row["idx"]] for row in numeric_output], axis=0) if numeric_output else np.empty((0, emb_last_token.shape[1]), dtype=emb_last_token.dtype)
	np.savez(
		result_pairs_npz,
		shard_indices=np.array([row["idx"] for row in numeric_output], dtype=np.int64),
		dec_input_ids=np.array([row["dec_input_ids"] for row in numeric_output], dtype=object),
		emb_last_token=shard_emb,
	)

	with open(result_text_csv, mode="w", newline="", encoding="utf-8") as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(["idx", "summary", "gt_text", "gen_text"])
		for row in text_output:
			writer.writerow([row["idx"], row["summary"], row["gt_text"], row["gen_text"]])

	print("=== OOD Clinical Summary Build Done ===")
	print(f"config             = {os.getenv('EINV_CONFIG')}")
	print(f"source_file        = {summary_path}")
	print(f"total_rows         = {len(all_rows)}")
	print(f"valid_summaries    = {available_valid_summaries} total / {len(summaries)} after filter (dec<={max_len})")
	print(f"paired_samples     = {pair_count}")
	print(f"emb_model          = {emb_id}")
	print(f"dec_model          = {dec_id}")
	print(f"max_len (32-token) = {max_len}")
	print(f"--- Token Length Stats (emb / {emb_id.split('/')[-1]}) ---")
	for k, v in emb_token_stats.items():
		print(f"  {k:20s} = {v}")
	print(f"--- Token Length Stats (dec / {dec_id.split('/')[-1]}) ---")
	for k, v in dec_token_stats.items():
		print(f"  {k:20s} = {v}")
	print(f"pairs_json_output  = {result_pairs_json}")
	print(f"pairs_npz_output   = {result_pairs_npz}")
	print(f"texts_json_output  = {result_text_json}")
	print(f"texts_csv_output   = {result_text_csv}")


if __name__ == "__main__":
	main()
