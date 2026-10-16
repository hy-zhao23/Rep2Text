import numpy as np
import torch
from torch.utils.data import Dataset
from utils import p_args, logger
import re
import json
import random
from functools import lru_cache
from ..models.constants import IGNORE_INDEX, DEFAULT_EM_TOKEN
import collections
from transformers import AutoTokenizer, AutoConfig

_file_label = f"l{p_args.data.max_len}_o{p_args.data.overlap_len}"


@lru_cache(maxsize=1)
def _get_embedding_dim() -> int:
    emb_cfg = AutoConfig.from_pretrained(p_args.model.emb_id)
    emb_hidden = getattr(emb_cfg, "hidden_size", None)
    if emb_hidden is None:
        raise ValueError("emb_id hidden_size not found; cannot build empty embedding")
    return int(emb_hidden)

class WikipediaEmbeddingDataset(Dataset):
    input_length_histogram = collections.Counter()
    def __init__(
        self,
        tokenizer,
        max_len,
        index_dict=None,  # Changed from index_list to index_dict
        shuffle_pairing=False,
        shuffle_pairing_seed=42,
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.prompts = self._load_prompts()
        self.sys_prompt = self._load_sys_prompts()
        self._pre_tokenizer = self._load_pre_tokenizer()

        # (rank, batch) → (emb_path, tok_path)
        self.file_map = {}
        pattern = re.compile(r"embeddings_r(\d+)_b(\d+)_{}\.npz".format(_file_label))  # full pattern for all files

        for emb_path in sorted(p_args.path.embedding_dir.glob("embeddings_r*_b*_{}.npz".format(_file_label))):
            match = pattern.match(emb_path.name)
            if not match:
                continue
            rank, batch = int(match[1]), int(match[2])
            tok_path = emb_path.with_name(f"tokens_r{rank}_b{batch}_{_file_label}.npz")

            if not tok_path.exists():
                raise FileNotFoundError(f"❌ Missing tokens for {emb_path.name}")

            self.file_map[(rank, batch)] = (emb_path, tok_path)

        # 🔁 Build index and pre-load data
        if index_dict is not None:
            # Use dictionary format for efficient loading
            self.index_dict = index_dict
            self.index_map = self._flatten_dict_to_list(index_dict)
        else:
            # Load all data if no index_dict provided
            self.index_dict = {}
            self.index_map = []
            for (rank, batch), (emb_path, tok_path) in self.file_map.items():
                with np.load(tok_path) as data:
                    n = len(data["tokens"])
                    self.index_dict[(rank, batch)] = list(range(n))
                    self.index_map.extend([(rank, batch, i) for i in range(n)])

        # Pre-load all needed data based on index_dict
        self._preload_data()

        # Ablation: random (embedding, text) pairing — permute tokens-side only.
        if shuffle_pairing:
            rng = np.random.default_rng(shuffle_pairing_seed)
            self._tok_perm = rng.permutation(len(self.index_map))
            logger.warning(
                "⚠️ shuffle_pairing=True: pairing %d (embedding, text) samples with random permutation (seed=%d). Ablation only.",
                len(self.index_map), shuffle_pairing_seed,
            )
        else:
            self._tok_perm = None

    @property
    def is_transfer(self):
        return p_args.model.dec_id != p_args.model.emb_id

    def _load_pre_tokenizer(self):
        if self.is_transfer:
            logger.info("🔁 Loading pre-tokenizer for transfer learning")
            return AutoTokenizer.from_pretrained(p_args.model.emb_id, use_fast=True)
        else:
            return None

    def _retokenize(self, tokens):
        if not self.is_transfer:
            return tokens
        if not hasattr(self, "_retokenize_logged"):
            logger.info(
                "🔁 Retokenizing: emb_id=%s -> dec_id=%s",
                p_args.model.emb_id,
                p_args.model.dec_id,
            )
            self._retokenize_logged = True
        text = self._pre_tokenizer.decode(tokens, skip_special_tokens=True).strip()
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        return token_ids

    def _flatten_dict_to_list(self, index_dict):
        """Convert dictionary format to flat list for compatibility."""
        flat_list = []
        for (rank, batch), indices in index_dict.items():
            for row_idx in indices:
                flat_list.append((rank, batch, row_idx))
        return flat_list

    def _preload_data(self):
        """Pre-load only the data needed based on index_dict"""
        try:
        
            # Pre-load embeddings and tokens
            self.embs = [None] * len(self.index_map)
            self.tokens = [None] * len(self.index_map)
            
            # Process files in sorted order for better I/O performance
            for file_key in sorted(self.index_dict.keys()):
                emb_path, tok_path = self.file_map[file_key]
                
                # Load embeddings
                with np.load(emb_path) as data:
                    file_embeddings = data["embeddings"]
                
                # Load tokens
                with np.load(tok_path) as data:
                    file_tokens = data["tokens"]
                
                # Find all positions in index_map that belong to this file
                file_positions = []
                for i, (rank, batch, row_idx) in enumerate(self.index_map):
                    if (rank, batch) == file_key:
                        file_positions.append((i, row_idx))
                
                # Extract only the needed rows (.copy() so parent file array can be freed)
                for pos, row_idx in file_positions:
                    self.embs[pos] = file_embeddings[row_idx].copy()
                    self.tokens[pos] = file_tokens[row_idx].copy()
        
        except Exception as e:
            logger.error(f"❌ Error pre-loading data: {e}")


    def _load_prompts(self):
        prompt_file = p_args.path.prompt_dir / "prompts.json"
        if prompt_file.exists():
            with open(prompt_file, 'r') as f:
                self.prompts = json.load(f)
            return self.prompts
        else:
            raise ValueError(f"❌ Prompt file not found: {prompt_file}")

    def _load_sys_prompts(self):
        prompt_file = p_args.path.prompt_dir / "sys_prompt.txt"
        if prompt_file.exists():
            with open(prompt_file, 'r') as f:
                self.sys_prompt = f.read()
            return self.sys_prompt
        else:
            raise ValueError(f"❌ System prompt file not found: {prompt_file}")

    def __len__(self):
        return len(self.index_map)

    def _decode_tokens_to_text(self, tokens):
        if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None:
            tokens = [t for t in tokens if t != self.tokenizer.pad_token_id]
        try:
            return self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
        except Exception as e:
            logger.error(f"❌ Failed to decode tokens: {e}")
            return ""

    def _prefix_str(self, user_prompt):
        """
        Create the conversation sequence in the correct order:
        system prompt, embedding token, user prompt, answer text (if any)
        """
        em_tokens = DEFAULT_EM_TOKEN * p_args.proj.n_vecs
        bos = self.tokenizer.bos_token if self.tokenizer.bos_token is not None else ""
        prefix_str = f"{bos}{em_tokens}\n{self.sys_prompt}\n{user_prompt}\nAnswer:"   # BEFORE the answer
        return prefix_str 


    def __getitem__(self, idx):
        def empty_sample(reason: str):
            logger.error(f"❌ {reason} (index {idx})")
            emb_dim = _get_embedding_dim()
            return {
                "embedding": torch.zeros(emb_dim, dtype=torch.bfloat16),
                "input_ids": torch.zeros(1, dtype=torch.long),
                "labels": torch.zeros(1, dtype=torch.long)
            }
        # Validate index
        if idx < 0 or idx >= len(self.index_map):
            return empty_sample("Invalid index")
        tok_idx = int(self._tok_perm[idx]) if self._tok_perm is not None else idx
        if self.embs[idx] is None:
            return empty_sample("Missing embedding")
        if self.tokens[tok_idx] is None:
            return empty_sample("Missing tokens")

        # ---- embedding ----
        embs = torch.tensor(self.embs[idx], dtype=torch.bfloat16)
        embs = torch.nn.functional.normalize(embs, p=2, dim=-1)  # L2 norm

        # ---- text & conversation ----
        ans_ids = self._retokenize(self.tokens[tok_idx])
        
        if ans_ids is None or len(ans_ids) == 0:
            logger.warning(f"⚠️ Skipping empty decoded answer (index {idx})")
            return None  # Dataloader will skip if your collate_fn handles None

        question     = random.choice(self.prompts)
        conversation = self._prefix_str(question)
        prefix_ids  = self.tokenizer.encode(conversation, add_special_tokens=False)

        input_ids = list(prefix_ids) + list(ans_ids)
        ans_start = len(prefix_ids)

        eos_id       = self.tokenizer.eos_token_id
        if eos_id is not None:
            input_ids.append(eos_id)

        if self.max_len and len(input_ids) > self.max_len:
            logger.warning(f"⚠️ Truncating tokens for index {idx} from {len(input_ids)} to {self.max_len}")
            input_ids = input_ids[:self.max_len]
            ans_start = min(ans_start, len(input_ids))

        input_ids = torch.tensor(input_ids, dtype=torch.long)

        # 5) Labels (standard next-token objective with loss mask via IGNORE_INDEX)
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        if input_ids.numel() > 1 and ans_start >= 0:
            labels[ans_start-1: -1] = input_ids[ans_start:]  # shift left

        return {
            "embedding": embs.to(dtype=torch.bfloat16),               # (n_vecs, dim)
            "input_ids": input_ids.to(dtype=torch.long),          # (seq_len)
            "labels": labels.to(dtype=torch.long),                # (seq_len)
        }

    def get_memory_usage(self):
        """Get approximate memory usage of loaded data"""
        total_size = 0
        for emb in self.embs:
            if emb is not None and hasattr(emb, 'nbytes'):
                total_size += emb.nbytes
        for tok in self.tokens:
            if tok is not None and hasattr(tok, 'nbytes'):
                total_size += tok.nbytes
        return total_size / (1024 * 1024)  # Return in MB

    def get_loading_stats(self):
        """Get statistics about data loading"""
        return {
            'total_samples': len(self.index_map),
            'unique_files': len(self.index_dict),
            'samples_per_file': {k: len(v) for k, v in self.index_dict.items()},
            'memory_usage_mb': self.get_memory_usage(),
            'file_loading_order': sorted(self.index_dict.keys())
        }

    def print_loading_stats(self):
        """Print detailed loading statistics"""
        stats = self.get_loading_stats()
        
        logger.info(f"📊 Dataset Loading Statistics:")
        logger.info(f"  Total samples: {stats['total_samples']}")
        logger.info(f"  Unique files: {stats['unique_files']}")
        logger.info(f"  Memory usage: {stats['memory_usage_mb']:.1f}MB")
        logger.info(f"  File loading order: {stats['file_loading_order']}")        
        logger.info(f"  Samples per file:")
        
        for file_key, count in sorted(stats['samples_per_file'].items()):
            logger.info(f"    {file_key}: {count} samples")

    @classmethod
    def print_input_length_histogram(cls):
        logger.info("[Input Length Histogram] Input length distribution across samples:")
        for length, count in sorted(cls.input_length_histogram.items()):
            logger.info(f"  Length {length}: {count} samples")
