import bisect
import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import transformers


@dataclass
class _Shard:
    emb_path: str
    tok_path: str
    size: int


class LazyNPZInversionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        npz_dir: str,
        source_tokenizer_name: str,
        target_tokenizer: transformers.PreTrainedTokenizer,
        max_seq_length: int,
        split: str,
        embeddings_glob: str = "embeddings_*.npz",
        use_less_data: int = -1,
        validation_fraction: float = 0.1,
        test_fraction: float = 0.1,
        max_split_samples: Optional[int] = None,
    ):
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unsupported split: {split}")
        if not os.path.isdir(npz_dir):
            raise ValueError(f"NPZ directory not found: {npz_dir}")

        self.npz_dir = npz_dir
        self.max_seq_length = int(max_seq_length)
        self.split = split
        self.source_tokenizer = transformers.AutoTokenizer.from_pretrained(
            source_tokenizer_name
        )
        if self.source_tokenizer.pad_token is None:
            self.source_tokenizer.pad_token = self.source_tokenizer.eos_token

        self.target_tokenizer = target_tokenizer
        if self.target_tokenizer.pad_token is None:
            self.target_tokenizer.pad_token = self.target_tokenizer.eos_token

        self._shards: List[_Shard] = self._discover_shards(
            npz_dir=npz_dir,
            embeddings_glob=embeddings_glob,
        )
        if not self._shards:
            raise ValueError(f"No valid embedding/token NPZ pairs found in: {npz_dir}")

        total = sum(s.size for s in self._shards)
        if use_less_data and (use_less_data > 0):
            total = min(total, int(use_less_data))

        if total < 3:
            raise ValueError(f"Too few samples for spliting: total={total}")

        n_val = max(1, int(total * validation_fraction))
        n_test = max(1, int(total * test_fraction))
        n_train = total - n_val - n_test
        if n_train < 1:
            raise ValueError(
                f"Invalid split sizes: total={total}, train={n_train}, val={n_val}, test={n_test}"
            )

        if split == "train":
            split_start, split_end = 0, n_train
        elif split == "validation":
            split_start, split_end = n_train, n_train + n_val
        else:
            split_start, split_end = n_train + n_val, total

        if max_split_samples is not None and max_split_samples > 0:
            split_end = min(split_end, split_start + int(max_split_samples))

        self._global_start = split_start
        self._global_end = split_end
        self._length = max(0, self._global_end - self._global_start)
        if self._length < 1:
            raise ValueError(f"Split {split} has zero samples")

        self._shard_prefix: List[int] = []
        running = 0
        for s in self._shards:
            running += s.size
            self._shard_prefix.append(running)

        self._opened: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    @staticmethod
    def _discover_shards(npz_dir: str, embeddings_glob: str) -> List[_Shard]:
        shards: List[_Shard] = []
        emb_files = sorted(glob.glob(os.path.join(npz_dir, embeddings_glob)))
        for emb_path in emb_files:
            tok_path = os.path.join(
                npz_dir, os.path.basename(emb_path).replace("embeddings_", "tokens_")
            )
            if not os.path.exists(tok_path):
                continue

            with np.load(emb_path, mmap_mode="r") as emb_npz, np.load(
                tok_path, mmap_mode="r"
            ) as tok_npz:
                if ("embeddings" not in emb_npz.files) or ("tokens" not in tok_npz.files):
                    continue
                size = min(len(emb_npz["embeddings"]), len(tok_npz["tokens"]))
                if size <= 0:
                    continue
            shards.append(_Shard(emb_path=emb_path, tok_path=tok_path, size=size))
        return shards

    def __len__(self) -> int:
        return self._length

    def _locate(self, global_idx: int) -> Tuple[int, int]:
        shard_idx = bisect.bisect_right(self._shard_prefix, global_idx)
        prev = 0 if shard_idx == 0 else self._shard_prefix[shard_idx - 1]
        local_idx = global_idx - prev
        return shard_idx, local_idx

    def _load_shard_arrays(self, shard_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        if shard_idx in self._opened:
            return self._opened[shard_idx]

        shard = self._shards[shard_idx]
        emb_npz = np.load(shard.emb_path, mmap_mode="r")
        tok_npz = np.load(shard.tok_path, mmap_mode="r")
        emb = emb_npz["embeddings"]
        tok = tok_npz["tokens"]
        self._opened[shard_idx] = (emb, tok)
        return emb, tok

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx < 0 or idx >= self._length:
            raise IndexError(idx)

        global_idx = self._global_start + idx
        shard_idx, local_idx = self._locate(global_idx)
        emb, tok = self._load_shard_arrays(shard_idx)

        source_ids = tok[local_idx].astype(np.int64, copy=False)
        text = self.source_tokenizer.decode(
            source_ids.tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        target = self.target_tokenizer(
            text,
            padding=False,
            truncation=True,
            max_length=self.max_seq_length,
        )
        input_ids = target["input_ids"]
        attention_mask = target["attention_mask"]
        labels = [
            (
                -100
                if token_id == self.target_tokenizer.pad_token_id
                else int(token_id)
            )
            for token_id in input_ids
        ]

        frozen = emb[local_idx].astype(np.float32, copy=False)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "embedder_input_ids": torch.tensor([0], dtype=torch.long),
            "embedder_attention_mask": torch.tensor([1], dtype=torch.long),
            "length": int(sum(attention_mask)),
            "frozen_embeddings": torch.tensor(frozen, dtype=torch.float32),
            "idx": int(global_idx),
        }

