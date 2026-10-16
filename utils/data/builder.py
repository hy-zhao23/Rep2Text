import torch
from functools import lru_cache
from transformers import AutoConfig
from torch.utils.data import DataLoader
from utils import logger, gh, p_args
from utils.data.dataloader import WikipediaEmbeddingDataset
from utils.data.data_splitter import split_data
from utils.models.constants import IGNORE_INDEX
from functools import partial


@lru_cache(maxsize=1)
def _get_embedding_dim() -> int:
    emb_cfg = AutoConfig.from_pretrained(p_args.model.emb_id)
    emb_hidden = getattr(emb_cfg, "hidden_size", None)
    if emb_hidden is None:
        raise ValueError("emb_id hidden_size not found; cannot build collate function")
    return int(emb_hidden)


def custom_collate_fn(batch, embedding_dim=None):
    if embedding_dim is None:
        embedding_dim = _get_embedding_dim()
    # if empty, return zero-sized tensors with shapes
    batch = [item for item in batch if item is not None]
    
    if not batch:
        return {
            'input_ids': torch.empty((0, 1), dtype=torch.long),
            'labels': torch.empty((0, 1), dtype=torch.long),
            'embedding': torch.empty((0, embedding_dim), dtype=torch.bfloat16)
        }
    
    # Pad to the longest sequence in the batch
    required_keys = ['input_ids', 'labels', 'embedding']
    max_len = max(item['input_ids'].shape[0] for item in batch if 'input_ids' in item and hasattr(item['input_ids'], 'shape'))
    collated = {}
    for key in required_keys:
        if key in ['input_ids', 'labels']:
            tensors = []
            for item in batch:
                t = item[key]
                if t.shape[0] < max_len:
                    pad_len = max_len - t.shape[0]
                    pad_value = IGNORE_INDEX if key == 'labels' else 0
                    pad = torch.full((pad_len,), pad_value, dtype=t.dtype, device=t.device)
                    t = torch.cat([t, pad], dim=0)
                tensors.append(t)
            collated[key] = torch.stack(tensors, dim=0) if tensors else torch.empty((0, max_len), dtype=torch.long)
        elif key == 'embedding':
            tensors = [item[key] for item in batch]
            collated[key] = torch.stack(tensors, dim=0) if tensors else torch.empty((0, embedding_dim), dtype=torch.bfloat16)
    return collated


def create_dataloader(config, tokenizer, embedding_dim, context_len, split_type="train", data_type="pretrain"):
    """Create a dataloader for the specified split type with robust error handling."""
    
    # Step 1: Get split indices
    train_indices, val_indices, test_indices = split_data(data_type=data_type)

    # Step 2: Validate split_type
    split_map = {"train": train_indices, "val": val_indices, "test": test_indices}
    if split_type not in split_map:
        valid_splits = list(split_map.keys())
        raise ValueError(f"❌ Unknown split type: '{split_type}'")
    split_indices = split_map[split_type]

    # Step 3: Validate split content
    total_samples = sum(len(indices) for indices in split_indices.values())
    if total_samples == 0:
        raise ValueError(f"❌ Split '{split_type}' has no data.")

    # Step 4: Create dataset
    seq_limit = context_len
    max_seq_len = getattr(p_args.data, "max_seq_len", None)
    if max_seq_len is not None:
        seq_limit = min(seq_limit, int(max_seq_len))

    shuffle_pairing = bool(getattr(p_args.data, "shuffle_pairing", False)) and split_type == "train"
    shuffle_pairing_seed = int(getattr(p_args.data, "shuffle_pairing_seed", 42))
    if shuffle_pairing and gh.is_main:
        logger.warning(
            "⚠️ Ablation: shuffle_pairing enabled for split='%s' (seed=%d).",
            split_type, shuffle_pairing_seed,
        )

    dataset = WikipediaEmbeddingDataset(
        tokenizer=tokenizer,
        max_len=seq_limit,
        index_dict=split_indices,
        shuffle_pairing=shuffle_pairing,
        shuffle_pairing_seed=shuffle_pairing_seed,
    )
    # Step 5: Collate function and dataloader
    collate_fn = partial(custom_collate_fn, embedding_dim=embedding_dim)
    
    batch_size = config['training']['batch_size']
    num_workers = max(config['data']['num_workers'], 1)
    
    # No sampler logic needed; let Accelerate handle distributed, and use shuffle for training
    shuffle = split_type == "train"
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=(split_type == "train"),
        persistent_workers=False,
        collate_fn=collate_fn,
        prefetch_factor=2 if num_workers > 0 else None
    )

    # Step 6: Try getting length to verify everything works
    num_batches = len(dataloader)
    if gh.is_main:
        logger.info(f"[Rank {gh.rank}] 🔄 Successfully created dataloader for split '{split_type}': {num_batches} batches")

    return dataloader

