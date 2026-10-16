from pathlib import Path
from utils import p_args, logger
import numpy as np
from sys import maxsize as INT_MAX
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

def get_token_files_and_nums():
    token_files = []
    file_nums = {}
    min_token_num = INT_MAX
    
    for file in Path(p_args.path.token_dir).glob("*.npz"):
        if "tokens" in file.name and f"l{p_args.data.max_len}_o{p_args.data.overlap_len}" in file.name:
            # Extract the number of tokens from filename
            token_num = int(str(file).split("_")[-1].split(".")[0])
            file_nums[file] = token_num
            min_token_num = min(min_token_num, token_num)
            token_files.append(file)
    
    return token_files, min_token_num, file_nums

def _clean_up_files(token_files: list[Path]):
    for file in token_files:
        if file.exists():
            os.remove(file)

def sample_file_index(min_token_num: int, token_files: list[Path], file_nums: dict): 
    idx_dicts = {}
    token_files = sorted(token_files)
    num_per_file = (p_args.data.train_size + len(token_files) - 1) // len(token_files)
    
    for i, file in enumerate(token_files):
        # Use simple integer range for seeds
        random.seed(i)
        # Get the actual number of sequences in this file
        actual_num_sequences = file_nums[file]
        # Generate num_per_file random indices for this file (unique indices)
        # Use the minimum of actual sequences and requested samples
        sample_size = min(num_per_file, actual_num_sequences)
        idx_dicts[file] = random.sample(range(actual_num_sequences), sample_size)
    
    logger.info(f"🔄 Sampled indices from {len(token_files)} files with min {min_token_num} sequences...")
    return idx_dicts

def save_train_tokens():
    file = p_args.path.token_dir / f"train_tokens_{p_args.data.train_size}_l{p_args.data.max_len}_o{p_args.data.overlap_len}.npy"

    if file.exists():
        logger.info(f"✅ Training token file already exists. Skipping sampling.")
        return

    data_arrays = []
    token_files, min_token_num, file_nums = get_token_files_and_nums()
    logger.info(f"🔄 Sampling {p_args.data.train_size} tokens from {len(token_files)} files")

    idx_dicts = sample_file_index(min_token_num, token_files, file_nums)

    def load_data_with_idx(file: Path, idx: list[int]):
        data = np.load(file)['arr_0']
        sampled_data = data[idx]
        
        # Log the shape for debugging
        logger.info(f"📊 Loaded {sampled_data.shape[0]} sequences from {file.name}, shape: {sampled_data.shape}")
        
        # Handle sequences that are longer than expected by truncating them
        if sampled_data.shape[1] != p_args.data.max_len:
            if sampled_data.shape[1] > p_args.data.max_len:
                logger.warning(f"⚠️ File {file.name} has sequences of length {sampled_data.shape[1]}, truncating to {p_args.data.max_len}")
                # Truncate longer sequences to max_len
                sampled_data = sampled_data[:, :p_args.data.max_len]
            else:
                logger.warning(f"⚠️ File {file.name} has sequences of length {sampled_data.shape[1]}, expected {p_args.data.max_len}")
                # For shorter sequences, we could pad them, but for now let's skip them
                logger.warning(f"⚠️ Skipping sequences shorter than {p_args.data.max_len}")
                return None
        
        return sampled_data

    with ThreadPoolExecutor(max_workers=len(token_files)) as executor:
        futures = {executor.submit(load_data_with_idx, file, idx_dicts[file]): file for file in token_files}
        processed_files = 0
        skipped_files = 0
        for future in as_completed(futures):
            result = future.result()
            if result is not None:  # Only add non-None results
                data_arrays.append(result)
                processed_files += 1
            else:
                skipped_files += 1
                logger.warning(f"⚠️ Skipped file due to length mismatch")
        
        logger.info(f"📊 File processing summary: {processed_files} processed, {skipped_files} skipped")

    # Concatenate all arrays along the first dimension
    if data_arrays:
        # Log summary of collected data
        total_sequences = sum(arr.shape[0] for arr in data_arrays)
        logger.info(f"📊 Collected {len(data_arrays)} arrays with {total_sequences} total sequences")
        
        # Verify all arrays have the same shape before concatenating
        expected_shape = (data_arrays[0].shape[1],)
        for i, arr in enumerate(data_arrays):
            if arr.shape[1] != expected_shape[0]:
                raise ValueError(f"Array shape mismatch: got array {i} with shape {arr.shape[1]}, expected {expected_shape[0]}")
        
        logger.info(f"✅ All arrays have the same shape, concatenating...")
        data = np.concatenate(data_arrays, axis=0)
        data = data[:p_args.data.train_size]
        
        os.makedirs(os.path.dirname(file), exist_ok=True)
        
        np.save(file, data)
        logger.info(f"✅ Training token file saved to {file} with shape {data.shape}")

        _clean_up_files(token_files)
    else:
        raise ValueError("No valid data arrays found to save. This suggests all token files have incompatible sequence lengths.")