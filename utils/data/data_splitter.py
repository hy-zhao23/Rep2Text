import os
from utils.core.files import write_pkl, read_pkl
import numpy as np
from utils import p_args, logger, gh
from utils.core.path import pt_data_idx_path, ft_data_idx_path
import re
from tqdm import tqdm
import time
from utils.project_utils import set_global_seed

set_global_seed()

def wait_for_split_files(*file_paths, max_wait_time=3000):
    """
    Wait for split files to be fully written and available for reading.
    This function ensures all ranks wait until the files are actually accessible.
    
    Args:
        *file_paths: Variable number of file paths to wait for
        max_wait_time: Maximum time to wait in seconds (default: 5 minutes)
    
    Returns:
        bool: True if all files are available, False if timeout
    """
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        all_files_ready = True
        
        for file_path in file_paths:
            if not os.path.exists(file_path):
                all_files_ready = False
                break
        
        if all_files_ready:
            return True
        # Wait a bit before checking again
        time.sleep(30)
    
    logger.error(f"❌ Timeout waiting for split files after {max_wait_time} seconds")
    return False

def generate_split():
    """Generate or load split indices with proper file-based grouping. Only rank 0 generates, all ranks wait, then all read and return."""
    if gh.is_main:
        # Check if split indices already exist
        if os.path.exists(pt_data_idx_path) and os.path.exists(ft_data_idx_path):
            pass  # Files already exist, nothing to do
        else:
            # Split indices don't exist → generate them
            _file_label = f"l{p_args.data.max_len}_o{p_args.data.overlap_len}"
            file_pattern = re.compile(r"embeddings_r(\d+)_b(\d+)_{}\.npz".format(_file_label))
            all_files = os.listdir(p_args.path.embedding_dir)
            file_groups = {}
            for emb_path in tqdm(all_files, desc="🔄 Scanning files"):
                match = file_pattern.match(emb_path)
                if not match or "_checkpoint" in emb_path:
                    continue
                rank = int(match.group(1))
                batch = int(match.group(2))
                emb_path = os.path.join(p_args.path.embedding_dir, emb_path)
                tok_path = emb_path.replace("embeddings_", "tokens_")
                if not os.path.exists(tok_path):
                    logger.error(f"⚠️ Token file not found for {emb_path}, skipping...")
                    continue
                with np.load(tok_path) as tok_data:
                    n = len(tok_data["tokens"])
                file_groups[(rank, batch)] = n
            sorted_files = sorted(file_groups.keys())
            pretrain_files = []
            finetune_files = []
            total_samples = sum(file_groups.values())
            target_pretrain_samples = int(p_args.data.pretrain_ratio * total_samples)
            current_pretrain_samples = 0
            for file_key in sorted_files:
                file_samples = file_groups[file_key]
                if current_pretrain_samples < target_pretrain_samples:
                    pretrain_files.append((file_key, file_samples))
                    current_pretrain_samples += file_samples
                else:
                    finetune_files.append((file_key, file_samples))
            pretrain_indices = {}
            finetune_indices = {}
            for (rank, batch), n_samples in tqdm(pretrain_files, desc="🔄 Pretrain files"):
                indices = list(range(n_samples))
                np.random.shuffle(indices)
                pretrain_indices[(rank, batch)] = indices
            for (rank, batch), n_samples in tqdm(finetune_files, desc="🔄 Finetune files"):
                indices = list(range(n_samples))
                np.random.shuffle(indices)
                finetune_indices[(rank, batch)] = indices
            pretrain_total = sum(len(indices) for indices in pretrain_indices.values())
            finetune_total = sum(len(indices) for indices in finetune_indices.values())
            write_pkl(pretrain_indices, pt_data_idx_path)
            write_pkl(finetune_indices, ft_data_idx_path)
            logger.info(f"✅ File-based split complete:")
            logger.info(f"   Pretrain: {pretrain_total} samples from {len(pretrain_files)} files")
            logger.info(f"   Finetune: {finetune_total} samples from {len(finetune_files)} files")

    gh.barrier()  # All ranks wait here after writing
    
    # Wait for files to be fully written and accessible
    logger.info(f"⏳ Waiting for split index files to be fully written...")
    files_ready = wait_for_split_files(pt_data_idx_path, ft_data_idx_path)
    
    if not files_ready:
        logger.error(f"❌ Split index files are not ready after waiting. This indicates a serious I/O issue.")
        logger.error(f"❌ Please check disk space, permissions, and network filesystem status.")
        raise RuntimeError("Split index files failed to be written or are not accessible")
    
    pretrain_indices = read_pkl(pt_data_idx_path)
    finetune_indices = read_pkl(ft_data_idx_path)
    return pretrain_indices, finetune_indices

def _split_data(data_type, train_ratio, val_ratio, test_ratio, target_total_samples=None, use_all_samples=False):
    """
    Generic function to split data into train/val/test.
    
    Args:
        data_type: "pretrain" or "finetune"
        train_ratio: Ratio for training data
        val_ratio: Ratio for validation data  
        test_ratio: Ratio for test data
        target_total_samples: Target total samples (for pretrain, limits to train_size)
        use_all_samples: If True, use all available samples (for finetune)
    """
    from utils.core.path import pt_train_idx_path, pt_val_idx_path, pt_test_idx_path
    from utils.core.path import ft_train_idx_path, ft_val_idx_path, ft_test_idx_path
    
    # Set paths based on data type
    if data_type == "pretrain":
        train_path, val_path, test_path = pt_train_idx_path, pt_val_idx_path, pt_test_idx_path
    elif data_type == "finetune":
        train_path, val_path, test_path = ft_train_idx_path, ft_val_idx_path, ft_test_idx_path
    else:
        raise ValueError(f"❌ Unknown data_type: {data_type}. Must be 'pretrain' or 'finetune'")
    
    if gh.is_main:
        if (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)):
            pass  # Files already exist, nothing to do
        else:
            # Get the appropriate indices
            base_indices = None
            if data_type == "pretrain":
                base_indices, _ = generate_split()
            else:  # finetune
                _, base_indices = generate_split()
            
            logger.info(f"🔁 Generating train/val/test split for {data_type} data...")
            
            # Calculate target samples
            total_available_samples = sum(len(indices) for indices in base_indices.values())
            if use_all_samples:
                target_samples = total_available_samples
            else:
                target_samples = min(total_available_samples, target_total_samples or p_args.data.train_size)
            
            logger.info(f"📊 {data_type.capitalize()} dataset size configuration:")
            logger.info(f"   Total available {data_type} samples: {total_available_samples}")
            logger.info(f"   Target total samples: {target_samples}")
            logger.info(f"   Split ratios: {train_ratio:.1%} train, {val_ratio:.1%} val, {test_ratio:.1%} test")
            
            # Calculate file sample counts
            sorted_files = sorted(base_indices.keys())
            file_sample_counts = {}
            
            if use_all_samples:
                # Use all samples from each file
                for file_key in sorted_files:
                    file_sample_counts[file_key] = len(base_indices[file_key])
            else:
                # Proportional sampling based on target
                for file_key in sorted_files:
                    file_total_samples = len(base_indices[file_key])
                    proportional_samples = int((file_total_samples / total_available_samples) * target_samples)
                    file_sample_counts[file_key] = min(proportional_samples, file_total_samples)
                
                # Adjust to reach exact target
                current_total = sum(file_sample_counts.values())
                if current_total != target_samples:
                    logger.info(f"🔄 Adjusting sample counts to reach target {target_samples} (current: {current_total})")
                    remaining_samples = {k: len(base_indices[k]) - v for k, v in file_sample_counts.items()}
                    sorted_by_remaining = sorted(remaining_samples.items(), key=lambda x: x[1], reverse=True)
                    
                    if current_total < target_samples:
                        samples_needed = target_samples - current_total
                        for file_key, remaining in sorted_by_remaining:
                            if samples_needed <= 0:
                                break
                            additional = min(samples_needed, remaining)
                            file_sample_counts[file_key] += additional
                            samples_needed -= additional
                    else:
                        samples_to_remove = current_total - target_samples
                        for file_key, current_count in sorted(file_sample_counts.items(), key=lambda x: x[1], reverse=True):
                            if samples_to_remove <= 0:
                                break
                            to_remove = min(samples_to_remove, current_count)
                            file_sample_counts[file_key] -= to_remove
                            samples_to_remove -= to_remove

            # Split into train/val/test
            train_indices = {}
            val_indices = {}
            test_indices = {}
            for (rank, batch), n_samples in tqdm(file_sample_counts.items(), desc=f"🔄 Sampling and splitting {data_type} per file"):
                if n_samples > 0:
                    available_indices = list(range(len(base_indices[(rank, batch)])))
                    sampled_indices = np.random.choice(available_indices, size=n_samples, replace=False)
                    np.random.shuffle(sampled_indices)
                    n_train = int(train_ratio * n_samples)
                    n_val = int(val_ratio * n_samples)
                    n_test = n_samples - n_train - n_val
                    train_idx = sampled_indices[:n_train]
                    val_idx = sampled_indices[n_train:n_train+n_val]
                    test_idx = sampled_indices[n_train+n_val:]
                    if len(train_idx) > 0:
                        train_indices[(rank, batch)] = list(train_idx)
                    if len(val_idx) > 0:
                        val_indices[(rank, batch)] = list(val_idx)
                    if len(test_idx) > 0:
                        test_indices[(rank, batch)] = list(test_idx)
            
            train_total = sum(len(indices) for indices in train_indices.values())
            val_total = sum(len(indices) for indices in val_indices.values())
            test_total = sum(len(indices) for indices in test_indices.values())
            
            # Write split files with error handling
            try:
                write_pkl(train_indices, train_path)
                write_pkl(val_indices, val_path)
                write_pkl(test_indices, test_path)
                logger.info(f"✅ {data_type.capitalize()} split complete:")
                logger.info(f"   Train: {train_total} samples")
                logger.info(f"   Val: {val_total} samples")
                logger.info(f"   Test: {test_total} samples")
            except Exception as e:
                logger.error(f"❌ Failed to write {data_type} split files: {e}")
                logger.error(f"❌ This will cause the training to fail. Please check disk space and permissions.")
                raise e
            
    gh.barrier()  # All ranks wait here after writing
    
    # Wait for files to be fully written and accessible
    files_ready = wait_for_split_files(train_path, val_path, test_path)
    
    if not files_ready:
        logger.error(f"❌ {data_type.capitalize()} split files are not ready after waiting. This indicates a serious I/O issue.")
        logger.error(f"❌ Please check disk space, permissions, and network filesystem status.")
        raise RuntimeError(f"{data_type.capitalize()} split files failed to be written or are not accessible")
    
    # Read split files with error handling
    try:
        train_indices = read_pkl(train_path)
        val_indices = read_pkl(val_path)
        test_indices = read_pkl(test_path)
        
        # Check if any files are missing
        if train_indices is None or val_indices is None or test_indices is None:
            missing_files = []
            if train_indices is None:
                missing_files.append(str(train_path))
            if val_indices is None:
                missing_files.append(str(val_path))
            if test_indices is None:
                missing_files.append(str(test_path))
            
            logger.error(f"❌ {data_type.capitalize()} split files are missing: {missing_files}")
            logger.error(f"❌ This indicates the {data_type} split generation process failed or was interrupted.")
            logger.error(f"❌ Please check the logs for errors during {data_type} split generation.")
            
    except Exception as e:
        logger.error(f"❌ Error reading {data_type} split files: {e}")
        raise e
    
    return train_indices, val_indices, test_indices


def split_data(data_type="pretrain"):
    """Split pretrain data into train/val/test for evaluation."""
    return _split_data(
        data_type=data_type,
        train_ratio=0.8,
        val_ratio=0.1, 
        test_ratio=0.1,
        target_total_samples=None,
        use_all_samples=True  # Use all available pretrain samples
    )

        
