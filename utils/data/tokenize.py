"""
The tokenization process:
1. First tokenize the entire text sequence, discard texts shorter than max_len
2. Then split into overlapping segments of max_len tokens
3. Each segment has overlap_len tokens in common with the next segment
"""
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from transformers import AutoTokenizer
from .. import p_args, logger
import tqdm
import numpy as np
import os

class TokenizeProcessor:
    def __init__(self):
        self.total_segments = 0
        logger.info(f"✅ Initializing TokenizeProcessor...")
        logger.info(f"✅ max_len: {p_args.data.max_len}, overlap_len: {p_args.data.overlap_len}")

    def _get_tokenizer(self):
        """Initialize and return the tokenizer with appropriate settings."""
        tokenizer = AutoTokenizer.from_pretrained(
            p_args.model.emb_id,
            cache_dir=p_args.path.weight_dir,
            local_files_only=True,
            use_fast=True,
            padding_side='right'
        )
        return tokenizer

    def _tokenize_and_truncate_with_overlap(self, texts: List[str], tokenizer) -> List[List]:
        """Tokenize a batch of texts and create overlapping segments."""
        stride = p_args.data.max_len - p_args.data.overlap_len
        max_segments_per_text = 4
        eos_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
        
        all_segments = []
        
        for i in range(0, len(texts), 500):
            batch_texts = texts[i:min(i + 500, len(texts))]
            # Tokenize the batch with padding
            tokens = tokenizer(
                batch_texts, 
                add_special_tokens=False, 
                padding=True,
                return_tensors="pt"
            )
            input_ids = tokens['input_ids'].tolist()  # Convert tensor to list
            
            # Process each text's segments
            for text_input_ids in input_ids:
                # Find actual length by finding first padding token
                try:
                    actual_len = text_input_ids.index(eos_token_id)
                except ValueError:
                    actual_len = len(text_input_ids)

                if actual_len < p_args.data.max_len:
                    continue
                    
                right_idx = min(actual_len, p_args.data.max_len + 3 * stride)
                
                segments = []
                for i in range(0, right_idx, stride):
                    segment = text_input_ids[i:i + p_args.data.max_len]
                    # Only keep segments that are exactly max_len and don't contain padding tokens
                    if len(segment) == p_args.data.max_len and eos_token_id not in segment:
                        segments.append(segment)
                    if len(segments) >= max_segments_per_text:
                        break
                
                all_segments.extend(segments)
        
        return all_segments

    def _process_batch(self, batch: List[str], batch_num: int, rank: int):
        results = []
        num_segments = 0

        num_cpus = cpu_count()
        
        with ThreadPoolExecutor(max_workers=num_cpus) as executor:
            # Create a tokenizer for each thread
            tokenizers = {thread_id: self._get_tokenizer() for thread_id in range(num_cpus)}
            
            def process_slice(thread_id):
                # Get texts for this thread using step slicing
                thread_texts = batch[thread_id::num_cpus]
                tokenizer = tokenizers[thread_id]
                return self._tokenize_and_truncate_with_overlap(thread_texts, tokenizer)

            futures = [executor.submit(process_slice, i) for i in range(num_cpus)]
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.extend(result)
                    num_segments += len(result)
        
        if num_segments > 0:
            output_file = p_args.path.token_dir / f"tokens_r{rank}_b{batch_num}_l{p_args.data.max_len}_o{p_args.data.overlap_len}_{num_segments}.npz"
            # Convert to numpy array and compress
            results = np.array(results, dtype=np.int32)
            
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            np.savez_compressed(output_file, results)
            self.total_segments += num_segments
        
            logger.info(f"✅ Rank {rank}: Saved batch {batch_num} to {output_file}, {num_segments} segments")

    def _preprocess_text(self, texts: List[str], rank: int) -> List[str]:
        new_texts = []
        num_discarded = 0

        # remove unrelated texts
        def _preprocess_text(text: str) -> str:
            # For GPT-2 tokenizer, roughly 4 characters per token
            # For 128*4=512 tokens, we need ~2048 characters, some texts are too long, we approximately truncate them to 2500 characters first
            if len(text) < 300: # some texts are too short, we discard them
                nonlocal num_discarded
                num_discarded += 1
                return None

            text = text[:min(len(text), 2500)]
            text = text.split("Reference")[0]
            text = text.split("Category:")[0].strip()
            return text

        with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
            futures = [executor.submit(_preprocess_text, text) for text in texts]
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    new_texts.append(result)
        
        # logger.info(f"📊 Rank {rank}: Discarded {num_discarded}/{len(texts)} texts")
        return new_texts

    def process_texts(self, texts: List[str], batch_size: int = 500000, rank: int = 0) -> None:
        """Process texts on each node and save results locally."""
        # 1. preprocess the texts
        texts = self._preprocess_text(texts, rank)

        # 2. process the texts
        for i in tqdm.trange(0, len(texts), batch_size):
            right_index = min(i + batch_size, len(texts))
            self._process_batch(texts[i:right_index], i // batch_size, rank)

        logger.info(f"📊 Rank {rank}: Total processed segments: {self.total_segments}")

