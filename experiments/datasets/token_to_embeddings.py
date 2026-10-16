from utils import p_args
from utils.data.embeddings import tokens_to_embeddings
from utils.core.gpu_distribute import gh
import numpy as np
from transformers import AutoTokenizer

def get_rank_train_tokens():
    token_file = p_args.path.token_dir / f"train_tokens_{p_args.data.train_size}_l{p_args.data.max_len}_o{p_args.data.overlap_len}.npy"
    if not token_file.exists():
        raise FileNotFoundError(f"Token file not found: {token_file}")

    train_tokens = np.load(token_file)

    tok = AutoTokenizer.from_pretrained(
        p_args.model.emb_id,
        use_fast=True,
        cache_dir=p_args.path.weight_dir,
    )
    max_token_id = int(train_tokens.max())
    vocab_size = int(tok.vocab_size)
    if max_token_id >= vocab_size:
        raise ValueError(
            f"Token ID out of vocab range for emb model. token_file={token_file}, "
            f"max_token_id={max_token_id}, emb_id={p_args.model.emb_id}, vocab_size={vocab_size}. "
            "Please regenerate train tokens with the same emb_id tokenizer."
        )

    return gh.get_rank_tasks(train_tokens)

def main():
    # 1. get rank tokens and save their embeddings in batches
    rank_tokens = get_rank_train_tokens()

    # 2. save their embeddings in batches
    tokens_to_embeddings(rank_tokens)

    gh.barrier()
    
    return 
    
if __name__ == "__main__":
    main()