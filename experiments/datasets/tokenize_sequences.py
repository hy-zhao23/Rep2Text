from utils.data.preprocess import download_wikipedia
from utils.data import TokenizeProcessor, save_train_tokens
from utils import p_args, logger
from pathlib import Path
from utils.core import read_pkl, write_pkl
from utils.core.cpu_distribute import mpi_helper
import gc

def combine_texts(dataset) -> None:
    all_texts = []
    from concurrent.futures import ThreadPoolExecutor

    if "train" in dataset:
        with ThreadPoolExecutor() as executor:
            texts = list(executor.map(lambda item: item["text"], dataset["train"]))
            all_texts.extend(texts)
    if "test" in dataset:
        with ThreadPoolExecutor() as executor:
            texts = list(executor.map(lambda item: item["text"], dataset["test"]))
            all_texts.extend(texts)

    write_pkl(all_texts, Path(p_args.path.data_dir) / "all_texts.pkl")

def read_rank_texts():
    file = Path(p_args.path.data_dir) / "all_texts.pkl"
    if file.exists():
        rank_texts = read_pkl(file)
        if rank_texts is not None:
            rank_texts = mpi_helper.get_rank_tasks(rank_texts)
            return rank_texts
    raise FileNotFoundError(f"❗️ Rank {mpi_helper.rank} texts file not found")

def main():
    file = Path(p_args.path.data_dir) / "all_texts.pkl"
    if not file.exists():
        # Building the corpus here used to run on rank 0 while 29 nodes idled at
        # the barrier, and HF's locks did not serialize it (the lock name embeds
        # the absolute path, so /path/... and /real/path/... never
        # collided) — two ranks downloaded at once and one wiped the other's
        # .incomplete dir. It belongs in a single-process job.
        raise FileNotFoundError(
            f"❗️ {file} not found. Run experiments/datasets/prepare_corpus.py first "
            f"(sh/data/prepare_corpus.sh)."
        )

    # 3. split the texts into rank_texts
    rank_texts = read_rank_texts()

    # 4. tokenize the texts on each rank
    tokenize_processor = TokenizeProcessor()
    tokenize_processor.process_texts(rank_texts, batch_size=100000, rank=mpi_helper.rank)

    # 5. wait for all ranks to finish tokenization before combining
    mpi_helper.barrier()

    if mpi_helper.is_main:
        logger.info("✅ All ranks finished tokenization. Now combining the tokens on main process...")
        save_train_tokens()

    mpi_helper.barrier()

if __name__ == "__main__":
    main()