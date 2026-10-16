from utils import p_args, logger
from datasets import load_dataset
from multiprocessing import cpu_count


# Commit the corpus was originally pulled at, recovered from the dataset cache
# directory name. Token sampling is seeded, so re-tokenizing reproduces the exact
# same 2M subset — but only against this revision. Left unpinned, a changed
# upstream repo would silently yield a different sample while the surviving
# split-index pkls stay structurally valid, so nothing would ever error out.
WIKIPEDIA_REVISION = "970296c12832ff2131ea13a04c35d8949cd7ee1e"


def download_wikipedia(subset_of_interest: str = "Wikipedia (en)", num_proc: int = None,
                       revision: str = WIKIPEDIA_REVISION):

    dir = p_args.path.data_dir
    if num_proc is None:
        # can not have too many requests to HF
        num_proc = min(cpu_count() - 1, 32)
    dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Rank 0: Downloading Wikipedia dataset to {dir} with {num_proc} processes (revision={revision})...")
    # Only rank 0 downloads from HF
    dataset = load_dataset(
        "ArmelR/the-pile-splitted",
        subset_of_interest,
        num_proc=num_proc,
        cache_dir=str(dir),
        revision=revision,
    )

    return dataset