import os
import argparse
from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-cache Hugging Face models to local/shared cache")
    parser.add_argument("--model", action="append", required=True, help="Model repo_id (repeatable)")
    parser.add_argument("--revision", type=str, default=None, help="Optional model revision")
    args = parser.parse_args()

    # No cache_dir override: hub files live in $HF_HOME/hub, which is where
    # from_pretrained looks. Passing $HF_HOME itself drops them one level too
    # high, so the pre-cache is invisible to training and every model downloads
    # twice.
    for model_id in args.model:
        print(f"Pre-caching {model_id}...")
        path = snapshot_download(
            repo_id=model_id,
            revision=args.revision,
            local_files_only=False,
        )
        print(f"Done: {model_id} -> {path}")


if __name__ == "__main__":
    main()
