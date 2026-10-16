#!/usr/bin/env python
"""Upload large artifacts (embedding caches, trained projectors) to a HF repo.

Uses `upload_large_folder` with `hf_transfer` enabled, which is resumable — re-running
skips blobs whose SHA already exists on the remote. The repo layout mirrors the local
paths relative to $PROJ.

Examples:
    # dry run: show what would go up
    python experiments/upload_to_hf.py --repo user/my-bundle \
        --paths tmp/embeddings/llama8B/l10t16 --dry-run

    # speed test on a single shard before committing to a multi-hour upload
    python experiments/upload_to_hf.py --repo user/my-bundle \
        --paths tmp/embeddings/llama8B/l10t16 --probe

    # the real thing
    python experiments/upload_to_hf.py --repo user/my-bundle \
        --paths tmp/embeddings/llama8B/l10t16 weights/llama8B

Requires HF_TOKEN (or HF_API_KEY) in the environment, or a token cached by
`huggingface-cli login`.
"""
import argparse
import os
import sys
import time
from pathlib import Path

PROJ = Path(os.environ.get("PROJ", Path(__file__).resolve().parents[1]))


def resolve_token() -> str:
    for var in ("HF_TOKEN", "HF_API_KEY"):
        if os.environ.get(var):
            return os.environ[var]
    cached = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "token"
    if cached.exists():
        return cached.read_text().strip()
    sys.exit("ERROR: no HF token. Set HF_TOKEN, or run `huggingface-cli login`.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=os.environ.get("HF_DATASET_REPO"), required=False,
                        help="target repo id, e.g. user/my-bundle (or set HF_DATASET_REPO)")
    parser.add_argument("--paths", nargs="+", required=True,
                        help="paths to upload, relative to $PROJ")
    parser.add_argument("--repo-type", default="dataset", choices=["dataset", "model"])
    parser.add_argument("--private", action="store_true", default=True)
    parser.add_argument("--public", dest="private", action="store_false")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the upload plan and exit")
    parser.add_argument("--probe", action="store_true",
                        help="upload only the first .npz found, as a throughput test")
    args = parser.parse_args()

    if not args.repo:
        sys.exit("ERROR: --repo not set and HF_DATASET_REPO is not in the environment")

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    try:
        import hf_transfer  # noqa: F401
    except ImportError:
        sys.exit("ERROR: pip install hf_transfer  (needed for fast multi-stream upload)")

    from huggingface_hub import HfApi, create_repo, upload_file
    try:
        from huggingface_hub.errors import HfHubHTTPError
    except ImportError:
        from huggingface_hub.utils import HfHubHTTPError

    token = resolve_token()
    api = HfApi(token=token)

    try:
        create_repo(args.repo, repo_type=args.repo_type, private=args.private,
                    exist_ok=True, token=token)
        print(f"[ok] {args.repo_type} repo ready: {args.repo}")
    except HfHubHTTPError as e:
        sys.exit(f"create_repo failed: {e}")

    if args.probe:
        for rel in args.paths:
            d = PROJ / rel
            files = sorted(d.glob("*.npz")) if d.is_dir() else []
            if not files:
                continue
            f = files[0]
            gb = f.stat().st_size / 1e9
            print(f"[probe] uploading {f.name} ({gb:.2f} GB)")
            if args.dry_run:
                return
            t0 = time.time()
            upload_file(path_or_fileobj=str(f), path_in_repo=str(f.relative_to(PROJ)),
                        repo_id=args.repo, repo_type=args.repo_type, token=token)
            dt = time.time() - t0
            print(f"[probe] {dt:.1f}s -> {f.stat().st_size / 1e6 / dt:.1f} MB/s")
            return
        sys.exit("probe found no .npz files in the given paths")

    # One call from $PROJ with allow_patterns, so the remote layout mirrors local paths.
    allow_patterns, total_files, total_gb = [], 0, 0.0
    for rel in args.paths:
        d = PROJ / rel
        if not d.exists():
            print(f"[skip] missing: {d}")
            continue
        if d.is_file():
            total_files += 1
            total_gb += d.stat().st_size / 1e9
            allow_patterns.append(rel)
            print(f"[plan] {rel}  (file, {d.stat().st_size / 1e9:.3f} GB)")
            continue
        files = [p for p in d.rglob("*") if p.is_file()]
        gb = sum(p.stat().st_size for p in files) / 1e9
        total_files += len(files)
        total_gb += gb
        allow_patterns.append(f"{rel}/**")
        print(f"[plan] {rel}  ({len(files)} files, {gb:.1f} GB)")

    if not allow_patterns:
        sys.exit("nothing to upload")

    print(f"[upload] {total_files} files, {total_gb:.1f} GB -> {args.repo}")
    if args.dry_run:
        return

    api.upload_large_folder(repo_id=args.repo, repo_type=args.repo_type,
                            folder_path=str(PROJ), allow_patterns=allow_patterns,
                            num_workers=args.workers)
    print("[done] upload complete")


if __name__ == "__main__":
    main()
