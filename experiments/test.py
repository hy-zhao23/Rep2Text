#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import socket

import torch

from utils import gh  # 改成你项目里的真实路径


def env_snapshot():
    keys = [
        "RANK", "WORLD_SIZE", "LOCAL_RANK",
        "SLURM_NODEID", "SLURM_PROCID", "SLURM_LOCALID", "SLURM_NTASKS",
        "MASTER_ADDR", "MASTER_PORT", "CUDA_VISIBLE_DEVICES",
    ]
    return {k: os.environ.get(k) for k in keys}


def p(msg: str):
    print(msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout_min", type=int, default=30)
    ap.add_argument("--do_allreduce", action="store_true")
    args = ap.parse_args()

    p(f"=== host={socket.gethostname()} torch={torch.__version__} cuda={torch.version.cuda} ===")
    p(f"[env] {env_snapshot()}")

    # 1) init via gh
    gh.init_distributed(backend="nccl", timeout_minutes=args.timeout_min, init_method="env://")

    # 2) basic cuda info
    p(f"[gh] rank={gh.rank} world={gh.world_size} local_rank={gh.local_rank} device={gh.device}")
    if torch.cuda.is_available():
        p(f"[cuda] current_device={torch.cuda.current_device()} name={torch.cuda.get_device_name(torch.cuda.current_device())}")

    # 3) barrier
    gh.barrier()
    if gh.is_main:
        p("[barrier] OK")

    # 4) optional allreduce sanity
    if args.do_allreduce:
        x = torch.tensor([float(gh.rank)], device=gh.device)
        gh.all_reduce(x, op="sum")
        p(f"[allreduce] sum(ranks)={x.item()} (rank={gh.rank})")

    # 5) clean exit
    try:
        gh.barrier()
        if gh.is_main:
            print("[barrier] OK", flush=True)
        gh.destroy()
        print("[done] exit", flush=True)
        pass
    finally:
        p("[done] exit")



if __name__ == "__main__":
    main()