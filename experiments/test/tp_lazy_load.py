"""Check that --lazy_load matches the eager load-then-shard path.

The lazy path hands the TP mesh to from_pretrained so each rank reads only its
own slice of every weight straight onto its GPU; the eager path materializes the
whole checkpoint in host RAM first and shards afterwards. Both should produce
identical hidden states — this script runs one mode per invocation, saves the
backbone output, and reports peak host RSS so the two can be diffed.

    torchrun --nproc_per_node=2 experiments/test/tp_lazy_load.py --mode lazy --out /tmp/lazy.pt
"""
import argparse
import importlib.util
import os

import torch

from utils import gh
from utils.models import load_pretrained_model

_PRETRAIN_TD = os.path.join(os.path.dirname(__file__), "..", "train", "pretrain-td.py")


def _load_pretrain_td():
    # Module name has a hyphen, so it cannot be imported the normal way. We want
    # the real _apply_tp_plan_llama_like, not a copy, or the test proves nothing.
    spec = importlib.util.spec_from_file_location("pretrain_td", _PRETRAIN_TD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def peak_host_rss_gb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / (1024 * 1024)
    return float("nan")


def main():
    ap = argparse.ArgumentParser("lazy vs eager TP load equivalence check")
    ap.add_argument("--mode", choices=["eager", "lazy"], required=True)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--seq_len", type=int, default=64)
    args = ap.parse_args()

    torch.cuda.set_device(gh.local_rank)
    gh.init_distributed()

    from torch.distributed.device_mesh import init_device_mesh

    world = int(os.environ["WORLD_SIZE"])
    if world != args.tp:
        raise RuntimeError(f"world_size={world} must equal tp={args.tp} (dp=1 for this test)")

    mesh = init_device_mesh("cuda", (1, args.tp), mesh_dim_names=("dp", "tp"))
    tp_mesh = mesh["tp"]

    lazy = args.mode == "lazy"
    lazy_kwargs = {"tp_plan": "auto", "device_mesh": tp_mesh} if lazy else {}

    _, model, _ = load_pretrained_model(
        model_type="einv",
        resume_from=False,
        device_map=None,
        **lazy_kwargs,
    )

    if not lazy:
        ptd = _load_pretrain_td()
        model = ptd._apply_tp_plan_llama_like(model, tp_mesh)
        model = model.to(gh.device)

    model.eval()

    # Deterministic input — no sampling, no dataset, so any mismatch is the load path.
    ids = torch.arange(1, args.seq_len + 1, device=gh.device).unsqueeze(0)
    with torch.no_grad():
        hidden = model.model(input_ids=ids).last_hidden_state

    rank = int(os.environ["RANK"])
    print(f"[rank{rank}] mode={args.mode} peak_host_rss={peak_host_rss_gb():.2f} GB "
          f"gpu_alloc={torch.cuda.max_memory_allocated() / 1024**3:.2f} GB", flush=True)

    if rank == 0:
        torch.save(hidden.float().cpu(), args.out)
        print(f"[rank0] saved {tuple(hidden.shape)} -> {args.out}", flush=True)

    gh.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
