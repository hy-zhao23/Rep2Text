from __future__ import annotations

import os
from datetime import timedelta
from typing import Any, List, Optional, Union

import numpy as np
import torch
import torch.distributed as dist


class GPUHelper:
    _instance: Optional["GPUHelper"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # 只设置“默认值”，不覆盖你在 slurm 里 export 的内容
        os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
        os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
        os.environ.setdefault("NCCL_TIMEOUT", "1800")

        self.accelerator = None  # optional

    # -------------------------
    # Rank / device info
    # -------------------------
    @property
    def local_rank(self) -> int:
        return int(os.getenv("LOCAL_RANK", os.getenv("SLURM_LOCALID", "0")))

    @property
    def rank(self) -> int:
        if self.accelerator is not None:
            return int(self.accelerator.process_index)
        return int(os.getenv("RANK", os.getenv("SLURM_PROCID", "0")))

    @property
    def world_size(self) -> int:
        if self.accelerator is not None:
            return int(self.accelerator.num_processes)
        return int(os.getenv("WORLD_SIZE", os.getenv("SLURM_NTASKS", "1")))

    @property
    def is_main(self) -> bool:
        if self.accelerator is not None:
            return bool(self.accelerator.is_main_process)
        return self.rank == 0

    @property
    def distributed(self) -> bool:
        if self.accelerator is not None:
            return self.accelerator.num_processes > 1
        return dist.is_available() and dist.is_initialized() and self.world_size > 1

    @property
    def device(self) -> torch.device:
        if self.accelerator is not None:
            return self.accelerator.device
        if torch.cuda.is_available():
            return torch.device(f"cuda:{self.local_rank}")
        return torch.device("cpu")

    # -------------------------
    # Accelerator hooks (optional)
    # -------------------------
    def enable_accelerator(self, accelerator) -> None:
        self.accelerator = accelerator

    def unwrap_model(self, model):
        if self.accelerator is not None:
            return self.accelerator.unwrap_model(model)
        return model

    # -------------------------
    # Distributed init (DEFAULT: NCCL)
    # -------------------------
    def init_distributed(
        self,
        backend: str = "nccl",
        timeout_minutes: int | None = None,
        init_method: str = "env://",
        require_cuda_for_nccl: bool = True,
    ) -> None:
        """
        显式初始化 default process group.
        - 默认 backend="nccl"
        - world_size<=1 则直接 return
        - 已初始化则做 backend sanity check + set_device 对齐
        """
        if self.world_size <= 1:
            return

        if dist.is_initialized():
            cur = dist.get_backend()
            if cur != backend:
                raise RuntimeError(
                    f"dist already initialized with backend={cur} but expected {backend}. "
                    "说明有地方提前 init 了（可能是 accelerate / 其他工具）。"
                )
            if backend == "nccl" and torch.cuda.is_available():
                torch.cuda.set_device(self.local_rank)
            return

        if backend == "nccl":
            if require_cuda_for_nccl and not torch.cuda.is_available():
                raise RuntimeError("NCCL backend requires CUDA but cuda is not available.")
            torch.cuda.set_device(self.local_rank)

        if timeout_minutes is None:
            timeout_minutes = int(os.getenv("DIST_TIMEOUT_MINUTES", "30"))

        init_kwargs = dict(
            backend=backend,
            init_method=init_method,
            timeout=timedelta(minutes=timeout_minutes),
        )

        # NOTE: Do NOT add device_id for NCCL - it triggers split_group() which NCCL doesn't support
        # This breaks DeviceMesh initialization for TP/DP mesh creation
        dist.init_process_group(**init_kwargs)

        if backend == "nccl" and torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)

    # -------------------------
    # Sync
    # -------------------------
    def barrier(self) -> None:
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
            return
        if dist.is_available() and dist.is_initialized():
            if torch.cuda.is_available():
                dist.barrier(device_ids=[torch.cuda.current_device()])
            else:
                dist.barrier()
    
    def destroy(self) -> None:
        if self.accelerator is not None:
            return
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(tensor, src=src)
        return tensor

    def all_reduce(self, tensor: torch.Tensor, op: str = "sum") -> torch.Tensor:
        if not (dist.is_available() and dist.is_initialized()):
            return tensor
        op_map = {"sum": dist.ReduceOp.SUM, "mean": dist.ReduceOp.SUM, "max": dist.ReduceOp.MAX}
        dist.all_reduce(tensor, op=op_map.get(op, dist.ReduceOp.SUM))
        if op == "mean":
            tensor.div_(self.world_size)
        return tensor

    def get_rank_tasks(self, task_list: Union[List[Any], np.ndarray]):
        if self.world_size <= 1:
            return task_list
        return task_list[self.rank :: self.world_size]


gh = GPUHelper()