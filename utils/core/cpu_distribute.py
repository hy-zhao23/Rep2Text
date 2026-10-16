import os
import time
from pathlib import Path
from typing import List, Any
from .logger import logger

try:
    from mpi4py import MPI
    _MPI_ERROR = None
except Exception as exc:  # mpi4py installed but no libmpi at runtime, or absent
    MPI = None
    _MPI_ERROR = exc


class MPIHelper:
    """
    Helper class for distributed cpu tasks using MPI.
    Only for 1 task per node.

    Falls back to SLURM's own rank environment when mpi4py cannot load an MPI
    runtime (the `emb` env is built without one -- see requirements_*_nompi.txt).
    Rank splitting is pure arithmetic and the barriers only need to be visible
    on the shared filesystem, so plain `srun` is enough; the MPI path is kept
    for environments that do have a working libmpi.
    """
    _BARRIER_POLL_SEC = 5
    _BARRIER_TIMEOUT_SEC = 24 * 3600  # rank 0 may be downloading the corpus

    def __init__(self):
        """Initialize MPI environment"""
        # Under srun, SLURM is the authority on how many ranks exist. mpi4py may
        # still import (login nodes keep an MPI in LD_LIBRARY_PATH, compute
        # nodes do not after `module purge`), but without PMI wiring every task
        # would see COMM_WORLD size 1, silently write the same shard filenames
        # and corrupt the sample pool. Trust SLURM whenever it says >1 task.
        slurm_ntasks = int(os.getenv("SLURM_NTASKS", "1"))
        self.comm = MPI.COMM_WORLD if (MPI is not None and slurm_ntasks <= 1) else None
        self._barrier_idx = 0

        if self.comm is not None:
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
            logger.info(f"✅ MPIHelper initialized on node {self.rank}/{self.size}")
        else:
            self.rank = int(os.getenv("SLURM_PROCID", "0"))
            self.size = slurm_ntasks
            reason = "multi-task srun step" if slurm_ntasks > 1 else f"mpi4py unavailable ({_MPI_ERROR})"
            logger.warning(
                f"⚠️ Using SLURM ranks {self.rank}/{self.size} with file-based "
                f"barriers ({reason})"
            )

    def get_rank_tasks(self, task_list: List[Any]) -> List[Any]:
        task_list = task_list[self.rank::self.size]
        logger.info(f"🔍 MPIHelper distributed {len(task_list)} tasks to node {self.rank}")
        return task_list

    def _file_barrier(self):
        """Rendezvous through the shared filesystem: every rank drops a marker
        and waits until all of them are visible."""
        if self.size <= 1:
            return

        self._barrier_idx += 1
        job = os.getenv("SLURM_JOB_ID", "nojobid")
        bdir = Path(os.getenv("PROJ", ".")) / "tmp" / "barriers" / job / f"b{self._barrier_idx}"
        bdir.mkdir(parents=True, exist_ok=True)
        (bdir / f"rank_{self.rank}").touch()

        deadline = time.time() + self._BARRIER_TIMEOUT_SEC
        while len(list(bdir.glob("rank_*"))) < self.size:
            if time.time() > deadline:
                arrived = len(list(bdir.glob("rank_*")))
                raise TimeoutError(
                    f"barrier {self._barrier_idx} timed out on rank {self.rank}: "
                    f"{arrived}/{self.size} ranks arrived"
                )
            time.sleep(self._BARRIER_POLL_SEC)

    def barrier(self):
        """Synchronize all processes"""
        if self.comm is not None:
            self.comm.Barrier()
        else:
            self._file_barrier()

    @property
    def is_main(self):
        """Check if current process is the main process (rank 0)"""
        return self.rank == 0

mpi_helper = MPIHelper()
