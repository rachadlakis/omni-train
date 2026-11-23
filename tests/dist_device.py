"""Where the multi-process (-m dist) tests run.

Default (CI, laptops): CPU processes over gloo, CUDA hidden.
OMNI_DIST_DEVICE=cuda (GPU machines, scripts/gpu_check.sh): one GPU per process over
NCCL. Cases that need more processes than there are GPUs are skipped, because NCCL
refuses two ranks on the same GPU.
"""

import os

import pytest
import torch

ON_GPU = os.environ.get("OMNI_DIST_DEVICE", "cpu").lower() == "cuda"


def dist_env(**extra) -> dict:
    env = dict(os.environ, OMP_NUM_THREADS="1", **extra)
    if not ON_GPU:
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def require_ranks(nproc: int) -> None:
    """Skip when GPU mode has fewer GPUs than ranks."""
    if ON_GPU and torch.cuda.device_count() < nproc:
        pytest.skip(f"needs {nproc} GPUs, found {torch.cuda.device_count()}")
