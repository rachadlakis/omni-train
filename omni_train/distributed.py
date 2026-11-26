"""Process groups: the backend (NCCL or gloo), setup, teardown and barriers."""

from datetime import timedelta
import os
import sys

import torch
import torch.distributed as dist

from omni_train.console import print_on_rank_0


if torch.cuda.is_available():
    if not sys.platform.startswith("linux"):
        raise RuntimeError(
            "The current distributed CUDA path uses NCCL, and NCCL is supported on Linux for now."
            "Non-Linux backends can be added later."
            f"Current platform: {sys.platform}"
        )
    BACKEND = "nccl"
else:
    BACKEND = "gloo"


def _detect_nvlink() -> bool:
    """Return True if NVML reports at least one active NVLink between local GPUs.

    Failure modes are deliberately split:
      - pynvml not installed         → RuntimeError. We have no way to know whether
                                       NVLink is present, so silently falling back
                                       to SHM would lie to the user on NVLink boxes.
                                       The dependency is pinned in requirements.txt
                                       (nvidia-ml-py); the user must install it.
      - pynvml present but NVML errs → return False. The probe was attempted, NVML
                                       just couldn't give us an answer (unsupported
                                       driver, transient error). SHM fallback is safe.
    """
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return False
    try:
        import pynvml  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "Cannot determine NVLink presence: 'pynvml' (nvidia-ml-py) is not installed.\n"
            "   Refusing to silently assume PCIe-only — that would force SHM transport\n"
            "   on NVLink-equipped machines and hide a real performance regression.\n"
            "   Install it with:\n\n"
            "       pip install -r requirements.txt\n\n"
            "   (or `pip install nvidia-ml-py==12.560.30`) and re-run."
        ) from e
    try:
        pynvml.nvmlInit()
        try:
            for i in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                for link in range(pynvml.NVML_NVLINK_MAX_LINKS):
                    try:
                        if pynvml.nvmlDeviceGetNvLinkState(handle, link) == pynvml.NVML_FEATURE_ENABLED:
                            return True
                    except pynvml.NVMLError:
                        break
            return False
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return False


def setup_dist_process_group():
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    try:
        print_on_rank_0(rank, f"Initializing process group with backend: {BACKEND}", "⚙️")

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            # On PCIe-only GPUs (e.g. RTX A4000) NCCL's P2P topology probe hangs
            # indefinitely; SHM (shared-memory) transport is fast and reliable instead.
            # On NVLink-equipped GPUs (A100/H100/V100) P2P is much faster than SHM, so
            # we leave it enabled. `setdefault` lets the user override either way via .env.
            p2p_default = "0" if _detect_nvlink() else "1"
            os.environ.setdefault("NCCL_P2P_DISABLE", p2p_default)
            print_on_rank_0(
                rank,
                f"NCCL_P2P_DISABLE={os.environ['NCCL_P2P_DISABLE']} "
                f"(NVLink {'detected' if p2p_default == '0' else 'not detected'})",
                "🔌",
            )

        # Do NOT pass device_id= here: it triggers an eager NCCL communicator that conflicts
        # with the device_ids=[local_rank] in barrier() calls, deadlocking on NCCL 2.21.5.
        dist.init_process_group(backend=BACKEND, timeout=timedelta(minutes=10))

        print_on_rank_0(rank, f"Process group initialized ✓ | rank: {rank} | local_rank: {local_rank}", "✅")

        # NCCL warm-up: absorb the first-collective cold-start here, where both ranks are
        # already synchronised, rather than silently mid-training.
        if BACKEND == "nccl":
            print_on_rank_0(rank, "NCCL warm-up barrier…", "🔥")
            dist.barrier(device_ids=[local_rank])
            print_on_rank_0(rank, "NCCL ready ✓", "🔥")

        return local_rank

    except Exception as e:
        print_on_rank_0(rank, f"❌ Failed to initialize process group: {e}", "❌")
        raise


def cleanup():
    try:
        if dist.is_available() and dist.is_initialized():
            # Drain pending CUDA work so NCCL aborts its communicator from a clean state.
            # Without this, NCCL 2.21.5 can report a misleading "unhandled cuda error /
            # out of memory" while tearing down — after all real work is already done.
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dist.destroy_process_group()
    except Exception as e:
        # Teardown runs after training + checkpointing complete, so a NCCL error here does
        # not affect results. Report it once, quietly, instead of a scary per-rank warning.
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"ℹ️  Process group teardown note (harmless — run already completed): {e}", flush=True)


def dist_barrier(local_rank: int) -> None:
    # Must pass device_ids=[local_rank] for NCCL so it knows which GPU each rank owns.
    # Do NOT use device_id= in init_process_group — that creates a conflicting communicator
    # on NCCL 2.21.5 + RTX A4000 and causes deadlocks.  device_ids here is sufficient.
    if dist.get_backend() == "nccl":
        dist.barrier(device_ids=[local_rank])
    else:
        dist.barrier()
