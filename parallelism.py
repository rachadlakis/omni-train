## parallelism.py
##
## Standalone module for 3D Parallelism (Data × Tensor × Pipeline).
## The original training files (train.py, utils.py, distributed_utils.py,
## config.yaml) are NOT touched by this module.
##
## Usage:
##   from parallelism import ParallelismArgs, resolve_device_mesh, setup_device_mesh
##
##   args = ParallelismArgs.from_config(cfg, num_gpus=4)
##   mesh = setup_device_mesh(args)          # None when dp=1, tp=1, pp=1
##   dp_mesh = mesh["dp"]
##   tp_mesh = mesh["tp"]
##   pp_mesh = mesh["pp"]

from __future__ import annotations

import torch
import torch.distributed as dist
from dataclasses import dataclass, field
from torch.distributed.device_mesh import init_device_mesh, DeviceMesh

# ---------------------------------------------------------------------------
# Auto-detection table
# Chosen to keep TP intra-node (NVLink) and minimise the PP bubble.
# dp × tp × pp MUST equal num_gpus in every row.
# ---------------------------------------------------------------------------
_DEFAULT_MESH: dict[int, tuple[int, int, int]] = {
    1:  (1,  1, 1),   # single GPU — no parallelism
    2:  (2,  1, 1),   # pure data-parallel
    4:  (4,  1, 1),   # pure data-parallel
    8:  (4,  2, 1),   # DP × intra-node TP (tp=2 needs NVLink)
    16: (4,  2, 2),   # 3D: 4 DP × 2 TP × 2 PP
    32: (8,  2, 2),   # 3D at 32 GPUs
    64: (16, 2, 2),   # 3D at 64 GPUs
}


# ---------------------------------------------------------------------------
# ParallelismArgs — typed container for resolved mesh dimensions
# ---------------------------------------------------------------------------

@dataclass
class ParallelismArgs:
    """Holds the resolved (dp, tp, pp) mesh topology and pipeline schedule."""

    dp_size: int = 1          ## data-parallel replicas
    pp_size: int = 1          ## pipeline stages
    tp_size: int = 1          ## tensor-parallel shards per replica
    n_microbatches: int = 4   ## micro-batches per step when pp_size > 1
    pp_schedule: str = "1f1b" ## pipeline schedule: {gpipe, 1f1b}
    auto_detected: bool = True ## True when at least one dim was auto-resolved

    # ------------------------------------------------------------------
    @classmethod
    def from_config(
        cls,
        cfg: dict,
        num_gpus: int,
    ) -> "ParallelismArgs":
        """Build a ParallelismArgs from the raw YAML config dict.

        Reads the top-level ``parallelism:`` section if present.
        Any value set to ``null`` (Python ``None``) is auto-resolved
        from the built-in ``_DEFAULT_MESH`` table.

        Args:
            cfg      : the full parsed YAML config dict.
            num_gpus : total GPU count (e.g. from ``cfg["num_gpus"]``).

        Returns:
            A fully resolved ``ParallelismArgs``.
        """
        parallelism = cfg.get("parallelism", {}) or {}

        user_dp = parallelism.get("dp_size", None)
        user_pp = parallelism.get("pp_size", None)
        user_tp = parallelism.get("tp_size", None)

        dp, pp, tp, auto = resolve_device_mesh(num_gpus, user_dp, user_pp, user_tp)

        schedule = str(parallelism.get("pp_schedule", "1f1b")).lower()
        if schedule not in {"gpipe", "1f1b"}:
            raise ValueError(
                f"parallelism.pp_schedule='{schedule}' is not supported. "
                "Choose from: gpipe, 1f1b"
            )

        return cls(
            dp_size=dp,
            pp_size=pp,
            tp_size=tp,
            n_microbatches=int(parallelism.get("n_microbatches", 4)),
            pp_schedule=schedule,
            auto_detected=auto,
        )

    # ------------------------------------------------------------------
    @property
    def is_3d(self) -> bool:
        """True when at least one of TP or PP is enabled."""
        return self.pp_size > 1 or self.tp_size > 1

    @property
    def is_flat_dp(self) -> bool:
        """True when only data parallelism is active (FSDP-only path)."""
        return self.pp_size == 1 and self.tp_size == 1

    def __str__(self) -> str:
        tag = "(auto)" if self.auto_detected else "(user)"
        return (
            f"Mesh {tag}: dp={self.dp_size} × pp={self.pp_size} × tp={self.tp_size} "
            f"[total={self.dp_size * self.pp_size * self.tp_size}]"
        )


# ---------------------------------------------------------------------------
# resolve_device_mesh — pure function, no side-effects
# ---------------------------------------------------------------------------

def resolve_device_mesh(
    num_gpus: int,
    user_dp: "int | None",
    user_pp: "int | None",
    user_tp: "int | None",
) -> "tuple[int, int, int, bool]":
    """Return ``(dp_size, tp_size, pp_size, auto_flag)``.

    Any dimension the caller passes as ``None`` is auto-filled from the
    built-in default table (or derived so that dp × tp × pp == num_gpus).

    ``auto_flag`` is ``True`` when at least one dimension was auto-resolved.

    Raises:
        ValueError: if the user-supplied combination is inconsistent.
    """
    any_auto = (user_dp is None) or (user_pp is None) or (user_tp is None)

    # ---- All three specified → just validate ----
    if not any_auto:
        dp, pp, tp = int(user_dp), int(user_pp), int(user_tp)   # type: ignore[arg-type]
        if dp * pp * tp != num_gpus:
            raise ValueError(
                f"parallelism: dp_size ({dp}) × pp_size ({pp}) × tp_size ({tp}) = "
                f"{dp * pp * tp}, but num_gpus = {num_gpus}. They must be equal."
            )
        for name, val in [("dp_size", dp), ("pp_size", pp), ("tp_size", tp)]:
            if val < 1:
                raise ValueError(f"parallelism.{name} must be ≥ 1, got {val}")
        return dp, pp, tp, False

    # ---- At least one is auto → look up defaults then fill gaps ----
    if num_gpus in _DEFAULT_MESH:
        default_dp, default_tp, default_pp = _DEFAULT_MESH[num_gpus]
    else:
        # Unusual GPU count: safe fallback is pure data-parallel
        default_dp, default_tp, default_pp = num_gpus, 1, 1


    dp = int(user_dp) if user_dp is not None else default_dp
    pp = int(user_pp) if user_pp is not None else default_pp
    tp = int(user_tp) if user_tp is not None else default_tp

    # If only one dim was pinned by the user, derive dp so the product holds
    n_pinned = sum(x is not None for x in [user_dp, user_pp, user_tp])
    if n_pinned == 1:
        if user_pp is not None:
            tp = 1
            dp = num_gpus // (pp * tp)
        elif user_tp is not None:
            pp = 1
            dp = num_gpus // (pp * tp)
        # user_dp only → pp and tp come from the table; dp is already set

    if dp * pp * tp != num_gpus:
        raise ValueError(
            f"parallelism auto-resolution produced dp={dp} × pp={pp} × tp={tp} = "
            f"{dp * pp * tp}, but num_gpus={num_gpus}. "
            "Please specify dp_size, pp_size, and tp_size explicitly."
        )
    for name, val in [("dp_size", dp), ("pp_size", pp), ("tp_size", tp)]:
        if val < 1:
            raise ValueError(f"parallelism.{name} must be ≥ 1, got {val}")

    return dp, pp, tp, True


# ---------------------------------------------------------------------------
# setup_device_mesh — builds the DeviceMesh after dist.init_process_group
# ---------------------------------------------------------------------------

def setup_device_mesh(args: ParallelismArgs) -> "DeviceMesh | None":
    """Build and return a 3D ``DeviceMesh`` when any parallelism dim > 1.

    Returns ``None`` when the topology is (1, 1, 1) so callers can skip
    mesh-aware code paths entirely (standard FSDP/DDP still works).

    Mesh dimension order is **(dp, pp, tp)** — the layout used by TorchTitan:

    ::

        dim 0 → data-parallel    (dp_size replicas)
        dim 1 → pipeline-parallel (pp_size stages)
        dim 2 → tensor-parallel  (tp_size shards)

    Sub-meshes:
        ``mesh["dp"]``, ``mesh["pp"]``, ``mesh["tp"]``

    Raises:
        RuntimeError: if called before ``dist.init_process_group()``.
    """
    dp, pp, tp = args.dp_size, args.pp_size, args.tp_size

    if dp == 1 and pp == 1 and tp == 1:
        _log("DeviceMesh: flat (1×1×1) — no mesh constructed 🔲")
        return None

    if not dist.is_initialized():
        raise RuntimeError(
            "setup_device_mesh() must be called AFTER dist.init_process_group()."
        )

    backend = "cuda" if torch.cuda.is_available() else "cpu"
    mesh = init_device_mesh(
        backend,
        (dp, pp, tp),
        mesh_dim_names=("dp", "pp", "tp"),
    )

    _log(f"DeviceMesh {args} 🔲")
    return mesh


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    """Print only from rank 0 (or when dist is not yet initialised)."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        print(f"\n   {msg}", flush=True)
