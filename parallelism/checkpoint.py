"""Sharded checkpoints for strategy: 5d.

    checkpoints/5d/<unix_ms_timestamp>/
        layout.json          # dp/tp/cp/pp/ep/sp/zero1 + model config
        rank_00000.pt ...    # this rank's model shard + optimizer state
        hf/                  # (llm only) normal Hugging Face checkpoint

Every rank writes its own file, so saving needs no gathering. Resuming needs
the SAME layout; to change the layout, start a new run from the exported hf/
folder (model_name: checkpoints/5d/<ts>/hf).
"""

from __future__ import annotations

import json
import os
import time

import torch
import torch.distributed as dist

from .process_groups import ParallelContext


def layout_of(pc: ParallelContext, zero1: bool) -> dict:
    return {"world_size": pc.world_size, "dp": pc.dp_size, "tp": pc.tp_size, "cp": pc.cp_size,
            "pp": pc.pp_size, "ep": pc.ep_size, "sp": pc.sequence_parallel, "zero1": zero1}


def new_run_dir(root: str) -> str:
    """Same timestamped folder on every rank (rank 0 picks it)."""
    name = [str(int(time.time() * 1000))]
    dist.broadcast_object_list(name, src=0)
    return os.path.join(root, "5d", name[0])


def save(run_dir: str, pc: ParallelContext, model, optimizer, meta: dict) -> None:
    if pc.rank == 0:
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "layout.json"), "w") as f:
            json.dump(meta, f, indent=2)
    dist.barrier()
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()},
               os.path.join(run_dir, f"rank_{pc.rank:05d}.pt"))
    dist.barrier()


def latest(root: str) -> str | None:
    base = os.path.join(root, "5d")
    if not os.path.isdir(base):
        return None
    runs = [d for d in os.listdir(base)
            if d.split("__")[0].isdigit() and os.path.exists(os.path.join(base, d, "layout.json"))]
    return os.path.join(base, max(runs, key=lambda d: int(d.split("__")[0]))) if runs else None


def load(run_dir: str, pc: ParallelContext, model, optimizer, expected_layout: dict) -> None:
    with open(os.path.join(run_dir, "layout.json")) as f:
        saved = json.load(f)["layout"]
    if saved != expected_layout:
        raise ValueError(
            f"checkpoint {run_dir} was saved with layout {saved}, but this run uses "
            f"{expected_layout}. Resume needs the same layout; to change it, start from the "
            f"exported Hugging Face folder ({os.path.join(run_dir, 'hf')}) instead.")
    state = torch.load(os.path.join(run_dir, f"rank_{pc.rank:05d}.pt"), map_location="cpu",
                       weights_only=True)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
