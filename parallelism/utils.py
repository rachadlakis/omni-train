"""Small helpers shared by the other modules."""

from __future__ import annotations

import torch


def seeded_normal(shape, seed: int, std: float = 0.02) -> torch.Tensor:
    """A normal tensor that depends only on `seed`.

    Every module builds its FULL weight from a fixed seed and keeps only its
    shard. So a model split with TP/PP/EP has exactly the same numbers as the
    unsplit model. This makes it easy to test that parallel == single GPU.
    """
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=gen) * std


def mark(param: torch.nn.Parameter, **flags) -> torch.nn.Parameter:
    """Attach flags such as tp_sharded=True or is_expert=True to a parameter."""
    for key, value in flags.items():
        setattr(param, key, value)
    return param


def split_layers(n_layers: int, pp_size: int, pp_rank: int) -> range:
    """Which layers live on this pipeline stage (as even as possible)."""
    base, extra = divmod(n_layers, pp_size)
    start = pp_rank * base + min(pp_rank, extra)
    end = start + base + (1 if pp_rank < extra else 0)
    return range(start, end)
