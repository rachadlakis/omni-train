"""Tensor parallelism (TP) and sequence parallelism (SP).

A "TP region" is a block such as attention or the MLP:

    enter_tp_region  ->  ColumnParallelLinear(s)  ->  ...  ->  RowParallelLinear  ->  exit_tp_region

Without SP the residual stream is full size on every TP rank:
    enter = identity   (backward: all-reduce)
    exit  = all-reduce (backward: identity)

With SP the residual stream is split along the sequence (dim 1) between TP ranks,
so LayerNorm/RMSNorm and dropout also use 1/tp of the memory:
    enter = all-gather on sequence     (backward: reduce-scatter)
    exit  = reduce-scatter on sequence (backward: all-gather)

The total traffic is the same: all-reduce = reduce-scatter + all-gather.

Every sharded parameter carries `tp_sharded=True` and `tp_dim` (the dimension it
is split on), which the HF loader and the HF exporter use to slice / gather it.
With `seed=None` a layer allocates an EMPTY shard (weights come from a
checkpoint); with a seed it builds the full weight and keeps its slice, so every
layout starts from exactly the same numbers.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import comm
from .process_groups import ParallelContext
from .utils import mark, seeded_normal

SEQ_DIM = 1  # activations are [batch, seq, hidden]


def enter_tp_region(x: torch.Tensor, pc: ParallelContext) -> torch.Tensor:
    if pc.sequence_parallel:
        return comm.gather_sequence(x, SEQ_DIM, pc.tp_group)
    return comm.copy_to_group(x, pc.tp_group)


def exit_tp_region(x: torch.Tensor, pc: ParallelContext) -> torch.Tensor:
    if pc.sequence_parallel:
        return comm.reduce_scatter_to_group(x, SEQ_DIM, pc.tp_group)
    return comm.reduce_from_group(x, pc.tp_group)


def _shard(shape, dim: int, pc: ParallelContext, seed, std: float, pad_to: int | None = None):
    """This rank's slice of a weight of `shape`, split on `dim`.

    pad_to: grow `dim` to this size with zero rows first (vocab padding)."""
    size = pad_to or shape[dim]
    if size % pc.tp_size:
        raise ValueError(f"dimension {size} is not divisible by tp={pc.tp_size}")
    local = list(shape)
    local[dim] = size // pc.tp_size
    if seed is None:
        return torch.empty(local)
    full = seeded_normal(shape, seed, std)
    if pad_to and pad_to > shape[dim]:
        pad = list(shape)
        pad[dim] = pad_to - shape[dim]
        full = torch.cat([full, torch.zeros(pad)], dim=dim)
    return full.chunk(pc.tp_size, dim=dim)[pc.tp_rank].clone()


class ColumnParallelLinear(nn.Module):
    """y = x W^T + b, with the OUTPUT features split between TP ranks.

    Each rank holds out_features / tp rows of W and produces its slice of y.
    No communication here: the caller enters the TP region first.
    """

    def __init__(self, in_features: int, out_features: int, pc: ParallelContext,
                 seed: int | None, std: float = 0.02, bias: bool = False,
                 pad_out_to: int | None = None):
        super().__init__()
        w = _shard((out_features, in_features), 0, pc, seed, std, pad_out_to)
        self.weight = mark(nn.Parameter(w), tp_sharded=True, tp_dim=0)
        self.bias = None
        if bias:
            # Bias rows follow the weight rows. Seeded runs start it at zero.
            b = torch.zeros(w.shape[0]) if seed is not None else torch.empty(w.shape[0])
            self.bias = mark(nn.Parameter(b), tp_sharded=True, tp_dim=0)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    """y = x W^T, with the INPUT features split between TP ranks.

    Each rank holds in_features / tp columns of W and gets its slice of x
    (which is exactly what a ColumnParallelLinear produced). The partial sums
    are added by the caller with exit_tp_region. (A bias, if any, must be added
    once AFTER that sum, so it lives in the caller.)
    """

    def __init__(self, in_features: int, out_features: int, pc: ParallelContext,
                 seed: int | None, std: float = 0.02):
        super().__init__()
        w = _shard((out_features, in_features), 1, pc, seed, std)
        self.weight = mark(nn.Parameter(w), tp_sharded=True, tp_dim=1)

    def forward(self, x):
        return F.linear(x, self.weight)


class VocabParallelEmbedding(nn.Module):
    """Embedding table split by vocabulary rows between TP ranks.

    Each rank looks up only the token ids it owns and writes zeros for the
    others. exit_tp_region then sums the partial results. The table may be
    padded so its row count divides tp; padded rows are never looked up.
    """

    def __init__(self, vocab_size: int, hidden: int, pc: ParallelContext, seed: int | None,
                 padded_vocab: int, std: float = 0.02):
        super().__init__()
        self.pc = pc
        w = _shard((vocab_size, hidden), 0, pc, seed, std, padded_vocab)
        self.start = pc.tp_rank * w.shape[0]
        self.end = self.start + w.shape[0]
        self.weight = mark(nn.Parameter(w), tp_sharded=True, tp_dim=0)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        outside = (ids < self.start) | (ids >= self.end)
        local_ids = (ids - self.start).masked_fill(outside, 0)
        out = F.embedding(local_ids, self.weight)
        out = out * (~outside).unsqueeze(-1).to(out.dtype)
        return exit_tp_region(out, self.pc)


def gather_logits(local_logits: torch.Tensor, pc: ParallelContext) -> torch.Tensor:
    """Each TP rank has logits for its part of the vocabulary; gather them all.

    (A production system would use a vocab-parallel cross-entropy instead,
    to avoid building the full logits tensor.)
    """
    return comm.gather_from_group(local_logits, -1, pc.tp_group)
