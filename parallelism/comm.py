"""Communication primitives, wrapped as autograd functions.

Each function says what happens in the forward pass and what happens to the
gradient in the backward pass. The pairs are "conjugates":

    copy_to_group        forward: identity        backward: all-reduce
    reduce_from_group    forward: all-reduce      backward: identity
    gather_from_group    forward: all-gather      backward: split
    reduce_scatter_to    forward: reduce-scatter  backward: all-gather
    gather_sequence      forward: all-gather      backward: reduce-scatter
    all_to_all           forward: all-to-all      backward: all-to-all (reverse)
    ring_shift           forward: send to next    backward: send to previous
"""

from __future__ import annotations

import torch
import torch.distributed as dist


# --------------------------------------------------------------- raw helpers
def _size(group) -> int:
    return dist.get_world_size(group) if group is not None else 1


def _all_reduce(x: torch.Tensor, group) -> torch.Tensor:
    if _size(group) == 1:
        return x
    x = x.contiguous().clone()
    dist.all_reduce(x, group=group)
    return x


def _all_gather(x: torch.Tensor, dim: int, group) -> torch.Tensor:
    n = _size(group)
    if n == 1:
        return x
    x = x.contiguous()
    parts = [torch.empty_like(x) for _ in range(n)]
    dist.all_gather(parts, x, group=group)
    return torch.cat(parts, dim=dim)


def _split(x: torch.Tensor, dim: int, group) -> torch.Tensor:
    n = _size(group)
    if n == 1:
        return x
    return x.chunk(n, dim=dim)[dist.get_rank(group)].contiguous()


def _reduce_scatter(x: torch.Tensor, dim: int, group) -> torch.Tensor:
    n = _size(group)
    if n == 1:
        return x
    rank = dist.get_rank(group)
    if dist.get_backend(group) == "nccl":
        chunks = [c.contiguous() for c in x.chunk(n, dim=dim)]
        out = torch.empty_like(chunks[rank])
        dist.reduce_scatter(out, chunks, group=group)
        return out
    # Gloo has no reduce-scatter: all-reduce, then keep our chunk (same result).
    return _all_reduce(x, group).chunk(n, dim=dim)[rank].contiguous()


# ------------------------------------------------------- autograd functions
class _CopyToGroup(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return x

    @staticmethod
    def backward(ctx, grad):
        return _all_reduce(grad, ctx.group), None


class _ReduceFromGroup(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        return _all_reduce(x, group)

    @staticmethod
    def backward(ctx, grad):
        return grad, None


class _GatherFromGroup(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim, group):
        ctx.dim, ctx.group = dim, group
        return _all_gather(x, dim, group)

    @staticmethod
    def backward(ctx, grad):
        return _split(grad, ctx.dim, ctx.group), None, None


class _ReduceScatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim, group):
        ctx.dim, ctx.group = dim, group
        return _reduce_scatter(x, dim, group)

    @staticmethod
    def backward(ctx, grad):
        return _all_gather(grad, ctx.dim, ctx.group), None, None


class _GatherSequence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim, group):
        ctx.dim, ctx.group = dim, group
        return _all_gather(x, dim, group)

    @staticmethod
    def backward(ctx, grad):
        return _reduce_scatter(grad, ctx.dim, ctx.group), None, None


class _AllToAll(torch.autograd.Function):
    """Rows of `x` are sent to other ranks: send_splits[i] rows go to rank i."""

    @staticmethod
    def forward(ctx, x, send_splits, recv_splits, group):
        ctx.send_splits, ctx.recv_splits, ctx.group = send_splits, recv_splits, group
        out = x.new_empty((sum(recv_splits),) + tuple(x.shape[1:]))
        dist.all_to_all_single(out, x.contiguous(), recv_splits, send_splits, group=group)
        return out

    @staticmethod
    def backward(ctx, grad):
        out = grad.new_empty((sum(ctx.send_splits),) + tuple(grad.shape[1:]))
        dist.all_to_all_single(out, grad.contiguous(), ctx.send_splits, ctx.recv_splits,
                               group=ctx.group)
        return out, None, None, None


def _p2p_shift(x: torch.Tensor, send_to: int, recv_from: int, group) -> torch.Tensor:
    x = x.contiguous()
    out = torch.empty_like(x)
    ops = [dist.P2POp(dist.isend, x, send_to, group),
           dist.P2POp(dist.irecv, out, recv_from, group)]
    for req in dist.batch_isend_irecv(ops):
        req.wait()
    return out


class _RingShift(torch.autograd.Function):
    """Send x to the next rank in a ring and receive from the previous one."""

    @staticmethod
    def forward(ctx, x, next_rank, prev_rank, group):
        ctx.next_rank, ctx.prev_rank, ctx.group = next_rank, prev_rank, group
        return _p2p_shift(x, next_rank, prev_rank, group)

    @staticmethod
    def backward(ctx, grad):
        # The gradient travels the ring in the opposite direction.
        return _p2p_shift(grad, ctx.prev_rank, ctx.next_rank, ctx.group), None, None, None


# ------------------------------------------------------------ public API
def copy_to_group(x, group):
    return _CopyToGroup.apply(x, group) if _size(group) > 1 else x


def reduce_from_group(x, group):
    return _ReduceFromGroup.apply(x, group) if _size(group) > 1 else x


def gather_from_group(x, dim, group):
    return _GatherFromGroup.apply(x, dim, group) if _size(group) > 1 else x


def reduce_scatter_to_group(x, dim, group):
    return _ReduceScatter.apply(x, dim, group) if _size(group) > 1 else x


def gather_sequence(x, dim, group):
    return _GatherSequence.apply(x, dim, group) if _size(group) > 1 else x


def all_to_all(x, send_splits, recv_splits, group):
    return _AllToAll.apply(x, send_splits, recv_splits, group)


def ring_shift(x, next_rank, prev_rank, group):
    return _RingShift.apply(x, next_rank, prev_rank, group)


def send_recv(sends, recvs, group=None):
    """Point-to-point helper for pipeline parallelism (no autograd).

    sends: list of (tensor, global_rank); recvs: list of (tensor, global_rank).
    """
    ops = [dist.P2POp(dist.isend, t.contiguous(), r, group) for t, r in sends]
    ops += [dist.P2POp(dist.irecv, t, r, group) for t, r in recvs]
    if ops:
        for req in dist.batch_isend_irecv(ops):
            req.wait()
