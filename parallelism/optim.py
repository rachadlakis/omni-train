"""Optimizer helpers for strategy: 5d: global gradient clipping and LR setting."""

from __future__ import annotations

import torch
import torch.distributed as dist

from .process_groups import ParallelContext


@torch.no_grad()
def global_grad_norm(params, pc: ParallelContext) -> torch.Tensor:
    """L2 norm of the FULL model's gradient, counting every element exactly once.

    After DataParallel.finish_grad_sync:
      * TP-sharded weights: each TP rank holds a different slice -> sum over TP
      * other weights: identical on every TP rank (SP grads were TP-summed) -> once
      * expert weights: different experts per EP rank -> sum over EP
      * pipeline stages hold different layers -> sum over PP
      * DP / CP replicas are identical -> once
      * a tied LM head on the last stage duplicates the embedding -> skipped
    """
    device = next(iter(params)).device if params else torch.device("cpu")
    sharded = torch.zeros((), device=device, dtype=torch.float32)
    replicated, expert = sharded.clone(), sharded.clone()
    for p in params:
        if p.grad is None or getattr(p, "tied_copy", False):
            continue
        sq = p.grad.detach().float().pow(2).sum()
        if getattr(p, "tp_sharded", False):
            sharded += sq
        elif getattr(p, "is_expert", False):
            expert += sq
        else:
            replicated += sq
    if pc.tp_size > 1:
        dist.all_reduce(sharded, group=pc.tp_group)
    if pc.ep_size > 1:
        dist.all_reduce(expert, group=pc.ep_group)
    total = sharded + replicated + expert
    if pc.pp_size > 1:
        dist.all_reduce(total, group=pc.pp_group)
    return total.sqrt()


@torch.no_grad()
def clip_grad_norm(params, max_norm: float, pc: ParallelContext) -> float:
    """Clip like torch.nn.utils.clip_grad_norm_, but with the global 5D norm."""
    params = [p for p in params if p.grad is not None]
    norm = global_grad_norm(params, pc)
    if max_norm > 0:
        coef = torch.clamp(max_norm / (norm + 1e-6), max=1.0)
        for p in params:
            p.grad.mul_(coef)
    return norm.item()


def set_lr(optimizer, lr: float) -> None:
    if hasattr(optimizer, "set_lr"):
        optimizer.set_lr(lr)
    else:
        for group in optimizer.param_groups:
            group["lr"] = lr
