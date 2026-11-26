"""ZeRO-1: split the optimizer states between data-parallel ranks.

Adam keeps two extra FP32 tensors per weight (momentum and variance). With
plain DP every rank stores all of them. With ZeRO-1 each rank:

    1. gets the averaged gradients (from DataParallel),
    2. runs Adam only for the weights it OWNS,
    3. broadcasts its updated weights to the other ranks.

So optimizer memory per GPU drops by about the DP size.

This version gives each rank whole parameters (balanced by size), which keeps
the code short. Real systems split one flat buffer and use reduce-scatter +
all-gather instead of all-reduce + broadcast; the memory saving is the same.

Expert weights are already different on each EP rank; they keep a normal
local optimizer here.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from .process_groups import ParallelContext


class ZeRO1AdamW:
    def __init__(self, params, pc: ParallelContext, **adam_kwargs):
        params = [p for p in params if p.requires_grad]
        self.group = pc.dp_cp_group
        self.world = dist.get_world_size(self.group)
        self.ranks = dist.get_process_group_ranks(self.group)
        my_rank = dist.get_rank(self.group)

        self.dense = [p for p in params if not getattr(p, "is_expert", False)]
        expert = [p for p in params if getattr(p, "is_expert", False)]

        # Greedy balance: biggest tensors first, each to the least-loaded rank.
        load = [0] * self.world
        self.owner = {}
        for idx in sorted(range(len(self.dense)), key=lambda i: (-self.dense[i].numel(), i)):
            r = min(range(self.world), key=lambda j: load[j])
            self.owner[idx] = r
            load[r] += self.dense[idx].numel()

        mine = [p for i, p in enumerate(self.dense) if self.owner[i] == my_rank] + expert
        self.all_params = params
        self.inner = torch.optim.AdamW(mine, **adam_kwargs) if mine else None

    @torch.no_grad()
    def step(self):
        if self.inner is not None:
            self.inner.step()
        if self.world > 1:
            for i, p in enumerate(self.dense):
                dist.broadcast(p.data, src=self.ranks[self.owner[i]], group=self.group)

    def zero_grad(self):
        for p in self.all_params:
            p.grad = None

    def set_lr(self, lr: float):
        if self.inner is not None:
            for group in self.inner.param_groups:
                group["lr"] = lr

    def state_dict(self):
        return self.inner.state_dict() if self.inner is not None else {}

    def load_state_dict(self, state):
        if self.inner is not None:
            self.inner.load_state_dict(state)
