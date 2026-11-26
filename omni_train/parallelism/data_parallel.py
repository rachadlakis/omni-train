"""Data parallelism (DP) with bucketing and compute/communication overlap.

Every DP rank has a full copy of its model part and sees different data.
After the backward pass the gradients are averaged, so all copies stay equal.

Three classic tricks from the playbook:
  1. Overlap  - a hook fires as soon as a parameter's gradient is ready.
  2. Buckets  - gradients are grouped, so we send a few big messages.
  3. no_sync  - with gradient accumulation we only sync on the LAST micro-batch
                (set `require_sync = False` for the other micro-batches).

Which group is used for which parameter:
  * normal weights -> DP x CP group (CP ranks also hold the same weights)
  * expert weights -> ranks that hold the same experts (see process_groups.py)
  * with SP, weights not split by TP are also summed over the TP group
  * tied embedding split by PP: first-stage embedding + last-stage LM head
    are summed over the 2-rank embed_tie_group, so both copies stay equal
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn

from .process_groups import ParallelContext


class _Bucket:
    def __init__(self, params: list[nn.Parameter], group):
        self.params, self.group = params, group
        self.ready = 0
        self.handle = None
        self.buffer = None


class _BucketList:
    """Buckets that share one process group. Launched strictly in order, so
    every rank issues the collectives in the same sequence."""

    def __init__(self, params: list[nn.Parameter], group, bucket_bytes: int, divisor: int):
        self.group = group
        self.divisor = divisor  # summed grads are divided by this to get the average
        self.buckets: list[_Bucket] = []
        current, current_bytes = [], 0
        for p in params:  # params arrive roughly in backward order
            current.append(p)
            current_bytes += p.numel() * p.element_size()
            if current_bytes >= bucket_bytes:
                self.buckets.append(_Bucket(current, group))
                current, current_bytes = [], 0
        if current:
            self.buckets.append(_Bucket(current, group))
        self.bucket_of = {id(p): b for b in self.buckets for p in b.params}
        self.next_to_launch = 0

    def _launch(self, bucket: _Bucket):
        grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in bucket.params]
        bucket.buffer = torch.cat([g.reshape(-1) for g in grads])
        bucket.handle = dist.all_reduce(bucket.buffer, group=self.group, async_op=True)

    def mark_ready(self, p: nn.Parameter):
        bucket = self.bucket_of[id(p)]
        bucket.ready += 1
        # Launch every bucket that is ready, in order.
        while (self.next_to_launch < len(self.buckets)
               and self.buckets[self.next_to_launch].ready == len(self.buckets[self.next_to_launch].params)):
            self._launch(self.buckets[self.next_to_launch])
            self.next_to_launch += 1

    def finish(self):
        while self.next_to_launch < len(self.buckets):  # anything not launched yet
            self._launch(self.buckets[self.next_to_launch])
            self.next_to_launch += 1
        for bucket in self.buckets:
            bucket.handle.wait()
            bucket.buffer /= self.divisor
            offset = 0
            for p in bucket.params:
                n = p.numel()
                p.grad = bucket.buffer[offset:offset + n].view_as(p).clone()
                offset += n
            bucket.ready, bucket.handle, bucket.buffer = 0, None, None
        self.next_to_launch = 0


class DataParallel(nn.Module):
    def __init__(self, module: nn.Module, pc: ParallelContext, bucket_mb: float = 25.0):
        super().__init__()
        self.module, self.pc = module, pc
        self.require_sync = True
        params = [p for p in module.parameters() if p.requires_grad][::-1]  # ~backward order
        bucket_bytes = int(bucket_mb * 2**20)
        dense = [p for p in params if not getattr(p, "is_expert", False)]
        expert = [p for p in params if getattr(p, "is_expert", False)]
        # Every weight's gradient is an average over ALL dp*cp replicas of the
        # loss. An expert already received (via the all-to-all backward) the
        # summed gradient of the ep ranks it served, so summing over its own
        # group and dividing by the group size would be ep times too big.
        n_replicas = pc.dp_size * pc.cp_size
        self.lists = []
        if dist.get_world_size(pc.dp_cp_group) > 1 and dense:
            self.lists.append(_BucketList(dense, pc.dp_cp_group, bucket_bytes, n_replicas))
        expert_sync = dist.get_world_size(pc.expert_dp_cp_group) > 1
        if expert_sync and expert:
            self.lists.append(_BucketList(expert, pc.expert_dp_cp_group, bucket_bytes, n_replicas))
        # No other rank holds these experts (dp == ep, cp == 1): just rescale.
        self.lone_experts = expert if (not expert_sync and pc.ep_size > 1) else []
        owner = {id(q): lst for lst in self.lists for b in lst.buckets for q in b.params}
        for p in params:
            if id(p) in owner:
                p.register_post_accumulate_grad_hook(self._make_hook(owner[id(p)]))
        self.sp_params = [p for p in params if getattr(p, "sp_replicated", False)]
        self.tied_params = [p for p in params if getattr(p, "tied_embedding", False)]

    def _make_hook(self, bucket_list: _BucketList):
        def hook(p):
            if self.require_sync:
                bucket_list.mark_ready(p)
        return hook

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_grad_sync(self):
        """Call once after the last backward pass of the step."""
        for lst in self.lists:
            lst.finish()
        for p in self.lone_experts:
            if p.grad is not None:
                p.grad /= self.pc.ep_size
        # SP: weights that are not TP-split saw only 1/tp of the tokens -> sum over TP.
        if self.sp_params:
            for p in self.sp_params:
                if p.grad is not None:
                    dist.all_reduce(p.grad, group=self.pc.tp_group)
        # Tied embedding across PP stages: each copy saw half the uses -> sum.
        if self.tied_params and self.pc.embed_tie_group is not None:
            for p in self.tied_params:
                dist.all_reduce(p.grad, group=self.pc.embed_tie_group)
