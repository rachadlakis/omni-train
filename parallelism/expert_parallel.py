"""Expert parallelism (EP): a Mixture-of-Experts layer with all-to-all routing.

    tokens -> router (top-k) -> sort by expert -> all-to-all -> local experts
           -> all-to-all back -> weighted sum per token

With ep_size ranks and E experts, every rank keeps E / ep_size whole experts.
No weight matrix is split (unlike TP); only tokens move.

Simplifications kept on purpose:
  * no load-balancing loss and no capacity limit (tokens are never dropped)
  * experts are plain SwiGLU MLPs
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from . import comm
from .process_groups import ParallelContext
from .utils import mark, seeded_normal


class Expert(nn.Module):
    def __init__(self, hidden: int, ffn: int, seed: int):
        super().__init__()
        self.gate = mark(nn.Parameter(seeded_normal((ffn, hidden), seed)), is_expert=True)
        self.up = mark(nn.Parameter(seeded_normal((ffn, hidden), seed + 1)), is_expert=True)
        self.down = mark(nn.Parameter(seeded_normal((hidden, ffn), seed + 2)), is_expert=True)

    def forward(self, x):
        return F.linear(F.silu(F.linear(x, self.gate)) * F.linear(x, self.up), self.down)


class MoELayer(nn.Module):
    def __init__(self, hidden: int, ffn: int, num_experts: int, top_k: int,
                 pc: ParallelContext, seed: int):
        super().__init__()
        if num_experts % pc.ep_size:
            raise ValueError(f"num_experts {num_experts} not divisible by ep {pc.ep_size}")
        self.pc, self.top_k, self.num_experts = pc, top_k, num_experts
        self.num_local = num_experts // pc.ep_size
        self.router = nn.Parameter(seeded_normal((num_experts, hidden), seed))
        first = pc.ep_rank * self.num_local  # global id of our first expert
        self.experts = nn.ModuleList(
            Expert(hidden, ffn, seed + 10 * (first + i + 1)) for i in range(self.num_local))

    def _run_experts(self, x: torch.Tensor, counts: list[int]) -> torch.Tensor:
        """x is sorted by local expert; counts[i] rows belong to expert i."""
        chunks = torch.split(x, counts, dim=0)
        # Call every expert, even with 0 tokens, so all grads exist (keeps DP sync simple).
        return torch.cat([expert(chunk) for expert, chunk in zip(self.experts, chunks)], dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(-1, shape[-1])                                  # [T, h]

        # 1. Route: pick top-k experts per token.
        probs = F.softmax(F.linear(x, self.router).float(), dim=-1)  # [T, E]
        top_w, top_e = probs.topk(self.top_k, dim=-1)                 # [T, k]
        top_w = (top_w / top_w.sum(-1, keepdim=True)).to(x.dtype)

        # 2. Sort the (token, expert) pairs by expert id.
        flat_e = top_e.reshape(-1)
        order = flat_e.argsort(stable=True)
        token_of = order // self.top_k
        x_sorted = x[token_of]                                        # [T*k, h]
        counts = torch.bincount(flat_e, minlength=self.num_experts)   # [E]

        if self.pc.ep_size == 1:
            y_sorted = self._run_experts(x_sorted, counts.tolist())
        else:
            y_sorted = self._dispatch_and_run(x_sorted, counts)

        # 3. Combine: weighted sum of the k expert outputs for each token.
        weights = top_w.reshape(-1)[order].unsqueeze(-1)
        out = torch.zeros_like(x).index_add(0, token_of, (y_sorted * weights).to(x.dtype))
        return out.view(shape)

    def _dispatch_and_run(self, x_sorted: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        ep, L, group = self.pc.ep_size, self.num_local, self.pc.ep_group

        # Tell every rank how many tokens it will get for each of its experts.
        recv_counts = torch.empty_like(counts)
        dist.all_to_all_single(recv_counts, counts, group=group)
        send_mat, recv_mat = counts.view(ep, L), recv_counts.view(ep, L)
        send_splits = send_mat.sum(1).tolist()
        recv_splits = recv_mat.sum(1).tolist()

        # Send tokens to the ranks that own their experts.
        x_recv = comm.all_to_all(x_sorted, send_splits, recv_splits, group)

        # x_recv is ordered [source rank][local expert]; reorder to [local expert][source].
        offsets = torch.cat([recv_mat.new_zeros(1), recv_mat.reshape(-1).cumsum(0)])[:-1].view(ep, L)
        perm = torch.cat([
            torch.arange(int(offsets[j, e]), int(offsets[j, e] + recv_mat[j, e]))
            for e in range(L) for j in range(ep)
        ] + [torch.zeros(0, dtype=torch.long)]).to(x_recv.device)
        y_local = self._run_experts(x_recv[perm], recv_mat.sum(0).tolist())
        y_recv = y_local[perm.argsort()]                              # undo the reorder

        # Send results back to where the tokens came from.
        return comm.all_to_all(y_recv, recv_splits, send_splits, group)
