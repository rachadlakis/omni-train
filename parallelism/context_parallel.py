"""Context parallelism (CP) with ring attention.

The sequence is cut into cp_size chunks, one per CP rank, for the WHOLE model.
Most layers (norms, MLPs) work on each token alone, so they need nothing.
Attention needs every key/value, so the K/V chunks travel around a ring:

    step 0: rank r uses its own K/V chunk          (causal mask inside the chunk)
    step s: rank r uses the chunk from rank r - s   (full attention if it is earlier,
                                                     skipped if it is later: causal)

Partial results are merged with the "online softmax" trick, the same idea
used in Flash Attention: keep a running max (m), running sum (l) and output.

The ring_shift op is an autograd function, so the backward pass sends the
K/V gradients back around the ring automatically.

Note: with a causal mask, early ranks do less work than late ones. The
"zig-zag" layout (each rank holds one early and one late chunk) fixes this;
it is left as an exercise to keep the code short.
"""

from __future__ import annotations

import math

import torch

from . import comm
from .process_groups import ParallelContext


def ring_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   pc: ParallelContext) -> torch.Tensor:
    """Causal attention over a sequence split between CP ranks.

    q: [batch, heads, local_seq, head_dim]; k, v: [batch, kv_heads, local_seq, head_dim]
    for this rank's chunk (grouped-query attention when kv_heads < heads: the small
    K/V travels around the ring and is expanded per step).
    Returns: [batch, heads, local_seq, head_dim].
    """
    group = q.shape[1] // k.shape[1]
    n, rank = pc.cp_size, pc.cp_rank
    in_dtype = q.dtype
    q, k, v = q.float(), k.float(), v.float()
    scale = 1.0 / math.sqrt(q.shape[-1])
    s_local = q.shape[-2]

    out = torch.zeros_like(q)
    row_max = torch.full(q.shape[:-1] + (1,), float("-inf"), device=q.device)
    row_sum = torch.zeros(q.shape[:-1] + (1,), device=q.device)
    causal = torch.ones(s_local, s_local, dtype=torch.bool, device=q.device).triu(1)

    kv = torch.stack([k, v])
    for step in range(n):
        source = (rank - step) % n  # which chunk of the sequence we hold now
        if source <= rank:
            k_blk, v_blk = kv[0], kv[1]
            if group > 1:
                k_blk = k_blk.repeat_interleave(group, dim=1)
                v_blk = v_blk.repeat_interleave(group, dim=1)
            scores = (q @ k_blk.transpose(-2, -1)) * scale
            if source == rank:
                scores = scores.masked_fill(causal, float("-inf"))
            new_max = torch.maximum(row_max, scores.amax(-1, keepdim=True))
            probs = torch.exp(scores - new_max)
            correction = torch.exp(row_max - new_max)
            row_sum = row_sum * correction + probs.sum(-1, keepdim=True)
            out = out * correction + probs @ v_blk
            row_max = new_max
        else:
            # Causal: this chunk is in our future, so we skip the math. But we keep a
            # zero-weight link to it, so autograd still runs its ring_shift backward.
            # Otherwise this rank would skip a send/recv that its neighbour expects.
            out = out + 0.0 * kv.sum()
        if step < n - 1:
            # Pass our current K/V to the next rank, get the previous rank's.
            kv = comm.ring_shift(kv, pc.cp_next_rank, pc.cp_prev_rank, pc.cp_group)

    return (out / row_sum).to(in_dtype)
