"""Builds every process group needed for 5D parallelism (strategy: 5d).

Every rank is placed on a 4D grid with shape [dp, pp, cp, tp]. TP is the
fastest-changing axis, so TP ranks are neighbours (ideally inside one node,
connected by NVLink). DP is the slowest-changing axis.

Expert parallelism (EP) does not add a new axis: experts are spread across
DP ranks. The DP axis is split into chunks of size `ep`:

    dp_rank = chunk * ep + ep_rank

Example with world_size = 8, tp = 2, pp = 2, dp = 2 (cp = 1):

    rank:   0  1  2  3  4  5  6  7
    dp:     0  0  0  0  1  1  1  1
    pp:     0  0  1  1  0  0  1  1
    tp:     0  1  0  1  0  1  0  1
"""

from __future__ import annotations

import torch
import torch.distributed as dist


class ParallelContext:
    """Holds the rank coordinates and the process groups of the current rank."""

    def __init__(self, tp: int = 1, cp: int = 1, pp: int = 1, dp: int = 1,
                 ep: int = 1, sequence_parallel: bool = False):
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        if tp * cp * pp * dp != self.world_size:
            raise ValueError(
                f"tp*cp*pp*dp = {tp * cp * pp * dp} but world size is {self.world_size}")
        if dp % ep != 0:
            raise ValueError(f"ep ({ep}) must divide dp ({dp})")

        self.tp_size, self.cp_size, self.pp_size, self.dp_size, self.ep_size = tp, cp, pp, dp, ep
        # Sequence parallelism only makes sense together with TP.
        self.sequence_parallel = sequence_parallel and tp > 1

        grid = torch.arange(self.world_size).view(dp, pp, cp, tp)
        self.grid = grid
        coords = (grid == self.rank).nonzero()[0].tolist()
        self.dp_rank, self.pp_rank, self.cp_rank, self.tp_rank = coords
        self.ep_rank = self.dp_rank % ep

        # One group per axis. `rows` lists, for each group, the ranks in it.
        self.tp_group = self._build(grid.movedim(3, -1).reshape(-1, tp))
        self.cp_group = self._build(grid.movedim(2, -1).reshape(-1, cp))
        self.pp_group = self._build(grid.movedim(1, -1).reshape(-1, pp))
        self.dp_group = self._build(grid.movedim(0, -1).reshape(-1, dp))

        # CP ranks hold the same weights but different tokens, so gradients
        # are averaged over DP *and* CP together.
        self.dp_cp_group = self._build(grid.permute(1, 3, 0, 2).reshape(-1, dp * cp))

        # Expert groups. g5 has shape [dp/ep, ep, pp, cp, tp].
        g5 = grid.view(dp // ep, ep, pp, cp, tp)
        self.ep_group = self._build(g5.movedim(1, -1).reshape(-1, ep))
        # Ranks holding the SAME experts (same ep_rank) sync expert grads.
        self.expert_dp_cp_group = self._build(
            g5.permute(1, 2, 4, 0, 3).reshape(-1, (dp // ep) * cp))

        # Tied embeddings (HF models with tie_word_embeddings) split by PP: the
        # first stage holds the embedding, the last the LM head. Both copies must
        # get the SAME gradient, so the pair is summed over this 2-rank group.
        self.embed_tie_group = None
        if pp > 1:
            pairs = grid[:, [0, pp - 1]].permute(0, 2, 3, 1).reshape(-1, 2)
            self.embed_tie_group = self._build(pairs)

        # Neighbours (global ranks) for point-to-point communication.
        d, p, c, t = coords
        self.pp_next_rank = int(grid[d, (p + 1) % pp, c, t])
        self.pp_prev_rank = int(grid[d, (p - 1) % pp, c, t])
        self.cp_next_rank = int(grid[d, p, (c + 1) % cp, t])
        self.cp_prev_rank = int(grid[d, p, (c - 1) % cp, t])

    # ------------------------------------------------------------------ utils
    def _build(self, rows: torch.Tensor):
        """Create one group per row. All ranks must create all groups, in the same order."""
        mine = None
        for row in rows.tolist():
            group = dist.new_group(row)
            if self.rank in row:
                mine = group
        return mine

    @property
    def is_first_stage(self) -> bool:
        return self.pp_rank == 0

    @property
    def is_last_stage(self) -> bool:
        return self.pp_rank == self.pp_size - 1

    @property
    def is_logging_rank(self) -> bool:
        """One rank that holds the loss: last pipeline stage, first everything else."""
        return self.is_last_stage and self.dp_rank == 0 and self.cp_rank == 0 and self.tp_rank == 0

    def __repr__(self) -> str:
        return (f"rank {self.rank}: dp={self.dp_rank}/{self.dp_size} pp={self.pp_rank}/{self.pp_size} "
                f"cp={self.cp_rank}/{self.cp_size} tp={self.tp_rank}/{self.tp_size} "
                f"ep={self.ep_rank}/{self.ep_size} sp={self.sequence_parallel}")
