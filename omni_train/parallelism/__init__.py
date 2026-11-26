"""5D parallelism for OMNI-Train (strategy: 5d). Plain torch.distributed, no extra deps.

The five dimensions:
    DP  - data parallelism (+ ZeRO-1 optimizer sharding)
    TP  - tensor parallelism (+ sequence parallelism, SP)
    CP  - context parallelism (ring attention)
    PP  - pipeline parallelism (AFAB and 1F1B schedules)
    EP  - expert parallelism (Mixture-of-Experts with all-to-all)

Entry point: parallelism.trainer.train_5d(args), called by train.py.
See docs/5D_PARALLELISM.md.
"""

from .process_groups import ParallelContext

__all__ = ["ParallelContext"]
