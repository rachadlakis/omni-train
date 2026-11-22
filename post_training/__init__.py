"""Post-training for OMNI-Train: SFT and preference optimisation (DPO, IPO, hinge,
SimPO, ORPO), selected with the `algorithm:` YAML block.

It reuses the normal solo / DDP / FSDP model setup (including LoRA/QLoRA) and
the normal training loop; only the data and the per-step loss change.

  config.py      parse + validate `algorithm:` (support matrix lives here)
  data.py        chat templates, answer-only masking, preference pairs
  losses.py      the loss formulas (pure tensor code)
  objectives.py  (model, batch) -> loss, reference log-prob precompute

GRPO and PPO (online RL with generation) are planned next.
"""

from .config import ALGORITHMS, PREFERENCE, parse_algorithm  # noqa: F401
