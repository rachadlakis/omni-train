"""Post-training for OMNI-Train: SFT, preference optimisation (DPO, IPO, hinge,
SimPO, ORPO) and online RL (GRPO), selected with the `algorithm:` YAML block.

It reuses the normal solo / DDP / FSDP model setup (including LoRA/QLoRA) and
the normal training loop; only the data and the per-step loss change.

  config.py      parse + validate `algorithm:` (support matrix lives here)
  data.py        chat templates, answer-only masking, preference pairs
  losses.py      the loss formulas (pure tensor code)
  objectives.py  (model, batch) -> loss, reference log-prob precompute
  grpo.py        GRPO: sample, score, group advantages, clipped policy-gradient loss
  rewards.py     reward functions for GRPO (built-in and "module:function")

PPO (with a reward model and a value head) is planned next.
"""

from .config import ALGORITHMS, PREFERENCE, parse_algorithm  # noqa: F401
