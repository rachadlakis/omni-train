"""Post-training for OMNI-Train: SFT, preference optimisation (DPO, IPO, hinge,
SimPO, ORPO) and online RL (GRPO, PPO), selected with the `algorithm:` YAML block.

It reuses the normal solo / DDP / FSDP model setup (including LoRA/QLoRA) and
the normal training loop; only the data and the per-step loss change.

  config.py      parse + validate `algorithm:` (support matrix lives here)
  data.py        chat templates, answer-only masking, preference pairs
  losses.py      the loss formulas (pure tensor code)
  objectives.py  (model, batch) -> loss, reference log-prob precompute
  grpo.py        GRPO: sample, score, group advantages, clipped policy-gradient loss
  rewards.py     reward functions for GRPO (built-in and "module:function")
  ppo.py         PPO: value head, GAE, several clipped steps per rollout
  reward_model.py  scores from Hugging Face reward models for GRPO and PPO
"""

from .config import ALGORITHMS, PREFERENCE, parse_algorithm  # noqa: F401
