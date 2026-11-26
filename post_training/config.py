"""Parse and check the `algorithm:` YAML block.

    algorithm:
      type: dpo            # pretrain (default) | sft | dpo | simpo | orpo | grpo
      beta: 0.1            # dpo: KL strength; simpo: reward scale; orpo: lambda;
                           # grpo: KL penalty against the start model (default 0)
      dpo_loss: sigmoid    # dpo only: sigmoid | ipo | hinge
      label_smoothing: 0.0 # dpo sigmoid only (conservative DPO), in [0, 0.5)
      simpo_gamma: 0.5     # simpo only: target reward margin
      max_prompt_length: 256   # preference / grpo: prompts longer than this are
                               # cut from the left (default: max_length // 2)
      completion_only: true    # sft: loss on the answer tokens only

    GRPO only:
      num_generations: 4         # completions sampled per prompt (the group), >= 2
      max_completion_length: 256 # tokens generated per completion
                                 # (default: max_length - max_prompt_length)
      temperature: 1.0           # sampling temperature (also divides the logits in the loss)
      top_p: 1.0
      top_k: 0                   # 0 = off
      epsilon: 0.2               # PPO clip range, lower side
      epsilon_high: 0.2          # upper side (DAPO "clip-higher": e.g. 0.28)
      loss_type: grpo            # grpo | dapo | dr_grpo (normalisation, see losses.grpo_loss)
      scale_rewards: true        # divide advantages by the group's reward std (false = Dr. GRPO)
      rewards: [accuracy]        # names, {name: weight}, or "module:function" (see rewards.py)
      system_prompt: null        # optional system message added to every prompt

Support matrix (anything else is a config error):
  * model_type: llm only (decoder-only causal LMs).
  * strategy: solo, ddp, fsdp. Not 5d yet (its own model has no LoRA and no
    reference-model path).
  * PEFT/LoRA and quantization work as they do for normal fine-tuning
    (quantization still needs solo/ddp).
  * grpo with beta > 0 needs LoRA: the KL reference is the same model with its
    adapters disabled, so no second model is kept.
"""

from __future__ import annotations

ALGORITHMS = ("pretrain", "sft", "dpo", "simpo", "orpo", "grpo")
PREFERENCE = ("dpo", "simpo", "orpo")
PLANNED = ("ppo",)
DPO_LOSSES = ("sigmoid", "ipo", "hinge")
DEFAULT_BETA = {"dpo": 0.1, "simpo": 2.0, "orpo": 0.1, "grpo": 0.0}
GRPO_KEYS = {"num_generations", "max_completion_length", "temperature", "top_p", "top_k",
             "epsilon", "epsilon_high", "loss_type", "scale_rewards", "rewards", "system_prompt"}
KNOWN_KEYS = {"type", "beta", "dpo_loss", "label_smoothing", "simpo_gamma",
              "max_prompt_length", "completion_only"} | GRPO_KEYS


def _bool(x) -> bool:
    if isinstance(x, str):
        return x.lower() in {"true", "1", "yes", "y", "on"}
    return bool(x)


def parse_algorithm(cfg: dict, args) -> None:
    block = cfg.get("algorithm") or {}
    if not isinstance(block, dict):
        raise ValueError("algorithm must be a mapping, e.g. `algorithm: {type: sft}`")
    unknown = set(block) - KNOWN_KEYS
    if unknown:
        raise ValueError(f"Unknown algorithm keys: {sorted(unknown)}. Known: {sorted(KNOWN_KEYS)}")

    algo = str(block.get("type", "pretrain")).lower()
    if algo in PLANNED:
        raise ValueError(f"algorithm.type={algo} is planned but not implemented yet. "
                         f"Available: {', '.join(ALGORITHMS)}")
    if algo not in ALGORITHMS:
        raise ValueError(f"Unsupported algorithm.type={algo}. Use one of: {', '.join(ALGORITHMS)}")
    args.algorithm = algo
    grpo_only = set(block) & GRPO_KEYS
    if grpo_only and algo != "grpo":
        raise ValueError(f"algorithm keys {sorted(grpo_only)} only apply to type: grpo")

    args.algo_beta = float(block.get("beta", DEFAULT_BETA.get(algo, 0.1)))
    args.algo_dpo_loss = str(block.get("dpo_loss", "sigmoid")).lower()
    args.algo_label_smoothing = float(block.get("label_smoothing", 0.0))
    args.algo_simpo_gamma = float(block.get("simpo_gamma", 0.5))
    mpl = block.get("max_prompt_length")
    args.algo_max_prompt_length = int(mpl) if mpl is not None else max(1, args.max_length // 2)
    args.algo_completion_only = _bool(block.get("completion_only", True))

    # GRPO
    args.algo_num_generations = int(block.get("num_generations", 4))
    mcl = block.get("max_completion_length")
    args.algo_max_completion_length = (int(mcl) if mcl is not None
                                       else max(1, args.max_length - args.algo_max_prompt_length))
    args.algo_temperature = float(block.get("temperature", 1.0))
    args.algo_top_p = float(block.get("top_p", 1.0))
    args.algo_top_k = int(block.get("top_k", 0) or 0)
    args.algo_epsilon = float(block.get("epsilon", 0.2))
    args.algo_epsilon_high = float(block.get("epsilon_high", args.algo_epsilon))
    args.algo_loss_type = str(block.get("loss_type", "grpo")).lower()
    args.algo_scale_rewards = _bool(block.get("scale_rewards", True))
    args.algo_rewards = block.get("rewards", ["accuracy"])
    sp = block.get("system_prompt")
    args.algo_system_prompt = str(sp) if sp else None

    if algo == "pretrain":
        return

    if args.model_type != "llm":
        raise ValueError(f"algorithm.type={algo} needs model_type: llm (got {args.model_type})")
    if args.strategy not in {"auto", "solo", "ddp", "fsdp"}:
        raise ValueError(f"algorithm.type={algo} supports strategy solo, ddp or fsdp (got {args.strategy})")
    if algo == "grpo":
        _check_grpo(args)
        return
    if args.algo_beta <= 0:
        raise ValueError("algorithm.beta must be > 0")
    if args.algo_dpo_loss not in DPO_LOSSES:
        raise ValueError(f"Unsupported algorithm.dpo_loss={args.algo_dpo_loss}. Use one of: {', '.join(DPO_LOSSES)}")
    if not 0.0 <= args.algo_label_smoothing < 0.5:
        raise ValueError("algorithm.label_smoothing must be in [0, 0.5)")
    if args.algo_label_smoothing and not (algo == "dpo" and args.algo_dpo_loss == "sigmoid"):
        raise ValueError("algorithm.label_smoothing only applies to dpo with dpo_loss: sigmoid")
    if algo in PREFERENCE and not 1 <= args.algo_max_prompt_length < args.max_length:
        raise ValueError(f"algorithm.max_prompt_length ({args.algo_max_prompt_length}) must be "
                         f">= 1 and < training.max_length ({args.max_length})")


def _check_grpo(args) -> None:
    from .losses import GRPO_LOSS_TYPES
    from .rewards import parse_reward_spec

    if args.algo_num_generations < 2:
        raise ValueError("algorithm.num_generations must be >= 2: advantages compare completions of one prompt")
    if args.algo_beta < 0:
        raise ValueError("algorithm.beta must be >= 0 for grpo")
    if args.algo_beta > 0 and not args.peft_enabled:
        raise ValueError("grpo with beta > 0 needs LoRA (peft.enabled: true): the KL reference is the model "
                         "with its adapters disabled. For a full fine-tune use beta: 0 (as DAPO / Dr. GRPO do).")
    if args.algo_temperature <= 0:
        raise ValueError("algorithm.temperature must be > 0 (sampling is required for GRPO)")
    if not 0 < args.algo_top_p <= 1:
        raise ValueError("algorithm.top_p must be in (0, 1]")
    if args.algo_top_k < 0:
        raise ValueError("algorithm.top_k must be >= 0 (0 = off)")
    if args.algo_epsilon <= 0 or args.algo_epsilon_high <= 0:
        raise ValueError("algorithm.epsilon and epsilon_high must be > 0")
    if args.algo_loss_type not in GRPO_LOSS_TYPES:
        raise ValueError(f"Unsupported algorithm.loss_type={args.algo_loss_type}. Use one of: {', '.join(GRPO_LOSS_TYPES)}")
    if args.algo_max_prompt_length < 1 or args.algo_max_completion_length < 1:
        raise ValueError("algorithm.max_prompt_length and max_completion_length must be >= 1")
    total = args.algo_max_prompt_length + args.algo_max_completion_length
    if total > args.max_length:
        raise ValueError(f"max_prompt_length + max_completion_length ({total}) must be <= training.max_length "
                         f"({args.max_length})")
    parse_reward_spec(args.algo_rewards)
