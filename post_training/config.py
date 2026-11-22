"""Parse and check the `algorithm:` YAML block.

    algorithm:
      type: dpo            # pretrain (default) | sft | dpo | simpo | orpo
      beta: 0.1            # dpo: KL strength; simpo: reward scale; orpo: lambda
      dpo_loss: sigmoid    # dpo only: sigmoid | ipo | hinge
      label_smoothing: 0.0 # dpo sigmoid only (conservative DPO), in [0, 0.5)
      simpo_gamma: 0.5     # simpo only: target reward margin
      max_prompt_length: 256   # preference data: prompts longer than this are
                               # cut from the left (default: max_length // 2)
      completion_only: true    # sft: loss on the answer tokens only

Support matrix (anything else is a config error):
  * model_type: llm only (decoder-only causal LMs).
  * strategy: solo, ddp, fsdp. Not 5d yet (its own model has no LoRA and no
    reference-model path).
  * PEFT/LoRA and quantization work as they do for normal fine-tuning
    (quantization still needs solo/ddp).
"""

from __future__ import annotations

ALGORITHMS = ("pretrain", "sft", "dpo", "simpo", "orpo")
PREFERENCE = ("dpo", "simpo", "orpo")
PLANNED = ("grpo", "ppo")
DPO_LOSSES = ("sigmoid", "ipo", "hinge")
DEFAULT_BETA = {"dpo": 0.1, "simpo": 2.0, "orpo": 0.1}
KNOWN_KEYS = {"type", "beta", "dpo_loss", "label_smoothing", "simpo_gamma",
              "max_prompt_length", "completion_only"}


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

    args.algo_beta = float(block.get("beta", DEFAULT_BETA.get(algo, 0.1)))
    args.algo_dpo_loss = str(block.get("dpo_loss", "sigmoid")).lower()
    args.algo_label_smoothing = float(block.get("label_smoothing", 0.0))
    args.algo_simpo_gamma = float(block.get("simpo_gamma", 0.5))
    mpl = block.get("max_prompt_length")
    args.algo_max_prompt_length = int(mpl) if mpl is not None else max(1, args.max_length // 2)
    args.algo_completion_only = _bool(block.get("completion_only", True))

    if algo == "pretrain":
        return

    if args.model_type != "llm":
        raise ValueError(f"algorithm.type={algo} needs model_type: llm (got {args.model_type})")
    if args.strategy not in {"solo", "ddp", "fsdp"}:
        raise ValueError(f"algorithm.type={algo} supports strategy solo, ddp or fsdp (got {args.strategy})")
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
