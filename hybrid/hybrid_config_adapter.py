"""
Configuration adapter for 3D parallelism.

Extends the existing Args dataclass with parallelism-specific settings
without modifying the original utils.py.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal
import os
import sys

# Add parent directory to path to import existing modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_config, Args


@dataclass
class TensorParallelConfig:
    """Tensor parallelism configuration."""

    style: Literal["colwise_rowwise", "megatron"] = "colwise_rowwise"
    sequence_parallel: bool = False
    plan: Optional[dict] = None  # Custom parallelization plan


@dataclass
class PipelineParallelConfig:
    """Pipeline parallelism configuration."""

    schedule: Literal["gpipe", "1f1b", "interleaved"] = "1f1b"
    num_microbatches: int = 4
    chunks: str = "auto"
    layer_assignment: Optional[list[int]] = None


@dataclass
class ParallelismConfig:
    """3D parallelism configuration."""

    enabled: bool = False

    # Mesh dimensions
    data_parallel_size: Optional[int] = None
    tensor_parallel_size: Optional[int] = None
    pipeline_parallel_size: Optional[int] = None

    # Sub-configs
    tensor_parallel: TensorParallelConfig = field(default_factory=TensorParallelConfig)
    pipeline_parallel: PipelineParallelConfig = field(default_factory=PipelineParallelConfig)

    # Mesh settings
    mesh_dim_names: tuple[str, ...] = ("dp", "tp", "pp")
    backend: str = "nccl"


@dataclass
class HybridArgs(Args):
    """
    Extended Args with 3D parallelism support.

    Inherits all fields from Args and adds parallelism configuration.
    """

    parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)

    # Resolved mesh (filled in by resolve_mesh_dimensions)
    resolved_dp_size: int = 1
    resolved_tp_size: int = 1
    resolved_pp_size: int = 1


def resolve_mesh_dimensions(
    num_gpus: int,
    dp: Optional[int] = None,
    tp: Optional[int] = None,
    pp: Optional[int] = None,
) -> tuple[int, int, int]:
    """
    Auto-resolve mesh dimensions when not fully specified.

    Priority: TP > PP > DP (TP needs fast interconnect, assign first)

    Args:
        num_gpus: Total number of GPUs
        dp: Data parallel size (optional)
        tp: Tensor parallel size (optional)
        pp: Pipeline parallel size (optional)

    Returns:
        Tuple of (dp_size, tp_size, pp_size)

    Raises:
        ValueError: If dimensions don't match GPU count
    """
    # Default lookup table for common GPU counts
    DEFAULT_MESH = {
        1: (1, 1, 1),
        2: (2, 1, 1),
        4: (2, 2, 1),
        8: (2, 2, 2),
        16: (4, 2, 2),
        32: (4, 4, 2),
        64: (8, 4, 2),
    }

    # If all specified, validate and return
    if dp is not None and tp is not None and pp is not None:
        if dp * tp * pp != num_gpus:
            raise ValueError(
                f"Mesh dimensions dp={dp} × tp={tp} × pp={pp} = {dp * tp * pp} "
                f"doesn't match num_gpus={num_gpus}"
            )
        return (dp, tp, pp)

    # If none specified, use default
    if dp is None and tp is None and pp is None:
        if num_gpus in DEFAULT_MESH:
            return DEFAULT_MESH[num_gpus]
        # For unlisted GPU counts, default to pure DP
        return (num_gpus, 1, 1)

    # Partial specification - resolve missing dimensions
    specified = {"dp": dp, "tp": tp, "pp": pp}
    unspecified = [k for k, v in specified.items() if v is None]

    # Calculate product of specified dimensions
    product = 1
    for k, v in specified.items():
        if v is not None:
            product *= v

    if num_gpus % product != 0:
        raise ValueError(
            f"Specified dimensions don't evenly divide num_gpus={num_gpus}"
        )

    remaining = num_gpus // product

    # Assign remaining to unspecified dimensions
    if len(unspecified) == 1:
        specified[unspecified[0]] = remaining
    elif len(unspecified) == 2:
        # Heuristic: prefer TP over PP over DP for the smaller factor
        # to keep TP within a node
        factors = _factorize(remaining)

        if "tp" in unspecified and "pp" in unspecified:
            # Assign smaller to TP, larger to PP
            specified["tp"] = min(factors)
            specified["pp"] = max(factors)
        elif "tp" in unspecified and "dp" in unspecified:
            specified["tp"] = min(factors)
            specified["dp"] = max(factors)
        else:  # dp and pp unspecified
            specified["pp"] = min(factors)
            specified["dp"] = max(factors)

    return (specified["dp"], specified["tp"], specified["pp"])


def _factorize(n: int) -> tuple[int, int]:
    """Find two factors of n closest to sqrt(n)."""
    import math

    sqrt_n = int(math.sqrt(n))
    for i in range(sqrt_n, 0, -1):
        if n % i == 0:
            return (i, n // i)
    return (1, n)


def build_hybrid_args(config_path: str) -> HybridArgs:
    """
    Build HybridArgs from config file.

    This function:
    1. Loads the YAML config
    2. Parses base Args fields using existing utils
    3. Parses parallelism-specific fields
    4. Resolves mesh dimensions

    Args:
        config_path: Path to config YAML file

    Returns:
        HybridArgs with all fields populated
    """
    cfg = load_config(config_path)

    # Build base Args (reuse existing validation)
    # We need to manually construct since build_args returns Args, not HybridArgs
    base_fields = _extract_base_args(cfg)

    # Parse parallelism config
    parallelism_cfg = cfg.get("parallelism", {})

    tp_cfg = parallelism_cfg.get("tensor_parallel", {})
    pp_cfg = parallelism_cfg.get("pipeline_parallel", {})

    tensor_parallel = TensorParallelConfig(
        style=tp_cfg.get("style", "colwise_rowwise"),
        sequence_parallel=tp_cfg.get("sequence_parallel", False),
        plan=tp_cfg.get("plan"),
    )

    pipeline_parallel = PipelineParallelConfig(
        schedule=pp_cfg.get("schedule", "1f1b"),
        num_microbatches=pp_cfg.get("num_microbatches", 4),
        chunks=pp_cfg.get("chunks", "auto"),
        layer_assignment=pp_cfg.get("layer_assignment"),
    )

    parallelism = ParallelismConfig(
        enabled=parallelism_cfg.get("enabled", False),
        data_parallel_size=parallelism_cfg.get("data_parallel_size"),
        tensor_parallel_size=parallelism_cfg.get("tensor_parallel_size"),
        pipeline_parallel_size=parallelism_cfg.get("pipeline_parallel_size"),
        tensor_parallel=tensor_parallel,
        pipeline_parallel=pipeline_parallel,
        mesh_dim_names=tuple(parallelism_cfg.get("mesh_dim_names", ["dp", "tp", "pp"])),
        backend=parallelism_cfg.get("backend", "nccl"),
    )

    # Get GPU count from config or environment
    num_gpus = cfg.get("num_gpus", 1)
    world_size = int(os.environ.get("WORLD_SIZE", num_gpus))

    # Resolve mesh dimensions
    resolved_dp, resolved_tp, resolved_pp = resolve_mesh_dimensions(
        world_size,
        parallelism.data_parallel_size,
        parallelism.tensor_parallel_size,
        parallelism.pipeline_parallel_size,
    )

    # Create HybridArgs
    args = HybridArgs(
        **base_fields,
        parallelism=parallelism,
        resolved_dp_size=resolved_dp,
        resolved_tp_size=resolved_tp,
        resolved_pp_size=resolved_pp,
    )

    # Validation
    _validate_hybrid_args(args)

    return args


def _extract_base_args(cfg: dict) -> dict:
    """Extract base Args fields from config dict."""
    # Training params
    training = cfg.get("training", {})

    # Dataset params
    dataset = cfg.get("dataset", {})

    # Dist params
    dist = cfg.get("dist_parameters", {})

    # PEFT params
    peft = cfg.get("peft", {})

    # Quantization params
    quant = cfg.get("quantization", {})

    return {
        "model_name": cfg.get("model_name", ""),
        "model_type": cfg.get("model_type", "llm"),
        "dataset_name": dataset.get("name", ""),
        "dataset_subset": dataset.get("subset"),
        "dataset_split": dataset.get("split", "train"),
        "epochs": training.get("epochs", 1),
        "batch_size": training.get("batch_size", 4),
        "max_length": training.get("max_length", 512),
        "learning_rate": training.get("learning_rate", 2e-5),
        "warmup_steps": training.get("warmup_steps", 100),
        "weight_decay": training.get("weight_decay", 0.01),
        "grad_clip": training.get("grad_clip", 1.0),
        "gradient_checkpointing": training.get("gradient_checkpointing", False),
        "strategy": cfg.get("strategy", "hybrid"),
        "num_gpus": cfg.get("num_gpus", 1),
        "checkpoint_dir": cfg.get("checkpoint_dir", "checkpoints/hybrid"),
        "save": cfg.get("save", True),
        "mixed_precision": dist.get("mixed_precision", True),
        "param_dtype": dist.get("param_dtype", "bfloat16"),
        "reduce_dtype": dist.get("reduce_dtype", "float32"),
        "output_dtype": dist.get("output_dtype", "float32"),
        "cast_forward_inputs": dist.get("cast_forward_inputs", True),
        "distribute_api": dist.get("distribute_api", "dcp_api"),
        "resume": cfg.get("save_load", {}).get("resume", False),
        "resume_path": cfg.get("save_load", {}).get("resume_path", ""),
        "load_model_from_hf": cfg.get("save_load", {}).get("load_model_from_hf", True),
        "prefetch_explicit": cfg.get("prefetch", {}).get("explicit", False),
        "forward_prefetch": cfg.get("prefetch", {}).get("forward", False),
        "backward_prefetch": cfg.get("prefetch", {}).get("backward", False),
        "peft_enabled": peft.get("enabled", False),
        "peft_type": peft.get("type", "lora"),
        "peft_r": peft.get("r", 8),
        "peft_alpha": peft.get("alpha", 32),
        "peft_dropout": peft.get("dropout", 0.1),
        "peft_target_modules": peft.get("target_modules", []),
        "peft_bias": peft.get("bias", "none"),
        "quantization_enabled": quant.get("enabled", False),
        "quantization_bits": quant.get("bits", 4),
        "quant_type": quant.get("quant_type", "nf4"),
        "compute_dtype": quant.get("compute_dtype", "bfloat16"),
        "double_quant": quant.get("double_quant", True),
        "wandb_log_with_train": cfg.get("wandb", {}).get("wandb_log_with_train", False),
        "wandb_entity": cfg.get("wandb", {}).get("wandb_entity", ""),
        "wandb_project": cfg.get("wandb", {}).get("wandb_project", "omni-train"),
        "wandb_run_name": cfg.get("wandb", {}).get("wandb_run_name", ""),
    }


def _validate_hybrid_args(args: HybridArgs) -> None:
    """Validate hybrid args for compatibility."""
    if not args.parallelism.enabled:
        return

    # Quantization is incompatible with TP (can't shard quantized tensors)
    if args.quantization_enabled and args.resolved_tp_size > 1:
        raise ValueError(
            "Quantization is incompatible with Tensor Parallelism. "
            "Quantized weights cannot be sharded across TP ranks. "
            "Use tp_size=1 or disable quantization."
        )

    # PEFT with TP requires special handling
    if args.peft_enabled and args.resolved_tp_size > 1:
        print(
            "Warning: PEFT with Tensor Parallelism requires adapter weight redistribution. "
            "This is experimental and may have correctness issues."
        )

    # Validate microbatch count for PP
    if args.resolved_pp_size > 1:
        pp_cfg = args.parallelism.pipeline_parallel
        if pp_cfg.num_microbatches < args.resolved_pp_size:
            print(
                f"Warning: num_microbatches ({pp_cfg.num_microbatches}) < pp_size ({args.resolved_pp_size}). "
                f"Pipeline will have low utilization. Recommend at least {args.resolved_pp_size * 2} microbatches."
            )
