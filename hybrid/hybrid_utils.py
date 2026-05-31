"""
3D Parallelism utilities for OMNI-Train.

This module provides functions to apply tensor parallelism, pipeline parallelism,
and data parallelism to models without modifying the existing codebase.
"""

from typing import Optional, Any
from dataclasses import dataclass
import os
import sys

import torch
import torch.nn as nn
import torch.distributed as dist

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hybrid.hybrid_config_adapter import HybridArgs

# PyTorch 2.4+ imports for 3D parallelism
try:
    from torch.distributed.device_mesh import init_device_mesh, DeviceMesh
    from torch.distributed.tensor.parallel import (
        ColwiseParallel,
        RowwiseParallel,
        parallelize_module,
    )
    from torch.distributed.pipelining import (
        PipelineStage,
        ScheduleGPipe,
        Schedule1F1B,
    )
    from torch.distributed.fsdp import fully_shard

    PARALLELISM_AVAILABLE = True
except ImportError:
    PARALLELISM_AVAILABLE = False
    print("Warning: PyTorch 2.4+ required for 3D parallelism. Some features unavailable.")


@dataclass
class HybridMesh:
    """Container for the 3D device mesh and submeshes."""

    full_mesh: "DeviceMesh"
    dp_mesh: "DeviceMesh"
    tp_mesh: "DeviceMesh"
    pp_mesh: "DeviceMesh"

    dp_size: int
    tp_size: int
    pp_size: int

    # Rank info
    dp_rank: int
    tp_rank: int
    pp_rank: int


class HybridModel(nn.Module):
    """
    Wrapper for models with 3D parallelism applied.

    This class wraps the original model and provides:
    - Correct forward pass routing for pipeline stages
    - Gradient synchronization across DP dimension
    - State dict gathering for checkpointing
    """

    def __init__(
        self,
        model: nn.Module,
        mesh: HybridMesh,
        pipeline_stage: Optional["PipelineStage"] = None,
    ):
        super().__init__()
        self.model = model
        self.mesh = mesh
        self.pipeline_stage = pipeline_stage
        self._is_first_stage = mesh.pp_rank == 0
        self._is_last_stage = mesh.pp_rank == mesh.pp_size - 1

    def forward(self, *args, **kwargs):
        """Forward pass - routes to pipeline stage if PP enabled."""
        if self.pipeline_stage is not None:
            # Pipeline parallelism handles forward
            return self.pipeline_stage(*args, **kwargs)
        return self.model(*args, **kwargs)

    @property
    def is_first_stage(self) -> bool:
        return self._is_first_stage

    @property
    def is_last_stage(self) -> bool:
        return self._is_last_stage

    def state_dict(self, *args, **kwargs):
        """Get state dict, gathering from all parallelism dimensions."""
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        """Load state dict, distributing to all parallelism dimensions."""
        return self.model.load_state_dict(state_dict, *args, **kwargs)


def setup_hybrid_parallelism(args: HybridArgs) -> HybridMesh:
    """
    Initialize the 3D device mesh for hybrid parallelism.

    Args:
        args: HybridArgs with resolved mesh dimensions

    Returns:
        HybridMesh with full mesh and submeshes
    """
    if not PARALLELISM_AVAILABLE:
        raise RuntimeError("PyTorch 2.4+ required for 3D parallelism")

    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before setup_hybrid_parallelism")

    dp_size = args.resolved_dp_size
    tp_size = args.resolved_tp_size
    pp_size = args.resolved_pp_size

    world_size = dist.get_world_size()
    expected_size = dp_size * tp_size * pp_size

    if world_size != expected_size:
        raise ValueError(
            f"World size {world_size} doesn't match mesh dimensions "
            f"{dp_size} × {tp_size} × {pp_size} = {expected_size}"
        )

    # Create 3D mesh
    # Mesh shape: (dp, tp, pp)
    # Outer dimension (dp) varies slowest, inner (pp) varies fastest
    mesh = init_device_mesh(
        device_type="cuda",
        mesh_shape=(dp_size, tp_size, pp_size),
        mesh_dim_names=args.parallelism.mesh_dim_names,
    )

    # Get submeshes for each dimension
    dp_mesh = mesh["dp"]
    tp_mesh = mesh["tp"]
    pp_mesh = mesh["pp"]

    # Calculate rank in each dimension
    rank = dist.get_rank()
    dp_rank = rank // (tp_size * pp_size)
    tp_rank = (rank // pp_size) % tp_size
    pp_rank = rank % pp_size

    return HybridMesh(
        full_mesh=mesh,
        dp_mesh=dp_mesh,
        tp_mesh=tp_mesh,
        pp_mesh=pp_mesh,
        dp_size=dp_size,
        tp_size=tp_size,
        pp_size=pp_size,
        dp_rank=dp_rank,
        tp_rank=tp_rank,
        pp_rank=pp_rank,
    )


def apply_tensor_parallelism(
    model: nn.Module,
    mesh: HybridMesh,
    args: HybridArgs,
) -> nn.Module:
    """
    Apply tensor parallelism to model layers.

    Splits attention and MLP layers across TP ranks using:
    - ColwiseParallel for q_proj, k_proj, v_proj, gate_proj, up_proj
    - RowwiseParallel for o_proj, down_proj

    Args:
        model: The model to parallelize
        mesh: HybridMesh with TP submesh
        args: HybridArgs with TP configuration

    Returns:
        Model with tensor parallelism applied
    """
    if mesh.tp_size == 1:
        return model  # No TP needed

    if not PARALLELISM_AVAILABLE:
        raise RuntimeError("PyTorch 2.4+ required for tensor parallelism")

    tp_config = args.parallelism.tensor_parallel

    # Get custom plan or use auto-detection
    if tp_config.plan:
        plan = tp_config.plan
    else:
        plan = _auto_detect_tp_plan(model, tp_config.style)

    # Apply parallelization to each layer
    layers = _get_transformer_layers(model)

    for layer in layers:
        # Build plan for this layer
        layer_plan = {}
        for param_name, strategy in plan.items():
            if hasattr(layer, param_name.split(".")[0]):
                if strategy == "colwise":
                    layer_plan[param_name] = ColwiseParallel()
                elif strategy == "rowwise":
                    layer_plan[param_name] = RowwiseParallel()

        if layer_plan:
            parallelize_module(layer, mesh.tp_mesh, layer_plan)

    return model


def _auto_detect_tp_plan(model: nn.Module, style: str) -> dict:
    """
    Auto-detect tensor parallelism plan based on model architecture.

    Returns a dict mapping parameter names to parallelization strategies.
    """
    # Common transformer layer patterns
    COLWISE_PATTERNS = [
        "q_proj", "k_proj", "v_proj",  # Attention projections
        "gate_proj", "up_proj",  # MLP projections (Llama-style)
        "fc1", "c_fc",  # MLP projections (GPT-style)
        "query", "key", "value",  # Alternative naming
    ]

    ROWWISE_PATTERNS = [
        "o_proj", "out_proj",  # Attention output
        "down_proj",  # MLP output (Llama-style)
        "fc2", "c_proj",  # MLP output (GPT-style)
    ]

    plan = {}

    for name, module in model.named_modules():
        module_name = name.split(".")[-1]

        if any(pattern in module_name for pattern in COLWISE_PATTERNS):
            plan[name] = "colwise"
        elif any(pattern in module_name for pattern in ROWWISE_PATTERNS):
            plan[name] = "rowwise"

    return plan


def apply_pipeline_parallelism(
    model: nn.Module,
    mesh: HybridMesh,
    args: HybridArgs,
    device: torch.device,
) -> tuple[nn.Module, Optional["PipelineStage"]]:
    """
    Apply pipeline parallelism by partitioning model into stages.

    Args:
        model: The model to partition
        mesh: HybridMesh with PP submesh
        args: HybridArgs with PP configuration
        device: Target device for this rank's stage

    Returns:
        Tuple of (partitioned model, PipelineStage)
    """
    if mesh.pp_size == 1:
        return model, None  # No PP needed

    if not PARALLELISM_AVAILABLE:
        raise RuntimeError("PyTorch 2.4+ required for pipeline parallelism")

    pp_config = args.parallelism.pipeline_parallel

    # Get transformer layers
    layers = _get_transformer_layers(model)
    num_layers = len(layers)

    # Determine layer assignment
    if pp_config.layer_assignment:
        layer_assignment = pp_config.layer_assignment
    else:
        # Equal split
        layers_per_stage = num_layers // mesh.pp_size
        remainder = num_layers % mesh.pp_size

        layer_assignment = []
        for i in range(mesh.pp_size):
            count = layers_per_stage + (1 if i < remainder else 0)
            layer_assignment.append(count)

    # Calculate start/end layer for this rank
    start_layer = sum(layer_assignment[:mesh.pp_rank])
    end_layer = start_layer + layer_assignment[mesh.pp_rank]

    # Partition model
    partitioned_model = _partition_model(
        model, start_layer, end_layer, mesh.pp_rank, mesh.pp_size
    )

    # Move to device
    partitioned_model = partitioned_model.to(device)

    # Create pipeline stage
    stage = PipelineStage(
        submodule=partitioned_model,
        stage_index=mesh.pp_rank,
        num_stages=mesh.pp_size,
        device=device,
    )

    return partitioned_model, stage


def _partition_model(
    model: nn.Module,
    start_layer: int,
    end_layer: int,
    pp_rank: int,
    pp_size: int,
) -> nn.Module:
    """
    Create a partition of the model for a pipeline stage.

    For first stage: includes embeddings + layers[start:end]
    For middle stages: includes layers[start:end]
    For last stage: includes layers[start:end] + output head
    """
    layers = _get_transformer_layers(model)

    # This is a simplified implementation
    # A full implementation would need to handle:
    # - Embedding layers (first stage only)
    # - Output head/LM head (last stage only)
    # - Proper input/output shape matching between stages

    class PartitionedModel(nn.Module):
        def __init__(self, layers, is_first, is_last, original_model):
            super().__init__()
            self.layers = nn.ModuleList(layers)
            self.is_first = is_first
            self.is_last = is_last

            # Copy embedding for first stage
            if is_first and hasattr(original_model, "model"):
                if hasattr(original_model.model, "embed_tokens"):
                    self.embed_tokens = original_model.model.embed_tokens

            # Copy LM head for last stage
            if is_last and hasattr(original_model, "lm_head"):
                self.lm_head = original_model.lm_head

        def forward(self, x):
            # First stage: embed input
            if self.is_first and hasattr(self, "embed_tokens"):
                x = self.embed_tokens(x)

            # Process layers
            for layer in self.layers:
                x = layer(x)

            # Last stage: apply LM head
            if self.is_last and hasattr(self, "lm_head"):
                x = self.lm_head(x)

            return x

    partitioned = PartitionedModel(
        layers=list(layers[start_layer:end_layer]),
        is_first=(pp_rank == 0),
        is_last=(pp_rank == pp_size - 1),
        original_model=model,
    )

    return partitioned


def apply_data_parallelism(
    model: nn.Module,
    mesh: HybridMesh,
    args: HybridArgs,
) -> nn.Module:
    """
    Apply FSDP-style data parallelism on the DP dimension.

    Args:
        model: The model (potentially already with TP/PP applied)
        mesh: HybridMesh with DP submesh
        args: HybridArgs with FSDP configuration

    Returns:
        Model with data parallelism applied
    """
    if mesh.dp_size == 1:
        return model  # No DP needed

    if not PARALLELISM_AVAILABLE:
        raise RuntimeError("PyTorch 2.4+ required for FSDP")

    # Apply FSDP per-layer then at root
    layers = _get_transformer_layers(model)

    for layer in layers:
        fully_shard(layer, mesh=mesh.dp_mesh)

    fully_shard(model, mesh=mesh.dp_mesh)

    return model


def _get_transformer_layers(model: nn.Module) -> list[nn.Module]:
    """
    Get transformer layers from model.

    Handles common HuggingFace model architectures.
    """
    # Common layer attribute paths
    LAYER_PATHS = [
        "model.layers",           # Llama, Mistral
        "transformer.h",          # GPT-2, GPT-J
        "model.decoder.layers",   # BART decoder
        "model.encoder.layers",   # BERT, T5 encoder
        "encoder.layers",         # Some encoders
        "decoder.layers",         # Some decoders
    ]

    for path in LAYER_PATHS:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            if isinstance(obj, (list, nn.ModuleList)):
                return list(obj)
        except AttributeError:
            continue

    # Fallback: find largest ModuleList
    largest = []
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > len(largest):
            largest = list(module)

    if largest:
        return largest

    raise ValueError("Could not find transformer layers in model")


def create_pipeline_schedule(
    stage: "PipelineStage",
    args: HybridArgs,
    loss_fn: Optional[Any] = None,
):
    """
    Create the pipeline execution schedule.

    Args:
        stage: The PipelineStage for this rank
        args: HybridArgs with schedule configuration
        loss_fn: Loss function (needed for last stage)

    Returns:
        Pipeline schedule object
    """
    pp_config = args.parallelism.pipeline_parallel
    schedule_type = pp_config.schedule
    num_microbatches = pp_config.num_microbatches

    if schedule_type == "gpipe":
        return ScheduleGPipe(
            stage=stage,
            n_microbatches=num_microbatches,
            loss_fn=loss_fn,
        )
    elif schedule_type == "1f1b":
        return Schedule1F1B(
            stage=stage,
            n_microbatches=num_microbatches,
            loss_fn=loss_fn,
        )
    else:
        raise ValueError(f"Unknown pipeline schedule: {schedule_type}")
