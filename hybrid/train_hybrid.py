#!/usr/bin/env python3
"""
3D Parallelism Training Entry Point for OMNI-Train.

This is a standalone training script that uses 3D parallelism
(Data × Tensor × Pipeline) without modifying the existing train.py.

Usage:
    torchrun --nproc_per_node=8 hybrid/train_hybrid.py --config configs/hybrid/llm_3d_8gpu.yaml
"""

import argparse
import os
import sys
from datetime import datetime

import torch
import torch.distributed as dist

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from existing OMNI-Train modules (unchanged)
from data import get_dataloader
from utils import get_tokenizer

# Import from hybrid plugin (new code)
from hybrid.hybrid_config_adapter import build_hybrid_args, HybridArgs
from hybrid.hybrid_utils import (
    setup_hybrid_parallelism,
    apply_tensor_parallelism,
    apply_pipeline_parallelism,
    apply_data_parallelism,
    create_pipeline_schedule,
    HybridModel,
    HybridMesh,
)
from hybrid.hybrid_checkpoint import HybridCheckpointer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="3D Parallelism Training")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--dp_size",
        type=int,
        default=None,
        help="Override data parallel size",
    )
    parser.add_argument(
        "--tp_size",
        type=int,
        default=None,
        help="Override tensor parallel size",
    )
    parser.add_argument(
        "--pp_size",
        type=int,
        default=None,
        help="Override pipeline parallel size",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum training steps (for testing)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable profiling",
    )
    return parser.parse_args()


def setup_distributed():
    """Initialize distributed training."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    return rank, local_rank, world_size, device


def load_model_for_hybrid(args: HybridArgs, device: torch.device):
    """
    Load model for 3D parallelism.

    Uses meta device initialization when possible to minimize memory.
    """
    from transformers import AutoModelForCausalLM, AutoConfig

    model_name = args.model_name

    # Load config first
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        config.use_cache = False

    # Load model
    # For large models, we could use meta device, but for simplicity
    # we load on CPU first then move to device
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.bfloat16 if args.param_dtype == "bfloat16" else torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    return model


def train_step_with_pipeline(
    schedule,
    input_batch,
    mesh: HybridMesh,
):
    """
    Execute one training step with pipeline parallelism.

    The schedule handles microbatching and stage synchronization.
    """
    # Split batch into microbatches (handled by schedule)
    # Execute forward/backward through pipeline

    if mesh.pp_size > 1:
        # Pipeline schedule handles forward/backward
        losses = schedule.step(input_batch)
        return sum(losses) / len(losses) if losses else torch.tensor(0.0)
    else:
        # No pipeline - direct forward/backward
        raise NotImplementedError("Use regular train_step for non-PP training")


def train_step_standard(
    model: HybridModel,
    batch,
    optimizer,
    device,
    mesh: HybridMesh,
):
    """
    Standard training step (no pipeline parallelism).

    Used when PP size is 1.
    """
    optimizer.zero_grad()

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    labels = batch.get("labels", input_ids).to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    loss = outputs.loss
    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()

    return loss.item()


def main():
    """Main training function."""
    cli_args = parse_args()

    # Setup distributed
    rank, local_rank, world_size, device = setup_distributed()

    # Load and build config
    args = build_hybrid_args(cli_args.config)

    # Override mesh dimensions from CLI if provided
    if cli_args.dp_size:
        args.resolved_dp_size = cli_args.dp_size
    if cli_args.tp_size:
        args.resolved_tp_size = cli_args.tp_size
    if cli_args.pp_size:
        args.resolved_pp_size = cli_args.pp_size

    # Print config on rank 0
    if rank == 0:
        print("=" * 60)
        print("OMNI-Train 3D Parallelism")
        print("=" * 60)
        print(f"Model: {args.model_name}")
        print(f"World size: {world_size}")
        print(f"Mesh: DP={args.resolved_dp_size} × TP={args.resolved_tp_size} × PP={args.resolved_pp_size}")
        print(f"Strategy: {args.strategy}")
        print("=" * 60)

    # Setup 3D mesh
    mesh = setup_hybrid_parallelism(args)

    if rank == 0:
        print(f"Mesh initialized: dp_rank={mesh.dp_rank}, tp_rank={mesh.tp_rank}, pp_rank={mesh.pp_rank}")

    # Load tokenizer
    tokenizer = get_tokenizer(args.model_name)

    # Load model
    if rank == 0:
        print("Loading model...")

    model = load_model_for_hybrid(args, device)

    # Apply 3D parallelism in order: TP → PP → DP
    # (TP splits layers, PP partitions stages, DP replicates)

    if rank == 0:
        print("Applying tensor parallelism...")
    model = apply_tensor_parallelism(model, mesh, args)

    if rank == 0:
        print("Applying pipeline parallelism...")
    model, pipeline_stage = apply_pipeline_parallelism(model, mesh, args, device)

    if rank == 0:
        print("Applying data parallelism...")
    model = apply_data_parallelism(model, mesh, args)

    # Wrap in HybridModel
    hybrid_model = HybridModel(model, mesh, pipeline_stage)
    hybrid_model = hybrid_model.to(device)

    # Create optimizer (only for trainable params)
    trainable_params = [p for p in hybrid_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Create checkpointer
    checkpointer = HybridCheckpointer(
        model=hybrid_model,
        optimizer=optimizer,
        mesh=mesh,
        args=args,
    )

    # Resume if requested
    start_epoch = 0
    start_step = 0
    if args.resume and args.resume_path:
        start_epoch, start_step = checkpointer.load()
        if rank == 0:
            print(f"Resumed from epoch {start_epoch}, step {start_step}")

    # Load data
    if rank == 0:
        print("Loading dataset...")

    dataloader = get_dataloader(
        args,
        tokenizer,
        rank=mesh.dp_rank,  # Use DP rank for data sharding
        world_size=mesh.dp_size,  # Only shard across DP dimension
    )

    # Create pipeline schedule if using PP
    schedule = None
    if mesh.pp_size > 1 and pipeline_stage is not None:
        loss_fn = torch.nn.CrossEntropyLoss() if hybrid_model.is_last_stage else None
        schedule = create_pipeline_schedule(pipeline_stage, args, loss_fn)

    # Training loop
    if rank == 0:
        print("Starting training...")
        print("-" * 60)

    global_step = start_step
    max_steps = cli_args.max_steps

    for epoch in range(start_epoch, args.epochs):
        hybrid_model.train()

        # Set epoch for distributed sampler
        if hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(epoch)

        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            if max_steps and global_step >= max_steps:
                break

            # Training step
            if mesh.pp_size > 1 and schedule is not None:
                loss = train_step_with_pipeline(schedule, batch, mesh)
            else:
                loss = train_step_standard(
                    hybrid_model, batch, optimizer, device, mesh
                )

            epoch_loss += loss
            num_batches += 1
            global_step += 1

            # Log progress
            if rank == 0 and batch_idx % 10 == 0:
                avg_loss = epoch_loss / num_batches
                print(f"Epoch {epoch+1}/{args.epochs} | Step {batch_idx} | Loss: {avg_loss:.4f}")

        if max_steps and global_step >= max_steps:
            break

        # Epoch summary
        if rank == 0:
            avg_loss = epoch_loss / max(num_batches, 1)
            print(f"Epoch {epoch+1} complete | Avg Loss: {avg_loss:.4f}")

        # Save checkpoint
        if args.save:
            checkpointer.save(epoch=epoch + 1, step=global_step)
            if rank == 0:
                print(f"Checkpoint saved at epoch {epoch+1}")

    # Final summary
    if rank == 0:
        print("=" * 60)
        print("Training complete!")
        print(f"Final step: {global_step}")
        print("=" * 60)

    # Cleanup
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
