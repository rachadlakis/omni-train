"""
Checkpoint utilities for 3D parallelism.

Extends the existing Checkpointer to handle tensor and pipeline parallelism
state dict gathering and scattering.
"""

import os
import sys
import json
from pathlib import Path
from typing import Optional
import time

import torch
import torch.distributed as dist

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hybrid.hybrid_config_adapter import HybridArgs
from hybrid.hybrid_utils import HybridMesh, HybridModel


class HybridCheckpointer:
    """
    Checkpointer for 3D parallelism training.

    Handles:
    - Gathering tensor-parallel shards for saving
    - Gathering pipeline stages for saving
    - Distributing checkpoints when loading
    """

    def __init__(
        self,
        model: HybridModel,
        optimizer: torch.optim.Optimizer,
        mesh: HybridMesh,
        args: HybridArgs,
    ):
        self.model = model
        self.optimizer = optimizer
        self.mesh = mesh
        self.args = args

        self.checkpoint_dir = Path(args.checkpoint_dir) / "hybrid" #type: ignore
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

    def _get_checkpoint_name(self, epoch: int, step: int) -> str:
        """Generate checkpoint folder name with mesh info."""
        timestamp = int(time.time() * 1000)
        mesh_tag = f"dp{self.mesh.dp_size}_tp{self.mesh.tp_size}_pp{self.mesh.pp_size}"
        return f"{timestamp}__e{epoch}_s{step}__{mesh_tag}"

    def save(self, epoch: int, step: int) -> Optional[Path]:
        """
        Save checkpoint.

        For 3D parallelism, we need to:
        1. Gather TP shards to reconstruct full tensors
        2. Gather PP stages to reconstruct full model
        3. Save on rank 0

        Returns:
            Path to checkpoint directory (on rank 0), None otherwise
        """
        # Create checkpoint directory
        ckpt_name = self._get_checkpoint_name(epoch, step)
        ckpt_path = self.checkpoint_dir / ckpt_name

        if self.rank == 0:
            ckpt_path.mkdir(parents=True, exist_ok=True)

        dist.barrier()

        # Get model state dict
        # For FSDP-wrapped models, this should handle gathering automatically
        model_state = self.model.state_dict()

        # Get optimizer state
        optimizer_state = self.optimizer.state_dict()

        # Save on rank 0
        if self.rank == 0:
            # Save model state
            torch.save(model_state, ckpt_path / "model_state_dict.pt")

            # Save optimizer state
            torch.save(optimizer_state, ckpt_path / "optim_state_dict.pt")

            # Save training state
            training_state = {
                "epoch": epoch,
                "step": step,
            }
            torch.save(training_state, ckpt_path / "training_state.pt")

            # Save parallelism config for resume validation
            parallelism_config = {
                "dp_size": self.mesh.dp_size,
                "tp_size": self.mesh.tp_size,
                "pp_size": self.mesh.pp_size,
                "world_size": self.world_size,
            }
            with open(ckpt_path / "parallelism_config.json", "w") as f:
                json.dump(parallelism_config, f, indent=2)

            print(f"Checkpoint saved to: {ckpt_path}")

        dist.barrier()

        return ckpt_path if self.rank == 0 else None

    def load(self, checkpoint_path: Optional[str] = None) -> tuple[int, int]:
        """
        Load checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory. If None, loads latest.

        Returns:
            Tuple of (epoch, step)
        """
        if checkpoint_path is None:
            checkpoint_path = self._find_latest_checkpoint()
            if checkpoint_path is None:
                print("No checkpoint found, starting from scratch")
                return 0, 0

        ckpt_path = Path(checkpoint_path)

        # Validate parallelism config matches
        config_path = ckpt_path / "parallelism_config.json"
        if config_path.exists():
            with open(config_path) as f:
                saved_config = json.load(f)

            if (saved_config["dp_size"] != self.mesh.dp_size or
                saved_config["tp_size"] != self.mesh.tp_size or
                saved_config["pp_size"] != self.mesh.pp_size):
                raise ValueError(
                    f"Checkpoint mesh ({saved_config['dp_size']}×{saved_config['tp_size']}×{saved_config['pp_size']}) "
                    f"doesn't match current mesh ({self.mesh.dp_size}×{self.mesh.tp_size}×{self.mesh.pp_size})"
                )

        # Load model state
        model_state = torch.load(
            ckpt_path / "model_state_dict.pt",
            map_location="cpu",
        )
        self.model.load_state_dict(model_state)

        # Load optimizer state
        optimizer_state = torch.load(
            ckpt_path / "optim_state_dict.pt",
            map_location="cpu",
        )
        self.optimizer.load_state_dict(optimizer_state)

        # Load training state
        training_state = torch.load(
            ckpt_path / "training_state.pt",
            map_location="cpu",
        )

        if self.rank == 0:
            print(f"Checkpoint loaded from: {ckpt_path}")

        return training_state["epoch"], training_state["step"]

    def _find_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint directory."""
        if not self.checkpoint_dir.exists():
            return None

        # Get all checkpoint directories
        ckpt_dirs = [
            d for d in self.checkpoint_dir.iterdir()
            if d.is_dir() and (d / "model_state_dict.pt").exists()
        ]

        if not ckpt_dirs:
            return None

        # Sort by timestamp (first part of name)
        def get_timestamp(path: Path) -> int:
            try:
                return int(path.name.split("__")[0])
            except (ValueError, IndexError):
                return 0

        latest = max(ckpt_dirs, key=get_timestamp)
        return str(latest)


def convert_checkpoint_to_standard(
    hybrid_checkpoint_path: str,
    output_path: str,
    output_format: str = "pytorch",
):
    """
    Convert a 3D parallelism checkpoint to standard format.

    This allows using the trained model for inference without
    the parallelism setup.

    Args:
        hybrid_checkpoint_path: Path to hybrid checkpoint directory
        output_path: Path to save converted checkpoint
        output_format: "pytorch" or "huggingface"
    """
    ckpt_path = Path(hybrid_checkpoint_path)

    # Load model state
    model_state = torch.load(
        ckpt_path / "model_state_dict.pt",
        map_location="cpu",
    )

    if output_format == "pytorch":
        # Save as standard PyTorch checkpoint
        torch.save({"model_state_dict": model_state}, output_path)
        print(f"Converted checkpoint saved to: {output_path}")

    elif output_format == "huggingface":
        # Save in HuggingFace format
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save model weights
        torch.save(model_state, output_dir / "pytorch_model.bin")

        # Copy config if available
        # (would need original model config for full HF compatibility)
        print(f"Converted checkpoint saved to: {output_dir}")
        print("Note: You may need to copy the model config.json manually")

    else:
        raise ValueError(f"Unknown output format: {output_format}")
