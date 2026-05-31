"""
OMNI-Train 3D Parallelism Plugin

This module provides 3D parallelism (Data × Tensor × Pipeline) support
as a drop-in plugin that works alongside existing OMNI-Train code.

No modifications to existing files required.
"""

from .hybrid_config_adapter import HybridArgs, build_hybrid_args
from .hybrid_utils import (
    setup_hybrid_parallelism,
    apply_tensor_parallelism,
    apply_pipeline_parallelism,
    HybridModel,
)

__all__ = [
    "HybridArgs",
    "build_hybrid_args",
    "setup_hybrid_parallelism",
    "apply_tensor_parallelism",
    "apply_pipeline_parallelism",
    "HybridModel",
]

__version__ = "1.0.0"
