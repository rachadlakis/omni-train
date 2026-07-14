## dist_kit ##
##
## Portable distributed-training kit extracted from OMNI-Train.
## Drop this folder into any project and use:
##
##   from dist_kit import DistConfig, setup_dist_process_group, apply_fsdp, ...

from .config import DistConfig, DTYPE_MAP
from .dist_setup import (
    BACKEND,
    setup_dist_process_group,
    cleanup,
    dist_barrier,
    print_on_rank_0,
    print_banner_on_rank_0,
    print_on_all_ranks,
    gpu_memory_snapshot,
)
from .strategies import (
    apply_solo,
    apply_ddp,
    apply_fsdp,
    save_checkpoint,
    get_model_layers,
    inspect_model,
)
from .checkpoint import Checkpointer, get_latest_checkpoint_folder
from .data import get_text_dataloader

__all__ = [
    "DistConfig", "DTYPE_MAP", "BACKEND",
    "setup_dist_process_group", "cleanup", "dist_barrier",
    "print_on_rank_0", "print_banner_on_rank_0", "print_on_all_ranks",
    "gpu_memory_snapshot",
    "apply_solo", "apply_ddp", "apply_fsdp", "save_checkpoint",
    "get_model_layers", "inspect_model",
    "Checkpointer", "get_latest_checkpoint_folder",
    "get_text_dataloader",
]
