"""How a model is placed on the devices: one module per strategy (solo, DDP, FSDP).

strategy: 5d has its own model and loop in omni_train.parallelism.
"""

from omni_train.strategies.common import get_model_layers, save_checkpoint
from omni_train.strategies.ddp import apply_ddp
from omni_train.strategies.fsdp import apply_fsdp
from omni_train.strategies.solo import apply_solo

__all__ = ["apply_ddp", "apply_fsdp", "apply_solo", "get_model_layers", "save_checkpoint"]
