## Checkpoint.py ##

import os
import time
import torch
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
    _init_optim_state,
)
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import distribute_tensor, DTensor

MODEL_CHECKPOINT = "model_state_dict.pt"
OPTIM_CHECKPOINT = "optim_state_dict.pt"
PARAMS = "params"

def get_latest_checkpoint_folder(path):
    max_num = None
    max_name = None
    if not os.path.exists(path):
        return max_name
    for name in os.listdir(path):
        folder_path = os.path.join(path, name)
        if os.path.isdir(folder_path):
            try:
                # Support tagged folders: <timestamp>__lora_q4
                # Parsing only the timestamp prefix preserves chronological ordering
                # while allowing suffix metadata for run mode.
                num = int(name.split("__", 1)[0])
                if max_num is None or num > max_num:
                    max_num = num
                    max_name = name
            except ValueError:
                pass  # Skip non-numeric folder names
    return max_name

class Checkpointer:
    """ Checkpointer class for FSDP2. Supports both DCP and DTensor APIs.
    
    Attributes:
        folder (str): Path to the checkpoint directory.
        dcp_api (bool): Whether to use DCP API.
        last_training_time (str | None): The latest checkpoint folder name.  
    """
    def __init__(self, folder: str, dcp_api: bool, run_tag: str = ""):
        self.folder: str = folder
        self.dcp_api: bool = dcp_api
        # run_tag is appended to newly written checkpoint folder names
        # so checkpoints remain self-descriptive (for example, __qlora_q4).
        self.run_tag: str = run_tag
        self.last_training_time = get_latest_checkpoint_folder(
            f"{folder}/fsdp/{'dcp_api' if dcp_api else 'dtensor_api'}"
        )

    def is_empty(self):
        return self.last_training_time is None

    def load_model(self, model: FSDPModule):
        last_model_checkpoint = (
            f"{self.folder}/fsdp/{'dcp_api' if self.dcp_api else 'dtensor_api'}"
            f"/{self.last_training_time}/{MODEL_CHECKPOINT}"
        )
        if self.dcp_api:
            # CHANGED: only rank 0 loads the file from disk; all other ranks pass an empty dict.
            #
            # Reason: set_model_state_dict with broadcast_from_rank0=True ignores the state
            # dict from every rank except rank 0 — it is discarded before any data is used.
            # The old code loaded the full checkpoint on ALL ranks simultaneously:
            #
            #   full_state_dict = torch.load(last_model_checkpoint, mmap=True, ...)  ← all ranks
            #   set_model_state_dict(..., broadcast_from_rank0=True)
            #
            # For a 70B model (140 GB checkpoint) with 8 ranks on 4 nodes, that was
            # 8 × 140 GB = 1.1 TB of pointless CPU RAM and network I/O per checkpoint load.
            # Now rank 0 is the single reader; all ranks receive their shard via NCCL broadcast.
            #
            # DTensor API does NOT use broadcast_from_rank0 — each rank calls distribute_tensor
            # on the full tensor locally, so all ranks must still load there (see below).
            import torch.distributed as _dist
            _rank = _dist.get_rank() if _dist.is_initialized() else 0
            if _rank == 0:
                full_state_dict = torch.load(
                    last_model_checkpoint,
                    mmap=True,
                    weights_only=True,
                    map_location="cpu",
                )
            else:
                full_state_dict = {}   # broadcast_from_rank0=True ignores this
            set_model_state_dict(
                model=model, # type: ignore
                model_state_dict=full_state_dict,
                options=StateDictOptions(
                    full_state_dict=True,
                    broadcast_from_rank0=True,
                ),
            )
            return

        # DTensor API path: each rank calls distribute_tensor() locally which requires the
        # full source tensor to be present — there is no broadcast protocol here, so all
        # ranks must load the checkpoint. mmap=True reduces physical RAM pressure via OS
        # page mapping but does not eliminate the per-rank file read.
        full_state_dict = torch.load(
            last_model_checkpoint,
            mmap=True,
            weights_only=True,
            map_location="cpu"
        )
        meta_sharded_state_dict = model.state_dict() # type: ignore
        sharded_state_dict = {}
        for param_name, full_tensor in full_state_dict.items():
            sharded_meta_param = meta_sharded_state_dict.get(param_name)
            sharded_tensor = distribute_tensor(
                full_tensor,
                sharded_meta_param.device_mesh,
                sharded_meta_param.placements,
            )
            sharded_state_dict[param_name] = nn.Parameter(sharded_tensor)

        ## Choose `assign=True` since we cannot call `copy_` on meta tensor
        model.load_state_dict(sharded_state_dict, strict=False, assign=True) # type: ignore

    def load_optim(self, model: FSDPModule, opt: torch.optim.Optimizer):
        last_optim_checkpoint = (
            f"{self.folder}/fsdp/{'dcp_api' if self.dcp_api else 'dtensor_api'}"
            f"/{self.last_training_time}/{OPTIM_CHECKPOINT}"
        )
        full_optimizer_state_dict = torch.load(
            last_optim_checkpoint, 
            mmap=True,
            weights_only=True,
            map_location=torch.device("cpu")
        )
        ## For DCP API
        if self.dcp_api:
            set_optimizer_state_dict(
                model=model, # type: ignore
                optimizers=opt,
                optim_state_dict=full_optimizer_state_dict,
                options=StateDictOptions(
                    full_state_dict=True,
                    broadcast_from_rank0=True,
                ),
            )
            return
        ## For DTensor API
        _init_optim_state(opt)
        param_groups = opt.state_dict()["param_groups"]
        state = opt.state_dict()["state"]

        full_param_groups = full_optimizer_state_dict["param_groups"]
        full_state = full_optimizer_state_dict["state"]

        for param_group, full_param_group in zip(param_groups, full_param_groups):
            for key, value in full_param_group.items():
                if key == PARAMS:
                    continue
                param_group[key] = value
            for pid, full_pid in zip(param_group[PARAMS], full_param_group[PARAMS]):
                if pid not in state:
                    continue
                param_state = state[pid]
                full_param_state = full_state[full_pid]
                for attr, full_tensor in full_param_state.items():
                    sharded_tensor = param_state[attr]
                    if isinstance(sharded_tensor, DTensor):
                        # exp_avg is DTensor
                        param_state[attr] = distribute_tensor(
                            full_tensor,
                            sharded_tensor.device_mesh,
                            sharded_tensor.placements,
                        )
                    else:
                        # step is plain tensor
                        param_state[attr] = full_tensor
        opt.load_state_dict(
            {
                "param_groups": param_groups,
                "state": state,
            }
        )

    def _get_full_model_state_dict(self, model: FSDPModule):
        if self.dcp_api:
            return get_model_state_dict( 
                model=model, # type: ignore
                options=StateDictOptions(
                    full_state_dict=True,
                    cpu_offload=True,
                ),
            )
        sharded_state_dict = model.state_dict() # type: ignore
        cpu_state_dict = {}
        for param_name, sharded_param in sharded_state_dict.items():
            full_param = sharded_param.full_tensor()
            if torch.distributed.get_rank() == 0:
                cpu_state_dict[param_name] = full_param.cpu()
            else:
                del full_param
        return cpu_state_dict

    def _get_full_optimizer_state_dict(self,model: FSDPModule,opt: torch.optim.Optimizer):
        if self.dcp_api:
            return get_optimizer_state_dict(
                model=model, # type: ignore
                optimizers=opt,
                options=StateDictOptions(
                    full_state_dict=True,
                    cpu_offload=True,
                ),
            )
        is_rank_zero = torch.distributed.get_rank() == 0
        sharded_state_dict = opt.state_dict()
        sharded_state = sharded_state_dict["state"]
        full_state = {}
        for group_id, sharded_group in sharded_state.items():
            group_state = {}
            for attr, sharded_tensor in sharded_group.items():
                if isinstance(sharded_tensor, DTensor):
                    # "exp_avg" in AdamW is `DTensor`
                    full_tensor = sharded_tensor.full_tensor()
                else:
                    # "step" in AdamW is plain tensor
                    full_tensor = sharded_tensor
                if is_rank_zero:
                    group_state[attr] = full_tensor.cpu()
                else:
                    del full_tensor
            if is_rank_zero:
                full_state[group_id] = group_state
            else:
                del group_state
        if is_rank_zero:
            return {
                "param_groups": sharded_state_dict["param_groups"],
                "state": full_state,
            }
        else:
            return {}

    def save(self, model: FSDPModule, optim: torch.optim.Optimizer):
        model_state_dict = self._get_full_model_state_dict(model)
        optim_state_dict = self._get_full_optimizer_state_dict(model, optim)
        
        if torch.distributed.get_rank() == 0:
            new_training_time = int(time.time() * 1000)
            # Keep timestamp first for sortable checkpoint folders and add optional mode suffix.
            checkpoint_folder_name = f"{new_training_time}{self.run_tag}" if self.run_tag else str(new_training_time)
            new_checkpoint_folder = f"{self.folder}/fsdp/{'dcp_api' if self.dcp_api else 'dtensor_api'}/{checkpoint_folder_name}"
            new_model_checkpoint = f"{new_checkpoint_folder}/{MODEL_CHECKPOINT}"
            new_optim_checkpoint = f"{new_checkpoint_folder}/{OPTIM_CHECKPOINT}"
            os.makedirs(new_checkpoint_folder, exist_ok=True)
            torch.save(model_state_dict, new_model_checkpoint)
            torch.save(optim_state_dict, new_optim_checkpoint)
