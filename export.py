"""Export a trained model in Hugging Face format (save_load.export_hf: true).

The result is a folder `from_pretrained` can load directly, with no OMNI-Train code:

    checkpoints/hf/<unix_ms>[__lora...]/
        config.json, model.safetensors (sharded if large), generation_config.json
        tokenizer / processor files
        omni_train.json            what produced it (base model, algorithm, strategy, peft)

LoRA adapters are MERGED into the base weights, so the export is a plain model.
QLoRA / 8-bit runs are the exception: merging into quantized weights is lossy, so
they export the adapter only (`adapter/`, loadable with peft). Merge it into the
full-precision base with:  python scripts/merge_lora.py --adapter <dir>/adapter --out <dir>

How the weights are collected:
  * solo / DDP: the trained model itself (DDP: rank 0's copy), merged in place.
    Export runs after training and checkpoint saving, so mutating it is fine.
  * FSDP: every rank joins one all-gather of the full state dict (CPU-offloaded to
    rank 0); rank 0 rebuilds the model on CPU, loads it, merges, saves.
"""

from __future__ import annotations

import copy
import json
import os
import time

import torch
import torch.distributed as dist

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def _unwrap(model, strategy):
    return model.module if strategy == "ddp" else model


def _is_peft(module) -> bool:
    return hasattr(module, "peft_config") and hasattr(module, "merge_and_unload")


def _base_config(module):
    base = module.get_base_model() if _is_peft(module) else module
    return base.config


def full_state_dict(model, strategy):
    """The full (unsharded) state dict on CPU. For FSDP this is a collective: every
    rank must call it; only rank 0 gets the tensors."""
    if strategy == "fsdp":
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
        return get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    return {k: v.detach().cpu() for k, v in _unwrap(model, strategy).state_dict().items()}


def rebuild_from_state_dict(module, state_dict, model_type: str):
    """A CPU copy of `module` (HF model or PeftModel) holding `state_dict`, with LoRA
    merged. Used for FSDP, whose live parameters are sharded across ranks."""
    from distributed_utils import _get_auto_model_class
    config = copy.deepcopy(_base_config(module))
    model = _get_auto_model_class(model_type).from_config(config, dtype=torch.float32)
    if _is_peft(module):
        from peft import get_peft_model
        peft_cfg = copy.deepcopy(module.peft_config[module.active_adapter])
        peft_cfg.inference_mode = False
        model = get_peft_model(model, peft_cfg)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # A missing name is fine only if that weight SHARES MEMORY with one that was loaded
    # (tied lm_head = embedding). Otherwise it would be exported with random values.
    own = model.state_dict()
    loaded = {own[k].data_ptr() for k in state_dict if k in own and own[k].numel()}
    missing = [k for k in missing if own[k].data_ptr() not in loaded]
    if missing or unexpected:
        raise RuntimeError(f"export: state dict does not match the model "
                           f"(missing={missing[:5]}, unexpected={unexpected[:5]})")
    return model.merge_and_unload() if _is_peft(model) else model


def export_hf(model, tokenizer, args, rank, strategy, out_dir=None):
    """Write the HF export. Call on EVERY rank (FSDP gathers collectively).
    Returns the output folder (rank 0) or None."""
    from distributed_utils import _checkpoint_run_tag
    if out_dir is None:
        out_dir = os.path.join(args.checkpoint_dir, "hf", f"{int(time.time() * 1000)}{_checkpoint_run_tag(args)}")
    module = _unwrap(model, strategy)
    quantized = bool(getattr(args, "quantization_enabled", False))

    state_dict = None
    if strategy == "fsdp":
        state_dict = full_state_dict(model, strategy)  # collective: all ranks

    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        if quantized:
            # Adapter only (merging into 4/8-bit weights would round them again).
            module.save_pretrained(os.path.join(out_dir, "adapter"))
            kind = "adapter"
        else:
            if strategy == "fsdp":
                merged = rebuild_from_state_dict(module, state_dict, args.model_type)
            else:
                merged = module.merge_and_unload() if _is_peft(module) else module
            merged = merged.to("cpu", dtype=DTYPES[args.export_dtype])
            merged.config.use_cache = True  # training turned it off; inference wants it
            merged.save_pretrained(out_dir, safe_serialization=True)
            kind = "merged" if getattr(args, "peft_enabled", False) else "full"
        if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(out_dir)
        with open(os.path.join(out_dir, "omni_train.json"), "w") as f:
            json.dump({
                "base_model": args.model_name,
                "model_type": args.model_type,
                "algorithm": getattr(args, "algorithm", "pretrain"),
                "strategy": strategy,
                "peft": getattr(args, "peft_type", None) if getattr(args, "peft_enabled", False) else None,
                "quantization_bits": args.quantization_bits if quantized else None,
                "export": kind,
                "dtype": None if quantized else args.export_dtype,
            }, f, indent=2)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    return out_dir if rank == 0 else None
