from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _default_config(project_root: Path) -> dict[str, Any]:
    config_path = project_root / "config.yaml"
    if config_path.exists():
        loaded = yaml.safe_load(config_path.read_text())
        if isinstance(loaded, dict):
            return loaded

    return {
        "model_name": "facebook/opt-125m",
        "dataset": {"name": "wikitext", "subset": "wikitext-2-raw-v1", "split": "train[:1%]"},
        "training": {
            "epochs": 3,
            "batch_size": 8,
            "max_length": 128,
            "learning_rate": 2e-4,
            "warmup_steps": 100,
            "weight_decay": 0.01,
            "grad_clip": 1.0,
            "gradient_checkpointing": True,
        },
        "strategy": "solo",
        "num_gpus": 1,
        "checkpoint_dir": "checkpoints",
        "save": True,
        "dist_parameters": {
            "mixed_precision": True,
            "param_dtype": "bfloat16",
            "reduce_dtype": "float32",
            "output_dtype": "bfloat16",
            "cast_forward_inputs": False,
            "distribute_api": "dcp_api",
        },
        "save_load": {"resume": False, "resume_path": "", "load_model_from_hf": True},
        "prefetch": {"explicit": True, "forward": 2, "backward": 2},
        "peft": {
            "enabled": False,
            "type": "lora",
            "r": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": "all-linear",
            "bias": "none",
        },
        "quantization": {
            "enabled": False,
            "bits": 4,
            "quant_type": "nf4",
            "compute_dtype": "bfloat16",
            "double_quant": True,
        },
        "wandb": {
            "wandb_log_with_train": False,
            "wandb_entity": "omni-train",
            "wandb_project": "omni-train",
            "wandb_run_name": "",
        },
    }


def is_mini_project_config(raw: dict[str, Any]) -> bool:
    # custom_transformer omits model_name (model is built from scratch), so accept
    # either model_name or model_type as the marker for the flat mini-project schema.
    has_model_key = "model_name" in raw or "model_type" in raw
    return has_model_key and "strategy" in raw and "training" in raw


def adapt_ui_config_to_mini(raw: dict[str, Any], project_root: Path) -> dict[str, Any]:
    base = _default_config(project_root)

    if is_mini_project_config(raw):
        return _deep_merge(base, raw)

    model = raw.get("model", {}) if isinstance(raw.get("model"), dict) else {}
    data = raw.get("data", {}) if isinstance(raw.get("data"), dict) else {}
    distributed = raw.get("distributed", {}) if isinstance(raw.get("distributed"), dict) else {}
    training = raw.get("training", {}) if isinstance(raw.get("training"), dict) else {}
    wandb = raw.get("wandb", {}) if isinstance(raw.get("wandb"), dict) else {}

    strategy = str(distributed.get("strategy", "solo")).lower()
    if strategy == "none":
        strategy = "solo"
    if strategy not in {"solo", "ddp", "fsdp", "hybrid"}:
        strategy = "solo"

    finetune_mode = str(model.get("finetune_mode", "lora")).lower()
    peft_enabled = finetune_mode in {"lora", "qlora"}
    quant_enabled = bool(model.get("quantize", False)) or finetune_mode == "qlora"

    # Map UI model type to backend model_type.
    # "vision" = HuggingFace Vision Transformer (ViT/Swin/DeiT…) → LoRA-compatible.
    # "cnn"/"detection" use their own training paths and default to "llm" only when
    # they reach this adapter (which currently only handles transformer-based models).
    _ui_to_model_type = {
        "llm":                "llm",
        "vlm":                "vlm",
        "vision":             "vision",
        "embedding":          "encoder",
        "detection":          "yolo",
        "cnn":                "llm",   # CNN goes through a separate training path
        "custom_transformer": "custom_transformer",
    }
    ui_model_type = str(model.get("type", "llm")).lower()
    model_type = _ui_to_model_type.get(ui_model_type, "llm")

    mapped = {
        "model_type": model_type,
        "model_name": model.get("name", base.get("model_name")),
        "dataset": {
            "name": data.get("name", base["dataset"].get("name", "wikitext")),
            "subset": data.get("subset", data.get("dataset_full_name", base["dataset"].get("subset", "wikitext-2-raw-v1"))),
            "split": data.get("split", data.get("train_split", base["dataset"].get("split", "train[:1%]"))),
        },
        "training": {
            "epochs": training.get("epochs", base["training"].get("epochs", 3)),
            "batch_size": training.get("batch_size", base["training"].get("batch_size", 8)),
            "max_length": training.get("max_length", data.get("max_seq_len", base["training"].get("max_length", 128))),
            "learning_rate": training.get("learning_rate", training.get("lr", base["training"].get("learning_rate", 2e-4))),
            "warmup_steps": training.get("warmup_steps", base["training"].get("warmup_steps", 100)),
            "weight_decay": training.get("weight_decay", base["training"].get("weight_decay", 0.01)),
            "grad_clip": training.get("grad_clip", base["training"].get("grad_clip", 1.0)),
            "gradient_checkpointing": training.get(
                "gradient_checkpointing",
                distributed.get("activation_checkpointing", base["training"].get("gradient_checkpointing", True)),
            ),
        },
        "strategy": strategy,
        "num_gpus": int(raw.get("num_gpus", distributed.get("gpu_count", base.get("num_gpus", 1))) or 1),
        "checkpoint_dir": training.get("checkpoint_dir", raw.get("checkpoint_dir", base.get("checkpoint_dir", "checkpoints"))),
        "save": raw.get("save", base.get("save", True)),
        "dist_parameters": {
            "mixed_precision": distributed.get("mixed_precision", base["dist_parameters"].get("mixed_precision", True)),
            "param_dtype": base["dist_parameters"].get("param_dtype", "bfloat16"),
            "reduce_dtype": base["dist_parameters"].get("reduce_dtype", "float32"),
            "output_dtype": base["dist_parameters"].get("output_dtype", "bfloat16"),
            "cast_forward_inputs": base["dist_parameters"].get("cast_forward_inputs", False),
            "distribute_api": base["dist_parameters"].get("distribute_api", "dcp_api"),
        },
        "save_load": {
            "resume": raw.get("resume", base["save_load"].get("resume", False)),
            "resume_path": raw.get("resume_path", base["save_load"].get("resume_path", "")),
            "load_model_from_hf": raw.get("load_model_from_hf", base["save_load"].get("load_model_from_hf", True)),
        },
        "prefetch": {
            "explicit": raw.get("explicit_prefetching", base["prefetch"].get("explicit", True)),
            "forward": raw.get("forward_prefetch", base["prefetch"].get("forward", 2)),
            "backward": raw.get("backward_prefetch", base["prefetch"].get("backward", 2)),
        },
        "peft": {
            "enabled": peft_enabled,
            "type": "qlora" if finetune_mode == "qlora" else "lora",
            "r": model.get("lora_r", base["peft"].get("r", 16)),
            "alpha": model.get("lora_alpha", base["peft"].get("alpha", 32)),
            "dropout": model.get("lora_dropout", base["peft"].get("dropout", 0.05)),
            "target_modules": base["peft"].get("target_modules", "all-linear"),
            "bias": base["peft"].get("bias", "none"),
        },
        "quantization": {
            "enabled": quant_enabled,
            "bits": model.get("quant_bits", base["quantization"].get("bits", 4)),
            "quant_type": base["quantization"].get("quant_type", "nf4"),
            "compute_dtype": base["quantization"].get("compute_dtype", "bfloat16"),
            "double_quant": base["quantization"].get("double_quant", True),
        },
        "wandb": {
            "wandb_log_with_train": wandb.get("wandb_log_with_train", base["wandb"].get("wandb_log_with_train", False)),
            "wandb_entity": wandb.get("wandb_entity", base["wandb"].get("wandb_entity", "omni-train")),
            "wandb_project": wandb.get("wandb_project", base["wandb"].get("wandb_project", "omni-train")),
            "wandb_run_name": wandb.get("wandb_run_name", base["wandb"].get("wandb_run_name", "")),
        },
    }

    # Pass custom transformer architecture args when present
    if model_type == "custom_transformer":
        arch = model.get("arch", {})
        mapped["custom_transformer_args"] = {
            "n_layers":   int(arch.get("n_layers",   6)),
            "vocab_size": int(arch.get("vocab_size", 8192)),
            "max_seq_len": int(arch.get("max_seq_len", 512)),
            "dim":        int(arch.get("dim",        512)),
            "n_heads":    int(arch.get("n_heads",    8)),
            "dropout_p":  float(arch.get("dropout_p", 0.1)),
        }
        # Override max_length with the model's max_seq_len so data loading uses the right length
        mapped["training"]["max_length"] = mapped["custom_transformer_args"]["max_seq_len"]

    # Forward 3D topology block for hybrid strategy
    if strategy == "hybrid":
        topology = raw.get("topology")
        if not topology and isinstance(distributed.get("topology"), dict):
            topology = distributed["topology"]
        if isinstance(topology, dict) and topology:
            mapped["topology"] = topology

    # Forward launch_mode and slurm config so app.py can dispatch correctly
    if raw.get("launch_mode"):
        mapped["launch_mode"] = str(raw["launch_mode"])
    if isinstance(raw.get("slurm"), dict):
        mapped["slurm"] = raw["slurm"]

    return _deep_merge(base, mapped)


def validate_mini_config(mini_cfg: dict[str, Any], project_root: Path) -> None:
    utils_path = project_root / "utils.py"
    spec = importlib.util.spec_from_file_location("fsdp_mini_utils", str(utils_path))
    if not spec or not spec.loader:
        raise RuntimeError(f"Failed to load utils module from {utils_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    build_args = getattr(module, "build_args", None)
    if build_args is None:
        raise RuntimeError("build_args() was not found in fsdp-mini-project utils.py")

    build_args(mini_cfg)
