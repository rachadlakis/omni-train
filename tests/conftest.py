"""Shared fixtures for all tests."""
import copy
import pytest


@pytest.fixture
def base_cfg():
    """A minimal valid config dict that passes build_args without errors."""
    return {
        "model_name": "facebook/opt-125m",
        "model_type": "llm",
        "dataset": {
            "name": "wikitext",
            "subset": "wikitext-2-raw-v1",
            "split": "train[:1%]",
        },
        "strategy": "fsdp",
        "num_gpus": 1,
        "checkpoint_dir": "checkpoints",
        "save": False,
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "max_length": 32,
            "learning_rate": 1e-4,
            "gradient_checkpointing": False,
            "warmup_steps": 0,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
        },
        "dist_parameters": {
            "mixed_precision": False,
            "param_dtype": "bfloat16",
            "reduce_dtype": "float32",
            "output_dtype": "bfloat16",
            "cast_forward_inputs": False,
            "distribute_api": "dcp_api",
        },
        "save_load": {
            "resume": False,
            "resume_path": "",
            "load_model_from_hf": False,
        },
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
        "prefetch": {
            "explicit": False,
            "forward": 1,
            "backward": 1,
        },
        "wandb": {
            "wandb_log_with_train": False,
            "wandb_entity": "test",
            "wandb_project": "test",
            "wandb_run_name": "test-run",
        },
        "MLFlow": {
            "mlflow_log_with_train": False,
            "mlflow_tracking_uri": "http://localhost:5000",
            "mlflow_experiment_name": "test-experiment",
        },
    }


@pytest.fixture
def cfg(base_cfg):
    """Deep-copy of base_cfg so tests can mutate freely."""
    return copy.deepcopy(base_cfg)
