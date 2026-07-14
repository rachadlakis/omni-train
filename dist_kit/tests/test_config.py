## dist_kit/tests/test_config.py ##
##
## DistConfig: nested config.yaml-schema parsing, validation rules,
## numeric coercion, and config-file resolution order.

import os

import pytest

from dist_kit.config import DistConfig, resolve_config_path, _KIT_DIR


# ---------------------------------------------------------------------------
# Defaults & validation
# ---------------------------------------------------------------------------

def test_defaults_are_valid():
    c = DistConfig()
    assert c.strategy == "fsdp"
    assert c.dcp_api is True
    assert c.dataset_name == ""  # empty → random-batch demo


def test_invalid_strategy_rejected():
    with pytest.raises(ValueError, match="strategy"):
        DistConfig(strategy="pipeline")


def test_invalid_distribute_api_rejected():
    with pytest.raises(ValueError, match="distribute_api"):
        DistConfig(distribute_api="magic_api")


def test_invalid_model_type_rejected():
    with pytest.raises(ValueError, match="model_type"):
        DistConfig(model_type="diffusion")


def test_invalid_dtype_rejected():
    with pytest.raises(ValueError, match="param_dtype"):
        DistConfig(param_dtype="float8")


def test_quantization_without_peft_rejected():
    for bits in (4, 8):
        with pytest.raises(ValueError, match="requires peft_enabled"):
            DistConfig(strategy="ddp", quantization_enabled=True, quantization_bits=bits)


def test_fsdp_with_quantization_rejected():
    with pytest.raises(ValueError, match="incompatible"):
        DistConfig(strategy="fsdp", peft_enabled=True, quantization_enabled=True)


def test_qlora_implies_quantization():
    c = DistConfig(strategy="ddp", peft_enabled=True, peft_type="qlora")
    assert c.quantization_enabled is True


def test_invalid_quantization_bits_rejected():
    with pytest.raises(ValueError, match="quantization_bits"):
        DistConfig(strategy="ddp", peft_enabled=True, quantization_enabled=True, quantization_bits=3)


def test_peft_unsupported_model_type_rejected():
    with pytest.raises(ValueError, match="peft_enabled"):
        DistConfig(model_type="yolo", peft_enabled=True)


def test_resume_without_path_rejected():
    with pytest.raises(ValueError, match="resume_path"):
        DistConfig(resume=True)


def test_dcp_api_property_tracks_distribute_api():
    assert DistConfig(distribute_api="dcp_api").dcp_api is True
    assert DistConfig(distribute_api="dtensor_api").dcp_api is False


# ---------------------------------------------------------------------------
# Nested from_dict parsing (config.yaml schema)
# ---------------------------------------------------------------------------

NESTED = {
    "model_name": "facebook/opt-125m",
    "model_type": "llm",
    "strategy": "ddp",
    "num_gpus": 2,
    "checkpoint_dir": "ckpts",
    "save": False,
    "dataset": {"name": "wikitext", "subset": "wikitext-2-raw-v1", "split": "train[:2%]"},
    "training": {
        "epochs": 4, "batch_size": 16, "max_length": 128,
        "learning_rate": "2e-5",  # PyYAML parses no-dot scientific notation as str
        "gradient_checkpointing": False, "warmup_steps": 5, "weight_decay": 0.1,
    },
    "dist_parameters": {
        "mixed_precision": False, "param_dtype": "float16", "reduce_dtype": "float32",
        "output_dtype": "float16", "cast_forward_inputs": True, "distribute_api": "dtensor_api",
    },
    "save_load": {"resume": False, "resume_path": "", "load_model_from_hf": False},
    "prefetch": {"explicit": True, "forward": 3, "backward": 1},
    "peft": {"enabled": True, "type": "lora", "r": 4, "alpha": 8, "dropout": 0.1,
             "target_modules": "q_proj,v_proj", "bias": "all"},
    "quantization": {"enabled": False, "bits": 4, "quant_type": "fp4",
                     "compute_dtype": "float16", "double_quant": False},
    "wandb": {"wandb_log_with_train": True},  # unknown section → ignored
}


def test_nested_sections_parse():
    c = DistConfig.from_dict(NESTED)
    assert (c.model_name, c.strategy, c.num_gpus) == ("facebook/opt-125m", "ddp", 2)
    assert (c.dataset_name, c.dataset_subset, c.dataset_split) == ("wikitext", "wikitext-2-raw-v1", "train[:2%]")
    assert (c.epochs, c.batch_size, c.max_length) == (4, 16, 128)
    assert c.learning_rate == 2e-5 and isinstance(c.learning_rate, float)
    assert (c.warmup_steps, c.weight_decay, c.gradient_checkpointing) == (5, 0.1, False)
    assert (c.mixed_precision, c.param_dtype, c.distribute_api) == (False, "float16", "dtensor_api")
    assert c.cast_forward_inputs is True
    assert (c.resume, c.resume_path, c.load_model_from_hf) == (False, "", False)
    assert (c.explicit_prefetching, c.forward_prefetch, c.backward_prefetch) == (True, 3, 1)
    assert (c.peft_enabled, c.peft_type, c.peft_r, c.peft_alpha) == (True, "lora", 4, 8)
    assert (c.peft_dropout, c.peft_target_modules, c.peft_bias) == (0.1, "q_proj,v_proj", "all")
    assert (c.quantization_enabled, c.quantization_bits, c.quantization_type) == (False, 4, "fp4")
    assert (c.quantization_compute_dtype, c.quantization_double_quant) == ("float16", False)


def test_flat_keys_win_over_nested():
    cfg = dict(NESTED)
    cfg["batch_size"] = 99  # flat key overrides training.batch_size
    assert DistConfig.from_dict(cfg).batch_size == 99


def test_empty_and_none_dicts():
    assert DistConfig.from_dict({}).strategy == "fsdp"
    assert DistConfig.from_dict(None).strategy == "fsdp"


def test_null_sections_tolerated():
    # A YAML section left as "peft:" with no keys parses to None
    c = DistConfig.from_dict({"peft": None, "dataset": None, "strategy": "solo"})
    assert c.strategy == "solo" and c.peft_enabled is False


def test_nested_validation_still_fires():
    with pytest.raises(ValueError, match="incompatible"):
        DistConfig.from_dict({"strategy": "fsdp",
                              "peft": {"enabled": True},
                              "quantization": {"enabled": True}})


# ---------------------------------------------------------------------------
# from_yaml + resolution order
# ---------------------------------------------------------------------------

def test_bundled_config_yaml_parses():
    c = DistConfig.from_yaml(os.path.join(_KIT_DIR, "config.yaml"))
    assert c.model_name and c.strategy in {"solo", "ddp", "fsdp"}
    assert c.dataset_name == "wikitext"


def test_resolve_explicit_path_wins(monkeypatch):
    monkeypatch.setenv("DIST_CONFIG", "env.yaml")
    assert resolve_config_path("explicit.yaml") == "explicit.yaml"


def test_resolve_env_wins_over_cwd(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text("strategy: solo\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DIST_CONFIG", "other.yaml")
    assert resolve_config_path() == "other.yaml"


def test_resolve_cwd_config(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text("strategy: solo\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DIST_CONFIG", raising=False)
    assert resolve_config_path() == "config.yaml"


def test_resolve_falls_back_to_bundled(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no config.yaml here
    monkeypatch.delenv("DIST_CONFIG", raising=False)
    p = resolve_config_path()
    assert os.path.isabs(p) and p == os.path.join(_KIT_DIR, "config.yaml")
