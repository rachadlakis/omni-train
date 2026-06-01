"""Tests for build_args config validation guards in utils.py."""
import pytest
from utils import build_args


# ------------------------------------------------------------------
# Happy path
# ------------------------------------------------------------------

def test_valid_config_returns_args(cfg):
    args = build_args(cfg)
    assert args.strategy == "fsdp"
    assert args.epochs == 1
    assert args.peft_enabled is False
    assert args.quantization_enabled is False


# ------------------------------------------------------------------
# Model / strategy validation
# ------------------------------------------------------------------

def test_invalid_strategy_raises(cfg):
    cfg["strategy"] = "tensor_parallel"
    with pytest.raises(ValueError, match="strategy"):
        build_args(cfg)


def test_invalid_model_type_raises(cfg):
    cfg["model_type"] = "diffusion"
    with pytest.raises(ValueError, match="model_type"):
        build_args(cfg)


# ------------------------------------------------------------------
# Training hyperparameter guards
# ------------------------------------------------------------------

@pytest.mark.parametrize("field,bad_value,match", [
    ("epochs",       0,   "epochs"),
    ("batch_size",   0,   "batch_size"),
    ("max_length",   0,   "max_length"),
    ("warmup_steps", -1,  "warmup_steps"),
    ("weight_decay", -1,  "weight_decay"),
    ("grad_clip",    -1,  "grad_clip"),
])
def test_invalid_training_params_raise(cfg, field, bad_value, match):
    cfg["training"][field] = bad_value
    with pytest.raises(ValueError, match=match):
        build_args(cfg)


# ------------------------------------------------------------------
# PEFT validation
# ------------------------------------------------------------------

def test_invalid_peft_type_raises(cfg):
    cfg["peft"]["enabled"] = True
    cfg["peft"]["type"] = "adapter"
    with pytest.raises(ValueError, match="peft.type"):
        build_args(cfg)


def test_peft_unsupported_model_type_raises(cfg):
    cfg["model_type"] = "yolo"
    cfg["peft"]["enabled"] = True
    with pytest.raises(ValueError, match="PEFT is not supported"):
        build_args(cfg)


# ------------------------------------------------------------------
# Quantization validation
# ------------------------------------------------------------------

def test_4bit_quantization_without_peft_raises(cfg):
    """4-bit without PEFT is mathematically impossible — hard error."""
    cfg["quantization"]["enabled"] = True
    cfg["quantization"]["bits"] = 4
    cfg["peft"]["enabled"] = False
    with pytest.raises(ValueError, match="requires peft"):
        build_args(cfg)


def test_8bit_quantization_without_peft_warns(cfg):
    """8-bit without PEFT is unstable but allowed — emits UserWarning, does not raise."""
    cfg["quantization"]["enabled"] = True
    cfg["quantization"]["bits"] = 8
    cfg["peft"]["enabled"] = False
    cfg["strategy"] = "solo"   # avoid fsdp+quant hard error
    with pytest.warns(UserWarning, match="NaN"):
        args = build_args(cfg)
    assert args.quantization_enabled is True
    assert args.peft_enabled is False


def test_invalid_quantization_bits_raises(cfg):
    cfg["quantization"]["bits"] = 2
    with pytest.raises(ValueError, match="quantization.bits"):
        build_args(cfg)


def test_qlora_supports_8bit(cfg):
    cfg["strategy"] = "ddp"  # quant+fsdp is blocked; use ddp
    cfg["peft"]["type"] = "qlora"
    cfg["quantization"]["bits"] = 8
    args = build_args(cfg)
    assert args.quantization_bits == 8
    assert args.peft_enabled is True
    assert args.quantization_enabled is True


# ------------------------------------------------------------------
# FSDP + quantization guard
# ------------------------------------------------------------------
# CHANGED: was pytest.raises(SystemExit) with code==1.
# Reason: build_args() raises ValueError (not sys.exit). sys.exit(1) only happens
# in train.py's __main__ block which wraps build_args in a try/except ValueError.
# The test was calling build_args() directly, so SystemExit was never raised.
# Old code:
#   with pytest.raises(SystemExit) as exc_info:
#       build_args(cfg)
#   assert exc_info.value.code == 1

def test_fsdp_plus_quantization_raises(cfg):
    cfg["strategy"] = "fsdp"
    cfg["peft"]["enabled"] = True
    cfg["quantization"]["enabled"] = True
    with pytest.raises(ValueError, match="FSDP"):
        build_args(cfg)


def test_ddp_plus_quantization_does_not_exit(cfg):
    """DDP + quantization is allowed — only FSDP is blocked."""
    cfg["strategy"] = "ddp"
    cfg["peft"]["enabled"] = True
    cfg["quantization"]["enabled"] = True
    args = build_args(cfg)
    assert args.quantization_enabled is True


# ------------------------------------------------------------------
# Custom transformer validation
# ------------------------------------------------------------------

def test_custom_transformer_dim_not_divisible_by_heads_raises(cfg):
    cfg["model_type"] = "custom_transformer"
    cfg["custom_transformer_args"] = {
        "n_layers": 2,
        "vocab_size": 8,
        "max_seq_len": 16,
        "dim": 17,    # not divisible by n_heads=4
        "n_heads": 4,
        "dropout_p": 0.0,
    }
    with pytest.raises(ValueError, match="divisible"):
        build_args(cfg)


def test_custom_transformer_valid(cfg):
    cfg["model_type"] = "custom_transformer"
    cfg["custom_transformer_args"] = {
        "n_layers": 2,
        "vocab_size": 8,
        "max_seq_len": 16,
        "dim": 16,
        "n_heads": 4,
        "dropout_p": 0.0,
    }
    args = build_args(cfg)
    assert args.custom_dim == 16
    assert args.custom_n_heads == 4
