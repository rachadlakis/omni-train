"""Tests for pure helper functions in distributed_utils.py."""
import pytest
import torch
from unittest.mock import MagicMock

from distributed_utils import (
    _checkpoint_run_tag,
    _normalize_target_modules,
    _dtype_from_name,
    get_model_layers,
    DTYPE_MAP,
)


# ------------------------------------------------------------------
# _dtype_from_name
# ------------------------------------------------------------------

@pytest.mark.parametrize("name,expected", [
    ("bfloat16", torch.bfloat16),
    ("float32",  torch.float32),
    ("float16",  torch.float16),
])
def test_dtype_from_name_valid(name, expected):
    assert _dtype_from_name(name) == expected


def test_dtype_from_name_invalid_raises():
    with pytest.raises(ValueError, match="Unsupported dtype"):
        _dtype_from_name("int8")


# ------------------------------------------------------------------
# _normalize_target_modules
# ------------------------------------------------------------------

def test_normalize_target_modules_string_single():
    assert _normalize_target_modules("q_proj") == "q_proj"


def test_normalize_target_modules_string_comma_separated():
    result = _normalize_target_modules("q_proj, k_proj, v_proj")
    assert result == ["q_proj", "k_proj", "v_proj"]


def test_normalize_target_modules_list_passthrough():
    modules = ["q_proj", "v_proj"]
    assert _normalize_target_modules(modules) == modules


def test_normalize_target_modules_fallback():
    assert _normalize_target_modules(None) == "all-linear"


def test_normalize_target_modules_all_linear_string():
    assert _normalize_target_modules("all-linear") == "all-linear"


# ------------------------------------------------------------------
# _checkpoint_run_tag
# ------------------------------------------------------------------

def _make_args(**kwargs):
    args = MagicMock()
    args.peft_enabled = kwargs.get("peft_enabled", False)
    args.peft_type = kwargs.get("peft_type", "lora")
    args.quantization_enabled = kwargs.get("quantization_enabled", False)
    args.quantization_bits = kwargs.get("quantization_bits", 4)
    return args


def test_checkpoint_run_tag_no_peft_no_quant():
    args = _make_args()
    assert _checkpoint_run_tag(args) == ""


def test_checkpoint_run_tag_lora_only():
    args = _make_args(peft_enabled=True, peft_type="lora")
    assert _checkpoint_run_tag(args) == "__lora"


def test_checkpoint_run_tag_lora_plus_quant_4bit():
    args = _make_args(peft_enabled=True, peft_type="lora", quantization_enabled=True, quantization_bits=4)
    assert _checkpoint_run_tag(args) == "__lora_q4"


def test_checkpoint_run_tag_quant_only_8bit():
    args = _make_args(peft_enabled=False, quantization_enabled=True, quantization_bits=8)
    assert _checkpoint_run_tag(args) == "__q8"


# ------------------------------------------------------------------
# get_model_layers — duck-typed model mocks
# ------------------------------------------------------------------

def _make_layers(n=4):
    return [MagicMock() for _ in range(n)]


def test_get_model_layers_decoder_style():
    model = MagicMock()
    del model.base_model  # ensure no PeftModel unwrap
    model.model.decoder.layers = _make_layers(4)
    layers, kind = get_model_layers(model)
    assert len(layers) == 4 # type: ignore
    assert kind == "decoder"


def test_get_model_layers_generic_layers():
    model = MagicMock(spec=[])  # no attributes at all
    model.layers = _make_layers(3)
    layers, kind = get_model_layers(model)
    assert len(layers) == 3 # type: ignore
    assert kind == "generic"


def test_get_model_layers_returns_none_when_unknown():
    model = MagicMock(spec=[])  # no matching attrs
    layers, kind = get_model_layers(model)
    assert layers is None
    assert kind is None
