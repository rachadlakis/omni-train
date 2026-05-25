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
#
# Returns list[(ModuleList, layer_type_name)] — one entry per stack.
# Multi-tower models (T5, BART, CLIP, VLM) return multiple stacks.
# ------------------------------------------------------------------

def _make_module_list(n=4):
    return torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(n)])


def test_get_model_layers_decoder_style():
    """A model exposing model.decoder.layers (OPT-style) should be detected via the
    known-attribute-paths fallback."""
    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = _make_module_list(4)
    class Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = Decoder()
    class Outer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()
    stacks = get_model_layers(Outer())
    assert len(stacks) == 1
    layers, kind = stacks[0]
    assert len(layers) == 4
    assert kind == "Linear"


def test_get_model_layers_generic_layers():
    """A flat .layers attribute should still be picked up by the known-paths fallback."""
    class Flat(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = _make_module_list(3)
    stacks = get_model_layers(Flat())
    assert len(stacks) == 1
    layers, kind = stacks[0]
    assert len(layers) == 3
    assert kind == "Linear"


def test_get_model_layers_returns_empty_when_unknown():
    """Model with no ModuleList anywhere returns an empty list."""
    class Empty(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(2, 2)
    assert get_model_layers(Empty()) == []


def test_get_model_layers_no_split_modules_hint():
    """When _no_split_modules is populated, the function must walk the tree and
    collect every ModuleList whose elements match. This unlocks Llama-style
    `model.model.layers` that the old implementation missed."""
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(2, 2)
    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Block() for _ in range(5)])
    class Wrapper(torch.nn.Module):
        _no_split_modules = ["Block"]
        def __init__(self):
            super().__init__()
            self.model = Backbone()
    stacks = get_model_layers(Wrapper())
    assert len(stacks) == 1
    layers, kind = stacks[0]
    assert len(layers) == 5
    assert kind == "Block"


def test_get_model_layers_multi_tower():
    """Multi-tower models (T5/BART/CLIP/VLM) should return one stack per tower."""
    class EncBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(2, 2)
    class DecBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(2, 2)
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.block = torch.nn.ModuleList([EncBlock() for _ in range(3)])
    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.block = torch.nn.ModuleList([DecBlock() for _ in range(2)])
    class TwoTower(torch.nn.Module):
        _no_split_modules = ["EncBlock", "DecBlock"]
        def __init__(self):
            super().__init__()
            self.encoder = Encoder()
            self.decoder = Decoder()
    stacks = get_model_layers(TwoTower())
    kinds = sorted(k for _, k in stacks)
    assert kinds == ["DecBlock", "EncBlock"]
    sizes = {k: len(layers) for layers, k in stacks}
    assert sizes == {"EncBlock": 3, "DecBlock": 2}
