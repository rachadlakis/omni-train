"""Tests for pure helper functions in distributed_utils.py."""
import pytest
import torch
from types import SimpleNamespace
from unittest.mock import MagicMock

from distributed_utils import (
    _load_dtype,
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


# ------------------------------------------------------------------
# _load_dtype — trainable weights must stay fp32
# ------------------------------------------------------------------

def _dtype_args(**kw):
    base = dict(mixed_precision=True, param_dtype="bfloat16", peft_enabled=False, strategy="solo")
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.mark.parametrize("strategy", ["solo", "ddp", "fsdp"])
def test_load_dtype_full_finetune_is_fp32(strategy):
    # Even with mixed precision on: compute is cast later (autocast / FSDP mp_policy).
    assert _load_dtype(_dtype_args(strategy=strategy)) == torch.float32


@pytest.mark.parametrize("strategy", ["solo", "ddp"])
def test_load_dtype_peft_base_uses_param_dtype(strategy):
    # Frozen base weights can be stored low-precision outside FSDP.
    assert _load_dtype(_dtype_args(strategy=strategy, peft_enabled=True)) == torch.bfloat16


def test_load_dtype_fsdp_peft_is_fp32():
    # FSDP2 needs one dtype per param group, so base + adapters both stay fp32.
    assert _load_dtype(_dtype_args(strategy="fsdp", peft_enabled=True)) == torch.float32


def test_load_dtype_no_mixed_precision_is_fp32():
    assert _load_dtype(_dtype_args(mixed_precision=False, peft_enabled=True)) == torch.float32


# ------------------------------------------------------------------
# GPU name → TFLOPS lookup
# ------------------------------------------------------------------

@pytest.mark.parametrize("name,expected", [
    ("NVIDIA RTX A4000", 77),
    ("NVIDIA A100-SXM4-80GB", 312),
    ("NVIDIA GeForce RTX 4090", 83),
    ("NVIDIA L40S", 362),
    ("NVIDIA L4", 120),
    ("NVIDIA RTX A2000", 40),   # must not match "a2"
    ("a100", 312),              # short keys still work
    ("CPU", 40),
])
def test_match_gpu_tflops(name, expected):
    from utils import _match_gpu_tflops
    table = {"a100": 312, "l40s": 362, "l40": 181, "l4": 120, "a4000": 77, "a2": 31, "4090": 83}
    assert _match_gpu_tflops(name, table, default=40) == expected


def test_estimate_training_time_uses_full_gpu_name():
    from utils import estimate_training_time
    common = dict(num_params=125_000_000, steps_per_epoch=100, epochs=1, batch_size=8)
    fast = estimate_training_time(gpu_type="NVIDIA H100 80GB HBM3", **common)
    unknown = estimate_training_time(gpu_type="Some Unknown GPU", **common)
    assert fast["total_seconds"] < unknown["total_seconds"]
