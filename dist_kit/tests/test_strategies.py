## dist_kit/tests/test_strategies.py ##
##
## Pure helpers in strategies.py: run tags, target-module normalization,
## PEFT/checkpoint compatibility, layer detection, meta-buffer fixes, and
## the solo save_checkpoint path. No GPU, no network.

import pytest
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModel,
    GPT2Config,
    GPT2LMHeadModel,
)

from dist_kit.config import DistConfig
from dist_kit.strategies import (
    _build_quantization_config,
    _check_checkpoint_peft_compat,
    _checkpoint_run_tag,
    _get_auto_model_class,
    _restore_buffers,
    _snapshot_buffers,
    _normalize_target_modules,
    get_model_layers,
    save_checkpoint,
)


# ---------------------------------------------------------------------------
# Run tags
# ---------------------------------------------------------------------------

def test_run_tag_plain():
    assert _checkpoint_run_tag(DistConfig()) == ""


def test_run_tag_lora():
    args = DistConfig(peft_enabled=True, peft_type="lora")
    assert _checkpoint_run_tag(args) == "__lora"


def test_run_tag_qlora_quantized():
    args = DistConfig(strategy="ddp", peft_enabled=True, peft_type="qlora", quantization_bits=4)
    assert _checkpoint_run_tag(args) == "__qlora_q4"


# ---------------------------------------------------------------------------
# Target modules normalization
# ---------------------------------------------------------------------------

def test_target_modules_single_token():
    assert _normalize_target_modules("all-linear") == "all-linear"


def test_target_modules_comma_string():
    assert _normalize_target_modules("q_proj, v_proj,k_proj") == ["q_proj", "v_proj", "k_proj"]


def test_target_modules_list_passthrough():
    assert _normalize_target_modules(["a", "b"]) == ["a", "b"]


def test_target_modules_fallback():
    assert _normalize_target_modules(None) == "all-linear"


# ---------------------------------------------------------------------------
# Model class mapping
# ---------------------------------------------------------------------------

def test_auto_model_class_map():
    assert _get_auto_model_class("llm") is AutoModelForCausalLM
    assert _get_auto_model_class("seq2seq") is AutoModelForSeq2SeqLM
    assert _get_auto_model_class("encoder") is AutoModel
    assert _get_auto_model_class("unknown") is AutoModelForCausalLM  # default


# ---------------------------------------------------------------------------
# PEFT / checkpoint compatibility guard
# ---------------------------------------------------------------------------

def test_peft_checkpoint_into_non_peft_config_rejected():
    sd = {"base_model.model.layer.weight": torch.zeros(1)}
    with pytest.raises(ValueError, match="WITH PEFT"):
        _check_checkpoint_peft_compat(sd, peft_enabled=False, resume_path="x.pt")


def test_plain_checkpoint_into_peft_config_rejected():
    sd = {"model.layers.0.weight": torch.zeros(1)}
    with pytest.raises(ValueError, match="WITHOUT PEFT"):
        _check_checkpoint_peft_compat(sd, peft_enabled=True, resume_path="x.pt")


def test_matching_checkpoints_pass():
    _check_checkpoint_peft_compat({"base_model.model.w": torch.zeros(1)}, True, "x.pt")
    _check_checkpoint_peft_compat({"model.w": torch.zeros(1)}, False, "x.pt")
    _check_checkpoint_peft_compat({}, True, "x.pt")  # empty → deferred to load_state_dict


# ---------------------------------------------------------------------------
# Quantization config
# ---------------------------------------------------------------------------

def test_quantization_config_disabled_returns_none():
    assert _build_quantization_config(DistConfig(), rank=0) is None


def test_quantization_config_4bit():
    args = DistConfig(strategy="ddp", peft_enabled=True, quantization_enabled=True, quantization_bits=4)
    cfg = _build_quantization_config(args, rank=0)
    assert cfg.load_in_4bit and not cfg.load_in_8bit


def test_quantization_config_8bit():
    args = DistConfig(strategy="ddp", peft_enabled=True, quantization_enabled=True, quantization_bits=8)
    cfg = _build_quantization_config(args, rank=0)
    assert cfg.load_in_8bit and not cfg.load_in_4bit


# ---------------------------------------------------------------------------
# Layer detection for FSDP sharding
# ---------------------------------------------------------------------------

class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)


def test_layers_found_via_known_path():
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([_Block() for _ in range(3)])

    stacks = get_model_layers(Toy())
    assert len(stacks) == 1
    layers, name = stacks[0]
    assert len(layers) == 3 and name == "_Block"


def test_layers_found_via_largest_modulelist_heuristic():
    class Odd(nn.Module):
        def __init__(self):
            super().__init__()
            self.small = nn.ModuleList([nn.Linear(2, 2) for _ in range(2)])
            self.blocks = nn.ModuleList([_Block() for _ in range(4)])  # more params

    stacks = get_model_layers(Odd())
    assert len(stacks) == 1
    layers, _ = stacks[0]
    assert len(layers) == 4


def test_no_layers_returns_empty():
    assert get_model_layers(nn.Linear(4, 4)) == []


def test_layers_found_on_hf_gpt2():
    config = GPT2Config(n_layer=2, n_embd=8, n_head=2, vocab_size=64, n_positions=16)
    model = GPT2LMHeadModel(config)  # built from config — no download
    stacks = get_model_layers(model)
    assert len(stacks) == 1
    layers, name = stacks[0]
    assert len(layers) == 2 and name == "GPT2Block"


# ---------------------------------------------------------------------------
# Meta-buffer materialization (RoPE fix)
# ---------------------------------------------------------------------------

def test_meta_init_restores_rope_with_theta_and_scaling():
    """Params on meta, buffers real → snapshot → to_empty → restore reproduces the
    HF-computed inv_freq (rope_theta=500k + llama3 scaling, not base 10000)."""
    from accelerate import init_empty_weights
    from transformers import LlamaConfig

    cfg = LlamaConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=256, rope_theta=500000.0,
        rope_scaling={"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                      "high_freq_factor": 4.0, "original_max_position_embeddings": 64},
    )
    ref = AutoModelForCausalLM.from_config(cfg).eval()
    with init_empty_weights(include_buffers=False):
        m = AutoModelForCausalLM.from_config(cfg)
    buffers = _snapshot_buffers(m)
    m.to_empty(device="cpu")
    m.load_state_dict(ref.state_dict(), strict=False)
    _restore_buffers(m, buffers, torch.device("cpu"), rank=0)
    m.eval()

    ids = torch.randint(0, 128, (2, 8))
    with torch.no_grad():
        assert torch.equal(ref(input_ids=ids).logits, m(input_ids=ids).logits)


def test_buffer_without_snapshot_zeroed():
    class B(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("mask", torch.empty(3, 3, device="meta"), persistent=False)

    b = B()
    _restore_buffers(b, _snapshot_buffers(b), torch.device("cpu"), rank=0)
    assert b.mask.device.type == "cpu"
    assert torch.all(b.mask == 0)


def test_persistent_buffers_untouched_by_default():
    class B(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("stats", torch.ones(2))

    b = B()
    snap = _snapshot_buffers(b)
    b.stats.fill_(5.0)  # value "loaded from checkpoint"
    _restore_buffers(b, snap, torch.device("cpu"), rank=0)
    assert torch.all(b.stats == 5)


# ---------------------------------------------------------------------------
# save_checkpoint (solo path — no process group required)
# ---------------------------------------------------------------------------

def test_save_checkpoint_solo(tmp_path):
    args = DistConfig(strategy="solo", checkpoint_dir=str(tmp_path))
    model = nn.Linear(4, 4)
    optim = torch.optim.AdamW(model.parameters())
    save_checkpoint("solo", model, optim, rank=0, args=args)
    files = list((tmp_path / "solo").glob("*.pt"))
    assert len(files) == 1
    sd = torch.load(files[0], weights_only=True)
    assert "weight" in sd and sd["weight"].shape == (4, 4)


def test_save_checkpoint_requires_checkpoint_dir():
    args = DistConfig(strategy="solo", checkpoint_dir="")
    with pytest.raises(ValueError, match="checkpoint_dir"):
        save_checkpoint("solo", nn.Linear(2, 2), None, rank=0, args=args)
