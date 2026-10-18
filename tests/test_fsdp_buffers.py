"""Buffer handling for the FSDP meta-init path (no GPU, no network).

The meta-init path builds the model with parameters on meta, calls to_empty(), then
loads weights. Non-persistent buffers (RoPE inv_freq, BERT position_ids, …) are not
in state_dict, so they must be snapshotted before to_empty() and restored after.
These tests run that exact sequence on CPU (minus fully_shard) and compare against
a normally constructed model.
"""

import pytest
import torch
import torch.nn as nn
from accelerate import init_empty_weights
from transformers import AutoModel, AutoModelForCausalLM, BertConfig, LlamaConfig

import distributed_utils
from distributed_utils import _remove_seed_dir, _restore_buffers, _snapshot_buffers


def _llama3_config():
    # rope_theta and llama3 rope scaling both change inv_freq away from the
    # textbook 1 / 10000^(2i/d) formula.
    return LlamaConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=256, rope_theta=500000.0,
        rope_scaling={
            "rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
            "high_freq_factor": 4.0, "original_max_position_embeddings": 64,
        },
    )


def _bert_config():
    return BertConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4)


@pytest.mark.parametrize("config_fn,model_cls,out_key", [
    (_llama3_config, AutoModelForCausalLM, "logits"),
    (_bert_config, AutoModel, "last_hidden_state"),
])
def test_meta_init_roundtrip_matches_reference(config_fn, model_cls, out_key):
    torch.manual_seed(0)
    cfg = config_fn()
    ref = model_cls.from_config(cfg, dtype=torch.float32).eval()

    with init_empty_weights(include_buffers=False):
        model = model_cls.from_config(cfg, dtype=torch.float32)
    assert next(model.parameters()).device.type == "meta"

    buffers = _snapshot_buffers(model)
    model.to_empty(device="cpu")  # every tensor is now uninitialized memory
    model.load_state_dict(ref.state_dict(), strict=False)
    _restore_buffers(model, buffers, torch.device("cpu"), rank=0)
    model.eval()

    ids = torch.randint(0, 128, (2, 16))
    with torch.no_grad():
        expected = getattr(ref(input_ids=ids), out_key)
        actual = getattr(model(input_ids=ids), out_key)
    assert torch.equal(expected, actual)


def test_restored_inv_freq_uses_rope_theta_and_scaling():
    """Regression: the old code recomputed inv_freq with base=10000, ignoring
    rope_theta and rope scaling (Llama 3, Qwen, Mistral)."""
    cfg = _llama3_config()
    ref = AutoModelForCausalLM.from_config(cfg)
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(cfg)
    buffers = _snapshot_buffers(model)
    model.to_empty(device="cpu")
    _restore_buffers(model, buffers, torch.device("cpu"), rank=0)

    expected = ref.model.rotary_emb.inv_freq
    assert torch.equal(model.model.rotary_emb.inv_freq, expected)
    dim = expected.shape[0] * 2
    textbook = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    assert not torch.allclose(expected, textbook)


class _WithBuffers(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("stats", torch.full((2,), 7.0))                      # persistent
        self.register_buffer("table", torch.arange(3.0), persistent=False)        # non-persistent


def test_persistent_buffers_left_to_checkpoint_by_default():
    m = _WithBuffers()
    buffers = _snapshot_buffers(m)
    m.stats.fill_(1.0)   # stands in for the value loaded from a checkpoint
    m.table.zero_()      # stands in for to_empty() garbage
    _restore_buffers(m, buffers, torch.device("cpu"), rank=0)
    assert torch.equal(m.stats, torch.ones(2))
    assert torch.equal(m.table, torch.arange(3.0))


def test_random_init_restores_persistent_buffers_too():
    m = _WithBuffers()
    buffers = _snapshot_buffers(m)
    m.stats.zero_()
    _restore_buffers(m, buffers, torch.device("cpu"), rank=0, include_persistent=True)
    assert torch.equal(m.stats, torch.full((2,), 7.0))


def test_buffer_without_snapshot_falls_back_to_zeros():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("mask", torch.empty(3, 3, device="meta"), persistent=False)

    m = M()
    buffers = _snapshot_buffers(m)
    assert buffers == {}  # meta buffers carry no value to snapshot
    _restore_buffers(m, buffers, torch.device("cpu"), rank=0)
    assert m.mask.device.type == "cpu"
    assert torch.all(m.mask == 0)


def test_remove_seed_dir_only_removes_own_run(tmp_path, monkeypatch):
    monkeypatch.setattr(distributed_utils, "dist_barrier", lambda local_rank: None)
    root = tmp_path / "pretrained_seed"
    mine = root / "fsdp" / "dcp_api" / "1000"
    other = root / "fsdp" / "dcp_api" / "2000"   # a concurrent run's seed
    for d in (mine, other):
        d.mkdir(parents=True)
        (d / "model_state_dict.pt").write_bytes(b"x")

    _remove_seed_dir(str(mine), str(root), local_rank=0, rank=0)
    assert not mine.exists()
    assert (other / "model_state_dict.pt").exists()


def test_remove_seed_dir_prunes_empty_parents(tmp_path, monkeypatch):
    monkeypatch.setattr(distributed_utils, "dist_barrier", lambda local_rank: None)
    root = tmp_path / "pretrained_seed"
    mine = root / "fsdp" / "dcp_api" / "1000"
    mine.mkdir(parents=True)
    _remove_seed_dir(str(mine), str(root), local_rank=0, rank=0)
    assert not root.exists()
    assert tmp_path.exists()  # never climbs above the seed root


def test_remove_seed_dir_non_zero_rank_keeps_files(tmp_path, monkeypatch):
    monkeypatch.setattr(distributed_utils, "dist_barrier", lambda local_rank: None)
    mine = tmp_path / "pretrained_seed" / "1000"
    mine.mkdir(parents=True)
    _remove_seed_dir(str(mine), str(tmp_path / "pretrained_seed"), local_rank=1, rank=1)
    assert mine.exists()
