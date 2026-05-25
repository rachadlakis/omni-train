"""Tests for the custom Transformer model — no GPU or HF download required."""
import pytest
import torch

from model import ModelArgs, Transformer, TransformerBlock
from distributed_utils import get_model_layers


TINY = ModelArgs(n_layers=2, vocab_size=16, max_seq_len=8, dim=16, n_heads=4, dropout_p=0.0)


# ------------------------------------------------------------------
# Construction
# ------------------------------------------------------------------

def test_transformer_builds():
    model = Transformer(TINY)
    assert len(model.layers) == TINY.n_layers


def test_transformer_param_count_is_positive():
    model = Transformer(TINY)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 0


# ------------------------------------------------------------------
# Forward pass (CPU, eval mode)
# ------------------------------------------------------------------

def test_transformer_forward_shape():
    model = Transformer(TINY).eval()
    x = torch.randint(0, TINY.vocab_size, (2, TINY.max_seq_len))  # (batch=2, seq=8)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (2, TINY.max_seq_len, TINY.vocab_size)


def test_transformer_forward_no_nan():
    model = Transformer(TINY).eval()
    x = torch.randint(0, TINY.vocab_size, (1, TINY.max_seq_len))
    with torch.no_grad():
        logits = model(x)
    assert not torch.isnan(logits).any()


def test_transformer_forward_shorter_sequence():
    """Model should handle seq_len < max_seq_len."""
    model = Transformer(TINY).eval()
    x = torch.randint(0, TINY.vocab_size, (1, 4))  # shorter than max_seq_len=8
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (1, 4, TINY.vocab_size)


# ------------------------------------------------------------------
# get_model_layers on real Transformer
# ------------------------------------------------------------------

def test_get_model_layers_finds_transformer_layers():
    model = Transformer(TINY)
    stacks = get_model_layers(model)
    assert len(stacks) == 1
    layers, _ = stacks[0]
    assert len(layers) == TINY.n_layers


# ------------------------------------------------------------------
# TransformerBlock in isolation
# ------------------------------------------------------------------

def test_transformer_block_forward_shape():
    block = TransformerBlock(TINY).eval()
    x = torch.randn(1, TINY.max_seq_len, TINY.dim)
    with torch.no_grad():
        out = block(x)
    assert out.shape == x.shape


def test_transformer_block_residual_preserves_shape():
    """Output shape must equal input shape (residual stream)."""
    block = TransformerBlock(TINY).eval()
    x = torch.randn(3, 5, TINY.dim)
    with torch.no_grad():
        out = block(x)
    assert out.shape == (3, 5, TINY.dim)
