"""A Llama-style transformer that supports all five parallel dimensions.

The same module serves two cases:
  * model_type: custom_transformer  -> weights built from fixed seeds
  * model_type: llm (Llama / Mistral / Qwen2 from Hugging Face) -> empty shards,
    then filled by parallelism.hf.load_hf_weights. The math matches HF exactly
    (rotate_half RoPE with HF's own inv_freq, GQA, optional q/k/v/o biases,
    tied embeddings), so loaded weights give the same logits as the HF model.

Shapes of the residual stream on one rank:
    [micro_batch, seq / cp / (tp if SP else 1), hidden]

One pipeline stage builds only its own layers. The first stage also owns the
embedding, the last stage owns the final norm and the LM head.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .context_parallel import ring_attention
from .expert_parallel import MoELayer
from .process_groups import ParallelContext
from .tensor_parallel import (ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding,
                              enter_tp_region, exit_tp_region, gather_logits)
from .utils import mark, split_layers


@dataclass
class ModelConfig:
    vocab_size: int = 256
    hidden: int = 128
    n_layers: int = 4
    n_heads: int = 4
    n_kv_heads: int | None = None  # None = n_heads (no GQA)
    head_dim: int | None = None    # None = hidden // n_heads
    ffn_hidden: int = 344          # about 8/3 * hidden, like Llama
    num_experts: int = 0           # 0 = dense model, >0 = every layer is MoE
    top_k: int = 2
    rope_theta: float = 10000.0
    inv_freq: list[float] | None = None  # from HF (handles rope scaling); None = from rope_theta
    attention_scaling: float = 1.0       # HF rope post-scaling of cos/sin
    norm_eps: float = 1e-5
    qkv_bias: bool = False         # Qwen2, Llama(attention_bias)
    o_bias: bool = False           # Llama(attention_bias)
    tie_embeddings: bool = False
    hf_model_type: str | None = None  # llama / mistral / qwen2 when loaded from HF
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        self.n_kv_heads = self.n_kv_heads or self.n_heads
        self.head_dim = self.head_dim or self.hidden // self.n_heads


def padded_vocab(vocab_size: int, tp: int) -> int:
    """Vocab rounded up so the embedding / LM head split evenly over TP."""
    return math.ceil(vocab_size / tp) * tp


def check_divisibility(cfg: ModelConfig, pc: ParallelContext) -> None:
    """Fail early, with the reason, instead of deep inside a layer."""
    errors = []
    if cfg.n_heads % pc.tp_size:
        errors.append(f"attention heads ({cfg.n_heads}) must be divisible by tp ({pc.tp_size})")
    if cfg.n_kv_heads % pc.tp_size:
        errors.append(f"key/value heads ({cfg.n_kv_heads}) must be divisible by tp ({pc.tp_size})")
    if cfg.n_heads % cfg.n_kv_heads:
        errors.append(f"heads ({cfg.n_heads}) must be a multiple of kv heads ({cfg.n_kv_heads})")
    if cfg.ffn_hidden % pc.tp_size:
        errors.append(f"ffn hidden size ({cfg.ffn_hidden}) must be divisible by tp ({pc.tp_size})")
    if cfg.n_layers < pc.pp_size:
        errors.append(f"need at least one layer per pipeline stage ({cfg.n_layers} layers, pp={pc.pp_size})")
    if cfg.num_experts and cfg.num_experts % pc.ep_size:
        errors.append(f"num_experts ({cfg.num_experts}) must be divisible by ep ({pc.ep_size})")
    if pc.ep_size > 1 and not cfg.num_experts:
        errors.append("ep > 1 needs a Mixture-of-Experts model (num_experts > 0)")
    if errors:
        raise ValueError("5d layout does not fit the model:\n  - " + "\n  - ".join(errors))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x_f = x.float()
        x_f = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x_f.to(x.dtype)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def rope_cos_sin(inv_freq: torch.Tensor, positions: torch.Tensor, scaling: float):
    """HF-convention RoPE tables for the given absolute positions: [s, head_dim]."""
    angles = positions.float()[:, None] * inv_freq.float()[None, :]
    emb = torch.cat((angles, angles), dim=-1)
    return emb.cos() * scaling, emb.sin() * scaling


def apply_rope(x, cos, sin):
    """x: [b, heads, s, d]; cos/sin: [s, d]."""
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return x * cos + rotate_half(x) * sin


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, pc: ParallelContext, seed: int | None):
        super().__init__()
        self.pc, self.cfg = pc, cfg
        hd = cfg.head_dim
        self.local_heads = cfg.n_heads // pc.tp_size
        self.local_kv = cfg.n_kv_heads // pc.tp_size
        s = (lambda k: None) if seed is None else (lambda k: seed + k)
        # TP: heads are split between ranks (column-parallel Q/K/V, row-parallel output).
        self.q = ColumnParallelLinear(cfg.hidden, cfg.n_heads * hd, pc, s(1), bias=cfg.qkv_bias)
        self.k = ColumnParallelLinear(cfg.hidden, cfg.n_kv_heads * hd, pc, s(2), bias=cfg.qkv_bias)
        self.v = ColumnParallelLinear(cfg.hidden, cfg.n_kv_heads * hd, pc, s(3), bias=cfg.qkv_bias)
        self.o = RowParallelLinear(cfg.n_heads * hd, cfg.hidden, pc, s(4))
        # Added once, after the TP sum. Not TP-split, so with SP it is TP-summed later.
        self.o_bias = nn.Parameter(torch.zeros(cfg.hidden)) if cfg.o_bias else None

    def forward(self, x, cos, sin):
        h = enter_tp_region(x, self.pc)                     # [b, s_cp, hidden]
        b, s, _ = h.shape
        hd = self.cfg.head_dim
        q = self.q(h).view(b, s, self.local_heads, hd).transpose(1, 2)
        k = self.k(h).view(b, s, self.local_kv, hd).transpose(1, 2)
        v = self.v(h).view(b, s, self.local_kv, hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        if self.pc.cp_size > 1:
            out = ring_attention(q, k, v, self.pc)
        else:
            group = self.local_heads // self.local_kv
            if group > 1:
                k = k.repeat_interleave(group, dim=1)
                v = v.repeat_interleave(group, dim=1)
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(b, s, self.local_heads * hd)
        out = exit_tp_region(self.o(out), self.pc)
        return out + self.o_bias if self.o_bias is not None else out


class MLP(nn.Module):
    """SwiGLU MLP: column-parallel gate/up, row-parallel down -> one all-reduce."""

    def __init__(self, cfg: ModelConfig, pc: ParallelContext, seed: int | None):
        super().__init__()
        self.pc = pc
        s = (lambda k: None) if seed is None else (lambda k: seed + k)
        self.gate = ColumnParallelLinear(cfg.hidden, cfg.ffn_hidden, pc, s(5))
        self.up = ColumnParallelLinear(cfg.hidden, cfg.ffn_hidden, pc, s(6))
        self.down = RowParallelLinear(cfg.ffn_hidden, cfg.hidden, pc, s(7))

    def forward(self, x):
        h = enter_tp_region(x, self.pc)
        return exit_tp_region(self.down(F.silu(self.gate(h)) * self.up(h)), self.pc)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, pc: ParallelContext, layer_id: int, seeded: bool):
        super().__init__()
        seed = 10_000 * (layer_id + 1) if seeded else None
        self.attn_norm = RMSNorm(cfg.hidden, cfg.norm_eps)
        self.attn = Attention(cfg, pc, seed)
        self.ffn_norm = RMSNorm(cfg.hidden, cfg.norm_eps)
        if cfg.num_experts > 0:
            # EP: the MoE layer works on this rank's local tokens, outside the TP region.
            self.ffn = MoELayer(cfg.hidden, cfg.ffn_hidden, cfg.num_experts, cfg.top_k, pc,
                                (seed or 0) + 100)
        else:
            self.ffn = MLP(cfg, pc, seed)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        return x + self.ffn(self.ffn_norm(x))


class PipelineStage(nn.Module):
    """The part of the model that lives on this pipeline stage.

    seeded=True builds weights from fixed seeds (custom model); seeded=False
    allocates empty shards for parallelism.hf.load_hf_weights to fill.
    """

    def __init__(self, cfg: ModelConfig, pc: ParallelContext, seeded: bool = True,
                 gradient_checkpointing: bool = False):
        super().__init__()
        check_divisibility(cfg, pc)
        self.cfg, self.pc = cfg, pc
        self.gradient_checkpointing = gradient_checkpointing
        self.vocab_padded = padded_vocab(cfg.vocab_size, pc.tp_size)
        self.layer_ids = split_layers(cfg.n_layers, pc.pp_size, pc.pp_rank)
        if pc.is_first_stage:
            self.embed = VocabParallelEmbedding(cfg.vocab_size, cfg.hidden, pc,
                                                seed=1 if seeded else None,
                                                padded_vocab=self.vocab_padded)
        self.layers = nn.ModuleList(Block(cfg, pc, i, seeded) for i in self.layer_ids)
        if pc.is_last_stage:
            self.final_norm = RMSNorm(cfg.hidden, cfg.norm_eps)
            self.lm_head = ColumnParallelLinear(cfg.hidden, cfg.vocab_size, pc,
                                                seed=2 if seeded else None,
                                                pad_out_to=self.vocab_padded)
        if cfg.tie_embeddings:
            if pc.pp_size == 1:
                self.lm_head.weight = self.embed.weight  # one shared parameter
            else:
                # Two copies on different stages, kept equal by summing their grads.
                param = self.embed.weight if pc.is_first_stage else (
                    self.lm_head.weight if pc.is_last_stage else None)
                if param is not None:
                    mark(param, tied_embedding=True)
                if pc.is_last_stage:
                    mark(self.lm_head.weight, tied_copy=True)  # counted once in grad norm / export

        # HF-convention RoPE frequencies (fp32, not a parameter, never saved).
        if cfg.inv_freq is not None:
            inv_freq = torch.tensor(cfg.inv_freq, dtype=torch.float32)
        else:
            hd = cfg.head_dim
            inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, hd, 2).float() / hd))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # With SP, any weight that is NOT split by TP only sees this rank's
        # tokens, so its gradient must be summed over the TP group later.
        for p in self.parameters():
            p.sp_replicated = pc.sequence_parallel and not getattr(p, "tp_sharded", False)

    def _positions(self, device):
        """Absolute positions of this rank's tokens (CP chunk; whole chunk, since
        attention runs after the SP all-gather)."""
        s = self.local_seq
        return torch.arange(s, device=device) + self.pc.cp_rank * s

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """First stage: x = token ids [b, s_cp]. Other stages: hidden states.
        Last stage returns logits [b, s_cp, vocab]; others return hidden states."""
        if self.pc.is_first_stage:
            self.local_seq = x.shape[1]
            x = self.embed(x)
        else:
            self.local_seq = x.shape[1] * (self.pc.tp_size if self.pc.sequence_parallel else 1)
        cos, sin = rope_cos_sin(self.inv_freq, self._positions(x.device), self.cfg.attention_scaling)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(layer, x, cos, sin, use_reentrant=False)
            else:
                x = layer(x, cos, sin)
        if self.pc.is_last_stage:
            x = enter_tp_region(self.final_norm(x), self.pc)
            x = gather_logits(self.lm_head(x), self.pc)[..., :self.cfg.vocab_size]
        return x

    # ------------------------------------------------------------ naming
    def global_name(self, local_name: str) -> str:
        """Layout-independent parameter name: local layer index -> global layer id,
        local expert index -> global expert id."""
        parts = local_name.split(".")
        if parts[0] == "layers":
            parts[1] = str(self.layer_ids[int(parts[1])])
        if "experts" in parts:
            i = parts.index("experts") + 1
            num_local = self.cfg.num_experts // self.pc.ep_size
            parts[i] = str(self.pc.ep_rank * num_local + int(parts[i]))
        return ".".join(parts)
