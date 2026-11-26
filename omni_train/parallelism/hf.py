"""Hugging Face <-> 5D model: config mapping, sharded weight loading, export.

Supported architectures: Llama, Mistral, Qwen2 (and checkpoints that use the
same layout, e.g. TinyLlama, Llama 3.x, SmolLM). They share one decoder:
RMSNorm, rotary embeddings, grouped-query attention and a SwiGLU MLP.

Loading never builds the full model: every rank opens the safetensors files and
reads only the slices it owns (its pipeline stage's layers, its TP shard).
"""

from __future__ import annotations

import importlib
import json
import os

import torch
import torch.distributed as dist

from .model import ModelConfig, PipelineStage

SUPPORTED = {"llama": "Llama", "mistral": "Mistral", "qwen2": "Qwen2"}

# our local name (inside one layer) -> HF name (inside model.layers.<i>.)
_LAYER_MAP = {
    "attn_norm.weight": "input_layernorm.weight",
    "attn.q.weight": "self_attn.q_proj.weight",
    "attn.q.bias": "self_attn.q_proj.bias",
    "attn.k.weight": "self_attn.k_proj.weight",
    "attn.k.bias": "self_attn.k_proj.bias",
    "attn.v.weight": "self_attn.v_proj.weight",
    "attn.v.bias": "self_attn.v_proj.bias",
    "attn.o.weight": "self_attn.o_proj.weight",
    "attn.o_bias": "self_attn.o_proj.bias",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "ffn.gate.weight": "mlp.gate_proj.weight",
    "ffn.up.weight": "mlp.up_proj.weight",
    "ffn.down.weight": "mlp.down_proj.weight",
}


def hf_name(global_name: str) -> str:
    """Map a layout-independent parameter name (PipelineStage.global_name) to HF."""
    if global_name == "embed.weight":
        return "model.embed_tokens.weight"
    if global_name == "final_norm.weight":
        return "model.norm.weight"
    if global_name == "lm_head.weight":
        return "lm_head.weight"
    _, layer, rest = global_name.split(".", 2)
    return f"model.layers.{layer}.{_LAYER_MAP[rest]}"


# ---------------------------------------------------------------- download
def resolve_model_dir(model_name: str, rank: int) -> str:
    """Local folder with config + safetensors. Rank 0 downloads first."""
    if os.path.isdir(model_name):
        return model_name
    from huggingface_hub import snapshot_download

    def fetch():
        return snapshot_download(
            model_name, token=os.getenv("HF_TOKEN"),
            allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.model", "*.txt", "*.tiktoken"])

    if rank == 0:
        path = fetch()
    dist.barrier()
    if rank != 0:
        path = fetch()  # cache hit
    return path


# ------------------------------------------------------------------ config
def config_from_hf(model_dir: str, max_length: int) -> ModelConfig:
    from transformers import AutoConfig

    hf = AutoConfig.from_pretrained(model_dir)
    mt = hf.model_type
    if mt not in SUPPORTED:
        raise ValueError(
            f"strategy: 5d supports Hugging Face models of type {sorted(SUPPORTED)}; "
            f"'{model_dir}' is '{mt}'. Use strategy: fsdp or ddp for other architectures.")
    if getattr(hf, "mlp_bias", False):
        raise ValueError("strategy: 5d does not support MLP biases (config.mlp_bias=true)")
    window = getattr(hf, "sliding_window", None)
    uses_window = window is not None and (mt == "mistral" or getattr(hf, "use_sliding_window", False))
    if uses_window and max_length > window:
        raise ValueError(f"max_length {max_length} > sliding_window {window}: sliding-window "
                         "attention is not implemented in strategy: 5d")

    # RoPE frequencies straight from HF's own rotary class: handles rope_theta and
    # every rope scaling type (llama3, linear, yarn, ...) without re-deriving them.
    module = importlib.import_module(f"transformers.models.{mt}.modeling_{mt}")
    rotary = getattr(module, f"{SUPPORTED[mt]}RotaryEmbedding")(config=hf)
    rope_type = getattr(rotary, "rope_type", "default")
    if rope_type == "dynamic":
        raise ValueError("dynamic RoPE scaling is not supported in strategy: 5d")

    attn_bias = bool(getattr(hf, "attention_bias", False))
    return ModelConfig(
        vocab_size=hf.vocab_size,
        hidden=hf.hidden_size,
        n_layers=hf.num_hidden_layers,
        n_heads=hf.num_attention_heads,
        n_kv_heads=getattr(hf, "num_key_value_heads", None) or hf.num_attention_heads,
        head_dim=getattr(hf, "head_dim", None) or hf.hidden_size // hf.num_attention_heads,
        ffn_hidden=hf.intermediate_size,
        inv_freq=rotary.inv_freq.tolist(),
        attention_scaling=float(getattr(rotary, "attention_scaling", 1.0)),
        norm_eps=hf.rms_norm_eps,
        qkv_bias=(mt == "qwen2") or attn_bias,
        o_bias=attn_bias and mt != "qwen2",
        tie_embeddings=bool(getattr(hf, "tie_word_embeddings", False)),
        hf_model_type=mt,
    )


# ----------------------------------------------------------------- loading
class _SafetensorsIndex:
    def __init__(self, model_dir: str):
        from safetensors import safe_open
        self._open = safe_open
        index = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(index):
            with open(index) as f:
                weight_map = json.load(f)["weight_map"]
        elif os.path.exists(os.path.join(model_dir, "model.safetensors")):
            with safe_open(os.path.join(model_dir, "model.safetensors"), "pt") as f:
                weight_map = {k: "model.safetensors" for k in f.keys()}
        else:
            raise FileNotFoundError(f"no safetensors weights in {model_dir} (.bin checkpoints are not supported)")
        self.dir, self.weight_map, self.handles = model_dir, weight_map, {}

    def __contains__(self, name):
        return name in self.weight_map

    def get_slice(self, name):
        file = self.weight_map[name]
        if file not in self.handles:
            self.handles[file] = self._open(os.path.join(self.dir, file), "pt")
        return self.handles[file].get_slice(name)


def _read_shard(sl, shape, param, pc, vocab_rows: int | None):
    """Read this rank's part of an HF tensor. vocab_rows: true vocab size when the
    parameter is a (possibly padded) vocab-split table."""
    full_shape = list(sl.get_shape())
    if not getattr(param, "tp_sharded", False):
        return sl[:]
    dim = param.tp_dim
    n = param.shape[dim]                  # rows/cols this rank owns
    start = pc.tp_rank * n
    stop = start + n
    if vocab_rows is not None:            # padded rows past the real vocab stay zero
        out = torch.zeros(tuple(param.shape), dtype=torch.float32)
        real_stop = min(stop, vocab_rows)
        if real_stop > start:
            out[: real_stop - start] = sl[start:real_stop].float()
        return out
    if dim == 0:
        return sl[start:stop]
    return sl[:, start:stop]


@torch.no_grad()
def load_hf_weights(stage: PipelineStage, model_dir: str) -> None:
    pc, cfg = stage.pc, stage.cfg
    index = _SafetensorsIndex(model_dir)
    for local, param in stage.named_parameters():
        name = hf_name(stage.global_name(local))
        if name == "lm_head.weight" and cfg.tie_embeddings:
            name = "model.embed_tokens.weight"   # tied copy on the last PP stage
        if name not in index:
            raise KeyError(f"{name} not found in {model_dir}")
        vocab_rows = cfg.vocab_size if local in ("embed.weight", "lm_head.weight") else None
        value = _read_shard(index.get_slice(name), param.shape, param, pc, vocab_rows)
        if tuple(value.shape) != tuple(param.shape):
            raise ValueError(f"{name}: checkpoint shard {tuple(value.shape)} != model {tuple(param.shape)}")
        param.copy_(value.float())


# ------------------------------------------------------------------ export
@torch.no_grad()
def export_hf(stage: PipelineStage, model_dir: str, out_dir: str, tokenizer=None) -> None:
    """Write a normal HF checkpoint (one safetensors file per pipeline stage).

    Collective over the TP group of DP/CP replica 0; every rank must call it."""
    from safetensors.torch import save_file

    pc, cfg = stage.pc, stage.cfg
    writer = pc.dp_rank == 0 and pc.cp_rank == 0
    tensors = {}
    if writer:
        for local, param in stage.named_parameters():
            if getattr(param, "tied_copy", False):
                continue                       # tied: HF keeps only embed_tokens
            if cfg.tie_embeddings and local == "lm_head.weight":
                continue                       # pp=1: same parameter as the embedding
            t = param.detach()
            if getattr(param, "tp_sharded", False) and pc.tp_size > 1:
                parts = [torch.empty_like(t) for _ in range(pc.tp_size)]
                dist.all_gather(parts, t.contiguous(), group=pc.tp_group)
                t = torch.cat(parts, dim=param.tp_dim)
            if local in ("embed.weight", "lm_head.weight"):
                t = t[: cfg.vocab_size]
            tensors[hf_name(stage.global_name(local))] = t.to(torch.bfloat16).cpu().contiguous()

    if writer and pc.tp_rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        fname = f"model-{pc.pp_rank + 1:05d}-of-{pc.pp_size:05d}.safetensors"
        save_file(tensors, os.path.join(out_dir, fname), metadata={"format": "pt"})
        index_part = {k: fname for k in tensors}
    else:
        index_part = {}

    parts = [None] * dist.get_world_size()
    dist.all_gather_object(parts, index_part)
    if dist.get_rank() == 0:
        from transformers import AutoConfig
        weight_map = {k: v for part in parts for k, v in part.items()}
        with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {}, "weight_map": dict(sorted(weight_map.items()))}, f, indent=2)
        hf_cfg = AutoConfig.from_pretrained(model_dir)
        hf_cfg.dtype = "bfloat16"
        hf_cfg.save_pretrained(out_dir)
        if tokenizer is not None:
            tokenizer.save_pretrained(out_dir)
    dist.barrier()
