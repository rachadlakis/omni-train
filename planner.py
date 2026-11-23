"""Capacity planner: does a training run need distributed training, and which layout?

    python planner.py --model meta-llama/Llama-3.1-8B --gpus 8 --gpu H100 --seq 4096
    python planner.py --config configs/llm_dpo_lora_single_gpu.yaml --gpus 1 --gpu-mem 6

`plan(model, workload, cluster)` estimates the peak memory of one GPU for every strategy
the trainer supports (solo, DDP, FSDP, and 5d layouts of TP x PP x CP x DP), then
recommends the simplest one that fits, with a throughput estimate, the minimum GPU count
for each strategy and concrete fixes when nothing fits.

The memory model follows how this repo trains (see CLAUDE.md, "Precision"):
  * trainable weights, gradients and AdamW moments are fp32: 16 bytes per trainable param;
  * a frozen LoRA base is bf16 outside FSDP (mixed precision) and fp32 under FSDP;
  * QLoRA stores linear weights in 4 or 8 bits, embeddings stay bf16;
  * activations are bf16 under mixed precision, fp32 otherwise; attention uses SDPA
    (no seq^2 score matrix); gradient checkpointing keeps one tensor per layer;
  * preference runs (DPO/SimPO/ORPO) forward chosen + rejected: 2 x batch_size rows;
  * GRPO samples num_generations completions per prompt: batch_size x G rows, and its
    peak is the larger of sampling (KV cache) and training; sampling time is added to
    the step time (decoding is memory-bandwidth bound);
  * the loss upcasts logits to fp32, which dominates for large vocabularies.
It is an estimate (typically within ~15% of the measured peak), not a guarantee.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass, field, replace

GB = 1024 ** 3
USABLE_FRACTION = 0.92      # leave room for the allocator and other processes
CUDA_CONTEXT_GB = 0.8       # CUDA context + cuBLAS workspace
COMM_BUFFERS_GB = 0.5       # NCCL buffers when a process group exists
FRAGMENTATION = 1.10        # caching-allocator fragmentation on dynamic memory

PREFERENCE_ALGOS = {"dpo", "simpo", "orpo"}
FIVE_D_HF_ARCHS = {"llama", "mistral", "qwen2"}
GATED_ARCHS = {"llama", "mistral", "mixtral", "qwen2", "qwen2_moe", "qwen3", "qwen3_moe",
               "gemma", "gemma2", "gemma3", "gemma3_text", "phi3", "olmo", "olmo2", "granite",
               "cohere", "deepseek_v2", "deepseek_v3", "smollm3"}

# name -> (display, memory GB, dense bf16 tensor TFLOPS, memory bandwidth GB/s)
GPU_CATALOG = {
    "b200":    ("NVIDIA B200 192GB", 192, 2250, 8000),
    "h200":    ("NVIDIA H200 141GB", 141, 989, 4800),
    "h100":    ("NVIDIA H100 80GB", 80, 989, 3350),
    "gh200":   ("NVIDIA GH200 96GB", 96, 989, 4000),
    "a100-80": ("NVIDIA A100 80GB", 80, 312, 2039),
    "a100":    ("NVIDIA A100 40GB", 40, 312, 1555),
    "l40s":    ("NVIDIA L40S 48GB", 48, 362, 864),
    "a6000":   ("NVIDIA RTX A6000 48GB", 48, 155, 768),
    "a10g":    ("NVIDIA A10G 24GB", 24, 125, 600),
    "l4":      ("NVIDIA L4 24GB", 24, 121, 300),
    "4090":    ("NVIDIA RTX 4090 24GB", 24, 165, 1008),
    "3090":    ("NVIDIA RTX 3090 24GB", 24, 71, 936),
    "v100":    ("NVIDIA V100 32GB", 32, 125, 900),
    "t4":      ("NVIDIA T4 16GB", 16, 65, 320),
    "4050":    ("NVIDIA RTX 4050 Laptop 6GB", 6, 25, 192),
    "mi300x":  ("AMD MI300X 192GB", 192, 1307, 5300),
}


def match_gpu(name: str) -> tuple[str, float, float, float] | None:
    """Map a device name ("NVIDIA H100 80GB HBM3", "a100-sxm4-40gb", "h100") to the catalog."""
    n = (name or "").lower().replace(" ", "")
    if not n:
        return None
    if "a100" in n:
        return GPU_CATALOG["a100-80" if "80" in n else "a100"]
    for key in GPU_CATALOG:
        if key.replace("-", "") in n:
            return GPU_CATALOG[key]
    return None


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    name: str
    layers: int
    hidden: int
    heads: int
    kv_heads: int
    ffn: int
    vocab: int
    gated_mlp: bool = True
    tied_embeddings: bool = False
    arch: str = ""               # HF model_type (llama, qwen2, ...) or "custom"
    head_dim: int | None = None
    num_experts: int = 0
    top_k: int = 0
    learned_positions: int = 0   # learned position-embedding rows (OPT/GPT-2/BERT)
    params_override: int | None = None
    source: str = "estimate"     # preset | hf-config | custom | estimate

    @property
    def hd(self) -> int:
        return self.head_dim or self.hidden // max(1, self.heads)

    @property
    def embed_params(self) -> int:
        return self.vocab * self.hidden * (1 if self.tied_embeddings else 2) + self.learned_positions * self.hidden

    @property
    def attn_params(self) -> int:
        return 2 * self.hidden * self.heads * self.hd + 2 * self.hidden * self.kv_heads * self.hd

    @property
    def mlp_params(self) -> int:
        per_expert = self.hidden * self.ffn * (3 if self.gated_mlp else 2)
        return per_expert * max(1, self.num_experts) + (self.hidden * self.num_experts if self.num_experts else 0)

    @property
    def layer_params(self) -> int:
        return self.attn_params + self.mlp_params + 2 * self.hidden

    @property
    def params(self) -> int:
        if self.params_override:
            return self.params_override
        return self.embed_params + self.layers * self.layer_params + self.hidden

    @property
    def active_params(self) -> int:
        """Params touched per token (MoE routes each token through top_k experts)."""
        if not self.num_experts:
            return self.params
        per_expert = self.hidden * self.ffn * (3 if self.gated_mlp else 2)
        idle = per_expert * (self.num_experts - max(1, self.top_k)) * self.layers
        return self.params - idle

    @property
    def five_d_ready(self) -> bool:
        return self.arch in FIVE_D_HF_ARCHS or self.arch == "custom"

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(params=self.params, active_params=self.active_params, five_d_ready=self.five_d_ready)
        return d


@dataclass
class Workload:
    method: str = "full"          # full | lora | qlora
    algorithm: str = "pretrain"   # pretrain | sft | dpo | simpo | orpo
    model_type: str = "llm"
    batch_size: int = 1           # per GPU (pairs per GPU for preference runs)
    seq_len: int = 512
    grad_ckpt: bool = True
    mixed_precision: bool = True
    lora_r: int = 16
    quant_bits: int = 4
    micro_batches: int = 1        # 5d gradient accumulation
    pp_schedule: str = "1f1b"
    num_samples: int | None = None
    epochs: int = 1
    num_generations: int = 4      # grpo: completions per prompt
    completion_len: int | None = None   # grpo: tokens generated per completion

    @property
    def rows(self) -> int:
        if self.algorithm == "grpo":
            return self.batch_size * max(2, self.num_generations)
        return self.batch_size * (2 if self.algorithm in PREFERENCE_ALGOS else 1)


@dataclass
class Cluster:
    gpu_mem_gb: float = 80.0
    gpus_per_node: int = 1
    nodes: int = 1
    gpu_name: str = "H100"
    tflops: float | None = None   # dense bf16; looked up from gpu_name when None

    @property
    def total_gpus(self) -> int:
        return max(1, self.gpus_per_node) * max(1, self.nodes)

    @property
    def peak_tflops(self) -> float:
        if self.tflops:
            return float(self.tflops)
        hit = match_gpu(self.gpu_name)
        return float(hit[2]) if hit else 100.0

    @property
    def bandwidth_gbs(self) -> float:
        hit = match_gpu(self.gpu_name)
        return float(hit[3]) if hit else 1000.0

    @property
    def usable_bytes(self) -> float:
        return self.gpu_mem_gb * GB * USABLE_FRACTION


@dataclass
class Layout:
    strategy: str                 # solo | ddp | fsdp | 5d
    gpus: int
    dp: int = 1
    tp: int = 1
    pp: int = 1
    cp: int = 1
    ep: int = 1
    sp: bool = False
    zero1: bool = False
    micro_batches: int = 1

    def label(self) -> str:
        if self.strategy != "5d":
            return f"{self.strategy.upper()} x{self.gpus}" if self.gpus > 1 else self.strategy.upper()
        parts = [f"DP{self.dp}"]
        for name in ("tp", "pp", "cp", "ep"):
            if getattr(self, name) > 1:
                parts.append(f"{name.upper()}{getattr(self, name)}")
        return " x ".join(parts) + (" +SP" if self.sp else "") + (" +ZeRO-1" if self.zero1 else "")

    def parallelism_block(self) -> dict | None:
        if self.strategy != "5d":
            return None
        return {"dp": self.dp, "tp": self.tp, "sp": self.sp, "cp": self.cp, "pp": self.pp,
                "pp_schedule": "1f1b", "micro_batches": self.micro_batches, "ep": self.ep,
                "zero1": self.zero1}


# ---------------------------------------------------------------------------
# Model specs: presets, HF configs, custom transformer, name heuristic
# ---------------------------------------------------------------------------

def _llama(name, L, h, heads, kv, ffn, V, tied, arch="llama", **kw):
    return ModelSpec(name, L, h, heads, kv, ffn, V, True, tied, arch, source="preset", **kw)


PRESETS: dict[str, ModelSpec] = {s.name.lower(): s for s in [
    ModelSpec("facebook/opt-125m", 12, 768, 12, 12, 3072, 50272, False, True, "opt", learned_positions=2050, source="preset"),
    ModelSpec("facebook/opt-1.3b", 24, 2048, 32, 32, 8192, 50272, False, True, "opt", learned_positions=2050, source="preset"),
    ModelSpec("gpt2", 12, 768, 12, 12, 3072, 50257, False, True, "gpt2", learned_positions=1024, source="preset"),
    _llama("HuggingFaceTB/SmolLM2-135M", 30, 576, 9, 3, 1536, 49152, True),
    _llama("HuggingFaceTB/SmolLM2-360M", 32, 960, 15, 5, 2560, 49152, True),
    _llama("HuggingFaceTB/SmolLM2-1.7B", 24, 2048, 32, 32, 8192, 49152, True),
    _llama("Qwen/Qwen2.5-0.5B", 24, 896, 14, 2, 4864, 151936, True, "qwen2"),
    _llama("Qwen/Qwen2.5-0.5B-Instruct", 24, 896, 14, 2, 4864, 151936, True, "qwen2"),
    _llama("Qwen/Qwen2.5-1.5B-Instruct", 28, 1536, 12, 2, 8960, 151936, True, "qwen2"),
    _llama("Qwen/Qwen2.5-3B-Instruct", 36, 2048, 16, 2, 11008, 151936, True, "qwen2"),
    _llama("Qwen/Qwen2.5-7B-Instruct", 28, 3584, 28, 4, 18944, 152064, False, "qwen2"),
    _llama("Qwen/Qwen2.5-14B-Instruct", 48, 5120, 40, 8, 13824, 152064, False, "qwen2"),
    _llama("Qwen/Qwen2.5-32B-Instruct", 64, 5120, 40, 8, 27648, 152064, False, "qwen2"),
    _llama("Qwen/Qwen2.5-72B-Instruct", 80, 8192, 64, 8, 29568, 152064, False, "qwen2"),
    _llama("meta-llama/Llama-3.2-1B", 16, 2048, 32, 8, 8192, 128256, True),
    _llama("meta-llama/Llama-3.2-1B-Instruct", 16, 2048, 32, 8, 8192, 128256, True),
    _llama("meta-llama/Llama-3.2-3B-Instruct", 28, 3072, 24, 8, 8192, 128256, True),
    _llama("meta-llama/Llama-3.1-8B", 32, 4096, 32, 8, 14336, 128256, False),
    _llama("meta-llama/Llama-3.1-8B-Instruct", 32, 4096, 32, 8, 14336, 128256, False),
    _llama("meta-llama/Llama-3.1-70B-Instruct", 80, 8192, 64, 8, 28672, 128256, False),
    _llama("meta-llama/Llama-3.1-405B", 126, 16384, 128, 8, 53248, 128256, False),
    _llama("mistralai/Mistral-7B-v0.3", 32, 4096, 32, 8, 14336, 32768, False, "mistral"),
    _llama("mistralai/Mistral-7B-Instruct-v0.3", 32, 4096, 32, 8, 14336, 32768, False, "mistral"),
    _llama("mistralai/Mixtral-8x7B-v0.1", 32, 4096, 32, 8, 14336, 32000, False, "mixtral", num_experts=8, top_k=2),
]}


def spec_from_hf_config(cfg: dict, name: str = "") -> ModelSpec:
    """Build a spec from a Hugging Face config.json dict (text_config for VLMs)."""
    if isinstance(cfg.get("text_config"), dict) and "hidden_size" not in cfg:
        inner = dict(cfg["text_config"])
        inner.setdefault("vocab_size", cfg.get("vocab_size"))
        return replace(spec_from_hf_config(inner, name), arch=str(cfg.get("model_type", "")))

    def pick(*keys, default=None):
        for k in keys:
            if cfg.get(k) is not None:
                return cfg[k]
        return default

    arch = str(cfg.get("model_type", "")).lower()
    h = int(pick("hidden_size", "n_embd", "d_model", "dim", default=768))
    L = int(pick("num_hidden_layers", "n_layer", "num_layers", "n_layers", default=12))
    heads = int(pick("num_attention_heads", "n_head", "num_heads", "n_heads", default=max(1, h // 64)))
    kv = int(pick("num_key_value_heads", "n_kv_heads", default=heads) or heads)
    ffn = int(pick("intermediate_size", "n_inner", "ffn_dim", "d_ff", "moe_intermediate_size", default=4 * h) or 4 * h)
    V = int(pick("vocab_size", default=32000))
    act = str(pick("hidden_act", "activation_function", default="")).lower()
    gated = arch in GATED_ARCHS or "glu" in act
    learned = int(pick("max_position_embeddings", "n_positions", default=0)) if arch in {"opt", "gpt2", "bert", "roberta", "distilbert", "gpt_neo"} else 0
    experts = int(pick("num_local_experts", "num_experts", "n_routed_experts", default=0) or 0)
    return ModelSpec(
        name=name or str(cfg.get("_name_or_path", "")), layers=L, hidden=h, heads=heads,
        kv_heads=kv, ffn=ffn, vocab=V, gated_mlp=gated,
        tied_embeddings=bool(cfg.get("tie_word_embeddings", True)), arch=arch,
        head_dim=cfg.get("head_dim"), num_experts=experts,
        top_k=int(pick("num_experts_per_tok", "top_k", default=2 if experts else 0) or 0),
        learned_positions=learned, source="hf-config",
    )


def spec_from_custom(args: dict) -> ModelSpec:
    """custom_transformer_args (model.py / parallelism/model.py)."""
    dim = int(args.get("dim", 512))
    heads = int(args.get("n_heads", 8))
    experts = int(args.get("num_experts", 0) or 0)
    llama_style = bool(experts or args.get("n_kv_heads") or args.get("ffn_hidden"))
    ffn = int(args.get("ffn_hidden") or (int(8 * dim / 3) if llama_style else 4 * dim))
    return ModelSpec(
        name="custom_transformer", layers=int(args.get("n_layers", 6)), hidden=dim, heads=heads,
        kv_heads=int(args.get("n_kv_heads") or heads), ffn=ffn,
        vocab=int(args.get("vocab_size", 8192)), gated_mlp=llama_style, tied_embeddings=False,
        arch="custom", num_experts=experts, top_k=int(args.get("top_k", 2)) if experts else 0,
        learned_positions=0 if llama_style else int(args.get("max_seq_len", 512)), source="custom",
    )


def params_from_name(name: str) -> int | None:
    """'Llama-3.1-8B' -> 8e9, 'opt-350m' -> 3.5e8; None when the name has no size."""
    n = (name or "").lower().split("/")[-1]
    m = re.search(r"(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)b", n)
    if m:                                    # Mixtral-style 8x7b: ~0.85 shared-weight discount
        return int(float(m.group(1)) * float(m.group(2)) * 0.83e9)
    m = re.search(r"(?<![\d.])(\d+(?:\.\d+)?)\s*([bm])(?![a-z])", n)
    if not m:
        return None
    return int(float(m.group(1)) * (1e9 if m.group(2) == "b" else 1e6))


def spec_from_params(params: int, name: str = "") -> ModelSpec:
    """Llama-shaped guess for a given size, used when no config is available."""
    table = [(2e8, 12, 768, 12), (6e8, 24, 1024, 16), (2e9, 24, 2048, 16), (4e9, 32, 3072, 24),
             (1e10, 32, 4096, 32), (2e10, 40, 5120, 40), (4e10, 64, 6144, 48), (1e11, 80, 8192, 64),
             (float("inf"), 126, 16384, 128)]
    for limit, L, h, heads in table:
        if params <= limit:
            break
    spec = ModelSpec(name or f"{params / 1e9:.1f}B model", L, h, heads, max(1, heads // 4),
                     int(h * 3.5), 128256 if params > 1e9 else 50272, True, params < 1e9, "",
                     params_override=int(params), source="estimate")
    return spec


def resolve_spec(model_name: str, model_type: str = "llm", custom_args: dict | None = None,
                 hf_config: dict | None = None) -> ModelSpec:
    if model_type == "custom_transformer":
        return spec_from_custom(custom_args or {})
    if hf_config:
        return spec_from_hf_config(hf_config, model_name)
    preset = PRESETS.get((model_name or "").lower())
    if preset:
        return preset
    guess = params_from_name(model_name)
    return spec_from_params(guess or 350_000_000, model_name)


# ---------------------------------------------------------------------------
# Memory model
# ---------------------------------------------------------------------------

def lora_trainable_params(m: ModelSpec, r: int) -> int:
    """LoRA on every linear layer (target_modules: all-linear): r * (in + out) each."""
    q = m.hidden + m.heads * m.hd
    kv = 2 * (m.hidden + m.kv_heads * m.hd)
    o = m.heads * m.hd + m.hidden
    mlp = (m.hidden + m.ffn) * (3 if m.gated_mlp else 2) * max(1, m.num_experts)
    return r * (q + kv + o + mlp) * m.layers


def _states(m: ModelSpec, w: Workload, lay: Layout) -> dict:
    """weights / gradients / optimizer bytes on one GPU."""
    P = m.params
    if w.method == "full":
        if lay.strategy == "5d":
            # worst pipeline stage: its share of the layers plus the embedding or LM head
            stage_layers = math.ceil(m.layers / lay.pp)
            emb_rows = m.vocab * m.hidden * (2 if lay.pp == 1 and not m.tied_embeddings else 1)
            local = (stage_layers * m.layer_params + emb_rows) / lay.tp
            if m.num_experts and lay.ep > 1:
                expert = m.hidden * m.ffn * (3 if m.gated_mlp else 2)
                local -= stage_layers * expert * m.num_experts * (1 - 1 / lay.ep) / lay.tp
            opt_div = lay.dp if lay.zero1 else 1
            return {"weights": 4 * local, "gradients": 4 * local, "optimizer": 8 * local / opt_div}
        if lay.strategy == "fsdp":
            n = lay.gpus
            # all-gathered working set: current layer + prefetched layer, bf16 params + fp32 grads
            gathered = 2 * m.layer_params * ((2 if w.mixed_precision else 4) + 4)
            return {"weights": 4 * P / n + gathered, "gradients": 4 * P / n, "optimizer": 8 * P / n}
        ddp_bucket = 4 * P if lay.strategy == "ddp" else 0
        return {"weights": 4 * P, "gradients": 4 * P + ddp_bucket, "optimizer": 8 * P}

    T = lora_trainable_params(m, w.lora_r)
    if w.method == "qlora":
        emb = m.vocab * m.hidden * (1 if m.tied_embeddings else 2)
        qbytes = 0.53 if w.quant_bits == 4 else 1.02   # NF4 + double-quant constants / int8 + scales
        base = (P - emb) * qbytes + emb * 2
        return {"weights": base + 4 * T, "gradients": 4 * T, "optimizer": 8 * T}
    if lay.strategy == "fsdp":
        n = lay.gpus
        gathered = 2 * m.layer_params * 4
        return {"weights": (4 * P + 4 * T) / n + gathered, "gradients": 4 * T / n, "optimizer": 8 * T / n}
    base = 2 if w.mixed_precision else 4
    ddp_bucket = 4 * T if lay.strategy == "ddp" else 0
    return {"weights": base * P + 4 * T, "gradients": 4 * T + ddp_bucket, "optimizer": 8 * T}


def _activations(m: ModelSpec, w: Workload, lay: Layout) -> dict:
    """Activation + logits bytes on one GPU at the peak of the backward pass."""
    act_bytes = 2 if w.mixed_precision else 4
    seq = w.seq_len / lay.cp
    rows = w.rows
    tokens = rows * seq
    ffn_eff = m.ffn * (max(1, m.top_k) if m.num_experts else 1)
    split = 4 * m.hidden + (3 if m.gated_mlp else 2) * ffn_eff      # qkv/attn-out/MLP: TP-split
    replicated = 4 * m.hidden                                        # norms + residuals: split by SP only
    tp_split = split / lay.tp
    tp_repl = replicated / lay.tp if lay.sp else replicated
    per_layer_full = tokens * act_bytes * (tp_split + tp_repl)
    stage_layers = math.ceil(m.layers / lay.pp)
    if w.grad_ckpt:
        boundary = tokens * act_bytes * m.hidden / (lay.tp if lay.sp else 1)
        per_micro = stage_layers * boundary + per_layer_full
    else:
        per_micro = stage_layers * per_layer_full
    if lay.pp > 1:
        inflight = lay.micro_batches if w.pp_schedule == "afab" else min(lay.pp, lay.micro_batches)
    else:
        inflight = 1
    acts = per_micro * inflight

    # logits: bf16/fp32 output + the fp32 upcast and its gradient inside the loss
    if w.algorithm in PREFERENCE_ALGOS:
        logits = tokens * m.vocab * act_bytes + seq * m.vocab * 8     # sequence_logps: one row in fp32
    elif w.algorithm == "grpo":
        comp = min(seq, w.completion_len or seq // 2)                # logits_to_keep: completion only
        logits = rows * comp * m.vocab * act_bytes + comp * m.vocab * 8
        # sampling phase: KV cache of every sequence + one step of fp32 logits; the peak is
        # whichever phase is larger, reported under "activations"
        kv = rows * seq * m.layers * 2 * m.kv_heads * m.hd * act_bytes
        sampling = kv + rows * m.vocab * 4 * 2
        if sampling > acts + logits:
            acts += sampling - (acts + logits)
    else:
        logits = tokens * m.vocab * (act_bytes + 8)
    return {"activations": acts, "logits": logits}


def memory_per_gpu(m: ModelSpec, w: Workload, lay: Layout) -> dict:
    states = _states(m, w, lay)
    acts = _activations(m, w, lay)
    overhead = CUDA_CONTEXT_GB * GB + (COMM_BUFFERS_GB * GB if lay.gpus > 1 else 0)
    dynamic = (acts["activations"] + acts["logits"]) * FRAGMENTATION
    total = sum(states.values()) + dynamic + overhead
    out = {k: v / GB for k, v in {**states, **acts}.items()}
    out["overhead"] = overhead / GB
    out["total"] = total / GB
    return {k: round(v, 3) for k, v in out.items()}


# ---------------------------------------------------------------------------
# Throughput
# ---------------------------------------------------------------------------

MFU = {"solo": 0.40, "ddp": 0.37, "fsdp": 0.34, "5d": 0.32}


def throughput(m: ModelSpec, w: Workload, lay: Layout, c: Cluster) -> dict:
    P = m.active_params
    if w.method == "full":
        dense = 6 * P
    else:   # frozen base: forward + activation-grad backward, adapters' weight grads are ~free
        dense = 4 * P
    if w.grad_ckpt:
        dense += 2 * P                                  # one extra forward
    attn = 12 * m.layers * m.hidden * w.seq_len         # SDPA fwd+bwd, causal ~ half of 2*s per token
    flops_per_token = dense + attn * 0.5
    tokens_per_step = w.rows * w.seq_len * lay.dp * (lay.micro_batches if lay.strategy == "5d" else 1)
    mfu = MFU[lay.strategy]
    if lay.strategy == "5d":
        mfu *= lay.micro_batches / (lay.micro_batches + lay.pp - 1)   # pipeline bubble
        if lay.tp > c.gpus_per_node:
            mfu *= 0.5                                                 # TP over the network
    if c.nodes > 1 and lay.strategy in {"ddp", "fsdp"}:
        mfu *= 0.85
    if w.method == "qlora":
        mfu *= 0.6                                                     # dequantize kernels
    flops_per_s = c.peak_tflops * 1e12 * mfu * lay.gpus
    step_s = tokens_per_step * flops_per_token / flops_per_s
    out = {"tokens_per_step": int(tokens_per_step), "step_seconds": round(step_s, 4),
           "tokens_per_second": round(tokens_per_step / step_s) if step_s else None, "mfu": round(mfu, 3)}
    if w.algorithm == "grpo":
        # decoding reads the (local) weights once per generated token: bandwidth bound
        weight_bytes = P * (0.53 if w.method == "qlora" else 2)
        comp = w.completion_len or w.seq_len // 2
        gen_s = comp * weight_bytes / (c.bandwidth_gbs * 1e9 * 0.6)
        if lay.strategy == "fsdp":
            gen_s *= 2          # every decode step all-gathers the full weights layer by layer
        elif lay.strategy == "5d":
            gen_s /= lay.tp     # (5d does not run grpo; kept for completeness)
        out["generation_seconds"] = round(gen_s, 4)
        step_s += gen_s
        out["step_seconds"] = round(step_s, 4)
        out["tokens_per_second"] = round(tokens_per_step / step_s) if step_s else None
    if w.num_samples:
        samples_per_step = w.batch_size * lay.dp * (lay.micro_batches if lay.strategy == "5d" else 1)  # prompts for grpo
        steps = math.ceil(w.num_samples / max(1, samples_per_step)) * max(1, w.epochs)
        out["total_steps"] = steps
        out["total_seconds"] = round(steps * step_s, 1)
    return out


# ---------------------------------------------------------------------------
# Layout search
# ---------------------------------------------------------------------------

def _divisors(n: int, cap: int | None = None) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0 and (cap is None or d <= cap)]


def five_d_eligibility(m: ModelSpec, w: Workload) -> str | None:
    """None if strategy: 5d can run this, else why not."""
    if w.model_type not in {"llm", "custom_transformer"}:
        return f"5d supports llm and custom_transformer, not {w.model_type}"
    if not m.five_d_ready:
        return f"5d loads Hugging Face llama / mistral / qwen2 only (this model is '{m.arch or 'unknown'}')"
    if w.method != "full":
        return "5d does not support LoRA / quantization yet"
    if w.algorithm != "pretrain":
        return f"5d does not support post-training ({w.algorithm}) yet"
    return None


def five_d_layouts(m: ModelSpec, w: Workload, gpus: int, gpus_per_node: int) -> list[Layout]:
    """Every valid tp x pp x cp x dp factorisation of `gpus` that the trainer accepts."""
    out = []
    for tp in _divisors(gpus, cap=min(gpus_per_node, 8)):
        if m.heads % tp or m.kv_heads % tp or m.ffn % tp:
            continue
        for pp in _divisors(gpus // tp, cap=m.layers):
            for cp in _divisors(gpus // (tp * pp), cap=16):
                seq_ok = w.seq_len % cp == 0 and (tp == 1 or (w.seq_len // cp) % tp == 0)
                if not seq_ok or (cp > 1 and w.seq_len // cp < 128):
                    continue
                dp = gpus // (tp * pp * cp)
                ep = 1
                if m.num_experts:
                    ep = max(e for e in _divisors(dp) if m.num_experts % e == 0)
                micro = max(w.micro_batches, 2 * pp if pp > 1 else 1)
                out.append(Layout("5d", gpus, dp=dp, tp=tp, pp=pp, cp=cp, ep=ep, sp=tp > 1,
                                  zero1=dp > 1, micro_batches=micro))
    return out


def _rank_5d(lay: Layout, w: Workload) -> tuple:
    # prefer: fewer model-parallel ranks (more DP), TP before PP before CP, CP for long context
    cp_pref = 0 if (lay.cp > 1 and w.seq_len >= 16384) else lay.cp
    return (lay.tp * lay.pp * lay.cp, cp_pref, lay.pp, lay.tp)


@dataclass
class Option:
    layout: Layout
    memory: dict
    fits: bool
    perf: dict
    note: str = ""

    def to_dict(self) -> dict:
        return {"strategy": self.layout.strategy, "label": self.layout.label(), "gpus": self.layout.gpus,
                "parallelism": self.layout.parallelism_block(), "memory_gb": self.memory,
                "per_gpu_gb": self.memory["total"], "fits": self.fits, "perf": self.perf, "note": self.note}


def _option(m, w, c, lay, note="") -> Option:
    mem = memory_per_gpu(m, w, lay)
    return Option(lay, mem, mem["total"] * GB <= c.usable_bytes, throughput(m, w, lay, c), note)


def _best_5d(m, w, c, gpus) -> Option | None:
    fitting = [_option(m, w, c, lay) for lay in five_d_layouts(m, w, gpus, c.gpus_per_node)]
    fitting = [o for o in fitting if o.fits]
    return min(fitting, key=lambda o: _rank_5d(o.layout, w)) if fitting else None


def min_gpus_fsdp(m, w, c) -> int | None:
    if w.method == "qlora":
        return None
    for n in [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]:
        lay = Layout("fsdp" if n > 1 else "solo", n, dp=n)
        if memory_per_gpu(m, w, lay)["total"] * GB <= c.usable_bytes:
            return n
    return None


def min_gpus_5d(m, w, c) -> tuple[int, Layout] | None:
    if five_d_eligibility(m, w):
        return None
    per_node = max(1, c.gpus_per_node)
    counts = sorted({n for n in [1, 2, 4, 8] if n <= per_node} | {per_node * k for k in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128)})
    for n in counts:
        best = _best_5d(m, w, c, n)
        if best:
            return n, best.layout
    return None


def plan(m: ModelSpec, w: Workload, c: Cluster) -> dict:
    """Recommend a strategy + layout for this model, workload and cluster."""
    N = c.total_gpus
    options: list[Option] = []
    notes: list[str] = []
    suggestions: list[dict] = []

    solo = _option(m, w, c, Layout("solo", 1, dp=1))
    options.append(solo)
    ddp = fsdp = five = None
    if N > 1:
        ddp = _option(m, w, c, Layout("ddp", N, dp=N))
        options.append(ddp)
        if w.method != "qlora":
            fsdp = _option(m, w, c, Layout("fsdp", N, dp=N))
            options.append(fsdp)
        else:
            notes.append("FSDP cannot shard bitsandbytes-quantized layers: QLoRA runs on solo or DDP.")
    why_not_5d = five_d_eligibility(m, w)
    if why_not_5d is None and N > 1:
        five = _best_5d(m, w, c, N)
        if five:
            options.append(five)

    if solo.fits:
        if N > 1 and ddp and ddp.fits:
            verdict, rec = "distributed_optional", ddp
            headline = "Fits on one GPU. Distributed training is optional."
            speed = solo.perf["step_seconds"] * N / ddp.perf["step_seconds"] if ddp.perf["step_seconds"] else N
            summary = (f"One GPU needs ~{solo.memory['total']:.1f} GB of {c.gpu_mem_gb:g} GB. "
                       f"DDP across {N} GPUs keeps a full copy on each and gives ~{speed:.1f}x the throughput.")
        else:
            verdict, rec = "single_gpu", solo
            headline = "Fits on one GPU. No distributed training needed."
            summary = f"Peak memory is ~{solo.memory['total']:.1f} GB of {c.gpu_mem_gb:g} GB available."
    elif fsdp and fsdp.fits:
        verdict, rec = "distributed_required", fsdp
        headline = f"Needs distributed training: shard with FSDP across {N} GPUs."
        summary = (f"One GPU would need ~{solo.memory['total']:.1f} GB (limit {c.gpu_mem_gb:g} GB). FSDP shards "
                   f"weights, gradients and AdamW states, bringing it to ~{fsdp.memory['total']:.1f} GB per GPU.")
    elif five:
        verdict, rec = "distributed_required", five
        headline = f"Needs 5D parallelism: {five.layout.label()}."
        why = ("activations dominate at this sequence length" if (fsdp and fsdp.memory["activations"] + fsdp.memory["logits"]) > (fsdp.memory["total"] / 2)
               else "even fully sharded model states do not fit")
        summary = (f"FSDP on {N} GPUs still needs ~{(fsdp or ddp or solo).memory['total']:.1f} GB per GPU ({why}). "
                   f"Splitting layers (TP/PP) and the sequence (CP) brings it to ~{five.memory['total']:.1f} GB.")
    else:
        verdict = "infeasible"
        rec = min((o for o in options), key=lambda o: o.memory["total"])
        headline = f"Does not fit on {N} x {c.gpu_mem_gb:g} GB GPUs as configured."
        summary = (f"The best option ({rec.layout.label()}) still needs ~{rec.memory['total']:.1f} GB per GPU. "
                   "See the suggestions below.")

    # How many GPUs each strategy would need
    mg_fsdp = min_gpus_fsdp(m, w, c)
    mg_5d = min_gpus_5d(m, w, c)
    min_gpus = {"single": 1 if solo.fits else None, "fsdp": mg_fsdp,
                "5d": mg_5d[0] if mg_5d else None,
                "5d_layout": mg_5d[1].label() if mg_5d else None}

    # What-if suggestions when it does not fit on this cluster
    if verdict == "infeasible":
        steps = []   # (label, changes), cheapest first; also applied cumulatively below
        if not w.grad_ckpt:
            steps.append(("Enable gradient checkpointing", {"grad_ckpt": True}))
        if w.batch_size > 1:
            steps.append(("Lower batch size to 1 (more steps, same data)", {"batch_size": 1}))
        if w.method == "full" and w.model_type in {"llm", "seq2seq", "encoder", "vlm", "vision"}:
            steps.append(("Switch to LoRA (r=16)", {"method": "lora", "lora_r": 16}))
        if w.method in {"full", "lora"} and w.model_type in {"llm", "vlm", "seq2seq"}:
            steps.append(("Switch to QLoRA (4-bit base)", {"method": "qlora", "quant_bits": 4}))
        if w.seq_len > 1024:
            steps.append((f"Shorten max_length to {w.seq_len // 2}", {"seq_len": w.seq_len // 2}))
        for label, changes in steps:
            sub = plan_core_fit(m, replace(w, **changes), c)
            if sub:
                suggestions.append({"change": label, "result": sub["label"], "per_gpu_gb": sub["per_gpu_gb"],
                                    "workload": changes})
        if not suggestions and len(steps) > 1:
            combined, labels = {}, []
            for label, changes in steps:
                combined.update(changes)
                labels.append(label.split(" (")[0])
                sub = plan_core_fit(m, replace(w, **combined), c)
                if sub:
                    suggestions.append({"change": " + ".join(labels), "result": sub["label"],
                                        "per_gpu_gb": sub["per_gpu_gb"], "workload": dict(combined)})
                    break
        if mg_fsdp and mg_fsdp > N:
            suggestions.append({"change": f"Use {mg_fsdp} GPUs", "result": f"FSDP x{mg_fsdp}", "per_gpu_gb": None, "workload": {}})
        if mg_5d and mg_5d[0] > N and (not mg_fsdp or mg_5d[0] < mg_fsdp):
            suggestions.append({"change": f"Use {mg_5d[0]} GPUs", "result": mg_5d[1].label(), "per_gpu_gb": None, "workload": {}})
        more = [n for n in (mg_fsdp, mg_5d[0] if mg_5d else None) if n and n > N]
        if N == 1 and more:
            headline = f"Needs distributed training: at least {min(more)} GPUs of {c.gpu_mem_gb:g} GB."

    if why_not_5d and N > 1 and verdict in {"distributed_required", "infeasible"}:
        notes.append(f"5D layouts not considered: {why_not_5d}.")
    if m.source == "estimate":
        notes.append("Architecture inferred from the model name; load its config for a sharper estimate.")
    if w.algorithm in PREFERENCE_ALGOS:
        notes.append(f"{w.algorithm.upper()} forwards chosen + rejected together: {w.rows} sequences per GPU per step.")
    if w.algorithm == "dpo":
        notes.append("DPO reference log-probs are precomputed once, so no second model is held in memory.")
    if w.algorithm == "grpo":
        notes.append(f"GRPO samples {max(2, w.num_generations)} completions per prompt ({w.rows} sequences per GPU "
                     "per step); the step time includes sampling, and the KL reference (LoRA) reuses the "
                     "model with adapters disabled.")
    if rec.layout.strategy == "5d" and rec.layout.tp > 1:
        notes.append("Keep TP inside a node (NVLink); PP and DP can cross nodes.")

    return {
        "verdict": verdict,
        "distributed_needed": verdict in {"distributed_required", "infeasible"} and not solo.fits,
        "headline": headline,
        "summary": summary,
        "recommended": rec.to_dict(),
        "options": [o.to_dict() for o in options],
        "min_gpus": min_gpus,
        "suggestions": suggestions,
        "notes": notes,
        "model": m.to_dict(),
        "workload": asdict(w),
        "cluster": {**asdict(c), "total_gpus": N, "usable_gb": round(c.usable_bytes / GB, 2),
                    "peak_tflops": c.peak_tflops},
    }


def plan_core_fit(m: ModelSpec, w: Workload, c: Cluster) -> dict | None:
    """Smallest-effort option that fits on this cluster, or None."""
    N = c.total_gpus
    candidates = [Layout("solo", 1, dp=1)]
    if N > 1 and w.method != "qlora":
        candidates.append(Layout("fsdp", N, dp=N))
    for lay in candidates:
        o = _option(m, w, c, lay)
        if o.fits:
            return {"label": lay.label(), "per_gpu_gb": o.memory["total"]}
    if N > 1 and not five_d_eligibility(m, w):
        best = _best_5d(m, w, c, N)
        if best:
            return {"label": best.layout.label(), "per_gpu_gb": best.memory["total"]}
    return None


def needs_distributed(model: ModelSpec, workload: Workload, cluster: Cluster) -> bool:
    """True when the run cannot fit on a single GPU of this cluster."""
    return not _option(model, workload, cluster, Layout("solo", 1, dp=1)).fits


# ---------------------------------------------------------------------------
# From a training config (config.yaml schema)
# ---------------------------------------------------------------------------

def workload_from_config(cfg: dict) -> Workload:
    tr = cfg.get("training") or {}
    peft = cfg.get("peft") or {}
    quant = cfg.get("quantization") or {}
    algo = cfg.get("algorithm") or {}
    par = cfg.get("parallelism") or {}
    dist = cfg.get("dist_parameters") or {}
    model_type = str(cfg.get("model_type", "llm")).lower()
    method = "full"
    if str(peft.get("enabled", False)).lower() in {"true", "1"}:
        method = "qlora" if (peft.get("type") == "qlora" or str(quant.get("enabled", False)).lower() in {"true", "1"}) else "lora"
    seq = int(tr.get("max_length", 512))
    if model_type == "custom_transformer":
        seq = int((cfg.get("custom_transformer_args") or {}).get("max_seq_len", seq))
    algo_type = str(algo.get("type", "pretrain")).lower()
    completion = None
    if algo_type == "grpo":
        prompt_len = int(algo.get("max_prompt_length") or seq // 2)
        completion = int(algo.get("max_completion_length") or seq - prompt_len)
        seq = min(seq, prompt_len + completion)
    return Workload(
        method=method, algorithm=str(algo.get("type", "pretrain")).lower(), model_type=model_type,
        batch_size=int(tr.get("batch_size", 1)), seq_len=seq,
        grad_ckpt=str(tr.get("gradient_checkpointing", False)).lower() in {"true", "1"},
        mixed_precision=str(dist.get("mixed_precision", True)).lower() in {"true", "1"},
        lora_r=int(peft.get("r", 16)), quant_bits=int(quant.get("bits", 4)),
        micro_batches=int(par.get("micro_batches", 1)), pp_schedule=str(par.get("pp_schedule", "1f1b")),
        epochs=int(tr.get("epochs", 1)),
        num_generations=int(algo.get("num_generations", 4)), completion_len=completion,
    )


def evaluate_layout(m: ModelSpec, w: Workload, c: Cluster, strategy: str, gpus: int,
                    parallelism: dict | None = None) -> dict:
    """Memory + throughput of one explicit layout (what the user picked)."""
    if strategy == "5d":
        p = parallelism or {}
        tp, pp, cp = int(p.get("tp", 1)), int(p.get("pp", 1)), int(p.get("cp", 1))
        dp = p.get("dp", "auto")
        dp = gpus // max(1, tp * pp * cp) if dp in (None, "auto") else int(dp)
        lay = Layout("5d", gpus, dp=max(1, dp), tp=tp, pp=pp, cp=cp, ep=int(p.get("ep", 1)),
                     sp=str(p.get("sp", False)).lower() in {"true", "1"},
                     zero1=str(p.get("zero1", False)).lower() in {"true", "1"},
                     micro_batches=int(p.get("micro_batches", 1)))
    else:
        n = 1 if strategy == "solo" else max(1, gpus)
        lay = Layout(strategy, n, dp=n)
    return _option(m, w, c, lay).to_dict()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_plan(p: dict) -> None:
    m, rec = p["model"], p["recommended"]
    print(f"\nModel     {m['name']}  ({m['params'] / 1e9:.2f}B params, {m['layers']} layers, "
          f"hidden {m['hidden']}, source: {m['source']})")
    c = p["cluster"]
    print(f"Cluster   {c['nodes']} node(s) x {c['gpus_per_node']} GPU(s), {c['gpu_mem_gb']:g} GB each "
          f"({c['usable_gb']:g} GB usable)")
    w = p["workload"]
    print(f"Workload  {w['algorithm']} / {w['method']}, batch {w['batch_size']} x seq {w['seq_len']}, "
          f"grad ckpt {'on' if w['grad_ckpt'] else 'off'}\n")
    print(f"=> {p['headline']}\n   {p['summary']}\n")
    print(f"{'option':<34}{'per GPU':>10}{'fits':>6}{'tok/s':>12}")
    for o in p["options"]:
        tps = o["perf"].get("tokens_per_second")
        print(f"{o['label']:<34}{o['per_gpu_gb']:>8.1f}GB{('yes' if o['fits'] else 'no'):>6}{(f'{tps:,}' if tps else '-'):>12}")
    mem = rec["memory_gb"]
    print(f"\nRecommended: {rec['label']}  ->  " + ", ".join(
        f"{k} {mem[k]:.1f}" for k in ("weights", "gradients", "optimizer", "activations", "logits", "overhead")) + " GB")
    if rec["parallelism"]:
        print("parallelism:", json.dumps(rec["parallelism"]))
    mg = p["min_gpus"]
    layout_5d = f" ({mg['5d_layout']})" if mg["5d_layout"] else ""
    print(f"Min GPUs: single={'yes' if mg['single'] else 'no'}, fsdp={mg['fsdp'] or '-'}, "
          f"5d={mg['5d'] or '-'}{layout_5d}")
    for s in p["suggestions"]:
        extra = f" -> {s['result']}" + (f" at {s['per_gpu_gb']:.1f} GB/GPU" if s.get("per_gpu_gb") else "")
        print(f"  * {s['change']}{extra}")
    for n in p["notes"]:
        print(f"  note: {n}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Does this run need distributed training? Which layout?")
    ap.add_argument("--config", help="training YAML (config.yaml schema)")
    ap.add_argument("--model", help="HF model id (presets are known offline)")
    ap.add_argument("--params", type=float, help="parameter count in billions, if the model is unknown")
    ap.add_argument("--method", choices=["full", "lora", "qlora"])
    ap.add_argument("--algorithm", choices=["pretrain", "sft", "dpo", "simpo", "orpo", "grpo"])
    ap.add_argument("--generations", type=int, help="grpo: completions per prompt (default 4)")
    ap.add_argument("--batch", type=int)
    ap.add_argument("--seq", type=int)
    ap.add_argument("--no-ckpt", action="store_true", help="gradient checkpointing off")
    ap.add_argument("--gpus", type=int, default=1, help="GPUs per node")
    ap.add_argument("--nodes", type=int, default=1)
    ap.add_argument("--gpu", default="H100", help="GPU type (h100, a100-80, 4090, ...)")
    ap.add_argument("--gpu-mem", type=float, help="GB per GPU (default: from --gpu)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    cfg = {}
    if a.config:
        import yaml
        with open(a.config) as f:
            cfg = yaml.safe_load(f) or {}
    w = workload_from_config(cfg) if cfg else Workload()
    for attr, val in (("method", a.method), ("algorithm", a.algorithm), ("batch_size", a.batch), ("seq_len", a.seq)):
        if val is not None:
            setattr(w, attr, val)
    if a.no_ckpt:
        w.grad_ckpt = False
    if a.generations:
        w.num_generations = a.generations
    name = a.model or cfg.get("model_name") or ""
    if a.params:
        m = spec_from_params(int(a.params * 1e9), name)
    else:
        m = resolve_spec(name, w.model_type, cfg.get("custom_transformer_args"))
    gpu = match_gpu(a.gpu)
    mem = a.gpu_mem or (gpu[1] if gpu else 80)
    c = Cluster(gpu_mem_gb=mem, gpus_per_node=a.gpus, nodes=a.nodes, gpu_name=a.gpu)
    p = plan(m, w, c)
    print(json.dumps(p, indent=2)) if a.json else _print_plan(p)


if __name__ == "__main__":
    main()
