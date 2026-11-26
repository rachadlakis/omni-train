"""Rough VRAM and time estimates printed at startup and shown by the console (see also planner.py)."""

from omni_train.console import print_banner_on_rank_0, print_on_rank_0


def _resolve_hf_config(model):
    """Best-effort fetch of the underlying HF config from a (possibly DDP/FSDP/PEFT) model.

    For VLMs the transformer dims live on a nested text_config, so we unwrap that too.
    Returns None for architectures we can't introspect (e.g. the toy custom_transformer).
    """
    for obj in (model, getattr(model, "module", None), getattr(model, "base_model", None)):
        cfg = getattr(obj, "config", None) if obj is not None else None
        if cfg is not None:
            return getattr(cfg, "text_config", cfg)
    return None


def _transformer_dims(cfg):
    """Pull (num_layers, hidden_dim) from an HF config across common naming schemes."""
    def first(names):
        for n in names:
            v = getattr(cfg, n, None)
            if v:
                return int(v)
        return None
    layers = first(("num_hidden_layers", "n_layer", "num_layers", "n_layers"))
    hidden = first(("hidden_size", "n_embd", "d_model", "hidden_dim"))
    return layers, hidden


# UI model_type labels → the set accepted by estimate_training_vram (below).
# Mirrors the _type_map in ui/app.py so terminal and UI feed the formula identically.
_VRAM_TYPE_MAP = {
    "llm": "llm", "seq2seq": "seq2seq", "vlm": "vlm",
    "encoder": "encoder", "embedding": "encoder",
    "vision": "vision", "cnn": "vision",
    "yolo": "yolo", "detection": "yolo",
}


def print_training_vram_estimate(rank, model, args, world_size):
    """Print the GPU memory needed to train `model`, kept numerically in sync with the UI.

    This does NOT reimplement the memory math — it feeds *measured* inputs (exact parameter
    count from numel(), real layer/hidden dims read from the model config) into the very same
    `estimate_training_vram` the web UI calls. So the conventions are identical by
    construction: decimal GB (÷1e9), AdamW = fp32 master + two fp32 moments (12 B/param),
    a 0.85 allocator-efficiency factor on the total, and the same ~10×(batch×seq×hidden)
    per-layer activation heuristic (halved under gradient checkpointing).

    Any residual gap vs. the UI is only because the UI *infers* the parameter count from the
    model name while we *measure* it here — i.e. this figure is the more accurate of the two.

    Activations do not shard under FSDP, so the per-GPU line divides only weights+grads+optim
    by `world_size` and adds activations back un-sharded — matching ui/app.py's fsdp-check.

    Falls back to a measured params+grads+optim figure (activations excluded) for
    architectures the formula doesn't cover (e.g. custom_transformer) or when the model
    config doesn't expose transformer dims.
    """
    if rank != 0:
        return

    # ---- Measured inputs ----
    num_params = 0
    param_dtype_bits = 16  # bf16/fp16 storage → 16, fp32 → 32
    for p in model.parameters():
        num_params += p.numel()
        if p.is_floating_point():
            param_dtype_bits = 32 if p.element_size() >= 4 else 16

    layers, hidden = (None, None)
    cfg = _resolve_hf_config(model)
    if cfg is not None:
        layers, hidden = _transformer_dims(cfg)

    seq_len = int(getattr(args, "max_length", 0) or 0)
    batch = int(getattr(args, "batch_size", 0) or 0)
    grad_ckpt = bool(getattr(args, "gradient_checkpointing", False))
    vram_type = _VRAM_TYPE_MAP.get(str(getattr(args, "model_type", "")).lower())
    quant_bits = int(getattr(args, "quantization_bits", 0)) if getattr(args, "quantization_enabled", False) else 0

    print_banner_on_rank_0(rank, "ESTIMATED VRAM TO TRAIN")

    # ---- Shared formula (same one the UI calls) ----
    vram = None
    if vram_type and layers and hidden and seq_len and batch:
        try:
            vram = estimate_training_vram(
                model_type=vram_type,
                num_params=num_params,
                param_dtype_bits=param_dtype_bits,
                batch_size=batch,
                seq_len=seq_len,
                num_layers=layers,
                hidden_dim=hidden,
                activation_checkpointing=grad_ckpt,
                peft_enabled=bool(getattr(args, "peft_enabled", False)),
                peft_r=int(getattr(args, "peft_r", 16)),
                quantization_bits=quant_bits,
            )
        except Exception as e:
            print_on_rank_0(rank, f"Falling back to static estimate ({e})", "⚠️")

    if vram is not None:
        ckpt = " | grad_ckpt" if grad_ckpt else ""
        print_on_rank_0(rank, f"Parameters        : {vram['weights_gb']:7.2f} GB", "🧮")
        print_on_rank_0(rank, f"Gradients         : {vram['gradients_gb']:7.2f} GB")
        print_on_rank_0(rank, f"Optimizer (AdamW) : {vram['optimizer_gb']:7.2f} GB")
        print_on_rank_0(rank, f"Activations (est.): {vram['activations_gb']:7.2f} GB | batch={batch} seq={seq_len}{ckpt}", "🔥")
        # total_gb already includes the 0.85 allocator-efficiency factor (UI convention).
        print_on_rank_0(rank, f"Est. peak total   : {vram['total_gb']:7.2f} GB (incl. 0.85 alloc factor)", "📈")
        if args.strategy == "fsdp" and world_size > 1:
            shardable = vram["weights_gb"] + vram["gradients_gb"] + vram["optimizer_gb"]
            per_gpu = (shardable / world_size + vram["activations_gb"]) * 0.85
            print_on_rank_0(rank, f"~Per-GPU under FSDP across {world_size} GPUs: {per_gpu:7.2f} GB")
        return

    # ---- Fallback: measured static buffers only (architecture not covered by the formula) ----
    GB = 1e9  # decimal GB, matching the UI
    weights_gb = num_params * (param_dtype_bits / 8) / GB
    trainable_elems = sum(p.numel() for p in model.parameters() if p.requires_grad)
    grad_gb = trainable_elems * (param_dtype_bits / 8) / GB
    optim_gb = trainable_elems * 12 / GB  # fp32 master + two fp32 moments, UI convention
    static_gb = weights_gb + grad_gb + optim_gb
    print_on_rank_0(rank, f"Parameters        : {weights_gb:7.2f} GB", "🧮")
    print_on_rank_0(rank, f"Gradients         : {grad_gb:7.2f} GB")
    print_on_rank_0(rank, f"Optimizer (AdamW) : {optim_gb:7.2f} GB")
    print_on_rank_0(rank, "Activations (est.): n/a — unknown architecture; total excludes activations", "🔥")
    print_on_rank_0(rank, f"Est. total (excl. activations): {static_gb:7.2f} GB", "📈")


def estimate_training_vram(
    model_type: str,               # llm, seq2seq, vision, yolo, vlm, encoder
    num_params: int,               # total model parameters
    param_dtype_bits: int,         # 16 for bfloat16/fp16, 32 for fp32
    batch_size: int,
    seq_len: int,
    num_layers: int,
    hidden_dim: int,
    activation_checkpointing: bool = False,
    peft_enabled: bool = False,    # LoRA/QLoRA: optimizer covers adapters only
    peft_r: int = 16,              # LoRA rank; used to estimate trainable param count
    quantization_bits: int = 0,    # 4 or 8 for quantized weights, 0 for full precision
) -> dict:
    """
    Estimates VRAM usage in GB for training with Adam optimizer.

    Key assumptions:
    - Quantized weights (4-bit/8-bit) use reduced byte-width for storage.
    - LoRA/PEFT: gradients + optimizer states only cover trainable adapter params,
      drastically reducing optimizer memory (from ~6-12× weights to near-zero for large models).
    - For mixed precision (param_dtype_bits=16): Adam keeps fp32 master weights + two fp32 moments
      = 12 bytes per param. Full fine-tune = 6× weight memory.
    - For fp32: Adam stores 3 fp32 states per param → 3× weight memory.
    - Transformer activation memory accounts for Q/K/V (3×), attention output (1×),
      FFN (4×), and layer norms/residuals (2×) ≈ 10 elements per hidden dimension per layer.
    - Activation checkpointing halves activation memory (recompute on backward).
    - For vision/YOLO, activation memory is a rough heuristic (~20% of weights).
    """
    valid_types = {"llm", "seq2seq", "vision", "yolo", "vlm", "encoder"}
    if model_type not in valid_types:
        raise ValueError(f"Unsupported model_type={model_type}. Use one of {valid_types}")

    quant_bits = int(quantization_bits) if quantization_bits else 0

    # ---- Weight storage (quantization reduces byte width) ----
    if quant_bits in {4, 8}:
        weight_bytes_per_param = quant_bits / 8   # 0.5 B for 4-bit, 1.0 B for 8-bit
    else:
        weight_bytes_per_param = param_dtype_bits / 8  # 2 B for fp16/bf16, 4 B for fp32
    weights_gb = num_params * weight_bytes_per_param / 1e9

    # ---- Trainable parameter estimate (for PEFT) ----
    if peft_enabled:
        # LoRA adds two adapter matrices (A: r×d, B: d×r) per target linear layer.
        # Typical all-linear targeting: q, k, v, o projections = 4 matrices per layer.
        # num_layers and hidden_dim are passed in directly — no approximation needed.
        trainable_params = min(num_params, 2 * peft_r * hidden_dim * num_layers * 4)
    else:
        trainable_params = num_params

    # ---- Gradient memory ----
    if peft_enabled:
        # Gradients only for adapter params (stored in fp16 training precision)
        gradients_gb = trainable_params * 2 / 1e9
    elif quant_bits in {4, 8}:
        # Quantized without PEFT: unsupported by build_args, but handle gracefully
        gradients_gb = 0.0
    else:
        # Full fine-tune: gradients same dtype as weights
        gradients_gb = num_params * (param_dtype_bits / 8) / 1e9

    # ---- Optimizer memory (Adam: weight + two moments in fp32) ----
    if peft_enabled:
        # Optimizer only for trainable adapter params (3 fp32 tensors each = 12B each)
        optimizer_gb = trainable_params * 12 / 1e9
    elif param_dtype_bits == 32:
        # fp32 weights (4B) + two fp32 Adam states (8B) = 12B/param → 3× weights_gb
        optimizer_gb = weights_gb * 3
    else:
        # fp16/bf16: optimizer keeps fp32 master copy + two fp32 moments = 12B/param
        optimizer_gb = num_params * 12 / 1e9

    # ---- Activation memory ----
    bytes_per_act = param_dtype_bits / 8  # activations stored in training precision

    if model_type in {"llm", "seq2seq", "encoder", "vlm"}:
        # Per transformer layer: Q/K/V (3×), attn output (1×), FFN intermediate (4×),
        # layer norms + residuals (2×) ≈ 10 elements per (batch × seq × hidden)
        act_bytes = batch_size * seq_len * hidden_dim * num_layers * 10 * bytes_per_act
    elif model_type in {"vision", "yolo"}:
        # CNN-style: rough heuristic ~20% of weight memory
        act_bytes = weights_gb * 0.2 * 1e9
    else:
        act_bytes = 0

    if activation_checkpointing:
        act_bytes /= 2.0   # ~2× memory saving from recompute

    activations_gb = act_bytes / 1e9
    total_gb_raw = weights_gb + gradients_gb + optimizer_gb + activations_gb

    # Apply efficiency factor: PyTorch's memory management (lazy allocation,
    # memory reuse, caching allocator) is typically ~15-20% more efficient
    # than worst-case theoretical estimates.
    efficiency_factor = 0.85
    total_gb = total_gb_raw * efficiency_factor

    return {
        "weights_gb":     round(weights_gb, 2),
        "gradients_gb":   round(gradients_gb, 2),
        "optimizer_gb":   round(optimizer_gb, 2),
        "activations_gb": round(activations_gb, 2),
        "total_gb":       round(total_gb, 2),
    }


def _match_gpu_tflops(gpu_name, tflops_map, default):
    """Look up a GPU in tflops_map by whole-word match, so a full device name like
    "NVIDIA RTX A4000" or "NVIDIA A100-SXM4-80GB" resolves to its short key, while
    "a2" does not match "A2000" and "l4" does not match "L40S"."""
    import re
    name = (gpu_name or "").lower()
    for key in sorted(tflops_map, key=len, reverse=True):
        if re.search(rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])", name):
            return tflops_map[key]
    return default


def estimate_training_time(
    num_params: int,
    steps_per_epoch: int,
    epochs: int,
    batch_size: int,
    seq_len: int = 256,            # actual sequence length (tokens per sample)
    num_gpus: int = 1,
    gpu_type: str = "unknown",
    strategy: str = "solo",
    peft_enabled: bool = False,
    peft_r: int = 16,
    gradient_checkpointing: bool = False,
    mfu: float = 0.25,            # model flop utilization (0.25 ≈ typical HF fine-tuning)
    extra_overhead: float = 1.0,
) -> dict:
    """
    Estimates training time based on FLOPs / effective throughput.

    Formula: total_flops = 6 * N * T * total_steps
    - 6× factor: 2× forward + 2× backward (activation gradients) + 2× backward (weight gradients)
    - N = num_params, T = seq_len (tokens per sample)

    LoRA reduces only the weight-gradient term (~1/3 of total FLOPs) because
    frozen weights still need a full forward pass and full activation-gradient backward.
    The remaining 2/3 (forward + activation-grad) are unchanged.

    MFU = 0.25 (25% model flop utilization) is a conservative default for typical
    HuggingFace fine-tuning without custom CUDA kernels. Well-tuned setups with
    FlashAttention can reach 0.35–0.45.
    """
    strategy = (strategy or "solo").strip().lower()

    # Peak TFLOPS per GPU by type (bf16/fp16 tensor core throughput)
    gpu_tflops_map = {
        "h100": 990, "a100": 312, "a10g": 125, "v100": 125,
        "l40s": 362, "l40": 181, "l4": 120, "t4": 65, "a6000": 155, "a5000": 110,
        "a4000": 77, "a2": 31, "4090": 83, "3090": 36, "3080": 30,
        "3070": 20, "3060": 13, "2080": 14, "1080": 11,
    }
    gpu_tflops = _match_gpu_tflops(gpu_type, gpu_tflops_map, default=40)  # 40 TFLOPS: conservative unknown GPU default

    # Distributed communication overhead (reduces effective GPU utilization)
    overhead = {"solo": 0.0, "ddp": 0.08, "fsdp": 0.18}.get(strategy, 0.1)

    # Total training FLOPs: 6 * params * tokens_per_sample * total_steps
    total_tokens = batch_size * steps_per_epoch * epochs * seq_len
    total_flops = 6.0 * num_params * total_tokens

    # LoRA: weight gradients only for adapters (tiny fraction), but forward +
    # activation-gradient backward still run on the full frozen model.
    #   full FLOPs   = forward(1/3) + act_grad(1/3) + weight_grad(1/3)
    #   LoRA FLOPs   = forward(1/3) + act_grad(1/3) + weight_grad(f/3)
    #                = (2 + f) / 3   where f = trainable_fraction
    if peft_enabled:
        # Lookup-table approach for (num_layers, hidden_dim) — same breakpoints as app.py
        if num_params <= 200_000_000:
            _nl, _hd = 12, 768
        elif num_params <= 500_000_000:
            _nl, _hd = 24, 1024
        elif num_params <= 2_000_000_000:
            _nl, _hd = 24, 2048
        elif num_params <= 9_000_000_000:
            _nl, _hd = 32, 4096
        elif num_params <= 20_000_000_000:
            _nl, _hd = 40, 5120
        else:
            _nl, _hd = 80, 8192
        # LoRA: 2 adapter matrices × 4 attention projection matrices per layer
        trainable_params = min(num_params, 2 * peft_r * _hd * _nl * 4)
        trainable_fraction = trainable_params / max(1, num_params)
        lora_multiplier = (2.0 + trainable_fraction) / 3.0
        total_flops *= lora_multiplier

    # Gradient checkpointing adds ~33% recompute cost during the backward pass
    if gradient_checkpointing:
        total_flops *= 1.33

    # Effective throughput across all GPUs
    effective_tflops_total = gpu_tflops * num_gpus * mfu * (1.0 - overhead)
    effective_flops_per_sec = effective_tflops_total * 1e12

    time_seconds = (total_flops / effective_flops_per_sec) * (1.0 + extra_overhead * 0.05)

    # Startup overhead: model load + tokenizer + dataset prep + process group init
    model_load_sec = (num_params / 1e9) * 1.0  # ~1 s per billion params from disk
    data_prep_sec = 15.0 + steps_per_epoch * 0.002
    dist_init_sec = {"solo": 5.0, "ddp": 20.0, "fsdp": 35.0}.get(strategy, 15.0)
    startup_sec = model_load_sec + data_prep_sec + dist_init_sec

    time_seconds += startup_sec + 10.0
    time_seconds = max(10.0, time_seconds)
    total_minutes = time_seconds / 60
    total_hours = time_seconds / 3600
    total_days = time_seconds / 86400

    if time_seconds < 60:
        human_readable = f"{time_seconds:.0f} seconds"
    elif time_seconds < 3600:
        human_readable = f"{total_minutes:.1f} minutes"
    elif time_seconds < 86400:
        human_readable = f"{total_hours:.1f} hours"
    else:
        human_readable = f"{total_days:.1f} days"

    return {
        "total_seconds": round(time_seconds, 2),
        "total_minutes": round(total_minutes, 2),
        "total_hours": round(total_hours, 2),
        "total_days": round(total_days, 3),
        "human_readable": human_readable,
    }
