
from math import sqrt

import plotext as plt   
import os
import yaml
import sys
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import Shard

import torch
import torch.distributed as dist

from torch.distributed.tensor import DTensor, Shard, Replicate

from torch.distributed import get_rank
from torch.nn import Module


# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------

def plot_losses_in_terminal(_losses):
    """Plot training loss curve, handling NaN values gracefully."""
    import math
    
    print("\nPlotting training loss...")
    print(f"Losses: {_losses}")
    
    # Check if losses list is empty
    if not _losses:
        print("⚠️ No loss data to plot (empty list)")
        return
    
    # Check if all losses are NaN
    all_nan = all(isinstance(x, (float, int)) and math.isnan(x) for x in _losses if x is not None)
    if all_nan or len(_losses) == 0:
        print("⚠️ All loss values are NaN. Cannot generate plot.")
        print("   Possible causes:")
        print("   - Gradient explosion (learning rate too high)")
        print("   - Numerical instability in quantization")
        print("   - Data type mismatches (try disabling mixed precision)")
        print("   - Try reducing learning rate or enabling gradient clipping")
        return
    
    # Filter out NaN values for plotting, but keep indices for context
    valid_epochs = []
    valid_losses = []
    nan_epochs = []
    
    for i, loss in enumerate(_losses, start=1):
        if loss is not None and not math.isnan(loss):
            valid_epochs.append(i)
            valid_losses.append(loss)
        else:
            nan_epochs.append(i)
    
    if not valid_losses:
        print("⚠️ No valid loss values to plot (all NaN)")
        return
    
    # Clear and create plot
    plt.clear_figure()
    
    # Plot valid loss points
    plt.plot(valid_epochs, valid_losses, marker="braille", label="loss curve")
    plt.scatter(valid_epochs, valid_losses, marker="dot", label="epoch loss")
    
    # Mark NaN epochs if any (as red X markers at the bottom)
    if nan_epochs:
        # Place markers at the minimum valid loss (or 0 if no valid losses)
        y_position = min(valid_losses) if valid_losses else 0
        plt.scatter(nan_epochs, [y_position] * len(nan_epochs), 
                   marker="x", color="red", label=f"NaN (epochs {nan_epochs})")
        print(f"⚠️ NaN losses detected at epoch(s): {nan_epochs}")
    
    # Configure plot
    plt.plot_size(80, 30)
    plt.title("Training Loss" + (" (with NaN values)" if nan_epochs else ""))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(list(range(1, len(_losses) + 1)))
    plt.theme("dark")
    
    # Add warning text if there were NaNs
    if nan_epochs:
        print("\n💡 Tip: NaN losses often indicate training instability.")
        print("   Try reducing learning rate, checking gradient clipping, or disabling mixed precision.")
    
    plt.show()
    

def print_config(args):
    """ Print the training configs """
    groups = {
        "Model & Dataset": ["model_name", "model_type", "dataset", "dataset_full_name", "dataset_split"],
        "System":          ["strategy", "num_gpus", "checkpoint_dir", "save"],
        "Training":        [ "epochs", "batch_size", "max_length", "learning_rate", "warmup_steps", "weight_decay", "grad_clip"],
        "Runtime":         ["gradient_checkpointing"],
        "Precision":       ["mixed_precision", "param_dtype", "reduce_dtype", "output_dtype", "cast_forward_inputs"],
        "PEFT":            ["peft_enabled", "peft_type", "peft_r", "peft_alpha", "peft_dropout", "peft_target_modules", "peft_bias"],
        "Quantization":    ["quantization_enabled", "quantization_bits", "quantization_type", "quantization_compute_dtype", "quantization_double_quant"],
        "Checkpoint":      ["distribute_api", "resume", "resume_path", "load_model_from_hf"],
        "Prefetch":        ["explicit_prefetching", "forward_prefetch", "backward_prefetch"],
        "Wandb":           ["wandb_log_with_train", "wandb_entity", "wandb_project", "wandb_run_name"]
    }

    cfg = vars(args)
    width = 72
    col = 32

    print(f"\n  ╔{'═' * width}╗")
    print(f"  ║{'  ⚙  TRAINING CONFIGURATION':^{width}}║")
    print(f"  ╠{'═' * width}╣")

    seen = set()
    for group, keys in groups.items():
        print(f"  ║  {'▸ ' + group:<{width - 2}}║")
        for key in keys:
            if key not in cfg:
                continue
            seen.add(key)
            val = cfg[key]
            label = f"    {key}"
            val_str = str(val) if val is not None else "—"
            # truncate long values
            if len(val_str) > width - col - 2:
                val_str = val_str[: width - col - 5] + "..."
            print(f"  ║  {label:<{col - 2}}  {val_str:<{width - col - 2}}║")
        print(f"  ║{' ' * width}║")

    # catch any attrs not covered by groups
    extras = [(k, v) for k, v in cfg.items() if k not in seen]
    if extras:
        print(f"  ║  {'▸ Other':<{width - 2}}║")
        for key, val in extras:
            label = f"    {key}"
            val_str = str(val) if val is not None else "—"
            if len(val_str) > width - col - 2:
                val_str = val_str[: width - col - 5] + "..."
            print(f"  ║  {label:<{col - 2}}  {val_str:<{width - col - 2}}║")
        print(f"  ║{' ' * width}║")

    print(f"  ╚{'═' * width}╝\n")


# ----------------------------------------------------------------------
# Model Inspection
# ----------------------------------------------------------------------


def inspect_model(model: Module, verbose: bool = True) -> None: 
    """
    Inspects and validates an FSDP2 / DTensor-backed model distribution state
    using standard python print operations.
    """
    rank = get_rank()
    
    # 1. Print full architecture if verbose is enabled
    if verbose and rank == 0: 
        print("\nModel Architecture:\n ", flush=True) 
        print(model, flush=True)

    sharded_params = 0
    replicated_params = 0
    total_elements = 0

    # 2. Inspect placement strategies across all parameters safely
    for name, param in model.named_parameters():
        total_elements += param.numel()
        
        # Check if the parameter is managed as a Distributed Tensor
        if isinstance(param, DTensor):
            placements = param.placements
            
            # Identify if it is sharded along dimension 0 or replicated
            if any(isinstance(p, Shard) and p.dim == 0 for p in placements):
                sharded_params += 1
            elif any(isinstance(p, Replicate) for p in placements):
                replicated_params += 1
        else:
            # Fallback for standard un-sharded torch.Tensors
            replicated_params += 1

    # 3. Print the execution summary
    if rank == 0: 
        print( " \n FSDP INSPECTION: ", flush=True) 
        print(f"   Total Weight & Bias Tensors: {sharded_params + replicated_params}", flush=True)
        print(f"   └─ Sharded Tensors  (Shard(0)): {sharded_params}", flush=True)
        print(f"   └─ Replicated Tensors (Replicate): {replicated_params}", flush=True)
        print(f"   Total Model Parameters (Floats): {total_elements:,}", flush=True)
        print("✅  Inspection completed successfully.\n", flush=True)




# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

CONFIG_PATH = os.getenv("CONFIG_PATH", "config.yaml")

def load_config():
    try:
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {CONFIG_PATH}")
        sys.exit(1)

class Args:
    def __init__(self):
        self.model_name = None
        self.model_type: str  # llm, seq2seq, vision, yolo, vlm, encoder

        self.dataset = None
        self.dataset_full_name = None
        self.dataset_split = None

        self.strategy = None # fsdp, ddp, or solo
        self.checkpoint_dir = None
        self.save: bool 

        self.num_gpus: int
        self.epochs: int 
        self.batch_size: int 
        self.max_length: int  
        self.learning_rate: float  
        self.warmup_steps: int
        self.weight_decay: float
        self.grad_clip: float
        self.gradient_checkpointing: bool

        self.mixed_precision: bool 
        self.param_dtype = None
        self.reduce_dtype = None
        self.output_dtype = None
        self.cast_forward_inputs: bool 

        self.distribute_api: str
        self.dcp_api: bool 
        self.dtensor_api: bool
        self.resume: bool 
        self.resume_path = None

        self.explicit_prefetching: bool 
        self.forward_prefetch: int 
        self.backward_prefetch: int

        self.load_model_from_hf: bool   

        self.peft_enabled: bool
        self.peft_type: str
        self.peft_r: int
        self.peft_alpha: int
        self.peft_dropout: float
        self.peft_target_modules = None
        self.peft_bias: str

        self.quantization_enabled: bool
        self.quantization_bits: int
        self.quantization_type: str
        self.quantization_compute_dtype: str
        self.quantization_double_quant: bool

        # Custom Transformer architecture (only used when model_type == "custom_transformer")
        self.custom_n_layers: int = 2
        self.custom_vocab_size: int = 8
        self.custom_max_seq_len: int = 16
        self.custom_dim: int = 16
        self.custom_n_heads: int = 4
        self.custom_dropout_p: float = 0.1

        self.wandb_log_with_train: bool
        self.wandb_entity: str
        self.wandb_project: str
        self.wandb_run_name: str 

def to_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.lower() in ["true", "1", "yes", "y", "on"]
    return bool(x)

def build_args(cfg):
    args = Args()

    # --------------------------------------------------
    # MODEL + DATASET
    # --------------------------------------------------
    args.model_name = cfg["model_name"]
    args.model_type = str(cfg.get("model_type", "llm")).lower()
    if args.model_type not in {"llm", "seq2seq", "yolo", "vlm", "vision", "encoder", "custom_transformer"}:
        raise ValueError(f"Unsupported model_type={args.model_type}. Use one of: llm, seq2seq, yolo, vlm, vision, encoder, custom_transformer")

    dataset = cfg["dataset"]
    args.dataset = dataset["name"]
    args.dataset_full_name = dataset["subset"]
    args.dataset_split = dataset.get("split", "train")

    # --------------------------------------------------
    # SYSTEM
    # --------------------------------------------------
    args.strategy = cfg["strategy"]
    args.num_gpus = int(cfg.get("num_gpus", 1))
    args.checkpoint_dir = cfg["checkpoint_dir"]
    args.save = to_bool(cfg["save"])

    if args.strategy not in {"solo", "ddp", "fsdp"}:
        raise ValueError(f"Unsupported strategy={args.strategy}. Use one of: solo, ddp, fsdp")

    # --------------------------------------------------
    # TRAINING
    # --------------------------------------------------
    t = cfg["training"] 
    args.epochs = int(t["epochs"])
    args.batch_size = int(t["batch_size"])
    args.max_length = int(t["max_length"])
    args.learning_rate = float(t["learning_rate"])
    args.warmup_steps = int(t.get("warmup_steps", 100))
    args.weight_decay = float(t.get("weight_decay", 0.01))
    args.grad_clip = float(t.get("grad_clip", 1.0))
    # Global activation checkpointing toggle used by both quantized and non-quantized paths.
    args.gradient_checkpointing = to_bool(t.get("gradient_checkpointing", True))

    if args.epochs < 1:
        raise ValueError("epochs must be at least 1")
    if args.batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if args.max_length < 1:
        raise ValueError("max_length must be at least 1")
    if args.warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if args.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if args.grad_clip < 0:
        raise ValueError("grad_clip must be non-negative")

    # --------------------------------------------------
    # FSDP
    # --------------------------------------------------
    f = cfg["dist_parameters"]
    args.mixed_precision = to_bool(f.get("mixed_precision", False))
    args.param_dtype = f.get("param_dtype", None)
    args.reduce_dtype = f.get("reduce_dtype", None)
    args.output_dtype = f.get("output_dtype", None)
    args.cast_forward_inputs = to_bool(f.get("cast_forward_inputs", False))
    args.distribute_api = f.get("distribute_api", "dcp_api")
    args.dcp_api = args.distribute_api == "dcp_api"
    args.dtensor_api = args.distribute_api == "dtensor_api"

    # --------------------------------------------------
    # SAVE LOAD
    # --------------------------------------------------
    sl = cfg["save_load"]
    args.resume = to_bool(sl.get("resume", False))
    args.resume_path = sl.get("resume_path", None)
    args.load_model_from_hf = to_bool(sl.get("load_model_from_hf", True))

    # --------------------------------------------------
    # PEFT
    # --------------------------------------------------
    # Parse PEFT settings once and pass as strongly typed runtime args.
    peft_cfg = cfg.get("peft", {})
    args.peft_enabled = to_bool(peft_cfg.get("enabled", False))
    args.peft_type = str(peft_cfg.get("type", "lora")).lower()
    args.peft_r = int(peft_cfg.get("r", 16))
    args.peft_alpha = int(peft_cfg.get("alpha", 32))
    args.peft_dropout = float(peft_cfg.get("dropout", 0.05))
    args.peft_target_modules = peft_cfg.get("target_modules", "all-linear")
    args.peft_bias = str(peft_cfg.get("bias", "none"))

    # Restrict supported adapter types to what distributed setup currently implements.
    if args.peft_type not in {"lora", "qlora"}:
        raise ValueError(f"Unsupported peft.type={args.peft_type}. Use one of: lora, qlora")

    _peft_compatible_types = {"llm", "seq2seq", "encoder", "vlm", "vision"}
    if args.peft_enabled and args.model_type not in _peft_compatible_types and args.model_type != "custom_transformer":
        raise ValueError(
            f"PEFT is not supported for model_type='{args.model_type}'.\n"
            f"PEFT (LoRA/QLoRA) requires a transformer architecture. "
            f"Supported types: {sorted(_peft_compatible_types)}"
        )

    # --------------------------------------------------
    # QUANTIZATION
    # --------------------------------------------------
    # Parse quantization block; defaults are QLoRA-friendly but disabled unless explicitly enabled.
    qcfg = cfg.get("quantization", {})
    args.quantization_enabled = to_bool(qcfg.get("enabled", False))
    args.quantization_bits = int(qcfg.get("bits", 4))
    args.quantization_type = str(qcfg.get("quant_type", "nf4")).lower()
    args.quantization_compute_dtype = str(qcfg.get("compute_dtype", "bfloat16")).lower()
    args.quantization_double_quant = to_bool(qcfg.get("double_quant", True))

    # Limit to bit widths currently handled by bitsandbytes wiring in this project.
    if args.quantization_bits not in {4, 8}:
        raise ValueError(f"Unsupported quantization.bits={args.quantization_bits}. Use 4 or 8")

    if args.peft_type == "qlora":
        args.peft_enabled = True
        args.quantization_enabled = True
        
        if args.quantization_bits != 4:
            raise ValueError("QLoRA requires quantization.bits=4")

    if args.quantization_enabled and not args.peft_enabled:
        if args.quantization_bits == 4:
            # 4-bit (NF4) has no differentiable backward — mathematically impossible to train.
            raise ValueError(
                "4-bit quantization requires peft.enabled=true. "
                "Use peft_type=qlora for QLoRA fine-tuning."
            )
        else:
            # CHANGED: was a hard ValueError. Downgraded to a warning so the user can proceed
            # if they choose to, while being clearly informed of the instability.
            # 8-bit training is allowed but will almost certainly produce NaN losses because:
            #   1. INT8 weights have a very limited range (-127 to 127); AdamW gradient updates
            #      overflow that range within a few steps, corrupting weights to NaN.
            #   2. LayerNorm and embedding layers stay in float16 without
            #      prepare_model_for_kbit_training — float16 overflows at ~65504, causing
            #      NaN in the forward pass itself before gradients are even computed.
            #   3. Optimizer states (exp_avg, exp_avg_sq) accumulate in the wrong dtype,
            #      amplifying the instability across steps.
            # Recommended alternatives:
            #   • peft.enabled=true + quantization.bits=8  → stable 8-bit LoRA
            #   • quantization.enabled=false               → full float fine-tuning
            import warnings
            warnings.warn(
                "\n⚠️  WARNING: 8-bit quantization without PEFT will likely produce NaN losses.\n"
                "   Reasons:\n"
                "   1. INT8 weight range is -127 to 127 — AdamW updates overflow it within steps.\n"
                "   2. LayerNorm/embeddings stay in float16, which overflows at ~65504.\n"
                "   3. Optimizer states accumulate in the wrong dtype, amplifying instability.\n"
                "   Alternatives:\n"
                "   • peft.enabled=true  (8-bit LoRA — stable)\n"
                "   • quantization.enabled=false  (full float fine-tuning)\n"
                "   Proceeding anyway...\n",
                UserWarning,
                stacklevel=2,
            )

    if args.strategy == "fsdp" and args.quantization_enabled:
        raise ValueError(
            "FSDP + bitsandbytes quantization is not supported: "
            "bitsandbytes 4-bit/8-bit layers cannot be sharded by FSDP. "
            "Use strategy=ddp to keep quantization, or set quantization.enabled=false to keep FSDP."
        )

    # --------------------------------------------------
    # PREFETCH
    # --------------------------------------------------
    p = cfg["prefetch"]
    args.explicit_prefetching = to_bool(p["explicit"])
    args.forward_prefetch = int(p["forward"])
    args.backward_prefetch = int(p["backward"])

    # --------------------------------------------------
    # CUSTOM TRANSFORMER ARCHITECTURE
    # --------------------------------------------------
    if args.model_type == "custom_transformer":
        cta = cfg.get("custom_transformer_args", {})
        args.custom_n_layers    = int(cta.get("n_layers",    2))
        args.custom_vocab_size  = int(cta.get("vocab_size",  8))
        args.custom_max_seq_len = int(cta.get("max_seq_len", 16))
        args.custom_dim         = int(cta.get("dim",         16))
        args.custom_n_heads     = int(cta.get("n_heads",     4))
        args.custom_dropout_p   = float(cta.get("dropout_p", 0.1))
        if args.custom_dim % args.custom_n_heads != 0:
            raise ValueError(
                f"custom_transformer: dim ({args.custom_dim}) must be divisible by n_heads ({args.custom_n_heads})"
            )

    # --------------------------------------------------
    # WANDB
    # --------------------------------------------------
    wb = cfg["wandb"]
    args.wandb_log_with_train = to_bool(wb.get("wandb_log_with_train", True))
    args.wandb_entity = wb.get("wandb_entity", "dist-train-project")
    args.wandb_project = wb.get("wandb_project", "dist-train-project") 
    args.wandb_run_name = wb.get("wandb_run_name", "")

    return args

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
    total_gb = weights_gb + gradients_gb + optimizer_gb + activations_gb

    return {
        "weights_gb":     round(weights_gb, 2),
        "gradients_gb":   round(gradients_gb, 2),
        "optimizer_gb":   round(optimizer_gb, 2),
        "activations_gb": round(activations_gb, 2),
        "total_gb":       round(total_gb, 2),
    }


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
        "l4": 120, "t4": 65, "a6000": 155, "a5000": 110,
        "a4000": 77, "a2": 31, "4090": 83, "3090": 36, "3080": 30,
        "3070": 20, "3060": 13, "2080": 14, "1080": 11,
    }
    gpu_tflops = gpu_tflops_map.get(gpu_type, 40)  # 40 TFLOPS: conservative unknown GPU default

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


def dist_barrier(local_rank: int) -> None:
    TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])

    if dist.get_backend() == "nccl" and TORCH_VERSION >= (2, 0):
        dist.barrier(device_ids=[local_rank])
    else:
        dist.barrier()