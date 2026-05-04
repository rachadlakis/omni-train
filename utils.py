
from math import sqrt

import plotext as plt   
import os
import yaml
import sys

# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------

def plot_losses_in_terminal(_losses):
    print("\nPlotting training loss...")
    print(f"Losses: {_losses}")
    epochs = list(range(1, len(_losses) + 1))
    plt.clear_figure()
    plt.plot(epochs, _losses, marker="braille", label="loss curve")
    plt.scatter(epochs, _losses, marker="dot", label="epoch loss")
    plt.plot_size(80, 30)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(epochs)
    plt.theme("dark")
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
        raise ValueError("Quantization-aware training in this project currently requires peft.enabled=true")

    if args.strategy == "fsdp" and args.quantization_enabled:
        print("\n❌ Config error: FSDP + quantization is not supported.")
        print("   FSDP cannot shard bitsandbytes 4-bit/8-bit layers.")
        print("   Fix one of the following in your config:")
        print("     • Set strategy: ddp       → to keep QLoRA")
        print("     • Set quantization: false → to keep FSDP\n")
        sys.exit(1)

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
    args.wandb_entity = wb.get("wandb_entity", "fsdp-mini-project")
    args.wandb_project = wb.get("wandb_project", "fsdp-mini-project") 
    args.wandb_run_name = wb.get("wandb_run_name", "")

    return args

# ----------------------------------------------------------------------
# VRAM Estimator
# ----------------------------------------------------------------------

# def estimate_training_vram(
#     model_type: str,       # llm, seq2seq, vision, yolo, vlm, encoder
#     num_params: int,        # e.g. 125_000_000 for opt-125m
#     param_dtype_bits: int,  # 16 for bfloat16, 32 for float32
#     batch_size: int,
#     seq_len: int,
#     num_layers: int,
#     hidden_dim: int,
#     num_heads: int,
#     activation_checkpointing: bool = False):
    
#     """ Estimates VRAM usage in GB for weights, gradients, optimizer states, and activations. """
#     if model_type not in {"llm", "seq2seq", "vision", "yolo", "vlm", "encoder"}:
#         raise ValueError(f"Unsupported model_type={model_type}. Use one of: llm, seq2seq, vision, yolo, vlm, encoder")
    

#     bytes_per_param = param_dtype_bits / 8 
#     weights_gb    = num_params * bytes_per_param / 1e9
#     gradients_gb  = weights_gb                          # same shape as weights
#     optimizer_gb  = weights_gb * 3
    
#     if model_type in {"vision", "yolo"}:
#         # Vision models often have large activations due to high-res inputs, but we'll use the same formula for simplicity.
#         pass
#     elif model_type == "vlm":
#         # VLMs can have additional components (e.g. vision tower), but we'll use the same formula for simplicity.
#         pass
#     elif model_type == "encoder":
#         # Encoder-only models may have slightly different activation patterns, but we'll use the same formula for simplicity.
#         pass
#     elif model_type in {"llm", "seq2seq"}:
#         act_bytes = (num_layers * 2 * seq_len * batch_size * hidden_dim *
#                     (16 + 2 / bytes_per_param +
#                     2 * num_heads * seq_len / hidden_dim))
#         activations_gb = act_bytes / 1e9
#         if activation_checkpointing:
#             activations_gb = activations_gb / (num_layers ** 0.5)
#         total_gb = weights_gb + gradients_gb + optimizer_gb + activations_gb
#         return {
#         "weights_gb":     round(weights_gb, 2),
#         "gradients_gb":   round(gradients_gb, 2),
#         "optimizer_gb":   round(optimizer_gb, 2),
#         "activations_gb": round(activations_gb, 2),
#         "total_gb":       round(total_gb, 2),
#     }

def estimate_training_vram(
    model_type: str,               # llm, seq2seq, vision, yolo, vlm, encoder
    num_params: int,               # number of trainable parameters
    param_dtype_bits: int,         # 16 for bfloat16/fp16, 32 for fp32
    batch_size: int,
    seq_len: int,
    num_layers: int,
    hidden_dim: int,
    activation_checkpointing: bool = False,
) -> dict:
    """
    Estimates VRAM usage in GB for training with Adam optimizer.

    Assumptions:
    - Gradients use the same dtype as weights.
    - For mixed precision (param_dtype_bits=16): Adam stores fp32 master weights + two fp32 moments
      → 12 bytes per original fp16/bf16 parameter → 6× weight memory.
    - For fp32: Adam stores fp32 weights + two fp32 moments → 12 bytes per param → 3× weight memory.
    - Activation memory for transformer‑based models (llm, seq2seq, encoder):
        approx = batch_size * seq_len * hidden_dim * num_layers * 2  (covers attention + MLP outputs).
    - Activation checkpointing halves the activation memory (common practice).
    - For vision/YOLO, activation memory is estimated as 20% of weights (very rough).
    - For VLM, the LLM part dominates, so same as transformer.
    """
    valid_types = {"llm", "seq2seq", "vision", "yolo", "vlm", "encoder"}
    if model_type not in valid_types:
        raise ValueError(f"Unsupported model_type={model_type}. Use one of {valid_types}")
    
    bytes_per_param = param_dtype_bits / 8
    weights_gb = num_params * bytes_per_param / 1e9
    gradients_gb = weights_gb                     # same dtype as weights

    # ---- Optimizer memory (Adam) ----
    if param_dtype_bits == 32:
        # fp32: weights (4B) + two fp32 states (8B) = 12B per param → 3× weights_gb
        optimizer_gb = weights_gb * 3
    else:  # 16-bit (fp16/bf16) mixed precision
        # weights are 2B, but optimizer keeps fp32 master (4B) + two fp32 moments (8B) = 12B per param
        # 12B / 2B = 6× weights_gb
        optimizer_gb = weights_gb * 6

    # ---- Activation memory ----
    if model_type in {"llm", "seq2seq", "encoder"}:
        # Transformer: each layer stores ~ batch * seq_len * hidden_dim * 2 (forward pass)
        act_bytes = batch_size * seq_len * hidden_dim * num_layers * 2
    elif model_type in {"vision", "yolo"}:
        # Crude heuristic: activations ~20% of weight memory (modify for your use case)
        act_bytes = weights_gb * 0.2 * 1e9
    elif model_type == "vlm":
        # Assume the LLM component dominates
        act_bytes = batch_size * seq_len * hidden_dim * num_layers * 2
    else:
        act_bytes = 0

    if activation_checkpointing:
        act_bytes /= 2.0   # typical reduction factor

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
    num_gpus: int = 1,
    gpu_type: str = "unknown",
    strategy: str = "solo", 
    peft_enabled: bool = False,
    peft_r: int = 16,
    gradient_checkpointing: bool = False,
    mfu: float = 0.35,
    extra_overhead: float = 1.0,
) -> dict:
    """
    Estimates training time based on FLOPs / effective throughput.

    Note:
    This estimator intentionally does not use sequence length to avoid very
    large swings in ETA caused by config defaults or template mismatches.
    """
    strategy = (strategy or "solo").strip().lower()

    # Peak TFLOPS per GPU by type (bf16/fp16)
    gpu_tflops_map = {
        "h100": 990, "a100": 312, "a10g": 125, "v100": 125,
        "l4": 120, "t4": 65, "a6000": 155, "a5000": 110,
        "a4000": 77, "a2": 31, "4090": 83, "3090": 36, "3080": 30,
    }
    gpu_tflops = gpu_tflops_map.get(gpu_type, 80)  # 80 TFLOPS as safe modern default

    # Distributed communication overhead
    overhead = {"solo": 0.0, "ddp": 0.08, "fsdp": 0.18}.get(strategy, 0.1)

    # Approximate work from parameter count and optimizer steps.
    # We intentionally avoid sequence-length dependence for a stabler ETA.
    # 256 token-equivalent units per sample keeps ETA realistic for small runs.
    token_equiv_per_sample = 256

    total_samples = batch_size * steps_per_epoch * epochs
    total_flops = 6.0 * num_params * total_samples * token_equiv_per_sample

    # LoRA reduces active parameters during backward
    if peft_enabled:
        lora_ratio = (2 * peft_r * num_params ** 0.5) / num_params
        total_flops *= (1.0 - 0.5 * min(lora_ratio, 0.9))

    # Activation checkpointing adds ~33% recompute cost
    if gradient_checkpointing:
        total_flops *= 1.33

    # Effective throughput across all GPUs
    effective_tflops_total = gpu_tflops * num_gpus * mfu * (1.0 - overhead)
    effective_flops_per_sec = effective_tflops_total * 1e12

    time_seconds = (total_flops / effective_flops_per_sec) * (1.0 + extra_overhead * 0.05)
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