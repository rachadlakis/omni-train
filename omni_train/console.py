"""Terminal output: rank-aware prints and banners, the config table, the loss plot, FSDP inspection."""

import socket

import plotext as plt
import torch
from torch.distributed import get_rank
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.nn import Module


# Terminal columns reserved for the leading emoji so that message text always
# starts at the same column no matter which emoji (or none) precedes it.
_GUTTER = 4


def _display_width(text):
    """Best-effort count of terminal columns occupied by `text`.

    Zero-width joiners and variation selectors take no cell; emoji and symbol
    glyphs take two. This is a small heuristic (no wcwidth dependency) that
    matches how modern terminals render the emoji used in these banners, so the
    gutter padding lines up instead of drifting by an emoji's render width.
    """
    import unicodedata
    width = 0
    for ch in text:
        code = ord(ch)
        if unicodedata.combining(ch) or code in (0x200D, 0xFE0E, 0xFE0F):
            continue  # ZWJ + variation selectors are zero-width
        if (code >= 0x1F000                  # emoji & pictographs
                or 0x2600 <= code <= 0x27BF  # misc symbols / dingbats
                or 0x2B00 <= code <= 0x2BFF  # arrows & symbols
                or 0x2190 <= code <= 0x21FF):
            width += 2
        else:
            width += 1
    return width


def _gutter(emoji):
    """Emoji padded out to the fixed-width left gutter, or blank spaces if none."""
    if not emoji:
        return " " * _GUTTER
    return emoji + " " * max(1, _GUTTER - _display_width(emoji))


def print_on_rank_0(rank, msg, emoji=""):
    if rank == 0:
        # Emoji lines are event headers (one blank line above for separation);
        # plain lines are tight detail lines under the preceding event.
        lead = "\n" if emoji else ""
        print(f"{lead}{_gutter(emoji)}{msg}", flush=True)


def print_banner_on_rank_0(rank, title):
    if rank == 0:
        bar = "=" * 60
        print(f"\n{bar}\n  {title}\n{bar}", flush=True)


def print_on_all_ranks(rank, msg, emoji="", local_rank=None, device=None):
    """Prints a message from all ranks, prefixed with rank and device info."""
    host = socket.gethostname().split(".")[0]
    prefix_parts = [f"host={host}", f"rank={rank}"]
    if local_rank is not None:
        prefix_parts.append(f"local_rank={local_rank}")
    if device is not None:
        prefix_parts.append(f"device={device}")
    prefix = " | ".join(prefix_parts)
    print(f"\n{_gutter(emoji)}[{prefix}] {msg}", flush=True)


def gpu_memory_snapshot(device):
    """Returns a string with the current GPU memory usage (allocated and reserved) for the given device."""
    if device.type != "cuda":
        return "CUDA not detected."
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
    return f"allocated={allocated:.2f} GB | reserved={reserved:.2f} GB"


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
        "Training":        [ "epochs", "batch_size", "max_length", "learning_rate", "lr_scheduler", "warmup_steps", "weight_decay", "grad_clip"],
        "Runtime":         ["gradient_checkpointing"],
        "Precision":       ["mixed_precision", "param_dtype", "reduce_dtype", "output_dtype", "cast_forward_inputs"],
        "PEFT":            ["peft_enabled", "peft_type", "peft_r", "peft_alpha", "peft_dropout", "peft_target_modules", "peft_bias"],
        "Quantization":    ["quantization_enabled", "quantization_bits", "quantization_type", "quantization_compute_dtype", "quantization_double_quant"],
        "Algorithm":       ["algorithm", "algo_beta", "algo_dpo_loss", "algo_label_smoothing", "algo_simpo_gamma", "algo_max_prompt_length", "algo_completion_only"],
        "Parallelism (5d)": ["par_dp", "par_tp", "par_sp", "par_cp", "par_pp", "par_pp_schedule", "par_micro_batches", "par_ep", "par_zero1"],
        "Checkpoint":      ["distribute_api", "resume", "resume_path", "load_model_from_hf", "export_hf", "export_dtype"],
        "Prefetch":        ["explicit_prefetching", "forward_prefetch", "backward_prefetch"],
        "Wandb":           ["wandb_log_with_train", "wandb_entity", "wandb_project", "wandb_run_name"],
    }

    cfg = vars(args)
    bar = "═" * 60      # horizontal rule only — no right border to drift out of alignment
    label_w = 28        # key column width; longest key (~26 chars) + breathing room
    max_val = 44        # truncate very long values so a row never wraps

    def _row(key, val):
        val_str = str(val) if val not in (None, "") else "—"
        if len(val_str) > max_val:
            val_str = val_str[: max_val - 3] + "..."
        print(f"      {key:<{label_w}}{val_str}", flush=True)

    print(f"\n{bar}", flush=True)
    print("  ⚙  TRAINING CONFIGURATION", flush=True)
    print(bar, flush=True)

    seen = set()
    for group, keys in groups.items():
        if (group == "Parallelism (5d)" and cfg.get("strategy") != "5d") or \
                (group == "Algorithm" and cfg.get("algorithm", "pretrain") == "pretrain"):
            seen.update(keys)
            continue
        present = [k for k in keys if k in cfg]
        if not present:
            continue
        print(f"\n  ▸ {group}", flush=True)
        for key in present:
            seen.add(key)
            _row(key, cfg[key])

    # catch any attrs not covered by groups
    extras = [(k, v) for k, v in cfg.items() if k not in seen]
    if extras:
        print("\n  ▸ Other", flush=True)
        for key, val in extras:
            _row(key, val)

    print(f"{bar}\n", flush=True)


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
