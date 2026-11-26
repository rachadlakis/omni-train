"""Optimizer helpers: the learning-rate schedule and restoring AdamW state on resume."""

import torch


def restore_optimizer_state(optimizer, args, device) -> str:
    """Solo/DDP resume: load the AdamW moments saved next to the weights.

    The LR schedule restarts on resume (warmup, then decay over the new run), so each
    group's lr / initial_lr / weight_decay are reset to this run's config instead of
    the values the previous run ended with (~0 after a cosine decay).
    Returns "restored", "absent" (older weights-only checkpoint) or "mismatch".
    """
    # CPU first: load_state_dict moves each state tensor to its parameter's device.
    ckpt = torch.load(args.resume_path, map_location="cpu", weights_only=True, mmap=True)
    state = ckpt.get("optimizer_state_dict") if isinstance(ckpt, dict) else None
    if state is None:
        return "absent"
    try:
        optimizer.load_state_dict(state)
    except (ValueError, KeyError, RuntimeError):
        return "mismatch"  # different trainable parameters (e.g. PEFT settings changed)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] = args.learning_rate
        group["weight_decay"] = args.weight_decay
    return "restored"


def lr_multiplier(step_idx: int, warmup_steps: int, total_steps: int, kind: str) -> float:
    """LR factor for one optimizer step: linear warmup, then decay to 0 over the
    remaining steps (cosine / linear) or hold (constant)."""
    import math
    if step_idx < warmup_steps:
        return float(step_idx + 1) / float(max(1, warmup_steps))
    if kind == "constant":
        return 1.0
    progress = min(1.0, (step_idx - warmup_steps) / float(max(1, total_steps - warmup_steps)))
    if kind == "linear":
        return 1.0 - progress
    return 0.5 * (1.0 + math.cos(math.pi * progress))  # cosine
