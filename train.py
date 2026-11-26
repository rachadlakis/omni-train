## train.py
## This script is built according to the PyTorch documentation and tutorials for FSDP2 : 
## https://github.com/pytorch/examples/tree/main/distributed/FSDP2


# ----------------------------------------------------------------------
# Import Libraries 
# ----------------------------------------------------------------------

import os
from dotenv import load_dotenv 
import sys
import traceback
# import argparse
# import yaml
# import time
import warnings
from contextlib import nullcontext
from transformers import AutoTokenizer
import torch
# import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DistributedSampler 

from omni_train import advisor
from omni_train.model import Transformer 
from omni_train.checkpoint import Checkpointer
from omni_train.data import get_dataloader
from omni_train.utils import (
    plot_losses_in_terminal,
    print_config,
    estimate_training_time,
)
from omni_train.distributed_utils import (
    BACKEND,
    print_on_rank_0,
    print_banner_on_rank_0,
    print_on_all_ranks,
    gpu_memory_snapshot,
    print_training_vram_estimate,
    setup_dist_process_group,
    cleanup,
    apply_solo,
    apply_ddp,
    apply_fsdp,
    save_checkpoint,
    restore_optimizer_state,
)
import wandb

from omni_train.utils import build_args, load_config, dist_barrier, lr_multiplier

import transformers
transformers.logging.disable_progress_bar()

## -------------------------------
# wandb 
# ----------------------------

# ----------------------------------------------------------------------
# Seed 
# ----------------------------------------------------------------------

torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

# ----------------------------------------------------------------------
# Env Variables
# ----------------------------------------------------------------------
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

# ----------------------------------------------------------------------
# Dtypes Map & BACKEND
# ----------------------------------------------------------------------

DTYPE_MAP = {
    "bfloat16": torch.bfloat16, 
    "float32": torch.float32, 
    "float16": torch.float16
}

if torch.cuda.is_available():
    if not sys.platform.startswith("linux"):
        raise RuntimeError(
            "The current distributed CUDA path uses NCCL, and NCCL is supported on Linux for now."
            "Non-Linux backends can be added later."
            f"Current platform: {sys.platform}"
        )
    BACKEND = "nccl"
else:
    BACKEND = "gloo"


# ----------------------------------------------------------------------
# Filter Warnings
# ---------------------------------------------------------------------

warnings.filterwarnings("ignore", message=".*_get_pg_default_device.*")
warnings.filterwarnings("ignore", message="Materializing param")
warnings.filterwarnings("ignore", message="Materializing param", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# ADDED: suppress bitsandbytes MatMul8bitLt dtype cast notice.
# Reason: this warning fires on every single forward pass when using 8-bit quantization
# without PEFT, printing inside the \r progress bar loop and breaking the overwrite display.
# The cast from float32 → float16 is expected behaviour for 8-bit matmul — it is already
# described in our startup warning to the user, so repeating it per-step adds no value.
warnings.filterwarnings("ignore", message=".*MatMul8bitLt.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*inputs will be cast.*", category=UserWarning)
# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

EXPERIMENTAL_MODEL_TYPES = {"seq2seq", "encoder", "vision", "yolo", "vlm"}


def main(args):
    """Main training loop for distributed training. 
    Initializes distributed environment, loads model and data, and runs training epochs."""
    try:
        checkpointer = None
        if args.model_type in EXPERIMENTAL_MODEL_TYPES and int(os.environ.get("RANK", "0")) == 0:
            print(f"🧪  model_type '{args.model_type}' is experimental: it works, but has less test "
                  "coverage than llm and no post-training support.", flush=True)

        ## 5D parallelism (DP, TP+SP, CP, PP, EP) has its own model and loop.
        if args.strategy == "5d":
            from omni_train.parallelism.trainer import train_5d
            train_5d(args)
            return

        ## 1. SETUP DISTRIBUTED ENVIRONMENT
        if args.strategy in ["ddp", "fsdp"]:
            local_rank = setup_dist_process_group()
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
            print_on_rank_0(rank, f"Distributed environment initialized | backend={BACKEND} | world_size={world_size} | local_rank={local_rank} | device={device}", "🌐")
        elif args.strategy == "solo":
            local_rank = 0
            rank = 0
            world_size = 1
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print_on_rank_0(rank, f"Running in solo mode on device={device}", "🧪")
        else:
            raise ValueError(f"Unknown strategy: {args.strategy}")
        
        ## WANDB setup - only on rank 0 to avoid multiple runs/logs for distributed strategies
        if args.wandb_log_with_train and rank == 0:
            wandb.login(key=WANDB_API_KEY) 
            print_on_rank_0(rank, f"WANDB logging enabled | project={args.wandb_project}", "📊")
            run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity or None,   # blank → your default W&B entity
                name=args.wandb_run_name or None,   # blank → W&B auto-generated name
                config=vars(args),
                tensorboard=True
                )


        ## Print environment info for debugging and verification
        print_on_all_ranks(rank, f"Process joined | world_size={world_size} | pid={os.getpid()}", "🚀",
                           local_rank=local_rank, device=device)
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(local_rank)
            gpu_mem  = torch.cuda.get_device_properties(local_rank).total_memory / 1e9
            print_on_all_ranks(rank, f"GPU: {gpu_name} ({gpu_mem:.1f} GB) | {gpu_memory_snapshot(device)}", "🖥️",
                               local_rank=local_rank, device=device)
 
        # --------------------------------------------------------------
        # 2. TOKENIZER
        # --------------------------------------------------------------
        tokenizer = None
        if args.model_type == "custom_transformer":
            print_on_rank_0(rank, "Custom Transformer: skipping tokenizer (synthetic data will be used)", "⏩")

        elif args.model_type in {"llm", "seq2seq", "encoder"}:
            print_banner_on_rank_0(rank, "LOADING TOKENIZER")
            print_on_rank_0(rank, f"Fetching tokenizer: {args.model_name}", "🔤")
            tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=HF_TOKEN)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            print_on_rank_0(rank, "Tokenizer ready ✓")
        elif args.model_type == "vlm":
            from transformers import AutoProcessor
            print_banner_on_rank_0(rank, "LOADING PROCESSOR")
            print_on_rank_0(rank, f"Fetching multimodal processor: {args.model_name}", "🖼️")
            tokenizer = AutoProcessor.from_pretrained(args.model_name, token=HF_TOKEN)
            # Disable image tiling/splitting (Idefics3/SmolVLM) so one image maps
            # to a bounded number of tokens — tiling can blow sequence length up
            # into the thousands and exhaust memory.
            image_processor = getattr(tokenizer, "image_processor", None)
            if image_processor is not None and hasattr(image_processor, "do_image_splitting"):
                image_processor.do_image_splitting = False
            inner_tok = getattr(tokenizer, "tokenizer", None)
            if inner_tok is not None and inner_tok.pad_token is None:
                inner_tok.pad_token = inner_tok.eos_token
            print_on_rank_0(rank, "Processor ready ✓")
        elif args.model_type in {"vision", "yolo"}:
            from transformers import AutoImageProcessor 
            print_banner_on_rank_0(rank, "LOADING IMAGE PROCESSOR")
            print_on_rank_0(rank, f"Fetching image processor: {args.model_name}", "🖼️")
            tokenizer = AutoImageProcessor.from_pretrained(args.model_name, token=HF_TOKEN)
            print_on_rank_0(rank, "Image processor ready ✓")
        
        # --------------------------------------------------------------
        # 3. MODEL
        # --------------------------------------------------------------
        print_banner_on_rank_0(rank, "LOADING MODEL")

        if args.strategy == "ddp":
            model = apply_ddp(local_rank, rank, device, args)

        elif args.strategy == "fsdp":
            model, checkpointer = apply_fsdp(local_rank, rank, device, args)

        elif args.strategy == "solo":
            model = apply_solo(device, rank, args)

        else:
            raise ValueError(f"Unknown strategy: {args.strategy}")
            
        # In PEFT/QLoRA runs, most base weights are frozen; optimize only trainable tensors.
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError("No trainable parameters were found. Check PEFT/quantization configuration.")

        # Report the memory footprint of training this model. FSDP prints this earlier,
        # before it shards (inside apply_fsdp), so here we only cover the non-sharded
        # strategies to avoid printing it twice.
        if args.strategy != "fsdp":
            print_training_vram_estimate(rank, model, args, world_size)

        ## Create the optimizer (AdamW) for trainable model parameters.
        # CHANGED: pass trainable_params instead of model.parameters() to AdamW.
        # Reason: for PEFT/QLoRA runs, most base weights are frozen (requires_grad=False).
        # Passing model.parameters() caused AdamW to allocate exp_avg and exp_avg_sq optimizer
        # states for every frozen parameter — potentially GBs of wasted GPU memory. trainable_params
        # was already computed above for the empty-check; reusing it here limits optimizer state
        # to only the parameters that will actually receive gradients during training.
        # Old code: optimizer = torch.optim.AdamW(model.parameters(), ...)
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        scheduler = None
        if rank == 0:
            # Report adapter efficiency explicitly so users can confirm PEFT is active.
            total_params = sum(p.numel() for p in model.parameters())
            trainable_count = sum(p.numel() for p in trainable_params)
            pct = (100.0 * trainable_count / total_params) if total_params else 0.0
            print_on_rank_0(rank, f"Trainable params: {trainable_count:,}/{total_params:,} ({pct:.3f}%)", "📊")

        print_on_rank_0(rank,f"Optimizer: {optimizer.__class__.__name__} | lr={args.learning_rate} | weight_decay={args.weight_decay} | grad_clip={args.grad_clip}","⚙️",)

        if args.strategy == "fsdp" and args.resume and checkpointer is not None and checkpointer.last_training_time is not None:
            print_on_rank_0(rank, "Loading optimizer state from checkpoint...", "♻️")
            try:
                checkpointer.load_optim(model, optimizer)
                print_on_rank_0(rank, "Optimizer state restored ✓")
            except Exception as e:
                print_on_rank_0(rank, f"⚠️ Optimizer state incompatible (model changed?), starting fresh. Reason: {e}", "⚠️")

        elif args.strategy in ("solo", "ddp") and args.resume and args.resume_path:
            status = restore_optimizer_state(optimizer, args, device)
            if status == "restored":
                print_on_rank_0(rank, "Optimizer state restored ✓ (LR schedule restarts from this config)", "♻️")
            elif status == "absent":
                print_on_rank_0(rank, "Checkpoint has no optimizer state (older format): AdamW starts fresh", "⚠️")
            else:
                print_on_rank_0(rank, "Optimizer state does not match the trainable parameters: AdamW starts fresh", "⚠️")
        
        ## Load the dataset and create the dataloader with DistributedSampler for sharding across GPUs
        print_banner_on_rank_0(rank, "PREPARING DATA")
        objective = None   # post-training (sft/dpo/...) replaces the per-step loss
        if args.algorithm != "pretrain":
            from omni_train.post_training.data import get_post_training_data
            from omni_train.post_training.objectives import build_objective
            objective = build_objective(args, tokenizer)
            pt_dataset, pt_collate, dataloader = get_post_training_data(args, tokenizer, rank, world_size)
            if hasattr(objective, "wrap_loader"):   # PPO: one batch -> a rollout, then several gradient steps
                dataloader = objective.wrap_loader(dataloader, model, device)
        else:
            dataloader = get_dataloader(
                args.dataset, 
                args.dataset_full_name, 
                args.dataset_split, 
                tokenizer, 
                rank, 
                world_size, 
                batch_size=args.batch_size,
                max_length=args.max_length, 
                model_type=args.model_type,
                vocab_size=getattr(args, "custom_vocab_size", None),
                custom_max_seq_len=getattr(args, "custom_max_seq_len", None),
                text_field=getattr(args, "dataset_text_field", "text"),
            )

        if hasattr(dataloader, "__len__"):
            total_steps = max(1, len(dataloader) * args.epochs)
            warmup_steps = min(args.warmup_steps, total_steps)

            def _lr_lambda(step_idx: int) -> float:
                return lr_multiplier(step_idx, warmup_steps, total_steps, args.lr_scheduler)

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
            print_on_rank_0(rank, f"LR schedule: {args.lr_scheduler} | warmup_steps={warmup_steps} | total_steps={total_steps}", "📈")

        # print_on_rank_0(rank, f"Counting Model Parameters", )
        print_on_rank_0(rank, "Counting model parameters for training time estimation... (this may take a moment)", "⏳")
        model_total_params = sum(p.numel() for p in model.parameters())
        print_on_rank_0(rank, f"Total model parameters: {model_total_params:,}")

        if rank == 0 and hasattr(dataloader, "__len__"):
            # Print a quick wall-clock estimate from static config knobs for faster experiment planning.
            time_est = estimate_training_time(
                num_params=model_total_params,                 
                steps_per_epoch=len(dataloader),
                epochs=args.epochs,
                batch_size=args.batch_size,
                seq_len=args.custom_max_seq_len if args.model_type == "custom_transformer" else args.max_length,
                num_gpus=world_size,
                gpu_type=torch.cuda.get_device_name(local_rank) if torch.cuda.is_available() else "CPU",
                strategy=args.strategy, 
                peft_enabled=args.peft_enabled,
                peft_r=args.peft_r,
                gradient_checkpointing=args.gradient_checkpointing 
            )
            print_on_rank_0(rank,f"Estimated time | total≈{time_est['human_readable']} | total minutes≈{time_est['total_minutes']} min | total hours≈{time_est['total_hours']} h ","⏱️",)
        
        # Mixed precision for solo/DDP: weights stay fp32 (see distributed_utils._load_dtype),
        # forward/backward run under autocast. FSDP gets this from its MixedPrecisionPolicy.
        amp_dtype = DTYPE_MAP.get(args.param_dtype, torch.float32)
        use_autocast = args.mixed_precision and args.strategy != "fsdp" and amp_dtype != torch.float32
        if use_autocast and device.type == "cpu" and amp_dtype == torch.float16:
            use_autocast = False  # CPU autocast is bf16-only
        # fp16 has a narrow exponent range, so its gradients need loss scaling; bf16 does not.
        scaler = torch.amp.GradScaler(device.type, enabled=use_autocast and amp_dtype == torch.float16)
        if use_autocast:
            print_on_rank_0(rank, f"Autocast enabled | compute dtype={amp_dtype} | fp32 weights{' | GradScaler on' if scaler.is_enabled() else ''}", "⚡")

        def autocast_ctx():
            return torch.autocast(device_type=device.type, dtype=amp_dtype) if use_autocast else nullcontext()

        if objective is not None and objective.prepare(model):
            print_on_rank_0(rank, f"Dropout disabled for {args.algorithm.upper()} (noisy log-probs otherwise)", "🔇")
        if hasattr(objective, "wrap_loader"):
            objective.autocast_ctx = autocast_ctx
        if hasattr(objective, "load_extra") and args.resume:
            start = checkpointer.last_folder() if checkpointer is not None else args.resume_path
            found = bool(start) and objective.load_extra(start)
            print_on_rank_0(rank, "Value head restored ✓" if found else "No saved value head next to the checkpoint: it starts fresh", "♻️")
        if objective is not None and objective.needs_reference:
            print_banner_on_rank_0(rank, "REFERENCE LOG-PROBS")
            print_on_rank_0(rank, "Scoring every pair with the initial model (= the DPO reference); "
                                  "no second model is kept in memory", "🧮")
            objective.precompute_reference(model, pt_dataset, pt_collate, rank, world_size,
                                           args.batch_size, device, autocast_ctx,
                                           log=lambda m: print_on_rank_0(rank, m))
            print_on_rank_0(rank, f"Reference log-probs ready for {len(pt_dataset)} pairs ✓")

        print_banner_on_rank_0(rank, "TRAINING")
        model.train()
        losses = []
        if dist.is_initialized():
            dist_barrier(local_rank)

        for epoch in range(args.epochs):
            print_on_rank_0(rank, f"Starting Epoch {epoch+1}/{args.epochs}", "🔁")
            
            if hasattr(dataloader, "sampler") and isinstance(dataloader.sampler, DistributedSampler):
                dataloader.sampler.set_epoch(epoch)

            total_loss = 0.0
            num_batches = len(dataloader) if hasattr(dataloader, "__len__") else None

            for step, batch in enumerate(dataloader):
                try:                     
                    ## FSDP shards the model parameters across GPUs — after each forward+backward pass           
                    # if args.explicit_prefetching and args.strategy == "fsdp":
                    #     model.unshard() # type: ignore

                    optimizer.zero_grad()
                    metrics = {}
                    with autocast_ctx():
                        if objective is not None:
                            loss, metrics = objective(model, batch, device)
                        elif args.model_type == "custom_transformer":
                            import torch.nn.functional as F
                            tokens = batch[0].to(device)    ## TensorDataset yields (tensor,) tuples
                            logits = model(tokens)          ## [bsz, seq_len, vocab_size]
                            # Next-token prediction loss
                            logits_shift = logits[:, :-1, :].contiguous()
                            labels_shift = tokens[:, 1:].contiguous()
                            loss = F.cross_entropy(
                                logits_shift.view(-1, logits_shift.size(-1)),
                                labels_shift.view(-1),
                            )
                        elif args.model_type == "vision":
                            pixel_values = batch["pixel_values"].to(device)
                            labels = batch["labels"].to(device)
                            outputs = model(pixel_values=pixel_values, labels=labels)
                            loss = outputs.loss
                        elif args.model_type == "yolo":
                            # Detection labels are a list of per-image dicts (boxes,
                            # class_labels, …) — move each tensor to the device but
                            # keep the list structure the model expects.
                            pixel_values = batch["pixel_values"].to(device)
                            labels = [
                                {k: v.to(device) for k, v in target.items()}
                                for target in batch["labels"]
                            ]
                            det_kwargs = {"pixel_values": pixel_values, "labels": labels}
                            if "pixel_mask" in batch:
                                det_kwargs["pixel_mask"] = batch["pixel_mask"].to(device)
                            outputs = model(**det_kwargs)
                            loss = outputs.loss
                        elif args.model_type == "vlm":
                            # Processor already produced input_ids, attention_mask,
                            # pixel_values, (pixel_attention_mask) and labels. Move
                            # tensors to the device; pass any non-tensor entries
                            # (some processors emit lists) through untouched.
                            model_inputs = {
                                k: (v.to(device) if torch.is_tensor(v) else v)
                                for k, v in batch.items()
                            }
                            outputs = model(**model_inputs)
                            loss = outputs.loss
                        else:
                            input_ids = batch["input_ids"].to(device)
                            labels = batch["labels"].to(device)
                            attention_mask = batch["attention_mask"].to(device)
                            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                            loss = outputs.loss

                    scaler.scale(loss).backward()

                    if args.grad_clip > 0:
                        scaler.unscale_(optimizer)  # clip the true gradients, not the scaled ones
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                    scaler.step(optimizer)
                    if hasattr(objective, "step_extra"):   # PPO's value head
                        objective.step_extra(scaler, args.grad_clip)
                    scaler.update()
                    if scheduler is not None:
                        scheduler.step()
                    total_loss += loss.item()

                    if rank == 0:
                        avg_loss = total_loss / (step + 1)
                        pct = (step + 1) / num_batches * 100 if num_batches is not None else 0
                        bar_len = 30
                        filled = int(bar_len * pct / 100)
                        bar = "█" * filled + "░" * (bar_len - filled)
                        # CHANGED: "batch_loss:" → "loss:" (saves 6 chars, prevents terminal truncation).
                        # ADDED: \033[K (erase to end of line) so leftover chars from a previously
                        # longer line (e.g. step 9 → step 10 adds one digit) are cleared cleanly.
                        # Old code: f"... | batch_loss: {loss.item():.4f} | avg: {avg_loss:.4f}"
                        step_w = len(str(num_batches))  # fixed-width step counter, e.g. " 1/14"
                        line = (
                            f"   Epoch {epoch+1}/{args.epochs} [{bar}] {pct:5.1f}% | "
                            f"step {step+1:>{step_w}}/{num_batches} | "
                            f"loss: {loss.item():.4f} | avg: {avg_loss:.4f}"
                            + "".join(f" | {k}: {v:.3f}" for k, v in metrics.items())
                        )
                        print(f"\r{line}\033[K", end="", flush=True)
                        
                except Exception as e:
                    print_on_rank_0(rank, f"❌ Failed in training step {step+1}: {e}", "❌")
                    raise
            
            epoch_loss = total_loss / max(num_batches, 1) if num_batches is not None else total_loss
            losses.append(epoch_loss)
                            
            if rank == 0:
                print_on_rank_0(rank, f"Epoch {epoch+1} complete | avg loss: {epoch_loss:.4f}", "✅")

            if args.wandb_log_with_train and rank == 0:
                run.log({"epoch": epoch+1, "loss": epoch_loss}) # type: ignore

        def in_terminal():
            return sys.stdout.isatty()
        
        if rank == 0 and in_terminal():
            plot_losses_in_terminal(losses) # not in UI only in terminal, but gives a nice visual of loss trend after training completes.

        
        if not args.save:
            print_on_rank_0(rank, "Checkpoint saving skipped (--save not specified)", "⚠️")
        else:
            saved_to = save_checkpoint(
                args.strategy, 
                model, 
                optimizer, 
                rank, 
                args, 
                checkpointer=checkpointer   # type: ignore
            )
            if hasattr(objective, "save_extra") and rank == 0:
                objective.save_extra(saved_to)

        if args.export_hf:
            from omni_train.export import export_hf
            print_banner_on_rank_0(rank, "EXPORTING (HUGGING FACE FORMAT)")
            out_dir = export_hf(model, tokenizer, args, rank, args.strategy)
            if rank == 0:
                note = (" (adapter only: merge with scripts/merge_lora.py)" if args.quantization_enabled else "")
                print_on_rank_0(rank, f"HF export written to {out_dir}{note} ✓", "📦")

        if dist.is_initialized():
            print_on_rank_0(rank, "Process group is being destroyed. All done!", "👋")
        cleanup()
        
        if args.wandb_log_with_train and rank == 0:
            run.finish() #type: ignore

    except ValueError as e:
        # ADDED: separate handler for user config errors (checkpoint compat, bad args, etc.)
        # These are not unexpected crashes — print once cleanly without a traceback, then
        # re-raise so __main__ can convert to sys.exit(1).
        # Old code: all exceptions fell into one handler that always printed a full traceback.
        _rank = dist.get_rank() if dist.is_initialized() else 0
        if _rank == 0:
            print(f"\n❌  {e}\n", flush=True)
        cleanup()
        raise
    except Exception as e:
        _rank = dist.get_rank() if dist.is_initialized() else 0
        print(f"\n[rank {_rank}] ❌ Training failed: {e}", flush=True)
        traceback.print_exc()
        cleanup()
        raise


if __name__ == "__main__":
    cfg = load_config()
    try:
        if advisor.is_auto(cfg):
            build_args(cfg)     # a mistake in the config must not wait for the dataset to be measured
            cfg = advisor.resolve_for_process(cfg)
        args = build_args(cfg)
    except ValueError as e:
        rank = int(os.environ.get("RANK", "0"))
        if rank == 0:
            print(f"\n❌  Configuration error: {e}\n")
        sys.exit(1)

    try: 
        if int(os.environ.get("RANK", "0")) == 0:
            print_config(args)

    except ValueError as e:
        rank = 0
        print_on_rank_0(rank, f"Could not parse RANK env variable, defaulting to 0 for config print. Error: {e}", "⚠️")

    if int(os.environ.get("RANK", "0")) == 0:
        try:
            warning = advisor.preflight(cfg, int(os.environ.get("WORLD_SIZE", "1")))
        except Exception:       # an estimate must never stop a run
            warning = None
        if warning:
            print(f"\n⚠️  {warning}\n", flush=True)

    # ADDED: catch ValueError from main() (checkpoint compat errors, bad resume configs, etc.)
    # main() already prints the message once cleanly and re-raises — we just convert to sys.exit(1).
    # Old code: ValueError from main() was unhandled here, printing a raw Python traceback.
    try:
        main(args)
    except ValueError:
        sys.exit(1)