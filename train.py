## train.py
## This script is built according to the PyTorch documentation and tutorials for FSDP2 : 
## https://github.com/pytorch/examples/tree/main/distributed/FSDP2


# ----------------------------------------------------------------------
# Import Libraries 
# ----------------------------------------------------------------------

import os
from dotenv import load_dotenv 
import sys
# import argparse
# import yaml
# import time
import shutil
import warnings
from transformers import AutoTokenizer
import torch
# import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DistributedSampler 

from model import Transformer 
from checkpoint import Checkpointer
from data import get_dataloader
from utils import (
    plot_losses_in_terminal,
    print_config,
    estimate_training_time,
)
from distributed_utils import (
    BACKEND,
    print_on_rank_0,
    print_banner_on_rank_0,
    print_on_all_ranks,
    gpu_memory_snapshot,
    setup_dist_process_group,
    cleanup,
    apply_solo,
    apply_ddp,
    apply_fsdp,
    save_checkpoint
)
import wandb

from utils import build_args, load_config

import transformers
transformers.logging.disable_progress_bar()

## If 3d parallelism is enabled
# from parallelism import ParallelismArgs, setup_device_mesh
# from parallelism import ParallelismArgs

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

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main(args):
    """Main training loop for distributed training. 
    Initializes distributed environment, loads model and data, and runs training epochs."""
    try:
        checkpointer = None
       
        ## 1. SETUP DISTRIBUTED ENVIRONMENT
        if args.strategy in ["ddp", "fsdp"]:
            local_rank = setup_dist_process_group()
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
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
            run = wandb.init(
                project=args.wandb_project,
                config=vars(args),
                tensorboard=True 
        )
            
        ## Print environment info for debugging and verification
        if args.strategy in ["ddp", "fsdp"]:
            print_on_rank_0(rank, f"backend={BACKEND}", "✅")
        print_on_all_ranks(rank, f"Process joined | world_size={world_size} | pid={os.getpid()}", "🚀",
                           local_rank=local_rank, device=device)
        if dist.is_initialized(): dist.barrier()
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(local_rank)
            gpu_mem  = torch.cuda.get_device_properties(local_rank).total_memory / 1e9
            print_on_all_ranks(rank, f"GPU: {gpu_name} ({gpu_mem:.1f} GB) | {gpu_memory_snapshot(device)}", "🖥️",
                               local_rank=local_rank, device=device)
        if dist.is_initialized(): 
            dist.barrier()
 
        # --------------------------------------------------------------
        # 2. TOKENIZER
        # --------------------------------------------------------------
        tokenizer = None
        if args.model_type == "custom_transformer":
            print_on_rank_0(rank, "Custom Transformer: skipping tokenizer (synthetic data will be used)", "⏩")
        elif args.model_type in {"llm", "seq2seq", "encoder", "vlm"}:
            print_banner_on_rank_0(rank, "LOADING TOKENIZER")
            print_on_rank_0(rank, f"Fetching tokenizer: {args.model_name}", "🔤")
            tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=HF_TOKEN)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            print_on_rank_0(rank, "Tokenizer ready ✓")
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

        ## Create the optimizer (AdamW) for trainable model parameters.
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
        print_on_rank_0(
            rank,
            f"Optimizer: AdamW | lr={args.learning_rate} | weight_decay={args.weight_decay} | grad_clip={args.grad_clip}",
            "⚙️",
        )

        if args.strategy == "fsdp" and args.resume and checkpointer is not None and checkpointer.last_training_time is not None:
            print_on_rank_0(rank, "Loading optimizer state from checkpoint...", "♻️")
            try:
                checkpointer.load_optim(model, optimizer)
                print_on_rank_0(rank, "Optimizer state restored ✓")
            except Exception as e:
                print_on_rank_0(rank, f"⚠️ Optimizer state incompatible (model changed?), starting fresh. Reason: {e}", "⚠️")

        elif args.strategy == "ddp":
            print_on_rank_0(rank, "Optimizer state will not be loaded since checkpointing is not implemented for DDP in this example.", "⚠️")    
        
        ## Load the dataset and create the dataloader with DistributedSampler for sharding across GPUs
        print_banner_on_rank_0(rank, "PREPARING DATA")
        dataloader = get_dataloader(
            args.dataset, args.dataset_full_name, args.dataset_split, 
            tokenizer, rank, world_size, batch_size=args.batch_size,
            max_length=args.max_length, model_type=args.model_type,
            vocab_size=getattr(args, "custom_vocab_size", None),
        )

        if hasattr(dataloader, "__len__"):
            total_steps = max(1, len(dataloader) * args.epochs)

            def _warmup_lambda(step_idx: int) -> float:
                if args.warmup_steps <= 0:
                    return 1.0
                return min(1.0, float(step_idx + 1) / float(max(1, args.warmup_steps)))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_warmup_lambda)
            print_on_rank_0(rank, f"LR warmup enabled: warmup_steps={args.warmup_steps}, total_steps={total_steps}", "📈")

        # print_on_rank_0(rank, f"Counting Model Parameters", )
        print_on_rank_0(
                rank,
                f"Counting model parameters for training time estimation... (this may take a moment)"
                "⏳",
            )
        model_total_params = sum(p.numel() for p in model.parameters())

        if rank == 0 and hasattr(dataloader, "__len__"):
            # Print a quick wall-clock estimate from static config knobs for faster experiment planning.
            time_est = estimate_training_time(
                num_params=model_total_params,                 
                steps_per_epoch=len(dataloader),
                epochs=args.epochs,
                batch_size=args.batch_size,
                num_gpus=world_size,
                gpu_type=torch.cuda.get_device_name(local_rank) if torch.cuda.is_available() else "CPU",
                strategy=args.strategy, 
                peft_enabled=args.peft_enabled,
                peft_r=args.peft_r,
                gradient_checkpointing=args.gradient_checkpointing 
            )
            print_on_rank_0(
                rank,
                f"Estimated time | total≈{time_est['human_readable']} | total minutes≈{time_est['total_minutes']} min | total hours≈{time_est['total_hours']} h ",
                "⏱️",
            )
        
        print_banner_on_rank_0(rank, "TRAINING")
        model.train()
        losses = []

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
                    if args.model_type == "custom_transformer":
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
                    elif args.model_type in {"vision", "yolo"}:
                        pixel_values = batch["pixel_values"].to(device)
                        labels = batch["labels"].to(device)
                        outputs = model(pixel_values=pixel_values, labels=labels)
                        loss = outputs.loss
                    else:
                        input_ids = batch["input_ids"].to(device)
                        labels = batch["labels"].to(device)
                        attention_mask = batch["attention_mask"].to(device)
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        loss = outputs.loss

                    loss.backward()

                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
                    
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    total_loss += loss.item()

                    if rank == 0:
                        avg_loss = total_loss / (step + 1)
                        pct = (step + 1) / num_batches * 100 if num_batches is not None else 0
                        bar_len = 30
                        filled = int(bar_len * pct / 100)
                        bar = "█" * filled + "░" * (bar_len - filled)
                        print(f"\r   Epoch {epoch+1}/{args.epochs} [{bar}] {pct:5.1f}% | "
                              f"step {step+1}/{num_batches} | batch_loss: {loss.item():.4f} | "
                              f"avg: {avg_loss:.4f}", end="", flush=True)
                        
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
            save_checkpoint(
                args.strategy, 
                model, 
                optimizer, 
                rank, 
                args, 
                checkpointer=checkpointer   # type: ignore
            )

        ## delete pretrained_seed folder
        if rank == 0:
            pretrained_seed_path = os.path.join(args.checkpoint_dir, "pretrained_seed")
            if os.path.exists(pretrained_seed_path):
                shutil.rmtree(pretrained_seed_path)
                print_on_rank_0(rank, "Pretrained seed folder deleted", "🧹")

        if dist.is_initialized():
            print_on_rank_0(rank, "Process group is being destroyed. All done!", "👋")
        cleanup()
        
        if args.wandb_log_with_train and rank == 0: 
            run.finish() #type: ignore

    except Exception as e:
        rank = dist.get_rank() if dist.is_initialized() else 0
        print_on_rank_0(rank, f"❌ Training failed: {e}", "❌")
        cleanup()
        raise


if __name__ == "__main__":
    cfg = load_config()
    args = build_args(cfg)

    try: 
        if int(os.environ.get("RANK", "0")) == 0:
            print_config(args)
    except ValueError:
        rank = 0
        print_on_rank_0(rank, "Could not parse RANK env variable, defaulting to 0 for config print. Error:", "⚠️")
    main(args)