## dist_kit/train.py ##
##
## End-to-end training entry point: wraps an HF model with the chosen strategy
## and trains on the configured HF text dataset (dataset: section). If no
## dataset name is configured, falls back to a few steps on random token
## batches so the kit can be smoke-tested with no downloads.
##
## Run from your project root (the folder containing dist_kit/):
##
##   python -m dist_kit                                        # solo
##   torchrun --nproc_per_node=2 -m dist_kit.train             # ddp / fsdp
##   bash dist_kit/launch.sh                                   # reads config.yaml
##
## Config resolution (same nested structure as OMNI-Train's config.yaml):
##   $DIST_CONFIG > ./config.yaml (project root) > dist_kit/config.yaml (bundled)

import os
import shutil

import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler
from dotenv import load_dotenv

from .config import DistConfig, resolve_config_path
from .dist_setup import (
    setup_dist_process_group,
    cleanup,
    dist_barrier,
    print_on_rank_0,
    print_banner_on_rank_0,
    print_on_all_ranks,
)
from .strategies import apply_solo, apply_ddp, apply_fsdp, save_checkpoint
from .data import get_text_dataloader

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Random-batch fallback knobs (used only when no dataset: is configured)
DEMO_STEPS = int(os.environ.get("DEMO_STEPS", "10"))


def load_dist_config() -> DistConfig:
    path = resolve_config_path()
    if os.path.exists(path):
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"   Config: {path}", flush=True)
        return DistConfig.from_yaml(path)
    print(f"   No config found at '{path}' — using DistConfig defaults (tiny demo model)", flush=True)
    return DistConfig(strategy="solo" if int(os.environ.get("WORLD_SIZE", "1")) == 1 else "fsdp")


def _demo_batches(args, device, vocab_size):
    """Random token batches — smoke-test fallback when no dataset is configured."""
    for _ in range(DEMO_STEPS):
        input_ids = torch.randint(0, vocab_size, (args.batch_size, args.max_length), device=device)
        yield {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": input_ids,
        }


def main():
    args = load_dist_config()
    checkpointer = None

    try:
        ## 1. Distributed environment
        if args.strategy in ("ddp", "fsdp"):
            local_rank = setup_dist_process_group()
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        else:
            local_rank, rank, world_size = 0, 0, 1
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print_on_rank_0(rank, f"Running in solo mode on device={device}", "🧪")

        print_on_all_ranks(rank, f"Process joined | world_size={world_size} | pid={os.getpid()}", "🚀",
                           local_rank=local_rank, device=device)

        ## 2. Tokenizer (text types; needed before the dataloader)
        tokenizer = None
        if args.dataset_name:
            from transformers import AutoTokenizer
            print_banner_on_rank_0(rank, "LOADING TOKENIZER")
            tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=HF_TOKEN)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            print_on_rank_0(rank, "Tokenizer ready ✓")

        ## 3. Model
        print_banner_on_rank_0(rank, "LOADING MODEL")
        if args.strategy == "ddp":
            model = apply_ddp(local_rank, rank, device, args)
        elif args.strategy == "fsdp":
            model, checkpointer = apply_fsdp(local_rank, rank, device, args)
        else:
            model = apply_solo(device, rank, args)

        # In PEFT/QLoRA runs, most base weights are frozen; optimize only trainable tensors.
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError("No trainable parameters were found. Check PEFT/quantization configuration.")
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        print_on_rank_0(rank, f"Optimizer: AdamW | lr={args.learning_rate} | weight_decay={args.weight_decay}", "⚙️")

        if args.strategy == "fsdp" and args.resume and checkpointer is not None and checkpointer.last_training_time is not None:
            print_on_rank_0(rank, "Loading optimizer state from checkpoint...", "♻️")
            try:
                checkpointer.load_optim(model, optimizer)
                print_on_rank_0(rank, "Optimizer state restored ✓")
            except Exception as e:
                print_on_rank_0(rank, f"⚠️ Optimizer state incompatible (model changed?), starting fresh. Reason: {e}", "⚠️")

        ## 4. Data
        print_banner_on_rank_0(rank, "PREPARING DATA")
        scheduler = None
        if args.dataset_name:
            dataloader = get_text_dataloader(args, tokenizer, rank, world_size)

            def _warmup_lambda(step_idx: int) -> float:
                if args.warmup_steps <= 0:
                    return 1.0
                return min(1.0, float(step_idx + 1) / float(max(1, args.warmup_steps)))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_warmup_lambda)
            print_on_rank_0(rank, f"LR warmup enabled: warmup_steps={args.warmup_steps}", "📈")
        else:
            print_on_rank_0(rank, f"No dataset configured — random-batch demo ({DEMO_STEPS} steps/epoch)", "🎲")
            dataloader = None

        ## 5. Train
        print_banner_on_rank_0(rank, "TRAINING")
        vocab_size = getattr(getattr(model, "config", None), "vocab_size", None) or 1000
        model.train()
        if dist.is_initialized():
            dist_barrier(local_rank)

        for epoch in range(args.epochs):
            print_on_rank_0(rank, f"Starting Epoch {epoch+1}/{args.epochs}", "🔁")
            if dataloader is not None and isinstance(getattr(dataloader, "sampler", None), DistributedSampler):
                dataloader.sampler.set_epoch(epoch)

            batches = dataloader if dataloader is not None else _demo_batches(args, device, vocab_size)
            num_batches = len(dataloader) if dataloader is not None else DEMO_STEPS
            total_loss = 0.0

            for step, batch in enumerate(batches):
                optimizer.zero_grad()
                model_inputs = {
                    k: (v.to(device) if torch.is_tensor(v) else v)
                    for k, v in batch.items()
                }
                outputs = model(**model_inputs)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                total_loss += loss.item()
                print_on_rank_0(rank, f"epoch {epoch+1}/{args.epochs} | step {step+1}/{num_batches} | loss: {loss.item():.4f}")

            print_on_rank_0(rank, f"Epoch {epoch+1} complete | avg loss: {total_loss / max(1, num_batches):.4f}", "✅")

        ## 6. Save
        if args.save:
            save_checkpoint(args.strategy, model, optimizer, rank, args, checkpointer=checkpointer)

        # Clean up the rank-0 seed weights written by fresh FSDP runs.
        if rank == 0:
            seed_path = os.path.join(args.checkpoint_dir, "pretrained_seed")
            if os.path.exists(seed_path):
                shutil.rmtree(seed_path)
                print_on_rank_0(rank, "Pretrained seed folder deleted", "🧹")

        if dist.is_initialized():
            print_on_rank_0(rank, "Process group is being destroyed. All done!", "👋")
        cleanup()

    except Exception:
        cleanup()
        raise


if __name__ == "__main__":
    main()
