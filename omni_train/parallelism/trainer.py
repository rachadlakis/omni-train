"""Training loop for strategy: 5d (any mix of DP, TP+SP, CP, PP, EP, ZeRO-1).

Called from train.py:  if args.strategy == "5d": train_5d(args)

One optimizer step:
  1. BlockSampler gives this rank `micro_batches` micro-batches (CP ranks keep their slice).
  2. PipelineEngine runs AFAB or 1F1B. With pp=1 it is plain gradient accumulation.
  3. During the last backward, DP hooks launch bucketed all-reduces (overlap).
  4. finish_grad_sync waits for them (+ SP sum over TP, + tied-embedding sum over PP).
  5. Global grad-norm clipping, then optimizer.step() (ZeRO-1: owner updates, then broadcast).
"""

from __future__ import annotations

import os
import socket
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict

import torch
import torch.distributed as dist
import torch.nn.functional as F

from . import checkpoint as ckpt
from .data import BlockSampler, synthetic_blocks, text_blocks
from .data_parallel import DataParallel
from .model import ModelConfig, PipelineStage
from .optim import clip_grad_norm, set_lr
from .pipeline_parallel import PipelineEngine
from .process_groups import ParallelContext
from .zero import ZeRO1AdamW

SYNTHETIC_SAMPLES = 2000


def _ensure_dist_env():
    """`python train.py` with strategy: 5d -> run as a 1-process group."""
    if "WORLD_SIZE" in os.environ:
        return
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    os.environ.update(RANK="0", LOCAL_RANK="0", WORLD_SIZE="1",
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))


def _custom_config(args) -> ModelConfig:
    dim = args.custom_dim
    ffn = args.custom_ffn_hidden or 64 * -(-int(8 * dim / 3) // 64)  # ~8/3 * dim, multiple of 64
    return ModelConfig(vocab_size=args.custom_vocab_size, hidden=dim, n_layers=args.custom_n_layers,
                       n_heads=args.custom_n_heads, n_kv_heads=args.custom_n_kv_heads,
                       ffn_hidden=ffn, num_experts=args.custom_num_experts, top_k=args.custom_top_k)


def count_params(stage: PipelineStage, pc: ParallelContext) -> int:
    """Total parameters of the whole model (each counted once across ranks)."""
    sharded = replicated = expert = 0
    for p in stage.parameters():
        if getattr(p, "tied_copy", False):
            continue
        if getattr(p, "tp_sharded", False):
            sharded += p.numel()
        elif getattr(p, "is_expert", False):
            expert += p.numel()
        else:
            replicated += p.numel()
    t = torch.tensor([sharded * pc.tp_size + replicated + expert * pc.ep_size], dtype=torch.float64)
    if dist.get_backend() == "nccl":
        t = t.cuda()
    dist.all_reduce(t, group=pc.pp_group)
    return int(t.item())


class FiveDRun:
    def __init__(self, args):
        self.args = args

    # ------------------------------------------------------------------ setup
    def setup(self):
        from omni_train.distributed_utils import print_banner_on_rank_0, print_on_rank_0, setup_dist_process_group

        a = self.args
        _ensure_dist_env()
        local_rank = setup_dist_process_group()
        self.device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
        world = dist.get_world_size()
        model_par = a.par_tp * a.par_cp * a.par_pp
        if world % model_par:
            raise ValueError(f"world size {world} is not divisible by tp*cp*pp = {model_par}")
        dp = a.par_dp or world // model_par
        pc = self.pc = ParallelContext(tp=a.par_tp, cp=a.par_cp, pp=a.par_pp, dp=dp, ep=a.par_ep,
                                       sequence_parallel=a.par_sp)
        self.say = lambda msg, emoji="": print_on_rank_0(pc.rank, msg, emoji)
        print_banner_on_rank_0(pc.rank, "5D PARALLELISM")
        self.say(f"world={world} | dp={dp} tp={pc.tp_size} sp={pc.sequence_parallel} cp={pc.cp_size} "
                 f"pp={pc.pp_size} ({a.par_pp_schedule}) ep={pc.ep_size} zero1={a.par_zero1}", "🧊")

        # ---- model
        self.tokenizer, self.model_dir = None, None
        if a.model_type == "custom_transformer":
            cfg = _custom_config(a)
            self.seq_len = a.custom_max_seq_len
            stage = PipelineStage(cfg, pc, seeded=True, gradient_checkpointing=a.gradient_checkpointing)
            self.say(f"Custom transformer: {cfg.n_layers} layers, hidden {cfg.hidden}"
                     f"{f', {cfg.num_experts} experts (top-{cfg.top_k})' if cfg.num_experts else ''}", "🧱")
        else:
            from .hf import config_from_hf, load_hf_weights, resolve_model_dir
            self.model_dir = resolve_model_dir(a.model_name, pc.rank)
            cfg = config_from_hf(self.model_dir, a.max_length)
            self.seq_len = a.max_length
            stage = PipelineStage(cfg, pc, seeded=not a.load_model_from_hf,
                                  gradient_checkpointing=a.gradient_checkpointing)
            if a.load_model_from_hf:
                self.say(f"Loading {a.model_name} ({cfg.hf_model_type}) shard by shard", "📥")
                load_hf_weights(stage, self.model_dir)
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            except Exception:
                if a.dataset != "synthetic":
                    raise
        self.cfg = cfg
        stage.to(self.device)
        self.stage = stage
        self.model = DataParallel(stage, pc)
        self.n_params = count_params(stage, pc)
        local = sum(p.numel() for p in stage.parameters())
        self.say(f"Parameters: {self.n_params:,} total | {local:,} on rank 0", "📊")

        # ---- optimizer
        adam = dict(lr=a.learning_rate, betas=(0.9, 0.95), weight_decay=a.weight_decay)
        params = list(self.model.parameters())
        self.optimizer = ZeRO1AdamW(params, pc, **adam) if a.par_zero1 else torch.optim.AdamW(params, **adam)
        self.layout = ckpt.layout_of(pc, a.par_zero1)
        if a.resume:
            run_dir = a.resume_path or ckpt.latest(a.checkpoint_dir)
            if run_dir:
                self.say(f"Resuming from {run_dir}", "♻️")
                ckpt.load(run_dir, pc, self.model, self.optimizer, self.layout)
            else:
                self.say("resume: no 5d checkpoint found, starting fresh", "⚠️")

        # ---- data
        print_banner_on_rank_0(pc.rank, "PREPARING DATA")
        if a.model_type == "custom_transformer" or a.dataset == "synthetic":
            blocks = synthetic_blocks(SYNTHETIC_SAMPLES, self.seq_len, cfg.vocab_size, seed=a.seed)
            self.say(f"Synthetic data | {SYNTHETIC_SAMPLES} sequences of {self.seq_len} tokens", "🎲")
        else:
            blocks = text_blocks(a, self.tokenizer, pc.rank)
        self.sampler = BlockSampler(blocks, pc, a.batch_size, a.par_micro_batches, seed=a.seed)
        self.say(f"Global batch = {a.batch_size} x {a.par_micro_batches} micro-batches x {dp} dp = "
                 f"{self.sampler.global_batch} sequences | {self.sampler.steps_per_epoch} steps/epoch", "🧺")

        # ---- precision + pipeline engine
        use_amp = a.mixed_precision and a.param_dtype in ("bfloat16", "float16")
        if use_amp and a.param_dtype == "float16":
            raise ValueError("strategy: 5d supports bfloat16 mixed precision only (set param_dtype: bfloat16)")
        self.autocast = (lambda: torch.autocast(self.device.type, dtype=torch.bfloat16)) if use_amp else nullcontext
        if use_amp:
            self.say("Autocast bf16 | fp32 weights and optimizer states", "⚡")
        s_act = self.sampler.local_seq // (pc.tp_size if pc.sequence_parallel else 1)
        vocab = cfg.vocab_size

        def loss_fn(logits, batch):
            return F.cross_entropy(logits.float().reshape(-1, vocab), batch["targets"].reshape(-1))

        self.engine = PipelineEngine(self.model, pc, (a.batch_size, s_act, cfg.hidden), torch.float32,
                                     self.device, loss_fn, schedule=a.par_pp_schedule, autocast=self.autocast)
        self.model.train()
        return self

    # ------------------------------------------------------------------- step
    def forward_backward(self, epoch: int, step: int):
        """Returns this step's loss averaged over DP x CP (on last-stage ranks; None elsewhere)."""
        batches = self.sampler.micro_batches(epoch, step, self.device)
        self.optimizer.zero_grad()
        loss = self.engine.train_step(batches)
        self.model.finish_grad_sync()
        if self.pc.is_last_stage:
            loss = loss.detach().clone()
            dist.all_reduce(loss, group=self.pc.dp_cp_group)
            return loss.item() / dist.get_world_size(self.pc.dp_cp_group)
        return None

    def clip_and_step(self, lr: float) -> float:
        norm = clip_grad_norm(list(self.model.parameters()), self.args.grad_clip, self.pc)
        set_lr(self.optimizer, lr)
        self.optimizer.step()
        return norm

    # ------------------------------------------------------------------- loop
    def train(self):
        from omni_train.distributed_utils import cleanup, print_banner_on_rank_0
        from omni_train.utils import lr_multiplier, plot_losses_in_terminal

        a, pc = self.args, self.pc
        steps = self.sampler.steps_per_epoch
        total = steps * a.epochs
        warmup = min(a.warmup_steps, total)
        logger = pc.is_logging_rank
        run = None
        if a.wandb_log_with_train and logger:
            import wandb
            wandb.login(key=os.getenv("WANDB_API_KEY"))
            run = wandb.init(project=a.wandb_project, entity=a.wandb_entity or None,
                             name=a.wandb_run_name or None, config=vars(a))
        self.say(f"LR schedule: {a.lr_scheduler} | warmup_steps={warmup} | total_steps={total}", "📈")
        print_banner_on_rank_0(pc.rank, "TRAINING")
        tokens = self.sampler.global_batch * self.seq_len
        epoch_losses, g = [], 0
        for epoch in range(a.epochs):
            total_loss = 0.0
            for step in range(steps):
                t0 = time.time()
                loss = self.forward_backward(epoch, step)
                norm = self.clip_and_step(a.learning_rate * lr_multiplier(g, warmup, total, a.lr_scheduler))
                g += 1
                if logger:
                    total_loss += loss
                    if self.device.type == "cuda":
                        torch.cuda.synchronize()
                    tps = tokens / max(time.time() - t0, 1e-9)
                    pct = 100 * (step + 1) / steps
                    bar = "█" * int(30 * pct / 100) + "░" * (30 - int(30 * pct / 100))
                    line = (f"   Epoch {epoch + 1}/{a.epochs} [{bar}] {pct:5.1f}% | step {step + 1}/{steps} | "
                            f"loss: {loss:.4f} | avg: {total_loss / (step + 1):.4f} | "
                            f"grad norm: {norm:.3f} | {tps:,.0f} tok/s")
                    print(f"\r{line}\033[K", end="", flush=True)
                    if run is not None:
                        run.log({"loss": loss, "grad_norm": norm, "tokens_per_s": tps, "step": g})
            if logger:
                epoch_losses.append(total_loss / steps)
                print(f"\n   ✅ Epoch {epoch + 1} complete | avg loss: {epoch_losses[-1]:.4f}", flush=True)
                if run is not None:
                    run.log({"epoch": epoch + 1, "epoch_loss": epoch_losses[-1]})
        if logger and sys.stdout.isatty():
            plot_losses_in_terminal(epoch_losses)

        if a.save:
            self.save()
        else:
            self.say("Checkpoint saving skipped (save: false)", "⚠️")
        if run is not None:
            run.finish()
        self.say("Process group is being destroyed. All done!", "👋")
        cleanup()
        return epoch_losses

    def save(self):
        run_dir = ckpt.new_run_dir(self.args.checkpoint_dir)
        meta = {"layout": self.layout, "model_name": self.args.model_name,
                "model_type": self.args.model_type,
                "model": {k: v for k, v in asdict(self.cfg).items() if k != "inv_freq"}}
        ckpt.save(run_dir, self.pc, self.model, self.optimizer, meta)
        self.say(f"Sharded checkpoint saved: {run_dir}", "💾")
        if self.model_dir is not None:
            from .hf import export_hf
            export_hf(self.stage, self.model_dir, os.path.join(run_dir, "hf"), self.tokenizer)
            self.say(f"Hugging Face export: {os.path.join(run_dir, 'hf')}", "🤗")


def train_5d(args):
    return FiveDRun(args).setup().train()
