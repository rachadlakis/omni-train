"""GRPO: online RL from rewards on sampled completions (Shao et al. 2024, DeepSeekMath).

Every training step, inside the normal train.py loop:

  1. rollout   sample G completions for each prompt of the local batch with the current
               policy (eval mode, no grad, KV cache), decode them and score them with the
               reward functions; advantages are the rewards normalised within each
               prompt's group, so no value model is needed.
  2. loss      one forward with gradients over prompt + completion; clipped
               policy-gradient loss on the completion tokens (+ beta * KL to the start
               model when beta > 0).

Groups never span ranks: each rank samples all G completions of its own prompts, so the
advantages need no communication and DDP / FSDP stay exact (tests/test_post_training_dist.py).

One gradient step per rollout (TRL's num_iterations = 1), so the importance ratio is 1 at
the step and old log-probs are the detached current ones; the clipped form is kept so that
reusing a rollout for several steps is a small change later.

KL reference: with LoRA it is the same model with the adapters disabled (no second model).
"""

from __future__ import annotations

import inspect
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.distributed as dist

from .losses import grpo_loss, group_advantages, selective_logps
from .objectives import disable_dropout
from .rewards import build_rewards


def _unwrap(model):
    """The module that owns .generate(): DDP -> .module; FSDP2 and PEFT wrap in place."""
    return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model


def _forward_params(model) -> set:
    m = _unwrap(model)
    base = m.get_base_model() if hasattr(m, "get_base_model") else m
    try:
        return set(inspect.signature(base.forward).parameters)
    except (TypeError, ValueError):
        return set()


@contextmanager
def _adapters_disabled(model):
    m = _unwrap(model)
    if not hasattr(m, "disable_adapter"):
        raise RuntimeError("grpo with beta > 0 needs a PEFT (LoRA) model for its reference")
    with m.disable_adapter():
        yield


def _drop_autocast_cache():
    """Autocast caches the low-precision copy of every trainable leaf weight it casts, until
    the outermost autocast region exits, and train.py wraps the whole step in one region.
    Copies made during sampling (no grad) are detached from the weights, so the training
    forward would reuse them and LoRA would get no gradient (and gradient checkpointing would
    fail with CheckpointError). Clearing the cache after the no-grad work prevents that."""
    torch.clear_autocast_cache()


@dataclass
class Rollout:
    """Everything the loss needs, for N = prompts x G completions."""
    input_ids: torch.Tensor        # [N, P + C] left-padded prompt + completion
    attention_mask: torch.Tensor   # [N, P + C] 0 on prompt padding and after the completion's EOS
    completion_mask: torch.Tensor  # [N, C] 1 on completion tokens up to and including EOS
    advantages: torch.Tensor       # [N]
    ref_logps: torch.Tensor | None = None  # [N, C] when beta > 0
    metrics: dict | None = None

    @property
    def completion_ids(self):
        return self.input_ids[:, -self.completion_mask.shape[1]:]


def completion_mask_from(ids: torch.Tensor, eos_ids) -> torch.Tensor:
    """1 up to and including the first EOS of each row, 0 after it."""
    is_eos = torch.zeros_like(ids, dtype=torch.bool)
    for e in eos_ids:
        is_eos |= ids == e
    after = (is_eos.cumsum(dim=1) - is_eos.long()) > 0   # strictly after the first EOS
    return (~after).long()


def build_rollout(prompt_ids, prompt_mask, completion_ids, completion_mask, rewards, group_size,
                  scale_rewards=True) -> Rollout:
    """Assemble a rollout from prompts [N, P] (left-padded) and completions [N, C]."""
    adv = group_advantages(rewards, group_size, scale=scale_rewards)
    return Rollout(
        input_ids=torch.cat([prompt_ids, completion_ids], dim=1),
        attention_mask=torch.cat([prompt_mask, completion_mask], dim=1),
        completion_mask=completion_mask,
        advantages=adv.to(prompt_ids.device),
    )


class GRPOObjective:
    needs_reference = False      # no DPO-style precompute; the KL reference is on the fly

    def __init__(self, args, tokenizer):
        self.tok = tokenizer
        self.G = args.algo_num_generations
        self.max_new = args.algo_max_completion_length
        self.temperature = args.algo_temperature
        self.top_p = args.algo_top_p
        self.top_k = args.algo_top_k
        self.beta = args.algo_beta
        self.eps, self.eps_high = args.algo_epsilon, args.algo_epsilon_high
        self.loss_type = args.algo_loss_type
        self.scale_rewards = args.algo_scale_rewards
        self.rewards = build_rewards(args.algo_rewards)
        self.sync_generation = args.strategy == "fsdp"   # FSDP: every rank must run the same forwards
        self._params = None

    def prepare(self, model):
        # Dropout would make the log-probs of the loss differ from those the samples came from.
        self._params = _forward_params(model)
        return disable_dropout(model)

    # ------------------------------------------------------------------
    def _eos_ids(self, model) -> list[int]:
        gen_cfg = getattr(_unwrap(model), "generation_config", None)
        ids = getattr(gen_cfg, "eos_token_id", None) if gen_cfg is not None else None
        ids = [ids] if isinstance(ids, int) else list(ids or [])
        if self.tok.eos_token_id is not None and self.tok.eos_token_id not in ids:
            ids.append(self.tok.eos_token_id)
        return ids

    def _model_inputs(self, input_ids, attention_mask, keep):
        kw = {"input_ids": input_ids, "attention_mask": attention_mask}
        params = self._params or set()
        if "position_ids" in params:
            # same positions as generation used for left-padded prompts
            kw["position_ids"] = (attention_mask.cumsum(-1) - 1).clamp(min=0)
        if "logits_to_keep" in params:
            kw["logits_to_keep"] = keep
        return kw

    def completion_logps(self, model, input_ids, attention_mask, n_completion):
        """log pi(token) for the last n_completion tokens: [N, C] fp32."""
        out = model(**self._model_inputs(input_ids, attention_mask, n_completion + 1))
        logits = out.logits[:, -(n_completion + 1):-1]
        return selective_logps(logits / self.temperature, input_ids[:, -n_completion:])

    @torch.no_grad()
    def rollout(self, model, batch, device) -> Rollout:
        from transformers import GenerationConfig

        prompt_ids = batch["input_ids"].to(device).repeat_interleave(self.G, dim=0)
        prompt_mask = batch["attention_mask"].to(device).repeat_interleave(self.G, dim=0)
        answers = [a for a in batch["answers"] for _ in range(self.G)]
        eos = self._eos_ids(model)
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else eos[0]
        gen_cfg = GenerationConfig(
            do_sample=True, temperature=self.temperature, top_p=self.top_p, top_k=self.top_k or None,
            max_new_tokens=self.max_new, eos_token_id=eos, pad_token_id=pad, use_cache=True,
        )
        was_training = model.training
        model.eval()
        try:
            synced = self.sync_generation and dist.is_available() and dist.is_initialized() \
                and dist.get_world_size() > 1
            seqs = _unwrap(model).generate(input_ids=prompt_ids, attention_mask=prompt_mask,
                                           generation_config=gen_cfg, synced_gpus=synced)
        finally:
            model.train(was_training)
        completion_ids = seqs[:, prompt_ids.shape[1]:]
        if completion_ids.shape[1] == 0:      # nothing generated (all prompts at EOS): one pad token
            completion_ids = torch.full((seqs.shape[0], 1), eos[0], device=device, dtype=seqs.dtype)
        completion_mask = completion_mask_from(completion_ids, eos)

        lengths = completion_mask.sum(1).tolist()
        texts = [self.tok.decode(row[:n], skip_special_tokens=True) for row, n in zip(completion_ids.tolist(), lengths)]
        prompts = self.tok.batch_decode(batch["input_ids"], skip_special_tokens=True)
        prompts = [p for p in prompts for _ in range(self.G)]
        total = torch.zeros(len(texts), dtype=torch.float32)
        metrics = {}
        for name, fn, weight in self.rewards:
            values = fn(prompts, texts, answers, completion_lengths=lengths, max_completion_length=self.max_new)
            values = torch.tensor([float(v) for v in values], dtype=torch.float32)
            if values.shape[0] != len(texts):
                raise ValueError(f"reward '{name}' returned {values.shape[0]} values for {len(texts)} completions")
            total += weight * values
            short = name.split(":")[-1] if len(self.rewards) > 1 else None
            if short:
                metrics[f"r_{short}"] = values.mean().item()

        roll = build_rollout(prompt_ids, prompt_mask, completion_ids, completion_mask, total.to(device),
                             self.G, self.scale_rewards)
        if self.beta > 0:
            self.attach_reference(model, roll)
        group_std = total.view(-1, self.G).std(dim=1).mean().item()
        metrics = {"reward": total.mean().item(), "reward_std": group_std, **metrics,
                   "len": float(sum(lengths)) / len(lengths)}
        roll.metrics = metrics
        _drop_autocast_cache()
        return roll

    @torch.no_grad()
    def attach_reference(self, model, roll: Rollout) -> None:
        """Reference log-probs for the KL term: the same model with LoRA adapters disabled."""
        with _adapters_disabled(model):
            roll.ref_logps = self.completion_logps(model, roll.input_ids, roll.attention_mask,
                                                   roll.completion_mask.shape[1])
        _drop_autocast_cache()

    def loss(self, model, roll: Rollout):
        C = roll.completion_mask.shape[1]
        logps = self.completion_logps(model, roll.input_ids, roll.attention_mask, C)
        norm = None
        if self.loss_type == "dapo":
            tokens = roll.completion_mask.sum().float()
            world = 1
            if dist.is_available() and dist.is_initialized():
                world = dist.get_world_size()
                dist.all_reduce(tokens)
            norm = tokens.clamp(min=1) / world
        loss, stats = grpo_loss(
            logps, logps.detach(), roll.advantages, roll.completion_mask, roll.ref_logps, self.beta,
            self.eps, self.eps_high, self.loss_type, self.max_new, norm,
        )
        metrics = dict(roll.metrics or {})
        if "kl" in stats:
            metrics["kl"] = stats["kl"].item()
        return loss, metrics

    def __call__(self, model, batch, device):
        roll = self.rollout(model, batch, device)
        return self.loss(model, roll)
