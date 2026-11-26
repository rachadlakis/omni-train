"""PPO: online RL with a value model (Schulman et al. 2017; RLHF as in Ouyang et al. 2022).

One dataloader batch becomes one ROLLOUT, then several gradient steps:

  1. rollout   sample completions with the current policy, score them (reward functions and/or
               a reward model), and record per token: log-prob, value, and the KL penalty
               against the start model. GAE turns them into advantages and returns.
  2. update    ppo_epochs passes over the rollout in num_minibatches pieces: clipped policy
               loss + vf_coef * clipped value loss.

`wrap_loader` turns the dataloader into one that yields the minibatches (doing the rollout when
it reaches a new batch), so train.py's loop, LR schedule and progress line see each gradient
step as a normal step.

Value model
-----------
One linear layer on the policy's last hidden state (the input of lm_head, caught by a hook). It
is NOT part of the wrapped model: it has its own AdamW and is replicated on every rank, with its
gradient averaged by hand (`step_extra`). So DDP/FSDP wrapping, checkpoints, resume and the HF
export of the policy are exactly those of any other run. It is saved next to the checkpoint.

Distributed exactness: advantages are normalised over the GLOBAL batch, and both losses divide by
the global token count of the minibatch, so the averaged gradients equal one process on the whole
batch (tests/test_post_training_dist.py compares every parameter).
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import dataclass, fields

import torch
import torch.distributed as dist

from .grpo import GRPOObjective, _adapters_disabled, _drop_autocast_cache, _unwrap
from .losses import gae, ppo_loss, selective_logps


@dataclass
class PPORollout:
    """[N, P + C] ids / attention mask; the rest are [N, C]."""
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    completion_mask: torch.Tensor
    old_logps: torch.Tensor
    old_values: torch.Tensor
    advantages: torch.Tensor       # normalised over the global batch
    returns: torch.Tensor
    metrics: dict | None = None

    def rows(self, index) -> "PPORollout":
        return PPORollout(**{f.name: (v[index] if torch.is_tensor(v := getattr(self, f.name)) else v)
                             for f in fields(self)})

    def __len__(self):
        return self.input_ids.shape[0]


def _world() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def _global_sum(*values):
    """Sums of the given scalars over all ranks (float64)."""
    ref = next(v for v in values if torch.is_tensor(v))
    t = torch.stack([torch.as_tensor(v, dtype=torch.float64, device=ref.device) for v in values])
    if _world() > 1:
        dist.all_reduce(t)
    return t.unbind()


def normalize_advantages(adv, mask, eps: float = 1e-8):
    """Zero mean, unit variance over the tokens of the global batch."""
    m = mask.float()
    n, s, sq = _global_sum(m.sum(), (adv.double() * m).sum(), (adv.double() ** 2 * m).sum())
    mean = s / n.clamp(min=1)
    std = (sq / n.clamp(min=1) - mean ** 2).clamp(min=0).sqrt()
    return ((adv - mean.float()) / (std.float() + eps)) * m


class RolloutLoader:
    """Yields PPO minibatches; len() counts gradient steps."""

    def __init__(self, loader, objective, model, device):
        self.loader, self.obj, self.model, self.device = loader, objective, model, device

    @property
    def sampler(self):
        return self.loader.sampler

    def __len__(self):
        return len(self.loader) * self.obj.epochs * self.obj.num_minibatches

    def __iter__(self):
        for batch in self.loader:
            self.model.zero_grad(set_to_none=True)      # sampling needs the memory
            with self.obj.autocast_ctx():
                roll = self.obj.rollout(self.model, batch, self.device)
            for _ in range(self.obj.epochs):
                for index in torch.randperm(len(roll)).chunk(self.obj.num_minibatches):
                    yield roll.rows(index.to(self.device))


class PPOObjective(GRPOObjective):
    def __init__(self, args, tokenizer):
        super().__init__(args, tokenizer)
        self.epochs, self.num_minibatches = args.algo_ppo_epochs, args.algo_num_minibatches
        self.gamma, self.lam = args.algo_gamma, args.algo_gae_lambda
        self.vf_coef, self.value_clip, self.value_lr = args.algo_vf_coef, args.algo_value_clip, args.algo_value_lr
        self.eos_penalty = args.algo_eos_penalty
        self.autocast_ctx = nullcontext      # train.py sets its own
        self.value_head = self.value_opt = None
        self._capture, self._hidden = False, None

    # ------------------------------------------------------------------ set-up
    def wrap_loader(self, loader, model, device) -> RolloutLoader:
        return RolloutLoader(loader, self, model, device)

    def prepare(self, model):
        changed = super().prepare(model)
        m = _unwrap(model)
        base = m.get_base_model() if hasattr(m, "get_base_model") else m
        lm_head = base.get_output_embeddings()
        if lm_head is None or not hasattr(lm_head, "in_features"):
            raise ValueError("ppo needs a causal LM with a linear output layer (its input feeds the value head)")
        # A hook INSIDE the model's forward: under FSDP the hidden state is only available there,
        # and models that slice with logits_to_keep hand lm_head just the positions we need.
        lm_head.register_forward_pre_hook(self._catch_hidden)
        device = next(model.parameters()).device
        self.value_head = torch.nn.Linear(lm_head.in_features, 1, device=device, dtype=torch.float32)
        # zeros: V = 0 at the start (advantages = returns) on every rank, no RNG involved
        torch.nn.init.zeros_(self.value_head.weight)
        torch.nn.init.zeros_(self.value_head.bias)
        self.value_opt = torch.optim.AdamW(self.value_head.parameters(), lr=self.value_lr, weight_decay=0.0)
        return changed

    def _catch_hidden(self, module, args):
        if self._capture:
            self._hidden = args[0]

    # ------------------------------------------------------------------ forward
    def logps_and_values(self, model, input_ids, attention_mask, C):
        """log pi(token) and V(state before the token) for the C completion tokens: [N, C] fp32 each."""
        self._capture = True
        try:
            out = model(**self._model_inputs(input_ids, attention_mask, C + 1))
        finally:
            self._capture = False
        hidden, self._hidden = self._hidden, None
        logits = out.logits[:, -(C + 1):-1]
        if self.temperature != 1.0:
            logits = logits / self.temperature
        with torch.autocast(hidden.device.type, enabled=False):
            values = self.value_head(hidden[:, -(C + 1):-1].float()).squeeze(-1)
        return selective_logps(logits, input_ids[:, -C:]), values

    # ------------------------------------------------------------------ rollout
    @torch.no_grad()
    def score_tokens(self, model, input_ids, attention_mask, C):
        """(logps, values, ref_logps or None), in minibatch-sized pieces: without gradients the
        peak must not exceed that of a training step."""
        size = max(1, input_ids.shape[0] // self.num_minibatches)
        was_training = model.training
        model.eval()
        logps, values, refs = [], [], []
        try:
            for i in range(0, input_ids.shape[0], size):
                ids, mask = input_ids[i:i + size], attention_mask[i:i + size]
                lp, v = self.logps_and_values(model, ids, mask, C)
                logps.append(lp)
                values.append(v)
                if self.beta > 0:
                    with _adapters_disabled(model):
                        refs.append(self.completion_logps(model, ids, mask, C))
        finally:
            model.train(was_training)
        _drop_autocast_cache()
        return torch.cat(logps), torch.cat(values), torch.cat(refs) if refs else None

    @torch.no_grad()
    def build(self, model, prompt_ids, prompt_mask, completion_ids, completion_mask, scores, finished=None):
        """PPORollout from sampled completions and their scores [N] (one per completion)."""
        C = completion_ids.shape[1]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logps, values, ref = self.score_tokens(model, input_ids, attention_mask, C)
        m = completion_mask.float()
        scores = scores.to(m.device).float()
        if self.eos_penalty and finished is not None:
            scores = scores - self.eos_penalty * (~finished).float()
        rewards = torch.zeros_like(m)
        rewards.scatter_(1, (completion_mask.sum(1, keepdim=True) - 1).clamp(min=0), scores.unsqueeze(1))
        metrics = {}
        if ref is not None:
            kl = (logps - ref) * m
            rewards = rewards - self.beta * kl
            metrics["kl"] = kl.sum(1).mean().item()
        adv, returns = gae(rewards, values, completion_mask, self.gamma, self.lam)
        # RMS error of the value head. (Explained variance blows up once returns stop varying.)
        metrics["v_err"] = (((returns - values) ** 2 * m).sum() / m.sum().clamp(min=1)).sqrt().item()
        return PPORollout(input_ids, attention_mask, completion_mask, logps, values * m,
                          normalize_advantages(adv, completion_mask), returns, metrics)

    @torch.no_grad()
    def rollout(self, model, batch, device) -> PPORollout:
        s = self.sample(model, batch, device)
        roll = self.build(model, s["prompt_ids"], s["prompt_mask"], s["completion_ids"], s["completion_mask"],
                          s["rewards"], s["finished"])
        roll.metrics = {**s["metrics"], **roll.metrics}
        _drop_autocast_cache()
        return roll

    # ------------------------------------------------------------------ update
    def loss(self, model, mb: PPORollout):
        C = mb.completion_mask.shape[1]
        logps, values = self.logps_and_values(model, mb.input_ids, mb.attention_mask, C)
        tokens, = _global_sum(mb.completion_mask.sum())
        norm = tokens.float().clamp(min=1) / _world()
        loss, stats = ppo_loss(logps, mb.old_logps, mb.advantages, mb.completion_mask, values, mb.old_values,
                               mb.returns, self.eps, self.value_clip, self.vf_coef, norm)
        return loss, {**(mb.metrics or {}), "vf": stats["vf"].item(), "clip": stats["clip_frac"].item()}

    def __call__(self, model, batch, device):
        self.value_opt.zero_grad()
        return self.loss(model, batch)

    def sync_value_grads(self):
        """Average the value head's gradient over ranks (DDP/FSDP do it for the policy)."""
        if _world() > 1:
            for p in self.value_head.parameters():
                dist.all_reduce(p.grad)
                p.grad /= _world()

    def step_extra(self, scaler, grad_clip: float):
        """Update the value head; train.py calls it between scaler.step(optimizer) and scaler.update()."""
        self.sync_value_grads()
        if grad_clip > 0:
            scaler.unscale_(self.value_opt)
            torch.nn.utils.clip_grad_norm_(self.value_head.parameters(), grad_clip)
        scaler.step(self.value_opt)

    # ------------------------------------------------------------------ save / resume
    @staticmethod
    def extra_path(checkpoint: str) -> str:
        """checkpoints/solo/123.pt -> 123.value_head; checkpoints/fsdp/dcp_api/123/ -> 123/value_head.pt"""
        if os.path.isdir(checkpoint):
            return os.path.join(checkpoint, "value_head.pt")
        return os.path.splitext(checkpoint)[0] + ".value_head"

    def save_extra(self, checkpoint: str):
        torch.save({"head": self.value_head.state_dict(), "optimizer": self.value_opt.state_dict()},
                   self.extra_path(checkpoint))

    def load_extra(self, checkpoint: str) -> bool:
        path = self.extra_path(checkpoint)
        if not os.path.exists(path):
            return False
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.value_head.load_state_dict(state["head"])
        self.value_opt.load_state_dict(state["optimizer"])
        for group in self.value_opt.param_groups:
            group["lr"] = self.value_lr
        return True
