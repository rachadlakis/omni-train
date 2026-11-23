"""Loss functions for post-training. Pure tensor math, no model or data code.

Conventions
-----------
* ``labels`` use -100 for positions that must not count (prompt, padding).
* A *sequence log-prob* is the SUM of log p(token) over the counted tokens of
  one sequence; ``counts`` is how many tokens were counted. ``sum / count`` is
  the length-normalised (average) log-prob used by IPO, SimPO and ORPO.
* Every loss returns one value PER EXAMPLE ([B]); the caller takes the mean.
  A per-example mean keeps DDP/FSDP exact: equal-sized batches on every rank,
  averaged by the gradient all-reduce, give the same gradient as one big batch.

References: DPO (Rafailov et al. 2023), IPO (Azar et al. 2023), SimPO
(Meng et al. 2024), ORPO (Hong et al. 2024).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def sequence_logps(logits: torch.Tensor, labels: torch.Tensor):
    """Sum of next-token log-probs per sequence.

    logits: [B, T, V] (any float dtype), labels: [B, T] aligned with the inputs
    (the shift by one happens here). Returns (sums [B] fp32, counts [B] fp32).

    Runs one row at a time in fp32, so peak extra memory is one [T, V] row
    instead of a full [B, T, V] fp32 log-softmax (151k-token vocabularies make
    that gigabytes).
    """
    logits = logits[:, :-1]
    labels = labels[:, 1:]
    mask = labels != IGNORE_INDEX
    safe = labels.masked_fill(~mask, 0)
    sums = []
    for b in range(logits.shape[0]):
        row = logits[b].float()
        token_logp = row.gather(-1, safe[b].unsqueeze(-1)).squeeze(-1) - torch.logsumexp(row, dim=-1)
        sums.append((token_logp * mask[b]).sum())
    return torch.stack(sums), mask.sum(-1).float()


def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected,
             beta: float, kind: str = "sigmoid", label_smoothing: float = 0.0):
    """DPO family. Inputs are per-example log-probs [B] (sums for sigmoid/hinge,
    length-normalised for IPO, as in the IPO paper).

    h = (log pi(y_w) - log ref(y_w)) - (log pi(y_l) - log ref(y_l))
      sigmoid: -log sigmoid(beta * h)    (label_smoothing: conservative DPO)
      ipo:     (h - 1 / (2 * beta))^2
      hinge:   max(0, 1 - beta * h)     (SLiC)
    Returns (losses [B], chosen_rewards [B], rejected_rewards [B]); rewards are
    beta * log-ratio, detached, for logging.
    """
    chosen_ratio = policy_chosen - ref_chosen
    rejected_ratio = policy_rejected - ref_rejected
    h = chosen_ratio - rejected_ratio
    if kind == "sigmoid":
        losses = (-(1 - label_smoothing) * F.logsigmoid(beta * h)
                  - label_smoothing * F.logsigmoid(-beta * h))
    elif kind == "ipo":
        losses = (h - 1 / (2 * beta)) ** 2
    elif kind == "hinge":
        losses = torch.relu(1 - beta * h)
    else:
        raise ValueError(f"unknown DPO loss {kind!r}")
    return losses, (beta * chosen_ratio).detach(), (beta * rejected_ratio).detach()


def simpo_loss(avg_chosen, avg_rejected, beta: float, gamma: float):
    """SimPO: reference-free, length-normalised, with a target margin gamma.

    loss = -log sigmoid(beta * (avg log pi(y_w) - avg log pi(y_l)) - gamma)
    """
    losses = -F.logsigmoid(beta * (avg_chosen - avg_rejected) - gamma)
    return losses, (beta * avg_chosen).detach(), (beta * avg_rejected).detach()


def _log1mexp(x: torch.Tensor) -> torch.Tensor:
    """log(1 - exp(x)) for x < 0, accurate everywhere (Maechler 2012)."""
    return torch.where(x > -math.log(2), torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x)))


def orpo_loss(avg_chosen, avg_rejected, lam: float):
    """ORPO: SFT on the chosen answer plus an odds-ratio preference term.

    odds(y) = p / (1 - p) with p = exp(avg log-prob)
    loss = -avg log pi(y_w) - lam * log sigmoid(log odds(y_w) - log odds(y_l))
    """
    # p = 1 exactly would make log(1 - p) = -inf; keep a hair below.
    avg_chosen, avg_rejected = avg_chosen.clamp(max=-1e-6), avg_rejected.clamp(max=-1e-6)
    log_odds = (avg_chosen - avg_rejected) - (_log1mexp(avg_chosen) - _log1mexp(avg_rejected))
    ratio = F.logsigmoid(log_odds)
    losses = -avg_chosen - lam * ratio
    return losses, (lam * avg_chosen).detach(), (lam * avg_rejected).detach()


# ----------------------------------------------------------------------
# GRPO (Shao et al. 2024), with the DAPO (Yu et al. 2025) and Dr. GRPO
# (Liu et al. 2025) normalisations
# ----------------------------------------------------------------------

GRPO_LOSS_TYPES = ("grpo", "dapo", "dr_grpo")


def selective_logps(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """log p(ids) per position. logits [B, T, V] already aligned with ids [B, T].

    One row at a time in fp32 (as sequence_logps), returns [B, T] fp32.
    """
    out = []
    for b in range(logits.shape[0]):
        row = logits[b].float()
        out.append(row.gather(-1, ids[b].unsqueeze(-1)).squeeze(-1) - torch.logsumexp(row, dim=-1))
    return torch.stack(out)


def group_advantages(rewards: torch.Tensor, group_size: int, scale: bool = True, eps: float = 1e-4):
    """A_i = (r_i - mean(group)) / (std(group) + eps), groups of `group_size`
    consecutive completions of the same prompt. scale=False (Dr. GRPO) skips the
    division, which otherwise up-weights prompts that are almost always right or wrong.
    A group whose rewards are all equal gets zero advantage: nothing to learn from it."""
    r = rewards.float().view(-1, group_size)
    adv = r - r.mean(dim=1, keepdim=True)
    if scale:
        adv = adv / (r.std(dim=1, keepdim=True) + eps)
    return adv.reshape(-1)


def grpo_loss(logps, old_logps, advantages, mask, ref_logps=None, beta: float = 0.0,
              epsilon: float = 0.2, epsilon_high: float | None = None, loss_type: str = "grpo",
              max_completion_length: int | None = None, token_normalizer=None):
    """Clipped policy-gradient loss over completion tokens.

    logps, old_logps, ref_logps, mask: [N, C]; advantages: [N].
      ratio    = exp(logp - old_logp)                       (1 when on-policy)
      token    = -min(ratio * A, clip(ratio, 1 - eps, 1 + eps_high) * A)
               + beta * KL,   KL = exp(ref - logp) - (ref - logp) - 1   (k3, >= 0)
    Aggregation (`loss_type`), all exact under DDP/FSDP with equal per-rank batches:
      grpo     mean over each completion's tokens, then mean over completions
      dapo     sum over all tokens / number of tokens in the GLOBAL batch; pass
               token_normalizer = global_tokens / world_size (gradients are averaged
               over ranks), or leave None for one process
      dr_grpo  sum over all tokens / (completions * max_completion_length)
    Returns (loss, stats) with detached stats: clip_frac, kl.
    """
    eps_high = epsilon if epsilon_high is None else epsilon_high
    m = mask.float()
    ratio = torch.exp(logps - old_logps)
    clipped = ratio.clamp(1 - epsilon, 1 + eps_high)
    a = advantages.unsqueeze(1)
    per_token = -torch.min(ratio * a, clipped * a)
    stats = {}
    if beta and ref_logps is not None:
        d = ref_logps - logps
        kl = torch.exp(d) - d - 1
        per_token = per_token + beta * kl
        stats["kl"] = ((kl * m).sum() / m.sum().clamp(min=1)).detach()
    if loss_type == "grpo":
        loss = ((per_token * m).sum(1) / m.sum(1).clamp(min=1)).mean()
    elif loss_type == "dapo":
        norm = m.sum().clamp(min=1) if token_normalizer is None else token_normalizer
        loss = (per_token * m).sum() / norm
    elif loss_type == "dr_grpo":
        if not max_completion_length:
            raise ValueError("dr_grpo needs max_completion_length")
        loss = (per_token * m).sum() / (m.shape[0] * max_completion_length)
    else:
        raise ValueError(f"unknown GRPO loss_type {loss_type!r}")
    was_clipped = ((ratio < 1 - epsilon) & (a < 0)) | ((ratio > 1 + eps_high) & (a > 0))
    stats["clip_frac"] = ((was_clipped.float() * m).sum() / m.sum().clamp(min=1)).detach()
    return loss, stats
