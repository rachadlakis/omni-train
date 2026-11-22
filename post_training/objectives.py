"""Per-step objectives. train.py keeps the model setup, optimizer, schedule,
clipping and checkpointing; an objective only turns (model, batch) into a loss.

Reference model (DPO)
---------------------
DPO compares the policy with a frozen reference: the model as it was before
training. Before step 1 the policy IS the reference, so one no-grad pass over
the dataset gives every reference log-prob we will ever need. They are cached
per example index ([N, 2] floats), and DPO then trains with no second model in
memory, for every strategy (solo/DDP/FSDP) and with or without LoRA.

Dropout is switched off for preference training (TRL does the same by default):
with dropout the policy's log-probs are noisy, so even at step 0 the implicit
reward would not be 0 and DPO would not start at ln 2.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.distributed as dist

from .losses import dpo_loss, orpo_loss, sequence_logps, simpo_loss


def _to(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def disable_dropout(model) -> int:
    """Zero every dropout in the model: nn.Dropout modules (LoRA dropout included)
    and the float `*dropout` attributes HF layers pass to F.dropout (OPT's
    `dropout`, Llama/Qwen's `attention_dropout`). Returns how many were changed."""
    changed = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout) and module.p:
            module.p = 0.0
            changed += 1
        for name, value in list(vars(module).items()):
            if name.endswith("dropout") and isinstance(value, float) and value:
                setattr(module, name, 0.0)
                changed += 1
    return changed


class SFTObjective:
    """Next-token cross-entropy on the unmasked (answer) tokens.

    Averaged over the tokens of the LOCAL batch (HF's loss), so with DDP/FSDP
    ranks with more answer tokens count the same as ranks with fewer. This is
    the usual convention (HF Trainer, TRL default)."""

    needs_reference = False

    def prepare(self, model):
        """SFT keeps the model's dropout."""
        return 0

    def __call__(self, model, batch, device):
        b = _to(batch, device)
        out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], labels=b["labels"])
        return out.loss, {}


class PreferenceObjective:
    """DPO (sigmoid / IPO / hinge), SimPO and ORPO on chosen/rejected pairs."""

    def __init__(self, args):
        self.algo = args.algorithm
        self.beta = args.algo_beta
        self.kind = args.algo_dpo_loss
        self.label_smoothing = args.algo_label_smoothing
        self.gamma = args.algo_simpo_gamma
        self.needs_reference = self.algo == "dpo"
        self.ref = None  # [N, 2] reference log-prob sums (chosen, rejected), on CPU

    def prepare(self, model):
        return disable_dropout(model)

    @staticmethod
    def logps(model, batch):
        """(sums, counts) for the 2B sequences of a collate_preference batch."""
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        return sequence_logps(out.logits, batch["labels"])

    @torch.no_grad()
    def precompute_reference(self, model, dataset, collate, rank, world_size, batch_size,
                             device, autocast_ctx=nullcontext, log=print):
        from .data import make_loader
        was_training = model.training
        model.eval()
        # No shuffling, no dropping: every example is scored. DistributedSampler pads
        # by repeating a few, so all ranks run the same number of forwards (FSDP).
        loader = make_loader(dataset, collate, rank, world_size, batch_size, shuffle=False, drop_last=False)
        rows = []
        for i, batch in enumerate(loader):
            b = _to(batch, device)
            with autocast_ctx():
                sums, _ = self.logps(model, b)
            n = b["idx"].shape[0]
            rows.append(torch.stack([b["idx"].double(), sums[:n].double(), sums[n:].double()], dim=1))
            if rank == 0 and (i + 1) % 20 == 0:
                log(f"   reference log-probs: {i + 1}/{len(loader)} batches")
        local = torch.cat(rows)
        if dist.is_available() and dist.is_initialized() and world_size > 1:
            parts = [torch.empty_like(local) for _ in range(world_size)]
            dist.all_gather(parts, local)
            local = torch.cat(parts)
        ref = torch.full((len(dataset), 2), float("nan"), dtype=torch.float64)
        ref[local[:, 0].long().cpu()] = local[:, 1:].cpu()
        if torch.isnan(ref).any():
            raise RuntimeError("reference log-probs missing for some examples")
        self.ref = ref.float()
        model.train(was_training)

    def __call__(self, model, batch, device):
        b = _to(batch, device)
        n = b["idx"].shape[0]
        sums, counts = self.logps(model, b)
        pc, pr = sums[:n], sums[n:]
        avg_c, avg_r = pc / counts[:n], pr / counts[n:]
        if self.algo == "dpo":
            ref = self.ref[b["idx"].cpu()].to(device)
            rc, rr = ref[:, 0], ref[:, 1]
            if self.kind == "ipo":  # IPO works on length-normalised log-probs
                pc, pr, rc, rr = avg_c, avg_r, rc / counts[:n], rr / counts[n:]
            losses, cr, rr_ = dpo_loss(pc, pr, rc, rr, self.beta, self.kind, self.label_smoothing)
        elif self.algo == "simpo":
            losses, cr, rr_ = simpo_loss(avg_c, avg_r, self.beta, self.gamma)
        else:  # orpo
            losses, cr, rr_ = orpo_loss(avg_c, avg_r, self.beta)
        metrics = {"acc": (cr > rr_).float().mean().item(), "margin": (cr - rr_).mean().item()}
        return losses.mean(), metrics


def build_objective(args):
    if args.algorithm == "sft":
        return SFTObjective()
    return PreferenceObjective(args)
