"""Pipeline parallelism (PP): AFAB and 1F1B schedules.

The model is split by layers into pp_size stages. Stages send activations
forward and gradients backward with point-to-point messages.

AFAB  (all-forward, all-backward): run every micro-batch forward, then every
      micro-batch backward. Simple, but each stage keeps activations for all
      micro-batches.

1F1B  (one-forward, one-backward): after a short warm-up, each stage alternates
      one forward and one backward. Same bubble, but at most pp_size
      micro-batches are kept in memory, so you can use many more of them.

Bubble ratio for both: (pp_size - 1) / num_micro_batches.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Callable

import torch

from .comm import send_recv
from .process_groups import ParallelContext


class PipelineEngine:
    def __init__(self, model, pc: ParallelContext, act_shape, dtype, device,
                 loss_fn: Callable, schedule: str = "1f1b", autocast: Callable = nullcontext):
        """
        model:     DataParallel-wrapped PipelineStage (needs .require_sync)
        act_shape: shape of the activations sent between stages
        dtype:     dtype of those activations (the fp32 residual stream)
        loss_fn:   (logits, batch) -> scalar loss for ONE micro-batch
        autocast:  context factory for mixed precision (torch.autocast or nullcontext)
        """
        if schedule not in ("afab", "1f1b"):
            raise ValueError(schedule)
        self.model, self.pc, self.loss_fn, self.schedule = model, pc, loss_fn, schedule
        self.act_shape, self.dtype, self.device = act_shape, dtype, device
        self.autocast = autocast

    # ------------------------------------------------------------ p2p comms
    def _empty(self):
        return torch.empty(self.act_shape, dtype=self.dtype, device=self.device)

    def _recv_forward(self):
        if self.pc.is_first_stage:
            return None
        x = self._empty()
        send_recv([], [(x, self.pc.pp_prev_rank)])
        return x.requires_grad_()

    def _send_forward(self, y):
        if not self.pc.is_last_stage:
            send_recv([(y.detach(), self.pc.pp_next_rank)], [])

    def _recv_backward(self):
        if self.pc.is_last_stage:
            return None
        g = self._empty()
        send_recv([], [(g, self.pc.pp_next_rank)])
        return g

    def _send_backward(self, g):
        if not self.pc.is_first_stage:
            send_recv([(g, self.pc.pp_prev_rank)], [])

    def _send_forward_recv_backward(self, y):
        if self.pc.is_last_stage:
            return None
        g = self._empty()
        send_recv([(y.detach(), self.pc.pp_next_rank)], [(g, self.pc.pp_next_rank)])
        return g

    def _send_backward_recv_forward(self, g):
        if self.pc.is_first_stage:
            return None
        x = self._empty()
        send_recv([(g, self.pc.pp_prev_rank)], [(x, self.pc.pp_prev_rank)])
        return x.requires_grad_()

    # ------------------------------------------------------ compute steps
    def _forward(self, x, batch, num_micro):
        inp = batch["input_ids"] if self.pc.is_first_stage else x
        with self.autocast():
            out = self.model(inp)
            if self.pc.is_last_stage:
                loss = self.loss_fn(out, batch) / num_micro
                self._losses.append(loss.detach())
                return loss
        # The p2p buffers have one fixed dtype; keep the cast inside the graph so
        # the gradient that comes back matches it.
        return out.to(self.dtype)

    def _backward(self, x, y, grad_y, is_last_micro):
        self.model.require_sync = is_last_micro  # DP: only sync on the last micro-batch
        torch.autograd.backward(y, grad_tensors=grad_y)
        return x.grad if x is not None else None

    # ------------------------------------------------------------ schedules
    def train_step(self, batches: list[dict]) -> torch.Tensor | None:
        """Runs forward + backward for all micro-batches. Returns the summed loss
        on the last stage (None elsewhere)."""
        self._losses = []
        if self.schedule == "afab":
            self._afab(batches)
        else:
            self._one_f_one_b(batches)
        return torch.stack(self._losses).sum() if self._losses else None

    def _afab(self, batches):
        m = len(batches)
        saved = []
        for i in range(m):
            x = self._recv_forward()
            y = self._forward(x, batches[i], m)
            self._send_forward(y)
            saved.append((x, y))
        for i in range(m):
            x, y = saved.pop(0)
            g = self._recv_backward()
            gx = self._backward(x, y, g, is_last_micro=(i == m - 1))
            self._send_backward(gx)

    def _one_f_one_b(self, batches):
        m = len(batches)
        warmup = min(self.pc.pp_size - self.pc.pp_rank - 1, m)
        steady = m - warmup
        saved, fwd_i, bwd_i = [], 0, 0

        for _ in range(warmup):                                # warm-up: forwards only
            x = self._recv_forward()
            y = self._forward(x, batches[fwd_i], m); fwd_i += 1
            self._send_forward(y)
            saved.append((x, y))

        x = self._recv_forward() if steady > 0 else None
        for i in range(steady):                                # steady state: 1F then 1B
            y = self._forward(x, batches[fwd_i], m); fwd_i += 1
            saved.append((x, y))
            g = self._send_forward_recv_backward(y)
            x_old, y_old = saved.pop(0)
            gx = self._backward(x_old, y_old, g, is_last_micro=(bwd_i == m - 1)); bwd_i += 1
            if i == steady - 1:
                self._send_backward(gx)
            else:
                x = self._send_backward_recv_forward(gx)

        for _ in range(warmup):                                # cool-down: backwards only
            x_old, y_old = saved.pop(0)
            g = self._recv_backward()
            gx = self._backward(x_old, y_old, g, is_last_micro=(bwd_i == m - 1)); bwd_i += 1
            self._send_backward(gx)
