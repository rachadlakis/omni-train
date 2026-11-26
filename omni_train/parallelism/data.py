"""Data for strategy: 5d.

All sequences are fixed-length token blocks of max_length + 1 tokens (inputs +
the shifted targets), held as one [N, max_length + 1] tensor on every rank:
  * dataset.name == "synthetic" (or model_type custom_transformer): arithmetic
    patterns t_i = (a + b * i) mod vocab, learnable from context.
  * otherwise: the HF dataset's "text" column, tokenized and PACKED (documents
    joined with EOS, then cut into blocks). Packing means no padding, so every
    token counts and CP chunks are always full.

Batching follows the playbook formula:  global batch = mbs x micro_batches x dp.
Which samples a step uses depends only on (epoch, step) and a seed, never on the
layout, so every parallel layout trains on exactly the same data.
"""

from __future__ import annotations

import os

import torch

from .process_groups import ParallelContext


def synthetic_blocks(n: int, seq_len: int, vocab: int, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    a = torch.randint(0, vocab, (n, 1), generator=gen)
    b = torch.randint(1, 8, (n, 1), generator=gen)
    return (a + b * torch.arange(seq_len + 1)) % vocab


def text_blocks(args, tokenizer, rank: int) -> torch.Tensor:
    from datasets import load_dataset

    from omni_train.data import _rank0_first  # rank 0 fills the HF cache, the others reuse it

    with _rank0_first():
        ds = load_dataset(args.dataset, args.dataset_full_name, split=args.dataset_split,
                          token=os.getenv("HF_TOKEN"))
        if "text" not in ds.column_names:
            raise ValueError(f"strategy: 5d needs a 'text' column; dataset has {ds.column_names}")
        ds = ds.filter(lambda x: len(x["text"].strip()) > 0)
        eos = tokenizer.eos_token_id
        ds = ds.map(lambda b: {"ids": [t + [eos] for t in tokenizer(b["text"])["input_ids"]]},
                    batched=True, remove_columns=ds.column_names)
    flat = torch.tensor([t for row in ds["ids"] for t in row], dtype=torch.long)
    block = args.max_length + 1
    n = flat.numel() // block
    if n == 0:
        raise ValueError(f"dataset has {flat.numel()} tokens, fewer than one block of {block}")
    if rank == 0:
        print(f"   packed {flat.numel():,} tokens into {n:,} blocks of {block}", flush=True)
    return flat[: n * block].view(n, block)


class BlockSampler:
    """Hands each rank its micro-batches for one optimizer step."""

    def __init__(self, blocks: torch.Tensor, pc: ParallelContext, micro_batch: int,
                 micro_batches: int, seed: int = 0):
        self.blocks, self.pc, self.mbs, self.n_micro, self.seed = blocks, pc, micro_batch, micro_batches, seed
        self.global_batch = micro_batch * micro_batches * pc.dp_size
        self.steps_per_epoch = blocks.shape[0] // self.global_batch
        if self.steps_per_epoch == 0:
            raise ValueError(
                f"only {blocks.shape[0]} sequences, but one step needs batch_size x micro_batches x dp = "
                f"{micro_batch} x {micro_batches} x {pc.dp_size} = {self.global_batch}")
        seq = blocks.shape[1] - 1
        if seq % pc.cp_size:
            raise ValueError(f"max_length {seq} must be divisible by cp ({pc.cp_size})")
        if pc.sequence_parallel and (seq // pc.cp_size) % pc.tp_size:
            raise ValueError(f"with sp, max_length {seq} must be divisible by cp x tp")
        self.local_seq = seq // pc.cp_size

    def micro_batches(self, epoch: int, step: int, device) -> list[dict]:
        """Every rank of the same DP replica (TP, PP, CP ranks) gets the same
        samples; CP ranks then keep only their slice of the sequence."""
        pc, dp = self.pc, self.pc.dp_size
        order = torch.randperm(self.blocks.shape[0],
                               generator=torch.Generator().manual_seed(self.seed * 1_000_003 + epoch))
        lo, hi = pc.cp_rank * self.local_seq, (pc.cp_rank + 1) * self.local_seq
        out = []
        for m in range(self.n_micro):
            first = step * self.global_batch + (m * dp + pc.dp_rank) * self.mbs
            seqs = self.blocks[order[first:first + self.mbs]]
            out.append({
                "input_ids": seqs[:, :-1][:, lo:hi].to(device, non_blocking=True),
                "targets": seqs[:, 1:][:, lo:hi].to(device, non_blocking=True),
            })
        return out
