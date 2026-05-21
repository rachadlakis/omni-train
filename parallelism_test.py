"""
3D Parallel LLM Training: TP + PP + FSDP (Data Parallel)
=========================================================
Combines three parallelism dimensions using PyTorch native APIs,
modelled closely on the official PyTorch distributed examples:

  - Tensor Parallel  (TP) — col-wise/row-wise weight sharding within a host
  - Pipeline Parallel (PP) — layer stages across hosts, GPipe schedule
  - Data Parallel    (DP) — FSDP2 (fully_shard) across DP ranks

Mesh layout (example: 8 GPUs, DP=2, PP=2, TP=2):

  Host A (GPUs 0-3)              Host B (GPUs 4-7)
  ┌──────────────────────┐       ┌──────────────────────┐
  │  PP stage 0          │       │  PP stage 0          │
  │  [GPU 0] [GPU 1] (TP)│  DP   │  [GPU 4] [GPU 5] (TP)│
  ├──────────────────────┤ ←───→ ├──────────────────────┤
  │  PP stage 1          │       │  PP stage 1          │
  │  [GPU 2] [GPU 3] (TP)│       │  [GPU 6] [GPU 7] (TP)│
  └──────────────────────┘       └──────────────────────┘

Mesh dim order: [dp, pp, tp]
  - TP ranks are innermost   (fastest NVLink / intra-node)
  - PP ranks are middle      (inter-node or intra-node)
  - DP ranks are outermost   (inter-node)

Requirements:
  pip install torch>=2.3
  torchrun --nproc_per_node=8 train_3d_parallel.py

  # Custom mesh sizes (must satisfy DP*PP*TP == nproc_per_node):
  DP_SIZE=2 PP_SIZE=2 TP_SIZE=2 torchrun --nproc_per_node=8 train_3d_parallel.py
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.distributed as dist

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard   
from torch.distributed._tensor import Shard, Replicate # type: ignore
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
    PrepareModuleInput,
    SequenceParallel,
)
from torch.distributed.pipelining import (
    PipelineStage,
    ScheduleGPipe,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DP_SIZE = int(os.environ.get("DP_SIZE", 2))
PP_SIZE = int(os.environ.get("PP_SIZE", 2))
TP_SIZE = int(os.environ.get("TP_SIZE", 2))

# Model
VOCAB_SIZE  = 32_000
SEQ_LEN     = 256
D_MODEL     = 512
N_HEADS     = 8        # must be divisible by TP_SIZE
N_KV_HEADS  = 8
N_LAYERS    = 8        # must be divisible by PP_SIZE
NORM_EPS    = 1e-5
MULTIPLE_OF = 256

# Training
GLOBAL_BATCH  = 16     # total sequences per step across all DP ranks
MICRO_BATCH   = 2      # sequences per pipeline micro-batch
LR            = 3e-4
MAX_STEPS     = 50
LOG_EVERY     = 10

assert N_LAYERS % PP_SIZE == 0, "N_LAYERS must be divisible by PP_SIZE"
assert N_HEADS  % TP_SIZE == 0, "N_HEADS must be divisible by TP_SIZE"
assert GLOBAL_BATCH % (DP_SIZE * MICRO_BATCH) == 0, (
    "GLOBAL_BATCH must be divisible by DP_SIZE * MICRO_BATCH"
)

# ---------------------------------------------------------------------------
# Model — Llama-style (mirrors llama2_model.py from official PyTorch examples)
# ---------------------------------------------------------------------------

def precompute_freqs_cis(head_dim: int, seq_len: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)   # complex64


def apply_rotary_emb(xq, xk, freqs_cis):
    def rotate(x):
        # x: [B, T, n_heads, head_dim]
        x_c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        fc   = freqs_cis.view(1, x_c.shape[1], 1, x_c.shape[-1])
        return torch.view_as_real(x_c * fc).flatten(3).type_as(x)
    return rotate(xq), rotate(xk)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = NORM_EPS):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

    def reset_parameters(self):
        nn.init.ones_(self.weight)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_heads    = N_HEADS
        self.n_kv_heads = N_KV_HEADS
        self.n_rep      = N_HEADS // N_KV_HEADS
        self.head_dim   = D_MODEL // N_HEADS

        self.wq = nn.Linear(D_MODEL, N_HEADS    * self.head_dim, bias=False)
        self.wk = nn.Linear(D_MODEL, N_KV_HEADS * self.head_dim, bias=False)
        self.wv = nn.Linear(D_MODEL, N_KV_HEADS * self.head_dim, bias=False)
        self.wo = nn.Linear(N_HEADS * self.head_dim, D_MODEL,    bias=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        bsz, seqlen, _ = x.shape

        xq = self.wq(x).view(bsz, seqlen, self.n_heads,    self.head_dim)
        xk = self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        if self.n_rep > 1:
            xk = xk.repeat_interleave(self.n_rep, dim=2)
            xv = xv.repeat_interleave(self.n_rep, dim=2)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        out = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(out)


def _ffn_hidden_dim(dim: int) -> int:
    """SwiGLU hidden dim rounded to nearest MULTIPLE_OF."""
    h = int(2 * (4 * dim) / 3)
    return MULTIPLE_OF * ((h + MULTIPLE_OF - 1) // MULTIPLE_OF)


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        h        = _ffn_hidden_dim(D_MODEL)
        self.w1  = nn.Linear(D_MODEL, h, bias=False)   # gate  (ColwiseParallel)
        self.w2  = nn.Linear(h, D_MODEL, bias=False)   # down  (RowwiseParallel)
        self.w3  = nn.Linear(D_MODEL, h, bias=False)   # up    (ColwiseParallel)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int):
        super().__init__()
        self.layer_id        = layer_id
        self.attention_norm  = RMSNorm(D_MODEL)
        self.attention       = Attention()
        self.ffn_norm        = RMSNorm(D_MODEL)
        self.feed_forward    = FeedForward()

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        h   = x + self.attention(self.attention_norm(x), freqs_cis)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


# ---------------------------------------------------------------------------
# Per-rank pipeline stage module
# Each PP rank owns a slice of the layers plus optionally the
# embedding (first stage) or norm + lm_head (last stage).
# freqs_cis is stored as a buffer so it moves to the right device with .to().
# ---------------------------------------------------------------------------

class StageModule(nn.Module):
    def __init__(
        self,
        layers: nn.ModuleList,
        freqs_cis: torch.Tensor,
        is_first: bool,
        is_last: bool,
    ):
        super().__init__()
        self.layers    = layers
        self.is_first  = is_first
        self.is_last   = is_last

        if is_first:
            self.tok_embeddings = nn.Embedding(VOCAB_SIZE, D_MODEL)
            nn.init.normal_(self.tok_embeddings.weight)

        if is_last:
            self.norm   = RMSNorm(D_MODEL)
            self.output = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)
            nn.init.trunc_normal_(
                self.output.weight, std=D_MODEL ** -0.5
            )

        self.register_buffer("freqs_cis", freqs_cis)

    def forward(self, x: torch.Tensor):
        """
        First stage : x is token ids  [B, T]  (torch.long)
        Other stages: x is hidden     [B, T, D]
        """
        if self.is_first:
            seqlen      = x.shape[1]
            h           = self.tok_embeddings(x)
            freqs_cis   = self.freqs_cis[:seqlen] # type: ignore
        else:
            h           = x
            freqs_cis   = self.freqs_cis[:h.shape[1]] # type: ignore

        for layer in self.layers:
            h = layer(h, freqs_cis)

        if self.is_last:
            h = self.output(self.norm(h)).float()

        return h


# ---------------------------------------------------------------------------
# Tensor Parallelism
# Mirrors fsdp_tp_example.py exactly: RowwiseParallel on embedding,
# PrepareModuleInput + Col/Row on attention and FFN, SequenceParallel on norms.
# ---------------------------------------------------------------------------

def apply_tp(stage: StageModule, tp_mesh):
    # --- Embedding: Replicate input → Shard(1) output [first stage only] ---
    if stage.is_first:
        parallelize_module(
            stage, tp_mesh,
            {
                "tok_embeddings": RowwiseParallel(
                    input_layouts=Replicate(),
                    output_layouts=Shard(1),
                ),
            },
        )

    # --- TransformerBlocks ---
    for blk in stage.layers:
        # Layout flow matches llama2 TP tutorial exactly:
        #   Shard(1) activations through norms (SequenceParallel)
        #   → Replicate for attention/FFN entry (PrepareModuleInput)
        #   → ColwiseParallel projects (output sharded on head/hidden dim)
        #   → RowwiseParallel output projection (all-reduce + Shard(1))
        parallelize_module(
            module=blk,
            device_mesh=tp_mesh,
            parallelize_plan={
                "attention_norm": SequenceParallel(),
                "attention": PrepareModuleInput(
                    input_layouts=(Shard(1), Replicate()),                      # type: ignore
                    desired_input_layouts=(Replicate(), Replicate()),           # type: ignore
                ),
                "attention.wq": ColwiseParallel(use_local_output=False),
                "attention.wk": ColwiseParallel(use_local_output=False),
                "attention.wv": ColwiseParallel(use_local_output=False),
                "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
                "ffn_norm": SequenceParallel(),
                "feed_forward": PrepareModuleInput(
                    input_layouts=(Shard(1),),
                    desired_input_layouts=(Replicate(),),
                ),
                "feed_forward.w1": ColwiseParallel(),
                "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
                "feed_forward.w3": ColwiseParallel(),
            },
        )

    # --- Head: SequenceParallel norm → ColwiseParallel lm_head [last stage] ---
    if stage.is_last:
        parallelize_module(
            stage, tp_mesh,
            {
                "norm":   SequenceParallel(),
                "output": ColwiseParallel(
                    input_layouts=Shard(1),
                    output_layouts=Replicate(),
                ),
            },
        )


# ---------------------------------------------------------------------------
# FSDP2 (Data Parallelism)
# Applied AFTER TP so that FSDP2 shards the DTensors produced by TP.
# Wrap each block independently first, then wrap the whole stage.
# Mirrors the per-layer wrapping pattern in fsdp_tp_example.py.
# ---------------------------------------------------------------------------

def apply_fsdp(stage: StageModule, dp_mesh):
    for layer in stage.layers:
        fully_shard(layer, mesh=dp_mesh)
    fully_shard(stage, mesh=dp_mesh)


# ---------------------------------------------------------------------------
# Random token dataset
# ---------------------------------------------------------------------------

class RandomTokenDataset(torch.utils.data.Dataset):
    def __init__(self, size: int = 4096):
        self.x = torch.randint(0, VOCAB_SIZE, (size, SEQ_LEN))
        self.y = torch.randint(0, VOCAB_SIZE, (size, SEQ_LEN))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ── Init distributed ────────────────────────────────────────────────────
    dist.init_process_group(backend="nccl")

    local_rank  = int(os.environ["LOCAL_RANK"])
    world_rank  = int(os.environ["RANK"])
    world_size  = int(os.environ["WORLD_SIZE"])

    assert world_size == DP_SIZE * PP_SIZE * TP_SIZE, (
        f"WORLD_SIZE ({world_size}) must equal "
        f"DP({DP_SIZE}) × PP({PP_SIZE}) × TP({TP_SIZE}) = "
        f"{DP_SIZE * PP_SIZE * TP_SIZE}"
    )

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # ── 3-D device mesh [dp, pp, tp] ────────────────────────────────────────
    device_mesh = init_device_mesh(
        "cuda",
        (DP_SIZE, PP_SIZE, TP_SIZE),
        mesh_dim_names=("dp", "pp", "tp"),
    )

    tp_mesh = device_mesh["tp"]
    pp_mesh = device_mesh["pp"]
    dp_mesh = device_mesh["dp"]

    dp_rank = dp_mesh.get_local_rank()
    pp_rank = pp_mesh.get_local_rank()

    is_first_pp = (pp_rank == 0)
    is_last_pp  = (pp_rank == PP_SIZE - 1)
    is_master   = (world_rank == 0)

    if is_master:
        print(f"\n{'='*60}")
        print(f"3D Parallel LLM Training")
        print(f"  DP={DP_SIZE} × PP={PP_SIZE} × TP={TP_SIZE} ({world_size} GPUs)")
        print(f"  Layers: {N_LAYERS} | d_model: {D_MODEL} | Heads: {N_HEADS}")
        print(f"  Vocab: {VOCAB_SIZE} | SeqLen: {SEQ_LEN}")
        print(f"{'='*60}\n")

    # ── Build this rank's stage ──────────────────────────────────────────────
    layers_per_stage = N_LAYERS // PP_SIZE
    start_layer      = pp_rank * layers_per_stage
    end_layer        = start_layer + layers_per_stage

    # Only instantiate the layers owned by this PP rank
    my_layers = nn.ModuleList(
        [TransformerBlock(layer_id=i) for i in range(start_layer, end_layer)]
    )

    freqs_cis = precompute_freqs_cis(D_MODEL // N_HEADS, SEQ_LEN * 2)

    stage = StageModule(
        layers=my_layers,
        freqs_cis=freqs_cis,
        is_first=is_first_pp,
        is_last=is_last_pp,
    ).to(device)

    # ── Step 1: Tensor Parallelism ───────────────────────────────────────────
    # Shard weight matrices across TP ranks. Must happen before FSDP2 so that
    # FSDP2 shards the already-distributed DTensor parameters.
    apply_tp(stage, tp_mesh)

    # ── Step 2: Pipeline Parallelism ─────────────────────────────────────────
    # Create example inputs for PipelineStage to trace shapes.
    if is_first_pp:
        example_input = torch.randint(
            0, VOCAB_SIZE, (MICRO_BATCH, SEQ_LEN), device=device
        )
    else:
        # Middle and last stages receive hidden states
        example_input = torch.zeros(
            MICRO_BATCH, SEQ_LEN, D_MODEL, device=device
        )

    pp_stage = PipelineStage(
        stage,
        pp_rank,
        PP_SIZE,
        device,
        input_args=(example_input,),
    )

    # Number of micro-batches = local batch size / micro-batch size
    # local batch size = GLOBAL_BATCH / DP_SIZE
    n_microbatches = GLOBAL_BATCH // (DP_SIZE * MICRO_BATCH)
    schedule = ScheduleGPipe(pp_stage, n_microbatches=n_microbatches)

    # ── Step 3: FSDP2 Data Parallelism ──────────────────────────────────────
    # Wraps each block and then the full stage with FSDP2 over the dp sub-mesh.
    # FSDP2 handles gradient sync automatically during backward.
    apply_fsdp(stage, dp_mesh)

    # ── Optimizer ────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        stage.parameters(), lr=LR, foreach=True
    )

    # ── Data loader ──────────────────────────────────────────────────────────
    dataset = RandomTokenDataset()
    sampler = torch.utils.data.distributed.DistributedSampler( # type: ignore
        dataset,
        num_replicas=DP_SIZE,
        rank=dp_rank,
        shuffle=True,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=GLOBAL_BATCH // DP_SIZE,   # each DP rank sees this many seqs
        sampler=sampler,
        drop_last=True,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    stage.train()
    step = 0

    for epoch in range(9999):
        sampler.set_epoch(epoch)

        for x, y in loader:
            if step >= MAX_STEPS:
                break

            # Mirror the official example: seed per (step, dp_rank) so that
            # all TP peers on the same DP rank see the same data.
            torch.manual_seed(step + dp_rank)

            x = x.to(device)   # [local_B, SEQ_LEN]  long
            y = y.to(device)   # [local_B, SEQ_LEN]  long

            optimizer.zero_grad()

            # The GPipe schedule coordinates forward/backward across PP stages.
            # - First PP rank:  pass input x
            # - Last  PP rank:  receives logits, computes loss, starts backward
            # - Middle ranks:   just call step() with no args
            losses = []

            if is_first_pp:
                schedule.step(x)
            elif is_last_pp:
                # last stage: pass targets for loss computation
                logits = schedule.step()
                # logits: [local_B, SEQ_LEN, VOCAB_SIZE]
                loss = F.cross_entropy(
                    logits.view(-1, VOCAB_SIZE), # type: ignore
                    y.view(-1),
                )
                loss.backward()
                losses.append(loss.detach())
            else:
                schedule.step()

            # FSDP2 syncs gradients across DP ranks during backward automatically.
            optimizer.step()

            if is_last_pp and dp_rank == 0 and step % LOG_EVERY == 0:
                avg_loss = losses[0].item() if losses else float("nan")
                print(f"  step {step:4d} | loss {avg_loss:.4f}")

            step += 1

        if step >= MAX_STEPS:
            break

    dist.barrier()
    if is_master:
        print("\n3D parallel training complete!")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()