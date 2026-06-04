# Training on Long Sequences (16K+ tokens): Why FSDP Isn't Enough

> **The question:** If `max_length` is, say, 16K tokens, how do GPUs handle it? Even
> the biggest GPUs can't — not even with FSDP. Why, and what actually solves it?

Your intuition is exactly right, and pinpointing *why* is the key insight:
**FSDP shards the wrong thing for this problem.**

---

## The core realization: it's not the parameters, it's the activations

FSDP shards **parameters + gradients + optimizer state** across GPUs. Those costs are
*independent of sequence length* — a 7B model needs the same parameter memory whether you
feed it 512 tokens or 16K.

What blows up with long sequences is **activations** (the intermediate tensors you must keep
for the backward pass), and **FSDP does not shard activations at all**. Every GPU processes
its own full-length sequence and holds the full activation stack for it. So throwing more
GPUs at it via pure data-parallel / FSDP doesn't reduce per-GPU activation memory by even
one byte. That's precisely why "more GPUs + FSDP" doesn't save you here.

---

## Two separate scaling problems at 16K

**1. Attention is quadratic — O(seq²).**
The attention score matrix is `batch × heads × seq × seq`. At seq = 16K that's `16384²`
≈ 268M entries *per head per layer*. If materialized, this is the killer.

**2. Activations are linear in seq, multiplied by depth — O(layers × seq).**
Every layer keeps activations proportional to `batch × seq × hidden`, and you stack that
across all layers.

### Concrete numbers

Model: 7B-ish — `hidden=4096`, `layers=32`, `heads=32`, `seq=16K`, `batch=1`.

Using the standard Megatron activation formula, per layer ≈ `s·b·h·(34 + 5·a·s/h)` bytes:

| Component | Per layer | × 32 layers |
|---|---|---|
| **Naive (full attention matrix materialized)** | ~45 GB | **~1.4 TB** 💀 |
| **+ FlashAttention** (kills the `5·a·s/h` quadratic term) | ~2.3 GB | ~73 GB |
| **+ gradient checkpointing** (store only layer inputs, recompute the rest) | ~0.13 GB | **~4.3 GB** ✅ |

That table *is* the answer. Watch what each technique removes — the quadratic attention term
first, then the layer multiplier.

---

## The toolkit, in the order you actually apply it

### 1. FlashAttention / memory-efficient attention — mandatory
It computes attention in tiles with an online softmax and **never materializes the full
seq×seq matrix**. This single change turns O(seq²) memory into O(seq). Without it, 16K is
basically impossible; with it, attention memory becomes linear.
*(In HuggingFace: `attn_implementation="flash_attention_2"` or `"sdpa"`.)*

### 2. Gradient (activation) checkpointing — almost mandatory for long seq
Instead of storing every layer's activations, store only each layer's *input* and
**recompute** the rest during the backward pass. Trades ~30% extra compute for a massive
memory cut (the 73 GB → 4.3 GB row above).
*(OMNI-Train already supports this — the `gradient_checkpointing` flag, wired into the FSDP
meta-init path.)*

### 3. Context / Sequence Parallelism — the real "FSDP for activations"
This is the piece that directly answers "how do I shard the sequence." You split the
**sequence dimension** across GPUs: with 8 GPUs, each holds `16K / 8 = 2K` tokens' worth of
activations. The catch is attention, where every token must attend to every other token — so
you need cross-GPU communication. The canonical algorithm is **Ring Attention**: each GPU
holds a chunk of the sequence, and K/V blocks are passed around a ring so each query block
sees all key blocks, without any single GPU ever holding the full sequence. This is how
frontier models train at 128K–1M context.

### 4. Tensor Parallelism
Shards the hidden dimension / attention heads, shrinking activation *width* per GPU.
Complements context parallelism.

### 5. Smaller levers
- Micro-batch = 1
- fp8 / bf16 activations
- CPU / NVMe activation offload (slow but works)
- **Architectural** fixes: sliding-window / sparse attention (Mistral, Longformer) that
  avoid full O(seq²) attention entirely.

---

## The full real-world recipe

Training at 16K+ in practice is **4D parallelism**:

```
Data / FSDP   →  shards params / grads / optimizer   (the seq-independent cost)
Tensor        →  shards hidden dim / heads            (activation width)
Pipeline      →  shards layers across GPUs            (depth)
Context / Seq →  shards the sequence itself           (the seq-dependent activation cost)  ← the missing axis
```

…all sitting on top of **FlashAttention + gradient checkpointing** per GPU.

---

## Tie-back to OMNI-Train

- **Gradient checkpointing** ✅ — already supported (`gradient_checkpointing` flag).
- **FlashAttention** — available via HuggingFace `attn_implementation`. These two alone get a
  moderate model to 16K on a single 80GB A100.
- **FSDP** ✅ — handles the param / grad / optimizer axis.
- The genuinely missing axis for "shard the sequence" is **context / sequence parallelism**.
  `parallelism.py` is currently an experimental **3D** mesh (data × tensor × pipeline) —
  context parallelism would be the **4th** dimension to add there, and it's the one
  specifically built for this problem.

---

## TL;DR

The gut feeling "even the biggest GPU with FSDP can't do this" is **correct for the naive
setup**. The escape isn't a bigger GPU; it's:

1. **Stop materializing the quadratic attention matrix** → FlashAttention
2. **Stop storing all activations** → gradient checkpointing
3. **Shard the sequence itself across GPUs** → context / sequence parallelism (Ring Attention)

FSDP shards the model state; long-sequence cost lives in activations. Different problem,
different parallelism axis.

---

## See also

- [`3d_parallelism_guide.md`](3d_parallelism_guide.md) — the data × tensor × pipeline mesh
- [`3d_parallelism_research.md`](3d_parallelism_research.md) — background research
- [`TECHNICAL.md`](TECHNICAL.md) — architecture internals
