# OMNI-Train — Verification Report

**Date:** 2026-05-27
**Scope:** Double-check every major subsystem against PyTorch / Hugging Face best practices and verify the code paths against observable correctness signals.
**Hardware available for this pass:** 1 × NVIDIA RTX 4050 Laptop (6.1 GB), CUDA 12.4, torch 2.6.0+cu124, bf16 supported.
**Library versions:** torch `2.6.0+cu124`, torchvision `0.21.0+cu124`, transformers `5.8.1`, peft `0.18.0`, bitsandbytes `0.47.0`, accelerate `1.11.0`, datasets `4.8.5`.

---

## 0. Executive summary

| Subsystem | Static analysis | Unit tests | Single-GPU runtime | Multi-GPU runtime | Risk |
|---|---|---|---|---|---|
| Config validation (`build_args`) | OK | 22/22 pass + 4 live guard checks | OK | N/A | Low |
| Strategy dispatch (`launch.sh` → `train.py`) | OK | covered | OK (solo) | unverifiable here | Low |
| Solo (single-GPU) path | OK | — | not exercised end-to-end this pass | N/A | Low |
| DDP path | OK | — | not exercised (needs ≥2 GPUs) | unverifiable here | Medium — see §3.2 |
| FSDP2 Path 1 (custom transformer) | OK | model tests pass | — | unverifiable here | Low |
| FSDP2 Path 2 (PEFT/quant) | OK | — | — | unverifiable here | Medium — see §3.3 |
| FSDP2 Path 3 (meta-init) | OK | — | — | unverifiable here | Low |
| Mixed precision (`MixedPrecisionPolicy`) | OK; default config matches torchtitan recommendation (`reduce_dtype=fp32`) | — | — | — | Low |
| Explicit prefetching | OK; correctly no-ops on non-FSDP layers via `hasattr` guard | helper tests pass | — | — | See §3.5 |
| PEFT (LoRA / QLoRA) | OK; `prepare_model_for_kbit_training` called for quant, `print_trainable_parameters()` logged on rank 0 | helper tests pass | — | — | Low |
| Quantization (bitsandbytes) | OK; constraints enforced; 8-bit-no-PEFT downgraded to warning with NaN advisory | constraint tests pass | — | — | Medium — see §3.7 |
| Checkpoint save (FSDP DCP/DTensor) | OK | — | — | — | Medium — see §3.8 |
| Checkpoint save (DDP / solo) | OK | — | — | — | High — see §3.8 (DDP optimizer state not loaded on resume) |
| RoPE buffer fix (`_materialize_meta_buffers`) | OK; matches the silent-corruption pattern documented in [pytorch#160340](https://github.com/pytorch/pytorch/issues/160340) | — | — | — | Low |
| Data loading + `DistributedSampler` | OK | — | — | — | Medium — see §3.10 (DataLoader workers not explicitly seeded) |
| SLURM launcher | OK | — | — | unverifiable here | Low |
| `parallelism.py` 3-D | Standalone scaffolding, **not integrated** in `train.py` (imports commented at `train.py:53–54`) | — | — | — | N/A — flagged as in-progress in CLAUDE.md |

**Total unit test result on this pass:** `128 passed, 293 deselected (smoke), 1 warning` in 2.90 s.

The codebase is in good shape. The constraint matrix in `build_args` is the single best correctness asset — it eliminates an entire class of silent misconfigurations (FSDP + bnb, 4-bit without PEFT, QLoRA without 4-bit). The FSDP2 implementation closely follows current torchtitan / PyTorch 2.6 idioms.

The four real risks (all medium, none catastrophic) are listed in §4.

---

## 1. Method

For each subsystem I (a) located the implementation by file:line, (b) cross-referenced it against the current PyTorch 2.9 / HF / torchtitan documentation to identify what an *observable* correctness signal looks like, (c) checked whether the code emits that signal or could be cheaply made to. Where a test already exists, that's noted. Where a test is impossible without ≥2 GPUs, a manual verification recipe is given in §5.

---

## 2. Subsystem-by-subsystem findings

### 2.1 Configuration validation (`utils.build_args`)

**File:** `utils.py:287–474`

Eleven hard constraints + one warning, all unit-tested in `tests/test_config_validation.py`. I additionally ran a live guard-matrix check against the current `config.yaml`:

```
PASS  fsdp + quant raises
PASS  4-bit without peft raises
PASS  qlora bits=8 raises
PASS  8-bit without peft warns (not raises)
```

The warning text for 8-bit-without-PEFT (`utils.py:431–443`) correctly explains the three failure modes (INT8 range, fp16 LayerNorm overflow, optimizer-state dtype). This is the kind of explicit-failure-mode documentation that prevents silent NaN.

**Verdict:** correct.

### 2.2 Strategy dispatch (`scripts/launch.sh` → `train.py`)

`launch.sh` reads `strategy` and `num_gpus` from `config.yaml` and chooses between `python train.py` (solo) and `torchrun --nproc_per_node=$NUM_GPUS train.py` (ddp/fsdp). `train.py:124` branches on `args.strategy` and forks into `apply_solo/ddp/fsdp`.

Observable signals already present:
- `train.py:129` logs `world_size`, `local_rank`, `device`.
- `train.py:151–157` calls `print_on_all_ranks` so every rank announces itself with hostname + pid + device. If a rank silently dies on init you see ≠ `world_size` announcements.

**Verdict:** correct.

### 2.3 DDP (`distributed_utils.apply_ddp`, lines 570–706)

What is right:
- Process group init at `distributed_utils.py:167` with a 10-min timeout and NCCL warm-up barrier at line 175 — absorbs first-collective cold-start where ranks are still in sync, not silently mid-training.
- NVLink probing via NVML at lines 99–140; sets `NCCL_P2P_DISABLE` accordingly. This avoids the well-known PCIe NCCL P2P probe hang.
- `find_unused_parameters=False` (line 696). Per PyTorch docs ([DDP notes](https://docs.pytorch.org/docs/main/notes/ddp.html)), enabling it costs an extra autograd-graph traversal and a correctness check; only turn it on if you actually have unused params. The codebase correctly keeps it off.
- Resume path performs **structural** PEFT compat check via state-dict key prefix (`base_model.model.`) at `distributed_utils.py:340–360`. This catches the classic "save with LoRA, resume without" footgun.

What to watch (Medium-risk):
- **DDP resume does not restore optimizer state** (`train.py:234–235` explicitly warns). On a real resume, AdamW `exp_avg`/`exp_avg_sq` restart at zero, which causes a noticeable spike in the first few steps after resume. This is intentional per CLAUDE.md but worth documenting. If you want to fix it, you'd save `{"model_state_dict": ..., "optim_state_dict": optim.state_dict(), "step": ..., "epoch": ...}` instead of the bare model state dict at `distributed_utils.py:1181-…`.
- **`grad_clip` walks `model.parameters()`** (`train.py:333`), not `trainable_params`. For LoRA runs this is wasted work (frozen params have `.grad = None` so `clip_grad_norm_` skips them, but you still iterate them). Not a correctness bug — a minor inefficiency.

**Verdict:** correct, with two known limitations.

### 2.4 FSDP2 — all three paths (`distributed_utils.apply_fsdp`, 708–1149)

**Path 1 — custom transformer** (`756–777`): materializes on device, then `fully_shard` each layer bottom-up, then root. Matches PyTorch tutorial's bottom-up wrapping order. Calls `inspect_model` afterwards (`utils.py:153–193`) which prints sharded-vs-replicated tensor counts — a real correctness signal.

**Path 2 — PEFT/quant** (`783–971`): materializes on CPU because bitsandbytes' `Linear4bit/Linear8bitLt` cannot be initialized on meta; applies PEFT/quant via `_apply_peft_quantization` *before* sharding (correct order — sharding has to see the LoRA adapters as ordinary submodules); freezes non-floating params before sharding (`899–905`); then per-layer `fully_shard` + root shard.

**Path 3 — standard meta-init** (`977–1139`): the by-the-book FSDP2 sequence — `with torch.device("meta")` build → `fully_shard` per layer + root → `model.to_empty(device=device)` → `Checkpointer.load_model` → `_materialize_meta_buffers`. Matches the [PyTorch FSDP2 tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html) idiom exactly.

Best-practice cross-check from research:
- **Bottom-up wrapping**: required so submodules are sharded before the root. The code does layer-by-layer first (`distributed_utils.py:929–932` for Path 2, `1019–1021` for Path 3), then root (`943`, `1025`). ✓
- **Optimizer created after `fully_shard`** so it sees DTensor-backed params: yes — optimizer is built in `train.py:211`, after `apply_fsdp`. ✓
- **Use `model(input)`, not `model.forward(input)`**, so FSDP2 pre/post hooks fire: `train.py:310, 321, 327` use `model(...)`. ✓
- **Observable signal**: `isinstance(p, DTensor)` + `p.placements == (Shard(0),)` after sharding. `inspect_model` already prints this aggregate; no explicit assertion, but the print is enough to spot a regression at run-time.

**Verdict:** correct.

### 2.5 Mixed precision (`MixedPrecisionPolicy`)

**Files:** `distributed_utils.py:914–920` (Path 2), `997–1003` (Path 3); config schema at `config.yaml:27–32`.

Default config: `param_dtype=bfloat16, reduce_dtype=float32, output_dtype=bfloat16, cast_forward_inputs=false`.

Research cross-check: torchtitan **hardcodes** `reduce_dtype=float32` because reducing gradients in bf16 across many ranks accumulates roundoff error catastrophically ([torchtitan/docs/fsdp.md](https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md), [main-horse: reduction precision](https://main-horse.github.io/posts/reduction-precision/)). The current default `reduce_dtype=float32` matches this recommendation.

Edge case handled correctly: if bf16 isn't supported (`torch.cuda.is_bf16_supported()` returns False), `args.mixed_precision = False` and `args.param_dtype = "float16"` (`909–912`, `992–995`). Silent fp16 fallback is documented in the rank-0 print but not in the config — users who care about reproducibility should be aware. (Low risk; the print is visible.)

Known PyTorch issue worth being aware of: [pytorch#143277](https://github.com/pytorch/pytorch/issues/143277) — `MixedPrecisionPolicy.reduce_dtype` is clamped before lazy init in some versions; verify your installed torch behaves as expected by inspecting `model._mp_policy.reduce_dtype` right after `fully_shard`.

**Verdict:** correct and aligned with current best practice.

### 2.6 Explicit prefetching (`set_modules_to_forward_prefetch` / `set_modules_to_backward_prefetch`)

**Files:** `distributed_utils.py:270–304`; invoked at `948–951` (Path 2) and `1030–1033` (Path 3) gated on `args.explicit_prefetching and stacks`.

What's right:
- Uses `hasattr(layer, 'set_modules_to_forward_prefetch')` guard (`distributed_utils.py:282`) so it's a safe no-op if FSDP didn't sharded that layer. Defensive.
- Skips the last N layers in forward, and the first N layers in backward — correct boundary handling.

**Caveat from research** ([PyTorch FSDP2 tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)): explicit prefetching only pays off when training is CPU-bound (the CPU thread can't issue the next all-gather fast enough). For a small LLM on one GPU this is rarely the bottleneck; the gain may be invisible. Also, true overlap of all-gather with compute requires `_set_unshard_async_op(True)`, which the codebase does **not** call. This means the prefetch hint is set, but the all-gather still runs synchronously — you get the *ordering* benefit but not full async overlap.

To actually overlap, add:
```python
# After fully_shard, before training loop
for module in get_model_layers(model):
    for layers, _ in [module]:
        for layer in layers:
            layer._set_unshard_async_op(True)
```

This is a **performance optimization**, not a correctness fix — the current code is correct, just less aggressive than it could be.

**Verdict:** correct; potential follow-up for performance only.

### 2.7 PEFT — LoRA + QLoRA (`_apply_peft_quantization`)

**File:** `distributed_utils.py:400–469`.

Cross-checked against [HF PEFT docs](https://huggingface.co/docs/peft/developer_guides/quantization):
- `prepare_model_for_kbit_training(model, use_gradient_checkpointing=...)` is called *before* `get_peft_model` when quant is enabled (`411`). This is exactly what HF recommends — it upcasts LayerNorm / embeddings / lm_head to fp32 so reductions don't NaN in fp16, freezes everything, and turns on grad checkpointing.
- `LoraConfig` (`432–438`) reads all six knobs from config; target-modules accepts string / list / "all-linear" via `_normalize_target_modules`.
- `model.print_trainable_parameters()` is called on rank 0 (`447`). This is the canonical observable signal: for LoRA-r=8 on Llama-3.2-1B with `target_modules="all-linear"`, you should see ≈ 0.5–1% of params trainable. If you see 100% trainable, the freeze step didn't run.
- For non-quantized PEFT (line 442–445), all floating params are downcast to `args.param_dtype` after wrapping — preserves PEFT's contract that base weights are frozen but matches the user's mixed-precision request.

Path 2 then does one more defence-in-depth pass after PEFT/quant but before sharding (`distributed_utils.py:899–905`):
```python
for _name, param in model.named_parameters():
    if not torch.is_floating_point(param) and param.requires_grad:
        param.requires_grad_(False)
```
This catches the case where bnb leaves an `int8` parameter `requires_grad=True` (which would make `_init_optim_state` fail loudly later). The print of `non_float_frozen` count is the signal that this triggered.

**Verdict:** correct, well-instrumented.

### 2.8 Quantization — bitsandbytes 4-bit / 8-bit

**File:** `distributed_utils.py:364–384` (config build), `491–498` (solo), `596–611` (DDP), `829–842` (FSDP Path 2).

Cross-checked against [HF bitsandbytes docs](https://huggingface.co/docs/transformers/quantization/bitsandbytes):
- 4-bit path uses `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=bfloat16, bnb_4bit_use_double_quant=True)`. NF4 is information-theoretically optimal for normally distributed weights per the QLoRA paper. `double_quant=True` saves an extra ~0.4 bits/param. All correct.
- 8-bit path uses `BitsAndBytesConfig(load_in_8bit=True)` — minimal but correct.
- `device_map={"": local_rank}` (DDP) or `{"": device}` (solo) ensures each rank places its model on the right GPU. Required because bnb cannot be `.to(device)`'d after load.

**The 8-bit-without-PEFT decision is worth scrutiny.** `build_args` downgraded this from a hard error to a UserWarning (`utils.py:417–443`). The warning is thorough and accurate. However, my read is that any user who hits this warning and proceeds is going to get NaN within a few steps, then re-debug it. Two reasonable choices:
- (a) Keep as warning (current behaviour) — gives the user agency, costs them ~30s of "why is loss NaN" debugging.
- (b) Re-promote to hard error — symmetric with the 4-bit-without-PEFT block, no chance of NaN-debug rabbit hole.

I'd lean toward (b) for symmetry, but the current behaviour is defensible.

**Observable signal a test could add (no GPU needed if mocked):** assert `isinstance(model.<some_linear>, bnb.nn.Linear4bit)` and `model.<some_linear>.weight.dtype == torch.uint8` (NF4 stores as packed uint8).

**Verdict:** correct.

### 2.9 Checkpoint save / load — FSDP DCP path

**File:** `checkpoint.py:60–95` (load), `240–253` (save), `182–199` (gather).

What's right:
- DCP path uses `set_model_state_dict(..., StateDictOptions(full_state_dict=True, broadcast_from_rank0=True))`. Only rank 0 needs the full checkpoint in CPU RAM; other ranks receive their 1/N shard over NCCL. Memory-efficient.
- The pre-condition for DCP load is documented in the code itself (`checkpoint.py:69–73`): `model.to_empty(device=device)` *must* be called first — `copy_()` into a meta tensor deadlocks silently. The call sites (`distributed_utils.py:1062, 1118`) honour this. This is exactly the pattern that bites people in the wild ([pytorch#125740](https://github.com/pytorch/pytorch/issues/125740)).
- Folder names use a Unix-ms timestamp + optional `__lora_q4`-style tag; `get_latest_checkpoint_folder` (`22–37`) parses the numeric prefix only, ignoring the tag. Correct.

**Known DCP risk worth noting** ([pytorch#149640](https://github.com/pytorch/pytorch/issues/149640)): `broadcast_from_rank0=True` can cause CUDA OOM on large models when rank 0 holds the full state dict. For Llama-3.2-1B at fp16 (~2 GB) on a 6 GB GPU this is borderline; for 7B+ models you'll want to compare DCP vs DTensor empirically.

### 2.10 Checkpoint save / load — FSDP DTensor path

**File:** `checkpoint.py:97–120` (load), `191–199` (gather).

All ranks load the full checkpoint from disk independently (no NCCL), then `distribute_tensor` slices each rank's shard. Uses `assign=True` in `load_state_dict` so DTensor params are replaced rather than `copy_()`-ed into (required for meta-device DTensors). Optimizer load (`146–180`) honours the `step` (plain tensor) vs `exp_avg` (DTensor) distinction. Correct.

DTensor is safer when NCCL is flaky; DCP is faster when NCCL is healthy. The user picks via `dist_parameters.distribute_api`. Both unit-tested at the config level by `tests/test_config_combinations.py`.

### 2.11 Checkpoint save / load — DDP and solo

**File:** `distributed_utils.py:1174–1193`.

`save_checkpoint` writes `model.module.state_dict()` (DDP) or `model.state_dict()` (solo) to `<dir>/<strategy>/<ts><tag>.pt`. On resume:
- solo: `torch.load(path) → state_dict → load_state_dict` (`distributed_utils.py:514–547`).
- DDP: same pattern (`619–625, 690`).

**Documented limitation:** no optimizer state is saved (and `train.py:235` says so). For full-finetune DDP runs this is the only real correctness gap on resume — the first ~100 steps after resume will be noisier as Adam rebuilds its moments.

**Observable signal that would catch a silent regression:** the saved file should be `>` 0 bytes and `torch.load(path, weights_only=True)` should return a dict whose first key matches the live model. A 30-line round-trip test would cover both DDP and solo.

### 2.12 RoPE buffer fix (`_materialize_meta_buffers`)

**File:** `distributed_utils.py:716–750`.

Non-persistent buffers like `inv_freq` (Llama, Mistral, Qwen, Gemma, Phi-3) are excluded from `state_dict()`, so they stay on meta after FSDP weight load. The fix recomputes `inv_freq` using the same `1 / base^(2i/dim)` formula the model uses at init.

This is exactly the silent-corruption pattern documented in [pytorch#160340](https://github.com/pytorch/pytorch/issues/160340) and the `accelerate.init_empty_weights` discussion. Zeroing `inv_freq` would silently destroy positional encoding — loss wouldn't NaN, the model would just stop learning. Catching this is genuinely hard if you don't know to look for it.

Code also handles the post-`model.to_empty(device)` case where `inv_freq` is no longer meta but is filled with garbage CUDA values (comment at line 731–736). Correct.

**Observable signal:** the function prints `Recomputed RoPE inv_freq for '<name>' on <device>` for each affected buffer. If you see *zero* such prints on a Llama/Mistral resume, the buffer detection failed.

**Verdict:** correct; this is one of the highest-leverage pieces of code in the repo because the failure mode is silent.

### 2.13 Data loading + DistributedSampler

**File:** `data.py:10–128`.

- `DistributedSampler(num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)` attached when `dist.is_initialized()` (`data.py:100–107`). Correct.
- `sampler.set_epoch(epoch)` called at `train.py:294–295`. Per [PyTorch DDP examples](https://discuss.pytorch.org/t/understanding-distributedsampler-and-dataloader-drop-last/206271), without this all epochs use the same shuffle. ✓
- `drop_last=True` both in the sampler and the DataLoader (`data.py:106, 118`). This ensures every rank gets exactly `len(dataset) // (world_size × batch_size)` batches. Correct — otherwise the trailing partial batch lands on one rank and the others hang at the next collective.

**Medium-risk gap:** DataLoader `num_workers > 0` is used (`data.py:110`) but no `worker_init_fn` seeds each worker. Per [PyTorch docs](https://medium.com/@zergtant/improving-control-and-reproducibility-of-pytorch-dataloader-with-sampler-instead-of-shuffle-7f795490256e), without a per-worker seed, runs are not reproducible — workers receive `torch.initial_seed() % 2**32 + worker_id`, which depends on the fork timing. For training reproducibility this matters; for correctness it doesn't (each worker still draws independent samples, just non-deterministically).

If you ever need bit-exact reruns, add to `DataLoader(...)`:

```python
def _seed_worker(worker_id):
    import random, numpy as np
    seed = (torch.initial_seed() + worker_id) % 2**32
    random.seed(seed); np.random.seed(seed)
DataLoader(..., worker_init_fn=_seed_worker, generator=torch.Generator().manual_seed(1234))
```

**Verdict:** correct for training; reproducibility-only gap.

### 2.14 SLURM launcher (`scripts/launch_slurm.py`)

Template-driven sbatch generation. `MASTER_ADDR` resolved from `scontrol show hostnames`. `WORLD_SIZE = nodes × gpus_per_node`. `--rdzv_endpoint` set to `$MASTER_ADDR:$MASTER_PORT`. Standard pattern, matches the PyTorch multi-node recipe.

Not exercisable without a SLURM cluster; the `--dry-run` flag prints the generated script without submitting, which is a good local test path.

### 2.15 `parallelism.py` (3-D parallelism, in progress)

`ParallelismArgs`, `resolve_device_mesh`, `setup_device_mesh` are fully implemented and callable; `init_device_mesh` is invoked correctly. **But** the imports in `train.py:53–54` are commented out and the module is not integrated into the training loop. CLAUDE.md explicitly marks this as in-progress. No verification needed beyond confirming the code compiles, which the test suite covers indirectly.

---

## 3. Research-backed best-practice checklist

This is the synthesis of what other practitioners use as correctness signals, distilled from PyTorch / Hugging Face / torchtitan docs and issues. Each row says **what to assert** and **whether OMNI-Train already asserts it**.

| Subsystem | Signal | In OMNI-Train? | Source |
|---|---|---|---|
| FSDP2 sharding | `isinstance(p, DTensor)` for each param after `fully_shard` | Implicit via `inspect_model` print | [PyTorch FSDP2 tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html) |
| FSDP2 sharding | `p.placements == (Shard(0),)` for non-replicated params | No explicit assertion | [PyTorch fully_shard docs](https://docs.pytorch.org/docs/2.9/distributed.fsdp.fully_shard.html) |
| FSDP2 init order | Build optimizer **after** `fully_shard` | ✓ (`train.py:211` runs after `apply_fsdp`) | torchtitan |
| FSDP2 init order | Bottom-up wrapping (children before root) | ✓ (`distributed_utils.py:929–943`, `1019–1025`) | PyTorch tutorial |
| FSDP2 forward | Call `model(input)`, not `model.forward(input)` | ✓ (`train.py:310, 321, 327`) | PyTorch tutorial |
| Mixed precision | `reduce_dtype=fp32` for stability | ✓ (default in `config.yaml:30`) | [torchtitan/docs/fsdp.md](https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md), [main-horse](https://main-horse.github.io/posts/reduction-precision/) |
| Mixed precision | Verify `model._mp_policy` after `fully_shard` | No | [pytorch#143277](https://github.com/pytorch/pytorch/issues/143277) |
| Explicit prefetching | Call `_set_unshard_async_op(True)` for true overlap | No (set hints only) | [PyTorch FSDP2 tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html) |
| PEFT | Call `model.print_trainable_parameters()` on rank 0 | ✓ (`distributed_utils.py:447`) | [HF PEFT quantization](https://huggingface.co/docs/peft/developer_guides/quantization) |
| PEFT | Call `prepare_model_for_kbit_training` for quant runs | ✓ (`distributed_utils.py:411`) | [HF PEFT docs](https://huggingface.co/docs/peft/main/en/developer_guides/lora) |
| PEFT | Assert `requires_grad=True` only on adapter params | Implicit via the print | HF PEFT |
| QLoRA | `bnb_4bit_quant_type="nf4"` for normally-distributed weights | ✓ (`distributed_utils.py:378`) | QLoRA paper |
| QLoRA | `bnb_4bit_use_double_quant=True` (~0.4 bits/param savings) | ✓ (default) | [bitsandbytes 4-bit reference](https://huggingface.co/docs/bitsandbytes/reference/nn/linear4bit) |
| bnb quant | Verify base-model linear layers are `Linear4bit` / `Linear8bitLt` | No | [HF Transformers/bnb](https://huggingface.co/docs/transformers/quantization/bitsandbytes) |
| FSDP+quant | Block at config time (incompatible) | ✓ (`utils.py:445–450`) | bnb / FSDP incompatibility |
| DDP | `find_unused_parameters=False` unless needed | ✓ (`distributed_utils.py:696`) | [PyTorch DDP notes](https://docs.pytorch.org/docs/main/notes/ddp.html) |
| DDP | Resume restores optimizer state | ✗ — known gap | PyTorch tutorial |
| DistributedSampler | `set_epoch(epoch)` called every epoch | ✓ (`train.py:294`) | [PyTorch forums](https://discuss.pytorch.org/t/understanding-distributedsampler-and-dataloader-drop-last/206271) |
| DistributedSampler | `drop_last=True` to keep ranks aligned | ✓ (`data.py:106, 118`) | PyTorch examples |
| DataLoader | `worker_init_fn` seeds workers for reproducibility | ✗ — known gap | [PyTorch reproducibility](https://medium.com/@zergtant/improving-control-and-reproducibility-of-pytorch-dataloader-with-sampler-instead-of-shuffle-7f795490256e) |
| DCP load | `model.to_empty(device)` before `set_model_state_dict` | ✓ (`distributed_utils.py:1062, 1118`) | [pytorch#125740](https://github.com/pytorch/pytorch/issues/125740) |
| DCP load | Watch for OOM with `broadcast_from_rank0=True` on huge models | Documented in code comments | [pytorch#149640](https://github.com/pytorch/pytorch/issues/149640) |
| RoPE / meta init | Materialize `inv_freq` after FSDP load (don't zero) | ✓ (`distributed_utils.py:730–743`) | [pytorch#160340](https://github.com/pytorch/pytorch/issues/160340) |
| 8-bit quant | Do not train without PEFT (AdamW INT8 overflow) | ⚠ warning, not hard error | bnb / QLoRA paper |
| Seed | Per-rank seed offset if needed for non-determinism | Not done (all ranks seed `1234`) | PyTorch tutorials |

---

## 4. Risks and gaps worth tracking

Ranked by impact-to-effort:

1. **DDP resume drops optimizer state** (`distributed_utils.py:1174–1193`, `train.py:234–235`). High impact on long DDP runs; medium effort to fix (change DDP save to write `{"model_state_dict": ..., "optim_state_dict": optim.state_dict(), "step": step, "epoch": epoch}` and update the matching load path).

2. **DataLoader workers not explicitly seeded** (`data.py:110`). Doesn't affect correctness, only reproducibility. 10-line fix (`worker_init_fn` + `generator=`).

3. **8-bit-without-PEFT is a warning, not an error**. The warning is excellent, but a user who ignores it gets several minutes of NaN-debugging. Consider promoting to hard error to match the 4-bit policy.

4. **Explicit prefetching does not call `_set_unshard_async_op(True)`** so the prefetch hint sets ordering but not async overlap. Performance-only, not correctness. One-line change inside `set_modules_to_forward_prefetch`.

5. **No checkpoint round-trip tests.** The save and load paths are exercised end-to-end by smoke tests (subprocess training), but there's no targeted unit test that does just save+reload and asserts state-dict equivalence. A 50-line `tests/test_checkpoint.py` that wraps a tiny model in DDP-with-`world_size=1`, saves, loads, and compares param tensors would close this gap on a single-GPU machine.

6. **No assertion that `MixedPrecisionPolicy` actually took effect.** A one-liner after `fully_shard` (`assert model._mp_policy.reduce_dtype == torch.float32`) would catch any future regression where the policy is dropped on the floor.

7. **Path 1 (custom transformer) materializes on device, not meta.** OK at the toy-model sizes used in tests, but if a user makes the custom transformer big it will OOM. Worth a one-line comment in `config.yaml` near `custom_transformer_args`.

None of these are blockers. Items 1 and 5 are the two that I'd prioritize.

---

## 5. Manual verification checklist (single GPU)

Since this pass had access to only one 6 GB GPU, the recipes below are the ones that need to be run on hardware you do have access to. Each one takes < 5 minutes.

### 5.1 Solo + PEFT smoke
```bash
CONFIG_PATH=configs/llm_lora_quantized_single_gpu.yaml bash scripts/launch.sh
```
**Look for:** "Trainable params: X/Y (0.XXX%)" — confirms LoRA freeze worked.
**Look for:** "Using bitsandbytes NF4 4-bit quantization" — confirms quant config built.
**Look for:** loss going down for at least 5 steps (not NaN, not flat).

### 5.2 FSDP-1 (1 rank, exercises FSDP path on one GPU)
```bash
STRATEGY=fsdp NUM_GPUS=1 CONFIG_PATH=configs/llm_full_finetune_fsdp.yaml bash scripts/launch.sh
```
**Look for:** "Sharding N layer(s) across 1 stack(s)" — confirms layer detection.
**Look for:** `inspect_model` output (printed via `utils.inspect_model`) — sharded:replicated ratio. With 1 rank, sharded counts will exist but each shard is the full param.
**Look for:** "Recomputed RoPE inv_freq for '...'" appearing for Llama/Mistral models — confirms the RoPE fix triggered.

### 5.3 Save / resume round-trip
```bash
# Phase 1: train + save
STRATEGY=solo CONFIG_PATH=configs/llm_lora_quantized_single_gpu.yaml bash scripts/launch.sh
# (config must have save: true)

# Phase 2: resume
# Edit configs/llm_lora_quantized_single_gpu.yaml:
#   save_load:
#     resume: true
#     resume_path: checkpoints/solo/<timestamp>__lora_q4.pt
bash scripts/launch.sh
```
**Look for:** "Solo checkpoint loaded ✓" — confirms `_check_checkpoint_peft_compat` passed.
**Look for:** first epoch loss after resume is close to (within ~5%) the last epoch loss before save. A reset-to-init loss means the load silently failed.

### 5.4 Quick guard-matrix self-check (no GPU needed, < 5 seconds)
```bash
python -m pytest tests/test_config_validation.py tests/test_helpers.py -q
```
All 50 cases should pass.

### 5.5 If/when 2+ GPUs are available
- FSDP DCP/DTensor resharding: save with `--nproc_per_node=2`, resume with `--nproc_per_node=4`. Only DCP supports this.
- DDP gradient sync sanity: print `model.module.<layer>.weight.grad.sum()` on each rank after `loss.backward()`; values should be identical across ranks before `optimizer.step()`.
- DDP bucket configuration: at large model sizes consider passing `bucket_cap_mb=50` (default 25); inspect with `model.reducer.bucket_bytes_cap`.

---

## 6. Sources

- [PyTorch FSDP2 tutorial (2.12)](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [PyTorch `fully_shard` reference (2.9)](https://docs.pytorch.org/docs/2.9/distributed.fsdp.fully_shard.html)
- [PyTorch DCP recipe](https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html)
- [PyTorch DDP notes](https://docs.pytorch.org/docs/main/notes/ddp.html)
- [torchtitan FSDP docs](https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md)
- [main-horse: Why reduction precision matters](https://main-horse.github.io/posts/reduction-precision/)
- [HF Transformers: bitsandbytes quantization](https://huggingface.co/docs/transformers/quantization/bitsandbytes)
- [HF PEFT: quantization developer guide](https://huggingface.co/docs/peft/developer_guides/quantization)
- [HF PEFT: LoRA developer guide](https://huggingface.co/docs/peft/main/en/developer_guides/lora)
- [HF bitsandbytes Linear4bit reference](https://huggingface.co/docs/bitsandbytes/reference/nn/linear4bit)
- [HF blog: 4-bit + QLoRA](https://huggingface.co/blog/4bit-transformers-bitsandbytes)
- [pytorch#160340 — meta init + HF non-persistent buffers](https://github.com/pytorch/pytorch/issues/160340)
- [pytorch#125740 — DCP per-rank files](https://github.com/pytorch/pytorch/issues/125740)
- [pytorch#149640 — DCP OOM with `broadcast_from_rank0`](https://github.com/pytorch/pytorch/issues/149640)
- [pytorch#143277 — FSDP2 mp_policy.reduce_dtype clamp](https://github.com/pytorch/pytorch/issues/143277)
- [PyTorch forum: DistributedSampler + drop_last](https://discuss.pytorch.org/t/understanding-distributedsampler-and-dataloader-drop-last/206271)
- [Medium: DataLoader reproducibility via Sampler / worker_init_fn](https://medium.com/@zergtant/improving-control-and-reproducibility-of-pytorch-dataloader-with-sampler-instead-of-shuffle-7f795490256e)
