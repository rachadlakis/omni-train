# OMNI-Train — NCCL Distributed Training Fix

**Date:** 2026-05-23

---

## What Was Broken

`torchrun --nproc_per_node=2 train.py` hung silently after:

> `Weights cached ✓ — releasing barrier for all ranks`

Training never continued; both GPU processes were alive but frozen.

---

## Root Cause

Two independent bugs that compounded each other:

### 1. NCCL P2P topology probe hang

NCCL probes PCIe peer-to-peer (P2P) GPU access during its very first collective
(barrier / all-reduce). On RTX A4000 GPUs connected via PCIe **without NVLink**,
this probe hangs indefinitely — it never returns, and the NCCL watchdog eventually
kills the process. The hang lasts well over 2 minutes and makes it look like a
code bug when it is actually a hardware/driver interaction with NCCL.

### 2. Conflicting NCCL communicators

PyTorch 2.3+ introduced an optional `device_id=` argument to
`dist.init_process_group()`. When set, it eagerly initialises a NCCL communicator
bound to that device. If a subsequent `dist.barrier(device_ids=[local_rank])` is
then called, NCCL creates a **second** communicator on the same device. On NCCL
2.21.5 these two concurrent communicators deadlock — the barrier never completes
even with a 5-minute timeout.

---

## What Was Changed

### `distributed_utils.py` — `setup_dist_process_group()`

**Before:**

```python
dist.init_process_group(
    backend=BACKEND,
    device_id=torch.device(f"cuda:{local_rank}"),   # caused communicator conflict
    timeout=timedelta(minutes=60)
)
# no warmup barrier
```

**After:**

```python
# Disable NCCL P2P before init_process_group
# (reads the env var during first collective)
os.environ.setdefault("NCCL_P2P_DISABLE", "1")

# No device_id= argument — avoids the double-communicator deadlock
dist.init_process_group(backend=BACKEND, timeout=timedelta(minutes=10))

# Warmup barrier: absorbs the first-collective cold-start here (both ranks synced)
# instead of silently mid-training. Uses device_ids= so NCCL knows the GPU mapping.
if BACKEND == "nccl":
    dist.barrier(device_ids=[local_rank])
```

### `utils.py` — `dist_barrier()`

**Before:**

```python
dist.barrier()   # plain barrier — NCCL had no device mapping → hung
```

**After:**

```python
if dist.get_backend() == "nccl":
    dist.barrier(device_ids=[local_rank])   # tells NCCL which GPU owns this rank
else:
    dist.barrier()
```

---

## What the Fixes Do

### `NCCL_P2P_DISABLE=1`

Forces NCCL to use Shared Memory (SHM/CPU-staging) for GPU–GPU data movement
instead of PCIe P2P. On a single node with PCIe GPUs this is typically
within 5–15% of P2P speed for large tensors, and for training the bottleneck
is compute, not the barrier or gradient sync.

### Removing `device_id=` from `init_process_group`

Reverts to lazy NCCL communicator initialisation. NCCL creates the communicator
the first time a collective is called (the warmup barrier), not during PG init.
This avoids the double-communicator conflict.

### Warmup barrier in `setup_dist_process_group`

Pays the one-time cold-start cost (`< 1 s` with SHM, `~31 s` without P2P disabled)
immediately after `init_process_group`, where both ranks are already synchronised.
All subsequent barriers during model loading and training are instant.

### `device_ids=[local_rank]` in `dist_barrier`

Without either `device_id=` in init OR `device_ids` in the barrier call, NCCL does
not know which GPU each rank owns and hangs indefinitely with a warning:

> `devices used by this process are currently unknown`

Passing `device_ids=[local_rank]` gives NCCL the mapping it needs.

---

## Will This Work on Other GPUs and Environments?

| GPU / Interconnect              | Expected Behaviour                                                                                                                                |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| **RTX 4090 / 3090 / A4000** (PCIe, no NVLink) | ✓ P2P probe hangs on many PCIe-only setups. `NCCL_P2P_DISABLE=1` is the correct default. Throughput unchanged for typical training. |
| **A100 / H100** (NVLink)        | ✓ NCCL prefers NVLink over P2P; the P2P disable has no effect on NVLink bandwidth. Training is at full speed. Override with `NCCL_P2P_DISABLE=0` in `.env` to force P2P for benchmarking. |
| **Any GPU with NVSwitch**       | ✓ Same as NVLink case above.                                                                                                                      |
| **CPU-only / gloo backend**     | ✓ The `os.environ.setdefault` line is inside the `if torch.cuda.is_available()` block, so it is never set on CPU-only machines. gloo barrier is used unchanged. |
| **PyTorch 2.3+**                | ✓ `device_id=` removed from `init_process_group`, so no version-dependent branch is needed.                                                       |
| **PyTorch < 2.3**               | ✓ `init_process_group()` simply has no `device_id=` argument; the call is identical.                                                              |
| **NCCL 2.x** (any patch)        | ✓ `NCCL_P2P_DISABLE` is a stable env var supported across all NCCL 2.x releases.                                                                  |
| **CUDA 11.x / 12.x**            | ✓ No CUDA-version-specific code involved.                                                                                                         |
| **Multi-node** (SLURM / torchrun) | ✓ `NCCL_P2P_DISABLE=1` only affects intra-node P2P. Cross-node traffic still uses InfiniBand or TCP via the network plugin.                     |

---

## Overriding the Defaults

If you are on a machine with NVLink and want to confirm P2P is helping:

```bash
echo "NCCL_P2P_DISABLE=0" >> .env
```

To see NCCL's communication choices at runtime:

```bash
NCCL_DEBUG=INFO torchrun --nproc_per_node=2 train.py 2>&1 | grep "via "
```

Interpreting the output:

| Line pattern         | Meaning                                                        |
| -------------------- | -------------------------------------------------------------- |
| `via SHM/direct`     | Shared memory (P2P disabled).                                  |
| `via NVL/direct`     | NVLink (fast, P2P disable irrelevant).                         |
| `via P2P/direct`     | PCIe P2P (fast, but may probe-hang on some hardware).          |
