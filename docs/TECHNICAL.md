# Technical Reference — OMNI-Train

For high-level setup and usage see the [README](../README.md). For a beginner-friendly introduction see [GUIDE.md](GUIDE.md).

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Distributed Setup](#2-distributed-setup-distributed_utilspy)
3. [DDP Path](#3-ddp-path)
4. [FSDP Path](#4-fsdp-path)
5. [Checkpointing](#5-checkpointing-checkpointpy)
6. [Data Pipeline](#6-data-pipeline-datapy)
7. [Training Loop](#7-training-loop-trainpy)
8. [Configuration Schema](#8-configuration-schema)
9. [LoRA / QLoRA](#9-lora--qlora)
10. [Learning Rate Schedules](#10-learning-rate-schedules)
11. [Gradient Accumulation](#11-gradient-accumulation)
12. [Mixed Precision](#12-mixed-precision)
13. [SLURM / Multi-Node](#13-slurm--multi-node)
14. [Web UI Internals](#14-web-ui-internals)

---

## 1. Architecture Overview

```
config.yaml
    │
    ▼
launch.sh  ──────────────►  torchrun / python
                                    │
                                    ▼
                               train.py  (main)
                          ┌────────┴────────┐
                          ▼                 ▼
              distributed_utils.py      data.py
              (DDP / FSDP setup)     (tokenize, sampler)
                          │
                    ┌─────┴─────┐
                    ▼           ▼
              checkpoint.py   utils.py
              (save/load)     (plots, formatting)
```

**Backend selection** (automatic at startup):
- `nccl` — CUDA on Linux (optimized NVIDIA multi-GPU comms)
- `gloo` — CPU fallback

---

## 2. Distributed Setup (`distributed_utils.py`)

`torchrun` injects `RANK`, `LOCAL_RANK`, and `WORLD_SIZE` as environment variables before Python starts.

```python
def setup_dist_process_group():
    rank       = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(
        backend=BACKEND,
        device_id=torch.device(f"cuda:{local_rank}")  # passed directly to NCCL
    )
    torch.cuda.set_device(f"cuda:{local_rank}")
```

Passing `device_id` to `init_process_group` avoids a separate `set_device` pre-call and ensures collective ops (barrier, gradient sync, DCP) map correctly to physical hardware ranks.

### Debug / Logging Helpers

| Function | Description |
|----------|-------------|
| `print_on_rank_0(rank, msg)` | Print only from rank 0 — prevents duplicate log spam |
| `print_banner_on_rank_0(rank, title)` | Section banner from rank 0 |
| `print_on_all_ranks(rank, msg)` | Print from every rank with `[host \| rank \| local_rank \| device]` prefix |
| `gather_rank_debug(rank, world_size, title, msg)` | `dist.all_gather_object` to collect diagnostics from all ranks, print on rank 0 |
| `gpu_memory_snapshot(device)` | Returns `memory_allocated` and `memory_reserved` in GB |

---

## 3. DDP Path

```python
def apply_ddp(local_rank, rank, device, args):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, low_cpu_mem_usage=True
    )
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    return model
```

All ranks load the full pretrained model independently. `low_cpu_mem_usage=True` reduces peak CPU RAM during load. Gradient synchronization happens automatically after each backward pass via all-reduce.

**Trade-offs:**
- Simple and well-understood
- Every GPU must hold a complete copy — memory scales poorly for large models

---

## 4. FSDP Path

### 4.1 Meta-Device Initialization

The model is instantiated on `torch.device("meta")` — the computation graph is built without allocating any real memory:

```python
config = AutoConfig.from_pretrained(args.model_name)
config.use_cache = False   # KV-cache not needed during training
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config)
```

### 4.2 Mixed Precision Policy

```python
MixedPrecisionPolicy(
    param_dtype  = DTYPE_MAP[args.param_dtype],    # e.g. bfloat16
    reduce_dtype = DTYPE_MAP[args.reduce_dtype],   # e.g. float32
    output_dtype = DTYPE_MAP[args.output_dtype],   # e.g. bfloat16
    cast_forward_inputs = args.cast_forward_inputs
)
```

- Parameters and outputs use the configured low precision.
- Reductions (all-reduce) stay in `float32` for numerical stability.

### 4.3 Layer Sharding, Gradient Checkpointing & Prefetching

`get_model_layers(model)` probes the architecture to identify transformer blocks. Each block is sharded independently:

```python
for layer in get_model_layers(model):
    fully_shard(layer, **fsdp_kwargs)
fully_shard(model, **fsdp_kwargs)   # root module
```

After sharding:
```python
model.gradient_checkpointing_enable()
```

If `--explicit-prefetching` is set, `set_modules_to_forward_prefetch` and `set_modules_to_backward_prefetch` configure per-layer lookahead. This overlaps all-gather communication with the current layer's compute.

### 4.4 Weight Loading Paths

| Path | Trigger | Behavior |
|------|---------|----------|
| **A — Resume** | `resume=true` + `resume_path` set | `Checkpointer.load_model()` + `load_optim()` |
| **B — Fresh from HF** | No checkpoint, `load_model_from_hf=true` | Rank 0 downloads model, saves `pretrained_seed/model_state_dict.pt`, all ranks load via `Checkpointer` |
| **C — Random init** | `load_model_from_hf=false` | `model.to_empty(device=device)` then `init_weights()` |

Path B: rank 0 saves the seed file, calls `dist.barrier()`, then all ranks load and shard it. This prevents every rank from independently downloading the model.

---

## 5. Checkpointing (`checkpoint.py`)

The `Checkpointer` class is initialized with a `folder` path and a `dcp_api` boolean.

| Method | Description |
|--------|-------------|
| `is_empty()` | Returns `True` if no checkpoint subfolder exists |
| `load_model(model)` | Load latest `model_state_dict.pt` |
| `load_optim(model, opt)` | Load latest `optim_state_dict.pt` and restore optimizer |
| `save(model, optim)` | Gather full state dicts and write `.pt` files |

### DCP API (`dcp_api: true`)

Uses PyTorch's native `torch.distributed.checkpoint.state_dict` helpers:

- **Load:** `set_model_state_dict(...)` — rank 0 loads the file; state is broadcast and auto-sharded to all ranks.
- **Save:** `get_model_state_dict(...)` — gathers the full unsharded state dict with CPU offloading on rank 0.

### DTensor API (`dcp_api: false`)

Manual sharding using `distribute_tensor`:

- **Load:** each parameter matched by name, then `distribute_tensor` distributes it according to the existing FSDP shard layout.
- **Save:** `sharded_param.full_tensor()` gathers the sharded DTensor back to a full tensor.
- **Optimizer:** `_init_optim_state(opt)` materializes optimizer state slots, then distributes DTensor optimizer states.

---

## 6. Data Pipeline (`data.py`)

```
Raw HF Dataset
      │
      ▼
Filter short sequences
      │
      ▼
Tokenize  (truncation, padding, labels = input_ids)
      │
      ▼
DistributedSampler(num_replicas=world_size, rank=rank, shuffle=True)
      │
      ▼
DataLoader(pin_memory=True, num_workers=N)
```

Key: `sampler.set_epoch(epoch)` must be called at the start of each epoch to reshuffle differently per epoch.

```python
class TokenizedDataset(Dataset):
    def __getitem__(self, idx):
        text = self.dataset[idx]["text"]
        enc  = self.tokenizer(text, truncation=True,
                              max_length=self.max_seq_len,
                              padding="max_length", return_tensors="pt")
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         enc["input_ids"].squeeze(0).clone(),
        }
```

---

## 7. Training Loop (`train.py`)

```
Stage 1  init_process_group()
Stage 2  load tokenizer (all ranks, same tokenizer)
Stage 3  apply_ddp() or apply_fsdp()
Stage 4  get_dataloader() → DistributedSampler
Stage 5  for epoch in range(epochs):
           sampler.set_epoch(epoch)
           for batch in dataloader:
             optimizer.zero_grad()
             loss = model(**batch).loss
             loss.backward()
             clip_grad_norm_(model.parameters(), 1.0)
             optimizer.step()
Stage 6  save_checkpoint()
Stage 7  dist.destroy_process_group()
```

Rank 0 prints an in-place progress bar. All other ranks are silent unless explicitly enabled.

---

## 8. Configuration Schema

All settings are in `config.yaml`. Full reference:

```yaml
model_name: "facebook/opt-125m"
strategy: "fsdp"         # solo | ddp | fsdp
num_gpus: 2

dataset:
  name: "wikitext"
  subset: "wikitext-2-raw-v1"
  split: "train[:1%]"

training:
  epochs: 3
  batch_size: 8
  max_length: 128
  learning_rate: 1e-5
  grad_clip: 1.0
  grad_accum_steps: 1
  warmup_steps: 100
  lr_schedule: cosine        # cosine | linear | constant
  checkpoint_dir: "checkpoints"

fsdp:
  mixed_precision: true
  param_dtype: "bfloat16"
  reduce_dtype: "float32"
  output_dtype: "bfloat16"
  cast_forward_inputs: true
  explicit_prefetching: true
  forward_prefetch: 2
  backward_prefetch: 2
  dcp_api: true              # true = DCP API  |  false = DTensor API

peft:
  enabled: false
  type: lora                 # lora | qlora
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: all-linear
  bias: none

quantization:
  enabled: false
  bits: 4                    # 4 | 8
  quant_type: nf4            # nf4 | fp4
  compute_dtype: bfloat16
  double_quant: true

save_load:
  resume: false
  resume_path: ""            # e.g. "checkpoints/dcp_api/1234567890"
  load_model_from_hf: true

wandb:
  enabled: false
  project: "dist-train-project"
  run_name: null             # auto-generated if null
```

**Enforced rules:**
- `peft.type: qlora` → `quantization.enabled` is forced `true` and `bits` must be `4`
- Quantization requires `peft.enabled: true`
- QLoRA + FSDP: the mixed-precision policy is intentionally skipped to avoid dtype conflicts on quantized base weights

---

## 9. LoRA / QLoRA

### LoRA

Freezes base model weights. Trains two small matrices $A \in \mathbb{R}^{d \times r}$ and $B \in \mathbb{R}^{r \times k}$ alongside each frozen weight $W$:

$$\text{output} = Wx + \alpha \cdot ABx$$

where $r \ll \min(d, k)$ (typical: 8–64) and $\alpha$ is a scaling factor (usually `2 * r`).

### QLoRA

4-bit NF4 quantization applied to the base model weights (frozen). LoRA adapters run in full precision (`bfloat16`). Enables 7B+ models on consumer GPUs.

```
Base model weights: 4-bit NF4 (frozen, ~4× memory reduction)
LoRA adapters:      bfloat16 (trainable)
```

---

## 10. Learning Rate Schedules

| Schedule | Formula |
|----------|---------|
| Cosine | $\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\!\left(\frac{\pi t}{T}\right)\right)$ |
| Linear | $\eta_t = \eta_{\max} - (\eta_{\max} - \eta_{\min}) \cdot \frac{t}{T}$ |
| Warmup | $\eta_t = \eta_{\max} \cdot \frac{t}{t_{\text{warmup}}}$ for $t < t_{\text{warmup}}$ |

---

## 11. Gradient Accumulation

Simulates a larger batch without extra memory. Optimizer step is deferred every `grad_accum_steps` micro-batches:

$$\text{Effective Batch} = \text{batch\_size} \times \text{grad\_accum\_steps} \times \text{world\_size}$$

---

## 12. Mixed Precision

```
Forward Pass   BF16  — fast matmuls
Backward Pass  BF16  — fast gradient computation
Grad Accum     FP32  — stable summation
Optimizer Step FP32  — stable weight update
```

`bfloat16` is preferred over `float16` for LLM training because it has the same exponent range as `float32`, making overflow/underflow less likely.

---

## 13. SLURM / Multi-Node

### Static Batch Script

```bash
sbatch scripts/slurm_train.sh configs/llm_full_finetune_fsdp.yaml

# Custom resources
sbatch --nodes=4 --gpus-per-node=8 \
    scripts/slurm_train.sh configs/my_config.yaml
```

`scripts/slurm_train.sh` uses `srun` + `torchrun` with `c10d` rendezvous:

```bash
srun --kill-on-bad-exit=1 \
    torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py --config $CONFIG_FILE
```

### Python SLURM Launcher

Generates and submits a SLURM job dynamically:

```bash
python scripts/launch_slurm.py \
    --config configs/llm_fsdp.yaml \
    --nodes 4 --gpus 8 \
    --venv /path/to/.venv \
    --nccl-debug WARN

# Dry run (print script, don't submit)
python scripts/launch_slurm.py --config configs/llm_fsdp.yaml --nodes 2 --dry-run
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | required | YAML config path |
| `--nodes` | 2 | Number of nodes |
| `--gpus` | 4 | GPUs per node |
| `--cpus` | 32 | CPUs per task |
| `--mem` | 256G | Memory per node |
| `--time` | 24:00:00 | Time limit |
| `--partition` | gpu | SLURM partition |
| `--venv PATH` | — | Virtualenv to activate |
| `--conda-env NAME` | — | Conda env to activate |
| `--nccl-debug` | WARN | `INFO` \| `WARN` \| `ERROR` |
| `--disable-ib` | — | Disable InfiniBand |
| `--dry-run` | — | Print without submitting |
| `--save-script PATH` | — | Save generated script to file |

**NCCL tuning env vars** (set in `slurm_train.sh`):

```bash
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
```

---

## 14. Web UI Internals

`ui/app.py` is a **FastAPI** server:

- `GET /` — serves the dark-themed launcher page
- `GET /config` — returns current `config.yaml` content
- `POST /config` — writes updated YAML
- `POST /start` — starts `launch.sh` as a subprocess, streams stdout/stderr via WebSocket
- `POST /stop` — kills the running training subprocess
- `GET /ws/logs` — WebSocket endpoint for live log streaming

`ui/queue.py` manages concurrent job state and GPU allocation. Static assets (HTML/CSS/JS) are in `ui/static/`.

Start:

```bash
source .venv/bin/activate
bash ui/launch_ui.sh         # default: http://127.0.0.1:8787

# Remote server
uvicorn ui.app:app --host 0.0.0.0 --port 8787
```
