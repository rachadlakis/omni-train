# Technical Explanation of the Distributed Training Project

## Overview

This project is a modularized **distributed language-model training codebase** built to compare and exercise parallel training strategies:

- **DDP**: `DistributedDataParallel`
- **FSDP**: `FullyShardedDataParallel`

It trains a Hugging Face causal language model (default: `facebook/opt-125m` or specified via `config.yaml`) on a split of the **WikiText dataset** and supports:

- Distributed process initialization
- Tokenizer and dataset preparation
- DDP and FSDP model wrapping
- Advanced FSDP configurations including mixed precision, gradient checkpointing, and explicit layer prefetching
- Flexible distributed checkpointing supporting both PyTorch DCP and DTensor APIs
- Dynamic pretrained weight loading to prevent Out-Of-Memory (OOM) errors

The codebase has been refactored into modular components (`train.py`, `checkpoint.py`, `data.py`, `fsdp_utils.py`, `utils.py`) for better organization and maintainability.

---

## 1. Project Structure

The project logic is split across the following core modules:

- **`train.py`**: The main entry point. Orchestrates the training loop, parses YAML configuration, handles process grouping, and coordinates model/data initialization.
- **`fsdp_utils.py`**: Contains all distributed computing logic, device setup, FSDP/DDP model wrappers, and logging tools.
- **`checkpoint.py`**: Manages state persistence, handling FSDP DCP and DTensor checkpoint saving and loading.
- **`data.py`**: Handles dataset downloading, tokenization, and construction of the `DistributedSampler` and `DataLoader`.
- **`utils.py`**: Provides visual utilities like training loss plotting in the terminal and formatted configuration printing.

---

## 2. Distributed Setup & Backends (`train.py` & `fsdp_utils.py`)

At the start of execution, the backend is selected automatically in `train.py`:

- **`nccl`** if CUDA is present in a Linux environment → optimized for NVIDIA multi-GPU communication.
- **`gloo`** otherwise → CPU fallback.

### `setup_dist_process_group()`

Located in `fsdp_utils.py`, this function initializes distributed execution:

1. Reads `RANK` and `LOCAL_RANK` from environment variables passed by `torchrun`.
2. Calls `dist.init_process_group(backend=BACKEND, device_id=torch.device(f"cuda:{local_rank}"))` — the `device_id` argument is passed directly to NCCL for correct device association without a separate `set_device` pre-call.
3. After process group init, binds the process to its GPU via `torch.cuda.set_device(f"cuda:{local_rank}")`.

This ensures collective operations (e.g., `barrier`, gradient synchronization, checkpointing coordination) map identically to physical hardware ranks.

---

## 3. Logging and Debug Utilities (`fsdp_utils.py`)

Several helper functions control output verbosity across a distributed run:

- **`print_on_rank_0(rank, msg, emoji="")`**: Prints a message only from rank 0, preventing duplicate log spam across all workers.
- **`print_banner_on_rank_0(rank, title)`**: Prints a section banner (`===...===`) only from rank 0, used to delimit training stages.
- **`print_on_all_ranks(rank, msg, emoji="", local_rank=None, device=None)`**: Prints from every rank with a `[host=... | rank=... | local_rank=... | device=...]` prefix so per-GPU diagnostics remain identifiable.
- **`gather_rank_debug(rank, world_size, title, message)`**: Uses `dist.all_gather_object` to collect a diagnostic string from every rank and print the full collection on rank 0 under a titled header.
- **`gpu_memory_snapshot(device)`**: Returns a formatted string with `cuda.memory_allocated` and `cuda.memory_reserved` in GB for the given device, used for memory diagnostics.

---

## 4. Data Pipeline (`data.py`)

### `get_dataloader(...)`

This pipeline processes text identically across all nodes:

1. **Load Data**: Retrieves the specified dataset and handles optional streaming limitations.
2. **Filter**: Removes very short sequences.
3. **Tokenize**: Tokenizes and structures input data for causal language modeling, notably setting `labels` equal to `input_ids`.
4. **Distributed Sampling**: Wraps the dataset in a `DistributedSampler(..., num_replicas=world_size, rank=rank, shuffle=True)` so each GPU sees a unique, interleaved shard of the data.
5. **DataLoader Construction**: Returns a `DataLoader` leveraging multiprocessing and `pin_memory=True` (for GPU environments).

---

## 5. DDP Path (`fsdp_utils.py`)

### `apply_ddp(local_rank, rank, device, args)`

This executes the simplest distributed training path:

1. **All ranks** independently call `AutoModelForCausalLM.from_pretrained(...)` to load the full pretrained model from Hugging Face. `low_cpu_mem_usage=True` is used to reduce peak CPU RAM during loading.
2. The model is moved to the rank's device (`.to(device)`).
3. Wrapped with `DDP`:
   ```python
   model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
   ```
   `device_ids` is set to `None` on CPU-only environments.

**DDP Trade-Offs**:

- **Cons**: Memory usage scales poorly because every GPU must hold the complete model.
- **Pros**: Conceptually simple, no sharding complexity, and straightforward gradient synchronization.

---

## 6. FSDP Path (`fsdp_utils.py`)

### `apply_fsdp(local_rank, rank, device, args)`

This prepares a model via **Fully Sharded Data Parallel** training.

### 6.1 Meta-Device Initialization

To prevent massive memory spikes, the model is initialized directly onto the `meta` device:

```python
config = AutoConfig.from_pretrained(args.model_name, token=HF_TOKEN)
config.use_cache = False  # disables KV-cache to save memory during training
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config, ...)
```

This creates the computational graph structure without allocating any physical RAM. `use_cache = False` is important during training because the KV-cache is only needed for inference.

### 6.2 Mixed Precision Policy

If enabled, an optimized `MixedPrecisionPolicy` is configured:

```python
MixedPrecisionPolicy(
    param_dtype=DTYPE_MAP[args.param_dtype],
    reduce_dtype=DTYPE_MAP[args.reduce_dtype],
    output_dtype=DTYPE_MAP[args.output_dtype],
    cast_forward_inputs=args.cast_forward_inputs
)
```

- Parameters and outputs operate in the configured precision (e.g. `bfloat16`).
- Reductions are typically handled in a stable precision like `float32`.
- Inputs can be automatically cast forward if needed.

### 6.3 Modular Sharding, Gradient Checkpointing & Prefetching

FSDP limits GPU memory by wrapping individual model sub-layers using `fully_shard`. The helper `get_model_layers(model)` identifies transformer blocks using architecture-agnostic probing.

For each discovered layer, `fully_shard(layer, **fsdp_kwargs)` is called, followed by `fully_shard(model, **fsdp_kwargs)` for the root module.

After sharding, gradient checkpointing is enabled:

```python
model.gradient_checkpointing_enable()
```

If `--explicit-prefetching` is set, `set_modules_to_forward_prefetch` and `set_modules_to_backward_prefetch` configure per-layer look-ahead prefetching.

### 6.4 Pretrained Weight Instantiation

The weight initialization handles **three explicit pathways**:

#### Path A: Resume from explicit checkpoint

A `Checkpointer` is instantiated pointing at the provided resume path. It calls `load_model(model)` and `checkpointer.load_optim(model, optimizer)` to restore the state.

#### Path B: Fresh run from Hugging Face

1. **Rank 0** downloads the model, saves its full `state_dict` as `model_state_dict.pt` inside a `pretrained_seed/` subfolder, then deletes the seed model and clears the CUDA cache.
2. All ranks sync via `dist.barrier()`.
3. A `Checkpointer` targeted at `pretrained_seed/` is used to load and shard the weights into the FSDP model.
4. A second `Checkpointer` is returned for future checkpoint saves.

#### Path C: Random Initialization

The model is materialized on the GPU via `model.to_empty(device=device)`, and parameters are initialized with `init_weights()`.

---

## 7. The `Checkpointer` Class (`checkpoint.py`)

The `Checkpointer` class handles both model and optimizer state persistence. It is initialized with a `folder` path and a `dcp_api` boolean.

Key methods:

- **`is_empty()`**: Returns `True` if no existing checkpoint subfolder is found.
- **`load_model(model)`**: Loads the latest `model_state_dict.pt` checkpoint.
- **`load_optim(model, opt)`**: Loads the latest `optim_state_dict.pt` and restores the optimizer state.
- **`save(model, optim)`**: Gathers full state dicts and saves them as `.pt` files.

### Option A — DCP API

Utilizes PyTorch's native `torch.distributed.checkpoint.state_dict` helpers:

- **Load**: `set_model_state_dict(...)` — rank 0 loads the `.pt` file, and the state is automatically broadcast and sharded to all ranks.
- **Save**: `get_model_state_dict(...)` — gathers the full unsharded state dict with CPU offloading on rank 0.

### Option B — DTensor API (default)

Manual sharding using `distribute_tensor`:

- **Load**: Each parameter is matched by name, then `distribute_tensor` distributes it according to the existing FSDP shard layout.
- **Save**: `sharded_param.full_tensor()` gathers the sharded DTensor back to a full tensor.
- **Optimizer load**: Uses `_init_optim_state(opt)` to materialize optimizer state slots, then distributes DTensor optimizer states.

---

## 8. Main Training Flow and The Loop Structure (`train.py`)

The `main(args)` function is organized into the following stages:

### Stage 1: Distributed Setup

`setup_dist_process_group()` initializes the process group. All ranks report their status and a `dist.barrier()` synchronizes them.

### Stage 2: Tokenizer

All ranks load the same tokenizer from Hugging Face.

### Stage 3: Model

Depending on the configuration, either `apply_ddp(...)` or `apply_fsdp(...)` is called.

### Stage 4: Data

`get_dataloader(...)` builds the distributed dataloader.

### Stage 5: Training Loop

For each epoch:

1. `dataloader.sampler.set_epoch(epoch)` shuffles data differently each epoch.
2. Standard optimization: `zero_grad()` → `model(...)` → `loss.backward()` → `clip_grad_norm_(..., 1.0)` → `optimizer.step()`.
3. Rank 0 prints an in-place progress bar.

### Stage 6: Checkpoint & Cleanup

`save_checkpoint(...)` handles post-training saving. Finally, `cleanup()` destroys the process group.

---

## 9. Visuals and Config Validation (`utils.py`)

- **`plot_losses_in_terminal(losses)`**: Utilizes the `plotext` library to draw a visual loss curve directly in the CLI.
- **`print_config(args)`**: Prints out a formatted CLI table showing all configuration options.

---

## 10. Configuration

All configuration is parsed from `config.yaml`. The `launch.sh` script sets sensible defaults and starts `torchrun`.
