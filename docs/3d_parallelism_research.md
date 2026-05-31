# 3D Parallelism Research Report for OMNI-Train

## Executive Summary

This report outlines how to add 3D parallelism (Data Parallel × Tensor Parallel × Pipeline Parallel) to OMNI-Train **without modifying existing code**. The approach uses a plugin-based architecture with new standalone modules that integrate through configuration.

---

## Table of Contents

1. [Background: What is 3D Parallelism?](#1-background-what-is-3d-parallelism)
2. [Current OMNI-Train Architecture](#2-current-omni-train-architecture)
3. [Integration Strategy: No-Modification Approach](#3-integration-strategy-no-modification-approach)
4. [Implementation Plan](#4-implementation-plan)
5. [New Files to Create](#5-new-files-to-create)
6. [Configuration Schema](#6-configuration-schema)
7. [PyTorch APIs Used](#7-pytorch-apis-used)
8. [Compatibility Matrix](#8-compatibility-matrix)
9. [Testing Strategy](#9-testing-strategy)
10. [References](#10-references)

---

## 1. Background: What is 3D Parallelism?

3D parallelism combines three orthogonal parallelization strategies:

### 1.1 Data Parallelism (DP)
- **What**: Replicate model across GPUs, partition data batches
- **How**: Each GPU processes different data, gradients synchronized via all-reduce
- **Memory**: Full model copy per GPU
- **Communication**: All-reduce gradients (O(params) per step)

### 1.2 Tensor Parallelism (TP)
- **What**: Split individual layers/tensors across GPUs
- **How**: Matrix multiplications distributed (column-wise or row-wise splits)
- **Memory**: 1/TP_SIZE of each layer per GPU
- **Communication**: All-reduce activations within each layer (high frequency)
- **Best for**: Intra-node (requires fast NVLink interconnect)

### 1.3 Pipeline Parallelism (PP)
- **What**: Partition model layers into stages across GPUs
- **How**: Each GPU holds consecutive layers, activations passed between stages
- **Memory**: 1/PP_SIZE of total layers per GPU
- **Communication**: Point-to-point activation transfer (lower bandwidth than TP)
- **Best for**: Inter-node (tolerates slower interconnect)

### 1.4 The 3D Mesh

```
                    ┌─────────────────────────────────────┐
                    │         3D Parallelism Mesh         │
                    │                                     │
       PP=0         │    TP=0        TP=1                 │
      ┌─────────────┼─────────────┬─────────────┐         │
      │   Stage 0   │  GPU 0      │  GPU 1      │  DP=0   │
      │  (Layers    │  (Col A)    │  (Col B)    │         │
      │   0-11)     ├─────────────┼─────────────┤         │
      │             │  GPU 2      │  GPU 3      │  DP=1   │
      │             │  (Col A)    │  (Col B)    │         │
      ├─────────────┼─────────────┼─────────────┤         │
       PP=1         │  GPU 4      │  GPU 5      │  DP=0   │
      │   Stage 1   │  (Col A)    │  (Col B)    │         │
      │  (Layers    ├─────────────┼─────────────┤         │
      │   12-23)    │  GPU 6      │  GPU 7      │  DP=1   │
      │             │  (Col A)    │  (Col B)    │         │
      └─────────────┴─────────────┴─────────────┴─────────┘

      Total: 8 GPUs = 2 (PP) × 2 (TP) × 2 (DP)
```

---

## 2. Current OMNI-Train Architecture

### 2.1 Existing Strategy Flow

```
config.yaml
    │
    ▼
utils.build_args() ──► Args dataclass
    │
    ▼
train.py main()
    │
    ├── strategy == "solo"  ──► apply_solo()
    ├── strategy == "ddp"   ──► apply_ddp()
    └── strategy == "fsdp"  ──► apply_fsdp()
```

### 2.2 Existing Parallelism Module

OMNI-Train already has `parallelism.py` with:
- `ParallelismArgs` dataclass
- `resolve_device_mesh()` - auto-resolves (dp, tp, pp) dimensions
- `setup_device_mesh()` - creates PyTorch `DeviceMesh`
- Default mesh table for 1-64 GPUs

**Status**: Module exists but is **not integrated** into training loop.

### 2.3 Extension Points

| Location | Extension Point | Purpose |
|----------|-----------------|---------|
| `train.py:124-196` | Strategy dispatcher | Add new strategy branch |
| `distributed_utils.py:1159` | `apply_fsdp()` | Model wrapping patterns |
| `utils.py:311` | Strategy validation | Allow new strategy values |
| `config.yaml` | YAML schema | Add parallelism config |

---

## 3. Integration Strategy: No-Modification Approach

### 3.1 Core Principle

Create **new files only** that:
1. Import from existing modules
2. Wrap existing functionality
3. Are invoked via a **new launcher script**

### 3.2 Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    NEW FILES (Plugin Layer)              │
│                                                          │
│  ┌─────────────────┐  ┌─────────────────┐               │
│  │ train_hybrid.py │  │ hybrid_utils.py │               │
│  │ (new entry pt)  │  │ (3D wrapping)   │               │
│  └────────┬────────┘  └────────┬────────┘               │
│           │                    │                         │
│           ▼                    ▼                         │
│  ┌─────────────────────────────────────────┐            │
│  │         hybrid_config_adapter.py        │            │
│  │    (extends config with 3D settings)    │            │
│  └─────────────────────────────────────────┘            │
└──────────────────────────────────────────────────────────┘
                           │
                           │ imports
                           ▼
┌──────────────────────────────────────────────────────────┐
│              EXISTING FILES (Unchanged)                  │
│                                                          │
│  train.py  │  distributed_utils.py  │  parallelism.py   │
│  utils.py  │  checkpoint.py         │  data.py          │
└──────────────────────────────────────────────────────────┘
```

### 3.3 Why This Works

1. **Python's import system**: New modules can import and extend existing ones
2. **Configuration override**: New config files override defaults
3. **Separate entry point**: `train_hybrid.py` replaces `train.py` for 3D runs
4. **Existing code stability**: Original files remain untouched and tested

---

## 4. Implementation Plan

### Phase 1: Configuration Layer
- Create `configs/hybrid/` directory with 3D parallelism configs
- Create `hybrid_config_adapter.py` to extend `Args` dataclass

### Phase 2: Model Wrapping Layer
- Create `hybrid_utils.py` with TP/PP/DP application functions
- Implement layer partitioning for pipeline stages
- Implement tensor parallelism for attention/MLP layers

### Phase 3: Training Loop
- Create `train_hybrid.py` as new entry point
- Import existing training utilities
- Add 3D-specific forward/backward logic

### Phase 4: Launch Scripts
- Create `scripts/launch_hybrid.sh` for multi-node 3D training
- Create `scripts/slurm_hybrid.sh` for SLURM clusters

### Phase 5: Checkpointing Extension
- Create `hybrid_checkpoint.py` extending `Checkpointer`
- Handle TP/PP state dict transformations

---

## 5. New Files to Create

```
dist-train-project/
├── hybrid/                          # New plugin directory
│   ├── __init__.py
│   ├── train_hybrid.py              # New entry point
│   ├── hybrid_utils.py              # 3D parallelism utilities
│   ├── hybrid_checkpoint.py         # Extended checkpointer
│   ├── tensor_parallel.py           # TP implementation
│   ├── pipeline_parallel.py         # PP implementation
│   └── hybrid_config_adapter.py     # Config extensions
│
├── configs/hybrid/                  # 3D parallelism configs
│   ├── llm_3d_8gpu.yaml
│   ├── llm_3d_16gpu.yaml
│   ├── llm_2d_dp_tp.yaml
│   └── llm_2d_dp_pp.yaml
│
├── scripts/
│   ├── launch_hybrid.sh             # 3D launcher
│   └── slurm_hybrid.sh              # SLURM 3D script
│
├── tests/
│   └── test_hybrid_parallelism.py   # 3D-specific tests
│
└── docs/
    ├── 3d_parallelism_research.md   # This document
    └── 3d_parallelism_guide.md      # User guide
```

---

## 6. Configuration Schema

### 6.1 New Config Structure

```yaml
# configs/hybrid/llm_3d_8gpu.yaml

# Inherit base model config
model_name: meta-llama/Llama-3.2-1B
model_type: llm

# Dataset (unchanged)
dataset:
  name: Salesforce/wikitext
  subset: wikitext-2-raw-v1
  split: train

# Training (unchanged)
training:
  epochs: 3
  batch_size: 4
  max_length: 512
  learning_rate: 2.0e-5

# NEW: 3D Parallelism Configuration
strategy: hybrid  # New strategy type

parallelism:
  enabled: true

  # Mesh dimensions (product must equal num_gpus)
  data_parallel_size: 2      # DP replicas
  tensor_parallel_size: 2    # TP shards per layer
  pipeline_parallel_size: 2  # PP stages

  # Tensor Parallelism settings
  tensor_parallel:
    style: colwise_rowwise   # or "megatron"
    sequence_parallel: false # experimental

  # Pipeline Parallelism settings
  pipeline_parallel:
    schedule: 1f1b           # or "gpipe", "interleaved"
    num_microbatches: 4      # microbatch count
    chunks: auto             # auto-calculate chunks

  # Device mesh
  mesh_dim_names: ["dp", "tp", "pp"]

# Checkpoint compatibility
checkpoint_dir: checkpoints/hybrid
save: true

# FSDP settings (applied within DP dimension)
dist_parameters:
  mixed_precision: true
  param_dtype: bfloat16
```

### 6.2 Mesh Resolution Logic

```python
# Auto-resolution when dimensions not specified
def resolve_mesh(num_gpus, dp=None, tp=None, pp=None):
    """
    Priority: TP > PP > DP (TP needs fast interconnect)

    Examples:
    - 8 GPUs, unspecified: (dp=2, tp=2, pp=2)
    - 8 GPUs, tp=4: (dp=2, tp=4, pp=1)
    - 16 GPUs, pp=4: (dp=2, tp=2, pp=4)
    """
```

---

## 7. PyTorch APIs Used

### 7.1 Device Mesh (PyTorch 2.0+)

```python
from torch.distributed.device_mesh import init_device_mesh, DeviceMesh

# Create 3D mesh
mesh = init_device_mesh(
    "cuda",
    mesh_shape=(dp_size, tp_size, pp_size),
    mesh_dim_names=("dp", "tp", "pp")
)

# Get submeshes for each dimension
dp_mesh = mesh["dp"]
tp_mesh = mesh["tp"]
pp_mesh = mesh["pp"]
```

### 7.2 Tensor Parallelism (torch.distributed.tensor.parallel)

```python
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

# Parallelize attention layers
parallelize_module(
    model.layers[i].self_attn,
    tp_mesh,
    {
        "q_proj": ColwiseParallel(),
        "k_proj": ColwiseParallel(),
        "v_proj": ColwiseParallel(),
        "o_proj": RowwiseParallel(),
    }
)

# Parallelize MLP layers
parallelize_module(
    model.layers[i].mlp,
    tp_mesh,
    {
        "gate_proj": ColwiseParallel(),
        "up_proj": ColwiseParallel(),
        "down_proj": RowwiseParallel(),
    }
)
```

### 7.3 Pipeline Parallelism (torch.distributed.pipelining)

```python
from torch.distributed.pipelining import (
    pipeline,
    PipelineStage,
    ScheduleGPipe,
    Schedule1F1B,
    ScheduleInterleaved1F1B,
)

# Split model into stages
def partition_model(model, pp_size, pp_rank):
    layers = get_model_layers(model)
    layers_per_stage = len(layers) // pp_size

    start = pp_rank * layers_per_stage
    end = start + layers_per_stage

    # Return only this rank's layers
    return layers[start:end]

# Create pipeline stage
stage = PipelineStage(
    submodule=partition_model(model, pp_size, pp_rank),
    stage_index=pp_rank,
    num_stages=pp_size,
    device=device,
)

# Create schedule
schedule = Schedule1F1B(
    stage=stage,
    n_microbatches=num_microbatches,
)
```

### 7.4 FSDP2 with Device Mesh

```python
from torch.distributed.fsdp import fully_shard

# Apply FSDP on DP dimension only
for layer in model.layers:
    fully_shard(layer, mesh=dp_mesh)
fully_shard(model, mesh=dp_mesh)
```

---

## 8. Compatibility Matrix

### 8.1 Strategy Combinations

| DP | TP | PP | Supported | Notes |
|----|----|----|-----------|-------|
| ✓  | ✗  | ✗  | ✓ (existing) | Current FSDP/DDP |
| ✓  | ✓  | ✗  | ✓ | 2D: DP × TP |
| ✓  | ✗  | ✓  | ✓ | 2D: DP × PP |
| ✗  | ✓  | ✓  | ✓ | 2D: TP × PP |
| ✓  | ✓  | ✓  | ✓ | Full 3D |

### 8.2 Feature Compatibility

| Feature | With 3D Parallelism | Notes |
|---------|---------------------|-------|
| PEFT/LoRA | ⚠️ Limited | TP requires adapter redistribution |
| Quantization | ❌ No | Incompatible with tensor sharding |
| Gradient Checkpointing | ✓ Yes | Per-stage activation checkpointing |
| Mixed Precision | ✓ Yes | Applied per-mesh dimension |
| Flash Attention | ✓ Yes | Works with TP |

### 8.3 Model Compatibility

| Model Type | TP Support | PP Support | Notes |
|------------|------------|------------|-------|
| LLM (Llama, Mistral) | ✓ Full | ✓ Full | Standard transformer |
| Seq2Seq (T5, BART) | ✓ Full | ⚠️ Partial | Encoder-decoder split |
| Vision (ViT, ResNet) | ⚠️ Partial | ✓ Full | ViT has attention |
| VLM (LLaVA) | ⚠️ Partial | ⚠️ Partial | Multi-tower complexity |
| Encoder (BERT) | ✓ Full | ✓ Full | Standard transformer |

---

## 9. Testing Strategy

### 9.1 Unit Tests (No GPU)

```python
# tests/test_hybrid_parallelism.py

def test_mesh_resolution():
    """Test auto-resolution of mesh dimensions"""
    result = resolve_mesh(8, dp=2)
    assert result == (2, 2, 2)

def test_config_validation():
    """Test parallelism config validation"""
    config = load_config("configs/hybrid/llm_3d_8gpu.yaml")
    args = build_hybrid_args(config)
    assert args.parallelism.enabled
    assert args.parallelism.dp_size * args.parallelism.tp_size * args.parallelism.pp_size == 8
```

### 9.2 Integration Tests (Multi-GPU)

```python
@pytest.mark.smoke
@pytest.mark.parametrize("mesh", [
    (2, 1, 1),  # Pure DP
    (1, 2, 1),  # Pure TP
    (1, 1, 2),  # Pure PP
    (2, 2, 1),  # 2D: DP × TP
])
def test_hybrid_training(mesh):
    """Test training with various mesh configurations"""
    # Requires 2+ GPUs
    pass
```

### 9.3 Smoke Tests

```bash
# Run with 4 GPUs
torchrun --nproc_per_node=4 hybrid/train_hybrid.py \
    --config configs/hybrid/llm_2d_dp_tp.yaml \
    --max_steps 10
```

---

## 10. References

### PyTorch Documentation
- [DeviceMesh Tutorial](https://pytorch.org/tutorials/recipes/distributed_device_mesh.html)
- [Tensor Parallelism](https://pytorch.org/docs/stable/distributed.tensor.parallel.html)
- [Pipeline Parallelism](https://pytorch.org/docs/stable/distributed.pipelining.html)
- [FSDP2](https://pytorch.org/docs/stable/fsdp.html)

### Research Papers
- Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism
- GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism
- PipeDream: Generalized Pipeline Parallelism for DNN Training

### Example Implementations
- [PyTorch Distributed Examples](https://github.com/pytorch/examples/tree/main/distributed)
- [torchtitan](https://github.com/pytorch/torchtitan) - PyTorch native large-scale training
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) - NVIDIA's implementation

---

## Appendix A: Memory Estimation

### Formula
```
Memory per GPU = (Model Params / (TP × PP)) × (1 + DP_overhead) + Activations + Optimizer
```

### Example: Llama-3-70B on 64 GPUs
```
Config: DP=8, TP=4, PP=2
Model: 70B params × 2 bytes (bf16) = 140GB

Per GPU:
- Model: 140GB / (4 × 2) = 17.5GB
- Optimizer (AdamW): 17.5GB × 2 = 35GB
- Activations: ~5GB (with checkpointing)
- Total: ~57.5GB per GPU (fits in 80GB A100)
```

---

## Appendix B: Communication Patterns

### Tensor Parallelism (High Frequency)
```
Forward:  AllReduce after each RowwiseParallel layer
Backward: AllReduce after each ColwiseParallel layer
Volume:   hidden_size × batch_size × seq_len per layer
```

### Pipeline Parallelism (Lower Frequency)
```
Forward:  Point-to-point send/recv between stages
Backward: Point-to-point send/recv (reverse direction)
Volume:   hidden_size × microbatch_size × seq_len per boundary
```

### Data Parallelism (Once per Step)
```
AllReduce gradients after backward pass
Volume:   total_params / (TP × PP)
```

---

*Document Version: 1.0*
*Last Updated: 2024*
