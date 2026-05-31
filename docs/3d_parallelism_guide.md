# 3D Parallelism User Guide for OMNI-Train

## Quick Start

This guide explains how to use 3D parallelism with OMNI-Train without modifying any existing code.

---

## Prerequisites

- PyTorch 2.4+ (for stable DeviceMesh and pipelining APIs)
- Multiple GPUs (minimum 2, recommended 8+)
- NCCL backend for distributed training

---

## Installation

The 3D parallelism plugin is a drop-in addition. No changes to existing OMNI-Train code required.

```bash
# Existing OMNI-Train setup
cd dist-train-project
source .venv/bin/activate

# The hybrid/ directory contains all 3D parallelism code
# No additional installation needed
```

---

## Basic Usage

### 1. Choose Your Configuration

```bash
# 2D: Data Parallel × Tensor Parallel (4 GPUs)
CONFIG_PATH=configs/hybrid/llm_2d_dp_tp.yaml bash scripts/launch_hybrid.sh

# 2D: Data Parallel × Pipeline Parallel (4 GPUs)
CONFIG_PATH=configs/hybrid/llm_2d_dp_pp.yaml bash scripts/launch_hybrid.sh

# Full 3D: DP × TP × PP (8 GPUs)
CONFIG_PATH=configs/hybrid/llm_3d_8gpu.yaml bash scripts/launch_hybrid.sh
```

### 2. Direct Launch with torchrun

```bash
# 8 GPU 3D parallelism
torchrun --nproc_per_node=8 hybrid/train_hybrid.py \
    --config configs/hybrid/llm_3d_8gpu.yaml

# Override mesh dimensions
torchrun --nproc_per_node=8 hybrid/train_hybrid.py \
    --config configs/hybrid/llm_3d_8gpu.yaml \
    --dp_size 2 --tp_size 2 --pp_size 2
```

### 3. Multi-Node with SLURM

```bash
# Submit 2-node job (16 GPUs total)
sbatch scripts/slurm_hybrid.sh configs/hybrid/llm_3d_16gpu.yaml

# Or use the launcher script
python scripts/launch_slurm.py \
    --config configs/hybrid/llm_3d_16gpu.yaml \
    --nodes 2 --gpus 8 \
    --entry hybrid/train_hybrid.py
```

---

## Configuration Reference

### Minimal 3D Config

```yaml
model_name: meta-llama/Llama-3.2-1B
model_type: llm
strategy: hybrid

parallelism:
  enabled: true
  data_parallel_size: 2
  tensor_parallel_size: 2
  pipeline_parallel_size: 2
```

### Full Configuration Options

```yaml
parallelism:
  enabled: true

  # Mesh Dimensions
  # Product must equal total GPU count
  data_parallel_size: 2
  tensor_parallel_size: 2
  pipeline_parallel_size: 2

  # Auto-resolution (alternative to specifying all dimensions)
  # Set one or two, the rest are calculated
  # auto_resolve: true

  # Tensor Parallelism Options
  tensor_parallel:
    # Parallelization style
    # - colwise_rowwise: Standard column/row splitting
    # - megatron: Megatron-LM style with fused kernels
    style: colwise_rowwise

    # Sequence parallelism (experimental)
    # Distributes sequence dimension in addition to hidden
    sequence_parallel: false

    # Custom parallel plan (optional)
    # Maps module names to parallelization strategies
    # plan:
    #   "self_attn.q_proj": "colwise"
    #   "self_attn.k_proj": "colwise"
    #   "mlp.gate_proj": "colwise"
    #   "mlp.down_proj": "rowwise"

  # Pipeline Parallelism Options
  pipeline_parallel:
    # Schedule type
    # - gpipe: Simple, all forward then all backward
    # - 1f1b: One-forward-one-backward (better memory)
    # - interleaved: Interleaved stages (best utilization)
    schedule: 1f1b

    # Number of microbatches
    # Higher = better pipeline utilization but more memory
    num_microbatches: 4

    # Chunks for interleaved schedule
    # auto: Calculate based on num_microbatches
    chunks: auto

    # Layer assignment (optional)
    # Default: Equal split across stages
    # layer_assignment: [12, 12, 12, 12]  # 48 layers across 4 stages

  # Device Mesh Configuration
  mesh_dim_names: ["dp", "tp", "pp"]

  # Process group backend
  backend: nccl
```

---

## Parallelism Strategies Explained

### When to Use Each Strategy

| Scenario | Recommended Config |
|----------|-------------------|
| Model fits in 1 GPU, want faster training | DP only (existing FSDP) |
| Model fits in 1 GPU with TP splitting | DP × TP |
| Large model, limited interconnect | DP × PP |
| Very large model, fast interconnect | Full 3D |

### Memory vs Communication Tradeoff

```
Strategy        Memory Efficiency    Communication Cost
─────────────────────────────────────────────────────
DP only         Low (full copy)      Low (once per step)
TP only         High                 Very High (per layer)
PP only         High                 Medium (per stage boundary)
DP × TP         High                 High
DP × PP         High                 Medium
Full 3D         Highest              Varies by config
```

### Recommended Configurations by GPU Count

| GPUs | Config | Mesh (DP×TP×PP) |
|------|--------|-----------------|
| 2 | 2D DP×TP or DP×PP | (2,1,1) or (1,2,1) |
| 4 | 2D | (2,2,1) or (2,1,2) |
| 8 | Full 3D | (2,2,2) |
| 16 | Full 3D | (4,2,2) or (2,4,2) |
| 32 | Full 3D | (4,4,2) or (8,2,2) |
| 64 | Full 3D | (8,4,2) or (4,4,4) |

---

## Model Compatibility

### Fully Supported Models

These models have automatic tensor parallelism plans:

- **Llama family**: Llama-2, Llama-3, Llama-3.2
- **Mistral family**: Mistral-7B, Mixtral
- **GPT-style**: GPT-2, GPT-J, GPT-NeoX
- **BERT-style**: BERT, RoBERTa, DeBERTa

### Partially Supported Models

These models work with pipeline parallelism but may need custom TP plans:

- **Seq2Seq**: T5, BART (encoder-decoder split)
- **Vision Transformers**: ViT, DeiT
- **Multimodal**: LLaVA, CLIP

### Not Supported

- Quantized models (4-bit, 8-bit) - incompatible with tensor sharding
- Models without clear layer structure

---

## Checkpoint Management

### Checkpoint Layout

3D parallelism checkpoints are stored separately:

```
checkpoints/
└── hybrid/
    └── <timestamp>__dp{}_tp{}_pp{}/
        ├── model_state_dict.pt      # Full model (gathered from all ranks)
        ├── optim_state_dict.pt      # Optimizer state
        └── parallelism_config.json  # Mesh configuration for resume
```

### Resuming Training

```yaml
# Resume from hybrid checkpoint
save_load:
  resume: true
  resume_path: checkpoints/hybrid/1234567890__dp2_tp2_pp2
```

### Converting Checkpoints

```bash
# Convert 3D checkpoint to standard format (for inference)
python hybrid/convert_checkpoint.py \
    --input checkpoints/hybrid/1234567890__dp2_tp2_pp2 \
    --output checkpoints/converted/model.pt \
    --format huggingface
```

---

## Troubleshooting

### Common Errors

#### "Mesh dimensions don't match GPU count"

```
Error: dp_size(2) × tp_size(2) × pp_size(4) = 16, but only 8 GPUs available
```

**Solution**: Adjust dimensions so product equals GPU count.

#### "NCCL timeout during TP all-reduce"

```
RuntimeError: NCCL timeout waiting for all-reduce
```

**Solution**:
- Ensure fast interconnect (NVLink) for TP
- Reduce TP size if using slow interconnect
- Increase `NCCL_TIMEOUT` environment variable

#### "Pipeline bubble too large"

```
Warning: Pipeline efficiency is only 45%. Consider increasing microbatches.
```

**Solution**: Increase `num_microbatches` in config (recommend PP_SIZE × 4 minimum).

#### "Out of memory on pipeline stage 0"

```
CUDA out of memory on rank 0
```

**Solution**:
- Stage 0 holds embeddings which can be large
- Try `layer_assignment` to give fewer layers to stage 0
- Enable gradient checkpointing

### Debug Mode

```bash
# Enable verbose logging
HYBRID_DEBUG=1 torchrun --nproc_per_node=8 hybrid/train_hybrid.py \
    --config configs/hybrid/llm_3d_8gpu.yaml

# Print mesh and communication patterns
HYBRID_TRACE=1 torchrun ...
```

---

## Performance Tuning

### General Tips

1. **TP size ≤ 8**: Tensor parallelism needs fast interconnect. Limit to intra-node.

2. **Microbatches ≥ PP × 2**: Ensures good pipeline utilization.

3. **Batch size divisibility**: `global_batch_size` must be divisible by `dp_size × num_microbatches`.

4. **Gradient checkpointing**: Enable for large models to trade compute for memory.

### Profiling

```bash
# Profile communication overhead
python -m torch.distributed.launch --nproc_per_node=8 \
    hybrid/train_hybrid.py \
    --config configs/hybrid/llm_3d_8gpu.yaml \
    --profile \
    --profile_output profiles/3d_run.json
```

### Monitoring

```bash
# Watch GPU utilization during training
watch -n 1 nvidia-smi

# W&B integration (enable in config)
wandb:
  wandb_log_with_train: true
  wandb_project: omni-train-3d
```

---

## Examples

### Example 1: Llama-3-8B on 8 GPUs

```yaml
# configs/hybrid/llama3_8b_8gpu.yaml
model_name: meta-llama/Llama-3-8B
model_type: llm
strategy: hybrid

parallelism:
  enabled: true
  data_parallel_size: 2
  tensor_parallel_size: 2
  pipeline_parallel_size: 2

  tensor_parallel:
    style: colwise_rowwise

  pipeline_parallel:
    schedule: 1f1b
    num_microbatches: 8

training:
  batch_size: 1  # Per microbatch per DP rank
  gradient_checkpointing: true
```

### Example 2: T5-XXL on 16 GPUs (Seq2Seq)

```yaml
# configs/hybrid/t5_xxl_16gpu.yaml
model_name: google/t5-xxl
model_type: seq2seq
strategy: hybrid

parallelism:
  enabled: true
  data_parallel_size: 4
  tensor_parallel_size: 2
  pipeline_parallel_size: 2

  pipeline_parallel:
    # Encoder on stages 0-1, decoder on stages 2-3
    schedule: 1f1b
    num_microbatches: 8
```

### Example 3: Pure Tensor Parallelism (4 GPUs)

```yaml
# configs/hybrid/tp_only_4gpu.yaml
model_name: meta-llama/Llama-3.2-3B
model_type: llm
strategy: hybrid

parallelism:
  enabled: true
  data_parallel_size: 1
  tensor_parallel_size: 4
  pipeline_parallel_size: 1

  tensor_parallel:
    style: colwise_rowwise
```

---

## API Reference

See [hybrid/README.md](../hybrid/README.md) for detailed API documentation.

---

*Guide Version: 1.0*
