# OMNI-Train 3D Parallelism Plugin

This plugin adds 3D parallelism (Data × Tensor × Pipeline) support to OMNI-Train without modifying any existing code.

## Quick Start

```bash
# 8 GPU 3D parallelism
bash scripts/launch_hybrid.sh

# Custom config
CONFIG_PATH=configs/hybrid/llm_2d_dp_tp.yaml bash scripts/launch_hybrid.sh

# Multi-node with SLURM
sbatch scripts/slurm_hybrid.sh configs/hybrid/llm_3d_16gpu.yaml
```

## Architecture

```
hybrid/
├── __init__.py              # Package exports
├── train_hybrid.py          # Training entry point
├── hybrid_config_adapter.py # Config parsing & validation
├── hybrid_utils.py          # 3D parallelism utilities
├── hybrid_checkpoint.py     # Checkpoint handling
└── README.md                # This file

configs/hybrid/
├── llm_3d_8gpu.yaml         # 8 GPU 3D config
├── llm_3d_16gpu.yaml        # 16 GPU 3D config (2 nodes)
├── llm_2d_dp_tp.yaml        # 4 GPU DP×TP config
└── llm_2d_dp_pp.yaml        # 4 GPU DP×PP config

scripts/
├── launch_hybrid.sh         # Single-node launcher
└── slurm_hybrid.sh          # SLURM multi-node launcher
```

## Key Components

### HybridArgs
Extended configuration dataclass with parallelism settings:
```python
from hybrid import build_hybrid_args

args = build_hybrid_args("configs/hybrid/llm_3d_8gpu.yaml")
print(args.resolved_dp_size)  # 2
print(args.resolved_tp_size)  # 2
print(args.resolved_pp_size)  # 2
```

### HybridMesh
Container for the 3D device mesh:
```python
from hybrid import setup_hybrid_parallelism

mesh = setup_hybrid_parallelism(args)
print(mesh.dp_rank)  # 0-1
print(mesh.tp_rank)  # 0-1
print(mesh.pp_rank)  # 0-1
```

### Parallelism Application
Apply parallelism in order: TP → PP → DP
```python
from hybrid import (
    apply_tensor_parallelism,
    apply_pipeline_parallelism,
    apply_data_parallelism,
)

model = apply_tensor_parallelism(model, mesh, args)
model, stage = apply_pipeline_parallelism(model, mesh, args, device)
model = apply_data_parallelism(model, mesh, args)
```

## Configuration Options

### Mesh Dimensions
```yaml
parallelism:
  enabled: true
  data_parallel_size: 2   # Replicate model
  tensor_parallel_size: 2 # Split layers
  pipeline_parallel_size: 2 # Partition stages
```

### Tensor Parallelism
```yaml
parallelism:
  tensor_parallel:
    style: colwise_rowwise  # or "megatron"
    sequence_parallel: false
```

### Pipeline Parallelism
```yaml
parallelism:
  pipeline_parallel:
    schedule: 1f1b  # or "gpipe", "interleaved"
    num_microbatches: 8
```

## Compatibility

| Feature | Supported | Notes |
|---------|-----------|-------|
| LLMs (Llama, Mistral) | ✅ | Full support |
| Seq2Seq (T5, BART) | ⚠️ | PP needs encoder-decoder split |
| PEFT/LoRA | ⚠️ | Experimental with TP |
| Quantization | ❌ | Incompatible with TP |
| Gradient Checkpointing | ✅ | Recommended for large models |

## Testing

```bash
# Unit tests (no GPU)
python -m pytest tests/test_hybrid_parallelism.py -v

# Integration test (requires 4+ GPUs)
torchrun --nproc_per_node=4 hybrid/train_hybrid.py \
    --config configs/hybrid/llm_2d_dp_tp.yaml \
    --max_steps 10
```

## Troubleshooting

### "Mesh dimensions don't match GPU count"
Ensure `dp_size × tp_size × pp_size = num_gpus`

### NCCL timeout
- Check network interface: `export NCCL_SOCKET_IFNAME=eth0`
- Increase timeout: `export NCCL_TIMEOUT=1800`

### Out of memory
- Enable gradient checkpointing
- Increase PP size to reduce per-GPU layers
- Reduce batch size or sequence length

## References

- [PyTorch DeviceMesh](https://pytorch.org/docs/stable/distributed.html#device-mesh)
- [Tensor Parallelism](https://pytorch.org/docs/stable/distributed.tensor.parallel.html)
- [Pipeline Parallelism](https://pytorch.org/docs/stable/distributed.pipelining.html)
