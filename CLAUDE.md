# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

A modular PyTorch distributed training framework called **OMNI-Train** that supports fine-tuning LLMs, vision, and embedding models across multiple GPUs using FSDP2, DDP, or single-GPU (solo) mode, with LoRA/QLoRA support, distributed checkpointing, SLURM multi-node launch, and a FastAPI browser UI.

---

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Create a `.env` file with:
```
HF_TOKEN=hf_...       # required for gated models (LLaMA, Mistral, Gemma)
WANDB_API_KEY=...     # optional
```

---

## Running Training

```bash
# Reads strategy and num_gpus from config.yaml automatically
bash scripts/launch.sh

# Use a different config file
CONFIG_PATH=configs/llm_lora_ddp.yaml bash scripts/launch.sh

# Override strategy and GPU count inline
STRATEGY=ddp NUM_GPUS=2 bash scripts/launch.sh

# Launch directly with torchrun (distributed)
torchrun --nproc_per_node=2 train.py

# Solo mode (single process, no torchrun)
python train.py
```

## Testing

```bash
# Unit tests (no GPU required) — default
python -m pytest

# Run a single test file
python -m pytest tests/test_config_validation.py -v

# Run a single test by name
python -m pytest tests/test_config_validation.py::TestFoo::test_bar -v

# GPU smoke tests (require GPU, ~30-60s each)
python -m pytest -m smoke
```

## Web UI

```bash
bash ui/launch_ui.sh       # http://127.0.0.1:8787

# For remote server
uvicorn ui.app:app --host 0.0.0.0 --port 8787
```

## SLURM / Multi-Node

```bash
sbatch scripts/slurm_train.sh configs/llm_full_finetune_fsdp.yaml

python scripts/launch_slurm.py \
    --config configs/llm_fsdp.yaml \
    --nodes 4 --gpus 8

# Dry run (print generated script, don't submit)
python scripts/launch_slurm.py --config configs/llm_fsdp.yaml --nodes 2 --dry-run
```

---

## Architecture

### Configuration Flow

```
config.yaml (or CONFIG_PATH env var)
    │
    ▼
utils.load_config() → utils.build_args()   # parses + validates YAML into Args dataclass
    │
    ▼
train.py main()
    ├── distributed_utils.apply_solo/ddp/fsdp()   # model setup
    ├── data.get_dataloader()                       # HF dataset + DistributedSampler
    └── checkpoint.Checkpointer                    # FSDP save/load
```

`scripts/launch.sh` reads `strategy` and `num_gpus` from the config and decides whether to run `python train.py` (solo) or `torchrun --nproc_per_node=N train.py` (ddp/fsdp).

`torchrun` injects `RANK`, `LOCAL_RANK`, and `WORLD_SIZE` env vars before Python starts.

### Training Strategies

| Strategy | When to use | Notes |
|----------|-------------|-------|
| `solo`   | Single GPU  | No torchrun, no process group |
| `ddp`    | Model fits in one GPU, scale throughput | Every GPU holds full model copy; gradient sync via all-reduce |
| `fsdp`   | Large models, limited GPU memory | Parameters/gradients/optimizer states sharded across GPUs |

**Hard constraint:** `bitsandbytes` quantization (4-bit/8-bit) is **incompatible with FSDP** — the sharding mechanism cannot work on quantized layers. Use `strategy: ddp` or `strategy: solo` when quantization is enabled.

### FSDP2 Meta-Device Initialization (Path 3, no PEFT/quant)

FSDP builds the model on `torch.device("meta")` first (no real memory allocated), shards layers with `fully_shard()`, then populates weights via one of three paths:
- **A — Resume:** `Checkpointer.load_model()` reads latest checkpoint
- **B — Fresh HF:** rank 0 downloads model, saves seed `.pt` to disk, barrier, all ranks load and shard
- **C — Random init:** `model.to_empty(device=device)` then `init_weights()`

Path B is why a `checkpoints/pretrained_seed/` folder appears on fresh runs; it is deleted at the end of training.

### FSDP + PEFT/Quantization (Path 2)

When `peft.enabled=true` or `quantization.enabled=true`, the model is **materialized before wrapping** (cannot use meta device). Rank 0 loads the full HF model, saves it as a seed `.pt`, all ranks load it CPU-side, PEFT adapters are attached, then `fully_shard()` is applied to the PEFT-wrapped model.

### Checkpoint Layout

```
checkpoints/
├── solo/solo_checkpoint.pt
├── ddp/ddp_checkpoint.pt
└── fsdp/
    ├── dcp_api/
    │   └── <unix_ms_timestamp>[__lora_q4]/
    │       ├── model_state_dict.pt
    │       └── optim_state_dict.pt
    └── dtensor_api/
        └── <unix_ms_timestamp>/
            ├── model_state_dict.pt
            └── optim_state_dict.pt
```

Checkpoint folders are named with Unix millisecond timestamps. PEFT/quantization runs append a tag (e.g., `__lora_q4`). `get_latest_checkpoint_folder()` picks the numerically largest timestamp prefix.

Two FSDP checkpoint APIs:
- **DCP API** (`distribute_api: dcp_api`): uses `torch.distributed.checkpoint.state_dict` — rank 0 broadcasts state, simpler
- **DTensor API** (`distribute_api: dtensor_api`): manual `distribute_tensor` / `full_tensor()` gathering

### Key Constraints Enforced in `build_args`

- `peft.type: qlora` → forces `quantization.enabled=true`; supports `bits=4` (NF4, the default) or `bits=8` (LLM.int8())
- 4-bit quantization requires `peft.enabled=true` — cannot train 4-bit weights directly (NF4 has no differentiable backward)
- 8-bit quantization requires `peft.enabled=true` — direct AdamW updates to INT8 weights overflow the quantization range immediately, causing NaN losses from step 1. Use 8-bit LoRA instead.
- Any quantization without PEFT → hard error in `build_args` (both 4-bit and 8-bit)
- `strategy: fsdp` + `quantization.enabled: true` → hard error
- `peft.enabled=true` only supports model types: `llm, seq2seq, encoder, vlm, vision`

### Model Types

| `model_type` | HF class used |
|---|---|
| `llm` | `AutoModelForCausalLM` |
| `seq2seq` | `AutoModelForSeq2SeqLM` |
| `vision` | `AutoModelForImageClassification` |
| `yolo` | `AutoModelForObjectDetection` |
| `vlm` | `AutoModelForImageTextToText` |
| `encoder` | `AutoModel` |
| `custom_transformer` | `model.Transformer` (toy model built from scratch, uses synthetic data) |

### Data Pipeline by Model Type (`data.py`)

`get_dataloader()` branches on `model_type` to decide how raw dataset rows become batches. Each type expects a specific dataset schema:

| `model_type` | Dataset columns expected | How batches are built | Train-loop call |
|---|---|---|---|
| `llm` / `seq2seq` / `encoder` | `text` | `dataset.map(tokenize)` → `input_ids`/`labels` | `model(input_ids, attention_mask, labels)` |
| `vision` | image (`img`/`image`) + `label`/`labels` | `dataset.map(process_images)` → `pixel_values` + int label | `model(pixel_values, labels)` |
| `yolo` (detection) | `image` + `objects` (`bbox` COCO `[x,y,w,h]`, `category`, optional `area`/`id`/`iscrowd`) + `image_id` | `_detection_collate` runs the image processor with COCO `annotations` → `pixel_values` + `labels` as a **list of per-image dicts** (`class_labels`, normalized `boxes`) | `model(pixel_values, labels=[{...}], pixel_mask?)` |
| `vlm` | image (`image`/`img`) + caption (`text`/`caption`/`captions`/`sentence`) | `_vlm_collate` frames each row via the processor chat template, expands `<image>` tokens, pads, and masks pad + image tokens to `-100` | `model(input_ids, attention_mask, pixel_values, pixel_attention_mask, labels)` |
| `custom_transformer` | — (synthetic) | random token tensors | next-token loss |

Detection and VLM use a `collate_fn` (not `dataset.map`) because their per-row outputs are variable-length structures that can't be default-collated into a tensor. Those paths keep the dataset raw (PIL images preserved) and run `num_workers` collation under fork. `load_dataset` is called with `token=HF_TOKEN` so gated/rate-limited datasets resolve instead of hanging on unauthenticated Hub requests.

For VLM, `train.py` loads an `AutoProcessor` (not a bare tokenizer) and disables image splitting/tiling (`do_image_splitting=False`) on Idefics3/SmolVLM-style processors so one image maps to a bounded number of tokens rather than blowing the sequence length up into the thousands.

### Layer Detection for FSDP Sharding

`get_model_layers(model)` in `distributed_utils.py` probes common HF architecture patterns (`model.decoder.layers`, `model.encoder.layers`, `transformer.h`, `model.layers`) to find transformer blocks for per-layer sharding. Unknown architectures fall back to root-only sharding.

### RoPE Buffer Fix

After FSDP checkpoint loading, non-persistent buffers (e.g., `inv_freq` in Llama's RoPE) remain on the meta device because they are excluded from `state_dict()`. `_materialize_meta_buffers()` recomputes `inv_freq` using the same formula (`1 / base^(2i/dim)`) rather than zeroing it — zeroing would silently corrupt positional encodings.

### Web UI (`ui/`)

FastAPI server (`ui/app.py`) with a REST API. The UI has its own config schema; `ui/config_adapter.py` converts it to the `config.yaml` format before launching. `ui/queue.py` manages a SQLite-backed job queue with priority scheduling. Active config is written to `ui/_active_config.yaml` before each run.

### Parallelism Module (`parallelism.py`)

**In progress / experimental.** Standalone 3D parallelism (Data × Tensor × Pipeline) using `init_device_mesh`. Mesh dimensions follow `(dp, pp, tp)`. Default mesh table covers 1–64 GPU counts. Not integrated into the main training loop yet — calls are commented out in `train.py`.

---

## Example Configs (`configs/`)

Pre-built configs cover common scenarios. Use them as starting points:

| File | Use case |
|------|----------|
| `llm_lora_ddp.yaml` | LoRA fine-tune, multi-GPU DDP |
| `llm_full_finetune_fsdp.yaml` | Full fine-tune with FSDP |
| `llm_lora_quantized_single_gpu.yaml` | QLoRA on a single GPU |
| `llm_fsdp_mini_project_style.yaml` | FSDP reference config |
| `cnn_resnet_single_gpu.yaml` | Vision CNN, single GPU |
| `embedding_bert_lora_triplet.yaml` | Embedding model with LoRA |
| `detection_yolo_single_gpu.yaml` | YOLO object detection |
| `vlm_llava_lora_single_gpu.yaml` | Vision-language model |

---

## Logging

- Rank 0 prints all progress banners and per-step loss bars
- Other ranks print only via `print_on_all_ranks()` (with `[host|rank|local_rank|device]` prefix)
- W&B logging is off by default; enable with `wandb.wandb_log_with_train: true`
- After training completes in a terminal (not UI), a loss curve is plotted inline using `plotext`
