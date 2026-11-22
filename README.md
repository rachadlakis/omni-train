# OMNI-Train

**A modular PyTorch framework for distributed fine-tuning of LLMs, vision, vision-language, and embedding models.**

Train across multiple GPUs and nodes with a single config file — from a single consumer GPU to a multi-node SLURM cluster.

```bash
bash scripts/launch.sh        # reads strategy + GPU count from config.yaml
```

---

## Highlights

- **Four execution modes** — `solo` (single GPU), `ddp` (Distributed Data Parallel), `fsdp` (Fully Sharded Data Parallel), and `5d`: any mix of data, tensor (+sequence), context, pipeline and expert parallelism, set in YAML.
- **Post-training** — SFT and preference optimisation (DPO, IPO, hinge, SimPO, ORPO), chosen with an `algorithm:` block and run on solo / DDP / FSDP, with or without LoRA. GRPO and PPO are next.
- **Parameter-efficient fine-tuning** — LoRA and QLoRA with 4-bit (NF4) / 8-bit quantization via `bitsandbytes`.
- **Memory-efficient FSDP2** — parameter, gradient, and optimizer-state sharding with meta-device initialization to avoid CPU RAM spikes.
- **Distributed checkpointing** — both DCP and DTensor APIs.
- **Multi-modality** — LLMs, seq2seq, encoders, image classification, object detection, and vision-language models.
- **SLURM multi-node** — batch script and Python launcher included.
- **Browser UI** — a FastAPI web app to configure and launch jobs.

---

## Table of Contents

- [Installation](#installation)
- [Hugging Face Authentication](#hugging-face-authentication)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Distributed Strategies](#distributed-strategies)
- [Training Modes](#training-modes)
- [Supported Models & Datasets](#supported-models--datasets)
- [Web UI](#web-ui)
- [SLURM / Multi-Node](#slurm--multi-node)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [License](#license)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/rachadlakis/omni-train.git
cd omni-train
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

> **Note:** Avoid upgrading `setuptools` manually — some PyTorch builds require specific versions.

### 3. Check your CUDA version

```bash
nvidia-smi    # use the "CUDA Version" shown in the top-right corner
```

### 4. Install PyTorch

Install PyTorch **before** the rest of the requirements:

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124
```

This wheel targets CUDA 12.4. CUDA is backward-compatible, so it also works on newer CUDA versions (13.x). For a different CUDA version, pick the matching wheel from [pytorch.org](https://pytorch.org/get-started/locally/) — but changing the PyTorch version may require bumping other packages in `requirements.txt` (e.g. `peft`, `transformers`, `accelerate`) to stay compatible.

### 5. Install project dependencies

```bash
pip install -r requirements.txt
```

### 6. Verify the installation

```bash
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

Expected output:

```text
2.6.0+cu124
True
NVIDIA GeForce RTX ...
```

---

## Hugging Face Authentication

Create a `.env` file in the project root:

```env
HF_TOKEN=hf_your_token_here
WANDB_API_KEY=your_key_here     # optional, for Weights & Biases logging
```

`HF_TOKEN` is required for gated models such as **LLaMA**, **Mistral**, and **Gemma**.

---

## Quick Start

Open `config.yaml`, pick your model, dataset, number of GPUs, and training parameters, then launch:

```bash
bash scripts/launch.sh
```

`launch.sh` reads `strategy` and `num_gpus` from the config and decides whether to run `python train.py` (solo) or `torchrun --nproc_per_node=N train.py` (ddp/fsdp/5d).

**Other ways to launch:**

```bash
# Use a different config file
CONFIG_PATH=configs/llm_lora_ddp.yaml bash scripts/launch.sh

# Override strategy and GPU count inline
STRATEGY=ddp NUM_GPUS=2 bash scripts/launch.sh

# Launch directly with torchrun (distributed)
torchrun --nproc_per_node=2 train.py

# Solo mode (single process, no torchrun)
python train.py
```

---

## Configuration

All settings live in `config.yaml`. A minimal example:

```yaml
model_name: "facebook/opt-125m"
model_type: llm

strategy: fsdp
num_gpus: 2

training:
  batch_size: 8
  learning_rate: 1e-5
  epochs: 3

dist_parameters:
  mixed_precision: true
  param_dtype: bfloat16

peft:
  enabled: false
```

Ready-made configs live in [`configs/`](configs/) — use them as starting points:

| Config | Use case |
| --- | --- |
| `llm_lora_ddp.yaml` | LoRA fine-tune, multi-GPU DDP |
| `llm_full_finetune_fsdp.yaml` | Full fine-tune with FSDP |
| `llm_lora_quantized_single_gpu.yaml` | QLoRA on a single GPU |
| `llm_fsdp_mini_project_style.yaml` | FSDP reference config |
| `cnn_resnet_single_gpu.yaml` | Vision CNN, single GPU |
| `embedding_bert_lora_triplet.yaml` | Embedding model with LoRA |
| `detection_yolo_single_gpu.yaml` | YOLO object detection |
| `vlm_llava_lora_single_gpu.yaml` | Vision-language model |
| `llm_5d_tp_pp_dp_8gpu.yaml` | 5D: TP=2 (+SP) × PP=2 × DP=2 + ZeRO-1, Llama-family full fine-tune, 8 GPUs |
| `llm_5d_long_context_cp.yaml` | 5D: 16K-token sequences with context parallelism (CP=4) × TP=2 |
| `custom_5d_moe_16gpu.yaml` | 5D: all five dimensions (DP, TP+SP, CP, PP, EP) on 2 × 8 GPUs, MoE |
| `custom_5d_cpu_demo.yaml` | 5D on a laptop: 4 CPU processes, TP=2 × PP=2 |
| `llm_sft_lora_single_gpu.yaml` | SFT with LoRA on chat data (Qwen2.5-0.5B-Instruct, fits 6 GB) |
| `llm_dpo_lora_single_gpu.yaml` | DPO with LoRA on UltraFeedback pairs (Qwen2.5-0.5B-Instruct, fits 6 GB) |

See [`docs/TECHNICAL.md`](docs/TECHNICAL.md) for the full configuration schema.

---

## Distributed Strategies

| Strategy | When to use | How it works |
| --- | --- | --- |
| `solo` | Single GPU | No torchrun, no process group |
| `ddp` | Model fits in one GPU; scale throughput | Every GPU holds a full model replica; gradients synced via all-reduce |
| `fsdp` | Large models, limited GPU memory | Parameters, gradients, and optimizer states sharded across GPUs |
| `5d` | Very large models, long sequences, MoE, many nodes | Any mix of DP (+ZeRO-1), TP (+SP), CP, PP and EP from the `parallelism:` block |

### 5D parallelism (`strategy: 5d`)

Pick the size of each dimension in YAML; `dp × tp × cp × pp` must equal `num_gpus`:

```yaml
strategy: 5d
num_gpus: 8
parallelism:
  tp: 2              # tensor parallel (split heads / MLP), keep within a node
  sp: true           # sequence parallel on top of TP
  cp: 1              # context parallel: ring attention over the sequence
  pp: 2              # pipeline stages
  pp_schedule: 1f1b
  micro_batches: 8   # global batch = batch_size × micro_batches × dp
  ep: 1              # expert parallel (MoE models)
  zero1: true        # shard AdamW states over DP
  dp: auto           # = num_gpus / (tp × cp × pp) = 2
```

Works with Hugging Face **Llama, Mistral and Qwen2** models (`model_type: llm`, full fine-tune,
weights loaded shard by shard, exported back to HF format after training) and with
`model_type: custom_transformer` (including Mixture-of-Experts). Every layout is tested to give the
same losses and gradients as a single process. Details: [`docs/5D_PARALLELISM.md`](docs/5D_PARALLELISM.md).

> **Constraint:** `bitsandbytes` quantization (4-bit / 8-bit) is **incompatible with FSDP**. Use `strategy: ddp` or `strategy: solo` when quantization is enabled.

---

## Training Modes

| Mode | Memory usage | Best for |
| --- | --- | --- |
| Full fine-tuning | Highest | Small models |
| LoRA | Medium | Most fine-tuning |
| QLoRA | Lowest | 7B+ models on consumer GPUs |

### Post-training (`algorithm:` block)

| `algorithm.type` | What it does | Data |
| --- | --- | --- |
| `pretrain` (default) | next-token loss on raw text | `text` |
| `sft` | supervised fine-tuning, loss on the answer tokens only | `messages`, `prompt`/`completion`, Alpaca |
| `dpo` | DPO (`dpo_loss: sigmoid`), IPO (`ipo`) or SLiC (`hinge`) | `prompt`/`chosen`/`rejected` or chosen/rejected chats |
| `simpo` / `orpo` | reference-free preference optimisation | same as DPO |

```bash
CONFIG_PATH=configs/llm_sft_lora_single_gpu.yaml bash scripts/launch.sh
CONFIG_PATH=configs/llm_dpo_lora_single_gpu.yaml bash scripts/launch.sh
```

DPO's reference log-probs are computed once before training, so no second model is kept in
memory. Details and dataset formats: [`docs/POST_TRAINING.md`](docs/POST_TRAINING.md).

---

## Supported Models & Datasets

Set `model_type` in `config.yaml` to match your architecture. Tiny picks below are smoke-tested.

### Large Language Models (`model_type: llm`)

| Tier | Example models | Auth |
| --- | --- | --- |
| Tiny | `erwanf/gpt2-mini` (11M), `openai-community/gpt2` (124M), `facebook/opt-125m` | No auth |
| Small | `openai-community/gpt2-xl` (1.5B), `Qwen/Qwen2.5-1.5B`, `meta-llama/Llama-3.2-1B` | Llama needs `HF_TOKEN` |
| Medium | `Qwen/Qwen2.5-7B`, `mistralai/Mistral-7B-v0.3`, `meta-llama/Llama-3.1-8B` | Mistral/Llama need `HF_TOKEN` |
| Large | `Qwen/Qwen2.5-14B`, `Qwen/Qwen2.5-72B`, `meta-llama/Llama-3.3-70B` | Llama needs `HF_TOKEN` |
| Very large | `meta-llama/Llama-3.1-405B`, `Qwen/Qwen2.5-72B` | Multi-node recommended |

### Vision — image classification (`model_type: vision`)

| Models | Datasets |
| --- | --- |
| `WinKawaks/vit-tiny-patch16-224` (6M, smoke-tested), `microsoft/resnet-18`, `microsoft/resnet-50`, `google/vit-base-patch16-224` | `beans` (smoke-tested), `cifar10`, `cifar100`, `food101` |

**Schema:** image column (`img`/`image`) + label column (`label`/`labels`).

### Object detection (`model_type: yolo`)

| Models | Datasets |
| --- | --- |
| `hustvl/yolos-tiny` (6M, smoke-tested), `hustvl/yolos-small`, `facebook/detr-resnet-50` | `cppe-5` (smoke-tested), or any HF detection set with an `objects` column |

**Schema:** `image` + `image_id` + `objects { bbox [x,y,w,h], category, area, id }`.

### Vision-language (`model_type: vlm`)

| Models | Datasets |
| --- | --- |
| `HuggingFaceTB/SmolVLM-256M-Instruct` (smoke-tested), `HuggingFaceTB/SmolVLM-500M-Instruct`, `llava-hf/llava-1.5-7b-hf` | `ybelkada/football-dataset` (smoke-tested), `jxie/coco_captions` |

**Schema:** image column (`image`/`img`) + caption column (`text`/`caption`/`captions`/`sentence`). Image splitting/tiling is auto-disabled to keep sequence length bounded.

### Custom

`custom_transformer` — a toy transformer built from scratch on synthetic data (no `model_name`/dataset required).

---

## Web UI

Launch the browser-based training launcher:

```bash
bash ui/launch_ui.sh          # http://127.0.0.1:8787
```

For a remote server:

```bash
uvicorn ui.app:app --host 0.0.0.0 --port 8787
```

---

## SLURM / Multi-Node

```bash
# Submit a batch job
sbatch scripts/slurm_train.sh configs/llm_full_finetune_fsdp.yaml

# Python launcher
python scripts/launch_slurm.py \
    --config configs/llm_full_finetune_fsdp.yaml \
    --nodes 4 --gpus 8

# Dry run (print the generated script without submitting)
python scripts/launch_slurm.py --config configs/llm_full_finetune_fsdp.yaml --nodes 2 --dry-run
```

---

## Testing

```bash
# Unit tests (no GPU required)
python -m pytest

# Run a single test file
python -m pytest tests/test_config_validation.py -v

# GPU smoke tests (require a GPU, ~30–60s each)
python -m pytest -m smoke

# 5D layout equivalence on CPU (multi-process, no GPU, a few minutes)
python -m pytest -m dist
```

---

## Project Structure

```text
omni-train/
├── train.py                # entry point
├── config.yaml             # active configuration
├── distributed_utils.py    # solo / ddp / fsdp model setup
├── checkpoint.py           # distributed checkpointing
├── data.py                 # dataloaders per model type
├── utils.py                # config parsing + validation
├── model.py                # custom_transformer toy model
├── parallelism/            # strategy: 5d — DP, TP+SP, CP, PP, EP, ZeRO-1
├── configs/                # ready-made example configs
├── scripts/                # launch + SLURM utilities
├── tests/                  # unit and smoke tests
├── ui/                     # FastAPI web UI
├── dist_kit/               # standalone solo/DDP/FSDP kit to copy into other projects
└── docs/                   # guides and technical reference
```

---

## Documentation

| File | Purpose |
| --- | --- |
| [`docs/GUIDE.md`](docs/GUIDE.md) | Beginner setup and usage |
| [`docs/TECHNICAL.md`](docs/TECHNICAL.md) | Distributed internals and architecture |
| [`docs/5D_PARALLELISM.md`](docs/5D_PARALLELISM.md) | `strategy: 5d`: layouts, rules, how each dimension works |

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| CUDA out of memory | Reduce batch size, enable gradient checkpointing, or switch to FSDP/QLoRA |
| `401` / `403` from Hugging Face | Verify `HF_TOKEN` in `.env` and that you accepted the model's license |
| `ImportError: VideoReader` | Recreate `.venv` and reinstall matching torch/torchvision versions |
| NCCL timeout | Check network / InfiniBand connectivity |
| W&B auth failure | Set `WANDB_API_KEY` or disable wandb in the config |

---

## Roadmap

- [ ] `strategy: 5d`: LoRA/QLoRA, more architectures (Qwen3, Gemma, MoE checkpoints such as Mixtral),
      vocab-parallel cross-entropy, zig-zag ring attention, interleaved pipeline schedules
- [ ] `strategy: 5d` in the web UI

---

## License

Free for **personal use**, **research**, and **education**. Commercial use requires a separate agreement.

See [LICENSE](LICENSE).
