# OMNI-Train

[![tests](https://github.com/rachadlakis/omni-train/actions/workflows/tests.yml/badge.svg)](https://github.com/rachadlakis/omni-train/actions/workflows/tests.yml)

**A modular PyTorch framework for distributed fine-tuning of LLMs, vision, vision-language, and embedding models.**

Train across multiple GPUs and nodes with a single config file — from a single consumer GPU to a multi-node SLURM cluster.

```bash
bash scripts/launch.sh        # reads strategy + GPU count from config.yaml
```

---

## Highlights

- **Four execution modes** — `solo` (single GPU), `ddp` (Distributed Data Parallel), `fsdp` (Fully Sharded Data Parallel), and `5d`: any mix of data, tensor (+sequence), context, pipeline and expert parallelism, set in YAML.
- **Post-training** — SFT, preference optimisation (DPO, IPO, hinge, SimPO, ORPO) and online RL with **GRPO** (verifiable rewards, DAPO / Dr. GRPO variants) and **PPO** (value head, reward functions or a reward model), chosen with an `algorithm:` block and run on solo / DDP / FSDP, with or without LoRA.
- **Parameter-efficient fine-tuning** — LoRA and QLoRA with 4-bit (NF4) / 8-bit quantization via `bitsandbytes`.
- **Memory-efficient FSDP2** — parameter, gradient, and optimizer-state sharding with meta-device initialization to avoid CPU RAM spikes.
- **Distributed checkpointing** — both DCP and DTensor APIs.
- **Multi-modality** — LLMs, seq2seq, encoders, image classification, object detection, and vision-language models.
- **SLURM multi-node** — batch script and Python launcher included.
- **Capacity planner** — `planner.py` answers “is distributed training needed?”: per-GPU memory for solo / DDP / FSDP / 5D, a recommended layout, the minimum GPU count and fixes when nothing fits.
- **Training console** — a browser UI and REST API to plan, queue and monitor pretraining, SFT and preference runs (5D included).

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

Open `config.yaml`, pick your model, dataset and training parameters, then launch:

```bash
bash scripts/launch.sh
```

You do not need to know DDP or FSDP. With `strategy: auto` (the default), `launch.sh` looks at the model, your
data and your GPUs, explains what it chose, and starts the run. An example (8 × H100):

```text
Automatic strategy: fsdp on 8 GPUs
   Your model needs about 121 GB of GPU memory to train. Each of your 8 GPUs has 80 GB.

   Recommended: split the model across your 8 GPUs (FSDP)
      19.6 GB per GPU

   Simpler or cheaper:
      - Train a small add-on instead (LoRA): Fits on 1 GPU, 17.3 GB per GPU
```

Set `strategy: solo`, `ddp`, `fsdp` or `5d` (and `num_gpus`) to choose yourself; then you are warned before the run
if it probably does not fit. `launch.sh` starts `python train.py` (solo) or `torchrun --nproc_per_node=N train.py`
(ddp/fsdp/5d).

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

strategy: auto        # or solo, ddp, fsdp, 5d
num_gpus: auto        # with auto: the most GPUs to use

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
| `auto` | Not sure (the default) | Measures the model, the data and the GPUs, and uses the simplest of the four below that fits |
| `solo` | Single GPU | No torchrun, no process group |
| `ddp` | Model fits in one GPU; scale throughput | Every GPU holds a full model replica; gradients synced via all-reduce |
| `fsdp` | Large models, limited GPU memory | Parameters, gradients, and optimizer states sharded across GPUs |
| `5d` | Very large models, long sequences, MoE, many nodes | Any mix of DP (+ZeRO-1), TP (+SP), CP, PP and EP from the `parallelism:` block |

### Automatic (`strategy: auto`)

```bash
python advisor.py --config config.yaml                        # what would it choose here, and why?
python advisor.py --config config.yaml --gpu h100 --gpus 8    # ... and on another machine?
```

Two questions decide: **does it fit in memory** (model, training method, batch, sequence length) and **is it fast
enough** (number of examples, GPU speed)? More examples only take longer; long examples need more memory.
`auto` takes the first of these that works: one GPU (unless it would take more than about 2 hours and there are
more), DDP, FSDP, the same with gradient checkpointing or a smaller batch, and only then 5D.

- It **keeps your training method**. If you ask for full fine-tuning it never switches to LoRA; it only tells you
  that LoRA would be cheaper. It may turn gradient checkpointing on or lower the batch per GPU, and says so.
- The dataset is measured with the model's tokenizer: 500 examples plus the longest ones. SFT and preference
  batches are padded to their longest example, so short data needs less than `max_length` suggests.
- When nothing fits, the run does not start and you get the changes that would work. The numbers are estimates; to
  try anyway, set the strategy yourself.
- `num_gpus: 4` means at most 4. Started with `torchrun`, all processes are used and rank 0 decides for every rank.

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

**Support levels.** OMNI-Train's focus is LLM fine-tuning and post-training (`model_type: llm`).
That path is covered by unit, multi-process and end-to-end tests. The other model types work
but are **experimental**: each has one smoke test, no post-training support, and they may lag
behind new features (export, resume, planner). `strategy: 5d` is an advanced mode for very
large models; for most fine-tuning, `fsdp` + LoRA is the recommended path.

| `model_type` | Status |
| --- | --- |
| `llm` | supported (SFT / DPO / SimPO / ORPO, LoRA/QLoRA, solo / DDP / FSDP, 5d) |
| `seq2seq`, `encoder`, `vision`, `yolo`, `vlm` | experimental |
| `custom_transformer` | demo / testing model (synthetic data) |

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

The console plans, launches and monitors runs on this machine's GPUs. It also gives you the commands for other
machines (plain `torchrun`, no scheduler) and for SLURM clusters, and submits to SLURM when a controller answers:

```bash
bash ui/launch_ui.sh          # http://127.0.0.1:8787   (REST docs: /docs)
```

- **New run**: pick a goal (pretrain, SFT, DPO / SimPO / ORPO, or GRPO / PPO reinforcement learning), a model,
  data and hyperparameters. The strategy is automatic unless you choose one (solo, DDP, FSDP or a 5D layout with a
  live rank map). The data step measures your dataset, and a plan panel says in plain words whether one GPU is
  enough, what to do if not, and what would be simpler or cheaper, each with a button to use it.
- **Runs**: a queue with live progress, loss curves and logs; logs persist in `ui/runs/` across restarts.
- **Planner**: a capacity map for any model. Rows are GPU counts, columns are training methods (full / LoRA / QLoRA)
  or sequence lengths; each square says what that setup needs (1 GPU, DDP, FSDP, 3D/5D or too big), the memory per
  GPU and, given a number of examples, the time. Click a square for the explanation and to start a run from it.
- **Checkpoints**, **System** (GPUs, which credentials are set) and **API** (copyable curl examples).

For a remote server (the UI has no authentication: put it behind a firewall or an authenticated proxy):

```bash
uvicorn ui.app:app --host 0.0.0.0 --port 8787
```

### Is distributed training needed?

```bash
python planner.py --model meta-llama/Llama-3.1-8B --gpus 8 --gpu h100 --seq 4096
python planner.py --config configs/llm_dpo_lora_single_gpu.yaml --gpu 4050
```

`advisor.py` (above) answers for your own GPUs and data; `planner.py` answers what-if questions for any model and
hardware. It prints the verdict (single GPU / distributed optional / recommended for speed / required / does not fit), the per-GPU
memory of every strategy split into weights, gradients, AdamW states, activations and logits, the recommended
layout (including a `parallelism:` block for 5D), the fewest GPUs each strategy needs and what-if fixes
(gradient checkpointing, LoRA, QLoRA, shorter sequences). The same plan is served at `POST /api/plan`.
Estimates follow this repo's precision rules (fp32 trainable weights, bf16 frozen LoRA base outside FSDP)
and match measured single-GPU peaks (RTX 4050: full, SFT, DPO, GRPO, PPO) within 15%. DDP, FSDP, 5D and
models above 0.5B parameters have not been measured, so treat those numbers as guidance.
Times are rougher than memory: one measured run (SFT + LoRA, Qwen2.5-0.5B, RTX 4050) took 0.26 s per step
where the planner said 0.15 s, so expect real runs to take up to about twice the estimate.

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
├── post_training/          # algorithm: sft / dpo / simpo / orpo
├── export.py               # Hugging Face export (+ LoRA merge)
├── configs/                # ready-made example configs
├── scripts/                # launch + SLURM utilities
├── tests/                  # unit and smoke tests
├── planner.py              # capacity planner: is distributed training needed?
├── advisor.py              # strategy: auto — GPUs, dataset lengths, the choice in plain words
├── ui/                     # training console + REST API (FastAPI)
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
| CUDA out of memory | Set `strategy: auto`, or reduce batch size, enable gradient checkpointing, or switch to FSDP/QLoRA |
| `401` / `403` from Hugging Face | Verify `HF_TOKEN` in `.env` and that you accepted the model's license |
| `ImportError: VideoReader` | Recreate `.venv` and reinstall matching torch/torchvision versions |
| NCCL timeout | Check network / InfiniBand connectivity |
| W&B auth failure | Set `WANDB_API_KEY` or disable wandb in the config |

---

## Roadmap

- [ ] `strategy: 5d`: LoRA/QLoRA, more architectures (Qwen3, Gemma, MoE checkpoints such as Mixtral),
      vocab-parallel cross-entropy, zig-zag ring attention, interleaved pipeline schedules
- [x] `strategy: 5d`, SFT and preference optimisation in the web UI
- [x] GRPO (online RL with reward functions), in the trainer and the web UI
- [x] PPO (value head, reward functions or a Hugging Face reward model)
- [ ] Reward-model training

---

## License

Free for **personal use**, **research**, and **education**. Commercial use requires a separate agreement.

See [LICENSE](LICENSE).
