# Distributed Training Project

A modular framework for training LLMs, Vision models, and embedding models across multiple GPUs using PyTorch FSDP and DDP.

> **License:** Free for personal, research, and educational use. Commercial use requires a separate agreement — see [LICENSE](LICENSE).

---

## 📚 Documentation

| Doc | Audience | Contents |
|-----|----------|----------|
| [GUIDE.md](Documentation/GUIDE.md) | Beginners | Key concepts, step-by-step setup and usage |
| [TECHNICAL.md](Documentation/TECHNICAL.md) | Developers | Architecture, distributed internals, checkpointing, config schema |

---

## 🚀 Key Features

| Feature | Description |
|---------|-------------|
| **FSDP** | Shards parameters, gradients, and optimizer state across GPUs |
| **DDP** | Full-model replication with gradient all-reduce |
| **Meta-device init** | Prevents host OOM when loading large pretrained models |
| **Mixed Precision** | Configurable `bfloat16`/`float16` params with `float32` reductions |
| **Gradient Checkpointing** | Trades recompute for memory on intermediate activations |
| **Layer Prefetching** | Overlaps communication and compute in forward and backward passes |
| **Distributed Checkpointing** | DCP and DTensor APIs for robust save/resume across all ranks |
| **LoRA / QLoRA** | Memory-efficient fine-tuning via PEFT adapters and 4-bit quantization |
| **Web UI** | Browser interface to configure, launch, and monitor training |
| **SLURM support** | Multi-node launch scripts and a Python job generator |

---

## 📂 Project Structure

```
dist-train-project/
├── train.py              # Main entry point
├── config.yaml           # All training settings
├── distributed_utils.py  # DDP / FSDP setup, mixed precision, prefetching
├── checkpoint.py         # Checkpoint save/load (DCP & DTensor APIs)
├── parallelism.py        # Parallelism helpers (TP, PP, hybrid strategies)
├── data.py               # Dataset loading, tokenization, DistributedSampler
├── utils.py              # Terminal loss plot, config formatting
├── model.py              # Optional custom model definitions
├── pytest.ini            # Pytest configuration
├── .env                  # Secrets: HF_TOKEN, WANDB_API_KEY
├── configs/              # Example YAML configs (LLM, Vision, LoRA, FSDP, etc.)
├── scripts/              # Launchers: launch.sh, SLURM scripts
├── tests/                # Unit and smoke test suite
├── ui/                   # Web UI (FastAPI + static frontend)
└── Documentation/        # GUIDE.md, TECHNICAL.md, SCRIPT.md
```

---

## ⚡ Quick Start

## ⚠️ Prerequisites

### CUDA Toolkit
Download and install from [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads).

> **Note:** The CUDA driver (via `nvidia-smi`) and the CUDA Toolkit (`nvcc`) are separate.
> PyTorch bundles its own CUDA runtime, so the toolkit is only required if you use
> `bitsandbytes`, `FlashAttention`, `DeepSpeed`, or compile custom CUDA kernels.
> For plain PyTorch training, the toolkit is optional.

**WSL users:** Install the WSL-Ubuntu variant of the toolkit — do not install the GPU driver inside WSL.

### Step 1 — Check your CUDA version

```bash
nvidia-smi        # use the "CUDA Version" shown here (top-right)
nvcc --version    # toolkit version (may differ — nvidia-smi is what matters)
```

| `nvidia-smi` CUDA | PyTorch wheel | Index URL |
|---|---|---|
| 13.2 | `torch==2.11.0` | `https://download.pytorch.org/whl/cu130` |
| 12.8 | `torch==2.10.0` | `https://download.pytorch.org/whl/cu128` |
| 12.4 | `torch==2.6.0`  | `https://download.pytorch.org/whl/cu124` |
| 12.1 | `torch==2.3.0`  | `https://download.pytorch.org/whl/cu121` |
| CPU  | `torch==2.6.0`  | `https://download.pytorch.org/whl/cpu`   |

---

### Step 2 — Clone and create a virtual environment

```bash
git clone https://github.com/rachadlakis/dist-train-project.git
cd dist-train-project

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

---

### Step 3 — Install PyTorch (pick your CUDA version)

```bash
# CUDA 13.2 - torch 2.11.0 (cu132 is not releases yet)
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130

# CUDA 12.8 — torch 2.10.0
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128

# CUDA 12.4 — torch 2.6.0
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# CUDA 12.1 — torch 2.3.0
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# CPU only — torch 2.6.0
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu

```

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

```

---

### Step 4 — Install project dependencies

```bash
pip install -r requirements.txt
```

---

### Step 5 — Set your Hugging Face token

Required for gated models (LLaMA, Mistral, etc.). The project reads `.env` automatically — no `huggingface-cli login` needed.

Create a `.env` file in the project root:

```
HF_TOKEN=hf_your_token_here
WANDB_API_KEY=your_key        # optional to log losses in wandb.ai
```

---

### Step 6 — Launch

**Command line** — reads model, strategy, and GPU count from `config.yaml`:

```bash
bash scripts/launch.sh

# Use a specific config
CONFIG_PATH=configs/llm_lora_ddp.yaml bash scripts/launch.sh

# Inline overrides
STRATEGY=ddp NUM_GPUS=2 bash scripts/launch.sh

# torchrun directly
torchrun --nproc_per_node=2 train.py
```

**Web UI** — browser-based config editor and launcher:

```bash
bash ui/launch_ui.sh
# → http://127.0.0.1:8787
```

---

## ⚙️ Configuration

All settings live in `config.yaml`. Key fields:

```yaml Example
model_name: "facebook/opt-125m"   # any HuggingFace model
model_type: llm                   # llm | seq2seq | vision | yolo | vlm | encoder
strategy: "fsdp"                  # solo | ddp | fsdp
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
  warmup_steps: 100
  grad_clip: 1.0

peft:
  enabled: false
  type: lora          # lora | qlora
  r: 16
  alpha: 32

dist_parameters:
  mixed_precision: true
  param_dtype: bfloat16
  reduce_dtype: float32

save_load:
  resume: false
  resume_path: ""
```

> **Rules:** `peft.type: qlora` requires `quantization.enabled: true`. QLoRA + FSDP skips the mixed-precision policy to avoid dtype conflicts.
>
> **Limitation:** FSDP and quantization (`bitsandbytes` 4-bit/8-bit) are not supported together. `bitsandbytes` quantizes weights into custom low-bit formats that live on a single device — FSDP cannot shard them because it needs to move parameter shards between ranks, which requires standard dtypes (`float32`, `bfloat16`, `float16`). Use QLoRA with `strategy: solo` (single GPU) or `strategy: ddp` instead.

---

## 🧠 Training Modes

| Mode | Config | Memory | Use when |
|------|--------|--------|----------|
| Full fine-tuning | `peft.enabled: false` | Highest | Small models or large/many GPUs |
| LoRA | `peft.type: lora` | ~10× less | Most fine-tuning tasks |
| QLoRA | `peft.type: qlora` | ~20× less | 7B+ on consumer GPUs |

**LoRA** adds low-rank adapter matrices $A \in \mathbb{R}^{d \times r}$, $B \in \mathbb{R}^{r \times k}$ to frozen weights:

$$\text{output} = Wx + \underbrace{ABx}_{\text{LoRA delta}}, \quad r \ll \min(d, k)$$

---

## 🌐 Distributed Strategies

**DDP** — each GPU holds a full model copy; gradients are all-reduced after each backward pass. Best for models that fit in a single GPU.

**FSDP** — parameters, gradients, and optimizer state are sharded; each GPU stores only $1/N$ of the model. Best for 7B+ models.

---

## 🖥️ Web UI

```bash
source .venv/bin/activate
bash ui/launch_ui.sh                               # localhost:8787
UI_HOST=0.0.0.0 UI_PORT=9000 bash ui/launch_ui.sh  # remote / custom port
```

Open **http://127.0.0.1:8787**. The UI lets you configure, launch, and monitor training from the browser.

---

## 🖧 SLURM / Multi-Node

```bash
# Batch script
sbatch scripts/slurm_train.sh configs/llm_full_finetune_fsdp.yaml

# Python launcher
python scripts/launch_slurm.py \
    --config configs/llm_fsdp.yaml \
    --nodes 4 --gpus 8

# Dry run
python scripts/launch_slurm.py --config configs/llm_fsdp.yaml --nodes 4 --dry-run
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | required | Path to YAML config |
| `--nodes` | 2 | Number of nodes |
| `--gpus` | 4 | GPUs per node |
| `--time` | 24:00:00 | Time limit |
| `--partition` | gpu | SLURM partition |
| `--venv PATH` | — | Activate a virtualenv |
| `--dry-run` | — | Print without submitting |

---

## 🧪 Testing

```bash
python -m pytest          # unit tests — fast, no GPU needed (~2 s)
python -m pytest -m smoke # smoke tests — real training jobs, requires GPU
```

| File | Covers |
|------|--------|
| `test_config_validation.py` | Invalid strategies, dtypes, PEFT/quant guard conditions |
| `test_config_combinations.py` | All valid strategy × PEFT × quantization × dtype combos |
| `test_helpers.py` | Helper functions (`_dtype_from_name`, `get_model_layers`, etc.) |
| `test_model.py` | Custom Transformer forward pass, output shape, no NaN |
| `test_smoke.py` | End-to-end training runs with `facebook/opt-125m` |

Smoke tests are excluded from the default run and auto-skipped on CPU-only machines. See `tests/test_smoke.py` to configure `SMOKE_GRID` and `SMOKE_GRID_FILTERS`.

---

## 🔍 Troubleshooting

| Error | Fix |
|-------|-----|
| `CUDA out of memory` | Reduce `batch_size` or `max_length`, or enable gradient checkpointing |
| `401 / 403` from HF | Verify `HF_TOKEN` in `.env` is valid |
| Port conflict on distributed init | Change `MASTER_PORT` in `scripts/launch.sh` |
| `ImportError: peft / bitsandbytes` | `pip install peft bitsandbytes accelerate` |
| W&B auth error | Set `WANDB_API_KEY` or set `wandb.enabled: false` |
| QLoRA config error | Ensure `peft.type: qlora`, `quantization.enabled: true`, `bits: 4` |
| `NCCL timeout` on multi-node | Increase `NCCL_TIMEOUT`, check network/IB connectivity |
| Checkpoint load mismatch | Use the same `dcp_api` setting for save and load |

---

## 🔧 Roadmap

### SLURM — True Multi-Node Training

Today `torchrun` spawns all processes on one node. The goal is to span them across many nodes, each with its own GPUs, over InfiniBand.

| What | Detail |
|------|--------|
| Auto-generated job scripts | `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE` derived from `config.yaml` automatically |
| Elastic rendezvous | `torchrun --nnodes=N --rdzv-backend=c10d` — nodes can join/leave without restarting the job |
| Auto-requeue on preemption | Checkpoint on signal → SLURM re-queues → training resumes from last checkpoint |

### Multi-Dimensional Parallelism *(in progress)*

Combining Data, Tensor, and Pipeline Parallelism via `DeviceMesh` — the path to training 1T+ parameter models.

| Strategy | How it works |
|----------|-------------|
| **Tensor Parallelism (TP)** | Weight matrices are split across GPUs within a node; each rank owns a row or column shard |
| **Pipeline Parallelism (PP)** | Model is sliced by layers across stages; only activations cross stage boundaries (tolerates slow inter-node links) |
| **DeviceMesh (DP × TP × PP)** | Named grid dimensions give each strategy its own process group — the same approach used by Megatron-LM |
| **4D / 6D parallelism** | Adds Context Parallel + Expert Parallel for MoE models at full scale |
