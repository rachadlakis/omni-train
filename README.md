# Distributed Training Project

A modular framework for training language models across multiple GPUs using PyTorch FSDP and DDP.

---

## 📚 Documentation

| Doc | Audience | Contents |
|-----|----------|----------|
| [GUIDE.md](Documentation/GUIDE.md) | Beginners | What this project does, key concepts explained simply, step-by-step setup and usage |
| [TECHNICAL.md](Documentation/TECHNICAL.md) | Developers | Architecture, distributed internals, checkpointing, config schema, SLURM |

---

## 📂 Project Structure

```
dist-train-project/
├── train.py              # Main entry point
├── config.yaml           # All training settings
├── distributed_utils.py  # DDP / FSDP setup, mixed precision, prefetching
├── checkpoint.py         # Checkpoint save/load (DCP & DTensor APIs)
├── data.py               # Dataset loading, tokenization, DistributedSampler
├── utils.py              # Terminal loss plot, config formatting
├── model.py              # Optional custom model definitions
├── launch.sh             # Launcher (reads config.yaml, calls torchrun)
├── .env                  # Secrets: HF_TOKEN, WANDB_API_KEY
├── configs/              # Example YAML configs (LLM, CNN, LoRA, FSDP, etc.)
├── scripts/              # setup_env.py, SLURM launchers
├── ui/                   # Web UI (FastAPI)
└── Documentation/        # GUIDE.md, TECHNICAL.md
```

---

## ⚡ Quick Start

```bash
# 1. Clone and enter
git clone https://github.com/rachadlakis/dist-train-project.git
cd dist-train-project

# 2. Install dependencies (auto-detects your CUDA version)
python scripts/setup_env.py

# 3. Set your Hugging Face token
echo "HF_TOKEN=hf_your_token_here" > .env

# 4. Launch (reads strategy and GPU count from config.yaml)
bash launch.sh
```

---

## ⚙️ Configuration

Edit `config.yaml` before running. Key settings:

```yaml
model_name: "facebook/opt-125m"   # any HuggingFace CausalLM
strategy: "fsdp"                  # solo | ddp | fsdp
num_gpus: 2

training:
  epochs: 3
  batch_size: 8
  learning_rate: 1e-5
```

Pass a different config file:

```bash
CONFIG_PATH=configs/llm_lora_ddp.yaml bash scripts/launch.sh
```

---

## 🛠️ Launch Options

```bash
# Auto mode — reads strategy and num_gpus from config.yaml
bash scripts/launch.sh

# Inline overrides
STRATEGY=ddp NUM_GPUS=2 bash scripts/launch.sh

# torchrun directly
CONFIG_PATH=config.yaml torchrun --nproc_per_node=2 train.py
```

---

## 🖥️ Web UI

```bash
source .venv/bin/activate
bash ui/launch_ui.sh
# → http://127.0.0.1:8787
```

---

## 🔍 Troubleshooting

| Error | Fix |
|-------|-----|
| `CUDA out of memory` | Reduce `batch_size` or `max_length` in `config.yaml` |
| `401 / 403` from Hugging Face | Check `HF_TOKEN` in `.env` |
| Port conflict on distributed init | Change `MASTER_PORT` in `launch.sh` |
| `ImportError: peft / bitsandbytes` | `pip install peft bitsandbytes accelerate` |
| W&B auth error | Set `WANDB_API_KEY` or disable with `wandb.enabled: false` | (e.g. `facebook/opt-125m`) on text datasets across multiple GPUs using two parallel strategies:

- **DDP** (`DistributedDataParallel`) — each GPU holds a full model copy; gradients are synchronized after each backward pass.
- **FSDP** (`FullyShardedDataParallel`) — model parameters, gradients, and optimizer states are sharded across GPUs, drastically reducing per-GPU memory.

It implements advanced memory and performance optimizations: mixed precision, gradient checkpointing, explicit layer prefetching, and distributed checkpointing via PyTorch's DCP and DTensor APIs.

---

## 📂 Project Structure

```
dist-train-project/
├── train.py                  # Main entry point — orchestrates the training loop
├── config.yaml               # Central configuration file
├── distributed_utils.py      # Distributed setup, DDP/FSDP wrappers, memory profiling
├── checkpoint.py             # Checkpoint save/load (DCP & DTensor APIs)
├── data.py                   # Dataset downloading, tokenization, DistributedSampler
├── utils.py                  # Utilities: terminal loss plot, config formatting
├── model.py                  # Optional custom model definitions
├── launch.sh                 # Launcher script (reads config.yaml, calls torchrun)
├── .env                      # Secrets (HF_TOKEN, WANDB_API_KEY)
├── configs/                  # Example YAML configurations per use case
│   ├── cnn_*.yaml
│   ├── llm_*.yaml
│   ├── vlm_*.yaml
│   ├── detection_*.yaml
│   └── embedding_*.yaml
├── scripts/
│   ├── launch.sh             # Alternative launcher (mirrors root launch.sh)
│   ├── setup_env.py          # Auto-detect GPU/CUDA and install correct torch wheel
│   ├── launch_slurm.py       # Python SLURM job launcher
│   └── slurm_train.sh        # SLURM batch script template
├── ui/
│   ├── app.py                # FastAPI web UI server
│   ├── queue.py              # Job queue manager
│   └── static/               # Frontend HTML/CSS/JS
└── Documentation/            # Detailed reference docs
```

---

## 🚀 Key Features

| Feature | Description |
|---------|-------------|
| **FSDP** | Shards parameters, gradients, and optimizer state across GPUs |
| **DDP** | Simple full-model replication with gradient all-reduce |
| **Meta-device init** | Prevents host OOM when loading large pretrained models |
| **Mixed Precision** | Configurable `bfloat16`/`float16` params with `float32` reductions |
| **Gradient Checkpointing** | Trades recompute for memory on intermediate activations |
| **Layer Prefetching** | Overlaps communication and compute in forward and backward passes |
| **Distributed Checkpointing** | DCP and DTensor APIs for robust save/resume across all ranks |
| **LoRA / QLoRA** | Memory-efficient fine-tuning via PEFT adapters and 4-bit quantization |
| **Web UI** | Dark-themed browser interface to configure, launch, and monitor training |
| **SLURM support** | Multi-node launch scripts and a Python job generator |

---

## ⚡ Quick Start

```bash
# 1. Clone and enter the project
git clone https://github.com/rachadlakis/dist-train-project.git
cd dist-train-project

# 2. Install dependencies (auto-detects CUDA version)
python scripts/setup_env.py

# 3. Set your Hugging Face token
echo "HF_TOKEN=hf_your_token_here" > .env

# 4. Launch training (reads strategy and GPU count from config.yaml)
bash launch.sh
```

---

## 💻 Setup Instructions

### Prerequisites

- Linux (WSL2, Ubuntu, or RunPod)
- Python 3.10+
- NVIDIA GPU drivers and CUDA runtime
- Hugging Face access token (`HF_TOKEN`) for gated models

Check CUDA availability:

```bash
nvcc --version
nvidia-smi
```

---

### Option A — Auto Setup (recommended)

The setup script detects your CUDA version and installs the correct PyTorch wheel automatically:

```bash
python scripts/setup_env.py
```

| Flag | Effect |
|------|--------|
| `--dry-run` | Print install command without running |
| `--cpu` | Force CPU-only install |
| `--requirements PATH` | Use a custom requirements file |

---

### Option B — Manual Setup (Local / WSL)

```bash
cd dist-train-project

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# CUDA 12.8
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu128

# CUDA 12.4
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt

# Optional: for LoRA / QLoRA workflows
pip install peft bitsandbytes accelerate
```

Select VS Code interpreter from: `dist-train-project/.venv/bin/python`

---

### Option C — RunPod Setup

```bash
# 1. SSH in
ssh root@<runpod-host> -p <port> -i ~/.ssh/id_ed25519

# 2. Clone and enter repo
git clone https://github.com/rachadlakis/dist-train-project.git
cd dist-train-project

# 3. Create venv and install
python3 -m venv .venv
source .venv/bin/activate
python scripts/setup_env.py   # or manual install above

# 4. Set token
echo "HF_TOKEN=hf_your_token_here" > .env
```

---

### Save Changes to GitHub

```bash
git config --global user.email "your@email.com"
git config --global user.name "Your Name"
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root (or export in shell):

```env
HF_TOKEN=hf_your_token_here          # Required for gated HF models
WANDB_API_KEY=your_wandb_key          # Optional — enables W&B logging
```

Or export directly:

```bash
export HF_TOKEN=hf_your_token_here
export WANDB_API_KEY=your_wandb_key
```

---

## ⚙️ Configuration

All training settings live in `config.yaml`. Edit this file before running.

### Common Fields

```yaml
# --- Model ---
model_name: "facebook/opt-125m"       # Any HuggingFace CausalLM model
strategy: "fsdp"                      # solo | ddp | fsdp
num_gpus: 2

# --- Dataset ---
dataset:
  name: "wikitext"
  subset: "wikitext-2-raw-v1"
  split: "train[:1%]"

# --- Training ---
training:
  epochs: 3
  batch_size: 8
  max_length: 128
  learning_rate: 1e-5
  grad_clip: 1.0
  warmup_steps: 100
  lr_schedule: cosine                 # cosine | linear | constant
  checkpoint_dir: "checkpoints"

# --- Mixed Precision ---
fsdp:
  mixed_precision: true
  param_dtype: "bfloat16"
  reduce_dtype: "float32"
  output_dtype: "bfloat16"
  cast_forward_inputs: true
  explicit_prefetching: true
  forward_prefetch: 2
  backward_prefetch: 2
  dcp_api: true                       # true = DCP API, false = DTensor API

# --- LoRA / PEFT ---
peft:
  enabled: false
  type: lora                          # lora | qlora
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: all-linear

# --- Quantization (for QLoRA) ---
quantization:
  enabled: false
  bits: 4                             # 4 | 8
  quant_type: nf4
  compute_dtype: bfloat16
  double_quant: true

# --- Resume ---
save_load:
  resume: false
  resume_path: ""                     # e.g. "checkpoints/dcp_api/1234567890"

# --- Logging ---
wandb:
  enabled: false
  project: "dist-train-project"
  run_name: null                      # auto-generated if null
```

> **Rules enforced by the code:**
> - `peft.type: qlora` implies `quantization.enabled: true` and `bits: 4`
> - Quantization requires PEFT enabled
> - QLoRA with FSDP: mixed-precision policy is skipped to avoid dtype conflicts on quantized weights

### Use a Specific Config File

```bash
CONFIG_PATH=configs/llm_lora_ddp.yaml bash launch.sh
```

---

## 🛠️ Running Training

### Activate Environment

```bash
cd dist-train-project
source .venv/bin/activate
```

### Option A — Launcher Script (recommended)

`launch.sh` reads `strategy` and `num_gpus` from `config.yaml` automatically:

```bash
bash launch.sh
```

- `strategy: solo` → runs `python train.py` (single process, no torchrun)
- `strategy: ddp` or `fsdp` → runs `torchrun --nproc_per_node=<num_gpus> train.py`

Override strategy or GPU count inline:

```bash
STRATEGY=ddp NUM_GPUS=2 bash launch.sh
CONFIG_PATH=configs/llm_lora_ddp.yaml bash launch.sh
```

### Option B — torchrun Directly

```bash
# Single GPU
CONFIG_PATH=config.yaml python train.py

# Multi-GPU (e.g. 2 GPUs)
CONFIG_PATH=config.yaml torchrun \
    --nproc_per_node=2 \
    --master_addr=localhost \
    --master_port=29500 \
    train.py
```

---

## 🧠 Training Modes

### Full Fine-tuning

Trains all model parameters. Best accuracy, highest memory cost. Use for small models or when you have large GPUs.

```yaml
peft:
  enabled: false
```

### LoRA (Low-Rank Adaptation)

Trains small adapter matrices while freezing the base model. ~10× less memory than full fine-tuning.

```yaml
peft:
  enabled: true
  type: lora
  r: 16
  alpha: 32
```

**How it works:** For a weight matrix $W \in \mathbb{R}^{d \times k}$, LoRA adds two smaller matrices $A \in \mathbb{R}^{d \times r}$ and $B \in \mathbb{R}^{r \times k}$:

$$\text{output} = Wx + \underbrace{ABx}_{\text{LoRA delta}}$$

where $r \ll \min(d, k)$. Typical $r$ values: 8, 16, 32, 64.

### QLoRA (Quantized LoRA)

Combines 4-bit quantization with LoRA adapters. Run 7B models on an 8 GB GPU.

```yaml
peft:
  enabled: true
  type: qlora
  r: 16
quantization:
  enabled: true
  bits: 4
  quant_type: nf4
  compute_dtype: bfloat16
```

---

## 🌐 Distributed Training Strategies

### DDP (DistributedDataParallel)

Each GPU holds a **complete copy** of the model. Gradients are all-reduced after each backward pass.

```
GPU 0: Full Model    GPU 1: Full Model    GPU 2: Full Model
       |                    |                    |
  [Forward]           [Forward]            [Forward]
       |                    |                    |
  [Backward]          [Backward]           [Backward]
       |                    |                    |
       +---------> All-Reduce Gradients <--------+
                            |
                   [Optimizer Step]
```

**Best for:** Models that fit in a single GPU's memory. Simplest to use.

### FSDP (FullyShardedDataParallel)

Parameters are sharded across GPUs — each GPU stores only $1/N$ of the model.

```
GPU 0: Shard 0    GPU 1: Shard 1    GPU 2: Shard 2
    |                  |                  |
    +------ All-Gather Parameters --------+
    |                  |                  |
[Forward]          [Forward]          [Forward]
    |                  |                  |
    +--- Reduce-Scatter Gradients --------+
    |                  |                  |
[Update]           [Update]           [Update]
```

**Best for:** Models larger than a single GPU's memory (7B+ parameters).

---

## 🔬 AI/ML Concepts

### Mixed Precision

Uses lower precision for speed while keeping FP32 for stability:

```
Forward Pass  : BF16 (fast)
Backward Pass : BF16 (fast)
Grad Accum    : FP32 (stable)
Optimizer Step: FP32 (stable)
```

### Gradient Checkpointing

Trades recompute for memory. Intermediate activations are discarded during forward and recomputed during backward. Reduces activation memory by $O(\sqrt{L})$ for $L$ layers.

### Layer Prefetching

Overlaps FSDP all-gather communication with compute by fetching the next layer's parameters before the current layer finishes — both in forward and backward passes.

### Gradient Accumulation

Simulates a larger effective batch size without extra memory:

$$\text{Effective Batch} = \text{batch\_size} \times \text{grad\_accum\_steps} \times \text{world\_size}$$

### Learning Rate Schedules

| Schedule | Formula |
|----------|---------|
| Cosine | $\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\pi \cdot t / T))$ |
| Linear | $\eta_t = \eta_{\max} - (\eta_{\max} - \eta_{\min}) \cdot t / T$ |
| Warmup | $\eta_t = \eta_{\max} \cdot t / t_{\text{warmup}}$ for $t < t_{\text{warmup}}$ |

---

## 🔧 Technical Deep Dive

### Distributed Setup (`distributed_utils.py`)

At startup, backend is selected automatically:
- **`nccl`** — CUDA + Linux (optimized NVIDIA multi-GPU comms)
- **`gloo`** — CPU fallback

`torchrun` injects `RANK`, `LOCAL_RANK`, and `WORLD_SIZE` as environment variables. The process group is initialized with `device_id` passed directly to NCCL for correct hardware binding.

### DDP Path

All ranks independently call `AutoModelForCausalLM.from_pretrained(low_cpu_mem_usage=True)`, move the model to their device, and wrap it:

```python
model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
```

### FSDP Path

1. **Meta-device init** — the model is created on `torch.device("meta")` (no memory allocated):
   ```python
   with torch.device("meta"):
       model = AutoModelForCausalLM.from_config(config)
   ```
2. **Mixed precision policy** — `MixedPrecisionPolicy(param_dtype, reduce_dtype, output_dtype)` is configured.
3. **Layer sharding** — each transformer block is wrapped with `fully_shard(layer)`, then the root module.
4. **Weight loading** — three paths:
   - **Resume:** load from existing checkpoint folder.
   - **Fresh from HF:** rank 0 downloads weights, saves a seed checkpoint, all ranks load and shard it.
   - **Random init:** materialize on GPU with `model.to_empty(device=device)`.

### Checkpointing (`checkpoint.py`)

Two checkpoint backends selectable via `fsdp.dcp_api`:

| Backend | Method | Notes |
|---------|--------|-------|
| **DCP API** (`dcp_api: true`) | `set_model_state_dict` / `get_model_state_dict` | PyTorch native; rank 0 loads, broadcasts + shards automatically |
| **DTensor API** (`dcp_api: false`) | `distribute_tensor` / `full_tensor()` | Manual per-parameter distribution |

### Training Loop (`train.py`)

```
Stage 1: init_process_group()
Stage 2: load tokenizer
Stage 3: apply_ddp() or apply_fsdp()
Stage 4: get_dataloader() with DistributedSampler
Stage 5: for each epoch:
           sampler.set_epoch(epoch)          # reshuffle
           zero_grad → forward → loss.backward
           clip_grad_norm_(1.0)
           optimizer.step()
Stage 6: save_checkpoint() → dist.destroy_process_group()
```

### Data Pipeline (`data.py`)

1. Load dataset from Hugging Face (with optional streaming)
2. Filter very short sequences
3. Tokenize with `labels = input_ids` for causal LM loss
4. Wrap with `DistributedSampler(num_replicas=world_size, rank=rank, shuffle=True)`
5. Return `DataLoader(pin_memory=True, num_workers=N)`

---

## 🖥️ Web UI

A lightweight dark-themed browser interface to configure, launch, and monitor training in real time.

### Features

- View and live-edit `config.yaml`
- Display key training settings and environment readiness (`HF_TOKEN`, CUDA, `.env`)
- Start / stop training (calls `launch.sh` underneath)
- Stream training logs live in the browser

### Start the UI

```bash
cd dist-train-project
source .venv/bin/activate

# Default (localhost:8787)
bash ui/launch_ui.sh

# Or with uvicorn directly
uvicorn ui.app:app --reload --port 8787

# For remote server (bind all interfaces)
uvicorn ui.app:app --host 0.0.0.0 --port 8787
```

Open: **http://127.0.0.1:8787**

Optional overrides:

```bash
UI_HOST=0.0.0.0 UI_PORT=9000 bash ui/launch_ui.sh
```

---

## 🖧 SLURM / Multi-Node

### Submit a Job

```bash
# Using the SLURM batch script
sbatch scripts/slurm_train.sh configs/llm_full_finetune_fsdp.yaml

# Custom resources
sbatch --nodes=4 --gpus-per-node=8 scripts/slurm_train.sh configs/my_config.yaml
```

### Python SLURM Launcher

```bash
# Generate and submit a job
python scripts/launch_slurm.py \
    --config configs/llm_fsdp.yaml \
    --nodes 4 \
    --gpus 8

# Dry run (print script without submitting)
python scripts/launch_slurm.py \
    --config configs/llm_fsdp.yaml \
    --nodes 4 \
    --dry-run

# With venv activation on the cluster
python scripts/launch_slurm.py \
    --config configs/llm_fsdp.yaml \
    --nodes 2 --gpus 4 \
    --venv /path/to/.venv
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | required | Path to YAML config |
| `--nodes` | 2 | Number of nodes |
| `--gpus` | 4 | GPUs per node |
| `--cpus` | 32 | CPUs per task |
| `--mem` | 256G | Memory per node |
| `--time` | 24:00:00 | Time limit |
| `--partition` | gpu | SLURM partition |
| `--venv PATH` | — | Activate a virtualenv |
| `--conda-env NAME` | — | Activate a conda env |
| `--nccl-debug` | WARN | NCCL log level |
| `--dry-run` | — | Print without submitting |

---

## 🔍 Troubleshooting

| Error | Fix |
|-------|-----|
| `CUDA out of memory` | Reduce `batch_size`, `max_length`, or use a smaller model |
| `401 / 403` from HF | Verify `HF_TOKEN` is valid and exported |
| Distributed init / port conflict | Change `MASTER_PORT` in `launch.sh` or `config.yaml` |
| W&B auth error | Set `WANDB_API_KEY` or set `wandb.enabled: false` |
| `ImportError: peft / bitsandbytes` | `pip install peft bitsandbytes accelerate` |
| QLoRA config error | Ensure `peft.type: qlora`, `quantization.enabled: true`, `bits: 4` |
| `NCCL timeout` on multi-node | Increase `NCCL_TIMEOUT`, check network/IB connectivity |
| Checkpoint load mismatch | Ensure same `dcp_api` setting used for save and load |
