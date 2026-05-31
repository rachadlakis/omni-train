# Beginner's Guide — Distributed Training Project

New to distributed training or this project? Start here.
For the full technical reference see [TECHNICAL.md](TECHNICAL.md).

---

## Table of Contents

1. [What Does This Project Do?](#1-what-does-this-project-do)
2. [Key Concepts Explained Simply](#2-key-concepts-explained-simply)
3. [Setup Step-by-Step](#3-setup-step-by-step)
4. [Environment Variables](#4-environment-variables)
5. [Understanding config.yaml](#5-understanding-configyaml)
6. [Running Your First Training](#6-running-your-first-training)
7. [Training Modes](#7-training-modes)
8. [Choosing a Strategy](#8-choosing-a-strategy)
9. [Monitoring Training](#9-monitoring-training)
10. [Common Errors & Fixes](#10-common-errors--fixes)

---

## 1. What Does This Project Do?

It **fine-tunes language models** (like GPT-style models) on text data, using one or more GPUs to do it faster and handle larger models.

Think of it like this:
- You have a pre-trained model (e.g. `facebook/opt-125m` — a model that already knows a lot about language).
- You have some text data you want the model to learn from.
- This project handles all the complexity of splitting that work across multiple GPUs.

---

## 2. Key Concepts Explained Simply

### What is fine-tuning?

A large model is pre-trained on billions of words from the internet. Fine-tuning means continuing that training on your specific dataset — so the model gets better at your particular task without starting from scratch.

### What is distributed training?

Training on multiple GPUs at the same time. Instead of one GPU doing all the work (slowly), you split the work across 2, 4, or 8 GPUs.

### DDP vs FSDP — what's the difference?

| | DDP | FSDP |
|---|-----|------|
| **Idea** | Every GPU has a full copy of the model | The model is split across GPUs |
| **Memory** | High — each GPU needs the whole model | Low — each GPU only holds a fraction |
| **When to use** | Model fits on 1 GPU | Model is too big for 1 GPU |
| **Complexity** | Simple | More complex but much more memory efficient |

### What is LoRA?

Instead of updating all the billions of weights in a model, LoRA only trains a small set of extra "adapter" weights. The original model stays frozen. This is ~10× cheaper in memory.

> Think of it as adding sticky notes to a textbook instead of rewriting the whole book.

### What is QLoRA?

QLoRA combines LoRA with **quantization** — compressing the model's base weights to 4-bit (instead of 32-bit). This lets you fine-tune a 7-billion-parameter model on a single 8 GB GPU.

### What is mixed precision?

Using lower-precision numbers (16-bit instead of 32-bit) for most operations to make training faster, while keeping 32-bit precision only where accuracy matters most.

### What is gradient checkpointing?

During training, the model temporarily stores intermediate calculations to compute gradients. Gradient checkpointing discards most of these and recomputes them when needed — trading a bit of speed for a lot of memory savings.

---

## 3. Setup Step-by-Step

### Prerequisites

- A Linux machine (WSL2, Ubuntu, or a cloud VM like RunPod)
- Python 3.10 or newer
- An NVIDIA GPU (optional for testing, required for real training)
- A free [Hugging Face](https://huggingface.co/) account and access token

### Step 1 — Get the code

```bash
git clone https://github.com/rachadlakis/dist-train-project.git
cd dist-train-project
```

### Step 2 — Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # run this every time you open a new terminal
```

### Step 3 — Install dependencies

The easiest way — the script detects your GPU and installs the right version of PyTorch automatically:

```bash
python scripts/setup_env.py
```

If you prefer to install manually:

```bash
# Check your CUDA version first
nvcc --version

# CUDA 12.8
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu128

# CUDA 12.4
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
```

### Step 4 — Set your Hugging Face token

```bash
echo "HF_TOKEN=hf_your_token_here" > .env
```

Get your token from: https://huggingface.co/settings/tokens

### Step 5 (VS Code only) — Select interpreter

Open the command palette (`Ctrl+Shift+P`) → "Python: Select Interpreter" → pick `.venv/bin/python`.

---

## 4. Environment Variables

Create a file called `.env` in the project folder:

```
HF_TOKEN=hf_your_token_here
WANDB_API_KEY=your_wandb_key      ← optional, only if you want experiment tracking
```

`HF_TOKEN` is needed to download gated models from Hugging Face (e.g. LLaMA). For open models like `facebook/opt-125m` it is still recommended.

---

## 5. Understanding config.yaml

`config.yaml` is where you control everything. You don't need to touch the Python code for most experiments.

Here are the most important settings:

```yaml
# Which model to use
model_name: "facebook/opt-125m"

# How to distribute the training
strategy: "fsdp"       # solo = 1 GPU no torchrun | ddp = multi-GPU | fsdp = multi-GPU (memory efficient)
num_gpus: 2            # how many GPUs to use

# The dataset
dataset:
  name: "wikitext"
  subset: "wikitext-2-raw-v1"
  split: "train[:1%]"  # use a small slice for quick tests

# Training settings
training:
  epochs: 3            # how many times to go through the dataset
  batch_size: 8        # samples per step per GPU (lower = less memory)
  learning_rate: 1e-5  # how fast the model learns
  max_length: 128      # max token length per sample
```

**Quick tips:**
- `batch_size: 4` or `batch_size: 2` if you get "out of memory" errors
- `split: "train[:1%]"` for a quick test run; change to `"train"` for real training
- `strategy: "solo"` if you only have 1 GPU and want to skip torchrun

---

## 6. Running Your First Training

```bash
# Make sure your env is activated
source .venv/bin/activate

# Run (reads config.yaml automatically)
bash scripts/launch.sh
```

That's it. The launcher reads `strategy` and `num_gpus` from your config and starts the right command.


Or lunch the UI with
```bash
# Make sure your env is activated
source .venv/bin/activate

# Run (reads config.yaml automatically)
bash ui/launch_ui.sh
```



### Other ways to launch

```bash
# Override strategy without editing config.yaml
STRATEGY=ddp NUM_GPUS=2 bash launch.sh

# Use a different config file
CONFIG_PATH=configs/llm_lora_ddp.yaml bash launch.sh

# Manual torchrun (2 GPUs)
CONFIG_PATH=config.yaml torchrun --nproc_per_node=2 train.py
```

---

## 7. Training Modes

### Full Fine-tuning
Train every weight in the model. Best results, but needs the most memory.

```yaml
peft:
  enabled: false
```

### LoRA
Only train small adapter layers. Much cheaper. Good for most use cases.

```yaml
peft:
  enabled: true
  type: lora
  r: 16          # adapter rank — higher = more capacity, more memory
  alpha: 32      # scaling factor (usually 2 × r)
```

### QLoRA
Compress the base model to 4-bit/8-bit and run LoRA on top. Lets you train large models on small GPUs.

```yaml
peft:
  enabled: true
  type: qlora
  r: 16
quantization:
  enabled: true
  bits: 4
```

---

## 8. Choosing a Strategy

| Situation | Recommended strategy |
|-----------|---------------------|
| Testing, 1 GPU, small model | `solo` |
| 1–2 GPUs, model fits in memory | `ddp` |
| Multiple GPUs, large model (1B+) | `fsdp` |
| Very large model, limited GPU memory | `fsdp` + `qlora` |

---

## 9. Monitoring Training

### Terminal logs

`launch.sh` prints a summary at start, then loss per step from rank 0.

### Web UI

A browser interface for a friendlier view:

```bash
bash ui/launch_ui.sh
```

Open **http://127.0.0.1:8787** — you can edit `config.yaml`, start/stop training, and watch live logs.

### Weights & Biases (optional)

Enable experiment tracking in `config.yaml`:

```yaml
wandb:
  enabled: true
  project: "my-project"
```

Then set `WANDB_API_KEY` in your `.env` file.

### Checkpoints

Saved to `checkpoints/` after training (if `save: true` in config). To resume from a checkpoint:

```yaml
save_load:
  resume: true
  resume_path: "checkpoints/dcp_api/1234567890"
```

---

## 10. Common Errors & Fixes

| What you see | What it means | Fix |
|---|---|---|
| `CUDA out of memory` | The model + batch don't fit in GPU RAM | Lower `batch_size`, lower `max_length`, or switch to `qlora` |
| `401 Unauthorized` from Hugging Face | Invalid or missing token | Check `HF_TOKEN` in `.env` |
| `Address already in use` / port conflict | Another process is using the training port | Change `MASTER_PORT` in `launch.sh` (e.g. to `29501`) |
| `ImportError: No module named 'peft'` | Missing package | `pip install peft bitsandbytes accelerate` |
| W&B login error | Bad API key | Set `WANDB_API_KEY` in `.env` or set `wandb.enabled: false` |
| `peft.type: qlora` + FSDP error | QLoRA and FSDP mixed precision conflict | Set `fsdp.mixed_precision: false` when using QLoRA |
| Training stuck / no output | Distributed processes may be hanging | Check with `nvidia-smi`; kill stale processes with `pkill -f torchrun` |
