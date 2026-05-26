# Distributed Training Project

A modular framework for training LLMs, vision models, and embedding models across multiple GPUs using PyTorch Distributed.

Supports:

* **Solo** (single-GPU, no process group)
* **DDP** (Distributed Data Parallel)
* **FSDP** (Fully Sharded Data Parallel)
* **LoRA / QLoRA**
* **4-bit / 8-bit Quantization** (bitsandbytes)
* **Mixed Precision**
* **Distributed Checkpointing**
* **SLURM multi-node training**
* **Browser-based Web UI**

---

## Features

| Feature                   | Description                                 |
| ------------------------- | ------------------------------------------- |
| FSDP                      | Parameter, gradient, and optimizer sharding |
| DDP                       | Multi-GPU synchronous training              |
| Mixed Precision           | `bfloat16` / `float16` training             |
| Gradient Checkpointing    | Reduced activation memory                   |
| LoRA / QLoRA              | Efficient fine-tuning                       |
| Meta-device Loading       | Prevents CPU RAM spikes                     |
| Distributed Checkpointing | DCP + DTensor APIs                          |
| Explicit Prefetching      | Communication/compute overlap               |
| SLURM Support             | Multi-node launch utilities                 |
| Web UI                    | Browser-based training launcher             |

---

# Documentation

| File                         | Purpose                                |
| ---------------------------- | -------------------------------------- |
| `Documentation/GUIDE.md`     | Beginner setup and usage               |
| `Documentation/TECHNICAL.md` | Distributed internals and architecture |
| `Documentation/SCRIPTS.md`   | Launch scripts and SLURM utilities     |

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/rachadlakis/dist-train-project.git
cd dist-train-project
```

---

## 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
```

> Avoid upgrading `setuptools` manually. Some PyTorch builds require specific versions.

---

## 3. Check your CUDA version

```bash
nvidia-smi
```

Use the CUDA version shown in the top-right corner.

Example:

```text
CUDA Version: 13.2
```

---

## 4. Install PyTorch and packages

Choose the wheel matching your CUDA version.


### PyTorch Installation

Install PyTorch before the rest of the requirements:

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

This wheel targets CUDA 12.4. CUDA is backward-compatible, so it will also work on newer CUDA versions (13.x, etc.). If you need a different CUDA version, pick the appropriate wheel from [pytorch.org](https://pytorch.org/get-started/locally/) — but note that changing the CUDA/PyTorch version may require updating several other package versions in `requirements.txt` (e.g. `peft`, `transformers`, `accelerate`) to maintain compatibility.

---

## 5. Verify installation

```bash
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

Expected:

```text
True
NVIDIA GeForce RTX ...
```

---

## 6. Install project dependencies

```bash
pip install -r requirements.txt
```

---

# Hugging Face Authentication

Create a `.env` file:

```env
HF_TOKEN=hf_your_token_here
WANDB_API_KEY=your_key_here
```

`HF_TOKEN` is required for gated models such as:

* LLaMA
* Mistral
* Gemma

---

# Quick Start

## Launch training

```bash
bash scripts/launch.sh
```

---

## Use a custom config

```bash
CONFIG_PATH=configs/llm_lora_ddp.yaml bash scripts/launch.sh
```

---

## Override settings inline

```bash
STRATEGY=ddp NUM_GPUS=2 bash scripts/launch.sh
```

---

## Launch directly with torchrun

```bash
torchrun --nproc_per_node=2 train.py
```

---

# Web UI

Launch the browser UI:

```bash
bash ui/launch_ui.sh
```

Open:

```text
http://127.0.0.1:8787
```

---

# Configuration

Main settings live in `config.yaml`.

Example:

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

See `Documentation/TECHNICAL.md` for the full schema.

---

# Training Modes

| Mode             | Memory Usage | Best For                    |
| ---------------- | ------------ | --------------------------- |
| Full Fine-tuning | Highest      | Small models                |
| LoRA             | Medium       | Most fine-tuning            |
| QLoRA            | Lowest       | 7B+ models on consumer GPUs |

---

# Distributed Strategies

## DDP

Each GPU stores a full model replica.

Best when:

* model fits in one GPU
* scaling throughput

---

## FSDP

Parameters, gradients, and optimizer states are sharded across GPUs.

Best when:

* training large models
* GPU memory is limited

---

# Project Structure

```text
dist-train-project/
├── train.py
├── config.yaml
├── distributed_utils.py
├── checkpoint.py
├── parallelism.py
├── data.py
├── utils.py
├── model.py
├── configs/
├── scripts/
├── tests/
├── ui/
└── Documentation/
```

---

# Testing

## Unit tests

```bash
python -m pytest
```

## GPU smoke tests

```bash
python -m pytest -m smoke
```

---

# SLURM / Multi-Node

## Submit a batch job

```bash
sbatch scripts/slurm_train.sh configs/llm_fsdp.yaml
```

## Python launcher

```bash
python scripts/launch_slurm.py \
    --config configs/llm_fsdp.yaml \
    --nodes 4 \
    --gpus 8
```

---

# Troubleshooting

| Error                        | Fix                                                                |
| ---------------------------- | ------------------------------------------------------------------ |
| CUDA out of memory           | Reduce batch size or enable checkpointing                          |
| `401 / 403` from HuggingFace | Verify `HF_TOKEN`                                                  |
| `ImportError: VideoReader`   | Recreate `.venv` and reinstall matching torch/torchvision versions |
| NCCL timeout                 | Check network / InfiniBand connectivity                            |
| W&B auth failure             | Set `WANDB_API_KEY` or disable wandb                               |

---

# Roadmap

* DeviceMesh parallelism
* Tensor parallelism
* Pipeline parallelism
* Expert parallelism (MoE)
* SLURM

---

# License

Free for:

* personal use
* research
* education

Commercial use requires a separate agreement.

See [LICENSE](LICENSE).
