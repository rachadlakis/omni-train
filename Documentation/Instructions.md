# OMNI Train: FSDP Mini Project Guide

This document explains how to set up and run the training workflow in `fsdp-mini-project` on both local Linux/WSL and RunPod.

## 1. Project Layout (Important Files)

- `train.py`: main training entrypoint.
- `launch.sh`: launcher for solo or distributed runs with `torchrun`.
- `config.yaml`: training/model/distributed settings.
- `.env`: secrets and tokens (`HF_TOKEN`, optional `WANDB_API_KEY`).
- `checkpoints/`: saved model and optimizer checkpoints.

## 2. Prerequisites

- Linux environment (WSL2, Ubuntu, or RunPod).
- Python 3.10+ recommended.
- NVIDIA GPU drivers and CUDA runtime available.
- Access token for Hugging Face models (`HF_TOKEN`).

Check CUDA:

```bash
nvcc --version
```

## 3. Local Setup (WSL/Linux)

### Step 1: Enter the project

```bash
cd /home/rachad_lakkis/projects/distributed-training/Local/omni_train/testing/fsdp-mini-project
```

### Step 2: Create and activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### Step 3: Install PyTorch (choose the CUDA build that matches your system)

```bash
# CUDA 12.8
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu128

# CUDA 12.4
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124
```

### Step 4: Install Python dependencies used by this mini-project

Note: this folder currently does not include a `requirements.txt`, so install dependencies directly:

```bash
pip install pyyaml python-dotenv transformers datasets wandb plotext
```

For PEFT LoRA / QLoRA training, also install:

```bash
pip install peft bitsandbytes accelerate
```

### Step 5: Select interpreter in VS Code

Use:

```text
.venv/bin/python
```

## 4. Environment Variables

### Option A: Use `.env` (recommended)

```bash
cp .env.template .env
```

Then edit `.env` and set at least:

```env
HF_TOKEN=hf_xxx
WANDB_API_KEY=...    # optional unless wandb logging is enabled
```

### Option B: Export in shell

```bash
export HF_TOKEN=hf_xxx
export WANDB_API_KEY=...   # optional
```

## 5. Configure Training (`config.yaml`)

Edit `config.yaml` before each run.

Common fields to tune:

- `model_name`: Hugging Face model name.
- `dataset.name`, `dataset.subset`, `dataset.split`.
- `training.epochs`, `training.batch_size`, `training.max_length`, `training.learning_rate`.
- `strategy`: `solo`, `ddp`, or `fsdp`.
- `fsdp.*`: mixed precision and checkpoint API options.
- `peft.*`: enable LoRA/QLoRA and choose adapter hyperparameters.
- `quantization.*`: enable 4-bit/8-bit bitsandbytes quantization.
- `save_load.*`: resume settings.
- `wandb.*`: enable/disable experiment logging.

### PEFT and Quantization Config Blocks

Use these blocks in `config.yaml`:

```yaml
training:
    gradient_checkpointing: true

peft:
    enabled: true
    type: lora        # lora or qlora
    r: 16
    alpha: 32
    dropout: 0.05
    target_modules: all-linear
    bias: none

quantization:
    enabled: false    # set true for QLoRA or 8-bit LoRA
    bits: 4           # 4 or 8
    quant_type: nf4   # nf4/fp4 for 4-bit
    compute_dtype: bfloat16
    double_quant: true
```

Rules enforced by the code:

- `peft.type: qlora` automatically implies PEFT + quantization enabled.
- QLoRA requires `quantization.bits: 4`.
- Quantization training currently requires PEFT enabled.

### Mixed Precision with PEFT/Quantization

- LoRA without quantization: mixed precision works normally in DDP/FSDP (`fsdp.mixed_precision`).
- QLoRA / quantized LoRA: quantization compute dtype (`quantization.compute_dtype`) controls low-bit compute path.
- In quantized FSDP runs, FSDP mixed precision policy is intentionally skipped to avoid conflicting dtype policies on quantized base weights.
- You can still keep mixed precision enabled for non-quantized runs in the same config workflow.

Current launcher behavior note:

- `launch.sh` uses `NUM_GPUS`  and `config_path` defined inside the script for `torchrun --nproc_per_node`.

## 6. Run Training

### Option A: launcher script (recommended)

```bash
bash launch.sh
```

`launch.sh` behavior:

- if `STRATEGY=solo`, it runs single-process Python.
- otherwise it uses `torchrun` distributed launch.

Example for solo mode:

```bash
STRATEGY=solo bash launch.sh
```

### Option B: direct torchrun

```bash
CONFIG_PATH=config.yaml torchrun \
    --nproc_per_node=1 \
    --master_addr=localhost \
    --master_port=29500 \
    train.py
```

## 7. Check Results

### Verify checkpoints

```bash
ls -lah checkpoints
```

### If W&B is enabled, inspect run logs

```bash
ls -lah wandb
```

## 8. RunPod Setup

### Step 1: SSH into RunPod

```bash
ssh root@<runpod-host> -p <port> -i ~/.ssh/id_ed25519
```

### Step 2: Clone and enter repo

```bash
git clone <your-repo-url>
cd <repo>/Local/omni_train/testing/fsdp-mini-project
```

### Step 3: Repeat local setup steps

Follow the same steps from sections 3, 4, and 5.

### Step 4: Launch training

```bash
bash launch.sh
```

## 9. Troubleshooting

- `CUDA out of memory`:
    - reduce `training.batch_size`.
    - reduce `training.max_length`.
    - use a smaller model (for example `facebook/opt-125m`).
- `401/403` when loading model/tokenizer:
    - verify `HF_TOKEN` is valid and exported.
- Distributed init errors / port conflicts:
    - change `MASTER_PORT` in `launch.sh`.
- W&B auth errors:
    - set `WANDB_API_KEY` or disable logging in `config.yaml`.
- `ImportError` for `peft` or `bitsandbytes`:
    - install `peft bitsandbytes accelerate` in your active environment.
- QLoRA config errors:
    - ensure `peft.type=qlora` with `quantization.enabled=true` and `quantization.bits=4`.

## 10. Quick Start

```bash
cd /home/rachad_lakkis/projects/distributed-training/Local/omni_train/testing/fsdp-mini-project
source .venv/bin/activate
bash launch.sh
```

Or with torchrun:

```bash
CONFIG_PATH=config.yaml torchrun \
    --nproc_per_node=nb_gpus \
    --master_addr=localhost \
    --master_port=29500 \
    train.py
```



## 11. Web UI

You can launch training from a local dark-themed browser UI.

### Start UI server

```bash
cd /home/rachad_lakkis/projects/distributed-training/Local/omni_train/testing/fsdp-mini-project
source /home/rachad_lakkis/projects/distributed-training/.venv/bin/activate
bash ui/launch_ui.sh
```

Open:

```text
http://127.0.0.1:8787
```

### UI features

- View and edit `config.yaml`
- Show key training settings and environment readiness
- Launch training via `launch.sh`
- Stop running training
- Stream logs/output live in the browser

You can launch a browser UI similar to `omni_train/ui` for this mini-project.

From the project folder:

```bash
cd /home/rachad_lakkis/projects/distributed-training/Local/omni_train/testing/fsdp-mini-project
pip install -r ui/requirements-ui.txt
uvicorn ui.app:app --reload --port 8010
```

Open:

```text
http://localhost:8010
```

UI features:

- edit and validate `config.yaml`
- start/stop training with selected strategy, GPU count, and master port
- monitor live logs
- inspect GPU availability
- view recent checkpoints
- get a quick training-time estimate
