# Distributed Training Mini-Project

This project is a modularized framework for distributed language model training using PyTorch. It is designed to compare and experiment with two parallel training strategies:

- **DDP** (`DistributedDataParallel`)
- **FSDP** (`FullyShardedDataParallel`)

It leverages the Hugging Face `transformers` and `datasets` ecosystems to train causal language models (such as `facebook/opt-125m`) efficiently across multiple GPUs. The codebase implements advanced optimization techniques like mixed precision, gradient checkpointing, explicit layer prefetching, and distributed checkpointing (DCP & DTensor APIs) to avoid Out-Of-Memory (OOM) errors during large model training.

## 📂 Project Structure

The codebase is organized into several modular components for ease of maintenance and extensibility:

- **`train.py`**: The main entry point. Orchestrates the training loop, sets up the environment, and triggers the FSDP/DDP workflows.
- **`config.yaml`**: The central configuration file. Defines model, dataset, training, and FSDP-specific settings.
- **`fsdp_utils.py`**: Contains core distributed computing logic. It includes device setup, FSDP and DDP model wrappers, mixed precision configurations, and GPU memory profiling.
- **`checkpoint.py`**: Manages distributed state persistence. It handles checkpoint saving and loading using both PyTorch DCP and DTensor APIs.
- **`data.py`**: Handles dataset downloading, tokenization, and construction of the PyTorch `DistributedSampler` and `DataLoader`.
- **`utils.py`**: Contains generic utilities, including training configuration formatting and terminal-based loss plotting.
- **`model.py`**: (Optional) Custom model definitions.

## 🚀 Key Features

1. **Fully Sharded Data Parallel (FSDP)**: Distributes model parameters, gradients, and optimizer states across multiple GPUs to drastically reduce memory overhead.
2. **Meta-Device Initialization**: Initializes models directly on the meta-device before sharding, preventing host OOM errors when loading large pretrained models.
3. **Mixed Precision Training**: Configurable precision policies (e.g., `bfloat16` for weights and `float32` for reductions) via PyTorch's `MixedPrecisionPolicy`.
4. **Gradient Checkpointing**: Trades compute for memory by discarding intermediate activations during the forward pass and recomputing them during the backward pass.
5. **Layer Prefetching**: Overlaps communication and computation by prefetching upcoming layers explicitly in both forward and backward passes.
6. **Robust Checkpointing**: Save and resume large distributed models seamlessly using PyTorch's latest Distributed Checkpoint (DCP) utilities.

---

## 💻 Setup Instructions

### Local Setup (WSL)

1. **Create Virtual Environment**

   ```bash
   # Navigate to your project
   cd ./dist-train-project
   
   # Create and activate the venv
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install Dependencies**
   Check your CUDA version first:

   ```bash
   nvcc --version
   ```

   Then install the matching PyTorch for GPU build:

   ```bash
   # For CUDA 12.4
   pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
       --index-url https://download.pytorch.org/whl/cu128
   
   # For CUDA 12.4
   pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
       --index-url https://download.pytorch.org/whl/cu124

   Etc.
   ```

   Then install the rest of the project requirements:

   ```bash
   pip install -r requirements.txt
   ```

3. **VS Code Interpreter**
   Select the Python interpreter from: `dist-train-project/.venv/bin/python`

### RunPod Setup

1. **Connect via SSH**

   ```bash
   ssh root@194.68.245.152 -p 22116 -i ~/.ssh/id_ed25519
   ```

2. **Clone and Open the Project**

   ```bash
   git clone <your-repo-url>
   cd repo_name
   ```

3. **Create Virtual Environment and Install Dependencies**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   
   # Check CUDA version
   nvcc --version
   
   # Install PyTorch (Choose the one matching your CUDA)
   # CUDA 12.8
   pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
       --index-url https://download.pytorch.org/whl/cu128
   # CUDA 12.4
   pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
       --index-url https://download.pytorch.org/whl/cu124
   
   pip install -r requirements.txt
   ```

### To save changes to GitHub

Add GitHub name and email in terminal:

```bash
git config --global user.email "[EMAIL_ADDRESS]"
git config --global user.name "Your Name"
```

---

## 🛠️ Usage & Running Tests

### 1. Set Environment Variables

A `.env` file is required to store your Hugging Face authentication token, or you can export it:

```bash
export HF_TOKEN=your_huggingface_token_here
```

### 2. Activate and Navigate

```bash
source .venv/bin/activate
cd dist-train-project
```

### 3. Launch

All training hyperparameters and system configurations are controlled via `config.yaml`. Before running, you can adjust settings like the batch size, epochs, and distributed strategy (`fsdp` vs `ddp`).

You can launch the distributed training run using `torchrun`:

```bash
# Option A: bash launcher (recommended — sets all config variables)
bash launch.sh

# Option B: torchrun directly (minimal example)
torchrun --nproc_per_node=1 train.py
```

## 📖 Further Reading

For a deep dive into the technical implementation details of how the script manages PyTorch distributed primitives, FSDP sharding strategies, and memory optimization techniques, see the [Technical Explanation](TRAIN_PY_TECHNICAL_EXPLANATION.md).

## 🌐 Simple Training UI (Dark Theme)

This mini-project now includes a lightweight local web UI to launch training, view and edit `config.yaml`, and stream training logs in real time.

### What the UI provides

- Dark themed launcher page
- Shows current `config.yaml` content (editable)
- Displays key training values (model, strategy, dataset, epochs, batch size, max length, lr)
- Shows required runtime info (`HF_TOKEN`, CUDA/GPU visibility, `.env` presence)
- Start/stop controls for training (`launch.sh` underneath)
- Live output panel with terminal logs from the training run

### Run the UI

From this folder:

```bash
cd /home/rachad_lakkis/projects/dist-train-project
source /home/rachad_lakkis/projects/dist-train-project/.venv/bin/activate
bash ui/launch_ui.sh
```

Then open:

```text
http://127.0.0.1:8787
```

Optional environment overrides:

```bash
UI_HOST=0.0.0.0 UI_PORT=8787 bash ui/launch_ui.sh
```

The UI start action passes these launcher variables to `launch.sh`:

- `STRATEGY`
- `NUM_GPUS`
- `MASTER_ADDR`
- `MASTER_PORT`
- `CONFIG_PATH` (auto-set to this folder's `config.yaml`)
