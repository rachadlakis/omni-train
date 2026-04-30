# OMNI-Train: Getting Started Guide

A unified framework for fine-tuning AI models with an easy-to-use Web UI.

---

## What Can You Train?

| Model Type | Examples | Use Cases |
|------------|----------|-----------|
| **Image Classification (CNN)** | ResNet, ViT, EfficientNet | Classify images into categories |
| **Language Models (LLM)** | LLaMA, Mistral, Phi-3 | Text generation, chatbots |
| **Vision-Language Models (VLM)** | LLaVA, BLIP-2, Qwen-VL | Image captioning, visual Q&A |
| **Object Detection** | YOLOv8, YOLOv11, YOLOv12 | Detect objects in images |
| **Embeddings** | BERT, E5, CLIP, Sentence Transformers | Similarity search, retrieval, RAG |

---

## Quick Start (5 Minutes)

### Step 1: Install Dependencies

The setup script auto-detects your GPU and CUDA version and installs the correct PyTorch wheel automatically:

```bash
python scripts/setup_env.py
```

Works on any machine — with or without a GPU. No need to know your CUDA version.

| Option      | Description                                           |
|-------------|-------------------------------------------------------|
| `--dry-run` | Print the install command without running it          |
| `--cpu`     | Force CPU-only install (e.g. on a server with no GPU) |

> **Manual install (Linux/macOS):**
>
> ```bash
> cd Local
> python -m venv ../.venv
> source ../.venv/bin/activate
> pip install -r omni_train/requirements.txt
> ```

### Manual CUDA + PyTorch Install (from fsdp-mini-project docs)

Use this when you want explicit control over the CUDA-specific PyTorch build.

1. Verify CUDA runtime/toolkit visibility:

```bash
nvcc --version
```

2. Install the matching PyTorch wheel:

```bash
# CUDA 12.8
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu128

# CUDA 12.4
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
```

3. Install OMNI-Train dependencies:

```bash
pip install -r omni_train/requirements.txt
```

4. Optional extras for PEFT/QLoRA workflows:

```bash
pip install peft bitsandbytes accelerate
```

### Step 2: Start the Web UI

```bash
cd Local
source ../.venv/bin/activate
python -m omni_train.ui.app
```

For a remote Linux server or VM:

```bash
cd Local
source ../.venv/bin/activate
uvicorn omni_train.ui.app:app --host 0.0.0.0 --port 8000
```

Open your browser to **<http://localhost:8000>** locally, or `http://<server-ip>:8000` on a remote Linux host.

### Step 3: Choose a Template & Train

1. Click **"Browse Templates"** on the landing page
2. Pick a model (e.g., "LLaMA 2 7B with LoRA")
3. Click **"Start Training"**

That's it! Watch the training progress in real-time.

---

## Using the Web UI

### Landing Page Options

| Option | What It Does |
|--------|--------------|
| **Browse Templates** | Start from pre-configured setups (recommended for beginners) |
| **Custom Setup** | Configure everything manually |
| **YAML Editor** | Edit configuration as code |

### Training Workflow

```
Choose Template → Adjust Settings → Start Training → Monitor Progress → Download Model
```

### Key Settings Explained

| Setting | What It Means |
|---------|---------------|
| **Model Type** | CNN (images), LLM (text), VLM (images+text), Detection (objects), Embedding (similarity) |
| **Fine-tune Mode** | Full (train all), LoRA (efficient), QLoRA (memory-efficient) |
| **Batch Size** | Samples per step (lower = less memory) |
| **Learning Rate** | How fast the model learns (default: 2e-5) |
| **Epochs** | How many times to go through the data |
| **Loss Type** | For embeddings: infonce, triplet, cosine, mnrl, contrastive |
| **Pooling Mode** | For embeddings: mean, cls, max |

---

## Training Modes

### Full Fine-tuning

- Trains **all** model parameters
- Best accuracy, but needs lots of memory
- Use for small models or when you have powerful GPUs

### LoRA (Low-Rank Adaptation)

- Only trains small adapter layers
- 10x less memory than full fine-tuning
- Great for most use cases

### QLoRA (Quantized LoRA)

- Combines LoRA with 4-bit quantization
- Run 7B models on 8GB GPU
- Recommended for consumer GPUs
- **Note**: Only available for LLM and VLM models

### Embedding Training

- Supports LoRA for efficient fine-tuning
- Multiple loss functions: InfoNCE, triplet, cosine, MNRL
- Works with text, vision, or CLIP models

---

## Command Line Usage (Without UI)

You can run all training directly from the command line without the Web UI.

### Single GPU Training

```bash
python -m omni_train.train --config configs/llm_lora_quantized_single_gpu.yaml
```

### Multi-GPU Training (DDP)

```bash
torchrun --nproc_per_node=4 -m omni_train.train --config configs/llm_lora_ddp.yaml
```

### Multi-GPU Training (FSDP)

```bash
torchrun --nproc_per_node=4 -m omni_train.train --config configs/llm_full_finetune_fsdp.yaml
```

### 2D Parallel Training (DP + TP)

```bash
torchrun --nproc_per_node=4 -m omni_train.train --config configs/llm_hybrid_2d_dp_tp.yaml
```

### Pipeline Parallel Training (PP)

```bash
# Dedicated pipeline trainer (manual GPipe-style)
torchrun --nproc_per_node=4 Local/pipeline-parallelism.py
```

### Multi-Node Training (SLURM)

```bash
# Using the SLURM script
sbatch scripts/slurm_train.sh configs/llm_full_finetune_fsdp.yaml

# Or using the Python launcher
python scripts/launch_slurm.py --config configs/llm_fsdp.yaml --nodes 4 --gpus 8
```

### CLI Arguments

| Argument | Description |
|----------|-------------|
| `--config PATH` | **(Required)** Path to YAML configuration file |
| `--lr FLOAT` | Override learning rate from config |
| `--batch-size INT` | Override batch size from config |
| `--epochs INT` | Override number of epochs from config |
| `--wandb_project STR` | Enable W&B logging with this project name |
| `--wandb_run_name STR` | Custom W&B run name (auto-generated if not set) |

### Override Settings

```bash
# Override learning rate and epochs
python -m omni_train.train --config configs/cnn_resnet.yaml --lr 1e-4 --epochs 10

# Override batch size
python -m omni_train.train --config configs/llm_lora.yaml --batch-size 2
```

> **Note:** The CLI uses a config-file-first design. All model, data, and distributed settings must be defined in a YAML config file. The CLI arguments only allow overriding common training parameters (lr, batch-size, epochs).

---

## Example Configurations

### Train LLaMA 2 with QLoRA (Low Memory)

```yaml
seed: 42

model:
  type: llm
  name: meta-llama/Llama-2-7b-hf
  finetune_mode: lora
  quantize: true
  quant_bits: 4
  lora_r: 16
  lora_alpha: 32

data:
  type: hf_dataset
  name: your-dataset-name
  max_seq_len: 2048

training:
  epochs: 3
  batch_size: 4
  lr: 2e-4
  grad_accum_steps: 4
```

### Train Image Classifier (ResNet)

```yaml
seed: 42

model:
  type: cnn
  name: resnet50
  num_classes: 10
  pretrained: true
  freeze_backbone: false

data:
  type: torchvision
  name: CIFAR10
  image_size: 224

training:
  epochs: 10
  batch_size: 32
  lr: 1e-3
```

### Train Vision-Language Model (LLaVA)

```yaml
seed: 42

model:
  type: vlm
  name: llava-hf/llava-1.5-7b-hf
  finetune_mode: lora
  quantize: true
  quant_bits: 4
  lora_target: llm_only

data:
  type: hf_dataset
  name: your-vlm-dataset
  image_size: 336
  max_seq_len: 2048

training:
  epochs: 1
  batch_size: 1
  lr: 2e-4
  grad_accum_steps: 16
```

### Train Object Detector (YOLOv8)

```yaml
seed: 42

model:
  type: detection
  yolo_model: yolov8n.pt

data:
  type: yolo
  data_yaml: path/to/data.yaml
  image_size: 640

training:
  epochs: 100
  batch_size: 16
  lr: 0.01
```

### Train Text Embeddings (Sentence Transformers)

```yaml
seed: 42

model:
  type: embedding
  name: sentence-transformers/all-MiniLM-L6-v2
  embedding_type: text
  pooling_mode: mean
  loss_type: infonce

data:
  type: hf_dataset
  name: sentence-transformers/all-nli
  embedding_format: pairs

training:
  epochs: 3
  batch_size: 64
  lr: 2e-5
```

---

## Data Formats

### For LLMs (Text)

- **HuggingFace datasets**: `type: hf_dataset`, `name: dataset-name`
- **Local JSONL**: `type: local_file`, `name: path/to/data.jsonl`
- **Format**: `{"text": "Your training text here"}`

### For CNNs (Images)

- **Torchvision**: `type: torchvision`, `name: CIFAR10`
- **Image folder**: `type: image_folder`, `name: path/to/images/`
- **Structure**: `images/class_name/image.jpg`

### For VLMs (Images + Text)

- **HuggingFace**: `type: hf_dataset`, `name: dataset-with-images`
- **Local JSON**: `type: local_file`, `name: path/to/data.json`
- **Format**: `{"image": "path/to/img.jpg", "caption": "Description"}`

### For Detection (YOLO)

- **YOLO format**: `type: yolo`, `data_yaml: path/to/data.yaml`
- **COCO format**: `type: coco`, `coco_train_json: annotations.json`
- **Note**: COCO format is automatically converted to YOLO format

### For Embeddings

- **Pairs format**: `{"sentence1": "text A", "sentence2": "text B", "score": 0.8}`
- **Triplet format**: `{"anchor": "query", "positive": "match", "negative": "non-match"}`
- **Labeled format**: `{"text": "sample", "label": 0}`
- Set `embedding_format` in config: `pairs`, `triplet`, or `labeled`

---

## GPU Memory Requirements

| Model | Full Fine-tune | LoRA | QLoRA (4-bit) |
|-------|----------------|------|---------------|
| LLaMA 7B | 60+ GB | 16 GB | 6 GB |
| LLaMA 13B | 100+ GB | 24 GB | 10 GB |
| LLaMA 70B | 500+ GB | 80 GB | 40 GB |
| ResNet-50 | 4 GB | - | - |
| YOLOv8-n | 4 GB | - | - |
| BERT-base | 2 GB | 1 GB | - |
| all-MiniLM-L6 | 1 GB | - | - |
| CLIP ViT-B/32 | 4 GB | 2 GB | - |

---

## Distributed Training Strategies

| Strategy | When to Use |
|----------|-------------|
| **none** | Single GPU |
| **ddp** | 1D data parallel training |
| **fsdp** | 1D sharded data parallel training for large models |
| **hybrid** | 2D/3D topology declaration (DP/TP/PP dimensions) |

### Hybrid Topology Fields

```yaml
distributed:
  strategy: hybrid
  parallelism_mode: 2d   # 1d | 2d | 3d
  data_parallel_size: 2
  tensor_parallel_size: 2
  pipeline_parallel_size: 1
```

> `data_parallel_size * tensor_parallel_size * pipeline_parallel_size` must equal `WORLD_SIZE`.
> 
> Current trainer supports DP and TP in `omni_train.train`. For PP execution, use `Local/pipeline-parallelism.py` or add a model-specific stage-partitioned trainer.

---

## Troubleshooting

### Out of Memory

1. Reduce `batch_size`
2. Increase `grad_accum_steps`
3. Enable quantization (`quantize: true`) - LLM/VLM only
4. Use LoRA instead of full fine-tuning
5. Enable `activation_checkpointing: true` for FSDP
6. Use FSDP for very large models (7B+)

### Training Too Slow

1. Increase `batch_size` (if memory allows)
2. Reduce `grad_accum_steps`
3. Use mixed precision (`mixed_precision: true`)
4. Use multiple GPUs with DDP
5. Enable `compile_model: true` (PyTorch 2.0+)
6. Increase `num_workers` for data loading

### Model Not Learning

1. Increase `lr` (learning rate)
2. Train for more `epochs`
3. Check your data format
4. Try a smaller model first
5. Add warmup: `warmup_steps: 100`

### Embedding Training Issues

1. **Wrong loss type**: Use `triplet` loss with `triplet` data format
2. **Low similarity scores**: Try `infonce` loss with in-batch negatives
3. **Overfitting**: Reduce `lr`, add dropout, use smaller model

---

## Project Structure

```
omni-train/
├── config/           # Configuration schema and loading
├── configs/          # Example YAML configurations
├── data/             # Data loading for each model type
├── distributed/      # Multi-GPU training setup (DDP, FSDP)
├── models/           # Model builders (CNN, LLM, VLM, Detection, Embedding)
├── training/         # Training loop and checkpointing
├── ui/               # Web interface (FastAPI)
├── scripts/          # Setup and SLURM launcher scripts
├── utils/            # Helpers (logging, seed, quantization)
└── train.py          # Main entry point
```

---

## Roadmap: Coming Soon

Additional model types planned for future releases:

| Model Type | Examples | Use Cases | Priority |
|------------|----------|-----------|----------|
| **Seq2Seq** | T5, BART, mT5, Flan-T5 | Translation, summarization, Q&A | High |
| **Speech/Audio** | Whisper, Wav2Vec2, HuBERT | Transcription, audio classification | High |
| **Segmentation** | SAM, Mask2Former, DeepLabV3 | Semantic/instance segmentation | High |
| **Diffusion** | Stable Diffusion, SDXL, Flux | Image generation with LoRA | High |
| **Document AI** | LayoutLM, Donut, Pix2Struct | OCR, document understanding | Medium |
| **Time Series** | Chronos, TimesFM, PatchTST | Forecasting, anomaly detection | Medium |
| **Alignment (RLHF)** | DPO, ORPO, PPO | LLM preference tuning | Medium |
| **Video** | VideoMAE, TimeSformer | Action recognition, video understanding | Medium |

---

## Need Help?

- Check the example configs in `configs/` folder
- Use the Web UI for guided setup
- Start with a small model and dataset to test your setup

Happy Training!
