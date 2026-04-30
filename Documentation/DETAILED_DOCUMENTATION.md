# OMNI-Train: Detailed Documentation

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [AI/ML Concepts](#aiml-concepts)
4. [Code Structure](#code-structure)
5. [Configuration System](#configuration-system)
6. [Model Types & Fine-Tuning](#model-types--fine-tuning)
7. [Data Loading](#data-loading)
8. [Distributed Training](#distributed-training)
9. [Training Loop](#training-loop)
10. [Web UI](#web-ui)
11. [API Reference](#api-reference)
12. [Troubleshooting](#troubleshooting)

---

## Overview

OMNI-Train is a unified framework for fine-tuning various deep learning model types with distributed training support. It provides a single interface to train:

- **CNN** (Convolutional Neural Networks) - Image classification
- **LLM** (Large Language Models) - Text generation
- **VLM** (Vision-Language Models) - Multimodal understanding
- **Detection** (YOLO models) - Object detection
- **Embedding** (Text/Vision/CLIP embeddings) - Similarity search and retrieval

### Installation

The setup script auto-detects your GPU and CUDA version and installs the correct PyTorch wheel:

```bash
python scripts/setup_env.py
```

| Option      | Description                                           |
|-------------|-------------------------------------------------------|
| `--dry-run` | Print the install command without running it          |
| `--cpu`     | Force CPU-only install (e.g. on a server with no GPU) |

For Linux/macOS manual setup from the parent `Local/` directory:

```bash
cd Local
python -m venv .venv
source .venv/bin/activate
pip install -r omni_train/requirements.txt
```

### Key Features

- Single configuration file for all model types
- Distributed training with DDP and FSDP
- LoRA and QLoRA support for memory-efficient fine-tuning
- Weights & Biases integration for experiment tracking
- Web UI for easy configuration and monitoring
- Checkpoint management with best model tracking

---

## Architecture

```
                                   +------------------+
                                   |    Web UI        |
                                   |  (FastAPI)       |
                                   +--------+---------+
                                            |
                                            v
+------------------+              +------------------+
|  YAML Config     | ----------> |   Config Loader  |
+------------------+              +--------+---------+
                                            |
                                            v
                                  +------------------+
                                  |   train.py       |
                                  | (Entry Point)    |
                                  +--------+---------+
                                            |
                    +-----------------------+-----------------------+
                    |                       |                       |
                    v                       v                       v
          +------------------+    +------------------+    +------------------+
          | Model Builder    |    | Data Builder     |    | Distributed      |
          | (models/)        |    | (data/)          |    | Setup            |
          +--------+---------+    +--------+---------+    +--------+---------+
                    |                       |                       |
                    +-----------------------+-----------------------+
                                            |
                                            v
                                  +------------------+
                                  |    Trainer       |
                                  | (training/)      |
                                  +--------+---------+
                                            |
                    +-----------------------+-----------------------+
                    |                       |                       |
                    v                       v                       v
          +------------------+    +------------------+    +------------------+
          | Checkpointing    |    | LR Scheduler     |    | W&B Logger       |
          +------------------+    +------------------+    +------------------+
```

---

## AI/ML Concepts

### Fine-Tuning Methods

#### Full Fine-Tuning
Updates all model parameters during training. Requires more memory and compute but can achieve best results.

```
Original Model (all weights)
         |
    [Training]
         |
         v
Updated Model (all weights modified)
```

#### LoRA (Low-Rank Adaptation)
Freezes original weights and trains small adapter matrices. Memory-efficient but may have slightly lower performance.

```
Original Model (frozen)
         |
    + LoRA Adapters (trainable, small matrices)
         |
         v
Adapted Model (original + adapters)
```

**How LoRA Works:**
- For a weight matrix W (d x k), LoRA adds two smaller matrices: A (d x r) and B (r x k)
- Output = W*x + (A*B)*x where r << min(d, k)
- Typical r values: 8, 16, 32, 64

#### QLoRA (Quantized LoRA)
Combines quantization with LoRA for even lower memory usage.

```
Original Model (4-bit quantized, frozen)
         |
    + LoRA Adapters (full precision, trainable)
         |
         v
Adapted Model (quantized base + adapters)
```

**Quantization Levels:**
- **4-bit (NF4)**: ~4x memory reduction, minimal quality loss
- **8-bit (INT8)**: ~2x memory reduction, very minimal quality loss

### Distributed Training Strategies

#### DDP (DistributedDataParallel)

Each GPU has a complete copy of the model. Gradients are synchronized after backward pass.

```
GPU 0: Full Model Copy    GPU 1: Full Model Copy    GPU 2: Full Model Copy
         |                         |                         |
    [Forward Pass]           [Forward Pass]           [Forward Pass]
         |                         |                         |
    [Backward Pass]          [Backward Pass]          [Backward Pass]
         |                         |                         |
         +------------> All-Reduce Gradients <---------------+
                              |
                    [Optimizer Step]
```

**Best for:** Models that fit in single GPU memory

#### FSDP (Fully Sharded Data Parallel)

Parameters are sharded across GPUs. Each GPU only stores 1/N of parameters.

```
GPU 0: Shard 0    GPU 1: Shard 1    GPU 2: Shard 2
    |                  |                  |
    +------ All-Gather Parameters --------+
    |                  |                  |
[Forward]          [Forward]          [Forward]
    |                  |                  |
    +------ Reduce-Scatter Gradients -----+
    |                  |                  |
[Update]           [Update]           [Update]
```

**Best for:** Models larger than single GPU memory (7B+ parameters)

### Loss Functions

#### Cross-Entropy Loss (Classification)
```python
L = -sum(y_i * log(p_i))
```
Used for CNN classification tasks.

#### Causal Language Modeling Loss (LLM)
```python
L = -sum(log P(token_i | tokens_0..i-1))
```
Predicts next token given previous tokens.

#### Contrastive Loss (Embeddings)
```python
L = -log(exp(sim(a, p)/τ) / sum(exp(sim(a, n)/τ)))
```
Pulls similar items together, pushes dissimilar items apart.

### Learning Rate Schedules

#### Cosine Annealing
```
LR = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(π * step / max_steps))
```
Smooth decay following cosine curve.

#### Linear Decay
```
LR = max_lr - (max_lr - min_lr) * (step / max_steps)
```
Linear decrease from max to min.

#### Warmup
Gradually increase LR at start of training to stabilize optimization.
```
if step < warmup_steps:
    LR = max_lr * (step / warmup_steps)
```

### Gradient Accumulation

Simulates larger batch sizes without more memory.

```
Effective Batch Size = batch_size * grad_accum_steps * world_size

Example:
  batch_size = 4
  grad_accum_steps = 8
  world_size = 2 GPUs
  Effective Batch = 4 * 8 * 2 = 64
```

### Mixed Precision Training

Uses lower precision (FP16/BF16) for faster computation while maintaining FP32 for critical operations.

```
Forward Pass: BF16 (fast)
Backward Pass: BF16 (fast)
Gradient Accumulation: FP32 (stable)
Optimizer Step: FP32 (stable)
```

---

## Code Structure

```
omni_train/
├── config/
│   ├── loader.py         # YAML loading + CLI overrides + validation
│   └── schema.py         # Dataclass definitions (FTAConfig, ModelConfig, etc.)
├── configs/              # Example YAML configurations
│   ├── cnn_*.yaml        # CNN training examples
│   ├── llm_*.yaml        # LLM training examples
│   ├── vlm_*.yaml        # VLM training examples
│   ├── detection_*.yaml  # YOLO training examples
│   └── embedding_*.yaml  # Embedding training examples
├── data/
│   ├── cnn_data.py       # Image dataset loaders (torchvision, ImageFolder)
│   ├── llm_data.py       # Text dataset loaders (HF, local, S3)
│   ├── vlm_data.py       # Vision-language loaders (image+text pairs)
│   ├── detection_data.py # YOLO/COCO loaders + COCO→YOLO converter
│   ├── embedding_data.py # Pair/triplet/labeled dataset loaders
│   └── registry.py       # Dataset dispatcher by model type
├── distributed/
│   ├── setup.py          # Process group init, device assignment, utilities
│   ├── strategies.py     # DDP/FSDP2 wrappers with activation checkpointing
│   └── checkpoint.py     # Strategy-aware checkpoint save/load
├── models/
│   ├── cnn.py            # Torchvision model builders (ResNet, ViT, etc.)
│   ├── llm.py            # HF CausalLM builders with LoRA/quantization
│   ├── vlm.py            # VLM builders (LLaVA, BLIP-2, Qwen-VL, etc.)
│   ├── detection.py      # YOLO builders via ultralytics
│   ├── embedding.py      # Text/vision/CLIP embedding builders
│   ├── losses.py         # Embedding loss function factory
│   └── registry.py       # Model dispatcher by type
├── training/
│   ├── trainer.py        # Main training loop with progress display
│   ├── checkpointing.py  # Checkpoint management with best model tracking
│   ├── lr_schedule.py    # Cosine/linear/constant LR schedulers
│   └── logging.py        # W&B integration and metrics logging
├── ui/
│   ├── app.py            # FastAPI server with REST API
│   ├── queue.py          # Job queue manager with GPU allocation
│   └── static/           # Frontend HTML/CSS/JS assets
├── scripts/
│   ├── setup_env.py      # Auto-detect GPU/CUDA and install PyTorch
│   ├── launch_slurm.py   # Python SLURM job launcher
│   └── slurm_train.sh    # SLURM batch script template
├── utils/
│   ├── logging.py        # Progress bar, rich terminal output
│   ├── seed.py           # Reproducibility (torch/numpy/python seeds)
│   └── quantization.py   # BitsAndBytes quantization config
└── train.py              # Main entry point
```

---

## Configuration System

### Schema Overview

The configuration is structured into four main sections:

```yaml
model:      # What to train
data:       # What to train on
distributed: # How to distribute
training:   # Training hyperparameters
seed: 42    # Random seed
```

### ModelConfig

```yaml
model:
  type: llm                          # cnn | llm | vlm | detection | embedding
  name: meta-llama/Llama-2-7b-hf     # Model name or path
  source: huggingface                # huggingface | url | local | upload

  # Fine-tuning mode
  finetune_mode: lora                # full | lora
  use_flash_attention: true          # Enable Flash Attention 2

  # LoRA parameters (LLM/VLM/Embedding)
  lora_r: 16                         # Rank (4-128 typical)
  lora_alpha: 32                     # Alpha scaling (usually 2*r)
  lora_dropout: 0.05                 # Dropout rate
  lora_target_modules: null          # null = auto-detect

  # Quantization (LLM/VLM only, incompatible with FSDP)
  quantize: true                     # Enable 4/8-bit quantization
  quant_bits: 4                      # 4 | 8

  # CNN specific
  num_classes: 10                    # Number of output classes
  pretrained: true                   # Use pretrained weights
  freeze_backbone: false             # Freeze feature extractor

  # VLM specific
  lora_target: llm_only              # llm_only | vision_only | both
  freeze_vision_encoder: false       # Freeze vision encoder

  # Embedding specific
  embedding_type: text               # text | vision | clip
  embedding_backend: sentence_transformers  # sentence_transformers | huggingface
  pooling_mode: mean                 # mean | cls | max
  loss_type: infonce                 # infonce | triplet | cosine | mnrl | contrastive

  # Detection specific
  yolo_model: yolov8m.pt             # YOLO model weights file
```

### DataConfig

```yaml
data:
  type: hf_dataset                   # torchvision | image_folder | hf_dataset |
                                     # local_file | s3 | yolo | coco | dummy
  name: tatsu-lab/alpaca             # Dataset name/path/URI

  train_split: train                 # Training split name
  val_split: validation              # Validation split name

  # Image settings (CNN/VLM/Detection)
  image_size: 224                    # Input image size
  augmentation: standard             # standard | autoaugment | none

  # LLM settings
  max_seq_len: 2048                  # Maximum sequence length
  tokenizer_name: null               # Tokenizer (null = same as model)
  text_field: text                   # JSON field for text

  # S3 settings
  s3_region: null                    # AWS region
  s3_endpoint_url: null              # Custom endpoint (MinIO)

  # Detection settings
  data_yaml: path/to/data.yaml       # YOLO data config
  coco_train_json: annotations.json  # COCO annotations
  coco_val_json: val_annotations.json  # COCO val annotations (optional)
  coco_images_dir: images/           # COCO images directory

  # Embedding settings
  embedding_format: pairs            # pairs | triplet | labeled
  # pairs: (text1, text2, score) for similarity
  # triplet: (anchor, positive, negative) for contrastive
  # labeled: (text, label) for classification-based contrastive

  num_workers: 4                     # DataLoader workers
```

### DistributedConfig

```yaml
distributed:
  strategy: ddp                      # none | ddp | fsdp
  mixed_precision: true              # Enable AMP
  param_dtype: bfloat16              # bfloat16 | float16
  activation_checkpointing: false    # Trade compute for memory

  # DDP options
  ddp_find_unused_parameters: false  # Handle unused params
  ddp_gradient_as_bucket_view: true  # Memory optimization
  ddp_static_graph: false            # Static graph optimization
  ddp_bucket_cap_mb: 25              # Gradient bucket size

  # FSDP options
  fsdp_forward_prefetch: true        # Prefetch during forward
  fsdp_backward_prefetch: true       # Prefetch during backward
  fsdp_num_prefetch: 2               # Layers to prefetch
  fsdp_reshard_after_forward: true   # Reshard after forward

  # torch.compile
  compile_model: false               # Enable compilation
  compile_mode: default              # default | reduce-overhead | max-autotune
```

### TrainingConfig

```yaml
training:
  epochs: 3                          # Number of epochs
  batch_size: 8                      # Batch size per GPU
  lr: 2e-5                           # Learning rate
  min_lr: 1e-6                       # Minimum LR for scheduling
  beta1: 0.9                         # Adam beta1
  beta2: 0.95                        # Adam beta2
  weight_decay: 0.01                 # L2 regularization
  grad_clip: 1.0                     # Gradient clipping norm
  grad_accum_steps: 1                # Gradient accumulation
  warmup_steps: 100                  # LR warmup steps
  lr_schedule: cosine                # cosine | linear | constant

  # Checkpointing
  checkpoint_dir: checkpoints        # Save directory
  save_every_n_steps: null           # Step-based saving
  save_every_n_epochs: 1             # Epoch-based saving
  resume_from: null                  # Checkpoint to resume from

  # Logging
  log_interval: 10                   # Steps between logs
  eval_interval: null                # Epochs between eval (null = every epoch)

  # Weights & Biases
  wandb_enabled: false               # Enable W&B
  wandb_project: null                # Project name
  wandb_run_name: null               # Run name (auto if null)
```

### Configuration Loading Flow

```python
# 1. Load YAML file
raw = yaml.safe_load(file)

# 2. Create typed dataclasses
cfg = FTAConfig(
    model=ModelConfig(**raw["model"]),
    data=DataConfig(**raw["data"]),
    distributed=DistributedConfig(**raw["distributed"]),
    training=TrainingConfig(**raw["training"]),
)

# 3. Apply CLI overrides
if cli_args.lr:
    cfg.training.lr = cli_args.lr

# 4. Validate configuration
_validate_config(cfg)  # Raises ValueError on invalid config
```

---

## Model Types & Fine-Tuning

### CNN Models

Supported architectures from torchvision:
- ResNet (18, 34, 50, 101, 152)
- ViT (Vision Transformer)
- EfficientNet (B0-B7)
- ConvNeXt
- MobileNet (V2, V3)

```yaml
model:
  type: cnn
  name: resnet50
  num_classes: 100
  pretrained: true
  freeze_backbone: false  # true = only train classifier
```

**Implementation** (`models/cnn.py`):
```python
def build_cnn_model(config):
    # Load pretrained model
    model = getattr(torchvision.models, config.name)(
        weights="IMAGENET1K_V1" if config.pretrained else None
    )

    # Replace classifier head
    if "resnet" in config.name:
        model.fc = nn.Linear(model.fc.in_features, config.num_classes)

    # Optionally freeze backbone
    if config.freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True

    return model
```

### LLM Models

Any HuggingFace CausalLM model:
- LLaMA, LLaMA 2, LLaMA 3
- Mistral, Mixtral
- GPT-2, GPT-J, GPT-NeoX
- Phi, Phi-2
- Falcon
- And more...

```yaml
model:
  type: llm
  name: meta-llama/Llama-2-7b-hf
  finetune_mode: lora
  quantize: true
  quant_bits: 4
  lora_r: 16
  lora_alpha: 32
```

**LoRA Implementation** (`models/llm.py`):
```python
def _build_lora_model(config):
    # Load base model (optionally quantized)
    kwargs = {"trust_remote_code": True}
    if config.quantize:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=(config.quant_bits == 4),
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(config.name, **kwargs)

    # Prepare for k-bit training
    if config.quantize:
        model = prepare_model_for_kbit_training(model)

    # Apply LoRA
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules or ["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)

    return model
```

### VLM Models

Vision-Language Models for multimodal tasks:
- LLaVA
- BLIP-2
- Qwen-VL
- InstructBLIP
- CogVLM

```yaml
model:
  type: vlm
  name: llava-hf/llava-1.5-7b-hf
  finetune_mode: lora
  lora_target: llm_only  # or vision_only, both
  freeze_vision_encoder: true
```

### Detection Models (YOLO)

All YOLO variants via Ultralytics:
- YOLOv5 (n/s/m/l/x)
- YOLOv8 (n/s/m/l/x)
- YOLOv9 (t/s/m/c/e)
- YOLOv10, YOLOv11, YOLOv12
- YOLO-World (open vocabulary)
- RT-DETR

```yaml
model:
  type: detection
  name: yolov8m
  yolo_model: yolov8m.pt
```

**Note:** Detection training delegates to Ultralytics trainer, bypassing the standard training loop.

### Embedding Models

For similarity search, retrieval, and representation learning:

**Text Embedding Models:**
- Sentence Transformers (all-MiniLM, all-mpnet, etc.)
- BERT, RoBERTa, E5, BGE
- Any HuggingFace encoder model

**Vision Embedding Models:**
- ResNet, EfficientNet, ViT (via torchvision)
- Classification head removed for pure embeddings

**Multimodal (CLIP) Models:**
- OpenAI CLIP variants
- SigLIP, EVA-CLIP

```yaml
model:
  type: embedding
  name: sentence-transformers/all-MiniLM-L6-v2
  embedding_type: text        # text | vision | clip
  embedding_backend: sentence_transformers  # sentence_transformers | huggingface
  pooling_mode: mean          # mean | cls | max
  loss_type: infonce          # infonce | triplet | cosine | mnrl | contrastive
  finetune_mode: full         # full | lora
  freeze_backbone: false      # Freeze encoder, train head only
```

**Supported Loss Functions** (`models/losses.py`):

| Loss Type | Description | Data Format |
|-----------|-------------|-------------|
| `infonce` | Contrastive with in-batch negatives | pairs |
| `triplet` | Anchor-positive-negative margin loss | triplet |
| `cosine` | Cosine similarity regression | pairs with scores |
| `mnrl` | Multiple Negatives Ranking Loss | pairs |
| `contrastive` | Pairwise contrastive with margin | pairs with labels |

---

## Data Loading

### Data Types

| Type | Description | Model Types | Example |
|------|-------------|-------------|---------|
| `torchvision` | Built-in datasets | CNN | CIFAR10, ImageNet |
| `image_folder` | Directory structure | CNN, Embedding (vision) | `root/class_name/image.jpg` |
| `hf_dataset` | HuggingFace Hub | All | `tatsu-lab/alpaca` |
| `local_file` | Local JSON/JSONL/CSV | LLM, VLM, Embedding | `data/train.jsonl` |
| `s3` | S3/MinIO bucket | LLM, VLM | `s3://bucket/data.jsonl` |
| `yolo` | YOLO format | Detection | `data.yaml` |
| `coco` | COCO JSON (auto-converts to YOLO) | Detection | `annotations/train.json` |
| `dummy` | Random data for testing | All | N/A |

### LLM Data Pipeline

```
Raw Data (JSON/HF Dataset)
         |
         v
+------------------+
| Tokenization     |
| - Truncation     |
| - Padding        |
| - Return tensors |
+------------------+
         |
         v
+------------------+
| Dataset Wrapper  |
| - input_ids      |
| - attention_mask |
| - labels         |
+------------------+
         |
         v
+------------------+
| DataLoader       |
| - Batching       |
| - Shuffling      |
| - DistSampler    |
+------------------+
```

**Tokenization Code** (`data/llm_data.py`):
```python
class TokenizedDataset(Dataset):
    def __getitem__(self, idx):
        text = self.dataset[idx]["text"]

        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_len,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encodings["input_ids"].squeeze(0),
            "attention_mask": encodings["attention_mask"].squeeze(0),
            "labels": encodings["input_ids"].squeeze(0).clone(),
        }
```

### Image Data Pipeline

```
Image Files
     |
     v
+------------------+
| Load & Resize    |
+------------------+
     |
     v
+------------------+
| Augmentation     |
| - RandomCrop     |
| - RandomFlip     |
| - ColorJitter    |
| - Normalize      |
+------------------+
     |
     v
+------------------+
| DataLoader       |
| - Batching       |
| - pin_memory     |
| - DistSampler    |
+------------------+
```

### Embedding Data Pipeline

```
Raw Data (JSON/HF Dataset)
         |
         v
+------------------+
| Format Detection |
| - pairs format   |
| - triplet format |
| - labeled format |
+------------------+
         |
         v
+------------------+
| Dataset Wrapper  |
| PairDataset or   |
| TripletDataset   |
+------------------+
         |
         v
+------------------+
| Loss Function    |
| (from factory)   |
+------------------+
```

**Embedding Data Formats:**

```python
# Pairs format (for InfoNCE, cosine, MNRL)
{"sentence1": "text A", "sentence2": "text B", "score": 0.8}

# Triplet format (for triplet loss)
{"anchor": "query", "positive": "relevant doc", "negative": "irrelevant doc"}

# Labeled format (for supervised contrastive)
{"text": "sample text", "label": 0}
```

### Distributed Data Loading

For multi-GPU training, use `DistributedSampler`:

```python
sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,  # Total GPUs
    rank=rank,                 # This GPU's rank
    shuffle=True,
)

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=sampler,
    # shuffle=False when using sampler
)

# IMPORTANT: Call each epoch
for epoch in range(epochs):
    sampler.set_epoch(epoch)  # Ensures different shuffling each epoch
    for batch in loader:
        ...
```

---

## Distributed Training

### Setup Process

```
torchrun --nproc_per_node=4 train.py
         |
         v
+------------------+
| Read ENV vars    |
| RANK, LOCAL_RANK |
| WORLD_SIZE       |
+------------------+
         |
         v
+------------------+
| Init Process     |
| Group (NCCL)     |
+------------------+
         |
         v
+------------------+
| Assign Device    |
| cuda:{local_rank}|
+------------------+
         |
         v
+------------------+
| Barrier Sync     |
+------------------+
```

**Setup Code** (`distributed/setup.py`):
```python
def setup_distributed(cfg):
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    dist.init_process_group(backend="nccl")

    return rank, local_rank, world_size, device
```

### DDP Wrapping

```python
def _apply_ddp(model, cfg, device, process_group=None):
    # Cast to target dtype if mixed precision is enabled
    if cfg.mixed_precision:
        model = model.to(dtype=torch.bfloat16)  # uses cfg.param_dtype
    model = model.to(device)

    ddp_kwargs = {
        "find_unused_parameters": cfg.ddp_find_unused_parameters,
        "gradient_as_bucket_view": cfg.ddp_gradient_as_bucket_view,
    }
    if device.type == "cuda":
        ddp_kwargs["device_ids"] = [device]
    if cfg.ddp_static_graph:
        ddp_kwargs["static_graph"] = True
    if cfg.ddp_bucket_cap_mb:
        ddp_kwargs["bucket_cap_mb"] = cfg.ddp_bucket_cap_mb
    if process_group is not None:
        ddp_kwargs["process_group"] = process_group

    return DDP(model, **ddp_kwargs)
```

### FSDP Wrapping

```python
def _apply_fsdp(model, cfg, device, pretrained_model_name=None):
    # Step 1: Activation checkpointing (before sharding)
    if cfg.activation_checkpointing:
        _apply_activation_checkpointing(model)

    # Step 2: Build FSDP kwargs (reshard_after_forward, etc.)
    fsdp_kwargs = _filter_supported_fully_shard_kwargs({
        "reshard_after_forward": cfg.fsdp_reshard_after_forward,
        "limit_all_gathers": cfg.fsdp_limit_all_gathers,
        "use_orig_params": cfg.fsdp_use_orig_params,
    })

    # Step 3: Configure mixed precision policy
    if cfg.mixed_precision:
        fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,     # from cfg.param_dtype
            reduce_dtype=torch.float32,      # from cfg.reduce_dtype
            output_dtype=torch.bfloat16,     # from cfg.output_dtype
            cast_forward_inputs=cfg.cast_forward_inputs,
        )

    # Step 4: Materialize meta model; init weights if no pretrained source
    if _is_meta_model(model):
        model.to_empty(device=device)
        if not pretrained_model_name:
            model.init_weights()  # or reset_parameters() fallback

    # Step 5: Shard transformer layers + root model
    _shard_layers(model, fsdp_kwargs)  # calls fully_shard per layer
    fully_shard(model, **fsdp_kwargs)

    # Step 6: Load pretrained weights (rank 0 downloads, DCP broadcast)
    if _is_meta_model(model) and pretrained_model_name:
        # rank 0 saves HF state_dict to disk -> barrier ->
        # all ranks load via set_model_state_dict(broadcast_from_rank0=True)
        ...

    # Step 7: Configure forward/backward prefetching if requested
    if cfg.fsdp_forward_prefetch or cfg.fsdp_backward_prefetch:
        set_fsdp_prefetching(model, ...)

    return model
```

### Multi-Node Training

For training across multiple machines:

```bash
# Node 0 (Master)
torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --node_rank=0 \
    --master_addr=192.168.1.1 \
    --master_port=29500 \
    train.py --config config.yaml

# Node 1
torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --node_rank=1 \
    --master_addr=192.168.1.1 \
    --master_port=29500 \
    train.py --config config.yaml
```

### SLURM Integration

**Option 1: Python Launcher (Recommended)**

```bash
# Basic usage
python scripts/launch_slurm.py --config configs/llm_fsdp.yaml --nodes 2 --gpus 4

# With custom options
python scripts/launch_slurm.py \
    --config configs/llm_fsdp.yaml \
    --nodes 4 \
    --gpus 8 \
    --partition gpu \
    --time 24:00:00 \
    --job-name my-training

# Dry run (print script without submitting)
python scripts/launch_slurm.py --config config.yaml --nodes 2 --gpus 4 --dry-run
```

**Option 2: Direct SLURM Script**

```bash
#!/bin/bash
#SBATCH --job-name=omni-train
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4

srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$SLURM_NODELIST:29500 \
    train.py --config config.yaml
```

**Option 3: Use the provided script**

```bash
sbatch scripts/slurm_train.sh configs/llm_full_finetune_fsdp.yaml
```

---

## Training Loop

### Main Training Flow

```python
class Trainer:
    def train(self):
        for epoch in range(self.start_epoch, num_epochs):
            # Set epoch for distributed sampler
            self.train_loader.sampler.set_epoch(epoch)

            # Train one epoch
            epoch_loss = self._train_epoch(epoch)

            # Save checkpoint
            if (epoch + 1) % save_every_n_epochs == 0:
                self.checkpointer.save(model, optimizer, step, epoch)

            # Validate
            val_loss = self._validate(epoch)

            # Save best model
            self.checkpointer.save_best(model, val_loss)

            # Log to W&B
            self.wandb.log_epoch(epoch, epoch_loss, val_loss)
```

### Single Epoch

```python
def _train_epoch(self, epoch):
    for batch_idx, batch in enumerate(self.train_loader):
        # Forward pass
        loss = self._train_step(batch)

        # Backward pass with gradient accumulation
        scaled_loss = loss / grad_accum_steps
        scaled_loss.backward()

        # Optimizer step (after accumulation)
        if (batch_idx + 1) % grad_accum_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            # Update LR
            lr = lr_scheduler(global_step)

            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
```

### Forward Pass by Model Type

```python
def _train_step(self, batch):
    if self.is_llm:
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        return outputs.loss

    elif self.is_cnn:
        images, labels = batch
        logits = self.model(images)
        return F.cross_entropy(logits, labels)

    elif self.is_embedding:
        return self._embedding_train_step(batch)
```

### Checkpointing

```python
class Checkpointer:
    def save(self, model, optimizer, step, epoch, metrics=None):
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
            "metrics": metrics,
        }
        path = self.dir / f"checkpoint_step{step}.pt"
        torch.save(state, path)
        self._cleanup_old_checkpoints()

    def save_best(self, model, metric_value):
        if metric_value < self._best_metric:
            self._best_metric = metric_value
            torch.save(model.state_dict(), self.dir / "best_model.pt")

    def load(self, model, optimizer, path):
        state = torch.load(path)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        return {"step": state["step"], "epoch": state["epoch"]}
```

---

## Web UI

### Architecture

```
Browser
   |
   | HTTP
   v
+------------------+
| FastAPI Server   |
| (ui/app.py)      |
+------------------+
   |
   +---> /api/configs     # Config templates
   +---> /api/train/*     # Training control
   +---> /api/queue/*     # Job queue
   +---> /api/system/*    # GPU info
   +---> /static/*        # Frontend files
```

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve frontend |
| `/api/configs` | GET | List config templates |
| `/api/configs/{name}` | GET | Get specific config |
| `/api/train/start` | POST | Start training |
| `/api/train/status` | GET | Get training status |
| `/api/train/stop` | POST | Stop training |
| `/api/queue/submit` | POST | Submit job to queue |
| `/api/queue/status` | GET | Queue status |
| `/api/system/gpus` | GET | GPU information |

### Job Queue System

The queue system manages multiple training jobs:

```python
class QueueManager:
    def submit_job(self, config, gpu_count, priority):
        job = Job(
            id=generate_id(),
            config=config,
            gpu_count=gpu_count,
            status=JobStatus.PENDING,
        )
        self.queue.append(job)
        return job

    def _worker(self):
        while self.running:
            job = self._get_next_job()
            if job:
                self._run_job(job)
            time.sleep(1)
```

### Starting the UI

From the parent `Local/` directory:

```bash
# Linux/macOS local development
cd Local
source .venv/bin/activate
python -m omni_train.ui.app

# Linux server / remote VM
cd Local
source .venv/bin/activate
uvicorn omni_train.ui.app:app --host 0.0.0.0 --port 8000
```

Use `http://localhost:8000` for local access, or `http://<server-ip>:8000` when the UI is running on a remote Linux machine.

---

## API Reference

### Config Loader

```python
from omni_train.config.loader import load_config

# Load config with optional CLI overrides
cfg = load_config("config.yaml", cli_args)
```

### Model Builders

```python
from omni_train.models.registry import build_model

model = build_model(cfg.model, device)
```

### Data Builders

```python
from omni_train.data.registry import build_dataloaders

train_loader, val_loader = build_dataloaders(
    cfg.data, cfg.model, rank, world_size, batch_size
)
```

### Distributed Setup

```python
from omni_train.distributed.setup import setup_distributed, cleanup_distributed

rank, local_rank, world_size, device = setup_distributed(cfg.distributed)
# ... training ...
cleanup_distributed(cfg.distributed)
```

### Training

```python
from omni_train.training.trainer import Trainer

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    train_loader=train_loader,
    val_loader=val_loader,
    checkpointer=checkpointer,
    cfg=cfg,
    device=device,
    rank=rank,
    world_size=world_size,
)
trainer.train()
```

---

## Troubleshooting

### Common Issues

#### CUDA Out of Memory

**Solutions:**
1. Reduce `batch_size`
2. Enable `activation_checkpointing: true`
3. Use smaller `grad_accum_steps` with larger accumulation
4. Enable quantization (`quantize: true, quant_bits: 4`)
5. Switch to FSDP for large models

#### DDP Hangs on Startup

**Causes:**
- Firewall blocking ports
- Wrong MASTER_ADDR/MASTER_PORT
- GPU mismatch between nodes

**Solutions:**
1. Check firewall: `sudo ufw allow 29500`
2. Verify env vars: `echo $MASTER_ADDR $MASTER_PORT`
3. Ensure all nodes have same GPU count

#### Slow Training

**Causes:**
- DataLoader bottleneck
- Small batch size
- No mixed precision

**Solutions:**
1. Increase `num_workers`
2. Enable `pin_memory: true`
3. Enable `mixed_precision: true`
4. Use `compile_model: true` (PyTorch 2.0+)

#### Gradient NaN/Inf

**Causes:**
- Learning rate too high
- Loss explosion
- Numerical instability

**Solutions:**
1. Reduce `lr`
2. Enable `grad_clip: 1.0`
3. Use `bfloat16` instead of `float16`
4. Add warmup: `warmup_steps: 100`

#### Flash Attention Not Working

**Requirements:**
- PyTorch 2.0+
- CUDA 11.6+
- Ampere GPU or newer (A100, RTX 30xx, RTX 40xx)

**Install:**
```bash
pip install flash-attn --no-build-isolation
```

### Logging & Debugging

Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check distributed environment:
```python
from omni_train.distributed.setup import get_env_info
print(get_env_info())
```

Profile memory usage:
```python
torch.cuda.memory_summary()
```

---

## Best Practices

### Memory Optimization Checklist

- [ ] Enable mixed precision (`mixed_precision: true`)
- [ ] Use gradient accumulation for large effective batch sizes
- [ ] Enable activation checkpointing for large models
- [ ] Use 4-bit quantization for inference-quality fine-tuning
- [ ] Use FSDP for models > 7B parameters

### Training Stability Checklist

- [ ] Use warmup steps (5-10% of total steps)
- [ ] Enable gradient clipping (1.0 is a good default)
- [ ] Start with small learning rate (1e-5 to 5e-5 for LLMs)
- [ ] Monitor loss for NaN/Inf values
- [ ] Use cosine LR schedule for smooth convergence

### Distributed Training Checklist

- [ ] Verify all nodes can communicate (ping test)
- [ ] Use NCCL backend for GPUs
- [ ] Set `set_epoch()` on DistributedSampler each epoch
- [ ] Only save/log from rank 0
- [ ] Use `barrier()` before critical sync points

---

## Appendix: Example Configurations

### LLM LoRA + QLoRA

```yaml
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
  name: tatsu-lab/alpaca
  max_seq_len: 2048

distributed:
  strategy: ddp
  mixed_precision: true

training:
  epochs: 3
  batch_size: 4
  lr: 2e-4
  grad_accum_steps: 8
```

### CNN ImageNet Fine-Tuning

```yaml
model:
  type: cnn
  name: resnet50
  num_classes: 1000
  pretrained: true

data:
  type: image_folder
  name: /path/to/imagenet
  image_size: 224
  augmentation: autoaugment

distributed:
  strategy: ddp
  mixed_precision: true

training:
  epochs: 90
  batch_size: 64
  lr: 0.1
  lr_schedule: cosine
  weight_decay: 1e-4
```

### YOLO Object Detection

```yaml
model:
  type: detection
  name: yolov8m
  yolo_model: yolov8m.pt

data:
  type: yolo
  data_yaml: coco128.yaml
  image_size: 640

training:
  epochs: 100
  batch_size: 16
  lr: 0.01
```

### Text Embedding with InfoNCE Loss

```yaml
model:
  type: embedding
  name: sentence-transformers/all-MiniLM-L6-v2
  embedding_type: text
  embedding_backend: sentence_transformers
  pooling_mode: mean
  loss_type: infonce
  finetune_mode: full

data:
  type: hf_dataset
  name: sentence-transformers/all-nli
  embedding_format: pairs

distributed:
  strategy: ddp
  mixed_precision: true

training:
  epochs: 3
  batch_size: 64
  lr: 2e-5
```

### CLIP Multimodal Fine-tuning

```yaml
model:
  type: embedding
  name: openai/clip-vit-base-patch32
  embedding_type: clip
  loss_type: infonce
  finetune_mode: lora
  lora_r: 8

data:
  type: hf_dataset
  name: your-image-text-dataset
  embedding_format: pairs
  image_size: 224

training:
  epochs: 5
  batch_size: 32
  lr: 1e-4
```

### BERT Embedding with Triplet Loss

```yaml
model:
  type: embedding
  name: bert-base-uncased
  embedding_type: text
  embedding_backend: huggingface
  pooling_mode: cls
  loss_type: triplet
  finetune_mode: lora
  lora_r: 16

data:
  type: local_file
  name: data/triplets.jsonl
  embedding_format: triplet

training:
  epochs: 5
  batch_size: 32
  lr: 2e-5
```

---

## Roadmap: Planned Model Types

The following model types are planned for future releases:

### High Priority

#### Seq2Seq (Encoder-Decoder Models)

```yaml
model:
  type: seq2seq
  name: google/flan-t5-base
  finetune_mode: lora          # full | lora
  quantize: true
```

- **Models**: T5, BART, mT5, Flan-T5, mBART, NLLB
- **Tasks**: Translation, summarization, question answering, text-to-text
- **Data format**: `{"input": "source text", "target": "target text"}`

#### Speech/Audio Models

```yaml
model:
  type: speech
  name: openai/whisper-large-v3
  finetune_mode: lora
  task: transcribe              # transcribe | translate
```

- **Models**: Whisper, Wav2Vec2, HuBERT, WavLM, SpeechT5
- **Tasks**: ASR (transcription), audio classification, speaker identification
- **Data format**: `{"audio": "path/to/audio.wav", "text": "transcription"}`

#### Segmentation Models

```yaml
model:
  type: segmentation
  name: facebook/sam-vit-base
  segmentation_type: instance   # semantic | instance | panoptic
  num_classes: 21
```

- **Models**: SAM, Mask2Former, DeepLabV3, SegFormer, UNet
- **Tasks**: Semantic segmentation, instance segmentation, panoptic segmentation
- **Data format**: COCO panoptic format or mask images

#### Diffusion Models

```yaml
model:
  type: diffusion
  name: stabilityai/stable-diffusion-xl-base-1.0
  finetune_mode: lora
  lora_r: 4
  train_text_encoder: false     # Also train text encoder
```

- **Models**: Stable Diffusion 1.5/2.1/XL, SDXL-Turbo, Flux, Kandinsky
- **Tasks**: Text-to-image generation, image-to-image, inpainting
- **Data format**: `{"image": "path/to/image.png", "caption": "description"}`
- **Popular use**: Custom character/style LoRAs

### Medium Priority

#### Document AI (OCR & Document Understanding)

```yaml
model:
  type: document
  name: microsoft/layoutlmv3-base
  finetune_mode: full
  task: extraction              # extraction | classification | qa
```

- **Models**: LayoutLM, LayoutLMv3, Donut, Pix2Struct, Nougat
- **Tasks**: Document classification, information extraction, document Q&A
- **Data format**: `{"image": "doc.png", "words": [...], "boxes": [...], "labels": [...]}`

#### Time Series Models

```yaml
model:
  type: timeseries
  name: amazon/chronos-t5-base
  prediction_length: 24
  context_length: 512
```

- **Models**: Chronos, TimesFM, PatchTST, Informer, Autoformer
- **Tasks**: Forecasting, anomaly detection, classification
- **Data format**: `{"timestamp": [...], "values": [...], "features": [...]}`

#### Alignment (RLHF/Preference Tuning)

```yaml
model:
  type: alignment
  name: meta-llama/Llama-2-7b-hf
  alignment_method: dpo         # dpo | orpo | ppo | kto
  beta: 0.1                     # KL penalty coefficient
```

- **Methods**: DPO, ORPO, PPO, KTO, IPO
- **Tasks**: Preference alignment, safety tuning, instruction following
- **Data format**: `{"prompt": "...", "chosen": "...", "rejected": "..."}`

#### Video Models

```yaml
model:
  type: video
  name: MCG-NJU/videomae-base
  num_frames: 16
  task: classification          # classification | captioning
```

- **Models**: VideoMAE, TimeSformer, X-CLIP, InternVideo
- **Tasks**: Action recognition, video classification, video captioning
- **Data format**: `{"video": "path/to/video.mp4", "label": 0}`

---

*Documentation generated for OMNI-Train v1.0*
