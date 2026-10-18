# dist_kit — portable distributed training

A self-contained folder extracted from [OMNI-Train](https://github.com/rachadlakis/omni-train)
that adds solo / DDP / FSDP2 training (with optional LoRA/QLoRA) to **any**
PyTorch + HuggingFace project. Copy the folder in, keep your own data pipeline
and training loop.

```
dist_kit/
├── __init__.py               # public API
├── __main__.py               # `python -m dist_kit` runs training
├── config.py                 # DistConfig — slim, validated config dataclass
├── dist_setup.py             # process-group init/teardown, barriers, rank-aware printing
├── strategies.py             # apply_solo / apply_ddp / apply_fsdp (+ PEFT/quant) + save_checkpoint
├── checkpoint.py             # FSDP2 Checkpointer (DCP + DTensor APIs)
├── data.py                   # HF text dataloader (llm/seq2seq/encoder) with DistributedSampler
├── train.py                  # training entry point (HF dataset, or random batches if none configured)
├── launch.sh                 # solo-vs-torchrun dispatcher
├── config.yaml               # kit config, same schema as OMNI-Train's config.yaml
├── .env.example              # HF_TOKEN etc. (copy to project root as .env)
├── requirements.txt          # kit-only dependencies (torch installed separately)
└── tests/                    # pytest suite (unit + integration + smoke)
```

## Quick start

```bash
# 1. Copy the folder into your project root
cp -r dist_kit /path/to/your-project/

cd /path/to/your-project

# 2. Install deps (torch first, matching your CUDA)
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r dist_kit/requirements.txt

# 3. Configure
cp dist_kit/.env.example .env    # add HF_TOKEN if needed
# Edit dist_kit/config.yaml (model/strategy/gpus), or put a config.yaml in your
# project root — it takes precedence and OMNI-Train's own config.yaml works as-is.
# Resolution: $DIST_CONFIG > ./config.yaml > dist_kit/config.yaml

# 4. Smoke-test with the built-in demo (random token batches)
bash dist_kit/launch.sh    # or: python -m dist_kit (solo, no torchrun)
```

## Using it in your own training loop

```python
import torch
import torch.distributed as dist
from dist_kit import (
    DistConfig, setup_dist_process_group, cleanup, dist_barrier,
    apply_solo, apply_ddp, apply_fsdp, save_checkpoint,
)

args = DistConfig.from_yaml()   # $DIST_CONFIG > ./config.yaml > dist_kit/config.yaml

# 1. Process group (torchrun injects RANK/LOCAL_RANK/WORLD_SIZE)
if args.strategy in ("ddp", "fsdp"):
    local_rank = setup_dist_process_group()
    rank = dist.get_rank()
    device = torch.device(f"cuda:{local_rank}")
else:
    local_rank, rank = 0, 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 2. Wrapped model (FSDP also returns a Checkpointer for save/resume)
checkpointer = None
if args.strategy == "fsdp":
    model, checkpointer = apply_fsdp(local_rank, rank, device, args)
elif args.strategy == "ddp":
    model = apply_ddp(local_rank, rank, device, args)
else:
    model = apply_solo(device, rank, args)

# 3. Your loop — only optimize trainable params (matters for PEFT)
trainable = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable, lr=1e-4)

for batch in your_dataloader:           # use DistributedSampler for ddp/fsdp
    optimizer.zero_grad()
    loss = model(**batch).loss
    loss.backward()
    optimizer.step()

# 4. Save + teardown
save_checkpoint(args.strategy, model, optimizer, rank, args, checkpointer=checkpointer)
cleanup()
```

To resume an FSDP run, set `resume: true` and `resume_path` to a checkpoint
folder (e.g. `checkpoints/fsdp/dcp_api/1720000000000`), then restore optimizer
state with `checkpointer.load_optim(model, optimizer)` after building the optimizer.

## Testing

```bash
# Unit + integration tests (no GPU; integration auto-skips when the HF cache is empty)
python -m pytest dist_kit/tests

# End-to-end smoke tests (subprocess training runs: solo, ddp, fsdp — fsdp needs a GPU)
python -m pytest dist_kit/tests -m smoke
```

## What it does for you

- **Process group setup** with NCCL warm-up barrier and automatic NVLink
  detection (falls back to SHM transport on PCIe-only boxes where NCCL's P2P
  probe hangs).
- **FSDP2** (`fully_shard`) with meta-device init — the full model never
  materializes on one GPU. Per-layer sharding via architecture detection that
  covers Llama/Mistral/Qwen/GPT-2/BERT/ViT/LLaVA families, with a
  largest-ModuleList fallback for unknown models.
- **Mixed precision** policy (bf16 params / fp32 reduce by default) with a
  fp16 fallback when bf16 is unsupported.
- **LoRA / QLoRA** — adapters attach before sharding; quantized bases are
  prepared with `prepare_model_for_kbit_training`.
- **Distributed checkpointing** — DCP API (rank-0 broadcast) or DTensor API
  (manual gather); resume validates PEFT/non-PEFT compatibility and restores
  non-persistent buffers (RoPE `inv_freq`, `position_ids`) that `to_empty()`
  wipes, using the values the HF constructor computed (zeroing or re-deriving
  them would silently corrupt positional encodings).

## Hard constraints (enforced by DistConfig)

- Quantization (4-bit or 8-bit) **requires** `peft_enabled: true` — quantized
  base weights cannot be trained directly.
- Quantization is **incompatible with FSDP** — use `ddp` or `solo`.
- `resume: true` requires `resume_path`.

## What was intentionally left out

Compared to OMNI-Train: the vision / detection / VLM data pipelines (only the
text path is included — bring your own DataLoader for those), the toy
`custom_transformer`, W&B logging, the web UI, SLURM scripts, and the
VRAM/time estimators. Those are app-level; this kit is the distributed
machinery plus enough to train a text model end to end.
