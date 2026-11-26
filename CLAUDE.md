# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

A modular PyTorch distributed training framework called **OMNI-Train** that supports fine-tuning LLMs, vision, and embedding models across multiple GPUs using FSDP2, DDP, or single-GPU (solo) mode, with LoRA/QLoRA support, distributed checkpointing, SLURM multi-node launch, and a FastAPI browser UI.

---

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Create a `.env` file with:
```
HF_TOKEN=hf_...       # required for gated models (LLaMA, Mistral, Gemma)
WANDB_API_KEY=...     # optional
```

---

## Running Training

```bash
# Reads strategy and num_gpus from config.yaml automatically
# (strategy: auto, the default: measures model, data and GPUs first, and explains the choice)
bash scripts/launch.sh

# What would strategy: auto choose, and why?
python advisor.py --config config.yaml

# Use a different config file
CONFIG_PATH=configs/llm_lora_ddp.yaml bash scripts/launch.sh

# Override strategy and GPU count inline
STRATEGY=ddp NUM_GPUS=2 bash scripts/launch.sh

# Launch directly with torchrun (distributed)
torchrun --nproc_per_node=2 train.py

# Solo mode (single process, no torchrun)
python train.py
```

## Testing

CI (`.github/workflows/tests.yml`, GitHub's CPU runners, offline: `HF_HUB_OFFLINE=1`) runs the unit and `-m e2e` suites on every push and the `-m dist` suite on PRs, pushes to `main` and manual runs. Only the GPU tests need a local machine: `-m smoke` and the FSDP cases of `test_post_training_dist.py` (≥ 2 GPUs). Tests in the CI suites must not download anything: build tiny models from configs (see `tests/post_training_helpers.py`).

On a rented GPU machine, `bash scripts/gpu_check.sh [--setup] [--full]` runs everything CI can't: an NCCL check (`scripts/nccl_check.py`), `-m dist` on GPUs over NCCL (`OMNI_DIST_DEVICE=cuda`, see `tests/dist_device.py`; cases needing more GPUs than present are skipped), and the curated smoke set (multi-GPU `min_gpus` cases, post-training, save/resume, QLoRA, vision/detection/VLM). Logs go to `gpu_check_logs/<ts>/`.

```bash
# Unit tests (no GPU required) — default
python -m pytest

# Run a single test file
python -m pytest tests/test_config_validation.py -v

# Run a single test by name
python -m pytest tests/test_config_validation.py::TestFoo::test_bar -v

# GPU smoke tests (require GPU, ~30-60s each)
python -m pytest -m smoke

# multi-process equivalence: strategy 5d layouts + post-training under DDP (CPU/gloo, ~5 min)
python -m pytest -m dist

# end to end: the real train.py as a subprocess on CPU (tiny local Llama + .jsonl, solo and
# 2-process DDP): loss decreases, DPO starts at ln 2, resume restores AdamW, HF export loads (~40 s)
python -m pytest -m e2e
```

## Web UI

```bash
bash ui/launch_ui.sh       # http://127.0.0.1:8787

# For remote server
uvicorn ui.app:app --host 0.0.0.0 --port 8787
```

## SLURM / Multi-Node

```bash
sbatch scripts/slurm_train.sh configs/llm_full_finetune_fsdp.yaml

python scripts/launch_slurm.py \
    --config configs/llm_fsdp.yaml \
    --nodes 4 --gpus 8

# Dry run (print generated script, don't submit)
python scripts/launch_slurm.py --config configs/llm_fsdp.yaml --nodes 2 --dry-run
```

---

## Architecture

### Configuration Flow

```
config.yaml (or CONFIG_PATH env var)
    │
    ▼
advisor.resolve_*()                        # only strategy: auto → a concrete strategy, num_gpus, parallelism
    │
    ▼
utils.load_config() → utils.build_args()   # parses + validates YAML into Args dataclass
    │
    ▼
train.py main()
    ├── strategy: 5d → parallelism.trainer.train_5d(args)   # own model + loop, returns
    ├── algorithm: sft|dpo|simpo|orpo|grpo|ppo → post_training data + objective (same model setup/loop)
    ├── distributed_utils.apply_solo/ddp/fsdp()   # model setup
    ├── data.get_dataloader()                       # HF dataset + DistributedSampler
    └── checkpoint.Checkpointer                    # FSDP save/load
```

`scripts/launch.sh` reads `strategy` and `num_gpus` from the config and decides whether to run `python train.py` (solo) or `torchrun --nproc_per_node=N train.py` (ddp/fsdp/5d).

`torchrun` injects `RANK`, `LOCAL_RANK`, and `WORLD_SIZE` env vars before Python starts.

### Training Strategies

| Strategy | When to use | Notes |
|----------|-------------|-------|
| `auto`   | Default     | `advisor.py` fills in one of the rows below before training starts; see "strategy: auto" |
| `solo`   | Single GPU  | No torchrun, no process group |
| `ddp`    | Model fits in one GPU, scale throughput | Every GPU holds full model copy; gradient sync via all-reduce |
| `fsdp`   | Large models, limited GPU memory | Parameters/gradients/optimizer states sharded across GPUs |
| `5d`     | Very large models, long sequences, MoE, multi-node | Any mix of DP(+ZeRO-1), TP(+SP), CP, PP, EP from the `parallelism:` block; see below |

**Hard constraint:** `bitsandbytes` quantization (4-bit/8-bit) is **incompatible with FSDP** — the sharding mechanism cannot work on quantized layers. Use `strategy: ddp` or `strategy: solo` when quantization is enabled.

### FSDP2 Meta-Device Initialization (Path 3, no PEFT/quant)

FSDP builds the model with its **parameters** on meta first (`accelerate.init_empty_weights(include_buffers=False)` — no real memory allocated; buffers are still computed for real), shards layers with `fully_shard()`, then populates weights via one of three paths:
- **A — Resume:** `Checkpointer.load_model()` reads latest checkpoint
- **B — Fresh HF:** rank 0 downloads model, saves seed `.pt` to disk, barrier, all ranks load and shard
- **C — Random init:** `model.to_empty(device=device)` then `init_weights()`

Path B is why a `checkpoints/pretrained_seed/<api>/<timestamp>/` folder appears on fresh runs. Each run loads its own timestamped seed (never "the latest", which could belong to a concurrent run) and deletes only that folder as soon as every rank has loaded it.

### FSDP + PEFT/Quantization (Path 2)

When `peft.enabled=true` or `quantization.enabled=true`, the model is **materialized before wrapping** (cannot use meta device). Rank 0 loads the full HF model, saves it as a seed `.pt`, all ranks load it CPU-side, PEFT adapters are attached, then `fully_shard()` is applied to the PEFT-wrapped model.

### Checkpoint Layout

```
checkpoints/
├── solo/<unix_ms_timestamp>[__lora_q4].pt     # {"model_state_dict", "optimizer_state_dict"}
├── ddp/<unix_ms_timestamp>[__lora_q4].pt      # same (rank 0's copy)
├── hf/<unix_ms_timestamp>[__lora]/            # save_load.export_hf: true (export.py)
│   ├── config.json, model.safetensors, tokenizer files, omni_train.json
│   └── adapter/                               # QLoRA/8-bit only: adapter, not merged
└── fsdp/
    ├── dcp_api/
    │   └── <unix_ms_timestamp>[__lora_q4]/
    │       ├── model_state_dict.pt
    │       └── optim_state_dict.pt
    └── dtensor_api/
        └── <unix_ms_timestamp>/
            ├── model_state_dict.pt
            └── optim_state_dict.pt
```

Checkpoint folders are named with Unix millisecond timestamps. PEFT/quantization runs append a tag (e.g., `__lora_q4`). `get_latest_checkpoint_folder()` picks the numerically largest timestamp prefix.

**Resume (solo/DDP):** `distributed_utils.restore_optimizer_state` reloads the AdamW moments and resets each group's `lr`/`initial_lr`/`weight_decay` to the new config, because the LR schedule restarts on resume. Older weights-only `.pt` files still load (AdamW starts fresh, with a notice). FSDP restores via `Checkpointer.load_optim`.

**HF export** (`export.py`, `save_load.export_hf: true`, `export_dtype` bfloat16 | float16 | float32): runs after training on every rank. Solo/DDP merge LoRA in place on the finished model. FSDP all-gathers the full state dict to rank 0 (`get_model_state_dict(full_state_dict, cpu_offload)`), which rebuilds the model on CPU (`rebuild_from_state_dict`) and merges. Quantized runs export only `adapter/`; `scripts/merge_lora.py` merges it into the full-precision base. Not for 5d (it has its own `<ts>/hf/` export) or `custom_transformer`.

Two FSDP checkpoint APIs:
- **DCP API** (`distribute_api: dcp_api`): uses `torch.distributed.checkpoint.state_dict` — rank 0 broadcasts state, simpler
- **DTensor API** (`distribute_api: dtensor_api`): manual `distribute_tensor` / `full_tensor()` gathering

### Key Constraints Enforced in `build_args`

- `peft.type: qlora` → forces `quantization.enabled=true`; supports `bits=4` (NF4, the default) or `bits=8` (LLM.int8())
- 4-bit quantization requires `peft.enabled=true` — cannot train 4-bit weights directly (NF4 has no differentiable backward)
- 8-bit quantization without PEFT → `UserWarning`, not an error. Only the non-quantized float params (embeddings, norms, biases) can learn, and NaN losses are likely. Use 8-bit LoRA instead.
- `strategy: fsdp` + `quantization.enabled: true` → hard error
- `strategy: fsdp` + `mixed_precision` with `output_dtype` ≠ a low-precision `param_dtype` → error for llm/encoder/vision/custom_transformer (final norm/head would get mixed dtypes; `cast_forward_inputs` does not help), warning otherwise (`_check_fsdp_dtypes`)
- `training.lr_scheduler` must be `cosine` (default), `linear` or `constant` — applied after the linear warmup
- `peft.enabled=true` only supports model types: `llm, seq2seq, encoder, vlm, vision`
- `strategy: auto` passes `build_args` (for validation only; train.py never runs it). `num_gpus: auto` needs `strategy: auto`
- `algorithm:` (`post_training.config.parse_algorithm`): `type` ∈ pretrain (default) | sft | dpo | simpo | orpo | grpo | ppo; unknown keys are errors, RL-, GRPO- and PPO-only keys on other types too; grpo: `num_generations` ≥ 2; ppo: `num_minibatches` divides `batch_size × num_generations`; both: `beta > 0` needs `peft.enabled`, `max_prompt_length + max_completion_length` ≤ `max_length`, rewards must be built-in names or `module:function`; non-pretrain needs `model_type: llm` and strategy solo | ddp | fsdp; `dpo_loss` ∈ sigmoid | ipo | hinge; `label_smoothing` only for dpo+sigmoid; `max_prompt_length` < `max_length`. `dataset.subset` is optional
- `strategy: 5d` (`_parse_5d`): `dp*tp*cp*pp == num_gpus` (`dp: auto` infers it), `ep` divides dp and needs `custom_transformer_args.num_experts`, `sp` needs tp>1, sequence length divisible by cp (by cp*tp with sp), only `model_type: llm | custom_transformer`, no PEFT/quantization, unknown `parallelism` keys are errors. Model-shape checks (heads/kv heads/ffn divisible by tp, layers ≥ pp) run at startup in `parallelism.model.check_divisibility`

### Model Types

| `model_type` | HF class used | Status |
|---|---|---|
| `llm` | `AutoModelForCausalLM` | supported: the product focus (post-training, export, e2e tests) |
| `seq2seq` | `AutoModelForSeq2SeqLM` | experimental |
| `vision` | `AutoModelForImageClassification` | experimental |
| `yolo` | `AutoModelForObjectDetection` | experimental |
| `vlm` | `AutoModelForImageTextToText` | experimental |
| `encoder` | `AutoModel` | experimental |
| `custom_transformer` | `model.Transformer` (toy model built from scratch, uses synthetic data) | demo/testing |

Experimental types print a one-line notice at startup (`train.py`, `EXPERIMENTAL_MODEL_TYPES`). New features target `llm` first; `strategy: 5d` is an advanced mode, not the default recommendation.

### Data Pipeline by Model Type (`data.py`)

`dataset.name` is a Hub dataset or a local `.json/.jsonl/.csv/.parquet/.txt` file (`data.load_any_dataset`, also used by post-training); `dataset.text_field` renames the pretrain text column to `text`. `get_dataloader()` branches on `model_type` to decide how raw dataset rows become batches. Each type expects a specific dataset schema:

| `model_type` | Dataset columns expected | How batches are built | Train-loop call |
|---|---|---|---|
| `llm` / `seq2seq` / `encoder` | `text` | `dataset.map(tokenize)` → `input_ids`/`labels` | `model(input_ids, attention_mask, labels)` |
| `vision` | image (`img`/`image`) + `label`/`labels` | `dataset.map(process_images)` → `pixel_values` + int label | `model(pixel_values, labels)` |
| `yolo` (detection) | `image` + `objects` (`bbox` COCO `[x,y,w,h]`, `category`, optional `area`/`id`/`iscrowd`) + `image_id` | `_detection_collate` runs the image processor with COCO `annotations` → `pixel_values` + `labels` as a **list of per-image dicts** (`class_labels`, normalized `boxes`) | `model(pixel_values, labels=[{...}], pixel_mask?)` |
| `vlm` | image (`image`/`img`) + caption (`text`/`caption`/`captions`/`sentence`) | `_vlm_collate` frames each row via the processor chat template, expands `<image>` tokens, pads, and masks pad + image tokens to `-100` | `model(input_ids, attention_mask, pixel_values, pixel_attention_mask, labels)` |
| `custom_transformer` | — (synthetic) | random token tensors | next-token loss |

Detection and VLM use a `collate_fn` (not `dataset.map`) because their per-row outputs are variable-length structures that can't be default-collated into a tensor. Those paths keep the dataset raw (PIL images preserved) and run `num_workers` collation under fork. `load_dataset` is called with `token=HF_TOKEN` so gated/rate-limited datasets resolve instead of hanging on unauthenticated Hub requests.

For VLM, `train.py` loads an `AutoProcessor` (not a bare tokenizer) and disables image splitting/tiling (`do_image_splitting=False`) on Idefics3/SmolVLM-style processors so one image maps to a bounded number of tokens rather than blowing the sequence length up into the thousands.

### Layer Detection for FSDP Sharding

`get_model_layers(model)` in `distributed_utils.py` probes common HF architecture patterns (`model.decoder.layers`, `model.encoder.layers`, `transformer.h`, `model.layers`) to find transformer blocks for per-layer sharding. Unknown architectures fall back to root-only sharding.

### Non-Persistent Buffers (RoPE `inv_freq`, `position_ids`)

`model.to_empty()` replaces every tensor with uninitialized memory, and non-persistent buffers (RoPE `inv_freq`/`original_inv_freq`, BERT `position_ids`, …) are excluded from `state_dict()`, so no checkpoint load refills them. The meta-init path therefore builds with buffers computed for real, `_snapshot_buffers()` copies them before `to_empty()`, and `_restore_buffers()` puts the non-persistent ones back after loading. Don't recompute them by formula: `rope_theta` and rope scaling (Llama 3, Qwen, Mistral) make the textbook `1 / 10000^(2i/dim)` wrong.

### Precision

Trainable weights are always fp32 (`_load_dtype()`), because bf16 rounds most fine-tuning-sized AdamW updates to zero. `mixed_precision: true` casts at compute time instead: FSDP via `MixedPrecisionPolicy`, solo/DDP via `torch.autocast` in `train.py` (plus `GradScaler` for float16). The only low-precision storage is a frozen PEFT base outside FSDP; PEFT keeps the adapters in fp32. FSDP2 needs one dtype per parameter group, so FSDP + PEFT stays fp32 throughout.

### Capacity planner (`planner.py`)

`plan(ModelSpec, Workload, Cluster)` answers "is distributed training needed?": per-GPU peak memory (weights / grads / AdamW / activations / logits / overhead) for solo, DDP, FSDP and every valid 5d layout, a verdict (`single_gpu | distributed_optional | distributed_required | infeasible`), the recommended option, min GPUs per strategy and what-if fixes. Pure Python (no torch). The memory model mirrors the Precision rules above (16 B/param trainable, bf16 frozen LoRA base outside FSDP, fp32 under FSDP, preference runs = 2 × batch rows, logits upcast to fp32 dominate for big vocabularies, no vocab-parallel CE under 5d). GRPO rows = batch × G; its peak is max(sampling KV cache, training) and its step time adds bandwidth-bound decoding (`GPU_CATALOG` carries GB/s). It only proposes what the trainer accepts: QLoRA never on FSDP; 5d only for full-FT pretrain on llama/mistral/qwen2/custom; 5d layouts obey `_parse_5d`. `ModelSpec` comes from `PRESETS` (offline), an HF `config.json`, `custom_transformer_args`, or a name heuristic. CLI: `python planner.py --model … --gpus 8 --gpu h100`. Tests: `tests/test_planner.py`.

`plan()` tries, in order (`_pick`): one GPU if fast enough (`TARGET_HOURS`, only when row count and GPU speed are known), DDP, FSDP, the same with `_gentle_changes` (checkpointing on, batch halved), then 5d. It never changes the training method; LoRA and QLoRA appear under `alternatives`. `recommended.changes` must be applied with the option. `explain()` / `advice` is the plain-words version used by the CLI, the run log and the console. Unknown GPU types need `--gpu-mem`. Anything the trainer refuses (local files / `text_field` / `export_hf` / fp16 under 5d, MoE outside 5d) belongs in `workload_from_config` (`no_5d`, `needs_5d`), or `auto` will write a config that fails.

### strategy: auto (`advisor.py`)

Supplies what the planner cannot know (GPUs, config.json, dataset) and fills in `strategy`, `num_gpus`, `parallelism`, `gradient_checkpointing`, `batch_size`. Resolved by `scripts/launch.sh` (writes a temporary config, also for `STRATEGY=`/`NUM_GPUS=` overrides), `ui/app.py` on submit (`_resolve_auto`), or `train.py` (`resolve_for_process`). Under torchrun rank 0 decides and shares it through the rendezvous store: ranks that measured for themselves could disagree and hang. `measure_data` tokenises 500 rows plus the longest by text length, with the trainer's own functions (`_row_lengths`: add new algorithms there); large Hub datasets are streamed. A failed measurement never stops a run (`try_measure`). `preflight` only warns. Console: `POST /api/data/inspect` fills `_DATA_CACHE`, which `/api/plan` and submit read. Tests: `tests/test_advisor.py`, `-m e2e`.

### Web UI (`ui/`)

FastAPI server (`ui/app.py`, REST docs at `/docs`) plus a single-page console in `ui/static/` (`index.html`, `app.js`, `charts.js` = charts, memory bars, log viewer, failure hints, `styles.css`, `fonts/` bundled so the console works offline: no asset may come from another host; hash routes `#/runs`, `#/run/<id>`, `#/compare/<id>,<id>` (max 3: the colours are validated for 3), `#/new`, `#/planner`, `#/checkpoints`, `#/system`, `#/api`). Requests must carry a Host header that is `localhost`, an IP address, this machine's hostname, or a name in `OMNI_UI_ALLOWED_HOSTS` (blocks DNS rebinding). Every torchrun launch, manual or queued, gets a free `--master_port`, and ddp/fsdp/5d jobs always go through torchrun, even on 1 GPU.

- The console builds configs in the **`config.yaml` schema** directly (algorithm, parallelism, custom_transformer_args included); `ui/config_adapter.py` deep-merges them over the root `config.yaml` (it still also accepts the old UI schema).
- Planner page = a capacity map (`POST /api/plan/grid`): rows = GPU counts (powers of two inside one machine, then 1/2/4/8 machines), columns = methods or sequence lengths, every cell is a full `plan()` (~0.1 s for the whole grid). It must not look like the wizard's plan panel: the wizard plans one run, the map compares many.
- Endpoints: `POST /api/plan` (planner + evaluation of the config's own strategy as `selected`), `GET /api/catalog`, `GET /api/models/inspect`, `POST /api/config/{validate,render,parse}`, `GET /api/env/status` (booleans only), `/api/queue/*`, `/api/checkpoints` (includes `5d/`), `/api/system/gpus`. `/api/train/start` is the older direct path (no queue).
- `ui/queue.py`: SQLite job queue (`ui/queue.db`) with priority scheduling and run names. `queue/submit` validates with `build_args` and rejects jobs asking for more GPUs than the machine has. Logs are also written to `ui/runs/<job_id>.log`; `parse_metrics` turns trainer step lines into the loss curve / progress.
- **Where a run executes** (`ui/launchers.py`, wizard "Where it runs"): `local` (default, the queue), `remote` (other machines: `launch.sh` on one, `torchrun --rdzv_backend c10d` on several; commands only) or `slurm`. SLURM is never assumed: `slurm_status()` asks `sinfo`, because `sbatch` can be installed with no controller. Each SLURM submission writes its own config under `ui/runs/slurm/` (the job reads it when it starts). Options that become `#SBATCH` lines are pattern-checked. `POST /api/launch/commands`, `GET /api/slurm/status`, `POST /api/slurm/submit`. FSDP on several machines needs a shared `checkpoint_dir` (rank 0 writes the seed). Tests: `tests/test_launchers.py`.
- Saved configs enter the wizard only through `loadConfig()` (a plain merge over `defaultConfig()` resurrects rewards and system prompts the config left out). Async pages check `stillHere()` after every await. `GET /api/configs` also returns `templates` described from the YAML; `/api/queue/jobs` rows carry a `summary`; `/api/queue/jobs/{id}/log` is the full log.
- UI tests must not use `with TestClient(...)` (the startup hook would open the real `ui/queue.db`); use a `QueueManager(db_path=tmp)` monkeypatched into `ui.app.queue_manager`.

### Post-training (`algorithm:` block, `post_training/`)

SFT and preference optimisation reuse the normal solo/DDP/FSDP setup (LoRA/QLoRA included) and the normal loop. `train.py` swaps in `post_training.data.get_post_training_data` for the dataloader and calls `objective(model, batch, device) -> (loss, metrics)` instead of the model-type branch.

- **Data** (`post_training/data.py`): SFT takes `messages` | `prompt`+`completion` | Alpaca `instruction`(+`input`)+`output` | `text`. Preference takes `prompt`+`chosen`+`rejected`, chosen/rejected conversations (last turn is the answer), or hh-rlhf strings. With a chat template the prompt is rendered with `add_generation_prompt=True` and must be a string prefix of the full conversation. Prompt and answer are tokenised separately so the mask boundary is exact. Without a template: BOS + prompt + answer + EOS. Preference batches are `[2B, T]` (chosen rows then rejected) so one forward covers both, which FSDP needs. Local `.json/.jsonl/.csv/.parquet` files work as `dataset.name`.
- **DPO reference:** the policy at step 0 is the reference, so `PreferenceObjective.precompute_reference` scores every pair once (eval mode, no grad, all ranks, all-gathered) and caches `[N, 2]` floats. No second model is kept, for every strategy.
- **GRPO** (`post_training/grpo.py`, `rewards.py`): the objective does the rollout inside the step: G samples per prompt (`generate`, eval mode, no grad; `synced_gpus` under FSDP), rewards on the decoded text, group-normalised advantages, then one forward with grad and `losses.grpo_loss` (clipped, `loss_type` grpo | dapo | dr_grpo; dapo all-reduces the global token count). `training.batch_size` = prompts per GPU. KL reference = LoRA adapters disabled (`attach_reference`). Keep: `position_ids` from the left-padded mask (absolute-position models), logits ÷ temperature, dropout off, and `torch.clear_autocast_cache()` after the no-grad work (train.py's autocast cache otherwise hands the training forward detached bf16 LoRA weights: no gradient / CheckpointError). The `Rollout` dataclass + `build_rollout` let the dist test feed fixed completions.
- **PPO** (`post_training/ppo.py`, subclass of `GRPOObjective`, shares `sample()`): `wrap_loader` replaces the dataloader with one that does a rollout per batch and yields `ppo_epochs × num_minibatches` minibatches, so each gradient step is a normal train.py step (LR schedule and step counts included). Values come from a forward pre-hook on `lm_head` (inside the model's forward, which FSDP needs) into `value_head`, one fp32 linear layer that is NOT part of the wrapped model: own AdamW, gradients averaged by hand in `step_extra` (called between `scaler.step` and `scaler.update`), saved by `save_extra` as `<checkpoint>.value_head` / `<folder>/value_head.pt`. So checkpoints, resume and HF export of the policy are unchanged. Advantages are normalised over the global batch and losses divide by global tokens ÷ world. `beta` = KL penalty in the per-token reward (LoRA adapters off), needs LoRA.
- **Reward models** (`post_training/reward_model.py`, `algorithm.reward_model`, GRPO and PPO): any 1-output sequence classifier, frozen, whole on every GPU; a checkpoint without its scoring head is refused. `rewards` defaults to `accuracy` only when no reward model is given.
- **Sampling** (`GRPOObjective.sample`): runs under `_blank_generation_config`, because `generate()` fills unset values from the model's own generation_config (Qwen: repetition_penalty 1.1, top_k 20) and the answers would not come from the policy being scored. Pass every sampling value explicitly (`top_k=0`, not None).
- **Optimizer order:** build the optimizer before any no-grad forward. After one, FSDP's root still holds full copies of its weights, and an optimizer built then updates those instead of the shards (bit a test worker; train.py is fine).
- **Losses** (`post_training/losses.py`) return per-example values and are averaged over pairs, so DDP/FSDP gradients equal one process on the global batch. SFT uses HF's per-rank token mean (the standard convention, not world-size invariant). `sequence_logps` goes one row at a time in fp32 to avoid a `[B, T, V]` fp32 log-softmax.
- **Testing rule:** `tests/test_post_training_dist.py -m dist` compares losses AND every parameter's gradient to 1 process. DDP runs on CPU. FSDP cases need ≥ 2 GPUs (FSDP2 has no CPU support in torch 2.6) and are skipped otherwise.
- **Not supported yet:** KTO, reward-model training, packing, 5d; a separate critic model; GRPO `num_iterations > 1` and full-FT GRPO with `beta > 0`.

### 5D Parallelism (`strategy: 5d`, `parallelism/`)

`train.py` hands 5d runs to `parallelism.trainer.FiveDRun`, which has its own Llama-style model (`parallelism/model.py`), not the HF/`model.py` classes. Plain `torch.distributed`; no Megatron/DeepSpeed.

- **Rank grid** `[dp, pp, cp, tp]`, TP fastest (neighbours → NVLink). EP reuses the DP axis (`dp_rank = chunk*ep + ep_rank`). `ParallelContext` builds every group; all ranks must create all groups in the same order.
- **Models:** `custom_transformer` (seeded weights, optional MoE/GQA) or HF `llm` of type llama/mistral/qwen2 (`parallelism/hf.py`). HF weights are read slice by slice from safetensors (no full-model load); RoPE `inv_freq` comes from HF's own `*RotaryEmbedding` class (rope scaling handled). Other HF types raise. Vocab is padded to a multiple of tp; padded logits are sliced off.
- **Gradient sync** (`data_parallel.py`): dense grads averaged over DP×CP; expert grads summed over the ranks holding the same experts and divided by **dp×cp**, not by that group's size (each expert already got the summed gradient of the EP ranks it served); non-TP weights summed over TP when SP is on; tied embeddings split by PP summed over `embed_tie_group`.
- **Grad clipping** (`optim.global_grad_norm`) counts each element once: TP shards summed over TP, experts over EP, stages over PP, the tied LM-head copy skipped.
- **Data** (`parallelism/data.py`): fixed `[N, max_length+1]` token blocks, packed HF text (`text` column) or `dataset.name: synthetic`. Sample order depends only on (seed, epoch, step), never on the layout. Global batch = `batch_size × micro_batches × dp`.
- **Precision:** fp32 weights + optimizer, bf16 autocast when `mixed_precision: true` (fp16 rejected). Activations between PP stages are fp32.
- **Checkpoints:** `checkpoints/5d/<ts>/rank_XXXXX.pt` + `layout.json`; resume needs the same layout. LLM runs also export a normal HF checkpoint to `<ts>/hf/`, which is how to change layouts.
- **Testing rule:** `tests/test_5d_parallel.py -m dist` compares losses, the global grad norm **and every parameter's gradient** to a 1-process run. Keep the gradient comparison: AdamW cancels per-tensor gradient scale, so a loss-only check passed an `ep`× expert-gradient bug. Default-suite tests check HF logit parity and export round trip.
- **Not supported yet:** PEFT/quantization, the web UI, vocab-parallel cross-entropy, zig-zag ring attention, interleaved PP, MoE HF checkpoints.

## Example Configs (`configs/`)

Pre-built configs cover common scenarios. Use them as starting points:

| File | Use case |
|------|----------|
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
| `llm_sft_lora_single_gpu.yaml` | SFT + LoRA on chat data, Qwen2.5-0.5B-Instruct, 6 GB GPU |
| `llm_dpo_lora_single_gpu.yaml` | DPO + LoRA on UltraFeedback pairs, Qwen2.5-0.5B-Instruct, 6 GB GPU |
| `llm_grpo_lora_single_gpu.yaml` | GRPO + LoRA (KL via disabled adapters) on GSM8K with the accuracy reward, Qwen2.5-0.5B-Instruct, 6 GB GPU |
| `llm_ppo_lora_single_gpu.yaml` | PPO + LoRA (value head, KL via disabled adapters) on GSM8K, Qwen2.5-0.5B-Instruct, 6 GB GPU |

---

## Logging

- Rank 0 prints all progress banners and per-step loss bars
- Other ranks print only via `print_on_all_ranks()` (with `[host|rank|local_rank|device]` prefix)
- W&B logging is off by default; enable with `wandb.wandb_log_with_train: true`
- After training completes in a terminal (not UI), a loss curve is plotted inline using `plotext`
