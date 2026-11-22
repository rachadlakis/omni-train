# 5D Parallelism (`strategy: 5d`)

`strategy: 5d` splits training over GPUs in up to five ways at once, chosen in YAML.
It follows the ideas in Hugging Face's *Ultra-Scale Playbook* and uses plain
`torch.distributed` (no Megatron or DeepSpeed).

| Dimension | YAML key | What it splits | Code |
|---|---|---|---|
| **DP** data parallel (+ bucketing, overlap, no_sync) | `dp` | the batch | `parallelism/data_parallel.py` |
| **ZeRO-1** | `zero1` | AdamW states, over DP | `parallelism/zero.py` |
| **TP** tensor parallel (+ **SP** sequence parallel) | `tp`, `sp` | attention heads / MLP width | `parallelism/tensor_parallel.py` |
| **CP** context parallel (ring attention) | `cp` | the sequence | `parallelism/context_parallel.py` |
| **PP** pipeline parallel (AFAB, 1F1B) | `pp`, `pp_schedule` | the layers | `parallelism/pipeline_parallel.py` |
| **EP** expert parallel (MoE, all-to-all) | `ep` | the experts | `parallelism/expert_parallel.py` |

## Quick start

```yaml
strategy: 5d
num_gpus: 8
model_name: meta-llama/Llama-3.1-8B
model_type: llm
parallelism:
  dp: auto           # num_gpus / (tp * cp * pp) = 2
  tp: 2
  sp: true
  cp: 1
  pp: 2
  pp_schedule: 1f1b
  micro_batches: 8   # global batch = training.batch_size * micro_batches * dp
  ep: 1
  zero1: true
```

```bash
CONFIG_PATH=configs/llm_5d_tp_pp_dp_8gpu.yaml bash scripts/launch.sh   # one node
python scripts/launch_slurm.py --config configs/custom_5d_moe_16gpu.yaml --nodes 2 --gpus 8
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 CONFIG_PATH=configs/custom_5d_cpu_demo.yaml bash scripts/launch.sh  # laptop, CPU
```

Ready-made layouts: `configs/llm_5d_tp_pp_dp_8gpu.yaml`, `configs/llm_5d_long_context_cp.yaml`,
`configs/custom_5d_moe_16gpu.yaml` (all five dimensions), `configs/custom_5d_cpu_demo.yaml`.

## What it trains

- **`model_type: llm`** — Hugging Face **Llama, Mistral and Qwen2** checkpoints (TinyLlama, Llama 3.x,
  SmolLM, … share the layout). Full fine-tuning. Weights are read slice by slice from the safetensors
  files, so no rank ever loads the whole model. RoPE frequencies come from HF's own rotary class, so
  `rope_theta` and rope scaling (llama3, linear, yarn) are exact. Tested to give the same logits as the
  HF model (real Llama-3.2-1B: max logit difference 4e-5).
- **`model_type: custom_transformer`** — the same Llama-style model built from
  `custom_transformer_args`, with optional `n_kv_heads` (GQA), `ffn_hidden`, and `num_experts` /
  `top_k` (Mixture-of-Experts in every layer, needed for `ep > 1`).
- **Data** — the dataset's `text` column, tokenized and *packed* into blocks of `max_length` tokens
  (no padding), or `dataset.name: synthetic` for a quick learnable synthetic task.

## Rules (checked at startup, with the reason)

- `dp × tp × cp × pp = num_gpus`; `ep` divides `dp`
- attention heads, key/value heads and the MLP width divisible by `tp`; at least one layer per stage
- `max_length` divisible by `cp` (and by `cp × tp` with `sp`)
- `sp` needs `tp > 1`; `ep > 1` needs a MoE model
- not supported with 5d: PEFT/LoRA, quantization, fp16 (use bf16), model types other than
  `llm` / `custom_transformer`, sliding-window attention longer than the window, dynamic RoPE

## Choosing a layout

1. Fits on one GPU? Use `ddp` (or `5d` with only `dp`).
2. Too big: add **TP** inside a node (2, 4 or 8, over NVLink) and turn on **SP**.
3. Still too big, or more than one node: add **PP** across nodes. Use `micro_batches ≥ pp`.
   The pipeline bubble is `(pp - 1) / micro_batches`.
4. Long sequences: add **CP**. Ring attention splits every sequence, so activation memory per GPU
   drops by `cp`.
5. MoE: add **EP** so each GPU holds `num_experts / ep` experts.
6. Turn on **ZeRO-1** to divide the AdamW memory by `dp`.

## Outputs

```
checkpoints/5d/<unix_ms_timestamp>/
├── layout.json        # dp/tp/cp/pp/ep/sp/zero1 + model config
├── rank_00000.pt …    # each rank's model shard + optimizer state
└── hf/                # llm only: a normal Hugging Face checkpoint (bf16)
```

`save_load.resume: true` resumes from the latest (or `resume_path`) run, but only with the **same
layout**. To change layouts, start a new run with `model_name: checkpoints/5d/<ts>/hf`.

## How the pieces fit

**Rank grid.** Ranks sit on a grid `[dp, pp, cp, tp]` with TP changing fastest, so TP ranks are
neighbours on the same node (NVLink). EP reuses the DP axis: `dp_rank = chunk × ep + ep_rank`.

**One training step** (`parallelism/trainer.py`):
1. `BlockSampler` gives this rank `micro_batches` micro-batches (CP ranks keep their slice).
2. `PipelineEngine.train_step` runs AFAB or 1F1B. With `pp=1` it is plain gradient accumulation.
3. During the last backward, DP hooks launch bucketed all-reduces (overlap with compute).
4. `finish_grad_sync` waits for them, sums non-TP weights over TP when SP is on, and sums tied
   embeddings across the first and last pipeline stage.
5. Global grad-norm clipping (every element counted once), then `optimizer.step()`
   (ZeRO-1: each rank updates the parameters it owns, then broadcasts them).

**Which gradients go where**

| Weight | Synced over |
|---|---|
| normal weights | DP × CP (average) |
| expert weights | ranks holding the same experts, × CP (sum, divided by dp × cp) |
| non-TP weights with SP (norms, router, experts, o-proj bias) | also TP (sum) |
| tied embedding / LM head split by PP | first + last stage (sum) |

**Mixed precision.** Weights and optimizer states stay fp32; `mixed_precision: true` runs compute
under bf16 autocast. Activations sent between pipeline stages are fp32.

## Testing

```bash
python -m pytest tests/test_5d_parallel.py      # config rules, HF logit parity, export round trip
python -m pytest -m dist                        # every layout vs 1 process, on CPU (~5 min)
```

Every module builds its weights from a fixed seed (or the same HF file) and keeps only its shard,
and the data depends only on the global sample id. So every layout must reproduce the 1-process
run. The `dist` tests compare the **losses, the global gradient norm and every parameter's
gradient**. Comparing gradients matters: AdamW divides out a per-tensor gradient scale, so a
loss-only test passes even when a gradient is 2× too large.

## Simplifications (on purpose)

- ZeRO-1 only, and it shards whole tensors (all-reduce + broadcast instead of reduce-scatter +
  all-gather). ZeRO-2/3 are not included; use `strategy: fsdp` for full sharding.
- Ring attention uses plain chunks, not the zig-zag layout, so early CP ranks do less work.
- No Zero-Bubble / DualPipe / interleaved pipeline schedules.
- MoE has no load-balancing loss and no capacity limit.
- Full logits are gathered for the loss (no vocab-parallel cross-entropy), which costs memory for
  large vocabularies.
- No dropout in the 5D model.
