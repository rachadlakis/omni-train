
# Handoff: OMNI-Train — `strategy: 5d` work

**Repo:** `/home/rachad_lakkis/projects/omni-train` — PyTorch distributed fine-tuning framework.
**Read first:** `CLAUDE.md`, then `docs/5D_PARALLELISM.md`.

---

## 1. Where things stand

| Item            | Status                                                                                                      |
| --------------- | ----------------------------------------------------------------------------------------------------------- |
| Branch          | `feat/5d-parallelism` (commit `47ab7d1`)                                                                |
| Pushed / PR     | ❌ Not pushed, no PR, `main` untouched                                                                    |
| Tests           | ✅ 190 default + 31 multi-process, all passing                                                              |
| Untracked files | `docs/OMNI-Train Field Guide.html` + its `:Zone.Identifier`. **Not mine, don't commit or delete** |
| Environment     | `.venv` has full `requirements.txt` installed                                                           |
| Hardware        | WSL2, 1× RTX 4050 (6 GB), 7.6 GB RAM → multi-process tests run on **CPU**                           |

---

## 2. What was built

**Goal:** choose any parallelism (up to 5D) in YAML and run it with the normal launcher.

```yaml
strategy: 5d
num_gpus: 8
parallelism:
  dp: auto          # = num_gpus / (tp × cp × pp)
  tp: 2             # tensor parallel
  sp: true          # sequence parallel (needs tp > 1)
  cp: 1             # context parallel (ring attention)
  pp: 2             # pipeline stages
  pp_schedule: 1f1b # or afab
  micro_batches: 8  # global batch = batch_size × micro_batches × dp
  ep: 1             # expert parallel (MoE only)
  zero1: true       # shard AdamW states
```

Run: `CONFIG_PATH=configs/<file>.yaml bash scripts/launch.sh`. torchrun and SLURM work unchanged.

### Supported

- **Models:** Hugging Face **Llama, Mistral, Qwen2** (full fine-tune), plus `custom_transformer` (can be MoE)
- **Data:** an HF dataset's `text` column (packed into fixed-length chunks), or `dataset.name: synthetic`
- **Training:** bf16 mixed precision, gradient checkpointing, grad clipping, LR schedules, W&B
- **Saving:** sharded checkpoints + resume (same layout only) + export to normal HF format

---

## 3. Where the code lives

| File                             | Role                                                          |
| -------------------------------- | ------------------------------------------------------------- |
| `parallelism/trainer.py`       | Training loop; `train.py` calls `train_5d(args)`          |
| `parallelism/model.py`         | Llama-style model that splits across all 5 dimensions         |
| `parallelism/hf.py`            | Load HF weights slice by slice; export back to HF             |
| `parallelism/data_parallel.py` | Gradient sync (DP, experts, SP, tied embeddings)              |
| `parallelism/optim.py`         | Global grad-norm clipping                                     |
| `parallelism/data.py`          | Packed text / synthetic data, per-rank batches                |
| `parallelism/checkpoint.py`    | Sharded save / resume                                         |
| `utils.py` → `_parse_5d`    | YAML validation for the `parallelism:` block                |
| `configs/*5d*.yaml`            | 4 example configs (8-GPU, long context, 16-GPU MoE, CPU demo) |
| `tests/test_5d_parallel.py`    | All 5d tests                                                  |

---

## 4. Testing

```bash
python -m pytest           # fast: config rules, HF parity, export round trip
python -m pytest -m dist   # ~4 min: 29 layouts vs a 1-process run
```

> ⚠️ **Rule:** comparing losses alone can't catch gradient-scale bugs (AdamW cancels them out).
> The `dist` tests compare **every parameter's gradient**. Keep them green after any gradient-sync change.

Also verified: real Llama-3.2-1B matches HF (max logit diff 4e-5), and a GPU run through `launch.sh` with real wikitext succeeds.

---

## 5. Other changes in the same commit

- Fixed: expert gradients were `ep`× too large
- Removed: old hybrid/3D leftovers, dev notes, field guide, `site/`, dead code
- Fixed: `scripts/setup_env.py` crashed on fresh machines (stray `sympy` import)

---

## 6. Not done yet

| Gap                                 | Note                                                                                |
| ----------------------------------- | ----------------------------------------------------------------------------------- |
| LoRA / QLoRA / quantization with 5d | Rejected at config load                                                             |
| Web UI support                      | 5d templates hidden in `ui/static/app.js`, because the UI would turn 5d into solo |
| More architectures                  | Qwen3, Gemma, Mixtral                                                               |
| Real multi-GPU run                  | Never run on more than 1 GPU; 8/16-GPU configs only validated                       |
| Performance extras                  | Vocab-parallel loss, zig-zag ring attention, interleaved pipeline                   |

## 7. Decisions waiting on the user

1. Push the branch and open a PR?
2. Add 5d to the web UI?
