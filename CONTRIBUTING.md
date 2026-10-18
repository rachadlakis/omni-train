# Contributing to OMNI-Train

Thanks for your interest in contributing. This guide covers everything you need to get started.

---

## Setup

```bash
git clone https://github.com/rachadlakis/omni-train.git
cd omni-train

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Required for gated models (LLaMA, Mistral, Gemma)
cp .env.example .env   # then add your HF_TOKEN
```

---

## Running Tests

All non-GPU tests must pass before opening a PR.

```bash
# Unit tests — no GPU required
python -m pytest

# Single test file
python -m pytest tests/test_config_validation.py -v

# GPU smoke tests (~30-60s each, requires a GPU)
python -m pytest -m smoke
```

---

## Good First Contributions

If you're not sure where to start, these are well-scoped and genuinely useful:

- **Add a new example config** — pick a model or use case not yet covered in `configs/`. Follow the structure of an existing config and add a row to the README table.
- **Extend layer detection** — `get_model_layers()` in `distributed_utils.py` probes common HF architecture patterns. If you find a model whose layers aren't detected, add the pattern there.
- **Write tests** — `tests/` has good coverage of config validation but light coverage of data loading and checkpoint paths.
- **Improve the Web UI** — `ui/static/app.js` and `ui/static/styles.css`. The UI is functional but there is room to improve the job history view and error display.

**Advanced:**
- **3D hybrid parallelism** (`hybrid/`) — the module exists and is tested in isolation but is not yet integrated into the main training loop. This is the highest-impact open task and the most complex.

---

## Hard Constraints

These are non-obvious rules that will break training silently if ignored. Read them before touching core files.

**Quantization + FSDP are incompatible.**
`bitsandbytes` quantized layers (4-bit NF4, 8-bit INT8) cannot be sharded. FSDP's sharding mechanism operates on float tensors and breaks on quantized weights. Never wire quantization with `strategy: fsdp`. This is enforced as a hard error in `build_args()` — keep it that way.

**Quantization requires PEFT.**
4-bit and 8-bit quantized weights have no differentiable backward pass. You cannot train them directly with AdamW. Any quantization without `peft.enabled: true` must raise an error in `build_args()`.

**Config changes must go through `build_args()`.**
All config validation lives in `utils.py::build_args()`. If you add a new config field, add it to the `Args` dataclass and validate it there. Do not add conditional logic inside `train.py` or `distributed_utils.py` based on raw config values.

**Meta-device init is FSDP-only.**
The meta-device loading path only applies when PEFT and quantization are both disabled. If you add a new loading path, make sure it does not break this invariant.

**Non-persistent buffers must be restored, not zeroed or re-derived.**
`to_empty()` wipes buffers like RoPE `inv_freq`, and they are not in `state_dict()`. The meta-init path snapshots the constructor-computed values (`_snapshot_buffers()`) and restores them after loading (`_restore_buffers()`). Zeroing them silently corrupts positional encodings, and recomputing with a fixed formula ignores `rope_theta`/rope scaling.

**Trainable weights stay fp32.**
Use `_load_dtype()` when building a model. Mixed precision is a compute-time cast (FSDP `MixedPrecisionPolicy`, or autocast in `train.py`); storing trainable weights in bf16 makes most AdamW updates round to zero.

---

## PR Guidelines

- **One feature or fix per PR.** Mixed PRs are hard to review and hard to revert.
- **Add a config example** if you're adding a new capability. If it runs, it works.
- **Touch `build_args()` or `distributed_utils.py`?** Add a unit test in `tests/test_config_validation.py` or `tests/test_config_combinations.py`.
- **Touch the checkpoint logic?** Verify both `dcp_api` and `dtensor_api` paths still work.
- Keep PR descriptions short: what changed and why, not how.

---

## Out of Scope

Please open an issue to discuss before submitting a PR that:

- Adds a new dependency to `requirements.txt`
- Changes the checkpoint folder layout or naming convention (requires a migration path)
- Modifies the config YAML schema in a backwards-incompatible way
- Adds a new training strategy at the same level as solo/DDP/FSDP

---

## Questions

Open a GitHub issue. If it's a quick question about the distributed internals or a specific config, feel free to ask — this project exists to make distributed training more accessible.
