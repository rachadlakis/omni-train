# OMNI-Train — FSDP Layer Detection Fix

**Date:** 2026-05-25

---

## What Was Broken

`get_model_layers(model)` in `distributed_utils.py` is the function that finds
the `nn.ModuleList` of transformer blocks so `apply_fsdp` can shard them one by
one (`fully_shard(layer)` per block). When it fails to find a stack, the code
falls back to root-only sharding — every rank effectively holds the whole model
and FSDP's memory savings collapse to ~0.

The previous implementation checked only four hard-coded patterns:

```python
model.model.decoder.layers      # OPT, BART decoder
model.model.encoder.layers      # BART encoder
model.transformer.h             # GPT-2
model.layers                    # (intended for Llama, but Llama puts it at
                                #  model.model.layers — so this branch never fires)
```

An empirical scan of every model in `configs/` showed that the function was
silently broken for almost every model family the framework claims to support:

| Model family                          | Real ModuleList path                                   | Old detection |
| ------------------------------------- | ------------------------------------------------------ | ------------- |
| Llama / Mistral / Qwen / Gemma / Phi  | `model.layers` → *DecoderLayer                         | ❌ None       |
| DeepSeek / Kimi / GLM                 | `model.layers` → *DecoderLayer                         | ❌ None       |
| OPT                                   | `model.decoder.layers` → OPTDecoderLayer               | ✅ found      |
| GPT-2                                 | `transformer.h` → GPT2Block                            | ✅ found      |
| T5                                    | `encoder.block` + `decoder.block` → T5Block            | ❌ None       |
| BART                                  | `model.encoder.layers` + `model.decoder.layers`        | ⚠️ decoder only |
| BERT / RoBERTa                        | `encoder.layer` (singular!) → BertLayer                | ❌ None       |
| ViT / DeiT / Swin                     | `vit.encoder.layer` → ViTLayer                         | ❌ None       |
| ResNet                                | `resnet.encoder.stages` → ResNetStage                  | ❌ None       |
| YOLOS                                 | `vit.encoder.layer` → YolosLayer                       | ❌ None       |
| CLIP                                  | `text_model.encoder.layers` + `vision_model.encoder.layers` | ❌ None  |
| LLaVA / VLMs                          | `language_model.model.layers` + vision tower           | ❌ None       |

The `set_modules_to_forward_prefetch` / `set_modules_to_backward_prefetch`
helpers duplicated the same four hard-coded checks and inherited the same
gaps — explicit prefetching silently became a no-op for the same models.

---

## Root Cause

Three independent design defects compounded:

1. **Missing `model.model.layers` branch.** This is the canonical decoder-only
   LLM structure (Llama, Mistral, Qwen, Gemma, DeepSeek, Phi). The function
   had `model.layers` but not the doubly-nested form.
2. **Plural-only key names.** HF inconsistently uses `layers` vs `layer` vs
   `block` vs `stages` (BERT and ViT use the singular `layer`; T5 uses `block`;
   ResNet uses `stages`). The function only looked for `layers` / `h`.
3. **Single-stack return type.** The signature was
   `(ModuleList, str) | (None, None)` — incapable of representing T5/BART/CLIP/VLM,
   which all have **two transformer stacks** that both need per-block sharding.

---

## What Was Changed

Two HF-provided introspection hints, available on every model in the
`transformers` package, anchor the new implementation:

```text
model._no_split_modules    # set[str]: class names that must not be split across devices.
                           # By convention these are the transformer-block classes.
model.base_model_prefix    # str: name of the inner backbone attr ("bert", "vit", …).
```

`_no_split_modules` is the same hint `device_map="auto"` uses to keep
transformer blocks intact when sharding across devices — exactly the granularity
FSDP wants.

### New `get_model_layers` — three-tier detection

```python
def get_model_layers(model):
    """Returns list[(ModuleList, layer_type_name)] — one entry per transformer stack."""

    # Unwrap PEFT
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        model = model.base_model.model

    # 1) HF _no_split_modules — walk the tree, collect every ModuleList whose
    #    element class name is on the no-split list. Naturally multi-tower.
    no_split = set(getattr(model, "_no_split_modules", None) or [])
    if no_split:
        stacks = [
            (m, type(m[0]).__name__)
            for m in model.modules()
            if isinstance(m, torch.nn.ModuleList) and len(m) > 0
               and type(m[0]).__name__ in no_split
        ]
        if stacks:
            return stacks

    # 2) Known attribute paths — covers models with empty _no_split_modules (YOLOS)
    for path in _KNOWN_LAYER_PATHS:    # see distributed_utils.py for full list
        target = model
        for part in path.split("."):
            if not hasattr(target, part):
                target = None; break
            target = getattr(target, part)
        if isinstance(target, torch.nn.ModuleList) and len(target) > 0:
            return [(target, type(target[0]).__name__)]

    # 3) Heuristic: pick the largest ModuleList by parameter count
    best, best_params = None, 0
    for m in model.modules():
        if isinstance(m, torch.nn.ModuleList) and len(m) > 1:
            p = sum(x.numel() for x in m.parameters())
            if p > best_params:
                best, best_params = m, p
    return [(best, type(best[0]).__name__)] if best else []
```

### Caller updates

`apply_fsdp` PATHs 2 and 3 (`distributed_utils.py:915, 1005`) — were:

```python
layers, layer_type = get_model_layers(model)
if layers is not None:
    for layer in layers:
        fully_shard(layer, **fsdp_kwargs)
```

Now:

```python
stacks = get_model_layers(model)
if stacks:
    for layers, _ in stacks:
        for layer in layers:
            fully_shard(layer, **fsdp_kwargs)
```

`set_modules_to_forward_prefetch` / `set_modules_to_backward_prefetch` — the
inline copies of the old detection logic were removed and both functions now
delegate to `get_model_layers`, iterating per stack.

---

## What the Fix Does

### Tier 1: `_no_split_modules` walk

The HF authors maintain `_no_split_modules` on every model so `device_map="auto"`
can shard parameters without splitting an attention block in half. The class
names listed are exactly the transformer-block classes we want FSDP to treat as
atomic shard units. Walking the module tree and collecting every `nn.ModuleList`
whose first element matches gives us:

- Single-tower models → one stack returned (Llama, Mistral, Qwen, GPT-2, OPT, …)
- Multi-tower models → multiple stacks returned (T5, BART, CLIP, VLMs)
- Models with `_no_split_modules = set()` (YOLOS) → falls through to tier 2

### Tier 2: known attribute paths

A small list of literal nested-attribute paths covers the few HF models whose
`_no_split_modules` is empty (YOLOS) and serves as a redundant safety net for
the common patterns. The list is data, not control flow — adding a new model
family that the heuristic misses is a one-line append to `_KNOWN_LAYER_PATHS`.

### Tier 3: parameter-count heuristic

For truly unknown architectures (private models, research code, future HF
additions before they ship `_no_split_modules`), pick the `nn.ModuleList` with
the most parameters. The transformer stack dominates every other ModuleList by
1–2 orders of magnitude in practice, so this is reliable as a last resort.

---

## Verification

Empirical scan of every model in `configs/`, after the fix:

```text
Qwen2 (Llama-ish)    24xQwen2DecoderLayer        ← was BROKEN
OPT                  24xOPTDecoderLayer
GPT-2                12xGPT2Block
T5  (2 towers)       6xT5Block, 6xT5Block        ← was BROKEN
BART (2 towers)      6xBartEncoderLayer, 6xBartDecoderLayer   ← only decoder before
BERT                 12xBertLayer                ← was BROKEN
ViT                  12xViTLayer                 ← was BROKEN
ResNet               4xResNetStage               ← was BROKEN
YOLOS                12xYolosLayer               ← was BROKEN
CLIP (2 towers)      12xCLIPEncoderLayer, 12xCLIPEncoderLayer  ← was BROKEN
```

Unit tests in `tests/test_helpers.py` and `tests/test_model.py` were updated
to the new list-return type and now cover:

- Decoder-only single-tower detection
- Flat `.layers` detection
- Empty-input → empty-list return
- `_no_split_modules` walk on a synthetic Llama-style wrapper
- Multi-tower stack collection on a synthetic T5-style wrapper

```bash
$ python -m pytest tests/test_helpers.py tests/test_model.py -q
26 passed in 5.06s
```

---

## Why This Matters

Before the fix, every Llama, Mistral, Qwen, BERT, ViT, T5, CLIP, and VLM run
under `strategy: fsdp` was silently falling back to root-only sharding. The
process would not crash — it would just consume far more memory than expected
(every rank holding the full model) and the user would see no warning beyond
a one-line `"No individual layers found, sharding root model only"` at startup.
For a Llama-3 70B that means the difference between fitting on 8×A100-80GB and
OOMing immediately.

The fix also unlocks the framework's claim that FSDP works for non-LLM
architectures: vision transformers, CNNs, seq2seq models, multimodal towers,
and detection backbones now all shard correctly with the same code path.

---

## Files Changed

| File                                        | Change                                                          |
| ------------------------------------------- | --------------------------------------------------------------- |
| `distributed_utils.py`                      | Rewrote `get_model_layers`; updated `set_modules_to_*_prefetch` and the two `apply_fsdp` sharding loops to iterate over a list of stacks. |
| `tests/test_helpers.py`                     | Updated to new list-return type; added `_no_split_modules` and multi-tower coverage. |
| `tests/test_model.py`                       | Updated single test to new return type.                         |
| `Documentation/LAYER_DETECTION_FIX_NOTES.md`| This file.                                                      |
