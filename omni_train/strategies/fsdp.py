"""strategy: fsdp (FSDP2): meta-device init, seeds, buffers, PEFT/quantized and custom models."""

import os
import shutil
import time
import traceback

import torch
import torch.distributed as dist
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from transformers import AutoConfig

from omni_train.checkpoint import Checkpointer
from omni_train.console import inspect_model, print_on_rank_0
from omni_train.distributed import cleanup, dist_barrier
from omni_train.estimates import print_training_vram_estimate
from omni_train.strategies.common import DTYPE_MAP, HF_TOKEN, _apply_peft_quantization, _build_quantization_config, _checkpoint_run_tag, _get_auto_model_class, _load_dtype, get_model_layers


def set_modules_to_forward_prefetch(model, num_to_forward_prefetch):
    """Configures each transformer stack to prefetch the next N layers in forward."""
    try:
        stacks = get_model_layers(model)
        if not stacks:
            print("Warning: Could not find layers for prefetching")
            return
        for layers, _ in stacks:
            for i, layer in enumerate(layers):
                if i >= len(layers) - num_to_forward_prefetch:
                    break
                layers_to_prefetch = [layers[i + j] for j in range(1, num_to_forward_prefetch + 1)]
                if hasattr(layer, 'set_modules_to_forward_prefetch'):
                    layer.set_modules_to_forward_prefetch(layers_to_prefetch) #type: ignore
    except Exception as e:
        print(f"❌ Failed to set forward prefetch: {e}")
        raise


def set_modules_to_backward_prefetch(model, num_to_backward_prefetch):
    """Configures each transformer stack to prefetch the previous N layers in backward."""
    try:
        stacks = get_model_layers(model)
        if not stacks:
            print("Warning: Could not find layers for prefetching")
            return
        for layers, _ in stacks:
            for i, layer in enumerate(layers):
                if i < num_to_backward_prefetch:
                    continue
                layers_to_prefetch = [layers[i - j] for j in range(1, num_to_backward_prefetch + 1)]
                if hasattr(layer, 'set_modules_to_backward_prefetch'):
                    layer.set_modules_to_backward_prefetch(layers_to_prefetch) #type: ignore
    except Exception as e:
        print(f"❌ Failed to set backward prefetch: {e}")
        raise


def _apply_fsdp_custom_transformer(args, device, rank):
    """PATH 1 of apply_fsdp: the toy custom Transformer.

    Builds on meta device so the full model never lands on a single GPU, shards each
    block + the root with FSDP2 (mixed precision applied if enabled), then materializes
    only each rank's 1/N shard via to_empty() and re-inits real weights. The toy model
    has no init_weights() and no non-persistent buffers (it uses learned positional
    embeddings, not RoPE), so the reset_parameters() walk fully populates it —
    no buffer snapshot/restore needed. Returns (model, None)."""
    from omni_train.model import Transformer, ModelArgs
    model_args = ModelArgs(
        n_layers=args.custom_n_layers,
        vocab_size=args.custom_vocab_size,
        max_seq_len=args.custom_max_seq_len,
        dim=args.custom_dim,
        n_heads=args.custom_n_heads,
        dropout_p=args.custom_dropout_p,
    )

    # Mixed-precision policy — same logic as PATH 3. Built before sharding so
    # fully_shard() applies it to every block and the root.
    fsdp_kwargs = {}
    if args.mixed_precision:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            print_on_rank_0(rank, "bfloat16 not supported on this device", "⚠️")
            args.mixed_precision = False
            args.param_dtype = "float16"
        else:
            fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
                param_dtype=DTYPE_MAP[args.param_dtype],
                reduce_dtype=DTYPE_MAP[args.reduce_dtype],
                output_dtype=DTYPE_MAP[args.output_dtype],
                cast_forward_inputs=args.cast_forward_inputs,
            )
            print_on_rank_0(rank, f"Mixed precision: {fsdp_kwargs['mp_policy'].param_dtype} for params, {fsdp_kwargs['mp_policy'].reduce_dtype} for reduce, {fsdp_kwargs['mp_policy'].output_dtype} for outputs", "⚡")

    with torch.device("meta"):
        model = Transformer(model_args)

    # Report training VRAM on the full model *before* sharding (toy model → activations
    # are reported as n/a since it exposes no HF config).
    print_training_vram_estimate(rank, model, args, dist.get_world_size())

    for layer in model.layers:
        fully_shard(layer, **fsdp_kwargs) # type: ignore
    fully_shard(model, **fsdp_kwargs)

    model.to_empty(device=device)
    for m in model.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters() # type: ignore

    if args.explicit_prefetching:
        print_on_rank_0(rank, f"Setting up explicit prefetching: forward={args.forward_prefetch}, backward={args.backward_prefetch}", "🔄")
        set_modules_to_forward_prefetch(model, args.forward_prefetch)
        set_modules_to_backward_prefetch(model, args.backward_prefetch)

    inspect_model(model) # type: ignore
    print_on_rank_0(rank, f"Custom Transformer (FSDP) built | layers={args.custom_n_layers} dim={args.custom_dim} heads={args.custom_n_heads} vocab={args.custom_vocab_size} ✓")
    return model, None


def _snapshot_buffers(model):
    """Copy every real (non-meta) buffer so it survives model.to_empty().

    The meta-init path builds the model with parameters on meta but buffers computed
    for real by the HF constructors (init_empty_weights(include_buffers=False)).
    to_empty() then replaces *every* tensor with uninitialized memory, and
    non-persistent buffers (RoPE inv_freq / original_inv_freq, BERT position_ids, …)
    are excluded from state_dict(), so no checkpoint load ever refills them.
    """
    return {name: buf.detach().clone() for name, buf in model.named_buffers() if buf.device.type != "meta"}


def _restore_buffers(model, snapshot, device, rank, include_persistent=False):
    """Put back buffers captured by _snapshot_buffers after to_empty() + weight load.

    By default only non-persistent buffers are restored — persistent ones come from the
    checkpoint. Pass include_persistent=True on the random-init path, where nothing is
    loaded and the constructor values are the right ones.
    """
    restored = 0
    for name, buf in list(model.named_buffers()):
        *path, attr = name.split(".")
        parent = model
        for part in path:
            parent = getattr(parent, part)
        persistent = attr not in parent._non_persistent_buffers_set
        if persistent and not include_persistent:
            continue
        if name in snapshot:
            setattr(parent, attr, snapshot[name].to(device))
            restored += 1
        else:
            # No real value was ever computed for this buffer. Zeros are a guess —
            # add an explicit case here if a model needs something else.
            setattr(parent, attr, torch.zeros(buf.shape, dtype=buf.dtype, device=device))
            print_on_rank_0(rank, f"Buffer '{name}' had no computed value — initialized as zeros", "⚠️")
    if restored:
        print_on_rank_0(rank, f"Restored {restored} buffer(s) computed at model construction (RoPE inv_freq, position ids, …)", "🔧")


def _remove_seed_dir(seed_subfolder, seed_root, local_rank, rank):
    """Delete this run's pretrained seed once every rank has loaded it.

    Only this run's timestamped folder is removed (plus parents left empty, up to
    seed_root) — never the whole pretrained_seed/ tree, which concurrent runs sharing
    checkpoint_dir may still be reading from.
    """
    dist_barrier(local_rank)  # every rank must be done reading before rank 0 deletes
    if rank != 0:
        return
    shutil.rmtree(seed_subfolder, ignore_errors=True)
    parent = os.path.dirname(os.path.normpath(seed_subfolder))
    root = os.path.normpath(seed_root)
    while parent.startswith(root):
        try:
            os.rmdir(parent)  # only succeeds when empty
        except OSError:
            break
        if parent == root:
            break
        parent = os.path.dirname(parent)
    print_on_rank_0(rank, "Pretrained seed deleted (weights are now on every rank)", "🧹")


def _apply_fsdp_peft_quant(local_rank, rank, device, args):
    """PATH 2: PEFT / quantization. Model is materialized before wrapping (cannot use
    meta device): rank 0 seeds full weights, all ranks load CPU-side, PEFT adapters
    attach, then layers are moved to GPU and sharded one at a time. Returns (model, checkpointer)."""
    quant_cfg = _build_quantization_config(args, rank)
    resuming           = args.resume and bool(args.resume_path)
    load_model_from_hf = not resuming and args.load_model_from_hf

    if resuming:
        _resume_folder_name = os.path.basename(os.path.normpath(os.path.abspath(args.resume_path)))
        _expected_tag = _checkpoint_run_tag(args)
        _checkpoint_is_plain = "__" not in _resume_folder_name
        if _expected_tag and _expected_tag not in _resume_folder_name:
            raise ValueError(f"Cannot resume PEFT run from non-PEFT checkpoint.")
        if not _expected_tag and not _checkpoint_is_plain:
            raise ValueError(f"Cannot resume non-PEFT run from PEFT checkpoint.")

        print_on_rank_0(rank, "Resuming — building PEFT model structure from config (no HF download)", "♻️")
        config = AutoConfig.from_pretrained(args.model_name, token=HF_TOKEN)
        config.use_cache = False
        config.tie_word_embeddings = False
        model = _get_auto_model_class(getattr(args, "model_type", "llm")).from_config(
            config,
            dtype=_load_dtype(args),
        )

    elif load_model_from_hf:
        print_on_rank_0(rank, "Fresh PEFT run — rank 0 loading pretrained weights from HuggingFace", "🧠")
        peft_seed_folder = f"{args.checkpoint_dir}/pretrained_seed"
        _ts = [int(time.time() * 1000) if rank == 0 else 0]
        dist.broadcast_object_list(_ts, src=0)
        peft_seed_timestamp = _ts[0]
        peft_seed_subfolder = f"{peft_seed_folder}/fsdp/{'dcp_api' if args.dcp_api else 'dtensor_api'}/{peft_seed_timestamp}"
        peft_seed_path = f"{peft_seed_subfolder}/model_state_dict.pt"

        # Do NOT pass tie_word_embeddings=False to from_pretrained.
        # Llama-style checkpoints only contain embed_tokens.weight and rely on
        # the tying to populate lm_head.weight. If we untie at construction
        # time, lm_head.weight has nothing to load and stays at random init
        # (you'll see "lm_head.weight | MISSING" in the load report and the
        # model can never learn — loss stuck at ln(vocab_size)).
        # We load tied, then clone+untie below so the saved seed has both
        # weights populated and FSDP can shard them independently.
        pretrained_kwargs = {
            "token": HF_TOKEN,
            "low_cpu_mem_usage": True,
        }
        if quant_cfg is not None:
            pretrained_kwargs["quantization_config"] = quant_cfg
            pretrained_kwargs["device_map"] = {"": local_rank} if torch.cuda.is_available() else {"": "cpu"}
        else:
            pretrained_kwargs["dtype"] = _load_dtype(args)

        if rank == 0:
            import transformers as _transformers
            print_on_rank_0(rank, "Downloading PEFT seed weights — progress shown below:", "⏳")
            _transformers.logging.enable_progress_bar()
            seed_model = _get_auto_model_class(getattr(args, "model_type", "llm")).from_pretrained(
                args.model_name, **pretrained_kwargs
            )
            _transformers.logging.disable_progress_bar()
            try:
                seed_model.config.use_cache = False
                seed_model.config.tie_word_embeddings = False
            except:
                raise

            # Fix truncated weight-tying cloning logic
            if hasattr(seed_model, "lm_head") and hasattr(seed_model, "model") and hasattr(seed_model.model, "embed_tokens"):
                if seed_model.lm_head.weight.data_ptr() == seed_model.model.embed_tokens.weight.data_ptr():
                    seed_model.lm_head.weight = torch.nn.Parameter(
                        seed_model.model.embed_tokens.weight.clone()
                    )
                    print_on_rank_0(rank, "Cloned embed_tokens → lm_head.weight (was tied)", "🔗")

            os.makedirs(peft_seed_subfolder, exist_ok=True)
            torch.save(seed_model.state_dict(), peft_seed_path)
            del seed_model

        dist_barrier(local_rank)  # ensure rank 0 has saved the seed checkpoint before others try to load
        config = AutoConfig.from_pretrained(args.model_name, token=HF_TOKEN)

        try:
            config.use_cache = False
            config.tie_word_embeddings = False
        except:
            raise

        model = _get_auto_model_class(getattr(args, "model_type", "llm")).from_config(
            config,
            dtype=_load_dtype(args),
        )
        model.load_state_dict(
            torch.load(peft_seed_path, mmap=True, weights_only=True, map_location="cpu")
        )
        print_on_rank_0(rank, "Pretrained seed weights loaded ✓")
        _remove_seed_dir(peft_seed_subfolder, peft_seed_folder, local_rank, rank)

    else:
        print_on_rank_0(rank, "No checkpoint dir — random weight init for PEFT path", "⚠️")
        config = AutoConfig.from_pretrained(args.model_name, token=HF_TOKEN)
        config.use_cache = False
        config.tie_word_embeddings = False
        model = _get_auto_model_class(getattr(args, "model_type", "llm")).from_config(
            config,
            dtype=_load_dtype(args),
        )

    try:
        model.config.use_cache = False
        model.config.tie_word_embeddings = False
    except:
        raise

    model = _apply_peft_quantization(model, args, rank)

    # Freeze any non-floating-point params before sharding.
    # Done here on CPU so the loop runs without any device dependency.
    non_float_frozen = 0
    for _name, param in model.named_parameters():
        if not torch.is_floating_point(param) and param.requires_grad:
            param.requires_grad_(False)
            non_float_frozen += 1
    if non_float_frozen > 0:
        print_on_rank_0(rank, f"Froze {non_float_frozen} non-floating parameter(s) before FSDP sharding", "🧊")

    fsdp_kwargs = {}
    if args.mixed_precision and not args.quantization_enabled:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            print_on_rank_0(rank, "bfloat16 not supported on this device", "⚠️")
            args.mixed_precision = False
            args.param_dtype = "float16"
        else:
            fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
                param_dtype=DTYPE_MAP[args.param_dtype],
                reduce_dtype=DTYPE_MAP[args.reduce_dtype],
                output_dtype=DTYPE_MAP[args.output_dtype],
                cast_forward_inputs=args.cast_forward_inputs,
            )
            print_on_rank_0(rank, f"Mixed precision: {fsdp_kwargs['mp_policy'].param_dtype} for params, {fsdp_kwargs['mp_policy'].reduce_dtype} for reduce, {fsdp_kwargs['mp_policy'].output_dtype} for outputs", "⚡")
    elif args.mixed_precision and args.quantization_enabled:
        print_on_rank_0(rank, "Mixed precision policy skipped for quantized base; using quantization compute dtype", "ℹ️")

    # Report training VRAM on the full model *before* sharding, so the user sees the
    # unsharded footprint (and why FSDP is needed) ahead of the sharding output.
    print_training_vram_estimate(rank, model, args, dist.get_world_size())

    stacks = get_model_layers(model)
    if stacks:
        total_layers = sum(len(layers) for layers, _ in stacks)
        summary = ", ".join(f"{len(layers)}x{ltype}" for layers, ltype in stacks)
        print_on_rank_0(rank, f"Sharding {total_layers} layer(s) across {len(stacks)} stack(s) (layer-by-layer): {summary}", "🔀")
        for layers, _ in stacks:
            for layer in layers:
                layer.to(device)                    # one block on GPU (~total / num_layers bytes)
                fully_shard(layer, **fsdp_kwargs)   # shard immediately → rank holds 1/N of block
    else:
        print_on_rank_0(rank, "No individual layers found — materializing full model on GPU (OOM risk for large models)", "⚠️")
        model = model.to(device)

    # Move any parameters still on CPU (embeddings, lm_head, layer norms, etc.)
    # that were not covered by the per-layer loop above.
    for param in model.parameters():
        if param.device.type == "cpu":
            param.data = param.data.to(device)

    fully_shard(model, **fsdp_kwargs)

    inspect_model(model)
    print_on_rank_0(rank, "FSDP sharding applied ✓. Model inspected ✓", "✅")

    if args.explicit_prefetching and stacks:
        print_on_rank_0(rank, f"Setting up explicit prefetching: forward={args.forward_prefetch}, backward={args.backward_prefetch}", "🔄")
        set_modules_to_forward_prefetch(model, args.forward_prefetch)
        set_modules_to_backward_prefetch(model, args.backward_prefetch)

    checkpointer = None
    if resuming:
        _resume_path = os.path.normpath(os.path.abspath(args.resume_path))
        timestamp = os.path.basename(_resume_path)
        api_dir = os.path.basename(os.path.dirname(_resume_path))
        base = str(os.path.dirname(os.path.dirname(os.path.dirname(_resume_path))))
        if (args.dcp_api and api_dir != "dcp_api") or (not args.dcp_api and api_dir != "dtensor_api"):
            print_on_rank_0(rank, f"Warning: resume_path API {api_dir} does not match dcp_api={args.dcp_api}. Attempting to load anyway.", "⚠️")
        checkpointer = Checkpointer(folder=base, dcp_api=args.dcp_api, run_tag=_checkpoint_run_tag(args))
        checkpointer.last_training_time = timestamp
        checkpointer.load_model(model) # type: ignore
        print_on_rank_0(rank, "Checkpoint loaded into PEFT model from saved ✓")
    else:
        checkpointer = Checkpointer(folder=args.checkpoint_dir, dcp_api=args.dcp_api, run_tag=_checkpoint_run_tag(args))
        print_on_rank_0(rank, "Checkpointer initialized for new run ✓")

    print(f"[Rank {rank}] num params: {sum(p.numel() for p in model.parameters())}")

    return model, checkpointer


def _apply_fsdp_meta_init(local_rank, rank, device, args):
    """PATH 3: standard non-PEFT / non-quantized FSDP2 meta-device init. Build on meta,
    shard layers + root, then populate weights via resume / fresh-HF / random init.
    Returns (model, checkpointer)."""
    ## FSDP Step 1: build model on meta device (no memory)
    print_on_rank_0(rank, "Instantiating model on meta device...", "🧠")
    config = AutoConfig.from_pretrained(args.model_name, token=HF_TOKEN)
    config.use_cache = False
    config.tie_word_embeddings = False

    # Parameters go on meta (no memory); buffers are still computed for real on CPU so
    # non-persistent ones (RoPE inv_freq, position_ids, …) keep the exact values the
    # HF constructor produced — see _snapshot_buffers / _restore_buffers below.
    from accelerate import init_empty_weights  # type: ignore[import-untyped]
    with init_empty_weights(include_buffers=False):
        model = _get_auto_model_class(getattr(args, "model_type", "llm")).from_config(
            config,
            dtype=_load_dtype(args),
        )

    ## FSDP Step 2: shard layers + root model (still on meta)
    fsdp_kwargs = {}
    if args.mixed_precision:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            print_on_rank_0(rank, "bfloat16 not supported on this device", "⚠️")
            args.mixed_precision = False
            args.param_dtype = "float16"
        else:
            fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
                param_dtype=DTYPE_MAP[args.param_dtype],
                reduce_dtype=DTYPE_MAP[args.reduce_dtype],
                output_dtype=DTYPE_MAP[args.output_dtype],
                cast_forward_inputs=args.cast_forward_inputs,
            )
            print_on_rank_0(rank, f"Mixed precision: {fsdp_kwargs['mp_policy'].param_dtype} for params, {fsdp_kwargs['mp_policy'].reduce_dtype} for reduce, {fsdp_kwargs['mp_policy'].output_dtype} for outputs", "⚡")

    if bool(getattr(args, "gradient_checkpointing", True)):
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            print_on_rank_0(rank, "Gradient Checkpointing (Activation Checkpointing) enabled", "💾")
        else:
            print_on_rank_0(rank, "This model does not support Gradient Checkpointing (Activation Checkpointing).", "⚠️")

    # Report training VRAM on the full model *before* sharding, so the user sees the
    # unsharded footprint (and why FSDP is needed) ahead of the sharding output.
    print_training_vram_estimate(rank, model, args, dist.get_world_size())

    stacks = get_model_layers(model)
    if stacks:
        total_layers = sum(len(layers) for layers, _ in stacks)
        summary = ", ".join(f"{len(layers)}x{ltype}" for layers, ltype in stacks)
        print_on_rank_0(rank, f"Sharding {total_layers} layer(s) across {len(stacks)} stack(s): {summary}", "🔀")
        for layers, _ in stacks:
            for layer in layers:
                fully_shard(layer, **fsdp_kwargs)
    else:
        print_on_rank_0(rank, "No individual layers found, sharding root model only", "⚠️")

    fully_shard(model, **fsdp_kwargs)

    inspect_model(model)
    print_on_rank_0(rank, "FSDP sharding applied ✓. Model inspected ✓", "✅")

    if args.explicit_prefetching and stacks:
        print_on_rank_0(rank, f"Setting up explicit prefetching: forward={args.forward_prefetch}, backward={args.backward_prefetch}", "🔄")
        set_modules_to_forward_prefetch(model, args.forward_prefetch)
        set_modules_to_backward_prefetch(model, args.backward_prefetch)

    # ------------------------------------------------------------------
    # FSDP Step 3: populate weights — three paths:
    #   A. Resume from checkpoint  → load checkpoint
    #   B. Fresh run               → load from HuggingFace on rank 0, shard, delete seed
    #   C. No checkpoint_dir       → random init
    # ------------------------------------------------------------------
    resuming           = args.resume and bool(args.resume_path)
    load_model_from_hf = not resuming and args.load_model_from_hf
    checkpointer       = None

    if resuming:
        if not (args.resume_path):
            print_on_rank_0(rank, "❌ Resume path not provided to resume training.", "❌")
            raise ValueError("No resume path provided")
        else:
            print_on_rank_0(rank, f"Resuming from: {args.resume_path}", "♻️")
            _resume_path = os.path.normpath(os.path.abspath(args.resume_path))
            timestamp = os.path.basename(_resume_path)
            api_dir = os.path.basename(os.path.dirname(_resume_path))
            base = str(os.path.dirname(os.path.dirname(os.path.dirname(_resume_path))))
            if (args.dcp_api and api_dir != "dcp_api") or (not args.dcp_api and api_dir != "dtensor_api"):
                print_on_rank_0(rank, f"Warning: resume_path API {api_dir} does not match dcp_api={args.dcp_api}. Attempting to load anyway.", "⚠️")
            checkpointer = Checkpointer(folder=base, dcp_api=args.dcp_api, run_tag=_checkpoint_run_tag(args))
            checkpointer.last_training_time = timestamp
            # Materialize meta-device DTensors to real (empty) CUDA storage so
            # set_model_state_dict can copy_() weights via NCCL into each rank's shard.
            # Without this, copy_() into a meta tensor deadlocks silently.
            buffers = _snapshot_buffers(model)
            model.to_empty(device=device)
            checkpointer.load_model(model)
            _restore_buffers(model, buffers, device, rank)

    elif load_model_from_hf:
        print_on_rank_0(rank, "Loading pretrained weights from HuggingFace on rank 0", "🆕")
        pretrained_seed_folder = f"{args.checkpoint_dir}/pretrained_seed"

        _ts = [int(time.time() * 1000) if rank == 0 else 0]
        dist.broadcast_object_list(_ts, src=0)
        timestamp = _ts[0]
        pretrained_seed_subfolder = f"{pretrained_seed_folder}/fsdp/{'dcp_api' if args.dcp_api else 'dtensor_api'}/{timestamp}"
        pretrained_seed_path = f"{pretrained_seed_subfolder}/model_state_dict.pt"

        if rank == 0:
            if os.path.exists(pretrained_seed_path):
                print_on_rank_0(rank, "Model already downloaded", "💾")
            else:
                import transformers as _transformers
                print_on_rank_0(rank, "Downloading model from HuggingFace — progress shown below:", "💾")
                _transformers.logging.enable_progress_bar()   # restore bars for the download
                # See PATH 2 comment: do NOT pass tie_word_embeddings=False to
                # from_pretrained — it leaves lm_head.weight at random init for
                # tied-weight checkpoints (Llama-style). Load tied, clone+untie below.
                seed_model = _get_auto_model_class(getattr(args, "model_type", "llm")).from_pretrained(
                    args.model_name,
                    token=HF_TOKEN,
                    dtype=_load_dtype(args),
                    low_cpu_mem_usage=True,
                )
                _transformers.logging.disable_progress_bar()  # re-suppress for the rest of training
                seed_model.config.tie_word_embeddings = False
                print_on_rank_0(rank, "Pretrained model loaded on rank 0 ✓", "🧠")

                # Fix truncated weight-tying cloning logic for the common case where lm_head is tied to embed_tokens
                if (
                    hasattr(seed_model, "lm_head")
                    and hasattr(seed_model, "model")
                    and hasattr(seed_model.model, "embed_tokens")
                    and seed_model.lm_head.weight.data_ptr() == seed_model.model.embed_tokens.weight.data_ptr()
                ):
                    seed_model.lm_head.weight = torch.nn.Parameter(
                        seed_model.model.embed_tokens.weight.clone()
                    )
                    print_on_rank_0(rank, "Cloned embed_tokens → lm_head.weight (was tied)", "🔗")

                os.makedirs(pretrained_seed_subfolder, exist_ok=True)
                print_on_rank_0(rank, "Saving seed weights to disk (other ranks waiting)...", "💾")
                torch.save(seed_model.state_dict(), pretrained_seed_path)
                print_on_rank_0(rank, "Seed weights saved ✓ | releasing barrier")
                del seed_model
                torch.cuda.empty_cache()

        dist_barrier(local_rank)
        # Materialize meta-device DTensors to real (empty) CUDA storage — same
        # reason as the resume path above. Each rank allocates only its 1/N shard.
        buffers = _snapshot_buffers(model)
        model.to_empty(device=device)
        seed_checkpointer = Checkpointer(folder=pretrained_seed_folder, dcp_api=args.dcp_api)
        # Pin to this run's seed — "latest folder" could be a concurrent run's seed
        # for a different model when several jobs share checkpoint_dir.
        seed_checkpointer.last_training_time = str(timestamp)
        seed_checkpointer.load_model(model)
        print_on_rank_0(rank, "Pretrained weights loaded and sharded ✓")

        _restore_buffers(model, buffers, device, rank)
        _remove_seed_dir(pretrained_seed_subfolder, pretrained_seed_folder, local_rank, rank)

        checkpointer = Checkpointer(folder=args.checkpoint_dir, dcp_api=args.dcp_api, run_tag=_checkpoint_run_tag(args))

    else:
        # PATH C: no checkpoint dir — random init
        print_on_rank_0(rank, "No checkpoint dir — random weight init", "⚠️")
        buffers = _snapshot_buffers(model)
        model.to_empty(device=device)
        if hasattr(model, "init_weights"):
            model.init_weights()
        else:
            for m in model.modules():
                if hasattr(m, "reset_parameters"):
                    m.reset_parameters()
        _restore_buffers(model, buffers, device, rank, include_persistent=True)
        checkpointer = None

    return model, checkpointer


def apply_fsdp(local_rank, rank, device, args):
    """Dispatches FSDP2 model setup to the path matching the run config, then returns
    (model, checkpointer). Each path builds, shards, and populates weights independently:
      - custom_transformer  -> _apply_fsdp_custom_transformer (meta init, toy model)
      - peft / quantization -> _apply_fsdp_peft_quant         (materialize before wrap)
      - everything else     -> _apply_fsdp_meta_init          (meta init: resume / HF / random)
    The shared try/except keeps user-config errors (ValueError) clean and routes any other
    failure through cleanup() so a dead process group never lingers."""
    try:

        ## Path 1: toy custom Transformer model (not from HuggingFace) — always meta init since it is tiny, no PEFT or quantization support
        if getattr(args, "model_type") == "custom_transformer":
            return _apply_fsdp_custom_transformer(args, device, rank)

        ## Path 2: any PEFT or quantization → materialize before wrap since both are incompatible with meta device init. This includes resume and fresh runs since the PEFT adapter structure must be in place before loading either from HF or a checkpoint.
        use_peft_or_quant = bool(getattr(args, "peft_enabled", False) or getattr(args, "quantization_enabled", False))
        if use_peft_or_quant:
            return _apply_fsdp_peft_quant(local_rank, rank, device, args)

        ## Path 3: standard FSDP2 meta init for non-PEFT, non-quantized runs
        return _apply_fsdp_meta_init(local_rank, rank, device, args)

    except ValueError:
        raise  # user config error — let train.py __main__ print it once cleanly
    except Exception as e:
        print(f"\n[rank {rank}] ❌ Failed in apply_fsdp: {e}", flush=True)
        traceback.print_exc()
        cleanup()
        raise
