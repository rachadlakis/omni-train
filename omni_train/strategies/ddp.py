"""strategy: ddp, a full model copy per GPU with averaged gradients."""

import traceback

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoConfig

from omni_train.console import print_on_rank_0
from omni_train.distributed import dist_barrier
from omni_train.strategies.common import HF_TOKEN, _apply_peft_quantization, _build_quantization_config, _check_checkpoint_peft_compat, _get_auto_model_class, _load_dtype


def apply_ddp(local_rank, rank, device, args):
    """Moves the model to the appropriate GPU and wraps it with DistributedDataParallel (DDP).
    Populates weights via one of three paths:
      A. Resume from checkpoint  (--resume --resume-path)
      B. Fresh run from HF       (--load-model-from-hf)  — every rank loads independently
      C. Random init             (no checkpoint dir, for experimentation only)
    Returns model wrapped with DDP.
    """
    try:
        if getattr(args, "model_type", "llm") == "custom_transformer":
            from omni_train.model import Transformer, ModelArgs
            model_args = ModelArgs(
                n_layers=args.custom_n_layers,
                vocab_size=args.custom_vocab_size,
                max_seq_len=args.custom_max_seq_len,
                dim=args.custom_dim,
                n_heads=args.custom_n_heads,
                dropout_p=args.custom_dropout_p,
            )
            model = Transformer(model_args).to(device)
            model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None,
                        find_unused_parameters=False)
            print_on_rank_0(rank, f"Custom Transformer (DDP) built | layers={args.custom_n_layers} dim={args.custom_dim} heads={args.custom_n_heads} vocab={args.custom_vocab_size} ✓")
            return model

        # Determine quant mode up front so resume/fresh/random paths share one config source.
        quant_cfg = _build_quantization_config(args, rank)

        def _pretrained_kwargs():
            # Single helper prevents drift in kwargs between different loading branches.
            kwargs = {
                "token": HF_TOKEN,
                "low_cpu_mem_usage": True,
            }
            if quant_cfg is not None:
                # Each rank owns its local shard/device placement for quantized loading.
                kwargs["quantization_config"] = quant_cfg
                kwargs["device_map"] = {"": local_rank} if torch.cuda.is_available() else {"": "cpu"}
            else:
                # Keep float model initialization behavior unchanged when quantization is off.
                kwargs["dtype"] = _load_dtype(args)
            return kwargs

        resuming           = args.resume and bool(args.resume_path)
        load_model_from_hf = not resuming and args.load_model_from_hf

        if resuming:
            # PATH A: resume from checkpoint
            print_on_rank_0(rank, f"Resuming from checkpoint: {args.resume_path}", "🔄")
            checkpoint = torch.load(
                args.resume_path,
                map_location="cpu",
                mmap=True,
                weights_only=True,
            )
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            _check_checkpoint_peft_compat(
                state_dict,
                peft_enabled=getattr(args, "peft_enabled", False),
                resume_path=args.resume_path,
            )

            if quant_cfg is not None:
                # Quantized resume: must use from_pretrained to build bnb quantized layer structure.
                model = _get_auto_model_class(getattr(args, "model_type", "llm")).from_pretrained(
                    args.model_name,
                    **_pretrained_kwargs(),
                )
            else:
                # Non-quantized resume: build architecture only — no HF weight download needed.
                # Keep the model's own tie_word_embeddings (as solo does): an untied copy of a tied
                # model has one more parameter, so the saved AdamW state would no longer match, and
                # the LM head would drift away from the embedding it was trained as.
                _resume_config = AutoConfig.from_pretrained(args.model_name, token=HF_TOKEN)
                _resume_config.use_cache = False
                model = _get_auto_model_class(getattr(args, "model_type", "llm")).from_config(
                    _resume_config,
                    dtype=_load_dtype(args),
                )
        elif load_model_from_hf:
            ## rank-0-first download, then all ranks load from cache
            if rank == 0:
                import transformers as _transformers
                print_on_rank_0(rank, f"Rank 0 fetching weights to cache: {args.model_name}", "🧠")
                print_on_rank_0(rank, "If not cached, weights will be downloaded now — progress shown below:", "⏳")
                _transformers.logging.enable_progress_bar()   # restore bars for the download
                _ = _get_auto_model_class(getattr(args, "model_type", "llm")).from_pretrained(
                    args.model_name,
                    **_pretrained_kwargs())
                _transformers.logging.disable_progress_bar()  # re-suppress for the rest of training
                del _
                print_on_rank_0(rank, "Weights cached ✓ — releasing barrier for all ranks", "✅")
                
            print(f"\n[rank {rank}] Waiting at barrier for rank 0 to cache weights...", flush=True)
            dist_barrier(local_rank)  # everyone waits until rank 0's cache write is done
            print_on_rank_0(rank, "All ranks loading model from cache...", "📦")
            model = _get_auto_model_class(getattr(args, "model_type", "llm")).from_pretrained(
                args.model_name,
                **_pretrained_kwargs())

        else:
            # PATH C: random init — for experimentation and debugging only
            print_on_rank_0(rank, "No checkpoint dir — random weight init", "⚠️")
            config = AutoConfig.from_pretrained(args.model_name, token=HF_TOKEN)
            config.use_cache = False
            config.tie_word_embeddings = False
            model = _get_auto_model_class(getattr(args, "model_type", "llm")).from_config(
                config,
                dtype=_load_dtype(args),
            )
            if hasattr(model, "init_weights"):
                model.init_weights()
            else:
                for m in model.modules():
                    if hasattr(m, "reset_parameters"):
                        m.reset_parameters()  # type: ignore

        model = _apply_peft_quantization(model, args, rank)
        if quant_cfg is None:
            model = model.to(device) 

        if resuming:
            model.load_state_dict(state_dict) # type: ignore
            print_on_rank_0(rank, "Checkpoint state dict loaded ✓")

        model = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            find_unused_parameters=False
            )
        print_on_rank_0(rank, "DDP wrapper applied ✓")
        return model

    except ValueError:
        raise  # user config error — let train.py __main__ print it once cleanly
    except Exception as e:
        print(f"\n[rank {rank}] ❌ Failed to apply DDP: {e}", flush=True)
        traceback.print_exc()
        raise
