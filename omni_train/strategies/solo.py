"""strategy: solo, one process on one device."""

import traceback

import torch
from transformers import AutoConfig

from omni_train.console import print_on_rank_0
from omni_train.strategies.common import HF_TOKEN, _apply_peft_quantization, _build_quantization_config, _check_checkpoint_peft_compat, _get_auto_model_class, _load_dtype


def apply_solo(device, rank, args):
    """Loads a single-process model and moves it to the selected device."""
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
            print_on_rank_0(rank, f"Custom Transformer built from scratch | layers={args.custom_n_layers} dim={args.custom_dim} heads={args.custom_n_heads} vocab={args.custom_vocab_size} ✓")
            return model

        quant_cfg = _build_quantization_config(args, rank)
        model_kwargs = {
            "token": HF_TOKEN,
            "low_cpu_mem_usage": True,
        }
        if quant_cfg is not None:
            model_kwargs["quantization_config"] = quant_cfg
            model_kwargs["device_map"] = {"": device}
        else:
            model_kwargs["dtype"] = _load_dtype(args)
            model_kwargs["device_map"] = None

        resuming = args.resume and bool(getattr(args, "resume_path", None))

        if resuming:
            # ADDED: resume path for solo — was completely missing before.
            # Old code always loaded from HuggingFace and silently ignored args.resume /
            # args.resume_path, so "resume" solo runs were actually fresh runs from HF weights.
            # This meant (a) the resumed weights were never actually loaded, and (b) a PEFT
            # checkpoint could be "resumed" into a non-PEFT model without any error.
            print_on_rank_0(rank, f"Resuming from checkpoint: {args.resume_path}", "🔄")

            # Load checkpoint first so we can validate compat before any expensive model build.
            checkpoint = torch.load(
                args.resume_path,
                map_location="cpu",
                mmap=True,
                weights_only=True,
            )
            state_dict = checkpoint.get("model_state_dict", checkpoint)

            # Fail fast if PEFT type doesn't match — same guard as apply_ddp.
            _check_checkpoint_peft_compat(
                state_dict,
                peft_enabled=getattr(args, "peft_enabled", False),
                resume_path=args.resume_path,
            )

            if quant_cfg is not None:
                # Quantized: must use from_pretrained to build the bnb layer structure.
                model = _get_auto_model_class(getattr(args, "model_type", "llm")).from_pretrained(
                    args.model_name, **model_kwargs,
                )
            else:
                # Non-quantized: build architecture only — no HF weight download needed.
                _resume_cfg = AutoConfig.from_pretrained(args.model_name, token=HF_TOKEN)
                _resume_cfg.use_cache = False
                model = _get_auto_model_class(getattr(args, "model_type", "llm")).from_config(
                    _resume_cfg,
                    dtype=_load_dtype(args),
                )
            # Apply PEFT wrapping BEFORE load_state_dict so key structure matches
            # (PEFT checkpoints use base_model.model.* keys).
            model = _apply_peft_quantization(model, args, rank)
            if quant_cfg is None:
                model = model.to(device)
            model.load_state_dict(state_dict)
            print_on_rank_0(rank, "Solo checkpoint loaded ✓")

        else:
            # Fresh run: load pretrained weights from HuggingFace.
            print_on_rank_0(rank, f"Fetching pretrained weights: {args.model_name}", "🧠")
            model = _get_auto_model_class(getattr(args, "model_type", "llm")).from_pretrained(
                args.model_name,
                **model_kwargs,
            )
            model = _apply_peft_quantization(model, args, rank)
            if quant_cfg is None:
                model = model.to(device)

        print_on_rank_0(rank, "Solo model ready ✓")
        return model
    except ValueError:
        raise  # user config error — let train.py __main__ print it once cleanly
    except Exception as e:
        print(f"\n[rank {rank}] ❌ Failed to apply solo model setup: {e}", flush=True)
        traceback.print_exc()
        raise
