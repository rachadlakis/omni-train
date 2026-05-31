
# def apply_fsdp(local_rank, rank, device, args):
#     """Instantiates the model on meta device, shards it with FSDP2, applies mixed precision
#     and prefetching, then populates weights via one of three paths:
#       A. Resume from checkpoint  (--resume --resume-path)
#       B. Fresh run from HF       (--load-model-from-hf)
#       C. Random init             (no checkpoint dir)
#     Returns (model, checkpointer)."""

#     try:
#         if getattr(args, "model_type", "llm") == "custom_transformer":
#             from model import Transformer, ModelArgs
#             model_args = ModelArgs(
#                 n_layers=args.custom_n_layers,
#                 vocab_size=args.custom_vocab_size,
#                 max_seq_len=args.custom_max_seq_len,
#                 dim=args.custom_dim,
#                 n_heads=args.custom_n_heads,
#                 dropout_p=args.custom_dropout_p,
#             )
#             model = Transformer(model_args).to(device)
#             # Wrap each TransformerBlock as a separate FSDP unit for fine-grained sharding
#             from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
#             from model import TransformerBlock
#             for i, layer in enumerate(model.layers):
#                 model.layers[i] = fully_shard(layer) # type: ignore
#             model = fully_shard(model)
#             print_on_rank_0(rank, f"Custom Transformer (FSDP) built | layers={args.custom_n_layers} dim={args.custom_dim} heads={args.custom_n_heads} vocab={args.custom_vocab_size} ✓")
#             return model, None

#         # PEFT/quantized runs must start from materialized pretrained weights (non-meta path) before FSDP wrapping.
#         use_peft_or_quant = bool(getattr(args, "peft_enabled", False) or getattr(args, "quantization_enabled", False))

#         if use_peft_or_quant:
#             ## PEFT and/or quantization enabled — meta init is not compatible with the complex weight loading logic required for these features, so we build real models on each rank and then apply FSDP.
#             quant_cfg = _build_quantization_config(args, rank)
#             resuming           = args.resume and bool(args.resume_path)
#             load_model_from_hf = not resuming and args.load_model_from_hf

#             if resuming:
#                 # PATH A: resuming — build model structure from config only; checkpoint supplies
#                 # both base weights and adapter weights so there is no need to hit HuggingFace.

#                 # Guard: the checkpoint folder name encodes the run tag (e.g. __lora, __lora_q4).
#                 # If the user is trying to resume a PEFT run from a non-PEFT checkpoint (or vice-versa)
#                 # the adapter weights will be missing and the load will silently produce wrong results.
#                 _resume_folder_name = os.path.basename(os.path.normpath(os.path.abspath(args.resume_path)))
#                 _expected_tag = _checkpoint_run_tag(args)  # e.g. "__lora" or "__lora_q4"
#                 _checkpoint_is_plain = "__" not in _resume_folder_name
#                 if _expected_tag and _expected_tag not in _resume_folder_name:
#                     raise ValueError(
#                         f"Cannot resume a PEFT/quantized run (tag='{_expected_tag}') from a non-PEFT checkpoint "
#                         f"at '{args.resume_path}'. The checkpoint was saved without adapter weights — "
#                         f"there is nothing to restore the LoRA layers from. "
#                         f"Start a fresh run or point --resume-path to a checkpoint with '{_expected_tag}' in its folder name."
#                     )
#                 if not _expected_tag and not _checkpoint_is_plain:
#                     raise ValueError(
#                         f"Cannot resume a non-PEFT run from a PEFT checkpoint at '{args.resume_path}'. "
#                         f"The checkpoint contains adapter weights that have no corresponding LoRA layers in the current model. "
#                         f"Enable PEFT in your config or point --resume-path to a plain checkpoint."
#                     )

#                 print_on_rank_0(rank, "Resuming — building PEFT model structure from config (no HF download)", "♻️")
#                 config = AutoConfig.from_pretrained(args.model_name, token=HF_TOKEN)
#                 config.use_cache = False
#                 config.tie_word_embeddings = False
#                 model = _get_auto_model_class(getattr(args, "model_type", "llm")).from_config(
#                     config,
#                     dtype=DTYPE_MAP[args.param_dtype] if args.mixed_precision else torch.float32,
#                 )

#             elif load_model_from_hf:
#                 # PATH B: fresh run — only rank 0 downloads and saves a seed checkpoint;
#                 # all ranks then load from that seed so HF is only hit once.
#                 print_on_rank_0(rank, "Fresh PEFT run — rank 0 loading pretrained weights from HuggingFace", "🧠")
#                 peft_seed_folder = f"{args.checkpoint_dir}/pretrained_seed"
#                 # Rank 0 generates the timestamp; broadcast it so every rank resolves
#                 # the identical path (each rank calling int(time.time()*1000)
#                 # independently can get a different millisecond value).
#                 _ts = [int(time.time() * 1000) if rank == 0 else 0]
#                 dist.broadcast_object_list(_ts, src=0)
#                 peft_seed_timestamp = _ts[0]
#                 peft_seed_subfolder = f"{peft_seed_folder}/fsdp/{'dcp_api' if args.dcp_api else 'dtensor_api'}/{peft_seed_timestamp}"
#                 peft_seed_path = f"{peft_seed_subfolder}/model_state_dict.pt"

#                 pretrained_kwargs = {
#                     "token": HF_TOKEN,
#                     "low_cpu_mem_usage": True,
#                     "tie_word_embeddings": False,
#                 }
#                 if quant_cfg is not None:
#                     pretrained_kwargs["quantization_config"] = quant_cfg
#                     pretrained_kwargs["device_map"] = {"": local_rank} if torch.cuda.is_available() else {"": "cpu"}
#                 else:
#                     pretrained_kwargs["dtype"] = DTYPE_MAP[args.param_dtype] if args.mixed_precision else torch.float32

#                 if rank == 0:
#                     seed_model = _get_auto_model_class(getattr(args, "model_type", "llm")).from_pretrained(
#                         args.model_name, **pretrained_kwargs
#                     )
#                     seed_model.config.use_cache = False
#                     seed_model.config.tie_word_embeddings = False
#                     os.makedirs(peft_seed_subfolder, exist_ok=True)
#                     torch.save(seed_model.state_dict(), peft_seed_path)
#                     del seed_model
#                     torch.cuda.empty_cache()
#                 dist.barrier(device_ids=[local_rank] if dist.get_backend() == "nccl" else None)

#                 # All ranks: build structure from config and load the seed weights.
#                 config = AutoConfig.from_pretrained(args.model_name, token=HF_TOKEN)
#                 config.use_cache = False
#                 config.tie_word_embeddings = False
#                 model = _get_auto_model_class(getattr(args, "model_type", "llm")).from_config(
#                     config,
#                     dtype=DTYPE_MAP[args.param_dtype] if args.mixed_precision else torch.float32,
#                 )
#                 model.load_state_dict(torch.load(peft_seed_path, map_location="cpu"))
#                 print_on_rank_0(rank, "Pretrained seed weights loaded ✓")

#             else:
#                 # PATH C: random init — for experimentation and debugging only.
#                 print_on_rank_0(rank, "No checkpoint dir — random weight init for PEFT path", "⚠️")
#                 config = AutoConfig.from_pretrained(args.model_name, token=HF_TOKEN)
#                 config.use_cache = False
#                 config.tie_word_embeddings = False
#                 model = _get_auto_model_class(getattr(args, "model_type", "llm")).from_config(
#                     config,
#                     dtype=DTYPE_MAP[args.param_dtype] if args.mixed_precision else torch.float32,
#                 )

#             # Disable inference-oriented cache to keep training + checkpoint states stable.
#             model.config.use_cache = False
#             model.config.tie_word_embeddings = False
#             model = _apply_peft_quantization(model, args, rank)

#             # Guardrail for quantized runs: non-floating tensors cannot require gradients.
#             # Some quantized modules may surface parameters with a stale requires_grad flag,
#             # which causes fully_shard(...) to fail during FSDP param materialization.
#             non_float_frozen = 0
#             for _name, param in model.named_parameters():
#                 if not torch.is_floating_point(param) and param.requires_grad:
#                     param.requires_grad_(False)
#                     non_float_frozen += 1

#             if non_float_frozen > 0:
#                 print_on_rank_0(
#                     rank,
#                     f"Froze {non_float_frozen} non-floating parameter(s) before FSDP sharding",
#                     "🧊",
#                 )

#             fsdp_kwargs = {}
#             if args.mixed_precision and not args.quantization_enabled:
#                 if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
#                     print_on_rank_0(rank, "bfloat16 not supported on this device", "⚠️")
#                     args.mixed_precision = False
#                     args.param_dtype = "float16"
#                 else:
#                     fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
#                         param_dtype=DTYPE_MAP[args.param_dtype],
#                         reduce_dtype=DTYPE_MAP[args.reduce_dtype],
#                         output_dtype=DTYPE_MAP[args.output_dtype],
#                         cast_forward_inputs=args.cast_forward_inputs,
#                     )
#                     print_on_rank_0(rank, f"Mixed precision: {fsdp_kwargs['mp_policy'].param_dtype} for params, {fsdp_kwargs['mp_policy'].reduce_dtype} for reduce, {fsdp_kwargs['mp_policy'].output_dtype} for outputs", "⚡")
#             elif args.mixed_precision and args.quantization_enabled:
#                 # Quantized kernels already define compute dtype; skip FSDP MP policy layering.
#                 print_on_rank_0(rank, "Mixed precision policy skipped for quantized base; using quantization compute dtype", "ℹ️")

#             layers, layer_type = get_model_layers(model)
#             if layers is not None:
#                 print_on_rank_0(rank, f"Sharding {len(layers)} {layer_type} layers...", "🔀")
#                 for layer in layers:
#                     fully_shard(layer, **fsdp_kwargs)
#             else:
#                 print_on_rank_0(rank, "No individual layers found, sharding root model only", "⚠️")

#             fully_shard(model, **fsdp_kwargs)
#             print_on_rank_0(rank, "FSDP sharding applied ✓", "✅")

#             if args.explicit_prefetching and layers is not None:
#                 print_on_rank_0(rank, f"Setting up explicit prefetching: forward={args.forward_prefetch}, backward={args.backward_prefetch}", "🔄")
#                 set_modules_to_forward_prefetch(model, args.forward_prefetch)
#                 set_modules_to_backward_prefetch(model, args.backward_prefetch)

#             checkpointer = None
#             if resuming:
#                 _rp = os.path.normpath(os.path.abspath(args.resume_path))
#                 timestamp = os.path.basename(_rp)
#                 api_dir = os.path.basename(os.path.dirname(_rp))
#                 base = str(os.path.dirname(os.path.dirname(os.path.dirname(_rp))))
#                 if (args.dcp_api and api_dir != "dcp_api") or (not args.dcp_api and api_dir != "dtensor_api"):
#                     print_on_rank_0(rank, f"Warning: resume_path API {api_dir} does not match dcp_api={args.dcp_api}. Attempting to load anyway.", "⚠️")
#                 checkpointer = Checkpointer(folder=base, dcp_api=args.dcp_api, run_tag=_checkpoint_run_tag(args))
#                 # Keep exact folder token (including optional mode suffix) for load consistency.
#                 checkpointer.last_training_time = timestamp
#                 checkpointer.load_model(model)  # type: ignore
#                 print_on_rank_0(rank, "Checkpoint loaded into PEFT model ✓")
#             else:
#                 checkpointer = Checkpointer(folder=args.checkpoint_dir, dcp_api=args.dcp_api, run_tag=_checkpoint_run_tag(args))

#             print(f"[Rank {rank}] num params: {sum(p.numel() for p in model.parameters())}")
#             return model, checkpointer


#         ## Non Peft: 
#         ## FSDP Step 1: build model on meta device (no memory) 
#         print_on_rank_0(rank, "Instantiating model on meta device...", "🧠")
#         config = AutoConfig.from_pretrained(args.model_name, token=HF_TOKEN)
#         config.use_cache = False  # important for saving memory during training
#         config.tie_word_embeddings = False  # prevents lm_head/embed_tokens KeyError in optimizer state dict

#         with torch.device("meta"): ## creates on all GPUs, but really does not create real models 
#             model = _get_auto_model_class(getattr(args, "model_type", "llm")).from_config(
#                 config,
#                 dtype=DTYPE_MAP[args.param_dtype] if args.mixed_precision else torch.float32,                
#             ) 

#         ## FSDP Step 2: shard layers + root model (still on meta) 
#         fsdp_kwargs = {}
#         if args.mixed_precision:
#             if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
#                 print_on_rank_0(rank, "bfloat16 not supported on this device", "⚠️")
#                 args.mixed_precision = False
#                 args.param_dtype = "float16"
#             else:
#                 fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
#                     param_dtype=DTYPE_MAP[args.param_dtype],        ## bfloat16 for weights and activations
#                     reduce_dtype=DTYPE_MAP[args.reduce_dtype],      ## float32 for gradient reduction
#                     output_dtype=DTYPE_MAP[args.output_dtype],      ## bfloat16 for outputs
#                     cast_forward_inputs=args.cast_forward_inputs    ## false: if FSDP auto-casts inputs entering the module
#                 )
#                 print_on_rank_0(rank, f"Mixed precision: {fsdp_kwargs['mp_policy'].param_dtype} for params, {fsdp_kwargs['mp_policy'].reduce_dtype} for reduce, {fsdp_kwargs['mp_policy'].output_dtype} for outputs", "⚡")

#         if bool(getattr(args, "gradient_checkpointing", True)):
#             if hasattr(model, 'gradient_checkpointing_enable'):
#                 model.gradient_checkpointing_enable()
#                 print_on_rank_0(rank, "Gradient Checkpointing (Activation Checkpointing) enabled", "💾")
#             else:
#                 print_on_rank_0(rank, "This model does not support Gradient Checkpointing (Activation Checkpointing).", "⚠️")
        
#         layers, layer_type = get_model_layers(model)
#         if layers is not None:
#             print_on_rank_0(rank, f"Sharding {len(layers)} {layer_type} layers...", "🔀")
#             for layer in layers:
#                 fully_shard(layer, **fsdp_kwargs)
#         else:
#             print_on_rank_0(rank, "No individual layers found, sharding root model only", "⚠️")

#         fully_shard(model, **fsdp_kwargs)
#         print_on_rank_0(rank, "FSDP sharding applied ✓")

        
#         if args.explicit_prefetching and layers is not None:
#             print_on_rank_0(rank, f"Setting up explicit prefetching: forward={args.forward_prefetch}, backward={args.backward_prefetch}", "🔄")
#             set_modules_to_forward_prefetch(model, args.forward_prefetch)
#             set_modules_to_backward_prefetch(model, args.backward_prefetch)

#         # ------------------------------------------------------------------
#         # FSDP Step 3: populate weights — three paths:
#         #   A. Resume from checkpoint  → load checkpoint
#         #   B. Fresh run               → load from HuggingFace on rank 0, shard, delete seed
#         #   C. No checkpoint_dir       → random init
#         # ------------------------------------------------------------------
#         resuming           = args.resume and bool(args.resume_path)
#         load_model_from_hf = not resuming and args.load_model_from_hf
#         checkpointer       = None

#         if resuming:
#             if not (args.resume_path):
#                 print_on_rank_0(rank, "❌ Resume path not provided to resume training.", "❌")
#                 raise ValueError("No resume path provided")
#             else:
#                 print_on_rank_0(rank, f"Resuming from: {args.resume_path}", "♻️") 
#                 _rp = os.path.normpath(os.path.abspath(args.resume_path))
#                 timestamp = os.path.basename(_rp)
#                 api_dir = os.path.basename(os.path.dirname(_rp))
#                 base = str(os.path.dirname(os.path.dirname(os.path.dirname(_rp))))
#                 if (args.dcp_api and api_dir != "dcp_api") or (not args.dcp_api and api_dir != "dtensor_api"):
#                     print_on_rank_0(rank, f"Warning: resume_path API {api_dir} does not match dcp_api={args.dcp_api}. Attempting to load anyway.", "⚠️")
#                 checkpointer = Checkpointer(folder=base, dcp_api=args.dcp_api, run_tag=_checkpoint_run_tag(args))
#                 checkpointer.last_training_time = timestamp
#                 checkpointer.load_model(model)
                
#         elif load_model_from_hf:
#             # PATH B: fresh run — load pretrained weights from HF on rank 0 only
#             print_on_rank_0(rank, "Fresh run — loading pretrained weights from HuggingFace on rank 0", "🆕")
#             pretrained_seed_folder = f"{args.checkpoint_dir}/pretrained_seed"

#             # Rank 0 generates the timestamp and broadcasts it so every rank
#             # resolves the identical seed path (independent calls to
#             # int(time.time()*1000) across ranks can differ by milliseconds).
#             _ts = [int(time.time() * 1000) if rank == 0 else 0]
#             dist.broadcast_object_list(_ts, src=0)
#             timestamp = _ts[0]
#             pretrained_seed_subfolder = f"{pretrained_seed_folder}/fsdp/{'dcp_api' if args.dcp_api else 'dtensor_api'}/{timestamp}"
#             pretrained_seed_path = f"{pretrained_seed_subfolder}/model_state_dict.pt"

#             if rank == 0:
#                 if os.path.exists(pretrained_seed_path):
#                     print_on_rank_0(rank, "Model already downloaded", "💾")
#                 else:
#                     print_on_rank_0(rank, "Downloading model from HuggingFace", "💾")
#                     seed_model = _get_auto_model_class(getattr(args, "model_type", "llm")).from_pretrained(
#                         args.model_name,
#                         token=HF_TOKEN,
#                         dtype=DTYPE_MAP[args.param_dtype] if args.mixed_precision else torch.float32,
#                         low_cpu_mem_usage=True,
#                         tie_word_embeddings=False,
#                     )
#                     seed_model.config.tie_word_embeddings = False
#                     os.makedirs(pretrained_seed_subfolder, exist_ok=True)
#                     print_on_rank_0(rank, "Saving seed weights to disk (other ranks waiting)...", "💾")
#                     torch.save(seed_model.state_dict(), pretrained_seed_path)
#                     print_on_rank_0(rank, "Seed weights saved ✓ | releasing barrier", "✅")
#                     del seed_model
#                     torch.cuda.empty_cache()
#             dist.barrier(device_ids=[local_rank] if dist.get_backend() == "nccl" else None)

#             seed_checkpointer = Checkpointer(folder=pretrained_seed_folder, dcp_api=args.dcp_api)
#             seed_checkpointer.load_model(model)



# def setup_dist_process_group():
#     """Initializes the distributed process group and sets the CUDA device for this process.
#     Expects environment variables RANK, LOCAL_RANK, and WORLD_SIZE to be set by the launcher (e.g. torchrun)."""
#     try:
#         rank = int(os.environ.get("RANK", "0"))
#         local_rank = int(os.environ.get("LOCAL_RANK", "0"))
#         print_on_rank_0(rank, f"Initializing process group with backend: {BACKEND}", "⚙️")
#         dist.init_process_group(backend=BACKEND, device_id=torch.device(f"cuda:{local_rank}"))

#         if torch.cuda.is_available():
#             torch.cuda.set_device(f"cuda:{local_rank}")
#             print_on_rank_0(rank, f"Process group initialized ✓ | rank: {rank} | local_rank: {local_rank}", "✅")
#         return local_rank
#     except Exception as e:
#         print_on_rank_0(int(os.environ.get("RANK", "0")), f"❌ Failed to initialize process group: {e}", "❌")
#         raise

# def setup_dist_process_group():
#     try:
#         rank = int(os.environ.get("RANK", "0"))
#         local_rank = int(os.environ.get("LOCAL_RANK", "0"))
#         print_on_rank_0(rank, f"Initializing process group with backend: {BACKEND}", "⚙️")

#         if torch.cuda.is_available():
#             device = torch.device(f"cuda:{local_rank}")
#             torch.cuda.set_device(device)
#             dist.init_process_group(backend=BACKEND, device_id=device)
#         else:
#             dist.init_process_group(backend=BACKEND)
#         print_on_rank_0(rank,f"Process group initialized ✓ | rank: {rank} | local_rank: {local_rank}","✅")
#         return local_rank

#     except Exception as e:
#         print_on_rank_0(int(os.environ.get("RANK", "0")),f"❌ Failed to initialize process group: {e}","❌")
#         raise