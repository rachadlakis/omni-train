## distributed_utils.py ##

import os
import sys
import socket
import time
import traceback
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import (
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM, 
    AutoModelForImageClassification, 
    AutoModelForObjectDetection, 
    AutoModelForImageTextToText, 
    AutoModel, 
    AutoConfig,
    BitsAndBytesConfig,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, fully_shard, MixedPrecisionPolicy
from checkpoint import Checkpointer
from dotenv import load_dotenv
from utils import dist_barrier, inspect_model
from datetime import timedelta


load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

DTYPE_MAP = {
    "bfloat16": torch.bfloat16, 
    "float32": torch.float32, 
    "float16": torch.float16
}

if torch.cuda.is_available():
    if not sys.platform.startswith("linux"):
        raise RuntimeError(
            "The current distributed CUDA path uses NCCL, and NCCL is supported on Linux for now."
            "Non-Linux backends can be added later."
            f"Current platform: {sys.platform}"
        )
    BACKEND = "nccl"
else:
    BACKEND = "gloo"

def print_on_rank_0(rank, msg, emoji=""):
    if rank == 0:
        print(f"\n{emoji}  {msg}" if emoji else f"   {msg}", flush=True)

def print_banner_on_rank_0(rank, title):
    if rank == 0:
        print(f"\n{'='*60}",flush=True)
        print(f"  {title}", flush=True)
        print(f"{'='*60}",  flush=True)

def print_on_all_ranks(rank, msg, emoji="", local_rank=None, device=None):
    """Prints a message from all ranks, prefixed with rank and device info."""
    host = socket.gethostname().split(".")[0]
    prefix_parts = [f"host={host}", f"rank={rank}"]
    if local_rank is not None:
        prefix_parts.append(f"local_rank={local_rank}")
    if device is not None:
        prefix_parts.append(f"device={device}")
    prefix = " | ".join(prefix_parts)
    print(f"\n{emoji}  [{prefix}] {msg}" if emoji else f"\n[{prefix}] {msg}", flush=True)
    
def gather_rank_debug(rank, world_size, title, message):
    """Utility to gather debug messages from all ranks and print them together on rank 0."""
    if not dist.is_available() or not dist.is_initialized():
        print(message, flush=True)
        return

    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, message) ## Gather messages from all ranks into 'gathered' list

    if rank == 0:
        print(f"\n🔎  {title}", flush=True)
        for item in gathered:
            print(f"   {item}", flush=True)

def gpu_memory_snapshot(device):
    """Returns a string with the current GPU memory usage (allocated and reserved) for the given device."""
    if device.type != "cuda":
        return "CUDA not detected."
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
    return f"allocated={allocated:.2f} GB | reserved={reserved:.2f} GB"



import os
from datetime import timedelta
import torch
import torch.distributed as dist


def _detect_nvlink() -> bool:
    """Return True if NVML reports at least one active NVLink between local GPUs.

    Failure modes are deliberately split:
      - pynvml not installed         → RuntimeError. We have no way to know whether
                                       NVLink is present, so silently falling back
                                       to SHM would lie to the user on NVLink boxes.
                                       The dependency is pinned in requirements.txt
                                       (nvidia-ml-py); the user must install it.
      - pynvml present but NVML errs → return False. The probe was attempted, NVML
                                       just couldn't give us an answer (unsupported
                                       driver, transient error). SHM fallback is safe.
    """
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return False
    try:
        import pynvml  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "Cannot determine NVLink presence: 'pynvml' (nvidia-ml-py) is not installed.\n"
            "   Refusing to silently assume PCIe-only — that would force SHM transport\n"
            "   on NVLink-equipped machines and hide a real performance regression.\n"
            "   Install it with:\n\n"
            "       pip install -r requirements.txt\n\n"
            "   (or `pip install nvidia-ml-py==12.560.30`) and re-run."
        ) from e
    try:
        pynvml.nvmlInit()
        try:
            for i in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                for link in range(pynvml.NVML_NVLINK_MAX_LINKS):
                    try:
                        if pynvml.nvmlDeviceGetNvLinkState(handle, link) == pynvml.NVML_FEATURE_ENABLED:
                            return True
                    except pynvml.NVMLError:
                        break
            return False
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return False


def setup_dist_process_group():
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    try:
        print_on_rank_0(rank, f"Initializing process group with backend: {BACKEND}", "⚙️")

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            # On PCIe-only GPUs (e.g. RTX A4000) NCCL's P2P topology probe hangs
            # indefinitely; SHM (shared-memory) transport is fast and reliable instead.
            # On NVLink-equipped GPUs (A100/H100/V100) P2P is much faster than SHM, so
            # we leave it enabled. `setdefault` lets the user override either way via .env.
            p2p_default = "0" if _detect_nvlink() else "1"
            os.environ.setdefault("NCCL_P2P_DISABLE", p2p_default)
            print_on_rank_0(
                rank,
                f"NCCL_P2P_DISABLE={os.environ['NCCL_P2P_DISABLE']} "
                f"(NVLink {'detected' if p2p_default == '0' else 'not detected'})",
                "🔌",
            )

        # Do NOT pass device_id= here: it triggers an eager NCCL communicator that conflicts
        # with the device_ids=[local_rank] in barrier() calls, deadlocking on NCCL 2.21.5.
        dist.init_process_group(backend=BACKEND, timeout=timedelta(minutes=10))

        print_on_rank_0(rank, f"Process group initialized ✓ | rank: {rank} | local_rank: {local_rank}", "✅")

        # NCCL warm-up: absorb the first-collective cold-start here, where both ranks are
        # already synchronised, rather than silently mid-training.
        if BACKEND == "nccl":
            print_on_rank_0(rank, "NCCL warm-up barrier…", "🔥")
            dist.barrier(device_ids=[local_rank])
            print_on_rank_0(rank, "NCCL ready ✓", "🔥")

        return local_rank

    except Exception as e:
        print_on_rank_0(rank, f"❌ Failed to initialize process group: {e}", "❌")
        raise



def cleanup():
    try:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception as e:
        print(f"⚠️ Warning: Failed to clean up process group: {e}")



def set_modules_to_forward_prefetch(model, num_to_forward_prefetch):
    """Configures the model layers to prefetch activations for the next N layers during the forward pass."""
    try:
        layers = None
        if hasattr(model, 'model') and hasattr(model.model, 'decoder') and hasattr(model.model.decoder, 'layers'):
            layers = model.model.decoder.layers
        elif hasattr(model, 'model') and hasattr(model.model, 'encoder') and hasattr(model.model.encoder, 'layers'):
            layers = model.model.encoder.layers
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            layers = model.transformer.h
        elif hasattr(model, 'layers'):
            layers = model.layers

        if layers is None:
            print("Warning: Could not find layers for prefetching")
            return
        for i, layer in enumerate(layers):
            if i >= len(layers) - num_to_forward_prefetch:
                break
            layers_to_prefetch = [layers[i + j] for j in range(1, num_to_forward_prefetch + 1)]
            if hasattr(layer, 'set_modules_to_forward_prefetch'):
                layer.set_modules_to_forward_prefetch(layers_to_prefetch)
    except Exception as e:
        print(f"❌ Failed to set forward prefetch: {e}")
        raise

def set_modules_to_backward_prefetch(model, num_to_backward_prefetch):
    """Configures the model layers to prefetch activations for the previous N layers during the backward pass."""
    try:
        layers = None
        if hasattr(model, 'model') and hasattr(model.model, 'decoder') and hasattr(model.model.decoder, 'layers'):
            layers = model.model.decoder.layers
        elif hasattr(model, 'model') and hasattr(model.model, 'encoder') and hasattr(model.model.encoder, 'layers'):
            layers = model.model.encoder.layers
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            layers = model.transformer.h
        elif hasattr(model, 'layers'):
            layers = model.layers

        if layers is None:
            print("Warning: Could not find layers for prefetching")
            return

        for i, layer in enumerate(layers):
            if i < num_to_backward_prefetch:
                continue
            layers_to_prefetch = [layers[i - j] for j in range(1, num_to_backward_prefetch + 1)]
            if hasattr(layer, 'set_modules_to_backward_prefetch'):
                layer.set_modules_to_backward_prefetch(layers_to_prefetch)
    except Exception as e:
        print(f"❌ Failed to set backward prefetch: {e}")
        raise
    
def get_model_layers(model):

    """Returns (layers, layer_type_name) or (None, None) if not found.""" 
    # Unwrap PeftModel safely if present
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        model = model.base_model.model
        
    if hasattr(model, 'model') and hasattr(model.model, 'decoder') and hasattr(model.model.decoder, 'layers'):
        return model.model.decoder.layers, type(model.model.decoder.layers[0]).__name__
    elif hasattr(model, 'model') and hasattr(model.model, 'encoder') and hasattr(model.model.encoder, 'layers'):
        return model.model.encoder.layers, type(model.model.encoder.layers[0]).__name__
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h, type(model.transformer.h[0]).__name__
    elif hasattr(model, 'layers'):
        return model.layers, type(model.layers[0]).__name__
    
    return None, None


def _dtype_from_name(dtype_name: str):
    # Validate config dtype strings early so downstream quantization and mixed precision paths
    # always receive a real torch dtype object.
    if dtype_name not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype '{dtype_name}'. Expected one of {list(DTYPE_MAP.keys())}")
    return DTYPE_MAP[dtype_name]


def _checkpoint_run_tag(args):
    # Encode run mode in checkpoint folder names (for example: __qlora_q4)
    # to distinguish checkpoints created under different adapter/quant settings.
    tags = []
    if getattr(args, "peft_enabled", False):
        tags.append(getattr(args, "peft_type", "lora"))
    if getattr(args, "quantization_enabled", False):
        tags.append(f"q{getattr(args, 'quantization_bits', 4)}")
    if not tags:
        return ""
    return "__" + "_".join(tags)


def _get_auto_model_class(model_type: str):
    """Returns the appropriate AutoModel class for the given model_type."""
    return {
        "llm":     AutoModelForCausalLM,
        "seq2seq": AutoModelForSeq2SeqLM,
        "vision":  AutoModelForImageClassification,
        "yolo":    AutoModelForObjectDetection,
        "vlm":     AutoModelForImageTextToText,
        "encoder": AutoModel,
    }.get(model_type, AutoModelForCausalLM)


def _check_checkpoint_peft_compat(state_dict: dict, peft_enabled: bool, resume_path: str):
    if not state_dict:
        return  # empty — let load_state_dict produce its own error

    first_key = next(iter(state_dict))
    ckpt_has_peft = first_key.startswith("base_model.model.")

    if ckpt_has_peft and not peft_enabled:
        raise ValueError(
            f"Checkpoint '{resume_path}' was saved WITH PEFT adapters "
            f"(keys start with 'base_model.model.') but current config has peft.enabled=false.\n"
            f"  → Enable peft.enabled=true to match the saved checkpoint, "
            f"or point to a non-PEFT checkpoint."
        )
    if not ckpt_has_peft and peft_enabled:
        raise ValueError(
            f"Checkpoint '{resume_path}' was saved WITHOUT PEFT adapters "
            f"(first key: '{first_key[:60]}') but current config has peft.enabled=true.\n"
            f"  → Disable peft.enabled=false to match the saved checkpoint, "
            f"or point to a PEFT checkpoint."
        )


# REPLACE the entire function with:
def _build_quantization_config(args, rank):
    if not getattr(args, "quantization_enabled", False):
        return None

    bits = int(getattr(args, "quantization_bits", 4))
    compute_dtype = DTYPE_MAP.get(
        getattr(args, "quantization_compute_dtype", "bfloat16"), torch.bfloat16
    )
    double_quant = bool(getattr(args, "quantization_double_quant", True))

    if bits == 4:
        print_on_rank_0(rank, f"Using bitsandbytes NF4 4-bit quantization (double_quant={double_quant})", "🧮")
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=getattr(args, "quantization_type", "nf4"),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=double_quant,
        )
    else:  # 8-bit
        print_on_rank_0(rank, "Using bitsandbytes LLM.int8() 8-bit quantization", "🧮")
        return BitsAndBytesConfig(load_in_8bit=True)


def _normalize_target_modules(target_modules):
    # Accept all supported user input formats from config.yaml and normalize into
    # what PEFT expects (single token, list, or "all-linear").
    if isinstance(target_modules, str):
        val = target_modules.strip()
        if "," in val:
            return [m.strip() for m in val.split(",") if m.strip()]
        return val
    if isinstance(target_modules, list):
        return target_modules
    return "all-linear"


def _apply_peft_quantization(model, args, rank):
    peft_enabled = getattr(args, "peft_enabled", False)
    quantization_enabled = getattr(args, "quantization_enabled", False)

    gradient_checkpointing = bool(getattr(args, "gradient_checkpointing", True))

    if peft_enabled:
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training  # type: ignore

        # If quantized, prepare the model first (handles gradient checkpointing internally)
        if quantization_enabled:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=gradient_checkpointing,
            )
            print_on_rank_0(rank, "Model prepared for k-bit training (prepare_model_for_kbit_training) ✓", "🔧")
        else:
            # Non-quantized path: handle gradient checkpointing manually
            if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
                print_on_rank_0(rank, "Gradient checkpointing enabled", "💾")

        _peft_task_type_map = {
            "llm":     TaskType.CAUSAL_LM,
            "seq2seq": TaskType.SEQ_2_SEQ_LM,
            "vision":  TaskType.FEATURE_EXTRACTION,
            "yolo":    TaskType.FEATURE_EXTRACTION,
            "vlm":     TaskType.CAUSAL_LM,
            "encoder": TaskType.FEATURE_EXTRACTION,
        }
        _task_type = _peft_task_type_map.get(getattr(args, "model_type", "llm"), TaskType.CAUSAL_LM)
        try:
            peft_cfg = LoraConfig(
                r=int(getattr(args, "peft_r", 16)),
                lora_alpha=int(getattr(args, "peft_alpha", 32)),
                lora_dropout=float(getattr(args, "peft_dropout", 0.05)),
                target_modules=_normalize_target_modules(getattr(args, "peft_target_modules", "all-linear")),
                bias=str(getattr(args, "peft_bias", "none")), # type: ignore
                task_type=_task_type,
            )
            model = get_peft_model(model, peft_cfg)
            if not quantization_enabled:
                target_dtype = DTYPE_MAP.get(getattr(args, "param_dtype", "float32"), torch.float32)
                for param in model.parameters():
                    if param.is_floating_point():
                        param.data = param.data.to(target_dtype)
            if rank == 0 and hasattr(model, "print_trainable_parameters"):
                model.print_trainable_parameters()
            print_on_rank_0(rank, f"PEFT adapter attached ({getattr(args, 'peft_type', 'lora')}) ✓", "🧩")
        except Exception as e:
            print_on_rank_0(rank, f"Failed to apply PEFT: {e}", "❌")
            traceback.print_exc()
            raise

    else:
        if quantization_enabled:
            print_on_rank_0(
                rank,
                "8-bit quantization without PEFT: training will proceed but NaN losses are likely. "
                "Only float16 layers (embeddings, norms, lm_head) are trainable — "
                "INT8 linear weights are frozen by bitsandbytes. "
                "See startup warning for details.",
                "⚠️",
            )
        # Non-quantized, no-PEFT path: handle gradient checkpointing if requested.
        elif gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            print_on_rank_0(rank, "Gradient checkpointing enabled", "💾")

    return model


######################---------------- Apply  policies ######################---------------- ##

def apply_solo(device, rank, args):
    """Loads a single-process model and moves it to the selected device."""
    try:
        if getattr(args, "model_type", "llm") == "custom_transformer":
            from model import Transformer, ModelArgs
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
            model_kwargs["dtype"] = DTYPE_MAP[args.param_dtype] if args.mixed_precision else torch.float32
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
                    dtype=DTYPE_MAP[args.param_dtype] if args.mixed_precision else torch.float32,
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
            from model import Transformer, ModelArgs
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
                kwargs["dtype"] = DTYPE_MAP[args.param_dtype] if args.mixed_precision else torch.float32
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
                _resume_config = AutoConfig.from_pretrained(args.model_name, token=HF_TOKEN)
                _resume_config.use_cache = False
                _resume_config.tie_word_embeddings = False
                model = _get_auto_model_class(getattr(args, "model_type", "llm")).from_config(
                    _resume_config,
                    dtype=DTYPE_MAP[args.param_dtype] if args.mixed_precision else torch.float32,
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
                dtype=DTYPE_MAP[args.param_dtype] if args.mixed_precision else torch.float32,
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

def apply_fsdp(local_rank, rank, device, args):
    """Instantiates the model on meta device, shards it with FSDP2, applies mixed precision
    and prefetching, then populates weights via one of three paths:
      A. Resume from checkpoint  (--resume --resume-path)
      B. Fresh run from HF       (--load-model-from-hf)
      C. Random init             (no checkpoint dir)
    Returns (model, checkpointer)."""

    def _materialize_meta_buffers(model, device):
        """
        After FSDP weight loading, some non-persistent buffers (e.g. RoPE inv_freq in Llama)
        may still live on the meta device because they are excluded from state_dict() by default.
        This function walks all buffers and recomputes / materializes any that are still on meta.
        Zeroing them out would silently break positional encoding, so we recompute inv_freq
        using the same formula the model uses at init time.
        """
        for name, buf in model.named_buffers():
            *path, attr = name.split(".")
            parent = model
            for part in path:
                parent = getattr(parent, part)

            if attr == "inv_freq":
                # Always recompute inv_freq regardless of device — it is registered
                # with persistent=False so it is NEVER in state_dict and therefore
                # NEVER restored by load_model(). Two cases both land here:
                #   (a) still on meta after load  → must materialize
                #   (b) garbage CUDA value after model.to_empty(device) → must recompute
                # Zeroing would silently corrupt all positional encodings.
                half_dim = buf.shape[0]
                dim = half_dim * 2
                base = getattr(parent, "base", 10000)
                inv_freq = 1.0 / (
                    base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
                )
                setattr(parent, attr, inv_freq)
                print_on_rank_0(rank, f"Recomputed RoPE inv_freq for '{name}' on {device}", "🔧")
            elif buf.device.type == "meta":
                # Other non-persistent meta buffers: materialize as zeros.
                # Add a named case above (like inv_freq) if zeros would be wrong.
                setattr(parent, attr, torch.zeros(buf.shape, dtype=buf.dtype, device=device))
                print_on_rank_0(rank, f"Materialized meta buffer '{name}' as zeros on {device}", "🔧")
            # else: buffer already on the right device — nothing to do

    try:
        # =================================================================================== #
        # PATH 1: Custom Transformer Path (Classic FSDP1/Standard Fallback API)
        # =================================================================================== #
        if getattr(args, "model_type") == "custom_transformer":
            from model import Transformer, ModelArgs
            model_args = ModelArgs(
                n_layers=args.custom_n_layers,
                vocab_size=args.custom_vocab_size,
                max_seq_len=args.custom_max_seq_len,
                dim=args.custom_dim,
                n_heads=args.custom_n_heads,
                dropout_p=args.custom_dropout_p,
            )
            model = Transformer(model_args).to(device)
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from model import TransformerBlock
            for i, layer in enumerate(model.layers):
                model.layers[i] = fully_shard(layer) # type: ignore
            model = fully_shard(model)

            # ADDED: inspect_model to verify sharding on the custom transformer path as well,
            # consistent with PATH 2 and PATH 3.
            inspect_model(model) # type: ignore
            print_on_rank_0(rank, f"Custom Transformer (FSDP) built | layers={args.custom_n_layers} dim={args.custom_dim} heads={args.custom_n_heads} vocab={args.custom_vocab_size} ✓")
            return model, None
        
        
        # =================================================================================== #
        # PATH 2: PEFT & Quantization Path (Requires Materialization Before Wrapping)
        # =================================================================================== #
        use_peft_or_quant = bool(getattr(args, "peft_enabled", False) or getattr(args, "quantization_enabled", False))

        if use_peft_or_quant:
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
                    dtype=DTYPE_MAP[args.param_dtype] if args.mixed_precision else torch.float32,
                )

            elif load_model_from_hf:
                print_on_rank_0(rank, "Fresh PEFT run — rank 0 loading pretrained weights from HuggingFace", "🧠")
                peft_seed_folder = f"{args.checkpoint_dir}/pretrained_seed"
                _ts = [int(time.time() * 1000) if rank == 0 else 0]
                dist.broadcast_object_list(_ts, src=0)
                peft_seed_timestamp = _ts[0]
                peft_seed_subfolder = f"{peft_seed_folder}/fsdp/{'dcp_api' if args.dcp_api else 'dtensor_api'}/{peft_seed_timestamp}"
                peft_seed_path = f"{peft_seed_subfolder}/model_state_dict.pt"

                pretrained_kwargs = {
                    "token": HF_TOKEN,
                    "low_cpu_mem_usage": True,
                    "tie_word_embeddings": False,
                }
                if quant_cfg is not None:
                    pretrained_kwargs["quantization_config"] = quant_cfg
                    pretrained_kwargs["device_map"] = {"": local_rank} if torch.cuda.is_available() else {"": "cpu"}
                else:
                    pretrained_kwargs["dtype"] = DTYPE_MAP[args.param_dtype] if args.mixed_precision else torch.float32

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
                    dtype=DTYPE_MAP[args.param_dtype] if args.mixed_precision else torch.float32,
                )
                model.load_state_dict(
                    torch.load(peft_seed_path, mmap=True, weights_only=True, map_location="cpu")
                )
                print_on_rank_0(rank, "Pretrained seed weights loaded ✓")

            else:
                print_on_rank_0(rank, "No checkpoint dir — random weight init for PEFT path", "⚠️")
                config = AutoConfig.from_pretrained(args.model_name, token=HF_TOKEN)
                config.use_cache = False
                config.tie_word_embeddings = False
                model = _get_auto_model_class(getattr(args, "model_type", "llm")).from_config(
                    config,
                    dtype=DTYPE_MAP[args.param_dtype] if args.mixed_precision else torch.float32,
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

            layers, layer_type = get_model_layers(model)
            if layers is not None:
                print_on_rank_0(rank, f"Sharding {len(layers)} {layer_type} layers (layer-by-layer)...", "🔀")
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

            if args.explicit_prefetching and layers is not None:
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
        
        # =================================================================================== #
        # PATH 3: Standard Non-PEFT / Non-Quantized FSDP Meta-Initialization Route
        # =================================================================================== #
        #         
        ## FSDP Step 1: build model on meta device (no memory)
        print_on_rank_0(rank, "Instantiating model on meta device...", "🧠")
        config = AutoConfig.from_pretrained(args.model_name, token=HF_TOKEN)
        config.use_cache = False
        config.tie_word_embeddings = False

        with torch.device("meta"):
            model = _get_auto_model_class(getattr(args, "model_type", "llm")).from_config(
                config,
                dtype=DTYPE_MAP[args.param_dtype] if args.mixed_precision else torch.float32,
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

        # fsdp_kwargs["mesh"] = DeviceMesh("cuda", mesh_shape=(dist.get_world_size(),), mesh_dim_names=["data_parallel"]) if torch.cuda.is_available() else None
        
        layers, layer_type = get_model_layers(model)
        if layers is not None:
            print_on_rank_0(rank, f"Sharding {len(layers)} {layer_type} layers...", "🔀")
            for layer in layers:
                fully_shard(layer, **fsdp_kwargs)
        else:
            print_on_rank_0(rank, "No individual layers found, sharding root model only", "⚠️")

        fully_shard(model, **fsdp_kwargs)

        inspect_model(model)
        print_on_rank_0(rank, "FSDP sharding applied ✓. Model inspected ✓", "✅")

        if args.explicit_prefetching and layers is not None:
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
                model.to_empty(device=device)
                checkpointer.load_model(model)
                _materialize_meta_buffers(model, device)

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
                    seed_model = _get_auto_model_class(getattr(args, "model_type", "llm")).from_pretrained(
                        args.model_name,
                        token=HF_TOKEN,
                        dtype=DTYPE_MAP[args.param_dtype] if args.mixed_precision else torch.float32,
                        low_cpu_mem_usage=True,
                        tie_word_embeddings=False,
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
            model.to_empty(device=device)
            seed_checkpointer = Checkpointer(folder=pretrained_seed_folder, dcp_api=args.dcp_api)
            seed_checkpointer.load_model(model)
            print_on_rank_0(rank, "Pretrained weights loaded and sharded ✓")

            _materialize_meta_buffers(model, device)

            checkpointer = Checkpointer(folder=args.checkpoint_dir, dcp_api=args.dcp_api, run_tag=_checkpoint_run_tag(args))

        else:
            # PATH C: no checkpoint dir — random init
            print_on_rank_0(rank, "No checkpoint dir — random weight init", "⚠️")
            model.to_empty(device=device)
            if hasattr(model, "init_weights"):
                model.init_weights()
            else:
                for m in model.modules():
                    if hasattr(m, "reset_parameters"):
                        m.reset_parameters()
            checkpointer = None

        return model, checkpointer   


    except ValueError:
        raise  # user config error — let train.py __main__ print it once cleanly
    except Exception as e:
        print(f"\n[rank {rank}] ❌ Failed in apply_fsdp: {e}", flush=True)
        traceback.print_exc()
        cleanup()
        raise

            


def save_checkpoint(strategy, model, optimizer, rank, args, checkpointer: Checkpointer = None): # type: ignore
    """Saves the model checkpoint.
    FSDP2: uses Checkpointer (from checkpoint.py) which supports both DCP and DTensor APIs.
      - DCP path  (--dcp-api): uses get_model/optimizer_state_dict with full_state_dict=True.
      - DTensor path         : gathers sharded tensors manually, rank 0 saves .pt files.
    DDP: rank 0 saves model.module.state_dict() as a single .pt file.
    SOLO: rank 0 saves model.state_dict() as a single .pt file.
    """
    try:
        print_banner_on_rank_0(rank, "SAVING CHECKPOINT")
        if not args.checkpoint_dir:
            raise ValueError("--checkpoint-dir is required when --save is specified")

        if strategy == "fsdp": 
            if checkpointer is None:
                raise ValueError("checkpointer must be provided for FSDP strategy")
            checkpointer.save(model, optimizer) 
            if rank == 0:
                api_path = f"{args.checkpoint_dir}/fsdp/{'dcp_api' if args.dcp_api else 'dtensor_api'}"
                print_on_rank_0(rank, f"FSDP checkpoint saved to {api_path}/ ✓", "🎉")

        elif strategy == "ddp":
            if rank == 0:
                import time as _time
                _ts  = int(_time.time() * 1000)
                _tag = _checkpoint_run_tag(args)   # e.g. "__lora_q4" or ""
                _fname = f"{_ts}{_tag}.pt"
                os.makedirs(args.checkpoint_dir + "/ddp", exist_ok=True)
                checkpoint_path = f"{args.checkpoint_dir}/ddp/{_fname}"
                torch.save(model.module.state_dict(), checkpoint_path)
                print_on_rank_0(rank, f"DDP checkpoint saved to {checkpoint_path} ✓", "🎉")

        elif strategy == "solo":
            import time as _time
            _ts  = int(_time.time() * 1000)
            _tag = _checkpoint_run_tag(args)
            _fname = f"{_ts}{_tag}.pt"
            os.makedirs(args.checkpoint_dir + "/solo", exist_ok=True)
            checkpoint_path = f"{args.checkpoint_dir}/solo/{_fname}"
            torch.save(model.state_dict(), checkpoint_path)
            print_on_rank_0(rank, f"Solo checkpoint saved to {checkpoint_path} ✓", "🎉")

        if dist.is_initialized():
            _local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            dist_barrier(_local_rank)  
            
    except Exception as e:
        print_on_rank_0(rank, f"❌ Failed to save checkpoint: {e}", "❌")
        raise
