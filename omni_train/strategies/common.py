"""What every strategy shares: model classes and dtypes, LoRA/QLoRA, layer detection, saving."""

import os
import traceback

from dotenv import load_dotenv
import torch
import torch.distributed as dist
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageClassification,
    AutoModelForImageTextToText,
    AutoModelForObjectDetection,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
)

from omni_train.checkpoint import Checkpointer
from omni_train.console import print_banner_on_rank_0, print_on_rank_0
from omni_train.distributed import dist_barrier


load_dotenv()


HF_TOKEN = os.getenv("HF_TOKEN")


DTYPE_MAP = {
    "bfloat16": torch.bfloat16, 
    "float32": torch.float32, 
    "float16": torch.float16
}


# Known nested attribute paths that hold a ModuleList of transformer blocks.
# Used as the second-tier fallback in get_model_layers() when the HF
# _no_split_modules hint is empty (e.g. YOLOS).
_KNOWN_LAYER_PATHS = (
    "model.decoder.layers",          # OPT, BART decoder
    "model.encoder.layers",          # BART encoder
    "model.layers",                  # Llama / Mistral / Qwen / Gemma / DeepSeek / Phi
    "decoder.layers",
    "encoder.layers",
    "encoder.layer",                 # BERT (singular)
    "transformer.h",                 # GPT-2
    "vit.encoder.layer",             # ViT, YOLOS
    "bert.encoder.layer",            # BERT (when wrapped)
    "language_model.model.layers",   # LLaVA (language tower)
    "layers",                        # custom transformers
)


def get_model_layers(model):
    """Detect transformer-block stacks for per-layer FSDP sharding.

    Returns list[(ModuleList, layer_type_name)] — one entry per stack.
    Multi-tower models (T5, BART, CLIP, VLMs) return multiple stacks.
    Returns [] when no shardable stack can be detected.

    Detection order:
      1. HF '_no_split_modules' hint — reliable across all HF model families;
         naturally supports multi-tower architectures.
      2. Known attribute paths — covers models with empty _no_split_modules
         (notably YOLOS)
      3. Largest ModuleList by parameter count — last-resort heuristic for
         truly unknown architectures.
    """
    # Unwrap PeftModel safely if present
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        model = model.base_model.model

    # 1) HF _no_split_modules
    no_split = set(getattr(model, "_no_split_modules", None) or [])
    if no_split:
        stacks = []
        for module in model.modules():
            if (
                isinstance(module, torch.nn.ModuleList)
                and len(module) > 0
                and type(module[0]).__name__ in no_split
            ):
                stacks.append((module, type(module[0]).__name__))
        if stacks:
            return stacks

    # 2) Known attribute paths
    for path in _KNOWN_LAYER_PATHS:
        target = model
        ok = True
        for part in path.split("."):
            if not hasattr(target, part):
                ok = False
                break
            target = getattr(target, part)
        if ok and isinstance(target, torch.nn.ModuleList) and len(target) > 0:
            return [(target, type(target[0]).__name__)]

    # 3) Heuristic: largest ModuleList by parameter count
    best, best_params = None, 0
    for m in model.modules():
        if isinstance(m, torch.nn.ModuleList) and len(m) > 1:
            params = sum(p.numel() for p in m.parameters())
            if params > best_params:
                best, best_params = m, params
    return [(best, type(best[0]).__name__)] if best is not None else []


def _load_dtype(args):
    """dtype to build/load model weights in.

    Weights the optimizer updates must stay fp32: bf16 has ~3 significant digits, so
    at fine-tuning learning rates (e.g. 2e-5) most AdamW steps round to zero. Mixed
    precision is applied at *compute* time instead — FSDP2 via MixedPrecisionPolicy,
    solo/DDP via torch.autocast in train.py.

    Only exception: PEFT outside FSDP. The base weights are frozen, so they can be
    stored in param_dtype to save memory; PEFT keeps the adapters in fp32
    (get_peft_model's autocast_adapter_dtype). FSDP2 requires one dtype per param
    group, so FSDP + PEFT keeps everything in fp32.
    """
    if (
        getattr(args, "mixed_precision", False)
        and getattr(args, "peft_enabled", False)
        and getattr(args, "strategy", None) != "fsdp"
    ):
        return DTYPE_MAP[args.param_dtype]
    return torch.float32


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
                r=int(getattr(args, "peft_r", 16)), # type: ignore
                lora_alpha=int(getattr(args, "peft_alpha", 32)), # type: ignore
                lora_dropout=float(getattr(args, "peft_dropout", 0.05)), # type: ignore
                target_modules=_normalize_target_modules(getattr(args, "peft_target_modules", "all-linear")),
                bias=str(getattr(args, "peft_bias", "none")), # type: ignore
                task_type=_task_type,
            )
            # Adapters stay fp32 (autocast_adapter_dtype=True upcasts them when the base
            # is bf16/fp16) so their optimizer updates are not rounded away.
            model = get_peft_model(model, peft_cfg)
            if rank == 0 and hasattr(model, "print_trainable_parameters"):
                model.print_trainable_parameters()
            print_on_rank_0(rank, f"PEFT adapter attached ({getattr(args, 'peft_type', 'lora')}) ✓", "🧩")
        except Exception as e:
            print_on_rank_0(rank, f"Failed to apply PEFT: {e}", "❌")
            traceback.print_exc()
            raise

    else:
        if quantization_enabled:
            # bitsandbytes froze the quantized (non-float) weight matrices; only the params
            # it left in float (embeddings, norms, biases) can still receive gradients.
            # Report the exact split so the user sees precisely what will and won't learn.
            total       = sum(p.numel() for p in model.parameters())
            quantized   = sum(p.numel() for p in model.parameters() if not torch.is_floating_point(p))
            trainable   = sum(p.numel() for p in model.parameters() if torch.is_floating_point(p) and p.requires_grad)
            pct         = (100.0 * trainable / total) if total else 0.0
            print_on_rank_0(
                rank,
                f"Quantization froze {quantized:,} quantized weights. Only the non-quantized float "
                f"params (embeddings, norms, biases) will learn: {trainable:,}/{total:,} ({pct:.2f}%). "
                "The quantized weight matrices receive no updates. "
                "Without PEFT this is not real fine-tuning and NaN losses are likely — "
                "set peft.enabled=true for stable 8-bit LoRA.",
                "⚠️",
            )
        # Non-quantized, no-PEFT path: handle gradient checkpointing if requested.
        elif gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            print_on_rank_0(rank, "Gradient checkpointing enabled", "💾")

    return model


def save_checkpoint(strategy, model, optimizer, rank, args, checkpointer: Checkpointer = None): # type: ignore
    """Saves the model checkpoint.
    FSDP2: uses Checkpointer (from checkpoint.py) which supports both DCP and DTensor APIs.
      - DCP path  (--dcp-api): uses get_model/optimizer_state_dict with full_state_dict=True.
      - DTensor path         : gathers sharded tensors manually, rank 0 saves .pt files.
    DDP / SOLO: rank 0 saves one .pt file {"model_state_dict", "optimizer_state_dict"}
      (DDP: model.module). Loaders also accept older files holding a bare state dict.
    """
    checkpoint_path = None   # returned on rank 0: the .pt file, or the FSDP checkpoint folder
    try:
        print_banner_on_rank_0(rank, "SAVING CHECKPOINT")
        if not args.checkpoint_dir:
            raise ValueError("--checkpoint-dir is required when --save is specified")

        if strategy == "fsdp": 
            if checkpointer is None:
                raise ValueError("checkpointer must be provided for FSDP strategy")
            checkpointer.save(model, optimizer) 
            checkpoint_path = checkpointer.saved_folder
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
                torch.save({"model_state_dict": model.module.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict()}, checkpoint_path)
                print_on_rank_0(rank, f"DDP checkpoint saved to {checkpoint_path} ✓", "🎉")

        elif strategy == "solo":
            import time as _time
            _ts  = int(_time.time() * 1000)
            _tag = _checkpoint_run_tag(args)
            _fname = f"{_ts}{_tag}.pt"
            os.makedirs(args.checkpoint_dir + "/solo", exist_ok=True)
            checkpoint_path = f"{args.checkpoint_dir}/solo/{_fname}"
            torch.save({"model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict()}, checkpoint_path)
            print_on_rank_0(rank, f"Solo checkpoint saved to {checkpoint_path} ✓", "🎉")

        if dist.is_initialized():
            _local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            dist_barrier(_local_rank)  
        return checkpoint_path

    except Exception as e:
        print_on_rank_0(rank, f"❌ Failed to save checkpoint: {e}", "❌")
        raise
