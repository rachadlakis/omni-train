"""Training config: load the YAML and turn it into a validated Args (build_args)."""

import os
import sys

import yaml


CONFIG_PATH = os.getenv("CONFIG_PATH", "config.yaml")


def load_config():
    try:
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {CONFIG_PATH}")
        sys.exit(1)


class Args:
    def __init__(self):
        self.model_name = None
        self.model_type: str  # llm, seq2seq, vision, yolo, vlm, encoder

        self.dataset = None
        self.dataset_full_name = None
        self.dataset_text_field = "text"
        self.dataset_split = None

        self.strategy = None # fsdp, ddp, solo, 5d, or auto (see advisor.py)
        self.checkpoint_dir = None
        self.save: bool 

        self.num_gpus: int
        self.epochs: int 
        self.batch_size: int 
        self.max_length: int  
        self.learning_rate: float  
        self.warmup_steps: int
        self.lr_scheduler: str
        self.weight_decay: float
        self.grad_clip: float
        self.gradient_checkpointing: bool

        self.mixed_precision: bool 
        self.param_dtype = None
        self.reduce_dtype = None
        self.output_dtype = None
        self.cast_forward_inputs: bool 

        self.distribute_api: str
        self.dcp_api: bool 
        self.dtensor_api: bool
        self.resume: bool 
        self.resume_path = None
        self.export_hf: bool = False
        self.export_dtype: str = "bfloat16"

        self.explicit_prefetching: bool 
        self.forward_prefetch: int 
        self.backward_prefetch: int

        self.load_model_from_hf: bool   

        self.peft_enabled: bool
        self.peft_type: str
        self.peft_r: int
        self.peft_alpha: int
        self.peft_dropout: float
        self.peft_target_modules = None
        self.peft_bias: str

        self.quantization_enabled: bool
        self.quantization_bits: int
        self.quantization_type: str
        self.quantization_compute_dtype: str
        self.quantization_double_quant: bool

        # Custom Transformer architecture (only used when model_type == "custom_transformer")
        self.custom_n_layers: int = 2
        self.custom_vocab_size: int = 8
        self.custom_max_seq_len: int = 16
        self.custom_dim: int = 16
        self.custom_n_heads: int = 4
        self.custom_dropout_p: float = 0.1
        self.custom_n_kv_heads = None   # None = n_heads (5d only: grouped-query attention)
        self.custom_ffn_hidden = None   # None = ~8/3 * dim (5d only)
        self.custom_num_experts: int = 0  # >0 = Mixture-of-Experts (5d only)
        self.custom_top_k: int = 2

        # 5D parallelism (only used when strategy == "5d")
        self.par_dp = None              # None = auto: world_size / (tp * cp * pp)
        self.par_tp: int = 1
        self.par_sp: bool = False
        self.par_cp: int = 1
        self.par_pp: int = 1
        self.par_pp_schedule: str = "1f1b"
        self.par_micro_batches: int = 1
        self.par_ep: int = 1
        self.par_zero1: bool = False
        self.seed: int = 0

        # Post-training (`algorithm:` block, parsed by post_training.config)
        self.algorithm: str = "pretrain"  # pretrain | sft | dpo | simpo | orpo
        self.algo_beta: float = 0.1
        self.algo_dpo_loss: str = "sigmoid"
        self.algo_label_smoothing: float = 0.0
        self.algo_simpo_gamma: float = 0.5
        self.algo_max_prompt_length: int = 0
        self.algo_completion_only: bool = True

        self.wandb_log_with_train: bool
        self.wandb_entity: str
        self.wandb_project: str
        self.wandb_run_name: str


def to_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.lower() in ["true", "1", "yes", "y", "on"]
    return bool(x)


def build_args(cfg):
    args = Args()

    # --------------------------------------------------
    # MODEL + DATASET
    # --------------------------------------------------
    args.model_type = str(cfg.get("model_type", "llm")).lower()
    args.model_name = cfg.get("model_name") if args.model_type == "custom_transformer" else cfg["model_name"]
    if args.model_type not in {"llm", "seq2seq", "yolo", "vlm", "vision", "encoder", "custom_transformer"}:
        raise ValueError(f"Unsupported model_type={args.model_type}. Use one of: llm, seq2seq, yolo, vlm, vision, encoder, custom_transformer")

    dataset = cfg["dataset"]
    args.dataset = dataset["name"]
    args.dataset_full_name = dataset.get("subset")  # None for datasets without subsets
    args.dataset_text_field = dataset.get("text_field", "text")  # pretrain text column
    args.dataset_split = dataset.get("split", "train")

    # --------------------------------------------------
    # SYSTEM
    # --------------------------------------------------
    # `auto` is filled in before training (advisor.py); here it only has to be a valid config.
    args.strategy = str(cfg["strategy"]).strip().lower()
    if args.strategy not in {"auto", "solo", "ddp", "fsdp", "5d"}:
        raise ValueError(f"Unsupported strategy={args.strategy}. Use one of: auto, solo, ddp, fsdp, 5d")
    num_gpus = cfg.get("num_gpus", 1)
    if str(num_gpus).strip().lower() == "auto":
        if args.strategy not in {"auto", "solo"}:
            raise ValueError(f"num_gpus: auto needs strategy: auto (got strategy: {args.strategy}). Set a number.")
        num_gpus = 1
    args.num_gpus = int(num_gpus)
    args.checkpoint_dir = cfg["checkpoint_dir"]
    args.save = to_bool(cfg["save"])
    args.seed = int(cfg.get("seed", 0))

    # --------------------------------------------------
    # TRAINING
    # --------------------------------------------------
    t = cfg["training"] 
    args.epochs = int(t["epochs"])
    args.batch_size = int(t["batch_size"])
    args.max_length = int(t["max_length"])
    args.learning_rate = float(t["learning_rate"])
    args.warmup_steps = int(t.get("warmup_steps", 100))
    # LR after warmup: cosine/linear decay to 0 over the run, or constant.
    args.lr_scheduler = str(t.get("lr_scheduler", "cosine")).lower()
    args.weight_decay = float(t.get("weight_decay", 0.01))
    args.grad_clip = float(t.get("grad_clip", 1.0))
    # Global activation checkpointing toggle used by both quantized and non-quantized paths.
    args.gradient_checkpointing = to_bool(t.get("gradient_checkpointing", True))

    if args.epochs < 1:
        raise ValueError("epochs must be at least 1")
    if args.batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if args.max_length < 1:
        raise ValueError("max_length must be at least 1")
    if args.warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if args.lr_scheduler not in {"cosine", "linear", "constant"}:
        raise ValueError(f"Unsupported training.lr_scheduler={args.lr_scheduler}. Use one of: cosine, linear, constant")
    if args.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if args.grad_clip < 0:
        raise ValueError("grad_clip must be non-negative")

    # --------------------------------------------------
    # FSDP
    # --------------------------------------------------
    f = cfg["dist_parameters"]
    args.mixed_precision = to_bool(f.get("mixed_precision", False))
    args.param_dtype = f.get("param_dtype", None)
    args.reduce_dtype = f.get("reduce_dtype", None)
    args.output_dtype = f.get("output_dtype", None)
    args.cast_forward_inputs = to_bool(f.get("cast_forward_inputs", False))
    args.distribute_api = f.get("distribute_api", "dcp_api")
    args.dcp_api = args.distribute_api == "dcp_api"
    args.dtensor_api = args.distribute_api == "dtensor_api"

    # --------------------------------------------------
    # SAVE LOAD
    # --------------------------------------------------
    sl = cfg["save_load"]
    args.resume = to_bool(sl.get("resume", False))
    args.resume_path = sl.get("resume_path", None)
    args.load_model_from_hf = to_bool(sl.get("load_model_from_hf", True))
    # Hugging Face export after training (export.py): a folder from_pretrained can load,
    # LoRA merged into the base. QLoRA runs export the adapter (merge: scripts/merge_lora.py).
    args.export_hf = to_bool(sl.get("export_hf", False))
    args.export_dtype = str(sl.get("export_dtype", "bfloat16")).lower()
    if args.export_dtype not in {"bfloat16", "float16", "float32"}:
        raise ValueError(f"save_load.export_dtype must be bfloat16, float16 or float32 (got {args.export_dtype})")

    # --------------------------------------------------
    # PEFT
    # --------------------------------------------------
    # Parse PEFT settings once and pass as strongly typed runtime args.
    peft_cfg = cfg.get("peft", {})
    args.peft_enabled = to_bool(peft_cfg.get("enabled", False))
    args.peft_type = str(peft_cfg.get("type", "lora")).lower()
    args.peft_r = int(peft_cfg.get("r", 16))
    args.peft_alpha = int(peft_cfg.get("alpha", 32))
    args.peft_dropout = float(peft_cfg.get("dropout", 0.05))
    args.peft_target_modules = peft_cfg.get("target_modules", "all-linear")
    args.peft_bias = str(peft_cfg.get("bias", "none"))

    # Restrict supported adapter types to what distributed setup currently implements.
    if args.peft_type not in {"lora", "qlora"}:
        raise ValueError(f"Unsupported peft.type={args.peft_type}. Use one of: lora, qlora")

    _peft_compatible_types = {"llm", "seq2seq", "encoder", "vlm", "vision"}
    if args.peft_enabled and args.model_type not in _peft_compatible_types and args.model_type != "custom_transformer":
        raise ValueError(
            f"PEFT is not supported for model_type='{args.model_type}'.\n"
            f"PEFT (LoRA/QLoRA) requires a transformer architecture. "
            f"Supported types: {sorted(_peft_compatible_types)}"
        )

    # --------------------------------------------------
    # QUANTIZATION
    # --------------------------------------------------
    # Parse quantization block; defaults are QLoRA-friendly but disabled unless explicitly enabled.
    qcfg = cfg.get("quantization", {})
    args.quantization_enabled = to_bool(qcfg.get("enabled", False))
    args.quantization_bits = int(qcfg.get("bits", 4))
    args.quantization_type = str(qcfg.get("quant_type", "nf4")).lower()
    args.quantization_compute_dtype = str(qcfg.get("compute_dtype", "bfloat16")).lower()
    args.quantization_double_quant = to_bool(qcfg.get("double_quant", True))

    # Limit to bit widths currently handled by bitsandbytes wiring in this project.
    if args.quantization_bits not in {4, 8}:
        raise ValueError(f"Unsupported quantization.bits={args.quantization_bits}. Use 4 or 8")

    if args.peft_type == "qlora":
        args.peft_enabled = True
        args.quantization_enabled = True

        # QLoRA supports both 4-bit (NF4) and 8-bit (LLM.int8()) base weights.
        # bits is already validated to {4, 8} above; default stays 4 (canonical QLoRA).
        if args.quantization_bits not in {4, 8}:
            raise ValueError("QLoRA requires quantization.bits=4 or 8")

    if args.quantization_enabled and not args.peft_enabled:
        if args.quantization_bits == 4:
            # 4-bit (NF4) has no differentiable backward — mathematically impossible to train.
            raise ValueError(
                "4-bit quantization requires peft.enabled=true. "
                "Use peft_type=qlora for QLoRA fine-tuning."
            )
        else:
            # CHANGED: was a hard ValueError. Downgraded to a warning so the user can proceed
            # if they choose to, while being clearly informed of the instability.
            # 8-bit training is allowed but will almost certainly produce NaN losses because:
            #   1. INT8 weights have a very limited range (-127 to 127); AdamW gradient updates
            #      overflow that range within a few steps, corrupting weights to NaN.
            #   2. LayerNorm and embedding layers stay in float16 without
            #      prepare_model_for_kbit_training — float16 overflows at ~65504, causing
            #      NaN in the forward pass itself before gradients are even computed.
            #   3. Optimizer states (exp_avg, exp_avg_sq) accumulate in the wrong dtype,
            #      amplifying the instability across steps.
            # Recommended alternatives:
            #   • peft.enabled=true + quantization.bits=8  → stable 8-bit LoRA
            #   • quantization.enabled=false               → full float fine-tuning
            import warnings
            warnings.warn(
                "\n⚠️  WARNING: 8-bit quantization without PEFT will likely produce NaN losses.\n"
                "   Reasons:\n"
                "   1. INT8 weight range is -127 to 127 — AdamW updates overflow it within steps.\n"
                "   2. LayerNorm/embeddings stay in float16, which overflows at ~65504.\n"
                "   3. Optimizer states accumulate in the wrong dtype, amplifying instability.\n"
                "   Alternatives:\n"
                "   • peft.enabled=true  (8-bit LoRA — stable)\n"
                "   • quantization.enabled=false  (full float fine-tuning)\n"
                "   Proceeding anyway...\n",
                UserWarning,
                stacklevel=2,
            )

    if args.strategy == "fsdp" and args.quantization_enabled:
        raise ValueError(
            "FSDP + bitsandbytes quantization is not supported: "
            "bitsandbytes 4-bit/8-bit layers cannot be sharded by FSDP. "
            "Use strategy=ddp to keep quantization, or set quantization.enabled=false to keep FSDP."
        )

    if args.strategy == "fsdp" and args.mixed_precision:
        _check_fsdp_dtypes(args)

    # --------------------------------------------------
    # PREFETCH
    # --------------------------------------------------
    p = cfg["prefetch"]
    args.explicit_prefetching = to_bool(p["explicit"])
    args.forward_prefetch = int(p["forward"])
    args.backward_prefetch = int(p["backward"])

    # --------------------------------------------------
    # CUSTOM TRANSFORMER ARCHITECTURE
    # --------------------------------------------------
    if args.model_type == "custom_transformer":
        cta = cfg.get("custom_transformer_args", {})
        args.custom_n_layers    = int(cta.get("n_layers",    2))
        args.custom_vocab_size  = int(cta.get("vocab_size",  8))
        args.custom_max_seq_len = int(cta.get("max_seq_len", 16))
        args.custom_dim         = int(cta.get("dim",         16))
        args.custom_n_heads     = int(cta.get("n_heads",     4))
        args.custom_dropout_p   = float(cta.get("dropout_p", 0.1))
        if args.custom_dim % args.custom_n_heads != 0:
            raise ValueError(
                f"custom_transformer: dim ({args.custom_dim}) must be divisible by n_heads ({args.custom_n_heads})"
            )
        args.custom_n_kv_heads = int(cta["n_kv_heads"]) if cta.get("n_kv_heads") else None
        args.custom_ffn_hidden = int(cta["ffn_hidden"]) if cta.get("ffn_hidden") else None
        args.custom_num_experts = int(cta.get("num_experts", 0) or 0)
        args.custom_top_k = int(cta.get("top_k", 2))
        if args.strategy not in {"5d", "auto"} and (args.custom_num_experts or args.custom_n_kv_heads):
            raise ValueError("custom_transformer_args.num_experts / n_kv_heads need strategy: 5d")
        if args.custom_num_experts and not 1 <= args.custom_top_k <= args.custom_num_experts:
            raise ValueError("custom_transformer_args.top_k must be between 1 and num_experts")

    # --------------------------------------------------
    # 5D PARALLELISM
    # --------------------------------------------------
    if args.strategy == "5d":
        _parse_5d(cfg, args)

    if args.export_hf and args.strategy == "5d":
        raise ValueError("save_load.export_hf is for solo/ddp/fsdp; strategy: 5d always exports "
                         "an HF checkpoint to checkpoints/5d/<ts>/hf/ when save: true")
    if args.export_hf and args.model_type == "custom_transformer":
        raise ValueError("save_load.export_hf needs a Hugging Face model (custom_transformer is not one)")

    # --------------------------------------------------
    # POST-TRAINING (SFT / DPO / SimPO / ORPO)
    # --------------------------------------------------
    from omni_train.post_training.config import parse_algorithm
    parse_algorithm(cfg, args)

    # --------------------------------------------------
    # WANDB
    # --------------------------------------------------
    wb = cfg["wandb"]
    args.wandb_log_with_train = to_bool(wb.get("wandb_log_with_train", False))
    args.wandb_entity = wb.get("wandb_entity", "")  # blank → your default W&B entity
    args.wandb_project = wb.get("wandb_project", "omni-train")
    args.wandb_run_name = wb.get("wandb_run_name", "")

    return args


# Every model of these types tested under FSDP fails with mismatched dtypes (T5 is the exception).
_FSDP_ONE_DTYPE = {"llm", "encoder", "vision", "custom_transformer"}


def _check_fsdp_dtypes(args):
    """FSDP casts each block's OUTPUT to output_dtype, but the final norm / head gets it with
    weights in param_dtype, so different dtypes crash in the first step ("mat1 and mat2 to have
    the same dtype"). cast_forward_inputs does not help: it only casts the inputs of the model
    and of each sharded block, not what the head receives."""
    param, output = (str(d).lower() if d else None for d in (args.param_dtype, args.output_dtype))
    if param is None or output is None or param == output:
        return
    msg = (f"FSDP with mixed_precision: dist_parameters.output_dtype ({output}) differs from "
           f"param_dtype ({param}), so the model's last layers would get {output} inputs with {param} "
           f"weights. Set output_dtype: {param} (the loss is still computed in float32).")
    if param != "float32" and args.model_type in _FSDP_ONE_DTYPE:
        raise ValueError(msg)
    import warnings  # some models survive it (T5; Llama/Qwen2 with float32 weights)
    warnings.warn(msg + " Most models fail with this setting.", UserWarning, stacklevel=3)


def _parse_5d(cfg, args):
    """Read and check the `parallelism:` block (strategy: 5d)."""
    par = cfg.get("parallelism") or {}
    known = {"dp", "tp", "sp", "cp", "pp", "pp_schedule", "micro_batches", "ep", "zero1"}
    unknown = set(par) - known
    if unknown:
        raise ValueError(f"Unknown parallelism option(s) {sorted(unknown)}. Known: {sorted(known)}")
    dp = par.get("dp", "auto")
    args.par_dp = None if dp in (None, "auto") else int(dp)
    args.par_tp = int(par.get("tp", 1))
    args.par_sp = to_bool(par.get("sp", False))
    args.par_cp = int(par.get("cp", 1))
    args.par_pp = int(par.get("pp", 1))
    args.par_pp_schedule = str(par.get("pp_schedule", "1f1b")).lower()
    args.par_micro_batches = int(par.get("micro_batches", 1))
    args.par_ep = int(par.get("ep", 1))
    args.par_zero1 = to_bool(par.get("zero1", False))

    sizes = {"tp": args.par_tp, "cp": args.par_cp, "pp": args.par_pp, "ep": args.par_ep,
             "micro_batches": args.par_micro_batches}
    if args.par_dp is not None:
        sizes["dp"] = args.par_dp
    for name, value in sizes.items():
        if value < 1:
            raise ValueError(f"parallelism.{name} must be at least 1 (got {value})")
    if args.par_pp_schedule not in {"1f1b", "afab"}:
        raise ValueError(f"parallelism.pp_schedule must be 1f1b or afab (got {args.par_pp_schedule})")
    if args.par_sp and args.par_tp == 1:
        raise ValueError("parallelism.sp (sequence parallelism) needs tp > 1")

    model_par = args.par_tp * args.par_cp * args.par_pp
    if args.num_gpus % model_par:
        raise ValueError(f"num_gpus ({args.num_gpus}) must be divisible by tp*cp*pp ({model_par})")
    dp = args.par_dp or args.num_gpus // model_par
    if dp * model_par != args.num_gpus:
        raise ValueError(f"dp*tp*cp*pp = {dp * model_par} must equal num_gpus ({args.num_gpus})")
    if dp % args.par_ep:
        raise ValueError(f"parallelism.ep ({args.par_ep}) must divide dp ({dp}): experts are spread over DP ranks")

    if args.model_type not in {"llm", "custom_transformer"}:
        raise ValueError(f"strategy: 5d supports model_type llm (Llama/Mistral/Qwen2) and "
                         f"custom_transformer, not '{args.model_type}'. Use ddp or fsdp.")
    if args.peft_enabled or args.quantization_enabled:
        raise ValueError("strategy: 5d does not support PEFT or quantization yet. Use ddp/fsdp for LoRA/QLoRA.")
    if args.par_ep > 1 and not (args.model_type == "custom_transformer" and args.custom_num_experts):
        raise ValueError("parallelism.ep > 1 needs a Mixture-of-Experts model "
                         "(model_type: custom_transformer with custom_transformer_args.num_experts > 0)")

    seq = args.custom_max_seq_len if args.model_type == "custom_transformer" else args.max_length
    if seq % args.par_cp:
        raise ValueError(f"sequence length {seq} must be divisible by cp ({args.par_cp})")
    if args.par_sp and (seq // args.par_cp) % args.par_tp:
        raise ValueError(f"with sp, sequence length {seq} must be divisible by cp*tp ({args.par_cp * args.par_tp})")
    if args.par_pp > 1 and args.par_micro_batches < args.par_pp:
        import warnings
        warnings.warn(f"parallelism.micro_batches ({args.par_micro_batches}) < pp ({args.par_pp}): "
                      f"the pipeline bubble is (pp-1)/micro_batches, so use micro_batches >= {args.par_pp}",
                      UserWarning, stacklevel=3)
