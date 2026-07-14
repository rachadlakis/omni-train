## dist_kit/config.py ##
##
## Slim, portable replacement for OMNI-Train's full Args/build_args.
## Carries ONLY the fields the distributed wrapping / checkpoint code reads,
## with the same validation rules for the combinations that matter
## (quantization ↔ PEFT ↔ strategy).

import os
from dataclasses import dataclass

import torch

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float16": torch.float16,
}

_KIT_DIR = os.path.dirname(os.path.abspath(__file__))

_VALID_STRATEGIES = {"solo", "ddp", "fsdp"}
_VALID_APIS = {"dcp_api", "dtensor_api"}
_VALID_MODEL_TYPES = {"llm", "seq2seq", "vision", "yolo", "vlm", "encoder"}
_PEFT_MODEL_TYPES = {"llm", "seq2seq", "encoder", "vlm", "vision"}


@dataclass
class DistConfig:
    """Everything the dist kit needs to wrap, train, and checkpoint a model."""

    # ---- model ----
    model_name: str = "sshleifer/tiny-gpt2"   # any HF hub id
    model_type: str = "llm"                   # llm | seq2seq | vision | yolo | vlm | encoder
    strategy: str = "fsdp"                    # solo | ddp | fsdp
    num_gpus: int = 1

    # ---- dataset (text types only in the kit; empty name → random-batch demo) ----
    dataset_name: str = ""                    # HF datasets hub id, e.g. "wikitext"
    dataset_subset: str = ""                  # subset name, e.g. "wikitext-2-raw-v1"
    dataset_split: str = "train[:2%]"         # split slice, e.g. "train[:2%]+validation[:3%]"

    # ---- training hyperparameters ----
    epochs: int = 1
    batch_size: int = 8
    max_length: int = 64
    learning_rate: float = 2e-5
    warmup_steps: int = 10
    weight_decay: float = 0.01

    # ---- checkpointing ----
    checkpoint_dir: str = "checkpoints"
    save: bool = True
    resume: bool = False
    resume_path: str = ""                     # required when resume=True
    load_model_from_hf: bool = True           # False → random init (debug only)
    distribute_api: str = "dcp_api"           # dcp_api | dtensor_api (FSDP only)

    # ---- precision ----
    mixed_precision: bool = True
    param_dtype: str = "bfloat16"
    reduce_dtype: str = "float32"
    output_dtype: str = "bfloat16"
    cast_forward_inputs: bool = True

    # ---- performance ----
    gradient_checkpointing: bool = True
    explicit_prefetching: bool = False
    forward_prefetch: int = 2
    backward_prefetch: int = 2

    # ---- PEFT (LoRA / QLoRA) ----
    peft_enabled: bool = False
    peft_type: str = "lora"                   # lora | qlora
    peft_r: int = 16
    peft_alpha: int = 32
    peft_dropout: float = 0.05
    peft_target_modules: object = "all-linear"  # str, "a,b,c", or list
    peft_bias: str = "none"

    # ---- quantization (bitsandbytes) ----
    quantization_enabled: bool = False
    quantization_bits: int = 4                # 4 (NF4) | 8 (LLM.int8())
    quantization_type: str = "nf4"
    quantization_compute_dtype: str = "bfloat16"
    quantization_double_quant: bool = True

    # ------------------------------------------------------------------
    @property
    def dcp_api(self) -> bool:
        return self.distribute_api == "dcp_api"

    # ------------------------------------------------------------------
    def __post_init__(self):
        # Coerce numerics — PyYAML parses "2e-5" (no dot) as a string.
        self.epochs = int(self.epochs)
        self.batch_size = int(self.batch_size)
        self.max_length = int(self.max_length)
        self.warmup_steps = int(self.warmup_steps)
        self.learning_rate = float(self.learning_rate)
        self.weight_decay = float(self.weight_decay)

        if self.strategy not in _VALID_STRATEGIES:
            raise ValueError(f"strategy='{self.strategy}' — expected one of {sorted(_VALID_STRATEGIES)}")
        if self.distribute_api not in _VALID_APIS:
            raise ValueError(f"distribute_api='{self.distribute_api}' — expected one of {sorted(_VALID_APIS)}")
        if self.model_type not in _VALID_MODEL_TYPES:
            raise ValueError(f"model_type='{self.model_type}' — expected one of {sorted(_VALID_MODEL_TYPES)}")

        for name in ("param_dtype", "reduce_dtype", "output_dtype", "quantization_compute_dtype"):
            val = getattr(self, name)
            if val not in DTYPE_MAP:
                raise ValueError(f"{name}='{val}' — expected one of {list(DTYPE_MAP.keys())}")

        # qlora is just lora + quantization; enabling it implies quantization.
        if self.peft_enabled and self.peft_type == "qlora":
            self.quantization_enabled = True

        if self.quantization_enabled:
            if self.quantization_bits not in (4, 8):
                raise ValueError(f"quantization_bits={self.quantization_bits} — only 4 and 8 are supported")
            # 4-bit: NF4 has no differentiable backward — weights can't train directly.
            # 8-bit: direct AdamW updates to INT8 weights overflow the quantization
            # range immediately (NaN losses from step 1). Both require adapters.
            if not self.peft_enabled:
                raise ValueError(
                    f"{self.quantization_bits}-bit quantization requires peft_enabled=True — "
                    "quantized base weights cannot be trained directly. Use LoRA/QLoRA."
                )
            # FSDP's sharding mechanism cannot operate on bitsandbytes quantized layers.
            if self.strategy == "fsdp":
                raise ValueError(
                    "strategy='fsdp' is incompatible with bitsandbytes quantization. "
                    "Use strategy='ddp' or 'solo' for quantized runs."
                )

        if self.peft_enabled and self.model_type not in _PEFT_MODEL_TYPES:
            raise ValueError(
                f"peft_enabled=True only supports model types {sorted(_PEFT_MODEL_TYPES)}, "
                f"got '{self.model_type}'"
            )

        if self.resume and not self.resume_path:
            raise ValueError("resume=True requires resume_path to be set")

    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, cfg: dict) -> "DistConfig":
        """Build from a config.yaml-style nested dict — the same schema as
        OMNI-Train's config.yaml (dist_parameters, save_load, prefetch, peft,
        quantization, training.gradient_checkpointing). Unknown keys and
        sections (dataset, wandb, ...) are ignored, so an existing config.yaml
        works as-is. Flat keys matching field names are also accepted and win
        over their nested equivalents."""
        cfg = cfg or {}

        def sect(name):
            val = cfg.get(name) or {}
            return val if isinstance(val, dict) else {}

        dataset   = sect("dataset")
        training  = sect("training")
        dist_p    = sect("dist_parameters")
        save_load = sect("save_load")
        prefetch  = sect("prefetch")
        peft      = sect("peft")
        quant     = sect("quantization")

        values = {
            "model_name": cfg.get("model_name"),
            "model_type": cfg.get("model_type"),
            "strategy": cfg.get("strategy"),
            "num_gpus": cfg.get("num_gpus"),
            "checkpoint_dir": cfg.get("checkpoint_dir"),
            "save": cfg.get("save"),

            "dataset_name": dataset.get("name"),
            "dataset_subset": dataset.get("subset"),
            "dataset_split": dataset.get("split"),

            "epochs": training.get("epochs"),
            "batch_size": training.get("batch_size"),
            "max_length": training.get("max_length"),
            "learning_rate": training.get("learning_rate"),
            "warmup_steps": training.get("warmup_steps"),
            "weight_decay": training.get("weight_decay"),
            "gradient_checkpointing": training.get("gradient_checkpointing"),

            "mixed_precision": dist_p.get("mixed_precision"),
            "param_dtype": dist_p.get("param_dtype"),
            "reduce_dtype": dist_p.get("reduce_dtype"),
            "output_dtype": dist_p.get("output_dtype"),
            "cast_forward_inputs": dist_p.get("cast_forward_inputs"),
            "distribute_api": dist_p.get("distribute_api"),

            "resume": save_load.get("resume"),
            "resume_path": save_load.get("resume_path"),
            "load_model_from_hf": save_load.get("load_model_from_hf"),

            "explicit_prefetching": prefetch.get("explicit"),
            "forward_prefetch": prefetch.get("forward"),
            "backward_prefetch": prefetch.get("backward"),

            "peft_enabled": peft.get("enabled"),
            "peft_type": peft.get("type"),
            "peft_r": peft.get("r"),
            "peft_alpha": peft.get("alpha"),
            "peft_dropout": peft.get("dropout"),
            "peft_target_modules": peft.get("target_modules"),
            "peft_bias": peft.get("bias"),

            "quantization_enabled": quant.get("enabled"),
            "quantization_bits": quant.get("bits"),
            "quantization_type": quant.get("quant_type"),
            "quantization_compute_dtype": quant.get("compute_dtype"),
            "quantization_double_quant": quant.get("double_quant"),
        }

        # Flat keys matching field names still work and win over nested ones.
        known = set(cls.__dataclass_fields__)
        for k, v in cfg.items():
            if k in known:
                values[k] = v

        # Drop unset entries so dataclass defaults apply.
        return cls(**{k: v for k, v in values.items() if v is not None})

    @classmethod
    def from_yaml(cls, path: str = "") -> "DistConfig":
        """Load from a config.yaml-style YAML file (see from_dict).
        With no path, resolves via resolve_config_path()."""
        import yaml
        path = resolve_config_path(path)
        with open(path, "r") as f:
            return cls.from_dict(yaml.safe_load(f) or {})


def resolve_config_path(path: str = "") -> str:
    """Config file resolution: explicit path > $DIST_CONFIG > ./config.yaml
    (project root) > the kit's bundled dist_kit/config.yaml."""
    if path:
        return path
    env = os.getenv("DIST_CONFIG", "")
    if env:
        return env
    if os.path.exists("config.yaml"):
        return "config.yaml"
    return os.path.join(_KIT_DIR, "config.yaml")
