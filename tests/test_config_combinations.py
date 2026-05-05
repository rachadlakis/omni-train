"""
Combinatorial tests for build_args — covers all meaningful cross-product combinations
of config.yaml variables (model/dataset held fixed via the shared cfg fixture).
"""
import copy
import pytest
from utils import build_args


# ======================================================================
# Strategy × PEFT × Quantization  (the main constraint matrix)
# ======================================================================
#
#  legend:  P = peft_enabled, Q = quant_enabled
#
#  strategy | P | Q | expected
#  ---------+---+---+-----------------------------
#  solo     | F | F | valid
#  solo     | T | F | valid
#  solo     | T | T | valid
#  solo     | F | T | ValueError (quant needs peft)
#  ddp      | F | F | valid
#  ddp      | T | F | valid
#  ddp      | T | T | valid
#  ddp      | F | T | ValueError (quant needs peft)
#  fsdp     | F | F | valid
#  fsdp     | T | F | valid
#  fsdp     | T | T | SystemExit(1) (fsdp + quant blocked)
#  fsdp     | F | T | ValueError (quant needs peft, checked first)

@pytest.mark.parametrize("strategy", ["solo", "ddp", "fsdp"])
def test_strategy_no_peft_no_quant(cfg, strategy):
    cfg["strategy"] = strategy
    args = build_args(cfg)
    assert args.strategy == strategy
    assert args.peft_enabled is False
    assert args.quantization_enabled is False


@pytest.mark.parametrize("strategy", ["solo", "ddp", "fsdp"])
def test_strategy_peft_only(cfg, strategy):
    cfg["strategy"] = strategy
    cfg["peft"]["enabled"] = True
    args = build_args(cfg)
    assert args.peft_enabled is True
    assert args.quantization_enabled is False


@pytest.mark.parametrize("strategy", ["solo", "ddp"])
def test_strategy_peft_and_quant_valid(cfg, strategy):
    """DDP and solo both allow peft + quant."""
    cfg["strategy"] = strategy
    cfg["peft"]["enabled"] = True
    cfg["quantization"]["enabled"] = True
    args = build_args(cfg)
    assert args.peft_enabled is True
    assert args.quantization_enabled is True


def test_fsdp_peft_and_quant_exits(cfg):
    """FSDP + quantization must exit with code 1."""
    cfg["strategy"] = "fsdp"
    cfg["peft"]["enabled"] = True
    cfg["quantization"]["enabled"] = True
    with pytest.raises(SystemExit) as exc_info:
        build_args(cfg)
    assert exc_info.value.code == 1


@pytest.mark.parametrize("strategy", ["solo", "ddp", "fsdp"])
def test_quant_without_peft_raises(cfg, strategy):
    """quantization_enabled=True always requires peft_enabled=True."""
    cfg["strategy"] = strategy
    cfg["peft"]["enabled"] = False
    cfg["quantization"]["enabled"] = True
    with pytest.raises((ValueError, SystemExit)):
        # fsdp path hits sys.exit; solo/ddp path hits ValueError
        build_args(cfg)


# ======================================================================
# PEFT type × quantization bits
# ======================================================================

@pytest.mark.parametrize("bits", [4, 8])
def test_lora_with_any_bits(cfg, bits):
    """LoRA does not constrain quantization bits (when quant is enabled separately)."""
    cfg["strategy"] = "ddp"  # quant+fsdp is blocked; use ddp
    cfg["peft"]["enabled"] = True
    cfg["peft"]["type"] = "lora"
    cfg["quantization"]["enabled"] = True
    cfg["quantization"]["bits"] = bits
    args = build_args(cfg)
    assert args.peft_type == "lora"
    assert args.quantization_bits == bits


def test_qlora_4bit_valid(cfg):
    """qlora implicitly enables peft+quant and is valid at bits=4."""
    cfg["strategy"] = "ddp"  # quant+fsdp is blocked; use ddp
    cfg["peft"]["type"] = "qlora"
    cfg["quantization"]["bits"] = 4
    args = build_args(cfg)
    assert args.peft_enabled is True
    assert args.quantization_enabled is True
    assert args.quantization_bits == 4


def test_qlora_8bit_raises(cfg):
    cfg["peft"]["type"] = "qlora"
    cfg["quantization"]["bits"] = 8
    with pytest.raises(ValueError, match="QLoRA requires"):
        build_args(cfg)


# ======================================================================
# PEFT configuration fields
# ======================================================================

@pytest.mark.parametrize("bias", ["none", "all", "lora_only"])
def test_peft_bias_values(cfg, bias):
    cfg["peft"]["enabled"] = True
    cfg["peft"]["bias"] = bias
    args = build_args(cfg)
    assert args.peft_bias == bias


@pytest.mark.parametrize("r,alpha", [(4, 8), (8, 16), (16, 32), (64, 128)])
def test_peft_rank_alpha_combinations(cfg, r, alpha):
    cfg["peft"]["enabled"] = True
    cfg["peft"]["r"] = r
    cfg["peft"]["alpha"] = alpha
    args = build_args(cfg)
    assert args.peft_r == r
    assert args.peft_alpha == alpha


@pytest.mark.parametrize("target_modules", [
    "all-linear",
    "q_proj",
    "q_proj, k_proj, v_proj",
    ["q_proj", "k_proj"],
])
def test_peft_target_modules_formats(cfg, target_modules):
    cfg["peft"]["enabled"] = True
    cfg["peft"]["target_modules"] = target_modules
    args = build_args(cfg)
    assert args.peft_target_modules == target_modules


@pytest.mark.parametrize("dropout", [0.0, 0.05, 0.1, 0.3])
def test_peft_dropout_values(cfg, dropout):
    cfg["peft"]["enabled"] = True
    cfg["peft"]["dropout"] = dropout
    args = build_args(cfg)
    assert abs(args.peft_dropout - dropout) < 1e-9


# ======================================================================
# Quantization configuration fields
# ======================================================================

@pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
def test_quantization_quant_type(cfg, quant_type):
    cfg["strategy"] = "ddp"
    cfg["peft"]["enabled"] = True
    cfg["quantization"]["enabled"] = True
    cfg["quantization"]["quant_type"] = quant_type
    args = build_args(cfg)
    assert args.quantization_type == quant_type


@pytest.mark.parametrize("compute_dtype", ["bfloat16", "float32", "float16"])
def test_quantization_compute_dtype(cfg, compute_dtype):
    cfg["strategy"] = "ddp"
    cfg["peft"]["enabled"] = True
    cfg["quantization"]["enabled"] = True
    cfg["quantization"]["compute_dtype"] = compute_dtype
    args = build_args(cfg)
    assert args.quantization_compute_dtype == compute_dtype


@pytest.mark.parametrize("double_quant", [True, False])
def test_quantization_double_quant(cfg, double_quant):
    cfg["strategy"] = "ddp"
    cfg["peft"]["enabled"] = True
    cfg["quantization"]["enabled"] = True
    cfg["quantization"]["double_quant"] = double_quant
    args = build_args(cfg)
    assert args.quantization_double_quant is double_quant


# ======================================================================
# Mixed precision × dtype fields
# ======================================================================

@pytest.mark.parametrize("mixed_precision", [True, False])
def test_mixed_precision_flag(cfg, mixed_precision):
    cfg["dist_parameters"]["mixed_precision"] = mixed_precision
    args = build_args(cfg)
    assert args.mixed_precision is mixed_precision


@pytest.mark.parametrize("dtype", ["bfloat16", "float32", "float16"])
def test_param_dtype_values(cfg, dtype):
    cfg["dist_parameters"]["param_dtype"] = dtype
    args = build_args(cfg)
    assert args.param_dtype == dtype


@pytest.mark.parametrize("dtype", ["bfloat16", "float32", "float16"])
def test_reduce_dtype_values(cfg, dtype):
    cfg["dist_parameters"]["reduce_dtype"] = dtype
    args = build_args(cfg)
    assert args.reduce_dtype == dtype


@pytest.mark.parametrize("dtype", ["bfloat16", "float32", "float16"])
def test_output_dtype_values(cfg, dtype):
    cfg["dist_parameters"]["output_dtype"] = dtype
    args = build_args(cfg)
    assert args.output_dtype == dtype


@pytest.mark.parametrize("cast", [True, False])
def test_cast_forward_inputs(cfg, cast):
    cfg["dist_parameters"]["cast_forward_inputs"] = cast
    args = build_args(cfg)
    assert args.cast_forward_inputs is cast


# ======================================================================
# distribute_api → dcp_api / dtensor_api booleans
# ======================================================================

def test_distribute_api_dcp(cfg):
    cfg["dist_parameters"]["distribute_api"] = "dcp_api"
    args = build_args(cfg)
    assert args.dcp_api is True
    assert args.dtensor_api is False


def test_distribute_api_dtensor(cfg):
    cfg["dist_parameters"]["distribute_api"] = "dtensor_api"
    args = build_args(cfg)
    assert args.dcp_api is False
    assert args.dtensor_api is True


# ======================================================================
# Save / Load / Resume flags
# ======================================================================

@pytest.mark.parametrize("save", [True, False])
def test_save_flag(cfg, save):
    cfg["save"] = save
    args = build_args(cfg)
    assert args.save is save


@pytest.mark.parametrize("resume,path", [
    (False, ""),
    (False, "/some/path"),
    (True,  ""),
    (True,  "/some/path"),
])
def test_resume_combinations(cfg, resume, path):
    cfg["save_load"]["resume"] = resume
    cfg["save_load"]["resume_path"] = path
    args = build_args(cfg)
    assert args.resume is resume
    assert args.resume_path == path


@pytest.mark.parametrize("load_from_hf", [True, False])
def test_load_model_from_hf(cfg, load_from_hf):
    cfg["save_load"]["load_model_from_hf"] = load_from_hf
    args = build_args(cfg)
    assert args.load_model_from_hf is load_from_hf


# ======================================================================
# Gradient checkpointing
# ======================================================================

@pytest.mark.parametrize("gc", [True, False])
def test_gradient_checkpointing(cfg, gc):
    cfg["training"]["gradient_checkpointing"] = gc
    args = build_args(cfg)
    assert args.gradient_checkpointing is gc


# ======================================================================
# Prefetch combinations
# ======================================================================

@pytest.mark.parametrize("explicit", [True, False])
def test_explicit_prefetching_flag(cfg, explicit):
    cfg["prefetch"]["explicit"] = explicit
    args = build_args(cfg)
    assert args.explicit_prefetching is explicit


@pytest.mark.parametrize("forward,backward", [(1, 1), (2, 2), (1, 2), (3, 1)])
def test_prefetch_forward_backward(cfg, forward, backward):
    cfg["prefetch"]["forward"] = forward
    cfg["prefetch"]["backward"] = backward
    args = build_args(cfg)
    assert args.forward_prefetch == forward
    assert args.backward_prefetch == backward


# ======================================================================
# to_bool — all truthy / falsy string representations
# ======================================================================

@pytest.mark.parametrize("truthy", ["true", "True", "TRUE", "1", "yes", "y", "on"])
def test_to_bool_truthy_strings(cfg, truthy):
    cfg["save"] = truthy
    args = build_args(cfg)
    assert args.save is True


@pytest.mark.parametrize("falsy", ["false", "False", "FALSE", "0", "no", "n", "off"])
def test_to_bool_falsy_strings(cfg, falsy):
    cfg["save"] = falsy
    args = build_args(cfg)
    assert args.save is False
