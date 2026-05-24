"""
Smoke tests — launch real training subprocesses for every parameter combination we care about, to catch config errors and basic runtime issues.
These are NOT unit tests — they run the full training script with a tiny config, on real models and datasets, using the actual training loop, optimizers, and distributed strategies. The goal is

Run with:
    python -m pytest -m smoke -v
    python -m pytest -m smoke -v -k "fsdp"   # filter by name

These tests are EXCLUDED from the default pytest run (`-m "not smoke"` in pytest.ini).
Requirements:
  • At least one GPU (CUDA) — tests are skipped automatically on CPU-only machines.
  • HuggingFace access to facebook/opt-125m (public, no token needed).
  • ~30–60 s per case.
"""
import itertools
import os
import sys
import subprocess
import tempfile

import pytest
import torch
import yaml

# ── Shared tiny-run config ────────────────────────────────────────────────────
#
# One epoch, 1% of wikitext, batch=2, seq=32, no save, no wandb.
# Each test case overrides only the fields that differ (strategy, peft, …).
#
BASE_SMOKE_CFG = {
    "model_name": "facebook/opt-125m",
    "model_type": "llm",
    "dataset": {
        "name": "wikitext",
        "subset": "wikitext-2-raw-v1",
        "split": "train[:1%]",
    },
    "strategy": "fsdp",         # overridden per test
    "num_gpus": 1,
    "checkpoint_dir": "",       # filled in with a tempdir per test
    "save": False,
    "training": {
        "epochs": 1,
        "batch_size": 2,
        "max_length": 32,
        "learning_rate": 1e-4,
        "gradient_checkpointing": False,
        "warmup_steps": 0,
        "weight_decay": 0.0,
        "grad_clip": 1.0,
    },
    "dist_parameters": {
        "mixed_precision": False,
        "param_dtype": "float32",
        "reduce_dtype": "float32",
        "output_dtype": "float32",
        "cast_forward_inputs": False,
        "distribute_api": "dcp_api",
    },
    "save_load": {
        "resume": False,
        "resume_path": "",
        "load_model_from_hf": True,
    },
    "peft": {
        "enabled": False,       # overridden per test
        "type": "lora",
        "r": 4,
        "alpha": 8,
        "dropout": 0.0,
        "target_modules": "all-linear",
        "bias": "none",
    },
    "quantization": {
        "enabled": False,
        "bits": 4,
        "quant_type": "nf4",
        "compute_dtype": "bfloat16",
        "double_quant": False,
    },
    "prefetch": {
        "explicit": False,
        "forward": 1,
        "backward": 1,
    },
    "wandb": {
        "wandb_log_with_train": False,
        "wandb_entity": "smoke-test",
        "wandb_project": "smoke-test",
        "wandb_run_name": "smoke",
    },
}

# ── Helpers ───────────────────────────────────────────────────────────────────

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_SCRIPT = os.path.join(REPO_ROOT, "train.py")
PYTHON = sys.executable

def _apply_overrides(cfg: dict, overrides: dict) -> dict:
    """Apply overrides onto a deep copy of cfg.

    Keys can be:
      - a plain string  → top-level field, e.g. "strategy"
      - a tuple         → nested path,     e.g. ("peft", "enabled")

    Reserved keys ("id", "phases", "min_gpus") are silently ignored.
    """
    import copy
    cfg = copy.deepcopy(cfg)
    _reserved = {"id", "phases", "min_gpus"}
    for key, value in overrides.items():
        if key in _reserved:
            continue
        if isinstance(key, tuple):
            node = cfg
            for part in key[:-1]:
                node = node[part]
            node[key[-1]] = value
        else:
            cfg[key] = value
    return cfg


def _find_checkpoint_path(tmpdir: str, strategy: str, distribute_api: str) -> str:
    """Return the resume_path for the checkpoint produced in the previous phase.

    Checkpoint locations (set by save_checkpoint in distributed_utils.py):
      solo  → {tmpdir}/solo/solo_checkpoint.pt
      ddp   → {tmpdir}/ddp/ddp_checkpoint.pt
      fsdp  → {tmpdir}/fsdp/{dcp_api|dtensor_api}/{latest_timestamp}/
    """
    if strategy == "solo":
        return os.path.join(tmpdir, "solo", "solo_checkpoint.pt")
    if strategy == "ddp":
        return os.path.join(tmpdir, "ddp", "ddp_checkpoint.pt")
    if strategy == "fsdp":
        fsdp_dir = os.path.join(tmpdir, "fsdp", distribute_api)
        # Find the highest-timestamp (most recent) checkpoint directory.
        latest = None
        latest_ts = -1
        if os.path.isdir(fsdp_dir):
            for name in os.listdir(fsdp_dir):
                if os.path.isdir(os.path.join(fsdp_dir, name)):
                    try:
                        ts = int(name.split("__", 1)[0])
                        if ts > latest_ts:
                            latest_ts = ts
                            latest = name
                    except ValueError:
                        pass
        if latest is None:
            raise RuntimeError(
                f"No FSDP checkpoint found in {fsdp_dir} after save phase."
            )
        return os.path.join(fsdp_dir, latest)
    raise ValueError(f"Unknown strategy: {strategy!r}")


def _launch(cfg: dict, tmpdir: str, timeout: int) -> subprocess.CompletedProcess:
    """Write cfg to tmpdir and launch train.py as a subprocess."""
    cfg_path = os.path.join(tmpdir, "smoke_cfg.yaml")
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f)

    strategy = cfg["strategy"]
    num_gpus = int(cfg.get("num_gpus", 1))
    env = {**os.environ, "CONFIG_PATH": cfg_path, "HF_HUB_DISABLE_XET": "1"}

    if strategy == "solo":
        cmd = [PYTHON, TRAIN_SCRIPT]
    else:
        cmd = [
            PYTHON, "-m", "torch.distributed.run",
            f"--nproc_per_node={num_gpus}",
            "--master_addr=localhost",
            "--master_port=29600",
            TRAIN_SCRIPT,
        ]

    return subprocess.run(
        cmd,
        env=env,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _run_case(case: dict, timeout: int = 300) -> subprocess.CompletedProcess:
    """Run a smoke case — single-phase or multi-phase — inside a shared tmpdir.

    Multi-phase cases use the "phases" key (list of override dicts).
    Phases share the same checkpoint_dir so phase 2 can resume from phase 1.
    If a phase has save_load.resume = True, resume_path is injected automatically.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        phases = case.get("phases", [case])
        result = None
        for i, phase_overrides in enumerate(phases):
            cfg = _apply_overrides(BASE_SMOKE_CFG, phase_overrides)
            cfg["checkpoint_dir"] = tmpdir

            # Auto-inject resume_path when this phase is a resume phase.
            if cfg["save_load"].get("resume"):
                prev_cfg = _apply_overrides(BASE_SMOKE_CFG, phases[i - 1])
                strategy = cfg["strategy"]
                distribute_api = cfg["dist_parameters"].get("distribute_api", "dcp_api")
                cfg["save_load"]["resume_path"] = _find_checkpoint_path(
                    tmpdir, strategy, distribute_api
                )

            result = _launch(cfg, tmpdir, timeout)
            if result.returncode != 0:
                return result   # fail fast, don't run remaining phases
        return result # type: ignore


def _assert_ok(result: subprocess.CompletedProcess, label: str):
    """Assert exit code 0, printing the last 40 lines on failure."""
    if result.returncode != 0:
        tail = "\n".join((result.stdout + result.stderr).splitlines()[-40:])
        pytest.fail(
            f"[{label}] exited with code {result.returncode}\n"
            f"--- last 40 lines ---\n{tail}"
        )


# ── Smoke grid ────────────────────────────────────────────────────────────────
#
# List every parameter you want to vary and all the values to try.
# The framework generates one test per combination automatically.
#
# Keys follow the same convention used elsewhere:
#   plain string → top-level config field   e.g. "strategy"
#   tuple        → nested config field      e.g. ("peft", "enabled")
#
# Test IDs are auto-built from each key=value pair, e.g.:
#   strategy=fsdp__peft_enabled=true__mixed_precision=false
#
# ┌── To test a new parameter, just add a line here: ───────────────────────────┐
# │  ("dist_parameters", "mixed_precision"):          [False, True],            │
# │  ("training", "gradient_checkpointing"):          [False, True],            │
# │  ("peft", "r"):                                   [4, 8],                   │
# │  "num_gpus":                                      [1, 2],                   │
# └─────────────────────────────────────────────────────────────────────────────┘

SMOKE_GRID = {
    "strategy":          ["solo", "ddp", "fsdp"],
    ("peft", "enabled"): [False, True],
    ("training", "batch_size"):       [2],  # keep batch_size=2 for all cases to limit GPU memory use
    ("model_name",):      ["facebook/opt-125m"],  # keep the same tiny model for all cases
    ("dataset",):         [{"name": "wikitext", "subset": "wikitext-2-raw-v1", "split": "train[:1%]"}],  # same tiny dataset
    "save":            [False, True],
    # "wandb":          [{"wandb_log_with_train": False}, {"wandb_log_with_train": True}],
    # Quantization is exercised via the named SMOKE_EXTRA_CASES (ddp_peft_qlora_quant)
    # so we don't combinatorially explode here, and we keep base dict structure intact.
    ("quantization", "enabled"): [False],
    "dist_parameters": [{"distribute_api": "dcp_api"}, {"distribute_api": "dtensor_api"}],
    "peft":          [{"type": "lora"}, {"type": "qlora"}],
    ("peft", "r"):                                 [4, 8, 16],
    ("peft", "dropout"):                             [0.0, 0.05],
    "num_gpus":       [1],  # keep num_gpus=1 for all cases to ensure they run on any GPU-equipped machine
    # "model_type":     ["llm", "custom_transformer"],
    # ("dist_parameters", "mixed_precision"):        [False, True],
    # ("training", "gradient_checkpointing"):        [False, True],
    # ("training", "epochs"):                        [1],  # keep epochs=1 for all cases to limit runtime
    # ("dist_parameters", "distribute_api"):         ["dcp_api", "dtensor_api"],

}

# Skip a combination when the filter returns True.
# cfg is the fully-resolved config dict (BASE_SMOKE_CFG + overrides applied).
SMOKE_GRID_FILTERS = [
    # FSDP + quantization: bitsandbytes layers can't be sharded — always invalid.
    lambda cfg: cfg["strategy"] == "fsdp" and cfg["quantization"]["enabled"],
    # Multi-GPU: skip automatically when the machine doesn't have enough GPUs.
    lambda cfg: int(cfg.get("num_gpus", 1)) > torch.cuda.device_count(),
]

# ── Extra named cases (save/resume, multi-phase, …) ──────────────────────────
#
# For scenarios that can't be expressed as a grid (e.g. save → resume).
# Same dict format: "id", optional "min_gpus", optional "phases".

SMOKE_EXTRA_CASES = [
    {
        "id": "solo_save_resume",
        "phases": [
            {"strategy": "solo", "save": True},
            {"strategy": "solo", ("save_load", "resume"): True},
        ],
    },
    {
        "id": "ddp_save_resume",
        "phases": [
            {"strategy": "ddp",  "save": True},
            {"strategy": "ddp",  ("save_load", "resume"): True},
        ],
    },
    {
        "id": "fsdp_save_resume",
        "phases": [
            {"strategy": "fsdp", "save": True},
            {"strategy": "fsdp", ("save_load", "resume"): True},
        ],
    },
    # DDP + QLoRA (4-bit bitsandbytes quantization). Exercises the PEFT+quant
    # path through apply_ddp + _build_quantization_config + prepare_model_for_kbit_training.
    # build_args enforces: peft.type=qlora ⇒ quantization.enabled=True, bits=4.
    {
        "id": "ddp_peft_qlora_quant",
        "strategy": "ddp",
        ("peft", "enabled"): True,
        ("peft", "type"): "qlora",
        ("quantization", "enabled"): True,
        ("quantization", "bits"): 4,
    },
    # FSDP + LoRA (no quantization — bnb is incompatible with FSDP sharding).
    # Exercises PATH 2 in apply_fsdp: materialize-before-shard + PEFT wrapping.
    {
        "id": "fsdp_peft_lora",
        "strategy": "fsdp",
        ("peft", "enabled"): True,
        ("peft", "type"): "lora",
    },
]

# ── Grid expansion ────────────────────────────────────────────────────────────

def _key_label(key) -> str:
    """Short readable label for a grid key used in test IDs."""
    if isinstance(key, tuple):
        return "_".join(str(p) for p in key)
    return str(key)


def _expand_grid() -> list:
    """Cross-product of SMOKE_GRID, filtered by SMOKE_GRID_FILTERS."""
    keys = list(SMOKE_GRID.keys())
    cases = []
    for combo in itertools.product(*SMOKE_GRID.values()):
        overrides = dict(zip(keys, combo))
        cfg = _apply_overrides(BASE_SMOKE_CFG, overrides)
        if any(f(cfg) for f in SMOKE_GRID_FILTERS):
            continue
        case_id = "__".join(
            f"{_key_label(k)}={str(v).lower()}" for k, v in zip(keys, combo)
        )
        cases.append({"id": case_id, **overrides})
    return cases


ALL_SMOKE_CASES = _expand_grid() + SMOKE_EXTRA_CASES

REQUIRES_GPU = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Smoke tests require at least one CUDA GPU",
)


@pytest.mark.smoke
@REQUIRES_GPU
@pytest.mark.parametrize("case", ALL_SMOKE_CASES, ids=[c["id"] for c in ALL_SMOKE_CASES])
def test_smoke(case):
    min_gpus = case.get("min_gpus", 1)
    if torch.cuda.device_count() < min_gpus:
        pytest.skip(f"Need {min_gpus} GPUs, only {torch.cuda.device_count()} available")

    result = _run_case(case)
    _assert_ok(result, case["id"])
