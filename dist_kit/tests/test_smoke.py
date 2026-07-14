## dist_kit/tests/test_smoke.py ##
##
## End-to-end subprocess runs of `python -m dist_kit` (random-batch demo,
## offline). Excluded by default (pytest.ini: -m "not smoke"); run with:
##
##   python -m pytest dist_kit/tests -m smoke

import os
import subprocess
import sys

import pytest

_KIT_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_MODEL_CACHED = os.path.isdir(
    os.path.expanduser("~/.cache/huggingface/hub/models--facebook--opt-125m")
)

CONFIG_TEMPLATE = """\
model_name: facebook/opt-125m
model_type: llm
strategy: {strategy}
num_gpus: 1
checkpoint_dir: checkpoints
save: true

training:
  epochs: 1
  batch_size: 2
  max_length: 32
  learning_rate: 2e-5
  gradient_checkpointing: false

dist_parameters:
  mixed_precision: false
  distribute_api: dcp_api
"""


def _run(tmp_path, strategy, launcher):
    (tmp_path / "config.yaml").write_text(CONFIG_TEMPLATE.format(strategy=strategy))
    env = dict(
        os.environ,
        PYTHONPATH=_KIT_PARENT,
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        DEMO_STEPS="2",
        DIST_KIT_NUM_WORKERS="0",
    )
    env.pop("DIST_CONFIG", None)  # let ./config.yaml resolution kick in
    result = subprocess.run(
        launcher, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=600
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    return result.stdout


@pytest.mark.smoke
@pytest.mark.skipif(not _MODEL_CACHED, reason="facebook/opt-125m not in HF cache (offline test)")
def test_e2e_solo(tmp_path):
    out = _run(tmp_path, "solo", [sys.executable, "-m", "dist_kit"])
    assert "checkpoint saved" in out
    assert list((tmp_path / "checkpoints" / "solo").glob("*.pt"))


@pytest.mark.smoke
@pytest.mark.skipif(not _MODEL_CACHED, reason="facebook/opt-125m not in HF cache (offline test)")
def test_e2e_ddp_torchrun(tmp_path):
    out = _run(
        tmp_path, "ddp",
        [sys.executable, "-m", "torch.distributed.run",
         "--nproc_per_node=1", "--master_port=29531", "-m", "dist_kit.train"],
    )
    assert "DDP checkpoint saved" in out
    assert list((tmp_path / "checkpoints" / "ddp").glob("*.pt"))


@pytest.mark.smoke
@pytest.mark.skipif(not _MODEL_CACHED, reason="facebook/opt-125m not in HF cache (offline test)")
@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="FSDP2 fully_shard requires CUDA")
def test_e2e_fsdp_torchrun(tmp_path):
    out = _run(
        tmp_path, "fsdp",
        [sys.executable, "-m", "torch.distributed.run",
         "--nproc_per_node=1", "--master_port=29532", "-m", "dist_kit.train"],
    )
    assert "FSDP checkpoint saved" in out
    assert (tmp_path / "checkpoints" / "fsdp" / "dcp_api").is_dir()
