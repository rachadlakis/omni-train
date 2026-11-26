"""Post-training (DPO/SimPO/ORPO/GRPO/PPO) under DDP and FSDP (2 CPU processes, gloo) must give the same
loss AND the same gradient for every parameter as 1 process on the whole batch.
DDP runs on CPU (gloo). FSDP2 needs CUDA, so its cases need >= 2 GPUs.

Per-parameter gradients are compared on purpose: AdamW cancels a per-tensor
gradient scale, so a loss-only check would miss e.g. a gradient 2x too large.

Run with:  python -m pytest -m dist tests/test_post_training_dist.py
"""

import os
import random
import subprocess
import sys

import pytest
import torch

from tests.dist_device import ON_GPU, dist_env, require_ranks

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKER = os.path.join(ROOT, "tests", "post_training_dist_worker.py")

CASES = [
    ("ddp", 0, "dpo"), ("ddp", 1, "dpo"), ("ddp", 0, "simpo"), ("ddp", 0, "orpo"),
    ("ddp", 0, "grpo"), ("ddp", 0, "grpo_dapo"), ("ddp", 0, "grpo_dr"), ("ddp", 1, "grpo_kl"),
    ("ddp", 0, "ppo"), ("ddp", 1, "ppo_kl"),
    ("fsdp", 0, "dpo"), ("fsdp", 1, "dpo"), ("fsdp", 0, "simpo"),
    ("fsdp", 0, "grpo_dapo"), ("fsdp", 1, "grpo_kl"), ("fsdp", 0, "ppo"), ("fsdp", 1, "ppo_kl"),
]


def _run(tmp_path, nproc, strategy, lora, algo):
    out = str(tmp_path / f"{strategy}_{lora}_{algo}_{nproc}")
    if strategy == "fsdp":  # FSDP2 always needs CUDA
        env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONPATH=ROOT)
    else:
        env = dist_env(PYTHONPATH=ROOT)
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", str(nproc),
           "--master_port", str(random.randint(20000, 29999)), WORKER, strategy, str(lora), algo, out]
    res = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    return torch.load(out + ".pt")


@pytest.mark.dist
@pytest.mark.parametrize("strategy,lora,algo", CASES)
def test_matches_single_process(tmp_path, strategy, lora, algo):
    if strategy == "fsdp" and torch.cuda.device_count() < 2:
        pytest.skip("FSDP2 needs CUDA (no CPU support in torch 2.6) and one GPU per rank: needs >= 2 GPUs")
    require_ranks(2)  # checked before the 1-process reference run
    cuda = strategy == "fsdp" or ON_GPU
    # GPU reductions are not bitwise reproducible; a scale bug is still off by >= 2x.
    rtol, atol = (1e-3, 1e-5) if cuda else (1e-4, 1e-6)
    ref = _run(tmp_path, 1, strategy, lora, algo)
    got = _run(tmp_path, 2, strategy, lora, algo)
    assert got["losses"] == pytest.approx(ref["losses"], rel=rtol / 10, abs=atol)
    assert set(got["grads"]) == set(ref["grads"]) and ref["grads"]
    if lora:
        assert all("lora_" in n for n in ref["grads"])
    for name, g in ref["grads"].items():
        assert g.abs().sum() > 0, f"{name}: zero gradient, the test would prove nothing"
        torch.testing.assert_close(got["grads"][name], g, rtol=rtol, atol=atol, msg=name)
