"""Where a run executes: this machine, other machines (torchrun, no scheduler) or SLURM.
A fake sbatch/sinfo on PATH stands in for a cluster."""

import os
import shlex
import stat
import subprocess
import sys

import pytest
import yaml
from fastapi.testclient import TestClient

import ui.app as ui_app
from ui import launchers as L

CFG = {"strategy": "ddp", "num_gpus": 2}


@pytest.fixture(autouse=True)
def fresh_status():
    L._STATUS.update(at=0.0, value=None)
    yield
    L._STATUS.update(at=0.0, value=None)


@pytest.fixture
def client():
    return TestClient(ui_app.app, base_url="http://localhost")


def fake_bin(tmp_path, monkeypatch, sinfo="echo gpu; echo long", sbatch="cat > sbatch_input.txt; echo Submitted batch job 4242"):
    d = tmp_path / "bin"
    d.mkdir()
    for name, body in (("sinfo", sinfo), ("sbatch", sbatch)):
        f = d / name
        f.write_text(f"#!/bin/sh\n{body}\n")
        f.chmod(f.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{d}{os.pathsep}{os.environ['PATH']}")


# ---- no SLURM needed -------------------------------------------------------

def test_local_and_single_other_machine_use_launch_sh_without_slurm():
    for target in ("local", "remote"):
        r = L.launch_commands(CFG, target, {"nodes": 1, "gpus_per_node": 2}, "configs/a.yaml")
        assert r["commands"][0]["command"] == "CONFIG_PATH=configs/a.yaml bash scripts/launch.sh"
        assert "slurm" not in str(r).lower() and "sbatch" not in str(r)


def test_several_machines_use_plain_torchrun():
    r = L.launch_commands(CFG, "remote", {"nodes": 3, "gpus_per_node": 4, "host": "10.0.0.5", "port": 29600}, "configs/a.yaml")
    words = shlex.split(r["commands"][0]["command"].replace("\\\n", " "))
    assert words[0] == "CONFIG_PATH=configs/a.yaml" and words[1] == "torchrun" and words[-1] == "train.py"
    opts = dict(zip(words[2:-1:2], words[3:-1:2]))
    assert opts == {"--nnodes": "3", "--nproc_per_node": "4", "--rdzv_backend": "c10d",
                    "--rdzv_endpoint": "10.0.0.5:29600", "--rdzv_id": "a"}
    assert "slurm" not in str(r).lower()


def test_missing_host_is_a_visible_placeholder_with_a_note():
    r = L.launch_commands(CFG, "remote", {"nodes": 2, "gpus_per_node": 1})
    assert "FIRST_MACHINE:29500" in r["commands"][0]["command"] and any("FIRST_MACHINE" in n for n in r["notes"])


def test_fsdp_across_machines_warns_about_shared_storage():
    fsdp = {"strategy": "fsdp", "num_gpus": 4}
    assert any("storage all machines share" in n for n in L.launch_commands(fsdp, "remote", {"nodes": 2, "gpus_per_node": 2})["notes"])
    assert not any("storage all" in n for n in L.launch_commands(fsdp, "remote", {"nodes": 1, "gpus_per_node": 4})["notes"])


@pytest.mark.parametrize("target", ["remote", "slurm"])
def test_solo_always_uses_one_gpu(target):
    hw = L.check_hardware({"strategy": "solo"}, target, {"nodes": 4, "gpus_per_node": 8})
    assert (hw["nodes"], hw["gpus_per_node"]) == (1, 1)


def test_5d_layout_must_match_the_machines():
    cfg = {"strategy": "5d", "num_gpus": 16}
    assert L.launch_commands(cfg, "remote", {"nodes": 2, "gpus_per_node": 8})
    with pytest.raises(ValueError, match="16 GPUs"):
        L.launch_commands(cfg, "remote", {"nodes": 1, "gpus_per_node": 8})


@pytest.mark.parametrize("target, hw", [
    ("slurm", {"partition": "gpu\n#SBATCH --wrap=id"}), ("slurm", {"time": "1; id"}), ("slurm", {"mem": "64G --x"}),
    ("slurm", {"cpus": 0}), ("remote", {"nodes": 2, "host": "a; id"}), ("remote", {"nodes": "x"}),
    ("remote", {"nodes": 2, "port": 99999}), ("k8s", {}),
])
def test_bad_options_are_rejected(target, hw):
    with pytest.raises(ValueError):
        L.launch_commands(CFG, target, hw)


def test_file_names_are_shell_quoted():
    r = L.launch_commands(CFG, "remote", {"nodes": 1}, "configs/a b; id.yaml")
    assert shlex.split(r["commands"][0]["command"])[0] == "CONFIG_PATH=configs/a b; id.yaml"


# ---- SLURM -----------------------------------------------------------------

def test_status_needs_an_answering_controller(tmp_path, monkeypatch):
    fake_bin(tmp_path, monkeypatch, sinfo="echo 'sinfo: error: no controller' >&2; exit 1")
    st = L.slurm_status()
    assert not st["available"] and "no SLURM controller" in st["reason"]


def test_status_without_slurm(monkeypatch):
    monkeypatch.setattr(L.shutil, "which", lambda name: None)
    assert "not installed" in L.slurm_status()["reason"]


def test_status_lists_partitions_and_is_cached(tmp_path, monkeypatch):
    fake_bin(tmp_path, monkeypatch)
    assert L.slurm_status() == {"available": True, "reason": "", "partitions": ["gpu", "long"]}
    monkeypatch.setattr(L.subprocess, "run", lambda *a, **k: pytest.fail("asked SLURM again"))
    assert L.slurm_status()["available"]


def test_slurm_command_parses_in_the_real_script():
    r = L.launch_commands(CFG, "slurm", {"nodes": 2, "gpus_per_node": 4, "partition": "long", "time": "2:00:00"}, "configs/llm_dpo_lora_single_gpu.yaml")
    out = subprocess.run([sys.executable, *shlex.split(r["commands"][0]["command"])[1:], "--dry-run"], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "--partition=long" in out.stdout and "--nodes=2" in out.stdout


def test_each_submission_keeps_its_own_config(tmp_path, monkeypatch):
    """A SLURM job reads its config when it starts, so a later submission must not overwrite it."""
    fake_bin(tmp_path, monkeypatch)
    folder = tmp_path / "slurm"
    root = tmp_path / "root"
    root.mkdir()
    (root / "scripts").symlink_to(os.path.abspath("scripts"))
    a = L.submit_slurm({**CFG, "model_name": "first", "launch_mode": "slurm"}, {}, root, folder, "run one")
    b = L.submit_slurm({**CFG, "model_name": "second"}, {}, root, folder, "run one")
    assert a["job_id"] == "4242" and a["config"] != b["config"]
    first = yaml.safe_load(open(a["config"]))
    assert first["model_name"] == "first" and "launch_mode" not in first
    assert a["config"] != b["config"] and b["config"] in (root / "sbatch_input.txt").read_text()


def test_sbatch_errors_reach_the_user(tmp_path, monkeypatch):
    fake_bin(tmp_path, monkeypatch, sbatch="cat >/dev/null; echo 'Invalid partition' >&2; exit 1")
    (tmp_path / "scripts").symlink_to(os.path.abspath("scripts"))
    with pytest.raises(ValueError, match="Invalid partition"):
        L.submit_slurm(CFG, {}, tmp_path, tmp_path / "slurm")


# ---- API -------------------------------------------------------------------

def test_api(client, tmp_path, monkeypatch):
    cfg = yaml.safe_load(open("configs/llm_sft_lora_single_gpu.yaml"))
    r = client.post("/api/launch/commands", json={"config": cfg, "target": "remote", "file": "configs/x.yaml"})
    assert r.status_code == 200 and "launch.sh" in r.json()["commands"][0]["command"]
    assert client.post("/api/launch/commands", json={"config": cfg, "target": "moon"}).status_code == 400
    monkeypatch.setattr(L.shutil, "which", lambda name: None)
    r = client.post("/api/slurm/submit", json={"config": cfg, "target": "slurm"})
    assert r.status_code == 400 and "login node" in r.json()["detail"]
