"""Where a run executes: this machine's queue, other machines with torchrun, or SLURM.

Only `local` runs are started and followed by this console. `remote` (any GPU server, or
several, with no scheduler) and `slurm` runs get the exact commands to run; SLURM runs can
also be submitted from here when a SLURM controller answers.
"""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

TARGETS = ("local", "remote", "slurm")
_STATUS: dict = {"at": 0.0, "value": None}

# These end up as #SBATCH lines, so nothing outside these patterns (a newline!) may pass.
_SAFE = {
    "partition": re.compile(r"[\w.,-]{1,64}"),
    "time": re.compile(r"(\d+-)?\d{1,3}(:\d{2}){0,2}"),
    "mem": re.compile(r"\d{1,6}[KMGT]?"),
}
_HOST = re.compile(r"[A-Za-z0-9._:\[\]-]{1,253}")


def slurm_status(max_age: float = 60.0) -> dict:
    """Is a SLURM controller reachable? Having the sbatch command installed is not enough."""
    if _STATUS["value"] is not None and time.monotonic() - _STATUS["at"] < max_age:
        return _STATUS["value"]
    if not shutil.which("sbatch") or not shutil.which("sinfo"):
        value = {"available": False, "reason": "SLURM is not installed on this machine.", "partitions": []}
    else:
        try:
            r = subprocess.run(["sinfo", "-h", "-o", "%R"], capture_output=True, text=True, timeout=5)
            parts = sorted({p.strip() for p in r.stdout.split() if p.strip()})
            if r.returncode == 0 and parts:
                value = {"available": True, "reason": "", "partitions": parts}
            else:
                value = {"available": False, "partitions": [],
                         "reason": "SLURM is installed here, but no SLURM controller answers."}
        except (OSError, subprocess.TimeoutExpired):
            value = {"available": False, "reason": "The SLURM controller did not answer in time.", "partitions": []}
    _STATUS.update(at=time.monotonic(), value=value)
    return value


def _int(hw: dict, key: str, default: int, top: int) -> int:
    try:
        value = default if hw.get(key) in (None, "") else int(hw[key])
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be a whole number (got {hw.get(key)!r})") from None
    if not 1 <= value <= top:
        raise ValueError(f"{key} must be between 1 and {top} (got {value})")
    return value


def check_hardware(cfg: dict, target: str, hw: Optional[dict]) -> dict:
    """Normalised hardware options. A solo run always uses one GPU on one machine."""
    if target not in TARGETS:
        raise ValueError(f"target must be one of {', '.join(TARGETS)} (got {target!r})")
    hw = dict(hw or {})
    out = {"nodes": _int(hw, "nodes", 1, 4096), "gpus_per_node": _int(hw, "gpus_per_node", 1, 64)}
    if str(cfg.get("strategy", "solo")).lower() == "solo":
        out.update(nodes=1, gpus_per_node=1)
    if target == "remote":
        host = str(hw.get("host") or "").strip()
        if host and not _HOST.fullmatch(host):
            raise ValueError(f"host must be a host name or an IP address (got {host!r})")
        out.update(host=host, port=_int(hw, "port", 29500, 65535))
    if target == "slurm":
        for key, default in (("partition", "gpu"), ("time", "24:00:00"), ("mem", "64G")):
            value = str(hw.get(key) or default).strip()
            if not _SAFE[key].fullmatch(value):
                raise ValueError(f"{key} is not valid for SLURM (got {value!r})")
            out[key] = value
        out["cpus"] = _int(hw, "cpus", 8, 1024)
    total = out["nodes"] * out["gpus_per_node"]
    if target != "local" and str(cfg.get("strategy")).lower() == "5d" and int(cfg.get("num_gpus") or 1) != total:
        raise ValueError(f"the 5D layout is for {cfg.get('num_gpus')} GPUs, but {out['nodes']} machine(s) x "
                         f"{out['gpus_per_node']} GPUs is {total}")
    return out


def slurm_argv(config_path: str, hw: dict, name: str = "") -> list[str]:
    argv = ["scripts/launch_slurm.py", "--config", config_path, "--nodes", str(hw["nodes"]),
            "--gpus", str(hw["gpus_per_node"]), "--partition", hw["partition"], "--time", hw["time"],
            "--cpus", str(hw["cpus"]), "--mem", hw["mem"]]
    return argv + (["--job-name", name] if name else [])


def launch_commands(cfg: dict, target: str, hw: Optional[dict] = None, file: str = "configs/my_run.yaml") -> dict:
    """The commands for a run: {"commands": [{"title", "command"}], "notes": [...]}. `file` is where the
    user saves the YAML, relative to the repository."""
    h = check_hardware(cfg, target, hw)
    strategy = str(cfg.get("strategy", "solo")).lower()
    path = shlex.quote(file)
    notes: list[str] = []
    if target == "slurm":
        command = "python " + " ".join(shlex.quote(a) for a in slurm_argv(file, h))
        notes.append("Run it on a login node of the cluster, in a copy of this repository every node can read.")
        notes.append("SLURM jobs are not listed under Runs: follow them with squeue and the files in logs/.")
        return {"target": target, "commands": [{"title": "On a login node", "command": command}], "notes": notes}

    if target == "local" or h["nodes"] == 1:
        commands = [{"title": "On this machine" if target == "local" else "On that machine, in this repository",
                     "command": f"CONFIG_PATH={path} bash scripts/launch.sh"}]
    else:
        host = h["host"] or "FIRST_MACHINE"
        commands = [{"title": f"On each of the {h['nodes']} machines, the same command", "command":
                     f"CONFIG_PATH={path} torchrun --nnodes {h['nodes']} --nproc_per_node {h['gpus_per_node']} \\\n"
                     f"    --rdzv_backend c10d --rdzv_endpoint {host}:{h['port']} --rdzv_id {shlex.quote(Path(file).stem)} \\\n"
                     f"    train.py"}]
        if not h["host"]:
            notes.append("Replace FIRST_MACHINE with the host name or IP address of one of the machines.")
        notes.append(f"Every machine must reach port {h['port']} on that machine, and have the same repository, "
                     "Python environment and YAML.")
        if strategy in {"fsdp", "auto"}:
            notes.append("FSDP across machines needs checkpoint_dir on storage all machines share: the first "
                         "process writes the start weights there and the others read them.")
        notes.append("Models and datasets are downloaded on each machine unless they share a Hugging Face cache.")
    if target == "remote":
        notes.append("This console cannot start or follow runs on other machines. To watch one here, run the console "
                     "on that machine and open it through an SSH tunnel: ssh -L 8787:localhost:8787 user@machine")
    return {"target": target, "commands": commands, "notes": notes}


def write_slurm_config(cfg: dict, folder: Path, name: str = "") -> Path:
    """Each submission gets its own file: the job reads it when it STARTS, which can be hours later."""
    folder.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^\w.-]+", "_", name)[:60] or "run"
    path = folder / f"{datetime.now():%Y%m%d-%H%M%S-%f}_{slug}.yaml"
    clean = {k: v for k, v in cfg.items() if k not in {"launch_mode", "slurm"}}
    path.write_text(yaml.safe_dump(clean, sort_keys=False), encoding="utf-8")
    return path


def submit_slurm(cfg: dict, hw: Optional[dict], project_root: Path, folder: Path, name: str = "") -> dict:
    status = slurm_status(max_age=0)
    if not status["available"]:
        raise ValueError(f"{status['reason']} Copy the command and run it on a login node instead.")
    h = check_hardware(cfg, "slurm", hw)
    path = write_slurm_config(cfg, folder, name)
    job_name = re.sub(r"[^\w.-]+", "_", name)[:60]
    r = subprocess.run([sys.executable, *slurm_argv(str(path), h, job_name)], cwd=str(project_root),
                       capture_output=True, text=True, timeout=60)
    output = (r.stdout + r.stderr).strip()
    if r.returncode != 0:
        raise ValueError(output[-800:] or "sbatch failed without a message")
    m = re.search(r"Submitted batch job (\d+)", output)
    return {"job_id": m.group(1) if m else None, "output": output, "config": str(path)}
