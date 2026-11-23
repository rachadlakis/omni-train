"""
OMNI-Train console and REST API.

Serves the single-page console (ui/static) and a REST API to plan, validate, queue and
monitor pretraining, SFT and preference-optimisation runs. Interactive API docs: /docs.

Usage:
    bash ui/launch_ui.sh                      # http://127.0.0.1:8787
    uvicorn ui.app:app --host 127.0.0.1 --port 8787
"""

import math
import os
import re
import json
import signal
import subprocess
import sys
import threading
import time
import importlib.util
from collections import deque
from pathlib import Path
from typing import Optional

import yaml
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from pydantic import BaseModel

try:
    from .queue import QueueManager, JobStatus, find_free_port, summarize_metrics
    from .config_adapter import adapt_ui_config_to_mini, validate_mini_config
except ImportError:
    from ui.queue import QueueManager, JobStatus, find_free_port, summarize_metrics
    from ui.config_adapter import adapt_ui_config_to_mini, validate_mini_config

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="OMNI-Train API",
    version="2.0",
    description="Plan, launch and monitor pretraining, SFT and preference-optimisation runs "
                "(solo, DDP, FSDP, 5D parallelism) on your own GPUs.",
)


class NoCacheStaticMiddleware(BaseHTTPMiddleware):
    """Prevent browsers from caching JS and CSS static assets during development."""
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        path = request.url.path
        if path.startswith("/static/") and (path.endswith(".js") or path.endswith(".css")):
            response.headers["Cache-Control"] = "no-store"
        return response


app.add_middleware(NoCacheStaticMiddleware)


def _allowed_host(host_header: str) -> bool:
    """Accept loopback names, this machine's hostnames, any IP literal, and anything
    listed in OMNI_UI_ALLOWED_HOSTS (comma-separated).

    This blocks DNS rebinding: a malicious page on attacker.example re-resolves its
    own domain to 127.0.0.1 so the browser treats this server as same-origin. Its
    requests still carry "Host: attacker.example", which is rejected here — otherwise
    it could read /api/env secrets (the client really is 127.0.0.1) or start jobs.
    """
    import ipaddress
    import socket
    host = (host_header or "").strip().lower()
    if host.startswith("["):            # [::1]:8787
        host = host[1:].split("]", 1)[0]
    elif host.count(":") == 1:          # localhost:8787
        host = host.split(":", 1)[0]
    if not host:
        return False
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        pass
    allowed = {"localhost", socket.gethostname().lower(), socket.getfqdn().lower()}
    allowed.update(h.strip().lower() for h in os.environ.get("OMNI_UI_ALLOWED_HOSTS", "").split(",") if h.strip())
    return host in allowed


class TrustedHostCheckMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if not _allowed_host(request.headers.get("host", "")):
            return JSONResponse(
                {"detail": "Host not allowed. Add it to OMNI_UI_ALLOWED_HOSTS to serve the UI under this name."},
                status_code=400,
            )
        return await call_next(request)


app.add_middleware(TrustedHostCheckMiddleware)

UI_DIR = Path(__file__).parent
STATIC_DIR = UI_DIR / "static"
PROJECT_ROOT = UI_DIR.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
IMAGES_DIR = PROJECT_ROOT / "images"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import planner  # noqa: E402  (needs PROJECT_ROOT on sys.path)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
if IMAGES_DIR.exists():
    app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")


# ---------------------------------------------------------------------------
# In-memory training state
# ---------------------------------------------------------------------------

class TrainingState:
    def __init__(self):
        self.process: subprocess.Popen | None = None
        self.status: str = "idle"  # "idle" | "running" | "stopped" | "finished" | "error"
        self.logs: deque[str] = deque(maxlen=2000)
        self.config: dict | None = None
        self.lock = threading.Lock()
        self.error_summary: str = ""
        self.exit_code: int | None = None

    def reset(self):
        self.process = None
        self.status = "idle"
        self.logs.clear()
        self.config = None
        self.error_summary = ""
        self.exit_code = None


def _extract_error_summary(log_lines: list[str]) -> str:
    """Scan log lines and return the most meaningful error message.

    Looks for the last Python traceback and returns the final exception line.
    Falls back to the last line containing an error keyword.
    """
    # Find the last traceback
    traceback_start = -1
    for i in range(len(log_lines) - 1, -1, -1):
        if "Traceback (most recent call last)" in log_lines[i]:
            traceback_start = i
            break

    if traceback_start != -1:
        error_context = log_lines[traceback_start:]
        # The final non-blank, non-indented line is the exception
        for line in reversed(error_context):
            stripped = line.strip()
            if stripped and not line.startswith((" ", "\t")):
                return stripped
        # Fallback: last non-blank line in traceback
        for line in reversed(error_context):
            if line.strip():
                return line.strip()

    # No traceback — find last line that looks like an error
    error_pattern = re.compile(
        r"(Error|Exception|CUDA error|RuntimeError|ValueError|TypeError|KeyError|ImportError|ModuleNotFoundError)",
        re.IGNORECASE,
    )
    for line in reversed(log_lines):
        if error_pattern.search(line) and line.strip():
            return line.strip()

    return ""


state = TrainingState()

# Global queue manager (initialized on startup)
queue_manager: Optional[QueueManager] = None


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ConfigPayload(BaseModel):
    config: dict


class QueueSubmitPayload(BaseModel):
    config: dict
    gpu_count: int = 1
    priority: int = 0
    name: Optional[str] = None


class PlanPayload(BaseModel):
    config: dict
    cluster: Optional[dict] = None   # {gpu_mem_gb, gpus_per_node, nodes, gpu_name, tflops}
    num_samples: Optional[int] = None


class ValidateResponse(BaseModel):
    valid: bool
    error: str | None = None


# ---------------------------------------------------------------------------
# Routes: Frontend
# ---------------------------------------------------------------------------

@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


# ---------------------------------------------------------------------------
# Routes: Config templates
# ---------------------------------------------------------------------------

@app.get("/api/configs")
async def list_configs():
    """List available example config templates."""
    configs = set()

    root_config = PROJECT_ROOT / "config.yaml"
    if root_config.exists():
        configs.add("config")

    for f in sorted(PROJECT_ROOT.glob("config*.yaml")):
        configs.add(f.stem)

    if CONFIGS_DIR.exists():
        for f in sorted(CONFIGS_DIR.glob("*.yaml")):
            configs.add(f.stem)

    return {"configs": sorted(configs), "templates": [_describe_template(n) for n in sorted(configs) if n != "config"]}


def _describe_template(name: str) -> dict:
    """What a config trains, read from the YAML itself (for the console's template picker)."""
    path = CONFIGS_DIR / f"{name}.yaml"
    path = path if path.exists() else PROJECT_ROOT / f"{name}.yaml"
    try:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        block = lambda key: cfg.get(key) if isinstance(cfg.get(key), dict) else {}  # noqa: E731
        peft, quant = block("peft"), block("quantization")
        method = "full"
        if peft.get("enabled"):
            method = "qlora" if (peft.get("type") == "qlora" or quant.get("enabled")) else "lora"
        return {
            "name": name, "model_type": str(cfg.get("model_type", "llm")), "model": cfg.get("model_name"),
            "algorithm": str(block("algorithm").get("type", "pretrain")), "method": method,
            "bits": int(quant.get("bits", 4)) if quant.get("enabled") else None,
            "strategy": str(cfg.get("strategy", "solo")), "num_gpus": int(cfg.get("num_gpus") or 1),
            "dataset": block("dataset").get("name"), "moe": bool(block("custom_transformer_args").get("num_experts")),
        }
    except Exception as e:  # a broken file must not hide the others
        return {"name": name, "error": str(e)[:200]}


@app.get("/api/configs/{name}")
async def get_config(name: str):
    """Load a specific config template as YAML text and parsed dict."""
    # Validate name to prevent path traversal attacks
    if ".." in name or "/" in name or "\\" in name:
        raise HTTPException(403, "Invalid config name")

    candidate_paths = [
        (PROJECT_ROOT / f"{name}.yaml").resolve(),
        (CONFIGS_DIR / f"{name}.yaml").resolve(),
    ]

    path = None
    for candidate in candidate_paths:
        if candidate.exists():
            path = candidate
            break

    if path is None:
        raise HTTPException(404, f"Config '{name}' not found")

    allowed_roots = [PROJECT_ROOT.resolve()]
    if CONFIGS_DIR.exists():
        allowed_roots.append(CONFIGS_DIR.resolve())

    if not any(_is_safe_path(path, root) for root in allowed_roots):
        raise HTTPException(403, "Access denied")

    raw = path.read_text()
    parsed = yaml.safe_load(raw)
    return {"name": name, "yaml": raw, "config": parsed}


def _is_safe_path(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _parse_dotenv(path: Path) -> dict:
    """Parse a .env file and return key-value pairs."""
    result = {}
    if not path.exists():
        return result
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, val = line.partition("=")
            result[key.strip()] = val.strip().strip('"').strip("'")
    return result


@app.get("/api/env")
async def get_env_keys(request: Request, keys: str = Query("", description="Comma-separated key names to fetch")):
    """Read specific keys from the project's .env file.

    Secret values are only returned to local (loopback) clients. If the UI is
    exposed on a non-loopback interface, callers receive empty values instead,
    so an exposed server cannot leak HF_TOKEN / WANDB_API_KEY over the network.
    """
    env_path = (PROJECT_ROOT / ".env").resolve()
    if not _is_safe_path(env_path, PROJECT_ROOT.resolve()):
        raise HTTPException(403, "Access denied")

    parsed = _parse_dotenv(env_path)

    if keys:
        requested = [k.strip() for k in keys.split(",") if k.strip()]
        selected = {k: parsed.get(k) for k in requested}
    else:
        selected = dict(parsed)

    client_host = request.client.host if request.client else None
    is_local = client_host in {"127.0.0.1", "::1", "localhost"}
    if is_local:
        result = selected
    else:
        # Non-local caller: never reveal values. Empty string makes the UI fall
        # back to manual key entry instead of autofilling a stored secret.
        result = {k: "" for k in selected}

    return {"found": env_path.exists(), "local": is_local, "keys": result}


@app.get("/api/env/status")
async def env_status():
    """Which credentials are configured (.env or the server's environment). Never returns values."""
    parsed = _parse_dotenv((PROJECT_ROOT / ".env").resolve())
    keys = ("HF_TOKEN", "WANDB_API_KEY")
    return {
        "found": (PROJECT_ROOT / ".env").exists(),
        "keys": {k: bool(parsed.get(k) or os.environ.get(k)) for k in keys},
    }


# ---------------------------------------------------------------------------
# Routes: System Info
# ---------------------------------------------------------------------------

@app.get("/api/system/gpus")
async def get_gpu_info():
    """Get information about available GPUs."""
    try:
        import torch
    except ImportError:
        return {
            "available": False,
            "error": "PyTorch not installed",
            "gpus": [],
            "count": 0,
        }

    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        return {
            "available": False,
            "error": "CUDA not available",
            "gpus": [],
            "count": 0,
        }

    gpu_count = torch.cuda.device_count()
    gpus = []

    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        # Get memory info
        total_mem = props.total_memory / (1024 ** 3)  # Convert to GB
        try:
            # Try to get current memory usage (may fail if device not initialized)
            torch.cuda.set_device(i)
            free_mem = torch.cuda.mem_get_info(i)[0] / (1024 ** 3)
            used_mem = total_mem - free_mem
        except Exception:
            free_mem = total_mem
            used_mem = 0

        gpus.append({
            "index": i,
            "name": props.name,
            "total_memory_gb": round(total_mem, 2),
            "free_memory_gb": round(free_mem, 2),
            "used_memory_gb": round(used_mem, 2),
            "compute_capability": f"{props.major}.{props.minor}",
            "multi_processor_count": props.multi_processor_count,
        })

    return {
        "available": True,
        "count": gpu_count,
        "gpus": gpus,
        "cuda_version": getattr(torch.version, "cuda", None), # type: ignore
        "pytorch_version": torch.__version__,
    }


# ---------------------------------------------------------------------------
# Routes: Training Time Estimation
# ---------------------------------------------------------------------------

def _format_time(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        return f"{seconds / 60:.1f} minutes"
    elif seconds < 86400:
        return f"{seconds / 3600:.1f} hours"
    else:
        return f"{seconds / 86400:.1f} days"


def get_model_params_from_name(model_name: str, model_type: str) -> float:
    """Extract model parameters (in millions) from model name or return default."""
    import re
    name_lower = model_name.lower()

    match = re.search(r"(\d+(?:\.\d+)?)\s*b(?:illion)?", name_lower)
    if match:
        return float(match.group(1)) * 1000

    match = re.search(r"(\d+(?:\.\d+)?)\s*m(?:illion)?", name_lower)
    if match:
        return float(match.group(1))

    known = {
        "resnet18": 11, "resnet34": 21, "resnet50": 25, "resnet101": 44,
        "vit_b": 86, "vit_l": 304, "vit_h": 632,
        "efficientnet_b0": 5, "efficientnet_b4": 19, "efficientnet_b7": 66,
        "yolov5n": 1.9, "yolov5s": 7.2, "yolov5m": 21, "yolov5l": 46,
        "yolov8n": 3.2, "yolov8s": 11, "yolov8m": 26, "yolov8l": 44,
        "bert-base": 110, "bert-large": 340,
        "llama": 7000, "mistral": 7000, "phi-3": 3800,
    }
    for key, params in known.items():
        if key in name_lower:
            return params

    defaults = {"cnn": 25, "llm": 7000, "vlm": 7000, "detection": 12, "embedding": 110}
    return defaults.get(model_type, 50)


def get_gpu_tflops() -> tuple[int, float]:
    """Detect GPU count and estimate TFLOPS."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0).lower()
            if "a100" in gpu_name:
                tflops = 312
            elif "h100" in gpu_name:
                tflops = 990
            elif "v100" in gpu_name:
                tflops = 125
            elif "4090" in gpu_name:
                tflops = 83
            elif "4080" in gpu_name:
                tflops = 49
            elif "3090" in gpu_name:
                tflops = 36
            elif "3080" in gpu_name:
                tflops = 30
            elif "3070" in gpu_name:
                tflops = 20
            elif "3060" in gpu_name:
                tflops = 13
            elif "2080" in gpu_name:
                tflops = 14
            elif "1080" in gpu_name:
                tflops = 11
            else:
                tflops = 15
            return gpu_count, tflops
    except Exception:
        pass
    return 1, 10.0


def _parse_dataset_split_ratio(split: str) -> float:
    match = re.search(r"\[:\s*(\d+(?:\.\d+)?)%\]", split or "")
    if not match:
        return 1.0
    pct = float(match.group(1))
    return max(0.0001, min(1.0, pct / 100.0))


def _infer_dataset_size(dataset_name: str, dataset_split: str) -> int:
    name = (dataset_name or "").lower()
    known_full_sizes = {
        "wikitext": 36718,
        "wikitext-2": 36718,
        "wikitext-103": 1801350,
        "ptb": 42068,
        "penn": 42068,
        "c4": 365000000,
        "alpaca": 52000,
        "squad": 87599,
        "ultrafeedback": 62135,
        "capybara": 16006,
        "hh-rlhf": 160800,
    }
    base = 100000
    for key, size in known_full_sizes.items():
        if key in name:
            base = size
            break
    # "train[:2000]" / "train[:2000]+validation[:500]": absolute row counts
    counts = re.findall(r"\[\s*:\s*(\d+)\s*\]", dataset_split or "")
    if counts:
        return max(1, sum(int(n) for n in counts))
    ratio = _parse_dataset_split_ratio(dataset_split)
    return max(1, int(base * ratio))


def _infer_num_params_from_model_name(model_name: str) -> int:
    raw = (model_name or "").strip().lower()
    name = re.split(r"\s+#", raw, maxsplit=1)[0].strip()
    name = re.split(r"\s+", name, maxsplit=1)[0].strip()
    name = re.split(r",", name, maxsplit=1)[0].strip()

    known = {
        "facebook/opt-125m": 125_000_000,
        "facebook/opt-350m": 350_000_000,
        "facebook/opt-1.3b": 1_300_000_000,
        "facebook/opt-2.7b": 2_700_000_000,
        "facebook/opt-6.7b": 6_700_000_000,
        "facebook/opt-13b": 13_000_000_000,
        "gpt2": 124_000_000,
        "gpt2-medium": 355_000_000,
        "gpt2-large": 774_000_000,
        "gpt2-xl": 1_500_000_000,
        "llama-3-8b": 8_000_000_000,
        "mistral-7b": 7_000_000_000,
    }
    if name in known:
        return known[name]

    alias_map = {
        "opt-125m": "facebook/opt-125m",
        "opt-350m": "facebook/opt-350m",
        "opt-1.3b": "facebook/opt-1.3b",
        "opt-2.7b": "facebook/opt-2.7b",
        "opt-6.7b": "facebook/opt-6.7b",
        "opt-13b": "facebook/opt-13b",
    }
    if name in alias_map:
        return known[alias_map[name]]

    b_match = re.search(r"(\d+(?:\.\d+)?)\s*b", name)
    if b_match:
        return int(float(b_match.group(1)) * 1_000_000_000)

    m_match = re.search(r"(\d+(?:\.\d+)?)\s*m", name)
    if m_match:
        return int(float(m_match.group(1)) * 1_000_000)

    return 350_000_000


def _infer_transformer_shape(num_params: int) -> tuple[int, int, int]:
    if num_params <= 200_000_000:
        return (12, 768, 12)
    if num_params <= 500_000_000:
        return (24, 1024, 16)
    if num_params <= 2_000_000_000:
        return (24, 2048, 16)
    if num_params <= 9_000_000_000:
        return (32, 4096, 32)
    return (48, 5120, 40)


def _normalize_gpu_type(gpu_name: str) -> str:
    n = (gpu_name or "").lower()
    keys = ["h100", "a100", "a10g", "v100", "l4", "t4", "a6000", "a5000", "a4000", "a2", "4090", "3090", "3080"]
    for key in keys:
        if key in n:
            return key
    return "unknown"


def _load_mini_utils_module(project_root: Path):
    utils_path = project_root / "utils.py"
    spec = importlib.util.spec_from_file_location("fsdp_mini_utils", str(utils_path))
    if not spec or not spec.loader:
        raise RuntimeError(f"Failed to load utils module from {utils_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module



@app.get("/api/checkpoints")
async def list_checkpoints(dir: str = "checkpoints"):
    """List checkpoints under the configured checkpoint directory.

    Layout (per save_checkpoint in distributed_utils.py):
      <dir>/solo/<ts>[<tag>].pt                            (file)
      <dir>/ddp/<ts>[<tag>].pt                             (file)
      <dir>/fsdp/dcp_api/<ts>[<tag>]/                      (dir)
      <dir>/fsdp/dtensor_api/<ts>[<tag>]/                  (dir)
    """
    from datetime import datetime
    import re

    # Resolve relative paths against PROJECT_ROOT; refuse to escape it.
    raw = Path(dir)
    ckpt_root = (raw if raw.is_absolute() else (PROJECT_ROOT / raw)).resolve()
    try:
        ckpt_root.relative_to(PROJECT_ROOT)
    except ValueError:
        raise HTTPException(400, f"checkpoint_dir must be under project root: {dir}")

    def _dir_size_bytes(p: Path) -> int:
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())

    def _parse_ts_tag(stem: str):
        m = re.match(r"^(\d+)(__.+)?$", stem)
        if not m:
            return None, ""
        return int(m.group(1)), (m.group(2) or "")

    entries: list[dict] = []
    scan_specs = [
        ("solo",         ckpt_root / "solo",              "file"),
        ("ddp",          ckpt_root / "ddp",               "file"),
        ("fsdp-dcp",     ckpt_root / "fsdp" / "dcp_api",  "dir"),
        ("fsdp-dtensor", ckpt_root / "fsdp" / "dtensor_api", "dir"),
        ("5d",           ckpt_root / "5d",                "dir"),
    ]
    for strategy, sub, kind in scan_specs:
        if not sub.exists():
            continue
        for child in sub.iterdir():
            if kind == "file" and child.is_file() and child.suffix == ".pt":
                ts, tag = _parse_ts_tag(child.stem)
                size = child.stat().st_size
            elif kind == "dir" and child.is_dir():
                ts, tag = _parse_ts_tag(child.name)
                size = _dir_size_bytes(child)
            else:
                continue
            mtime = child.stat().st_mtime
            entries.append({
                "strategy": strategy,
                "path": str(child.relative_to(PROJECT_ROOT)),
                "abs_path": str(child),
                "is_dir": child.is_dir(),
                "size_bytes": size,
                "timestamp_ms": ts,
                "tag": tag,
                "mtime": mtime,
                "mtime_iso": datetime.fromtimestamp(mtime).isoformat(timespec="seconds"),
            })

    entries.sort(key=lambda e: e["mtime"], reverse=True)
    return {
        "base_dir": str(ckpt_root),
        "base_dir_relative": str(ckpt_root.relative_to(PROJECT_ROOT)) if ckpt_root != PROJECT_ROOT else ".",
        "exists": ckpt_root.exists(),
        "count": len(entries),
        "checkpoints": entries,
    }


@app.get("/api/browse")
async def browse_dir(path: str = "."):
    """List subdirectories at `path` for an in-UI folder picker. Restricted to PROJECT_ROOT."""
    raw = Path(path)
    target = (raw if raw.is_absolute() else (PROJECT_ROOT / raw)).resolve()
    try:
        target.relative_to(PROJECT_ROOT)
    except ValueError:
        raise HTTPException(400, f"path must be under project root: {path}")
    if not target.exists():
        raise HTTPException(404, f"path does not exist: {target}")
    if not target.is_dir():
        raise HTTPException(400, f"not a directory: {target}")

    dirs = []
    try:
        for child in sorted(target.iterdir(), key=lambda p: p.name.lower()):
            if child.is_dir() and not child.name.startswith("."):
                dirs.append({
                    "name": child.name,
                    "path": str(child.relative_to(PROJECT_ROOT)),
                })
    except PermissionError:
        raise HTTPException(403, f"permission denied: {target}")

    parent_rel = None
    if target != PROJECT_ROOT:
        parent_rel = str(target.parent.relative_to(PROJECT_ROOT)) or "."

    rel = str(target.relative_to(PROJECT_ROOT)) or "."
    return {
        "path": rel,
        "abs_path": str(target),
        "parent": parent_rel,
        "is_root": target == PROJECT_ROOT,
        "dirs": dirs,
    }


@app.post("/api/estimate-time")
async def api_estimate_training_time(payload: ConfigPayload):
    """
    Estimate training time based on model size and dataset size.
    """
    try:
        mini_cfg = adapt_ui_config_to_mini(payload.config, PROJECT_ROOT)
        utils_mod = _load_mini_utils_module(PROJECT_ROOT)
        args = utils_mod.build_args(mini_cfg)

        try:
            import torch
            detected_gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown"
        except Exception:
            detected_gpu_count = 1
            gpu_name = "unknown"

        strategy = str(getattr(args, "strategy", mini_cfg.get("strategy", "solo"))).lower()
        cfg_num_gpus = int(mini_cfg.get("num_gpus", 1) or 1)
        num_gpus = 1 if strategy == "solo" else max(1, cfg_num_gpus)

        dataset_name = str(getattr(args, "dataset", mini_cfg.get("dataset", {}).get("name", "")))
        dataset_split = str(getattr(args, "dataset_split", mini_cfg.get("dataset", {}).get("split", "train")))
        dataset_size = _infer_dataset_size(dataset_name, dataset_split)

        batch_size = int(getattr(args, "batch_size", mini_cfg.get("training", {}).get("batch_size", 8)))
        epochs = int(getattr(args, "epochs", mini_cfg.get("training", {}).get("epochs", 3)))
        max_length = int(getattr(args, "max_length", mini_cfg.get("training", {}).get("max_length", 128)))
        steps_per_epoch = max(1, math.ceil(dataset_size / max(1, batch_size)))

        model_name = str(getattr(args, "model_name", mini_cfg.get("model_name", "")))
        num_params = _infer_num_params_from_model_name(model_name)
        gpu_type = _normalize_gpu_type(gpu_name)

        time_result = utils_mod.estimate_training_time(
            num_params=num_params,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            batch_size=batch_size,
            seq_len=max_length,
            num_gpus=num_gpus,
            gpu_type=gpu_type,
            peft_enabled=bool(getattr(args, "peft_enabled", False)),
            peft_r=int(getattr(args, "peft_r", 16)),
            gradient_checkpointing=bool(getattr(args, "gradient_checkpointing", False)),
            strategy=strategy,
            mfu=0.25,
            extra_overhead=1.0,
        )

        num_layers, hidden_dim, num_heads = _infer_transformer_shape(num_params)
        param_dtype = str(getattr(args, "param_dtype", "bfloat16")).lower()
        param_dtype_bits = 32 if "32" in param_dtype else 16
        model_type = str(getattr(args, "model_type", mini_cfg.get("model", {}).get("type", "llm"))).lower()

        quant_bits = 0
        if bool(getattr(args, "quantization_enabled", False)):
            quant_bits = int(getattr(args, "quantization_bits", 4))

        vram_result = utils_mod.estimate_training_vram(
            model_type=model_type,
            num_params=num_params,
            param_dtype_bits=param_dtype_bits,
            batch_size=batch_size,
            seq_len=max_length,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            activation_checkpointing=bool(getattr(args, "gradient_checkpointing", False)),
            peft_enabled=bool(getattr(args, "peft_enabled", False)),
            peft_r=int(getattr(args, "peft_r", 16)),
            quantization_bits=quant_bits,
        )

        est_total_minutes = float(time_result.get("total_minutes", 0.0))
        est_total_seconds = est_total_minutes * 60.0
        total_steps = steps_per_epoch * max(1, epochs)

        return {
            "model_type": model_type,
            "dataset_size": dataset_size,
            "epochs": epochs,
            "batch_size": batch_size,
            "gpu_count": num_gpus,
            "gpu_type": gpu_type,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "est_total_seconds": round(est_total_seconds, 1),
            "est_total_minutes": round(est_total_minutes, 2),
            "est_total_hours": round(float(time_result.get("total_hours", 0.0)), 2),
            "est_total_days": round(float(time_result.get("total_days", 0.0)), 3),
            "readable": str(time_result.get("human_readable", _format_time(est_total_seconds))),
            "vram": vram_result,
        }

    except Exception:
        # Fallback to heuristic if utils module invocation fails
        config = payload.config
        model_config = config.get("model", {})
        data_config = config.get("data", {})
        training_config = config.get("training", {})
        distributed_config = config.get("distributed", {})

        model_type = model_config.get("type", "llm")
        model_name = model_config.get("name", "") or mini_cfg.get("model_name", "") # type: ignore
        epochs = training_config.get("epochs") or mini_cfg.get("training", {}).get("epochs", 3)  # type: ignore 
        batch_size = training_config.get("batch_size") or mini_cfg.get("training", {}).get("batch_size", 8) # type: ignore 
        seq_length = training_config.get("max_length") or mini_cfg.get("training", {}).get("max_length", 128) # type: ignore 

        model_params = get_model_params_from_name(model_name, model_type)
        nb_parameters = max(1, int(model_params * 1_000_000))
        dataset_size = data_config.get("dataset_size") or data_config.get("num_samples") or 10000
        gpu_count, gpu_tflops = get_gpu_tflops()

        strategy = distributed_config.get("strategy", "none")

        steps_per_epoch = max(1, math.ceil(dataset_size / max(1, batch_size)))
        total_steps = steps_per_epoch * max(1, epochs)
        total_tokens = seq_length * batch_size * steps_per_epoch * epochs
        total_flops = 6.0 * nb_parameters * total_tokens
        effective_flops_per_sec = gpu_tflops * 1e12 * gpu_count * 0.5
        est_total_seconds = (total_flops / effective_flops_per_sec) * 1.10

        return {
            "model_type": model_type,
            "dataset_size": dataset_size,
            "epochs": epochs,
            "batch_size": batch_size,
            "gpu_count": gpu_count,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "est_total_seconds": round(est_total_seconds, 1),
            "est_total_minutes": round(est_total_seconds / 60, 2),
            "est_total_hours": round(est_total_seconds / 3600, 2),
            "est_total_days": round(est_total_seconds / 86400, 3),
            "readable": _format_time(est_total_seconds),
        }

# ---------------------------------------------------------------------------
# Routes: Capacity planner ("is distributed training needed?")
# ---------------------------------------------------------------------------

_HF_CONFIG_CACHE: dict[str, Optional[dict]] = {}


def _hf_config(model_name: str) -> Optional[dict]:
    """config.json of a Hub model: local cache first, then the Hub (unless offline)."""
    if not model_name or model_name in _HF_CONFIG_CACHE:
        return _HF_CONFIG_CACHE.get(model_name)
    result = None
    try:
        from transformers import AutoConfig
        token = os.environ.get("HF_TOKEN") or _parse_dotenv(PROJECT_ROOT / ".env").get("HF_TOKEN") or None
        for local_only in (True, False):
            if not local_only and os.environ.get("HF_HUB_OFFLINE") == "1":
                break
            try:
                result = AutoConfig.from_pretrained(model_name, token=token, local_files_only=local_only).to_dict()
                break
            except Exception:
                continue
    except Exception:
        result = None
    _HF_CONFIG_CACHE[model_name] = result
    return result


def _model_spec(cfg: dict) -> "planner.ModelSpec":
    model_type = str(cfg.get("model_type", "llm")).lower()
    name = str(cfg.get("model_name") or "")
    if model_type == "custom_transformer":
        return planner.spec_from_custom(cfg.get("custom_transformer_args") or {})
    if name.lower() in planner.PRESETS:
        return planner.PRESETS[name.lower()]
    return planner.resolve_spec(name, model_type, hf_config=_hf_config(name))


def _local_cluster() -> dict:
    """GPUs on this machine, as planner.Cluster fields."""
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return {"gpu_mem_gb": round(props.total_memory / GB_BYTES, 2), "gpus_per_node": torch.cuda.device_count(),
                    "nodes": 1, "gpu_name": props.name, "detected": True}
    except Exception:
        pass
    return {"gpu_mem_gb": 80.0, "gpus_per_node": 1, "nodes": 1, "gpu_name": "H100", "detected": False}


GB_BYTES = 1024 ** 3


def _plan(payload: PlanPayload) -> dict:
    cfg = adapt_ui_config_to_mini(payload.config, PROJECT_ROOT)
    spec = _model_spec(cfg)
    work = planner.workload_from_config(cfg)
    work.num_samples = payload.num_samples or _infer_dataset_size(
        str((cfg.get("dataset") or {}).get("name", "")), str((cfg.get("dataset") or {}).get("split", "train")))
    local = _local_cluster()
    fields = {k: v for k, v in {**local, **(payload.cluster or {})}.items()
              if k in {"gpu_mem_gb", "gpus_per_node", "nodes", "gpu_name", "tflops"} and v not in (None, "")}
    cluster = planner.Cluster(**fields)
    result = planner.plan(spec, work, cluster)
    strategy = str(cfg.get("strategy", "solo")).lower()
    result["selected"] = planner.evaluate_layout(
        spec, work, cluster, strategy, int(cfg.get("num_gpus", 1) or 1), cfg.get("parallelism"))
    result["selected"]["strategy"] = strategy
    result["local_cluster"] = local
    return result


@app.post("/api/plan")
async def plan_run(payload: PlanPayload):
    """Is distributed training needed? Recommends solo / DDP / FSDP / a 5D layout.

    Returns the verdict, per-GPU memory breakdown and throughput of every strategy,
    the minimum GPU count per strategy, what-if fixes, and an evaluation of the
    strategy currently in the config (`selected`).
    """
    from starlette.concurrency import run_in_threadpool
    try:
        return await run_in_threadpool(_plan, payload)
    except Exception as e:
        raise HTTPException(400, f"Could not plan this config: {e}")


@app.get("/api/models/inspect")
async def inspect_model(name: str = Query(..., description="Hugging Face model id"),
                        model_type: str = Query("llm")):
    """Architecture summary of a model (presets offline, otherwise its config.json)."""
    from starlette.concurrency import run_in_threadpool
    spec = await run_in_threadpool(_model_spec, {"model_name": name, "model_type": model_type})
    return spec.to_dict()


ALGORITHM_CATALOG = [
    {"id": "pretrain", "stage": "pretrain", "title": "Pretraining / continued pretraining",
     "summary": "Next-token prediction on raw text. Supports every strategy including 5D.",
     "data": "text column (packed into fixed blocks under 5d)", "status": "ready",
     "strategies": ["solo", "ddp", "fsdp", "5d"], "lr": {"full": 2e-5, "lora": 2e-4}},
    {"id": "sft", "stage": "sft", "title": "Supervised fine-tuning (SFT)",
     "summary": "Instruction / chat tuning. Loss on the assistant tokens only by default.",
     "data": "messages | prompt + completion | instruction (+input) + output | text", "status": "ready",
     "strategies": ["solo", "ddp", "fsdp"], "lr": {"full": 1e-5, "lora": 2e-4},
     "example_dataset": {"name": "trl-lib/Capybara", "split": "train[:2000]"}},
    {"id": "dpo", "stage": "preference", "title": "DPO", "summary": "Direct Preference Optimisation against a frozen reference (precomputed once, no second model).",
     "data": "prompt + chosen + rejected | chosen/rejected conversations | hh-rlhf", "status": "ready",
     "strategies": ["solo", "ddp", "fsdp"], "lr": {"full": 1e-6, "lora": 5e-5}, "beta": 0.1,
     "example_dataset": {"name": "trl-lib/ultrafeedback_binarized", "split": "train[:2000]"}},
    {"id": "simpo", "stage": "preference", "title": "SimPO", "summary": "Reference-free, length-normalised preference loss with a target margin.",
     "data": "prompt + chosen + rejected", "status": "ready", "strategies": ["solo", "ddp", "fsdp"],
     "lr": {"full": 1e-6, "lora": 5e-5}, "beta": 2.0,
     "example_dataset": {"name": "trl-lib/ultrafeedback_binarized", "split": "train[:2000]"}},
    {"id": "orpo", "stage": "preference", "title": "ORPO", "summary": "SFT + odds-ratio preference term in one pass, no reference model.",
     "data": "prompt + chosen + rejected", "status": "ready", "strategies": ["solo", "ddp", "fsdp"],
     "lr": {"full": 5e-6, "lora": 8e-5}, "beta": 0.1,
     "example_dataset": {"name": "trl-lib/ultrafeedback_binarized", "split": "train[:2000]"}},
    {"id": "grpo", "stage": "rl", "title": "GRPO",
     "summary": "Online RL: sample a group of answers per prompt, score them with reward functions, "
                "reinforce the ones that beat the group average. No reward model or value model.",
     "data": "prompt | question | problem | messages, plus an optional reference answer (answer | solution | ...)",
     "status": "ready", "strategies": ["solo", "ddp", "fsdp"], "lr": {"full": 1e-6, "lora": 2e-5}, "beta": 0.04,
     "rewards": ["accuracy", "format", "exact_match", "contains", "length"],
     "example_dataset": {"name": "openai/gsm8k", "subset": "main", "split": "train[:1000]"}},
    {"id": "ppo", "stage": "rl", "title": "PPO (RLHF)", "summary": "Online RL with a reward model and value head.",
     "data": "prompts + reward model", "status": "planned", "strategies": []},
]


@app.get("/api/catalog")
async def catalog():
    """Algorithms, strategies, model presets and GPU types the console offers."""
    models = [{"name": s.name, "params": s.params, "arch": s.arch, "five_d_ready": s.five_d_ready,
               "layers": s.layers, "hidden": s.hidden, "vocab": s.vocab, "num_experts": s.num_experts}
              for s in planner.PRESETS.values()]
    gpus = [{"id": k, "name": v[0], "memory_gb": v[1], "tflops": v[2]} for k, v in planner.GPU_CATALOG.items()]
    return {"algorithms": ALGORITHM_CATALOG, "models": models, "gpus": gpus,
            "local_cluster": _local_cluster()}


# ---------------------------------------------------------------------------
# Routes: Validation
# ---------------------------------------------------------------------------

class YamlPayload(BaseModel):
    yaml: str


@app.post("/api/config/render")
async def render_config(payload: ConfigPayload):
    """The full training YAML (config.yaml schema) this config launches with."""
    mini_cfg = adapt_ui_config_to_mini(payload.config, PROJECT_ROOT)
    for key in ("launch_mode", "slurm"):
        mini_cfg.pop(key, None)
    return {"yaml": yaml.safe_dump(mini_cfg, sort_keys=False, default_flow_style=False), "config": mini_cfg}


@app.post("/api/config/parse")
async def parse_config(payload: YamlPayload):
    """Parse YAML text into a config dict (for the console's YAML editor)."""
    try:
        parsed = yaml.safe_load(payload.yaml)
    except yaml.YAMLError as e:
        raise HTTPException(400, f"YAML syntax error: {e}")
    if not isinstance(parsed, dict):
        raise HTTPException(400, "The YAML must be a mapping (key: value pairs) at the top level.")
    return {"config": parsed}


@app.post("/api/config/validate")
async def validate_config(payload: ConfigPayload):
    """Validate a config dict against the schema."""
    try:
        mini_cfg = adapt_ui_config_to_mini(payload.config, PROJECT_ROOT)
        validate_mini_config(mini_cfg, PROJECT_ROOT)
        return ValidateResponse(valid=True)

    except ValueError as e:
        # Validation logic errors (from _validate_config)
        return ValidateResponse(valid=False, error=str(e))
    except TypeError as e:
        # Type-related errors
        return ValidateResponse(valid=False, error=f"Type error: {str(e)}")
    except Exception as e:
        # Catch-all with more detail
        error_msg = str(e) if str(e) else f"{type(e).__name__} occurred"
        return ValidateResponse(valid=False, error=error_msg)

# ---------------------------------------------------------------------------
# Routes: Training
# ---------------------------------------------------------------------------

@app.post("/api/train/start")
async def start_training(payload: ConfigPayload):
    """Start a training run with the given config."""
    with state.lock:
        if state.status == "running":
            raise HTTPException(409, "Training is already running")

        # Write config to a temp file
        tmp_config = UI_DIR / "_active_config.yaml"
        mini_cfg = adapt_ui_config_to_mini(payload.config, PROJECT_ROOT)
        try:
            validate_mini_config(mini_cfg, PROJECT_ROOT)
        except Exception as e:
            raise HTTPException(400, f"Invalid training config: {e}")
        with open(tmp_config, "w") as f:
            yaml.dump(mini_cfg, f, default_flow_style=False)

        state.reset()
        state.config = mini_cfg
        state.status = "running"

    strategy = str(mini_cfg.get("strategy", "solo")).lower()
    strategy = strategy if strategy in {"solo", "ddp", "fsdp", "5d"} else "solo"
    gpu_count = int(mini_cfg.get("num_gpus", 1) or 1)
    if strategy == "solo":
        gpu_count = 1
    else:
        gpu_count = max(1, gpu_count)

    launch_mode = str(mini_cfg.get("launch_mode", "torchrun")).lower()

    if launch_mode == "slurm":
        slurm_cfg = mini_cfg.get("slurm", {}) if isinstance(mini_cfg.get("slurm"), dict) else {}
        slurm_nodes = int(slurm_cfg.get("nodes", 1))
        slurm_partition = str(slurm_cfg.get("partition", "gpu"))
        slurm_time = str(slurm_cfg.get("time", "2:00:00"))
        # Use explicit gpus_per_node from UI; fall back to dividing total only if not provided
        slurm_gpus_per_node = int(slurm_cfg.get("gpus_per_node") or max(1, gpu_count // max(1, slurm_nodes)))
        cmd = [
            sys.executable,
            "-u",
            str(PROJECT_ROOT / "scripts" / "launch_slurm.py"),
            "--config", str(tmp_config),
            "--nodes", str(slurm_nodes),
            "--gpus", str(slurm_gpus_per_node),
            "--partition", slurm_partition,
            "--time", slurm_time,
        ]
    elif strategy == "solo":
        cmd = [sys.executable, "-u", "train.py"]
    else:
        # ddp, fsdp and 5d use torchrun
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={gpu_count}",
            "--master_addr=localhost",
            f"--master_port={find_free_port()}",  # a queued job may already hold 29500
            "train.py",
        ]

    env = os.environ.copy()
    env["CONFIG_PATH"] = str(tmp_config)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTHONIOENCODING", "utf-8")
    elastic_error_file = UI_DIR / "_torchelastic_error.json"
    env["TORCHELASTIC_ERROR_FILE"] = str(elastic_error_file)

    # Ensure fsdp-mini-project root is on PYTHONPATH
    python_path = env.get("PYTHONPATH", "")
    project_root_str = str(PROJECT_ROOT)
    if project_root_str not in python_path:
        env["PYTHONPATH"] = project_root_str + os.pathsep + python_path

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=project_root_str,
            env=env,
            start_new_session=sys.platform != "win32",
        )
        state.process = proc
    except Exception as e:
        state.status = "error"
        state.logs.append(f"Failed to start process: {e}")
        raise HTTPException(500, str(e))

    # Background thread to read output
    def _reader():
        try:
            if proc.stdout:
                for line in proc.stdout:
                    state.logs.append(line.rstrip("\n"))
        except Exception:
            pass
        proc.wait()
        with state.lock:
            state.exit_code = proc.returncode
            if proc.returncode == 0:
                state.status = "finished"
            elif state.status == "running":
                state.status = "error"
                if elastic_error_file.exists():
                    try:
                        raw = elastic_error_file.read_text().strip()
                        if raw:
                            data = json.loads(raw)
                            msg = data.get("message") or ""
                            extra = data.get("extraInfo", {}) if isinstance(data, dict) else {}
                            py_callstack = extra.get("py_callstack") if isinstance(extra, dict) else None
                            if msg:
                                state.logs.append(f"TorchElastic error: {msg}")
                            if py_callstack:
                                for line in str(py_callstack).splitlines():
                                    state.logs.append(line)
                    except Exception:
                        pass
                state.logs.append(f"Process exited with code {proc.returncode}")
                state.error_summary = _extract_error_summary(list(state.logs))

    t = threading.Thread(target=_reader, daemon=True)
    t.start()

    return {"status": "started"}


@app.get("/api/train/status")
async def training_status():
    """Get current training status and recent logs."""
    return {
        "status": state.status,
        "logs": list(state.logs),
        "config": state.config,
        "error_summary": state.error_summary,
        "exit_code": state.exit_code,
    }

@app.post("/api/train/stop")
async def stop_training():
    """Stop the running training process."""
    with state.lock:
        if state.status != "running" or state.process is None:
            raise HTTPException(400, "No training is running")

        state.logs.append("--- Stopping training ---")
        try:
            if sys.platform == "win32":
                state.process.terminate()
            else:
                os.killpg(os.getpgid(state.process.pid), signal.SIGTERM)
        except Exception:
            state.process.kill()

        state.status = "stopped"

    return {"status": "stopped"}

# ---------------------------------------------------------------------------
# Application Lifecycle
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    """Initialize queue manager on startup."""
    global queue_manager
    queue_manager = QueueManager()
    queue_manager.cleanup_stale_jobs()
    queue_manager.start_worker()
    print("Queue manager initialized")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global queue_manager
    if queue_manager:
        queue_manager.stop_worker()
        print("Queue manager stopped")

# ---------------------------------------------------------------------------
# Routes: Job Queue
# ---------------------------------------------------------------------------

@app.post("/api/queue/submit")
async def queue_submit(payload: QueueSubmitPayload):
    """Submit a job to the queue."""
    if not queue_manager:
        raise HTTPException(500, "Queue manager not initialized")

    mini_cfg = adapt_ui_config_to_mini(payload.config, PROJECT_ROOT)
    try:
        validate_mini_config(mini_cfg, PROJECT_ROOT)
    except Exception as e:
        raise HTTPException(400, f"Invalid training config: {e}")

    strategy = str(mini_cfg.get("strategy", "solo")).lower()
    requested_gpus = 1 if strategy == "solo" else max(1, int(mini_cfg.get("num_gpus", payload.gpu_count) or 1))
    total = queue_manager.get_total_gpus()
    if requested_gpus > total:
        raise HTTPException(
            400,
            f"This machine has {total} GPU(s) but the run asks for {requested_gpus}. Lower num_gpus, "
            f"or launch it on a cluster (python scripts/launch_slurm.py --config <yaml>).",
        )

    # Calculate estimated training time
    estimate = await api_estimate_training_time(ConfigPayload(config=payload.config))
    estimated_seconds = estimate.get("est_total_seconds")

    name = (payload.name or "").strip()[:120] or None
    job = queue_manager.submit_job(
        config=mini_cfg,
        gpu_count=requested_gpus,
        priority=payload.priority,
        estimated_seconds=estimated_seconds,
        name=name,
    )

    response = {
        "job_id": job.id,
        "status": job.status.value,
        "gpu_count": job.gpu_count,
        "gpu_indices": job.gpu_indices,
        "estimated_duration": estimated_seconds,
    }

    if job.status == JobStatus.PENDING:
        response["queue_position"] = queue_manager.get_queue_position(job.id)
        response["estimated_wait"] = queue_manager.get_queue_eta(job.id)
    else:
        response["queue_position"] = None
        response["estimated_wait"] = None

    return response

@app.get("/api/queue/status")
async def queue_status():
    """Get overall queue status and GPU availability."""
    if not queue_manager:
        raise HTTPException(500, "Queue manager not initialized")

    return queue_manager.get_queue_status()

@app.get("/api/queue/jobs")
async def queue_list_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """List all jobs with optional filtering."""
    if not queue_manager:
        raise HTTPException(500, "Queue manager not initialized")

    job_status = None
    if status:
        try:
            job_status = JobStatus(status)
        except ValueError:
            raise HTTPException(400, f"Invalid status: {status}")

    jobs = queue_manager.list_jobs(status=job_status, limit=limit, offset=offset)
    out = []
    for job in jobs:
        item = job.to_dict()
        if job.status != JobStatus.PENDING:
            metrics = queue_manager.get_job_metrics(job.id)
            item["progress"] = metrics["progress"] if job.status == JobStatus.RUNNING else None
            item["summary"] = summarize_metrics(metrics)
        out.append(item)
    return {"jobs": out}

@app.get("/api/queue/jobs/{job_id}")
async def queue_get_job(job_id: str, tail: int = Query(2000, ge=1, le=20000)):
    """Get details for a specific job."""
    if not queue_manager:
        raise HTTPException(500, "Queue manager not initialized")

    job = queue_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job '{job_id}' not found")

    result = job.to_dict()
    result["logs"] = queue_manager.get_job_logs(job_id, tail=tail)
    result["metrics"] = queue_manager.get_job_metrics(job_id)

    if job.status == JobStatus.PENDING:
        result["queue_position"] = queue_manager.get_queue_position(job_id)
        result["estimated_wait"] = queue_manager.get_queue_eta(job_id)

    return result

@app.get("/api/queue/jobs/{job_id}/log", response_class=PlainTextResponse)
async def queue_job_log(job_id: str):
    """The run's complete log as plain text."""
    if not queue_manager:
        raise HTTPException(500, "Queue manager not initialized")
    if not queue_manager.get_job(job_id):   # also keeps arbitrary names out of the file path
        raise HTTPException(404, f"Job '{job_id}' not found")
    return PlainTextResponse("\n".join(queue_manager.get_full_log(job_id)) + "\n")


@app.post("/api/queue/jobs/{job_id}/cancel")
async def queue_cancel_job(job_id: str):
    """Cancel a pending or running job."""
    if not queue_manager:
        raise HTTPException(500, "Queue manager not initialized")

    job = queue_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job '{job_id}' not found")

    success = queue_manager.cancel_job(job_id)
    if not success:
        raise HTTPException(
            400,
            f"Cannot cancel job with status '{job.status.value}'"
        )

    return {
        "success": True,
        "job_id": job_id,
        "status": "cancelled",
    }

@app.delete("/api/queue/jobs/{job_id}")
async def queue_delete_job(job_id: str):
    """Delete a completed/failed/cancelled job from history."""
    if not queue_manager:
        raise HTTPException(500, "Queue manager not initialized")

    job = queue_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job '{job_id}' not found")

    success = queue_manager.delete_job(job_id)
    if not success:
        raise HTTPException(
            400,
            f"Cannot delete job with status '{job.status.value}'. Only completed, failed, or cancelled jobs can be deleted."
        )

    return {"success": True, "job_id": job_id}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    import uvicorn
    host = os.getenv("UI_HOST", "127.0.0.1")
    port = int(os.getenv("UI_PORT", "8787"))
    print("\n OMNI-Train console")
    print(f"  http://{host}:{port}\n")
    if host not in ("127.0.0.1", "localhost"):
        print("  WARNING: this UI has no authentication. Binding to a non-loopback")
        print("  address exposes training control and .env key presence to your")
        print("  network — put it behind a firewall or an authenticated proxy.\n")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
