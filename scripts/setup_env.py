"""
setup_env.py  –  Auto-detect GPU/CUDA and install requirements with the right torch wheel.

Usage:
    python scripts/setup_env.py              # install into current environment
    python scripts/setup_env.py --dry-run    # print the pip command without running it
    python scripts/setup_env.py --cpu        # force CPU-only torch
"""

import subprocess
import sys
import re
import argparse


# Maps detected CUDA major.minor to the closest available PyTorch wheel tag.
# Add newer entries here as PyTorch releases them.
CUDA_TO_WHEEL = {
    (12, 8): "cu128",
    (12, 6): "cu126",
    (12, 4): "cu124",
    (12, 1): "cu121",
    (11, 8): "cu118",
}

PYTORCH_INDEX = "https://download.pytorch.org/whl/{tag}"
PYPI_INDEX    = "https://pypi.org/simple"


def detect_cuda_version():
    """Return (major, minor) from nvidia-smi, or None if no GPU found."""

    # Method 1: Try nvidia-smi directly
    try:
        out = subprocess.check_output(
            ["nvidia-smi"], text=True, stderr=subprocess.DEVNULL
        )
        match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out)
        if match:
            return int(match.group(1)), int(match.group(2))
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    # Method 2: Try nvidia-smi from common Windows paths
    import os
    nvidia_smi_paths = [
        r"C:\Windows\System32\nvidia-smi.exe",
        r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
        os.path.expandvars(r"%ProgramFiles%\NVIDIA Corporation\NVSMI\nvidia-smi.exe"),
    ]

    for path in nvidia_smi_paths:
        if os.path.exists(path):
            try:
                out = subprocess.check_output(
                    [path], text=True, stderr=subprocess.DEVNULL
                )
                match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out)
                if match:
                    print(f"  Found nvidia-smi at: {path}")
                    return int(match.group(1)), int(match.group(2))
            except (FileNotFoundError, subprocess.CalledProcessError):
                pass

    # Method 3: Try nvcc (CUDA toolkit compiler)
    try:
        out = subprocess.check_output(
            ["nvcc", "--version"], text=True, stderr=subprocess.DEVNULL
        )
        match = re.search(r"release (\d+)\.(\d+)", out)
        if match:
            print("  Detected CUDA via nvcc")
            return int(match.group(1)), int(match.group(2))
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    # Method 4: Check if torch is already installed and has CUDA
    try:
        result = subprocess.run(
            [sys.executable, "-c",
             "import torch; print(torch.version.cuda if torch.cuda.is_available() else 'none')"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            cuda_ver = result.stdout.strip()
            if cuda_ver and cuda_ver != 'none':
                match = re.match(r"(\d+)\.(\d+)", cuda_ver)
                if match:
                    print("  Detected CUDA via existing torch installation")
                    return int(match.group(1)), int(match.group(2))
    except Exception:
        pass

    return None


def pick_wheel_tag(cuda: tuple[int, int]) -> str:
    """Return the best matching wheel tag for the detected CUDA version."""
    # Exact match first
    if cuda in CUDA_TO_WHEEL:
        return CUDA_TO_WHEEL[cuda]
    # Fall back to the highest supported tag that is <= detected version
    compatible = [k for k in CUDA_TO_WHEEL if k <= cuda]
    if compatible:
        best = max(compatible)
        print(f"  No exact wheel for CUDA {cuda[0]}.{cuda[1]}, using {CUDA_TO_WHEEL[best]} (closest compatible).")
        return CUDA_TO_WHEEL[best]
    # Nothing compatible
    print(f"  Warning: No PyTorch wheel found for CUDA {cuda[0]}.{cuda[1]}. Falling back to CPU.")
    return "cpu"


def build_pip_command(wheel_tag: str, requirements_file: str) -> list[str]:
    if wheel_tag == "cpu":
        index_url = PYTORCH_INDEX.format(tag="cpu")
    else:
        index_url = PYTORCH_INDEX.format(tag=wheel_tag)

    return [
        sys.executable, "-m", "pip", "install",
        "--index-url", index_url,
        "--extra-index-url", PYPI_INDEX,
        "-r", requirements_file,
    ]


def main():
    parser = argparse.ArgumentParser(description="Install dependencies with the correct torch wheel.")
    parser.add_argument("--dry-run", action="store_true", help="Print the pip command but do not run it.")
    parser.add_argument("--cpu",     action="store_true", help="Force CPU-only torch regardless of GPU.")
    parser.add_argument("--requirements", default="requirements.txt",
                        help="Path to requirements file (default: requirements.txt)")
    args = parser.parse_args()

    print("=== Environment Setup ===\n")

    if args.cpu:
        wheel_tag = "cpu"
        print("  Mode: CPU-only (forced)")
    else:
        cuda = detect_cuda_version()
        if cuda:
            print(f"  Detected GPU  : yes")
            print(f"  CUDA version  : {cuda[0]}.{cuda[1]}")
            wheel_tag = pick_wheel_tag(cuda)
            print(f"  PyTorch wheel : {wheel_tag}")
        else:
            print("  No NVIDIA GPU detected — installing CPU-only torch.")
            wheel_tag = "cpu"

    print(f"  Python        : {sys.executable}")
    print(f"  Requirements  : {args.requirements}\n")

    cmd = build_pip_command(wheel_tag, args.requirements)
    print("Running:", " ".join(cmd), "\n")

    if args.dry_run:
        print("(dry-run — nothing installed)")
        return

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("\nInstallation failed. Check the output above.")
        sys.exit(result.returncode)
    else:
        print("\nDone! Verifying torch import...")
        verify = subprocess.run([sys.executable, "-c", "import torch; print(f'  torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"])
        if verify.returncode != 0:
            print("  torch import failed — something went wrong.")


if __name__ == "__main__":
    main()
