#!/bin/bash
# GPU check: run every test that needs real GPUs, on a rented machine (RunPod,
# Lambda, vast.ai, Kaggle 2xT4...). CI on GitHub covers the CPU suites; this covers
# what CI cannot: NCCL, multi-GPU DDP/FSDP, FSDP2 at all, bitsandbytes, bf16 on GPU.
#
#   bash scripts/gpu_check.sh --setup     # first time on a fresh pod: venv + installs
#   bash scripts/gpu_check.sh             # quick (default): ~15-40 min depending on GPU count
#   bash scripts/gpu_check.sh --full      # + the whole 1-GPU smoke grid (~300 cases, hours)
#
# What runs (cases that need more GPUs than the machine has are skipped, not failed):
#   1. environment report (GPUs, topology, driver, torch)
#   2. NCCL: correctness of all-reduce / all-gather / reduce-scatter / broadcast + bandwidth
#   3. unit tests (sanity of the install)
#   4. -m dist on GPUs over NCCL (OMNI_DIST_DEVICE=cuda): 5d layouts and post-training
#      DDP/FSDP vs 1 process, on losses AND every parameter's gradient
#   5. smoke: real train.py runs (multi-GPU DDP/FSDP, save/resume, QLoRA, post-training,
#      vision/detection/VLM); downloads opt-125m and small datasets from the HF Hub
#
# Logs: gpu_check_logs/<timestamp>/. Exit code 0 only if every step passed.
# Put HF_TOKEN in .env for gated models (not needed by the default cases).

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODE=quick
SETUP=0
for arg in "$@"; do
    case "$arg" in
        --quick) MODE=quick ;;
        --full) MODE=full ;;
        --setup) SETUP=1 ;;
        -h|--help) sed -n 2,22p "$0"; exit 0 ;;
        *) echo "unknown option: $arg (use --setup, --quick, --full)"; exit 2 ;;
    esac
done

LOG_DIR="gpu_check_logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
declare -a NAMES RESULTS TIMES

step() {  # step "name" command...  -> runs, logs, records PASS/FAIL, never aborts
    local name="$1"; shift
    local log="$LOG_DIR/$(echo "$name" | tr -cs 'A-Za-z0-9' '_' | sed 's/_*$//').log"
    echo
    echo "=== $name"
    echo "    log: $log"
    local start=$SECONDS
    "$@" 2>&1 | tee "$log" | grep -E --line-buffered "passed|failed|skipped|error|OK |FAIL|bandwidth|GB/s|check" | tail -n 20
    local rc=${PIPESTATUS[0]}  # exit code of the command itself, not of the filters
    NAMES+=("$name"); TIMES+=("$((SECONDS - start))s")
    if [ "$rc" -eq 0 ]; then RESULTS+=("PASS"); else RESULTS+=("FAIL (exit $rc)"); fi
}

# ------------------------------------------------------------------ setup
if [ "$SETUP" -eq 1 ]; then
    echo "=== setup: .venv + torch 2.6.0 (cu124) + requirements.txt"
    python3 -m venv .venv
    # shellcheck disable=SC1091
    source .venv/bin/activate
    pip install --upgrade pip
    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
    pip install -r requirements.txt
elif [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
else
    echo "note: no .venv found, using $(command -v python) (run with --setup for the pinned versions)"
fi

set -a; [ -f .env ] && . ./.env; set +a
export HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1

# ------------------------------------------------------------ environment
N_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
{
    echo "date: $(date -Is)   host: $(hostname)   mode: $MODE"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv 2>&1
    echo; echo "topology (NV# = NVLink, PIX/PXB/PHB/SYS = PCIe paths):"
    nvidia-smi topo -m 2>&1
    echo
    python -c "import torch, transformers, peft; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| transformers', transformers.__version__, '| peft', peft.__version__)"
    python -c "import torch; [print(f'  cuda:{i}', torch.cuda.get_device_name(i), 'bf16' if torch.cuda.is_bf16_supported() else 'no-bf16') for i in range(torch.cuda.device_count())]"
} 2>&1 | tee "$LOG_DIR/environment.txt"

if [ "$N_GPUS" -lt 1 ]; then
    echo "No CUDA GPU visible to PyTorch; nothing to check here (CI covers the CPU suites)."
    exit 1
fi
echo
echo ">>> $N_GPUS GPU(s). Multi-GPU cases need >= 2 (some need 4 or 8); the rest are skipped."

# ------------------------------------------------------------------ steps
if [ "$N_GPUS" -ge 2 ]; then
    step "nccl" torchrun --nproc_per_node="$N_GPUS" --master_port=29511 scripts/nccl_check.py
else
    step "nccl (1 GPU: collectives only)" torchrun --nproc_per_node=1 --master_port=29511 scripts/nccl_check.py
fi

step "unit tests" python -m pytest -q -p no:cacheprovider

step "dist on GPU (NCCL)" env OMNI_DIST_DEVICE=cuda python -m pytest -m dist -q -rs -p no:cacheprovider

SMOKE_QUICK="2gpu or 4gpu or post_ or save_resume or qlora_quant or fsdp_peft_lora or vision or detection or vlm"
step "smoke (curated)" python -m pytest -m smoke -q -rs -p no:cacheprovider -k "$SMOKE_QUICK"

if [ "$MODE" = full ]; then
    step "smoke (full grid)" python -m pytest -m smoke -q -rs -p no:cacheprovider -k "not ($SMOKE_QUICK)"
fi

# ---------------------------------------------------------------- summary
echo
echo "================ GPU check summary ($N_GPUS GPU(s), $MODE) ================"
for i in "${!NAMES[@]}"; do
    printf "  %-34s %-16s %s\n" "${NAMES[$i]}" "${RESULTS[$i]}" "${TIMES[$i]}"
done | tee "$LOG_DIR/summary.txt"
FAILED=0
grep -q FAIL "$LOG_DIR/summary.txt" && FAILED=1
echo "logs: $LOG_DIR"
exit $FAILED
