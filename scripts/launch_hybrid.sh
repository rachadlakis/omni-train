#!/bin/bash
# Launch script for 3D parallelism training
#
# Usage:
#   bash scripts/launch_hybrid.sh
#   CONFIG_PATH=configs/hybrid/llm_3d_16gpu.yaml bash scripts/launch_hybrid.sh
#   NUM_GPUS=4 bash scripts/launch_hybrid.sh

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default config
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_DIR/configs/hybrid/llm_3d_8gpu.yaml}"

# Parse num_gpus from config if not overridden
if [ -z "$NUM_GPUS" ]; then
    NUM_GPUS=$(grep "^num_gpus:" "$CONFIG_PATH" | awk '{print $2}')
    NUM_GPUS="${NUM_GPUS:-8}"
fi

echo "=============================================="
echo "OMNI-Train 3D Parallelism Launcher"
echo "=============================================="
echo "Config: $CONFIG_PATH"
echo "GPUs: $NUM_GPUS"
echo "=============================================="

# Activate virtual environment if exists
if [ -f "$PROJECT_DIR/.venv/bin/activate" ]; then
    source "$PROJECT_DIR/.venv/bin/activate"
fi

# Load environment variables
if [ -f "$PROJECT_DIR/.env" ]; then
    export $(grep -v '^#' "$PROJECT_DIR/.env" | xargs)
fi

# Set NCCL environment for multi-node
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-2}"

# Launch with torchrun
cd "$PROJECT_DIR"

torchrun \
    --nproc_per_node="$NUM_GPUS" \
    --master_port="${MASTER_PORT:-29500}" \
    hybrid/train_hybrid.py \
    --config "$CONFIG_PATH" \
    "$@"

echo "=============================================="
echo "Training complete!"
echo "=============================================="
