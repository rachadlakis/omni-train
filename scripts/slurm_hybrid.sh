#!/bin/bash
#SBATCH --job-name=omni-train-3d
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# SLURM script for multi-node 3D parallelism training
#
# Usage:
#   sbatch scripts/slurm_hybrid.sh configs/hybrid/llm_3d_16gpu.yaml
#
# Customize the #SBATCH directives above for your cluster.

set -e

# Get config from argument or use default
CONFIG_PATH="${1:-configs/hybrid/llm_3d_16gpu.yaml}"

# Project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# Create logs directory
mkdir -p logs

echo "=============================================="
echo "OMNI-Train 3D Parallelism (SLURM)"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"
echo "Config: $CONFIG_PATH"
echo "=============================================="

# Load modules (customize for your cluster)
# module load cuda/12.4
# module load nccl/2.20

# Activate environment
if [ -f "$PROJECT_DIR/.venv/bin/activate" ]; then
    source "$PROJECT_DIR/.venv/bin/activate"
fi

# Load environment variables
if [ -f "$PROJECT_DIR/.env" ]; then
    export $(grep -v '^#' "$PROJECT_DIR/.env" | xargs)
fi

# NCCL settings for multi-node
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_SOCKET_IFNAME=eth0  # Adjust for your network interface

# Get master node info
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500

export MASTER_ADDR
export MASTER_PORT

echo "Master: $MASTER_ADDR:$MASTER_PORT"

# Calculate total GPUs
GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-8}
NNODES=${SLURM_JOB_NUM_NODES:-1}
WORLD_SIZE=$((GPUS_PER_NODE * NNODES))

echo "World size: $WORLD_SIZE"

# Launch with srun + torchrun
srun --kill-on-bad-exit=1 \
    torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    hybrid/train_hybrid.py \
    --config "$CONFIG_PATH"

echo "=============================================="
echo "Training complete!"
echo "Job ID: $SLURM_JOB_ID"
echo "=============================================="
