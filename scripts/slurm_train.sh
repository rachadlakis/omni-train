#!/bin/bash
#SBATCH --job-name=omni-train
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# =============================================================================
# OMNI-Train SLURM Multi-Node Training Script
# =============================================================================
# Usage:
#   sbatch scripts/slurm_train.sh configs/llm_full_finetune_fsdp.yaml
#
# Or with custom parameters:
#   sbatch --nodes=4 --gpus-per-node=8 scripts/slurm_train.sh configs/my_config.yaml
# =============================================================================

set -e

# Get config file from argument or use default
CONFIG_FILE="${1:-configs/llm_full_finetune_fsdp.yaml}"

# Create logs directory
mkdir -p logs

# Print job info
echo "=============================================="
echo "SLURM Job ID      : $SLURM_JOB_ID"
echo "Running on nodes  : $SLURM_NODELIST"
echo "Number of nodes   : $SLURM_NNODES"
echo "GPUs per node     : $SLURM_GPUS_PER_NODE"
echo "Config file       : $CONFIG_FILE"
echo "=============================================="

# Get master node address
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=${MASTER_PORT:-29500}

# Calculate total processes
GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-4}
WORLD_SIZE=$((SLURM_NNODES * GPUS_PER_NODE))

echo "Master address    : $MASTER_ADDR:$MASTER_PORT"
echo "World size        : $WORLD_SIZE"
echo "=============================================="

# Set environment variables for NCCL
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Run training with srun + torchrun
srun --kill-on-bad-exit=1 \
    torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    -m omni_train.train \
    --config $CONFIG_FILE

echo "Training complete!"
