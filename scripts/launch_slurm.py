#!/usr/bin/env python3
"""SLURM job launcher for OMNI-Train.

This script generates and submits SLURM jobs for distributed training.

Usage:
    python scripts/launch_slurm.py --config configs/llm_fsdp.yaml --nodes 4 --gpus 8

    # Dry run (print sbatch script without submitting)
    python scripts/launch_slurm.py --config configs/llm_fsdp.yaml --nodes 4 --dry-run
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime


SLURM_TEMPLATE = '''#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node={gpus_per_node}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --partition={partition}
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
{extra_sbatch}

set -e

# Print job info
echo "=============================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on nodes: $SLURM_NODELIST"
echo "Number of nodes: $SLURM_NNODES"
echo "GPUs per node: {gpus_per_node}"
echo "Config file: {config}"
echo "=============================================="

# Create logs directory
mkdir -p logs

# Master node setup
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT={master_port}
WORLD_SIZE=$((SLURM_NNODES * {gpus_per_node}))

echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "World size: $WORLD_SIZE"

# Environment setup
{env_setup}

# NCCL settings
export NCCL_DEBUG={nccl_debug}
export NCCL_IB_DISABLE={nccl_ib_disable}
{extra_env}

# Activate conda/venv if specified
{activate_env}

# Run training
srun --kill-on-bad-exit=1 \\
    torchrun \\
    --nnodes=$SLURM_NNODES \\
    --nproc_per_node={gpus_per_node} \\
    --rdzv_id=$SLURM_JOB_ID \\
    --rdzv_backend=c10d \\
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \\
    -m omni_train.train \\
    --config {config} {extra_args}

echo "Training complete!"
'''


def parse_args():
    parser = argparse.ArgumentParser(description="Launch SLURM training job")

    # Required
    parser.add_argument("--config", type=str, required=True,
                        help="Path to training config YAML")

    # SLURM resources
    parser.add_argument("--nodes", type=int, default=2,
                        help="Number of nodes (default: 2)")
    parser.add_argument("--gpus", type=int, default=4,
                        help="GPUs per node (default: 4)")
    parser.add_argument("--cpus", type=int, default=32,
                        help="CPUs per task (default: 32)")
    parser.add_argument("--mem", type=str, default="256G",
                        help="Memory per node (default: 256G)")
    parser.add_argument("--time", type=str, default="24:00:00",
                        help="Time limit (default: 24:00:00)")
    parser.add_argument("--partition", type=str, default="gpu",
                        help="SLURM partition (default: gpu)")

    # Job settings
    parser.add_argument("--job-name", type=str, default=None,
                        help="Job name (default: auto-generated)")
    parser.add_argument("--master-port", type=int, default=29500,
                        help="Master port (default: 29500)")

    # Environment
    parser.add_argument("--conda-env", type=str, default=None,
                        help="Conda environment to activate")
    parser.add_argument("--venv", type=str, default=None,
                        help="Path to virtualenv to activate")
    parser.add_argument("--module", type=str, action="append", default=[],
                        help="Module to load (can specify multiple)")

    # NCCL settings
    parser.add_argument("--nccl-debug", type=str, default="WARN",
                        choices=["INFO", "WARN", "ERROR"],
                        help="NCCL debug level (default: WARN)")
    parser.add_argument("--disable-ib", action="store_true",
                        help="Disable InfiniBand")

    # Extra arguments
    parser.add_argument("--extra-sbatch", type=str, default="",
                        help="Extra SBATCH directives")
    parser.add_argument("--extra-env", type=str, default="",
                        help="Extra environment variables (KEY=VALUE,...)")
    parser.add_argument("--extra-args", type=str, default="",
                        help="Extra arguments to pass to train.py")

    # Actions
    parser.add_argument("--dry-run", action="store_true",
                        help="Print script without submitting")
    parser.add_argument("--save-script", type=str, default=None,
                        help="Save script to file")

    return parser.parse_args()


def generate_script(args):
    """Generate SLURM batch script."""

    # Job name
    if args.job_name:
        job_name = args.job_name
    else:
        config_name = os.path.basename(args.config).replace(".yaml", "")
        timestamp = datetime.now().strftime("%m%d_%H%M")
        job_name = f"omni_{config_name}_{timestamp}"

    # Environment setup
    env_lines = [f"module load {module}" for module in args.module]
    env_setup = "\n".join(env_lines) if env_lines else "# No modules to load"

    # Conda/venv activation
    if args.conda_env:
        activate_env = f"source $(conda info --base)/etc/profile.d/conda.sh\nconda activate {args.conda_env}"
    elif args.venv:
        activate_env = f"source {args.venv}/bin/activate"
    else:
        activate_env = "# No virtual environment"

    # Extra environment variables
    extra_env_lines = [
        f"export {item.strip()}"
        for item in args.extra_env.split(",")
        if "=" in item
    ]
    extra_env = "\n".join(extra_env_lines) if extra_env_lines else ""

    # Extra SBATCH
    extra_sbatch = ""
    if args.extra_sbatch:
        extra_sbatch = "\n".join(f"#SBATCH {x.strip()}" for x in args.extra_sbatch.split(","))

    script = SLURM_TEMPLATE.format(
        job_name=job_name,
        nodes=args.nodes,
        gpus_per_node=args.gpus,
        cpus_per_task=args.cpus,
        mem=args.mem,
        time=args.time,
        partition=args.partition,
        master_port=args.master_port,
        config=args.config,
        env_setup=env_setup,
        activate_env=activate_env,
        nccl_debug=args.nccl_debug,
        nccl_ib_disable="1" if args.disable_ib else "0",
        extra_env=extra_env,
        extra_sbatch=extra_sbatch,
        extra_args=args.extra_args,
    )

    return script, job_name


def main():
    args = parse_args()

    # Verify config exists
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    # Generate script
    script, job_name = generate_script(args)

    # Dry run - just print
    if args.dry_run:
        print("=" * 60)
        print("Generated SLURM Script (dry run)")
        print("=" * 60)
        print(script)
        print("=" * 60)
        return

    # Save script if requested
    if args.save_script:
        with open(args.save_script, "w") as f:
            f.write(script)
        print(f"Script saved to: {args.save_script}")

    # Create logs directory
    os.makedirs("logs", exist_ok=True)

    # Submit job
    try:
        result = subprocess.run(
            ["sbatch"],
            input=script,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print(f"Job submitted: {result.stdout.strip()}")
            print(f"Job name: {job_name}")
            print(f"Nodes: {args.nodes} x {args.gpus} GPUs = {args.nodes * args.gpus} total GPUs")
            print(f"Config: {args.config}")
            print(f"\nMonitor with: squeue -u $USER")
            print(f"Cancel with: scancel <job_id>")
        else:
            print(f"Error submitting job: {result.stderr}")
            sys.exit(1)

    except FileNotFoundError:
        print("Error: sbatch command not found. Are you on a SLURM cluster?")
        sys.exit(1)


if __name__ == "__main__":
    main()
