#!/bin/bash

set -euo pipefail ## safer bash scripting

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

CONFIG_PATH="${CONFIG_PATH:-$SCRIPT_DIR/../config.yaml}"
NUM_GPUS=""
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"
STRATEGY="${STRATEGY:-}"

if [ -z "$STRATEGY" ] || [ -z "$NUM_GPUS" ]; then
    read -r STRATEGY_FROM_CFG NUM_GPUS_FROM_CFG <<EOF
$(CONFIG_PATH="$CONFIG_PATH" python - <<'PY'
import os
import yaml
from pathlib import Path
p = Path(os.getenv("CONFIG_PATH", "config.yaml"))
if p.exists():
        cfg = yaml.safe_load(p.read_text()) or {}
        strategy = cfg.get("strategy", "fsdp")
        num_gpus = int(cfg.get("num_gpus", 1) or 1)
        print(strategy, max(1, num_gpus))
else:
        print("fsdp 1")
PY
)
EOF

    if [ -z "$STRATEGY" ]; then
        STRATEGY="$STRATEGY_FROM_CFG"
    fi
    if [ -z "$NUM_GPUS" ]; then
        NUM_GPUS="$NUM_GPUS_FROM_CFG"
    fi
fi

echo "🚀 Running training..."

if [ "$STRATEGY" = "solo" ]; then
    echo "   Running in solo mode (single process, no torchrun)"
        CONFIG_PATH="$CONFIG_PATH" python train.py
else
    echo "   Running in distributed mode with torchrun | GPUs: $NUM_GPUS"
        CONFIG_PATH="$CONFIG_PATH" torchrun \
            --nproc_per_node="$NUM_GPUS" \
            --master_addr="$MASTER_ADDR" \
            --master_port="$MASTER_PORT" \
      train.py
fi