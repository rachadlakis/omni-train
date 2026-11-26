#!/bin/bash

set -euo pipefail ## safer bash scripting

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

CONFIG_PATH="${CONFIG_PATH:-$SCRIPT_DIR/../config.yaml}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"

# "<strategy> <num_gpus>" of a config
read_config() {
    CONFIG_PATH="$1" python - <<'PY'
import os
import sys
import yaml
from pathlib import Path
p = Path(os.environ["CONFIG_PATH"])
if not p.exists():
    sys.exit(f"Config file not found: {p}")
cfg = yaml.safe_load(p.read_text()) or {}
num_gpus = cfg.get("num_gpus", 1) or 1
auto = str(num_gpus).strip().lower() == "auto"
print(str(cfg.get("strategy", "auto")).strip().lower(), "auto" if auto else max(1, int(num_gpus)))
PY
}

FROM_CFG="$(read_config "$CONFIG_PATH")"
read -r CFG_STRATEGY CFG_NUM_GPUS <<< "$FROM_CFG"
STRATEGY="${STRATEGY:-$CFG_STRATEGY}"
NUM_GPUS="${NUM_GPUS:-$CFG_NUM_GPUS}"

# strategy: auto -> advisor.py looks at the model, the data and the GPUs, explains its choice and writes
# the config to run. A STRATEGY / NUM_GPUS given on the command line is written into that config too,
# because train.py reads the strategy from the config.
if [ "$STRATEGY" = "auto" ] || [ "$STRATEGY" != "$CFG_STRATEGY" ] || [ "$NUM_GPUS" != "$CFG_NUM_GPUS" ]; then
    RESOLVED="$(mktemp "${TMPDIR:-/tmp}/omni_run_XXXXXX")"
    trap 'rm -f "$RESOLVED"' EXIT
    GPU_ARGS=()
    if [ "$NUM_GPUS" != "auto" ]; then GPU_ARGS=(--gpus "$NUM_GPUS"); fi
    python advisor.py --config "$CONFIG_PATH" --write "$RESOLVED" --strategy "$STRATEGY" ${GPU_ARGS[@]+"${GPU_ARGS[@]}"}
    CONFIG_PATH="$RESOLVED"
    FROM_CFG="$(read_config "$CONFIG_PATH")"
    read -r STRATEGY NUM_GPUS <<< "$FROM_CFG"
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
