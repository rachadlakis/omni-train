#!/bin/bash
##
## dist_kit launcher — adapted from OMNI-Train's scripts/launch.sh.
## Reads top-level strategy and num_gpus from the config (same schema as
## OMNI-Train's config.yaml) and decides between `python -m` (solo) and
## `torchrun` (ddp/fsdp). Config resolution:
##   $DIST_CONFIG > ./config.yaml (project root) > dist_kit/config.yaml (bundled)
##
## Usage (run from your project root, the folder containing dist_kit/):
##   bash dist_kit/launch.sh
##   DIST_CONFIG=my_config.yaml bash dist_kit/launch.sh
##   STRATEGY=ddp NUM_GPUS=2 bash dist_kit/launch.sh
##   TRAIN_MODULE=my_pkg.train bash dist_kit/launch.sh    # your own entry point

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

if [ -z "${DIST_CONFIG:-}" ]; then
    if [ -f "config.yaml" ]; then
        DIST_CONFIG="config.yaml"
    else
        DIST_CONFIG="$SCRIPT_DIR/config.yaml"
    fi
fi
echo "   Config: $DIST_CONFIG"
TRAIN_MODULE="${TRAIN_MODULE:-dist_kit.train}"
NUM_GPUS="${NUM_GPUS:-}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"
STRATEGY="${STRATEGY:-}"

if [ -z "$STRATEGY" ] || [ -z "$NUM_GPUS" ]; then
    read -r STRATEGY_FROM_CFG NUM_GPUS_FROM_CFG <<EOF
$(DIST_CONFIG="$DIST_CONFIG" python - <<'PY'
import os
import yaml
from pathlib import Path
p = Path(os.getenv("DIST_CONFIG", "config.yaml"))
if p.exists():
        cfg = yaml.safe_load(p.read_text()) or {}
        strategy = cfg.get("strategy", "fsdp")
        num_gpus = int(cfg.get("num_gpus", 1) or 1)
        print(strategy, max(1, num_gpus))
else:
        print("solo 1")
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
    DIST_CONFIG="$DIST_CONFIG" python -m "$TRAIN_MODULE"
else
    echo "   Running in distributed mode with torchrun | GPUs: $NUM_GPUS"
    DIST_CONFIG="$DIST_CONFIG" torchrun \
        --nproc_per_node="$NUM_GPUS" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        -m "$TRAIN_MODULE"
fi
