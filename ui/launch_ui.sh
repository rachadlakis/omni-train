#!/usr/bin/env bash
set -euo pipefail

HOST="${UI_HOST:-127.0.0.1}"
PORT="${UI_PORT:-8787}"

cd "$(dirname "$0")/.."

if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
fi

echo "Starting UI at http://${HOST}:${PORT}"
uvicorn ui.app:app --host "$HOST" --port "$PORT"
