#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESEARCH_PY="${ONCOTWIN_RESEARCH_PY:-/home/henryxie/orcd/scratch/.conda/envs/oncotwin/bin/python3}"
export ONCOTWIN_REPO="$REPO"
export ONCOTWIN_MODEL_DEVICE="${ONCOTWIN_MODEL_DEVICE:-cpu}"
PORT="${ONCOTWIN_MODEL_PORT:-8766}"
exec "$RESEARCH_PY" "$REPO/services/oncotwin-model/model_service.py" --host 127.0.0.1 --port "$PORT"
