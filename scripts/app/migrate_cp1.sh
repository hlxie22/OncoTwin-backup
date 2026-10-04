#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
if [[ -x /home/henryxie/orcd/scratch/.conda/envs/oncotwin/bin/python3 ]]; then DEFAULT_PY=/home/henryxie/orcd/scratch/.conda/envs/oncotwin/bin/python3; else DEFAULT_PY=python3; fi
PY="${ONCOTWIN_PYTHON:-$DEFAULT_PY}"
. "$ROOT/scripts/app/common_env.sh"
cd "$ROOT/services/oncotwin-api"
"$PY" -m alembic upgrade head
