#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  DEFAULT_PY="${CONDA_PREFIX}/bin/python"
else
  DEFAULT_PY=python3
fi
PY="${ONCOTWIN_PYTHON:-$DEFAULT_PY}"
. "$ROOT/scripts/app/common_env.sh"

echo "[setup] repo: $ROOT"
echo "[setup] python: $PY"
"$PY" --version
command -v node >/dev/null || { echo "ERROR: node is not on PATH. Load your cluster Node module first (Node 20+)."; exit 2; }
command -v npm >/dev/null || { echo "ERROR: npm is not on PATH."; exit 2; }
node --version
npm --version

"$PY" -m pip install -e "$ROOT/services/oncotwin-api[dev]"
cd "$ROOT/apps/oncotwin-web"
npm install

echo "[setup] complete"
