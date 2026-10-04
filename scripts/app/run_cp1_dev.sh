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
DATA="${ONCOTWIN_APP_DATA_DIR:-$ROOT/.oncotwin-dev}"
WEB_PORT="${ONCOTWIN_WEB_PORT:-3000}"
API_PORT="${ONCOTWIN_API_PORT:-8000}"
mkdir -p "$DATA/storage"

export ONCOTWIN_ENVIRONMENT=development
export ONCOTWIN_DATABASE_URL="${ONCOTWIN_DATABASE_URL:-sqlite:///$DATA/oncotwin.db}"
export ONCOTWIN_STORAGE_BACKEND="${ONCOTWIN_STORAGE_BACKEND:-local}"
export ONCOTWIN_STORAGE_ROOT="${ONCOTWIN_STORAGE_ROOT:-$DATA/storage}"
export ONCOTWIN_TASKS_EAGER="${ONCOTWIN_TASKS_EAGER:-true}"
export ONCOTWIN_AUTO_CREATE_SCHEMA="${ONCOTWIN_AUTO_CREATE_SCHEMA:-false}"
export ONCOTWIN_DEMO_MODE="${ONCOTWIN_DEMO_MODE:-true}"
export ONCOTWIN_JWT_SECRET="${ONCOTWIN_JWT_SECRET:-cp1-dev-$(id -u)-change-before-hosting}"
export ONCOTWIN_EXTRACTOR_PROVIDER="${ONCOTWIN_EXTRACTOR_PROVIDER:-rules}"
export ONCOTWIN_API_INTERNAL="http://127.0.0.1:$API_PORT"

"$ROOT/scripts/app/migrate_cp1.sh" "$ROOT"

cleanup() {
  [[ -n "${API_PID:-}" ]] && kill "$API_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

cd "$ROOT"
"$PY" -m uvicorn oncotwin_api.main:app --host 127.0.0.1 --port "$API_PORT" >"$DATA/api.log" 2>&1 &
API_PID=$!
api_healthcheck() {
  "$PY" - "$API_PORT" >/dev/null 2>&1 <<'PYHEALTH'
import sys
import urllib.request

port = int(sys.argv[1])

try:
    with urllib.request.urlopen(
        f"http://127.0.0.1:{port}/healthz",
        timeout=0.75,
    ) as response:
        if response.status != 200:
            raise SystemExit(1)
except Exception:
    raise SystemExit(1)
PYHEALTH
}

API_READY=0
for _ in $(seq 1 80); do
  if api_healthcheck; then
    API_READY=1
    break
  fi
  sleep .25
done

if [[ "$API_READY" != "1" ]]; then
  echo "API health check failed; see $DATA/api.log"
  tail -100 "$DATA/api.log"
  exit 3
fi

NODE_NAME="$(hostname -f 2>/dev/null || hostname)"
echo
echo "=============================================================="
echo " OncoTwin checkpoint-1 preview is starting on: $NODE_NAME"
echo " Browser port: $WEB_PORT"
echo " API is private on 127.0.0.1:$API_PORT and proxied by Next.js."
echo " Data dir: $DATA"
echo " Extractor: $ONCOTWIN_EXTRACTOR_PROVIDER"
echo "=============================================================="
echo
echo "From YOUR LAPTOP, open an SSH tunnel through your cluster login node:"
echo "  ssh -N -L ${WEB_PORT}:${NODE_NAME}:${WEB_PORT} YOUR_USER@YOUR_LOGIN_HOST"
echo
echo "Then open: http://localhost:${WEB_PORT}"
echo
echo "If your login node cannot resolve ${NODE_NAME}, use the compute node's short hostname from: hostname"
echo "Keep this Slurm allocation and this command running while you preview."
echo

cd "$ROOT/apps/oncotwin-web"
exec npm run dev -- --hostname 0.0.0.0 --port "$WEB_PORT"
