#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
source scripts/app/common_env.sh
PORT="${ONCOTWIN_MODEL_PORT:-8766}"
export ONCOTWIN_MODEL_URL="http://127.0.0.1:${PORT}"
mkdir -p artifacts/app_cp2/dev
MODEL_LOG="artifacts/app_cp2/dev/model_service.log"
bash scripts/app/run_model_service.sh >"$MODEL_LOG" 2>&1 &
MODEL_PID=$!
cleanup() { kill "$MODEL_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
"$PY" - <<PY
from urllib.request import urlopen
import time
url='http://127.0.0.1:${PORT}/healthz'
for _ in range(80):
    try:
        print(urlopen(url, timeout=2).read().decode())
        break
    except Exception:
        time.sleep(.25)
else:
    raise SystemExit('model service failed to become healthy')
PY
bash scripts/app/run_cp1_dev.sh
