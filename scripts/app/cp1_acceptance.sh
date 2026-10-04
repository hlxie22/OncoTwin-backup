#!/usr/bin/env bash
# Outer shell survives a strict child so a failed acceptance does not kill an interactive allocation.
set +e
ROOT="${1:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  DEFAULT_PY="${CONDA_PREFIX}/bin/python"
else
  DEFAULT_PY=python3
fi
PY="${ONCOTWIN_PYTHON:-$DEFAULT_PY}"
. "$ROOT/scripts/app/common_env.sh"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$ROOT/artifacts/app_cp1/acceptance_$STAMP"
mkdir -p "$OUT"
export ROOT PY OUT

bash -euo pipefail <<'STRICT' >"$OUT/full.log" 2>&1
ROOT="${ROOT:?}"
PY="${PY:?}"
OUT="${OUT:?}"
TMP="$OUT/runtime"
mkdir -p "$TMP/storage"
export ONCOTWIN_ENVIRONMENT=test
export ONCOTWIN_DATABASE_URL="sqlite:///$TMP/acceptance.db"
export ONCOTWIN_STORAGE_BACKEND=local
export ONCOTWIN_STORAGE_ROOT="$TMP/storage"
export ONCOTWIN_TASKS_EAGER=true
export ONCOTWIN_AUTO_CREATE_SCHEMA=false
export ONCOTWIN_DEMO_MODE=true
export ONCOTWIN_JWT_SECRET=cp1-acceptance-secret
export ONCOTWIN_EXTRACTOR_PROVIDER=rules

cd "$ROOT/services/oncotwin-api"
"$PY" -m alembic upgrade head
"$PY" -m pytest --disable-warnings --maxfail=1 | tee "$OUT/pytest.txt"

cd "$ROOT/apps/oncotwin-web"
npm run build | tee "$OUT/next_build.txt"

"$PY" - <<'PY'
import json, os, shutil, subprocess
summary={
  "status":"PASS",
  "python": subprocess.check_output([os.environ.get("PY", "python3"), "--version"], text=True, stderr=subprocess.STDOUT).strip() if False else "see full.log",
  "node": subprocess.check_output(["node","--version"], text=True).strip(),
  "tesseract_available": shutil.which("tesseract") is not None,
  "database":"fresh sqlite migration for cluster acceptance; production schema targets PostgreSQL via SQLAlchemy/Alembic",
  "extractor":"deterministic golden-test provider",
  "notes":["Set ONCOTWIN_EXTRACTOR_PROVIDER=openai plus ONCOTWIN_OPENAI_API_KEY for LLM structured extraction."]
}
with open(os.path.join(os.environ["OUT"],"summary.json"),"w") as f: json.dump(summary,f,indent=2)
PY
STRICT
STATUS=$?
if [[ $STATUS -ne 0 ]]; then
  cat >"$OUT/summary.json" <<JSON
{"status":"FAIL","exit_code":$STATUS,"full_log":"$OUT/full.log"}
JSON
fi

echo "=== ONCOTWIN CP1 ACCEPTANCE ==="
cat "$OUT/summary.json"
echo "--- tail(full.log) ---"
tail -120 "$OUT/full.log"
echo "ARTIFACT_DIR=$OUT"
echo "EXIT_CODE=$STATUS"
exit 0
