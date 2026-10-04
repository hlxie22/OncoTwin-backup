#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
source scripts/app/common_env.sh
RESEARCH_PY="${ONCOTWIN_RESEARCH_PY:-/home/henryxie/orcd/scratch/.conda/envs/oncotwin/bin/python3}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="artifacts/app_cp2/acceptance_${STAMP}"
mkdir -p "$OUT"
LOG="$OUT/full.log"
SUMMARY="$OUT/summary.json"
PORT="${ONCOTWIN_CP2_TEST_MODEL_PORT:-18766}"
MODEL_PID=""

cleanup() {
  if [ -n "$MODEL_PID" ]; then kill "$MODEL_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM

{
  echo "[CP2] repository=$REPO"
  echo "[CP2] app_python=$PY"
  echo "[CP2] research_python=$RESEARCH_PY"

  echo "[CP2] 1/6 compile new Python"
  "$PY" -m py_compile services/oncotwin-api/src/oncotwin_api/forecast_service.py
  "$RESEARCH_PY" -m py_compile services/oncotwin-model/model_service.py

  echo "[CP2] 2/6 frozen model + repaired adapter self-test"
  ONCOTWIN_REPO="$REPO" ONCOTWIN_MODEL_DEVICE=cpu \
    "$RESEARCH_PY" services/oncotwin-model/model_service.py --self-test \
    > "$OUT/model_self_test.raw.log"

  "$PY" - "$OUT/model_self_test.raw.log" "$OUT/model_self_test.json" <<'PYSELFTEST'
import json
import sys
from pathlib import Path

raw_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])

raw = raw_path.read_text()
lines = raw.splitlines()

# Research preprocessing prints diagnostic lines before the final JSON payload.
# Locate the top-level JSON object and persist a clean machine-readable copy.
start = next(
    (i for i, line in enumerate(lines) if line == "{"),
    None,
)
if start is None:
    raise SystemExit(f"No JSON object found in {raw_path}")

payload = json.loads("\n".join(lines[start:]))
out_path.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
PYSELFTEST

  cat "$OUT/model_self_test.json"

  echo "[CP2] 3/6 start localhost-only research model service"
  ONCOTWIN_REPO="$REPO" ONCOTWIN_MODEL_DEVICE=cpu \
    "$RESEARCH_PY" services/oncotwin-model/model_service.py --host 127.0.0.1 --port "$PORT" \
    > "$OUT/model_service.log" 2>&1 &
  MODEL_PID=$!
  "$PY" - <<PY
from urllib.request import urlopen
import json,time
url='http://127.0.0.1:${PORT}/healthz'
for _ in range(120):
    try:
        payload=json.loads(urlopen(url, timeout=2).read())
        assert payload['ok'] is True
        assert payload['locked_alpha'] == 0.52
        assert payload['upstream_hashes_verified'] is True
        print(json.dumps(payload, indent=2, sort_keys=True))
        break
    except Exception:
        time.sleep(.25)
else:
    raise SystemExit('model service health timeout')
PY

  echo "[CP2] 4/6 CP1 regression + Alembic-head migration + backend tests + web build"
  export ONCOTWIN_MODEL_URL="http://127.0.0.1:${PORT}"
  bash scripts/app/cp1_acceptance.sh

echo "[CP2] 5/6 verify frontend forecast route exists in production build source"

FORECAST_PAGE="apps/oncotwin-web/app/app/forecast/page.tsx"

test -f "$FORECAST_PAGE"

# CP2 owns the quantitative trajectory capability, not exact patient-facing copy.
# CP3/CP4 are allowed to evolve wording and layout while preserving this contract.
grep -Fq "ForecastChart" "$FORECAST_PAGE"
grep -Fq "ModelSupport" "$FORECAST_PAGE"
grep -Fq "/forecast" "$FORECAST_PAGE"

# The immediately preceding `npm run build` must have emitted the route.
# Accept either the Next app-path manifest or the generated route artifact,
# rather than relying on an old literal UI phrase.
"$PY" - <<'PY2'
import json
from pathlib import Path

root = Path("apps/oncotwin-web/.next")

candidates = [
    root / "server" / "app-paths-manifest.json",
    root / "app-build-manifest.json",
    root / "build-manifest.json",
]

seen = []
route_found = False

for path in candidates:
    if not path.is_file():
        continue
    seen.append(str(path))
    text = path.read_text(errors="replace")
    if "/app/forecast" in text or "app/forecast" in text:
        route_found = True
        break

if not route_found:
    generated = list((root / "server" / "app" / "app" / "forecast").glob("*"))
    if generated:
        route_found = True
        seen.extend(str(x) for x in generated[:10])

assert route_found, (
    "Production Next.js build did not expose /app/forecast. "
    f"Inspected: {seen}"
)

print("CP2_FRONTEND_FORECAST_ROUTE=PASS")
PY2

  echo "[CP2] 6/6 immutable release hashes"
  DEPLOY_SHA="$(sha256sum artifacts/dynamic_scan_v2/v2_07/deploy_model.pt | awk '{print $1}')"
  TEMP_SHA="$(sha256sum artifacts/checkpoint7r1_line_agnostic_temporal/temporal_encoder.pt | awk '{print $1}')"
  test "$DEPLOY_SHA" = "a2de9909b2f7323dbafb1e9d9343946815bea160791b53f10eb31b85c4eaa4be"
  test "$TEMP_SHA" = "daa47e1d30d61a3c3a6c86a146c683a0a1ac0a635e0d8e6247f3ce47c4ae1411"

  "$PY" - <<PY > "$SUMMARY"
import json
from pathlib import Path
selftest=json.loads(Path('$OUT/model_self_test.json').read_text())
print(json.dumps({
  'status':'PASS',
  'checkpoint':'APP_CP2',
  'artifact_dir':'$OUT',
  'deploy_sha256':'a2de9909b2f7323dbafb1e9d9343946815bea160791b53f10eb31b85c4eaa4be',
  'temporal_encoder_sha256':'daa47e1d30d61a3c3a6c86a146c683a0a1ac0a635e0d8e6247f3ce47c4ae1411',
  'locked_alpha':0.52,
  'research_status':'EXTERNAL_CONFIRMATION_PENDING',
  'fixture':selftest['release_fixture'],
  'adapter_smoke':selftest['adapter_smoke'],
  'acceptance':{
    'python_compile':True,
    'frozen_fixture':True,
    'r1_adapter_smoke':True,
    'model_service_health':True,
    'cp1_regression_and_migration':True,
    'frontend_route':True,
    'immutable_hashes':True,
  }
}, indent=2, sort_keys=True))
PY
  cat "$SUMMARY"
} 2>&1 | tee "$LOG"

echo "CP2_ACCEPTANCE_ARTIFACT_DIR=$OUT"
echo "CP2_ACCEPTANCE_SUMMARY=$SUMMARY"
echo "CP2_ACCEPTANCE_PASS"
