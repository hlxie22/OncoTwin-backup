#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
source scripts/app/common_env.sh
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="artifacts/app_cp3/acceptance_${STAMP}"
mkdir -p "$OUT"
LOG="$OUT/full.log"
SUMMARY="$OUT/summary.json"
PORT="${ONCOTWIN_CP3_WEB_PORT:-19003}"
WEB_PID=""

cleanup() {
  if [ -n "$WEB_PID" ]; then kill "$WEB_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM

{
  echo "[CP3] repository=$REPO"
  echo "[CP3] app_python=$PY"

  echo "[CP3] 1/5 full CP2 scientific/platform regression"
  bash scripts/app/cp2_acceptance.sh

  echo "[CP3] 2/5 patient-experience source contract"
  "$PY" scripts/app/cp3_ui_contract.py | tee "$OUT/ui_contract.json"

  echo "[CP3] 3/5 verify final routes are part of the production build"
  for route in update forecast records verify timeline; do
    test -f "apps/oncotwin-web/.next/server/app/app/${route}.html" \
      -o -f "apps/oncotwin-web/.next/server/app/app/${route}/page.js" \
      -o -f "apps/oncotwin-web/.next/server/app/app/${route}/page_client-reference-manifest.js"
  done

  echo "[CP3] 4/5 production-server route smoke"
  (
    cd apps/oncotwin-web
    npm start -- -p "$PORT"
  ) > "$OUT/next_start.log" 2>&1 &
  WEB_PID=$!

  "$PY" - <<PY | tee "$OUT/route_smoke.json"
from urllib.request import urlopen
import json, time
base='http://127.0.0.1:${PORT}'
routes=['/app','/app/update','/app/forecast','/app/records','/app/verify','/app/timeline']
for _ in range(120):
    try:
        urlopen(base + '/app', timeout=2).read(64)
        break
    except Exception:
        time.sleep(.25)
else:
    raise SystemExit('Next production server failed to become ready')
result={}
for route in routes:
    with urlopen(base + route, timeout=5) as response:
        body=response.read()
        result[route]={'status':response.status,'bytes':len(body)}
        if response.status != 200 or len(body) < 100:
            raise SystemExit(f'Bad route response: {route} {result[route]}')
print(json.dumps(result, indent=2, sort_keys=True))
PY

  echo "[CP3] 5/5 accessibility/responsive contract + release packet"
  grep -q ':focus-visible' apps/oncotwin-web/app/globals.css
  grep -q 'prefers-reduced-motion' apps/oncotwin-web/app/globals.css
  grep -q 'aria-modal="true"' apps/oncotwin-web/components/SourceViewer.tsx
  grep -q "event.key === 'Escape'" apps/oncotwin-web/components/SourceViewer.tsx

  "$PY" - <<PY > "$SUMMARY"
import json
from pathlib import Path
ui=json.loads(Path('$OUT/ui_contract.json').read_text())
routes=json.loads(Path('$OUT/route_smoke.json').read_text())
print(json.dumps({
  'status':'PASS',
  'checkpoint':'APP_CP3',
  'artifact_dir':'$OUT',
  'cp2_regression':True,
  'patient_experience':{
    'overview':True,
    'signature_new_scan_flow':True,
    'trajectory':True,
    'source_viewer':True,
    'records_review_timeline':True,
    'responsive_contract':True,
    'keyboard_source_dialog':True,
  },
  'ui_contract':ui['checks'],
  'route_smoke':routes,
  'scientific_boundary':{
    'model_release':'V2-07',
    'locked_alpha':0.52,
    'external_confirmation':'PENDING',
    'treatment_recommendation':False,
  }
}, indent=2, sort_keys=True))
PY
  cat "$SUMMARY"
} 2>&1 | tee "$LOG"

echo "CP3_ACCEPTANCE_ARTIFACT_DIR=$OUT"
echo "CP3_ACCEPTANCE_SUMMARY=$SUMMARY"
echo "CP3_ACCEPTANCE_PASS"
