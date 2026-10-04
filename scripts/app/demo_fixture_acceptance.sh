#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
source scripts/app/common_env.sh
PY_BIN="${PY:-${ONCOTWIN_PYTHON:-python3}}"
export PYTHONPATH="$REPO_ROOT/services/oncotwin-api/src:${PYTHONPATH:-}"

export ONCOTWIN_DEMO_MODE=true
export ONCOTWIN_DEMO_FIXTURE_ID="${ONCOTWIN_DEMO_FIXTURE_ID:-maya_rowan_v1}"
export ONCOTWIN_DEMO_PATIENT_ROOT="${ONCOTWIN_DEMO_PATIENT_ROOT:-$REPO_ROOT/demo/synthetic_patient_v1}"
unset ONCOTWIN_OPENAI_API_KEY || true

echo "=== DEMO FIXTURE BUNDLE VALIDATION ==="
"$PY_BIN" scripts/app/validate_demo_fixture_bundle.py

echo
echo "=== DEMO FIXTURE TESTS ==="
(
  cd services/oncotwin-api
  "$PY_BIN" -m pytest -q tests/test_demo_fixtures.py
)

echo
echo "=== NORMAL-MODE BACKEND REGRESSION ==="
(
  cd services/oncotwin-api
  env \
    -u ONCOTWIN_DEMO_MODE \
    -u ONCOTWIN_DEMO_FIXTURE_ID \
    -u ONCOTWIN_DEMO_PATIENT_ROOT \
    "$PY_BIN" -m pytest -q tests/test_semantics.py tests/test_checkpoint1_flow.py
)

echo
echo "DEMO_FIXTURE_ACCEPTANCE_PASS"
