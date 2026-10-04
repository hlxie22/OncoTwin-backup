#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
source scripts/app/common_env.sh

export ONCOTWIN_DEMO_MODE=true
export ONCOTWIN_DEMO_FIXTURE_ID="${ONCOTWIN_DEMO_FIXTURE_ID:-maya_rowan_v1}"
export ONCOTWIN_DEMO_PATIENT_ROOT="${ONCOTWIN_DEMO_PATIENT_ROOT:-$REPO_ROOT/demo/synthetic_patient_v1}"
export PYTHONPATH="$REPO_ROOT/services/oncotwin-api/src:${PYTHONPATH:-}"

# Extraction fixture mode should never require a paid LLM key.
unset ONCOTWIN_OPENAI_API_KEY || true

echo "OncoTwin deterministic demo-fixture mode"
echo "  fixture_id=$ONCOTWIN_DEMO_FIXTURE_ID"
echo "  patient_root=$ONCOTWIN_DEMO_PATIENT_ROOT"
echo "  live LLM extraction calls: disabled"
echo "  frozen V2 forecast: real"
echo

"${PY:-${ONCOTWIN_PYTHON:-python3}}" scripts/app/validate_demo_fixture_bundle.py

echo
bash scripts/app/run_cp3_dev.sh
