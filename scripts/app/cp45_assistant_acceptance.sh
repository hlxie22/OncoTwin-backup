#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source scripts/app/common_env.sh

STAMP="$(date +%Y%m%d_%H%M%S)_$$_${RANDOM}"
OUT="artifacts/app_cp45/assistant_acceptance_${STAMP}"
mkdir -p "$OUT"
LOG="$OUT/full.log"
exec > >(tee "$LOG") 2>&1

echo "=== CP4.5 Ask OncoTwin: Python compile ==="
"$PY" -m py_compile \
  services/oncotwin-api/src/oncotwin_api/assistant_schemas.py \
  services/oncotwin-api/src/oncotwin_api/demo_assistant.py \
  services/oncotwin-api/src/oncotwin_api/assistant_service.py \
  scripts/app/cp45_assistant_contract.py

echo "=== CP4.5 Ask OncoTwin: backend unit tests ==="
(
  cd services/oncotwin-api
  "$PY" -m pytest -q tests/test_cp45_assistant.py
)

echo "=== CP4.5 Ask OncoTwin: source contract ==="
"$PY" scripts/app/cp45_assistant_contract.py

echo "=== CP4.5 Ask OncoTwin: production web build ==="
(
  cd apps/oncotwin-web
  npm run build
)

echo "=== CP4.5 Ask OncoTwin: full CP1-CP4 regression ==="
bash scripts/app/cp4_acceptance.sh

echo "CP45_ASSISTANT_ACCEPTANCE_PASS"
echo "ARTIFACT_DIR=$OUT"
