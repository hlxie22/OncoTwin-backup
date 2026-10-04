#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source scripts/app/common_env.sh

STAMP="$(date +%Y%m%d_%H%M%S)_$$_${RANDOM}"
OUT="artifacts/app_cp4/acceptance_${STAMP}"
mkdir -p "$OUT"
LOG="$OUT/cp4_acceptance.log"
SUMMARY="$OUT/summary.json"

exec > >(tee "$LOG") 2>&1

echo "=== CP4 migration smoke ==="
TEST_DB="$OUT/cp4_acceptance.db"

# Acceptance must always exercise Alembic against a genuinely empty database.
# SQLite DDL is non-transactional, so a partially migrated smoke DB must never
# be reused on a subsequent acceptance attempt.
rm -f "$TEST_DB" "$TEST_DB-shm" "$TEST_DB-wal"

export ONCOTWIN_DATABASE_URL="sqlite:///$ROOT/$TEST_DB"
export ONCOTWIN_STORAGE_ROOT="$OUT/storage"
export ONCOTWIN_EXTRACTOR_PROVIDER="rules"
export ONCOTWIN_DEMO_MODE="true"
export ONCOTWIN_AUTO_CREATE_SCHEMA="false"
unset ONCOTWIN_DEMO_FIXTURE_ID || true

if [ -e "$TEST_DB" ]; then
  echo "ERROR: CP4 smoke database unexpectedly exists before Alembic migration"
  exit 1
fi

echo "CP4_SMOKE_DB_FRESH=PASS"
(
  cd services/oncotwin-api
  "$PY" -m alembic upgrade head
  "$PY" -m alembic current
)

echo "=== CP4 backend tests ==="
(
  cd services/oncotwin-api
  "$PY" -m pytest tests/test_cp4_router.py tests/test_cp4_workflows.py tests/test_semantics.py tests/test_checkpoint1_flow.py
)

echo "=== CP4 source contract ==="
"$PY" scripts/app/cp4_contract.py

echo "=== Maya fixture regression ==="
bash scripts/app/demo_fixture_acceptance.sh

echo "=== CP1-CP3 + frozen model regression ==="
bash scripts/app/cp3_acceptance.sh

cat > "$SUMMARY" <<JSON
{
  "checkpoint": "CP4",
  "status": "PASS",
  "migration_head": "20261004_cp4_intelligence",
  "llm_forecast_boundary": "PASS",
  "provider_router_contract": "PASS",
  "cache_and_cooldown_tests": "PASS",
  "fixture_no_live_call_regression": "PASS",
  "cp1_cp2_cp3_regression": "PASS",
  "artifact_dir": "$OUT"
}
JSON

echo "CP4_ACCEPTANCE_PASS"
echo "CP4_ACCEPTANCE_DIR=$OUT"
echo "CP4_ACCEPTANCE_SUMMARY=$SUMMARY"
cat "$SUMMARY"
