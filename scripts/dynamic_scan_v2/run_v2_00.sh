#!/usr/bin/env bash
set -euo pipefail

REPO="${1:-/home/henryxie/OncoTwin-backup}"
PY="${PYTHON:-/home/henryxie/orcd/scratch/.conda/envs/oncotwin/bin/python3}"
OUT="$REPO/artifacts/dynamic_scan_v2/v2_00"

cd "$REPO"

echo "[V2-00 runner] repo=$REPO"
echo "[V2-00 runner] python=$PY"

echo "[V2-00 runner] persistent methodology tests"
"$PY" -m pytest -q tests/dynamic_scan_v2/test_v2_00_metrics.py

echo "[V2-00 runner] end-to-end benchmark establishment"
"$PY" scripts/dynamic_scan_v2/v2_00_establish_benchmark.py \
  --repo "$REPO" \
  --out "$OUT"

echo "[V2-00 runner] artifact manifest"
cat "$OUT/result_packet.json"
