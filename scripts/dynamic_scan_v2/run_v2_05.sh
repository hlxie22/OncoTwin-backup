#!/usr/bin/env bash
set -euo pipefail
REPO="${REPO:-/home/henryxie/OncoTwin-backup}"
PYTHON="${PYTHON:-/home/henryxie/orcd/scratch/.conda/envs/oncotwin/bin/python3}"
export PYTHONPATH="$REPO/scripts/dynamic_scan_v2${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"
"$PYTHON" -m pytest -q tests/dynamic_scan_v2/test_v2_05_robustness.py
"$PYTHON" -u scripts/dynamic_scan_v2/v2_05_robustness.py --repo "$REPO"
