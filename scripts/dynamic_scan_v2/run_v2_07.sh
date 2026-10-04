#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$(pwd)}"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="$REPO/scripts/dynamic_scan_v2${PYTHONPATH:+:$PYTHONPATH}"

cd "$REPO"
"$PYTHON" -m pytest -q \
  tests/dynamic_scan_v2/test_v2_07_inference.py \
  tests/dynamic_scan_v2/test_v2_07_release_contract.py

"$PYTHON" -u scripts/dynamic_scan_v2/v2_07_release.py --repo "$REPO"
