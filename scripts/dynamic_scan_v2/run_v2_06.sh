#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="$REPO/scripts/dynamic_scan_v2${PYTHONPATH:+:$PYTHONPATH}"

cd "$REPO"
"$PYTHON" -m pytest -q tests/dynamic_scan_v2/test_v2_06_external_readiness.py
"$PYTHON" -u scripts/dynamic_scan_v2/v2_06_prepare.py --repo "$REPO"
