#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="$REPO/scripts/dynamic_scan_v2${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" -m pytest -q \
  "$REPO/tests/dynamic_scan_v2/test_v2_08_metrics.py"

"$PYTHON" "$REPO/scripts/dynamic_scan_v2/v2_08_characterize.py" --repo "$REPO"
