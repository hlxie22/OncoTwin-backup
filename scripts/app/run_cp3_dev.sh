#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
# CP3 adds no new runtime service; use the accepted CP2 model/API/web stack.
exec bash scripts/app/run_cp2_dev.sh
