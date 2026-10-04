#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/henryxie/OncoTwin-backup}"
PYTHON="${PYTHON:-/home/henryxie/orcd/scratch/.conda/envs/oncotwin/bin/python3}"
cd "$REPO"

# V2 modules intentionally use flat imports (metrics, access, paired_scan_encoder,
# etc.) so they can also be executed directly as scripts. Pytest imports test
# modules before v2_02_train.py can amend sys.path, therefore expose the source
# root explicitly for the whole runner.
export PYTHONPATH="$REPO/scripts/dynamic_scan_v2${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" -m pytest -q \
  tests/dynamic_scan_v2/test_v2_02_models.py \
  tests/dynamic_scan_v2/test_v2_02_training_utils.py

"$PYTHON" -u scripts/dynamic_scan_v2/v2_02_train.py --repo "$REPO"
