#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/henryxie/OncoTwin-backup}"
PYTHON="${PYTHON:-/home/henryxie/orcd/scratch/.conda/envs/oncotwin/bin/python3}"

cd "$REPO"

"$PYTHON" - <<'PY'
import json
from pathlib import Path
p = Path("artifacts/dynamic_scan_v2/v2_00/result_packet.json")
if not p.is_file():
    raise SystemExit("V2-00 result packet missing")
x = json.loads(p.read_text())
if x.get("status") != "PASS":
    raise SystemExit(f"V2-00 is not PASS: {x.get('status')}")
print("V2_00_PREREQUISITE=PASS")
PY

"$PYTHON" -m pytest -q \
  tests/dynamic_scan_v2/test_v2_00_metrics.py \
  tests/dynamic_scan_v2/test_v2_01_scan_representation.py \
  tests/dynamic_scan_v2/test_v2_01_crosswalk.py

"$PYTHON" scripts/dynamic_scan_v2/v2_01_build_scan_representation.py \
  --repo "$REPO"

"$PYTHON" - <<'PY'
import json
from pathlib import Path
p = Path("artifacts/dynamic_scan_v2/v2_01/result_packet.json")
x = json.loads(p.read_text())
if x.get("status") != "PASS":
    raise SystemExit(f"V2-01 failed: {x}")
if not all(x.get("acceptance", {}).values()):
    raise SystemExit(f"V2-01 acceptance failure: {x['acceptance']}")
print("V2_01=PASS")
print("rows=", x["rows"])
print("patients=", x["patients"])
print("next=", x["next_checkpoint"])
PY
