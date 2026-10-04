from __future__ import annotations

import json
import os

os.environ.setdefault("ONCOTWIN_DEMO_MODE", "true")
os.environ.setdefault("ONCOTWIN_DEMO_FIXTURE_ID", "maya_rowan_v1")

from oncotwin_api.demo_fixtures import validate_demo_fixture_bundle

report = validate_demo_fixture_bundle()
print(json.dumps(report, indent=2, sort_keys=True))
raise SystemExit(0 if report["ok"] else 1)
