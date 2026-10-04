from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "apps" / "oncotwin-web"


def read(rel: str) -> str:
    path = WEB / rel
    if not path.is_file():
        return ""
    return path.read_text(errors="replace")


overview = read("app/app/page.tsx")
forecast = read("app/app/forecast/page.tsx")
update = read("app/app/update/page.tsx")
records = read("app/app/records/page.tsx")
prepare = read("app/app/prepare/page.tsx")
public_home = read("app/page.tsx")
shell = read("components/AppShell.tsx")
chart = read("components/ForecastChart.tsx")
source_viewer = read("components/SourceViewer.tsx")
types = read("lib/types.ts")

css = ""
for rel in ("app/globals.css", "styles/globals.css"):
    candidate = read(rel)
    if candidate:
        css = candidate
        break

checks: dict[str, bool] = {}

# Accessibility and responsive behavior.
checks["focus_visible_css"] = ":focus-visible" in css
checks["reduced_motion_css"] = "prefers-reduced-motion" in css
checks["responsive_breakpoints"] = "@media" in css and "max-width" in css

checks["source_dialog_keyboard_escape"] = (
    "Escape" in source_viewer
    and (
        "keydown" in source_viewer.lower()
        or "onkeydown" in source_viewer.lower()
        or "onKeyDown" in source_viewer
    )
)

# Typed frozen-forecast frontend contract.
checks["typed_forecast_contract"] = (
    "export type Forecast" in types
    and "pre_pfs" in types
    and "selected_pfs" in types
)

checks["forecast_before_selected_chart"] = (
    "before" in chart.lower()
    and (
        "selected" in chart.lower()
        or "after newest scan" in chart.lower()
        or "updated" in chart.lower()
    )
)

checks["forecast_uses_proxy_api"] = (
    "api<Forecast>" in forecast
    and "/forecast" in forecast
)

# The newest-scan flow must rebuild committed state first and then invoke
# the frozen forecast through the normal application API boundary.
checks["scan_flow_uses_committed_proxy_api"] = (
    "api<StateResponse>" in update
    and "/state" in update
    and "api<Forecast>" in update
    and "/forecast" in update
    and "method: 'POST'" in update
)

# Navigation semantics. CP3 does not require a particular exact nav layout:
# the patient must have clear routes to both newest-scan and trajectory flows.
nav_surface = "\n".join((shell, overview, records, forecast))
checks["navigation_has_scan_and_trajectory"] = (
    "/app/update" in nav_surface
    and "/app/forecast" in nav_surface
)

# Signature CP3 result: keep scan report, quantitative update, and structured
# state change separate. These are the accepted current patient-facing labels.
checks["signature_three_panel_language"] = all(
    phrase in update
    for phrase in (
        "WHAT THE SCAN REPORTED",
        "HOW YOUR OUTLOOK CHANGED",
        "WHAT NEW INFORMATION WAS ADDED",
    )
)

# Patient-facing PRE/POST language intentionally avoids research terminology.
checks["pre_post_patient_language"] = (
    "BEFORE THIS SCAN" in update
    and "AFTER THIS SCAN" in update
    and "BEFORE THIS SCAN" in forecast
    and "AFTER THIS SCAN" in forecast
)

# The model must continue to be visibly framed as a research release, while
# detailed checkpoint metadata may remain secondary.
checks["forecast_research_disclosure"] = (
    "Research release" in shell
    and "Release" in forecast
    and "V2-07" in forecast
)

# Overview must still expose the major verified-state domains with source
# access and retain trajectory + recent-history hierarchy.
checks["overview_current_state_hierarchy"] = all(
    token in overview
    for token in (
        "Check diagnosis source",
        "data.sources.treatment",
        "data.sources.scan_assessment",
        "data.sources.genomic_alteration",
        "Your latest outlook",
        "Latest changes in your record",
    )
)

# Patient-facing language may surface options/questions for clinician
# discussion, but must not become a treatment directive.
patient_facing = "\n".join(
    (overview, forecast, update, records, prepare, public_home)
).lower()

prohibited_directives = (
    "you should start",
    "you should stop",
    "you should switch",
    "you should take",
    "we recommend starting",
    "we recommend stopping",
    "the best treatment for you is",
)

checks["no_treatment_recommendation_copy"] = (
    not any(x in patient_facing for x in prohibited_directives)
    and (
        "worth discussing with your care team" in patient_facing
        or "questions" in prepare.lower()
        or "discussion" in prepare.lower()
    )
)

# Missing information must render as unavailable/not established rather than
# being silently converted to a negative clinical fact.
missing_fallback_visible = any(
    phrase.lower() in patient_facing
    for phrase in (
        "not calculated",
        "not established",
        "not available",
        "unavailable",
    )
)

negative_default = re.search(
    r"(?:\?\?|\|\|)\s*['\"]negative['\"]",
    "\n".join((overview, forecast, update, records)),
    flags=re.IGNORECASE,
)

checks["missing_not_negative_visible"] = (
    missing_fallback_visible
    and negative_default is None
)

result = {
    "checks": checks,
    "failed": [name for name, passed in checks.items() if not passed],
    "status": "PASS" if all(checks.values()) else "FAIL",
}

print(json.dumps(result, indent=2, sort_keys=True))

if result["failed"]:
    raise SystemExit(f"CP3 UI contract failed: {result['failed']}")
