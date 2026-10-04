#!/usr/bin/env bash

# Installs deterministic LLM extraction fixtures for the canonical synthetic patient.
# The frozen V2 forecast is never fixture-backed.
# Safe for an interactive Slurm allocation: failures are reported but the parent shell survives.

set +e
(
  set -euo pipefail

  REPO="/home/henryxie/OncoTwin-backup"
  cd "$REPO"

  source scripts/app/common_env.sh
  PY_BIN="${PY:-${ONCOTWIN_PYTHON:-python3}}"
  export PYTHONPATH="$REPO/services/oncotwin-api/src:${PYTHONPATH:-}"

  DEMO_ROOT="${ONCOTWIN_DEMO_PATIENT_ROOT:-$REPO/demo/synthetic_patient_v1}"
  FIXTURE_ID="${ONCOTWIN_DEMO_FIXTURE_ID:-maya_rowan_v1}"

  if [ ! -f "$DEMO_ROOT/demo_fixtures/extraction_manifest.json" ]; then
    echo "ERROR: demo fixture manifest not found:"
    echo "  $DEMO_ROOT/demo_fixtures/extraction_manifest.json"
    echo "Expected the synthetic patient ZIP to be unpacked at:"
    echo "  $REPO/demo/synthetic_patient_v1"
    exit 2
  fi

  TS="$(date +%Y%m%d_%H%M%S)"
  ART="artifacts/app_demo_fixture/install_${TS}"
  mkdir -p "$ART"

  PROCESSING="services/oncotwin-api/src/oncotwin_api/processing.py"
  cp "$PROCESSING" "$ART/processing.py.before"

  cat > services/oncotwin-api/src/oncotwin_api/demo_fixtures.py <<'PY'
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .schemas import ExtractionPayload

DEMO_MODE_ENV = "ONCOTWIN_DEMO_MODE"
DEMO_FIXTURE_ID_ENV = "ONCOTWIN_DEMO_FIXTURE_ID"
DEMO_PATIENT_ROOT_ENV = "ONCOTWIN_DEMO_PATIENT_ROOT"
FIXTURE_MODE = "fixture"
DEFAULT_FIXTURE_ID = "maya_rowan_v1"


class DemoFixtureError(RuntimeError):
    """Raised when deterministic demo-fixture mode cannot safely serve a request."""


def fixture_mode_enabled() -> bool:
    return os.getenv(DEMO_MODE_ENV, "").strip().lower() == FIXTURE_MODE


def repo_root() -> Path:
    # services/oncotwin-api/src/oncotwin_api/demo_fixtures.py -> repository root
    return Path(__file__).resolve().parents[4]


def demo_patient_root() -> Path:
    configured = os.getenv(DEMO_PATIENT_ROOT_ENV, "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (repo_root() / "demo" / "synthetic_patient_v1").resolve()


def fixture_root() -> Path:
    return demo_patient_root() / "demo_fixtures"


def expected_fixture_id() -> str:
    return os.getenv(DEMO_FIXTURE_ID_ENV, DEFAULT_FIXTURE_ID).strip() or DEFAULT_FIXTURE_ID


def _json_file(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DemoFixtureError(f"Required demo fixture file is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DemoFixtureError(f"Demo fixture JSON is invalid: {path}: {exc}") from exc


def _manifest() -> dict[str, Any]:
    path = fixture_root() / "extraction_manifest.json"
    manifest = _json_file(path)
    fixture_id = str(manifest.get("fixture_id") or "")
    if fixture_id != expected_fixture_id():
        raise DemoFixtureError(
            f"Demo fixture id mismatch: configured={expected_fixture_id()!r}, manifest={fixture_id!r}"
        )
    documents = manifest.get("documents")
    if not isinstance(documents, dict) or not documents:
        raise DemoFixtureError(f"Demo fixture manifest has no documents: {path}")
    return manifest


def _safe_child(root: Path, relative: str) -> Path:
    root = root.resolve()
    candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise DemoFixtureError(f"Unsafe demo fixture path outside root: {relative}")
    return candidate


def extraction_identity(document_sha256: str | None = None) -> dict[str, str]:
    # Do not validate the PDF here. process_document creates ExtractionRun before its
    # protected processing block. Unknown PDFs are rejected by load_extraction_fixture(),
    # allowing the normal failure path to mark the run/document as failed.
    if not fixture_mode_enabled():
        raise DemoFixtureError("extraction_identity called while fixture mode is disabled")
    fixture_id = expected_fixture_id()
    return {
        "provider": "demo_fixture",
        "model_name": f"fixture:{fixture_id}",
        "schema_version": "demo.fixture.extraction.v1",
        "prompt_version": "demo.fixture.extraction.v1",
    }


def load_extraction_fixture(document_sha256: str) -> ExtractionPayload:
    if not fixture_mode_enabled():
        raise DemoFixtureError("load_extraction_fixture called while fixture mode is disabled")
    manifest = _manifest()
    entry = manifest["documents"].get(document_sha256)
    if not isinstance(entry, dict):
        raise DemoFixtureError(
            "Unknown PDF in deterministic demo-fixture mode. "
            f"sha256={document_sha256}. Live LLM fallback is intentionally disabled."
        )
    fixture_rel = str(entry.get("fixture_file") or "")
    fixture_path = _safe_child(fixture_root(), fixture_rel)
    fixture = _json_file(fixture_path)
    if str(fixture.get("fixture_id") or "") != str(manifest["fixture_id"]):
        raise DemoFixtureError(f"Fixture id mismatch in {fixture_path}")
    if str(fixture.get("document_sha256") or "") != document_sha256:
        raise DemoFixtureError(f"Document checksum mismatch inside {fixture_path}")
    if fixture.get("task") != "document_extraction":
        raise DemoFixtureError(f"Unexpected fixture task in {fixture_path}: {fixture.get('task')!r}")
    try:
        return ExtractionPayload.model_validate(fixture["response"])
    except Exception as exc:
        raise DemoFixtureError(f"Fixture response failed ExtractionPayload validation: {fixture_path}: {exc}") from exc


def load_workflow_fixtures() -> dict[str, Any]:
    data = _json_file(fixture_root() / "workflows_v1.json")
    if str(data.get("fixture_id") or "") != expected_fixture_id():
        raise DemoFixtureError("Workflow fixture id does not match configured demo fixture id")
    return data


def load_workflow_fixture(name: str) -> Any:
    data = load_workflow_fixtures()
    if name not in data:
        raise DemoFixtureError(f"Unknown workflow fixture: {name}")
    return data[name]


def validate_demo_fixture_bundle() -> dict[str, Any]:
    manifest = _manifest()
    patient_root = demo_patient_root()
    fixture_dir = fixture_root()
    checked: list[dict[str, Any]] = []
    problems: list[str] = []

    for expected_sha, entry in sorted(manifest["documents"].items(), key=lambda item: item[1].get("source_file", "")):
        source_rel = str(entry.get("source_file") or "")
        fixture_rel = str(entry.get("fixture_file") or "")
        try:
            source_path = _safe_child(patient_root, source_rel)
            fixture_path = _safe_child(fixture_dir, fixture_rel)
            if not source_path.is_file():
                raise DemoFixtureError(f"Missing source PDF: {source_path}")
            actual_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
            if actual_sha != expected_sha:
                raise DemoFixtureError(
                    f"Source PDF checksum mismatch: {source_rel}: expected {expected_sha}, got {actual_sha}"
                )
            if not fixture_path.is_file():
                raise DemoFixtureError(f"Missing extraction fixture: {fixture_path}")
            payload = load_extraction_fixture(expected_sha)
            checked.append(
                {
                    "source_file": source_rel,
                    "sha256": expected_sha,
                    "document_type": payload.document_type,
                    "fact_count": len(payload.facts),
                }
            )
        except Exception as exc:
            problems.append(str(exc))

    workflow = load_workflow_fixtures()
    expected_workflow_keys = {
        "record_explanations",
        "what_changed_after_progression_scan",
        "visit_questions_after_progression_scan",
        "visit_questions_after_esr1_result",
        "visit_brief_static",
    }
    missing_workflow_keys = sorted(expected_workflow_keys - set(workflow))
    if missing_workflow_keys:
        problems.append(f"Workflow fixture keys missing: {missing_workflow_keys}")

    return {
        "fixture_id": str(manifest["fixture_id"]),
        "patient_root": str(patient_root),
        "fixture_root": str(fixture_dir),
        "document_count": len(manifest["documents"]),
        "checked_count": len(checked),
        "documents": checked,
        "workflow_keys": sorted(k for k in workflow if k not in {"fixture_id", "note"}),
        "problems": problems,
        "ok": not problems and len(checked) == len(manifest["documents"]),
    }
PY

  "$PY_BIN" - "$PROCESSING" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
s = path.read_text(encoding="utf-8")

import_marker = "from .extractor import classify_document, extract_facts\n"
import_addition = import_marker + """from .demo_fixtures import (\n    extraction_identity,\n    fixture_mode_enabled,\n    load_extraction_fixture,\n)\n"""
if "from .demo_fixtures import (" not in s:
    if import_marker not in s:
        raise SystemExit("Could not find extractor import marker in processing.py")
    s = s.replace(import_marker, import_addition, 1)

old_run = '''    run = ExtractionRun(\n        document_id=doc.id,\n        provider=settings.extractor_provider,\n        model_name=settings.openai_model if settings.extractor_provider == "openai" else "deterministic-rules-cp1",\n        status="running",\n    )\n'''
new_run = '''    if fixture_mode_enabled():\n        fixture_identity = extraction_identity(doc.sha256)\n        run = ExtractionRun(\n            document_id=doc.id,\n            provider=fixture_identity["provider"],\n            model_name=fixture_identity["model_name"],\n            schema_version=fixture_identity["schema_version"],\n            prompt_version=fixture_identity["prompt_version"],\n            status="running",\n        )\n    else:\n        run = ExtractionRun(\n            document_id=doc.id,\n            provider=settings.extractor_provider,\n            model_name=settings.openai_model if settings.extractor_provider == "openai" else "deterministic-rules-cp1",\n            status="running",\n        )\n'''
if "fixture_identity = extraction_identity(doc.sha256)" not in s:
    if old_run not in s:
        raise SystemExit("Could not find ExtractionRun block in processing.py")
    s = s.replace(old_run, new_run, 1)

old_extract = '''        combined = text_with_page_markers(pages)\n        doc.document_type = classify_document(combined)\n        payload = extract_facts(combined, doc.document_type)\n        doc.document_type = payload.document_type\n'''
new_extract = '''        combined = text_with_page_markers(pages)\n        doc.document_type = classify_document(combined)\n        if fixture_mode_enabled():\n            payload = load_extraction_fixture(doc.sha256)\n        else:\n            payload = extract_facts(combined, doc.document_type)\n        doc.document_type = payload.document_type\n'''
if "payload = load_extraction_fixture(doc.sha256)" not in s:
    if old_extract not in s:
        raise SystemExit("Could not find extraction call block in processing.py")
    s = s.replace(old_extract, new_extract, 1)

path.write_text(s, encoding="utf-8")
PY

  cat > services/oncotwin-api/tests/test_demo_fixtures.py <<'PY'
from __future__ import annotations

import hashlib

import pytest

from oncotwin_api.demo_fixtures import (
    DemoFixtureError,
    demo_patient_root,
    extraction_identity,
    load_extraction_fixture,
    load_workflow_fixture,
    validate_demo_fixture_bundle,
)


def _enable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ONCOTWIN_DEMO_MODE", "fixture")
    monkeypatch.setenv("ONCOTWIN_DEMO_FIXTURE_ID", "maya_rowan_v1")


def _auth_headers(client):
    r = client.post("/v1/auth/demo")
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def _create_patient(client, headers):
    r = client.post(
        "/v1/patients",
        headers=headers,
        json={"display_name": "Maya Rowan - synthetic demo", "disease_pack": "mbc_v2"},
    )
    assert r.status_code == 200, r.text
    return r.json()["id"]


def test_demo_fixture_bundle_valid(monkeypatch):
    _enable(monkeypatch)
    report = validate_demo_fixture_bundle()
    assert report["ok"], report
    assert report["document_count"] == 12
    assert report["checked_count"] == 12


def test_unknown_pdf_fails_closed(monkeypatch):
    _enable(monkeypatch)
    unknown = hashlib.sha256(b"not a canonical demo pdf").hexdigest()
    with pytest.raises(DemoFixtureError, match="Unknown PDF"):
        load_extraction_fixture(unknown)


def test_fixture_identity_is_explicit(monkeypatch):
    _enable(monkeypatch)
    ident = extraction_identity()
    assert ident["provider"] == "demo_fixture"
    assert ident["model_name"] == "fixture:maya_rowan_v1"
    assert ident["schema_version"] == "demo.fixture.extraction.v1"


def test_fixture_payload_has_dated_scan_landmarks(monkeypatch):
    _enable(monkeypatch)
    report = validate_demo_fixture_bundle()
    scans = []
    for row in report["documents"]:
        payload = load_extraction_fixture(row["sha256"])
        for fact in payload.facts:
            if fact.fact_type == "scan_assessment":
                assert fact.effective_date
                assert fact.value in {"stable", "progression", "indeterminate"}
                scans.append((fact.effective_date, fact.value))
    scans.sort()
    assert scans == [
        ("2025-01-27", "indeterminate"),
        ("2025-04-28", "stable"),
        ("2025-07-28", "stable"),
        ("2025-10-27", "stable"),
        ("2026-01-26", "stable"),
        ("2026-04-29", "progression"),
    ]


def test_fixture_contains_metastatic_anchor_and_pr_review_conflict(monkeypatch):
    _enable(monkeypatch)
    report = validate_demo_fixture_bundle()
    values = []
    for row in report["documents"]:
        payload = load_extraction_fixture(row["sha256"])
        values.extend((f.fact_type, f.value, f.effective_date) for f in payload.facts)
    assert ("diagnosis", "metastatic breast cancer", "2025-01-15") in values
    assert ("pr_status", "positive", "2025-01-04") in values
    assert ("pr_status", "negative", "2025-01-15") in values


def test_future_cp4_workflow_fixtures_are_loadable(monkeypatch):
    _enable(monkeypatch)
    changed = load_workflow_fixture("what_changed_after_progression_scan")
    assert "scan_reported" in changed
    assert "forecast_language_template" in changed
    assert "{pre_pfs_6m_pct}" in changed["forecast_language_template"]
    questions = load_workflow_fixture("visit_questions_after_esr1_result")
    assert len(questions) >= 4


def test_known_demo_pdfs_use_fixture_path_and_real_review_state(client, monkeypatch):
    _enable(monkeypatch)

    def _live_extractor_must_not_run(*args, **kwargs):
        raise AssertionError("live extractor was called in deterministic demo-fixture mode")

    monkeypatch.setattr("oncotwin_api.processing.extract_facts", _live_extractor_must_not_run)

    headers = _auth_headers(client)
    patient_id = _create_patient(client, headers)
    root = demo_patient_root()

    primary = (root / "baseline/01_primary_breast_pathology.pdf").read_bytes()
    r = client.post(
        f"/v1/patients/{patient_id}/documents?process=true",
        headers=headers,
        files={"file": ("01_primary_breast_pathology.pdf", primary, "application/pdf")},
    )
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "ready"

    state = client.get(f"/v1/patients/{patient_id}/state", headers=headers).json()["state"]
    assert state["diagnosis"] == "invasive ductal carcinoma"
    assert state["receptors"] == {"ER": "positive", "PR": "positive", "HER2": "negative"}

    metastatic = (root / "baseline/02_metastatic_liver_pathology.pdf").read_bytes()
    r = client.post(
        f"/v1/patients/{patient_id}/documents?process=true",
        headers=headers,
        files={"file": ("02_metastatic_liver_pathology.pdf", metastatic, "application/pdf")},
    )
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "needs_review"

    review = client.get(f"/v1/patients/{patient_id}/review", headers=headers)
    assert review.status_code == 200, review.text
    review_rows = review.json()
    pr_rows = [x for x in review_rows if x["fact_type"] == "pr_status" and x["value"] == "negative"]
    assert len(pr_rows) == 1, review_rows
    assert "conflicts with an existing committed fact" in (pr_rows[0]["review_reason"] or "")

    accepted = client.post(
        f"/v1/candidates/{pr_rows[0]['id']}/decision",
        headers=headers,
        json={"decision": "accept"},
    )
    assert accepted.status_code == 200, accepted.text
    state = client.get(f"/v1/patients/{patient_id}/state", headers=headers).json()["state"]
    assert state["diagnosis"] == "metastatic breast cancer"
    assert state["receptors"]["PR"] == "negative"
    assert "liver" in state["disease_sites"]
PY

  cat > scripts/app/validate_demo_fixture_bundle.py <<'PY'
from __future__ import annotations

import json
import os

os.environ.setdefault("ONCOTWIN_DEMO_MODE", "fixture")
os.environ.setdefault("ONCOTWIN_DEMO_FIXTURE_ID", "maya_rowan_v1")

from oncotwin_api.demo_fixtures import validate_demo_fixture_bundle

report = validate_demo_fixture_bundle()
print(json.dumps(report, indent=2, sort_keys=True))
raise SystemExit(0 if report["ok"] else 1)
PY

  cat > scripts/app/run_demo_fixture_dev.sh <<'SH2'
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
source scripts/app/common_env.sh

export ONCOTWIN_DEMO_MODE=fixture
export ONCOTWIN_DEMO_FIXTURE_ID="${ONCOTWIN_DEMO_FIXTURE_ID:-maya_rowan_v1}"
export ONCOTWIN_DEMO_PATIENT_ROOT="${ONCOTWIN_DEMO_PATIENT_ROOT:-$REPO_ROOT/demo/synthetic_patient_v1}"
export PYTHONPATH="$REPO_ROOT/services/oncotwin-api/src:${PYTHONPATH:-}"

# Extraction fixture mode should never require a paid LLM key.
unset ONCOTWIN_OPENAI_API_KEY || true

echo "OncoTwin deterministic demo-fixture mode"
echo "  fixture_id=$ONCOTWIN_DEMO_FIXTURE_ID"
echo "  patient_root=$ONCOTWIN_DEMO_PATIENT_ROOT"
echo "  live LLM extraction calls: disabled"
echo "  frozen V2 forecast: real"
echo

"${PY:-${ONCOTWIN_PYTHON:-python3}}" scripts/app/validate_demo_fixture_bundle.py

echo
bash scripts/app/run_cp3_dev.sh
SH2

  cat > scripts/app/demo_fixture_acceptance.sh <<'SH2'
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
source scripts/app/common_env.sh
PY_BIN="${PY:-${ONCOTWIN_PYTHON:-python3}}"
export PYTHONPATH="$REPO_ROOT/services/oncotwin-api/src:${PYTHONPATH:-}"

export ONCOTWIN_DEMO_MODE=fixture
export ONCOTWIN_DEMO_FIXTURE_ID="${ONCOTWIN_DEMO_FIXTURE_ID:-maya_rowan_v1}"
export ONCOTWIN_DEMO_PATIENT_ROOT="${ONCOTWIN_DEMO_PATIENT_ROOT:-$REPO_ROOT/demo/synthetic_patient_v1}"
unset ONCOTWIN_OPENAI_API_KEY || true

echo "=== DEMO FIXTURE BUNDLE VALIDATION ==="
"$PY_BIN" scripts/app/validate_demo_fixture_bundle.py

echo
echo "=== DEMO FIXTURE TESTS ==="
(
  cd services/oncotwin-api
  "$PY_BIN" -m pytest -q tests/test_demo_fixtures.py
)

echo
echo "=== NORMAL-MODE BACKEND REGRESSION ==="
(
  cd services/oncotwin-api
  env \
    -u ONCOTWIN_DEMO_MODE \
    -u ONCOTWIN_DEMO_FIXTURE_ID \
    -u ONCOTWIN_DEMO_PATIENT_ROOT \
    "$PY_BIN" -m pytest -q tests/test_semantics.py tests/test_checkpoint1_flow.py
)

echo
echo "DEMO_FIXTURE_ACCEPTANCE_PASS"
SH2

  cat > demo/DEMO_FIXTURE_RUNTIME.md <<'MD'
# OncoTwin deterministic demo-fixture runtime

The canonical synthetic patient lives at `demo/synthetic_patient_v1`.

Use fixture mode when repeatedly testing or presenting this exact patient:

```bash
bash scripts/app/run_demo_fixture_dev.sh
```

The mode is intentionally narrow:

- PDF upload, storage, page rendering/OCR and source viewing still run normally.
- Extraction results for the 12 canonical demo PDFs are loaded by exact SHA256.
- Unknown PDFs fail closed in fixture mode; the app never silently falls back to a paid LLM.
- Candidate facts, verification/conflicts, committed state, timeline and source provenance are real.
- The frozen Dynamic Scan V2 forecast is always real and is never hard-coded.
- Extraction provenance is stored as `provider=demo_fixture`, `model=fixture:maya_rowan_v1`.
- `demo_fixtures/workflows_v1.json` is a typed-content source for future CP4 fixture-backed explanations/questions/briefs. Numerical forecast values must always be injected from the real forecast run.

Normal development behavior is unchanged when `ONCOTWIN_DEMO_MODE` is unset.
MD

  chmod +x scripts/app/run_demo_fixture_dev.sh scripts/app/demo_fixture_acceptance.sh

  "$PY_BIN" -m py_compile \
    services/oncotwin-api/src/oncotwin_api/demo_fixtures.py \
    services/oncotwin-api/src/oncotwin_api/processing.py \
    services/oncotwin-api/tests/test_demo_fixtures.py \
    scripts/app/validate_demo_fixture_bundle.py

  cp "$PROCESSING" "$ART/processing.py.after"
  diff -u "$ART/processing.py.before" "$ART/processing.py.after" > "$ART/processing.diff" || true

  echo "=== RUNNING ACCEPTANCE ==="
  bash scripts/app/demo_fixture_acceptance.sh 2>&1 | tee "$ART/acceptance.log"

  cat > "$ART/summary.txt" <<EOF
DEMO_FIXTURE_INSTALL=PASS
fixture_id=$FIXTURE_ID
patient_root=$DEMO_ROOT
processing_diff=$ART/processing.diff
acceptance_log=$ART/acceptance.log
normal_mode_unchanged_when_demo_env_unset=true
forecast_fixture_backed=false
EOF

  echo
  cat "$ART/summary.txt"
  echo "ARTIFACT_DIR=$ART"
) 
RC=$?

echo
echo "INSTALL_CHILD_EXIT_CODE=$RC"
if [ "$RC" -eq 0 ]; then
  echo "COMMAND_STATUS=PASS"
else
  echo "COMMAND_STATUS=FAIL"
fi
echo "Interactive Slurm shell preserved."
true
