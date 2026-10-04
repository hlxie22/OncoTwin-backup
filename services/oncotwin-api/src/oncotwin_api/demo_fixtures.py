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
    """Return whether deterministic patient-fixture extraction is active.

    Generic ONCOTWIN_DEMO_MODE=true is used elsewhere in the application
    and test suite and therefore MUST NOT by itself activate synthetic
    patient fixtures.

    Fixture extraction is enabled only when demo mode is on AND an
    explicit fixture ID has been selected.
    """
    raw = os.getenv("ONCOTWIN_DEMO_MODE", "").strip().lower()
    fixture_id = os.getenv("ONCOTWIN_DEMO_FIXTURE_ID", "").strip()

    demo_enabled = raw in {"1", "true", "yes", "on", "fixture"}
    return demo_enabled and bool(fixture_id)


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
