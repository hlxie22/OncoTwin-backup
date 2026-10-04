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
    monkeypatch.setenv("ONCOTWIN_DEMO_MODE", "true")
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

    import json

    manifest_path = demo_patient_root() / "case_manifest.json"
    manifest = json.loads(manifest_path.read_text())

    expected = sorted(
        (row["date"], row["assessment"])
        for row in manifest["scan_landmarks"]
    )

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

    assert scans == expected


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

    docs = client.get(f"/v1/patients/{patient_id}/documents", headers=headers)
    assert docs.status_code == 200, docs.text
    metastatic_doc = next(
        x for x in docs.json() if x["filename"] == "02_metastatic_liver_pathology.pdf"
    )
    assert metastatic_doc["status"] == "ready", metastatic_doc
    print("METASTATIC_DOCUMENT_STATUS_READY_AFTER_REVIEW")
