from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from oncotwin_api.intelligence_service import deterministic_state_changes
from oncotwin_api.models import Base, GeneratedArtifact, Patient, PatientStateSnapshot, User
from oncotwin_api.evidence_service import build_evidence_queries


def test_state_change_diff_keeps_scan_and_genomics_distinct():
    before = {
        "latest_scan": {"fact_id": "a", "assessment": "stable", "date": "2026-01-28"},
        "current_treatment": {"fact_id": "t", "name": "letrozole + ribociclib"},
        "disease_sites": ["liver", "bone"],
        "genomics": [],
        "missing_information": [],
    }
    after = {
        "latest_scan": {"fact_id": "b", "assessment": "progression", "date": "2026-04-29"},
        "current_treatment": {"fact_id": "t", "name": "letrozole + ribociclib"},
        "disease_sites": ["liver", "bone"],
        "genomics": [{"alteration": "ESR1", "date": "2026-05-08", "fact_id": "g"}],
        "missing_information": [],
    }
    rows = deterministic_state_changes(before, after)
    text = " ".join(rows).lower()
    assert "progression" in text
    assert "esr1" in text
    assert "caused" not in text


def test_evidence_query_is_deterministic_and_state_based():
    state = {
        "diagnosis": "metastatic breast cancer",
        "subtype": "HR-positive / HER2-negative",
        "receptors": {"ER": "positive", "HER2": "negative"},
        "latest_scan": {"assessment": "progression"},
        "genomics": [{"alteration": "ESR1"}],
    }
    a = build_evidence_queries(state)
    b = build_evidence_queries(state)
    assert a == b
    assert "ESR1" in a["literature"]
    assert "progression" in a["trials"]


def test_generated_artifact_schema_is_persistable():
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        user = User(display_name="test")
        db.add(user); db.flush()
        patient = Patient(owner_user_id=user.id, display_name="test")
        db.add(patient); db.flush()
        snap = PatientStateSnapshot(patient_id=patient.id, state_hash="a" * 64, state_json={}, fact_revision_ids_json=[])
        db.add(snap)
        artifact = GeneratedArtifact(
            patient_id=patient.id,
            artifact_type="visit_questions_final",
            cache_key="b" * 64,
            status="COMPLETED",
            provider="test",
            model_name="test",
            prompt_version="v1",
            schema_version="v1",
            input_hash="c" * 64,
            state_hash=snap.state_hash,
            source_hashes_json=[],
            content_json={"questions": []},
            grounding_json=[],
            metadata_json={},
        )
        db.add(artifact); db.commit()
        assert artifact.id
