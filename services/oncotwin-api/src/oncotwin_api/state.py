from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import (
    AuditEvent,
    CandidateFact,
    CommittedFact,
    Document,
    FactSource,
    PatientStateSnapshot,
    TimelineEvent,
)

HIGH_IMPACT_TYPES = {"er_status", "pr_status", "her2_status", "treatment", "scan_assessment", "genomic_alteration"}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def audit(db: Session, *, user_id: str | None, patient_id: str | None, action: str, object_type: str | None = None,
          object_id: str | None = None, outcome: str = "success", metadata: dict | None = None) -> None:
    db.add(AuditEvent(
        user_id=user_id,
        patient_id=patient_id,
        action=action,
        object_type=object_type,
        object_id=object_id,
        outcome=outcome,
        metadata_json=metadata or {},
    ))


def derive_subtype(latest: dict[str, CommittedFact]) -> str | None:
    er = latest.get("er_status")
    pr = latest.get("pr_status")
    her2 = latest.get("her2_status")
    if her2 and her2.value_json.get("value") == "positive":
        return "HER2-positive"
    if (er and er.value_json.get("value") == "positive") or (pr and pr.value_json.get("value") == "positive"):
        return "HR-positive / HER2-negative" if her2 and her2.value_json.get("value") == "negative" else "HR-positive"
    if er and pr and her2 and all(x.value_json.get("value") == "negative" for x in [er, pr, her2]):
        return "triple-negative"
    return None


def build_state(db: Session, patient_id: str) -> tuple[dict, list[str]]:
    facts = list(db.scalars(
        select(CommittedFact)
        .where(CommittedFact.patient_id == patient_id, CommittedFact.status == "active")
        .order_by(CommittedFact.effective_date.asc().nullsfirst(), CommittedFact.created_at.asc())
    ))
    latest: dict[str, CommittedFact] = {}
    for f in facts:
        latest[f.fact_type] = f

    sites = []
    genomics = []
    treatments = []
    scans = []
    for f in facts:
        value = f.value_json.get("value")
        if f.fact_type == "disease_site" and value and value not in sites:
            sites.append(value)
        elif f.fact_type == "genomic_alteration" and value:
            genomics.append({"alteration": value, "date": f.effective_date, "fact_id": f.id})
        elif f.fact_type == "treatment" and value:
            treatments.append({"name": value, "date": f.effective_date, "active": f.value_json.get("attributes", {}).get("active", True), "fact_id": f.id})
        elif f.fact_type == "scan_assessment" and value:
            scans.append({"assessment": value, "date": f.effective_date, "fact_id": f.id})

    current_treatment = treatments[-1] if treatments else None
    latest_scan = scans[-1] if scans else None
    diagnosis = latest.get("diagnosis")
    subtype = derive_subtype(latest)
    receptors = {
        k.replace("_status", "").upper(): latest[k].value_json.get("value") if k in latest else None
        for k in ["er_status", "pr_status", "her2_status"]
    }
    missing = []
    if diagnosis is None:
        missing.append("diagnosis")
    if subtype is None:
        missing.append("receptor/subtype information")
    if current_treatment is None:
        missing.append("current treatment")
    if latest_scan is None:
        missing.append("recent scan assessment")

    state = {
        "diagnosis": diagnosis.value_json.get("value") if diagnosis else None,
        "subtype": subtype,
        "receptors": receptors,
        "current_treatment": current_treatment,
        "disease_sites": sites,
        "latest_scan": latest_scan,
        "genomics": genomics,
        "missing_information": missing,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    return state, [f.id for f in facts]


def rebuild_state(db: Session, patient_id: str) -> PatientStateSnapshot:
    state, fact_ids = build_state(db, patient_id)
    stable_state = {k: v for k, v in state.items() if k != "updated_at"}
    state_hash = hashlib.sha256(canonical_json({"state": stable_state, "facts": fact_ids}).encode()).hexdigest()
    latest = db.scalar(
        select(PatientStateSnapshot)
        .where(PatientStateSnapshot.patient_id == patient_id)
        .order_by(PatientStateSnapshot.created_at.desc())
        .limit(1)
    )
    if latest and latest.state_hash == state_hash:
        return latest
    snap = PatientStateSnapshot(
        patient_id=patient_id,
        state_hash=state_hash,
        state_json=state,
        fact_revision_ids_json=fact_ids,
    )
    db.add(snap)
    db.flush()
    return snap


def event_for_fact(fact: CommittedFact) -> tuple[str, str, str]:
    value = str(fact.value_json.get("value", ""))
    mapping = {
        "diagnosis": ("diagnosis", "Diagnosis", value),
        "er_status": ("pathology", "ER status", value),
        "pr_status": ("pathology", "PR status", value),
        "her2_status": ("pathology", "HER2 status", value),
        "treatment": ("treatment", "Treatment", value),
        "disease_site": ("disease_site", "Disease site", value),
        "scan_assessment": ("scan", "Scan assessment", value),
        "scan_modality": ("scan", "Scan modality", value),
        "genomic_assay": ("genomics", "Genomic assay", value),
        "genomic_alteration": ("genomics", "Genomic alteration", value),
        "lab": ("lab", "Laboratory result", value),
    }
    return mapping.get(fact.fact_type, ("other", fact.fact_type.replace("_", " ").title(), value))


def commit_candidate(
    db: Session,
    candidate: CandidateFact,
    *,
    user_id: str,
    verification_state: str,
    edited_value: str | None = None,
    edited_effective_date: str | None = None,
) -> CommittedFact:
    value_json = dict(candidate.value_json or {})
    if edited_value is not None:
        value_json["value"] = edited_value
    effective_date = edited_effective_date if edited_effective_date is not None else candidate.effective_date
    fact = CommittedFact(
        patient_id=candidate.patient_id,
        candidate_fact_id=candidate.id,
        source_document_id=candidate.document_id,
        fact_type=candidate.fact_type,
        value_json=value_json,
        effective_date=effective_date,
        unit=candidate.unit,
        confidence=candidate.confidence,
        verification_state=verification_state,
        status="active",
    )
    db.add(fact)
    db.flush()
    db.add(FactSource(
        committed_fact_id=fact.id,
        document_id=candidate.document_id,
        page_number=candidate.source_page,
        source_snippet=candidate.source_snippet,
        source_locator_json={"kind": "page_snippet"},
    ))
    event_type, title, summary = event_for_fact(fact)
    db.add(TimelineEvent(
        patient_id=fact.patient_id,
        source_fact_id=fact.id,
        event_type=event_type,
        event_date=fact.effective_date,
        title=title,
        summary=summary,
    ))
    candidate.status = "committed"
    candidate.decided_at = datetime.now(timezone.utc)
    rebuild_state(db, candidate.patient_id)
    audit(db, user_id=user_id, patient_id=candidate.patient_id, action="fact.commit", object_type="candidate_fact", object_id=candidate.id,
          metadata={"fact_type": candidate.fact_type, "verification_state": verification_state})
    return fact


def supersede_document_facts(db: Session, document_id: str, patient_id: str, user_id: str) -> None:
    facts = list(db.scalars(select(CommittedFact).where(
        CommittedFact.source_document_id == document_id,
        CommittedFact.status == "active",
    )))
    ids = [f.id for f in facts]
    for f in facts:
        f.status = "superseded"
    if ids:
        for ev in db.scalars(select(TimelineEvent).where(TimelineEvent.source_fact_id.in_(ids))):
            db.delete(ev)
    db.flush()
    rebuild_state(db, patient_id)
    audit(db, user_id=user_id, patient_id=patient_id, action="document.derived_facts_superseded", object_type="document", object_id=document_id,
          metadata={"fact_count": len(facts)})


def state_with_sources(db: Session, patient_id: str) -> tuple[PatientStateSnapshot | None, dict[str, dict]]:
    snap = db.scalar(
        select(PatientStateSnapshot)
        .where(PatientStateSnapshot.patient_id == patient_id)
        .order_by(PatientStateSnapshot.created_at.desc())
        .limit(1)
    )
    sources: dict[str, dict] = {}
    facts = list(db.scalars(
        select(CommittedFact)
        .where(CommittedFact.patient_id == patient_id, CommittedFact.status == "active")
        .order_by(CommittedFact.created_at.asc())
    ))
    for fact in facts:
        src = db.scalar(select(FactSource).where(FactSource.committed_fact_id == fact.id).limit(1))
        doc = db.get(Document, src.document_id) if src else None
        if src and doc:
            sources[fact.fact_type] = {
                "document_id": doc.id,
                "document_name": doc.filename,
                "page_number": src.page_number,
                "source_snippet": src.source_snippet,
            }
    return snap, sources
