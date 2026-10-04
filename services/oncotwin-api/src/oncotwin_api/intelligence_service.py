from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from .evidence_service import search_evidence
from .forecast_service import latest_forecast
from .intelligence_schemas import (
    ArtifactEnvelope,
    GroundingRef,
    RecordExplanationLLM,
    VisitBriefLLM,
    VisitQuestionsLLM,
    WhatChangedLLM,
)
from .llm_router import RouterUnavailable, run_structured
from .models import CandidateFact, CommittedFact, Document, DocumentPage, FactSource, GeneratedArtifact, PatientStateSnapshot
from .state import canonical_json

FINAL_SCHEMA_VERSION = "cp4.workflow.final.v1"


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _latest_state(db: Session, patient_id: str) -> PatientStateSnapshot:
    row = db.scalar(
        select(PatientStateSnapshot)
        .where(PatientStateSnapshot.patient_id == patient_id)
        .order_by(PatientStateSnapshot.created_at.desc())
        .limit(1)
    )
    if row is None:
        raise ValueError("Patient state is not available")
    return row


def _state_by_hash(db: Session, patient_id: str, state_hash: str) -> PatientStateSnapshot:
    row = db.scalar(
        select(PatientStateSnapshot)
        .where(PatientStateSnapshot.patient_id == patient_id, PatientStateSnapshot.state_hash == state_hash)
        .order_by(PatientStateSnapshot.created_at.desc())
        .limit(1)
    )
    if row is None:
        raise ValueError("The requested earlier patient-state snapshot is not available")
    return row


def _source_for_fact(db: Session, fact: CommittedFact) -> GroundingRef | None:
    fs = db.scalar(select(FactSource).where(FactSource.committed_fact_id == fact.id).limit(1))
    if fs is None:
        return None
    doc = db.get(Document, fs.document_id)
    if doc is None or doc.deleted_at is not None:
        return None
    return GroundingRef(
        key=f"fact:{fact.id}",
        kind="record",
        label=f"{doc.filename} · page {fs.page_number}",
        document_id=doc.id,
        document_name=doc.filename,
        page_number=fs.page_number,
        source_snippet=fs.source_snippet,
    )


def patient_grounding_catalog(db: Session, patient_id: str) -> tuple[dict[str, Any], list[GroundingRef]]:
    state_row = _latest_state(db, patient_id)
    facts = list(db.scalars(
        select(CommittedFact)
        .where(CommittedFact.patient_id == patient_id, CommittedFact.status == "active")
        .order_by(CommittedFact.effective_date.asc().nullsfirst(), CommittedFact.created_at.asc())
    ))
    refs: list[GroundingRef] = []
    fact_rows: list[dict[str, Any]] = []
    for fact in facts:
        ref = _source_for_fact(db, fact)
        if ref is not None:
            refs.append(ref)
        fact_rows.append({
            "source_key": ref.key if ref else None,
            "fact_id": fact.id,
            "fact_type": fact.fact_type,
            "value": fact.value_json.get("value"),
            "attributes": fact.value_json.get("attributes") or {},
            "effective_date": fact.effective_date,
            "verification_state": fact.verification_state,
        })
    return {
        "state": state_row.state_json,
        "state_hash": state_row.state_hash,
        "verified_facts": fact_rows,
    }, refs


def _resolve_refs(keys: list[str], catalog: list[GroundingRef]) -> list[GroundingRef]:
    by_key = {x.key: x for x in catalog}
    out: list[GroundingRef] = []
    seen = set()
    for key in keys:
        if key in by_key and key not in seen:
            seen.add(key)
            out.append(by_key[key])
    return out


def _artifact_envelope(artifact: GeneratedArtifact, *, content: dict[str, Any] | None = None, grounding: list[GroundingRef] | None = None, cache_hit: bool = False) -> ArtifactEnvelope:
    return ArtifactEnvelope(
        id=artifact.id,
        artifact_type=artifact.artifact_type,
        status=artifact.status,
        cache_hit=cache_hit,
        patient_id=artifact.patient_id,
        document_id=artifact.document_id,
        state_hash=artifact.state_hash,
        provider=artifact.provider,
        model_name=artifact.model_name,
        prompt_version=artifact.prompt_version,
        generated_at=artifact.created_at.isoformat() if artifact.created_at else None,
        content=content if content is not None else (artifact.content_json or {}),
        grounding=grounding if grounding is not None else [GroundingRef.model_validate(x) for x in (artifact.grounding_json or [])],
    )


def _persist_final(
    db: Session,
    *,
    artifact_type: str,
    patient_id: str,
    document_id: str | None,
    state_hash: str | None,
    content: dict[str, Any],
    grounding: list[GroundingRef],
    provider: str,
    model_name: str,
    prompt_version: str,
    basis: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> tuple[GeneratedArtifact, bool]:
    key = _sha({"artifact_type": artifact_type, "basis": basis, "schema": FINAL_SCHEMA_VERSION})
    existing = db.scalar(select(GeneratedArtifact).where(GeneratedArtifact.cache_key == key, GeneratedArtifact.status == "COMPLETED"))
    if existing is not None:
        return existing, True
    artifact = GeneratedArtifact(
        patient_id=patient_id,
        document_id=document_id,
        artifact_type=artifact_type,
        cache_key=key,
        status="COMPLETED",
        provider=provider,
        model_name=model_name,
        prompt_version=prompt_version,
        schema_version=FINAL_SCHEMA_VERSION,
        input_hash=_sha(basis),
        state_hash=state_hash,
        source_hashes_json=[_sha(x.model_dump()) for x in grounding],
        content_json=content,
        grounding_json=[x.model_dump() for x in grounding],
        metadata_json=metadata or {},
    )
    db.add(artifact)
    db.flush()
    return artifact, False


def explain_record(db: Session, patient_id: str, document_id: str, *, refresh: bool = False) -> ArtifactEnvelope:
    doc = db.get(Document, document_id)
    if doc is None or doc.patient_id != patient_id or doc.deleted_at is not None:
        raise ValueError("Document not found")
    pages = list(db.scalars(select(DocumentPage).where(DocumentPage.document_id == doc.id).order_by(DocumentPage.page_number.asc())))
    if not pages:
        raise ValueError("Document has no extracted pages")
    page_payload = []
    catalog: list[GroundingRef] = []
    for page in pages[:20]:
        key = f"page:{page.page_number}"
        text = (page.text or "").strip()
        page_payload.append({"source_key": key, "page_number": page.page_number, "text": text[:8000]})
        catalog.append(GroundingRef(
            key=key,
            kind="record",
            label=f"{doc.filename} · page {page.page_number}",
            document_id=doc.id,
            document_name=doc.filename,
            page_number=page.page_number,
            source_snippet=text[:700],
        ))
    payload = {
        "filename": doc.filename,
        "document_type": doc.document_type,
        "pages": page_payload,
        "default_source_keys": [x.key for x in catalog],
    }
    result = run_structured(
        db,
        task="record_explanation",
        payload=payload,
        response_model=RecordExplanationLLM,
        patient_id=patient_id,
        document_id=document_id,
        grounding=[x.model_dump() for x in catalog],
        refresh=refresh,
        cache_basis={"document_sha256": doc.sha256},
    )
    refs = _resolve_refs(result.value.source_keys, catalog) or catalog[:1]
    content = result.value.model_dump()
    content["source_keys"] = [x.key for x in refs]
    final, hit = _persist_final(
        db,
        artifact_type="record_explanation_final",
        patient_id=patient_id,
        document_id=document_id,
        state_hash=None,
        content=content,
        grounding=refs,
        provider=result.identity.provider,
        model_name=result.identity.model_name,
        prompt_version=result.identity.prompt_version,
        basis={"router_artifact": result.artifact.id if result.artifact else None, "document_sha256": doc.sha256},
        metadata={"router_cache_hit": result.cache_hit},
    )
    return _artifact_envelope(final, content=content, grounding=refs, cache_hit=hit or result.cache_hit)


def latest_record_explanation(db: Session, patient_id: str, document_id: str) -> ArtifactEnvelope | None:
    row = db.scalar(
        select(GeneratedArtifact)
        .where(
            GeneratedArtifact.patient_id == patient_id,
            GeneratedArtifact.document_id == document_id,
            GeneratedArtifact.artifact_type == "record_explanation_final",
            GeneratedArtifact.status == "COMPLETED",
        )
        .order_by(GeneratedArtifact.created_at.desc())
        .limit(1)
    )
    return _artifact_envelope(row) if row else None


def deterministic_state_changes(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    out: list[str] = []
    a_scan = before.get("latest_scan") or {}
    b_scan = after.get("latest_scan") or {}
    if a_scan.get("fact_id") != b_scan.get("fact_id"):
        text = f"Latest scan updated to {b_scan.get('assessment') or 'a new assessment'}"
        if b_scan.get("date"):
            text += f" ({b_scan['date']})"
        out.append(text + ".")
    a_t = before.get("current_treatment") or {}
    b_t = after.get("current_treatment") or {}
    if a_t.get("fact_id") != b_t.get("fact_id") and b_t.get("name"):
        out.append(f"Current treatment record updated to {b_t['name']}.")
    old_sites = set(before.get("disease_sites") or [])
    added_sites = [x for x in (after.get("disease_sites") or []) if x not in old_sites]
    if added_sites:
        out.append("New disease-site information: " + ", ".join(added_sites) + ".")
    old_g = {x.get("alteration") for x in (before.get("genomics") or [])}
    added_g = [x.get("alteration") for x in (after.get("genomics") or []) if x.get("alteration") not in old_g]
    if added_g:
        out.append("New genomic information: " + ", ".join(x for x in added_g if x) + ".")
    old_missing = set(before.get("missing_information") or [])
    resolved = [x for x in old_missing if x not in set(after.get("missing_information") or [])]
    if resolved:
        out.append("Previously missing information now resolved: " + ", ".join(resolved) + ".")
    if not out:
        out.append("No other headline information changed in the current-state summary.")
    return out


def _forecast_change(forecast: dict[str, Any] | None) -> dict[str, Any]:
    if not forecast or forecast.get("run_status") != "COMPLETED" or not forecast.get("horizons"):
        return {"status": "UNAVAILABLE"}
    horizons = {}
    for m in (3, 6, 12, 18):
        row = forecast["horizons"].get(str(m)) or forecast["horizons"].get(m)
        if not row:
            continue
        pre = row.get("pre_pfs")
        selected = row.get("selected_pfs")
        horizons[str(m)] = {
            "pre_pfs": pre,
            "selected_pfs": selected,
            "delta_percentage_points": (selected - pre) * 100 if isinstance(pre, (int, float)) and isinstance(selected, (int, float)) else None,
        }
    return {"status": "COMPLETED", "horizons": horizons, "support": forecast.get("support"), "forecast_run_id": forecast.get("id")}


def generate_what_changed(db: Session, patient_id: str, before_state_hash: str, document_id: str | None = None, *, refresh: bool = False) -> ArtifactEnvelope:
    before = _state_by_hash(db, patient_id, before_state_hash)
    after = _latest_state(db, patient_id)
    if before.state_hash == after.state_hash:
        raise ValueError("Before and after states are identical")
    forecast = latest_forecast(db, patient_id)
    if forecast and forecast.get("state_hash") != after.state_hash:
        raise ValueError("The latest forecast is stale. Recalculate the trajectory before generating what changed.")
    context, catalog = patient_grounding_catalog(db, patient_id)
    state_changes = deterministic_state_changes(before.state_json, after.state_json)
    current_scan_fact_id = (after.state_json.get("latest_scan") or {}).get("fact_id")
    if current_scan_fact_id:
        scan_fact = db.get(CommittedFact, current_scan_fact_id)
        if scan_fact:
            scan_ref = _source_for_fact(db, scan_fact)
            if scan_ref and scan_ref.key not in {x.key for x in catalog}:
                catalog.append(scan_ref)
    if document_id:
        catalog = [x for x in catalog if x.document_id == document_id] + [x for x in catalog if x.document_id != document_id]
    forecast_change = _forecast_change(forecast)
    payload = {
        "before_verified_state": before.state_json,
        "after_verified_state": after.state_json,
        "deterministic_state_changes": state_changes,
        "forecast_update_status": forecast_change.get("status"),
        "forecast_support": forecast_change.get("support"),
        "source_catalog": [x.model_dump() for x in catalog[:20]],
        "default_source_keys": [x.key for x in catalog[:8]],
    }
    result = run_structured(
        db,
        task="what_changed",
        payload=payload,
        response_model=WhatChangedLLM,
        patient_id=patient_id,
        document_id=document_id,
        state_hash=after.state_hash,
        grounding=[x.model_dump() for x in catalog],
        refresh=refresh,
        cache_basis={"before_state_hash": before.state_hash, "after_state_hash": after.state_hash, "forecast_run_id": forecast.get("id") if forecast else None},
    )
    refs = _resolve_refs(result.value.source_keys, catalog) or catalog[:2]
    narrative = result.value.model_dump()
    content = {
        "narrative": narrative,
        "scan_reported": narrative.get("scan_summary"),
        "verified_state_changes": state_changes,
        "forecast_change": forecast_change,
        "before_state_hash": before.state_hash,
        "after_state_hash": after.state_hash,
    }
    final, hit = _persist_final(
        db,
        artifact_type="what_changed_final",
        patient_id=patient_id,
        document_id=document_id,
        state_hash=after.state_hash,
        content=content,
        grounding=refs,
        provider=result.identity.provider,
        model_name=result.identity.model_name,
        prompt_version=result.identity.prompt_version,
        basis={"router_artifact": result.artifact.id if result.artifact else None, "forecast_change": forecast_change, "state_changes": state_changes},
        metadata={"router_cache_hit": result.cache_hit},
    )
    return _artifact_envelope(final, content=content, grounding=refs, cache_hit=hit or result.cache_hit)


def _latest_final(db: Session, patient_id: str, artifact_type: str, state_hash: str | None = None) -> GeneratedArtifact | None:
    q = select(GeneratedArtifact).where(
        GeneratedArtifact.patient_id == patient_id,
        GeneratedArtifact.artifact_type == artifact_type,
        GeneratedArtifact.status == "COMPLETED",
    )
    if state_hash is not None:
        q = q.where(GeneratedArtifact.state_hash == state_hash)
    return db.scalar(q.order_by(GeneratedArtifact.created_at.desc()).limit(1))


def generate_visit_questions(db: Session, patient_id: str, *, refresh: bool = False) -> ArtifactEnvelope:
    context, catalog = patient_grounding_catalog(db, patient_id)
    state_hash = context["state_hash"]
    recent = _latest_final(db, patient_id, "what_changed_final", state_hash)
    evidence = _latest_final(db, patient_id, "evidence_bundle", state_hash)
    unresolved = list(db.scalars(select(CandidateFact).where(CandidateFact.patient_id == patient_id, CandidateFact.status == "needs_review")))
    payload = {
        "state": context["state"],
        "verified_facts": context["verified_facts"],
        "recent_change": recent.content_json if recent else None,
        "unresolved_review_items": [{"fact_type": x.fact_type, "review_reason": x.review_reason} for x in unresolved[:10]],
        "evidence_titles": [x.get("title") for x in ((evidence.content_json.get("literature") or []) + (evidence.content_json.get("trials") or []))[:8]] if evidence else [],
        "default_source_keys": [x.key for x in catalog[:12]],
    }
    result = run_structured(
        db,
        task="visit_questions",
        payload=payload,
        response_model=VisitQuestionsLLM,
        patient_id=patient_id,
        state_hash=state_hash,
        grounding=[x.model_dump() for x in catalog],
        refresh=refresh,
        cache_basis={"state_hash": state_hash, "recent_change": recent.id if recent else None, "evidence": evidence.id if evidence else None, "unresolved": [x.id for x in unresolved]},
    )
    all_keys = []
    for q in result.value.questions:
        all_keys.extend(q.source_keys)
    refs = _resolve_refs(all_keys, catalog) or catalog[:4]
    content = result.value.model_dump()
    final, hit = _persist_final(
        db,
        artifact_type="visit_questions_final",
        patient_id=patient_id,
        document_id=None,
        state_hash=state_hash,
        content=content,
        grounding=refs,
        provider=result.identity.provider,
        model_name=result.identity.model_name,
        prompt_version=result.identity.prompt_version,
        basis={"router_artifact": result.artifact.id if result.artifact else None, "state_hash": state_hash},
        metadata={"router_cache_hit": result.cache_hit},
    )
    return _artifact_envelope(final, content=content, grounding=refs, cache_hit=hit or result.cache_hit)


def generate_visit_brief(db: Session, patient_id: str, *, refresh: bool = False) -> ArtifactEnvelope:
    context, catalog = patient_grounding_catalog(db, patient_id)
    state_hash = context["state_hash"]
    questions_art = _latest_final(db, patient_id, "visit_questions_final", state_hash)
    if questions_art is None:
        q = generate_visit_questions(db, patient_id, refresh=refresh)
        questions_art = db.get(GeneratedArtifact, q.id) if q.id else None
    recent = _latest_final(db, patient_id, "what_changed_final", state_hash)
    question_texts = []
    if questions_art:
        question_texts = [str(x.get("question")) for x in (questions_art.content_json.get("questions") or [])]
    recent_bullets = (recent.content_json.get("verified_state_changes") or []) if recent else []
    payload = {
        "state": context["state"],
        "verified_facts": context["verified_facts"],
        "recent_change_bullets": recent_bullets,
        "question_texts": question_texts,
        "default_source_keys": [x.key for x in catalog[:16]],
    }
    result = run_structured(
        db,
        task="visit_brief",
        payload=payload,
        response_model=VisitBriefLLM,
        patient_id=patient_id,
        state_hash=state_hash,
        grounding=[x.model_dump() for x in catalog],
        refresh=refresh,
        cache_basis={"state_hash": state_hash, "questions": questions_art.id if questions_art else None, "recent_change": recent.id if recent else None},
    )
    refs = _resolve_refs(result.value.source_keys, catalog) or catalog[:6]
    forecast = latest_forecast(db, patient_id)
    trajectory = _forecast_change(forecast) if forecast and forecast.get("state_hash") == state_hash else {"status": "UNAVAILABLE"}
    content = result.value.model_dump()
    content["trajectory"] = trajectory
    content["research_forecast_note"] = "The trajectory is a research-model estimate and is shown separately from the language-generated visit brief."
    final, hit = _persist_final(
        db,
        artifact_type="visit_brief_final",
        patient_id=patient_id,
        document_id=None,
        state_hash=state_hash,
        content=content,
        grounding=refs,
        provider=result.identity.provider,
        model_name=result.identity.model_name,
        prompt_version=result.identity.prompt_version,
        basis={"router_artifact": result.artifact.id if result.artifact else None, "trajectory": trajectory, "state_hash": state_hash},
        metadata={"router_cache_hit": result.cache_hit},
    )
    return _artifact_envelope(final, content=content, grounding=refs, cache_hit=hit or result.cache_hit)


def latest_intelligence_bundle(db: Session, patient_id: str) -> dict[str, Any]:
    state = _latest_state(db, patient_id)
    out: dict[str, Any] = {"patient_id": patient_id, "state_hash": state.state_hash}
    for label, kind in [
        ("what_changed", "what_changed_final"),
        ("visit_questions", "visit_questions_final"),
        ("visit_brief", "visit_brief_final"),
        ("evidence", "evidence_bundle"),
        ("export", "second_opinion_export"),
    ]:
        row = _latest_final(db, patient_id, kind, state.state_hash)
        out[label] = _artifact_envelope(row).model_dump() if row else None
    return out


def generate_evidence(db: Session, patient_id: str, *, refresh: bool = False, max_literature: int = 4, max_trials: int = 4) -> dict[str, Any]:
    return search_evidence(db, patient_id, refresh=refresh, max_literature=max_literature, max_trials=max_trials)
