from __future__ import annotations

import hashlib
import io
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from .auth import create_token, get_current_user
from .config import get_settings
from .db import get_db, get_engine
from .models import (
    Base,
    CandidateFact,
    Document,
    DocumentPage,
    FactSource,
    Patient,
    TimelineEvent,
    User,
)
from .schemas import (
    CandidateDecision,
    CandidateFactOut,
    DemoAuthResponse,
    DocumentOut,
    PatientCreate,
    PatientOut,
    SourceOut,
    StateResponse,
    TimelineEventOut,
)
from .state import audit, commit_candidate, rebuild_state, state_with_sources, supersede_document_facts
from .storage import get_object_store
from .tasks import process_document_task


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    if settings.auto_create_schema:
        Base.metadata.create_all(bind=get_engine())
    yield


settings = get_settings()
app = FastAPI(title="OncoTwin API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def owned_patient(db: Session, patient_id: str, user: User) -> Patient:
    patient = db.scalar(select(Patient).where(Patient.id == patient_id, Patient.owner_user_id == user.id))
    if patient is None:
        raise HTTPException(404, "Patient not found")
    return patient


def owned_document(db: Session, document_id: str, user: User) -> Document:
    doc = db.scalar(select(Document).where(Document.id == document_id, Document.owner_user_id == user.id, Document.deleted_at.is_(None)))
    if doc is None:
        raise HTTPException(404, "Document not found")
    return doc


def refresh_document_review_status(db: Session, document_id: str) -> None:
    """Keep Document.status aligned with unresolved candidate review items."""
    doc = db.get(Document, document_id)
    if doc is None or doc.deleted_at is not None or doc.status in {"failed", "deleted"}:
        return
    pending = db.scalar(
        select(CandidateFact.id)
        .where(
            CandidateFact.document_id == document_id,
            CandidateFact.status == "needs_review",
        )
        .limit(1)
    )
    doc.status = "needs_review" if pending is not None else "ready"


@app.get("/healthz")
def healthz():
    return {"status": "ok", "version": app.version}


@app.post("/v1/auth/demo", response_model=DemoAuthResponse)
def demo_auth(db: Session = Depends(get_db)):
    if not settings.demo_mode:
        raise HTTPException(404, "Demo authentication is disabled")
    user = db.scalar(select(User).order_by(User.created_at.asc()).limit(1))
    if user is None:
        user = User(display_name="OncoTwin Demo")
        db.add(user)
        db.commit()
    return DemoAuthResponse(access_token=create_token(user.id), user_id=user.id, display_name=user.display_name)


@app.get("/v1/patients", response_model=list[PatientOut])
def list_patients(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    patients = list(db.scalars(select(Patient).where(Patient.owner_user_id == user.id).order_by(Patient.created_at.asc())))
    return [PatientOut(id=p.id, display_name=p.display_name, disease_pack=p.disease_pack) for p in patients]


@app.post("/v1/patients", response_model=PatientOut)
def create_patient(payload: PatientCreate, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    patient = Patient(owner_user_id=user.id, display_name=payload.display_name, disease_pack=payload.disease_pack)
    db.add(patient)
    db.flush()
    rebuild_state(db, patient.id)
    audit(db, user_id=user.id, patient_id=patient.id, action="patient.create", object_type="patient", object_id=patient.id)
    db.commit()
    return PatientOut(id=patient.id, display_name=patient.display_name, disease_pack=patient.disease_pack)


@app.get("/v1/patients/{patient_id}/documents", response_model=list[DocumentOut])
def list_documents(patient_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    owned_patient(db, patient_id, user)
    docs = list(db.scalars(
        select(Document).where(Document.patient_id == patient_id, Document.deleted_at.is_(None)).order_by(Document.created_at.desc())
    ))
    # RECONCILE_STALE_DOCUMENT_REVIEW_STATUS
    changed = False
    for doc in docs:
        before = doc.status
        if before == "needs_review":
            refresh_document_review_status(db, doc.id)
            changed = changed or doc.status != before
    if changed:
        db.commit()
    return [DocumentOut(
        id=d.id, filename=d.filename, document_type=d.document_type, status=d.status,
        created_at=d.created_at.isoformat(), processing_error=d.processing_error
    ) for d in docs]


@app.post("/v1/patients/{patient_id}/documents", response_model=DocumentOut)
async def upload_document(
    patient_id: str,
    file: UploadFile = File(...),
    process: bool = Query(True),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    owned_patient(db, patient_id, user)
    if file.content_type not in {"application/pdf", "application/octet-stream"} and not file.filename.lower().endswith(".pdf"):
        raise HTTPException(415, "Checkpoint 1 accepts PDF reports only")
    data = await file.read()
    if not data.startswith(b"%PDF"):
        raise HTTPException(400, "File does not look like a PDF")
    sha = hashlib.sha256(data).hexdigest()
    existing = db.scalar(select(Document).where(Document.patient_id == patient_id, Document.sha256 == sha))
    if existing and existing.deleted_at is None:
        return DocumentOut(id=existing.id, filename=existing.filename, document_type=existing.document_type, status=existing.status,
                           created_at=existing.created_at.isoformat(), processing_error=existing.processing_error)
    doc = Document(
        patient_id=patient_id,
        owner_user_id=user.id,
        filename=Path(file.filename or "record.pdf").name,
        mime_type="application/pdf",
        sha256=sha,
        storage_key="pending",
        status="uploaded",
    )
    db.add(doc)
    db.flush()
    doc.storage_key = f"patients/{patient_id}/documents/{doc.id}/original.pdf"
    get_object_store().put_bytes(doc.storage_key, data, "application/pdf")
    audit(db, user_id=user.id, patient_id=patient_id, action="document.upload", object_type="document", object_id=doc.id,
          metadata={"sha256": sha, "filename": doc.filename})
    db.commit()
    if process:
        process_document_task.delay(doc.id, user.id)
        db.refresh(doc)
    return DocumentOut(id=doc.id, filename=doc.filename, document_type=doc.document_type, status=doc.status,
                       created_at=doc.created_at.isoformat(), processing_error=doc.processing_error)


@app.delete("/v1/documents/{document_id}")
def delete_document(document_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    doc = owned_document(db, document_id, user)
    doc.deleted_at = datetime.now(timezone.utc)
    doc.status = "deleted"
    supersede_document_facts(db, doc.id, doc.patient_id, user.id)
    audit(db, user_id=user.id, patient_id=doc.patient_id, action="document.delete", object_type="document", object_id=doc.id)
    db.commit()
    return {"status": "deleted", "document_id": doc.id}


@app.get("/v1/documents/{document_id}/pages/{page_number}")
def get_page(document_id: str, page_number: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    owned_document(db, document_id, user)
    page = db.scalar(select(DocumentPage).where(DocumentPage.document_id == document_id, DocumentPage.page_number == page_number))
    if page is None:
        raise HTTPException(404, "Page not found")
    return {"document_id": document_id, "page_number": page_number, "text": page.text, "ocr_used": page.ocr_used}


@app.get("/v1/documents/{document_id}/pages/{page_number}/image")
def get_page_image(document_id: str, page_number: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    owned_document(db, document_id, user)
    page = db.scalar(select(DocumentPage).where(DocumentPage.document_id == document_id, DocumentPage.page_number == page_number))
    if page is None or not page.image_storage_key:
        raise HTTPException(404, "Page image not found")
    return Response(get_object_store().get_bytes(page.image_storage_key), media_type="image/png")


@app.get("/v1/patients/{patient_id}/review", response_model=list[CandidateFactOut])
def review_queue(patient_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    owned_patient(db, patient_id, user)
    candidates = list(db.scalars(
        select(CandidateFact).where(CandidateFact.patient_id == patient_id, CandidateFact.status == "needs_review").order_by(CandidateFact.created_at.asc())
    ))
    out = []
    for c in candidates:
        doc = db.get(Document, c.document_id)
        out.append(CandidateFactOut(
            id=c.id, document_id=c.document_id, document_name=doc.filename if doc else "Unknown document",
            fact_type=c.fact_type, value=str(c.value_json.get("value", "")), effective_date=c.effective_date,
            confidence=c.confidence, source_page=c.source_page, source_snippet=c.source_snippet,
            review_reason=c.review_reason, status=c.status,
        ))
    return out


@app.post("/v1/candidates/{candidate_id}/decision")
def decide_candidate(payload: CandidateDecision, candidate_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    candidate = db.get(CandidateFact, candidate_id)
    if candidate is None:
        raise HTTPException(404, "Candidate fact not found")
    owned_patient(db, candidate.patient_id, user)
    if candidate.status != "needs_review":
        raise HTTPException(409, f"Candidate is not awaiting review: {candidate.status}")
    if payload.decision == "reject":
        candidate.status = "rejected"
        candidate.decided_at = datetime.now(timezone.utc)
        audit(db, user_id=user.id, patient_id=candidate.patient_id, action="fact.reject", object_type="candidate_fact", object_id=candidate.id,
              metadata={"note": payload.note or ""})
    else:
        if payload.decision == "edit" and not payload.edited_value:
            raise HTTPException(422, "edited_value is required for edit")
        commit_candidate(
            db,
            candidate,
            user_id=user.id,
            verification_state="user_verified" if payload.decision == "accept" else "user_edited",
            edited_value=payload.edited_value,
            edited_effective_date=payload.edited_effective_date,
        )
    db.flush()
    refresh_document_review_status(db, candidate.document_id)
    db.commit()
    return {"status": candidate.status}


@app.get("/v1/patients/{patient_id}/state", response_model=StateResponse)
def get_state(patient_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    owned_patient(db, patient_id, user)
    snap, sources = state_with_sources(db, patient_id)
    if snap is None:
        snap = rebuild_state(db, patient_id)
        db.commit()
    return StateResponse(snapshot_id=snap.id, state_hash=snap.state_hash, state=snap.state_json, sources=sources)


@app.get("/v1/patients/{patient_id}/timeline", response_model=list[TimelineEventOut])
def get_timeline(patient_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    owned_patient(db, patient_id, user)
    events = list(db.scalars(
        select(TimelineEvent).where(TimelineEvent.patient_id == patient_id).order_by(TimelineEvent.event_date.asc().nullsfirst(), TimelineEvent.created_at.asc())
    ))
    out = []
    for ev in events:
        source = None
        fs = db.scalar(select(FactSource).where(FactSource.committed_fact_id == ev.source_fact_id).limit(1))
        if fs:
            doc = db.get(Document, fs.document_id)
            if doc and doc.deleted_at is None:
                source = SourceOut(document_id=doc.id, document_name=doc.filename, page_number=fs.page_number, source_snippet=fs.source_snippet)
        out.append(TimelineEventOut(id=ev.id, event_type=ev.event_type, event_date=ev.event_date, title=ev.title, summary=ev.summary, source=source))
    return out

# === ONCOTWIN CP2 FORECAST ROUTES ===
from .forecast_service import create_forecast as cp2_create_forecast
from .forecast_service import latest_forecast as cp2_latest_forecast
from .forecast_service import model_health as cp2_model_health
from .models import Patient as CP2Patient


def _cp2_owned_patient(patient_id: str, user: User, db: Session):
    patient = db.get(CP2Patient, patient_id)
    if patient is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    owner_attr = None
    for candidate in ("user_id", "owner_id", "owner_user_id"):
        if hasattr(patient, candidate):
            owner_attr = candidate
            break
    if owner_attr is None:
        raise HTTPException(status_code=500, detail="Patient ownership field not configured")
    if str(getattr(patient, owner_attr)) != str(user.id):
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient


@app.get("/v1/model/health")
def cp2_model_health_endpoint(user: User = Depends(get_current_user)):
    return cp2_model_health()


@app.get("/v1/patients/{patient_id}/forecast")
def cp2_get_forecast(patient_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    _cp2_owned_patient(patient_id, user, db)
    result = cp2_latest_forecast(db, patient_id)
    if result is None:
        return {"status": "NOT_RUN", "patient_id": patient_id, "research_status": "EXTERNAL_CONFIRMATION_PENDING"}
    return result


@app.post("/v1/patients/{patient_id}/forecast")
def cp2_post_forecast(patient_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    _cp2_owned_patient(patient_id, user, db)
    try:
        result = cp2_create_forecast(db, patient_id, str(user.id))
        db.commit()
        return result
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=503, detail=f"Research forecast unavailable: {exc}") from exc
# === END ONCOTWIN CP2 FORECAST ROUTES ===

# === ONCOTWIN CP4 PATIENT INTELLIGENCE ROUTES ===
from .evidence_service import search_evidence as cp4_search_evidence
from .export_service import create_second_opinion_export as cp4_create_export
from .export_service import get_export_bytes as cp4_get_export_bytes
from .intelligence_schemas import EvidenceGenerateRequest, ExportRequest, IntelligenceGenerateRequest, WhatChangedRequest
from .intelligence_service import (
    explain_record as cp4_explain_record,
    generate_visit_brief as cp4_generate_visit_brief,
    generate_visit_questions as cp4_generate_visit_questions,
    generate_what_changed as cp4_generate_what_changed,
    latest_intelligence_bundle as cp4_latest_intelligence_bundle,
    latest_record_explanation as cp4_latest_record_explanation,
)
from .llm_router import RouterUnavailable
from .models import GeneratedArtifact as CP4GeneratedArtifact


def _cp4_commit_or_http(db: Session, fn):
    try:
        result = fn()
        db.commit()
        return result
    except RouterUnavailable as exc:
        db.rollback()
        raise HTTPException(status_code=503, detail=f"Patient-intelligence service is temporarily unavailable: {exc}") from exc
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=503, detail=f"Patient-intelligence workflow failed: {exc}") from exc


@app.get("/v1/documents/{document_id}/explanation")
def cp4_get_record_explanation(document_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    doc = owned_document(db, document_id, user)
    result = cp4_latest_record_explanation(db, doc.patient_id, doc.id)
    return result.model_dump() if result else {"status": "NOT_RUN", "document_id": doc.id}


@app.post("/v1/documents/{document_id}/explanation")
def cp4_post_record_explanation(
    document_id: str,
    payload: IntelligenceGenerateRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    doc = owned_document(db, document_id, user)
    return _cp4_commit_or_http(db, lambda: cp4_explain_record(db, doc.patient_id, doc.id, refresh=payload.refresh).model_dump())


@app.get("/v1/patients/{patient_id}/intelligence")
def cp4_get_intelligence(patient_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    owned_patient(db, patient_id, user)
    return cp4_latest_intelligence_bundle(db, patient_id)


@app.post("/v1/patients/{patient_id}/intelligence/what-changed")
def cp4_post_what_changed(
    patient_id: str,
    payload: WhatChangedRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    owned_patient(db, patient_id, user)
    if payload.document_id:
        doc = owned_document(db, payload.document_id, user)
        if doc.patient_id != patient_id:
            raise HTTPException(404, "Document not found")
    return _cp4_commit_or_http(
        db,
        lambda: cp4_generate_what_changed(db, patient_id, payload.before_state_hash, payload.document_id, refresh=False).model_dump(),
    )


@app.post("/v1/patients/{patient_id}/intelligence/questions")
def cp4_post_visit_questions(
    patient_id: str,
    payload: IntelligenceGenerateRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    owned_patient(db, patient_id, user)
    return _cp4_commit_or_http(db, lambda: cp4_generate_visit_questions(db, patient_id, refresh=payload.refresh).model_dump())


@app.post("/v1/patients/{patient_id}/intelligence/visit-brief")
def cp4_post_visit_brief(
    patient_id: str,
    payload: IntelligenceGenerateRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    owned_patient(db, patient_id, user)
    return _cp4_commit_or_http(db, lambda: cp4_generate_visit_brief(db, patient_id, refresh=payload.refresh).model_dump())


@app.post("/v1/patients/{patient_id}/evidence")
def cp4_post_evidence(
    patient_id: str,
    payload: EvidenceGenerateRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    owned_patient(db, patient_id, user)
    return _cp4_commit_or_http(
        db,
        lambda: cp4_search_evidence(
            db,
            patient_id,
            refresh=payload.refresh,
            max_literature=payload.max_literature,
            max_trials=payload.max_trials,
        ),
    )


@app.post("/v1/patients/{patient_id}/export")
def cp4_post_export(
    patient_id: str,
    payload: ExportRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    owned_patient(db, patient_id, user)
    return _cp4_commit_or_http(
        db,
        lambda: cp4_create_export(
            db,
            patient_id,
            include_evidence=payload.include_evidence,
            include_forecast=payload.include_forecast,
        ),
    )


@app.get("/v1/exports/{artifact_id}/download")
def cp4_download_export(artifact_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    artifact = db.get(CP4GeneratedArtifact, artifact_id)
    if artifact is None or not artifact.patient_id:
        raise HTTPException(404, "Export not found")
    owned_patient(db, artifact.patient_id, user)
    try:
        data, filename = cp4_get_export_bytes(db, artifact_id, artifact.patient_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    return Response(
        data,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
# === END ONCOTWIN CP4 PATIENT INTELLIGENCE ROUTES ===

# === ONCOTWIN CP4.5 ASK ONCOTWIN ROUTES ===
from .assistant_schemas import AssistantAskRequest
from .assistant_service import answer_question as cp45_answer_question
from .assistant_service import bootstrap_assistant as cp45_bootstrap_assistant


@app.get("/v1/patients/{patient_id}/assistant")
def cp45_get_assistant(
    patient_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    owned_patient(db, patient_id, user)
    return _cp4_commit_or_http(db, lambda: cp45_bootstrap_assistant(db, patient_id).model_dump())


@app.post("/v1/patients/{patient_id}/assistant/messages")
def cp45_post_assistant_message(
    patient_id: str,
    payload: AssistantAskRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    owned_patient(db, patient_id, user)
    return _cp4_commit_or_http(db, lambda: cp45_answer_question(db, patient_id, payload).model_dump())
# === END ONCOTWIN CP4.5 ASK ONCOTWIN ROUTES ===

