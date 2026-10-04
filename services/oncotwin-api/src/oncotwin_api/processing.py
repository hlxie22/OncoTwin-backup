from __future__ import annotations

import hashlib
import io
import shutil
from datetime import datetime, timezone

import pymupdf as fitz
from PIL import Image
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import get_settings
from .extractor import classify_document, extract_facts
from .llm_router import extraction_result_via_router
from .demo_fixtures import (
    extraction_identity,
    fixture_mode_enabled,
    load_extraction_fixture,
)
from .models import CandidateFact, CommittedFact, Document, DocumentPage, ExtractionRun, FactConflict
from .state import HIGH_IMPACT_TYPES, commit_candidate, rebuild_state
from .storage import get_object_store


def _ocr_image(png_bytes: bytes) -> tuple[str, bool]:
    if shutil.which("tesseract") is None:
        return "", False
    try:
        import pytesseract
        image = Image.open(io.BytesIO(png_bytes))
        return pytesseract.image_to_string(image), True
    except Exception:
        return "", False


def extract_pdf_pages(pdf_bytes: bytes) -> list[dict]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages: list[dict] = []
    for idx, page in enumerate(doc):
        text = page.get_text("text").strip()
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
        png = pix.tobytes("png")
        ocr_used = False
        if len(text) < 24:
            ocr_text, used = _ocr_image(png)
            if used and len(ocr_text.strip()) > len(text):
                text = ocr_text.strip()
                ocr_used = True
        pages.append({"page_number": idx + 1, "text": text, "png": png, "ocr_used": ocr_used})
    return pages


def text_with_page_markers(pages: list[dict]) -> str:
    return "\n\n".join(f"[PAGE {p['page_number']}]\n{p['text']}" for p in pages)


def _find_conflict(db: Session, patient_id: str, candidate: CandidateFact) -> CommittedFact | None:
    if candidate.fact_type not in {"er_status", "pr_status", "her2_status"}:
        return None
    current = db.scalar(
        select(CommittedFact)
        .where(
            CommittedFact.patient_id == patient_id,
            CommittedFact.fact_type == candidate.fact_type,
            CommittedFact.status == "active",
        )
        .order_by(CommittedFact.created_at.desc())
        .limit(1)
    )
    if current and current.value_json.get("value") != candidate.value_json.get("value"):
        return current
    return None


def process_document(db: Session, document_id: str, user_id: str) -> None:
    settings = get_settings()
    store = get_object_store()
    doc = db.get(Document, document_id)
    if doc is None or doc.deleted_at is not None:
        return
    doc.status = "processing"
    doc.processing_error = None
    db.commit()

    if fixture_mode_enabled():
        fixture_identity = extraction_identity(doc.sha256)
        run = ExtractionRun(
            document_id=doc.id,
            provider=fixture_identity["provider"],
            model_name=fixture_identity["model_name"],
            schema_version=fixture_identity["schema_version"],
            prompt_version=fixture_identity["prompt_version"],
            status="running",
        )
    else:
        run = ExtractionRun(
            document_id=doc.id,
            provider=settings.extractor_provider,
            model_name=settings.openai_model if settings.extractor_provider == "openai" else "deterministic-rules-cp1",
            status="running",
        )
    db.add(run)
    db.commit()

    try:
        pdf_bytes = store.get_bytes(doc.storage_key)
        pages = extract_pdf_pages(pdf_bytes)
        for p in pages:
            image_key = f"patients/{doc.patient_id}/documents/{doc.id}/pages/{p['page_number']}.png"
            store.put_bytes(image_key, p["png"], "image/png")
            existing = db.scalar(select(DocumentPage).where(DocumentPage.document_id == doc.id, DocumentPage.page_number == p["page_number"]))
            if existing is None:
                db.add(DocumentPage(
                    document_id=doc.id,
                    page_number=p["page_number"],
                    text=p["text"],
                    image_storage_key=image_key,
                    ocr_used=p["ocr_used"],
                ))
            else:
                existing.text = p["text"]
                existing.image_storage_key = image_key
                existing.ocr_used = p["ocr_used"]
        combined = text_with_page_markers(pages)
        doc.document_type = classify_document(combined)
        if fixture_mode_enabled():
            payload = load_extraction_fixture(doc.sha256)
        elif settings.extractor_provider == "router":
            payload, router_meta = extraction_result_via_router(
                db,
                text_with_pages=combined,
                document_type=doc.document_type,
                document_sha256=doc.sha256,
                patient_id=doc.patient_id,
                document_id=doc.id,
            )
            run.provider = router_meta["provider"]
            run.model_name = router_meta["model_name"]
            run.schema_version = router_meta["schema_version"]
            run.prompt_version = router_meta["prompt_version"]
        else:
            payload = extract_facts(combined, doc.document_type)
        doc.document_type = payload.document_type

        response_json = payload.model_dump_json()
        run.response_sha256 = hashlib.sha256(response_json.encode()).hexdigest()
        for extracted in payload.facts:
            page = max(1, min(extracted.source_page, len(pages) if pages else 1))
            candidate = CandidateFact(
                patient_id=doc.patient_id,
                document_id=doc.id,
                extraction_run_id=run.id,
                fact_type=extracted.fact_type,
                value_json={"value": extracted.value, "attributes": extracted.attributes},
                effective_date=extracted.effective_date,
                unit=extracted.unit,
                confidence=extracted.confidence,
                source_page=page,
                source_snippet=extracted.source_snippet[:1200],
                attributes_json=extracted.attributes,
                status="candidate",
            )
            db.add(candidate)
            db.flush()
            conflict = _find_conflict(db, doc.patient_id, candidate)
            needs_review = False
            reasons = []
            if conflict:
                needs_review = True
                reasons.append("conflicts with an existing committed fact")
                db.add(FactConflict(
                    patient_id=doc.patient_id,
                    candidate_fact_id=candidate.id,
                    conflicting_fact_id=conflict.id,
                ))
            if candidate.fact_type in HIGH_IMPACT_TYPES and candidate.confidence < settings.extraction_confidence_threshold:
                needs_review = True
                reasons.append("high-impact field below verification threshold")
            if needs_review:
                candidate.status = "needs_review"
                candidate.review_reason = "; ".join(reasons)
            else:
                commit_candidate(db, candidate, user_id=user_id, verification_state="auto")

        run.status = "complete"
        run.finished_at = datetime.now(timezone.utc)
        doc.status = "ready" if not any(
            c.status == "needs_review" for c in db.scalars(select(CandidateFact).where(CandidateFact.document_id == doc.id))
        ) else "needs_review"
        rebuild_state(db, doc.patient_id)
        db.commit()
    except Exception as exc:
        db.rollback()
        doc = db.get(Document, document_id)
        run = db.get(ExtractionRun, run.id)
        if doc:
            doc.status = "failed"
            doc.processing_error = f"{type(exc).__name__}: {exc}"[:4000]
        if run:
            run.status = "failed"
            run.error = f"{type(exc).__name__}: {exc}"[:4000]
            run.finished_at = datetime.now(timezone.utc)
        db.commit()
        raise
