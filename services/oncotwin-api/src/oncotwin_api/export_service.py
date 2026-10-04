from __future__ import annotations

import hashlib
import textwrap
from datetime import datetime, timezone
from typing import Any

import pymupdf as fitz
from sqlalchemy import select
from sqlalchemy.orm import Session

from .forecast_service import latest_forecast
from .models import GeneratedArtifact, PatientStateSnapshot
from .state import canonical_json
from .storage import get_object_store

EXPORT_VERSION = "cp4.second_opinion.v1"


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


def _latest_artifact(db: Session, patient_id: str, kind: str, state_hash: str) -> GeneratedArtifact | None:
    return db.scalar(
        select(GeneratedArtifact)
        .where(
            GeneratedArtifact.patient_id == patient_id,
            GeneratedArtifact.artifact_type == kind,
            GeneratedArtifact.status == "COMPLETED",
            GeneratedArtifact.state_hash == state_hash,
        )
        .order_by(GeneratedArtifact.created_at.desc())
        .limit(1)
    )


def _pct(value: Any) -> str:
    return f"{float(value) * 100:.0f}%" if isinstance(value, (int, float)) else "Unavailable"


def _plain_lines(state: dict[str, Any], brief: dict[str, Any] | None, questions: dict[str, Any] | None, forecast: dict[str, Any] | None, evidence: dict[str, Any] | None, include_evidence: bool, include_forecast: bool) -> list[tuple[str, str]]:
    lines: list[tuple[str, str]] = []
    lines.append(("title", "OncoTwin cancer summary for a visit or second opinion"))
    lines.append(("body", "Generated from verified information in the OncoTwin record. Review against the original reports and with the oncology team."))
    lines.append(("h1", "Current cancer picture"))
    lines.append(("body", f"Diagnosis: {state.get('diagnosis') or 'Not established'}"))
    lines.append(("body", f"Subtype: {state.get('subtype') or 'Not established'}"))
    tx = (state.get("current_treatment") or {}).get("name")
    lines.append(("body", f"Current treatment: {tx or 'Not established'}"))
    sites = ", ".join(state.get("disease_sites") or []) or "Not established"
    lines.append(("body", f"Known disease sites: {sites}"))
    scan = state.get("latest_scan") or {}
    lines.append(("body", f"Latest scan: {scan.get('assessment') or 'Not established'}" + (f" ({scan.get('date')})" if scan.get("date") else "")))
    genomics = ", ".join(x.get("alteration") for x in (state.get("genomics") or []) if x.get("alteration")) or "None recorded"
    lines.append(("body", f"Recorded genomic alterations: {genomics}"))

    if brief:
        lines.append(("h1", "Recent changes and visit summary"))
        if brief.get("headline"):
            lines.append(("body", str(brief["headline"])))
        for item in brief.get("recent_changes") or []:
            lines.append(("bullet", str(item)))
        if brief.get("uncertainties"):
            lines.append(("h2", "Information still uncertain or missing"))
            for item in brief.get("uncertainties") or []:
                lines.append(("bullet", str(item)))

    if include_forecast and forecast:
        lines.append(("h1", "Research-model trajectory"))
        if forecast.get("run_status") == "COMPLETED" and forecast.get("horizons"):
            h6 = forecast["horizons"].get("6") or {}
            lines.append(("body", f"6-month progression-free estimate before newest scan: {_pct(h6.get('pre_pfs'))}."))
            lines.append(("body", f"6-month selected estimate after newest scan: {_pct(h6.get('selected_pfs'))}."))
        else:
            lines.append(("body", "Quantitative research forecast unavailable for this state."))
        support = forecast.get("support") or {}
        if support.get("status"):
            lines.append(("body", f"Model support: {support.get('status')}."))
        lines.append(("body", "This is a research-model estimate, not a treatment recommendation or a substitute for clinical judgment."))

    if questions:
        lines.append(("h1", "Questions for the oncology visit"))
        for row in questions.get("questions") or []:
            q = row.get("question") if isinstance(row, dict) else row
            if q:
                lines.append(("bullet", str(q)))

    if include_evidence and evidence:
        lines.append(("h1", "Selected evidence and trials"))
        for row in (evidence.get("literature") or [])[:4]:
            lines.append(("bullet", f"{row.get('title')} — {row.get('url')}"))
        for row in (evidence.get("trials") or [])[:4]:
            lines.append(("bullet", f"{row.get('title')} ({row.get('source_id')}) — {row.get('url')}"))
        lines.append(("body", "Trial relevance is not an eligibility determination. Confirm eligibility and treatment implications with the oncology team and trial site."))

    lines.append(("h1", "Source and scope note"))
    lines.append(("body", "OncoTwin keeps verified facts connected to uploaded records. This export is a concise patient-controlled summary and does not replace the original medical record."))
    return lines


def _render_pdf(lines: list[tuple[str, str]]) -> bytes:
    doc = fitz.open()
    page = None
    y = 58.0
    width = 612.0
    height = 792.0
    left = 58.0
    right = 554.0

    def new_page():
        nonlocal page, y
        page = doc.new_page(width=width, height=height)
        y = 58.0
        page.insert_text((left, 30), "ONCOTWIN", fontsize=9, fontname="helv", color=(0.15, 0.4, 0.45))
        page.insert_text((right - 105, 30), "Patient-controlled summary", fontsize=7.5, fontname="helv", color=(0.4, 0.4, 0.4))

    def emit(text: str, size: float, *, bold: bool = False, indent: float = 0, gap_after: float = 5):
        nonlocal y, page
        font = "hebo" if bold else "helv"
        wrap_chars = max(45, int((right - left - indent) / (size * 0.52)))
        chunks = textwrap.wrap(text, width=wrap_chars, break_long_words=False, replace_whitespace=False) or [""]
        needed = len(chunks) * (size * 1.35) + gap_after
        if page is None or y + needed > height - 52:
            new_page()
        for chunk in chunks:
            page.insert_text((left + indent, y), chunk, fontsize=size, fontname=font, color=(0.08, 0.12, 0.15))
            y += size * 1.35
        y += gap_after

    for kind, text in lines:
        if kind == "title":
            emit(text, 17, bold=True, gap_after=10)
        elif kind == "h1":
            y += 5
            emit(text, 12, bold=True, gap_after=5)
        elif kind == "h2":
            emit(text, 10.5, bold=True, gap_after=4)
        elif kind == "bullet":
            emit("• " + text, 9.2, indent=8, gap_after=2.5)
        else:
            emit(text, 9.2, gap_after=4)
    pdf = doc.tobytes(garbage=4, deflate=True)
    doc.close()
    return pdf


def create_second_opinion_export(db: Session, patient_id: str, *, include_evidence: bool = True, include_forecast: bool = True) -> dict[str, Any]:
    state = _latest_state(db, patient_id)
    brief_art = _latest_artifact(db, patient_id, "visit_brief_final", state.state_hash)
    questions_art = _latest_artifact(db, patient_id, "visit_questions_final", state.state_hash)
    evidence_art = _latest_artifact(db, patient_id, "evidence_bundle", state.state_hash)
    forecast = latest_forecast(db, patient_id)
    if forecast and forecast.get("state_hash") != state.state_hash:
        forecast = None

    basis = {
        "state_hash": state.state_hash,
        "brief": brief_art.id if brief_art else None,
        "questions": questions_art.id if questions_art else None,
        "evidence": evidence_art.id if evidence_art and include_evidence else None,
        "forecast": forecast.get("id") if forecast and include_forecast else None,
        "version": EXPORT_VERSION,
    }
    cache_key = _sha({"artifact_type": "second_opinion_export", **basis})
    existing = db.scalar(select(GeneratedArtifact).where(GeneratedArtifact.cache_key == cache_key, GeneratedArtifact.status == "COMPLETED"))
    if existing is not None:
        return {"artifact_id": existing.id, "cache_hit": True, **existing.content_json}

    lines = _plain_lines(
        state.state_json,
        brief_art.content_json if brief_art else None,
        questions_art.content_json if questions_art else None,
        forecast if include_forecast else None,
        evidence_art.content_json if evidence_art and include_evidence else None,
        include_evidence,
        include_forecast,
    )
    pdf = _render_pdf(lines)
    sha = hashlib.sha256(pdf).hexdigest()
    artifact = GeneratedArtifact(
        patient_id=patient_id,
        artifact_type="second_opinion_export",
        cache_key=cache_key,
        status="COMPLETED",
        provider="deterministic_export",
        model_name="pymupdf",
        prompt_version=EXPORT_VERSION,
        schema_version=EXPORT_VERSION,
        input_hash=_sha(basis),
        state_hash=state.state_hash,
        source_hashes_json=[],
        content_json={},
        grounding_json=[],
        metadata_json=basis,
    )
    db.add(artifact)
    db.flush()
    storage_key = f"patients/{patient_id}/exports/{artifact.id}/oncotwin_second_opinion.pdf"
    get_object_store().put_bytes(storage_key, pdf, "application/pdf")
    artifact.content_json = {
        "filename": "OncoTwin_second_opinion_summary.pdf",
        "storage_key": storage_key,
        "sha256": sha,
        "size_bytes": len(pdf),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "include_evidence": include_evidence,
        "include_forecast": include_forecast,
    }
    db.flush()
    return {"artifact_id": artifact.id, "cache_hit": False, **artifact.content_json}


def get_export_bytes(db: Session, artifact_id: str, patient_id: str) -> tuple[bytes, str]:
    artifact = db.get(GeneratedArtifact, artifact_id)
    if artifact is None or artifact.patient_id != patient_id or artifact.artifact_type != "second_opinion_export" or artifact.status != "COMPLETED":
        raise ValueError("Export not found")
    content = artifact.content_json or {}
    key = content.get("storage_key")
    if not key:
        raise ValueError("Export file is unavailable")
    return get_object_store().get_bytes(key), str(content.get("filename") or "OncoTwin_summary.pdf")
