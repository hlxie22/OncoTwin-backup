from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from .demo_assistant import lookup_question, render_answer, suggested_questions
from .assistant_schemas import (
    AssistantAnswerLLM,
    AssistantAnswerResponse,
    AssistantAskRequest,
    AssistantBootstrapResponse,
)
from .demo_fixtures import fixture_mode_enabled
from .forecast_service import latest_forecast
from .intelligence_schemas import GroundingRef
from .intelligence_service import patient_grounding_catalog
from .llm_router import run_structured
from .models import GeneratedArtifact


ASSISTANT_PROMPT_VERSION = "cp4.5.patient_assistant.v1"


def assistant_stage(state: dict[str, Any]) -> str:
    genomics = state.get("genomics") or []
    if any("ESR1" in str((x or {}).get("alteration") or "").upper() for x in genomics if isinstance(x, dict)):
        return "after_esr1"
    assessment = str((state.get("latest_scan") or {}).get("assessment") or "").lower()
    if "progress" in assessment:
        return "after_progression"
    return "baseline"


def _pct(value: Any) -> str:
    return f"{round(float(value) * 100)}%" if isinstance(value, (int, float)) else "not available"


def _forecast_refs(db: Session, patient_id: str, state_hash: str) -> dict[str, Any]:
    forecast = latest_forecast(db, patient_id)
    if not forecast:
        return {"status": "NOT_RUN"}
    current = forecast.get("state_hash") == state_hash
    horizons = forecast.get("horizons") or {}
    h6 = horizons.get("6") or horizons.get(6) or {}
    pre = h6.get("pre_pfs")
    selected = h6.get("selected_pfs")
    delta = (float(selected) - float(pre)) * 100 if isinstance(pre, (int, float)) and isinstance(selected, (int, float)) else None
    return {
        "status": "CURRENT" if current else "STALE",
        "forecast_run_id": forecast.get("id"),
        "support": forecast.get("support"),
        "pre_pfs_6m": pre,
        "selected_pfs_6m": selected,
        "delta_6m_pp": delta,
        "pre_pfs_6m_display": _pct(pre),
        "selected_pfs_6m_display": _pct(selected),
        "delta_6m_pp_display": f"{delta:+.0f}" if isinstance(delta, (int, float)) else "not available",
        "horizons": horizons if current else {},
        "research_status": forecast.get("research_status") or "EXTERNAL_CONFIRMATION_PENDING",
    }


def _refs_for_fact_types(context: dict[str, Any], catalog: list[GroundingRef], fact_types: tuple[str, ...]) -> list[GroundingRef]:
    by_key = {x.key: x for x in catalog}
    wanted = set(fact_types)
    out: list[GroundingRef] = []
    seen: set[str] = set()
    for fact in context.get("verified_facts") or []:
        if wanted and fact.get("fact_type") not in wanted:
            continue
        key = fact.get("source_key")
        if key and key in by_key and key not in seen:
            seen.add(key)
            out.append(by_key[key])
        if len(out) >= 5:
            break
    return out


def _evidence_context(db: Session, patient_id: str, state_hash: str) -> dict[str, Any] | None:
    row = db.scalar(
        select(GeneratedArtifact)
        .where(
            GeneratedArtifact.patient_id == patient_id,
            GeneratedArtifact.artifact_type == "evidence_bundle",
            GeneratedArtifact.state_hash == state_hash,
            GeneratedArtifact.status == "COMPLETED",
        )
        .order_by(GeneratedArtifact.created_at.desc())
        .limit(1)
    )
    if row is None:
        return None
    content = row.content_json or {}
    return {
        "literature": [
            {"source_id": x.get("source_id"), "title": x.get("title"), "patient_summary": x.get("patient_summary")}
            for x in (content.get("literature") or [])[:4]
        ],
        "trials": [
            {"source_id": x.get("source_id"), "title": x.get("title"), "patient_relevance": x.get("patient_relevance")}
            for x in (content.get("trials") or [])[:4]
        ],
    }


def bootstrap_assistant(db: Session, patient_id: str) -> AssistantBootstrapResponse:
    context, _ = patient_grounding_catalog(db, patient_id)
    stage = assistant_stage(context["state"])
    return AssistantBootstrapResponse(
        stage=stage,
        mode="fixture" if fixture_mode_enabled() else "llm",
        greeting="Ask about your cancer history, your trajectory, what changed, or what may be useful to discuss with your care team.",
        suggested_questions=suggested_questions(stage),
    )


def answer_question(db: Session, patient_id: str, request: AssistantAskRequest) -> AssistantAnswerResponse:
    context, catalog = patient_grounding_catalog(db, patient_id)
    state = context["state"]
    state_hash = context["state_hash"]
    stage = assistant_stage(state)
    forecast_refs = _forecast_refs(db, patient_id, state_hash)

    # The deterministic Maya demo must never reach an LLM provider. Only exact
    # authored questions (ignoring cosmetic case/terminal punctuation) resolve.
    if fixture_mode_enabled():
        authored, row = lookup_question(stage, request.question)
        if row is None:
            return AssistantAnswerResponse(
                stage=stage,
                mode="fixture",
                matched_question=False,
                answer="I don't have a prepared answer for that question in this demo. Try one of the suggested questions below, or type one of the other pre-authored questions available for this point in Maya's journey.",
                suggested_questions=suggested_questions(stage),
                suggested_followups=suggested_questions(stage)[:4],
                grounding=[],
                forecast_refs=forecast_refs,
                provider="demo_fixture",
                model_name="fixture:maya_rowan_v1",
            )
        refs = _refs_for_fact_types(context, catalog, row.source_fact_types)
        if not refs and row.source_fact_types:
            refs = catalog[:3]
        followups = list(row.followups) if row.followups else [x for x in suggested_questions(stage) if x != authored][:4]
        return AssistantAnswerResponse(
            stage=stage,
            mode="fixture",
            matched_question=True,
            answer=render_answer(row.answer, forecast_refs),
            suggested_questions=suggested_questions(stage),
            suggested_followups=followups[:4],
            grounding=refs,
            forecast_refs=forecast_refs,
            provider="demo_fixture",
            model_name="fixture:maya_rowan_v1",
        )

    history = [x.model_dump() for x in request.history[-8:]]
    payload = {
        "question": request.question,
        "recent_conversation": history,
        "verified_patient_state": state,
        "verified_facts": context.get("verified_facts") or [],
        "forecast_context": forecast_refs,
        "source_catalog": [x.model_dump() for x in catalog[:24]],
        "existing_retrieved_evidence": _evidence_context(db, patient_id, state_hash),
        "suggestion_style": "Return up to four short patient-friendly follow-up questions. Prefer broad useful next questions over technical labels.",
    }
    system_extra = """
Answer the user's free-form question as a patient-facing interpreter of the verified OncoTwin record.
Keep three categories distinct whenever relevant: (1) what the medical record says, (2) what the frozen forecasting model estimates, and (3) general/evidence context.
For patient-specific factual claims, cite only source_keys present in source_catalog. Never invent a source key.
Do not claim that one lesion or feature caused a forecast change. Do not calculate new prognosis values; if you mention a forecast number, copy it exactly from forecast_context.
Do not tell the patient to start, stop, switch, or choose a treatment. Do not assert clinical-trial eligibility.
If the supplied context cannot answer the question safely, say what is not established and redirect to an answerable question.
If current literature or trial evidence is required but existing_retrieved_evidence is absent, explain that the Evidence section should run a source-linked search rather than inventing current evidence from memory.
""".strip()
    result = run_structured(
        db,
        task="patient_assistant",
        payload=payload,
        response_model=AssistantAnswerLLM,
        patient_id=patient_id,
        state_hash=state_hash,
        grounding=[x.model_dump() for x in catalog],
        refresh=True,
        cache_basis={"state_hash": state_hash, "question": request.question, "history": history},
        system_extra=system_extra,
    )
    by_key = {x.key: x for x in catalog}
    refs: list[GroundingRef] = []
    seen: set[str] = set()
    for key in result.value.source_keys:
        if key in by_key and key not in seen:
            seen.add(key)
            refs.append(by_key[key])
    return AssistantAnswerResponse(
        stage=stage,
        mode="llm",
        matched_question=True,
        answer=result.value.answer,
        suggested_questions=suggested_questions(stage),
        suggested_followups=result.value.suggested_followups[:4],
        grounding=refs,
        forecast_refs=forecast_refs,
        provider=result.identity.provider,
        model_name=result.identity.model_name,
        uncertainty_note=result.value.uncertainty_note,
    )
