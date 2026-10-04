from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from .models import CommittedFact, PatientStateSnapshot
from .state import audit

MODEL_URL = os.environ.get("ONCOTWIN_MODEL_URL", "http://127.0.0.1:8766").rstrip("/")
ADAPTER_VERSION = "app_cp2_r1_temporal_v1"
RESEARCH_STATUS = "EXTERNAL_CONFIRMATION_PENDING"


def _json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha(obj: Any) -> str:
    return hashlib.sha256(_json(obj).encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _id() -> str:
    return str(uuid.uuid4())


def model_health(timeout: float = 5.0) -> dict[str, Any]:
    try:
        with urlopen(f"{MODEL_URL}/healthz", timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return {"ok": False, "error": str(exc), "url": MODEL_URL}


def _call_model(payload: dict[str, Any], timeout: float = 180.0) -> dict[str, Any]:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = Request(
        f"{MODEL_URL}/v1/predict",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Frozen model service HTTP {exc.code}: {detail[:1200]}") from exc
    except URLError as exc:
        raise RuntimeError(f"Frozen model service unavailable at {MODEL_URL}: {exc}") from exc
    if "error" in result:
        raise RuntimeError(f"Frozen model service error: {result['error']}")
    return result


def _latest_state(db: Session, patient_id: str) -> PatientStateSnapshot | None:
    return db.scalar(
        select(PatientStateSnapshot)
        .where(PatientStateSnapshot.patient_id == patient_id)
        .order_by(PatientStateSnapshot.created_at.desc())
        .limit(1)
    )


def _fact_payload(f: CommittedFact) -> dict[str, Any]:
    value_json = dict(f.value_json or {})
    return {
        "id": f.id,
        "fact_type": f.fact_type,
        "value": value_json.get("value"),
        "value_json": value_json,
        "attributes": value_json.get("attributes") if isinstance(value_json.get("attributes"), dict) else {},
        "effective_date": f.effective_date,
        "verification_state": f.verification_state,
        "confidence": f.confidence,
        "status": f.status,
    }


def build_model_request(db: Session, patient_id: str) -> tuple[PatientStateSnapshot, dict[str, Any]]:
    state = _latest_state(db, patient_id)
    if state is None:
        raise ValueError("Patient state has not been built yet")
    facts = list(
        db.scalars(
            select(CommittedFact)
            .where(CommittedFact.patient_id == patient_id, CommittedFact.status == "active")
            .order_by(CommittedFact.effective_date.asc().nullsfirst(), CommittedFact.created_at.asc())
        )
    )
    request = {
        "patient_id": patient_id,
        "state_snapshot_id": state.id,
        "state_hash": state.state_hash,
        "fact_revision_ids": list(state.fact_revision_ids_json or []),
        "facts": [_fact_payload(f) for f in facts],
    }
    return state, request


def _existing_run(db: Session, patient_id: str, state_hash: str, request_hash: str) -> dict[str, Any] | None:
    row = db.execute(
        text(
            """
            SELECT fr.id
            FROM forecast_runs fr
            JOIN model_input_snapshots mi ON mi.id = fr.model_input_snapshot_id
            WHERE fr.patient_id = :patient_id
              AND fr.state_hash = :state_hash
              AND mi.request_hash = :request_hash
              AND fr.run_status IN ('COMPLETED','UNAVAILABLE')
            ORDER BY fr.created_at DESC
            LIMIT 1
            """
        ),
        {"patient_id": patient_id, "state_hash": state_hash, "request_hash": request_hash},
    ).mappings().first()
    return get_forecast_run(db, row["id"]) if row else None


def create_forecast(db: Session, patient_id: str, user_id: str | None) -> dict[str, Any]:
    state, request_payload = build_model_request(db, patient_id)
    request_hash = _sha(request_payload)
    existing = _existing_run(db, patient_id, state.state_hash, request_hash)
    if existing is not None:
        existing["reused"] = True
        return existing

    model_result = _call_model(request_payload)
    now = _now()
    support_id = _id()
    input_id = _id()
    run_id = _id()

    supported = bool(model_result.get("supported"))
    support_status = str(model_result.get("support_status") or ("STANDARD" if supported else "UNAVAILABLE"))
    reasons = list(model_result.get("support_reasons") or [])
    explanations = list(model_result.get("support_explanations") or [])

    db.execute(
        text(
            """
            INSERT INTO support_assessments
              (id, patient_id, status, reason_codes_json, explanations_json, created_at)
            VALUES
              (:id, :patient_id, :status, :reasons, :explanations, :created_at)
            """
        ),
        {
            "id": support_id,
            "patient_id": patient_id,
            "status": support_status,
            "reasons": _json(reasons),
            "explanations": _json(explanations),
            "created_at": now,
        },
    )

    current_scan_fact_id = model_result.get("current_scan_fact_id")
    input_metadata = {
        "fact_revision_ids": request_payload["fact_revision_ids"],
        "fact_count": len(request_payload["facts"]),
        "current_scan_fact_id": current_scan_fact_id,
        "current_scan_date": model_result.get("current_scan_date"),
        "anchor_date": model_result.get("anchor_date"),
        "anchor_fact_id": model_result.get("anchor_fact_id"),
        "portable_context": model_result.get("portable_context"),
        "adapter_version": model_result.get("adapter_version", ADAPTER_VERSION),
    }
    db.execute(
        text(
            """
            INSERT INTO model_input_snapshots
              (id, patient_id, state_snapshot_id, state_hash, current_scan_fact_id,
               adapter_version, request_hash, input_metadata_json, prepared_hashes_json, created_at)
            VALUES
              (:id, :patient_id, :state_snapshot_id, :state_hash, :current_scan_fact_id,
               :adapter_version, :request_hash, :input_metadata_json, :prepared_hashes_json, :created_at)
            """
        ),
        {
            "id": input_id,
            "patient_id": patient_id,
            "state_snapshot_id": state.id,
            "state_hash": state.state_hash,
            "current_scan_fact_id": current_scan_fact_id,
            "adapter_version": str(model_result.get("adapter_version") or ADAPTER_VERSION),
            "request_hash": request_hash,
            "input_metadata_json": _json(input_metadata),
            "prepared_hashes_json": _json(model_result.get("prepared_hashes") or {}),
            "created_at": now,
        },
    )

    run_status = "COMPLETED" if supported else "UNAVAILABLE"
    db.execute(
        text(
            """
            INSERT INTO forecast_runs
              (id, patient_id, state_snapshot_id, state_hash, model_input_snapshot_id,
               support_assessment_id, current_scan_fact_id, run_status, checkpoint,
               selected_rule, locked_alpha, deploy_sha256, temporal_encoder_sha256,
               research_status, pre_logits_json, post_logits_json, selected_logits_json,
               curves_json, horizons_json, error_message, created_at)
            VALUES
              (:id, :patient_id, :state_snapshot_id, :state_hash, :model_input_snapshot_id,
               :support_assessment_id, :current_scan_fact_id, :run_status, :checkpoint,
               :selected_rule, :locked_alpha, :deploy_sha256, :temporal_encoder_sha256,
               :research_status, :pre_logits_json, :post_logits_json, :selected_logits_json,
               :curves_json, :horizons_json, NULL, :created_at)
            """
        ),
        {
            "id": run_id,
            "patient_id": patient_id,
            "state_snapshot_id": state.id,
            "state_hash": state.state_hash,
            "model_input_snapshot_id": input_id,
            "support_assessment_id": support_id,
            "current_scan_fact_id": current_scan_fact_id,
            "run_status": run_status,
            "checkpoint": str(model_result.get("checkpoint") or "V2-07"),
            "selected_rule": "v2_03_frozen_temporal_constant_alpha",
            "locked_alpha": float(model_result.get("locked_alpha") or 0.52),
            "deploy_sha256": str(model_result.get("deploy_sha256") or ""),
            "temporal_encoder_sha256": str(model_result.get("temporal_encoder_sha256") or ""),
            "research_status": RESEARCH_STATUS,
            "pre_logits_json": _json(model_result.get("pre_logits")) if supported else None,
            "post_logits_json": _json(model_result.get("post_logits")) if supported else None,
            "selected_logits_json": _json(model_result.get("selected_logits")) if supported else None,
            "curves_json": _json(model_result.get("curves")) if supported else None,
            "horizons_json": _json(model_result.get("horizons")) if supported else None,
            "created_at": now,
        },
    )

    audit(
        db,
        user_id=user_id,
        patient_id=patient_id,
        action="forecast.create",
        object_type="forecast_run",
        object_id=run_id,
        metadata={
            "state_hash": state.state_hash,
            "support_status": support_status,
            "checkpoint": "V2-07",
            "research_status": RESEARCH_STATUS,
        },
    )
    db.flush()
    result = get_forecast_run(db, run_id)
    if result is None:
        raise RuntimeError("Forecast run persistence failed")
    result["reused"] = False
    return result


def _loads(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return default


def get_forecast_run(db: Session, run_id: str) -> dict[str, Any] | None:
    row = db.execute(
        text(
            """
            SELECT
              fr.*, sa.status AS support_status,
              sa.reason_codes_json, sa.explanations_json,
              mi.adapter_version, mi.request_hash, mi.input_metadata_json, mi.prepared_hashes_json
            FROM forecast_runs fr
            JOIN support_assessments sa ON sa.id = fr.support_assessment_id
            JOIN model_input_snapshots mi ON mi.id = fr.model_input_snapshot_id
            WHERE fr.id = :run_id
            LIMIT 1
            """
        ),
        {"run_id": run_id},
    ).mappings().first()
    if row is None:
        return None
    curves = _loads(row["curves_json"], None)
    horizons = _loads(row["horizons_json"], None)
    input_meta = _loads(row["input_metadata_json"], {})
    return {
        "id": row["id"],
        "patient_id": row["patient_id"],
        "state_hash": row["state_hash"],
        "run_status": row["run_status"],
        "support": {
            "status": row["support_status"],
            "reason_codes": _loads(row["reason_codes_json"], []),
            "explanations": _loads(row["explanations_json"], []),
        },
        "research_status": row["research_status"],
        "model": {
            "checkpoint": row["checkpoint"],
            "selected_rule": row["selected_rule"],
            "locked_alpha": float(row["locked_alpha"]),
            "deploy_sha256": row["deploy_sha256"],
            "temporal_encoder_sha256": row["temporal_encoder_sha256"],
            "adapter_version": row["adapter_version"],
        },
        "current_scan": {
            "fact_id": row["current_scan_fact_id"],
            "date": input_meta.get("current_scan_date"),
        },
        "curves": curves,
        "horizons": horizons,
        "prepared_hashes": _loads(row["prepared_hashes_json"], {}),
        "created_at": row["created_at"],
    }


def latest_forecast(db: Session, patient_id: str) -> dict[str, Any] | None:
    row = db.execute(
        text(
            """
            SELECT id FROM forecast_runs
            WHERE patient_id = :patient_id
            ORDER BY created_at DESC
            LIMIT 1
            """
        ),
        {"patient_id": patient_id},
    ).mappings().first()
    return get_forecast_run(db, row["id"]) if row else None
