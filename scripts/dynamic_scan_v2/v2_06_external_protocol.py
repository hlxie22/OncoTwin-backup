from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

OUTCOME_TOKENS = {
    "outcome", "survival", "progression", "death", "censor", "event_time",
    "event_day", "cause", "pfs", "os_time", "endpoint"
}
OLD_EXTERNAL_LABELS = {"dfci", "vicc", "dfci_vicc", "legacy_dfci", "legacy_vicc"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_patient_token(namespace: str, patient_id: str) -> str:
    payload = f"{namespace.strip()}::{str(patient_id).strip()}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_stage_order(stages: list[str]) -> None:
    expected = [
        "A_OUTCOME_BLIND_COHORT_INTAKE_AND_OVERLAP",
        "B_OUTCOME_BLIND_PREDICTION_FREEZE",
        "C_ENDPOINT_REVEAL_AND_ONE_SHOT_PRIMARY_EVALUATION",
        "D_OPTIONAL_SEPARATE_RECALIBRATION_ANALYSIS",
    ]
    require(stages == expected, f"External stage order drifted: {stages}")


def looks_like_outcome_field(name: str) -> bool:
    x = name.strip().lower()
    return any(tok in x for tok in OUTCOME_TOKENS)


def validate_predictor_inventory(columns: Iterable[str]) -> dict[str, Any]:
    cols = [str(x).strip() for x in columns]
    forbidden = sorted([x for x in cols if looks_like_outcome_field(x)])
    require(not forbidden, f"Outcome-like fields present in outcome-blind predictor inventory: {forbidden}")
    return {"status": "PASS", "columns": cols, "outcome_like_fields": forbidden}


def validate_new_cohort_label(label: str) -> None:
    normalized = str(label).strip().lower().replace("-", "_").replace("/", "_")
    require(normalized not in OLD_EXTERNAL_LABELS, f"Prior external cohort cannot be relabeled as new evidence: {label}")


def overlap_report(new_tokens: set[str], upstream_tokens: set[str]) -> dict[str, Any]:
    overlap = sorted(new_tokens & upstream_tokens)
    return {
        "status": "PASS" if not overlap else "FAIL",
        "new_patient_tokens": len(new_tokens),
        "upstream_patient_tokens": len(upstream_tokens),
        "overlap_count": len(overlap),
        "overlap_tokens": overlap[:100],
    }


def predictor_crosswalk_template() -> dict[str, Any]:
    return {
        "version": "v2_06_1",
        "model_candidate": "v2_03_frozen_temporal_constant_alpha",
        "required": [
            {
                "concept": "patient_identity",
                "source_field": None,
                "availability_clock": "identity linkage only; never model-facing",
                "notes": "Required for overlap checks and patient-level evaluation; excluded from model inputs."
            },
            {
                "concept": "landmark_day",
                "source_field": None,
                "availability_clock": "known at prediction time",
                "notes": "Defines time origin; treatment-line membership is not used."
            },
            {
                "concept": "historical_scan_prefix",
                "source_field": None,
                "availability_clock": "strictly before current scan episode start",
                "notes": "Must support repaired line-agnostic R1 temporal tokenization."
            },
            {
                "concept": "current_scan_three_state_assessment",
                "source_field": None,
                "availability_clock": "current episode only",
                "allowed_values": ["NON_PROGRESSIVE", "INDETERMINATE", "PROGRESSIVE"],
                "notes": "No retrospective future adjudication."
            },
            {
                "concept": "current_scan_anatomic_coverage",
                "source_field": None,
                "availability_clock": "current episode only",
                "allowed_regions": ["chest", "abdomen", "pelvis", "head", "other"],
                "notes": "Coverage is not disease positivity."
            },
            {
                "concept": "genomic_embedding_128d",
                "source_field": None,
                "availability_clock": "sample availability strictly before landmark",
                "notes": "Derived using frozen checkpoint3 encoder/gene-space contract when source data permit."
            },
            {
                "concept": "genomic_available",
                "source_field": None,
                "availability_clock": "strictly before landmark",
                "allowed_values": [0, 1],
                "notes": "If 0, embedding contribution is zero and genomic_age_scaled is 0."
            },
            {
                "concept": "genomic_age_scaled",
                "source_field": None,
                "availability_clock": "strictly before landmark",
                "notes": "clip(age_days,0,3650)/365 if genomic_available=1; else 0."
            },
            {
                "concept": "scan_number_patient_log",
                "source_field": None,
                "availability_clock": "count of scans known by landmark",
                "notes": "log1p(scan_number_patient)/5; line-relative counts are prohibited."
            }
        ],
        "optional_or_provenance_only": [
            "center_id", "source_system", "component_id", "modality", "positive_site_mentions"
        ],
        "prohibited_model_facing": [
            "treatment_line_number", "elapsed_on_line", "scan_number_within_line", "future events", "outcomes"
        ],
        "status": "FROZEN_TEMPLATE"
    }


def endpoint_contract() -> dict[str, Any]:
    return {
        "version": "v2_06_1",
        "reveal_stage": "C_ENDPOINT_REVEAL_AND_ONE_SHOT_PRIMARY_EVALUATION",
        "time_origin": "current scan landmark day",
        "causes": {
            "0": "censored/no observed competing event by follow-up",
            "1": "progression",
            "2": "death",
            "3": "treatment switch as distinct competing event"
        },
        "horizons_months": [3, 6, 12, 18],
        "month_width_days": 30.4375,
        "primary_target": "PFS-style cumulative progression/death risk under the locked competing-risk contract",
        "switch_policy": "switch remains its own competing event; do not silently convert to progression",
        "partial_censor_policy": "fractional-censor NLL V1 for supporting likelihood; primary Brier uses prespecified censoring nuisance policy",
        "status": "FROZEN_BEFORE_NEW_OUTCOMES"
    }


def nuisance_policy() -> dict[str, Any]:
    return {
        "version": "v2_06_1",
        "primary_brier": {
            "weighting": "patient-balanced",
            "censoring": "evaluation-only censoring nuisance estimation prespecified before outcomes are inspected",
            "horizons_months": [3, 6, 12, 18],
            "aggregation": "mean across four horizons"
        },
        "supporting_nll": {
            "name": "FRACTIONAL_CENSOR_NLL_V1",
            "weighting": "patient-balanced",
            "selection_role": "supporting only; does not replace locked primary metric"
        },
        "recalibration": {
            "primary_evaluation": "none",
            "secondary": "allowed only as separately labeled analysis with calibration/evaluation separation",
            "may_replace_primary": False
        },
        "status": "FROZEN_BEFORE_NEW_OUTCOMES"
    }


def abstention_policy() -> dict[str, Any]:
    return {
        "version": "v2_06_1",
        "hard_abstain": [
            "unknown or invalid prediction landmark clock",
            "cannot construct verified historical prefix",
            "unknown current three-state assessment",
            "patient identity/overlap status unresolved for a claimed independent cohort",
            "predictor crosswalk violates source availability timing"
        ],
        "not_a_reason_to_invent_normal": [
            "missing genomics", "missing optional modality", "missing optional positive-site stream"
        ],
        "genomics_missing_policy": {
            "genomic_available": 0,
            "genomic_age_scaled": 0.0,
            "embedding_contribution": "zero"
        },
        "quality_flags_are_not_calibrated_uncertainty": True,
        "status": "FROZEN_BEFORE_NEW_OUTCOMES"
    }


def cohort_manifest_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "OncoTwin V2-06 new independent cohort intake manifest",
        "type": "object",
        "required": [
            "cohort_id", "cohort_label", "centers", "identity_namespace",
            "patient_identity_file", "predictor_inventory_file", "outcomes_unavailable_to_prediction_team"
        ],
        "properties": {
            "cohort_id": {"type": "string"},
            "cohort_label": {"type": "string"},
            "centers": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "identity_namespace": {"type": "string"},
            "patient_identity_file": {"type": "string"},
            "predictor_inventory_file": {"type": "string"},
            "outcomes_unavailable_to_prediction_team": {"const": True},
            "notes": {"type": "string"}
        },
        "additionalProperties": True
    }


def evaluation_stage_contract() -> dict[str, Any]:
    return {
        "version": "v2_06_1",
        "stages": [
            {
                "stage": "A_OUTCOME_BLIND_COHORT_INTAKE_AND_OVERLAP",
                "may_access": ["identity linkage", "predictor metadata", "source timestamps", "predictor values"],
                "must_not_access": ["outcomes", "event times", "censoring", "future clinical summaries"],
                "outputs": ["overlap report", "predictor crosswalk", "eligibility counts", "missingness profile", "abstention counts"]
            },
            {
                "stage": "B_OUTCOME_BLIND_PREDICTION_FREEZE",
                "may_access": ["eligible predictor inputs", "frozen model bundle"],
                "must_not_access": ["outcomes", "event times", "censoring"],
                "outputs": ["immutable PRE logits", "immutable POST logits", "locked-alpha logits", "prediction SHA256 manifest"]
            },
            {
                "stage": "C_ENDPOINT_REVEAL_AND_ONE_SHOT_PRIMARY_EVALUATION",
                "may_access": ["frozen predictions", "endpoint file", "prespecified nuisance estimation"],
                "must_not_change": ["candidate", "alpha", "predictor mapping", "eligibility", "primary missingness profile", "horizons", "primary metric"],
                "outputs": ["separate-center primary results", "coverage/abstention report", "calibration", "discrimination", "PRE-to-POST changes"]
            },
            {
                "stage": "D_OPTIONAL_SEPARATE_RECALIBRATION_ANALYSIS",
                "may_run": "only after immutable primary results exist",
                "rule": "must use a separate calibration/evaluation split or independent calibration data and cannot replace primary results"
            }
        ]
    }


def validate_candidate_contract(v205: dict[str, Any], v203: dict[str, Any], v202: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    require(v205.get("status") == "PASS", "V2-05 must be PASS")
    lock = v205["candidate_lock"]
    require(lock["selected"] == cfg["candidate"]["selected_rule"], "V2-05 selected rule drift")
    require(lock["v2_04_candidate_retained"] is False, "Unexpected V2-04 candidate retention")
    require(abs(float(v205["locked_alpha"]) - float(cfg["candidate"]["locked_alpha"])) < 1e-12, "V2-05 alpha drift")
    require(v203.get("status") == "PASS", "V2-03 must be PASS")
    require(v203.get("branch_family") == cfg["candidate"]["branch_family"], "V2-03 branch family drift")
    require(v202.get("status") == "PASS", "V2-02 must be PASS")
    require(v202["selection"]["best_screening_family"] == cfg["candidate"]["branch_family"], "V2-02 selected family drift")
    return {
        "status": "PASS",
        "selected_rule": lock["selected"],
        "branch_family": v203["branch_family"],
        "locked_alpha": float(v205["locked_alpha"]),
        "v2_04_candidate_retained": bool(lock["v2_04_candidate_retained"]),
    }


def validate_outcome_blind_manifest(manifest: dict[str, Any], predictor_columns: Iterable[str]) -> dict[str, Any]:
    required = [
        "cohort_id", "cohort_label", "centers", "identity_namespace",
        "patient_identity_file", "predictor_inventory_file", "outcomes_unavailable_to_prediction_team"
    ]
    missing = [x for x in required if x not in manifest]
    require(not missing, f"Missing manifest fields: {missing}")
    validate_new_cohort_label(manifest["cohort_label"])
    require(manifest["outcomes_unavailable_to_prediction_team"] is True,
            "Stage A requires outcomes to be unavailable to the prediction team")
    inv = validate_predictor_inventory(predictor_columns)
    return {
        "status": "PASS",
        "cohort_id": manifest["cohort_id"],
        "centers": list(manifest["centers"]),
        "predictor_inventory": inv,
    }
