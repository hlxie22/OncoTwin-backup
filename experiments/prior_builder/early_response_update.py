"""Auditable post-treatment early MRI response update.

This module applies a versioned, bounded log-linear correction to an existing
point prediction. It does not construct or modify uncertainty intervals.

Post-hoc candidate artifacts require explicit opt-in and must not be described
as externally validated models.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path


EARLY_RESPONSE_UPDATE_VERSION = (
    "oncotwin_early_response_update_v1"
)
EARLY_RESPONSE_METHOD = (
    "bounded_log_linear_early_volume_residual_update"
)
CANDIDATE_STATUS = "candidate_posthoc"
MISSING_EARLY_POLICY = (
    "global_mean_static_log_residual"
)


@dataclass(frozen=True)
class EarlyResponseUpdater:
    """Validated early-volume point-update artifact."""

    artifact_version: str
    model_version: str
    status: str
    method: str
    source_prediction: str
    intercept: float
    slope: float
    missing_early_log_correction: float
    min_log_correction: float
    max_log_correction: float
    ridge_penalty: float
    source_path: str
    validation_note: str

    def audit_payload(self) -> dict[str, object]:
        """Return artifact-level provenance."""

        return {
            "artifact_version": self.artifact_version,
            "model_version": self.model_version,
            "status": self.status,
            "method": self.method,
            "source_prediction": self.source_prediction,
            "source_artifact": self.source_path,
            "point_prediction_changed": True,
            "interval_prediction_changed": False,
            "parameters": {
                "intercept": self.intercept,
                "slope": self.slope,
                "missing_early_log_correction": (
                    self.missing_early_log_correction
                ),
                "min_log_correction": (
                    self.min_log_correction
                ),
                "max_log_correction": (
                    self.max_log_correction
                ),
                "ridge_penalty": self.ridge_penalty,
            },
            "validation_note": self.validation_note,
        }


def load_early_response_updater(
    path: Path,
    *,
    allow_candidate: bool = False,
) -> EarlyResponseUpdater:
    """Load and validate a versioned early-response artifact."""

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8")
        )
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Early-response artifact is not valid JSON: {path}"
        ) from exc

    root = _require_mapping(
        payload,
        "early-response artifact",
    )

    artifact_version = _require_text(
        root.get("artifact_version"),
        "artifact_version",
    )

    if artifact_version != EARLY_RESPONSE_UPDATE_VERSION:
        raise ValueError(
            "Unsupported early-response artifact version: "
            f"{artifact_version!r}"
        )

    model_version = _require_text(
        root.get("model_version"),
        "model_version",
    )
    status = _require_text(
        root.get("status"),
        "status",
    )

    if status == CANDIDATE_STATUS and not allow_candidate:
        raise ValueError(
            "Post-hoc candidate early-response updater "
            "requires allow_candidate=True."
        )

    method_payload = _require_mapping(
        root.get("method"),
        "method",
    )
    method = _require_text(
        method_payload.get("name"),
        "method.name",
    )

    if method != EARLY_RESPONSE_METHOD:
        raise ValueError(
            f"Unsupported early-response method: {method!r}"
        )

    if method_payload.get("point_prediction_changed") is not True:
        raise ValueError(
            "Early-response update must change the point prediction."
        )

    if method_payload.get("interval_prediction_changed") is not False:
        raise ValueError(
            "The early-response artifact must not define intervals."
        )

    missing_policy = _require_text(
        method_payload.get("missing_early_policy"),
        "method.missing_early_policy",
    )

    if missing_policy != MISSING_EARLY_POLICY:
        raise ValueError(
            "Unsupported missing-early policy: "
            f"{missing_policy!r}"
        )

    training_source = _require_mapping(
        root.get("training_source"),
        "training_source",
    )
    source_prediction = _require_text(
        training_source.get("source_prediction"),
        "training_source.source_prediction",
    )

    parameters = _require_mapping(
        root.get("parameters"),
        "parameters",
    )

    intercept = _require_finite_number(
        parameters.get("intercept"),
        "parameters.intercept",
    )
    slope = _require_finite_number(
        parameters.get("slope"),
        "parameters.slope",
    )
    missing_correction = _require_finite_number(
        parameters.get(
            "missing_early_log_correction"
        ),
        "parameters.missing_early_log_correction",
    )
    minimum = _require_finite_number(
        parameters.get("min_log_correction"),
        "parameters.min_log_correction",
    )
    maximum = _require_finite_number(
        parameters.get("max_log_correction"),
        "parameters.max_log_correction",
    )
    ridge_penalty = _require_finite_number(
        parameters.get("ridge_penalty"),
        "parameters.ridge_penalty",
    )

    if minimum >= maximum:
        raise ValueError(
            "min_log_correction must be less than "
            "max_log_correction."
        )

    if ridge_penalty < 0.0:
        raise ValueError(
            "ridge_penalty must be nonnegative."
        )

    validation_note = _require_text(
        root.get("validation_note"),
        "validation_note",
    )

    return EarlyResponseUpdater(
        artifact_version=artifact_version,
        model_version=model_version,
        status=status,
        method=method,
        source_prediction=source_prediction,
        intercept=intercept,
        slope=slope,
        missing_early_log_correction=(
            missing_correction
        ),
        min_log_correction=minimum,
        max_log_correction=maximum,
        ridge_penalty=ridge_penalty,
        source_path=str(path),
        validation_note=validation_note,
    )


def apply_early_response_update(
    source_prediction: Mapping[str, object],
    case: Mapping[str, object],
    updater: EarlyResponseUpdater,
) -> dict[str, object]:
    """Apply the bounded point update without constructing intervals."""

    source_point_ml = _require_positive_number(
        source_prediction.get("point_ml"),
        "source_prediction.point_ml",
    )
    baseline_volume_ml = _require_positive_number(
        case.get("baseline_volume_ml"),
        "case.baseline_volume_ml",
    )
    early_volume_ml = _optional_positive_number(
        case.get("early_volume_ml"),
        "case.early_volume_ml",
    )

    if early_volume_ml is None:
        early_available = False
        early_to_baseline_ratio = None
        raw_log_correction = (
            updater.missing_early_log_correction
        )
        correction_source = "missing_early_fallback"
    else:
        early_available = True
        early_to_baseline_ratio = (
            early_volume_ml / baseline_volume_ml
        )
        raw_log_correction = (
            updater.intercept
            + updater.slope
            * math.log(early_to_baseline_ratio)
        )
        correction_source = "continuous_early_volume"

    applied_log_correction = min(
        max(
            raw_log_correction,
            updater.min_log_correction,
        ),
        updater.max_log_correction,
    )
    multiplicative_factor = math.exp(
        applied_log_correction
    )
    updated_point_ml = (
        source_point_ml * multiplicative_factor
    )

    return {
        "point_ml": updated_point_ml,
        "early_response_update": {
            **updater.audit_payload(),
            "source_point_ml": source_point_ml,
            "updated_point_ml": updated_point_ml,
            "baseline_volume_ml": baseline_volume_ml,
            "early_volume_ml": early_volume_ml,
            "early_available": early_available,
            "early_to_baseline_ratio": (
                early_to_baseline_ratio
            ),
            "correction_source": correction_source,
            "raw_log_correction": raw_log_correction,
            "applied_log_correction": (
                applied_log_correction
            ),
            "correction_clipped": not math.isclose(
                raw_log_correction,
                applied_log_correction,
                rel_tol=0.0,
                abs_tol=1e-15,
            ),
            "multiplicative_factor": (
                multiplicative_factor
            ),
        },
    }


def _require_mapping(
    value: object,
    name: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object.")

    return {
        str(key): item
        for key, item in value.items()
    }


def _require_text(
    value: object,
    name: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{name} must be a non-empty string."
        )

    return value.strip()


def _require_finite_number(
    value: object,
    name: str,
) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric.")

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be numeric."
        ) from exc

    if not math.isfinite(number):
        raise ValueError(
            f"{name} must be finite."
        )

    return number


def _require_positive_number(
    value: object,
    name: str,
) -> float:
    number = _require_finite_number(value, name)

    if number <= 0.0:
        raise ValueError(
            f"{name} must be positive."
        )

    return number


def _optional_positive_number(
    value: object,
    name: str,
) -> float | None:
    if value is None:
        return None

    if isinstance(value, str) and not value.strip():
        return None

    return _require_positive_number(value, name)
