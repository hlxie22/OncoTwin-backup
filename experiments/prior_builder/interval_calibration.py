"""Auditable interval calibration for OncoTwin prior predictions.

The raw mechanistic prior-predictive intervals remain unchanged. This module
creates a separate calibrated prediction from an existing point prediction
and a versioned calibration artifact.

Post-hoc candidate artifacts require explicit opt-in and must not be described
as externally validated calibration models.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Mapping


INTERVAL_CALIBRATOR_VERSION = "oncotwin_interval_calibrator_v1"
SYMMETRIC_LOG_RADIUS_METHOD = (
    "symmetric_absolute_log_residual_split_conformal"
)
CANDIDATE_STATUS = "candidate_posthoc"


@dataclass(frozen=True)
class IntervalCalibrator:
    """Validated symmetric log-radius interval calibrator."""

    artifact_version: str
    model_version: str
    status: str
    method: str
    log_radius_80: float
    log_radius_95: float
    source_path: str
    validation_note: str

    def audit_payload(
        self,
        *,
        source_prediction: str,
    ) -> dict[str, object]:
        """Return prediction-level calibration provenance."""

        return {
            "artifact_version": self.artifact_version,
            "model_version": self.model_version,
            "status": self.status,
            "method": self.method,
            "source_prediction": source_prediction,
            "source_artifact": self.source_path,
            "point_prediction_changed": False,
            "parameters": {
                "80": {
                    "log_radius": self.log_radius_80,
                    "multiplicative_factor": math.exp(
                        self.log_radius_80
                    ),
                },
                "95": {
                    "log_radius": self.log_radius_95,
                    "multiplicative_factor": math.exp(
                        self.log_radius_95
                    ),
                },
            },
            "validation_note": self.validation_note,
        }


def load_interval_calibrator(
    path: Path,
    *,
    allow_candidate: bool = False,
) -> IntervalCalibrator:
    """Load and validate a versioned interval-calibration artifact.

    A ``candidate_posthoc`` artifact is rejected unless ``allow_candidate`` is
    explicitly true. This prevents accidental treatment of a selected-on-D1
    artifact as a validated production calibrator.
    """

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8")
        )
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Interval calibrator is not valid JSON: {path}"
        ) from exc

    root = _require_mapping(
        payload,
        "interval calibrator",
    )

    artifact_version = _require_text(
        root.get("artifact_version"),
        "artifact_version",
    )

    if artifact_version != INTERVAL_CALIBRATOR_VERSION:
        raise ValueError(
            "Unsupported interval-calibrator artifact version: "
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
            "Post-hoc candidate interval calibrator requires "
            "allow_candidate=True."
        )

    method_payload = _require_mapping(
        root.get("method"),
        "method",
    )
    method = _require_text(
        method_payload.get("name"),
        "method.name",
    )

    if method != SYMMETRIC_LOG_RADIUS_METHOD:
        raise ValueError(
            f"Unsupported interval-calibration method: {method!r}"
        )

    if method_payload.get("point_prediction_changed") is not False:
        raise ValueError(
            "V1 interval calibration must not change the point prediction."
        )

    parameters = _require_mapping(
        root.get("parameters"),
        "parameters",
    )

    log_radius_80 = _read_radius(
        parameters,
        level="80",
        expected_coverage=0.80,
    )
    log_radius_95 = _read_radius(
        parameters,
        level="95",
        expected_coverage=0.95,
    )

    if log_radius_95 < log_radius_80:
        raise ValueError(
            "95% log radius must be at least as large as "
            "the 80% log radius."
        )

    validation_note = _require_text(
        root.get("validation_note"),
        "validation_note",
    )

    return IntervalCalibrator(
        artifact_version=artifact_version,
        model_version=model_version,
        status=status,
        method=method,
        log_radius_80=log_radius_80,
        log_radius_95=log_radius_95,
        source_path=str(path),
        validation_note=validation_note,
    )


def apply_interval_calibrator(
    prediction: Mapping[str, object],
    calibrator: IntervalCalibrator,
    *,
    source_prediction: str,
) -> dict[str, object]:
    """Create a separately calibrated prediction.

    The input mapping is never modified. The calibrated intervals are
    symmetric in log-volume space around the original point prediction.
    """

    point_ml = _require_positive_number(
        prediction.get("point_ml"),
        "prediction.point_ml",
    )

    lower_80_ml, upper_80_ml = _symmetric_interval(
        point_ml,
        calibrator.log_radius_80,
    )
    lower_95_ml, upper_95_ml = _symmetric_interval(
        point_ml,
        calibrator.log_radius_95,
    )

    return {
        "point_ml": point_ml,
        "lower_80_ml": lower_80_ml,
        "upper_80_ml": upper_80_ml,
        "lower_95_ml": lower_95_ml,
        "upper_95_ml": upper_95_ml,
        "calibration": calibrator.audit_payload(
            source_prediction=source_prediction,
        ),
    }


def _symmetric_interval(
    point_ml: float,
    log_radius: float,
) -> tuple[float, float]:
    factor = math.exp(log_radius)

    return (
        point_ml / factor,
        point_ml * factor,
    )


def _read_radius(
    parameters: Mapping[str, object],
    *,
    level: str,
    expected_coverage: float,
) -> float:
    level_payload = _require_mapping(
        parameters.get(level),
        f"parameters.{level}",
    )

    nominal_coverage = _require_finite_number(
        level_payload.get("nominal_coverage"),
        f"parameters.{level}.nominal_coverage",
    )

    if not math.isclose(
        nominal_coverage,
        expected_coverage,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"parameters.{level}.nominal_coverage must be "
            f"{expected_coverage}."
        )

    radius = _require_positive_number(
        level_payload.get("log_radius"),
        f"parameters.{level}.log_radius",
    )

    expected_factor = math.exp(radius)
    recorded_factor = _require_positive_number(
        level_payload.get("multiplicative_factor"),
        f"parameters.{level}.multiplicative_factor",
    )

    if not math.isclose(
        recorded_factor,
        expected_factor,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"parameters.{level}.multiplicative_factor "
            "does not match exp(log_radius)."
        )

    return radius


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
