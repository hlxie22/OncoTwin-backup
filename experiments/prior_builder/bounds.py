"""Layer 1 biologic bounds and observation-noise policy for V1 priors.

The first V1-A prior stack stays deliberately conservative: bounds are used to
reject numerically unstable particles and surface high-but-allowed values for
review instead of silently clipping samples. Observation noise is resolved from
measurement source and quality-control metadata so low-quality observations
widen updates rather than pretending to be precise.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Mapping, Sequence, Tuple

from .parameter_contract import LEARNABLE_PARAMETER_NAMES


@dataclass(frozen=True)
class ParameterBounds:
    """Biologic and numerical guardrails for one simulator parameter."""

    normal_min: float
    normal_max: float
    warning_high: float
    hard_min: float
    hard_max: float

    def validate(self, parameter_name: str, value: float) -> Tuple[str, ...]:
        """Validate a value and return non-fatal warning messages."""

        numeric = _require_finite(value, parameter_name)
        if numeric < self.hard_min:
            raise ValueError(
                f"{parameter_name}={numeric:g} is below hard minimum "
                f"{self.hard_min:g}"
            )
        if numeric > self.hard_max:
            raise ValueError(
                f"{parameter_name}={numeric:g} exceeds hard stop "
                f"{self.hard_max:g}"
            )

        if self.normal_min <= numeric <= self.normal_max:
            return ()
        if numeric > self.warning_high:
            return (
                f"{parameter_name}={numeric:g} exceeds high-warning threshold "
                f"{self.warning_high:g}; verify trajectory stability.",
            )
        return (
            f"{parameter_name}={numeric:g} is outside V1-A normal range "
            f"{self.normal_min:g}-{self.normal_max:g}.",
        )


@dataclass(frozen=True)
class BoundsValidationResult:
    """Validated simulator-facing parameters plus traceable warnings."""

    parameters: Dict[str, float]
    warnings: Tuple[str, ...]


@dataclass(frozen=True)
class ObservationNoise:
    """Resolved log-scale observation-noise setting for one measurement."""

    measurement_kind: str
    log_scale_noise: float
    warnings: Tuple[str, ...] = ()


PARAMETER_BOUNDS: Mapping[str, ParameterBounds] = {
    "growth_rate_per_day": ParameterBounds(
        normal_min=0.0005,
        normal_max=0.020,
        warning_high=0.030,
        hard_min=0.0,
        hard_max=0.100,
    ),
    "active_treatment_sensitivity": ParameterBounds(
        normal_min=0.015,
        normal_max=0.200,
        warning_high=0.300,
        hard_min=0.0,
        hard_max=0.500,
    ),
    "resistant_fraction": ParameterBounds(
        normal_min=0.0,
        normal_max=0.65,
        warning_high=0.75,
        hard_min=0.0,
        hard_max=0.90,
    ),
}

OBSERVATION_NOISE_BY_KIND: Mapping[str, float] = {
    "high_qc_mri_volume": 0.08,
    "medium_qc_mri_volume": 0.12,
    "low_qc_mri_volume": 0.20,
    "manual_volume": 0.25,
    "diameter_derived_volume": 0.35,
}

_SOURCE_FIELDS = (
    "source",
    "measurement_source",
    "source_type",
    "measurement_type",
    "modality",
)
_QC_FIELDS = (
    "confidence",
    "qc",
    "qc_quality",
    "quality",
    "segmentation_confidence",
)
_DIAMETER_TOKENS = ("diameter", "longest diameter", "ellipsoid")
_MANUAL_TOKENS = ("manual", "entered", "user entered", "clinician", "caliper")
_MRI_TOKENS = ("mri", "dce", "imaging", "segmentation", "mask", "ftv")
_HIGH_QC_TOKENS = ("high", "good", "excellent")
_LOW_QC_TOKENS = (
    "low",
    "poor",
    "uncertain",
    "motion",
    "artifact",
    "manual review",
)


def validate_parameter_bounds(
    parameters: Mapping[str, float],
    *,
    required_parameter_names: Sequence[str] = LEARNABLE_PARAMETER_NAMES,
) -> BoundsValidationResult:
    """Validate V1-A learnable parameters against Layer 1 bounds.

    The default requires a complete V1-A learnable particle. Callers that need
    to validate a deliberate subset can pass ``required_parameter_names``.
    """

    unexpected = sorted(set(parameters) - set(PARAMETER_BOUNDS))
    if unexpected:
        raise ValueError(
            "No Layer 1 bounds are configured for: " + ", ".join(unexpected)
        )

    missing = [name for name in required_parameter_names if name not in parameters]
    if missing:
        raise KeyError(f"Missing bounded parameters: {', '.join(missing)}")

    validated: Dict[str, float] = {}
    warnings = []
    for name in required_parameter_names:
        bounds = PARAMETER_BOUNDS[name]
        numeric = _require_finite(parameters[name], name)
        warnings.extend(bounds.validate(name, numeric))
        validated[name] = numeric

    return BoundsValidationResult(parameters=validated, warnings=tuple(warnings))


def resolve_observation_noise(measurement: Mapping[str, object]) -> ObservationNoise:
    """Resolve the V1-A log-scale noise policy for a tumor-volume observation."""

    source_text = _combined_normalized_text(measurement, _SOURCE_FIELDS)
    qc_text = _combined_normalized_text(measurement, _QC_FIELDS)
    qc_flags_text = _normalize_text(" ".join(_as_text_list(measurement.get("qc_flags"))))
    combined_qc_text = " ".join(text for text in (qc_text, qc_flags_text) if text)

    if _is_diameter_derived(measurement, source_text):
        return _noise("diameter_derived_volume")
    if _contains_any(source_text, _MANUAL_TOKENS):
        return _noise("manual_volume")
    if _contains_any(source_text, _MRI_TOKENS):
        return _noise(_mri_noise_kind(measurement, combined_qc_text))

    return _noise(
        "manual_volume",
        warnings=(
            "Measurement source is missing or unknown; using conservative "
            "manual-volume observation noise.",
        ),
    )


def _noise(
    measurement_kind: str,
    *,
    warnings: Tuple[str, ...] = (),
) -> ObservationNoise:
    return ObservationNoise(
        measurement_kind=measurement_kind,
        log_scale_noise=OBSERVATION_NOISE_BY_KIND[measurement_kind],
        warnings=warnings,
    )


def _mri_noise_kind(measurement: Mapping[str, object], qc_text: str) -> str:
    numeric_confidence = _numeric_confidence(measurement.get("segmentation_confidence"))
    if numeric_confidence is not None:
        if numeric_confidence >= 0.80:
            return "high_qc_mri_volume"
        if numeric_confidence < 0.50:
            return "low_qc_mri_volume"

    if _contains_any(qc_text, _LOW_QC_TOKENS):
        return "low_qc_mri_volume"
    if _contains_any(qc_text, _HIGH_QC_TOKENS):
        return "high_qc_mri_volume"
    return "medium_qc_mri_volume"


def _is_diameter_derived(measurement: Mapping[str, object], source_text: str) -> bool:
    if _is_truthy(measurement.get("derived_from_diameter")):
        return True
    if _contains_any(source_text, _DIAMETER_TOKENS):
        return True
    return "longest_diameter_cm" in measurement and "tumor_volume_ml" not in measurement


def _require_finite(value: float, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _combined_normalized_text(
    context: Mapping[str, object], fields: Tuple[str, ...]
) -> str:
    values = []
    for field in fields:
        value = context.get(field)
        if value is not None:
            values.extend(_as_text_list(value))
    return _normalize_text(" ".join(values))


def _as_text_list(value: object) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item) for item in value)
    return (str(value),)


def _normalize_text(value: str) -> str:
    return " ".join(value.lower().replace("_", " ").replace("-", " ").split())


def _contains_any(value: str, tokens: Tuple[str, ...]) -> bool:
    return any(token in value for token in tokens)


def _numeric_confidence(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if 0.0 <= numeric <= 1.0:
        return numeric
    return None


def _is_truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return _normalize_text(str(value)) in {"1", "true", "yes", "y"}
