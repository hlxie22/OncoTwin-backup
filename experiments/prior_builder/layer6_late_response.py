"""Layer 6 AI-inferred late-response digital-twin mechanics.

Layer 6 activates only when a valid early tumor-volume observation is
available before the requested final prediction day.

The neural or linear inference model predicts a distribution over the
patient-specific late log-volume response rate. This module provides the
stable feature contract and mechanistic decoder shared by all inference
models.
"""

from __future__ import annotations

from math import exp, isfinite, log
from statistics import NormalDist
from typing import Mapping, Sequence


LAYER6_DECODER_VERSION = (
    "oncotwin_layer6_late_response_decoder_v0_1"
)

EPSILON_VOLUME_ML = 1e-6
EARLY_RATE_REFERENCE_DAYS = 42.0
FINAL_DAY_REFERENCE_DAYS = 126.0
REMAINING_HORIZON_REFERENCE_DAYS = 84.0

LAYER6_FEATURE_NAMES = (
    "log_baseline_volume",
    "static_layer4_log_change",
    "static_layer4_log_width_80",
    "final_day_scaled",
    "early_day_scaled",
    "remaining_horizon_scaled",
    "observed_early_log_rate_42d",
    "mammaprint_value",
    "mammaprint_missing",
)


def _finite_number(
    value: object,
    name: str,
) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{name} must be a finite number"
        ) from error

    if not isfinite(numeric):
        raise ValueError(
            f"{name} must be a finite number"
        )

    return numeric


def _positive_number(
    value: object,
    name: str,
) -> float:
    numeric = _finite_number(value, name)

    if numeric <= 0:
        raise ValueError(
            f"{name} must be positive"
        )

    return numeric


def _optional_finite_number(
    value: object,
) -> float | None:
    if value in (None, ""):
        return None

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None

    if not isfinite(numeric):
        return None

    return numeric


def _recursive_items(
    value: object,
    prefix: str = "",
):
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = (
                f"{prefix}.{key}"
                if prefix
                else str(key)
            )

            yield from _recursive_items(
                item,
                path,
            )

        return

    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            path = (
                f"{prefix}[{index}]"
                if prefix
                else f"[{index}]"
            )

            yield from _recursive_items(
                item,
                path,
            )

        return

    yield prefix, value


def extract_mammaprint_feature(
    context: Mapping[str, object],
) -> tuple[float, float]:
    """Return numeric MammaPrint value and missingness indicator."""

    candidates = []

    for path, value in _recursive_items(context):
        normalized_path = (
            path.lower()
            .replace("_", "")
            .replace("-", "")
            .replace(" ", "")
        )

        if "mammaprint" in normalized_path:
            candidates.append(value)

    for value in candidates:
        if value is None:
            continue

        if isinstance(value, bool):
            return float(value), 0.0

        if isinstance(value, (int, float)):
            numeric = float(value)

            if isfinite(numeric):
                return numeric, 0.0

        text = str(value).strip().lower()

        if text in {
            "1",
            "positive",
            "high",
            "high risk",
            "high-risk",
            "yes",
            "true",
        }:
            return 1.0, 0.0

        if text in {
            "0",
            "negative",
            "low",
            "low risk",
            "low-risk",
            "no",
            "false",
        }:
            return 0.0, 0.0

    return 0.0, 1.0


def layer6_activation(
    case: Mapping[str, object],
) -> dict[str, object]:
    """Return whether Layer 6 may replace the previous-layer prediction."""

    baseline_volume = _optional_finite_number(
        case.get("baseline_volume_ml")
    )
    early_volume = _optional_finite_number(
        case.get("early_volume_ml")
    )
    early_day = _optional_finite_number(
        case.get("early_day")
    )
    final_day = _optional_finite_number(
        case.get("final_day")
    )

    if baseline_volume is None or baseline_volume <= 0:
        return {
            "active": False,
            "status": "inactive_invalid_baseline",
            "reason": (
                "Layer 6 requires a positive baseline volume."
            ),
        }

    if early_volume is None or early_volume <= 0:
        return {
            "active": False,
            "status": "inactive_missing_early_volume",
            "reason": (
                "Layer 6 requires a positive early MRI volume."
            ),
        }

    if early_day is None or early_day <= 0:
        return {
            "active": False,
            "status": "inactive_missing_early_day",
            "reason": (
                "Layer 6 requires a positive early MRI day."
            ),
        }

    if final_day is None or final_day <= early_day:
        return {
            "active": False,
            "status": "inactive_no_late_horizon",
            "reason": (
                "Layer 6 requires a prediction day after "
                "the early MRI."
            ),
        }

    return {
        "active": True,
        "status": "active",
        "reason": (
            "A valid early MRI and a later prediction "
            "horizon are available."
        ),
    }


def observed_early_log_rate_per_day(
    case: Mapping[str, object],
) -> float:
    """Calculate the observed baseline-to-early log-volume rate."""

    activation = layer6_activation(case)

    if not activation["active"]:
        raise ValueError(
            str(activation["reason"])
        )

    baseline_volume = _positive_number(
        case["baseline_volume_ml"],
        "baseline_volume_ml",
    )
    early_volume = _positive_number(
        case["early_volume_ml"],
        "early_volume_ml",
    )
    early_day = _positive_number(
        case["early_day"],
        "early_day",
    )

    return (
        log(early_volume)
        - log(baseline_volume)
    ) / early_day


def build_layer6_features(
    case: Mapping[str, object],
    static_layer4_prediction: Mapping[str, object],
) -> dict[str, float]:
    """Build the frozen Layer 6 feature contract.

    All features must be available by the early MRI time. The final day is
    an allowed prediction-horizon input, not an outcome-derived feature.
    """

    activation = layer6_activation(case)

    if not activation["active"]:
        raise ValueError(
            str(activation["reason"])
        )

    baseline_volume = _positive_number(
        case["baseline_volume_ml"],
        "baseline_volume_ml",
    )
    early_day = _positive_number(
        case["early_day"],
        "early_day",
    )
    final_day = _positive_number(
        case["final_day"],
        "final_day",
    )

    static_point = _positive_number(
        static_layer4_prediction["point_ml"],
        "static_layer4_prediction.point_ml",
    )
    static_lower = _positive_number(
        static_layer4_prediction["lower_80_ml"],
        "static_layer4_prediction.lower_80_ml",
    )
    static_upper = _positive_number(
        static_layer4_prediction["upper_80_ml"],
        "static_layer4_prediction.upper_80_ml",
    )

    if static_upper < static_lower:
        raise ValueError(
            "Static Layer 4 upper interval must not be "
            "below its lower interval."
        )

    mammaprint_value, mammaprint_missing = (
        extract_mammaprint_feature(
            case.get("context", {})
        )
    )

    early_rate = observed_early_log_rate_per_day(
        case
    )

    features = {
        "log_baseline_volume": log(
            baseline_volume
        ),
        "static_layer4_log_change": (
            log(static_point)
            - log(baseline_volume)
        ),
        "static_layer4_log_width_80": (
            log(static_upper)
            - log(static_lower)
        ),
        "final_day_scaled": (
            final_day
            / FINAL_DAY_REFERENCE_DAYS
        ),
        "early_day_scaled": (
            early_day
            / EARLY_RATE_REFERENCE_DAYS
        ),
        "remaining_horizon_scaled": (
            (final_day - early_day)
            / REMAINING_HORIZON_REFERENCE_DAYS
        ),
        "observed_early_log_rate_42d": (
            early_rate
            * EARLY_RATE_REFERENCE_DAYS
        ),
        "mammaprint_value": mammaprint_value,
        "mammaprint_missing": (
            mammaprint_missing
        ),
    }

    if tuple(features) != LAYER6_FEATURE_NAMES:
        raise RuntimeError(
            "Layer 6 feature ordering does not match "
            "the frozen feature contract."
        )

    return features


def layer6_feature_vector(
    case: Mapping[str, object],
    static_layer4_prediction: Mapping[str, object],
) -> tuple[float, ...]:
    """Return Layer 6 features in frozen model-input order."""

    features = build_layer6_features(
        case,
        static_layer4_prediction,
    )

    return tuple(
        features[name]
        for name in LAYER6_FEATURE_NAMES
    )


def late_response_target_per_day(
    case: Mapping[str, object],
    observed_final_volume_ml: float,
) -> float:
    """Calculate the known-truth late rate for model training."""

    activation = layer6_activation(case)

    if not activation["active"]:
        raise ValueError(
            str(activation["reason"])
        )

    early_volume = _positive_number(
        case["early_volume_ml"],
        "early_volume_ml",
    )
    early_day = _positive_number(
        case["early_day"],
        "early_day",
    )
    final_day = _positive_number(
        case["final_day"],
        "final_day",
    )
    final_volume = _positive_number(
        observed_final_volume_ml,
        "observed_final_volume_ml",
    )

    return (
        log(final_volume)
        - log(early_volume)
    ) / (
        final_day - early_day
    )


def predict_volume_from_late_rate(
    case: Mapping[str, object],
    late_log_rate_per_day: float,
    *,
    prediction_day: float | None = None,
) -> float:
    """Decode one late-response parameter into a tumor-volume prediction."""

    activation = layer6_activation(case)

    if not activation["active"]:
        raise ValueError(
            str(activation["reason"])
        )

    baseline_volume = _positive_number(
        case["baseline_volume_ml"],
        "baseline_volume_ml",
    )
    early_volume = _positive_number(
        case["early_volume_ml"],
        "early_volume_ml",
    )
    early_day = _positive_number(
        case["early_day"],
        "early_day",
    )
    final_day = _positive_number(
        case["final_day"],
        "final_day",
    )

    target_day = (
        final_day
        if prediction_day is None
        else _finite_number(
            prediction_day,
            "prediction_day",
        )
    )

    late_rate = _finite_number(
        late_log_rate_per_day,
        "late_log_rate_per_day",
    )

    if target_day <= 0:
        return baseline_volume

    if target_day <= early_day:
        early_rate = (
            log(early_volume)
            - log(baseline_volume)
        ) / early_day

        predicted_log_volume = (
            log(baseline_volume)
            + early_rate * target_day
        )

    else:
        predicted_log_volume = (
            log(early_volume)
            + late_rate
            * (target_day - early_day)
        )

    predicted_log_volume = min(
        max(predicted_log_volume, -30.0),
        30.0,
    )

    return max(
        exp(predicted_log_volume),
        EPSILON_VOLUME_ML,
    )


def summarize_late_rate_posterior(
    case: Mapping[str, object],
    *,
    late_rate_mean_per_day: float,
    late_rate_sd_per_day: float,
    prediction_day: float | None = None,
) -> dict[str, object]:
    """Propagate a Gaussian late-rate posterior into volume intervals."""

    mean = _finite_number(
        late_rate_mean_per_day,
        "late_rate_mean_per_day",
    )
    sd = _finite_number(
        late_rate_sd_per_day,
        "late_rate_sd_per_day",
    )

    if sd < 0:
        raise ValueError(
            "late_rate_sd_per_day must be nonnegative"
        )

    activation = layer6_activation(case)

    if not activation["active"]:
        raise ValueError(
            str(activation["reason"])
        )

    final_day = _positive_number(
        case["final_day"],
        "final_day",
    )
    target_day = (
        final_day
        if prediction_day is None
        else _finite_number(
            prediction_day,
            "prediction_day",
        )
    )

    early_day = _positive_number(
        case["early_day"],
        "early_day",
    )

    if target_day <= early_day:
        point = predict_volume_from_late_rate(
            case,
            mean,
            prediction_day=target_day,
        )

        return {
            "decoder_version": (
                LAYER6_DECODER_VERSION
            ),
            "point_ml": point,
            "lower_80_ml": point,
            "upper_80_ml": point,
            "lower_95_ml": point,
            "upper_95_ml": point,
            "late_rate_mean_per_day": mean,
            "late_rate_sd_per_day": sd,
            "prediction_day": target_day,
            "activation": activation,
        }

    normal = NormalDist()
    z80 = normal.inv_cdf(0.90)
    z95 = normal.inv_cdf(0.975)

    return {
        "decoder_version": (
            LAYER6_DECODER_VERSION
        ),
        "point_ml": predict_volume_from_late_rate(
            case,
            mean,
            prediction_day=target_day,
        ),
        "lower_80_ml": (
            predict_volume_from_late_rate(
                case,
                mean - z80 * sd,
                prediction_day=target_day,
            )
        ),
        "upper_80_ml": (
            predict_volume_from_late_rate(
                case,
                mean + z80 * sd,
                prediction_day=target_day,
            )
        ),
        "lower_95_ml": (
            predict_volume_from_late_rate(
                case,
                mean - z95 * sd,
                prediction_day=target_day,
            )
        ),
        "upper_95_ml": (
            predict_volume_from_late_rate(
                case,
                mean + z95 * sd,
                prediction_day=target_day,
            )
        ),
        "late_rate_mean_per_day": mean,
        "late_rate_sd_per_day": sd,
        "prediction_day": target_day,
        "activation": activation,
    }


def validate_feature_vector(
    values: Sequence[float],
) -> tuple[float, ...]:
    """Validate a serialized Layer 6 feature vector."""

    if len(values) != len(LAYER6_FEATURE_NAMES):
        raise ValueError(
            "Layer 6 feature vector has "
            f"{len(values)} values; expected "
            f"{len(LAYER6_FEATURE_NAMES)}."
        )

    return tuple(
        _finite_number(
            value,
            LAYER6_FEATURE_NAMES[index],
        )
        for index, value in enumerate(values)
    )
