"""Runtime candidate policy for safe Layer 7 posterior prediction."""

from __future__ import annotations

import math
from typing import Mapping


LAYER7_FAIL_CLOSED_CANDIDATE_VERSION = (
    "oncotwin_layer7_fail_closed_candidate_v0_2"
)
LAYER7_FAIL_CLOSED_CANDIDATE_NAME = (
    "layer7_fail_closed_posterior_candidate"
)

_REQUIRED_PREDICTION_KEYS = (
    "point_ml",
    "lower_80_ml",
    "upper_80_ml",
    "lower_95_ml",
    "upper_95_ml",
)


def _finite(
    value: object,
    name: str,
) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be numeric"
        ) from exc

    if not math.isfinite(number):
        raise ValueError(
            f"{name} must be finite"
        )

    return number


def _prediction_copy(
    value: object,
    name: str,
) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError(
            f"{name} must be a mapping"
        )

    prediction = {
        key: _finite(
            value.get(key),
            f"{name}.{key}",
        )
        for key in _REQUIRED_PREDICTION_KEYS
    }

    ordered = [
        prediction["lower_95_ml"],
        prediction["lower_80_ml"],
        prediction["point_ml"],
        prediction["upper_80_ml"],
        prediction["upper_95_ml"],
    ]

    if ordered != sorted(ordered):
        raise ValueError(
            f"{name} intervals must be nested"
        )

    return prediction


def apply_layer7_fail_closed_candidate(
    case_result: Mapping[str, object],
    *,
    ess_threshold_fraction: float = 0.10,
    accepted_health_statuses: tuple[str, ...] = (
        "updated_healthy",
    ),
) -> dict[str, object]:
    """Apply the deterministic Layer 7 fail-closed policy.

    The held-out final outcome is deliberately not read by this function.
    """

    threshold = _finite(
        ess_threshold_fraction,
        "ess_threshold_fraction",
    )
    if not 0 < threshold <= 1:
        raise ValueError(
            "ess_threshold_fraction must be in (0, 1]"
        )

    prior = _prediction_copy(
        case_result.get("prior_prediction"),
        "prior_prediction",
    )
    posterior = _prediction_copy(
        case_result.get(
            "posterior_prediction"
        ),
        "posterior_prediction",
    )

    if case_result.get("early_day") is None:
        selected = prior
        status = (
            "not_updated_no_valid_observation"
        )
        fallback_used = True
        fallback_reason = (
            "No valid early observation was available."
        )
        ess_fraction = None
        raw_health_status = (
            "not_updated_no_valid_observation"
        )
    else:
        health = case_result.get(
            "posterior_health"
        )
        if not isinstance(health, Mapping):
            raise ValueError(
                "posterior_health must be a mapping"
            )

        raw_health_status = str(
            health.get("status", "unknown")
        )
        weights = health.get("weights")
        if not isinstance(weights, Mapping):
            raise ValueError(
                "posterior_health.weights must be a mapping"
            )

        ess_fraction = _finite(
            weights.get(
                "effective_sample_size_fraction"
            ),
            "effective_sample_size_fraction",
        )

        accepted = (
            raw_health_status
            in accepted_health_statuses
            and ess_fraction >= threshold
        )

        if accepted:
            selected = posterior
            status = "updated_accepted"
            fallback_used = False
            fallback_reason = None
        else:
            selected = prior
            status = (
                "fallback_to_prior_unhealthy_posterior"
            )
            fallback_used = True
            fallback_reason = (
                "The raw posterior did not satisfy the "
                "predeclared posterior-health contract."
            )

    return {
        **selected,
        "layer7": {
            "candidate_version": (
                LAYER7_FAIL_CLOSED_CANDIDATE_VERSION
            ),
            "candidate_name": (
                LAYER7_FAIL_CLOSED_CANDIDATE_NAME
            ),
            "status": status,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "ess_threshold_fraction": threshold,
            "raw_ess_fraction": ess_fraction,
            "raw_health_status": (
                raw_health_status
            ),
            "accepted_health_statuses": list(
                accepted_health_statuses
            ),
            "selected_prediction": (
                "mechanistic_prior"
                if fallback_used
                else "mechanistic_posterior"
            ),
            "final_outcome_used_for_selection": (
                False
            ),
        },
    }
