"""Fail-closed and likelihood-tempered safety policies for Layer 7."""

from __future__ import annotations

import math
from typing import Sequence


POSTERIOR_SAFETY_VERSION = (
    "oncotwin_layer7_posterior_safety_v0_1"
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


def normalize_log_weights(
    log_weights: Sequence[float],
) -> list[float]:
    values = [
        _finite(value, "log weight")
        for value in log_weights
    ]
    if not values:
        raise ValueError(
            "log_weights must not be empty"
        )

    maximum = max(values)
    shifted = [
        math.exp(value - maximum)
        for value in values
    ]
    total = sum(shifted)

    if not math.isfinite(total) or total <= 0:
        raise ValueError(
            "log weights contain no finite mass"
        )

    return [
        value / total
        for value in shifted
    ]


def effective_sample_size(
    weights: Sequence[float],
) -> float:
    values = [
        _finite(value, "weight")
        for value in weights
    ]
    if not values:
        raise ValueError(
            "weights must not be empty"
        )
    if any(value < 0 for value in values):
        raise ValueError(
            "weights must be nonnegative"
        )

    total = sum(values)
    if total <= 0:
        raise ValueError(
            "weights must contain positive mass"
        )

    normalized = [
        value / total
        for value in values
    ]
    return 1.0 / sum(
        value * value
        for value in normalized
    )


def fail_closed_weights(
    *,
    log_prior_weights: Sequence[float],
    log_likelihoods: Sequence[float],
    ess_threshold_fraction: float,
) -> dict[str, object]:
    if (
        len(log_prior_weights)
        != len(log_likelihoods)
        or not log_prior_weights
    ):
        raise ValueError(
            "prior weights and likelihoods must have "
            "equal nonzero length"
        )

    threshold = _finite(
        ess_threshold_fraction,
        "ess_threshold_fraction",
    )
    if not 0 < threshold <= 1:
        raise ValueError(
            "ess_threshold_fraction must be in (0, 1]"
        )

    prior_weights = normalize_log_weights(
        log_prior_weights
    )
    posterior_weights = normalize_log_weights(
        [
            prior + likelihood
            for prior, likelihood in zip(
                log_prior_weights,
                log_likelihoods,
            )
        ]
    )

    raw_ess = effective_sample_size(
        posterior_weights
    )
    raw_fraction = (
        raw_ess / len(posterior_weights)
    )

    if raw_fraction < threshold:
        selected_weights = prior_weights
        status = "fallback_to_prior_low_ess"
        likelihood_power = 0.0
    else:
        selected_weights = posterior_weights
        status = "full_update_accepted"
        likelihood_power = 1.0

    selected_ess = effective_sample_size(
        selected_weights
    )

    return {
        "safety_version": (
            POSTERIOR_SAFETY_VERSION
        ),
        "strategy": "fail_closed",
        "status": status,
        "ess_threshold_fraction": threshold,
        "likelihood_power": (
            likelihood_power
        ),
        "raw_effective_sample_size": (
            raw_ess
        ),
        "raw_ess_fraction": raw_fraction,
        "selected_effective_sample_size": (
            selected_ess
        ),
        "selected_ess_fraction": (
            selected_ess
            / len(selected_weights)
        ),
        "weights": selected_weights,
    }


def tempered_likelihood_weights(
    *,
    log_prior_weights: Sequence[float],
    log_likelihoods: Sequence[float],
    ess_target_fraction: float,
    grid_size: int = 1000,
    refinement_iterations: int = 60,
) -> dict[str, object]:
    """Use the largest likelihood power whose ESS meets the target."""

    if (
        len(log_prior_weights)
        != len(log_likelihoods)
        or not log_prior_weights
    ):
        raise ValueError(
            "prior weights and likelihoods must have "
            "equal nonzero length"
        )

    target = _finite(
        ess_target_fraction,
        "ess_target_fraction",
    )
    if not 0 < target <= 1:
        raise ValueError(
            "ess_target_fraction must be in (0, 1]"
        )
    if grid_size < 10:
        raise ValueError(
            "grid_size must be at least 10"
        )
    if refinement_iterations < 1:
        raise ValueError(
            "refinement_iterations must be positive"
        )

    n = len(log_prior_weights)
    target_ess = target * n

    def weights_at(
        likelihood_power: float,
    ) -> list[float]:
        return normalize_log_weights(
            [
                prior
                + likelihood_power * likelihood
                for prior, likelihood in zip(
                    log_prior_weights,
                    log_likelihoods,
                )
            ]
        )

    prior_weights = weights_at(0.0)
    raw_weights = weights_at(1.0)

    prior_ess = effective_sample_size(
        prior_weights
    )
    raw_ess = effective_sample_size(
        raw_weights
    )

    if raw_ess >= target_ess:
        return {
            "safety_version": (
                POSTERIOR_SAFETY_VERSION
            ),
            "strategy": (
                "likelihood_tempering"
            ),
            "status": (
                "full_update_accepted"
            ),
            "ess_target_fraction": target,
            "likelihood_power": 1.0,
            "prior_effective_sample_size": (
                prior_ess
            ),
            "raw_effective_sample_size": (
                raw_ess
            ),
            "selected_effective_sample_size": (
                raw_ess
            ),
            "selected_ess_fraction": (
                raw_ess / n
            ),
            "weights": raw_weights,
        }

    if prior_ess < target_ess:
        return {
            "safety_version": (
                POSTERIOR_SAFETY_VERSION
            ),
            "strategy": (
                "likelihood_tempering"
            ),
            "status": (
                "prior_support_insufficient"
            ),
            "ess_target_fraction": target,
            "likelihood_power": 0.0,
            "prior_effective_sample_size": (
                prior_ess
            ),
            "raw_effective_sample_size": (
                raw_ess
            ),
            "selected_effective_sample_size": (
                prior_ess
            ),
            "selected_ess_fraction": (
                prior_ess / n
            ),
            "weights": prior_weights,
        }

    # Search the complete interval first so the procedure is robust to
    # small numerical non-monotonicities.
    feasible_power = 0.0
    feasible_weights = prior_weights

    for index in range(1, grid_size + 1):
        power = index / grid_size
        weights = weights_at(power)
        ess = effective_sample_size(
            weights
        )

        if ess >= target_ess:
            feasible_power = power
            feasible_weights = weights

    if feasible_power >= 1.0:
        selected_power = 1.0
        selected_weights = raw_weights
    else:
        low = feasible_power
        high = min(
            1.0,
            feasible_power + 1.0 / grid_size,
        )
        selected_power = low
        selected_weights = feasible_weights

        for _ in range(
            refinement_iterations
        ):
            midpoint = (low + high) / 2.0
            midpoint_weights = weights_at(
                midpoint
            )
            midpoint_ess = (
                effective_sample_size(
                    midpoint_weights
                )
            )

            if midpoint_ess >= target_ess:
                low = midpoint
                selected_power = midpoint
                selected_weights = (
                    midpoint_weights
                )
            else:
                high = midpoint

    selected_ess = effective_sample_size(
        selected_weights
    )

    return {
        "safety_version": (
            POSTERIOR_SAFETY_VERSION
        ),
        "strategy": (
            "likelihood_tempering"
        ),
        "status": "tempered_to_ess_target",
        "ess_target_fraction": target,
        "likelihood_power": (
            selected_power
        ),
        "prior_effective_sample_size": (
            prior_ess
        ),
        "raw_effective_sample_size": (
            raw_ess
        ),
        "selected_effective_sample_size": (
            selected_ess
        ),
        "selected_ess_fraction": (
            selected_ess / n
        ),
        "weights": selected_weights,
    }
