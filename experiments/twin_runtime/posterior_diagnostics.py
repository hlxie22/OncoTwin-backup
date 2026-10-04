"""Detailed posterior-health diagnostics for Layer 7 importance sampling."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from experiments.prior_builder.bounds import (
    PARAMETER_BOUNDS,
)


LAYER7_POSTERIOR_DIAGNOSTICS_VERSION = (
    "oncotwin_layer7_posterior_diagnostics_v0_1"
)

_PARAMETER_NAMES = (
    "growth_rate_per_day",
    "active_treatment_sensitivity",
    "resistant_fraction",
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


def _normalize_weights(
    values: Sequence[float],
) -> list[float]:
    weights = [
        max(_finite(value, "weight"), 0.0)
        for value in values
    ]
    total = sum(weights)

    if total <= 0:
        raise ValueError(
            "weights must contain positive mass"
        )

    return [
        value / total
        for value in weights
    ]


def _normalize_log_weights(
    values: Sequence[float],
) -> list[float]:
    logs = [
        _finite(value, "log weight")
        for value in values
    ]
    maximum = max(logs)
    shifted = [
        math.exp(value - maximum)
        for value in logs
    ]
    return _normalize_weights(shifted)


def _weighted_quantile(
    values: Sequence[float],
    weights: Sequence[float],
    probability: float,
) -> float:
    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            "probability must be in [0, 1]"
        )
    if len(values) != len(weights) or not values:
        raise ValueError(
            "values and weights must have equal nonzero length"
        )

    normalized = _normalize_weights(weights)
    ordered = sorted(
        zip(
            (
                _finite(value, "quantile value")
                for value in values
            ),
            normalized,
        ),
        key=lambda pair: pair[0],
    )

    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= probability:
            return value

    return ordered[-1][0]


def _parameter_value(
    parameters: Mapping[str, object],
    name: str,
) -> float:
    if name == "growth_rate_per_day":
        return _finite(
            parameters.get("growth_rate"),
            name,
        )

    if name == "resistant_fraction":
        return _finite(
            parameters.get("resistant_fraction"),
            name,
        )

    if name == "active_treatment_sensitivity":
        sensitivities = parameters.get(
            "drug_sensitivity"
        )
        if not isinstance(sensitivities, Mapping):
            raise ValueError(
                "drug_sensitivity must be a mapping"
            )

        active = [
            _finite(
                sensitivities.get(drug),
                f"drug_sensitivity.{drug}",
            )
            for drug in (
                "anthracycline",
                "taxane",
            )
        ]
        return sum(active) / len(active)

    raise KeyError(name)


def _parameter_summary(
    *,
    name: str,
    values: Sequence[float],
    prior_weights: Sequence[float],
    posterior_weights: Sequence[float],
    boundary_fraction: float,
) -> dict[str, object]:
    prior_lower = _weighted_quantile(
        values,
        prior_weights,
        0.10,
    )
    prior_median = _weighted_quantile(
        values,
        prior_weights,
        0.50,
    )
    prior_upper = _weighted_quantile(
        values,
        prior_weights,
        0.90,
    )

    posterior_lower = _weighted_quantile(
        values,
        posterior_weights,
        0.10,
    )
    posterior_median = _weighted_quantile(
        values,
        posterior_weights,
        0.50,
    )
    posterior_upper = _weighted_quantile(
        values,
        posterior_weights,
        0.90,
    )

    prior_width = max(
        prior_upper - prior_lower,
        0.0,
    )
    posterior_width = max(
        posterior_upper - posterior_lower,
        0.0,
    )

    bounds = PARAMETER_BOUNDS[name]
    hard_range = (
        float(bounds.hard_max)
        - float(bounds.hard_min)
    )
    boundary_width = (
        boundary_fraction * hard_range
    )

    def boundary_mass(
        weights: Sequence[float],
    ) -> dict[str, float]:
        lower = sum(
            weight
            for value, weight
            in zip(values, weights)
            if value
            <= float(bounds.hard_min)
            + boundary_width
        )
        upper = sum(
            weight
            for value, weight
            in zip(values, weights)
            if value
            >= float(bounds.hard_max)
            - boundary_width
        )
        return {
            "lower": lower,
            "upper": upper,
            "either": lower + upper,
        }

    prior_boundary = boundary_mass(
        prior_weights
    )
    posterior_boundary = boundary_mass(
        posterior_weights
    )

    prior_mean = sum(
        value * weight
        for value, weight
        in zip(values, prior_weights)
    )
    posterior_mean = sum(
        value * weight
        for value, weight
        in zip(values, posterior_weights)
    )

    prior_variance = sum(
        weight * (value - prior_mean) ** 2
        for value, weight
        in zip(values, prior_weights)
    )
    prior_sd = math.sqrt(
        max(prior_variance, 0.0)
    )

    standardized_shift = (
        (posterior_mean - prior_mean)
        / prior_sd
        if prior_sd > 1e-12
        else 0.0
    )

    return {
        "parameter": name,
        "prior": {
            "mean": prior_mean,
            "median": prior_median,
            "lower_80": prior_lower,
            "upper_80": prior_upper,
            "width_80": prior_width,
            "boundary_mass": prior_boundary,
        },
        "posterior": {
            "mean": posterior_mean,
            "median": posterior_median,
            "lower_80": posterior_lower,
            "upper_80": posterior_upper,
            "width_80": posterior_width,
            "boundary_mass": posterior_boundary,
        },
        "contraction_ratio_80": (
            posterior_width / prior_width
            if prior_width > 1e-12
            else 1.0
        ),
        "standardized_mean_shift": (
            standardized_shift
        ),
        "boundary_mass_increase": (
            posterior_boundary["either"]
            - prior_boundary["either"]
        ),
        "hard_bounds": {
            "minimum": float(
                bounds.hard_min
            ),
            "maximum": float(
                bounds.hard_max
            ),
            "boundary_fraction": (
                boundary_fraction
            ),
        },
    }


def summarize_posterior_health(
    posterior_update: Mapping[str, object],
    *,
    boundary_fraction: float = 0.02,
    boundary_collapse_mass: float = 0.50,
    boundary_mass_increase: float = 0.25,
    material_mass_target: float = 0.95,
) -> dict[str, object]:
    """Summarize weight and parameter health for one posterior update."""

    if not 0 < boundary_fraction < 0.5:
        raise ValueError(
            "boundary_fraction must be in (0, 0.5)"
        )
    if not 0 < boundary_collapse_mass <= 1:
        raise ValueError(
            "boundary_collapse_mass must be in (0, 1]"
        )
    if not 0 <= boundary_mass_increase <= 1:
        raise ValueError(
            "boundary_mass_increase must be in [0, 1]"
        )
    if not 0 < material_mass_target <= 1:
        raise ValueError(
            "material_mass_target must be in (0, 1]"
        )

    rows = posterior_update.get(
        "particle_trajectories"
    )
    if (
        not isinstance(rows, Sequence)
        or isinstance(rows, (str, bytes))
        or not rows
    ):
        raise ValueError(
            "particle_trajectories must contain rows"
        )

    particle_rows = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(
                f"particle row {index} must be a mapping"
            )
        particle_rows.append(row)

    posterior_weights = _normalize_weights(
        [
            _finite(
                row.get("weight"),
                "particle weight",
            )
            for row in particle_rows
        ]
    )

    if all(
        row.get("log_prior_weight")
        is not None
        for row in particle_rows
    ):
        prior_weights = _normalize_log_weights(
            [
                _finite(
                    row.get("log_prior_weight"),
                    "log prior weight",
                )
                for row in particle_rows
            ]
        )
    else:
        prior_weights = [
            1.0 / len(particle_rows)
            for _ in particle_rows
        ]

    ess = 1.0 / sum(
        weight * weight
        for weight in posterior_weights
    )
    ess_fraction = (
        ess / len(posterior_weights)
    )

    entropy = -sum(
        weight * math.log(weight)
        for weight in posterior_weights
        if weight > 0
    )
    perplexity = math.exp(entropy)
    normalized_entropy = (
        entropy
        / math.log(len(posterior_weights))
        if len(posterior_weights) > 1
        else 1.0
    )

    sorted_weights = sorted(
        posterior_weights,
        reverse=True,
    )
    cumulative = 0.0
    material_count = 0
    for weight in sorted_weights:
        cumulative += weight
        material_count += 1
        if cumulative >= material_mass_target:
            break

    kl_posterior_prior = sum(
        posterior * math.log(
            posterior / prior
        )
        for posterior, prior
        in zip(
            posterior_weights,
            prior_weights,
        )
        if posterior > 0 and prior > 0
    )

    log_weights = [
        _finite(
            row.get("log_weight"),
            "log weight",
        )
        for row in particle_rows
    ]
    log_likelihoods = [
        _finite(
            row.get("log_likelihood"),
            "log likelihood",
        )
        for row in particle_rows
    ]

    parameter_summaries = {}
    collapse_parameters = []

    for name in _PARAMETER_NAMES:
        values = []
        for row in particle_rows:
            parameters = row.get(
                "parameters"
            )
            if not isinstance(
                parameters,
                Mapping,
            ):
                raise ValueError(
                    "particle parameters must be a mapping"
                )
            values.append(
                _parameter_value(
                    parameters,
                    name,
                )
            )

        summary = _parameter_summary(
            name=name,
            values=values,
            prior_weights=prior_weights,
            posterior_weights=posterior_weights,
            boundary_fraction=(
                boundary_fraction
            ),
        )
        parameter_summaries[name] = summary

        posterior_boundary = summary[
            "posterior"
        ]["boundary_mass"]["either"]
        increase = summary[
            "boundary_mass_increase"
        ]

        if (
            posterior_boundary
            >= boundary_collapse_mass
            and increase
            >= boundary_mass_increase
        ):
            collapse_parameters.append(name)

    runtime_fallback = str(
        posterior_update.get(
            "fallback_status",
            "unknown",
        )
    )

    if collapse_parameters:
        status = "updated_boundary_collapse"
    elif runtime_fallback == (
        "tempered_smc_recommended"
    ):
        status = "updated_low_ess"
    else:
        status = "updated_healthy"

    return {
        "diagnostics_version": (
            LAYER7_POSTERIOR_DIAGNOSTICS_VERSION
        ),
        "status": status,
        "particle_count": len(
            posterior_weights
        ),
        "weights": {
            "effective_sample_size": ess,
            "effective_sample_size_fraction": (
                ess_fraction
            ),
            "maximum_weight": max(
                posterior_weights
            ),
            "minimum_weight": min(
                posterior_weights
            ),
            "entropy": entropy,
            "normalized_entropy": (
                normalized_entropy
            ),
            "effective_perplexity": (
                perplexity
            ),
            "material_particle_count": (
                material_count
            ),
            "material_particle_fraction": (
                material_count
                / len(posterior_weights)
            ),
            "material_mass_target": (
                material_mass_target
            ),
            "posterior_to_prior_kl": (
                kl_posterior_prior
            ),
            "log_weight_range": (
                max(log_weights)
                - min(log_weights)
            ),
            "log_likelihood_range": (
                max(log_likelihoods)
                - min(log_likelihoods)
            ),
        },
        "parameters": parameter_summaries,
        "boundary_collapse_parameters": (
            collapse_parameters
        ),
        "runtime_fallback_status": (
            runtime_fallback
        ),
    }
