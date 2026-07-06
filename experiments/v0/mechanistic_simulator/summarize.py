"""Trajectory summaries for simulator ensembles."""

from __future__ import annotations

import math
from typing import Any


DEFAULT_RESIDUAL_THRESHOLD_ML = 1.0


def quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile requires at least one value.")
    if probability <= 0:
        return min(values)
    if probability >= 1:
        return max(values)
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    weight = position - lower_index
    return ordered[lower_index] * (1.0 - weight) + ordered[upper_index] * weight


def _pearson_abs(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3 or len(xs) != len(ys):
        return 0.0
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    centered_x = [value - mean_x for value in xs]
    centered_y = [value - mean_y for value in ys]
    denom_x = math.sqrt(sum(value * value for value in centered_x))
    denom_y = math.sqrt(sum(value * value for value in centered_y))
    if denom_x == 0 or denom_y == 0:
        return 0.0
    corr = sum(x * y for x, y in zip(centered_x, centered_y)) / (denom_x * denom_y)
    return abs(corr)


def flatten_driver_params(params: dict[str, Any]) -> dict[str, float]:
    flattened: dict[str, float] = {}
    for key in (
        "growth_rate",
        "carrying_capacity_ml",
        "resistant_fraction",
        "resistant_sensitivity_scale",
        "observation_noise_fraction",
    ):
        if key in params and isinstance(params[key], (int, float)):
            flattened[key] = float(params[key])
    for drug, value in params.get("drug_sensitivity", {}).items():
        if isinstance(value, (int, float)):
            flattened[f"drug_sensitivity.{drug}"] = float(value)
    return flattened


def summarize_trajectories(
    particle_trajectories: list[dict[str, Any]],
    residual_threshold_ml: float = DEFAULT_RESIDUAL_THRESHOLD_ML,
) -> dict[str, Any]:
    """Summarize particle trajectories without making clinical claims."""

    if not particle_trajectories:
        raise ValueError("particle_trajectories must contain at least one particle.")
    times = list(particle_trajectories[0]["times"])
    for trajectory in particle_trajectories:
        if list(trajectory["times"]) != times:
            raise ValueError("all particle trajectories must use the same output times.")

    volumes_by_day: list[list[float]] = []
    for index in range(len(times)):
        volumes_by_day.append(
            [float(particle["predicted_volume_ml"][index]) for particle in particle_trajectories]
        )

    median = [quantile(values, 0.5) for values in volumes_by_day]
    interval_80 = [[quantile(values, 0.1), quantile(values, 0.9)] for values in volumes_by_day]
    interval_95 = [[quantile(values, 0.025), quantile(values, 0.975)] for values in volumes_by_day]
    uncertainty_width_by_day = [upper - lower for lower, upper in interval_80]
    max_uncertainty_index = max(
        range(len(uncertainty_width_by_day)),
        key=lambda index: uncertainty_width_by_day[index],
    )
    final_volumes = volumes_by_day[-1]
    probability_below_threshold = (
        sum(1 for value in final_volumes if value < residual_threshold_ml) / len(final_volumes)
    )
    denominator = max(sum(max(value, 1e-9) for value in median) / len(median), 1e-9)
    uncertainty_score = (
        sum(uncertainty_width_by_day) / len(uncertainty_width_by_day)
    ) / denominator

    final_values = [float(particle["predicted_volume_ml"][-1]) for particle in particle_trajectories]
    parameter_values: dict[str, list[float]] = {}
    for particle in particle_trajectories:
        for key, value in flatten_driver_params(particle["parameters"]).items():
            parameter_values.setdefault(key, []).append(value)

    correlations = {
        key: _pearson_abs(values, final_values)
        for key, values in parameter_values.items()
    }
    ordered_drivers = sorted(correlations, key=lambda key: correlations[key], reverse=True)
    dominant_factors = [key for key in ordered_drivers[:3] if correlations[key] >= 0.2]
    prior_dominated = sorted(key for key, corr in correlations.items() if corr < 0.1)

    warning_set = {
        warning
        for particle in particle_trajectories
        for warning in particle.get("warnings", [])
    }
    warning_set.add("Exploratory simulation; parameters are not clinically validated.")

    return {
        "times": times,
        "median_volume_ml": median,
        "interval_80_volume_ml": interval_80,
        "interval_95_volume_ml": interval_95,
        "probability_final_volume_below_research_threshold": probability_below_threshold,
        "research_residual_burden_threshold_ml": residual_threshold_ml,
        "uncertainty_width_by_day": uncertainty_width_by_day,
        "maximum_uncertainty_day": times[max_uncertainty_index],
        "uncertainty_score": uncertainty_score,
        "final_residual_burden_distribution": {
            "median_ml": quantile(final_volumes, 0.5),
            "interval_80_ml": [quantile(final_volumes, 0.1), quantile(final_volumes, 0.9)],
            "interval_95_ml": [quantile(final_volumes, 0.025), quantile(final_volumes, 0.975)],
            "probability_below_research_threshold": probability_below_threshold,
        },
        "driver_summary": {
            "dominant_factors": dominant_factors,
            "prior_dominated_parameters": prior_dominated,
            "correlation_to_final_volume": correlations,
        },
        "warnings": sorted(warning_set),
    }
