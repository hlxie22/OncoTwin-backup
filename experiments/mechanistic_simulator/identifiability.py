"""Practical identifiability checks for simulator particle outputs."""

from __future__ import annotations

import math
from typing import Any

from .summarize import flatten_driver_params, quantile


def _weighted_quantile(values: list[float], weights: list[float], probability: float) -> float:
    ordered = sorted(zip(values, weights), key=lambda pair: pair[0])
    total_weight = sum(weights)
    threshold = probability * total_weight
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= threshold:
            return value
    return ordered[-1][0]


def _normalized_rmse(xs: list[float], ys: list[float]) -> float:
    mean_scale = max((sum(xs) + sum(ys)) / (2.0 * len(xs)), 1.0)
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(xs, ys)) / len(xs)) / mean_scale


def _normalized_param_distance(
    a: dict[str, float],
    b: dict[str, float],
    parameter_ranges: dict[str, float],
) -> float:
    common = sorted(set(a).intersection(b))
    if not common:
        return 0.0
    distances = []
    for key in common:
        scale = max(parameter_ranges.get(key, 0.0), 1e-9)
        distances.append(abs(a[key] - b[key]) / scale)
    return sum(distances) / len(distances)


def analyze_identifiability(
    particle_trajectories: list[dict[str, Any]],
    observations: list[dict[str, Any]] | None = None,
    active_drugs: list[str] | None = None,
    trajectory_tolerance_fraction: float = 0.08,
    parameter_gap_fraction: float = 0.25,
    max_pair_scan: int = 250,
) -> dict[str, Any]:
    """Classify parameters and find similar trajectories from different params."""

    if not particle_trajectories:
        raise ValueError("particle_trajectories must not be empty.")
    weights = [float(particle.get("weight", 0.0) or 0.0) for particle in particle_trajectories]
    if sum(weights) <= 0:
        weights = [1.0 / len(particle_trajectories)] * len(particle_trajectories)
    else:
        total = sum(weights)
        weights = [weight / total for weight in weights]
    effective_sample_size = 1.0 / sum(weight * weight for weight in weights)
    low_effective_sample_size = effective_sample_size < max(10.0, 0.02 * len(particle_trajectories))
    active_drug_set = set(active_drugs) if active_drugs is not None else None

    parameter_keys = sorted(
        {
            key
            for particle in particle_trajectories
            for key in flatten_driver_params(particle["parameters"])
        }
    )
    parameter_ranges: dict[str, float] = {}
    constrained: list[str] = []
    weakly_constrained: list[str] = []
    prior_dominated: list[str] = []
    parameter_reports: dict[str, dict[str, float | str]] = {}
    sparse_observations = observations is not None and len(observations) < 3

    for key in parameter_keys:
        values = [flatten_driver_params(particle["parameters"])[key] for particle in particle_trajectories]
        prior_width = quantile(values, 0.9) - quantile(values, 0.1)
        posterior_width = _weighted_quantile(values, weights, 0.9) - _weighted_quantile(
            values, weights, 0.1
        )
        parameter_ranges[key] = max(values) - min(values)
        ratio = posterior_width / prior_width if prior_width > 0 else 1.0
        reason = ""
        if (
            active_drug_set is not None
            and key.startswith("drug_sensitivity.")
            and key.split(".", 1)[1] not in active_drug_set
        ):
            classification = "prior-dominated"
            prior_dominated.append(key)
            reason = "drug is not present in the treatment schedule"
        elif sparse_observations:
            classification = "prior-dominated"
            prior_dominated.append(key)
            reason = "too few observations"
        elif ratio < 0.35:
            classification = "constrained"
            if low_effective_sample_size:
                classification = "weakly constrained"
                weakly_constrained.append(key)
                reason = "low effective sample size prevents a strong constrained call"
            else:
                constrained.append(key)
        elif ratio < 0.55:
            classification = "weakly constrained"
            weakly_constrained.append(key)
        else:
            classification = "prior-dominated"
            prior_dominated.append(key)
        parameter_reports[key] = {
            "prior_width_80": prior_width,
            "posterior_width_80": posterior_width,
            "width_ratio": ratio,
            "classification": classification,
            "reason": reason,
        }

    similar_pairs: list[dict[str, Any]] = []
    scan_particles = particle_trajectories[:max_pair_scan]
    for i, first in enumerate(scan_particles):
        first_params = flatten_driver_params(first["parameters"])
        for second in scan_particles[i + 1 :]:
            distance = _normalized_rmse(
                [float(value) for value in first["predicted_volume_ml"]],
                [float(value) for value in second["predicted_volume_ml"]],
            )
            if distance > trajectory_tolerance_fraction:
                continue
            param_gap = _normalized_param_distance(
                first_params,
                flatten_driver_params(second["parameters"]),
                parameter_ranges,
            )
            if param_gap >= parameter_gap_fraction:
                similar_pairs.append(
                    {
                        "particle_id_a": first["particle_id"],
                        "particle_id_b": second["particle_id"],
                        "trajectory_normalized_rmse": distance,
                        "parameter_gap_fraction": param_gap,
                    }
                )
            if len(similar_pairs) >= 20:
                break
        if len(similar_pairs) >= 20:
            break

    recommendation = "Parameter-level explanations should be limited to constrained parameters."
    if prior_dominated or similar_pairs:
        recommendation = (
            "Reduce free parameters, fix weak parameters to curated priors, or show trajectory-level "
            "uncertainty without parameter-level certainty."
        )

    warnings = [
        "Identifiability report is based on synthetic/demo particle behavior, not clinical evidence."
    ]
    if low_effective_sample_size:
        warnings.append(
            "Low effective sample size; narrow posterior intervals may reflect too few surviving particles."
        )

    return {
        "constrained_parameters": constrained,
        "weakly_constrained_parameters": weakly_constrained,
        "prior_dominated_parameters": prior_dominated,
        "parameter_reports": parameter_reports,
        "similar_trajectory_pairs": similar_pairs,
        "recommendation": recommendation,
        "effective_sample_size": effective_sample_size,
        "warnings": warnings,
    }
