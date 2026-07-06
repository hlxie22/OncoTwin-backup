"""Synthetic parameter-recovery utilities for the v0 simulator."""

from __future__ import annotations

import math
import random
from typing import Any

from .summarize import flatten_driver_params, quantile
from .volume_ode import simulate_volume_trajectory


def make_noisy_observations(
    trajectory: dict[str, Any],
    noise_fraction: float,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    observations: list[dict[str, Any]] = []
    for record in trajectory["trajectory"]:
        true_volume = float(record["tumor_volume_ml"])
        sigma = max(noise_fraction * max(true_volume, 1.0), 0.1)
        observed = max(0.0, rng.gauss(true_volume, sigma))
        observations.append(
            {
                "day": record["day"],
                "tumor_volume_ml": observed,
                "source": "synthetic",
                "confidence": "demo",
                "true_tumor_volume_ml": true_volume,
            }
        )
    return observations


def _rmse(predicted: list[float], observed: list[float]) -> float:
    return math.sqrt(
        sum((pred - obs) ** 2 for pred, obs in zip(predicted, observed)) / len(observed)
    )


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    total_weight = sum(weights)
    if total_weight == 0:
        return sum(values) / len(values)
    return sum(value * weight for value, weight in zip(values, weights)) / total_weight


def reweight_particles_from_observations(
    initial_volume_ml: float,
    treatment_schedule: dict[str, Any],
    parameter_particles: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    dt_days: float = 0.5,
    likelihood_noise_fraction: float | None = None,
    prediction_days: list[float] | None = None,
) -> dict[str, Any]:
    """Fit by particle reweighting against longitudinal volume observations."""

    observation_days = [float(obs["day"]) for obs in observations]
    observed_volumes = [float(obs["tumor_volume_ml"]) for obs in observations]
    output_days = sorted(set(observation_days + [float(day) for day in (prediction_days or [])]))
    log_likelihoods: list[float] = []
    particle_trajectories: list[dict[str, Any]] = []

    for index, params in enumerate(parameter_particles):
        result = simulate_volume_trajectory(
            initial_volume_ml=initial_volume_ml,
            treatment_schedule=treatment_schedule,
            params=params,
            output_days=output_days,
            dt_days=dt_days,
        )
        predicted_all = [float(record["tumor_volume_ml"]) for record in result["trajectory"]]
        predicted_by_day = {
            float(day): predicted for day, predicted in zip(output_days, predicted_all)
        }
        predicted_scored = [predicted_by_day[day] for day in observation_days]
        log_likelihood = 0.0
        noise_fraction = (
            float(likelihood_noise_fraction)
            if likelihood_noise_fraction is not None
            else float(params.get("observation_noise_fraction", 0.12))
        )
        for predicted_volume, observed_volume in zip(predicted_scored, observed_volumes):
            sigma = max(noise_fraction * max(observed_volume, 1.0), 0.25)
            residual = observed_volume - predicted_volume
            log_likelihood += -0.5 * (residual / sigma) ** 2 - math.log(sigma)
        log_likelihoods.append(log_likelihood)
        particle_trajectories.append(
            {
                "particle_id": params.get("particle_id", f"p{index:06d}"),
                "parameters": params,
                "times": output_days,
                "predicted_volume_ml": predicted_all,
                "predicted_longest_diameter_cm": [
                    record["predicted_longest_diameter_cm"] for record in result["trajectory"]
                ],
                "likelihood": log_likelihood,
                "weight": None,
                "warnings": list(result["warnings"]),
            }
        )

    max_log_likelihood = max(log_likelihoods)
    raw_weights = [math.exp(value - max_log_likelihood) for value in log_likelihoods]
    weight_total = sum(raw_weights)
    weights = [value / weight_total for value in raw_weights]
    for particle, weight in zip(particle_trajectories, weights):
        particle["weight"] = weight

    prior_mean_prediction = [
        sum(
            particle["predicted_volume_ml"][output_days.index(day)]
            for particle in particle_trajectories
        )
        / len(particle_trajectories)
        for day in observation_days
    ]
    posterior_mean_prediction = [
        sum(
            particle["predicted_volume_ml"][output_days.index(day)] * float(particle["weight"])
            for particle in particle_trajectories
        )
        for day in observation_days
    ]
    prior_mean_all_times = [
        sum(particle["predicted_volume_ml"][day_index] for particle in particle_trajectories)
        / len(particle_trajectories)
        for day_index in range(len(output_days))
    ]
    posterior_mean_all_times = [
        sum(
            particle["predicted_volume_ml"][day_index] * float(particle["weight"])
            for particle in particle_trajectories
        )
        for day_index in range(len(output_days))
    ]

    parameter_keys = sorted(
        {
            key
            for particle in particle_trajectories
            for key in flatten_driver_params(particle["parameters"])
        }
    )
    prior_parameter_means: dict[str, float] = {}
    posterior_parameter_means: dict[str, float] = {}
    parameter_interval_report: dict[str, dict[str, float]] = {}
    for key in parameter_keys:
        values = [flatten_driver_params(particle["parameters"])[key] for particle in particle_trajectories]
        prior_parameter_means[key] = sum(values) / len(values)
        posterior_parameter_means[key] = _weighted_mean(values, weights)
        ordered_pairs = sorted(zip(values, weights), key=lambda pair: pair[0])
        cumulative = 0.0
        weighted_q10 = ordered_pairs[0][0]
        weighted_q90 = ordered_pairs[-1][0]
        for value, weight in ordered_pairs:
            cumulative += weight
            if cumulative >= 0.1:
                weighted_q10 = value
                break
        cumulative = 0.0
        for value, weight in ordered_pairs:
            cumulative += weight
            if cumulative >= 0.9:
                weighted_q90 = value
                break
        prior_width = quantile(values, 0.9) - quantile(values, 0.1)
        posterior_width = weighted_q90 - weighted_q10
        parameter_interval_report[key] = {
            "prior_width_80": prior_width,
            "posterior_width_80": posterior_width,
            "width_ratio": posterior_width / prior_width if prior_width > 0 else 1.0,
        }

    effective_sample_size = 1.0 / sum(weight * weight for weight in weights)
    return {
        "observations": observations,
        "scored_times": observation_days,
        "prediction_times": output_days,
        "particle_trajectories": particle_trajectories,
        "prior_mean_prediction_ml": prior_mean_prediction,
        "posterior_mean_prediction_ml": posterior_mean_prediction,
        "prior_mean_all_times_ml": prior_mean_all_times,
        "posterior_mean_all_times_ml": posterior_mean_all_times,
        "prior_rmse": _rmse(prior_mean_prediction, observed_volumes),
        "posterior_rmse": _rmse(posterior_mean_prediction, observed_volumes),
        "prior_parameter_means": prior_parameter_means,
        "posterior_parameter_means": posterior_parameter_means,
        "parameter_interval_report": parameter_interval_report,
        "effective_sample_size": effective_sample_size,
        "warnings": [
            "Synthetic fitting is a particle-reweighting sanity check, not clinical calibration."
        ],
    }
