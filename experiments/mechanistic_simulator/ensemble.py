"""Ensemble simulation for v0 volume trajectories."""

from __future__ import annotations

from typing import Any

from .summarize import summarize_trajectories
from .volume_ode import SIMULATION_VERSION, simulate_volume_trajectory


def simulate_volume_ensemble(
    initial_volume_ml: float,
    treatment_schedule: dict[str, Any],
    parameter_particles: list[dict[str, Any]],
    output_days: list[float],
    dt_days: float = 0.5,
    residual_threshold_ml: float = 1.0,
) -> dict[str, Any]:
    """Run all parameter particles and return particle-level and summary output."""

    particle_trajectories: list[dict[str, Any]] = []
    for index, params in enumerate(parameter_particles):
        particle_id = params.get("particle_id", f"p{index:06d}")
        result = simulate_volume_trajectory(
            initial_volume_ml=initial_volume_ml,
            treatment_schedule=treatment_schedule,
            params=params,
            output_days=output_days,
            dt_days=dt_days,
        )
        records = result["trajectory"]
        particle_trajectories.append(
            {
                "particle_id": particle_id,
                "parameters": params,
                "times": list(result["times"]),
                "predicted_volume_ml": [record["tumor_volume_ml"] for record in records],
                "predicted_longest_diameter_cm": [
                    record["predicted_longest_diameter_cm"] for record in records
                ],
                "likelihood_placeholder": None,
                "weight_placeholder": None,
                "warnings": list(result["warnings"]),
            }
        )

    summary = summarize_trajectories(
        particle_trajectories,
        residual_threshold_ml=residual_threshold_ml,
    )

    return {
        "simulation_version": SIMULATION_VERSION,
        "times": list(output_days),
        "n_particles": len(particle_trajectories),
        "particle_trajectories": particle_trajectories,
        **summary,
    }
