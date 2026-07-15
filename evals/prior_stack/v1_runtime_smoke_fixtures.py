"""Deterministic smoke fixtures for V1 runtime-layer evals.

These helpers intentionally use tiny synthetic particles. They verify that the
newly added V1 runtime layers are wired and produce inspectable artifacts; they
are not clinical performance evidence.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.twin_runtime.explanations import build_twin_update_explanation
from experiments.twin_runtime.posterior import update_volume_posterior
from experiments.twin_runtime.scenario_lab import run_scenario_lab
from experiments.v0.mechanistic_simulator.volume_ode import simulate_volume_trajectory


INITIAL_VOLUME_ML = 30.0
OUTPUT_DAYS = (21.0, 42.0, 84.0)


def chemo_schedule(relative_dose: float = 1.0) -> dict[str, object]:
    return {
        "schedule_id": "v1_runtime_smoke_chemo",
        "regimen_name": "V1 runtime smoke A/C-T chemotherapy",
        "total_duration_days": 84,
        "events": [
            {"drug": "anthracycline", "day": 0, "relative_dose": relative_dose},
            {"drug": "anthracycline", "day": 14, "relative_dose": relative_dose},
            {"drug": "anthracycline", "day": 28, "relative_dose": relative_dose},
            {"drug": "taxane", "day": 42, "relative_dose": relative_dose},
            {"drug": "taxane", "day": 49, "relative_dose": relative_dose},
        ],
    }


def observation_only_schedule() -> dict[str, object]:
    return {
        "schedule_id": "v1_runtime_smoke_observation_only",
        "regimen_name": "Observation-only modeled comparator",
        "total_duration_days": 84,
        "events": [],
    }


def parameter_particles() -> list[dict[str, object]]:
    return [
        _params(
            "strong_response",
            anthracycline_sensitivity=0.12,
            taxane_sensitivity=0.11,
            resistant_fraction=0.04,
        ),
        _params(
            "moderate_response",
            anthracycline_sensitivity=0.06,
            taxane_sensitivity=0.055,
            resistant_fraction=0.12,
        ),
        _params(
            "weak_response",
            anthracycline_sensitivity=0.02,
            taxane_sensitivity=0.02,
            resistant_fraction=0.25,
        ),
    ]


def deterministic_observations() -> list[dict[str, object]]:
    strong = parameter_particles()[0]
    trajectory = simulate_volume_trajectory(
        initial_volume_ml=INITIAL_VOLUME_ML,
        treatment_schedule=chemo_schedule(),
        params=strong,
        output_days=list(OUTPUT_DAYS),
    )["trajectory"]
    by_day = {float(row["day"]): float(row["tumor_volume_ml"]) for row in trajectory}
    return [
        {
            "day": 21.0,
            "tumor_volume_ml": by_day[21.0],
            "source": "mask_derived",
            "confidence": "high",
            "segmentation_qc": "high",
            "observation_id": "smoke_day21",
        },
        {
            "day": 42.0,
            "tumor_volume_ml": by_day[42.0],
            "source": "mask_derived",
            "confidence": "high",
            "segmentation_qc": "high",
            "observation_id": "smoke_day42",
        },
    ]


def heldout_final_observation() -> dict[str, object]:
    strong = parameter_particles()[0]
    trajectory = simulate_volume_trajectory(
        initial_volume_ml=INITIAL_VOLUME_ML,
        treatment_schedule=chemo_schedule(),
        params=strong,
        output_days=[84.0],
    )["trajectory"]
    return {
        "day": 84.0,
        "tumor_volume_ml": float(trajectory[0]["tumor_volume_ml"]),
        "source": "mask_derived",
        "confidence": "high",
        "segmentation_qc": "high",
        "observation_id": "smoke_day84_heldout",
    }


def posterior_update_fixture(
    *,
    observations: Sequence[Mapping[str, object]] | None = None,
    prediction_days: Sequence[float] = (84.0,),
    likelihood_noise_fraction: float = 0.06,
) -> dict[str, object]:
    return update_volume_posterior(
        initial_volume_ml=INITIAL_VOLUME_ML,
        treatment_schedule=chemo_schedule(),
        parameter_particles=parameter_particles(),
        observations=list(observations or deterministic_observations()),
        prediction_days=list(prediction_days),
        likelihood_noise_fraction=likelihood_noise_fraction,
    )


def scenario_inputs() -> list[dict[str, object]]:
    return [
        {
            "scenario_id": "observation_only",
            "label": "Observation only",
            "reference": True,
            "treatment_schedule": observation_only_schedule(),
        },
        {
            "scenario_id": "continue_chemo",
            "label": "Continue chemo",
            "treatment_schedule": chemo_schedule(),
        },
        {
            "scenario_id": "unsafe_overdose",
            "label": "Unsafe overdose preflight check",
            "treatment_schedule": chemo_schedule(relative_dose=2.0),
        },
    ]


def scenario_lab_fixture(
    *,
    posterior_update: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    return run_scenario_lab(
        posterior_update=with_positive_particle_weights(
            posterior_update or positive_weight_posterior_update_fixture()
        ),
        scenarios=scenario_inputs(),
        output_days=[84.0],
        residual_burden_threshold_ml=20.0,
    )


def positive_weight_posterior_update_fixture() -> dict[str, object]:
    return with_positive_particle_weights(
        posterior_update_fixture(likelihood_noise_fraction=0.12)
    )


def with_positive_particle_weights(posterior_update: Mapping[str, Any]) -> dict[str, object]:
    posterior = dict(posterior_update)
    rows = [dict(row) for row in posterior["particle_trajectories"]]
    positive_weights = [max(float(row.get("weight", 0.0)), 1e-12) for row in rows]
    total = sum(positive_weights)
    for row, weight in zip(rows, positive_weights):
        row["weight"] = weight / total
    posterior["particle_trajectories"] = rows
    return posterior


def explanation_fixture() -> dict[str, object]:
    posterior_update = posterior_update_fixture()
    scenario_lab = scenario_lab_fixture(posterior_update=posterior_update)
    return build_twin_update_explanation(
        posterior_update=posterior_update,
        scenario_lab=scenario_lab,
        prior_context={
            "layer_contributions": [
                {"layer": "population_prior", "rule_id": "smoke_layer1_population"},
                {"layer": "mri_feature_rules", "rule_id": "smoke_layer4_mri"},
            ]
        },
        audience="clinician",
    )


def write_json(path: Path | None, payload: Mapping[str, object]) -> str | None:
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(path)


def write_markdown_report(
    path: Path | None,
    *,
    title: str,
    summary: str,
    metrics: Mapping[str, object],
    warnings: Sequence[str] = (),
) -> str | None:
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", "", summary, "", "## Metrics", ""]
    for key, value in metrics.items():
        lines.append(f"- {key}: {value}")
    if warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in warnings:
            lines.append(f"- {warning}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _params(
    particle_id: str,
    *,
    anthracycline_sensitivity: float,
    taxane_sensitivity: float,
    resistant_fraction: float,
) -> dict[str, object]:
    return {
        "particle_id": particle_id,
        "growth_law": "logistic",
        "growth_rate": 0.006,
        "carrying_capacity_ml": 260.0,
        "drug_sensitivity": {
            "anthracycline": anthracycline_sensitivity,
            "taxane": taxane_sensitivity,
        },
        "drug_ec50": {"anthracycline": 0.5, "taxane": 0.5},
        "drug_decay": {"anthracycline": 0.25, "taxane": 0.2},
        "resistant_fraction": resistant_fraction,
        "resistant_sensitivity_scale": 0.08,
        "observation_noise_fraction": 0.10,
    }
