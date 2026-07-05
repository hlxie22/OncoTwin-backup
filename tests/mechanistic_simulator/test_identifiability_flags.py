from __future__ import annotations

import unittest

from experiments.mechanistic_simulator.ensemble import simulate_volume_ensemble
from experiments.mechanistic_simulator.identifiability import analyze_identifiability
from experiments.mechanistic_simulator.params import sample_volume_params
from experiments.mechanistic_simulator.synthetic_fit import reweight_particles_from_observations
from experiments.mechanistic_simulator.volume_ode import simulate_volume_trajectory
from tests.mechanistic_simulator.helpers import load_json


class IdentifiabilityFlagTests(unittest.TestCase):
    def test_sparse_observations_mark_parameters_as_prior_dominated(self) -> None:
        case = load_json("fixtures/mechanistic_simulator/cases/tnbc_demo_case.json")
        schedule = load_json(case["treatment_schedule"]["path"])
        prior = load_json("fixtures/mechanistic_simulator/params/generic_volume_prior.json")
        particles = sample_volume_params(prior, 40, seed=3)
        ensemble = simulate_volume_ensemble(
            case["baseline_measurement"]["tumor_volume_ml"],
            schedule,
            particles,
            [0, 42],
        )
        report = analyze_identifiability(
            ensemble["particle_trajectories"],
            observations=[{"day": 0, "tumor_volume_ml": 28.0}],
        )
        self.assertIn("growth_rate", report["prior_dominated_parameters"])
        self.assertIn("resistant_fraction", report["prior_dominated_parameters"])

    def test_more_observations_reduce_but_do_not_eliminate_uncertainty(self) -> None:
        case = load_json("fixtures/mechanistic_simulator/cases/longitudinal_measurement_demo_case.json")
        schedule = load_json(case["treatment_schedule"]["path"])
        truth_params = load_json("fixtures/mechanistic_simulator/params/high_response_params.json")
        truth = simulate_volume_trajectory(
            case["baseline_measurement"]["tumor_volume_ml"],
            schedule,
            truth_params,
            [0, 42, 84, 126],
            dt_days=0.25,
        )
        observations = [
            {"day": row["day"], "tumor_volume_ml": row["tumor_volume_ml"]}
            for row in truth["trajectory"]
        ]
        prior = load_json("fixtures/mechanistic_simulator/params/generic_volume_prior.json")
        fit = reweight_particles_from_observations(
            case["baseline_measurement"]["tumor_volume_ml"],
            schedule,
            sample_volume_params(prior, 220, seed=17),
            observations,
        )
        report = analyze_identifiability(fit["particle_trajectories"], observations=observations)
        classifications = {
            details["classification"]
            for details in report["parameter_reports"].values()
        }
        self.assertTrue(classifications.intersection({"constrained", "weakly constrained"}))
        self.assertTrue(report["prior_dominated_parameters"])

    def test_similar_trajectories_from_different_parameters_are_reported(self) -> None:
        particles = [
            {
                "particle_id": "a",
                "parameters": {
                    "growth_rate": 0.002,
                    "carrying_capacity_ml": 200.0,
                    "resistant_fraction": 0.02,
                    "resistant_sensitivity_scale": 0.1,
                    "observation_noise_fraction": 0.1,
                    "drug_sensitivity": {"demo": 0.02},
                },
                "times": [0, 7, 14],
                "predicted_volume_ml": [10.0, 9.0, 8.0],
                "warnings": [],
            },
            {
                "particle_id": "b",
                "parameters": {
                    "growth_rate": 0.012,
                    "carrying_capacity_ml": 450.0,
                    "resistant_fraction": 0.34,
                    "resistant_sensitivity_scale": 0.2,
                    "observation_noise_fraction": 0.1,
                    "drug_sensitivity": {"demo": 0.12},
                },
                "times": [0, 7, 14],
                "predicted_volume_ml": [10.0, 9.1, 8.1],
                "warnings": [],
            },
        ]
        report = analyze_identifiability(particles)
        self.assertTrue(report["similar_trajectory_pairs"])


if __name__ == "__main__":
    unittest.main()
