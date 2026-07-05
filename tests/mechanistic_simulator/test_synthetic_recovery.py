from __future__ import annotations

import unittest

from experiments.mechanistic_simulator.params import sample_volume_params
from experiments.mechanistic_simulator.synthetic_fit import reweight_particles_from_observations
from experiments.mechanistic_simulator.volume_ode import simulate_volume_trajectory
from tests.mechanistic_simulator.helpers import REPO_ROOT, load_json


def exact_observations_from_params(params_path: str) -> tuple[dict, dict, list[dict]]:
    case = load_json("fixtures/mechanistic_simulator/cases/longitudinal_measurement_demo_case.json")
    schedule = load_json(case["treatment_schedule"]["path"])
    params = load_json(params_path)
    trajectory = simulate_volume_trajectory(
        initial_volume_ml=case["baseline_measurement"]["tumor_volume_ml"],
        treatment_schedule=schedule,
        params=params,
        output_days=[0, 42, 84, 126],
        dt_days=0.25,
    )
    observations = [
        {
            "day": row["day"],
            "tumor_volume_ml": row["tumor_volume_ml"],
            "source": "synthetic",
            "confidence": "demo",
        }
        for row in trajectory["trajectory"]
    ]
    return case, schedule, observations


class SyntheticRecoveryTests(unittest.TestCase):
    def test_strong_response_data_favors_high_sensitivity_particles(self) -> None:
        case, schedule, observations = exact_observations_from_params(
            "fixtures/mechanistic_simulator/params/high_response_params.json"
        )
        prior = load_json("fixtures/mechanistic_simulator/params/generic_volume_prior.json")
        particles = sample_volume_params(prior, 240, seed=7)
        report = reweight_particles_from_observations(
            case["baseline_measurement"]["tumor_volume_ml"],
            schedule,
            particles,
            observations,
        )
        self.assertLess(report["posterior_rmse"], report["prior_rmse"])
        self.assertGreater(
            report["posterior_parameter_means"]["drug_sensitivity.anthracycline"],
            report["prior_parameter_means"]["drug_sensitivity.anthracycline"],
        )
        self.assertGreater(
            report["posterior_parameter_means"]["drug_sensitivity.taxane"],
            report["prior_parameter_means"]["drug_sensitivity.taxane"],
        )

    def test_weak_response_data_favors_low_sensitivity_particles(self) -> None:
        case, schedule, observations = exact_observations_from_params(
            "fixtures/mechanistic_simulator/params/weak_response_params.json"
        )
        prior = load_json("fixtures/mechanistic_simulator/params/generic_volume_prior.json")
        particles = sample_volume_params(prior, 240, seed=11)
        report = reweight_particles_from_observations(
            case["baseline_measurement"]["tumor_volume_ml"],
            schedule,
            particles,
            observations,
        )
        self.assertLess(report["posterior_rmse"], report["prior_rmse"])
        self.assertLess(
            report["posterior_parameter_means"]["drug_sensitivity.anthracycline"],
            report["prior_parameter_means"]["drug_sensitivity.anthracycline"],
        )

    def test_noisy_observations_do_not_collapse_uncertainty_too_aggressively(self) -> None:
        case, schedule, observations = exact_observations_from_params(
            "fixtures/mechanistic_simulator/params/high_response_params.json"
        )
        noisy = [
            {**observation, "tumor_volume_ml": observation["tumor_volume_ml"] * 1.15}
            for observation in observations
        ]
        prior = load_json("fixtures/mechanistic_simulator/params/generic_volume_prior.json")
        particles = sample_volume_params(prior, 240, seed=13)
        report = reweight_particles_from_observations(
            case["baseline_measurement"]["tumor_volume_ml"],
            schedule,
            particles,
            noisy,
        )
        self.assertGreater(report["effective_sample_size"], 1.5)


if __name__ == "__main__":
    unittest.main()
