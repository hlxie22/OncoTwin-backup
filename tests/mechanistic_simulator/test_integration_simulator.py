from __future__ import annotations

import unittest

from experiments.mechanistic_simulator.ensemble import simulate_volume_ensemble
from experiments.mechanistic_simulator.params import sample_volume_params
from experiments.mechanistic_simulator.safety import assert_no_unsafe_language
from tests.mechanistic_simulator.helpers import REPO_ROOT, load_json


class SimulatorIntegrationTests(unittest.TestCase):
    def test_fixture_case_prior_ensemble_output_is_app_ready_json_shape(self) -> None:
        case = load_json("fixtures/mechanistic_simulator/cases/tnbc_demo_case.json")
        schedule = load_json(case["treatment_schedule"]["path"])
        prior = load_json("fixtures/mechanistic_simulator/params/generic_volume_prior.json")
        result = simulate_volume_ensemble(
            case["baseline_measurement"]["tumor_volume_ml"],
            schedule,
            sample_volume_params(prior, 30, seed=19),
            [0, 42, 84, 126, 168],
        )
        schema = load_json("schemas/mechanistic_simulator/simulation_output.schema.json")
        for required_key in schema["required"]:
            self.assertIn(required_key, result)
        self.assertEqual(result["n_particles"], 30)
        self.assertEqual(len(result["particle_trajectories"]), 30)
        self.assertTrue(result["warnings"])
        self.assertIn("parameters", result["particle_trajectories"][0])
        self.assertIn("likelihood_placeholder", result["particle_trajectories"][0])
        self.assertIn("weight_placeholder", result["particle_trajectories"][0])
        for median, interval_80, interval_95 in zip(
            result["median_volume_ml"],
            result["interval_80_volume_ml"],
            result["interval_95_volume_ml"],
        ):
            self.assertLessEqual(interval_80[0], median)
            self.assertLessEqual(interval_95[0], interval_80[0])
            self.assertGreaterEqual(interval_80[1], median)
            self.assertGreaterEqual(interval_95[1], interval_80[1])
        assert_no_unsafe_language(result)

    def test_schema_files_exist(self) -> None:
        for filename in (
            "tumor_measurement.schema.json",
            "treatment_schedule.schema.json",
            "mechanistic_params.schema.json",
            "simulation_output.schema.json",
        ):
            self.assertTrue((REPO_ROOT / "schemas/mechanistic_simulator" / filename).exists())


if __name__ == "__main__":
    unittest.main()
