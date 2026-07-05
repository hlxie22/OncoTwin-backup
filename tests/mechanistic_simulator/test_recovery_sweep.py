from __future__ import annotations

import unittest

from experiments.mechanistic_simulator.recovery_sweep import (
    baseline_predictions,
    make_prior_variant,
    run_recovery_sweep,
)
from tests.mechanistic_simulator.helpers import load_json


class RecoverySweepTests(unittest.TestCase):
    def test_baseline_predictions_have_expected_methods(self) -> None:
        observations = [
            {"day": 0.0, "tumor_volume_ml": 30.0},
            {"day": 42.0, "tumor_volume_ml": 15.0},
            {"day": 84.0, "tumor_volume_ml": 8.0},
        ]
        predictions = baseline_predictions(observations, 126.0)
        self.assertEqual(
            set(predictions),
            {"last_observation", "last_slope", "linear", "exponential"},
        )
        self.assertTrue(all(value >= 0 for value in predictions.values()))

    def test_active_drugs_only_prior_removes_inactive_sensitivities(self) -> None:
        prior = load_json("fixtures/mechanistic_simulator/params/generic_volume_prior.json")
        variant = make_prior_variant(
            prior,
            "active_drugs_only",
            ["anthracycline", "taxane"],
            assumed_noise_fraction=0.12,
        )
        self.assertEqual(
            set(variant["distributions"]["drug_sensitivity"]),
            {"anthracycline", "taxane"},
        )
        self.assertEqual(set(variant["fixed"]["drug_decay"]), {"anthracycline", "taxane"})

    def test_tiny_recovery_sweep_produces_summaries(self) -> None:
        case = load_json("fixtures/mechanistic_simulator/cases/longitudinal_measurement_demo_case.json")
        schedule = load_json(case["treatment_schedule"]["path"])
        report = run_recovery_sweep(
            case=case,
            schedule=schedule,
            base_prior=load_json("fixtures/mechanistic_simulator/params/generic_volume_prior.json"),
            truth_params=load_json("fixtures/mechanistic_simulator/params/high_response_params.json"),
            seeds=[1],
            particle_counts=[20],
            assumed_noise_levels=[0.12],
            variants=["full", "fixed_core"],
        )
        self.assertEqual(report["n_runs"], 2)
        self.assertIn("by_variant", report["summaries"])
        self.assertIn("heldout_baseline_comparison", report["summaries"])
        self.assertIn("posterior_particle_mean", report["summaries"]["heldout_baseline_comparison"])
        self.assertTrue(report["insights"])


if __name__ == "__main__":
    unittest.main()
