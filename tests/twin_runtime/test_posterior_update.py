from __future__ import annotations

import copy
import unittest

from experiments.twin_runtime.posterior import (
    VolumeObservation,
    resolve_volume_observation_noise_fraction,
    update_volume_posterior,
)
from experiments.v0.mechanistic_simulator.volume_ode import simulate_volume_trajectory


def _schedule() -> dict[str, object]:
    return {
        "schedule_id": "test_chemo",
        "regimen_name": "Test A/C-T chemotherapy",
        "total_duration_days": 84,
        "events": [
            {"drug": "anthracycline", "day": 0, "relative_dose": 1.0},
            {"drug": "anthracycline", "day": 14, "relative_dose": 1.0},
            {"drug": "anthracycline", "day": 28, "relative_dose": 1.0},
            {"drug": "taxane", "day": 42, "relative_dose": 1.0},
            {"drug": "taxane", "day": 49, "relative_dose": 1.0},
        ],
    }


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


class PosteriorUpdateRuntimeTest(unittest.TestCase):
    def test_update_weights_particle_matching_observed_shrinkage(self):
        strong = _params(
            "strong_response",
            anthracycline_sensitivity=0.12,
            taxane_sensitivity=0.11,
            resistant_fraction=0.04,
        )
        weak = _params(
            "weak_response",
            anthracycline_sensitivity=0.02,
            taxane_sensitivity=0.02,
            resistant_fraction=0.25,
        )
        strong_trajectory = simulate_volume_trajectory(
            initial_volume_ml=30.0,
            treatment_schedule=_schedule(),
            params=strong,
            output_days=[42.0],
        )
        observed_volume = strong_trajectory["trajectory"][0]["tumor_volume_ml"]

        result = update_volume_posterior(
            initial_volume_ml=30.0,
            treatment_schedule=_schedule(),
            parameter_particles=[strong, weak],
            observations=[
                {
                    "day": 42,
                    "tumor_volume_ml": observed_volume,
                    "source": "mask_derived",
                    "confidence": "high",
                    "segmentation_qc": "high",
                }
            ],
            prediction_days=[84],
            likelihood_noise_fraction=0.05,
        )

        weights = {
            row["particle_id"]: row["weight"]
            for row in result["particle_trajectories"]
        }
        self.assertGreater(weights["strong_response"], weights["weak_response"])
        self.assertIn("day 42", result["update_explanation"])
        self.assertIn("posterior median", result["update_explanation"])
        self.assertEqual(result["fallback_status"], "not_needed")
        self.assertIn(84.0, result["posterior_trajectory_summary"]["times"])

    def test_low_qc_widens_likelihood_noise(self):
        high_qc = VolumeObservation(
            day=21,
            tumor_volume_ml=20,
            source="mask_derived",
            confidence="high",
            segmentation_qc="high",
        )
        low_qc = VolumeObservation(
            day=21,
            tumor_volume_ml=20,
            source="mask_derived",
            confidence="low",
            segmentation_qc="low",
        )

        self.assertGreater(
            resolve_volume_observation_noise_fraction(low_qc),
            resolve_volume_observation_noise_fraction(high_qc),
        )

    def test_failed_qc_observation_is_ignored_by_default(self):
        params = _params(
            "particle",
            anthracycline_sensitivity=0.08,
            taxane_sensitivity=0.08,
            resistant_fraction=0.10,
        )

        result = update_volume_posterior(
            initial_volume_ml=30.0,
            treatment_schedule=_schedule(),
            parameter_particles=[params],
            observations=[
                {
                    "day": 14,
                    "tumor_volume_ml": 25,
                    "source": "mask_derived",
                    "confidence": "high",
                    "segmentation_qc": "failed",
                },
                {
                    "day": 28,
                    "tumor_volume_ml": 20,
                    "source": "mask_derived",
                    "confidence": "high",
                    "segmentation_qc": "high",
                },
            ],
        )

        self.assertEqual(result["n_observations"], 1)
        self.assertEqual(len(result["skipped_observations"]), 1)
        self.assertTrue(any("ignored failed-QC" in warning for warning in result["warnings"]))

    def test_low_ess_surfaces_tempered_smc_recommendation(self):
        strong = _params(
            "strong_response",
            anthracycline_sensitivity=0.12,
            taxane_sensitivity=0.11,
            resistant_fraction=0.04,
        )
        weak = _params(
            "weak_response",
            anthracycline_sensitivity=0.02,
            taxane_sensitivity=0.02,
            resistant_fraction=0.25,
        )
        resistant = _params(
            "resistant",
            anthracycline_sensitivity=0.05,
            taxane_sensitivity=0.04,
            resistant_fraction=0.40,
        )
        trajectory = simulate_volume_trajectory(
            initial_volume_ml=30.0,
            treatment_schedule=_schedule(),
            params=strong,
            output_days=[42.0],
        )
        observed_volume = trajectory["trajectory"][0]["tumor_volume_ml"]

        result = update_volume_posterior(
            initial_volume_ml=30.0,
            treatment_schedule=_schedule(),
            parameter_particles=[strong, weak, resistant],
            observations=[
                {
                    "day": 42,
                    "tumor_volume_ml": observed_volume,
                    "source": "mask_derived",
                    "confidence": "high",
                    "segmentation_qc": "high",
                }
            ],
            likelihood_noise_fraction=0.01,
            ess_threshold_fraction=0.9,
        )

        self.assertEqual(result["fallback_status"], "tempered_smc_recommended")
        self.assertLess(result["effective_sample_size_fraction"], 0.9)
        self.assertIn("low effective sample size", result["uncertainty_summary"]["top_drivers"])


if __name__ == "__main__":
    unittest.main()
