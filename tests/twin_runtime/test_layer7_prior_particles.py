from __future__ import annotations

import copy
import unittest

from experiments.prior_builder.bounds import (
    validate_parameter_bounds,
)
from experiments.twin_runtime.layer7_prior_particles import (
    DEFAULT_POLICY_PATH,
    build_layer7_prior_particles,
)


class Layer7PriorParticlesTest(unittest.TestCase):
    def test_particle_count_and_ids(self):
        artifact = self._build(
            seed=101,
            count=32,
        )

        self.assertEqual(
            artifact["particle_count"],
            32,
        )
        self.assertEqual(
            len(artifact["prior_samples"]),
            32,
        )
        self.assertEqual(
            len(artifact["parameter_particles"]),
            32,
        )

        particle_ids = [
            row["particle_id"]
            for row in artifact[
                "prior_samples"
            ]
        ]
        self.assertEqual(
            len(set(particle_ids)),
            32,
        )

    def test_same_seed_is_deterministic(self):
        first = self._build(
            seed=202,
            count=32,
        )
        second = self._build(
            seed=202,
            count=32,
        )

        self.assertEqual(
            first["particle_sha256"],
            second["particle_sha256"],
        )
        self.assertEqual(
            first["prior_samples"],
            second["prior_samples"],
        )
        self.assertEqual(
            first["parameter_particles"],
            second["parameter_particles"],
        )

    def test_different_seed_changes_particles(self):
        first = self._build(
            seed=202,
            count=32,
        )
        second = self._build(
            seed=203,
            count=32,
        )

        self.assertNotEqual(
            first["particle_sha256"],
            second["particle_sha256"],
        )

    def test_future_information_cannot_change_prior(self):
        first_case = self._case()
        second_case = copy.deepcopy(
            first_case
        )

        second_case.update(
            {
                "early_day": 90.0,
                "early_volume_ml": 99999.0,
                "final_day": 999.0,
                "final_volume_ml": 0.00001,
                "response_label": (
                    "fabricated_future_response"
                ),
                "observed_response": (
                    "fabricated_future_response"
                ),
            }
        )
        second_case["context"].update(
            {
                "early_volume_ml": 99999.0,
                "final_volume_ml": 0.00001,
                "response": (
                    "fabricated_future_response"
                ),
                "observed_outcome": (
                    "fabricated_future_outcome"
                ),
            }
        )

        first = build_layer7_prior_particles(
            first_case,
            n_particles=32,
            seed=404,
            policy_path=DEFAULT_POLICY_PATH,
        )
        second = build_layer7_prior_particles(
            second_case,
            n_particles=32,
            seed=404,
            policy_path=DEFAULT_POLICY_PATH,
        )

        self.assertEqual(
            first["particle_sha256"],
            second["particle_sha256"],
        )
        self.assertEqual(
            first["prior_samples"],
            second["prior_samples"],
        )

    def test_rejection_sampling_does_not_clip(self):
        artifact = self._build(
            seed=303,
            count=32,
        )
        sampling = artifact["sampling"]

        self.assertEqual(
            sampling["method"],
            "deterministic_rejection_sampling",
        )
        self.assertEqual(
            sampling["accepted_count"],
            32,
        )
        self.assertFalse(
            sampling["clipping_performed"]
        )
        self.assertGreaterEqual(
            sampling["examined_count"],
            32,
        )
        self.assertEqual(
            sampling["rejected_count"],
            sampling["examined_count"] - 32,
        )

        # This seed previously exposed the unbounded transformed draw.
        self.assertGreaterEqual(
            sampling["rejected_count"],
            1,
        )
        self.assertGreaterEqual(
            sampling["rejected_by_parameter"].get(
                "active_treatment_sensitivity",
                0,
            ),
            1,
        )

    def test_prior_samples_respect_hard_bounds(self):
        artifact = self._build(
            seed=303,
            count=32,
        )

        for sample in artifact[
            "prior_samples"
        ]:
            validate_parameter_bounds(
                {
                    key: value
                    for key, value
                    in sample.items()
                    if key != "particle_id"
                }
            )

    def test_particles_are_simulator_ready(self):
        artifact = self._build(
            seed=505,
            count=32,
        )

        for particle in artifact[
            "parameter_particles"
        ]:
            self.assertIn(
                "growth_rate",
                particle,
            )
            self.assertIn(
                "drug_sensitivity",
                particle,
            )
            self.assertIn(
                "resistant_fraction",
                particle,
            )

            sensitivities = particle[
                "drug_sensitivity"
            ]
            self.assertIn(
                "anthracycline",
                sensitivities,
            )
            self.assertIn(
                "taxane",
                sensitivities,
            )

            self.assertGreater(
                float(particle["growth_rate"]),
                0.0,
            )
            self.assertGreater(
                float(
                    sensitivities[
                        "anthracycline"
                    ]
                ),
                0.0,
            )
            self.assertGreater(
                float(
                    sensitivities["taxane"]
                ),
                0.0,
            )
            self.assertGreaterEqual(
                float(
                    particle[
                        "resistant_fraction"
                    ]
                ),
                0.0,
            )
            self.assertLessEqual(
                float(
                    particle[
                        "resistant_fraction"
                    ]
                ),
                0.90,
            )

    def _build(
        self,
        *,
        seed,
        count,
    ):
        return build_layer7_prior_particles(
            self._case(),
            n_particles=count,
            seed=seed,
            policy_path=DEFAULT_POLICY_PATH,
        )

    @staticmethod
    def _case():
        return {
            "case_id": "layer7-prior-test",
            "data_origin": (
                "retrospective_clinical_registry"
            ),
            "subtype": "TNBC",
            "treatment_regimen": (
                "A/C-T neoadjuvant chemotherapy"
            ),
            "baseline_day": 0.0,
            "baseline_volume_ml": 28.0,
            "early_day": 42.0,
            "early_volume_ml": 18.0,
            "final_day": 126.0,
            "final_volume_ml": 7.5,
            "context": {
                "data_origin": (
                    "retrospective_clinical_registry"
                ),
                "subtype": "TNBC",
                "treatment_context": (
                    "A/C-T neoadjuvant chemotherapy"
                ),
                "er_status": "negative",
                "pr_status": "negative",
                "her2_status": "negative",
                "grade": 3,
                "ki67_percent": 45,
                "brca_status": "negative",
                "hrd_status": "positive",
                "segmentation_qc": "high",
                "source": (
                    "clinical_mri_report"
                ),
            },
        }


if __name__ == "__main__":
    unittest.main()
