from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evals.prior_stack.v1_real_data_eval import (
    CALIBRATED_LAYER4_CANDIDATE,
    STATIC_LAYER4_SOURCE_PREDICTION,
    run_real_data_eval,
)


ARTIFACT = Path(
    "configs/prior/"
    "v1_d1_constant_log_radius_posthoc_v0_1.json"
)


class V1IntervalCalibrationIntegrationTest(
    unittest.TestCase
):
    def test_default_run_is_unchanged_and_has_no_candidate(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            cohort = self._write_cohort(
                Path(tmpdir)
            )

            result = run_real_data_eval(
                cohort,
                n_samples=200,
                seed=17,
            )

        prediction = result["case_predictions"][0][
            "predictions"
        ]

        self.assertNotIn(
            CALIBRATED_LAYER4_CANDIDATE,
            prediction,
        )
        self.assertNotIn(
            CALIBRATED_LAYER4_CANDIDATE,
            result["metrics"],
        )
        self.assertIsNone(
            result["interval_calibration"]
        )

    def test_candidate_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cohort = self._write_cohort(
                Path(tmpdir)
            )

            with self.assertRaisesRegex(
                ValueError,
                "allow_candidate=True",
            ):
                run_real_data_eval(
                    cohort,
                    n_samples=200,
                    seed=17,
                    interval_calibrator_path=ARTIFACT,
                )

    def test_candidate_is_separate_and_preserves_raw_layers(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            cohort = self._write_cohort(
                Path(tmpdir)
            )

            default_result = run_real_data_eval(
                cohort,
                n_samples=200,
                seed=17,
            )

            candidate_result = run_real_data_eval(
                cohort,
                n_samples=200,
                seed=17,
                interval_calibrator_path=ARTIFACT,
                allow_candidate_interval_calibrator=True,
            )

        default_row = default_result[
            "case_predictions"
        ][0]
        candidate_row = candidate_result[
            "case_predictions"
        ][0]

        self.assertEqual(
            candidate_row["predictions"][
                "layer4_mri_qc"
            ],
            default_row["predictions"][
                "layer4_mri_qc"
            ],
        )
        self.assertEqual(
            candidate_row["predictions"][
                "layer5_ai_residual"
            ],
            default_row["predictions"][
                "layer5_ai_residual"
            ],
        )

        calibrated = candidate_row["predictions"][
            CALIBRATED_LAYER4_CANDIDATE
        ]

        self.assertEqual(
            calibrated["calibration"][
                "source_prediction"
            ],
            STATIC_LAYER4_SOURCE_PREDICTION,
        )
        self.assertEqual(
            calibrated["calibration"]["status"],
            "candidate_posthoc",
        )
        self.assertFalse(
            calibrated["calibration"][
                "point_prediction_changed"
            ]
        )
        self.assertIn(
            CALIBRATED_LAYER4_CANDIDATE,
            candidate_result["metrics"],
        )
        self.assertEqual(
            candidate_result["metrics"][
                CALIBRATED_LAYER4_CANDIDATE
            ]["n"],
            1,
        )

    def test_candidate_uses_static_pre_early_response_point(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            cohort = self._write_cohort(
                Path(tmpdir)
            )

            candidate_result = run_real_data_eval(
                cohort,
                n_samples=200,
                seed=17,
                interval_calibrator_path=ARTIFACT,
                allow_candidate_interval_calibrator=True,
            )

            with patch(
                "evals.prior_stack.v1_real_data_eval."
                "_apply_layer4_early_response_rules",
                side_effect=self._no_early_rule,
            ):
                static_result = run_real_data_eval(
                    cohort,
                    n_samples=200,
                    seed=17,
                )

        calibrated = candidate_result[
            "case_predictions"
        ][0]["predictions"][
            CALIBRATED_LAYER4_CANDIDATE
        ]
        static_layer4 = static_result[
            "case_predictions"
        ][0]["predictions"]["layer4_mri_qc"]

        self.assertEqual(
            calibrated["point_ml"],
            static_layer4["point_ml"],
        )
        self.assertLess(
            calibrated["lower_80_ml"],
            calibrated["point_ml"],
        )
        self.assertGreater(
            calibrated["upper_80_ml"],
            calibrated["point_ml"],
        )
        self.assertLess(
            calibrated["lower_95_ml"],
            calibrated["lower_80_ml"],
        )
        self.assertGreater(
            calibrated["upper_95_ml"],
            calibrated["upper_80_ml"],
        )

    @staticmethod
    def _no_early_rule(case, samples):
        del case
        return [
            dict(sample)
            for sample in samples
        ], None

    @staticmethod
    def _write_cohort(root: Path) -> Path:
        cohort = root / "real_cohort.jsonl"
        cohort.write_text(
            json.dumps(
                {
                    "case_id": "registry_001",
                    "data_origin": "ISPY2",
                    "subtype": "TNBC",
                    "cancer_subtype": "TNBC",
                    "disease_context": "TNBC",
                    "treatment_context": (
                        "neoadjuvant chemotherapy"
                    ),
                    "schedule_type": (
                        "neoadjuvant chemotherapy"
                    ),
                    "treatment_regimen": (
                        "AC-T chemotherapy"
                    ),
                    "regimen_name": "AC-T chemotherapy",
                    "er_status": "negative",
                    "pr_status": "negative",
                    "hr_status": "negative",
                    "her2_status": "negative",
                    "baseline_day": 0,
                    "baseline_volume_ml": 0.8,
                    "early_day": 42,
                    "early_volume_ml": 0.35,
                    "final_day": 126,
                    "final_volume_ml": 0.2,
                    "volume_ml": 0.8,
                    "functional_tumor_volume_ml": 0.8,
                    "segmentation_qc": "high",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return cohort


if __name__ == "__main__":
    unittest.main()
