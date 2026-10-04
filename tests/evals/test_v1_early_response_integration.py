from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

from evals.prior_stack.v1_real_data_eval import (
    EARLY_RESPONSE_CANDIDATE,
    run_real_data_eval,
)


UPDATER = Path(
    "configs/prior/"
    "v1_d1_continuous_early_volume_posthoc_v0_1.json"
)
CALIBRATOR = Path(
    "configs/prior/"
    "v1_d1_early_response_constant_log_radius_"
    "posthoc_v0_1.json"
)


class EarlyResponseIntegrationTest(unittest.TestCase):
    def test_default_run_has_no_candidate(self):
        cohort = self._cohort(early=0.4)

        result = run_real_data_eval(
            cohort,
            n_samples=200,
            seed=17,
        )

        self.assertNotIn(
            EARLY_RESPONSE_CANDIDATE,
            result["case_predictions"][0][
                "predictions"
            ],
        )

    def test_artifacts_must_be_paired(self):
        cohort = self._cohort(early=0.4)

        with self.assertRaisesRegex(
            ValueError,
            "must be provided together",
        ):
            run_real_data_eval(
                cohort,
                n_samples=200,
                seed=17,
                early_response_updater_path=UPDATER,
            )

    def test_candidate_is_complete_and_separate(self):
        cohort = self._cohort(early=0.4)

        baseline = run_real_data_eval(
            cohort,
            n_samples=200,
            seed=17,
        )
        candidate = run_real_data_eval(
            cohort,
            n_samples=200,
            seed=17,
            early_response_updater_path=UPDATER,
            early_response_interval_calibrator_path=(
                CALIBRATOR
            ),
            allow_candidate_early_response_updater=True,
            allow_candidate_interval_calibrator=True,
        )

        old = baseline["case_predictions"][0]
        new = candidate["case_predictions"][0]

        self.assertEqual(
            old["predictions"]["layer4_mri_qc"],
            new["predictions"]["layer4_mri_qc"],
        )
        self.assertEqual(
            old["predictions"]["layer5_ai_residual"],
            new["predictions"]["layer5_ai_residual"],
        )

        prediction = new["predictions"][
            EARLY_RESPONSE_CANDIDATE
        ]

        source_point = prediction[
            "early_response_update"
        ]["source_point_ml"]
        expected_log_correction = (
            -0.07390443919179766
            + 0.7127777605265024
            * math.log(0.4 / 0.8)
        )
        expected_point = (
            source_point
            * math.exp(expected_log_correction)
        )

        self.assertAlmostEqual(
            prediction["point_ml"],
            expected_point,
        )
        self.assertLess(
            prediction["lower_80_ml"],
            prediction["point_ml"],
        )
        self.assertGreater(
            prediction["upper_80_ml"],
            prediction["point_ml"],
        )
        self.assertEqual(
            prediction["calibration"][
                "source_prediction"
            ],
            "early_response_update_candidate_point",
        )
        self.assertIn(
            EARLY_RESPONSE_CANDIDATE,
            candidate["metrics"],
        )

    def test_missing_early_uses_fallback(self):
        cohort = self._cohort(early=None)

        result = run_real_data_eval(
            cohort,
            n_samples=200,
            seed=17,
            early_response_updater_path=UPDATER,
            early_response_interval_calibrator_path=(
                CALIBRATOR
            ),
            allow_candidate_early_response_updater=True,
            allow_candidate_interval_calibrator=True,
        )

        audit = result["case_predictions"][0][
            "predictions"
        ][EARLY_RESPONSE_CANDIDATE][
            "early_response_update"
        ]

        self.assertFalse(audit["early_available"])
        self.assertEqual(
            audit["correction_source"],
            "missing_early_fallback",
        )

    def _cohort(self, *, early):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)

        path = Path(directory.name) / "cohort.jsonl"

        row = {
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
            "treatment_regimen": "AC-T chemotherapy",
            "regimen_name": "AC-T chemotherapy",
            "er_status": "negative",
            "pr_status": "negative",
            "hr_status": "negative",
            "her2_status": "negative",
            "baseline_day": 0,
            "baseline_volume_ml": 0.8,
            "early_day": 42 if early is not None else None,
            "early_volume_ml": early,
            "final_day": 126,
            "final_volume_ml": 0.2,
            "volume_ml": 0.8,
            "functional_tumor_volume_ml": 0.8,
            "segmentation_qc": "high",
        }

        path.write_text(
            json.dumps(row) + "\n",
            encoding="utf-8",
        )

        return path


if __name__ == "__main__":
    unittest.main()
