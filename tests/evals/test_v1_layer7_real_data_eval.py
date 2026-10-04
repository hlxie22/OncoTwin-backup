from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from evals.prior_stack.v1_layer7_real_data_eval import (
    DEFAULT_EVAL_POLICY,
    canonical_act_schedule,
    evaluate_layer7_case,
    run_layer7_real_data_eval,
)
from experiments.twin_runtime.layer7_prior_particles import (
    DEFAULT_POLICY_PATH,
)


class Layer7RealDataEvalTest(
    unittest.TestCase
):
    @classmethod
    def setUpClass(cls) -> None:
        cls.eval_policy = json.loads(
            DEFAULT_EVAL_POLICY.read_text(
                encoding="utf-8"
            )
        )

    def test_schedule_covers_prediction_horizon(self):
        schedule = canonical_act_schedule(
            126.0,
            policy=self.eval_policy,
        )

        self.assertEqual(
            schedule["total_duration_days"],
            126.0,
        )
        self.assertTrue(
            any(
                event["drug"] == "anthracycline"
                for event in schedule["events"]
            )
        )
        self.assertTrue(
            any(
                event["drug"] == "taxane"
                for event in schedule["events"]
            )
        )
        self.assertTrue(
            all(
                event["day"] <= 126.0
                for event in schedule["events"]
            )
        )

    def test_missing_early_is_exact_no_update(self):
        case = self._case(
            case_id="missing_early",
            early_day=None,
            early_volume_ml=None,
        )

        result = evaluate_layer7_case(
            case,
            case_index=0,
            particle_count=32,
            seed=1000,
            particle_policy_path=(
                DEFAULT_POLICY_PATH
            ),
            eval_policy=self.eval_policy,
        )

        self.assertEqual(
            result["status"],
            "not_updated_no_valid_observation",
        )
        self.assertEqual(
            result["prior_prediction"],
            result["posterior_prediction"],
        )
        self.assertTrue(
            result[
                "exact_prior_posterior_match"
            ]
        )

    def test_final_outcome_cannot_change_inference(self):
        first_case = self._case(
            case_id="active_case",
        )
        second_case = copy.deepcopy(
            first_case
        )
        second_case[
            "final_volume_ml"
        ] = 999999.0

        first = evaluate_layer7_case(
            first_case,
            case_index=0,
            particle_count=32,
            seed=2000,
            particle_policy_path=(
                DEFAULT_POLICY_PATH
            ),
            eval_policy=self.eval_policy,
        )
        second = evaluate_layer7_case(
            second_case,
            case_index=0,
            particle_count=32,
            seed=2000,
            particle_policy_path=(
                DEFAULT_POLICY_PATH
            ),
            eval_policy=self.eval_policy,
        )

        self.assertEqual(
            first["particle_sha256"],
            second["particle_sha256"],
        )
        self.assertEqual(
            first["prior_prediction"],
            second["prior_prediction"],
        )
        self.assertEqual(
            first["posterior_prediction"],
            second["posterior_prediction"],
        )
        self.assertEqual(
            first["posterior_health"],
            second["posterior_health"],
        )

        for prediction_name in (
            "prior_prediction",
            "posterior_prediction",
        ):
            prediction = first[
                prediction_name
            ]
            self.assertLessEqual(
                prediction["lower_95_ml"],
                prediction["lower_80_ml"],
            )
            self.assertLessEqual(
                prediction["lower_80_ml"],
                prediction["point_ml"],
            )
            self.assertLessEqual(
                prediction["point_ml"],
                prediction["upper_80_ml"],
            )
            self.assertLessEqual(
                prediction["upper_80_ml"],
                prediction["upper_95_ml"],
            )

    def test_small_cohort_runs_end_to_end(self):
        cases = [
            self._case(
                case_id="active_case",
            ),
            self._case(
                case_id="missing_case",
                early_day=None,
                early_volume_ml=None,
            ),
        ]

        with tempfile.TemporaryDirectory() as directory:
            cohort = Path(directory) / "cohort.jsonl"
            cohort.write_text(
                "".join(
                    json.dumps(case) + "\n"
                    for case in cases
                ),
                encoding="utf-8",
            )

            result = run_layer7_real_data_eval(
                cohort,
                particle_count=32,
                seed=3000,
                particle_policy_path=(
                    DEFAULT_POLICY_PATH
                ),
                eval_policy_path=(
                    DEFAULT_EVAL_POLICY
                ),
            )

        self.assertEqual(
            result["dataset"]["case_count"],
            2,
        )
        self.assertEqual(
            result["dataset"][
                "updated_case_count"
            ],
            1,
        )
        self.assertEqual(
            result["dataset"][
                "early_missing_count"
            ],
            1,
        )
        self.assertEqual(
            result["dataset"][
                "exact_no_update_count"
            ],
            1,
        )

    @staticmethod
    def _case(
        *,
        case_id,
        early_day=42.0,
        early_volume_ml=18.0,
    ):
        return {
            "case_id": case_id,
            "data_origin": (
                "retrospective_clinical_registry"
            ),
            "subtype": "TNBC",
            "treatment_regimen": (
                "A/C-T neoadjuvant chemotherapy"
            ),
            "baseline_day": 0.0,
            "baseline_volume_ml": 28.0,
            "early_day": early_day,
            "early_volume_ml": (
                early_volume_ml
            ),
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
