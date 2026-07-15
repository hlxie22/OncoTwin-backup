from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evals.prior_stack.v1_uncertainty_calibration_eval import (
    run_uncertainty_calibration_eval,
)


class V1UncertaintyCalibrationEvalTest(unittest.TestCase):
    def test_reports_subgroup_calibration_when_groups_are_large_enough(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cohort = root / "cohort.jsonl"
            cohort.write_text(
                "".join(json.dumps(case) + "\n" for case in _cases(6)),
                encoding="utf-8",
            )
            report = root / "calibration.md"

            result = run_uncertainty_calibration_eval(
                cohort,
                report_path=report,
                n_samples=120,
                seed=23,
            )
            report_text = report.read_text(encoding="utf-8")

        subgroup_metrics = result.metrics["subgroup_calibration"]
        self.assertIn("data_origin=ispy2", subgroup_metrics)
        self.assertEqual(subgroup_metrics["data_origin=ispy2"]["n"], 6)
        self.assertIn("Subgroup calibration", report_text)
        self.assertIn("data_origin=ispy2", report_text)
        self.assertIn("Aggregate calibration", report_text)

    def test_flags_low_interval_coverage_as_overconfident(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report = Path(tmpdir) / "calibration.md"
            with patch(
                "evals.prior_stack.v1_uncertainty_calibration_eval.run_real_data_eval",
                return_value=_overconfident_eval_result(),
            ):
                result = run_uncertainty_calibration_eval(
                    Path(tmpdir) / "cohort.jsonl",
                    report_path=report,
                    n_samples=120,
                    seed=23,
                )
            report_text = report.read_text(encoding="utf-8")

        self.assertEqual(result.metrics["final_layer"], "layer5_ai_residual")
        self.assertTrue(any("overconfident" in warning for warning in result.warnings))
        self.assertIn("may be overconfident", report_text)
        self.assertEqual(result.metrics["final_layer_coverage_80"], 0.5)

    def test_rejects_too_small_subgroup_threshold(self):
        with self.assertRaisesRegex(ValueError, "min_subgroup_size"):
            run_uncertainty_calibration_eval(
                Path("unused.jsonl"),
                min_subgroup_size=1,
            )


def _cases(count: int) -> list[dict[str, object]]:
    cases = []
    for index in range(count):
        baseline = 18.0 + index
        cases.append(
            {
                "case_id": f"registry_{index:03d}",
                "data_origin": "ISPY2",
                "subtype": "TNBC",
                "treatment_regimen": "AC-T chemotherapy",
                "er_status": "negative",
                "pr_status": "negative",
                "her2_status": "negative",
                "baseline_day": 0,
                "baseline_volume_ml": baseline,
                "early_day": 28,
                "early_volume_ml": baseline * 0.78,
                "final_day": 84,
                "final_volume_ml": baseline * 0.38,
                "volume_ml": baseline,
                "functional_tumor_volume_ml": baseline * 0.9,
                "segmentation_qc": "high",
            }
        )
    return cases


def _overconfident_eval_result() -> dict[str, object]:
    case_predictions = [
        {
            "case_id": f"registry_{index:03d}",
            "observed_final_volume_ml": 10.0,
            "calibration_groups": {
                "data_origin": "ispy2",
                "early_followup": "available",
            },
            "predictions": {
                "layer5_ai_residual": {
                    "point_ml": 8.0,
                    "lower_80_ml": 7.0,
                    "upper_80_ml": 9.0,
                    "lower_95_ml": 6.5,
                    "upper_95_ml": 9.5,
                },
                "layer4_mri_qc": {
                    "point_ml": 8.0,
                    "lower_80_ml": 7.0,
                    "upper_80_ml": 9.0,
                    "lower_95_ml": 6.5,
                    "upper_95_ml": 9.5,
                },
            },
        }
        for index in range(6)
    ]
    return {
        "in_scope_case_count": 6,
        "metrics": {
            "layer5_ai_residual": {
                "n": 6,
                "coverage_80": 0.5,
                "coverage_95": 0.5,
                "width_80_ml": 2.0,
            },
            "layer4_mri_qc": {
                "n": 6,
                "coverage_80": 0.5,
                "coverage_95": 0.5,
                "width_80_ml": 2.0,
            },
        },
        "case_predictions": case_predictions,
        "warnings": [],
    }
