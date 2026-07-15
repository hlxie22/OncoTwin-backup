from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

from evals.prior_stack.v1_cohort_validation import validate_v1_prior_eval_cohort
from evals.prior_stack.v1_real_data_eval import run_real_data_eval
from scripts.apply_v1_layer5_global_calibrator import (
    DEFAULT_CALIBRATOR_VERSION,
    DEFAULT_LOG_GROWTH_RATE_SHIFT,
    apply_layer5_global_calibrator,
)


class V1Layer5GlobalCalibratorTest(unittest.TestCase):
    def test_applies_validated_growth_down_residual_and_sidecars(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cohort = root / "cohort.jsonl"
            output = root / "cohort.layer5.jsonl"
            cohort.write_text(json.dumps(_case("registry_001")) + "\n", encoding="utf-8")

            result = apply_layer5_global_calibrator(cohort, output)

            rows = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            summary_path = output.with_name("cohort.layer5.summary.json")
            exclusions_path = output.with_name("cohort.layer5.exclusions.jsonl")
            validation = validate_v1_prior_eval_cohort(
                output,
                require_in_scope=True,
                require_sidecars=True,
            )

            summary_exists = summary_path.exists()
            exclusions_exists = exclusions_path.exists()
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(result["rows"], 1)
        self.assertEqual(rows[0]["data_origin"], "ISPY2")
        self.assertEqual(rows[0]["ai_residual"]["validated"], True)
        self.assertEqual(rows[0]["ai_residual"]["model_version"], DEFAULT_CALIBRATOR_VERSION)
        self.assertAlmostEqual(
            rows[0]["ai_residual"]["log_growth_rate_shift"],
            DEFAULT_LOG_GROWTH_RATE_SHIFT,
        )
        self.assertTrue(summary_exists)
        self.assertTrue(exclusions_exists)
        self.assertEqual(validation.case_count, 1)
        self.assertEqual(validation.in_scope_case_count, 1)
        self.assertEqual(summary["included_rows"], 1)
        self.assertEqual(
            summary["layer5_calibrator"]["calibrator_version"],
            DEFAULT_CALIBRATOR_VERSION,
        )
        self.assertEqual(summary["layer5_calibrator"]["uses_heldout_outcomes"], False)

    def test_rejects_existing_residual_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cohort = root / "cohort.jsonl"
            output = root / "cohort.layer5.jsonl"
            case = _case("registry_001")
            case["ai_residual"] = {
                "validated": True,
                "model_version": "existing_residual",
                "log_growth_rate_shift": -0.01,
            }
            cohort.write_text(json.dumps(case) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "already has ai_residual"):
                apply_layer5_global_calibrator(cohort, output)

            result = apply_layer5_global_calibrator(
                cohort,
                output,
                calibrator_version="replacement",
                log_growth_rate_shift=-math.log(1.07),
                overwrite_existing_residual=True,
            )

            row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(result["rows"], 1)
        self.assertEqual(row["ai_residual"]["model_version"], "replacement")
        self.assertAlmostEqual(row["ai_residual"]["log_growth_rate_shift"], -math.log(1.07))

    def test_validated_noop_layer5_is_exact_layer4_noop(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cohort = Path(tmpdir) / "cohort.jsonl"
            case = _case("registry_001")
            case["ai_residual"] = {
                "validated": True,
                "model_version": "validated_noop_unit_test",
            }
            cohort.write_text(json.dumps(case) + "\n", encoding="utf-8")

            result = run_real_data_eval(cohort, n_samples=200, seed=7)

        row = result["case_predictions"][0]
        layer4 = row["predictions"]["layer4_mri_qc"]
        layer5 = row["predictions"]["layer5_ai_residual"]
        layer5_audit = row["layer_audit"]["layer5_ai_residual"]
        delta = result["layer_delta"]["layer5_vs_layer4"]["cases"][0]

        self.assertEqual(layer5_audit["policy_mode"], "validated_noop")
        self.assertEqual(layer5, layer4)
        self.assertEqual(
            row["layer_debug"]["layer5_ai_residual"],
            row["layer_debug"]["layer4_mri_qc"],
        )
        self.assertEqual(delta["outcome"], "unchanged")
        self.assertAlmostEqual(delta["mae_delta_ml"], 0.0)


def _case(case_id: str) -> dict[str, object]:
    return {
        "case_id": case_id,
        "data_origin": "ISPY2",
        "subtype": "TNBC",
        "cancer_subtype": "TNBC",
        "disease_context": "TNBC",
        "treatment_context": "neoadjuvant chemotherapy",
        "schedule_type": "neoadjuvant chemotherapy",
        "treatment_regimen": "AC-T chemotherapy",
        "regimen_name": "AC-T chemotherapy",
        "er_status": "negative",
        "pr_status": "negative",
        "hr_status": "negative",
        "her2_status": "negative",
        "baseline_day": 0,
        "baseline_volume_ml": 24,
        "early_day": 28,
        "early_volume_ml": 18,
        "final_day": 84,
        "final_volume_ml": 6,
        "volume_ml": 24,
        "functional_tumor_volume_ml": 20,
        "segmentation_qc": "high",
    }


if __name__ == "__main__":
    unittest.main()
