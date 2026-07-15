from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from evals.prior_stack.v1_real_data_eval import (
    load_real_cohort,
    run_real_data_eval,
    write_markdown_report,
)


class V1RealDataEvalTest(unittest.TestCase):
    def test_runs_layer_ablation_on_real_cohort_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cohort = Path(tmpdir) / "cohort.json"
            report = Path(tmpdir) / "report.md"
            cohort.write_text(json.dumps({"cases": _cases()}), encoding="utf-8")

            result = run_real_data_eval(cohort, n_samples=200, seed=7)
            write_markdown_report(result, report)
            text = report.read_text(encoding="utf-8")

        self.assertEqual(result["case_count"], 2)
        self.assertEqual(result["in_scope_case_count"], 2)
        for name in (
            "baseline_no_change",
            "layer2_population",
            "layer3_pathology",
            "layer4_mri_qc",
            "layer5_ai_residual",
        ):
            self.assertIn(name, result["metrics"])
        self.assertIn("layer3_vs_layer2", result["layer_delta"])
        self.assertIn("layer5_vs_layer4", result["layer_delta"])
        delta_cases = result["layer_delta"]["layer3_vs_layer2"]["cases"]
        self.assertEqual(
            {case["case_id"] for case in delta_cases},
            {"tnbc_registry_001", "tnbc_registry_002"},
        )
        self.assertTrue(
            all(
                case["outcome"] in {"helped", "harmed", "unchanged"}
                for case in delta_cases
            )
        )
        self.assertTrue(
            all(
                "old_abs_error_ml" in case and "new_abs_error_ml" in case
                for case in delta_cases
            )
        )
        self.assertIn("V1 real-data prior-layer evaluation", text)
        self.assertIn("Cases helped/harmed by layer", text)
        self.assertIn("layer4_mri_qc", text)
        self.assertIn("layer5_ai_residual", text)

    def test_loads_csv_cohort(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cohort = Path(tmpdir) / "cohort.csv"
            cohort.write_text(
                "case_id,subtype,treatment_regimen,baseline_day,baseline_volume_ml,final_day,final_volume_ml\n"
                "registry_001,TNBC,AC-T chemotherapy,0,12,84,3.5\n",
                encoding="utf-8",
            )
            cases = load_real_cohort(cohort)

        self.assertEqual(cases[0]["case_id"], "registry_001")
        self.assertEqual(cases[0]["baseline_volume_ml"], 12.0)

    def test_rejects_missing_case_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cohort = Path(tmpdir) / "cohort.jsonl"
            cohort.write_text(
                json.dumps(
                    {
                        "subtype": "TNBC",
                        "treatment_regimen": "AC-T chemotherapy",
                        "baseline_day": 0,
                        "baseline_volume_ml": 30,
                        "final_day": 84,
                        "final_volume_ml": 7.5,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "case_id or patient_id"):
                load_real_cohort(cohort)

    def test_rejects_missing_required_volume(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cohort = Path(tmpdir) / "cohort.jsonl"
            cohort.write_text(
                json.dumps(
                    {
                        "case_id": "registry_001",
                        "subtype": "TNBC",
                        "treatment_regimen": "AC-T chemotherapy",
                        "baseline_day": 0,
                        "baseline_volume_ml": 30,
                        "final_day": 84,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "missing required numeric field"):
                load_real_cohort(cohort)

    def test_rejects_invalid_time_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cohort = Path(tmpdir) / "cohort.jsonl"
            cohort.write_text(
                json.dumps(
                    {
                        "case_id": "registry_001",
                        "subtype": "TNBC",
                        "treatment_regimen": "AC-T chemotherapy",
                        "baseline_day": 28,
                        "baseline_volume_ml": 30,
                        "final_day": 28,
                        "final_volume_ml": 7.5,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "final_day after baseline_day"):
                load_real_cohort(cohort)

    def test_rejects_unpaired_or_out_of_order_early_followup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cohort = Path(tmpdir) / "cohort.jsonl"
            cohort.write_text(
                json.dumps(
                    {
                        "case_id": "registry_001",
                        "subtype": "TNBC",
                        "treatment_regimen": "AC-T chemotherapy",
                        "baseline_day": 0,
                        "baseline_volume_ml": 30,
                        "early_day": 100,
                        "early_volume_ml": 20,
                        "final_day": 84,
                        "final_volume_ml": 7.5,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "early follow-up"):
                load_real_cohort(cohort)

    def test_rejects_demo_fixture_by_default(self):
        fixture = Path("fixtures/mechanistic_simulator/cases/tnbc_demo_case.json")
        with self.assertRaisesRegex(ValueError, "refused demo/synthetic/fixture"):
            load_real_cohort(fixture)

    def test_allows_demo_fixture_for_smoke_testing_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture = Path(tmpdir) / "demo_fixture.json"
            fixture.write_text(
                json.dumps(
                    {
                        "case_id": "demo_case",
                        "subtype": "TNBC",
                        "treatment_regimen": "AC-T chemotherapy",
                        "baseline_day": 0,
                        "baseline_volume_ml": 30,
                        "final_day": 84,
                        "final_volume_ml": 7.5,
                    }
                ),
                encoding="utf-8",
            )
            cases = load_real_cohort(fixture, allow_demo_data=True)

        self.assertEqual(cases[0]["baseline_volume_ml"], 30.0)
        self.assertEqual(cases[0]["final_volume_ml"], 7.5)


def _cases():
    return [
        {
            "case_id": "tnbc_registry_001",
            "data_origin": "retrospective_clinical_registry",
            "subtype": "Triple-negative breast cancer",
            "treatment_regimen": "A/C-T neoadjuvant chemotherapy",
            "er_status": "negative",
            "pr_status": "negative",
            "her2_status": "negative",
            "grade": 3,
            "ki67_percent": 45,
            "brca_status": "pathogenic",
            "baseline_day": 0,
            "baseline_volume_ml": 28,
            "early_day": 42,
            "early_volume_ml": 18,
            "final_day": 126,
            "final_volume_ml": 7.5,
            "volume_ml": 28,
            "functional_tumor_volume_ml": 24,
            "segmentation_qc": "high",
            "registration_qc": "high",
            "source": "clinical_mri_report",
            "ai_residual": {
                "validated": True,
                "model_version": "unit_residual_v1",
                "log_active_treatment_sensitivity_shift": 0.03,
                "resistant_variance_multiplier": 1.10,
            },
        },
        {
            "case_id": "tnbc_registry_002",
            "data_origin": "retrospective_clinical_registry",
            "subtype": "TNBC",
            "treatment_regimen": "AC-T chemotherapy",
            "er_status": "negative",
            "pr_status": "negative",
            "her2_status": "negative",
            "grade": 2,
            "ki67_percent": 18,
            "brca_status": "not detected",
            "baseline_day": 0,
            "baseline_volume_ml": 20,
            "early_day": 35,
            "early_volume_ml": 17.5,
            "final_day": 105,
            "final_volume_ml": 14,
            "volume_ml": 20,
            "functional_tumor_volume_ml": 9,
            "enhancement_std": 0.35,
            "low_enhancement_fraction": 0.40,
            "segmentation_qc": "medium",
            "registration_qc": "low",
            "source": "clinical_mri_report",
        },
    ]


if __name__ == "__main__":
    unittest.main()
