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
        ):
            self.assertIn(name, result["metrics"])
        self.assertIn("layer3_vs_layer2", result["layer_delta"])
        self.assertIn("V1 real-data prior-layer evaluation", text)
        self.assertIn("layer4_mri_qc", text)

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
