from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from evals.prior_stack.common import EvalResult, EvalUnavailable, require_modules
from evals.prior_stack.run_v1_eval_suite import run_suite
from evals.prior_stack.v1_posterior_health_eval import run_eval as run_posterior_health


RUNTIME_EVAL_NAMES = {
    "posterior_health",
    "sequential_forecasting",
    "update_value",
    "scenario_lab_stability",
    "explanation_quality",
}


class V1EvalSuiteTest(unittest.TestCase):
    def test_suite_runs_without_cohort_and_writes_runtime_analysis(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            report = root / "suite.md"
            analysis_dir = root / "analysis"
            results = run_suite(report_path=report, analysis_dir=analysis_dir)
            payload = json.loads(report.with_suffix(".summary.json").read_text(encoding="utf-8"))

            self.assertTrue(report.exists())
            self.assertTrue(report.with_suffix(".summary.json").exists())
            self.assertEqual(payload["analysis_dir"], str(analysis_dir))
            for name in RUNTIME_EVAL_NAMES:
                self.assertTrue((analysis_dir / f"v1_{name}.analysis.json").exists())

        self.assertTrue(any(r.name == "real_data_prior_layer_performance" for r in results))
        self.assertTrue(any(r.status == "unavailable" for r in results))
        self.assertTrue(all(isinstance(r, EvalResult) for r in results))
        runtime_results = {result.name: result for result in results if result.name in RUNTIME_EVAL_NAMES}
        self.assertEqual(set(runtime_results), RUNTIME_EVAL_NAMES)
        self.assertTrue(all(result.status == "pass" for result in runtime_results.values()))

    def test_suite_writes_v1_d1_machine_summary_with_curation_sidecars(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cohort = root / "ispy2_v1_prior_eval_cohort.jsonl"
            cohort.write_text(
                "\n".join(json.dumps(case) for case in _real_cases()) + "\n",
                encoding="utf-8",
            )
            cohort.with_suffix(".summary.json").write_text(
                json.dumps(
                    {
                        "total_input_rows": 3,
                        "included_rows": 2,
                        "excluded_rows": 1,
                        "v1a_in_scope_count": 2,
                        "excluded_reason_counts": {"missing_final_volume": 1},
                        "biomarker_completeness": {"er_status": 2, "her2_status": 2},
                        "mri_feature_completeness": {"functional_tumor_volume_ml": 2},
                        "source_files": ["local/ispy2_longitudinal.csv"],
                    }
                ),
                encoding="utf-8",
            )
            cohort.with_suffix(".exclusions.jsonl").write_text(
                json.dumps(
                    {
                        "case_id": "registry_003",
                        "excluded_reason": "missing_final_volume",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            report = root / "v1_eval_suite.md"
            summary = root / "v1_eval_suite.summary.json"

            results = run_suite(
                cohort_path=cohort,
                report_path=report,
                summary_path=summary,
                analysis_dir=root / "analysis",
                n_samples=120,
                seed=11,
            )
            payload = json.loads(summary.read_text(encoding="utf-8"))
            report_text = report.read_text(encoding="utf-8")

        self.assertEqual(payload["v1_d1_status"], "pass")
        self.assertEqual(payload["cohort_path"], str(cohort))
        self.assertEqual(payload["seed"], 11)
        self.assertEqual(payload["n_samples"], 120)
        self.assertEqual(
            payload["cohort_curation"]["cohort_summary"]["total_input_rows"],
            3,
        )
        self.assertEqual(
            payload["cohort_curation"]["exclusion_report"][
                "excluded_reason_counts"
            ],
            {"missing_final_volume": 1},
        )
        self.assertTrue(any(result.name == "uncertainty_calibration" for result in results))
        self.assertIn("V1-D1 cohort evidence", report_text)
        self.assertIn("missing_final_volume", report_text)
        self.assertIn("local/ispy2_longitudinal.csv", report_text)

    def test_posterior_health_runtime_eval_passes_and_writes_analysis(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            analysis = Path(tmpdir) / "posterior.analysis.json"
            result = run_posterior_health(analysis_path=analysis)
            payload = json.loads(analysis.read_text(encoding="utf-8"))

        self.assertEqual(result.status, "pass")
        self.assertGreater(result.metrics["effective_sample_size_fraction"], 0)
        self.assertEqual(payload["eval_name"], "posterior_health")
        self.assertIn("posterior_trajectory_summary", payload)

    def test_require_modules_raises_clear_error(self):
        with self.assertRaises(EvalUnavailable) as raised:
            require_modules(
                "missing_eval",
                ("not_a_real_oncotwin_runtime",),
                "Missing runtime",
            )
        self.assertEqual(raised.exception.name, "missing_eval")
        self.assertEqual(
            raised.exception.missing,
            ("not_a_real_oncotwin_runtime",),
        )


def _real_cases() -> list[dict[str, object]]:
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
