from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from evals.prior_stack.v1_cohort_validation import validate_v1_prior_eval_cohort


class V1CohortValidationTest(unittest.TestCase):
    def test_validates_cohort_with_required_sidecars_and_in_scope_cases(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cohort = root / "ispy2_v1_prior_eval_cohort.jsonl"
            cohort.write_text(json.dumps(_case("registry_001")) + "\n", encoding="utf-8")
            cohort.with_name("ispy2_v1_prior_eval_cohort.summary.json").write_text(
                json.dumps(
                    {
                        "included_rows": 1,
                        "excluded_rows": 1,
                        "v1a_in_scope_count": 1,
                        "excluded_reason_counts": {"missing_final_volume": 1},
                    }
                ),
                encoding="utf-8",
            )
            cohort.with_name("ispy2_v1_prior_eval_cohort.exclusions.jsonl").write_text(
                json.dumps(
                    {
                        "case_id": "registry_002",
                        "excluded_reasons": ["missing_final_volume"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = validate_v1_prior_eval_cohort(
                cohort,
                require_in_scope=True,
                require_sidecars=True,
            )

        self.assertEqual(result.case_count, 1)
        self.assertEqual(result.in_scope_case_count, 1)
        self.assertEqual(result.data_origin_counts, {"ISPY2": 1})
        self.assertEqual(
            result.exclusion_report["excluded_reason_counts"],
            {"missing_final_volume": 1},
        )
        self.assertIn("fewer than 20 cases", result.warnings[0])

    def test_rejects_missing_data_origin(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cohort = Path(tmpdir) / "cohort.jsonl"
            case = _case("registry_001")
            case.pop("data_origin")
            cohort.write_text(json.dumps(case) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "require data_origin"):
                validate_v1_prior_eval_cohort(cohort)

    def test_rejects_duplicate_case_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cohort = Path(tmpdir) / "cohort.jsonl"
            cohort.write_text(
                json.dumps(_case("registry_001"))
                + "\n"
                + json.dumps(_case("registry_001"))
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "duplicate case_id"):
                validate_v1_prior_eval_cohort(cohort)

    def test_require_in_scope_rejects_out_of_scope_only_cohort(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cohort = Path(tmpdir) / "cohort.jsonl"
            cohort.write_text(
                json.dumps(
                    {
                        **_case("registry_001"),
                        "subtype": "HR positive",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "0 V1-A in-scope cases"):
                validate_v1_prior_eval_cohort(cohort, require_in_scope=True)


def _case(case_id: str) -> dict[str, object]:
    return {
        "case_id": case_id,
        "data_origin": "ISPY2",
        "subtype": "TNBC",
        "treatment_regimen": "AC-T chemotherapy",
        "er_status": "negative",
        "pr_status": "negative",
        "her2_status": "negative",
        "baseline_day": 0,
        "baseline_volume_ml": 24,
        "final_day": 84,
        "final_volume_ml": 6,
    }


if __name__ == "__main__":
    unittest.main()
