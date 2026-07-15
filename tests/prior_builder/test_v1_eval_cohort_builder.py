from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from evals.prior_stack.v1_real_data_eval import load_real_cohort
from experiments.prior_builder.v1_eval_cohort_builder import (
    build_v1_prior_eval_cohort,
    inspect_v1_data_columns,
)


class V1EvalCohortBuilderTest(unittest.TestCase):
    def test_builds_wide_v1a_cohort_with_summary_and_exclusions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            measurements = root / "ispy2_measurements.csv"
            measurements.write_text(
                "case_id,subtype,treatment_regimen,baseline_day,baseline_volume_ml,final_day,final_volume_ml,early_day,early_volume_ml,er_status,pr_status,her2_status,segmentation_qc\n"
                "registry_001,TNBC,AC-T chemotherapy,0,24,84,6,21,18,negative,negative,negative,high\n"
                "registry_002,TNBC,AC-T chemotherapy,0,31,84,,21,22,negative,negative,negative,high\n"
                "registry_003,HR positive,AC-T chemotherapy,0,19,84,11,,,,,\n"
                "synthetic_004,TNBC,AC-T chemotherapy,0,12,84,4,,,,,\n",
                encoding="utf-8",
            )
            output = root / "ispy2_v1_prior_eval_cohort.jsonl"

            result = build_v1_prior_eval_cohort(measurements, output_path=output)
            cohort_rows = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            exclusions = [
                json.loads(line)
                for line in result.exclusions_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            eval_cases = load_real_cohort(output)

        self.assertEqual([row["case_id"] for row in cohort_rows], ["registry_001"])
        self.assertEqual(eval_cases[0]["case_id"], "registry_001")
        self.assertEqual(cohort_rows[0]["baseline_volume_ml"], 24.0)
        self.assertEqual(cohort_rows[0]["early_volume_ml"], 18.0)
        self.assertEqual(cohort_rows[0]["data_origin"], "ISPY2")
        self.assertEqual(result.summary["included_rows"], 1)
        self.assertEqual(result.summary["excluded_rows"], 3)
        self.assertEqual(result.summary["v1a_in_scope_count"], 1)
        self.assertEqual(result.summary["biomarker_completeness"]["er_status"], 1)
        self.assertEqual(result.summary["mri_feature_completeness"]["segmentation_qc"], 1)
        reason_counts = result.summary["excluded_reason_counts"]
        self.assertEqual(reason_counts["missing_final_volume"], 1)
        self.assertEqual(reason_counts["out_of_v1a_scope"], 1)
        self.assertEqual(reason_counts["synthetic_or_demo_data"], 1)
        self.assertTrue(any(row["excluded_reason"] == "missing_final_volume" for row in exclusions))

    def test_builds_long_cohort_with_clinical_merge_and_nominal_days(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            measurements = root / "long.csv"
            measurements.write_text(
                "patient_id,timepoint,functional_tumor_volume_ml\n"
                "registry_001,T0,30\n"
                "registry_001,T1,20\n"
                "registry_001,T3,5\n",
                encoding="utf-8",
            )
            clinical = root / "clinical.csv"
            clinical.write_text(
                "patient_id,er_status,pr_status,her2_status,treatment_context,grade\n"
                "registry_001,negative,negative,negative,neoadjuvant chemotherapy,3\n",
                encoding="utf-8",
            )
            output = root / "cohort.jsonl"

            result = build_v1_prior_eval_cohort(
                measurements,
                output_path=output,
                clinical_path=clinical,
                use_nominal_ispy2_days=True,
            )
            row = json.loads(output.read_text(encoding="utf-8").strip())

        self.assertEqual(result.summary["included_rows"], 1)
        self.assertEqual(result.summary["excluded_rows"], 0)
        self.assertEqual(row["case_id"], "registry_001")
        self.assertEqual(row["baseline_day"], 0.0)
        self.assertEqual(row["early_day"], 21.0)
        self.assertEqual(row["final_day"], 140.0)
        self.assertEqual(row["baseline_volume_ml"], 30.0)
        self.assertEqual(row["final_volume_ml"], 5.0)
        self.assertEqual(row["grade"], "3")

    def test_default_treatment_context_only_fills_missing_treatment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            measurements = root / "measurements.csv"
            measurements.write_text(
                "case_id,er_status,pr_status,her2_status,baseline_day,baseline_volume_ml,final_day,final_volume_ml\n"
                "registry_001,negative,negative,negative,0,10,84,2\n",
                encoding="utf-8",
            )
            output = root / "cohort.jsonl"

            result = build_v1_prior_eval_cohort(
                measurements,
                output_path=output,
                default_treatment_context="neoadjuvant chemotherapy",
            )
            row = json.loads(output.read_text(encoding="utf-8").strip())

        self.assertEqual(result.summary["included_rows"], 1)
        self.assertEqual(row["treatment_context"], "neoadjuvant chemotherapy")

    def test_explicit_data_origin_fills_only_missing_origin(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            measurements = root / "measurements.csv"
            measurements.write_text(
                "case_id,data_origin,er_status,pr_status,her2_status,treatment_context,baseline_day,baseline_volume_ml,final_day,final_volume_ml\n"
                "registry_001,source_kept,negative,negative,negative,neoadjuvant chemotherapy,0,10,84,2\n"
                "registry_002,,negative,negative,negative,neoadjuvant chemotherapy,0,12,84,3\n",
                encoding="utf-8",
            )
            output = root / "cohort.jsonl"

            result = build_v1_prior_eval_cohort(
                measurements,
                output_path=output,
                data_origin="explicit_registry",
            )
            rows = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(result.summary["included_rows"], 2)
        self.assertEqual(
            {row["case_id"]: row["data_origin"] for row in rows},
            {
                "registry_001": "source_kept",
                "registry_002": "explicit_registry",
            },
        )


    def test_source_dataset_is_normalized_to_primary_data_origin_field(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            measurements = root / "measurements.csv"
            measurements.write_text(
                "case_id,source_dataset,er_status,pr_status,her2_status,treatment_context,baseline_day,baseline_volume_ml,final_day,final_volume_ml\n"
                "registry_001,ISPY2_metadata_export,negative,negative,negative,neoadjuvant chemotherapy,0,10,84,2\n",
                encoding="utf-8",
            )
            output = root / "cohort.jsonl"

            result = build_v1_prior_eval_cohort(measurements, output_path=output)
            row = json.loads(output.read_text(encoding="utf-8").strip())

        self.assertEqual(result.summary["included_rows"], 1)
        self.assertEqual(row["source_dataset"], "ISPY2_metadata_export")
        self.assertEqual(row["data_origin"], "ISPY2_metadata_export")

    def test_inspects_candidate_table_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            table = Path(tmpdir) / "columns.csv"
            table.write_text(
                "patient_id,day,tumor_volume_ml,er_status,treatment_regimen\n"
                "registry_001,0,12,negative,AC-T chemotherapy\n"
                "registry_001,84,4,negative,AC-T chemotherapy\n",
                encoding="utf-8",
            )

            inspection = inspect_v1_data_columns(table)

        self.assertEqual(inspection.row_count, 2)
        self.assertIn("patient_id", inspection.suspected_roles["case_id"])
        self.assertIn("day", inspection.suspected_roles["long_day"])
        self.assertIn("tumor_volume_ml", inspection.suspected_roles["long_volume"])
        self.assertEqual(inspection.non_empty_counts["tumor_volume_ml"], 2)


if __name__ == "__main__":
    unittest.main()
