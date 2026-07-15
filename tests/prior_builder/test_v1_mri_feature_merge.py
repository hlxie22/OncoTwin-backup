from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from experiments.prior_builder.v1_mri_feature_merge import (
    merge_mri_features_into_v1_cohort,
)


class V1MRIFeatureMergeTest(unittest.TestCase):
    def test_merges_available_features_and_preserves_missing_rows_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cohort = root / "cohort.jsonl"
            features = root / "mri_features.jsonl"
            output = root / "cohort_with_mri.jsonl"

            _write_jsonl(
                cohort,
                [
                    _cohort_row("registry_001"),
                    _cohort_row("registry_002"),
                ],
            )
            _write_jsonl(
                features,
                [
                    {
                        "case_id": "registry_001",
                        "source_image": "images/registry_001.nii.gz",
                        "source_mask": "masks/registry_001.nii.gz",
                        "tumor_volume_ml": 23.5,
                        "functional_tumor_volume_ml": 21.0,
                        "enhancement_mean": 1.3,
                        "enhancement_std": 0.22,
                        "low_enhancement_fraction": 0.18,
                        "mask_voxels": 23500,
                        "voxel_volume_ml": 0.001,
                        "connected_component_count": 1,
                        "segmentation_qc": "high",
                        "registration_qc": "medium",
                    }
                ],
            )

            result = merge_mri_features_into_v1_cohort(
                cohort,
                features,
                output_path=output,
            )
            rows = _read_jsonl(output)
            summary = json.loads(result.summary_path.read_text(encoding="utf-8"))

        self.assertEqual([row["case_id"] for row in rows], ["registry_001", "registry_002"])
        self.assertEqual(rows[0]["mri_features"]["mri_feature_status"], "available")
        self.assertEqual(rows[0]["mri_features"]["tumor_volume_ml"], 23.5)
        self.assertEqual(rows[0]["mri_features"]["volume_ml"], 23.5)
        self.assertEqual(rows[0]["mri_features"]["segmentation_qc"], "high")
        self.assertEqual(rows[1]["mri_features"]["mri_feature_status"], "missing")
        self.assertEqual(result.summary["matched_feature_rows"], 1)
        self.assertEqual(result.summary["missing_feature_rows"], 1)
        self.assertEqual(result.summary["output_rows"], 2)
        self.assertEqual(summary["mri_feature_completeness"]["tumor_volume_ml"], 1)
        self.assertEqual(summary["mri_feature_completeness"]["segmentation_qc"], 2)

    def test_drop_missing_features_creates_feature_complete_subset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cohort = root / "cohort.jsonl"
            features = root / "mri_features.jsonl"
            output = root / "complete.jsonl"

            _write_jsonl(cohort, [_cohort_row("registry_001"), _cohort_row("registry_002")])
            _write_jsonl(
                features,
                [
                    {
                        "case_id": "registry_001",
                        "tumor_volume_ml": 11.0,
                        "mask_voxels": 11000,
                        "segmentation_qc": "medium",
                    }
                ],
            )

            result = merge_mri_features_into_v1_cohort(
                cohort,
                features,
                output_path=output,
                drop_missing_features=True,
            )
            rows = _read_jsonl(output)

        self.assertEqual([row["case_id"] for row in rows], ["registry_001"])
        self.assertEqual(result.summary["dropped_missing_feature_rows"], 1)
        self.assertEqual(result.summary["output_rows"], 1)

    def test_failed_or_empty_mask_features_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cohort = root / "cohort.jsonl"
            features = root / "mri_features.jsonl"
            output = root / "cohort_with_mri.jsonl"

            _write_jsonl(cohort, [_cohort_row("registry_001")])
            _write_jsonl(
                features,
                [
                    {
                        "case_id": "registry_001",
                        "source_image": "images/registry_001.nii.gz",
                        "source_mask": "masks/registry_001.nii.gz",
                        "tumor_volume_ml": 0,
                        "functional_tumor_volume_ml": 0,
                        "enhancement_std": 0.9,
                        "low_enhancement_fraction": 0.5,
                        "mask_voxels": 0,
                        "segmentation_qc": "high",
                    }
                ],
            )

            result = merge_mri_features_into_v1_cohort(
                cohort,
                features,
                output_path=output,
            )
            row = _read_jsonl(output)[0]
            mri = row["mri_features"]

        self.assertEqual(mri["mri_feature_status"], "failed")
        self.assertEqual(mri["segmentation_qc"], "failed")
        self.assertEqual(mri["source_mask"], "masks/registry_001.nii.gz")
        self.assertIn("empty_or_nonpositive_mask", mri["warnings"])
        self.assertIn("non_positive_tumor_volume_ml", mri["warnings"])
        self.assertNotIn("tumor_volume_ml", mri)
        self.assertNotIn("functional_tumor_volume_ml", mri)
        self.assertNotIn("enhancement_std", mri)
        self.assertEqual(result.summary["feature_status_counts"]["failed"], 1)
        self.assertNotIn("tumor_volume_ml", result.summary["numeric_feature_copy_counts"])

    def test_duplicate_feature_rows_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cohort = root / "cohort.jsonl"
            features = root / "mri_features.jsonl"
            _write_jsonl(cohort, [_cohort_row("registry_001")])
            _write_jsonl(
                features,
                [
                    {"case_id": "registry_001", "tumor_volume_ml": 10, "mask_voxels": 100},
                    {"case_id": "registry_001", "tumor_volume_ml": 11, "mask_voxels": 110},
                ],
            )

            with self.assertRaisesRegex(ValueError, "duplicate MRI feature row"):
                merge_mri_features_into_v1_cohort(cohort, features, output_path=root / "out.jsonl")


def _cohort_row(case_id: str) -> dict[str, object]:
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


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

