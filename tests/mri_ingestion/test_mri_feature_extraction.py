from __future__ import annotations

import unittest

from experiments.mri_ingestion.features.case_features import extract_case_feature_record
from experiments.mri_ingestion.features.qc import qc_mri_feature_record
from experiments.mri_ingestion.features.tumor_volume import tumor_volume_features


class MRIFeatureExtractionTest(unittest.TestCase):
    def test_computes_tumor_volume_from_spacing(self):
        features = tumor_volume_features(
            mask_voxels=500,
            voxel_spacing_mm=[1.0, 1.0, 2.0],
        )

        self.assertEqual(features["mask_voxels"], 500)
        self.assertAlmostEqual(features["voxel_volume_ml"], 0.002)
        self.assertAlmostEqual(features["tumor_volume_ml"], 1.0)

    def test_extracts_qced_feature_record_from_cached_metadata(self):
        record = extract_case_feature_record(
            {
                "case_id": "ISPY2_001",
                "source_image": "images/ISPY2_001.nii.gz",
                "source_mask": "masks/ISPY2_001.nii.gz",
                "mask_voxels": 100,
                "voxel_volume_ml": 0.01,
                "connected_component_count": 1,
                "enhancement_values": [0.5, 1.5, 2.5],
                "segmentation_qc": "medium",
                "registration_qc": "pass",
            }
        )

        self.assertEqual(record["mri_feature_status"], "available")
        self.assertEqual(record["layer4_feature_policy"], "conservative_numeric")
        self.assertAlmostEqual(record["tumor_volume_ml"], 1.0)
        self.assertAlmostEqual(record["low_enhancement_fraction"], 1.0 / 3.0)
        self.assertAlmostEqual(record["functional_tumor_volume_ml"], 2.0 / 3.0)

    def test_qc_fails_closed_for_empty_masks(self):
        record = qc_mri_feature_record(
            {
                "case_id": "ISPY2_002",
                "mask_voxels": 0,
                "tumor_volume_ml": 0,
                "voxel_volume_ml": 0.01,
                "connected_component_count": 0,
                "segmentation_qc": "high",
            }
        )

        self.assertEqual(record["mri_feature_status"], "failed")
        self.assertEqual(record["segmentation_qc"], "failed")
        self.assertEqual(record["layer4_feature_policy"], "report_only")
        self.assertIn("empty_or_nonpositive_mask", record["warnings"])

    def test_low_qc_keeps_features_but_restricts_layer4_policy(self):
        record = qc_mri_feature_record(
            {
                "case_id": "ISPY2_003",
                "mask_voxels": 50,
                "tumor_volume_ml": 1.0,
                "voxel_volume_ml": 0.02,
                "connected_component_count": 1,
                "segmentation_qc": "low",
            }
        )

        self.assertEqual(record["mri_feature_status"], "available")
        self.assertEqual(record["layer4_feature_policy"], "uncertainty_only")


if __name__ == "__main__":
    unittest.main()
