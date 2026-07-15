from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from experiments.mri_ingestion.segmenters.mamamia_nnunet import (
    MAMAMIANnUNetConfig,
    build_predict_command,
    environment_warnings,
    validate_prediction_inputs,
)


class MAMAMIANnUNetAdapterTest(unittest.TestCase):
    def test_builds_expected_predict_command(self):
        command = build_predict_command(
            Path("data/curated/nnunet_inputs"),
            Path("data/curated/segmentations/mamamia_nnunet"),
            config=MAMAMIANnUNetConfig(folds=("0",), checkpoint="checkpoint_final.pth"),
        )

        self.assertEqual(command[:7], [
            "nnUNetv2_predict",
            "-i",
            "data/curated/nnunet_inputs",
            "-o",
            "data/curated/segmentations/mamamia_nnunet",
            "-d",
            "101",
        ])
        self.assertIn("3d_fullres", command)
        self.assertIn("checkpoint_final.pth", command)

    def test_validates_compressed_nifti_inputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "inputs"
            output_dir = root / "outputs"
            input_dir.mkdir()
            (input_dir / "case_0000.nii.gz").write_bytes(b"placeholder")

            files = validate_prediction_inputs(input_dir, output_dir)

            self.assertEqual([path.name for path in files], ["case_0000.nii.gz"])
            self.assertTrue(output_dir.exists())

    def test_rejects_uncompressed_or_non_nifti_inputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "inputs"
            input_dir.mkdir()
            (input_dir / "case_0000.nii").write_bytes(b"placeholder")

            with self.assertRaisesRegex(ValueError, "compressed NIfTI"):
                validate_prediction_inputs(input_dir, root / "outputs")

    def test_environment_warnings_are_actionable_without_running_nnunet(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            warnings = environment_warnings(
                predictor="definitely_missing_nnunet_predictor",
                nnunet_results=Path(tmpdir),
                require_weights=True,
            )

        self.assertTrue(any("PATH" in warning for warning in warnings))
        self.assertTrue(any("model weights" in warning for warning in warnings))


if __name__ == "__main__":
    unittest.main()
