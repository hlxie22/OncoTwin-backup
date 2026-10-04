from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from experiments.prior_builder.interval_calibration import (
    apply_interval_calibrator,
    load_interval_calibrator,
)


ARTIFACT = Path(
    "configs/prior/"
    "v1_d1_constant_log_radius_posthoc_v0_1.json"
)


class IntervalCalibrationTest(unittest.TestCase):
    def test_candidate_requires_explicit_opt_in(self):
        with self.assertRaisesRegex(
            ValueError,
            "allow_candidate=True",
        ):
            load_interval_calibrator(ARTIFACT)

    def test_loads_fitted_candidate_with_opt_in(self):
        calibrator = load_interval_calibrator(
            ARTIFACT,
            allow_candidate=True,
        )

        self.assertEqual(
            calibrator.artifact_version,
            "oncotwin_interval_calibrator_v1",
        )
        self.assertEqual(
            calibrator.model_version,
            "v1_d1_constant_log_radius_posthoc_v0_1",
        )
        self.assertEqual(
            calibrator.status,
            "candidate_posthoc",
        )
        self.assertAlmostEqual(
            calibrator.log_radius_80,
            0.7588707883051214,
        )
        self.assertAlmostEqual(
            calibrator.log_radius_95,
            1.1440564261771882,
        )

    def test_application_preserves_point_and_input(self):
        calibrator = load_interval_calibrator(
            ARTIFACT,
            allow_candidate=True,
        )

        raw = {
            "point_ml": 10.0,
            "lower_80_ml": 7.0,
            "upper_80_ml": 14.0,
            "lower_95_ml": 5.0,
            "upper_95_ml": 20.0,
        }
        original = copy.deepcopy(raw)

        calibrated = apply_interval_calibrator(
            raw,
            calibrator,
            source_prediction="layer4_mri_qc_static",
        )

        factor_80 = math.exp(
            calibrator.log_radius_80
        )
        factor_95 = math.exp(
            calibrator.log_radius_95
        )

        self.assertEqual(raw, original)
        self.assertEqual(
            calibrated["point_ml"],
            raw["point_ml"],
        )
        self.assertAlmostEqual(
            calibrated["lower_80_ml"],
            10.0 / factor_80,
        )
        self.assertAlmostEqual(
            calibrated["upper_80_ml"],
            10.0 * factor_80,
        )
        self.assertAlmostEqual(
            calibrated["lower_95_ml"],
            10.0 / factor_95,
        )
        self.assertAlmostEqual(
            calibrated["upper_95_ml"],
            10.0 * factor_95,
        )

        audit = calibrated["calibration"]

        self.assertFalse(
            audit["point_prediction_changed"]
        )
        self.assertEqual(
            audit["source_prediction"],
            "layer4_mri_qc_static",
        )
        self.assertEqual(
            audit["status"],
            "candidate_posthoc",
        )

    def test_rejects_nonpositive_point_prediction(self):
        calibrator = load_interval_calibrator(
            ARTIFACT,
            allow_candidate=True,
        )

        with self.assertRaisesRegex(
            ValueError,
            "prediction.point_ml must be positive",
        ):
            apply_interval_calibrator(
                {"point_ml": 0.0},
                calibrator,
                source_prediction="layer4_mri_qc_static",
            )

    def test_rejects_inverted_interval_levels(self):
        payload = json.loads(
            ARTIFACT.read_text(encoding="utf-8")
        )
        payload["parameters"]["95"]["log_radius"] = 0.5
        payload["parameters"]["95"][
            "multiplicative_factor"
        ] = math.exp(0.5)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.json"
            path.write_text(
                json.dumps(payload),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "95% log radius",
            ):
                load_interval_calibrator(
                    path,
                    allow_candidate=True,
                )


if __name__ == "__main__":
    unittest.main()
