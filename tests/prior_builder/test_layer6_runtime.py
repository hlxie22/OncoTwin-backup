from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

from experiments.prior_builder.layer6_runtime import (
    LAYER6_RUNTIME_VERSION,
    load_layer6_ensemble,
    predict_layer6,
)


ARTIFACT = Path(
    "artifacts/prior_builder/layer6/"
    "v1_late_response_ensemble"
)


@unittest.skipUnless(
    ARTIFACT.is_dir(),
    "deployable Layer 6 artifact has not been generated",
)
class Layer6RuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ensemble = load_layer6_ensemble(
            ARTIFACT
        )

    def test_loads_five_member_frozen_ensemble(self):
        self.assertEqual(
            len(self.ensemble.members),
            5,
        )
        self.assertEqual(
            self.ensemble.training_scale_days,
            84.0,
        )
        self.assertAlmostEqual(
            self.ensemble.neural_weight,
            0.75,
        )
        self.assertAlmostEqual(
            self.ensemble.fallback_weight,
            0.25,
        )
        self.assertLess(
            self.ensemble.q80,
            self.ensemble.q95,
        )

    def test_unsupported_active_case_falls_back(self):
        fallback = self._fallback_prediction()

        result = predict_layer6(
            self._case(),
            self._static_prediction(),
            fallback,
            self.ensemble,
        )

        self.assertEqual(
            result["layer6"]["runtime_version"],
            LAYER6_RUNTIME_VERSION,
        )
        self.assertEqual(
            result["layer6"]["status"],
            "active_ood_fallback",
        )
        self.assertTrue(
            result["layer6"]["fallback_used"]
        )
        self.assertIn(
            "support_diagnostics",
            result["layer6"],
        )

        for key, value in fallback.items():
            self.assertEqual(
                result[key],
                value,
            )

    def test_inactive_case_returns_fallback_unchanged(self):
        fallback = self._fallback_prediction()
        case = self._case()
        case["early_day"] = None
        case["early_volume_ml"] = None

        result = predict_layer6(
            case,
            self._static_prediction(),
            fallback,
            self.ensemble,
        )

        for key, value in fallback.items():
            self.assertEqual(
                result[key],
                value,
            )

        self.assertEqual(
            result["layer6"]["status"],
            "inactive_fallback",
        )
        self.assertTrue(
            result["layer6"]["fallback_used"]
        )
        self.assertEqual(
            result["layer6"]["activation"]["status"],
            "inactive_missing_early_volume",
        )

    def test_prediction_is_deterministic(self):
        first = predict_layer6(
            self._case(),
            self._static_prediction(),
            self._fallback_prediction(),
            self.ensemble,
        )
        second = predict_layer6(
            self._case(),
            self._static_prediction(),
            self._fallback_prediction(),
            self.ensemble,
        )

        for key in (
            "point_ml",
            "lower_80_ml",
            "upper_80_ml",
            "lower_95_ml",
            "upper_95_ml",
        ):
            self.assertEqual(
                first[key],
                second[key],
            )

    def test_manifest_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "artifact"
            shutil.copytree(
                ARTIFACT,
                copied,
            )

            checksums_path = (
                copied / "checksums.json"
            )
            checksums = json.loads(
                checksums_path.read_text(
                    encoding="utf-8"
                )
            )
            checksums["manifest.json"] = "0" * 64
            checksums_path.write_text(
                json.dumps(
                    checksums,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "manifest hash mismatch",
            ):
                load_layer6_ensemble(copied)

    @staticmethod
    def _case() -> dict[str, object]:
        return {
            "case_id": "runtime_test_case",
            "baseline_day": 0.0,
            "baseline_volume_ml": 0.8,
            "early_day": 42.0,
            "early_volume_ml": 0.4,
            "final_day": 126.0,
            "context": {},
        }

    @staticmethod
    def _static_prediction() -> dict[str, float]:
        return {
            "point_ml": 0.35,
            "lower_80_ml": 0.15,
            "upper_80_ml": 0.75,
            "lower_95_ml": 0.08,
            "upper_95_ml": 1.10,
        }

    @staticmethod
    def _fallback_prediction() -> dict[str, float]:
        return {
            "point_ml": 0.25,
            "lower_80_ml": 0.12,
            "upper_80_ml": 0.48,
            "lower_95_ml": 0.06,
            "upper_95_ml": 0.72,
        }


if __name__ == "__main__":
    unittest.main()
