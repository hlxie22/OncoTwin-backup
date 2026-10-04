from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from experiments.prior_builder.early_response_update import (
    apply_early_response_update,
    load_early_response_updater,
)


ARTIFACT = Path(
    "configs/prior/"
    "v1_d1_continuous_early_volume_posthoc_v0_1.json"
)


class EarlyResponseUpdateTest(unittest.TestCase):
    def test_candidate_requires_explicit_opt_in(self):
        with self.assertRaisesRegex(
            ValueError,
            "allow_candidate=True",
        ):
            load_early_response_updater(ARTIFACT)

    def test_loads_exact_fitted_parameters(self):
        updater = load_early_response_updater(
            ARTIFACT,
            allow_candidate=True,
        )

        self.assertEqual(
            updater.artifact_version,
            "oncotwin_early_response_update_v1",
        )
        self.assertEqual(
            updater.model_version,
            "v1_d1_continuous_early_volume_posthoc_v0_1",
        )
        self.assertAlmostEqual(
            updater.intercept,
            -0.07390443919179766,
        )
        self.assertAlmostEqual(
            updater.slope,
            0.7127777605265024,
        )
        self.assertAlmostEqual(
            updater.missing_early_log_correction,
            -0.04955888829518541,
        )

    def test_applies_continuous_early_volume_update(self):
        updater = load_early_response_updater(
            ARTIFACT,
            allow_candidate=True,
        )
        source = {"point_ml": 10.0}
        case = {
            "baseline_volume_ml": 20.0,
            "early_volume_ml": 10.0,
        }
        original_source = copy.deepcopy(source)
        original_case = copy.deepcopy(case)

        result = apply_early_response_update(
            source,
            case,
            updater,
        )

        expected_log_correction = (
            updater.intercept
            + updater.slope * math.log(0.5)
        )
        expected_point = (
            10.0 * math.exp(expected_log_correction)
        )

        self.assertEqual(source, original_source)
        self.assertEqual(case, original_case)
        self.assertAlmostEqual(
            result["point_ml"],
            expected_point,
        )

        audit = result["early_response_update"]

        self.assertTrue(audit["early_available"])
        self.assertEqual(
            audit["correction_source"],
            "continuous_early_volume",
        )
        self.assertAlmostEqual(
            audit["early_to_baseline_ratio"],
            0.5,
        )
        self.assertFalse(
            audit["correction_clipped"]
        )

    def test_missing_early_uses_fallback(self):
        updater = load_early_response_updater(
            ARTIFACT,
            allow_candidate=True,
        )

        result = apply_early_response_update(
            {"point_ml": 10.0},
            {
                "baseline_volume_ml": 20.0,
                "early_volume_ml": None,
            },
            updater,
        )

        expected = 10.0 * math.exp(
            updater.missing_early_log_correction
        )

        self.assertAlmostEqual(
            result["point_ml"],
            expected,
        )
        self.assertFalse(
            result["early_response_update"][
                "early_available"
            ]
        )
        self.assertEqual(
            result["early_response_update"][
                "correction_source"
            ],
            "missing_early_fallback",
        )

    def test_correction_is_bounded(self):
        updater = load_early_response_updater(
            ARTIFACT,
            allow_candidate=True,
        )

        cases = (
            (
                1_000_000.0,
                updater.max_log_correction,
            ),
            (
                0.000001,
                updater.min_log_correction,
            ),
        )

        for early_volume, expected_bound in cases:
            with self.subTest(
                early_volume=early_volume
            ):
                result = apply_early_response_update(
                    {"point_ml": 10.0},
                    {
                        "baseline_volume_ml": 1.0,
                        "early_volume_ml": early_volume,
                    },
                    updater,
                )

                audit = result[
                    "early_response_update"
                ]

                self.assertTrue(
                    audit["correction_clipped"]
                )
                self.assertAlmostEqual(
                    audit["applied_log_correction"],
                    expected_bound,
                )

    def test_rejects_invalid_volumes(self):
        updater = load_early_response_updater(
            ARTIFACT,
            allow_candidate=True,
        )

        invalid_cases = (
            (
                {"point_ml": 0.0},
                {
                    "baseline_volume_ml": 1.0,
                    "early_volume_ml": 0.5,
                },
                "source_prediction.point_ml",
            ),
            (
                {"point_ml": 1.0},
                {
                    "baseline_volume_ml": 0.0,
                    "early_volume_ml": 0.5,
                },
                "case.baseline_volume_ml",
            ),
            (
                {"point_ml": 1.0},
                {
                    "baseline_volume_ml": 1.0,
                    "early_volume_ml": 0.0,
                },
                "case.early_volume_ml",
            ),
        )

        for source, case, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(
                    ValueError,
                    message,
                ):
                    apply_early_response_update(
                        source,
                        case,
                        updater,
                    )

    def test_rejects_invalid_correction_bounds(self):
        payload = json.loads(
            ARTIFACT.read_text(encoding="utf-8")
        )
        payload["parameters"][
            "min_log_correction"
        ] = 1.0
        payload["parameters"][
            "max_log_correction"
        ] = -1.0

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.json"
            path.write_text(
                json.dumps(payload),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "min_log_correction",
            ):
                load_early_response_updater(
                    path,
                    allow_candidate=True,
                )


if __name__ == "__main__":
    unittest.main()
