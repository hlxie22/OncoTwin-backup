from __future__ import annotations

import copy
import unittest

from experiments.twin_runtime.layer7_candidate import (
    apply_layer7_fail_closed_candidate,
)


class Layer7FailClosedCandidateTest(
    unittest.TestCase
):
    def test_healthy_posterior_is_accepted(self):
        case = self._case(
            health_status="updated_healthy",
            ess_fraction=0.50,
        )

        result = (
            apply_layer7_fail_closed_candidate(
                case
            )
        )

        self.assertEqual(
            result["point_ml"],
            case["posterior_prediction"][
                "point_ml"
            ],
        )
        self.assertFalse(
            result["layer7"]["fallback_used"]
        )
        self.assertEqual(
            result["layer7"]["status"],
            "updated_accepted",
        )

    def test_low_ess_is_exact_prior_fallback(self):
        case = self._case(
            health_status="updated_low_ess",
            ess_fraction=0.04,
        )

        result = (
            apply_layer7_fail_closed_candidate(
                case
            )
        )

        for key, value in (
            case["prior_prediction"].items()
        ):
            self.assertEqual(
                result[key],
                value,
            )

        self.assertTrue(
            result["layer7"]["fallback_used"]
        )
        self.assertEqual(
            result["layer7"][
                "selected_prediction"
            ],
            "mechanistic_prior",
        )

    def test_boundary_collapse_is_prior_fallback(self):
        case = self._case(
            health_status=(
                "updated_boundary_collapse"
            ),
            ess_fraction=0.80,
        )

        result = (
            apply_layer7_fail_closed_candidate(
                case
            )
        )

        self.assertEqual(
            result["point_ml"],
            case["prior_prediction"][
                "point_ml"
            ],
        )
        self.assertTrue(
            result["layer7"]["fallback_used"]
        )

    def test_missing_observation_is_exact_prior(self):
        case = self._case(
            health_status="updated_healthy",
            ess_fraction=0.80,
        )
        case["early_day"] = None

        result = (
            apply_layer7_fail_closed_candidate(
                case
            )
        )

        self.assertEqual(
            {
                key: result[key]
                for key in (
                    "point_ml",
                    "lower_80_ml",
                    "upper_80_ml",
                    "lower_95_ml",
                    "upper_95_ml",
                )
            },
            case["prior_prediction"],
        )
        self.assertEqual(
            result["layer7"]["status"],
            "not_updated_no_valid_observation",
        )

    def test_final_outcome_cannot_change_selection(self):
        first = self._case(
            health_status="updated_healthy",
            ess_fraction=0.50,
        )
        second = copy.deepcopy(first)

        first[
            "observed_final_volume_ml"
        ] = 0.001
        second[
            "observed_final_volume_ml"
        ] = 999999.0

        self.assertEqual(
            apply_layer7_fail_closed_candidate(
                first
            ),
            apply_layer7_fail_closed_candidate(
                second
            ),
        )

    @staticmethod
    def _case(
        *,
        health_status,
        ess_fraction,
    ):
        return {
            "case_id": "test-case",
            "early_day": 42.0,
            "observed_final_volume_ml": 4.0,
            "prior_prediction": {
                "point_ml": 3.0,
                "lower_80_ml": 2.0,
                "upper_80_ml": 4.0,
                "lower_95_ml": 1.0,
                "upper_95_ml": 5.0,
            },
            "posterior_prediction": {
                "point_ml": 4.0,
                "lower_80_ml": 3.0,
                "upper_80_ml": 5.0,
                "lower_95_ml": 2.0,
                "upper_95_ml": 6.0,
            },
            "posterior_health": {
                "status": health_status,
                "weights": {
                    "effective_sample_size_fraction": (
                        ess_fraction
                    )
                },
            },
        }


if __name__ == "__main__":
    unittest.main()
