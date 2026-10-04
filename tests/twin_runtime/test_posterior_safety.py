from __future__ import annotations

import math
import unittest

from experiments.twin_runtime.posterior_safety import (
    effective_sample_size,
    fail_closed_weights,
    tempered_likelihood_weights,
)


class PosteriorSafetyTest(unittest.TestCase):
    def test_fail_closed_uses_prior_for_collapsed_update(self):
        result = fail_closed_weights(
            log_prior_weights=[
                math.log(0.25)
                for _ in range(4)
            ],
            log_likelihoods=[
                0.0,
                -100.0,
                -100.0,
                -100.0,
            ],
            ess_threshold_fraction=0.50,
        )

        self.assertEqual(
            result["status"],
            "fallback_to_prior_low_ess",
        )
        self.assertEqual(
            result["likelihood_power"],
            0.0,
        )
        self.assertEqual(
            result["weights"],
            [0.25, 0.25, 0.25, 0.25],
        )

    def test_fail_closed_accepts_healthy_update(self):
        result = fail_closed_weights(
            log_prior_weights=[
                math.log(0.25)
                for _ in range(4)
            ],
            log_likelihoods=[
                0.0,
                -0.1,
                -0.2,
                -0.3,
            ],
            ess_threshold_fraction=0.50,
        )

        self.assertEqual(
            result["status"],
            "full_update_accepted",
        )
        self.assertEqual(
            result["likelihood_power"],
            1.0,
        )

    def test_tempering_meets_ess_target(self):
        result = tempered_likelihood_weights(
            log_prior_weights=[
                math.log(0.25)
                for _ in range(4)
            ],
            log_likelihoods=[
                0.0,
                -100.0,
                -100.0,
                -100.0,
            ],
            ess_target_fraction=0.75,
        )

        self.assertEqual(
            result["status"],
            "tempered_to_ess_target",
        )
        self.assertGreater(
            result["likelihood_power"],
            0.0,
        )
        self.assertLess(
            result["likelihood_power"],
            1.0,
        )
        self.assertGreaterEqual(
            result["selected_ess_fraction"],
            0.75 - 1e-10,
        )

    def test_tempering_is_deterministic(self):
        kwargs = {
            "log_prior_weights": [
                -1.0,
                -1.0,
                -1.0,
            ],
            "log_likelihoods": [
                0.0,
                -5.0,
                -10.0,
            ],
            "ess_target_fraction": 0.50,
        }

        first = tempered_likelihood_weights(
            **kwargs
        )
        second = tempered_likelihood_weights(
            **kwargs
        )

        self.assertEqual(first, second)
        self.assertAlmostEqual(
            effective_sample_size(
                first["weights"]
            ),
            first[
                "selected_effective_sample_size"
            ],
        )


if __name__ == "__main__":
    unittest.main()
