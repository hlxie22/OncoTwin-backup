from __future__ import annotations

import unittest

from experiments.twin_runtime.posterior_diagnostics import (
    summarize_posterior_health,
)


class PosteriorDiagnosticsTest(
    unittest.TestCase
):
    def test_uniform_weights_are_healthy(self):
        result = summarize_posterior_health(
            self._posterior(
                [0.25, 0.25, 0.25, 0.25]
            )
        )

        self.assertEqual(
            result["status"],
            "updated_healthy",
        )
        self.assertAlmostEqual(
            result["weights"][
                "effective_sample_size"
            ],
            4.0,
        )
        self.assertAlmostEqual(
            result["weights"][
                "maximum_weight"
            ],
            0.25,
        )

    def test_runtime_low_ess_status_is_preserved(self):
        posterior = self._posterior(
            [0.97, 0.01, 0.01, 0.01]
        )
        posterior["fallback_status"] = (
            "tempered_smc_recommended"
        )

        result = summarize_posterior_health(
            posterior
        )

        self.assertEqual(
            result["status"],
            "updated_low_ess",
        )
        self.assertLess(
            result["weights"][
                "effective_sample_size_fraction"
            ],
            0.30,
        )

    @staticmethod
    def _posterior(
        weights,
    ):
        parameters = [
            (0.004, 0.04, 0.10),
            (0.006, 0.08, 0.20),
            (0.008, 0.12, 0.30),
            (0.010, 0.16, 0.40),
        ]

        rows = []
        for index, (
            growth,
            sensitivity,
            resistance,
        ) in enumerate(parameters):
            rows.append(
                {
                    "particle_id": f"p{index}",
                    "parameters": {
                        "growth_rate": growth,
                        "drug_sensitivity": {
                            "anthracycline": (
                                sensitivity
                            ),
                            "taxane": sensitivity,
                        },
                        "resistant_fraction": (
                            resistance
                        ),
                    },
                    "weight": weights[index],
                    "log_prior_weight": (
                        -1.3862943611198906
                    ),
                    "log_likelihood": (
                        -float(index)
                    ),
                    "log_weight": (
                        -float(index)
                    ),
                }
            )

        return {
            "particle_trajectories": rows,
            "fallback_status": "not_needed",
        }


if __name__ == "__main__":
    unittest.main()
