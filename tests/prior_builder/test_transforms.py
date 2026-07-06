import math
import unittest

from experiments.prior_builder.transforms import (
    NormalApproximation,
    from_transformed,
    median_interval_to_normal,
    safe_logit,
    safe_sigmoid,
    sample_correlated_transformed_prior,
    to_transformed,
    validate_covariance,
)


class TransformTests(unittest.TestCase):
    def test_round_trip_transforms_learnable_parameters(self):
        params = {
            "growth_rate_per_day": 0.0067,
            "active_treatment_sensitivity": 0.09,
            "resistant_fraction": 0.20,
        }

        round_tripped = from_transformed(to_transformed(params))

        self.assertAlmostEqual(round_tripped["growth_rate_per_day"], 0.0067)
        self.assertAlmostEqual(round_tripped["active_treatment_sensitivity"], 0.09)
        self.assertAlmostEqual(round_tripped["resistant_fraction"], 0.20)

    def test_near_zero_resistant_fraction_stays_finite(self):
        z = safe_logit(1e-12)

        self.assertTrue(math.isfinite(z))
        self.assertGreater(safe_sigmoid(z), 0.0)

    def test_near_one_resistant_fraction_stays_finite(self):
        z = safe_logit(1.0 - 1e-12)

        self.assertTrue(math.isfinite(z))
        self.assertLess(safe_sigmoid(z), 1.0)

    def test_fraction_outside_unit_interval_fails(self):
        with self.assertRaises(ValueError):
            safe_logit(-0.01)
        with self.assertRaises(ValueError):
            safe_logit(1.01)


class NormalApproximationTests(unittest.TestCase):
    def test_median_interval_to_normal_in_log_space(self):
        approximation = median_interval_to_normal(
            0.0067,
            0.003,
            0.015,
            transform="log",
        )

        self.assertIsInstance(approximation, NormalApproximation)
        self.assertAlmostEqual(approximation.mean, math.log(0.0067))
        self.assertGreater(approximation.std, 0.0)

    def test_invalid_interval_fails(self):
        with self.assertRaises(ValueError):
            median_interval_to_normal(0.2, 0.3, 0.1, transform="logit")
        with self.assertRaises(ValueError):
            median_interval_to_normal(0.8, 0.1, 0.6, transform="logit")


class CovarianceTests(unittest.TestCase):
    def test_positive_semidefinite_covariance_passes(self):
        covariance = [
            [0.10, 0.05, 0.00],
            [0.05, 0.10, 0.00],
            [0.00, 0.00, 0.02],
        ]

        self.assertEqual(validate_covariance(covariance, dimension=3), covariance)

    def test_singular_positive_semidefinite_covariance_passes(self):
        covariance = [
            [1.0, 1.0],
            [1.0, 1.0],
        ]

        self.assertEqual(validate_covariance(covariance, dimension=2), covariance)

    def test_invalid_covariance_fails(self):
        with self.assertRaises(ValueError):
            validate_covariance([[1.0, 0.0], [0.0]], dimension=2)
        with self.assertRaises(ValueError):
            validate_covariance([[1.0, 2.0], [2.0, 1.0]], dimension=2)
        with self.assertRaises(ValueError):
            validate_covariance([[1.0, 0.2], [0.1, 1.0]], dimension=2)


class SamplingTests(unittest.TestCase):
    def test_sampling_is_reproducible_with_fixed_seed(self):
        means = {
            "log_growth_rate_per_day": -5.0,
            "log_active_treatment_sensitivity": -2.4,
            "logit_resistant_fraction": -1.4,
        }
        covariance = [
            [0.20, 0.02, 0.00],
            [0.02, 0.15, -0.04],
            [0.00, -0.04, 0.30],
        ]

        first = sample_correlated_transformed_prior(
            means,
            covariance,
            n_samples=5,
            seed=17,
        )
        second = sample_correlated_transformed_prior(
            means,
            covariance,
            n_samples=5,
            seed=17,
        )

        self.assertEqual(first, second)

    def test_sample_quantiles_approximately_match_configured_normal(self):
        approximation = median_interval_to_normal(0.20, 0.10, 0.35, transform="logit")
        means = {"logit_resistant_fraction": approximation.mean}
        covariance = [[approximation.std**2]]

        samples = sample_correlated_transformed_prior(
            means,
            covariance,
            n_samples=5000,
            seed=23,
        )
        fractions = sorted(
            safe_sigmoid(sample["logit_resistant_fraction"]) for sample in samples
        )

        self.assertAlmostEqual(_percentile(fractions, 0.50), 0.20, delta=0.02)
        self.assertAlmostEqual(_percentile(fractions, 0.10), 0.10, delta=0.03)
        self.assertAlmostEqual(_percentile(fractions, 0.90), 0.35, delta=0.04)


def _percentile(values, fraction):
    index = (len(values) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return values[int(index)]
    weight = index - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


if __name__ == "__main__":
    unittest.main()
