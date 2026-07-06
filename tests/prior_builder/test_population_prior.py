import math
import unittest

from experiments.prior_builder.parameter_contract import resolve_parameter_contract
from experiments.prior_builder.population_prior import (
    TNBC_CHEMO_POPULATION_GROUP,
    resolve_population_prior,
    sample_population_prior,
)
from experiments.prior_builder.transforms import safe_sigmoid, validate_covariance


class PopulationPriorResolutionTests(unittest.TestCase):
    def test_tnbc_chemo_prior_exposes_configured_medians(self):
        prior = resolve_population_prior(_tnbc_chemo_contract())

        self.assertEqual(prior.population_group, TNBC_CHEMO_POPULATION_GROUP)
        self.assertEqual(
            prior.parameter_names,
            (
                "growth_rate_per_day",
                "active_treatment_sensitivity",
                "resistant_fraction",
            ),
        )
        medians = {
            parameter.parameter_name: parameter.median
            for parameter in prior.parameters
        }
        self.assertEqual(medians["growth_rate_per_day"], 0.0067)
        self.assertEqual(medians["active_treatment_sensitivity"], 0.090)
        self.assertEqual(medians["resistant_fraction"], 0.20)

    def test_covariance_is_positive_semidefinite(self):
        prior = resolve_population_prior(_tnbc_chemo_contract())

        self.assertEqual(
            validate_covariance(prior.transformed_covariance, dimension=3),
            [list(row) for row in prior.transformed_covariance],
        )

    def test_correlation_signs_match_config(self):
        prior = resolve_population_prior(_tnbc_chemo_contract())

        self.assertGreater(
            prior.parameter_correlations[
                ("growth_rate_per_day", "active_treatment_sensitivity")
            ],
            0.0,
        )
        self.assertLess(
            prior.parameter_correlations[
                ("active_treatment_sensitivity", "resistant_fraction")
            ],
            0.0,
        )
        self.assertLess(
            prior.parameter_correlations[
                ("growth_rate_per_day", "resistant_fraction")
            ],
            0.0,
        )

    def test_out_of_scope_population_fails_explicitly(self):
        contract = resolve_parameter_contract(
            {
                "subtype": "HR-positive / HER2-negative",
                "treatment_context": "endocrine therapy",
            }
        )

        with self.assertRaises(ValueError):
            resolve_population_prior(contract)

    def test_layer_contribution_is_traceable_and_json_friendly(self):
        prior = resolve_population_prior(_tnbc_chemo_contract())

        contribution = prior.layer_contribution()

        self.assertEqual(contribution["layer"], "population_prior")
        self.assertEqual(contribution["medians"]["resistant_fraction"], 0.20)
        self.assertEqual(
            contribution["correlations"][
                "active_treatment_sensitivity__resistant_fraction"
            ],
            -0.40,
        )
        self.assertIsInstance(contribution["transformed_covariance"][0], list)


class PopulationPriorSamplingTests(unittest.TestCase):
    def test_sampling_reproduces_configured_medians_and_intervals(self):
        prior = resolve_population_prior(_tnbc_chemo_contract())

        result = sample_population_prior(prior, n_samples=6000, seed=29)

        samples = result.samples
        growth = sorted(sample["growth_rate_per_day"] for sample in samples)
        sensitivity = sorted(
            sample["active_treatment_sensitivity"] for sample in samples
        )
        resistance = sorted(sample["resistant_fraction"] for sample in samples)

        self.assertAlmostEqual(_percentile(growth, 0.50), 0.0067, delta=0.0005)
        self.assertAlmostEqual(_percentile(growth, 0.10), 0.0030, delta=0.0005)
        self.assertAlmostEqual(_percentile(growth, 0.90), 0.0150, delta=0.0015)
        self.assertAlmostEqual(_percentile(sensitivity, 0.50), 0.090, delta=0.006)
        self.assertAlmostEqual(_percentile(sensitivity, 0.10), 0.040, delta=0.004)
        self.assertAlmostEqual(_percentile(sensitivity, 0.90), 0.180, delta=0.015)
        self.assertAlmostEqual(_percentile(resistance, 0.50), 0.20, delta=0.02)
        self.assertAlmostEqual(_percentile(resistance, 0.10), 0.08, delta=0.02)
        self.assertAlmostEqual(_percentile(resistance, 0.90), 0.45, delta=0.04)

    def test_sampling_is_reproducible_with_fixed_seed(self):
        prior = resolve_population_prior(_tnbc_chemo_contract())

        first = sample_population_prior(prior, n_samples=5, seed=11)
        second = sample_population_prior(prior, n_samples=5, seed=11)

        self.assertEqual(first.samples, second.samples)

    def test_transformed_medians_map_back_to_configured_values(self):
        prior = resolve_population_prior(_tnbc_chemo_contract())

        self.assertAlmostEqual(
            math.exp(prior.transformed_means["log_growth_rate_per_day"]),
            0.0067,
        )
        self.assertAlmostEqual(
            math.exp(prior.transformed_means["log_active_treatment_sensitivity"]),
            0.090,
        )
        self.assertAlmostEqual(
            safe_sigmoid(prior.transformed_means["logit_resistant_fraction"]),
            0.20,
        )


def _tnbc_chemo_contract():
    return resolve_parameter_contract(
        {
            "subtype": "TNBC",
            "regimen_name": "A/C-T neoadjuvant chemotherapy",
        }
    )


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
