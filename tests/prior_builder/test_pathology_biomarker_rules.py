import math
import unittest

from experiments.prior_builder.parameter_contract import resolve_parameter_contract
from experiments.prior_builder.pathology_biomarker_rules import (
    apply_pathology_biomarker_rules,
    sample_pathology_biomarker_prior,
)
from experiments.prior_builder.population_prior import resolve_population_prior
from experiments.prior_builder.transforms import safe_sigmoid, validate_covariance


class PathologyBiomarkerRuleTests(unittest.TestCase):
    def test_unknown_ki67_is_audit_only_without_inventing_status(self):
        base = _population_prior()

        adjusted = apply_pathology_biomarker_rules(
            base,
            {
                "ki67_percent": "unknown",
                "brca_status": "negative",
                "hrd_status": "negative",
            },
        )

        self.assertEqual(
            adjusted.transformed_means["log_growth_rate_per_day"],
            base.transformed_means["log_growth_rate_per_day"],
        )
        self.assertEqual(
            adjusted.transformed_covariance[0][0],
            base.transformed_covariance[0][0],
        )
        rule_ids = _rule_ids(adjusted)
        self.assertIn("ki67_missing_v1", rule_ids)
        self.assertNotIn("ki67_high_v1", rule_ids)
        self.assertNotIn("ki67_low_v1", rule_ids)

    def test_low_ki67_lowers_growth_prior(self):
        base = _population_prior()

        adjusted = apply_pathology_biomarker_rules(
            base,
            _known_context(ki67_percent=4),
        )

        self.assertLess(_growth_median(adjusted), _growth_median(base))
        self.assertIn("ki67_low_v1", _rule_ids(adjusted))

    def test_high_ki67_raises_growth_and_lowers_resistant_fraction(self):
        base = _population_prior()

        adjusted = apply_pathology_biomarker_rules(
            base,
            _known_context(ki67_percent=45),
        )

        self.assertGreater(_growth_median(adjusted), _growth_median(base))
        self.assertGreater(_sensitivity_median(adjusted), _sensitivity_median(base))
        self.assertLess(
            _resistant_fraction_median(adjusted),
            _resistant_fraction_median(base),
        )
        self.assertIn("ki67_high_v1", _rule_ids(adjusted))

    def test_intermediate_ki67_has_no_mean_shift_when_other_markers_known(self):
        base = _population_prior()

        adjusted = apply_pathology_biomarker_rules(
            base,
            _known_context(ki67_percent=15, grade=2),
        )

        self.assertEqual(adjusted.transformed_means, base.transformed_means)
        self.assertEqual(adjusted.layer_contributions, ())

    def test_grade3_raises_growth_modestly(self):
        base = _population_prior()

        adjusted = apply_pathology_biomarker_rules(
            base,
            _known_context(ki67_percent=15, grade="grade 3"),
        )

        self.assertAlmostEqual(_growth_median(adjusted), _growth_median(base) * 1.15)
        self.assertIn("grade3_high_growth_v1", _rule_ids(adjusted))

    def test_brca_hrd_unknown_is_audit_only_and_not_treated_as_negative(self):
        base = _population_prior()
        negative = apply_pathology_biomarker_rules(
            base,
            _known_context(ki67_percent=15, grade=2),
        )
        unknown = apply_pathology_biomarker_rules(
            base,
            {
                "ki67_percent": 15,
                "grade": 2,
                "brca_status": "unknown",
                "hrd_status": "not tested",
            },
        )
        positive = apply_pathology_biomarker_rules(
            base,
            {
                "ki67_percent": 15,
                "grade": 2,
                "brca_status": "pathogenic mutation",
                "hrd_status": "negative",
            },
        )

        self.assertEqual(unknown.transformed_means, negative.transformed_means)
        self.assertEqual(
            unknown.transformed_covariance[1][1],
            negative.transformed_covariance[1][1],
        )
        self.assertEqual(
            unknown.transformed_covariance[2][2],
            negative.transformed_covariance[2][2],
        )
        self.assertGreater(_sensitivity_median(positive), _sensitivity_median(negative))
        self.assertLess(
            _resistant_fraction_median(positive),
            _resistant_fraction_median(negative),
        )
        self.assertIn("brca_hrd_missing_v1", _rule_ids(unknown))
        self.assertIn("brca_hrd_positive_v1", _rule_ids(positive))

    def test_every_applied_rule_appears_in_layer_contribution(self):
        adjusted = apply_pathology_biomarker_rules(
            _population_prior(),
            {
                "ki67_percent": 50,
                "grade": 3,
                "er_status": "positive",
                "pr_status": "negative",
                "her2_status": "negative",
                "brca_status": "negative",
                "hrd_status": "positive",
            },
        )

        contribution = adjusted.layer_contribution()
        rule_ids = [rule["rule_id"] for rule in contribution["rules"]]
        self.assertEqual(
            rule_ids,
            [
                "ki67_high_v1",
                "grade3_high_growth_v1",
                "tnbc_receptor_inconsistency_v1",
                "brca_hrd_positive_v1",
            ],
        )
        self.assertIn(
            "tnbc_receptor_inconsistency_v1",
            contribution["uncertainty_drivers"],
        )
        self.assertIn("Receptor markers", contribution["warnings"][0])

    def test_adjusted_covariance_remains_positive_semidefinite(self):
        adjusted = apply_pathology_biomarker_rules(
            _population_prior(),
            {
                "ki67_percent": 52,
                "grade": 3,
                "er_status": "positive",
                "brca_status": "unknown",
                "hrd_status": "pending",
            },
        )

        self.assertEqual(
            validate_covariance(adjusted.transformed_covariance, dimension=3),
            [list(row) for row in adjusted.transformed_covariance],
        )

    def test_sampling_layer3_prior_is_reproducible(self):
        adjusted = apply_pathology_biomarker_rules(
            _population_prior(),
            _known_context(ki67_percent=45, grade=3),
        )

        first = sample_pathology_biomarker_prior(adjusted, n_samples=4, seed=7)
        second = sample_pathology_biomarker_prior(adjusted, n_samples=4, seed=7)

        self.assertEqual(first.samples, second.samples)


def _population_prior():
    contract = resolve_parameter_contract(
        {
            "subtype": "TNBC",
            "regimen_name": "A/C-T neoadjuvant chemotherapy",
        }
    )
    return resolve_population_prior(contract)


def _known_context(*, ki67_percent, grade=2):
    return {
        "ki67_percent": ki67_percent,
        "grade": grade,
        "er_status": "negative",
        "pr_status": "negative",
        "her2_status": "negative",
        "brca_status": "negative",
        "hrd_status": "negative",
    }


def _rule_ids(adjusted):
    return [rule.rule_id for rule in adjusted.layer_contributions]


def _growth_median(prior):
    return math.exp(prior.transformed_means["log_growth_rate_per_day"])


def _sensitivity_median(prior):
    return math.exp(prior.transformed_means["log_active_treatment_sensitivity"])


def _resistant_fraction_median(prior):
    return safe_sigmoid(prior.transformed_means["logit_resistant_fraction"])


if __name__ == "__main__":
    unittest.main()
