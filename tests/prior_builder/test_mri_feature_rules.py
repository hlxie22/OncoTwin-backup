import unittest

from experiments.prior_builder.bounds import OBSERVATION_NOISE_BY_KIND
from experiments.prior_builder.mri_feature_rules import (
    apply_mri_feature_rules,
    sample_mri_feature_prior,
)
from experiments.prior_builder.parameter_contract import resolve_parameter_contract
from experiments.prior_builder.pathology_biomarker_rules import apply_pathology_biomarker_rules
from experiments.prior_builder.population_prior import resolve_population_prior
from experiments.prior_builder.transforms import validate_covariance


class MRIFeatureRuleTests(unittest.TestCase):
    def test_low_segmentation_qc_inflates_observation_noise(self):
        base = _layer3_prior()
        high_qc = apply_mri_feature_rules(
            base,
            {
                "volume_ml": 24.0,
                "functional_tumor_volume_ml": 18.0,
                "segmentation_qc": 0.92,
            },
        )
        low_qc = apply_mri_feature_rules(
            base,
            {
                "volume_ml": 24.0,
                "functional_tumor_volume_ml": 18.0,
                "segmentation_qc": 0.35,
            },
        )

        self.assertEqual(high_qc.observation_noise_kind, "high_qc_mri_volume")
        self.assertEqual(low_qc.observation_noise_kind, "low_qc_mri_volume")
        self.assertGreater(
            low_qc.log_scale_observation_noise,
            high_qc.log_scale_observation_noise,
        )
        self.assertIn("low_segmentation_qc_v1", _rule_ids(low_qc))
        self.assertIn("low_segmentation_qc_v1", low_qc.uncertainty_drivers)

    def test_high_heterogeneity_widens_resistant_fraction_uncertainty(self):
        base = _layer3_prior()

        adjusted = apply_mri_feature_rules(
            base,
            {
                "volume_ml": 22.0,
                "functional_tumor_volume_ml": 17.0,
                "segmentation_qc": "high",
                "enhancement_std": 0.42,
            },
        )

        self.assertEqual(adjusted.transformed_means, base.transformed_means)
        self.assertGreater(
            adjusted.transformed_covariance[2][2],
            base.transformed_covariance[2][2],
        )
        self.assertIn("high_enhancement_heterogeneity_v1", _rule_ids(adjusted))

    def test_low_enhancement_fraction_widens_response_and_resistance_uncertainty(self):
        base = _layer3_prior()

        adjusted = apply_mri_feature_rules(
            base,
            {
                "volume_ml": 31.0,
                "functional_tumor_volume_ml": 18.0,
                "segmentation_qc": "good",
                "low_enhancement_fraction": 0.48,
            },
        )

        self.assertGreater(
            adjusted.transformed_covariance[1][1],
            base.transformed_covariance[1][1],
        )
        self.assertGreater(
            adjusted.transformed_covariance[2][2],
            base.transformed_covariance[2][2],
        )
        self.assertIn("low_enhancement_fraction_v1", _rule_ids(adjusted))

    def test_missing_mri_features_do_not_fail_prior_builder(self):
        base = _layer3_prior()

        adjusted = apply_mri_feature_rules(base, {})

        self.assertEqual(adjusted.update_mode, "volume_update")
        self.assertEqual(adjusted.transformed_means, base.transformed_means)
        self.assertEqual(adjusted.transformed_covariance, base.transformed_covariance)
        self.assertEqual(adjusted.layer_contributions, ())
        self.assertEqual(adjusted.observation_noise_kind, "manual_volume")

    def test_ftv_anatomic_volume_inconsistency_creates_qc_warning(self):
        adjusted = apply_mri_feature_rules(
            _layer3_prior(),
            {
                "volume_ml": 20.0,
                "functional_tumor_volume_ml": 28.0,
                "segmentation_qc": 0.86,
            },
        )

        self.assertIn("ftv_anatomic_volume_inconsistency_v1", _rule_ids(adjusted))
        self.assertTrue(
            any("Functional tumor volume" in warning for warning in adjusted.warnings)
        )
        self.assertIn(
            "ftv_anatomic_volume_inconsistency_v1",
            adjusted.uncertainty_drivers,
        )

    def test_diameter_only_case_falls_back_to_report_only_mode(self):
        adjusted = apply_mri_feature_rules(
            _layer3_prior(),
            {
                "longest_diameter_cm": 3.1,
                "segmentation_qc": "high",
            },
        )

        self.assertTrue(adjusted.report_only)
        self.assertEqual(adjusted.update_mode, "report_only")
        self.assertEqual(adjusted.observation_noise_kind, "diameter_derived_volume")
        self.assertEqual(
            adjusted.log_scale_observation_noise,
            OBSERVATION_NOISE_BY_KIND["diameter_derived_volume"],
        )
        self.assertIn("diameter_only_report_only_v1", _rule_ids(adjusted))

    def test_layer4_contribution_is_traceable_and_covariance_stays_valid(self):
        adjusted = apply_mri_feature_rules(
            _layer3_prior(),
            {
                "volume_ml": 20.0,
                "functional_tumor_volume_ml": 28.0,
                "segmentation_qc": 0.25,
                "registration_qc": "failed",
                "enhancement_std": 0.40,
                "low_enhancement_fraction": 0.45,
            },
        )

        contribution = adjusted.layer_contribution()
        self.assertEqual(contribution["layer"], "mri_feature_rules")
        self.assertEqual(contribution["observation_noise_kind"], "low_qc_mri_volume")
        self.assertEqual(contribution["update_mode"], "volume_update")
        self.assertIn("warnings", contribution)
        self.assertEqual(
            validate_covariance(adjusted.transformed_covariance, dimension=3),
            [list(row) for row in adjusted.transformed_covariance],
        )
        self.assertEqual(
            [rule["rule_id"] for rule in contribution["rules"]],
            [
                "low_segmentation_qc_v1",
                "low_registration_qc_v1",
                "high_enhancement_heterogeneity_v1",
                "low_enhancement_fraction_v1",
                "ftv_anatomic_volume_inconsistency_v1",
            ],
        )

    def test_sampling_layer4_prior_is_reproducible(self):
        adjusted = apply_mri_feature_rules(
            _layer3_prior(),
            {
                "volume_ml": 27.0,
                "functional_tumor_volume_ml": 12.0,
                "segmentation_qc": "high",
                "enhancement_std": 0.38,
            },
        )

        first = sample_mri_feature_prior(adjusted, n_samples=4, seed=11)
        second = sample_mri_feature_prior(adjusted, n_samples=4, seed=11)

        self.assertEqual(first.samples, second.samples)


def _layer3_prior():
    contract = resolve_parameter_contract(
        {
            "subtype": "TNBC",
            "regimen_name": "A/C-T neoadjuvant chemotherapy",
        }
    )
    population_prior = resolve_population_prior(contract)
    return apply_pathology_biomarker_rules(
        population_prior,
        {
            "ki67_percent": 15,
            "grade": 2,
            "er_status": "negative",
            "pr_status": "negative",
            "her2_status": "negative",
            "brca_status": "negative",
            "hrd_status": "negative",
        },
    )


def _rule_ids(adjusted):
    return [rule.rule_id for rule in adjusted.layer_contributions]


if __name__ == "__main__":
    unittest.main()
