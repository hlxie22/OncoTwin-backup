import math
import unittest

from experiments.prior_builder.ai_residual_policy import (
    apply_ai_residual_policy,
    sample_ai_residual_prior,
)
from experiments.prior_builder.mri_feature_rules import apply_mri_feature_rules
from experiments.prior_builder.parameter_contract import resolve_parameter_contract
from experiments.prior_builder.pathology_biomarker_rules import apply_pathology_biomarker_rules
from experiments.prior_builder.population_prior import resolve_population_prior
from experiments.prior_builder.transforms import validate_covariance


class AIResidualPolicyTests(unittest.TestCase):
    def test_missing_signal_is_exact_noop(self):
        base = _layer4_prior()

        adjusted = apply_ai_residual_policy(base, {})

        self.assertEqual(adjusted.policy_mode, "inactive_noop")
        self.assertFalse(adjusted.active)
        self.assertIsNone(adjusted.residual_model_version)
        self.assertEqual(adjusted.transformed_means, base.transformed_means)
        self.assertEqual(adjusted.transformed_covariance, base.transformed_covariance)
        self.assertEqual(adjusted.layer_contributions, ())

    def test_validated_signal_shifts_means_and_can_widen_uncertainty(self):
        base = _layer4_prior()

        adjusted = apply_ai_residual_policy(
            base,
            {
                "ai_residual": {
                    "validated": True,
                    "model_version": "unit_residual_v1",
                    "log_growth_rate_shift": math.log(1.05),
                    "log_active_treatment_sensitivity_shift": -math.log(1.04),
                    "resistant_variance_multiplier": 1.20,
                }
            },
        )

        self.assertEqual(adjusted.policy_mode, "validated_residual")
        self.assertTrue(adjusted.active)
        self.assertEqual(adjusted.residual_model_version, "unit_residual_v1")
        self.assertGreater(
            adjusted.transformed_means["log_growth_rate_per_day"],
            base.transformed_means["log_growth_rate_per_day"],
        )
        self.assertLess(
            adjusted.transformed_means["log_active_treatment_sensitivity"],
            base.transformed_means["log_active_treatment_sensitivity"],
        )
        self.assertGreater(adjusted.transformed_covariance[2][2], base.transformed_covariance[2][2])
        self.assertEqual(validate_covariance(adjusted.transformed_covariance, dimension=3), [list(row) for row in adjusted.transformed_covariance])
        self.assertEqual(adjusted.layer_contributions[0].rule_id, "validated_ai_residual_v1")

    def test_unvalidated_signal_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "explicitly validated"):
            apply_ai_residual_policy(
                _layer4_prior(),
                {"ai_residual": {"validated": False, "model_version": "unit", "log_growth_rate_shift": 0.01}},
            )

    def test_model_version_is_required(self):
        with self.assertRaisesRegex(ValueError, "requires model_version"):
            apply_ai_residual_policy(
                _layer4_prior(),
                {"ai_residual": {"validated": True, "log_growth_rate_shift": 0.01}},
            )

    def test_out_of_bounds_or_covariance_narrowing_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exceeds Layer 5 bound"):
            apply_ai_residual_policy(
                _layer4_prior(),
                {"ai_residual": {"validated": True, "model_version": "unit", "log_growth_rate_shift": math.log(1.4)}},
            )
        with self.assertRaisesRegex(ValueError, "may not narrow covariance"):
            apply_ai_residual_policy(
                _layer4_prior(),
                {"ai_residual": {"validated": True, "model_version": "unit", "sensitivity_variance_multiplier": 0.90}},
            )

    def test_unknown_field_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported Layer 5"):
            apply_ai_residual_policy(
                _layer4_prior(),
                {"ai_residual": {"validated": True, "model_version": "unit", "growth_shift_typo": 0.01}},
            )

    def test_sampling_is_reproducible(self):
        adjusted = apply_ai_residual_policy(
            _layer4_prior(),
            {"ai_residual": {"validated": True, "model_version": "unit", "logit_resistant_fraction_shift": 0.02}},
        )

        first = sample_ai_residual_prior(adjusted, n_samples=4, seed=13)
        second = sample_ai_residual_prior(adjusted, n_samples=4, seed=13)

        self.assertEqual(first.samples, second.samples)


def _layer4_prior():
    contract = resolve_parameter_contract({"subtype": "TNBC", "regimen_name": "A/C-T neoadjuvant chemotherapy"})
    population_prior = resolve_population_prior(contract)
    layer3 = apply_pathology_biomarker_rules(
        population_prior,
        {
            "ki67_percent": 20,
            "grade": 2,
            "er_status": "negative",
            "pr_status": "negative",
            "her2_status": "negative",
            "brca_status": "negative",
            "hrd_status": "negative",
        },
    )
    return apply_mri_feature_rules(
        layer3,
        {"volume_ml": 24.0, "functional_tumor_volume_ml": 18.0, "segmentation_qc": "high"},
    )


if __name__ == "__main__":
    unittest.main()
