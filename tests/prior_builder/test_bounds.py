import unittest

from experiments.prior_builder.bounds import (
    OBSERVATION_NOISE_BY_KIND,
    resolve_observation_noise,
    validate_parameter_bounds,
)


class ParameterBoundsTests(unittest.TestCase):
    def test_normal_values_pass_without_warning(self):
        result = validate_parameter_bounds(
            {
                "growth_rate_per_day": 0.0067,
                "active_treatment_sensitivity": 0.09,
                "resistant_fraction": 0.20,
            }
        )

        self.assertEqual(result.warnings, ())
        self.assertEqual(result.parameters["growth_rate_per_day"], 0.0067)

    def test_warning_values_pass_with_warning(self):
        result = validate_parameter_bounds(
            {
                "growth_rate_per_day": 0.035,
                "active_treatment_sensitivity": 0.32,
                "resistant_fraction": 0.80,
            }
        )

        self.assertEqual(len(result.warnings), 3)
        self.assertIn("growth_rate_per_day", result.warnings[0])
        self.assertIn("active_treatment_sensitivity", result.warnings[1])
        self.assertIn("resistant_fraction", result.warnings[2])

    def test_hard_stop_values_fail_loudly(self):
        for parameter_name, value in (
            ("growth_rate_per_day", 0.101),
            ("active_treatment_sensitivity", 0.501),
            ("resistant_fraction", 0.901),
        ):
            with self.subTest(parameter_name=parameter_name):
                params = {
                    "growth_rate_per_day": 0.0067,
                    "active_treatment_sensitivity": 0.09,
                    "resistant_fraction": 0.20,
                }
                params[parameter_name] = value

                with self.assertRaises(ValueError):
                    validate_parameter_bounds(params)

    def test_negative_growth_or_sensitivity_fails(self):
        for parameter_name in (
            "growth_rate_per_day",
            "active_treatment_sensitivity",
        ):
            with self.subTest(parameter_name=parameter_name):
                params = {
                    "growth_rate_per_day": 0.0067,
                    "active_treatment_sensitivity": 0.09,
                    "resistant_fraction": 0.20,
                }
                params[parameter_name] = -0.001

                with self.assertRaises(ValueError):
                    validate_parameter_bounds(params)


class ObservationNoisePolicyTests(unittest.TestCase):
    def test_low_qc_mri_uses_larger_noise_than_high_qc_mri(self):
        high_qc = resolve_observation_noise(
            {
                "source": "DCE-MRI segmentation volume",
                "segmentation_confidence": 0.90,
            }
        )
        low_qc = resolve_observation_noise(
            {
                "source": "DCE-MRI segmentation volume",
                "qc_flags": ["motion_artifact_possible", "manual_review_recommended"],
            }
        )

        self.assertEqual(high_qc.measurement_kind, "high_qc_mri_volume")
        self.assertEqual(low_qc.measurement_kind, "low_qc_mri_volume")
        self.assertGreater(low_qc.log_scale_noise, high_qc.log_scale_noise)

    def test_diameter_derived_volume_uses_larger_noise_than_mri_volume(self):
        mri_volume = resolve_observation_noise(
            {
                "source": "MRI volume",
                "confidence": "medium",
                "tumor_volume_ml": 18.4,
            }
        )
        diameter_volume = resolve_observation_noise(
            {
                "source": "diameter-derived volume",
                "longest_diameter_cm": 3.2,
                "tumor_volume_ml": 17.2,
            }
        )

        self.assertEqual(mri_volume.measurement_kind, "medium_qc_mri_volume")
        self.assertEqual(diameter_volume.measurement_kind, "diameter_derived_volume")
        self.assertEqual(
            diameter_volume.log_scale_noise,
            OBSERVATION_NOISE_BY_KIND["diameter_derived_volume"],
        )
        self.assertGreater(diameter_volume.log_scale_noise, mri_volume.log_scale_noise)

    def test_manual_volume_uses_manual_noise_policy(self):
        manual = resolve_observation_noise(
            {
                "source": "manual volume entry",
                "tumor_volume_ml": 12.0,
            }
        )

        self.assertEqual(manual.measurement_kind, "manual_volume")
        self.assertEqual(manual.log_scale_noise, 0.25)


if __name__ == "__main__":
    unittest.main()
