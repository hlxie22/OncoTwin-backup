import math
import unittest

from experiments.prior_builder.layer6_late_response import (
    LAYER6_FEATURE_NAMES,
    build_layer6_features,
    extract_mammaprint_feature,
    late_response_target_per_day,
    layer6_activation,
    layer6_feature_vector,
    observed_early_log_rate_per_day,
    predict_volume_from_late_rate,
    summarize_late_rate_posterior,
    validate_feature_vector,
)


def _case(**overrides):
    case = {
        "case_id": "case_001",
        "baseline_volume_ml": 1.0,
        "early_volume_ml": 0.5,
        "early_day": 42.0,
        "final_day": 126.0,
        "context": {
            "biomarkers": {
                "mammaprint_status": 1,
            }
        },
    }
    case.update(overrides)
    return case


def _static_prediction():
    return {
        "point_ml": 0.4,
        "lower_80_ml": 0.2,
        "upper_80_ml": 0.8,
        "lower_95_ml": 0.1,
        "upper_95_ml": 1.2,
    }


class Layer6LateResponseTest(unittest.TestCase):
    def test_activation_requires_early_mri(self):
        active = layer6_activation(_case())
        self.assertTrue(active["active"])
        self.assertEqual(active["status"], "active")

        missing = layer6_activation(
            _case(early_volume_ml=None)
        )
        self.assertFalse(missing["active"])
        self.assertEqual(
            missing["status"],
            "inactive_missing_early_volume",
        )

    def test_activation_requires_late_horizon(self):
        result = layer6_activation(
            _case(final_day=42.0)
        )

        self.assertFalse(result["active"])
        self.assertEqual(
            result["status"],
            "inactive_no_late_horizon",
        )

    def test_observed_early_rate_matches_log_ratio(self):
        rate = observed_early_log_rate_per_day(
            _case()
        )

        self.assertAlmostEqual(
            rate,
            math.log(0.5) / 42.0,
            places=12,
        )

    def test_feature_contract_is_frozen(self):
        features = build_layer6_features(
            _case(),
            _static_prediction(),
        )
        vector = layer6_feature_vector(
            _case(),
            _static_prediction(),
        )

        self.assertEqual(
            tuple(features),
            LAYER6_FEATURE_NAMES,
        )
        self.assertEqual(
            vector,
            tuple(
                features[name]
                for name in LAYER6_FEATURE_NAMES
            ),
        )
        self.assertAlmostEqual(
            features[
                "observed_early_log_rate_42d"
            ],
            math.log(0.5),
            places=12,
        )
        self.assertEqual(
            features["mammaprint_value"],
            1.0,
        )
        self.assertEqual(
            features["mammaprint_missing"],
            0.0,
        )

    def test_decoder_passes_through_early_volume(self):
        predicted = predict_volume_from_late_rate(
            _case(),
            late_log_rate_per_day=-0.01,
            prediction_day=42.0,
        )

        self.assertAlmostEqual(
            predicted,
            0.5,
            places=12,
        )

    def test_zero_late_rate_produces_plateau(self):
        predicted = predict_volume_from_late_rate(
            _case(),
            late_log_rate_per_day=0.0,
        )

        self.assertAlmostEqual(
            predicted,
            0.5,
            places=12,
        )

    def test_late_target_reconstructs_final_volume(self):
        case = _case()
        final_volume = 0.25

        target = late_response_target_per_day(
            case,
            final_volume,
        )
        reconstructed = (
            predict_volume_from_late_rate(
                case,
                target,
            )
        )

        self.assertAlmostEqual(
            reconstructed,
            final_volume,
            places=12,
        )

    def test_posterior_intervals_are_nested(self):
        summary = summarize_late_rate_posterior(
            _case(),
            late_rate_mean_per_day=-0.002,
            late_rate_sd_per_day=0.003,
        )

        self.assertLess(
            summary["lower_95_ml"],
            summary["lower_80_ml"],
        )
        self.assertLess(
            summary["lower_80_ml"],
            summary["point_ml"],
        )
        self.assertLess(
            summary["point_ml"],
            summary["upper_80_ml"],
        )
        self.assertLess(
            summary["upper_80_ml"],
            summary["upper_95_ml"],
        )

    def test_negative_posterior_sd_is_rejected(self):
        with self.assertRaises(ValueError):
            summarize_late_rate_posterior(
                _case(),
                late_rate_mean_per_day=0.0,
                late_rate_sd_per_day=-0.1,
            )

    def test_nested_mammaprint_extraction(self):
        value, missing = extract_mammaprint_feature(
            {
                "pathology": {
                    "assays": [
                        {
                            "name": "MammaPrint",
                            "mammaprint_result": "high risk",
                        }
                    ]
                }
            }
        )

        self.assertEqual(value, 1.0)
        self.assertEqual(missing, 0.0)

    def test_feature_vector_validation(self):
        values = [0.0] * len(
            LAYER6_FEATURE_NAMES
        )

        self.assertEqual(
            validate_feature_vector(values),
            tuple(values),
        )

        with self.assertRaises(ValueError):
            validate_feature_vector(
                values[:-1]
            )


if __name__ == "__main__":
    unittest.main()
