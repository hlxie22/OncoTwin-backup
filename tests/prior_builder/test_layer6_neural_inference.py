import unittest

import numpy as np

from experiments.prior_builder.layer6_neural_inference import (
    Layer6NeuralConfig,
    combine_gaussian_ensemble,
    fit_layer6_neural_model,
    predict_layer6_distribution,
)


class Layer6NeuralInferenceTest(unittest.TestCase):
    def _synthetic_data(self):
        rng = np.random.default_rng(7)

        features = rng.normal(
            size=(120, 4)
        )

        targets = (
            0.50 * features[:, 0]
            - 0.25 * features[:, 1]
            + 0.15 * features[:, 0]
            * features[:, 2]
            + rng.normal(
                scale=0.08,
                size=120,
            )
        )

        return features, targets

    def test_predictions_have_valid_shape_and_sd(self):
        features, targets = (
            self._synthetic_data()
        )

        fitted = fit_layer6_neural_model(
            features[:90],
            targets[:90],
            validation_features=features[90:],
            validation_targets=targets[90:],
            config=Layer6NeuralConfig(
                hidden_dim=8,
                max_epochs=150,
                patience=20,
                min_epochs=25,
            ),
            seed=11,
        )

        mean, sd = predict_layer6_distribution(
            fitted,
            features[90:],
        )

        self.assertEqual(
            mean.shape,
            (30,),
        )
        self.assertEqual(
            sd.shape,
            (30,),
        )
        self.assertTrue(
            np.all(sd > 0)
        )

    def test_training_learns_synthetic_signal(self):
        features, targets = (
            self._synthetic_data()
        )

        fitted = fit_layer6_neural_model(
            features[:90],
            targets[:90],
            validation_features=features[90:],
            validation_targets=targets[90:],
            config=Layer6NeuralConfig(
                hidden_dim=12,
                max_epochs=220,
                patience=30,
                min_epochs=30,
            ),
            seed=13,
        )

        mean, _ = predict_layer6_distribution(
            fitted,
            features[90:],
        )

        mse = float(
            np.mean(
                (
                    mean - targets[90:]
                ) ** 2
            )
        )

        self.assertLess(
            mse,
            0.08,
        )

    def test_same_seed_is_deterministic(self):
        features, targets = (
            self._synthetic_data()
        )

        config = Layer6NeuralConfig(
            hidden_dim=8,
            dropout=0.0,
            max_epochs=80,
            patience=15,
            min_epochs=20,
        )

        first = fit_layer6_neural_model(
            features[:90],
            targets[:90],
            validation_features=features[90:],
            validation_targets=targets[90:],
            config=config,
            seed=19,
        )

        second = fit_layer6_neural_model(
            features[:90],
            targets[:90],
            validation_features=features[90:],
            validation_targets=targets[90:],
            config=config,
            seed=19,
        )

        first_mean, first_sd = (
            predict_layer6_distribution(
                first,
                features[90:],
            )
        )

        second_mean, second_sd = (
            predict_layer6_distribution(
                second,
                features[90:],
            )
        )

        np.testing.assert_allclose(
            first_mean,
            second_mean,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            first_sd,
            second_sd,
            atol=1e-8,
        )

    def test_ensemble_moment_matching(self):
        means = [
            np.asarray([0.0, 1.0]),
            np.asarray([2.0, 3.0]),
        ]

        standard_deviations = [
            np.asarray([1.0, 1.0]),
            np.asarray([1.0, 1.0]),
        ]

        mean, sd = combine_gaussian_ensemble(
            means,
            standard_deviations,
        )

        np.testing.assert_allclose(
            mean,
            np.asarray([1.0, 2.0]),
        )

        np.testing.assert_allclose(
            sd,
            np.sqrt(
                np.asarray([2.0, 2.0])
            ),
        )


if __name__ == "__main__":
    unittest.main()
