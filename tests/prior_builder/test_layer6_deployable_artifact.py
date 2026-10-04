from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.prior_builder.layer6_deployable_artifact import (
    load_fitted_layer6_member,
    save_fitted_layer6_member,
    sha256_file,
)
from experiments.prior_builder.layer6_neural_inference import (
    Layer6NeuralConfig,
    fit_layer6_neural_model,
    predict_layer6_distribution,
)


class Layer6DeployableArtifactTest(unittest.TestCase):
    def _fitted(self):
        rng = np.random.default_rng(31)
        features = rng.normal(size=(64, 4))
        targets = (
            0.35 * features[:, 0]
            - 0.20 * features[:, 1]
            + 0.10 * features[:, 2]
            + rng.normal(scale=0.05, size=64)
        )
        fitted = fit_layer6_neural_model(
            features,
            targets,
            config=Layer6NeuralConfig(
                hidden_dim=8,
                dropout=0.0,
                weight_decay=0.01,
                learning_rate=0.01,
                max_epochs=12,
                patience=5,
                min_epochs=5,
            ),
            seed=41,
            fixed_epochs=12,
        )
        return fitted, features

    def test_state_dict_round_trip_preserves_predictions(self):
        fitted, features = self._fitted()
        feature_names = ("a", "b", "c", "d")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "member.pt"
            metadata = save_fitted_layer6_member(
                fitted,
                path,
                member_index=0,
                feature_names=feature_names,
            )
            loaded = load_fitted_layer6_member(
                path,
                expected_sha256=str(metadata["sha256"]),
                expected_feature_names=feature_names,
            )

            before_mean, before_sd = predict_layer6_distribution(
                fitted,
                features[:10],
            )
            after_mean, after_sd = predict_layer6_distribution(
                loaded,
                features[:10],
            )

        np.testing.assert_allclose(before_mean, after_mean, atol=1e-8)
        np.testing.assert_allclose(before_sd, after_sd, atol=1e-8)

    def test_feature_order_is_enforced(self):
        fitted, _ = self._fitted()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "member.pt"
            save_fitted_layer6_member(
                fitted,
                path,
                member_index=0,
                feature_names=("a", "b", "c", "d"),
            )

            with self.assertRaisesRegex(ValueError, "feature order mismatch"):
                load_fitted_layer6_member(
                    path,
                    expected_feature_names=("b", "a", "c", "d"),
                )

    def test_hash_mismatch_fails_before_loading(self):
        fitted, _ = self._fitted()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "member.pt"
            save_fitted_layer6_member(
                fitted,
                path,
                member_index=0,
                feature_names=("a", "b", "c", "d"),
            )
            self.assertEqual(len(sha256_file(path)), 64)

            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_fitted_layer6_member(
                    path,
                    expected_sha256="0" * 64,
                )


if __name__ == "__main__":
    unittest.main()
