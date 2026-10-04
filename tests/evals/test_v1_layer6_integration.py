from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

from evals.prior_stack.v1_real_data_eval import (
    CALIBRATED_LAYER4_CANDIDATE,
    EARLY_RESPONSE_CANDIDATE,
    LAYER6_CANDIDATE,
    load_real_cohort,
    run_real_data_eval,
)
from experiments.prior_builder.layer6_runtime import (
    load_layer6_ensemble,
    predict_layer6,
)


COHORT = Path(
    "data/processed/v1_prior_stack/"
    "ispy2_v1_prior_eval_cohort.jsonl"
)
STATIC_CALIBRATOR = Path(
    "configs/prior/"
    "v1_d1_constant_log_radius_posthoc_v0_1.json"
)
EARLY_UPDATER = Path(
    "configs/prior/"
    "v1_d1_continuous_early_volume_posthoc_v0_1.json"
)
EARLY_CALIBRATOR = Path(
    "configs/prior/"
    "v1_d1_early_response_constant_log_radius_"
    "posthoc_v0_1.json"
)
LAYER6_ARTIFACT = Path(
    "artifacts/prior_builder/layer6/"
    "v1_late_response_ensemble"
)

REQUIRED = (
    COHORT,
    STATIC_CALIBRATOR,
    EARLY_UPDATER,
    EARLY_CALIBRATOR,
    LAYER6_ARTIFACT / "manifest.json",
)


@unittest.skipUnless(
    all(path.exists() for path in REQUIRED),
    "Layer 6 integration artifacts are unavailable",
)
class Layer6EvaluatorIntegrationTest(
    unittest.TestCase
):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = run_real_data_eval(
            COHORT,
            n_samples=100,
            seed=2026,
            interval_calibrator_path=(
                STATIC_CALIBRATOR
            ),
            allow_candidate_interval_calibrator=True,
            early_response_updater_path=(
                EARLY_UPDATER
            ),
            early_response_interval_calibrator_path=(
                EARLY_CALIBRATOR
            ),
            allow_candidate_early_response_updater=True,
            layer6_artifact_dir=LAYER6_ARTIFACT,
        )

    def test_layer6_is_separate_candidate(self):
        self.assertIn(
            LAYER6_CANDIDATE,
            self.result["metrics"],
        )
        self.assertIn(
            EARLY_RESPONSE_CANDIDATE,
            self.result["metrics"],
        )
        self.assertIn(
            CALIBRATED_LAYER4_CANDIDATE,
            self.result["metrics"],
        )

        runtime = self.result["layer6_runtime"]
        self.assertEqual(
            runtime["candidate_count"],
            270,
        )
        self.assertEqual(
            runtime["status_counts"],
            {
                "active": 252,
                "inactive_fallback": 18,
            },
        )

    def test_inactive_cases_exactly_match_d1(self):
        inactive_count = 0

        for row in self.result["case_predictions"]:
            predictions = row["predictions"]
            if LAYER6_CANDIDATE not in predictions:
                continue

            layer6 = predictions[LAYER6_CANDIDATE]
            if (
                layer6["layer6"]["status"]
                != "inactive_fallback"
            ):
                continue

            inactive_count += 1
            fallback = predictions[
                EARLY_RESPONSE_CANDIDATE
            ]

            for key in (
                "point_ml",
                "lower_80_ml",
                "upper_80_ml",
                "lower_95_ml",
                "upper_95_ml",
            ):
                self.assertEqual(
                    layer6[key],
                    fallback[key],
                )

        self.assertEqual(inactive_count, 18)

    def test_active_cases_use_loaded_ensemble(self):
        active_count = 0

        for row in self.result["case_predictions"]:
            prediction = row["predictions"].get(
                LAYER6_CANDIDATE
            )
            if prediction is None:
                continue

            if (
                prediction["layer6"]["status"]
                == "active"
            ):
                active_count += 1
                self.assertFalse(
                    prediction["layer6"][
                        "fallback_used"
                    ]
                )

        self.assertEqual(active_count, 252)

    def test_final_outcome_cannot_change_inference(self):
        cases = load_real_cohort(COHORT)
        cases_by_id = {
            str(case["case_id"]): case
            for case in cases
        }

        active_row = next(
            row
            for row in self.result[
                "case_predictions"
            ]
            if row["predictions"].get(
                LAYER6_CANDIDATE,
                {},
            ).get("layer6", {}).get("status")
            == "active"
        )

        case = copy.deepcopy(
            cases_by_id[
                str(active_row["case_id"])
            ]
        )
        predictions = active_row["predictions"]
        ensemble = load_layer6_ensemble(
            LAYER6_ARTIFACT
        )

        original = predict_layer6(
            case,
            predictions[
                CALIBRATED_LAYER4_CANDIDATE
            ],
            predictions[
                EARLY_RESPONSE_CANDIDATE
            ],
            ensemble,
        )

        mutated = copy.deepcopy(case)
        mutated["final_volume_ml"] = 1e9

        context = mutated.get("context")
        if isinstance(context, dict):
            for key in tuple(context):
                normalized = str(key).lower()
                if (
                    "final_volume" in normalized
                    or "observed_final" in normalized
                    or "outcome_volume" in normalized
                    or "heldout_volume" in normalized
                ):
                    context[key] = 1e9

        changed = predict_layer6(
            mutated,
            predictions[
                CALIBRATED_LAYER4_CANDIDATE
            ],
            predictions[
                EARLY_RESPONSE_CANDIDATE
            ],
            ensemble,
        )

        self.assertEqual(original, changed)

        serialized = json.dumps(
            original,
            sort_keys=True,
        )
        self.assertNotIn(
            "observed_final_volume_ml",
            serialized,
        )


if __name__ == "__main__":
    unittest.main()
