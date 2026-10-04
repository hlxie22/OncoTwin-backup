from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


SPEC = Path(
    "configs/prior/"
    "layer7_locked_specification_v1.json"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


class Layer7LockedSpecificationTest(
    unittest.TestCase
):
    def test_locked_runtime_values(self):
        payload = json.loads(
            SPEC.read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(
            payload["status"],
            "locked_internal_specification",
        )
        self.assertEqual(
            payload["particle_runtime"][
                "particle_count"
            ],
            256,
        )
        self.assertEqual(
            payload["particle_runtime"][
                "seed"
            ],
            730000,
        )
        self.assertEqual(
            payload["posterior_update"][
                "ess_threshold_fraction"
            ],
            0.1,
        )
        self.assertEqual(
            payload["posterior_update"][
                "low_ess_policy"
            ],
            "exact_mechanistic_prior_fallback",
        )
        self.assertFalse(
            payload[
                "independent_confirmatory_evidence"
            ]
        )
        self.assertFalse(
            payload["locked_evaluation"][
                "selection_performed_in_locked_run"
            ]
        )

    def test_locked_source_hashes(self):
        payload = json.loads(
            SPEC.read_text(
                encoding="utf-8"
            )
        )
        expected = payload[
            "expected_hashes"
        ]

        paths = {
            "cohort": Path(
                "data/processed/v1_prior_stack/"
                "ispy2_v1_prior_eval_cohort.jsonl"
            ),
            "particle_policy": Path(
                "configs/prior/"
                "layer7_baseline_particle_policy_v0_1.json"
            ),
            "evaluation_policy": Path(
                "configs/prior/"
                "layer7_exploratory_eval_policy_v0_1.json"
            ),
            "candidate_policy": Path(
                "configs/prior/"
                "layer7_fail_closed_candidate_v0_2.json"
            ),
            "prelock_robustness": Path(
                "artifacts/prior_builder/layer7/"
                "layer7_prelock_robustness_v0_2.json"
            ),
            "layer6_closure": Path(
                "artifacts/prior_builder/layer6/"
                "layer6_runtime_closure_v1.json"
            ),
        }

        for name, path in paths.items():
            self.assertEqual(
                sha256_file(path),
                expected[name],
                name,
            )


if __name__ == "__main__":
    unittest.main()
