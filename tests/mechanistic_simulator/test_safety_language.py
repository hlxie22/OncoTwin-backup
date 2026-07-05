from __future__ import annotations

import unittest

from experiments.mechanistic_simulator.ensemble import simulate_volume_ensemble
from experiments.mechanistic_simulator.params import sample_volume_params
from experiments.mechanistic_simulator.safety import assert_no_unsafe_language, find_unsafe_language
from tests.mechanistic_simulator.helpers import load_json


class SafetyLanguageTests(unittest.TestCase):
    def test_forbidden_language_is_rejected(self) -> None:
        payload = {"summary": "This treatment will cure disease and is guaranteed."}
        matches = find_unsafe_language(payload)
        self.assertIn("will cure", matches)
        self.assertIn("guaranteed", matches)
        with self.assertRaises(AssertionError):
            assert_no_unsafe_language(payload)

    def test_allowed_uncertainty_language_passes(self) -> None:
        payload = {
            "summary": "Exploratory simulation with uncertain estimate below a research threshold; discuss with care team."
        }
        assert_no_unsafe_language(payload)

    def test_ensemble_output_contains_no_forbidden_language(self) -> None:
        case = load_json("fixtures/mechanistic_simulator/cases/tnbc_demo_case.json")
        schedule = load_json(case["treatment_schedule"]["path"])
        prior = load_json("fixtures/mechanistic_simulator/params/generic_volume_prior.json")
        result = simulate_volume_ensemble(
            case["baseline_measurement"]["tumor_volume_ml"],
            schedule,
            sample_volume_params(prior, 20, seed=5),
            [0, 42, 84],
        )
        assert_no_unsafe_language(result)


if __name__ == "__main__":
    unittest.main()
