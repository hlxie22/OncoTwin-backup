from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from experiments.twin_runtime.scenario_lab import (
    DECISION_SUPPORT_DISCLAIMER,
    run_scenario_lab,
)


def _params(
    particle_id: str,
    *,
    anthracycline_sensitivity: float,
    taxane_sensitivity: float,
    resistant_fraction: float,
) -> dict[str, object]:
    return {
        "particle_id": particle_id,
        "growth_law": "logistic",
        "growth_rate": 0.006,
        "carrying_capacity_ml": 260.0,
        "drug_sensitivity": {
            "anthracycline": anthracycline_sensitivity,
            "taxane": taxane_sensitivity,
        },
        "drug_ec50": {"anthracycline": 0.5, "taxane": 0.5},
        "drug_decay": {"anthracycline": 0.25, "taxane": 0.2},
        "resistant_fraction": resistant_fraction,
        "resistant_sensitivity_scale": 0.08,
        "observation_noise_fraction": 0.10,
    }


def _posterior_update() -> dict[str, object]:
    strong = _params(
        "strong_response",
        anthracycline_sensitivity=0.12,
        taxane_sensitivity=0.11,
        resistant_fraction=0.04,
    )
    weak = _params(
        "weak_response",
        anthracycline_sensitivity=0.02,
        taxane_sensitivity=0.02,
        resistant_fraction=0.25,
    )
    return {
        "posterior_runtime_version": "oncotwin_posterior_update_v1",
        "initial_volume_ml": 30.0,
        "prediction_days": [42.0, 84.0],
        "particle_trajectories": [
            {"particle_id": "strong_response", "parameters": strong, "weight": 0.85},
            {"particle_id": "weak_response", "parameters": weak, "weight": 0.15},
        ],
    }


def _chemo_schedule(relative_dose: float = 1.0) -> dict[str, object]:
    return {
        "schedule_id": "chemo",
        "regimen_name": "Test A/C-T chemotherapy",
        "total_duration_days": 84,
        "events": [
            {"drug": "anthracycline", "day": 0, "relative_dose": relative_dose},
            {"drug": "anthracycline", "day": 14, "relative_dose": relative_dose},
            {"drug": "anthracycline", "day": 28, "relative_dose": relative_dose},
            {"drug": "taxane", "day": 42, "relative_dose": relative_dose},
            {"drug": "taxane", "day": 49, "relative_dose": relative_dose},
        ],
    }


def _no_treatment_schedule() -> dict[str, object]:
    return {
        "schedule_id": "observation_only",
        "regimen_name": "Observation-only modeled comparator",
        "total_duration_days": 84,
        "events": [],
    }


class ScenarioLabRuntimeTest(unittest.TestCase):
    def test_ranks_scenarios_with_posterior_weights_and_bands(self):
        result = run_scenario_lab(
            posterior_update=_posterior_update(),
            scenarios=[
                {
                    "scenario_id": "observation_only",
                    "label": "Observation only",
                    "reference": True,
                    "treatment_schedule": _no_treatment_schedule(),
                },
                {
                    "scenario_id": "chemo",
                    "label": "Continue chemo",
                    "treatment_schedule": _chemo_schedule(),
                },
            ],
            output_days=[84],
            residual_burden_threshold_ml=20.0,
        )

        self.assertTrue(result["not_a_treatment_recommendation"])
        self.assertIn("not treatment recommendations", result["warnings"][-1])
        self.assertEqual(
            result["comparison_summary"]["ranked_scenario_ids_by_low_residual_probability"][0],
            "chemo",
        )

        scenarios = {scenario["scenario_id"]: scenario for scenario in result["scenarios"]}
        self.assertEqual(scenarios["chemo"]["status"], "ok")
        self.assertEqual(scenarios["observation_only"]["status"], "ok")
        self.assertLess(
            scenarios["chemo"]["trajectory_summary"]["median_volume_ml"][-1],
            scenarios["observation_only"]["trajectory_summary"]["median_volume_ml"][-1],
        )
        self.assertIn("lower80_volume_ml", scenarios["chemo"]["trajectory_summary"])
        self.assertIn("comparison_to_reference", scenarios["chemo"])
        self.assertIn(DECISION_SUPPORT_DISCLAIMER, scenarios["chemo"]["explanation"])

    def test_unsafe_dose_fails_closed_without_blocking_other_scenarios(self):
        result = run_scenario_lab(
            posterior_update=_posterior_update(),
            scenarios=[
                {
                    "scenario_id": "safe_chemo",
                    "treatment_schedule": _chemo_schedule(),
                },
                {
                    "scenario_id": "unsafe_dose",
                    "treatment_schedule": _chemo_schedule(relative_dose=2.0),
                },
            ],
            output_days=[84],
        )

        scenarios = {scenario["scenario_id"]: scenario for scenario in result["scenarios"]}
        self.assertEqual(scenarios["safe_chemo"]["status"], "ok")
        self.assertEqual(scenarios["unsafe_dose"]["status"], "failed_safety")
        self.assertTrue(
            any("exceeds safety cap" in warning for warning in scenarios["unsafe_dose"]["warnings"])
        )
        self.assertEqual(result["comparison_summary"]["top_scenario_id"], "safe_chemo")

    def test_empty_scenarios_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one scenario"):
            run_scenario_lab(posterior_update=_posterior_update(), scenarios=[])

    def test_output_is_json_serializable(self):
        result = run_scenario_lab(
            posterior_update=_posterior_update(),
            scenarios=[
                {
                    "scenario_id": "chemo",
                    "treatment_schedule": _chemo_schedule(),
                }
            ],
            output_days=[84],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scenario_lab.json"
            path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
            loaded = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(loaded["scenario_lab_version"], "oncotwin_scenario_lab_v1")
        self.assertEqual(loaded["n_scenarios"], 1)


if __name__ == "__main__":
    unittest.main()
