from __future__ import annotations

import unittest

from experiments.twin_runtime.explanations import (
    SAFETY_AND_SCOPE_NOTE,
    build_twin_update_explanation,
    render_markdown_explanation,
)


def _posterior_update() -> dict[str, object]:
    return {
        "posterior_runtime_version": "oncotwin_posterior_update_v1",
        "n_observations": 2,
        "effective_sample_size_fraction": 0.42,
        "fallback_status": "not_needed",
        "update_explanation": (
            "The posterior shifted toward particles close to the observed tumor-volume trend."
        ),
        "posterior_trajectory_summary": {
            "times": [0.0, 42.0, 84.0],
            "median_volume_ml": [30.0, 18.0, 9.0],
            "lower80_volume_ml": [28.0, 12.0, 4.0],
            "upper80_volume_ml": [32.0, 25.0, 18.0],
            "probability_low_residual_burden": 0.35,
        },
        "uncertainty_summary": {
            "top_drivers": [
                "unknown or medium segmentation QC",
                "posterior update used available tumor-volume observations",
            ],
        },
        "warnings": ["measurement uncertainty remains material"],
    }


def _scenario_lab() -> dict[str, object]:
    return {
        "scenario_lab_version": "oncotwin_scenario_lab_v1",
        "decision_support_disclaimer": "not recommendations",
        "scenarios": [
            {
                "scenario_id": "current_plan",
                "label": "Current plan",
                "status": "ok",
                "trajectory_summary": {
                    "times": [84.0],
                    "median_volume_ml": [12.0],
                    "lower80_volume_ml": [6.0],
                    "upper80_volume_ml": [22.0],
                },
                "probabilities": {
                    "probability_low_residual_burden": 0.30,
                    "probability_progression": 0.10,
                    "probability_insufficient_response": 0.25,
                },
            },
            {
                "scenario_id": "alternate_plan",
                "label": "Alternate modeled plan",
                "status": "ok",
                "trajectory_summary": {
                    "times": [84.0],
                    "median_volume_ml": [8.0],
                    "lower80_volume_ml": [3.0],
                    "upper80_volume_ml": [18.0],
                },
                "probabilities": {
                    "probability_low_residual_burden": 0.55,
                    "probability_progression": 0.05,
                    "probability_insufficient_response": 0.18,
                },
            },
            {
                "scenario_id": "unsafe_plan",
                "label": "Unsafe plan",
                "status": "failed_safety",
                "warnings": ["relative_dose exceeds safety cap"],
            },
        ],
        "comparison_summary": {
            "reference_scenario_id": "current_plan",
            "top_scenario_id": "alternate_plan",
            "ranked_scenario_ids_by_low_residual_probability": [
                "alternate_plan",
                "current_plan",
            ],
        },
        "warnings": ["1 scenario(s) failed and were not used for ranking"],
    }


class ExplanationRuntimeTest(unittest.TestCase):
    def test_builds_clinician_structured_explanation(self):
        explanation = build_twin_update_explanation(
            posterior_update=_posterior_update(),
            scenario_lab=_scenario_lab(),
            prior_context={
                "layer_contributions": [
                    {"layer": "population_prior"},
                    {"rule_id": "pathology_tnbc_v1"},
                ]
            },
            audience="clinician",
        )

        self.assertEqual(explanation["explanation_runtime_version"], "oncotwin_explanation_v1")
        self.assertEqual(explanation["audience"], "clinician")
        self.assertTrue(explanation["not_a_treatment_recommendation"])
        self.assertIn("not a treatment recommendation", explanation["summary"])
        self.assertIn(SAFETY_AND_SCOPE_NOTE, explanation["safety_and_scope_note"])
        self.assertEqual(
            explanation["scenario_comparison_explanation"]["top_scenario"]["scenario_id"],
            "alternate_plan",
        )
        self.assertTrue(
            any(factor["factor_id"] == "effective_sample_size" for factor in explanation["key_factors"])
        )
        self.assertTrue(
            any("failed" in driver["description"] for driver in explanation["uncertainty_drivers"])
        )

    def test_patient_explanation_uses_patient_audience_and_guardrails(self):
        explanation = build_twin_update_explanation(
            posterior_update=_posterior_update(),
            scenario_lab=_scenario_lab(),
            audience="patient",
        )

        self.assertEqual(explanation["audience"], "patient")
        self.assertIn("available tumor measurements", explanation["summary"])
        self.assertIn("Modeled what-if comparisons", [section["title"] for section in explanation["sections"]])
        self.assertTrue(explanation["not_a_treatment_recommendation"])

    def test_markdown_renderer_includes_sections(self):
        explanation = build_twin_update_explanation(
            posterior_update=_posterior_update(),
            scenario_lab=_scenario_lab(),
        )

        markdown = render_markdown_explanation(explanation)

        self.assertIn("# OncoTwin explanation", markdown)
        self.assertIn("## Key factors", markdown)
        self.assertIn("## Scenario comparison", markdown)
        self.assertIn("not a treatment recommendation", markdown)

    def test_rejects_missing_posterior_trajectory(self):
        with self.assertRaisesRegex(ValueError, "posterior_trajectory_summary"):
            build_twin_update_explanation(posterior_update={"n_observations": 1})

    def test_rejects_unknown_audience(self):
        with self.assertRaisesRegex(ValueError, "audience"):
            build_twin_update_explanation(
                posterior_update=_posterior_update(),
                audience="surgeon",
            )


if __name__ == "__main__":
    unittest.main()
