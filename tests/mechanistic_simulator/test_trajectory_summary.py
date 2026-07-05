from __future__ import annotations

import unittest

from experiments.mechanistic_simulator.summarize import summarize_trajectories


def particle(particle_id: str, volumes: list[float], warning: str | None = None) -> dict:
    return {
        "particle_id": particle_id,
        "parameters": {
            "growth_rate": 0.005 + 0.001 * int(particle_id[-1]),
            "carrying_capacity_ml": 250.0,
            "resistant_fraction": 0.1 * int(particle_id[-1]),
            "resistant_sensitivity_scale": 0.1,
            "observation_noise_fraction": 0.1,
            "drug_sensitivity": {"demo": 0.05 + 0.01 * int(particle_id[-1])},
        },
        "times": [0, 7],
        "predicted_volume_ml": volumes,
        "predicted_longest_diameter_cm": [2.0, 2.0],
        "warnings": [warning] if warning else [],
    }


class TrajectorySummaryTests(unittest.TestCase):
    def test_median_and_intervals_are_ordered(self) -> None:
        summary = summarize_trajectories(
            [
                particle("p1", [10.0, 8.0]),
                particle("p2", [12.0, 10.0]),
                particle("p3", [14.0, 12.0]),
            ]
        )
        self.assertEqual(summary["median_volume_ml"], [12.0, 10.0])
        for median, interval_80, interval_95 in zip(
            summary["median_volume_ml"],
            summary["interval_80_volume_ml"],
            summary["interval_95_volume_ml"],
        ):
            self.assertLessEqual(interval_80[0], median)
            self.assertGreaterEqual(interval_80[1], median)
            self.assertLessEqual(interval_95[0], interval_80[0])
            self.assertGreaterEqual(interval_95[1], interval_80[1])

    def test_uncertainty_score_increases_for_wider_trajectories(self) -> None:
        narrow = summarize_trajectories(
            [
                particle("p1", [10.0, 9.9]),
                particle("p2", [10.1, 10.0]),
                particle("p3", [10.2, 10.1]),
            ]
        )
        wide = summarize_trajectories(
            [
                particle("p1", [5.0, 1.0]),
                particle("p2", [10.0, 10.0]),
                particle("p3", [20.0, 30.0]),
            ]
        )
        self.assertGreater(wide["uncertainty_score"], narrow["uncertainty_score"])

    def test_summary_preserves_warning_messages(self) -> None:
        summary = summarize_trajectories([particle("p1", [10.0, 8.0], "demo warning")])
        self.assertIn("demo warning", summary["warnings"])
        self.assertIn("Exploratory simulation; parameters are not clinically validated.", summary["warnings"])


if __name__ == "__main__":
    unittest.main()
