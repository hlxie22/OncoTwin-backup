from __future__ import annotations

import math
import unittest

from experiments.mechanistic_simulator.validation import SimulatorInputError
from experiments.mechanistic_simulator.volume_ode import simulate_volume_trajectory
from tests.mechanistic_simulator.helpers import (
    no_drug_params,
    no_treatment_schedule,
    simple_drug_params,
    simple_drug_schedule,
)


class VolumeOdeTests(unittest.TestCase):
    def test_no_treatment_volume_grows_toward_carrying_capacity(self) -> None:
        result = simulate_volume_trajectory(
            10.0,
            no_treatment_schedule(),
            no_drug_params(),
            [0, 30, 60],
            dt_days=0.5,
        )
        volumes = [row["tumor_volume_ml"] for row in result["trajectory"]]
        self.assertGreater(volumes[-1], volumes[0])
        self.assertLess(volumes[-1], 250.0)

    def test_zero_growth_and_zero_treatment_keeps_volume_constant(self) -> None:
        result = simulate_volume_trajectory(
            10.0,
            no_treatment_schedule(),
            no_drug_params(growth_rate=0.0),
            [0, 30, 60],
            dt_days=1.0,
        )
        volumes = [row["tumor_volume_ml"] for row in result["trajectory"]]
        self.assertTrue(all(abs(volume - 10.0) < 1e-9 for volume in volumes))

    def test_high_treatment_sensitivity_causes_shrinkage(self) -> None:
        params = simple_drug_params(drug_sensitivity={"demo_drug": 0.2})
        result = simulate_volume_trajectory(
            20.0,
            simple_drug_schedule(),
            params,
            [0, 14, 28],
            dt_days=0.25,
        )
        volumes = [row["tumor_volume_ml"] for row in result["trajectory"]]
        self.assertLess(volumes[-1], volumes[0])

    def test_volume_never_becomes_negative(self) -> None:
        params = simple_drug_params(drug_sensitivity={"demo_drug": 1.0})
        result = simulate_volume_trajectory(
            20.0,
            simple_drug_schedule(),
            params,
            [0, 14, 28],
            dt_days=1.0,
        )
        for row in result["trajectory"]:
            self.assertGreaterEqual(row["tumor_volume_ml"], 0.0)
            self.assertTrue(math.isfinite(row["tumor_volume_ml"]))

    def test_smaller_dt_gives_similar_result(self) -> None:
        coarse = simulate_volume_trajectory(
            20.0,
            simple_drug_schedule(),
            simple_drug_params(),
            [0, 14, 28],
            dt_days=1.0,
        )
        fine = simulate_volume_trajectory(
            20.0,
            simple_drug_schedule(),
            simple_drug_params(),
            [0, 14, 28],
            dt_days=0.25,
        )
        coarse_final = coarse["trajectory"][-1]["tumor_volume_ml"]
        fine_final = fine["trajectory"][-1]["tumor_volume_ml"]
        self.assertLess(abs(coarse_final - fine_final) / fine_final, 0.08)

    def test_invalid_inputs_fail_loudly(self) -> None:
        with self.assertRaises(SimulatorInputError):
            simulate_volume_trajectory(
                -1.0,
                no_treatment_schedule(),
                no_drug_params(),
                [0, 1],
            )
        with self.assertRaises(SimulatorInputError):
            simulate_volume_trajectory(
                10.0,
                no_treatment_schedule(),
                no_drug_params(carrying_capacity_ml=0.0),
                [0, 1],
            )

    def test_output_days_beyond_schedule_require_explicit_permission(self) -> None:
        with self.assertRaises(SimulatorInputError):
            simulate_volume_trajectory(
                10.0,
                no_treatment_schedule(total_duration_days=10),
                no_drug_params(),
                [0, 20],
            )


if __name__ == "__main__":
    unittest.main()
