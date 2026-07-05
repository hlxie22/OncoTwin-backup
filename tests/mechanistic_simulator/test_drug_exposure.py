from __future__ import annotations

import copy
import unittest

from experiments.mechanistic_simulator.exposure import compute_exposure
from experiments.mechanistic_simulator.validation import SimulatorInputError
from tests.mechanistic_simulator.helpers import simple_drug_params, simple_drug_schedule


class DrugExposureTests(unittest.TestCase):
    def test_exposure_is_zero_before_first_dose(self) -> None:
        schedule = simple_drug_schedule()
        schedule["events"][0]["day"] = 10
        schedule["events"][1]["day"] = 20
        exposure = compute_exposure(schedule, simple_drug_params(), [0, 9.9])
        self.assertEqual(exposure["demo_drug"], [0.0, 0.0])

    def test_single_dose_decays_over_time(self) -> None:
        schedule = simple_drug_schedule()
        schedule["events"] = [schedule["events"][0]]
        exposure = compute_exposure(schedule, simple_drug_params(), [0, 1, 7])
        self.assertAlmostEqual(exposure["demo_drug"][0], 1.0)
        self.assertGreater(exposure["demo_drug"][0], exposure["demo_drug"][1])
        self.assertGreater(exposure["demo_drug"][1], exposure["demo_drug"][2])

    def test_multiple_doses_accumulate(self) -> None:
        schedule = simple_drug_schedule()
        exposure = compute_exposure(schedule, simple_drug_params(), [7])
        self.assertGreater(exposure["demo_drug"][0], 1.0)

    def test_relative_dose_scales_exposure(self) -> None:
        schedule = simple_drug_schedule()
        schedule["events"] = [{"drug": "demo_drug", "day": 0, "relative_dose": 0.5}]
        exposure = compute_exposure(schedule, simple_drug_params(), [0])
        self.assertAlmostEqual(exposure["demo_drug"][0], 0.5)

    def test_zero_relative_dose_adds_no_exposure(self) -> None:
        schedule = simple_drug_schedule()
        schedule["events"] = [{"drug": "demo_drug", "day": 0, "relative_dose": 0.0}]
        exposure = compute_exposure(schedule, simple_drug_params(), [0])
        self.assertEqual(exposure["demo_drug"][0], 0.0)

    def test_unknown_drug_fails_validation(self) -> None:
        schedule = simple_drug_schedule()
        schedule["events"][0]["drug"] = "unknown_drug"
        with self.assertRaises(SimulatorInputError):
            compute_exposure(schedule, simple_drug_params(), [0])

    def test_negative_dose_fails_validation(self) -> None:
        schedule = copy.deepcopy(simple_drug_schedule())
        schedule["events"][0]["relative_dose"] = -1.0
        with self.assertRaises(SimulatorInputError):
            compute_exposure(schedule, simple_drug_params(), [0])


if __name__ == "__main__":
    unittest.main()
