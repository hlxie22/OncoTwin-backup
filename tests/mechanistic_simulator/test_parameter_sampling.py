from __future__ import annotations

import unittest

from experiments.mechanistic_simulator.params import sample_volume_params
from tests.mechanistic_simulator.helpers import load_json


class ParameterSamplingTests(unittest.TestCase):
    def test_sampler_returns_requested_number_of_particles(self) -> None:
        particles = sample_volume_params(load_json("fixtures/mechanistic_simulator/params/generic_volume_prior.json"), 7, seed=1)
        self.assertEqual(len(particles), 7)

    def test_same_seed_gives_same_particles(self) -> None:
        prior = load_json("fixtures/mechanistic_simulator/params/generic_volume_prior.json")
        self.assertEqual(
            sample_volume_params(prior, 5, seed=42),
            sample_volume_params(prior, 5, seed=42),
        )

    def test_different_seed_gives_different_particles(self) -> None:
        prior = load_json("fixtures/mechanistic_simulator/params/generic_volume_prior.json")
        self.assertNotEqual(
            sample_volume_params(prior, 5, seed=42),
            sample_volume_params(prior, 5, seed=43),
        )

    def test_all_sampled_values_are_inside_bounds(self) -> None:
        prior = load_json("fixtures/mechanistic_simulator/params/generic_volume_prior.json")
        particles = sample_volume_params(prior, 50, seed=99)
        for particle in particles:
            self.assertGreaterEqual(particle["growth_rate"], 0.001)
            self.assertLessEqual(particle["growth_rate"], 0.014)
            self.assertGreaterEqual(particle["resistant_fraction"], 0.02)
            self.assertLessEqual(particle["resistant_fraction"], 0.35)
            self.assertIn("anthracycline", particle["drug_sensitivity"])
            self.assertIn("particle_id", particle)


if __name__ == "__main__":
    unittest.main()
