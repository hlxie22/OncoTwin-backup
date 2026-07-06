import unittest

from experiments.prior_builder.adapter_to_volume_ode import (
    DEFAULT_GROWTH_LAW,
    adapt_v1_prior_sample_to_volume_ode,
    adapt_v1_prior_samples_to_volume_ode,
)
from experiments.prior_builder.parameter_contract import resolve_parameter_contract


class VolumeOdeAdapterTests(unittest.TestCase):
    def test_adapter_emits_valid_simulator_params(self):
        result = adapt_v1_prior_sample_to_volume_ode(
            _tnbc_chemo_contract(),
            _sampled_parameters(),
            _fixed_parameters(),
            particle_id="particle_001",
        )

        params = result.parameters
        self.assertEqual(result.warnings, ())
        self.assertEqual(params["particle_id"], "particle_001")
        self.assertEqual(params["growth_law"], DEFAULT_GROWTH_LAW)
        self.assertEqual(params["growth_rate"], 0.0067)
        self.assertEqual(params["resistant_fraction"], 0.20)
        self.assertNotIn("growth_rate_per_day", params)
        self.assertNotIn("active_treatment_sensitivity", params)
        self.assertEqual(
            set(params),
            {
                "growth_law",
                "growth_rate",
                "carrying_capacity_ml",
                "drug_sensitivity",
                "drug_ec50",
                "drug_decay",
                "resistant_fraction",
                "resistant_sensitivity_scale",
                "observation_noise_fraction",
                "particle_id",
            },
        )

    def test_shared_sensitivity_maps_to_active_chemo_agents_only(self):
        result = adapt_v1_prior_sample_to_volume_ode(
            _tnbc_chemo_contract(),
            _sampled_parameters(active_treatment_sensitivity=0.11),
            _fixed_parameters(),
        )

        drug_sensitivity = result.parameters["drug_sensitivity"]
        self.assertEqual(drug_sensitivity["anthracycline"], 0.11)
        self.assertEqual(drug_sensitivity["taxane"], 0.11)
        self.assertEqual(drug_sensitivity["endocrine_agent"], 0.0)
        self.assertEqual(drug_sensitivity["her2_directed_agent"], 0.0)

    def test_fixed_nuisance_parameters_do_not_vary_across_particles(self):
        fixed = _fixed_parameters()
        results = adapt_v1_prior_samples_to_volume_ode(
            _tnbc_chemo_contract(),
            [
                _sampled_parameters(active_treatment_sensitivity=0.08),
                _sampled_parameters(
                    growth_rate_per_day=0.009,
                    active_treatment_sensitivity=0.13,
                    resistant_fraction=0.31,
                ),
            ],
            fixed,
            particle_id_prefix="particle",
        )

        first = results[0].parameters
        second = results[1].parameters
        self.assertEqual(first["particle_id"], "particle_0000")
        self.assertEqual(second["particle_id"], "particle_0001")
        for fixed_name in (
            "carrying_capacity_ml",
            "drug_ec50",
            "drug_decay",
            "resistant_sensitivity_scale",
            "observation_noise_fraction",
        ):
            self.assertEqual(first[fixed_name], second[fixed_name])
        self.assertEqual(first["drug_sensitivity"]["endocrine_agent"], 0.0)
        self.assertEqual(second["drug_sensitivity"]["endocrine_agent"], 0.0)
        self.assertNotEqual(
            first["drug_sensitivity"]["anthracycline"],
            second["drug_sensitivity"]["anthracycline"],
        )

    def test_adapter_rejects_out_of_scope_contract(self):
        contract = resolve_parameter_contract(
            {
                "subtype": "HR-positive / HER2-negative",
                "treatment_context": "endocrine therapy",
            }
        )

        with self.assertRaises(ValueError):
            adapt_v1_prior_sample_to_volume_ode(contract, {}, _fixed_parameters())

    def test_inactive_drug_sensitivities_cannot_override_active_agents(self):
        fixed = _fixed_parameters()
        fixed["inactive_drug_sensitivities"] = {
            "taxane": 0.03,
            "endocrine_agent": 0.0,
            "her2_directed_agent": 0.0,
        }

        with self.assertRaises(ValueError):
            adapt_v1_prior_sample_to_volume_ode(
                _tnbc_chemo_contract(),
                _sampled_parameters(),
                fixed,
            )

    def test_adapter_rejects_particles_outside_layer1_bounds(self):
        with self.assertRaises(ValueError):
            adapt_v1_prior_sample_to_volume_ode(
                _tnbc_chemo_contract(),
                _sampled_parameters(active_treatment_sensitivity=0.501),
                _fixed_parameters(),
            )

    def test_adapter_requires_kinetics_for_all_drug_sensitivity_entries(self):
        fixed = _fixed_parameters()
        fixed["drug_ec50"] = {
            "anthracycline": 0.50,
            "taxane": 0.50,
            "endocrine_agent": 0.70,
        }

        with self.assertRaises(KeyError):
            adapt_v1_prior_sample_to_volume_ode(
                _tnbc_chemo_contract(),
                _sampled_parameters(),
                fixed,
            )


def _tnbc_chemo_contract():
    return resolve_parameter_contract(
        {
            "subtype": "TNBC",
            "regimen_name": "A/C-T neoadjuvant chemotherapy",
        }
    )


def _sampled_parameters(
    *,
    growth_rate_per_day=0.0067,
    active_treatment_sensitivity=0.09,
    resistant_fraction=0.20,
):
    return {
        "growth_rate_per_day": growth_rate_per_day,
        "active_treatment_sensitivity": active_treatment_sensitivity,
        "resistant_fraction": resistant_fraction,
    }


def _fixed_parameters():
    return {
        "carrying_capacity_ml": 300.0,
        "drug_decay": {
            "anthracycline": 0.25,
            "taxane": 0.20,
            "endocrine_agent": 0.06,
            "her2_directed_agent": 0.16,
        },
        "drug_ec50": {
            "anthracycline": 0.50,
            "taxane": 0.50,
            "endocrine_agent": 0.70,
            "her2_directed_agent": 0.50,
        },
        "resistant_sensitivity_scale": 0.05,
        "observation_noise_fraction": 0.12,
        "inactive_drug_sensitivities": {
            "endocrine_agent": 0.0,
            "her2_directed_agent": 0.0,
        },
    }


if __name__ == "__main__":
    unittest.main()
