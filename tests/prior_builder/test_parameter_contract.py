import unittest

from experiments.prior_builder.parameter_contract import (
    CONSERVATIVE_GENERIC_CONTRACT_ID,
    FIXED_PARAMETER_NAMES,
    LEARNABLE_PARAMETER_NAMES,
    TNBC_CHEMO_CONTRACT_ID,
    merge_sampled_and_fixed_parameters,
    resolve_parameter_contract,
)


class ParameterContractResolutionTests(unittest.TestCase):
    def test_tnbc_chemo_activates_only_v1a_learnable_parameters(self):
        contract = resolve_parameter_contract(
            {
                "subtype": "Triple-negative breast cancer",
                "regimen_name": "A/C-T neoadjuvant chemotherapy",
            }
        )

        self.assertEqual(contract.contract_id, TNBC_CHEMO_CONTRACT_ID)
        self.assertEqual(contract.base_group, "tnbc_chemo")
        self.assertEqual(contract.learnable_parameters, LEARNABLE_PARAMETER_NAMES)
        self.assertEqual(contract.fixed_parameters, FIXED_PARAMETER_NAMES)
        self.assertEqual(contract.active_treatment_drugs, ("anthracycline", "taxane"))
        self.assertEqual(contract.warnings, ())

    def test_inactive_drug_sensitivities_cannot_be_personalized(self):
        contract = resolve_parameter_contract(
            {
                "subtype": "TNBC",
                "treatment_context": "anthracycline taxane chemotherapy",
            }
        )

        self.assertTrue(contract.can_personalize("active_treatment_sensitivity"))
        self.assertFalse(contract.can_personalize("drug_sensitivity.endocrine_agent"))
        self.assertFalse(
            contract.can_personalize("drug_sensitivity.her2_directed_agent")
        )
        with self.assertRaises(ValueError):
            contract.require_personalizable("drug_sensitivity.endocrine_agent")

    def test_unknown_regimen_falls_back_to_conservative_generic_contract(self):
        contract = resolve_parameter_contract(
            {
                "subtype": "TNBC",
                "regimen_name": "unknown",
            }
        )

        self.assertEqual(contract.contract_id, CONSERVATIVE_GENERIC_CONTRACT_ID)
        self.assertEqual(contract.learnable_parameters, ())
        self.assertIn("growth_rate_per_day", contract.fixed_parameters)
        self.assertIn("missing or unknown", contract.warnings[0])

    def test_out_of_scope_context_emits_warning(self):
        contract = resolve_parameter_contract(
            {
                "subtype": "HR-positive / HER2-negative",
                "treatment_context": "endocrine therapy",
            }
        )

        self.assertEqual(contract.contract_id, CONSERVATIVE_GENERIC_CONTRACT_ID)
        self.assertEqual(contract.learnable_parameters, ())
        self.assertIn("supports only TNBC", contract.warnings[0])


class ParameterMergeTests(unittest.TestCase):
    def test_fixed_parameters_remain_fixed_after_sampling(self):
        contract = resolve_parameter_contract(
            {
                "subtype": "TNBC",
                "regimen_name": "AC-T chemotherapy",
            }
        )
        fixed = {
            "carrying_capacity_ml": 300.0,
            "drug_decay": {"anthracycline": 0.25, "taxane": 0.20},
            "drug_ec50": {"anthracycline": 0.50, "taxane": 0.50},
            "resistant_sensitivity_scale": 0.05,
            "observation_noise_fraction": 0.12,
            "inactive_drug_sensitivities": {
                "endocrine_agent": 0.0,
                "her2_directed_agent": 0.0,
            },
        }
        sampled = {
            "growth_rate_per_day": 0.0067,
            "active_treatment_sensitivity": 0.09,
            "resistant_fraction": 0.20,
        }

        merged = merge_sampled_and_fixed_parameters(contract, sampled, fixed)

        for name, value in fixed.items():
            self.assertEqual(merged[name], value)
        for name, value in sampled.items():
            self.assertEqual(merged[name], value)

    def test_sampling_outside_the_contract_fails(self):
        contract = resolve_parameter_contract(
            {
                "subtype": "TNBC",
                "regimen_name": "AC-T chemotherapy",
            }
        )

        with self.assertRaises(ValueError):
            merge_sampled_and_fixed_parameters(
                contract,
                {
                    "growth_rate_per_day": 0.0067,
                    "active_treatment_sensitivity": 0.09,
                    "resistant_fraction": 0.20,
                    "carrying_capacity_ml": 500.0,
                },
                {},
            )

    def test_missing_learnable_sample_fails(self):
        contract = resolve_parameter_contract(
            {
                "subtype": "TNBC",
                "regimen_name": "AC-T chemotherapy",
            }
        )

        with self.assertRaises(ValueError):
            merge_sampled_and_fixed_parameters(
                contract,
                {
                    "growth_rate_per_day": 0.0067,
                    "active_treatment_sensitivity": 0.09,
                },
                {},
            )


if __name__ == "__main__":
    unittest.main()
