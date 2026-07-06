"""Layer 0 parameter-contract rules for V1 prior construction.

The V1-A contract is intentionally narrow: TNBC + A/C-T-style chemotherapy may
personalize only the small parameter set that is identifiable in the first
volume-only prior stack. All unsupported or ambiguous contexts fall back to a
conservative generic contract with no learnable parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple


LEARNABLE_PARAMETER_NAMES = (
    "growth_rate_per_day",
    "active_treatment_sensitivity",
    "resistant_fraction",
)

FIXED_PARAMETER_NAMES = (
    "carrying_capacity_ml",
    "drug_decay",
    "drug_ec50",
    "resistant_sensitivity_scale",
    "observation_noise_fraction",
    "inactive_drug_sensitivities",
)

TNBC_CHEMO_CONTRACT_ID = "v1a_tnbc_chemo_parameter_contract"
CONSERVATIVE_GENERIC_CONTRACT_ID = "v1a_conservative_generic_parameter_contract"

_SUBTYPE_FIELDS = ("subtype", "disease_context", "cancer_subtype")
_TREATMENT_FIELDS = (
    "treatment_context",
    "treatment_regimen",
    "regimen_name",
    "schedule_type",
)
_TNBC_TOKENS = ("tnbc", "triple negative", "triple-negative")
_CHEMO_TOKENS = (
    "a/c-t",
    "ac-t",
    "ac t",
    "adriamycin",
    "cyclophosphamide",
    "anthracycline",
    "taxane",
    "taxol",
    "paclitaxel",
    "chemo",
    "chemotherapy",
)
_UNKNOWN_TOKENS = ("", "unknown", "unspecified", "not specified", "n/a", "na", "none")


@dataclass(frozen=True)
class ParameterContract:
    """Resolved Layer 0 contract for a prior-building context."""

    contract_id: str
    base_group: str
    learnable_parameters: Tuple[str, ...]
    fixed_parameters: Tuple[str, ...]
    active_treatment_drugs: Tuple[str, ...]
    warnings: Tuple[str, ...] = ()

    def can_personalize(self, parameter_name: str) -> bool:
        """Return whether a parameter may be personalized for this context."""

        return parameter_name in self.learnable_parameters

    def require_personalizable(self, parameter_name: str) -> None:
        """Fail loudly when code tries to personalize outside the contract."""

        if not self.can_personalize(parameter_name):
            raise ValueError(
                f"{parameter_name} is not learnable under {self.contract_id}"
            )


def resolve_parameter_contract(context: Mapping[str, object]) -> ParameterContract:
    """Resolve the V1-A parameter contract from patient and treatment context."""

    subtype_text = _combined_normalized_text(context, _SUBTYPE_FIELDS)
    treatment_text = _combined_normalized_text(context, _TREATMENT_FIELDS)
    is_tnbc = _contains_any(subtype_text, _TNBC_TOKENS)
    is_chemo = _contains_any(treatment_text, _CHEMO_TOKENS)

    if is_tnbc and is_chemo:
        return ParameterContract(
            contract_id=TNBC_CHEMO_CONTRACT_ID,
            base_group="tnbc_chemo",
            learnable_parameters=LEARNABLE_PARAMETER_NAMES,
            fixed_parameters=FIXED_PARAMETER_NAMES,
            active_treatment_drugs=("anthracycline", "taxane"),
        )

    if is_tnbc and _is_unknown_text(treatment_text):
        warning = (
            "Treatment regimen is missing or unknown; using conservative generic "
            "V1-A contract with no learnable parameters."
        )
    else:
        warning = (
            "V1-A supports only TNBC + A/C-T-style chemotherapy; using "
            "conservative generic contract with no learnable parameters."
        )

    return ParameterContract(
        contract_id=CONSERVATIVE_GENERIC_CONTRACT_ID,
        base_group="conservative_generic",
        learnable_parameters=(),
        fixed_parameters=LEARNABLE_PARAMETER_NAMES + FIXED_PARAMETER_NAMES,
        active_treatment_drugs=(),
        warnings=(warning,),
    )


def merge_sampled_and_fixed_parameters(
    contract: ParameterContract,
    sampled_parameters: Mapping[str, float],
    fixed_parameters: Mapping[str, object],
) -> Dict[str, object]:
    """Merge a sampled learnable particle with fixed parameters under a contract."""

    sampled_names = set(sampled_parameters)
    learnable_names = set(contract.learnable_parameters)
    unexpected = sorted(sampled_names - learnable_names)
    if unexpected:
        raise ValueError(
            "Sample contains parameters outside the Layer 0 contract: "
            + ", ".join(unexpected)
        )

    missing = sorted(learnable_names - sampled_names)
    if missing:
        raise ValueError(
            "Sample is missing learnable parameters required by the contract: "
            + ", ".join(missing)
        )

    fixed_conflicts = sorted(set(fixed_parameters) & learnable_names)
    if fixed_conflicts:
        raise ValueError(
            "Fixed parameter map includes learnable parameters: "
            + ", ".join(fixed_conflicts)
        )

    merged: Dict[str, object] = dict(fixed_parameters)
    merged.update(sampled_parameters)
    return merged


def _combined_normalized_text(
    context: Mapping[str, object], fields: Tuple[str, ...]
) -> str:
    values = []
    for field in fields:
        value = context.get(field)
        if value is not None:
            values.append(str(value))
    return _normalize_text(" ".join(values))


def _normalize_text(value: str) -> str:
    return " ".join(value.lower().replace("_", " ").split())


def _contains_any(value: str, tokens: Tuple[str, ...]) -> bool:
    return any(token in value for token in tokens)


def _is_unknown_text(value: str) -> bool:
    return value in _UNKNOWN_TOKENS
