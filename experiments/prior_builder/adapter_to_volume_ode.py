"""Adapter from V1 prior samples to volume-ODE simulator parameters.

The V1 prior stack personalizes a deliberately small set of biologic
parameters. The current volume-only simulator expects a fully resolved
parameter dictionary, including per-drug sensitivity maps and fixed nuisance
parameters. This module keeps that translation explicit so V1 particles cannot
silently personalize inactive drugs or simulator constants.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Mapping, Sequence, Tuple

from .bounds import validate_parameter_bounds
from .parameter_contract import ParameterContract, merge_sampled_and_fixed_parameters


DEFAULT_GROWTH_LAW = "logistic"
_REQUIRED_LEARNABLE_FOR_VOLUME_ODE = (
    "growth_rate_per_day",
    "active_treatment_sensitivity",
    "resistant_fraction",
)


@dataclass(frozen=True)
class VolumeOdeAdapterResult:
    """Simulator-ready parameters plus any non-fatal prior-layer warnings."""

    parameters: Dict[str, object]
    warnings: Tuple[str, ...] = ()


def adapt_v1_prior_sample_to_volume_ode(
    contract: ParameterContract,
    sampled_parameters: Mapping[str, float],
    fixed_parameters: Mapping[str, object],
    *,
    particle_id: str | None = None,
) -> VolumeOdeAdapterResult:
    """Convert one V1-A learnable sample into resolved volume-ODE params.

    ``sampled_parameters`` must contain only the Layer 0 learnable parameters.
    ``fixed_parameters`` must contain exactly the fixed nuisance terms from the
    contract, including an ``inactive_drug_sensitivities`` map. The shared
    ``active_treatment_sensitivity`` is copied to each active chemotherapy drug.
    """

    _require_volume_ode_contract(contract)
    _require_exact_fixed_parameters(contract, fixed_parameters)
    merge_sampled_and_fixed_parameters(contract, sampled_parameters, fixed_parameters)
    bounded = validate_parameter_bounds(
        sampled_parameters,
        required_parameter_names=contract.learnable_parameters,
    )

    active_sensitivity = bounded.parameters["active_treatment_sensitivity"]
    inactive_sensitivities = _require_unit_interval_mapping(
        fixed_parameters["inactive_drug_sensitivities"],
        "inactive_drug_sensitivities",
    )
    active_drugs = tuple(contract.active_treatment_drugs)
    inactive_active_overlap = sorted(set(inactive_sensitivities) & set(active_drugs))
    if inactive_active_overlap:
        raise ValueError(
            "inactive_drug_sensitivities includes active treatment drugs: "
            + ", ".join(inactive_active_overlap)
        )

    drug_sensitivity = {
        drug: active_sensitivity
        for drug in active_drugs
    }
    drug_sensitivity.update(inactive_sensitivities)

    drug_decay = _require_positive_mapping(fixed_parameters["drug_decay"], "drug_decay")
    drug_ec50 = _require_positive_mapping(fixed_parameters["drug_ec50"], "drug_ec50")
    _require_kinetics_for_sensitivity_drugs("drug_decay", drug_decay, drug_sensitivity)
    _require_kinetics_for_sensitivity_drugs("drug_ec50", drug_ec50, drug_sensitivity)

    parameters: Dict[str, object] = {
        "growth_law": DEFAULT_GROWTH_LAW,
        "growth_rate": bounded.parameters["growth_rate_per_day"],
        "carrying_capacity_ml": _require_positive(
            fixed_parameters["carrying_capacity_ml"],
            "carrying_capacity_ml",
        ),
        "drug_sensitivity": drug_sensitivity,
        "drug_ec50": drug_ec50,
        "drug_decay": drug_decay,
        "resistant_fraction": bounded.parameters["resistant_fraction"],
        "resistant_sensitivity_scale": _require_unit_interval(
            fixed_parameters["resistant_sensitivity_scale"],
            "resistant_sensitivity_scale",
        ),
        "observation_noise_fraction": _require_unit_interval(
            fixed_parameters["observation_noise_fraction"],
            "observation_noise_fraction",
        ),
    }
    if particle_id is not None:
        parameters["particle_id"] = str(particle_id)

    return VolumeOdeAdapterResult(parameters=parameters, warnings=bounded.warnings)


def adapt_v1_prior_samples_to_volume_ode(
    contract: ParameterContract,
    sampled_particles: Sequence[Mapping[str, float]],
    fixed_parameters: Mapping[str, object],
    *,
    particle_id_prefix: str = "v1_prior_particle",
) -> List[VolumeOdeAdapterResult]:
    """Convert a sequence of V1-A particles into simulator-ready parameters."""

    return [
        adapt_v1_prior_sample_to_volume_ode(
            contract,
            sample,
            fixed_parameters,
            particle_id=f"{particle_id_prefix}_{index:04d}",
        )
        for index, sample in enumerate(sampled_particles)
    ]


def _require_volume_ode_contract(contract: ParameterContract) -> None:
    missing = sorted(
        set(_REQUIRED_LEARNABLE_FOR_VOLUME_ODE) - set(contract.learnable_parameters)
    )
    if missing:
        raise ValueError(
            "Volume-ODE adapter requires an in-scope V1-A contract; missing "
            "learnable parameters: " + ", ".join(missing)
        )
    if not contract.active_treatment_drugs:
        raise ValueError(
            "Volume-ODE adapter requires active treatment drugs in the contract."
        )


def _require_exact_fixed_parameters(
    contract: ParameterContract,
    fixed_parameters: Mapping[str, object],
) -> None:
    expected = set(contract.fixed_parameters)
    actual = set(fixed_parameters)
    missing = sorted(expected - actual)
    if missing:
        raise KeyError(f"Missing fixed parameters: {', '.join(missing)}")
    unexpected = sorted(actual - expected)
    if unexpected:
        raise ValueError(
            "Fixed parameter map includes fields outside the Layer 0 contract: "
            + ", ".join(unexpected)
        )


def _require_positive_mapping(value: object, name: str) -> Dict[str, float]:
    mapping = _require_mapping(value, name)
    if not mapping:
        raise ValueError(f"{name} must not be empty")
    return {
        key: _require_positive(entry, f"{name}.{key}")
        for key, entry in mapping.items()
    }


def _require_unit_interval_mapping(value: object, name: str) -> Dict[str, float]:
    mapping = _require_mapping(value, name)
    return {
        key: _require_unit_interval(entry, f"{name}.{key}")
        for key, entry in mapping.items()
    }


def _require_mapping(value: object, name: str) -> Dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return {str(key): entry for key, entry in value.items()}


def _require_kinetics_for_sensitivity_drugs(
    name: str,
    kinetics: Mapping[str, float],
    drug_sensitivity: Mapping[str, float],
) -> None:
    missing = sorted(set(drug_sensitivity) - set(kinetics))
    if missing:
        raise KeyError(
            f"{name} is missing entries for drugs with sensitivity values: "
            + ", ".join(missing)
        )


def _require_positive(value: object, name: str) -> float:
    numeric = _require_finite_number(value, name)
    if numeric <= 0.0:
        raise ValueError(f"{name} must be positive")
    return numeric


def _require_unit_interval(value: object, name: str) -> float:
    numeric = _require_finite_number(value, name)
    if numeric < 0.0 or numeric > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return numeric


def _require_finite_number(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric
