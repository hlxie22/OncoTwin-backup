"""Layer 2 subtype/treatment population priors for V1 prior construction.

Layer 2 provides the transparent population-level prior before patient-specific
pathology, biomarker, MRI, or AI residual layers are applied. The first V1-A
configuration is deliberately narrow: TNBC treated with A/C-T-style
chemotherapy, represented in the three learnable volume-only parameters from
Layer 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .parameter_contract import (
    LEARNABLE_PARAMETER_NAMES,
    TNBC_CHEMO_CONTRACT_ID,
    ParameterContract,
)
from .transforms import (
    from_transformed,
    median_interval_to_normal,
    sample_correlated_transformed_prior,
    validate_covariance,
)


POPULATION_PRIOR_VERSION = "oncotwin_prior_v1_layer2"
TNBC_CHEMO_POPULATION_GROUP = "tnbc_chemo"


@dataclass(frozen=True)
class PopulationPriorParameter:
    """Configured marginal prior for one learnable simulator parameter."""

    parameter_name: str
    transformed_name: str
    median: float
    interval_80: Tuple[float, float]
    transform: str
    transformed_mean: float
    transformed_std: float


@dataclass(frozen=True)
class PopulationPrior:
    """Resolved Layer 2 correlated population prior."""

    prior_version: str
    population_group: str
    contract_id: str
    parameters: Tuple[PopulationPriorParameter, ...]
    transformed_parameter_names: Tuple[str, ...]
    transformed_means: Dict[str, float]
    transformed_covariance: Tuple[Tuple[float, ...], ...]
    parameter_correlations: Dict[Tuple[str, str], float]
    warnings: Tuple[str, ...] = ()

    @property
    def parameter_names(self) -> Tuple[str, ...]:
        """Return simulator-facing learnable parameter names in covariance order."""

        return tuple(parameter.parameter_name for parameter in self.parameters)

    def layer_contribution(self) -> Dict[str, object]:
        """Return a JSON-friendly trace of the Layer 2 prior contribution."""

        return {
            "layer": "population_prior",
            "prior_version": self.prior_version,
            "population_group": self.population_group,
            "contract_id": self.contract_id,
            "medians": {
                parameter.parameter_name: parameter.median
                for parameter in self.parameters
            },
            "central_80_intervals": {
                parameter.parameter_name: list(parameter.interval_80)
                for parameter in self.parameters
            },
            "correlations": {
                f"{left}__{right}": correlation
                for (left, right), correlation in self.parameter_correlations.items()
            },
            "transformed_parameter_names": list(self.transformed_parameter_names),
            "transformed_means": dict(self.transformed_means),
            "transformed_covariance": [list(row) for row in self.transformed_covariance],
        }


@dataclass(frozen=True)
class PopulationPriorSampleResult:
    """Layer 2 samples in simulator-facing parameter space."""

    samples: List[Dict[str, float]]


@dataclass(frozen=True)
class _ParameterPriorSpec:
    parameter_name: str
    transformed_name: str
    median: float
    lower_80: float
    upper_80: float
    transform: str


_TNBC_CHEMO_PARAMETER_PRIORS: Tuple[_ParameterPriorSpec, ...] = (
    _ParameterPriorSpec(
        parameter_name="growth_rate_per_day",
        transformed_name="log_growth_rate_per_day",
        median=0.0067,
        lower_80=0.0030,
        upper_80=0.0150,
        transform="log",
    ),
    _ParameterPriorSpec(
        parameter_name="active_treatment_sensitivity",
        transformed_name="log_active_treatment_sensitivity",
        median=0.090,
        lower_80=0.040,
        upper_80=0.180,
        transform="log",
    ),
    _ParameterPriorSpec(
        parameter_name="resistant_fraction",
        transformed_name="logit_resistant_fraction",
        median=0.20,
        lower_80=0.08,
        upper_80=0.45,
        transform="logit",
    ),
)

_TNBC_CHEMO_PARAMETER_CORRELATIONS: Dict[Tuple[str, str], float] = {
    ("growth_rate_per_day", "active_treatment_sensitivity"): 0.25,
    ("active_treatment_sensitivity", "resistant_fraction"): -0.40,
    ("growth_rate_per_day", "resistant_fraction"): -0.05,
}


def resolve_population_prior(contract: ParameterContract) -> PopulationPrior:
    """Resolve the Layer 2 population prior for an in-scope V1-A contract."""

    _require_tnbc_chemo_contract(contract)
    parameters = tuple(
        _build_parameter_prior(spec) for spec in _TNBC_CHEMO_PARAMETER_PRIORS
    )
    covariance = _build_covariance(parameters, _TNBC_CHEMO_PARAMETER_CORRELATIONS)
    validate_covariance(covariance, dimension=len(parameters))

    return PopulationPrior(
        prior_version=POPULATION_PRIOR_VERSION,
        population_group=TNBC_CHEMO_POPULATION_GROUP,
        contract_id=contract.contract_id,
        parameters=parameters,
        transformed_parameter_names=tuple(
            parameter.transformed_name for parameter in parameters
        ),
        transformed_means={
            parameter.transformed_name: parameter.transformed_mean
            for parameter in parameters
        },
        transformed_covariance=tuple(tuple(row) for row in covariance),
        parameter_correlations=dict(_TNBC_CHEMO_PARAMETER_CORRELATIONS),
    )


def sample_population_prior(
    prior: PopulationPrior,
    *,
    n_samples: int,
    seed: Optional[int] = None,
) -> PopulationPriorSampleResult:
    """Sample Layer 2 particles in simulator-facing parameter space."""

    transformed_samples = sample_correlated_transformed_prior(
        prior.transformed_means,
        prior.transformed_covariance,
        n_samples=n_samples,
        seed=seed,
    )

    return PopulationPriorSampleResult(
        samples=[from_transformed(sample) for sample in transformed_samples]
    )


def _require_tnbc_chemo_contract(contract: ParameterContract) -> None:
    if contract.contract_id != TNBC_CHEMO_CONTRACT_ID:
        raise ValueError(
            "Layer 2 population prior is configured only for the TNBC "
            "A/C-T chemotherapy V1-A contract."
        )
    if contract.base_group != TNBC_CHEMO_POPULATION_GROUP:
        raise ValueError(
            "Layer 2 population prior does not match contract base group: "
            f"{contract.base_group}"
        )
    if tuple(contract.learnable_parameters) != LEARNABLE_PARAMETER_NAMES:
        raise ValueError(
            "Layer 2 population prior requires the V1-A learnable parameter set."
        )


def _build_parameter_prior(spec: _ParameterPriorSpec) -> PopulationPriorParameter:
    approximation = median_interval_to_normal(
        spec.median,
        spec.lower_80,
        spec.upper_80,
        transform=spec.transform,
    )
    return PopulationPriorParameter(
        parameter_name=spec.parameter_name,
        transformed_name=spec.transformed_name,
        median=spec.median,
        interval_80=(spec.lower_80, spec.upper_80),
        transform=spec.transform,
        transformed_mean=approximation.mean,
        transformed_std=approximation.std,
    )


def _build_covariance(
    parameters: Sequence[PopulationPriorParameter],
    correlations: Mapping[Tuple[str, str], float],
) -> List[List[float]]:
    covariance: List[List[float]] = []
    for left in parameters:
        row = []
        for right in parameters:
            if left.parameter_name == right.parameter_name:
                row.append(left.transformed_std**2)
            else:
                correlation = _lookup_correlation(
                    correlations,
                    left.parameter_name,
                    right.parameter_name,
                )
                row.append(correlation * left.transformed_std * right.transformed_std)
        covariance.append(row)
    return covariance


def _lookup_correlation(
    correlations: Mapping[Tuple[str, str], float],
    left: str,
    right: str,
) -> float:
    if (left, right) in correlations:
        return correlations[(left, right)]
    if (right, left) in correlations:
        return correlations[(right, left)]
    raise KeyError(f"Missing Layer 2 correlation for {left} and {right}")
