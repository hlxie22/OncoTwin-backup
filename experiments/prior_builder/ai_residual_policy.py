"""Layer 5 guarded AI residual policy for V1 prior construction.

Layer 5 is intentionally fail-closed in V1: it preserves Layer 4 exactly unless
an explicitly validated residual signal supplies small bounded adjustments.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Mapping, Sequence, Tuple

from .mri_feature_rules import MRIFeaturePrior
from .population_prior import PopulationPriorSampleResult
from .transforms import (
    TRANSFORMED_PARAMETER_NAMES,
    from_transformed,
    sample_correlated_transformed_prior,
    validate_covariance,
)


AI_RESIDUAL_LAYER_VERSION = "oncotwin_prior_v1_layer5"
RESIDUAL_SIGNAL_FIELDS = ("ai_residual", "ai_residual_policy", "layer5_ai_residual")
SHIFT_BOUNDS = {
    "log_growth_rate_shift": ("log_growth_rate_per_day", math.log(1.15)),
    "log_active_treatment_sensitivity_shift": (
        "log_active_treatment_sensitivity",
        math.log(1.15),
    ),
    "logit_resistant_fraction_shift": ("logit_resistant_fraction", 0.25),
}
VARIANCE_MULTIPLIERS = {
    "growth_variance_multiplier": "log_growth_rate_per_day",
    "sensitivity_variance_multiplier": "log_active_treatment_sensitivity",
    "resistant_variance_multiplier": "logit_resistant_fraction",
}
MAX_VARIANCE_MULTIPLIER = 1.50
ALLOWED_FIELDS = frozenset(
    {
        "validated",
        "is_validated",
        "model_version",
        "residual_model_version",
        "policy_version",
        "model_name",
        "source",
        *SHIFT_BOUNDS,
        *VARIANCE_MULTIPLIERS,
    }
)


@dataclass(frozen=True)
class Layer5Contribution:
    rule_id: str
    effects: Dict[str, float]
    explanation: str
    uncertainty_driver: bool = False

    def as_dict(self) -> Dict[str, object]:
        return {
            "rule_id": self.rule_id,
            "rule_family": "ai_residual",
            "effects": dict(self.effects),
            "evidence_level": "validated_model_policy",
            "explanation": self.explanation,
            "uncertainty_driver": self.uncertainty_driver,
        }


@dataclass(frozen=True)
class AIResidualPrior:
    prior_version: str
    base_prior_version: str
    population_group: str
    contract_id: str
    transformed_parameter_names: Tuple[str, ...]
    transformed_means: Dict[str, float]
    transformed_covariance: Tuple[Tuple[float, ...], ...]
    policy_mode: str
    residual_model_version: str | None
    layer_contributions: Tuple[Layer5Contribution, ...]
    warnings: Tuple[str, ...] = ()
    uncertainty_drivers: Tuple[str, ...] = ()

    @property
    def active(self) -> bool:
        return self.policy_mode == "validated_residual"

    def layer_contribution(self) -> Dict[str, object]:
        return {
            "layer": "ai_residual_policy",
            "prior_version": self.prior_version,
            "base_prior_version": self.base_prior_version,
            "contract_id": self.contract_id,
            "policy_mode": self.policy_mode,
            "residual_model_version": self.residual_model_version,
            "rules": [rule.as_dict() for rule in self.layer_contributions],
            "transformed_parameter_names": list(self.transformed_parameter_names),
            "transformed_means": dict(self.transformed_means),
            "transformed_covariance": [list(row) for row in self.transformed_covariance],
            "warnings": list(self.warnings),
            "uncertainty_drivers": list(self.uncertainty_drivers),
        }


def apply_ai_residual_policy(
    prior: MRIFeaturePrior,
    context: Mapping[str, object],
) -> AIResidualPrior:
    """Apply bounded validated residual shifts to a Layer 4 prior."""

    _require_expected_prior_order(prior)
    signal = _signal(context)
    if signal is None:
        return _copy_prior(prior, policy_mode="inactive_noop")

    unexpected = sorted(set(signal) - ALLOWED_FIELDS)
    if unexpected:
        raise ValueError("unsupported Layer 5 residual fields: " + ", ".join(unexpected))
    if not _truthy(signal.get("validated", signal.get("is_validated"))):
        raise ValueError("Layer 5 AI residual input must be explicitly validated")
    model_version = _model_version(signal)
    if model_version is None:
        raise ValueError("Layer 5 AI residual input requires model_version")

    means = dict(prior.transformed_means)
    effects: Dict[str, float] = {}
    for field, (transformed_name, max_abs_shift) in SHIFT_BOUNDS.items():
        shift = _optional_float(signal.get(field), field)
        if shift is None:
            continue
        if abs(shift) > max_abs_shift:
            raise ValueError(f"{field} exceeds Layer 5 bound")
        means[transformed_name] += shift
        effects[field] = shift

    variance_multipliers = {name: 1.0 for name in prior.transformed_parameter_names}
    for field, transformed_name in VARIANCE_MULTIPLIERS.items():
        multiplier = _optional_float(signal.get(field), field)
        if multiplier is None:
            continue
        if multiplier < 1.0:
            raise ValueError(f"{field} may not narrow covariance")
        if multiplier > MAX_VARIANCE_MULTIPLIER:
            raise ValueError(f"{field} exceeds Layer 5 maximum")
        variance_multipliers[transformed_name] *= multiplier
        effects[field] = multiplier

    covariance = _scale_covariance(
        prior.transformed_covariance,
        prior.transformed_parameter_names,
        variance_multipliers,
    )
    validate_covariance(covariance, dimension=len(prior.transformed_parameter_names))

    if not effects:
        contribution = Layer5Contribution(
            rule_id="validated_ai_residual_no_effect_v1",
            effects={},
            explanation="Validated residual metadata was present, but no bounded shifts were supplied.",
        )
        return _copy_prior(
            prior,
            policy_mode="validated_noop",
            residual_model_version=model_version,
            layer_contributions=(contribution,),
            warnings=("Validated Layer 5 residual signal had no effect fields.",),
        )

    contribution = Layer5Contribution(
        rule_id="validated_ai_residual_v1",
        effects=effects,
        explanation=(
            "A validated residual model supplied bounded transformed-space shifts; "
            "V1 permits only small shifts and covariance preservation or widening."
        ),
        uncertainty_driver=any(field in effects for field in VARIANCE_MULTIPLIERS),
    )
    return _copy_prior(
        prior,
        policy_mode="validated_residual",
        residual_model_version=model_version,
        transformed_means=means,
        transformed_covariance=covariance,
        layer_contributions=(contribution,),
        uncertainty_drivers=(contribution.rule_id,) if contribution.uncertainty_driver else (),
    )


def sample_ai_residual_prior(
    prior: AIResidualPrior,
    *,
    n_samples: int,
    seed: int | None = None,
) -> PopulationPriorSampleResult:
    transformed_samples = sample_correlated_transformed_prior(
        prior.transformed_means,
        prior.transformed_covariance,
        n_samples=n_samples,
        seed=seed,
    )
    return PopulationPriorSampleResult(
        samples=[from_transformed(sample) for sample in transformed_samples]
    )


def _copy_prior(
    prior: MRIFeaturePrior,
    *,
    policy_mode: str,
    residual_model_version: str | None = None,
    transformed_means: Mapping[str, float] | None = None,
    transformed_covariance: Sequence[Sequence[float]] | None = None,
    layer_contributions: Tuple[Layer5Contribution, ...] = (),
    warnings: Tuple[str, ...] = (),
    uncertainty_drivers: Tuple[str, ...] = (),
) -> AIResidualPrior:
    return AIResidualPrior(
        prior_version=AI_RESIDUAL_LAYER_VERSION,
        base_prior_version=prior.prior_version,
        population_group=prior.population_group,
        contract_id=prior.contract_id,
        transformed_parameter_names=prior.transformed_parameter_names,
        transformed_means=dict(transformed_means or prior.transformed_means),
        transformed_covariance=tuple(
            tuple(row) for row in (transformed_covariance or prior.transformed_covariance)
        ),
        policy_mode=policy_mode,
        residual_model_version=residual_model_version,
        layer_contributions=layer_contributions,
        warnings=warnings,
        uncertainty_drivers=uncertainty_drivers,
    )


def _signal(context: Mapping[str, object]) -> Mapping[str, object] | None:
    for field in RESIDUAL_SIGNAL_FIELDS:
        value = context.get(field)
        if value in (None, ""):
            continue
        if not isinstance(value, Mapping):
            raise ValueError(f"{field} must be a mapping")
        return value
    return None


def _model_version(signal: Mapping[str, object]) -> str | None:
    for field in ("model_version", "residual_model_version", "policy_version"):
        value = signal.get(field)
        if value not in (None, ""):
            return str(value)
    return None


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "validated"}


def _optional_float(value: object, name: str) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _scale_covariance(
    covariance: Sequence[Sequence[float]],
    parameter_names: Sequence[str],
    variance_multipliers: Mapping[str, float],
) -> list[list[float]]:
    scales = [math.sqrt(variance_multipliers[name]) for name in parameter_names]
    return [
        [float(value) * scales[row_index] * scales[column_index] for column_index, value in enumerate(row)]
        for row_index, row in enumerate(covariance)
    ]


def _require_expected_prior_order(prior: MRIFeaturePrior) -> None:
    if tuple(prior.transformed_parameter_names) != TRANSFORMED_PARAMETER_NAMES:
        raise ValueError("Layer 5 requires the V1-A transformed parameter order")
    missing = sorted(set(TRANSFORMED_PARAMETER_NAMES) - set(prior.transformed_means))
    if missing:
        raise KeyError("Layer 5 prior is missing transformed means: " + ", ".join(missing))
