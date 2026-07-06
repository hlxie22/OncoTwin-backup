"""Layer 4 MRI feature and QC rules for V1 prior construction.

V1-A treats MRI features as conservative uncertainty and measurement-quality
signals. MRI/QC can widen covariance, increase observation noise, add warnings,
or mark uncertainty drivers, but it does not shift prior means.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Mapping, Sequence, Tuple

from .bounds import resolve_observation_noise
from .pathology_biomarker_rules import PathologyBiomarkerPrior
from .population_prior import PopulationPrior, PopulationPriorSampleResult
from .transforms import (
    TRANSFORMED_PARAMETER_NAMES,
    from_transformed,
    sample_correlated_transformed_prior,
    validate_covariance,
)


MRI_FEATURE_LAYER_VERSION = "oncotwin_prior_v1_layer4"

_GROWTH_Z = "log_growth_rate_per_day"
_SENSITIVITY_Z = "log_active_treatment_sensitivity"
_RESISTANCE_Z = "logit_resistant_fraction"
_ANATOMIC_VOLUME_FIELDS = ("volume_ml", "tumor_volume_ml")
_FUNCTIONAL_VOLUME_FIELDS = ("functional_tumor_volume_ml", "ftv_ml", "ftv")
_DIAMETER_FIELDS = ("longest_diameter_cm", "diameter_cm")
_SEGMENTATION_QC_FIELDS = (
    "segmentation_qc",
    "segmentation_confidence",
    "segmentation_quality",
)
_REGISTRATION_QC_FIELDS = ("registration_qc", "registration_confidence")
_LOW_QC_TOKENS = (
    "low",
    "poor",
    "fail",
    "failed",
    "motion",
    "artifact",
    "manual review",
    "uncertain",
)
_HIGH_QC_TOKENS = ("high", "good", "excellent", "pass", "passed")


@dataclass(frozen=True)
class Layer4RuleContribution:
    """One MRI/QC rule and its transparent effect."""

    rule_id: str
    rule_family: str
    condition: str
    effects: Dict[str, object]
    evidence_level: str
    explanation: str
    uncertainty_driver: bool = False

    def as_dict(self) -> Dict[str, object]:
        return {
            "rule_id": self.rule_id,
            "rule_family": self.rule_family,
            "condition": self.condition,
            "effects": dict(self.effects),
            "evidence_level": self.evidence_level,
            "explanation": self.explanation,
            "uncertainty_driver": self.uncertainty_driver,
        }


@dataclass(frozen=True)
class MRIFeaturePrior:
    """Layer 4 prior after MRI feature and QC rules are applied."""

    prior_version: str
    base_prior_version: str
    population_group: str
    contract_id: str
    transformed_parameter_names: Tuple[str, ...]
    transformed_means: Dict[str, float]
    transformed_covariance: Tuple[Tuple[float, ...], ...]
    observation_noise_kind: str
    log_scale_observation_noise: float
    update_mode: str
    layer_contributions: Tuple[Layer4RuleContribution, ...]
    warnings: Tuple[str, ...] = ()
    uncertainty_drivers: Tuple[str, ...] = ()

    @property
    def report_only(self) -> bool:
        return self.update_mode == "report_only"

    def layer_contribution(self) -> Dict[str, object]:
        return {
            "layer": "mri_feature_rules",
            "prior_version": self.prior_version,
            "base_prior_version": self.base_prior_version,
            "contract_id": self.contract_id,
            "observation_noise_kind": self.observation_noise_kind,
            "log_scale_observation_noise": self.log_scale_observation_noise,
            "update_mode": self.update_mode,
            "rules": [rule.as_dict() for rule in self.layer_contributions],
            "transformed_parameter_names": list(self.transformed_parameter_names),
            "transformed_means": dict(self.transformed_means),
            "transformed_covariance": [list(row) for row in self.transformed_covariance],
            "warnings": list(self.warnings),
            "uncertainty_drivers": list(self.uncertainty_drivers),
        }


@dataclass(frozen=True)
class _RuleEffect:
    contribution: Layer4RuleContribution
    variance_multipliers: Dict[str, float]


def apply_mri_feature_rules(
    prior: PopulationPrior | PathologyBiomarkerPrior,
    context: Mapping[str, object],
) -> MRIFeaturePrior:
    """Apply V1 Layer 4 MRI feature and QC rules to a Layer 2/3 prior."""

    _require_expected_prior_order(prior)
    observation_noise = _resolve_observation_noise(context)
    report_only = _is_diameter_only(context)
    warnings = list(observation_noise.warnings)

    if report_only:
        effects = [_diameter_only_effect()]
        warnings.append(
            "Only a longest-diameter measurement is available; Layer 4 is using "
            "report-only mode rather than a high-confidence volumetric MRI update."
        )
    else:
        effects = _rule_effects(context)

    multipliers = {name: 1.0 for name in prior.transformed_parameter_names}
    uncertainty_drivers: List[str] = []
    for effect in effects:
        for name, multiplier in effect.variance_multipliers.items():
            if multiplier <= 0 or not math.isfinite(multiplier):
                raise ValueError("variance multipliers must be positive and finite")
            multipliers[name] *= multiplier
        if effect.contribution.uncertainty_driver:
            uncertainty_drivers.append(effect.contribution.rule_id)

    covariance = _scale_covariance(
        prior.transformed_covariance,
        prior.transformed_parameter_names,
        multipliers,
    )
    validate_covariance(covariance, dimension=len(prior.transformed_parameter_names))

    if any(
        effect.contribution.rule_id == "ftv_anatomic_volume_inconsistency_v1"
        for effect in effects
    ):
        warnings.append(
            "Functional tumor volume is inconsistent with anatomic volume; Layer "
            "4 widened response/resistance uncertainty and marked MRI QC for review."
        )

    return MRIFeaturePrior(
        prior_version=MRI_FEATURE_LAYER_VERSION,
        base_prior_version=prior.prior_version,
        population_group=prior.population_group,
        contract_id=prior.contract_id,
        transformed_parameter_names=prior.transformed_parameter_names,
        transformed_means=dict(prior.transformed_means),
        transformed_covariance=tuple(tuple(row) for row in covariance),
        observation_noise_kind=observation_noise.measurement_kind,
        log_scale_observation_noise=observation_noise.log_scale_noise,
        update_mode="report_only" if report_only else "volume_update",
        layer_contributions=tuple(effect.contribution for effect in effects),
        warnings=tuple(warnings),
        uncertainty_drivers=tuple(dict.fromkeys(uncertainty_drivers)),
    )


def sample_mri_feature_prior(
    prior: MRIFeaturePrior,
    *,
    n_samples: int,
    seed: int | None = None,
) -> PopulationPriorSampleResult:
    """Sample Layer 4 particles in simulator-facing parameter space."""

    transformed_samples = sample_correlated_transformed_prior(
        prior.transformed_means,
        prior.transformed_covariance,
        n_samples=n_samples,
        seed=seed,
    )
    return PopulationPriorSampleResult(
        samples=[from_transformed(sample) for sample in transformed_samples]
    )


def _rule_effects(context: Mapping[str, object]) -> List[_RuleEffect]:
    effects: List[_RuleEffect] = []
    if _qc_level(_first_present(context, _SEGMENTATION_QC_FIELDS)) == "low":
        effects.append(
            _effect(
                "low_segmentation_qc_v1",
                "segmentation_qc",
                "segmentation_qc is low or segmentation confidence < 0.50",
                {
                    "observation_noise": "low_qc_mri_volume",
                    "growth_variance_multiplier": 1.05,
                    "sensitivity_variance_multiplier": 1.10,
                    "resistant_variance_multiplier": 1.10,
                },
                "Low segmentation QC inflates measurement noise and modestly widens uncertainty.",
            )
        )
    if _qc_level(_first_present(context, _REGISTRATION_QC_FIELDS)) == "low":
        effects.append(
            _effect(
                "low_registration_qc_v1",
                "registration_qc",
                "registration_qc is low for longitudinal imaging",
                {
                    "growth_variance_multiplier": 1.10,
                    "resistant_variance_multiplier": 1.10,
                },
                "Low registration QC widens growth and resistance uncertainty.",
            )
        )
    enhancement_std = _optional_number(context.get("enhancement_std"), "enhancement_std")
    if enhancement_std is not None and enhancement_std >= 0.30:
        effects.append(
            _effect(
                "high_enhancement_heterogeneity_v1",
                "enhancement_heterogeneity",
                "enhancement_std >= 0.30",
                {"resistant_variance_multiplier": 1.25},
                "High enhancement heterogeneity widens resistance/delivery uncertainty without shifting means.",
            )
        )
    low_enhancement = _optional_fraction(
        context.get("low_enhancement_fraction"), "low_enhancement_fraction"
    )
    if low_enhancement is not None and low_enhancement >= 0.35:
        effects.append(
            _effect(
                "low_enhancement_fraction_v1",
                "enhancement_delivery",
                "low_enhancement_fraction >= 0.35",
                {
                    "sensitivity_variance_multiplier": 1.15,
                    "resistant_variance_multiplier": 1.20,
                },
                "A large low-enhancement fraction widens treatment sensitivity and resistance uncertainty.",
            )
        )
    effects.extend(_ftv_consistency_effects(context))
    return effects


def _ftv_consistency_effects(context: Mapping[str, object]) -> List[_RuleEffect]:
    volume = _optional_number(_first_present(context, _ANATOMIC_VOLUME_FIELDS), "volume_ml")
    ftv = _optional_number(
        _first_present(context, _FUNCTIONAL_VOLUME_FIELDS),
        "functional_tumor_volume_ml",
    )
    if volume is None or ftv is None:
        return []
    if volume <= 0 or ftv < 0:
        raise ValueError("MRI volumes must be non-negative and anatomic volume positive")
    if 0.05 <= ftv / volume <= 1.10:
        return []
    return [
        _effect(
            "ftv_anatomic_volume_inconsistency_v1",
            "mri_qc_consistency",
            "functional_tumor_volume_ml / volume_ml is outside 0.05-1.10",
            {
                "sensitivity_variance_multiplier": 1.10,
                "resistant_variance_multiplier": 1.10,
            },
            "FTV/anatomic-volume mismatch is treated as a QC warning, not as biology.",
        )
    ]


def _diameter_only_effect() -> _RuleEffect:
    return _effect(
        "diameter_only_report_only_v1",
        "measurement_mode",
        "longest_diameter_cm is present without MRI volume features",
        {"update_mode": "report_only"},
        "Diameter-only input is retained for reporting but not used as a volumetric MRI update.",
    )


def _effect(
    rule_id: str,
    rule_family: str,
    condition: str,
    effects: Dict[str, object],
    explanation: str,
) -> _RuleEffect:
    return _RuleEffect(
        contribution=Layer4RuleContribution(
            rule_id=rule_id,
            rule_family=rule_family,
            condition=condition,
            effects=dict(effects),
            evidence_level="data_quality_policy",
            explanation=explanation,
            uncertainty_driver=True,
        ),
        variance_multipliers=_variance_multipliers(effects),
    )


def _variance_multipliers(effects: Mapping[str, object]) -> Dict[str, float]:
    names = {
        "growth_variance_multiplier": _GROWTH_Z,
        "sensitivity_variance_multiplier": _SENSITIVITY_Z,
        "resistant_variance_multiplier": _RESISTANCE_Z,
    }
    return {
        transformed_name: float(effects[effect_name])
        for effect_name, transformed_name in names.items()
        if effect_name in effects
    }


def _scale_covariance(
    covariance: Sequence[Sequence[float]],
    parameter_names: Sequence[str],
    variance_multipliers: Mapping[str, float],
) -> List[List[float]]:
    scales = [math.sqrt(variance_multipliers[name]) for name in parameter_names]
    return [
        [
            float(value) * scales[row_index] * scales[column_index]
            for column_index, value in enumerate(row)
        ]
        for row_index, row in enumerate(covariance)
    ]


def _resolve_observation_noise(context: Mapping[str, object]):
    measurement = dict(context)
    if _is_diameter_only(context):
        measurement.setdefault("source", "diameter-derived volume")
        measurement.setdefault("derived_from_diameter", True)
    elif _has_any_field(context, _ANATOMIC_VOLUME_FIELDS + _FUNCTIONAL_VOLUME_FIELDS):
        measurement.setdefault("source", "DCE-MRI segmentation volume")

    segmentation_qc = _first_present(context, _SEGMENTATION_QC_FIELDS)
    if "segmentation_confidence" not in measurement:
        numeric_qc = _unit_interval_number(segmentation_qc)
        if numeric_qc is not None:
            measurement["segmentation_confidence"] = numeric_qc
        elif segmentation_qc is not None:
            measurement.setdefault("qc", str(segmentation_qc))
    return resolve_observation_noise(measurement)


def _is_diameter_only(context: Mapping[str, object]) -> bool:
    return _has_any_field(context, _DIAMETER_FIELDS) and not _has_any_field(
        context,
        _ANATOMIC_VOLUME_FIELDS + _FUNCTIONAL_VOLUME_FIELDS,
    )


def _has_any_field(context: Mapping[str, object], fields: Tuple[str, ...]) -> bool:
    return any(field in context and context[field] is not None for field in fields)


def _first_present(context: Mapping[str, object], fields: Tuple[str, ...]) -> object | None:
    for field in fields:
        if field in context:
            return context[field]
    return None


def _qc_level(value: object) -> str | None:
    numeric = _unit_interval_number(value)
    if numeric is not None:
        if numeric >= 0.80:
            return "high"
        if numeric < 0.50:
            return "low"
        return "medium"
    if value is None:
        return None
    text = _normalize_text(str(value))
    if any(token in text for token in _LOW_QC_TOKENS):
        return "low"
    if any(token in text for token in _HIGH_QC_TOKENS):
        return "high"
    return None


def _optional_fraction(value: object, name: str) -> float | None:
    numeric = _optional_number(value, name)
    if numeric is None:
        return None
    if numeric < 0 or numeric > 1:
        raise ValueError(f"{name} must be in [0, 1]")
    return numeric


def _optional_number(value: object, name: str) -> float | None:
    if value is None or _normalize_text(str(value)) in {
        "",
        "unknown",
        "unspecified",
        "not specified",
        "not assessed",
        "n/a",
        "na",
        "none",
        "missing",
    }:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _unit_interval_number(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if 0 <= numeric <= 1:
        return numeric
    return None


def _normalize_text(value: str) -> str:
    return " ".join(value.lower().replace("_", " ").replace("-", " ").split())


def _require_expected_prior_order(
    prior: PopulationPrior | PathologyBiomarkerPrior,
) -> None:
    if tuple(prior.transformed_parameter_names) != TRANSFORMED_PARAMETER_NAMES:
        raise ValueError(
            "Layer 4 MRI feature rules require the V1-A transformed parameter order."
        )
    missing = sorted(set(TRANSFORMED_PARAMETER_NAMES) - set(prior.transformed_means))
    if missing:
        raise KeyError("Layer 4 prior is missing transformed means: " + ", ".join(missing))
