"""Layer 3 pathology and biomarker rules for V1 prior construction.

Layer 3 applies deliberately modest, traceable shifts to the Layer 2 TNBC
chemotherapy population prior. Rules operate in transformed parameter space so
multiplicative biologic assumptions remain explicit: growth and treatment
sensitivity multipliers become log-space offsets, while resistant-fraction odds
multipliers become logit-space offsets. Missing biomarkers widen uncertainty;
they never act like negative biomarker evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Dict, List, Mapping, Sequence, Tuple

from .population_prior import PopulationPrior, PopulationPriorSampleResult
from .transforms import (
    TRANSFORMED_PARAMETER_NAMES,
    from_transformed,
    sample_correlated_transformed_prior,
    validate_covariance,
)


PATHOLOGY_BIOMARKER_LAYER_VERSION = "oncotwin_prior_v1_layer3"

_GROWTH_Z = "log_growth_rate_per_day"
_SENSITIVITY_Z = "log_active_treatment_sensitivity"
_RESISTANCE_Z = "logit_resistant_fraction"

_KI67_FIELDS = (
    "ki67_percent",
    "ki67",
    "ki_67",
    "ki67_index",
    "proliferation_index",
)
_GRADE_FIELDS = (
    "grade",
    "tumor_grade",
    "nottingham_grade",
    "histologic_grade",
)
_ER_FIELDS = ("er_status", "estrogen_receptor_status", "er")
_PR_FIELDS = ("pr_status", "progesterone_receptor_status", "pr")
_HER2_FIELDS = ("her2_status", "her2", "erbb2_status")
_BRCA_HRD_FIELDS = (
    "brca_status",
    "brca1_status",
    "brca2_status",
    "hrd_status",
    "homologous_recombination_deficiency",
    "hrd_positive",
)
_UNKNOWN_TOKENS = {
    "",
    "unknown",
    "unspecified",
    "not specified",
    "not assessed",
    "not tested",
    "pending",
    "n/a",
    "na",
    "none",
    "missing",
}
_POSITIVE_TOKENS = (
    "positive",
    "pathogenic",
    "mutated",
    "mutation",
    "variant detected",
    "deficient",
    "hrd+",
    "+",
)
_NEGATIVE_TOKENS = (
    "negative",
    "wild type",
    "wild-type",
    "wildtype",
    "proficient",
    "no mutation",
    "not detected",
    "hrd-",
    "-",
)


@dataclass(frozen=True)
class Layer3RuleContribution:
    """One applied pathology/biomarker rule and its transparent effect."""

    rule_id: str
    rule_family: str
    condition: str
    effects: Dict[str, float]
    evidence_level: str
    explanation: str
    uncertainty_driver: bool = False

    def as_dict(self) -> Dict[str, object]:
        """Return a JSON-friendly contribution record."""

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
class PathologyBiomarkerPrior:
    """Layer 3 prior after pathology and biomarker rules are applied."""

    prior_version: str
    base_prior_version: str
    population_group: str
    contract_id: str
    transformed_parameter_names: Tuple[str, ...]
    transformed_means: Dict[str, float]
    transformed_covariance: Tuple[Tuple[float, ...], ...]
    layer_contributions: Tuple[Layer3RuleContribution, ...]
    warnings: Tuple[str, ...] = ()
    uncertainty_drivers: Tuple[str, ...] = ()

    def layer_contribution(self) -> Dict[str, object]:
        """Return the traceable Layer 3 contribution to prior composition."""

        return {
            "layer": "pathology_biomarker_rules",
            "prior_version": self.prior_version,
            "base_prior_version": self.base_prior_version,
            "contract_id": self.contract_id,
            "rules": [rule.as_dict() for rule in self.layer_contributions],
            "transformed_parameter_names": list(self.transformed_parameter_names),
            "transformed_means": dict(self.transformed_means),
            "transformed_covariance": [list(row) for row in self.transformed_covariance],
            "warnings": list(self.warnings),
            "uncertainty_drivers": list(self.uncertainty_drivers),
        }


@dataclass(frozen=True)
class _Layer3RuleEffect:
    contribution: Layer3RuleContribution
    mean_multipliers: Dict[str, float]
    variance_multipliers: Dict[str, float]


def apply_pathology_biomarker_rules(
    prior: PopulationPrior,
    context: Mapping[str, object],
) -> PathologyBiomarkerPrior:
    """Apply V1 Layer 3 pathology/biomarker rules to a Layer 2 prior."""

    _require_expected_prior_order(prior)
    rule_effects = _resolve_rule_effects(context)

    transformed_means = dict(prior.transformed_means)
    variance_multipliers = {name: 1.0 for name in prior.transformed_parameter_names}
    warnings: List[str] = []
    uncertainty_drivers: List[str] = []

    for effect in rule_effects:
        for transformed_name, multiplier in effect.mean_multipliers.items():
            transformed_means[transformed_name] += _log_multiplier(multiplier)
        for transformed_name, multiplier in effect.variance_multipliers.items():
            if multiplier <= 0 or not math.isfinite(multiplier):
                raise ValueError("variance multipliers must be positive and finite")
            variance_multipliers[transformed_name] *= multiplier
        if effect.contribution.uncertainty_driver:
            uncertainty_drivers.append(effect.contribution.rule_id)

    transformed_covariance = _scale_covariance(
        prior.transformed_covariance,
        prior.transformed_parameter_names,
        variance_multipliers,
    )
    validate_covariance(
        transformed_covariance,
        dimension=len(prior.transformed_parameter_names),
    )

    if any(
        effect.contribution.rule_id == "tnbc_receptor_inconsistency_v1"
        for effect in rule_effects
    ):
        warnings.append(
            "Receptor markers are not fully consistent with TNBC; Layer 3 widened "
            "prior uncertainty rather than treating the context as certain."
        )

    return PathologyBiomarkerPrior(
        prior_version=PATHOLOGY_BIOMARKER_LAYER_VERSION,
        base_prior_version=prior.prior_version,
        population_group=prior.population_group,
        contract_id=prior.contract_id,
        transformed_parameter_names=prior.transformed_parameter_names,
        transformed_means=transformed_means,
        transformed_covariance=tuple(tuple(row) for row in transformed_covariance),
        layer_contributions=tuple(effect.contribution for effect in rule_effects),
        warnings=tuple(warnings),
        uncertainty_drivers=tuple(dict.fromkeys(uncertainty_drivers)),
    )


def sample_pathology_biomarker_prior(
    prior: PathologyBiomarkerPrior,
    *,
    n_samples: int,
    seed: int | None = None,
) -> PopulationPriorSampleResult:
    """Sample Layer 3 particles in simulator-facing parameter space."""

    transformed_samples = sample_correlated_transformed_prior(
        prior.transformed_means,
        prior.transformed_covariance,
        n_samples=n_samples,
        seed=seed,
    )
    return PopulationPriorSampleResult(
        samples=[from_transformed(sample) for sample in transformed_samples]
    )


def _resolve_rule_effects(context: Mapping[str, object]) -> List[_Layer3RuleEffect]:
    effects: List[_Layer3RuleEffect] = []
    effects.extend(_ki67_rule_effects(context))
    effects.extend(_grade_rule_effects(context))
    effects.extend(_receptor_consistency_rule_effects(context))
    effects.extend(_brca_hrd_rule_effects(context))
    return effects


def _ki67_rule_effects(context: Mapping[str, object]) -> List[_Layer3RuleEffect]:
    value = _extract_percent(context, _KI67_FIELDS, "Ki-67")
    if value is None:
        return [
            _effect(
                rule_id="ki67_missing_v1",
                rule_family="ki67",
                condition="ki67_percent is missing or unknown",
                effects={"growth_variance_multiplier": 1.25},
                evidence_level="missingness_policy",
                explanation=(
                    "Missing Ki-67 widens proliferation uncertainty without "
                    "inventing a high or low proliferation status."
                ),
                uncertainty_driver=True,
            )
        ]

    if value <= 5.0:
        return [
            _effect(
                rule_id="ki67_low_v1",
                rule_family="ki67",
                condition="ki67_percent <= 5",
                effects={
                    "growth_multiplier": 0.70,
                    "growth_variance_multiplier": 1.05,
                },
                evidence_level="B",
                explanation=(
                    "Very low Ki-67 shifts proliferation assumptions downward "
                    "while keeping uncertainty slightly wider."
                ),
            )
        ]

    if value >= 30.0:
        return [
            _effect(
                rule_id="ki67_high_v1",
                rule_family="ki67",
                condition="ki67_percent >= 30",
                effects={
                    "growth_multiplier": 1.50,
                    "chemo_sensitivity_multiplier": 1.10,
                    "resistant_odds_multiplier": 0.90,
                    "growth_variance_multiplier": 1.10,
                },
                evidence_level="A/B",
                explanation=(
                    "High Ki-67 shifts proliferation upward with a modest "
                    "chemotherapy-response effect."
                ),
            )
        ]

    return []


def _grade_rule_effects(context: Mapping[str, object]) -> List[_Layer3RuleEffect]:
    grade = _extract_grade(context)
    if grade is None:
        return []
    if grade >= 3:
        return [
            _effect(
                rule_id="grade3_high_growth_v1",
                rule_family="grade",
                condition="grade >= 3",
                effects={
                    "growth_multiplier": 1.15,
                    "growth_variance_multiplier": 1.05,
                },
                evidence_level="B",
                explanation=(
                    "High histologic grade modestly shifts proliferation "
                    "assumptions upward."
                ),
            )
        ]
    if grade <= 1:
        return [
            _effect(
                rule_id="grade1_lower_growth_v1",
                rule_family="grade",
                condition="grade <= 1",
                effects={"growth_multiplier": 0.90},
                evidence_level="C",
                explanation=(
                    "Low histologic grade modestly shifts proliferation "
                    "assumptions downward."
                ),
            )
        ]
    return []


def _receptor_consistency_rule_effects(
    context: Mapping[str, object],
) -> List[_Layer3RuleEffect]:
    statuses = {
        "ER": _extract_status(context, _ER_FIELDS),
        "PR": _extract_status(context, _PR_FIELDS),
        "HER2": _extract_status(context, _HER2_FIELDS),
    }
    positive_markers = tuple(
        marker for marker, status in statuses.items() if status == "positive"
    )
    if not positive_markers:
        return []

    return [
        _effect(
            rule_id="tnbc_receptor_inconsistency_v1",
            rule_family="receptor_consistency",
            condition="TNBC context has positive receptor marker(s): "
            + ", ".join(positive_markers),
            effects={
                "growth_variance_multiplier": 1.10,
                "sensitivity_variance_multiplier": 1.15,
                "resistant_variance_multiplier": 1.15,
            },
            evidence_level="data_quality_policy",
            explanation=(
                "Receptor markers conflict with a strict TNBC interpretation, "
                "so the prior widens uncertainty instead of making a strong "
                "biology claim."
            ),
            uncertainty_driver=True,
        )
    ]


def _brca_hrd_rule_effects(context: Mapping[str, object]) -> List[_Layer3RuleEffect]:
    statuses = [
        _extract_status(context, (field,))
        for field in _BRCA_HRD_FIELDS
        if field in context
    ]
    known_statuses = [status for status in statuses if status is not None]
    if any(status == "positive" for status in known_statuses):
        return [
            _effect(
                rule_id="brca_hrd_positive_v1",
                rule_family="brca_hrd",
                condition="BRCA/HRD positive or pathogenic/deficient status present",
                effects={
                    "chemo_sensitivity_multiplier": 1.15,
                    "resistant_odds_multiplier": 0.85,
                    "sensitivity_variance_multiplier": 1.05,
                },
                evidence_level="B",
                explanation=(
                    "BRCA/HRD positivity modestly shifts chemotherapy "
                    "sensitivity upward and resistant odds downward."
                ),
            )
        ]

    if not known_statuses or any(status == "unknown" for status in known_statuses):
        return [
            _effect(
                rule_id="brca_hrd_missing_v1",
                rule_family="brca_hrd",
                condition="BRCA/HRD status is missing, pending, or unknown",
                effects={
                    "sensitivity_variance_multiplier": 1.10,
                    "resistant_variance_multiplier": 1.15,
                },
                evidence_level="missingness_policy",
                explanation=(
                    "Unknown BRCA/HRD status widens response and resistance "
                    "uncertainty rather than behaving like negative evidence."
                ),
                uncertainty_driver=True,
            )
        ]

    return []


def _effect(
    *,
    rule_id: str,
    rule_family: str,
    condition: str,
    effects: Dict[str, float],
    evidence_level: str,
    explanation: str,
    uncertainty_driver: bool = False,
) -> _Layer3RuleEffect:
    return _Layer3RuleEffect(
        contribution=Layer3RuleContribution(
            rule_id=rule_id,
            rule_family=rule_family,
            condition=condition,
            effects=dict(effects),
            evidence_level=evidence_level,
            explanation=explanation,
            uncertainty_driver=uncertainty_driver,
        ),
        mean_multipliers=_mean_multipliers(effects),
        variance_multipliers=_variance_multipliers(effects),
    )


def _mean_multipliers(effects: Mapping[str, float]) -> Dict[str, float]:
    result = {}
    for effect_name, transformed_name in (
        ("growth_multiplier", _GROWTH_Z),
        ("chemo_sensitivity_multiplier", _SENSITIVITY_Z),
        ("resistant_odds_multiplier", _RESISTANCE_Z),
    ):
        if effect_name in effects:
            result[transformed_name] = effects[effect_name]
    return result


def _variance_multipliers(effects: Mapping[str, float]) -> Dict[str, float]:
    result = {}
    for effect_name, transformed_name in (
        ("growth_variance_multiplier", _GROWTH_Z),
        ("sensitivity_variance_multiplier", _SENSITIVITY_Z),
        ("resistant_variance_multiplier", _RESISTANCE_Z),
    ):
        if effect_name in effects:
            result[transformed_name] = effects[effect_name]
    return result


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


def _extract_percent(
    context: Mapping[str, object],
    fields: Tuple[str, ...],
    display_name: str,
) -> float | None:
    value = _first_present(context, fields)
    if value is None or _is_unknown(value):
        return None
    numeric = _parse_float(value, display_name)
    if numeric < 0.0 or numeric > 100.0:
        raise ValueError(f"{display_name} percent must be in [0, 100]")
    return numeric


def _extract_grade(context: Mapping[str, object]) -> int | None:
    value = _first_present(context, _GRADE_FIELDS)
    if value is None or _is_unknown(value):
        return None
    if isinstance(value, bool):
        raise ValueError("grade must not be boolean")
    if isinstance(value, (int, float)):
        grade = int(value)
        if float(value) != float(grade):
            raise ValueError("grade must be 1, 2, or 3")
        return _validate_grade(grade)

    text = _normalize_text(str(value))
    roman_matches = {
        "i": 1,
        "ii": 2,
        "iii": 3,
    }
    if text in roman_matches:
        return roman_matches[text]
    match = re.search(r"\b([123])\b", text)
    if not match:
        raise ValueError("grade must be 1, 2, or 3")
    return _validate_grade(int(match.group(1)))


def _validate_grade(value: int) -> int:
    if value not in (1, 2, 3):
        raise ValueError("grade must be 1, 2, or 3")
    return value


def _extract_status(
    context: Mapping[str, object],
    fields: Tuple[str, ...],
) -> str | None:
    value = _first_present(context, fields)
    if value is None:
        return None
    if isinstance(value, bool):
        return "positive" if value else "negative"
    text = _normalize_text(str(value))
    if text in _UNKNOWN_TOKENS:
        return "unknown"
    if _contains_any(text, _NEGATIVE_TOKENS):
        return "negative"
    if _contains_any(text, _POSITIVE_TOKENS):
        return "positive"
    return "unknown"


def _first_present(
    context: Mapping[str, object],
    fields: Tuple[str, ...],
) -> object | None:
    for field in fields:
        if field in context:
            return context[field]
    return None


def _parse_float(value: object, display_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{display_name} must be numeric")
    if isinstance(value, (int, float)):
        numeric = float(value)
    else:
        text = _normalize_text(str(value)).replace("%", "")
        try:
            numeric = float(text)
        except ValueError as exc:
            raise ValueError(f"{display_name} must be numeric") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{display_name} must be finite")
    return numeric


def _log_multiplier(multiplier: float) -> float:
    if multiplier <= 0 or not math.isfinite(multiplier):
        raise ValueError("mean multipliers must be positive and finite")
    return math.log(multiplier)


def _contains_any(value: str, tokens: Tuple[str, ...]) -> bool:
    return any(token in value for token in tokens)


def _is_unknown(value: object) -> bool:
    if value is None:
        return True
    return _normalize_text(str(value)) in _UNKNOWN_TOKENS


def _normalize_text(value: str) -> str:
    return " ".join(value.lower().replace("_", " ").split())


def _require_expected_prior_order(prior: PopulationPrior) -> None:
    if tuple(prior.transformed_parameter_names) != TRANSFORMED_PARAMETER_NAMES:
        raise ValueError(
            "Layer 3 pathology/biomarker rules require the V1-A transformed "
            "parameter order."
        )
    missing_means = sorted(
        set(TRANSFORMED_PARAMETER_NAMES) - set(prior.transformed_means)
    )
    if missing_means:
        raise KeyError(
            "Layer 3 prior is missing transformed means: " + ", ".join(missing_means)
        )
