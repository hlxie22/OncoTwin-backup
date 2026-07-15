"""Structured explanations for V1 posterior and scenario runtime artifacts.

The explanation layer is deterministic and structured-first. It summarizes model
artifacts for UI/API surfaces while preserving uncertainty, scope limits, and
decision-support guardrails.
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any, Iterable, Mapping, Sequence


EXPLANATION_RUNTIME_VERSION = "oncotwin_explanation_v1"
SUPPORTED_EXPLANATION_AUDIENCES = ("clinician", "patient")
SAFETY_AND_SCOPE_NOTE = (
    "This explanation is research decision support, not a diagnosis or treatment "
    "recommendation. Clinical decisions require clinician review, source-data "
    "verification, and patient-specific context."
)


def build_twin_update_explanation(
    *,
    posterior_update: Mapping[str, Any],
    scenario_lab: Mapping[str, Any] | None = None,
    prior_context: Mapping[str, Any] | None = None,
    audience: str = "clinician",
    max_key_factors: int = 6,
    max_uncertainty_drivers: int = 6,
) -> dict[str, object]:
    """Build a deterministic explanation from V1 runtime artifacts."""

    if audience not in SUPPORTED_EXPLANATION_AUDIENCES:
        raise ValueError(
            "audience must be one of: " + ", ".join(SUPPORTED_EXPLANATION_AUDIENCES)
        )
    if max_key_factors < 1:
        raise ValueError("max_key_factors must be at least 1")
    if max_uncertainty_drivers < 1:
        raise ValueError("max_uncertainty_drivers must be at least 1")

    posterior_summary = _posterior_summary(posterior_update, audience=audience)
    scenario_summary = (
        _scenario_summary(scenario_lab, audience=audience)
        if scenario_lab is not None
        else None
    )
    prior_summary = _prior_summary(prior_context) if prior_context is not None else None

    key_factors = _key_factors(
        posterior_update=posterior_update,
        posterior_summary=posterior_summary,
        scenario_summary=scenario_summary,
        prior_summary=prior_summary,
        audience=audience,
    )[:max_key_factors]
    uncertainty_drivers = _uncertainty_drivers(
        posterior_update=posterior_update,
        scenario_lab=scenario_lab,
    )[:max_uncertainty_drivers]

    summary = _overall_summary(
        posterior_summary=posterior_summary,
        scenario_summary=scenario_summary,
        audience=audience,
    )

    sections = [
        {
            "section_id": "posterior_update",
            "title": "Posterior update" if audience == "clinician" else "What changed after the new measurements",
            "body": posterior_summary["text"],
        }
    ]
    if scenario_summary is not None:
        sections.append(
            {
                "section_id": "scenario_comparison",
                "title": "Scenario comparison" if audience == "clinician" else "Modeled what-if comparisons",
                "body": scenario_summary["text"],
            }
        )
    if prior_summary is not None:
        sections.append(
            {
                "section_id": "prior_context",
                "title": "Prior context" if audience == "clinician" else "Starting assumptions",
                "body": prior_summary["text"],
            }
        )
    sections.append(
        {
            "section_id": "safety_and_scope",
            "title": "Safety and scope",
            "body": SAFETY_AND_SCOPE_NOTE,
        }
    )

    return {
        "explanation_runtime_version": EXPLANATION_RUNTIME_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "audience": audience,
        "summary": summary,
        "key_factors": key_factors,
        "uncertainty_drivers": uncertainty_drivers,
        "posterior_update_explanation": posterior_summary,
        "scenario_comparison_explanation": scenario_summary,
        "prior_context_explanation": prior_summary,
        "sections": sections,
        "safety_and_scope_note": SAFETY_AND_SCOPE_NOTE,
        "not_a_treatment_recommendation": True,
        "source_versions": {
            "posterior_runtime_version": posterior_update.get("posterior_runtime_version"),
            "scenario_lab_version": scenario_lab.get("scenario_lab_version") if scenario_lab else None,
        },
    }


def render_markdown_explanation(explanation: Mapping[str, Any]) -> str:
    """Render a structured explanation as simple Markdown for review."""

    lines = [
        f"# OncoTwin explanation ({explanation.get('audience', 'unknown')})",
        "",
        str(explanation.get("summary", "")),
        "",
        "## Key factors",
    ]
    key_factors = explanation.get("key_factors", [])
    if isinstance(key_factors, Sequence) and not isinstance(key_factors, (str, bytes)):
        for factor in key_factors:
            if isinstance(factor, Mapping):
                lines.append(
                    f"- **{factor.get('label', factor.get('factor_id', 'factor'))}:** "
                    f"{factor.get('description', '')}"
                )
    lines.extend(["", "## Uncertainty drivers"])
    uncertainty_drivers = explanation.get("uncertainty_drivers", [])
    if isinstance(uncertainty_drivers, Sequence) and not isinstance(uncertainty_drivers, (str, bytes)):
        for driver in uncertainty_drivers:
            if isinstance(driver, Mapping):
                lines.append(
                    f"- **{driver.get('driver_id', 'uncertainty')}:** "
                    f"{driver.get('description', '')}"
                )
            else:
                lines.append(f"- {driver}")
    lines.append("")
    sections = explanation.get("sections", [])
    if isinstance(sections, Sequence) and not isinstance(sections, (str, bytes)):
        for section in sections:
            if not isinstance(section, Mapping):
                continue
            lines.extend(
                [
                    f"## {section.get('title', section.get('section_id', 'Section'))}",
                    str(section.get("body", "")),
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def _posterior_summary(
    posterior_update: Mapping[str, Any],
    *,
    audience: str,
) -> dict[str, object]:
    trajectory = _required_mapping(
        posterior_update.get("posterior_trajectory_summary"),
        "posterior_update.posterior_trajectory_summary",
    )
    times = _required_number_list(trajectory.get("times"), "posterior_trajectory_summary.times")
    medians = _required_number_list(
        trajectory.get("median_volume_ml"),
        "posterior_trajectory_summary.median_volume_ml",
    )
    if len(times) != len(medians):
        raise ValueError("posterior trajectory times and medians must have matching lengths")

    lower80 = _optional_number_list(trajectory.get("lower80_volume_ml"))
    upper80 = _optional_number_list(trajectory.get("upper80_volume_ml"))
    final_day = times[-1]
    final_median = medians[-1]
    final_interval = None
    if lower80 and upper80 and len(lower80) == len(times) and len(upper80) == len(times):
        final_interval = {"lower80_volume_ml": lower80[-1], "upper80_volume_ml": upper80[-1]}

    ess_fraction = _optional_finite(posterior_update.get("effective_sample_size_fraction"))
    fallback_status = str(posterior_update.get("fallback_status", "unknown"))
    observation_count = int(_optional_finite(posterior_update.get("n_observations")) or 0)
    posterior_text = str(posterior_update.get("update_explanation") or "")

    if audience == "patient":
        text = (
            f"After using {observation_count} tumor measurement(s), the model's middle "
            f"estimate for day {final_day:g} is {final_median:.3g} mL."
        )
        if final_interval is not None:
            text += (
                f" Most modeled outcomes fall between "
                f"{final_interval['lower80_volume_ml']:.3g} and "
                f"{final_interval['upper80_volume_ml']:.3g} mL."
            )
    else:
        text = (
            f"The posterior update used {observation_count} tumor-volume observation(s). "
            f"The posterior median volume at day {final_day:g} is {final_median:.3g} mL."
        )
        if final_interval is not None:
            text += (
                f" The posterior 80% interval is "
                f"{final_interval['lower80_volume_ml']:.3g}-"
                f"{final_interval['upper80_volume_ml']:.3g} mL."
            )
    if posterior_text:
        text += " " + posterior_text
    if fallback_status != "not_needed":
        text += (
            " The posterior update reported a fallback warning, so uncertainty should "
            "remain prominent."
        )

    return {
        "final_day": final_day,
        "final_median_volume_ml": final_median,
        "final_interval": final_interval,
        "observation_count": observation_count,
        "effective_sample_size_fraction": ess_fraction,
        "fallback_status": fallback_status,
        "text": text,
    }


def _scenario_summary(
    scenario_lab: Mapping[str, Any],
    *,
    audience: str,
) -> dict[str, object]:
    scenarios = _required_sequence(scenario_lab.get("scenarios"), "scenario_lab.scenarios")
    comparison = _required_mapping(
        scenario_lab.get("comparison_summary"),
        "scenario_lab.comparison_summary",
    )
    top_id = comparison.get("top_scenario_id")
    reference_id = comparison.get("reference_scenario_id")
    ok_scenarios = [
        scenario for scenario in scenarios
        if isinstance(scenario, Mapping) and scenario.get("status") == "ok"
    ]

    top_scenario = _find_scenario(ok_scenarios, top_id)
    reference_scenario = _find_scenario(ok_scenarios, reference_id)
    if top_scenario is None and ok_scenarios:
        top_scenario = ok_scenarios[0]

    top_payload = _scenario_brief(top_scenario) if top_scenario else None
    reference_payload = _scenario_brief(reference_scenario) if reference_scenario else None

    failed_count = sum(
        1 for scenario in scenarios
        if isinstance(scenario, Mapping) and scenario.get("status") != "ok"
    )
    if top_payload is None:
        text = "No scenario produced a modeled trajectory."
    elif audience == "patient":
        text = (
            f"Among the modeled what-if scenarios, {top_payload['label']} had the most "
            "favorable modeled low-residual-burden probability."
        )
    else:
        text = (
            f"The highest-ranked modeled scenario is {top_payload['scenario_id']} "
            f"({top_payload['label']}) with low-residual-burden probability "
            f"{top_payload['probability_low_residual_burden']:.1%} and final median "
            f"{top_payload['final_median_volume_ml']:.3g} mL."
        )

    if reference_payload is not None and top_payload is not None:
        delta = top_payload["final_median_volume_ml"] - reference_payload["final_median_volume_ml"]
        text += (
            f" Compared with reference scenario {reference_payload['scenario_id']}, "
            f"the modeled final median volume difference is {delta:.3g} mL."
        )
    if failed_count:
        text += f" {failed_count} scenario(s) failed preflight/runtime checks and were not ranked."
    text += " Scenario rankings are modeled summaries, not treatment recommendations."

    return {
        "top_scenario": top_payload,
        "reference_scenario": reference_payload,
        "failed_scenario_count": failed_count,
        "text": text,
    }


def _prior_summary(prior_context: Mapping[str, Any]) -> dict[str, object]:
    layers = []
    for field in ("layer_contributions", "prior_layers", "rules"):
        value = prior_context.get(field)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            layers.extend(item for item in value if isinstance(item, Mapping))

    labels = []
    for layer in layers:
        label = (
            layer.get("rule_id")
            or layer.get("layer")
            or layer.get("rule_family")
            or layer.get("prior_version")
        )
        if label:
            labels.append(str(label))
    labels = list(_stable_unique(labels))

    if labels:
        text = "The prior context included these modeled evidence/rule contributions: " + ", ".join(labels[:8]) + "."
    else:
        text = "Prior-layer context was provided, but no structured contribution labels were found."

    return {
        "contribution_labels": labels,
        "text": text,
    }


def _key_factors(
    *,
    posterior_update: Mapping[str, Any],
    posterior_summary: Mapping[str, object],
    scenario_summary: Mapping[str, object] | None,
    prior_summary: Mapping[str, object] | None,
    audience: str,
) -> list[dict[str, object]]:
    factors: list[dict[str, object]] = []
    obs_count = posterior_summary.get("observation_count", 0)
    factors.append(
        {
            "factor_id": "tumor_volume_observations",
            "label": "Tumor-volume observations",
            "description": (
                f"The update used {obs_count} tumor-volume observation(s) to reweight "
                "posterior particles."
            ),
            "direction": "posterior_update_input",
        }
    )

    ess = posterior_summary.get("effective_sample_size_fraction")
    if isinstance(ess, (float, int)):
        factors.append(
            {
                "factor_id": "effective_sample_size",
                "label": "Particle support",
                "description": (
                    f"The posterior effective sample size fraction is {float(ess):.1%}. "
                    "Lower values mean fewer particles explain the observations well."
                ),
                "direction": "uncertainty" if float(ess) < 0.25 else "support",
            }
        )

    final_day = posterior_summary["final_day"]
    final_median = posterior_summary["final_median_volume_ml"]
    factors.append(
        {
            "factor_id": "posterior_final_volume",
            "label": "Posterior final volume",
            "description": (
                f"The modeled posterior median tumor volume at day {float(final_day):g} "
                f"is {float(final_median):.3g} mL."
            ),
            "direction": "trajectory_summary",
        }
    )

    fallback = posterior_summary.get("fallback_status")
    if fallback and fallback != "not_needed":
        factors.append(
            {
                "factor_id": "fallback_status",
                "label": "Fallback warning",
                "description": (
                    f"The posterior runtime reported fallback_status={fallback}; "
                    "interpret the update as a high-uncertainty diagnostic."
                ),
                "direction": "caution",
            }
        )

    if scenario_summary and scenario_summary.get("top_scenario"):
        top = scenario_summary["top_scenario"]
        if isinstance(top, Mapping):
            factors.append(
                {
                    "factor_id": "top_modeled_scenario",
                    "label": "Top modeled scenario",
                    "description": (
                        f"{top['label']} had the highest modeled low-residual-burden "
                        f"probability among simulated scenarios."
                    ),
                    "direction": "modeled_comparison",
                }
            )

    if prior_summary and prior_summary.get("contribution_labels"):
        labels = prior_summary["contribution_labels"]
        if isinstance(labels, Sequence) and not isinstance(labels, (str, bytes)):
            factors.append(
                {
                    "factor_id": "prior_layer_context",
                    "label": "Prior-layer context",
                    "description": (
                        "The explanation includes prior-layer context from "
                        + ", ".join(str(label) for label in labels[:4])
                        + "."
                    ),
                    "direction": "prior_context",
                }
            )

    if audience == "patient":
        for factor in factors:
            factor["description"] = str(factor["description"]).replace(
                "posterior particles",
                "modeled possibilities",
            )
    return factors


def _uncertainty_drivers(
    *,
    posterior_update: Mapping[str, Any],
    scenario_lab: Mapping[str, Any] | None,
) -> list[dict[str, object]]:
    drivers: list[str] = []

    posterior_uncertainty = posterior_update.get("uncertainty_summary")
    if isinstance(posterior_uncertainty, Mapping):
        top_drivers = posterior_uncertainty.get("top_drivers")
        if isinstance(top_drivers, Sequence) and not isinstance(top_drivers, (str, bytes)):
            drivers.extend(str(item) for item in top_drivers)

    posterior_warnings = posterior_update.get("warnings")
    if isinstance(posterior_warnings, Sequence) and not isinstance(posterior_warnings, (str, bytes)):
        drivers.extend(str(item) for item in posterior_warnings)

    if scenario_lab is not None:
        scenario_warnings = scenario_lab.get("warnings")
        if isinstance(scenario_warnings, Sequence) and not isinstance(scenario_warnings, (str, bytes)):
            drivers.extend(str(item) for item in scenario_warnings)
        scenarios = scenario_lab.get("scenarios")
        if isinstance(scenarios, Sequence) and not isinstance(scenarios, (str, bytes)):
            for scenario in scenarios:
                if not isinstance(scenario, Mapping):
                    continue
                if scenario.get("status") != "ok":
                    drivers.append(
                        f"scenario {scenario.get('scenario_id', 'unknown')} was not simulated successfully"
                    )

    output = []
    for index, driver in enumerate(_stable_unique(drivers), start=1):
        output.append(
            {
                "driver_id": f"uncertainty_{index}",
                "description": driver,
            }
        )
    if not output:
        output.append(
            {
                "driver_id": "uncertainty_1",
                "description": "No additional uncertainty drivers were reported by the runtime artifacts.",
            }
        )
    return output


def _overall_summary(
    *,
    posterior_summary: Mapping[str, object],
    scenario_summary: Mapping[str, object] | None,
    audience: str,
) -> str:
    final_day = float(posterior_summary["final_day"])
    final_median = float(posterior_summary["final_median_volume_ml"])
    if audience == "patient":
        summary = (
            f"The model updated its estimate using the available tumor measurements. "
            f"The middle modeled estimate at day {final_day:g} is {final_median:.3g} mL."
        )
    else:
        summary = (
            f"The V1 twin posterior update estimates a median tumor volume of "
            f"{final_median:.3g} mL at day {final_day:g}."
        )

    if scenario_summary and scenario_summary.get("top_scenario"):
        top = scenario_summary["top_scenario"]
        if isinstance(top, Mapping):
            summary += (
                f" Among simulated scenarios, {top['label']} had the highest modeled "
                "low-residual-burden probability."
            )
    summary += " This is not a treatment recommendation."
    return summary


def _scenario_brief(scenario: Mapping[str, Any] | None) -> dict[str, object] | None:
    if scenario is None:
        return None
    trajectory = _required_mapping(
        scenario.get("trajectory_summary"),
        f"scenario {scenario.get('scenario_id', 'unknown')}.trajectory_summary",
    )
    probabilities = _required_mapping(
        scenario.get("probabilities"),
        f"scenario {scenario.get('scenario_id', 'unknown')}.probabilities",
    )
    medians = _required_number_list(trajectory.get("median_volume_ml"), "scenario median_volume_ml")
    return {
        "scenario_id": str(scenario.get("scenario_id", "unknown")),
        "label": str(scenario.get("label") or scenario.get("scenario_id", "unknown")),
        "final_median_volume_ml": medians[-1],
        "probability_low_residual_burden": _required_finite(
            probabilities.get("probability_low_residual_burden"),
            "probability_low_residual_burden",
        ),
        "probability_progression": _required_finite(
            probabilities.get("probability_progression"),
            "probability_progression",
        ),
    }


def _find_scenario(
    scenarios: Sequence[Mapping[str, Any]],
    scenario_id: object,
) -> Mapping[str, Any] | None:
    if scenario_id in (None, ""):
        return None
    for scenario in scenarios:
        if str(scenario.get("scenario_id")) == str(scenario_id):
            return scenario
    return None


def _required_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _required_sequence(value: object, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a list")
    return value


def _required_number_list(value: object, name: str) -> list[float]:
    sequence = _required_sequence(value, name)
    numbers = [_required_finite(item, name) for item in sequence]
    if not numbers:
        raise ValueError(f"{name} must not be empty")
    return numbers


def _optional_number_list(value: object) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    try:
        return [_required_finite(item, "number list") for item in value]
    except ValueError:
        return None


def _optional_finite(value: object) -> float | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _required_finite(value: object, name: str) -> float:
    if value in (None, "") or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _stable_unique(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value and value not in seen:
            output.append(value)
            seen.add(value)
    return tuple(output)
