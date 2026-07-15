"""V1 scenario-lab runtime over posterior-weighted simulator particles.

This module compares candidate treatment schedules under the current posterior.
It is deliberately decision-support only: it summarizes modeled trajectories and
uncertainty, but never recommends treatment choices.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from experiments.twin_runtime.posterior import (
    DEFAULT_LOW_RESIDUAL_BURDEN_ML,
    normalize_log_weights,
    summarize_weighted_trajectories,
)
from experiments.v0.mechanistic_simulator.volume_ode import simulate_volume_trajectory


SCENARIO_LAB_VERSION = "oncotwin_scenario_lab_v1"
DECISION_SUPPORT_DISCLAIMER = (
    "Scenario outputs are research decision-support summaries, not treatment "
    "recommendations. Clinical decisions require clinician review and patient-specific context."
)
DEFAULT_PROGRESSION_MULTIPLIER = 1.20
DEFAULT_INSUFFICIENT_RESPONSE_REMAINING_FRACTION = 0.70
DEFAULT_MAX_RELATIVE_DOSE = 1.25
DEFAULT_MAX_EVENTS = 96


@dataclass(frozen=True)
class _PosteriorParticle:
    particle_id: str
    parameters: Mapping[str, Any]
    weight: float


def run_scenario_lab(
    *,
    posterior_update: Mapping[str, Any],
    scenarios: Sequence[Mapping[str, Any]],
    output_days: Sequence[float] = (),
    dt_days: float = 0.5,
    allow_beyond_schedule: bool = False,
    residual_burden_threshold_ml: float = DEFAULT_LOW_RESIDUAL_BURDEN_ML,
    progression_multiplier: float = DEFAULT_PROGRESSION_MULTIPLIER,
    insufficient_response_remaining_fraction: float = DEFAULT_INSUFFICIENT_RESPONSE_REMAINING_FRACTION,
    max_relative_dose: float = DEFAULT_MAX_RELATIVE_DOSE,
    max_events: int = DEFAULT_MAX_EVENTS,
    include_particle_trajectories: bool = False,
) -> dict[str, object]:
    """Compare candidate schedules using posterior particle weights."""

    if not scenarios:
        raise ValueError("scenarios must contain at least one scenario")
    if dt_days <= 0 or not math.isfinite(dt_days):
        raise ValueError("dt_days must be finite and positive")

    initial_volume_ml = _required_nonnegative(
        posterior_update.get("initial_volume_ml"),
        "posterior_update.initial_volume_ml",
    )
    particles = _posterior_particles(posterior_update)
    normalized_weights = _normalize_particle_weights(particles)

    scenario_results: list[dict[str, object]] = []
    for index, scenario in enumerate(scenarios):
        scenario_results.append(
            _run_one_scenario(
                scenario,
                scenario_index=index,
                initial_volume_ml=initial_volume_ml,
                particles=particles,
                normalized_weights=normalized_weights,
                posterior_update=posterior_update,
                global_output_days=output_days,
                dt_days=dt_days,
                allow_beyond_schedule=allow_beyond_schedule,
                residual_burden_threshold_ml=residual_burden_threshold_ml,
                progression_multiplier=progression_multiplier,
                insufficient_response_remaining_fraction=insufficient_response_remaining_fraction,
                max_relative_dose=max_relative_dose,
                max_events=max_events,
                include_particle_trajectories=include_particle_trajectories,
            )
        )

    comparison_summary = _comparison_summary(scenario_results)
    _attach_reference_comparisons(scenario_results, comparison_summary.get("reference_scenario_id"))

    return {
        "scenario_lab_version": SCENARIO_LAB_VERSION,
        "decision_support_disclaimer": DECISION_SUPPORT_DISCLAIMER,
        "not_a_treatment_recommendation": True,
        "posterior_runtime_version": posterior_update.get("posterior_runtime_version"),
        "initial_volume_ml": initial_volume_ml,
        "n_posterior_particles": len(particles),
        "n_scenarios": len(scenario_results),
        "residual_burden_threshold_ml": residual_burden_threshold_ml,
        "progression_multiplier": progression_multiplier,
        "insufficient_response_remaining_fraction": insufficient_response_remaining_fraction,
        "scenarios": scenario_results,
        "comparison_summary": comparison_summary,
        "warnings": _scenario_lab_warnings(scenario_results),
    }


def _run_one_scenario(
    scenario: Mapping[str, Any],
    *,
    scenario_index: int,
    initial_volume_ml: float,
    particles: Sequence[_PosteriorParticle],
    normalized_weights: Sequence[float],
    posterior_update: Mapping[str, Any],
    global_output_days: Sequence[float],
    dt_days: float,
    allow_beyond_schedule: bool,
    residual_burden_threshold_ml: float,
    progression_multiplier: float,
    insufficient_response_remaining_fraction: float,
    max_relative_dose: float,
    max_events: int,
    include_particle_trajectories: bool,
) -> dict[str, object]:
    scenario_id = _scenario_id(scenario, scenario_index)
    label = str(scenario.get("label") or scenario.get("name") or scenario_id)
    schedule = _scenario_schedule(scenario)
    safety = _safety_assessment(
        schedule,
        max_relative_dose=max_relative_dose,
        max_events=max_events,
    )
    output_days = _resolve_output_days(
        scenario,
        schedule,
        posterior_update=posterior_update,
        global_output_days=global_output_days,
        allow_beyond_schedule=allow_beyond_schedule,
    )

    base_payload: dict[str, object] = {
        "scenario_id": scenario_id,
        "label": label,
        "schedule_id": schedule.get("schedule_id"),
        "regimen_name": schedule.get("regimen_name"),
        "is_reference": _truthy(scenario.get("reference", scenario.get("is_reference"))),
        "safety_assessment": safety,
        "decision_support_disclaimer": DECISION_SUPPORT_DISCLAIMER,
        "not_a_treatment_recommendation": True,
    }

    if safety["status"] != "ok":
        return {
            **base_payload,
            "status": "failed_safety",
            "warnings": safety["warnings"],
            "explanation": (
                f"Scenario {scenario_id} was not simulated because it failed safety preflight. "
                + DECISION_SUPPORT_DISCLAIMER
            ),
        }

    particle_rows: list[dict[str, object]] = []
    scenario_warnings: list[str] = list(safety["warnings"])
    try:
        for particle in particles:
            result = simulate_volume_trajectory(
                initial_volume_ml=initial_volume_ml,
                treatment_schedule=dict(schedule),
                params=dict(particle.parameters),
                output_days=output_days,
                dt_days=dt_days,
                allow_beyond_schedule=allow_beyond_schedule,
            )
            warnings = [str(warning) for warning in result.get("warnings", [])]
            scenario_warnings.extend(warnings)
            particle_rows.append(
                {
                    "particle_id": particle.particle_id,
                    "weight": particle.weight,
                    "parameters": dict(particle.parameters),
                    "times": output_days,
                    "predicted_volume_ml": [
                        float(record["tumor_volume_ml"])
                        for record in result["trajectory"]
                    ],
                    "predicted_longest_diameter_cm": [
                        float(record["predicted_longest_diameter_cm"])
                        for record in result["trajectory"]
                    ],
                    "warnings": warnings,
                }
            )
    except Exception as exc:  # fail closed for malformed or unsupported schedules
        return {
            **base_payload,
            "status": "failed_runtime",
            "warnings": [*scenario_warnings, str(exc)],
            "explanation": (
                f"Scenario {scenario_id} could not be simulated: {exc}. "
                + DECISION_SUPPORT_DISCLAIMER
            ),
        }

    trajectory_summary = summarize_weighted_trajectories(
        particle_rows,
        output_days=output_days,
        weights=normalized_weights,
        residual_burden_threshold_ml=residual_burden_threshold_ml,
    )
    probabilities = _scenario_probabilities(
        particle_rows,
        weights=normalized_weights,
        initial_volume_ml=initial_volume_ml,
        residual_burden_threshold_ml=residual_burden_threshold_ml,
        progression_multiplier=progression_multiplier,
        insufficient_response_remaining_fraction=insufficient_response_remaining_fraction,
    )
    explanation = _scenario_explanation(
        scenario_id=scenario_id,
        label=label,
        trajectory_summary=trajectory_summary,
        probabilities=probabilities,
    )

    payload = {
        **base_payload,
        "status": "ok",
        "output_days": output_days,
        "trajectory_summary": trajectory_summary,
        "probabilities": probabilities,
        "warnings": list(_stable_unique(scenario_warnings)),
        "explanation": explanation,
    }
    if include_particle_trajectories:
        payload["particle_trajectories"] = particle_rows
    else:
        payload["particle_summaries"] = [
            {
                "particle_id": row["particle_id"],
                "weight": row["weight"],
                "final_volume_ml": row["predicted_volume_ml"][-1],  # type: ignore[index]
                "warnings": row["warnings"],
            }
            for row in particle_rows
        ]
    return payload


def _posterior_particles(posterior_update: Mapping[str, Any]) -> list[_PosteriorParticle]:
    rows = posterior_update.get("particle_trajectories")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise ValueError("posterior_update.particle_trajectories must contain particles")

    particles: list[_PosteriorParticle] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"posterior particle {index} must be an object")
        params = row.get("parameters")
        if not isinstance(params, Mapping):
            raise ValueError(f"posterior particle {index} is missing parameters")
        weight = _optional_positive(row.get("weight"))
        if weight is None:
            log_weight = row.get("log_weight")
            if log_weight in (None, ""):
                raise ValueError(f"posterior particle {index} is missing weight")
            weight = math.exp(_required_finite(log_weight, f"posterior particle {index} log_weight"))
        particles.append(
            _PosteriorParticle(
                particle_id=str(row.get("particle_id", f"p{index:06d}")),
                parameters=dict(params),
                weight=weight,
            )
        )
    return particles


def _normalize_particle_weights(particles: Sequence[_PosteriorParticle]) -> list[float]:
    return normalize_log_weights([math.log(particle.weight) for particle in particles])


def _scenario_schedule(scenario: Mapping[str, Any]) -> Mapping[str, Any]:
    for field in ("treatment_schedule", "schedule", "counterfactual_schedule"):
        schedule = scenario.get(field)
        if isinstance(schedule, Mapping):
            return dict(schedule)
    raise ValueError("each scenario requires treatment_schedule, schedule, or counterfactual_schedule")


def _scenario_id(scenario: Mapping[str, Any], index: int) -> str:
    for field in ("scenario_id", "id", "name"):
        value = scenario.get(field)
        if value not in (None, ""):
            return str(value)
    return f"scenario_{index + 1}"


def _resolve_output_days(
    scenario: Mapping[str, Any],
    schedule: Mapping[str, Any],
    *,
    posterior_update: Mapping[str, Any],
    global_output_days: Sequence[float],
    allow_beyond_schedule: bool,
) -> list[float]:
    for field in ("output_days", "prediction_days"):
        value = scenario.get(field)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and value:
            return _sorted_unique_floats(value)

    if global_output_days:
        return _sorted_unique_floats(global_output_days)

    posterior_days = posterior_update.get("prediction_days")
    duration = _optional_positive(schedule.get("total_duration_days"))
    if isinstance(posterior_days, Sequence) and not isinstance(posterior_days, (str, bytes)) and posterior_days:
        days = _sorted_unique_floats(posterior_days)
        if not allow_beyond_schedule and duration is not None:
            days = [day for day in days if day <= duration]
        if days:
            return days

    if duration is None:
        raise ValueError("scenario output days could not be resolved")
    return [duration]


def _safety_assessment(
    schedule: Mapping[str, Any],
    *,
    max_relative_dose: float,
    max_events: int,
) -> dict[str, object]:
    warnings: list[str] = []
    status = "ok"

    missing = [
        key for key in ("schedule_id", "regimen_name", "total_duration_days", "events")
        if key not in schedule
    ]
    if missing:
        return {
            "status": "failed",
            "warnings": ["schedule missing required fields: " + ", ".join(missing)],
        }

    duration = _optional_positive(schedule.get("total_duration_days"), allow_zero=True)
    if duration is None:
        status = "failed"
        warnings.append("total_duration_days must be finite and non-negative")

    events = schedule.get("events")
    if not isinstance(events, list):
        status = "failed"
        warnings.append("schedule events must be a list")
        events = []

    if len(events) == 0:
        warnings.append("scenario has no treatment events")
    if len(events) > max_events:
        status = "failed"
        warnings.append(f"scenario has more than {max_events} treatment events")

    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            status = "failed"
            warnings.append(f"schedule event {index} must be an object")
            continue
        if not event.get("drug"):
            status = "failed"
            warnings.append(f"schedule event {index} is missing drug")
        day = _optional_finite(event.get("day"))
        if day is None or day < 0:
            status = "failed"
            warnings.append(f"schedule event {index} day must be finite and non-negative")
        elif duration is not None and day > duration:
            status = "failed"
            warnings.append(f"schedule event {index} occurs after total_duration_days")
        dose = _optional_finite(event.get("relative_dose"))
        if dose is None or dose < 0:
            status = "failed"
            warnings.append(f"schedule event {index} relative_dose must be finite and non-negative")
        elif dose > max_relative_dose:
            status = "failed"
            warnings.append(
                f"schedule event {index} relative_dose exceeds safety cap {max_relative_dose:g}"
            )

    return {
        "status": status,
        "warnings": list(_stable_unique(warnings)),
    }


def _scenario_probabilities(
    particle_rows: Sequence[Mapping[str, object]],
    *,
    weights: Sequence[float],
    initial_volume_ml: float,
    residual_burden_threshold_ml: float,
    progression_multiplier: float,
    insufficient_response_remaining_fraction: float,
) -> dict[str, float]:
    final_values = [
        float(row["predicted_volume_ml"][-1])  # type: ignore[index]
        for row in particle_rows
    ]
    return {
        "probability_low_residual_burden": sum(
            weight
            for value, weight in zip(final_values, weights)
            if value <= residual_burden_threshold_ml
        ),
        "probability_progression": sum(
            weight
            for value, weight in zip(final_values, weights)
            if value >= initial_volume_ml * progression_multiplier
        ),
        "probability_insufficient_response": sum(
            weight
            for value, weight in zip(final_values, weights)
            if value >= initial_volume_ml * insufficient_response_remaining_fraction
        ),
    }


def _scenario_explanation(
    *,
    scenario_id: str,
    label: str,
    trajectory_summary: Mapping[str, object],
    probabilities: Mapping[str, float],
) -> str:
    final_median = float(trajectory_summary["median_volume_ml"][-1])  # type: ignore[index]
    lower80 = float(trajectory_summary["lower80_volume_ml"][-1])  # type: ignore[index]
    upper80 = float(trajectory_summary["upper80_volume_ml"][-1])  # type: ignore[index]
    low_residual = probabilities["probability_low_residual_burden"]
    progression = probabilities["probability_progression"]
    return (
        f"Scenario {scenario_id} ({label}) has a modeled final median tumor volume "
        f"of {final_median:.3g} mL with an 80% interval of {lower80:.3g}-{upper80:.3g} mL. "
        f"The posterior-weighted probability of low residual burden is {low_residual:.1%}, "
        f"and the probability of progression is {progression:.1%}. "
        + DECISION_SUPPORT_DISCLAIMER
    )


def _comparison_summary(scenario_results: Sequence[Mapping[str, object]]) -> dict[str, object]:
    ok_results = [result for result in scenario_results if result.get("status") == "ok"]
    if not ok_results:
        return {
            "reference_scenario_id": None,
            "ranked_scenario_ids_by_low_residual_probability": [],
            "warnings": ["no scenarios were simulated successfully"],
        }

    reference = next(
        (result for result in ok_results if _truthy(result.get("is_reference"))),
        ok_results[0],
    )
    ranked = sorted(
        ok_results,
        key=lambda result: (
            -float(result["probabilities"]["probability_low_residual_burden"]),  # type: ignore[index]
            float(result["trajectory_summary"]["median_volume_ml"][-1]),  # type: ignore[index]
            str(result["scenario_id"]),
        ),
    )
    return {
        "reference_scenario_id": reference["scenario_id"],
        "ranked_scenario_ids_by_low_residual_probability": [
            result["scenario_id"] for result in ranked
        ],
        "top_scenario_id": ranked[0]["scenario_id"],
        "comparison_note": (
            "Ranks summarize modeled posterior outcomes only and are not treatment recommendations."
        ),
    }


def _attach_reference_comparisons(
    scenario_results: Sequence[dict[str, object]],
    reference_id: object,
) -> None:
    if reference_id is None:
        return
    reference = next(
        (
            result for result in scenario_results
            if result.get("scenario_id") == reference_id and result.get("status") == "ok"
        ),
        None,
    )
    if reference is None:
        return

    reference_final = float(reference["trajectory_summary"]["median_volume_ml"][-1])  # type: ignore[index]
    reference_low_residual = float(
        reference["probabilities"]["probability_low_residual_burden"]  # type: ignore[index]
    )
    reference_progression = float(
        reference["probabilities"]["probability_progression"]  # type: ignore[index]
    )
    for result in scenario_results:
        if result.get("status") != "ok":
            continue
        final = float(result["trajectory_summary"]["median_volume_ml"][-1])  # type: ignore[index]
        low_residual = float(result["probabilities"]["probability_low_residual_burden"])  # type: ignore[index]
        progression = float(result["probabilities"]["probability_progression"])  # type: ignore[index]
        result["comparison_to_reference"] = {
            "reference_scenario_id": reference_id,
            "delta_final_median_volume_ml": final - reference_final,
            "delta_probability_low_residual_burden": low_residual - reference_low_residual,
            "delta_probability_progression": progression - reference_progression,
        }


def _scenario_lab_warnings(scenario_results: Sequence[Mapping[str, object]]) -> list[str]:
    warnings: list[str] = []
    failed = [result for result in scenario_results if result.get("status") != "ok"]
    if failed:
        warnings.append(f"{len(failed)} scenario(s) failed and were not used for ranking")
    if not any(result.get("status") == "ok" for result in scenario_results):
        warnings.append("no scenario produced modeled trajectories")
    warnings.append("scenario outputs are not treatment recommendations")
    return list(_stable_unique(warnings))


def _sorted_unique_floats(values: Sequence[object]) -> list[float]:
    output = sorted({_required_finite(value, "output_day") for value in values})
    if not output:
        raise ValueError("output_days must not be empty")
    if output[0] < 0:
        raise ValueError("output_days must be non-negative")
    return output


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "reference"}


def _optional_positive(value: object, *, allow_zero: bool = False) -> float | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    number = _optional_finite(value)
    if number is None:
        return None
    if allow_zero:
        return number if number >= 0 else None
    return number if number > 0 else None


def _optional_finite(value: object) -> float | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _required_nonnegative(value: object, name: str) -> float:
    number = _required_finite(value, name)
    if number < 0:
        raise ValueError(f"{name} must be non-negative")
    return number


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


def _stable_unique(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value and value not in seen:
            output.append(value)
            seen.add(value)
    return tuple(output)
