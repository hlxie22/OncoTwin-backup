"""Validation helpers for the volume-only simulator."""

from __future__ import annotations

import math
from typing import Any, Iterable


class SimulatorInputError(ValueError):
    """Raised when simulator input would make outputs invalid or misleading."""


MAX_GROWTH_RATE_PER_DAY = 0.1
MAX_SENSITIVITY_PER_DAY = 1.0


def append_warning(warnings: list[str], message: str) -> None:
    if message not in warnings:
        warnings.append(message)


def require_finite_number(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SimulatorInputError(f"{name} must be a finite number.")
    value = float(value)
    if not math.isfinite(value):
        raise SimulatorInputError(f"{name} must be finite.")
    return value


def schedule_drugs(schedule: dict[str, Any]) -> list[str]:
    drugs: list[str] = []
    for event in schedule.get("events", []):
        drug = event.get("drug")
        if isinstance(drug, str) and drug not in drugs:
            drugs.append(drug)
    return drugs


def validate_times(times: Iterable[float], name: str = "times") -> list[float]:
    normalized = [require_finite_number(value, f"{name} value") for value in times]
    if not normalized:
        raise SimulatorInputError(f"{name} must contain at least one value.")
    return normalized


def validate_output_days(output_days: Iterable[float]) -> list[float]:
    days = validate_times(output_days, "output_days")
    for day in days:
        if day < 0:
            raise SimulatorInputError("output_days cannot contain negative days.")
    ordered = sorted(days)
    if ordered != days:
        raise SimulatorInputError("output_days must be sorted in ascending order.")
    return days


def validate_schedule(schedule: dict[str, Any], params: dict[str, Any] | None = None) -> None:
    if not isinstance(schedule, dict):
        raise SimulatorInputError("treatment schedule must be an object.")
    for key in ("schedule_id", "regimen_name", "total_duration_days", "events"):
        if key not in schedule:
            raise SimulatorInputError(f"treatment schedule is missing {key}.")
    total_duration = require_finite_number(schedule["total_duration_days"], "total_duration_days")
    if total_duration < 0:
        raise SimulatorInputError("total_duration_days cannot be negative.")
    if not isinstance(schedule["events"], list):
        raise SimulatorInputError("schedule events must be a list.")

    known_decay = params.get("drug_decay", {}) if params else {}
    for index, event in enumerate(schedule["events"]):
        if not isinstance(event, dict):
            raise SimulatorInputError(f"schedule event {index} must be an object.")
        drug = event.get("drug")
        if not isinstance(drug, str) or not drug:
            raise SimulatorInputError(f"schedule event {index} must include a drug.")
        day = require_finite_number(event.get("day"), f"schedule event {index} day")
        if day < 0:
            raise SimulatorInputError(f"schedule event {index} day cannot be negative.")
        relative_dose = require_finite_number(
            event.get("relative_dose"), f"schedule event {index} relative_dose"
        )
        if relative_dose < 0:
            raise SimulatorInputError(f"schedule event {index} relative_dose cannot be negative.")
        if params is not None and drug not in known_decay:
            raise SimulatorInputError(f"missing drug_decay parameter for scheduled drug {drug}.")


def validate_params(params: dict[str, Any], schedule: dict[str, Any] | None = None) -> list[str]:
    if not isinstance(params, dict):
        raise SimulatorInputError("mechanistic params must be an object.")
    warnings: list[str] = []

    growth_law = params.get("growth_law", "logistic")
    if growth_law != "logistic":
        raise SimulatorInputError("v0 only supports logistic growth_law.")

    growth_rate = require_finite_number(params.get("growth_rate"), "growth_rate")
    if growth_rate < 0 or growth_rate > MAX_GROWTH_RATE_PER_DAY:
        raise SimulatorInputError(
            f"growth_rate must be between 0 and {MAX_GROWTH_RATE_PER_DAY} per day."
        )
    if growth_rate > 0.05:
        append_warning(warnings, "High growth_rate used; trajectory should be treated as stress-test output.")

    carrying_capacity = require_finite_number(
        params.get("carrying_capacity_ml"), "carrying_capacity_ml"
    )
    if carrying_capacity <= 0:
        raise SimulatorInputError("carrying_capacity_ml must be positive.")

    for dictionary_name in ("drug_sensitivity", "drug_ec50", "drug_decay"):
        if not isinstance(params.get(dictionary_name), dict):
            raise SimulatorInputError(f"{dictionary_name} must be an object keyed by drug.")

    drugs_to_validate = set(params["drug_decay"].keys())
    if schedule is not None:
        drugs_to_validate.update(schedule_drugs(schedule))

    for drug in sorted(drugs_to_validate):
        if drug not in params["drug_sensitivity"]:
            raise SimulatorInputError(f"missing drug_sensitivity parameter for {drug}.")
        if drug not in params["drug_ec50"]:
            raise SimulatorInputError(f"missing drug_ec50 parameter for {drug}.")
        if drug not in params["drug_decay"]:
            raise SimulatorInputError(f"missing drug_decay parameter for {drug}.")

        sensitivity = require_finite_number(
            params["drug_sensitivity"][drug], f"drug_sensitivity.{drug}"
        )
        if sensitivity < 0 or sensitivity > MAX_SENSITIVITY_PER_DAY:
            raise SimulatorInputError(
                f"drug_sensitivity.{drug} must be between 0 and {MAX_SENSITIVITY_PER_DAY}."
            )
        if sensitivity > 0.5:
            append_warning(
                warnings,
                f"High drug_sensitivity.{drug} used; trajectory should be treated as stress-test output.",
            )
        ec50 = require_finite_number(params["drug_ec50"][drug], f"drug_ec50.{drug}")
        if ec50 <= 0:
            raise SimulatorInputError(f"drug_ec50.{drug} must be positive.")
        decay = require_finite_number(params["drug_decay"][drug], f"drug_decay.{drug}")
        if decay <= 0:
            raise SimulatorInputError(f"drug_decay.{drug} must be positive.")

    resistant_fraction = require_finite_number(
        params.get("resistant_fraction", 0.0), "resistant_fraction"
    )
    if resistant_fraction < 0 or resistant_fraction > 1:
        raise SimulatorInputError("resistant_fraction must be between 0 and 1.")
    if resistant_fraction > 0.6:
        append_warning(
            warnings,
            "High resistant_fraction used; residual-burden behavior should be treated as stress-test output.",
        )

    resistant_scale = require_finite_number(
        params.get("resistant_sensitivity_scale", 0.0), "resistant_sensitivity_scale"
    )
    if resistant_scale < 0 or resistant_scale > 1:
        raise SimulatorInputError("resistant_sensitivity_scale must be between 0 and 1.")

    noise = require_finite_number(
        params.get("observation_noise_fraction", 0.0), "observation_noise_fraction"
    )
    if noise < 0 or noise > 1:
        raise SimulatorInputError("observation_noise_fraction must be between 0 and 1.")

    return warnings
