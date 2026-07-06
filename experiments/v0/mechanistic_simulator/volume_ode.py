"""Volume-only tumor ODE simulator."""

from __future__ import annotations

import math
from typing import Any

from .exposure import exposure_at_time
from .validation import (
    SimulatorInputError,
    append_warning,
    validate_output_days,
    validate_params,
    validate_schedule,
)


SIMULATION_VERSION = "volume_ode_v0"


def estimate_longest_diameter_cm(volume_ml: float) -> float:
    """Estimate spherical equivalent diameter from volume in mL/cm^3."""

    volume_ml = max(float(volume_ml), 0.0)
    if volume_ml == 0:
        return 0.0
    return (6.0 * volume_ml / math.pi) ** (1.0 / 3.0)


def _kill_rate_per_day(exposures: dict[str, float], params: dict[str, Any]) -> float:
    total = 0.0
    for drug, exposure in exposures.items():
        if exposure <= 0:
            continue
        sensitivity = float(params["drug_sensitivity"][drug])
        ec50 = float(params["drug_ec50"][drug])
        total += sensitivity * exposure / (exposure + ec50)
    return total


def _state_record(
    day: float,
    sensitive_volume: float,
    resistant_volume: float,
    treatment_schedule: dict[str, Any],
    params: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    volume = max(sensitive_volume + resistant_volume, 0.0)
    carrying_capacity = float(params["carrying_capacity_ml"])
    growth_rate = float(params["growth_rate"])
    exposures = exposure_at_time(treatment_schedule, params, day)
    kill_rate = _kill_rate_per_day(exposures, params)
    growth_component = growth_rate * volume * (1.0 - volume / carrying_capacity)
    sensitive_kill = kill_rate * sensitive_volume
    resistant_kill = (
        kill_rate
        * float(params.get("resistant_sensitivity_scale", 0.0))
        * resistant_volume
    )

    if volume > 1.25 * carrying_capacity:
        append_warning(warnings, "Volume exceeded carrying capacity by an implausible margin.")

    return {
        "day": round(day, 8),
        "tumor_volume_ml": volume,
        "predicted_longest_diameter_cm": estimate_longest_diameter_cm(volume),
        "sensitive_volume_ml": max(sensitive_volume, 0.0),
        "resistant_volume_ml": max(resistant_volume, 0.0),
        "total_exposure_by_drug": exposures,
        "growth_component": growth_component,
        "kill_component": sensitive_kill + resistant_kill,
        "warnings": list(warnings),
    }


def simulate_volume_trajectory(
    initial_volume_ml: float,
    treatment_schedule: dict[str, Any],
    params: dict[str, Any],
    output_days: list[float],
    dt_days: float = 0.5,
    allow_beyond_schedule: bool = False,
) -> dict[str, Any]:
    """Simulate one volume trajectory with sensitive and resistant compartments."""

    initial_volume = float(initial_volume_ml)
    if not math.isfinite(initial_volume) or initial_volume < 0:
        raise SimulatorInputError("initial_volume_ml must be finite and non-negative.")
    if initial_volume == 0:
        base_warning = "Initial volume is zero; trajectory remains a zero-volume mathematical case."
    else:
        base_warning = ""

    output_days = validate_output_days(output_days)
    dt_days = float(dt_days)
    if not math.isfinite(dt_days) or dt_days <= 0:
        raise SimulatorInputError("dt_days must be finite and positive.")

    validate_schedule(treatment_schedule, params)
    warnings = validate_params(params, treatment_schedule)
    if base_warning:
        append_warning(warnings, base_warning)

    total_duration = float(treatment_schedule["total_duration_days"])
    if output_days[-1] > total_duration and not allow_beyond_schedule:
        raise SimulatorInputError(
            "output days extend beyond schedule duration; set allow_beyond_schedule=True explicitly."
        )
    if dt_days > 2.0:
        append_warning(warnings, "Large dt_days used; compare against a smaller step before trusting shape.")

    carrying_capacity = float(params["carrying_capacity_ml"])
    if initial_volume > carrying_capacity:
        append_warning(warnings, "Initial volume exceeds carrying capacity; logistic growth will shrink it.")

    resistant_fraction = float(params.get("resistant_fraction", 0.0))
    sensitive_volume = initial_volume * (1.0 - resistant_fraction)
    resistant_volume = initial_volume * resistant_fraction
    current_day = 0.0
    records: list[dict[str, Any]] = []

    for target_day in output_days:
        while current_day + 1e-12 < target_day:
            step = min(dt_days, target_day - current_day)
            total_volume = sensitive_volume + resistant_volume
            exposures = exposure_at_time(treatment_schedule, params, current_day)
            kill_rate = _kill_rate_per_day(exposures, params)
            growth_modifier = 1.0 - total_volume / carrying_capacity
            growth_rate = float(params["growth_rate"])
            resistant_scale = float(params.get("resistant_sensitivity_scale", 0.0))

            sensitive_delta = (
                growth_rate * sensitive_volume * growth_modifier
                - kill_rate * sensitive_volume
            ) * step
            resistant_delta = (
                growth_rate * resistant_volume * growth_modifier
                - kill_rate * resistant_scale * resistant_volume
            ) * step

            sensitive_volume += sensitive_delta
            resistant_volume += resistant_delta
            if sensitive_volume < 0 or resistant_volume < 0:
                append_warning(
                    warnings,
                    "Numerical step drove a compartment below zero; value was floored at zero.",
                )
                sensitive_volume = max(sensitive_volume, 0.0)
                resistant_volume = max(resistant_volume, 0.0)

            total_volume = sensitive_volume + resistant_volume
            if not math.isfinite(total_volume):
                raise SimulatorInputError("trajectory became non-finite during integration.")
            if total_volume > 1.25 * carrying_capacity:
                append_warning(
                    warnings,
                    "Volume exceeded carrying capacity by an implausible margin.",
                )

            current_day += step

        records.append(
            _state_record(
                target_day,
                sensitive_volume,
                resistant_volume,
                treatment_schedule,
                params,
                warnings,
            )
        )

    return {
        "simulation_version": SIMULATION_VERSION,
        "initial_volume_ml": initial_volume,
        "times": output_days,
        "trajectory": records,
        "warnings": list(warnings),
    }
