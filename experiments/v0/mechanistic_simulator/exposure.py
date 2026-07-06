"""Treatment exposure curves for the v0 volume simulator."""

from __future__ import annotations

import math
from typing import Any

from .validation import SimulatorInputError, schedule_drugs, validate_schedule, validate_times


def exposure_at_time(
    treatment_schedule: dict[str, Any],
    params: dict[str, Any],
    time_day: float,
) -> dict[str, float]:
    """Return per-drug exposure at one time point.

    Exposure is a simple sum of exponentially decaying dose events. This is a
    coarse mechanism category signal, not pharmacokinetics.
    """

    time_day = float(time_day)
    exposures = {drug: 0.0 for drug in schedule_drugs(treatment_schedule)}
    decay_by_drug = params.get("drug_decay", {})

    for event in treatment_schedule.get("events", []):
        drug = event["drug"]
        if drug not in decay_by_drug:
            raise SimulatorInputError(f"missing drug_decay parameter for scheduled drug {drug}.")
        dose_day = float(event["day"])
        relative_dose = float(event["relative_dose"])
        if time_day < dose_day or relative_dose == 0:
            continue
        elapsed = time_day - dose_day
        exposures[drug] += relative_dose * math.exp(-float(decay_by_drug[drug]) * elapsed)

    return exposures


def compute_exposure(
    treatment_schedule: dict[str, Any],
    params: dict[str, Any],
    times: list[float],
) -> dict[str, list[float]]:
    """Compute exposure by drug for every requested time point."""

    validate_schedule(treatment_schedule, params)
    normalized_times = validate_times(times)
    exposures_by_drug = {drug: [] for drug in schedule_drugs(treatment_schedule)}
    for time_day in normalized_times:
        exposures = exposure_at_time(treatment_schedule, params, time_day)
        for drug in exposures_by_drug:
            exposures_by_drug[drug].append(exposures.get(drug, 0.0))
    return exposures_by_drug
