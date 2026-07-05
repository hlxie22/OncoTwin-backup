from __future__ import annotations

import copy
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_json(relative_path: str) -> dict:
    with (REPO_ROOT / relative_path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def no_treatment_schedule(total_duration_days: int = 120) -> dict:
    return {
        "schedule_id": "none",
        "regimen_name": "No treatment",
        "total_duration_days": total_duration_days,
        "events": [],
    }


def no_drug_params(**overrides) -> dict:
    params = {
        "growth_law": "logistic",
        "growth_rate": 0.006,
        "carrying_capacity_ml": 250.0,
        "drug_sensitivity": {},
        "drug_ec50": {},
        "drug_decay": {},
        "resistant_fraction": 0.0,
        "resistant_sensitivity_scale": 0.0,
        "observation_noise_fraction": 0.1,
    }
    params.update(overrides)
    return params


def simple_drug_schedule() -> dict:
    return {
        "schedule_id": "simple",
        "regimen_name": "Simple demo drug",
        "total_duration_days": 60,
        "events": [
            {"drug": "demo_drug", "day": 0, "relative_dose": 1.0},
            {"drug": "demo_drug", "day": 7, "relative_dose": 1.0},
        ],
    }


def simple_drug_params(**overrides) -> dict:
    params = {
        "growth_law": "logistic",
        "growth_rate": 0.004,
        "carrying_capacity_ml": 250.0,
        "drug_sensitivity": {"demo_drug": 0.08},
        "drug_ec50": {"demo_drug": 0.5},
        "drug_decay": {"demo_drug": 0.25},
        "resistant_fraction": 0.05,
        "resistant_sensitivity_scale": 0.1,
        "observation_noise_fraction": 0.1,
    }
    merged = copy.deepcopy(params)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return merged
