"""Run input-validation and numerical-guard stress tests."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.mechanistic_simulator.io_utils import (
    DEFAULT_OUTPUT_DIR,
    REPO_ROOT,
    load_case_and_schedule,
    load_json,
    write_json,
)
from experiments.mechanistic_simulator.validation import SimulatorInputError
from experiments.mechanistic_simulator.volume_ode import simulate_volume_trajectory


def _run_case(name: str, case: dict, schedule: dict, params: dict) -> dict:
    try:
        result = simulate_volume_trajectory(
            initial_volume_ml=case["baseline_measurement"]["tumor_volume_ml"],
            treatment_schedule=schedule,
            params=params,
            output_days=[0, 14, 28],
            dt_days=params.pop("_dt_days", 0.5),
        )
        return {"name": name, "status": "completed", "warnings": result["warnings"]}
    except SimulatorInputError as exc:
        return {"name": name, "status": "failed_validation", "message": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR / "v0_stress_test_report.json"))
    args = parser.parse_args()

    case, schedule = load_case_and_schedule(
        REPO_ROOT / "fixtures/mechanistic_simulator/cases/tnbc_demo_case.json"
    )
    base_params = load_json(REPO_ROOT / "fixtures/mechanistic_simulator/params/resistant_disease_params.json")

    scenarios = []
    high_growth = copy.deepcopy(base_params)
    high_growth["growth_rate"] = 0.2
    scenarios.append(("very_high_growth", high_growth))

    high_kill = copy.deepcopy(base_params)
    high_kill["drug_sensitivity"]["anthracycline"] = 0.95
    high_kill["drug_sensitivity"]["taxane"] = 0.95
    scenarios.append(("very_high_kill", high_kill))

    high_resistant = copy.deepcopy(base_params)
    high_resistant["resistant_fraction"] = 0.9
    scenarios.append(("very_high_resistant_fraction", high_resistant))

    low_capacity = copy.deepcopy(base_params)
    low_capacity["carrying_capacity_ml"] = 1.0
    scenarios.append(("very_low_carrying_capacity", low_capacity))

    large_dt = copy.deepcopy(base_params)
    large_dt["_dt_days"] = 7.0
    scenarios.append(("large_dt", large_dt))

    missing_decay = copy.deepcopy(base_params)
    del missing_decay["drug_decay"]["taxane"]
    scenarios.append(("missing_drug_decay", missing_decay))

    unknown_schedule = copy.deepcopy(base_params)
    modified_schedule = copy.deepcopy(schedule)
    modified_schedule["events"][0]["drug"] = "unknown_demo_drug"
    results = [_run_case("unknown_drug_category", case, modified_schedule, unknown_schedule)]
    results.extend(_run_case(name, case, schedule, params) for name, params in scenarios)

    report = {"stress_tests": results}
    write_json(args.output, report)
    print(f"wrote {args.output}")
    for result in results:
        print(f"{result['name']}: {result['status']}")


if __name__ == "__main__":
    main()
