"""Run a single v0 volume trajectory from fixture inputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.mechanistic_simulator.io_utils import (
    DEFAULT_OUTPUT_DIR,
    REPO_ROOT,
    load_case_and_schedule,
    load_json,
    output_days,
    write_json,
)
from experiments.mechanistic_simulator.volume_ode import simulate_volume_trajectory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        default="fixtures/mechanistic_simulator/cases/tnbc_demo_case.json",
    )
    parser.add_argument(
        "--params",
        default="fixtures/mechanistic_simulator/params/resistant_disease_params.json",
    )
    parser.add_argument("--dt-days", type=float, default=0.5)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR / "v0_single_trajectory.json"))
    args = parser.parse_args()

    case, schedule = load_case_and_schedule(REPO_ROOT / args.case)
    params = load_json(REPO_ROOT / args.params)
    result = simulate_volume_trajectory(
        initial_volume_ml=case["baseline_measurement"]["tumor_volume_ml"],
        treatment_schedule=schedule,
        params=params,
        output_days=output_days(schedule["total_duration_days"], 7),
        dt_days=args.dt_days,
    )
    result["case_id"] = case["case_id"]
    write_json(args.output, result)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
