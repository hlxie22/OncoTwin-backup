"""Run a v0 parameter ensemble from fixture inputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.mechanistic_simulator.ensemble import simulate_volume_ensemble
from experiments.mechanistic_simulator.io_utils import (
    DEFAULT_OUTPUT_DIR,
    REPO_ROOT,
    load_case_and_schedule,
    load_json,
    output_days,
    write_json,
    write_svg_trajectory_plot,
)
from experiments.mechanistic_simulator.params import sample_volume_params
from experiments.mechanistic_simulator.safety import assert_no_unsafe_language


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        default="fixtures/mechanistic_simulator/cases/tnbc_demo_case.json",
    )
    parser.add_argument(
        "--prior",
        default="fixtures/mechanistic_simulator/params/generic_volume_prior.json",
    )
    parser.add_argument("--particles", type=int, default=250)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--dt-days", type=float, default=0.5)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR / "v0_ensemble_output.json"))
    parser.add_argument("--plot", default=str(DEFAULT_OUTPUT_DIR / "v0_ensemble_plot.svg"))
    args = parser.parse_args()

    case, schedule = load_case_and_schedule(REPO_ROOT / args.case)
    prior = load_json(REPO_ROOT / args.prior)
    particles = sample_volume_params(prior, n_particles=args.particles, seed=args.seed)
    result = simulate_volume_ensemble(
        initial_volume_ml=case["baseline_measurement"]["tumor_volume_ml"],
        treatment_schedule=schedule,
        parameter_particles=particles,
        output_days=output_days(schedule["total_duration_days"], 7),
        dt_days=args.dt_days,
    )
    result["case_id"] = case["case_id"]
    assert_no_unsafe_language(result)
    write_json(args.output, result)
    write_svg_trajectory_plot(
        args.plot,
        result["times"],
        result["median_volume_ml"],
        result["interval_80_volume_ml"],
        f"{case['display_name']} - v0 ensemble",
    )
    print(f"wrote {args.output}")
    print(f"wrote {args.plot}")
    print(f"final median volume: {result['median_volume_ml'][-1]:.3f} mL")
    print(f"uncertainty score: {result['uncertainty_score']:.3f}")


if __name__ == "__main__":
    main()
