"""Run practical identifiability analysis after synthetic reweighting."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.mechanistic_simulator.identifiability import analyze_identifiability
from experiments.mechanistic_simulator.io_utils import (
    DEFAULT_OUTPUT_DIR,
    REPO_ROOT,
    load_case_and_schedule,
    load_json,
    write_json,
)
from experiments.mechanistic_simulator.params import sample_volume_params
from experiments.mechanistic_simulator.synthetic_fit import (
    make_noisy_observations,
    reweight_particles_from_observations,
)
from experiments.mechanistic_simulator.validation import schedule_drugs
from experiments.mechanistic_simulator.volume_ode import simulate_volume_trajectory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        default="fixtures/mechanistic_simulator/cases/longitudinal_measurement_demo_case.json",
    )
    parser.add_argument(
        "--truth-params",
        default="fixtures/mechanistic_simulator/params/high_response_params.json",
    )
    parser.add_argument(
        "--prior",
        default="fixtures/mechanistic_simulator/params/generic_volume_prior.json",
    )
    parser.add_argument("--particles", type=int, default=600)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR / "v0_identifiability_report.json"))
    args = parser.parse_args()

    case, schedule = load_case_and_schedule(REPO_ROOT / args.case)
    truth = simulate_volume_trajectory(
        initial_volume_ml=case["baseline_measurement"]["tumor_volume_ml"],
        treatment_schedule=schedule,
        params=load_json(REPO_ROOT / args.truth_params),
        output_days=[0, 42, 84, 126],
        dt_days=0.25,
    )
    observations = make_noisy_observations(truth, noise_fraction=0.08, seed=args.seed)
    fit = reweight_particles_from_observations(
        initial_volume_ml=case["baseline_measurement"]["tumor_volume_ml"],
        treatment_schedule=schedule,
        parameter_particles=sample_volume_params(
            load_json(REPO_ROOT / args.prior),
            n_particles=args.particles,
            seed=args.seed,
        ),
        observations=observations,
        dt_days=0.5,
    )
    report = analyze_identifiability(
        fit["particle_trajectories"],
        observations=observations,
        active_drugs=schedule_drugs(schedule),
    )
    report["case_id"] = case["case_id"]
    report["effective_sample_size"] = fit["effective_sample_size"]
    write_json(args.output, report)
    print(f"wrote {args.output}")
    print(f"constrained: {', '.join(report['constrained_parameters']) or 'none'}")
    print(f"prior-dominated: {', '.join(report['prior_dominated_parameters']) or 'none'}")


if __name__ == "__main__":
    main()
