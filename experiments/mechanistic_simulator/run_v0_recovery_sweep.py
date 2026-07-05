"""Run stability, noise, ablation, held-out, and baseline checks for v0."""

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
    write_json,
)
from experiments.mechanistic_simulator.recovery_sweep import (
    run_recovery_sweep,
    write_markdown_report,
)


def _parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _parse_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


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
    parser.add_argument("--seeds", default="101,202,303")
    parser.add_argument("--particle-counts", default="300,900")
    parser.add_argument("--assumed-noise-levels", default="0.08,0.12,0.20,0.30")
    parser.add_argument(
        "--variants",
        default="full,active_drugs_only,fixed_core,shared_chemo_fixed_core",
    )
    parser.add_argument("--generated-noise-fraction", type=float, default=0.08)
    parser.add_argument(
        "--output-json",
        default=str(DEFAULT_OUTPUT_DIR / "v0_recovery_sweep_report.json"),
    )
    parser.add_argument(
        "--output-md",
        default=str(DEFAULT_OUTPUT_DIR / "v0_recovery_sweep_report.md"),
    )
    args = parser.parse_args()

    case, schedule = load_case_and_schedule(REPO_ROOT / args.case)
    report = run_recovery_sweep(
        case=case,
        schedule=schedule,
        base_prior=load_json(REPO_ROOT / args.prior),
        truth_params=load_json(REPO_ROOT / args.truth_params),
        seeds=_parse_ints(args.seeds),
        particle_counts=_parse_ints(args.particle_counts),
        assumed_noise_levels=_parse_floats(args.assumed_noise_levels),
        variants=_parse_strings(args.variants),
        generated_noise_fraction=args.generated_noise_fraction,
    )
    write_json(args.output_json, report)
    write_markdown_report(args.output_md, report)
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")
    for insight in report["insights"]:
        print(f"- {insight}")


if __name__ == "__main__":
    main()
