"""Real-data uncertainty calibration eval for V1 prior-predictive intervals."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .common import EvalResult, print_result, unavailable
from .v1_real_data_eval import run_real_data_eval


DEFAULT_REPORT_PATH = Path("evals/reports/v1_uncertainty_calibration.md")


def run_uncertainty_calibration_eval(
    cohort: Path | None,
    *,
    report_path: Path = DEFAULT_REPORT_PATH,
    n_samples: int = 2000,
    seed: int = 2026,
    allow_demo_data: bool = False,
) -> EvalResult:
    if cohort is None:
        return unavailable(
            "uncertainty_calibration",
            ("real_longitudinal_cohort",),
            "No cohort was provided, so real-data interval calibration could not be computed.",
        )

    result = run_real_data_eval(
        cohort,
        n_samples=n_samples,
        seed=seed,
        allow_demo_data=allow_demo_data,
    )
    metrics = result["metrics"].get("layer4_mri_qc", {})

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join(
            [
                "# V1 uncertainty calibration",
                "",
                f"- Cohort path: `{cohort}`",
                f"- In-scope cases: {result['in_scope_case_count']}",
                f"- Layer 4 80% coverage: {metrics.get('coverage_80')}",
                f"- Layer 4 95% coverage: {metrics.get('coverage_95')}",
                f"- Layer 4 mean 80% interval width ml: {metrics.get('width_80_ml')}",
                "",
                "This uses prior-predictive intervals; posterior calibration requires the Bayesian update runtime.",
            ]
        ),
        encoding="utf-8",
    )

    return EvalResult(
        name="uncertainty_calibration",
        status="pass",
        summary="Computed prior-predictive interval coverage on real held-out observations.",
        metrics={
            "layer4_coverage_80": metrics.get("coverage_80"),
            "layer4_coverage_95": metrics.get("coverage_95"),
            "layer4_width_80_ml": metrics.get("width_80_ml"),
        },
        report_path=str(report_path),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run V1 uncertainty calibration on a real cohort."
    )
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--allow-demo-data", action="store_true")
    args = parser.parse_args(argv)

    result = run_uncertainty_calibration_eval(
        args.cohort,
        report_path=args.report,
        n_samples=args.n_samples,
        seed=args.seed,
        allow_demo_data=args.allow_demo_data,
    )
    print_result(result)
    return 0 if result.available else 2


if __name__ == "__main__":
    raise SystemExit(main())
