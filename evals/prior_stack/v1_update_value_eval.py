"""Update-value smoke eval for benefit from patient-specific evidence."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

from .common import EvalResult, print_result
from .v1_runtime_smoke_fixtures import (
    heldout_final_observation,
    posterior_update_fixture,
    write_json,
    write_markdown_report,
)


NAME = "update_value"


def run_eval(
    *,
    report_path: Path | None = None,
    analysis_path: Path | None = None,
    strict: bool = False,
) -> EvalResult:
    posterior = posterior_update_fixture()
    heldout = heldout_final_observation()
    times = list(posterior["posterior_trajectory_summary"]["times"])  # type: ignore[index]
    final_index = times.index(float(heldout["day"]))
    heldout_volume = float(heldout["tumor_volume_ml"])
    prior_median = float(
        posterior["prior_trajectory_summary"]["median_volume_ml"][final_index]  # type: ignore[index]
    )
    posterior_median = float(
        posterior["posterior_trajectory_summary"]["median_volume_ml"][final_index]  # type: ignore[index]
    )
    prior_error = abs(prior_median - heldout_volume)
    posterior_error = abs(posterior_median - heldout_volume)
    improvement_ml = prior_error - posterior_error
    status = "pass" if math.isfinite(improvement_ml) and improvement_ml >= 0 else "fail"
    summary = (
        "Update-value smoke check compared population-prior and updated posterior "
        f"median final-volume errors; improvement={improvement_ml:.3g} mL."
    )
    warnings = () if improvement_ml >= 0 else ("posterior smoke update did not improve held-out final-volume error",)
    analysis = {
        "eval_name": NAME,
        "status": status,
        "heldout_day": heldout["day"],
        "heldout_volume_ml": heldout_volume,
        "prior_median_volume_ml": prior_median,
        "posterior_median_volume_ml": posterior_median,
        "prior_absolute_error_ml": prior_error,
        "posterior_absolute_error_ml": posterior_error,
        "improvement_ml": improvement_ml,
        "warnings": list(warnings),
    }
    written_analysis = write_json(analysis_path, analysis)
    metrics = {
        "prior_absolute_error_ml": prior_error,
        "posterior_absolute_error_ml": posterior_error,
        "improvement_ml": improvement_ml,
    }
    if written_analysis:
        metrics["analysis_path"] = written_analysis
    written_report = write_markdown_report(
        report_path,
        title="V1 update-value smoke eval",
        summary=summary,
        metrics=metrics,
        warnings=warnings,
    )
    return EvalResult(
        name=NAME,
        status=status,
        summary=summary,
        metrics=metrics,
        warnings=warnings,
        report_path=written_report,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--analysis", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    result = run_eval(report_path=args.report, analysis_path=args.analysis, strict=args.strict)
    print_result(result)
    return 0 if result.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
