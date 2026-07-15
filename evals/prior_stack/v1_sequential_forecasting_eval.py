"""Sequential forecasting smoke eval for the V1 patient-update stack."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

from .common import EvalResult, print_result
from .v1_runtime_smoke_fixtures import (
    deterministic_observations,
    heldout_final_observation,
    posterior_update_fixture,
    write_json,
    write_markdown_report,
)


NAME = "sequential_forecasting"


def run_eval(
    *,
    report_path: Path | None = None,
    analysis_path: Path | None = None,
    strict: bool = False,
) -> EvalResult:
    early_observation = deterministic_observations()[:1]
    heldout = heldout_final_observation()
    posterior = posterior_update_fixture(
        observations=early_observation,
        prediction_days=[42.0, float(heldout["day"])],
    )
    times = list(posterior["posterior_trajectory_summary"]["times"])  # type: ignore[index]
    final_index = times.index(float(heldout["day"]))
    forecast_median = float(
        posterior["posterior_trajectory_summary"]["median_volume_ml"][final_index]  # type: ignore[index]
    )
    heldout_volume = float(heldout["tumor_volume_ml"])
    absolute_error_ml = abs(forecast_median - heldout_volume)
    relative_error = absolute_error_ml / max(heldout_volume, 1.0)
    status = "pass" if math.isfinite(relative_error) else "fail"
    summary = (
        "Sequential forecasting smoke check fit the day-21 observation and "
        f"forecast held-out day {heldout['day']:.0f} volume with "
        f"absolute error {absolute_error_ml:.3g} mL."
    )
    analysis = {
        "eval_name": NAME,
        "status": status,
        "train_observation_days": [row["day"] for row in early_observation],
        "heldout_day": heldout["day"],
        "heldout_volume_ml": heldout_volume,
        "forecast_median_volume_ml": forecast_median,
        "absolute_error_ml": absolute_error_ml,
        "relative_error": relative_error,
        "posterior_trajectory_summary": posterior.get("posterior_trajectory_summary"),
    }
    written_analysis = write_json(analysis_path, analysis)
    metrics = {
        "heldout_day": heldout["day"],
        "heldout_volume_ml": heldout_volume,
        "forecast_median_volume_ml": forecast_median,
        "absolute_error_ml": absolute_error_ml,
        "relative_error": relative_error,
    }
    if written_analysis:
        metrics["analysis_path"] = written_analysis
    written_report = write_markdown_report(
        report_path,
        title="V1 sequential-forecasting smoke eval",
        summary=summary,
        metrics=metrics,
    )
    return EvalResult(
        name=NAME,
        status=status,
        summary=summary,
        metrics=metrics,
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
