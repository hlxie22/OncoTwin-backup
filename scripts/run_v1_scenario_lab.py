#!/usr/bin/env python3
"""Run V1 scenario-lab comparisons from a posterior-update JSON artifact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.twin_runtime.scenario_lab import run_scenario_lab


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare counterfactual treatment schedules under a V1 posterior update."
    )
    parser.add_argument("--posterior-update", required=True, type=Path)
    parser.add_argument("--scenarios", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--output-days", nargs="*", type=float, default=[])
    parser.add_argument("--dt-days", type=float, default=0.5)
    parser.add_argument("--allow-beyond-schedule", action="store_true")
    parser.add_argument("--residual-burden-threshold-ml", type=float, default=1.0)
    parser.add_argument("--include-particle-trajectories", action="store_true")
    args = parser.parse_args()

    result = run_scenario_lab(
        posterior_update=_read_json_object(args.posterior_update),
        scenarios=_read_scenarios(args.scenarios),
        output_days=args.output_days,
        dt_days=args.dt_days,
        allow_beyond_schedule=args.allow_beyond_schedule,
        residual_burden_threshold_ml=args.residual_burden_threshold_ml,
        include_particle_trajectories=args.include_particle_trajectories,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "wrote scenario lab output for "
        f"{result['n_scenarios']} scenario(s) to {args.output}; "
        f"top={result['comparison_summary'].get('top_scenario_id')}"
    )
    return 0


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return dict(payload)


def _read_scenarios(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(_require_mapping(row, path)) for row in payload]
    if isinstance(payload, Mapping):
        scenarios = payload.get("scenarios")
        if isinstance(scenarios, list):
            return [dict(_require_mapping(row, path)) for row in scenarios]
    raise ValueError(f"{path} must contain a scenario array or an object with scenarios")


def _require_mapping(value: object, path: Path) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} scenario rows must be JSON objects")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
