#!/usr/bin/env python3
"""Run a local V1 volume-posterior update from JSON/JSONL artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.twin_runtime.posterior import update_volume_posterior


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch-reweight volume-ODE particles against tumor-volume observations."
    )
    parser.add_argument("--initial-volume-ml", required=True, type=float)
    parser.add_argument("--schedule", required=True, type=Path, help="Treatment schedule JSON.")
    parser.add_argument("--particles", required=True, type=Path, help="Particle JSON or JSONL.")
    parser.add_argument("--observations", required=True, type=Path, help="Observation JSON or JSONL.")
    parser.add_argument("--output", required=True, type=Path, help="Posterior update JSON output.")
    parser.add_argument("--prediction-days", nargs="*", type=float, default=[])
    parser.add_argument("--dt-days", type=float, default=0.5)
    parser.add_argument("--likelihood-noise-fraction", type=float)
    parser.add_argument("--include-failed-qc-observations", action="store_true")
    args = parser.parse_args()

    result = update_volume_posterior(
        initial_volume_ml=args.initial_volume_ml,
        treatment_schedule=_read_json(args.schedule),
        parameter_particles=_read_rows(args.particles),
        observations=_read_rows(args.observations),
        prediction_days=args.prediction_days,
        dt_days=args.dt_days,
        likelihood_noise_fraction=args.likelihood_noise_fraction,
        include_failed_qc_observations=args.include_failed_qc_observations,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote posterior update with ESS={result['effective_sample_size']:.3f} to {args.output}")
    return 0


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return dict(payload)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(dict(payload))
        return rows
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(_require_mapping(row, path)) for row in payload]
    if isinstance(payload, Mapping):
        for key in ("particles", "observations", "rows", "data"):
            rows = payload.get(key)
            if isinstance(rows, list):
                return [dict(_require_mapping(row, path)) for row in rows]
        return [dict(payload)]
    raise ValueError(f"{path} must contain a JSON object, JSON array, or JSONL rows")


def _require_mapping(value: object, path: Path) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} rows must be JSON objects")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
