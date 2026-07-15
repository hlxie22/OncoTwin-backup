#!/usr/bin/env python3
"""Validate a normalized V1-D1 prior-stack eval cohort."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.prior_stack.v1_cohort_validation import validate_v1_prior_eval_cohort


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a normalized V1-D1 real-data prior eval cohort."
    )
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument(
        "--allow-demo-data",
        action="store_true",
        help="Allow demo/synthetic/fixture rows for smoke tests only.",
    )
    parser.add_argument(
        "--require-in-scope",
        action="store_true",
        help="Require at least one V1-A TNBC + chemotherapy case.",
    )
    parser.add_argument(
        "--min-in-scope-cases",
        type=int,
        default=1,
        help="Minimum V1-A in-scope cases when --require-in-scope is set.",
    )
    parser.add_argument(
        "--require-sidecars",
        action="store_true",
        help="Require <cohort>.summary.json and <cohort>.exclusions.jsonl.",
    )
    parser.add_argument(
        "--cohort-summary",
        type=Path,
        help="Optional explicit cohort-builder summary JSON path.",
    )
    parser.add_argument(
        "--exclusions",
        type=Path,
        help="Optional explicit cohort-builder exclusions JSONL path.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the validation result as machine-readable JSON.",
    )
    args = parser.parse_args()

    try:
        result = validate_v1_prior_eval_cohort(
            args.cohort,
            allow_demo_data=args.allow_demo_data,
            require_in_scope=args.require_in_scope,
            min_in_scope_cases=args.min_in_scope_cases,
            require_sidecars=args.require_sidecars,
            cohort_summary_path=args.cohort_summary,
            exclusions_path=args.exclusions,
        )
    except Exception as exc:  # pragma: no cover - CLI presentation only.
        print(f"Validation failed: {exc}", file=sys.stderr)
        return 2

    payload = result.as_dict()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    print(
        "Validation passed: "
        f"{result.case_count} loadable cases, "
        f"{result.in_scope_case_count} V1-A in-scope cases."
    )
    print("Data origins:")
    for origin, count in result.data_origin_counts.items():
        print(f"- {origin}: {count}")
    print(f"Summary sidecar: {result.summary_path or 'not found'}")
    print(f"Exclusions sidecar: {result.exclusions_path or 'not found'}")
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"- {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
