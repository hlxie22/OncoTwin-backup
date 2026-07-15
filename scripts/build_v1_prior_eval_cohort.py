#!/usr/bin/env python3
"""CLI wrapper for building the V1-D1 prior-stack eval cohort."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.prior_builder.v1_eval_cohort_builder import (
    DEFAULT_OUTPUT_PATH,
    build_v1_prior_eval_cohort,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a normalized V1-D1 real-data prior eval cohort."
    )
    parser.add_argument(
        "--measurements",
        required=True,
        type=Path,
        help="CSV/JSON/JSONL longitudinal measurement table.",
    )
    parser.add_argument(
        "--clinical",
        type=Path,
        help="Optional CSV/JSON/JSONL clinical/context table keyed by case_id/patient_id.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output cohort JSONL path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--default-treatment-context",
        help="Fill missing treatment context only when source documentation supports it.",
    )
    parser.add_argument(
        "--data-origin",
        help="Dataset/provenance label to write when source rows omit data_origin.",
    )
    parser.add_argument(
        "--use-nominal-ispy2-days",
        action="store_true",
        help="Map T0/T1/T2/T3 labels to approximate I-SPY2 nominal days.",
    )
    parser.add_argument(
        "--allow-demo-data",
        action="store_true",
        help="Allow rows with demo/synthetic/fixture tokens. Use only for smoke tests.",
    )
    args = parser.parse_args()

    result = build_v1_prior_eval_cohort(
        args.measurements,
        output_path=args.output,
        clinical_path=args.clinical,
        default_treatment_context=args.default_treatment_context,
        data_origin=args.data_origin,
        use_nominal_ispy2_days=args.use_nominal_ispy2_days,
        allow_demo_data=args.allow_demo_data,
    )
    print(f"Cohort: {result.cohort_path}")
    print(f"Exclusions: {result.exclusions_path}")
    print(f"Summary: {result.summary_path}")
    print(
        "Included "
        f"{result.summary['included_rows']}/{result.summary['total_input_rows']} "
        "input rows as V1-A evaluable cases."
    )
    if result.summary["excluded_rows"]:
        print("Excluded reason counts:")
        for reason, count in result.summary["excluded_reason_counts"].items():
            print(f"- {reason}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
