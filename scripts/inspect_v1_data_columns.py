#!/usr/bin/env python3
"""Inspect source-table columns before building a V1-D1 eval cohort."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.prior_builder.v1_eval_cohort_builder import inspect_v1_data_columns


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect columns in a candidate V1 prior-stack data table."
    )
    parser.add_argument("path", type=Path, help="CSV/JSON/JSONL source table.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Write the profile as machine-readable JSON.",
    )
    args = parser.parse_args()

    inspection = inspect_v1_data_columns(args.path)
    payload = inspection.as_dict()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    print(f"Path: {inspection.path}")
    print(f"Rows: {inspection.row_count}")
    print("Columns:")
    for column in inspection.columns:
        print(f"- {column}: {inspection.non_empty_counts[column]} non-empty")
    if inspection.suspected_roles:
        print("Suspected V1 roles:")
        for role, columns in inspection.suspected_roles.items():
            print(f"- {role}: {', '.join(columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
