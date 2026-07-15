#!/usr/bin/env python3
"""CLI wrapper for merging cached MRI features into a V1 eval cohort."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.prior_builder.v1_mri_feature_merge import (
    DEFAULT_OUTPUT_PATH,
    merge_mri_features_into_v1_cohort,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge cached MRI feature rows into a normalized V1-D1 cohort."
    )
    parser.add_argument(
        "--cohort",
        required=True,
        type=Path,
        help="Normalized V1-D1 cohort JSON/JSONL/CSV path.",
    )
    parser.add_argument(
        "--mri-features",
        required=True,
        type=Path,
        help="Case-level MRI feature table as JSON/JSONL/CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output merged cohort JSONL path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--drop-missing-features",
        action="store_true",
        help="Drop cohort rows without a matching MRI feature row.",
    )
    args = parser.parse_args()

    result = merge_mri_features_into_v1_cohort(
        args.cohort,
        args.mri_features,
        output_path=args.output,
        drop_missing_features=args.drop_missing_features,
    )

    print(f"Merged cohort: {result.output_path}")
    print(f"Summary: {result.summary_path}")
    print(
        "Matched "
        f"{result.summary['matched_feature_rows']}/"
        f"{result.summary['input_cohort_rows']} cohort rows to MRI features."
    )
    if result.summary["missing_feature_rows"]:
        print(f"Missing MRI features: {result.summary['missing_feature_rows']}")
    if result.summary["feature_status_counts"]:
