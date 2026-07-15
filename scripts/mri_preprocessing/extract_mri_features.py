#!/usr/bin/env python3
"""Extract cached V1 MRI feature rows from per-case mask/intensity metadata."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mri_ingestion.features.case_features import extract_feature_rows
from experiments.mri_ingestion.schemas import read_jsonl, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build data/curated/features/mri_features.jsonl from cached MRI metadata."
    )
    parser.add_argument("--input", required=True, type=Path, help="JSONL metadata table.")
    parser.add_argument("--output", required=True, type=Path, help="Output feature JSONL path.")
    args = parser.parse_args()

    rows = extract_feature_rows(read_jsonl(args.input))
    write_jsonl(args.output, rows)
    print(f"wrote {len(rows)} MRI feature rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
