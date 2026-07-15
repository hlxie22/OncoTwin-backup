#!/usr/bin/env python3
"""Apply fail-closed V1 MRI QC gates to an existing feature JSONL table."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mri_ingestion.features.qc import qc_mri_feature_record
from experiments.mri_ingestion.schemas import read_jsonl, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="QC a V1 MRI feature JSONL table.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows = [qc_mri_feature_record(row) for row in read_jsonl(args.input)]
    write_jsonl(args.output, rows)
    failed = sum(row.get("mri_feature_status") == "failed" for row in rows)
    print(f"wrote {len(rows)} QC rows to {args.output}; failed={failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
