#!/usr/bin/env python3
"""Thin dcm2niix wrapper for the V1 MRI preprocessing lane."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys


def build_dcm2niix_command(
    dicom_dir: Path,
    output_dir: Path,
    *,
    case_id: str | None = None,
    compress: bool = True,
) -> list[str]:
    command = ["dcm2niix", "-z", "y" if compress else "n", "-o", str(output_dir)]
    if case_id:
        command.extend(["-f", case_id])
    command.append(str(dicom_dir))
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert one DICOM directory to NIfTI with dcm2niix.")
    parser.add_argument("--dicom-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--case-id", help="Optional output filename stem.")
    parser.add_argument("--no-compress", action="store_true", help="Write .nii instead of .nii.gz.")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without executing it.")
    args = parser.parse_args()

    if not args.dicom_dir.exists() or not args.dicom_dir.is_dir():
        print(f"DICOM directory not found: {args.dicom_dir}", file=sys.stderr)
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    command = build_dcm2niix_command(
        args.dicom_dir,
        args.output_dir,
        case_id=args.case_id,
        compress=not args.no_compress,
    )
    if args.dry_run:
        print(" ".join(command))
        return 0
    if shutil.which("dcm2niix") is None:
        print("dcm2niix is not available on PATH", file=sys.stderr)
        return 2
    subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
