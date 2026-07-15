#!/usr/bin/env python3
"""Prepare compressed NIfTI files for MAMA-MIA/nnU-Net inference."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mri_ingestion.schemas import read_jsonl, require_case_id, write_jsonl


def prepare_nnunet_inputs(
    manifest_path: Path,
    output_dir: Path,
    *,
    copy_files: bool = False,
) -> list[dict[str, object]]:
    rows = read_jsonl(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared: list[dict[str, object]] = []
    for row in rows:
        case_id = _safe_case_id(require_case_id(row))
        source_image = _source_image(row)
        if source_image is None:
            raise ValueError(f"missing source_image/image_path for case_id {case_id}")
        if not source_image.exists():
            raise FileNotFoundError(source_image)
        if not source_image.name.endswith(".nii.gz"):
            raise ValueError(f"MAMA-MIA nnU-Net inputs must be .nii.gz: {source_image}")
        target = output_dir / f"{case_id}_0000.nii.gz"
        if target.exists() or target.is_symlink():
            target.unlink()
        if copy_files:
            shutil.copy2(source_image, target)
        else:
            target.symlink_to(source_image.resolve())
        prepared.append({"case_id": case_id, "source_image": str(source_image), "nnunet_image": str(target)})
    return prepared


def _source_image(row: dict[str, object]) -> Path | None:
    for field in ("source_image", "image_path", "input_image"):
        value = row.get(field)
        if value not in (None, ""):
            return Path(str(value))
    return None


def _safe_case_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare MAMA-MIA/nnU-Net case inputs from JSONL manifest.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prepared-manifest", type=Path)
    parser.add_argument("--copy", action="store_true", help="Copy files instead of creating symlinks.")
    args = parser.parse_args()

    prepared = prepare_nnunet_inputs(args.manifest, args.output_dir, copy_files=args.copy)
    if args.prepared_manifest:
        write_jsonl(args.prepared_manifest, prepared)
    else:
        for row in prepared:
            print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
