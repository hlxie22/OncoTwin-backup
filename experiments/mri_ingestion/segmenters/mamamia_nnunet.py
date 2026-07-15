"""Replaceable MAMA-MIA/nnU-Net adapter helpers.

The prior stack consumes cached feature tables, not nnU-Net internals. This
module keeps the command shape and local preflight checks in one place so the
heavy inference runtime can be swapped or smoke-tested independently.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
from typing import Iterable, Sequence


DEFAULT_DATASET_ID = "101"
DEFAULT_CONFIGURATION = "3d_fullres"
DEFAULT_PREDICTOR = "nnUNetv2_predict"
NNUNET_RESULTS_ENV = "nnUNet_results"


@dataclass(frozen=True)
class MAMAMIANnUNetConfig:
    dataset_id: str = DEFAULT_DATASET_ID
    configuration: str = DEFAULT_CONFIGURATION
    predictor: str = DEFAULT_PREDICTOR
    folds: tuple[str, ...] = ()
    checkpoint: str | None = None
    trainer: str | None = None
    plans: str | None = None


def build_predict_command(
    input_dir: Path,
    output_dir: Path,
    *,
    config: MAMAMIANnUNetConfig | None = None,
    extra_args: Sequence[str] = (),
) -> list[str]:
    """Build the nnUNetv2_predict command used by the MAMA-MIA smoke lane."""

    resolved = config or MAMAMIANnUNetConfig()
    command = [
        resolved.predictor,
        "-i",
        str(input_dir),
        "-o",
        str(output_dir),
        "-d",
        str(resolved.dataset_id),
        "-c",
        resolved.configuration,
    ]
    if resolved.folds:
        command.extend(["-f", *resolved.folds])
    if resolved.checkpoint:
        command.extend(["-chk", resolved.checkpoint])
    if resolved.trainer:
        command.extend(["-tr", resolved.trainer])
    if resolved.plans:
        command.extend(["-p", resolved.plans])
    command.extend(extra_args)
    return command


def validate_prediction_inputs(
    input_dir: Path,
    output_dir: Path,
    *,
    require_compressed_nifti: bool = True,
) -> tuple[Path, ...]:
    """Fail fast on common local setup issues before launching nnU-Net."""

    if not input_dir.exists():
        raise FileNotFoundError(f"nnU-Net input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise ValueError(f"nnU-Net input path must be a directory: {input_dir}")
    input_files = tuple(sorted(path for path in input_dir.iterdir() if path.is_file()))
    if not input_files:
        raise ValueError(f"nnU-Net input directory contains no files: {input_dir}")
    if require_compressed_nifti:
        invalid = [path.name for path in input_files if not path.name.endswith(".nii.gz")]
        if invalid:
            raise ValueError(
                "nnU-Net inputs must be compressed NIfTI files (*.nii.gz): "
                + ", ".join(invalid)
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    if not os.access(output_dir, os.W_OK):
        raise PermissionError(f"nnU-Net output directory is not writable: {output_dir}")
    return input_files


def environment_warnings(
    *,
    predictor: str = DEFAULT_PREDICTOR,
    nnunet_results: Path | None = None,
    dataset_id: str = DEFAULT_DATASET_ID,
    require_weights: bool = False,
) -> tuple[str, ...]:
    """Return actionable setup warnings without importing or invoking nnU-Net."""

    warnings: list[str] = []
    if shutil.which(predictor) is None:
        warnings.append(f"{predictor} is not available on PATH")

    results_value = nnunet_results or _env_path(NNUNET_RESULTS_ENV)
    if results_value is None:
        warnings.append(f"{NNUNET_RESULTS_ENV} is not set")
    elif require_weights and not _dataset_weight_dirs(results_value, dataset_id):
        warnings.append(
            f"no nnU-Net model weights found for Dataset{dataset_id} under {results_value}"
        )
    return tuple(warnings)


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


def _dataset_weight_dirs(root: Path, dataset_id: str) -> tuple[Path, ...]:
    if not root.exists():
        return ()
    prefixes = (f"Dataset{dataset_id}", f"{int(dataset_id):03d}") if dataset_id.isdigit() else (dataset_id,)
    matches: list[Path] = []
    for path in root.rglob("*"):
        if path.is_dir() and any(path.name.startswith(prefix) for prefix in prefixes):
            matches.append(path)
    return tuple(matches)


def command_text(command: Iterable[str]) -> str:
    """Render a command for dry-run logs without shell-specific quoting magic."""

    return " ".join(str(part) for part in command)
