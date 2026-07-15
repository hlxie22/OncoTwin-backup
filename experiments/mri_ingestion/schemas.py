"""Shared schemas and JSONL helpers for the V1 MRI preprocessing lane."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence


CASE_ID_FIELDS = ("case_id", "patient_id", "subject_id", "participant_id", "ptid")
QC_LABELS = ("high", "medium", "low", "failed", "unknown")
UNKNOWN_TOKENS = {
    "",
    "unknown",
    "unspecified",
    "not specified",
    "not available",
    "not assessed",
    "n/a",
    "na",
    "none",
    "null",
}


@dataclass(frozen=True)
class SegmentationRequest:
    """Stable request shape for replaceable tumor segmentation adapters."""

    case_id: str
    image_path: Path
    output_dir: Path
    model_config: Mapping[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "image_path": str(self.image_path),
            "output_dir": str(self.output_dir),
            "model_config": dict(self.model_config),
        }


@dataclass(frozen=True)
class SegmentationResult:
    """Stable result shape returned by segmentation adapters."""

    case_id: str
    segmentation_path: Path
    inference_metadata: Mapping[str, object]
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "segmentation_path": str(self.segmentation_path),
            "inference_metadata": dict(self.inference_metadata),
            "warnings": list(self.warnings),
        }


def read_jsonl(path: Path) -> list[dict[str, object]]:
    """Read a JSONL table and require every non-empty row to be a JSON object."""

    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        rows.append(dict(payload))
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    """Write stable JSONL with parent directory creation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def case_id_from_row(row: Mapping[str, object]) -> str:
    for field in CASE_ID_FIELDS:
        value = row.get(field)
        if present(value):
            return str(value).strip()
    return ""


def require_case_id(row: Mapping[str, object]) -> str:
    case_id = case_id_from_row(row)
    if not case_id:
        raise ValueError("MRI preprocessing rows require case_id or patient_id")
    return case_id


def normalize_qc_label(value: object) -> str:
    if not present(value):
        return "unknown"
    text = normalize_text(value)
    if text in QC_LABELS:
        return text
    if any(token in text for token in ("fail", "empty", "all zero", "no mask")):
        return "failed"
    if any(token in text for token in ("low", "poor", "artifact", "motion", "review")):
        return "low"
    if any(token in text for token in ("good", "excellent", "pass", "passed", "high")):
        return "high"
    if "medium" in text or "moderate" in text:
        return "medium"
    return "unknown"


def present(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return normalize_text(value) not in UNKNOWN_TOKENS
    return True


def normalize_text(value: object) -> str:
    return " ".join(str(value).lower().replace("_", " ").split())


def as_text_list(value: object) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def stable_unique(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            output.append(value)
            seen.add(value)
    return tuple(output)
