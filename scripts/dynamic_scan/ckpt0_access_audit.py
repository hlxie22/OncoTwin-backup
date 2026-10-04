#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DATA_SUFFIXES = (
    ".csv",
    ".csv.gz",
    ".tsv",
    ".tsv.gz",
    ".txt",
    ".txt.gz",
    ".maf",
    ".maf.gz",
    ".parquet",
    ".pq",
    ".json",
    ".jsonl",
    ".ndjson",
    ".xlsx",
    ".xls",
    ".zip",
    ".tar",
    ".tar.gz",
    ".tgz",
)

SKIP_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "checkpoints",
    "wandb",
}

DATASET_KEYWORDS = (
    "genie",
    "chord",
    "bpc",
    "dfci",
    "dana-farber",
    "dana_farber",
    "danafarber",
    "vanderbilt",
    "vicc",
)

PATIENT_ID_EXACT = {
    "patient_id",
    "patientid",
    "patient",
    "pt_id",
    "ptid",
    "subject_id",
    "subjectid",
    "person_id",
    "record_id",
    "recordid",
    "deid_patient_id",
    "deidentified_patient_id",
}

SAMPLE_ID_EXACT = {
    "sample_id",
    "sampleid",
    "sample",
    "specimen_id",
    "specimenid",
    "tumor_sample_barcode",
    "tumor_sample_id",
    "aliquot_id",
}

INSTITUTION_TOKENS = (
    "institution",
    "center",
    "centre",
    "hospital",
    "facility",
    "site_id",
    "site_name",
    "institution_id",
)

TIME_TOKENS = (
    "date",
    "datetime",
    "timestamp",
    "time",
    "start",
    "stop",
    "end_date",
    "report",
    "result",
    "available",
    "availability",
    "collection",
    "collected",
    "diagnos",
    "death",
    "deceased",
    "scan_date",
    "imaging_date",
    "treatment_date",
)

PANEL_TOKENS = (
    "panel",
    "assay",
    "sequencing",
    "seq_",
    "platform",
    "coverage",
    "covered",
    "test_name",
)

GENE_TOKENS = (
    "gene",
    "hugo",
    "symbol",
    "chromosome",
    "variant",
    "mutation",
    "alteration",
    "cna",
    "fusion",
)

IMAGING_TOKENS = (
    "scan",
    "radiol",
    "imaging",
    "modality",
    "lesion",
    "metasta",
    "response",
    "progress",
    "recist",
    "anatomic",
    "body_region",
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def norm_col(name: str) -> str:
    x = re.sub(r"[^a-zA-Z0-9]+", "_", str(name).strip().lower())
    return x.strip("_")


def human_gib(n: int | float | None) -> str:
    if n is None:
        return "NA"
    return f"{float(n) / (1024 ** 3):.3f}"


def safe_rel(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except Exception:
        return str(path.resolve())


def data_kind(path: Path) -> str | None:
    name = path.name.lower()
    for suffix in sorted(DATA_SUFFIXES, key=len, reverse=True):
        if name.endswith(suffix):
            return suffix.lstrip(".")
    return None


def guess_dataset(path: Path) -> str:
    p = str(path).lower()

    if any(x in p for x in ("dfci", "dana-farber", "dana_farber", "danafarber")):
        return "BPC_DFCI"

    if any(x in p for x in ("vanderbilt", "vicc")):
        return "BPC_VICC"

    if "bpc" in p and "msk" in p:
        return "BPC_MSK"

    if "chord" in p:
        return "CHORD"

    if "genie" in p and "bpc" not in p:
        return "GENIE"

    if "bpc" in p:
        return "BPC_UNASSIGNED"

    return "UNKNOWN"


def classify_columns(columns: Iterable[str]) -> dict[str, list[str]]:
    original = [str(x) for x in columns]
    normalized = {c: norm_col(c) for c in original}

    out: dict[str, list[str]] = {
        "patient_id": [],
        "sample_id": [],
        "institution": [],
        "timestamp": [],
        "panel_or_assay": [],
        "gene_or_variant": [],
        "imaging_or_response": [],
    }

    for col, n in normalized.items():
        if (
            n in PATIENT_ID_EXACT
            or n.endswith("_patient_id")
            or n.startswith("patient_id_")
        ):
            out["patient_id"].append(col)

        if (
            n in SAMPLE_ID_EXACT
            or n.endswith("_sample_id")
            or n.endswith("_specimen_id")
        ):
            out["sample_id"].append(col)

        if any(tok in n for tok in INSTITUTION_TOKENS):
            out["institution"].append(col)

        if any(tok in n for tok in TIME_TOKENS):
            out["timestamp"].append(col)

        if any(tok in n for tok in PANEL_TOKENS):
            out["panel_or_assay"].append(col)

        if any(tok in n for tok in GENE_TOKENS):
            out["gene_or_variant"].append(col)

        if any(tok in n for tok in IMAGING_TOKENS):
            out["imaging_or_response"].append(col)

    return out


def discover_roots(repo_root: Path, explicit_roots: list[str]) -> list[Path]:
    roots: list[Path] = []

    def add_root(p: str | Path) -> None:
        path = Path(p).expanduser().resolve()
        if path.exists() and path.is_dir() and path not in roots:
            roots.append(path)

    for x in explicit_roots:
        add_root(x)

    env_roots = os.environ.get("ONCOTWIN_DATA_ROOTS", "").strip()
    if env_roots:
        for x in env_roots.split(os.pathsep):
            if x.strip():
                add_root(x.strip())

    common = [
        repo_root / "data",
        repo_root / "datasets",
        repo_root / "raw",
        repo_root / "data_raw",
        repo_root / "resources" / "data",
        repo_root / "artifacts" / "data",
    ]
    for p in common:
        add_root(p)

    try:
        for child in repo_root.iterdir():
            if not child.is_dir():
                continue
            name = child.name.lower()
            if any(
                token in name
                for token in ("data", "dataset", "genie", "chord", "bpc")
            ):
                add_root(child)
    except OSError:
        pass

    # If no obvious source directory exists, search the repository itself,
    # but repository-root scanning is filtered to dataset-looking paths.
    if not roots:
        add_root(repo_root)

    # Remove nested duplicates when a parent root is already present.
    unique: list[Path] = []
    for candidate in sorted(roots, key=lambda p: (len(p.parts), str(p))):
        if any(
            candidate == parent or parent in candidate.parents
            for parent in unique
        ):
            continue
        unique.append(candidate)

    return unique


def iter_data_files(
    roots: list[Path],
    repo_root: Path,
    max_files: int = 50000,
) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()

    for root in roots:
        root_is_repo = root.resolve() == repo_root.resolve()

        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                d
                for d in dirnames
                if d not in SKIP_DIR_NAMES
                and not d.startswith(".cache")
            ]

            for filename in filenames:
                path = Path(dirpath) / filename
                if data_kind(path) is None:
                    continue

                if root_is_repo:
                    low = str(path).lower()
                    if not any(k in low for k in DATASET_KEYWORDS):
                        continue

                rp = path.resolve()
                if rp in seen:
                    continue

                seen.add(rp)
                files.append(rp)

                if len(files) >= max_files:
                    return sorted(files)

    return sorted(files)


def content_fingerprint(path: Path) -> str:
    size = path.stat().st_size
    h = hashlib.sha256()

    # Full SHA256 for modest files.
    if size <= 64 * 1024 * 1024:
        with path.open("rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
        return "sha256:" + h.hexdigest()

    # Large files: low-I/O diagnostic fingerprint, explicitly marked sampled.
    h.update(str(size).encode("ascii"))
    with path.open("rb") as f:
        first = f.read(1024 * 1024)
        h.update(first)

        if size > 1024 * 1024:
            f.seek(max(0, size - 1024 * 1024))
            last = f.read(1024 * 1024)
            h.update(last)

    return "sampled_sha256:" + h.hexdigest()


def open_text(path: Path):
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8-sig", errors="replace")
    return path.open("rt", encoding="utf-8-sig", errors="replace")


def infer_delimiter(path: Path, text_sample: str) -> str:
    name = path.name.lower()

    if name.endswith((".tsv", ".tsv.gz", ".maf", ".maf.gz")):
        return "\t"

    if name.endswith((".csv", ".csv.gz")):
        return ","

    try:
        dialect = csv.Sniffer().sniff(
            text_sample,
            delimiters="\t,|;",
        )
        return dialect.delimiter
    except Exception:
        return "\t"


def inspect_delimited(path: Path, max_rows: int = 200) -> dict[str, Any]:
    lines: list[str] = []

    with open_text(path) as f:
        for line in f:
            # MAF and some genomics tables start with comment metadata.
            if not lines and line.startswith("#"):
                continue
            lines.append(line)
            if len(lines) >= max_rows + 1:
                break

    if not lines:
        return {
            "columns": [],
            "column_count": 0,
            "sample_rows_read": 0,
            "candidate_columns": classify_columns([]),
        }

    sample_text = "".join(lines[:20])
    delimiter = infer_delimiter(path, sample_text)
    reader = csv.reader(lines, delimiter=delimiter)
    rows = list(reader)

    if not rows:
        columns: list[str] = []
        data_rows: list[list[str]] = []
    else:
        columns = [
            str(x).strip() if str(x).strip() else f"unnamed_{i}"
            for i, x in enumerate(rows[0])
        ]
        data_rows = rows[1:]

    nonempty_fraction: dict[str, float] = {}
    for i, col in enumerate(columns):
        if not data_rows:
            nonempty_fraction[col] = 0.0
            continue

        observed = 0
        for row in data_rows:
            if i < len(row) and str(row[i]).strip() not in ("", "NA", "NaN", "nan", "NULL", "null"):
                observed += 1

        nonempty_fraction[col] = observed / len(data_rows)

    return {
        "delimiter": "\\t" if delimiter == "\t" else delimiter,
        "columns": columns,
        "column_count": len(columns),
        "sample_rows_read": len(data_rows),
        "candidate_columns": classify_columns(columns),
        "sample_nonempty_fraction": nonempty_fraction,
        "row_count": None,
        "row_count_note": "not counted for delimited text during low-I/O audit",
    }


def inspect_parquet(path: Path) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        return {
            "error": f"pyarrow unavailable: {exc}",
            "columns": [],
            "candidate_columns": classify_columns([]),
        }

    pf = pq.ParquetFile(path)
    columns = list(pf.schema_arrow.names)
    return {
        "columns": columns,
        "column_count": len(columns),
        "row_count": pf.metadata.num_rows,
        "row_groups": pf.metadata.num_row_groups,
        "candidate_columns": classify_columns(columns),
    }


def inspect_json(path: Path, max_rows: int = 100) -> dict[str, Any]:
    columns: set[str] = set()
    rows = 0
    name = path.name.lower()

    if name.endswith((".jsonl", ".ndjson")):
        with path.open("rt", encoding="utf-8-sig", errors="replace") as f:
            for line in f:
                if rows >= max_rows:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    columns.update(str(k) for k in obj.keys())
                rows += 1
    else:
        if path.stat().st_size > 16 * 1024 * 1024:
            return {
                "columns": [],
                "candidate_columns": classify_columns([]),
                "note": "JSON >16MB not loaded during low-I/O audit",
            }

        with path.open("rt", encoding="utf-8-sig", errors="replace") as f:
            obj = json.load(f)

        if isinstance(obj, dict):
            columns.update(str(k) for k in obj.keys())
            rows = 1
        elif isinstance(obj, list):
            for item in obj[:max_rows]:
                if isinstance(item, dict):
                    columns.update(str(k) for k in item.keys())
                rows += 1

    cols = sorted(columns)
    return {
        "columns": cols,
        "column_count": len(cols),
        "sample_rows_read": rows,
        "candidate_columns": classify_columns(cols),
    }


def inspect_excel(path: Path) -> dict[str, Any]:
    try:
        import openpyxl
    except Exception as exc:
        return {
            "error": f"openpyxl unavailable: {exc}",
            "columns": [],
            "candidate_columns": classify_columns([]),
        }

    wb = openpyxl.load_workbook(
        filename=path,
        read_only=True,
        data_only=True,
    )

    sheets: dict[str, Any] = {}
    all_columns: list[str] = []

    for ws in wb.worksheets[:20]:
        header = None
        for row in ws.iter_rows(min_row=1, max_row=10, values_only=True):
            values = [
                "" if x is None else str(x).strip()
                for x in row
            ]
            if any(values):
                header = values
                break

        cols = header or []
        sheets[ws.title] = {
            "columns": cols,
            "column_count": len(cols),
            "candidate_columns": classify_columns(cols),
        }
        all_columns.extend(cols)

    wb.close()

    return {
        "sheets": sheets,
        "columns": sorted(set(all_columns)),
        "candidate_columns": classify_columns(sorted(set(all_columns))),
    }


def inspect_archive(path: Path, limit: int = 200) -> dict[str, Any]:
    members: list[str] = []

    name = path.name.lower()

    if name.endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            members = names[:limit]
            return {
                "archive_member_count": len(names),
                "archive_members_first_200": members,
                "candidate_columns": classify_columns([]),
            }

    if name.endswith((".tar", ".tar.gz", ".tgz")):
        with tarfile.open(path, "r:*") as tf:
            count = 0
            for member in tf:
                count += 1
                if len(members) < limit:
                    members.append(member.name)

        return {
            "archive_member_count": count,
            "archive_members_first_200": members,
            "candidate_columns": classify_columns([]),
        }

    return {
        "candidate_columns": classify_columns([]),
    }


def inspect_schema(path: Path, kind: str) -> dict[str, Any]:
    try:
        if kind in {
            "csv",
            "csv.gz",
            "tsv",
            "tsv.gz",
            "txt",
            "txt.gz",
            "maf",
            "maf.gz",
        }:
            return inspect_delimited(path)

        if kind in {"parquet", "pq"}:
            return inspect_parquet(path)

        if kind in {"json", "jsonl", "ndjson"}:
            return inspect_json(path)

        if kind in {"xlsx", "xls"}:
            return inspect_excel(path)

        if kind in {"zip", "tar", "tar.gz", "tgz"}:
            return inspect_archive(path)

        return {
            "columns": [],
            "candidate_columns": classify_columns([]),
        }

    except Exception as exc:
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "columns": [],
            "candidate_columns": classify_columns([]),
        }


def inspect_file(path: Path, repo_root: Path) -> dict[str, Any]:
    stat = path.stat()
    kind = data_kind(path) or "unknown"

    result = {
        "path": str(path),
        "path_display": safe_rel(path, repo_root),
        "dataset_guess": guess_dataset(path),
        "kind": kind,
        "size_bytes": stat.st_size,
        "size_gib": round(stat.st_size / (1024 ** 3), 6),
        "mtime_utc": dt.datetime.fromtimestamp(
            stat.st_mtime,
            tz=dt.timezone.utc,
        ).isoformat(),
    }

    try:
        result["content_fingerprint"] = content_fingerprint(path)
    except Exception as exc:
        result["content_fingerprint"] = f"ERROR:{type(exc).__name__}:{exc}"

    result["schema"] = inspect_schema(path, kind)
    return result


def package_versions() -> dict[str, str | None]:
    packages = [
        "numpy",
        "pandas",
        "pyarrow",
        "torch",
        "torch-geometric",
        "scikit-learn",
        "xgboost",
        "lifelines",
        "sksurv",
        "scikit-survival",
        "polars",
        "duckdb",
        "openpyxl",
    ]

    out: dict[str, str | None] = {}

    for package in packages:
        try:
            out[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            out[package] = None
        except Exception:
            out[package] = "UNKNOWN"

    return out


def gpu_environment() -> dict[str, Any]:
    out: dict[str, Any] = {
        "torch_cuda_available": None,
        "torch_cuda_device_count": None,
        "torch_devices": [],
        "nvidia_smi": [],
    }

    try:
        import torch

        out["torch_cuda_available"] = bool(torch.cuda.is_available())
        out["torch_cuda_device_count"] = int(torch.cuda.device_count())

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            out["torch_devices"].append(
                {
                    "index": i,
                    "name": props.name,
                    "total_memory_gib": round(
                        props.total_memory / (1024 ** 3),
                        3,
                    ),
                    "compute_capability": f"{props.major}.{props.minor}",
                }
            )
    except Exception as exc:
        out["torch_error"] = f"{type(exc).__name__}: {exc}"

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            proc = subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=index,name,memory.total,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
            )

            for line in proc.stdout.splitlines():
                parts = [x.strip() for x in line.split(",")]
                if len(parts) >= 4:
                    out["nvidia_smi"].append(
                        {
                            "index": parts[0],
                            "name": parts[1],
                            "memory_total_mib": parts[2],
                            "driver_version": parts[3],
                        }
                    )
        except Exception as exc:
            out["nvidia_smi_error"] = f"{type(exc).__name__}: {exc}"

    return out


def disk_record(path: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    return {
        "path": str(path),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "total_gib": round(usage.total / (1024 ** 3), 3),
        "used_gib": round(usage.used / (1024 ** 3), 3),
        "free_gib": round(usage.free / (1024 ** 3), 3),
    }


def validate_policy(policy: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    roles = policy.get("dataset_roles", {})

    for name in ("BPC_DFCI", "BPC_VICC"):
        cfg = roles.get(name)

        if cfg is None:
            errors.append(f"missing required policy entry: {name}")
            continue

        if cfg.get("development_access") is not False:
            errors.append(f"{name} must have development_access=false")

        if cfg.get("supervised_dynamic_training") is not False:
            errors.append(f"{name} must not be used for supervised training")

        if cfg.get("hyperparameter_tuning") is not False:
            errors.append(f"{name} must not be used for tuning")

        if cfg.get("external_validation") is not True:
            errors.append(f"{name} must be marked external_validation=true")

    return errors


def summarize_files(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for rec in records:
        grouped[rec["dataset_guess"]].append(rec)

    for dataset, items in sorted(grouped.items()):
        candidate_counts = Counter()

        for item in items:
            candidates = item.get("schema", {}).get("candidate_columns", {})
            for key, values in candidates.items():
                if values:
                    candidate_counts[key] += 1

        summary[dataset] = {
            "file_count": len(items),
            "total_bytes": sum(x["size_bytes"] for x in items),
            "total_gib": round(
                sum(x["size_bytes"] for x in items) / (1024 ** 3),
                6,
            ),
            "files_with_schema_errors": sum(
                1
                for x in items
                if x.get("schema", {}).get("error")
            ),
            "files_with_patient_id_candidates": candidate_counts["patient_id"],
            "files_with_sample_id_candidates": candidate_counts["sample_id"],
            "files_with_institution_candidates": candidate_counts["institution"],
            "files_with_timestamp_candidates": candidate_counts["timestamp"],
            "files_with_panel_candidates": candidate_counts["panel_or_assay"],
            "files_with_gene_candidates": candidate_counts["gene_or_variant"],
            "files_with_imaging_candidates": candidate_counts["imaging_or_response"],
        }

    return summary


def sample_overlap_candidates(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Checkpoint 0 deliberately does not assert patient overlap from heuristics.
    This function tells us whether there are plausible ID columns that can be
    mapped exactly on the next pass.
    """
    out: dict[str, Any] = {}

    for dataset in (
        "GENIE",
        "CHORD",
        "BPC_MSK",
        "BPC_DFCI",
        "BPC_VICC",
        "BPC_UNASSIGNED",
    ):
        files = [
            r
            for r in records
            if r["dataset_guess"] == dataset
        ]

        candidate_files = []

        for rec in files:
            ids = (
                rec.get("schema", {})
                .get("candidate_columns", {})
                .get("patient_id", [])
            )

            if ids:
                candidate_files.append(
                    {
                        "path": rec["path_display"],
                        "patient_id_candidates": ids,
                    }
                )

        out[dataset] = {
            "files_with_patient_id_candidates": len(candidate_files),
            "candidate_files": candidate_files[:50],
            "exact_overlap_computed": False,
            "note": (
                "Exact cross-dataset overlap is intentionally deferred until "
                "the authoritative patient identifier fields are confirmed."
            ),
        }

    return out


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []

    lines.append("# Checkpoint 0 - Access, Environment, and Data Audit")
    lines.append("")
    lines.append(f"Generated: `{report['generated_utc']}`")
    lines.append("")
    lines.append(f"Status: **{report['checkpoint_status']}**")
    lines.append("")

    lines.append("## Environment")
    lines.append("")
    env = report["environment"]

    lines.append(f"- Python: `{env['python_version']}`")
    lines.append(f"- Executable: `{env['python_executable']}`")
    lines.append(f"- Platform: `{env['platform']}`")
    lines.append(
        f"- Torch CUDA available: `{env['gpu'].get('torch_cuda_available')}`"
    )
    lines.append(
        f"- Torch CUDA devices: `{env['gpu'].get('torch_cuda_device_count')}`"
    )

    for device in env["gpu"].get("torch_devices", []):
        lines.append(
            "- GPU "
            f"{device['index']}: `{device['name']}`, "
            f"{device['total_memory_gib']} GiB, "
            f"CC {device['compute_capability']}"
        )

    lines.append("")
    lines.append("## Storage")
    lines.append("")
    lines.append("| Path | Total GiB | Used GiB | Free GiB |")
    lines.append("|---|---:|---:|---:|")

    for d in report["disk"]:
        lines.append(
            f"| `{d['path']}` | {d['total_gib']} | "
            f"{d['used_gib']} | {d['free_gib']} |"
        )

    lines.append("")
    lines.append("## Data roots scanned")
    lines.append("")

    for root in report["data_roots"]:
        lines.append(f"- `{root}`")

    lines.append("")
    lines.append("## Dataset detection")
    lines.append("")
    lines.append(
        "| Dataset guess | Files | Size GiB | Patient-ID files | "
        "Timestamp files | Panel files | Imaging/response files |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|"
    )

    for dataset, s in report["dataset_summary"].items():
        lines.append(
            f"| {dataset} | {s['file_count']} | {s['total_gib']} | "
            f"{s['files_with_patient_id_candidates']} | "
            f"{s['files_with_timestamp_candidates']} | "
            f"{s['files_with_panel_candidates']} | "
            f"{s['files_with_imaging_candidates']} |"
        )

    lines.append("")
    lines.append("## Required source detection")
    lines.append("")

    for dataset, detected in report["required_source_detection"].items():
        mark = "YES" if detected else "NO"
        lines.append(f"- {dataset}: **{mark}**")

    lines.append("")
    lines.append("Important: `NO` means **not automatically detected by path/name**,")
    lines.append("not necessarily that the cohort is absent. A combined BPC file may")
    lines.append("contain several institutions and require site-column mapping.")
    lines.append("")

    lines.append("## External-validation quarantine")
    lines.append("")

    external = report["external_quarantine_files"]

    if external:
        for path in external:
            lines.append(f"- `{path}`")
    else:
        lines.append("- No DFCI/Vanderbilt/VICC file was automatically identified.")

    lines.append("")
    lines.append("These files are inventory-only in Checkpoint 0 and must not be")
    lines.append("used for model fitting, endpoint-rule selection, or tuning.")
    lines.append("")

    lines.append("## Candidate schema mapping")
    lines.append("")

    shown = 0

    for rec in report["files"]:
        cand = rec.get("schema", {}).get("candidate_columns", {})
        interesting = any(cand.get(k) for k in cand)

        if not interesting:
            continue

        shown += 1

        if shown > 80:
            lines.append("")
            lines.append(
                "_Additional candidate mappings are available in "
                "`schema_inventory.jsonl`._"
            )
            break

        lines.append(
            f"### {rec['dataset_guess']}: `{rec['path_display']}`"
        )
        lines.append("")

        for key in (
            "patient_id",
            "sample_id",
            "institution",
            "timestamp",
            "panel_or_assay",
            "gene_or_variant",
            "imaging_or_response",
        ):
            values = cand.get(key, [])
            if values:
                lines.append(
                    f"- {key}: "
                    + ", ".join(f"`{x}`" for x in values)
                )

        if rec.get("schema", {}).get("row_count") is not None:
            lines.append(
                f"- rows: `{rec['schema']['row_count']}`"
            )

        lines.append("")

    lines.append("## Patient overlap status")
    lines.append("")
    lines.append(
        "Exact patient overlap has **not** been guessed from column names. "
        "The audit identifies candidate patient-ID fields first. We will "
        "compute exact overlap only after confirming the authoritative ID "
        "column(s), preventing an accidental comparison of sample IDs, "
        "specimen IDs, or institution-local row IDs."
    )
    lines.append("")

    lines.append("## Policy validation")
    lines.append("")

    if report["policy_errors"]:
        for error in report["policy_errors"]:
            lines.append(f"- ERROR: {error}")
    else:
        lines.append(
            "- PASS: DFCI and Vanderbilt/VICC are locked to external validation."
        )

    lines.append("")
    lines.append("## Next decision")
    lines.append("")

    if report["checkpoint_status"] == "PASS_SOURCE_DETECTION":
        lines.append(
            "All required source families were automatically identifiable. "
            "Review the candidate patient/timestamp/site fields and then "
            "perform the authoritative schema mapping and overlap audit."
        )
    else:
        lines.append(
            "One or more source families could not be identified automatically. "
            "Use this report to identify the correct data roots or combined "
            "BPC site field. Do not download or restructure anything yet."
        )

    lines.append("")
    lines.append("## Artifact files")
    lines.append("")
    lines.append("- `artifacts/checkpoint0/audit.json`")
    lines.append("- `artifacts/checkpoint0/audit.md`")
    lines.append("- `artifacts/checkpoint0/file_inventory.tsv`")
    lines.append("- `artifacts/checkpoint0/schema_inventory.jsonl`")
    lines.append("- `artifacts/checkpoint0/external_test_quarantine.txt`")
    lines.append("- `artifacts/checkpoint0/environment.json`")
    lines.append("- `artifacts/handoff/checkpoint_00.json`")
    lines.append("")

    return "\n".join(lines) + "\n"


def write_outputs(
    report: dict[str, Any],
    output_dir: Path,
    handoff_path: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    handoff_path.parent.mkdir(parents=True, exist_ok=True)

    audit_json = output_dir / "audit.json"
    audit_md = output_dir / "audit.md"
    inventory_tsv = output_dir / "file_inventory.tsv"
    schema_jsonl = output_dir / "schema_inventory.jsonl"
    quarantine_txt = output_dir / "external_test_quarantine.txt"
    environment_json = output_dir / "environment.json"

    audit_json.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    audit_md.write_text(
        render_markdown(report),
        encoding="utf-8",
    )

    with inventory_tsv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "dataset_guess",
                "kind",
                "size_bytes",
                "size_gib",
                "path",
                "content_fingerprint",
                "schema_error",
            ]
        )

        for rec in report["files"]:
            writer.writerow(
                [
                    rec["dataset_guess"],
                    rec["kind"],
                    rec["size_bytes"],
                    rec["size_gib"],
                    rec["path_display"],
                    rec["content_fingerprint"],
                    rec.get("schema", {}).get("error", ""),
                ]
            )

    with schema_jsonl.open("w", encoding="utf-8") as f:
        for rec in report["files"]:
            f.write(
                json.dumps(
                    {
                        "dataset_guess": rec["dataset_guess"],
                        "path": rec["path_display"],
                        "kind": rec["kind"],
                        "schema": rec["schema"],
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    quarantine_txt.write_text(
        "\n".join(report["external_quarantine_files"])
        + ("\n" if report["external_quarantine_files"] else ""),
        encoding="utf-8",
    )

    environment_json.write_text(
        json.dumps(
            report["environment"],
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    handoff = {
        "checkpoint": 0,
        "name": "access_environment_data_audit",
        "generated_utc": report["generated_utc"],
        "status": report["checkpoint_status"],
        "repo_root": report["repo_root"],
        "data_roots": report["data_roots"],
        "required_source_detection": report["required_source_detection"],
        "dataset_summary": report["dataset_summary"],
        "policy_errors": report["policy_errors"],
        "external_quarantine_files": report["external_quarantine_files"],
        "artifacts": {
            "audit_json": str(audit_json),
            "audit_markdown": str(audit_md),
            "file_inventory": str(inventory_tsv),
            "schema_inventory": str(schema_jsonl),
            "external_quarantine": str(quarantine_txt),
            "environment": str(environment_json),
        },
        "frozen_decisions": [
            "GENIE is genomic-pretraining data.",
            "CHORD is temporal-pretraining and primary dynamic-training data.",
            "BPC-MSK is development-time schema/endpoint/label-audit data only.",
            "DFCI and Vanderbilt/VICC are external-validation-only.",
            "No exact patient overlap will be inferred until authoritative patient IDs are mapped."
        ],
        "next_required_action": (
            "Review candidate source/schema detection and map the authoritative "
            "patient/sample/site/timestamp fields before canonical timeline construction."
        ),
    }

    handoff_path.write_text(
        json.dumps(handoff, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    for required in (
        audit_json,
        audit_md,
        inventory_tsv,
        schema_jsonl,
        environment_json,
        handoff_path,
    ):
        if not required.exists() or required.stat().st_size == 0:
            raise RuntimeError(
                f"required output missing or empty: {required}"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        required=True,
    )
    parser.add_argument(
        "--data-root",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--policy",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/checkpoint0",
    )
    parser.add_argument(
        "--handoff",
        default="artifacts/handoff/checkpoint_00.json",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).expanduser().resolve()
    policy_path = Path(args.policy)

    if not policy_path.is_absolute():
        policy_path = repo_root / policy_path

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir

    handoff_path = Path(args.handoff)
    if not handoff_path.is_absolute():
        handoff_path = repo_root / handoff_path

    with policy_path.open("r", encoding="utf-8") as f:
        policy = json.load(f)

    policy_errors = validate_policy(policy)
    data_roots = discover_roots(repo_root, args.data_root)
    files = iter_data_files(data_roots, repo_root)

    records: list[dict[str, Any]] = []

    print(f"[CKPT0] repo_root={repo_root}")
    print(f"[CKPT0] python={sys.executable}")
    print(f"[CKPT0] discovered_data_roots={len(data_roots)}")
    print(f"[CKPT0] candidate_data_files={len(files)}")

    for i, path in enumerate(files, start=1):
        if i == 1 or i % 100 == 0 or i == len(files):
            print(
                f"[CKPT0] inspecting_file={i}/{len(files)} "
                f"path={safe_rel(path, repo_root)}"
            )

        records.append(inspect_file(path, repo_root))

    dataset_summary = summarize_files(records)

    required = (
        "GENIE",
        "CHORD",
        "BPC_MSK",
        "BPC_DFCI",
        "BPC_VICC",
    )

    detection = {
        dataset: dataset_summary.get(dataset, {}).get("file_count", 0) > 0
        for dataset in required
    }

    # A combined BPC release may contain all institutions in one file.
    # Therefore this is a path/name detection status, not a claim about absence.
    if all(detection.values()) and not policy_errors:
        checkpoint_status = "PASS_SOURCE_DETECTION"
    else:
        checkpoint_status = "NEEDS_SOURCE_OR_SITE_MAPPING"

    external_files = sorted(
        rec["path_display"]
        for rec in records
        if rec["dataset_guess"] in {"BPC_DFCI", "BPC_VICC"}
    )

    unique_disks: list[dict[str, Any]] = []
    seen_devices: set[tuple[int, int]] = set()

    for path in [repo_root] + data_roots:
        try:
            st = os.stat(path)
            key = (st.st_dev, st.st_ino if path == repo_root else st.st_dev)
        except Exception:
            key = (hash(str(path)), 0)

        if key in seen_devices:
            continue

        seen_devices.add(key)
        unique_disks.append(disk_record(path))

    environment = {
        "python_version": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": package_versions(),
        "gpu": gpu_environment(),
    }

    report = {
        "generated_utc": utc_now(),
        "checkpoint_status": checkpoint_status,
        "repo_root": str(repo_root),
        "policy_path": safe_rel(policy_path, repo_root),
        "policy_errors": policy_errors,
        "data_roots": [str(x) for x in data_roots],
        "disk": unique_disks,
        "environment": environment,
        "files": records,
        "dataset_summary": dataset_summary,
        "required_source_detection": detection,
        "patient_overlap_readiness": sample_overlap_candidates(records),
        "external_quarantine_files": external_files,
        "notes": [
            "File sizes are exact.",
            "Files <=64 MiB receive full SHA256 hashes.",
            "Larger files receive explicitly labeled sampled fingerprints to avoid unnecessary I/O.",
            "Delimited text schemas are sampled without reading the full table.",
            "Parquet row counts are taken from metadata when pyarrow is available.",
            "No raw data are modified.",
            "No external-validation data are used for modeling."
        ],
    }

    write_outputs(
        report=report,
        output_dir=output_dir,
        handoff_path=handoff_path,
    )

    print("")
    print("========== CKPT0 SUMMARY ==========")
    print(f"status={checkpoint_status}")
    print(f"files_inspected={len(records)}")

    for dataset in (
        "GENIE",
        "CHORD",
        "BPC_MSK",
        "BPC_DFCI",
        "BPC_VICC",
        "BPC_UNASSIGNED",
        "UNKNOWN",
    ):
        s = dataset_summary.get(dataset)
        if s:
            print(
                f"{dataset}: "
                f"files={s['file_count']} "
                f"size_gib={s['total_gib']} "
                f"patient_id_files={s['files_with_patient_id_candidates']} "
                f"timestamp_files={s['files_with_timestamp_candidates']} "
                f"panel_files={s['files_with_panel_candidates']} "
                f"imaging_files={s['files_with_imaging_candidates']}"
            )

    gpu = environment["gpu"]
    print(
        "cuda_available="
        f"{gpu.get('torch_cuda_available')} "
        "cuda_devices="
        f"{gpu.get('torch_cuda_device_count')}"
    )

    for d in unique_disks:
        print(
            f"disk={d['path']} "
            f"free_gib={d['free_gib']}"
        )

    print(f"policy_errors={len(policy_errors)}")
    print(f"external_quarantine_files={len(external_files)}")
    print("audit_md=artifacts/checkpoint0/audit.md")
    print("audit_json=artifacts/checkpoint0/audit.json")
    print("handoff=artifacts/handoff/checkpoint_00.json")
    print("========== CKPT0 END ==========")
    print("")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
