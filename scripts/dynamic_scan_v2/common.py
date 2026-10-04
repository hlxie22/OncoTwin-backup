from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def atomic_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )
    json.loads(tmp.read_text(encoding="utf-8"))
    tmp.replace(path)


def atomic_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    if tmp.stat().st_size == 0:
        raise RuntimeError(f"Refusing empty write: {path}")
    tmp.replace(path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    frame.to_parquet(tmp, index=False)
    check = pd.read_parquet(tmp)
    if len(check) != len(frame):
        raise RuntimeError(f"Parquet verification failed for {path}")
    tmp.replace(path)


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_fold(patient_id: str, folds: int, salt: str) -> int:
    if folds <= 1:
        raise ValueError("folds must be > 1")
    raw = f"{salt}\0{patient_id}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")
    return int(value % folds)


def import_module_from_path(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def assert_unique(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    columns = list(columns)
    duplicated = frame.duplicated(columns, keep=False)
    if duplicated.any():
        sample = frame.loc[duplicated, columns].head(10).to_dict("records")
        raise RuntimeError(f"Duplicate {label} keys: {sample}")


def normalized_string(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")
