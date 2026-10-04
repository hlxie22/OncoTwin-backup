#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

from torch.utils.data import (
    DataLoader,
    Dataset,
)


SEED = 20260926

GENOMIC_DIM = 128
TEMPORAL_DIM = 192

MONTHS = 24
MONTH_DAYS = 730.0 / MONTHS

CAUSE_CENSOR = 0
CAUSE_PROGRESSION = 1
CAUSE_DEATH = 2
CAUSE_SWITCH = 3

CAUSE_NAMES = {
    CAUSE_CENSOR: "CENSOR",
    CAUSE_PROGRESSION: "PROGRESSION",
    CAUSE_DEATH: "DEATH",
    CAUSE_SWITCH: "SWITCH",
}

NEXT_SCAN_CLASSES = {
    "NON_PROGRESSIVE": 0,
    "INDETERMINATE": 1,
    "PROGRESSIVE": 2,
}

POSTPROG_CLASSES = {
    "NEXT_TREATMENT": 0,
    "CONTINUED_OR_CENSORED": 1,
    "DEATH": 2,
}

EPS = 1e-7


###############################################################################
# General utilities
###############################################################################


def norm(value: Any) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value).strip().lower(),
    ).strip("_")


def atomic_json(
    path: Path,
    payload: Any,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=lambda value:
                int(value)
                if isinstance(value, np.integer)
                else float(value)
                if isinstance(value, np.floating)
                else bool(value)
                if isinstance(value, np.bool_)
                else str(value),
        ),
        encoding="utf-8",
    )

    if (
        not tmp.exists()
        or tmp.stat().st_size == 0
    ):
        raise RuntimeError(
            f"empty JSON write: {path}"
        )

    json.loads(
        tmp.read_text(
            encoding="utf-8"
        )
    )

    tmp.replace(path)


def atomic_text(
    path: Path,
    text: str,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_text(
        text,
        encoding="utf-8",
    )

    if (
        not tmp.exists()
        or tmp.stat().st_size == 0
    ):
        raise RuntimeError(
            f"empty text write: {path}"
        )

    tmp.replace(path)


def atomic_parquet(
    path: Path,
    frame: pd.DataFrame,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    frame.to_parquet(
        tmp,
        index=False,
    )

    check = pd.read_parquet(
        tmp
    )

    if len(check) != len(frame):
        raise RuntimeError(
            f"Parquet validation failed: {path}"
        )

    tmp.replace(path)


def find_col(
    columns: Iterable[str],
    candidates: Iterable[str],
    *,
    required: bool = True,
) -> str | None:

    lookup = {
        norm(column): column
        for column in columns
    }

    for candidate in candidates:

        key = norm(
            candidate
        )

        if key in lookup:
            return lookup[
                key
            ]

    if required:
        raise KeyError(
            f"Unable to resolve {list(candidates)} "
            f"from columns={list(columns)}"
        )

    return None


def to_numeric(
    series: pd.Series,
) -> pd.Series:

    raw = (
        series
        .astype(str)
        .str.strip()
    )

    lower = raw.str.lower()

    mapped = lower.map(
        {
            "true": 1.0,
            "false": 0.0,
            "yes": 1.0,
            "no": 0.0,
            "y": 1.0,
            "n": 0.0,
            "present": 1.0,
            "absent": 0.0,
        }
    )

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    return numeric.where(
        numeric.notna(),
        mapped,
    )


def stable_hash(
    value: str,
) -> int:

    return int(
        hashlib.sha256(
            value.encode(
                "utf-8"
            )
        ).hexdigest()[:16],
        16,
    )


def import_ckpt3(
    repo: Path,
):

    path = (
        repo
        / "scripts"
        / "dynamic_scan"
        / "ckpt3_genie_pretrain.py"
    )

    if not path.exists():
        raise RuntimeError(
            f"Frozen CKPT3 implementation missing: {path}"
        )

    spec = importlib.util.spec_from_file_location(
        "frozen_ckpt3",
        path,
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            "Unable to import frozen CKPT3 implementation."
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[
        spec.name
    ] = module

    spec.loader.exec_module(
        module
    )

    return module


###############################################################################
# Frozen upstream validation
###############################################################################


def validate_frozen_upstream(
    repo: Path,
) -> dict[str, Any]:

    ###########################################################################
    # QC JSON establishes checkpoint PASS state only.
    #
    # Counts/shapes below are derived directly from the frozen artifacts rather
    # than assuming a particular bookkeeping schema inside final_qc.json.
    ###########################################################################

    ckpt2_path = (
        repo
        / "artifacts"
        / "checkpoint2"
        / "qc.json"
    )

    ckpt3_path = (
        repo
        / "artifacts"
        / "checkpoint3"
        / "qc.json"
    )

    ckpt4_path = (
        repo
        / "artifacts"
        / "checkpoint4"
        / "final_qc.json"
    )

    ckpt2 = json.loads(
        ckpt2_path.read_text(
            encoding="utf-8"
        )
    )

    ckpt3 = json.loads(
        ckpt3_path.read_text(
            encoding="utf-8"
        )
    )

    ckpt4 = json.loads(
        ckpt4_path.read_text(
            encoding="utf-8"
        )
    )

    ckpt2_status = (
        ckpt2.get(
            "status"
        )
        or ckpt2.get(
            "checkpoint_status"
        )
    )

    ckpt3_status = (
        ckpt3.get(
            "status"
        )
        or ckpt3.get(
            "checkpoint_status"
        )
    )

    ckpt4_status = (
        ckpt4.get(
            "checkpoint_status"
        )
        or ckpt4.get(
            "status"
        )
    )

    if (
        ckpt2_status
        != "PASS_SCAN_UPDATE_INFORMATION_GAIN"
    ):

        raise RuntimeError(
            "CKPT2 is not frozen PASS: "
            f"{ckpt2_status!r}"
        )

    if (
        ckpt3_status
        != "PASS_GENOMIC_PRETRAINING"
    ):

        raise RuntimeError(
            "CKPT3 is not frozen PASS: "
            f"{ckpt3_status!r}"
        )

    if (
        ckpt4_status
        != "PASS_TEMPORAL_PRETRAINING"
    ):

        raise RuntimeError(
            "CKPT4 is not frozen PASS: "
            f"{ckpt4_status!r}"
        )

    ###########################################################################
    # Authoritative CKPT1 W3 scan population.
    ###########################################################################

    scan_path = (
        repo
        / "artifacts"
        / "checkpoint1"
        / "canonical"
        / "chord_mbc_scan_episodes_w3.parquet"
    )

    scan = pd.read_parquet(
        scan_path
    )

    scan_patient_col = find_col(
        scan.columns,
        [
            "patient_id",
            "PATIENT_ID",
        ],
    )

    scan_episode_col = find_col(
        scan.columns,
        [
            "scan_episode_id",
            "episode_id",
        ],
    )

    scan_patient = (
        scan[
            scan_patient_col
        ]
        .astype(str)
        .str.strip()
    )

    scan_episode = (
        scan[
            scan_episode_col
        ]
        .astype(str)
        .str.strip()
    )

    authoritative_keys = pd.DataFrame(
        {
            "patient_id":
                scan_patient,

            "scan_episode_id":
                scan_episode,
        }
    )

    if (
        authoritative_keys
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Authoritative CKPT1 W3 scan keys "
            "contain duplicates."
        )

    authoritative_rows = int(
        len(
            authoritative_keys
        )
    )

    authoritative_patients = int(
        authoritative_keys[
            "patient_id"
        ].nunique()
    )

    ###########################################################################
    # Frozen CKPT4 scan embedding index.
    ###########################################################################

    cache_index_path = (
        repo
        / "artifacts"
        / "checkpoint4"
        / "breast_scan_prepost_index.parquet"
    )

    cache_index = pd.read_parquet(
        cache_index_path
    )

    cache_patient_col = find_col(
        cache_index.columns,
        [
            "patient_id",
            "PATIENT_ID",
        ],
    )

    cache_episode_col = find_col(
        cache_index.columns,
        [
            "scan_episode_id",
            "episode_id",
        ],
    )

    cache_keys = pd.DataFrame(
        {
            "patient_id":
                cache_index[
                    cache_patient_col
                ]
                .astype(str)
                .str.strip(),

            "scan_episode_id":
                cache_index[
                    cache_episode_col
                ]
                .astype(str)
                .str.strip(),
        }
    )

    if (
        cache_keys
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "CKPT4 scan-cache keys contain duplicates."
        )

    cached_rows = int(
        len(
            cache_keys
        )
    )

    cached_patients = int(
        cache_keys[
            "patient_id"
        ].nunique()
    )

    ###########################################################################
    # Exact key equality rather than relying merely on counts.
    ###########################################################################

    authoritative_key_set = set(
        zip(
            authoritative_keys[
                "patient_id"
            ],
            authoritative_keys[
                "scan_episode_id"
            ],
        )
    )

    cache_key_set = set(
        zip(
            cache_keys[
                "patient_id"
            ],
            cache_keys[
                "scan_episode_id"
            ],
        )
    )

    missing_from_cache = (
        authoritative_key_set
        - cache_key_set
    )

    extra_in_cache = (
        cache_key_set
        - authoritative_key_set
    )

    if (
        missing_from_cache
        or extra_in_cache
    ):

        raise RuntimeError(
            "CKPT4 cache keys do not exactly equal "
            "authoritative CKPT1 W3 scan keys: "
            f"missing={len(missing_from_cache)} "
            f"extra={len(extra_in_cache)}"
        )

    ###########################################################################
    # Frozen CKPT4 embedding tensor.
    ###########################################################################

    embedding_path = (
        repo
        / "artifacts"
        / "checkpoint4"
        / "breast_scan_prepost_embeddings_f16.npy"
    )

    embedding = np.load(
        embedding_path,
        mmap_mode="r",
    )

    expected_shape = (
        authoritative_rows,
        2,
        TEMPORAL_DIM,
    )

    if tuple(
        embedding.shape
    ) != expected_shape:

        raise RuntimeError(
            "Unexpected CKPT4 PRE/POST embedding shape: "
            f"observed={tuple(embedding.shape)} "
            f"expected={expected_shape}"
        )

    ###########################################################################
    # Derive eligible residual-PFS row count directly from the cache index
    # when that field is present.
    ###########################################################################

    eligible_col = find_col(
        cache_index.columns,
        [
            "eligible_residual_pfs",
            "residual_pfs_eligible",
            "eligible_primary",
        ],
        required=False,
    )

    if eligible_col is not None:

        raw = (
            cache_index[
                eligible_col
            ]
        )

        if pd.api.types.is_bool_dtype(
            raw.dtype
        ):

            eligible_mask = (
                raw.fillna(
                    False
                )
                .astype(bool)
            )

        elif pd.api.types.is_numeric_dtype(
            raw.dtype
        ):

            eligible_mask = (
                pd.to_numeric(
                    raw,
                    errors="coerce",
                )
                .fillna(
                    0
                )
                > 0
            )

        else:

            eligible_mask = (
                raw
                .astype(str)
                .str.strip()
                .str.lower()
                .isin(
                    [
                        "1",
                        "true",
                        "yes",
                        "y",
                    ]
                )
            )

        eligible_residual_rows = int(
            eligible_mask.sum()
        )

        eligibility_source = (
            "checkpoint4_cache_index:"
            + str(
                eligible_col
            )
        )

    else:

        #######################################################################
        # This count is descriptive only at this point. The downstream
        # preparation reconstructs/validates landmark eligibility again.
        #######################################################################

        eligible_residual_rows = None

        eligibility_source = (
            "not_materialized_in_cache_index;"
            "downstream_reconstruction_required"
        )

    ###########################################################################
    # Frozen invariants established by completed CKPT4.
    ###########################################################################

    if authoritative_rows != 49189:

        raise RuntimeError(
            "Unexpected authoritative W3 scan count: "
            f"{authoritative_rows}"
        )

    if authoritative_patients != 3351:

        raise RuntimeError(
            "Unexpected authoritative W3 scan patient count: "
            f"{authoritative_patients}"
        )

    if cached_rows != authoritative_rows:

        raise RuntimeError(
            "CKPT4 cache row count does not equal "
            "authoritative scan count."
        )

    if cached_patients != authoritative_patients:

        raise RuntimeError(
            "CKPT4 cache patient count does not equal "
            "authoritative W3 scan patient count."
        )

    if (
        eligible_residual_rows
        is not None
        and eligible_residual_rows
        != 32477
    ):

        raise RuntimeError(
            "Unexpected CKPT4 cached residual-PFS eligibility count: "
            f"{eligible_residual_rows}"
        )

    return {
        "ckpt2":
            ckpt2_status,

        "ckpt3":
            ckpt3_status,

        "ckpt4":
            ckpt4_status,

        "w3_scans":
            authoritative_rows,

        "w3_scan_patients":
            authoritative_patients,

        "cached_scan_rows":
            cached_rows,

        "cached_scan_patients":
            cached_patients,

        "embedding_shape":
            list(
                embedding.shape
            ),

        "scan_key_validation":
            "patient_id+scan_episode_id_exact",

        "eligible_residual_pfs_rows":
            eligible_residual_rows,

        "eligible_residual_pfs_source":
            eligibility_source,

        "ckpt4_qc_top_level_keys":
            sorted(
                ckpt4.keys()
            ),
    }


###############################################################################
# Temporal cache alignment
###############################################################################


def load_temporal_cache(
    repo: Path,
) -> tuple[
    pd.DataFrame,
    np.ndarray,
]:

    index_path = (
        repo
        / "artifacts"
        / "checkpoint4"
        / "breast_scan_prepost_index.parquet"
    )

    array_path = (
        repo
        / "artifacts"
        / "checkpoint4"
        / "breast_scan_prepost_embeddings_f16.npy"
    )

    index = pd.read_parquet(
        index_path
    )

    embeddings = np.load(
        array_path,
        mmap_mode="r",
    )

    if embeddings.shape != (
        49189,
        2,
        TEMPORAL_DIM,
    ):
        raise RuntimeError(
            "Unexpected CKPT4 temporal embedding shape: "
            f"{embeddings.shape}"
        )

    if len(index) != len(
        embeddings
    ):
        raise RuntimeError(
            "CKPT4 index/embedding length mismatch."
        )

    patient_col = find_col(
        index.columns,
        [
            "patient_id",
            "PATIENT_ID",
        ],
    )

    scan_col = find_col(
        index.columns,
        [
            "scan_episode_id",
            "episode_id",
        ],
    )

    day_col = find_col(
        index.columns,
        [
            "landmark_day",
            "episode_end_day",
            "scan_day",
        ],
        required=False,
    )

    line_col = find_col(
        index.columns,
        [
            "treatment_line",
            "line",
        ],
        required=False,
    )

    eligible_col = find_col(
        index.columns,
        [
            "eligible_residual_pfs",
            "residual_pfs_eligible",
            "eligible_primary",
        ],
        required=False,
    )

    normalized = index.copy()

    normalized[
        "patient_id"
    ] = (
        normalized[
            patient_col
        ]
        .astype(str)
        .str.strip()
    )

    normalized[
        "scan_episode_id"
    ] = (
        normalized[
            scan_col
        ]
        .astype(str)
        .str.strip()
    )

    normalized[
        "temporal_row"
    ] = np.arange(
        len(
            normalized
        ),
        dtype=np.int64,
    )

    if day_col is not None:

        normalized[
            "cache_landmark_day"
        ] = pd.to_numeric(
            normalized[
                day_col
            ],
            errors="coerce",
        )

    if line_col is not None:

        normalized[
            "cache_line"
        ] = pd.to_numeric(
            normalized[
                line_col
            ],
            errors="coerce",
        )

    if eligible_col is not None:

        normalized[
            "cache_eligible_residual_pfs"
        ] = (
            normalized[
                eligible_col
            ]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(
                [
                    "1",
                    "true",
                    "yes",
                    "y",
                ]
            )
        )

    return (
        normalized,
        embeddings,
    )


###############################################################################
# Authoritative scan normalization
###############################################################################


def load_scans(
    repo: Path,
) -> pd.DataFrame:

    path = (
        repo
        / "artifacts"
        / "checkpoint1"
        / "canonical"
        / "chord_mbc_scan_episodes_w3.parquet"
    )

    frame = pd.read_parquet(
        path
    )

    patient_col = find_col(
        frame.columns,
        [
            "patient_id",
            "PATIENT_ID",
        ],
    )

    scan_col = find_col(
        frame.columns,
        [
            "scan_episode_id",
            "episode_id",
        ],
    )

    day_col = find_col(
        frame.columns,
        [
            "landmark_day",
            "episode_end_day",
        ],
    )

    end_col = find_col(
        frame.columns,
        [
            "episode_end_day",
            "landmark_day",
        ],
    )

    line_col = find_col(
        frame.columns,
        [
            "treatment_line",
            "line",
        ],
    )

    state_col = find_col(
        frame.columns,
        [
            "progression_state_3",
            "scan_state",
        ],
    )

    out = frame.copy()

    out[
        "patient_id"
    ] = (
        out[
            patient_col
        ]
        .astype(str)
        .str.strip()
    )

    out[
        "scan_episode_id"
    ] = (
        out[
            scan_col
        ]
        .astype(str)
        .str.strip()
    )

    out[
        "landmark_day"
    ] = pd.to_numeric(
        out[
            day_col
        ],
        errors="raise",
    ).astype(
        float
    )

    out[
        "episode_end_day_norm"
    ] = pd.to_numeric(
        out[
            end_col
        ],
        errors="raise",
    ).astype(
        float
    )

    out[
        "line"
    ] = pd.to_numeric(
        out[
            line_col
        ],
        errors="raise",
    ).astype(
        int
    )

    out[
        "scan_state"
    ] = (
        out[
            state_col
        ]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    if not np.allclose(
        out[
            "landmark_day"
        ],
        out[
            "episode_end_day_norm"
        ],
    ):
        raise RuntimeError(
            "Scan landmark is not at episode END day."
        )

    if (
        out[
            [
                "patient_id",
                "scan_episode_id",
            ]
        ]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "Authoritative W3 scan keys are not unique."
        )

    return out


###############################################################################
# Patient splits and treatment-line context
###############################################################################


def load_splits(
    repo: Path,
) -> pd.DataFrame:

    frame = pd.read_parquet(
        repo
        / "artifacts"
        / "checkpoint1"
        / "canonical"
        / "chord_patient_splits.parquet"
    )

    patient_col = find_col(
        frame.columns,
        [
            "patient_id",
            "PATIENT_ID",
        ],
    )

    split_col = find_col(
        frame.columns,
        [
            "split",
            "SPLIT",
        ],
    )

    out = frame[
        [
            patient_col,
            split_col,
        ]
    ].copy()

    out.columns = [
        "patient_id",
        "split",
    ]

    out[
        "patient_id"
    ] = (
        out[
            "patient_id"
        ]
        .astype(str)
        .str.strip()
    )

    out[
        "split"
    ] = (
        out[
            "split"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    if (
        out[
            "patient_id"
        ]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "Patient split table is not one row per patient."
        )

    return out


def load_lines(
    repo: Path,
) -> pd.DataFrame:

    frame = pd.read_parquet(
        repo
        / "artifacts"
        / "checkpoint1"
        / "canonical"
        / "chord_mbc_lines.parquet"
    )

    patient_col = find_col(
        frame.columns,
        [
            "patient_id",
            "PATIENT_ID",
        ],
    )

    line_col = find_col(
        frame.columns,
        [
            "line",
            "LINE",
        ],
    )

    start_col = find_col(
        frame.columns,
        [
            "line_start_day",
            "LINE_START",
            "line_start",
        ],
    )

    out = pd.DataFrame(
        {
            "patient_id":
                frame[
                    patient_col
                ]
                .astype(str)
                .str.strip(),

            "line":
                pd.to_numeric(
                    frame[
                        line_col
                    ],
                    errors="raise",
                )
                .astype(int),

            "line_start_day":
                pd.to_numeric(
                    frame[
                        start_col
                    ],
                    errors="raise",
                )
                .astype(float),
        }
    )

    out = (
        out
        .drop_duplicates(
            subset=[
                "patient_id",
                "line",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return out


###############################################################################
# Scan feature construction
###############################################################################


def scan_features(
    scans: pd.DataFrame,
) -> tuple[
    np.ndarray,
    list[str],
]:

    feature_columns: dict[
        str,
        np.ndarray,
    ] = {}

    state = (
        scans[
            "scan_state"
        ]
        .astype(str)
        .str.upper()
    )

    for state_name in (
        "NON_PROGRESSIVE",
        "INDETERMINATE",
        "PROGRESSIVE",
    ):

        feature_columns[
            "state_"
            + state_name.lower()
        ] = (
            state
            == state_name
        ).to_numpy(
            dtype=np.float32
        )

    ###########################################################################
    # Explicit imaging coverage.
    ###########################################################################

    allowed_numeric_tokens = (
        "coverage",
        "chest",
        "abdomen",
        "pelvis",
        "head",
        "other",
        "has_cancer",
        "cancer_presence",
    )

    deny_tokens = (
        "target",
        "next",
        "future",
        "pfs",
        "death",
        "switch",
        "outcome",
        "time_to",
    )

    for column in scans.columns:

        normalized = norm(
            column
        )

        if any(
            token in normalized
            for token in deny_tokens
        ):
            continue

        if not any(
            token in normalized
            for token in allowed_numeric_tokens
        ):
            continue

        numeric = to_numeric(
            scans[
                column
            ]
        )

        nonmissing = float(
            numeric
            .notna()
            .mean()
        )

        if nonmissing < 0.50:
            continue

        values = (
            numeric
            .fillna(
                0.0
            )
            .to_numpy(
                dtype=np.float32
            )
        )

        if (
            np.nanstd(
                values
            )
            < 1e-8
        ):
            continue

        feature_columns[
            "raw_"
            + normalized
        ] = values

    ###########################################################################
    # Modalities/sites from explicitly observed text fields only.
    ###########################################################################

    text_candidates = []

    for column in scans.columns:

        normalized = norm(
            column
        )

        if any(
            token in normalized
            for token in (
                "procedure",
                "modality",
                "source_specific",
                "tumor_site",
                "tumor_sites",
                "site",
            )
        ):

            text_candidates.append(
                column
            )

    if text_candidates:

        combined = (
            scans[
                text_candidates
            ]
            .fillna("")
            .astype(str)
            .agg(
                " | ".join,
                axis=1,
            )
            .str.upper()
        )

    else:

        combined = pd.Series(
            "",
            index=scans.index,
        )

    patterns = {
        "modality_ct":
            r"\bCT\b|COMPUTED TOMOGRAPH",

        "modality_pet":
            r"\bPET\b|PET-CT",

        "modality_mr":
            r"\bMRI?\b|MAGNETIC",

        "modality_bone_scan":
            r"BONE SCAN",

        "site_bone":
            r"\bBONE\b|SKELET|VERTEBR|RIB",

        "site_liver":
            r"\bLIVER\b",

        "site_lung":
            r"\bLUNG\b|PULMON",

        "site_brain":
            r"\bBRAIN\b|\bCNS\b|CEREBR",

        "site_lymph":
            r"LYMPH",

        "site_pleura":
            r"PLEURA",
    }

    for name, pattern in patterns.items():

        feature_columns[
            name
        ] = (
            combined
            .str.contains(
                pattern,
                regex=True,
                na=False,
            )
            .to_numpy(
                dtype=np.float32
            )
        )

    names = list(
        feature_columns
    )

    matrix = np.column_stack(
        [
            feature_columns[
                name
            ]
            for name in names
        ]
    ).astype(
        np.float32
    )

    if matrix.shape[
        1
    ] < 8:
        raise RuntimeError(
            "Unexpectedly weak current-scan feature surface: "
            f"{names}"
        )

    return (
        matrix,
        names,
    )


###############################################################################
# Survival labels from frozen CKPT1 landmarks
###############################################################################


def resolve_target_columns(
    frame: pd.DataFrame,
) -> tuple[
    str,
    str,
]:

    ###########################################################################
    # Frozen CKPT1 supervised endpoint interface.
    #
    # These fields already encode the canonical endpoint construction,
    # including the 730-day administrative horizon. CKPT5 consumes this
    # interface; it must not reconstruct the endpoint from raw fields.
    ###########################################################################

    if (
        "train_event_type"
        in frame.columns
        and "train_time_days"
        in frame.columns
    ):

        return (
            "train_event_type",
            "train_time_days",
        )

    ###########################################################################
    # Explicit compatibility aliases only. Deliberately DO NOT include
    # raw_event_type or raw_time_days.
    ###########################################################################

    type_col = find_col(
        frame.columns,
        [
            "training_event_type",
            "target_event_type",
            "target_type",
            "training_target",
        ],
        required=False,
    )

    time_col = find_col(
        frame.columns,
        [
            "training_time_days",
            "target_time_days",
            "residual_time_days",
        ],
        required=False,
    )

    if (
        type_col is not None
        and time_col is not None
    ):

        return (
            type_col,
            time_col,
        )

    raise RuntimeError(
        "Unable to resolve canonical CKPT1 TRAIN survival target schema.\n"
        "Expected train_event_type + train_time_days.\n"
        f"columns={list(frame.columns)}"
    )


def load_survival_targets(
    repo: Path,
) -> pd.DataFrame:

    frame = pd.read_parquet(
        repo
        / "artifacts"
        / "checkpoint1"
        / "canonical"
        / "chord_landmarks.parquet"
    )

    patient_col = find_col(
        frame.columns,
        [
            "patient_id",
        ],
    )

    landmark_type_col = find_col(
        frame.columns,
        [
            "landmark_type",
        ],
    )

    scan_id_col = find_col(
        frame.columns,
        [
            "scan_episode_id",
            "episode_id",
        ],
        required=False,
    )

    day_col = find_col(
        frame.columns,
        [
            "landmark_day",
            "day",
        ],
    )

    eligible_col = find_col(
        frame.columns,
        [
            "eligible_primary",
            "eligible_residual_pfs",
        ],
        required=False,
    )

    (
        target_type_col,
        target_time_col,
    ) = resolve_target_columns(
        frame
    )

    scan = frame[
        frame[
            landmark_type_col
        ]
        .astype(str)
        .str.upper()
        == "SCAN"
    ].copy()

    out = pd.DataFrame(
        {
            "patient_id":
                scan[
                    patient_col
                ]
                .astype(str)
                .str.strip(),

            "target_landmark_day":
                pd.to_numeric(
                    scan[
                        day_col
                    ],
                    errors="coerce",
                ),

            "target_type":
                scan[
                    target_type_col
                ]
                .astype(str)
                .str.strip()
                .str.upper(),

            "target_time_days":
                pd.to_numeric(
                    scan[
                        target_time_col
                    ],
                    errors="coerce",
                ),
        }
    )

    if scan_id_col is not None:

        out[
            "scan_episode_id"
        ] = (
            scan[
                scan_id_col
            ]
            .astype(str)
            .str.strip()
        )

    if eligible_col is not None:

        out[
            "landmark_eligible"
        ] = (
            scan[
                eligible_col
            ]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(
                [
                    "1",
                    "true",
                    "yes",
                    "y",
                ]
            )
        )

    return out


def cause_from_target(
    value: str,
) -> int | None:

    text = str(
        value
    ).upper()

    if "PROGRESSION" in text:
        return CAUSE_PROGRESSION

    if "DEATH" in text:
        return CAUSE_DEATH

    if "SWITCH" in text:
        return CAUSE_SWITCH

    if (
        "CENSOR"
        in text
    ):
        return CAUSE_CENSOR

    if (
        "NO_FOLLOWUP"
        in text
        or "NO FOLLOW"
        in text
    ):
        return None

    return None


###############################################################################
# Next-scan targets derived from authoritative W3 sequence
###############################################################################


def load_next_scan_targets(
    repo: Path,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
]:

    path = (
        repo
        / "artifacts"
        / "checkpoint1"
        / "canonical"
        / "chord_next_scan_targets.parquet"
    )

    frame = pd.read_parquet(
        path
    )

    patient_col = find_col(
        frame.columns,
        [
            "patient_id",
            "PATIENT_ID",
        ],
    )

    source_scan_col = find_col(
        frame.columns,
        [
            "scan_episode_id",
            "current_scan_episode_id",
            "landmark_scan_episode_id",
            "source_scan_episode_id",
        ],
        required=False,
    )

    if source_scan_col is None:

        raise RuntimeError(
            "Frozen CKPT1 next-scan target artifact does not expose "
            "an exact source scan_episode_id."
        )

    state_col = find_col(
        frame.columns,
        [
            "next_scan_state",
            "target_scan_state",
            "next_state",
            "scan_state_next",
        ],
        required=False,
    )

    if state_col is None:

        allowed = {
            "NON_PROGRESSIVE",
            "INDETERMINATE",
            "PROGRESSIVE",
        }

        candidates = []

        for column in frame.columns:

            name = norm(
                column
            )

            if not (
                "next"
                in name
                or "target"
                in name
            ):
                continue

            values = {
                str(
                    value
                )
                .strip()
                .upper()
                for value
                in frame[
                    column
                ]
                .dropna()
                .unique()
            }

            values.discard(
                ""
            )

            if (
                values
                and values
                <= allowed
            ):

                candidates.append(
                    column
                )

        if len(
            candidates
        ) != 1:

            raise RuntimeError(
                "Unable to uniquely resolve frozen next-scan target state: "
                f"candidates={candidates} "
                f"columns={list(frame.columns)}"
            )

        state_col = candidates[
            0
        ]

    result = pd.DataFrame(
        {
            "patient_id":
                frame[
                    patient_col
                ]
                .astype(str)
                .str.strip(),

            "scan_episode_id":
                frame[
                    source_scan_col
                ]
                .astype(str)
                .str.strip(),

            "next_scan_state":
                frame[
                    state_col
                ]
                .astype(str)
                .str.strip()
                .str.upper(),
        }
    )

    allowed_states = {
        "NON_PROGRESSIVE",
        "INDETERMINATE",
        "PROGRESSIVE",
    }

    unexpected = (
        set(
            result[
                "next_scan_state"
            ].unique()
        )
        - allowed_states
    )

    if unexpected:

        raise RuntimeError(
            "Unexpected frozen next-scan states: "
            f"{sorted(unexpected)}"
        )

    if result[
        [
            "patient_id",
            "scan_episode_id",
        ]
    ].duplicated().any():

        raise RuntimeError(
            "Frozen next-scan target keys are not unique."
        )

    result[
        "next_scan_label"
    ] = (
        result[
            "next_scan_state"
        ]
        .map(
            NEXT_SCAN_CLASSES
        )
        .astype(
            int
        )
    )

    if len(
        result
    ) != 23429:

        raise RuntimeError(
            "Frozen CKPT1 next-scan target count changed: "
            f"{len(result)} != 23429"
        )

    audit = {
        "source_rows":
            int(
                len(
                    frame
                )
            ),

        "source_patients":
            int(
                result[
                    "patient_id"
                ].nunique()
            ),

        "patient_column":
            patient_col,

        "source_scan_episode_column":
            source_scan_col,

        "target_state_column":
            state_col,

        "target_state_counts":
            result[
                "next_scan_state"
            ]
            .value_counts()
            .to_dict(),

        "frozen_reference_count":
            23429,
    }

    return (
        result,
        audit,
    )


def attach_next_scan_targets(
    repo: Path,
    index: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
]:

    targets, audit = (
        load_next_scan_targets(
            repo
        )
    )

    scan_keys = index[
        [
            "patient_id",
            "scan_episode_id",
        ]
    ]

    if scan_keys.duplicated().any():

        raise RuntimeError(
            "CKPT5 scan keys are not unique before next-scan merge."
        )

    source_keys = targets[
        [
            "patient_id",
            "scan_episode_id",
        ]
    ]

    matched = source_keys.merge(
        scan_keys,
        on=[
            "patient_id",
            "scan_episode_id",
        ],
        how="inner",
        validate="one_to_one",
    )

    if len(
        matched
    ) != len(
        targets
    ):

        raise RuntimeError(
            "Frozen CKPT1 next-scan target keys do not all exist "
            "in the authoritative W3 scan population: "
            f"matched={len(matched)} "
            f"targets={len(targets)}"
        )

    merged = index.merge(
        targets,
        on=[
            "patient_id",
            "scan_episode_id",
        ],
        how="left",
        validate="one_to_one",
    )

    merged[
        "next_scan_mask"
    ] = merged[
        "next_scan_label"
    ].notna()

    observed = int(
        merged[
            "next_scan_mask"
        ].sum()
    )

    if observed != 23429:

        raise RuntimeError(
            "Frozen next-scan target count changed after merge: "
            f"{observed} != 23429"
        )

    audit[
        "exact_matched_rows"
    ] = int(
        len(
            matched
        )
    )

    audit[
        "merged_target_rows"
    ] = observed

    audit[
        "merge_key"
    ] = (
        "patient_id+scan_episode_id"
    )

    return (
        merged,
        audit,
    )


###############################################################################
# Post-progression targets
###############################################################################


def normalize_postprog_class(
    value: Any,
) -> int | None:

    ###########################################################################
    # Frozen CKPT1 post-progression transition semantics.
    ###########################################################################

    text = (
        str(
            value
        )
        .strip()
        .upper()
        .replace(
            "-",
            "_",
        )
        .replace(
            " ",
            "_",
        )
    )

    if not text:
        return None

    if (
        "NEXT_TREAT"
        in text
        or "NEW_TREAT"
        in text
        or "NEW_LINE"
        in text
    ):

        return POSTPROG_CLASSES[
            "NEXT_TREATMENT"
        ]

    ###########################################################################
    # In the frozen post-progression artifact, plain DEATH means death occurred
    # before another systemic-treatment transition.
    ###########################################################################

    if "DEATH" in text:

        return POSTPROG_CLASSES[
            "DEATH"
        ]

    if (
        "CONTINU"
        in text
        or "SAME_THERAP"
        in text
    ):

        return POSTPROG_CLASSES[
            "CONTINUED_OR_CENSORED"
        ]

    return None


def load_postprog_targets(
    repo: Path,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
]:

    path = (
        repo
        / "artifacts"
        / "checkpoint1"
        / "canonical"
        / "chord_post_progression_targets.parquet"
    )

    frame = pd.read_parquet(
        path
    )

    required = [
        "patient_id",
        "scan_episode_id",
        "progression_day",
        "treatment_line",
        "transition_type",
        "transition_day",
        "time_to_transition_days",
    ]

    missing = [
        column
        for column in required
        if column not in frame.columns
    ]

    if missing:

        raise RuntimeError(
            "Frozen CKPT1 post-progression schema changed: "
            f"missing={missing} "
            f"columns={list(frame.columns)}"
        )

    result = pd.DataFrame(
        {
            "patient_id":
                frame[
                    "patient_id"
                ]
                .astype(str)
                .str.strip(),

            "scan_episode_id":
                frame[
                    "scan_episode_id"
                ]
                .astype(str)
                .str.strip(),

            "postprog_progression_day":
                pd.to_numeric(
                    frame[
                        "progression_day"
                    ],
                    errors="coerce",
                ),

            "postprog_treatment_line":
                pd.to_numeric(
                    frame[
                        "treatment_line"
                    ],
                    errors="coerce",
                ),

            "postprog_transition_type_raw":
                frame[
                    "transition_type"
                ]
                .astype(str)
                .str.strip(),

            "postprog_transition_day":
                pd.to_numeric(
                    frame[
                        "transition_day"
                    ],
                    errors="coerce",
                ),

            "time_to_transition_days":
                pd.to_numeric(
                    frame[
                        "time_to_transition_days"
                    ],
                    errors="coerce",
                ),
        }
    )

    if result[
        [
            "patient_id",
            "scan_episode_id",
        ]
    ].duplicated().any():

        raise RuntimeError(
            "Frozen post-progression patient+scan_episode key "
            "is unexpectedly non-unique."
        )

    result[
        "postprog_label"
    ] = [
        normalize_postprog_class(
            value
        )
        for value
        in result[
            "postprog_transition_type_raw"
        ]
    ]

    unresolved = (
        result[
            "postprog_label"
        ].isna()
    )

    if unresolved.any():

        unresolved_counts = (
            result.loc[
                unresolved,
                "postprog_transition_type_raw",
            ]
            .value_counts()
            .to_dict()
        )

        raise RuntimeError(
            "Unresolved frozen post-progression transition types: "
            f"{unresolved_counts}"
        )

    if (
        result[
            "time_to_transition_days"
        ]
        .dropna()
        .lt(
            0
        )
        .any()
    ):

        raise RuntimeError(
            "Negative time_to_transition_days found in frozen "
            "post-progression targets."
        )

    ###########################################################################
    # Generic time-to-post-progression-transition bucket.
    #
    # This is intentionally NOT called "time to next treatment": for DEATH or
    # continued-therapy/censor transitions, the same field is time to that
    # observed transition.
    ###########################################################################

    time = result[
        "time_to_transition_days"
    ]

    bucket = np.full(
        len(
            result
        ),
        np.nan,
    )

    bucket[
        (
            time
            >= 0
        )
        & (
            time
            <= 28
        )
    ] = 0

    bucket[
        (
            time
            > 28
        )
        & (
            time
            <= 56
        )
    ] = 1

    bucket[
        (
            time
            > 56
        )
        & (
            time
            <= 90
        )
    ] = 2

    bucket[
        time
        > 90
    ] = 3

    result[
        "postprog_time_bucket"
    ] = bucket

    if len(
        result
    ) != 10955:

        raise RuntimeError(
            "Frozen post-progression target count changed: "
            f"{len(result)} != 10955"
        )

    audit = {
        "source_rows":
            int(
                len(
                    frame
                )
            ),

        "source_patients":
            int(
                result[
                    "patient_id"
                ].nunique()
            ),

        "merge_key":
            (
                "patient_id+scan_episode_id"
            ),

        "unique_event_keys":
            int(
                result[
                    [
                        "patient_id",
                        "scan_episode_id",
                    ]
                ]
                .drop_duplicates()
                .shape[
                    0
                ]
            ),

        "transition_type_counts":
            result[
                "postprog_transition_type_raw"
            ]
            .str.upper()
            .value_counts()
            .to_dict(),

        "resolved_class_rows":
            int(
                result[
                    "postprog_label"
                ]
                .notna()
                .sum()
            ),

        "resolved_time_rows":
            int(
                result[
                    "postprog_time_bucket"
                ]
                .notna()
                .sum()
            ),

        "time_field":
            "time_to_transition_days",

        "time_head_semantics":
            "time_to_observed_postprogression_transition",
    }

    return (
        result,
        audit,
    )


def attach_postprog_targets(
    index: pd.DataFrame,
    postprog: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
]:

    ###########################################################################
    # Exact event-level reconciliation.
    #
    # CKPT1 gives each post-progression target the exact canonical W3
    # scan_episode_id that generated it. No fuzzy date/line matching is needed.
    ###########################################################################

    left_keys = index[
        [
            "patient_id",
            "scan_episode_id",
            "scan_state",
        ]
    ].copy()

    if left_keys[
        [
            "patient_id",
            "scan_episode_id",
        ]
    ].duplicated().any():

        raise RuntimeError(
            "Authoritative CKPT5 scan keys are not unique."
        )

    right_keys = postprog[
        [
            "patient_id",
            "scan_episode_id",
        ]
    ]

    if right_keys.duplicated().any():

        raise RuntimeError(
            "Post-progression source event keys are not unique."
        )

    key_check = postprog[
        [
            "patient_id",
            "scan_episode_id",
        ]
    ].merge(
        left_keys,
        on=[
            "patient_id",
            "scan_episode_id",
        ],
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    unmatched = (
        key_check[
            "_merge"
        ]
        != "both"
    )

    if unmatched.any():

        raise RuntimeError(
            "Frozen post-progression target keys missing from "
            "authoritative W3 scans: "
            f"{int(unmatched.sum())}"
        )

    nonprogressive = (
        key_check[
            "scan_state"
        ]
        .astype(str)
        .str.upper()
        != "PROGRESSIVE"
    )

    if nonprogressive.any():

        raise RuntimeError(
            "Frozen post-progression target attached to "
            "non-progressive authoritative scan: "
            f"{int(nonprogressive.sum())}"
        )

    merge_columns = [
        "patient_id",
        "scan_episode_id",
        "postprog_progression_day",
        "postprog_treatment_line",
        "postprog_transition_type_raw",
        "postprog_transition_day",
        "time_to_transition_days",
        "postprog_label",
        "postprog_time_bucket",
    ]

    right = postprog[
        merge_columns
    ].copy()

    right[
        "postprog_source_match"
    ] = True

    merged = index.merge(
        right,
        on=[
            "patient_id",
            "scan_episode_id",
        ],
        how="left",
        validate="one_to_one",
    )

    matched = (
        merged[
            "postprog_source_match"
        ]
        .fillna(
            False
        )
        .astype(
            bool
        )
    )

    if int(
        matched.sum()
    ) != 10955:

        raise RuntimeError(
            "Exact post-progression merge did not preserve all "
            "10,955 frozen targets: "
            f"matched={int(matched.sum())}"
        )

    if not (
        merged.loc[
            matched,
            "scan_state",
        ]
        .astype(str)
        .str.upper()
        .eq(
            "PROGRESSIVE"
        )
        .all()
    ):

        raise RuntimeError(
            "Post-progression exact merge produced a "
            "non-progressive matched scan."
        )

    audit = {
        "selected_strategy":
            "patient_scan_episode_exact",

        "selected_keys": [
            "patient_id",
            "scan_episode_id",
        ],

        "source_rows":
            int(
                len(
                    postprog
                )
            ),

        "progressive_scan_rows":
            int(
                merged[
                    "scan_state"
                ]
                .astype(str)
                .str.upper()
                .eq(
                    "PROGRESSIVE"
                )
                .sum()
            ),

        "matched_progressive_scan_rows":
            int(
                matched.sum()
            ),

        "unmatched_source_rows":
            0,

        "matched_class_rows":
            int(
                (
                    matched
                    & merged[
                        "postprog_label"
                    ]
                    .notna()
                )
                .sum()
            ),

        "matched_time_rows":
            int(
                (
                    matched
                    & merged[
                        "postprog_time_bucket"
                    ]
                    .notna()
                )
                .sum()
            ),
    }

    return (
        merged,
        audit,
    )


###############################################################################
# Frozen CKPT3 encoder -> CHORD genomic samples
###############################################################################


def chord_genomic_samples(
    repo: Path,
    out: Path,
    patient_ids: set[str],
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    dict[str, Any],
]:

    ###########################################################################
    # Frozen CKPT3 implementation and local source discovery.
    ###########################################################################

    ckpt3 = import_ckpt3(
        repo
    )

    chord_root = (
        repo
        / "data"
        / "external_sources"
        / "msk_chord_2024"
        / "raw"
    )

    genie_root = (
        repo
        / "data"
        / "external_sources"
        / "genie_20_0_public"
    )

    chord_sources = (
        ckpt3.discover_sources(
            chord_root
        )
    )

    genie_sources = (
        ckpt3.discover_sources(
            genie_root
        )
    )

    if not chord_sources[
        "clinical_sample"
    ]:

        raise RuntimeError(
            "CHORD clinical sample table not found."
        )

    if not chord_sources[
        "mutation"
    ]:

        raise RuntimeError(
            "CHORD mutation table not found."
        )

    if not genie_sources[
        "gene_panel_definition"
    ]:

        raise RuntimeError(
            "GENIE20 shared gene-panel definitions not found."
        )

    ###########################################################################
    # CHORD clinical genomic samples.
    #
    # CHORD explicitly records GENE_PANEL. The study bundle's
    # data_gene_panel_matrix is blank for these samples, so GENE_PANEL is the
    # authoritative sample assay assignment.
    ###########################################################################

    clinical_path = Path(
        chord_sources[
            "clinical_sample"
        ][
            0
        ]
    )

    clinical = ckpt3.read_table(
        clinical_path
    )

    sample_col = find_col(
        clinical.columns,
        [
            "SAMPLE_ID",
            "sample_id",
        ],
    )

    patient_col = find_col(
        clinical.columns,
        [
            "PATIENT_ID",
            "patient_id",
        ],
    )

    panel_col = find_col(
        clinical.columns,
        [
            "GENE_PANEL",
        ],
    )

    seq_col = find_col(
        clinical.columns,
        [
            "SEQ_DATE",
            "SEQUENCING_DATE",
            "SEQ_RESULT_DATE",
            "SEQUENCING_RESULT_DATE",
        ],
        required=False,
    )

    sample = pd.DataFrame(
        {
            "sample_id":
                clinical[
                    sample_col
                ]
                .astype(str)
                .str.strip(),

            "patient_id":
                clinical[
                    patient_col
                ]
                .astype(str)
                .str.strip(),

            "chord_panel_id":
                clinical[
                    panel_col
                ]
                .astype(str)
                .str.strip()
                .str.upper(),
        }
    )

    if seq_col is not None:

        sample[
            "availability_day"
        ] = pd.to_numeric(
            clinical[
                seq_col
            ],
            errors="coerce",
        )

    else:

        sample[
            "availability_day"
        ] = np.nan

    sample = sample[
        sample[
            "patient_id"
        ].isin(
            patient_ids
        )
    ].copy()

    ###########################################################################
    # Remove byte-identical duplicate sample rows only. Conflicting duplicate
    # sample IDs are forbidden.
    ###########################################################################

    if sample[
        "sample_id"
    ].duplicated(
        keep=False
    ).any():

        duplicate = sample[
            sample[
                "sample_id"
            ].duplicated(
                keep=False
            )
        ]

        conflict = (
            duplicate
            .groupby(
                "sample_id",
                observed=True,
            )[
                [
                    "patient_id",
                    "chord_panel_id",
                ]
            ]
            .nunique()
        )

        if (
            conflict
            > 1
        ).any().any():

            raise RuntimeError(
                "Conflicting duplicate CHORD genomic sample IDs."
            )

        sample = (
            sample
            .drop_duplicates(
                subset=[
                    "sample_id",
                ],
                keep="first",
            )
            .copy()
        )

    sample = sample.reset_index(
        drop=True
    )

    ###########################################################################
    # Frozen mBC CHORD cohort currently has exactly one genomic sample per
    # patient. This makes the patient-level canonical availability event an
    # unambiguous fallback when raw SEQ_DATE is not exposed by this release.
    ###########################################################################

    patient_sample_count = (
        sample[
            "patient_id"
        ]
        .value_counts()
    )

    ###########################################################################
    # Fill unresolved availability from CKPT1's canonical
    # GENOMIC_RESULT_AVAILABLE event.
    ###########################################################################

    unresolved_availability = (
        sample[
            "availability_day"
        ].isna()
    )

    if unresolved_availability.any():

        events = pd.read_parquet(
            repo
            / "artifacts"
            / "checkpoint1"
            / "canonical"
            / "chord_canonical_events.parquet"
        )

        event_patient_col = find_col(
            events.columns,
            [
                "patient_id",
            ],
        )

        event_type_col = find_col(
            events.columns,
            [
                "event_type",
            ],
        )

        event_availability_col = find_col(
            events.columns,
            [
                "availability_day",
                "event_day",
            ],
        )

        genomic_events = events[
            events[
                event_type_col
            ]
            .astype(str)
            .str.strip()
            .str.upper()
            == "GENOMIC_RESULT_AVAILABLE"
        ].copy()

        genomic_events[
            "patient_id"
        ] = (
            genomic_events[
                event_patient_col
            ]
            .astype(str)
            .str.strip()
        )

        genomic_events[
            "availability_day"
        ] = pd.to_numeric(
            genomic_events[
                event_availability_col
            ],
            errors="coerce",
        )

        genomic_patient_day = (
            genomic_events
            .dropna(
                subset=[
                    "availability_day",
                ]
            )
            .groupby(
                "patient_id",
                observed=True,
            )[
                "availability_day"
            ]
            .min()
        )

        fallback = (
            sample[
                "patient_id"
            ]
            .map(
                genomic_patient_day
            )
        )

        #######################################################################
        # Patient-level fallback is safe only for a single genomic sample.
        #######################################################################

        unique_sample_patient = (
            sample[
                "patient_id"
            ]
            .map(
                patient_sample_count
            )
            == 1
        )

        fill = (
            unresolved_availability
            & unique_sample_patient
            & fallback.notna()
        )

        sample.loc[
            fill,
            "availability_day",
        ] = fallback[
            fill
        ]

    if (
        seq_col is not None
        and clinical[
            seq_col
        ]
        .notna()
        .any()
    ):

        availability_source = (
            "SEQ_DATE_WHEN_PRESENT_WITH_"
            "CKPT1_GENOMIC_RESULT_AVAILABLE_SINGLE_SAMPLE_FALLBACK"
        )

    else:

        availability_source = (
            "CKPT1_GENOMIC_RESULT_AVAILABLE_SINGLE_SAMPLE_FALLBACK"
        )

    ###########################################################################
    # Exact shared assay definitions.
    #
    # GENIE20 contributes only the stable MSK-IMPACT assay gene lists.
    # No GENIE patient/sample/mutation/outcome row is used here.
    ###########################################################################

    genie_panel_defs_raw = (
        ckpt3.build_panel_definitions(
            genie_sources[
                "gene_panel_definition"
            ]
        )
    )

    genie_panel_defs = {
        str(
            panel_id
        ):
            {
                str(
                    gene
                )
                .strip()
                .upper()
                for gene
                in genes
            }
        for panel_id, genes
        in genie_panel_defs_raw.items()
    }

    impact_crosswalk = {
        "IMPACT341":
            "MSK-IMPACT341",

        "IMPACT410":
            "MSK-IMPACT410",

        "IMPACT468":
            "MSK-IMPACT468",

        "IMPACT505":
            "MSK-IMPACT505",
    }

    expected_panel_sizes = {
        "IMPACT341":
            341,

        "IMPACT410":
            410,

        "IMPACT468":
            468,

        "IMPACT505":
            505,
    }

    for (
        chord_panel,
        stable_panel,
    ) in impact_crosswalk.items():

        if (
            stable_panel
            not in genie_panel_defs
        ):

            raise RuntimeError(
                "Missing exact shared assay definition: "
                f"{stable_panel}"
            )

        observed_count = len(
            genie_panel_defs[
                stable_panel
            ]
        )

        expected_count = (
            expected_panel_sizes[
                chord_panel
            ]
        )

        if (
            observed_count
            != expected_count
        ):

            raise RuntimeError(
                "Shared MSK-IMPACT definition size changed: "
                f"{stable_panel}={observed_count}, "
                f"expected={expected_count}"
            )

    ###########################################################################
    # Every CHORD sample must use exactly one of the four audited solid-tumor
    # IMPACT assays.
    ###########################################################################

    unexpected_panels = (
        set(
            sample[
                "chord_panel_id"
            ]
            .unique()
        )
        - set(
            impact_crosswalk
        )
    )

    if unexpected_panels:

        raise RuntimeError(
            "Unexpected CHORD GENE_PANEL values: "
            f"{sorted(unexpected_panels)}"
        )

    sample[
        "panel_id"
    ] = (
        sample[
            "chord_panel_id"
        ]
        .map(
            impact_crosswalk
        )
    )

    ###########################################################################
    # Frozen CKPT3 468-gene encoder space.
    ###########################################################################

    gene_space = [
        line.strip()
        for line
        in (
            repo
            / "artifacts"
            / "checkpoint3"
            / "prepared"
            / "gene_space_468.txt"
        )
        .read_text(
            encoding="utf-8"
        )
        .splitlines()
        if line.strip()
    ]

    if len(
        gene_space
    ) != 468:

        raise RuntimeError(
            "Frozen CKPT3 gene space is not 468 genes."
        )

    if len(
        set(
            gene_space
        )
    ) != 468:

        raise RuntimeError(
            "Frozen CKPT3 gene space contains duplicate genes."
        )

    gene_space = [
        gene.upper()
        for gene in gene_space
    ]

    gene_index = {
        gene:
            index
        for index, gene
        in enumerate(
            gene_space
        )
    }

    ###########################################################################
    # Build panel-aware coverage matrix.
    ###########################################################################

    n = len(
        sample
    )

    coverage = np.zeros(
        (
            n,
            468,
        ),
        dtype=np.float32,
    )

    alteration = np.zeros(
        (
            n,
            468,
        ),
        dtype=np.float32,
    )

    panel_selected_coverage = {}

    for chord_panel, stable_panel in impact_crosswalk.items():

        covered_genes = (
            genie_panel_defs[
                stable_panel
            ]
            & set(
                gene_space
            )
        )

        panel_selected_coverage[
            chord_panel
        ] = int(
            len(
                covered_genes
            )
        )

        if not covered_genes:

            raise RuntimeError(
                f"{stable_panel} covers zero frozen CKPT3 genes."
            )

    for row, (
        chord_panel,
        stable_panel,
    ) in enumerate(
        zip(
            sample[
                "chord_panel_id"
            ],
            sample[
                "panel_id"
            ],
        )
    ):

        if (
            impact_crosswalk[
                chord_panel
            ]
            != stable_panel
        ):

            raise RuntimeError(
                "Internal IMPACT crosswalk inconsistency."
            )

        for gene in genie_panel_defs[
            stable_panel
        ]:

            column = gene_index.get(
                gene
            )

            if column is not None:

                coverage[
                    row,
                    column,
                ] = 1.0

    ###########################################################################
    # Exact frozen coverage sizes established by the preceding independent
    # assay audit.
    ###########################################################################

    expected_selected_coverage = {
        "IMPACT341":
            288,

        "IMPACT410":
            325,

        "IMPACT468":
            354,

        "IMPACT505":
            362,
    }

    if (
        panel_selected_coverage
        != expected_selected_coverage
    ):

        raise RuntimeError(
            "Frozen-space assay coverage changed: "
            f"observed={panel_selected_coverage} "
            f"expected={expected_selected_coverage}"
        )

    ###########################################################################
    # CHORD mutations.
    ###########################################################################

    sample_row = {
        sample_id:
            row
        for row, sample_id
        in enumerate(
            sample[
                "sample_id"
            ]
        )
    }

    selected_mutation_rows = 0
    outside_coverage_rows = 0

    for raw_path in chord_sources[
        "mutation"
    ]:

        mutation_path = Path(
            raw_path
        )

        columns = ckpt3.mutation_header(
            mutation_path
        )

        for chunk in pd.read_csv(
            mutation_path,
            sep="\t",
            comment="#",
            dtype=str,
            usecols=[
                columns[
                    "sample"
                ],
                columns[
                    "gene"
                ],
            ],
            chunksize=200000,
            low_memory=False,
        ):

            mutation_samples = (
                chunk[
                    columns[
                        "sample"
                    ]
                ]
                .astype(str)
                .str.strip()
            )

            mutation_genes = (
                chunk[
                    columns[
                        "gene"
                    ]
                ]
                .astype(str)
                .str.strip()
                .str.upper()
            )

            for (
                sample_id,
                gene,
            ) in zip(
                mutation_samples,
                mutation_genes,
            ):

                row = sample_row.get(
                    sample_id
                )

                column = gene_index.get(
                    gene
                )

                if (
                    row is None
                    or column is None
                ):

                    continue

                selected_mutation_rows += 1

                if (
                    coverage[
                        row,
                        column,
                    ]
                    <= 0
                ):

                    outside_coverage_rows += 1
                    continue

                alteration[
                    row,
                    column,
                ] = 1.0

    ###########################################################################
    # Exact source integrity check from the completed bridge audit.
    ###########################################################################

    bridge_path = (
        repo
        / "artifacts"
        / "checkpoint5"
        / "genomic_panel_bridge_audit.json"
    )

    bridge = json.loads(
        bridge_path.read_text(
            encoding="utf-8"
        )
    )

    if (
        selected_mutation_rows
        != int(
            bridge[
                "selected_mutation_rows"
            ]
        )
    ):

        raise RuntimeError(
            "Selected mutation-row count differs from frozen bridge audit: "
            f"{selected_mutation_rows} != "
            f"{bridge['selected_mutation_rows']}"
        )

    if outside_coverage_rows != 0:

        raise RuntimeError(
            "Mutation observed outside exact MSK-IMPACT coverage: "
            f"{outside_coverage_rows}"
        )

    ###########################################################################
    # Resolution / descriptive columns.
    ###########################################################################

    panel_resolved = (
        coverage.sum(
            axis=1
        )
        > 0
    )

    availability_resolved = (
        sample[
            "availability_day"
        ]
        .notna()
        .to_numpy()
    )

    usable = (
        panel_resolved
        & availability_resolved
    )

    sample[
        "panel_resolved"
    ] = panel_resolved

    sample[
        "availability_resolved"
    ] = availability_resolved

    sample[
        "coverage_gene_count"
    ] = (
        coverage.sum(
            axis=1
        )
        .astype(
            int
        )
    )

    sample[
        "selected_alteration_count"
    ] = (
        alteration.sum(
            axis=1
        )
        .astype(
            int
        )
    )

    ###########################################################################
    # Frozen mBC genomic cohort invariants.
    ###########################################################################

    if len(
        sample
    ) != 3351:

        raise RuntimeError(
            "Unexpected CHORD mBC genomic sample count: "
            f"{len(sample)} != 3351"
        )

    if (
        sample[
            "patient_id"
        ]
        .nunique()
        != 3351
    ):

        raise RuntimeError(
            "Unexpected CHORD mBC genomic patient count."
        )

    if not bool(
        panel_resolved.all()
    ):

        raise RuntimeError(
            "Not all CHORD mBC genomic samples have resolved coverage."
        )

    if not bool(
        availability_resolved.all()
    ):

        unresolved = sample.loc[
            ~availability_resolved,
            [
                "patient_id",
                "sample_id",
            ],
        ]

        raise RuntimeError(
            "Not all CHORD mBC genomic samples have availability timing: "
            f"{len(unresolved)} unresolved"
        )

    ###########################################################################
    # Load frozen CKPT3 encoder.
    ###########################################################################

    checkpoint = torch.load(
        repo
        / "artifacts"
        / "checkpoint3"
        / "genomic_encoder.pt",
        map_location="cpu",
        weights_only=False,
    )

    if int(
        checkpoint[
            "gene_count"
        ]
    ) != 468:

        raise RuntimeError(
            "Frozen CKPT3 checkpoint gene_count != 468."
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = ckpt3.GenomicEncoder(
        gene_count=checkpoint[
            "gene_count"
        ],
        cancer_classes=checkpoint[
            "cancer_classes"
        ],
        sample_type_classes=checkpoint[
            "sample_type_classes"
        ],
        hidden=checkpoint[
            "hidden"
        ],
        heads=checkpoint[
            "heads"
        ],
        dropout=checkpoint[
            "dropout"
        ],
    ).to(
        device
    )

    model.load_state_dict(
        checkpoint[
            "model_state"
        ]
    )

    model.eval()

    edge_index = (
        torch.from_numpy(
            np.load(
                repo
                / "artifacts"
                / "checkpoint3"
                / "prepared"
                / "graph_edges.npy"
            )
        )
        .long()
        .to(
            device
        )
    )

    embeddings = np.zeros(
        (
            n,
            GENOMIC_DIM,
        ),
        dtype=np.float16,
    )

    usable_rows = np.where(
        usable
    )[
        0
    ]

    batch_size = 512

    with torch.no_grad():

        for start_row in range(
            0,
            len(
                usable_rows
            ),
            batch_size,
        ):

            rows = usable_rows[
                start_row:
                start_row
                + batch_size
            ]

            mutation_batch = (
                torch.from_numpy(
                    alteration[
                        rows
                    ]
                )
                .to(
                    device
                )
            )

            coverage_batch = (
                torch.from_numpy(
                    coverage[
                        rows
                    ]
                )
                .to(
                    device
                )
            )

            output = model(
                mutation_batch,
                coverage_batch,
                edge_index,
            )

            embedding_batch = (
                output[
                    "embedding"
                ]
                .float()
                .cpu()
                .numpy()
            )

            if (
                embedding_batch.shape[
                    1
                ]
                != GENOMIC_DIM
            ):

                raise RuntimeError(
                    "Frozen CKPT3 embedding dimension changed: "
                    f"{embedding_batch.shape}"
                )

            embeddings[
                rows
            ] = embedding_batch.astype(
                np.float16
            )

    if not np.isfinite(
        embeddings[
            usable_rows
        ].astype(
            np.float32
        )
    ).all():

        raise RuntimeError(
            "Non-finite CHORD genomic embedding detected."
        )

    if float(
        np.abs(
            embeddings[
                usable_rows
            ].astype(
                np.float32
            )
        ).sum()
    ) <= 0:

        raise RuntimeError(
            "All usable CHORD genomic embeddings are zero."
        )

    ###########################################################################
    # Persistent genomic preparation audit.
    ###########################################################################

    panel_counts = (
        sample[
            "chord_panel_id"
        ]
        .value_counts()
        .to_dict()
    )

    audit = {
        "samples_in_mbc_patients":
            int(
                n
            ),

        "panel_resolved_samples":
            int(
                panel_resolved.sum()
            ),

        "availability_resolved_samples":
            int(
                availability_resolved.sum()
            ),

        "usable_embedding_samples":
            int(
                usable.sum()
            ),

        "patients_with_usable_embedding":
            int(
                sample.loc[
                    usable,
                    "patient_id",
                ]
                .nunique()
            ),

        "availability_source":
            availability_source,

        "panel_assignment_source":
            "CHORD_CLINICAL_SAMPLE.GENE_PANEL",

        "panel_definition_count":
            4,

        "panel_definition_source":
            "GENIE20_EXACT_SHARED_MSK_IMPACT_ASSAY_METADATA",

        "panel_definition_semantics":
            (
                "GENIE20 contributes only stable MSK-IMPACT assay "
                "gene-list definitions. CHORD patient/sample/mutation/"
                "timing/outcome data remain CHORD-derived."
            ),

        "impact_crosswalk":
            impact_crosswalk,

        "panel_counts": {
            str(
                key
            ):
                int(
                    value
                )
            for key, value
            in panel_counts.items()
        },

        "frozen_gene_space_coverage":
            panel_selected_coverage,

        "selected_mutation_rows":
            int(
                selected_mutation_rows
            ),

        "outside_coverage_rows":
            int(
                outside_coverage_rows
            ),

        "mutation_files":
            chord_sources[
                "mutation"
            ],

        "strict_landmark_rule":
            (
                "A genomic sample is available at a scan only if "
                "sample availability_day < scan landmark_day. "
                "Same-day genomic results are excluded from both "
                "PRE and POST views."
            ),
    }

    ###########################################################################
    # Persist.
    ###########################################################################

    atomic_parquet(
        out
        / "prepared"
        / "chord_genomic_samples.parquet",
        sample,
    )

    embedding_path = (
        out
        / "prepared"
        / "chord_genomic_sample_embeddings_f16.npy"
    )

    embedding_tmp = embedding_path.with_suffix(
        ".npy.tmp"
    )

    with open(
        embedding_tmp,
        "wb",
    ) as handle:

        np.save(
            handle,
            embeddings,
        )

    validation_embedding = np.load(
        embedding_tmp,
        mmap_mode="r",
    )

    if validation_embedding.shape != (
        3351,
        GENOMIC_DIM,
    ):

        raise RuntimeError(
            "Genomic sample embedding write verification failed: "
            f"{validation_embedding.shape}"
        )

    os.replace(
        embedding_tmp,
        embedding_path,
    )

    atomic_json(
        out
        / "prepared"
        / "chord_genomic_audit.json",
        audit,
    )

    return (
        sample,
        embeddings,
        audit,
    )


def align_genomics_to_scans(
    scan_index: pd.DataFrame,
    genomic_samples: pd.DataFrame,
    genomic_embeddings: np.ndarray,
) -> tuple[
    pd.DataFrame,
    np.ndarray,
]:

    ###########################################################################
    # Prepare usable genomic samples.
    ###########################################################################

    samples = genomic_samples.copy()

    samples[
        "genomic_sample_row"
    ] = np.arange(
        len(
            samples
        ),
        dtype=np.int64,
    )

    samples = samples[
        samples[
            "panel_resolved"
        ]
        .fillna(
            False
        )
        .astype(
            bool
        )
        & samples[
            "availability_resolved"
        ]
        .fillna(
            False
        )
        .astype(
            bool
        )
    ].copy()

    samples[
        "patient_id"
    ] = (
        samples[
            "patient_id"
        ]
        .astype(str)
        .str.strip()
    )

    samples[
        "availability_day"
    ] = pd.to_numeric(
        samples[
            "availability_day"
        ],
        errors="coerce",
    )

    samples = samples.dropna(
        subset=[
            "patient_id",
            "availability_day",
        ]
    ).copy()

    if samples.empty:

        raise RuntimeError(
            "No usable genomic samples available for scan alignment."
        )

    if genomic_embeddings.ndim != 2:

        raise RuntimeError(
            "Genomic sample embedding array must be rank 2."
        )

    if genomic_embeddings.shape[
        1
    ] != GENOMIC_DIM:

        raise RuntimeError(
            "Unexpected genomic embedding dimension: "
            f"{genomic_embeddings.shape}"
        )

    if genomic_embeddings.shape[
        0
    ] != len(
        genomic_samples
    ):

        raise RuntimeError(
            "Genomic sample table / embedding row mismatch: "
            f"samples={len(genomic_samples)} "
            f"embeddings={genomic_embeddings.shape[0]}"
        )

    ###########################################################################
    # Prepare scan rows while preserving their exact input order.
    ###########################################################################

    merged = (
        scan_index
        .copy()
        .reset_index(
            drop=True
        )
    )

    merged[
        "patient_id"
    ] = (
        merged[
            "patient_id"
        ]
        .astype(str)
        .str.strip()
    )

    merged[
        "landmark_day"
    ] = pd.to_numeric(
        merged[
            "landmark_day"
        ],
        errors="raise",
    ).astype(
        float
    )

    n_scans = len(
        merged
    )

    ###########################################################################
    # Output metadata columns are deliberately empty until a sample has
    # actually become available. This prevents future assay metadata from
    # leaking into an earlier scan row.
    ###########################################################################

    merged[
        "sample_id"
    ] = pd.Series(
        [
            None
        ]
        * n_scans,
        dtype="object",
    )

    merged[
        "availability_day"
    ] = np.nan

    merged[
        "genomic_sample_row"
    ] = pd.Series(
        [
            pd.NA
        ]
        * n_scans,
        dtype="Int64",
    )

    merged[
        "panel_id"
    ] = pd.Series(
        [
            None
        ]
        * n_scans,
        dtype="object",
    )

    merged[
        "coverage_gene_count"
    ] = np.nan

    ###########################################################################
    # Patient-specific latest-prior-sample lookup.
    #
    # np.searchsorted(..., side="left") - 1 implements STRICTLY PRIOR:
    #
    #     availability_day < landmark_day
    #
    # A same-day genomic result is therefore not available in either PRE or
    # POST prediction at that scan.
    ###########################################################################

    sample_groups = {}

    for patient_id, group in samples.groupby(
        "patient_id",
        observed=True,
        sort=False,
    ):

        ordered = (
            group
            .sort_values(
                [
                    "availability_day",
                    "genomic_sample_row",
                ],
                kind="mergesort",
            )
            .reset_index(
                drop=True
            )
        )

        availability = ordered[
            "availability_day"
        ].to_numpy(
            dtype=float
        )

        if (
            len(
                availability
            )
            > 1
            and np.any(
                np.diff(
                    availability
                )
                < 0
            )
        ):

            raise RuntimeError(
                "Internal genomic availability ordering failure "
                f"for patient {patient_id}"
            )

        sample_groups[
            patient_id
        ] = (
            ordered,
            availability,
        )

    matched_scan_rows = 0

    for patient_id, scan_positions in merged.groupby(
        "patient_id",
        observed=True,
        sort=False,
    ).groups.items():

        if patient_id not in sample_groups:

            continue

        (
            patient_samples,
            availability_days,
        ) = sample_groups[
            patient_id
        ]

        scan_positions = np.asarray(
            list(
                scan_positions
            ),
            dtype=np.int64,
        )

        scan_days = (
            merged.loc[
                scan_positions,
                "landmark_day",
            ]
            .to_numpy(
                dtype=float
            )
        )

        sample_positions = (
            np.searchsorted(
                availability_days,
                scan_days,
                side="left",
            )
            - 1
        )

        valid = (
            sample_positions
            >= 0
        )

        if not valid.any():

            continue

        target_scan_rows = scan_positions[
            valid
        ]

        source_positions = sample_positions[
            valid
        ]

        selected = (
            patient_samples
            .iloc[
                source_positions
            ]
            .reset_index(
                drop=True
            )
        )

        selected_availability = (
            selected[
                "availability_day"
            ]
            .to_numpy(
                dtype=float
            )
        )

        selected_scan_days = (
            merged.loc[
                target_scan_rows,
                "landmark_day",
            ]
            .to_numpy(
                dtype=float
            )
        )

        #######################################################################
        # Critical leakage assertion.
        #######################################################################

        if not np.all(
            selected_availability
            < selected_scan_days
        ):

            raise RuntimeError(
                "Same-day/future genomic sample selected during "
                "patient-specific alignment."
            )

        merged.loc[
            target_scan_rows,
            "sample_id",
        ] = (
            selected[
                "sample_id"
            ]
            .astype(str)
            .to_numpy()
        )

        merged.loc[
            target_scan_rows,
            "availability_day",
        ] = selected_availability

        merged.loc[
            target_scan_rows,
            "genomic_sample_row",
        ] = (
            selected[
                "genomic_sample_row"
            ]
            .astype(
                int
            )
            .to_numpy()
        )

        merged.loc[
            target_scan_rows,
            "panel_id",
        ] = (
            selected[
                "panel_id"
            ]
            .astype(str)
            .to_numpy()
        )

        merged.loc[
            target_scan_rows,
            "coverage_gene_count",
        ] = (
            selected[
                "coverage_gene_count"
            ]
            .astype(
                float
            )
            .to_numpy()
        )

        matched_scan_rows += int(
            len(
                target_scan_rows
            )
        )

    ###########################################################################
    # Scan-level genomic embedding.
    ###########################################################################

    scan_embeddings = np.zeros(
        (
            n_scans,
            GENOMIC_DIM,
        ),
        dtype=np.float16,
    )

    available = (
        merged[
            "genomic_sample_row"
        ].notna()
    )

    if available.any():

        embedding_rows = (
            merged.loc[
                available,
                "genomic_sample_row",
            ]
            .astype(
                int
            )
            .to_numpy()
        )

        if (
            embedding_rows.min()
            < 0
            or embedding_rows.max()
            >= genomic_embeddings.shape[
                0
            ]
        ):

            raise RuntimeError(
                "Aligned genomic sample row is outside embedding array."
            )

        scan_embeddings[
            available.to_numpy()
        ] = genomic_embeddings[
            embedding_rows
        ]

    merged[
        "genomic_available"
    ] = available.astype(
        np.float32
    )

    merged[
        "genomic_age_days"
    ] = (
        merged[
            "landmark_day"
        ]
        - merged[
            "availability_day"
        ]
    )

    ###########################################################################
    # Persistent leakage checks.
    ###########################################################################

    if (
        merged.loc[
            available,
            "genomic_age_days",
        ]
        <= 0
    ).any():

        bad = merged.loc[
            available
            & (
                merged[
                    "genomic_age_days"
                ]
                <= 0
            ),
            [
                "patient_id",
                "scan_episode_id",
                "landmark_day",
                "sample_id",
                "availability_day",
                "genomic_age_days",
            ],
        ].head(
            20
        )

        raise RuntimeError(
            "Same-day/future genomic result leaked into "
            "scan representation:\n"
            + bad.to_string(
                index=False
            )
        )

    ###########################################################################
    # Unavailable scans must contain no genomic metadata and exact-zero
    # embedding.
    ###########################################################################

    unavailable = (
        ~available
    )

    if unavailable.any():

        if (
            merged.loc[
                unavailable,
                "sample_id",
            ]
            .notna()
            .any()
        ):

            raise RuntimeError(
                "Unavailable scan exposes future genomic sample_id."
            )

        if (
            merged.loc[
                unavailable,
                "availability_day",
            ]
            .notna()
            .any()
        ):

            raise RuntimeError(
                "Unavailable scan exposes future genomic availability_day."
            )

        if not np.all(
            scan_embeddings[
                unavailable.to_numpy()
            ]
            == 0
        ):

            raise RuntimeError(
                "Unavailable scan has nonzero genomic embedding."
            )

    ###########################################################################
    # Available scan embeddings must be finite.
    ###########################################################################

    if available.any():

        available_embedding = (
            scan_embeddings[
                available.to_numpy()
            ]
            .astype(
                np.float32
            )
        )

        if not np.isfinite(
            available_embedding
        ).all():

            raise RuntimeError(
                "Non-finite aligned genomic embedding."
            )

    ###########################################################################
    # Descriptive alignment attributes used for audit.
    ###########################################################################

    merged[
        "genomic_alignment_rule"
    ] = np.where(
        available,
        "LATEST_SAMPLE_STRICTLY_BEFORE_LANDMARK",
        "NO_SAMPLE_AVAILABLE_BEFORE_LANDMARK",
    )

    merged[
        "genomic_prior_sample_count"
    ] = 0

    ###########################################################################
    # Count how many genomic samples were already available at each scan.
    # This is descriptive only and uses the same strict-before rule.
    ###########################################################################

    for patient_id, scan_positions in merged.groupby(
        "patient_id",
        observed=True,
        sort=False,
    ).groups.items():

        if patient_id not in sample_groups:

            continue

        (
            _,
            availability_days,
        ) = sample_groups[
            patient_id
        ]

        scan_positions = np.asarray(
            list(
                scan_positions
            ),
            dtype=np.int64,
        )

        scan_days = (
            merged.loc[
                scan_positions,
                "landmark_day",
            ]
            .to_numpy(
                dtype=float
            )
        )

        prior_counts = np.searchsorted(
            availability_days,
            scan_days,
            side="left",
        )

        merged.loc[
            scan_positions,
            "genomic_prior_sample_count",
        ] = prior_counts

    merged[
        "genomic_prior_sample_count"
    ] = (
        merged[
            "genomic_prior_sample_count"
        ]
        .astype(
            int
        )
    )

    ###########################################################################
    # Internal row-count/order contract.
    ###########################################################################

    if len(
        merged
    ) != len(
        scan_index
    ):

        raise RuntimeError(
            "Genomic alignment changed scan row count."
        )

    if matched_scan_rows != int(
        available.sum()
    ):

        raise RuntimeError(
            "Genomic alignment accounting mismatch: "
            f"matched={matched_scan_rows} "
            f"available={int(available.sum())}"
        )

    return (
        merged,
        scan_embeddings,
    )


###############################################################################
# Model preparation
###############################################################################


def prepare(
    repo: Path,
    out: Path,
) -> None:

    upstream = validate_frozen_upstream(
        repo
    )

    (
        cache_index,
        temporal_embeddings,
    ) = load_temporal_cache(
        repo
    )

    scans = load_scans(
        repo
    )

    authoritative_keys = set(
        zip(
            scans[
                "patient_id"
            ],
            scans[
                "scan_episode_id"
            ],
        )
    )

    cache_keys = set(
        zip(
            cache_index[
                "patient_id"
            ],
            cache_index[
                "scan_episode_id"
            ],
        )
    )

    if (
        authoritative_keys
        != cache_keys
    ):

        raise RuntimeError(
            "CKPT4 temporal-cache keys do not exactly equal "
            "authoritative CKPT1 W3 scan keys."
        )

    ###########################################################################
    # Align cache rows to authoritative scans.
    ###########################################################################

    index = scans.merge(
        cache_index[
            [
                "patient_id",
                "scan_episode_id",
                "temporal_row",
            ]
            + (
                [
                    "cache_eligible_residual_pfs",
                ]
                if (
                    "cache_eligible_residual_pfs"
                    in cache_index.columns
                )
                else []
            )
        ],
        on=[
            "patient_id",
            "scan_episode_id",
        ],
        how="inner",
        validate="one_to_one",
    )

    if len(
        index
    ) != 49189:

        raise RuntimeError(
            f"Expected 49,189 aligned scans, got {len(index)}"
        )

    temporal_rows = index[
        "temporal_row"
    ].to_numpy(
        dtype=int
    )

    aligned_temporal = np.asarray(
        temporal_embeddings[
            temporal_rows
        ],
        dtype=np.float16,
    )

    ###########################################################################
    # Patient splits / line context.
    ###########################################################################

    splits = load_splits(
        repo
    )

    lines = load_lines(
        repo
    )

    index = index.merge(
        splits,
        on="patient_id",
        how="inner",
        validate="many_to_one",
    )

    index = index.merge(
        lines,
        on=[
            "patient_id",
            "line",
        ],
        how="left",
        validate="many_to_one",
    )

    if len(
        index
    ) != 49189:

        raise RuntimeError(
            "Split/line merge unexpectedly removed scan rows."
        )

    index[
        "elapsed_on_line_days"
    ] = (
        index[
            "landmark_day"
        ]
        - index[
            "line_start_day"
        ]
    )

    if (
        index[
            "elapsed_on_line_days"
        ]
        < 0
    ).any():

        raise RuntimeError(
            "A scan occurs before its reconstructed treatment-line start."
        )

    index = (
        index
        .sort_values(
            [
                "patient_id",
                "landmark_day",
                "scan_episode_id",
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    index[
        "scan_number_patient"
    ] = (
        index.groupby(
            "patient_id",
            observed=True,
        )
        .cumcount()
        + 1
    )

    index[
        "scan_number_line"
    ] = (
        index.groupby(
            [
                "patient_id",
                "line",
            ],
            observed=True,
        )
        .cumcount()
        + 1
    )

    ###########################################################################
    # Need to realign temporal array because index was re-sorted.
    ###########################################################################

    cache_lookup = {
        (
            patient,
            episode,
        ):
            int(
                row
            )
        for patient, episode, row
        in zip(
            cache_index[
                "patient_id"
            ],
            cache_index[
                "scan_episode_id"
            ],
            cache_index[
                "temporal_row"
            ],
        )
    }

    temporal_rows = np.asarray(
        [
            cache_lookup[
                (
                    patient,
                    episode,
                )
            ]
            for patient, episode
            in zip(
                index[
                    "patient_id"
                ],
                index[
                    "scan_episode_id"
                ],
            )
        ],
        dtype=int,
    )

    aligned_temporal = np.asarray(
        temporal_embeddings[
            temporal_rows
        ],
        dtype=np.float16,
    )

    ###########################################################################
    # Current scan feature matrix.
    ###########################################################################

    (
        current_scan_features,
        current_scan_feature_names,
    ) = scan_features(
        index
    )

    ###########################################################################
    # Survival labels.
    ###########################################################################

    targets = load_survival_targets(
        repo
    )

    if (
        "scan_episode_id"
        in targets.columns
    ):

        index = index.merge(
            targets[
                [
                    "patient_id",
                    "scan_episode_id",
                    "target_type",
                    "target_time_days",
                ]
                + (
                    [
                        "landmark_eligible",
                    ]
                    if (
                        "landmark_eligible"
                        in targets.columns
                    )
                    else []
                )
            ],
            on=[
                "patient_id",
                "scan_episode_id",
            ],
            how="left",
            validate="one_to_one",
        )

    else:

        index = index.merge(
            targets[
                [
                    "patient_id",
                    "target_landmark_day",
                    "target_type",
                    "target_time_days",
                ]
                + (
                    [
                        "landmark_eligible",
                    ]
                    if (
                        "landmark_eligible"
                        in targets.columns
                    )
                    else []
                )
            ],
            left_on=[
                "patient_id",
                "landmark_day",
            ],
            right_on=[
                "patient_id",
                "target_landmark_day",
            ],
            how="left",
            validate="one_to_one",
        )

    index[
        "survival_cause"
    ] = [
        cause_from_target(
            value
        )
        for value
        in index[
            "target_type"
        ]
    ]

    index[
        "survival_time_days"
    ] = pd.to_numeric(
        index[
            "target_time_days"
        ],
        errors="coerce",
    )

    survival_resolved = (
        index[
            "survival_cause"
        ].notna()
        & index[
            "survival_time_days"
        ].notna()
        & (
            index[
                "survival_time_days"
            ]
            > 0
        )
    )

    ###########################################################################
    # Supervised residual-PFS eligibility comes from CKPT1 landmarks.
    #
    # CKPT4's `cache_eligible_residual_pfs` is a broader representation-cache
    # attribute over all authoritative W3 scans. It must NOT redefine the
    # supervised landmark population.
    ###########################################################################

    if (
        "landmark_eligible"
        not in index.columns
    ):

        raise RuntimeError(
            "CKPT1 eligible_primary was not materialized into "
            "the supervised scan table."
        )

    eligible = (
        index[
            "landmark_eligible"
        ]
        .fillna(
            False
        )
        .astype(
            bool
        )
    )

    index[
        "ckpt1_primary_landmark"
    ] = eligible

    ###########################################################################
    # Keep CKPT4 cache eligibility only as a descriptive/audit field.
    ###########################################################################

    if (
        "cache_eligible_residual_pfs"
        in index.columns
    ):

        index[
            "ckpt4_cache_eligible_residual_pfs"
        ] = (
            index[
                "cache_eligible_residual_pfs"
            ]
            .fillna(
                False
            )
            .astype(
                bool
            )
        )

    index[
        "survival_mask"
    ] = (
        eligible
        & survival_resolved
        & (
            index[
                "scan_state"
            ]
            != "PROGRESSIVE"
        )
    )

    # Administrative 24-month cap.
    beyond = (
        index[
            "survival_mask"
        ]
        & (
            index[
                "survival_time_days"
            ]
            > 730.0
        )
    )

    index.loc[
        beyond,
        "survival_time_days",
    ] = 730.0

    index.loc[
        beyond,
        "survival_cause",
    ] = CAUSE_CENSOR

    ###########################################################################
    # Next scan.
    ###########################################################################

    (
        index,
        next_scan_audit,
    ) = attach_next_scan_targets(
        repo,
        index,
    )

    atomic_json(
        out
        / "prepared"
        / "next_scan_audit.json",
        next_scan_audit,
    )

    ###########################################################################
    # Post-progression target.
    ###########################################################################

    (
        postprog,
        postprog_audit,
    ) = load_postprog_targets(
        repo
    )

    ###########################################################################
    # Attach post-progression labels at the exact progression event/scan.
    # Never broadcast a target across every scan in a treatment line.
    ###########################################################################

    (
        index,
        postprog_merge_audit,
    ) = attach_postprog_targets(
        index,
        postprog,
    )

    postprog_audit[
        "merge"
    ] = postprog_merge_audit

    index[
        "postprog_mask"
    ] = (
        (
            index[
                "scan_state"
            ]
            == "PROGRESSIVE"
        )
        & index[
            "postprog_label"
        ].notna()
    )

    index[
        "postprog_time_mask"
    ] = (
        index[
            "postprog_mask"
        ]
        & index[
            "postprog_time_bucket"
        ].notna()
    )

    ###########################################################################
    # Frozen CKPT3 tumor representations.
    ###########################################################################

    (
        genomic_samples,
        genomic_sample_embeddings,
        genomic_audit,
    ) = chord_genomic_samples(
        repo,
        out,
        set(
            index[
                "patient_id"
            ]
        ),
    )

    (
        index,
        tumor_embeddings,
    ) = align_genomics_to_scans(
        index,
        genomic_samples,
        genomic_sample_embeddings,
    )

    ###########################################################################
    # Context features available before scan.
    ###########################################################################

    context_names = [
        "line_number_scaled",
        "elapsed_on_line_scaled",
        "scan_number_patient_log",
        "scan_number_line_log",
        "genomic_available",
        "genomic_age_scaled",
    ]

    genomic_age = (
        index[
            "genomic_age_days"
        ]
        .fillna(
            0.0
        )
        .clip(
            lower=0.0,
            upper=3650.0,
        )
    )

    context = np.column_stack(
        [
            (
                index[
                    "line"
                ]
                .clip(
                    1,
                    10,
                )
                .to_numpy(
                    dtype=np.float32
                )
                / 10.0
            ),

            (
                index[
                    "elapsed_on_line_days"
                ]
                .clip(
                    0,
                    1460,
                )
                .to_numpy(
                    dtype=np.float32
                )
                / 365.0
            ),

            np.log1p(
                index[
                    "scan_number_patient"
                ]
                .to_numpy(
                    dtype=np.float32
                )
            )
            / 5.0,

            np.log1p(
                index[
                    "scan_number_line"
                ]
                .to_numpy(
                    dtype=np.float32
                )
            )
            / 5.0,

            index[
                "genomic_available"
            ]
            .to_numpy(
                dtype=np.float32
            ),

            genomic_age.to_numpy(
                dtype=np.float32
            )
            / 365.0,
        ]
    ).astype(
        np.float32
    )

    ###########################################################################
    # Patient-balanced task weights.
    ###########################################################################

    for task, mask_col in (
        (
            "survival",
            "survival_mask",
        ),
        (
            "next_scan",
            "next_scan_mask",
        ),
        (
            "postprog",
            "postprog_mask",
        ),
    ):

        counts = (
            index.loc[
                index[
                    mask_col
                ],
                [
                    "patient_id",
                ]
            ]
            .groupby(
                "patient_id",
                observed=True,
            )
            .size()
        )

        weight = (
            index[
                "patient_id"
            ]
            .map(
                counts
            )
        )

        index[
            f"{task}_weight"
        ] = np.where(
            index[
                mask_col
            ],
            1.0
            / weight.fillna(
                1.0
            ),
            0.0,
        )

    ###########################################################################
    # Persistent leakage/split checks.
    ###########################################################################

    split_sets = {
        split:
            set(
                group[
                    "patient_id"
                ]
            )
        for split, group
        in index.groupby(
            "split"
        )
    }

    if (
        split_sets.get(
            "train",
            set(),
        )
        & split_sets.get(
            "val",
            set(),
        )
    ):

        raise RuntimeError(
            "train/val patient overlap"
        )

    if (
        split_sets.get(
            "train",
            set(),
        )
        & split_sets.get(
            "test",
            set(),
        )
    ):

        raise RuntimeError(
            "train/test patient overlap"
        )

    if (
        split_sets.get(
            "val",
            set(),
        )
        & split_sets.get(
            "test",
            set(),
        )
    ):

        raise RuntimeError(
            "val/test patient overlap"
        )

    same_day_genomic = (
        index[
            "genomic_available"
        ]
        > 0
    ) & (
        index[
            "genomic_age_days"
        ]
        <= 0
    )

    if same_day_genomic.any():

        raise RuntimeError(
            "same-day/future genomic leakage detected"
        )

    ###########################################################################
    # Save aligned arrays/index.
    ###########################################################################

    atomic_parquet(
        out
        / "prepared"
        / "scan_index.parquet",
        index,
    )

    np.save(
        out
        / "prepared"
        / "temporal_prepost_f16.npy",
        aligned_temporal,
    )

    np.save(
        out
        / "prepared"
        / "tumor_embeddings_f16.npy",
        tumor_embeddings,
    )

    np.save(
        out
        / "prepared"
        / "current_scan_features_f32.npy",
        current_scan_features,
    )

    np.save(
        out
        / "prepared"
        / "context_features_f32.npy",
        context,
    )

    atomic_json(
        out
        / "prepared"
        / "feature_schema.json",
        {
            "current_scan":
                current_scan_feature_names,

            "context":
                context_names,

            "temporal_dim":
                TEMPORAL_DIM,

            "tumor_dim":
                GENOMIC_DIM,
        },
    )

    ###########################################################################
    # Preparation QC.
    ###########################################################################

    split_counts = {}

    for split, group in index.groupby(
        "split"
    ):

        split_counts[
            split
        ] = {
            "patients":
                int(
                    group[
                        "patient_id"
                    ].nunique()
                ),

            "scans":
                int(
                    len(
                        group
                    )
                ),

            "survival_landmarks":
                int(
                    group[
                        "survival_mask"
                    ].sum()
                ),

            "next_scan_targets":
                int(
                    group[
                        "next_scan_mask"
                    ].sum()
                ),

            "postprog_targets":
                int(
                    group[
                        "postprog_mask"
                    ].sum()
                ),
        }

    prepare_report = {
        "status":
            "PREPARED",

        "frozen_upstream":
            upstream,

        "scan_rows":
            int(
                len(
                    index
                )
            ),

        "scan_patients":
            int(
                index[
                    "patient_id"
                ].nunique()
            ),

        "survival_landmarks":
            int(
                index[
                    "survival_mask"
                ].sum()
            ),

        "survival_cause_counts": {
            CAUSE_NAMES[
                int(
                    key
                )
            ]:
                int(
                    value
                )
            for key, value
            in index.loc[
                index[
                    "survival_mask"
                ],
                "survival_cause",
            ]
            .astype(int)
            .value_counts()
            .to_dict()
            .items()
        },

        "next_scan_targets":
            int(
                index[
                    "next_scan_mask"
                ].sum()
            ),

        "postprog_targets":
            int(
                index[
                    "postprog_mask"
                ].sum()
            ),

        "postprog_time_targets":
            int(
                index[
                    "postprog_time_mask"
                ].sum()
            ),

        "genomic_available_scan_rows":
            int(
                index[
                    "genomic_available"
                ].sum()
            ),

        "genomic_available_fraction":
            float(
                index[
                    "genomic_available"
                ].mean()
            ),

        "genomic_age_days_median":
            (
                float(
                    index.loc[
                        index[
                            "genomic_available"
                        ]
                        > 0,
                        "genomic_age_days",
                    ]
                    .median()
                )
                if (
                    index[
                        "genomic_available"
                    ]
                    .sum()
                    > 0
                )
                else None
            ),

        "genomic_audit":
            genomic_audit,

        "postprog_schema_audit":
            postprog_audit,

        "current_scan_feature_count":
            int(
                current_scan_features.shape[
                    1
                ]
            ),

        "split_counts":
            split_counts,

        "timing_contract": {
            "pre":
                "events available strictly before scan day",

            "post":
                (
                    "same pre-scan state plus exactly current scan; "
                    "same-day non-scan events remain excluded"
                ),

            "genomic":
                (
                    "latest resolved sample with "
                    "availability_day < landmark_day"
                ),
        },
    }

    atomic_json(
        out
        / "prepared"
        / "prepare_report.json",
        prepare_report,
    )

    print("")
    print(
        "========== CKPT5 PREP SUMMARY =========="
    )

    print(
        "scan_rows="
        f"{prepare_report['scan_rows']}"
    )

    print(
        "scan_patients="
        f"{prepare_report['scan_patients']}"
    )

    print(
        "survival_landmarks="
        f"{prepare_report['survival_landmarks']}"
    )

    print(
        "survival_cause_counts="
        f"{prepare_report['survival_cause_counts']}"
    )

    print(
        "next_scan_targets="
        f"{prepare_report['next_scan_targets']}"
    )

    print(
        "postprog_targets="
        f"{prepare_report['postprog_targets']}"
    )

    print(
        "genomic_available_scan_rows="
        f"{prepare_report['genomic_available_scan_rows']}"
    )

    print(
        "genomic_available_fraction="
        f"{prepare_report['genomic_available_fraction']}"
    )

    print(
        "current_scan_feature_count="
        f"{prepare_report['current_scan_feature_count']}"
    )

    print(
        "split_counts="
        f"{prepare_report['split_counts']}"
    )

    print(
        "========== CKPT5 PREP SUMMARY END =========="
    )


###############################################################################
# Dataset
###############################################################################


class DynamicScanDataset(
    Dataset
):

    def __init__(
        self,
        prepared: Path,
        split: str,
        *,
        paired_views: bool,
    ):

        full_index = pd.read_parquet(
            prepared
            / "scan_index.parquet"
        )

        selected = np.where(
            full_index[
                "split"
            ].to_numpy()
            == split
        )[
            0
        ]

        self.index = (
            full_index
            .iloc[
                selected
            ]
            .reset_index(
                drop=True
            )
        )

        self.base_rows = selected

        self.temporal = np.load(
            prepared
            / "temporal_prepost_f16.npy",
            mmap_mode="r",
        )

        self.tumor = np.load(
            prepared
            / "tumor_embeddings_f16.npy",
            mmap_mode="r",
        )

        self.scan = np.load(
            prepared
            / "current_scan_features_f32.npy",
            mmap_mode="r",
        )

        self.context = np.load(
            prepared
            / "context_features_f32.npy",
            mmap_mode="r",
        )

        self.paired_views = (
            paired_views
        )

    def __len__(
        self,
    ) -> int:

        multiplier = (
            2
            if self.paired_views
            else 1
        )

        return (
            len(
                self.base_rows
            )
            * multiplier
        )

    def __getitem__(
        self,
        item: int,
    ):

        if self.paired_views:

            local_row = (
                item
                // 2
            )

            view = (
                item
                % 2
            )

        else:

            local_row = item

            # Evaluation dataset is not normally used directly.
            view = 1

        global_row = int(
            self.base_rows[
                local_row
            ]
        )

        row = self.index.iloc[
            local_row
        ]

        temporal = np.asarray(
            self.temporal[
                global_row,
                view,
            ],
            dtype=np.float32,
        ).copy()

        tumor = np.asarray(
            self.tumor[
                global_row
            ],
            dtype=np.float32,
        ).copy()

        context = np.asarray(
            self.context[
                global_row
            ],
            dtype=np.float32,
        ).copy()

        if view == 0:

            scan = np.zeros(
                self.scan.shape[
                    1
                ],
                dtype=np.float32,
            )

        else:

            scan = np.asarray(
                self.scan[
                    global_row
                ],
                dtype=np.float32,
            ).copy()

        survival_mask = bool(
            row[
                "survival_mask"
            ]
        )

        next_scan_mask = bool(
            row[
                "next_scan_mask"
            ]
        )

        postprog_mask = bool(
            row[
                "postprog_mask"
            ]
        )

        postprog_time_mask = bool(
            row[
                "postprog_time_mask"
            ]
        )

        survival_cause = (
            int(
                row[
                    "survival_cause"
                ]
            )
            if survival_mask
            else 0
        )

        survival_time = (
            float(
                row[
                    "survival_time_days"
                ]
            )
            if survival_mask
            else 1.0
        )

        next_scan_label = (
            int(
                row[
                    "next_scan_label"
                ]
            )
            if next_scan_mask
            else 0
        )

        postprog_label = (
            int(
                row[
                    "postprog_label"
                ]
            )
            if postprog_mask
            else 0
        )

        postprog_time_bucket = (
            int(
                row[
                    "postprog_time_bucket"
                ]
            )
            if postprog_time_mask
            else 0
        )

        # Half weight because every base scan appears as PRE and POST.
        view_factor = (
            0.5
            if self.paired_views
            else 1.0
        )

        return {
            "temporal":
                torch.from_numpy(
                    temporal
                ),

            "tumor":
                torch.from_numpy(
                    tumor
                ),

            "scan":
                torch.from_numpy(
                    scan
                ),

            "context":
                torch.from_numpy(
                    context
                ),

            "view":
                torch.tensor(
                    view,
                    dtype=torch.long,
                ),

            "survival_mask":
                torch.tensor(
                    survival_mask,
                    dtype=torch.bool,
                ),

            "survival_time":
                torch.tensor(
                    survival_time,
                    dtype=torch.float32,
                ),

            "survival_cause":
                torch.tensor(
                    survival_cause,
                    dtype=torch.long,
                ),

            "survival_weight":
                torch.tensor(
                    float(
                        row[
                            "survival_weight"
                        ]
                    )
                    * view_factor,
                    dtype=torch.float32,
                ),

            "next_scan_mask":
                torch.tensor(
                    next_scan_mask,
                    dtype=torch.bool,
                ),

            "next_scan_label":
                torch.tensor(
                    next_scan_label,
                    dtype=torch.long,
                ),

            "next_scan_weight":
                torch.tensor(
                    float(
                        row[
                            "next_scan_weight"
                        ]
                    )
                    * view_factor,
                    dtype=torch.float32,
                ),

            "postprog_mask":
                torch.tensor(
                    postprog_mask,
                    dtype=torch.bool,
                ),

            "postprog_label":
                torch.tensor(
                    postprog_label,
                    dtype=torch.long,
                ),

            "postprog_weight":
                torch.tensor(
                    float(
                        row[
                            "postprog_weight"
                        ]
                    )
                    * view_factor,
                    dtype=torch.float32,
                ),

            "postprog_time_mask":
                torch.tensor(
                    postprog_time_mask,
                    dtype=torch.bool,
                ),

            "postprog_time_bucket":
                torch.tensor(
                    postprog_time_bucket,
                    dtype=torch.long,
                ),

            "global_row":
                torch.tensor(
                    global_row,
                    dtype=torch.long,
                ),
        }


###############################################################################
# Supervised fusion model
###############################################################################


class DynamicFusionModel(
    nn.Module
):

    def __init__(
        self,
        scan_dim: int,
        context_dim: int,
    ):

        super().__init__()

        self.temporal = nn.Sequential(
            nn.LayerNorm(
                TEMPORAL_DIM
            ),
            nn.Linear(
                TEMPORAL_DIM,
                256,
            ),
            nn.GELU(),
            nn.Dropout(
                0.15
            ),
        )

        self.tumor = nn.Sequential(
            nn.LayerNorm(
                GENOMIC_DIM
            ),
            nn.Linear(
                GENOMIC_DIM,
                96,
            ),
            nn.GELU(),
        )

        self.scan = nn.Sequential(
            nn.LayerNorm(
                scan_dim
            ),
            nn.Linear(
                scan_dim,
                64,
            ),
            nn.GELU(),
        )

        self.context = nn.Sequential(
            nn.LayerNorm(
                context_dim
            ),
            nn.Linear(
                context_dim,
                32,
            ),
            nn.GELU(),
        )

        self.aux = nn.Sequential(
            nn.Linear(
                96
                + 64
                + 32,
                256,
            ),
            nn.GELU(),
            nn.Dropout(
                0.15
            ),
        )

        self.gate = nn.Sequential(
            nn.Linear(
                512,
                256,
            ),
            nn.Sigmoid(),
        )

        self.fusion = nn.Sequential(
            nn.LayerNorm(
                256
            ),
            nn.Linear(
                256,
                512,
            ),
            nn.GELU(),
            nn.Dropout(
                0.15
            ),
            nn.Linear(
                512,
                256,
            ),
            nn.GELU(),
            nn.LayerNorm(
                256
            ),
        )

        # Monthly conditional categories:
        # [no event, progression, death, switch]
        self.survival_head = nn.Linear(
            256,
            MONTHS
            * 4,
        )

        self.next_scan_head = nn.Linear(
            256,
            3,
        )

        self.postprog_head = nn.Linear(
            256,
            3,
        )

        self.postprog_time_head = nn.Linear(
            256,
            4,
        )

    def forward(
        self,
        temporal: torch.Tensor,
        tumor: torch.Tensor,
        scan: torch.Tensor,
        context: torch.Tensor,
    ) -> dict[str, torch.Tensor]:

        temporal_state = self.temporal(
            temporal
        )

        auxiliary = self.aux(
            torch.cat(
                [
                    self.tumor(
                        tumor
                    ),
                    self.scan(
                        scan
                    ),
                    self.context(
                        context
                    ),
                ],
                dim=-1,
            )
        )

        gate = self.gate(
            torch.cat(
                [
                    temporal_state,
                    auxiliary,
                ],
                dim=-1,
            )
        )

        fused = (
            temporal_state
            + gate
            * auxiliary
        )

        hidden = self.fusion(
            fused
        )

        survival_logits = (
            self.survival_head(
                hidden
            )
            .view(
                -1,
                MONTHS,
                4,
            )
        )

        return {
            "hidden":
                hidden,

            "survival_logits":
                survival_logits,

            "next_scan_logits":
                self.next_scan_head(
                    hidden
                ),

            "postprog_logits":
                self.postprog_head(
                    hidden
                ),

            "postprog_time_logits":
                self.postprog_time_head(
                    hidden
                ),
        }


###############################################################################
# Losses
###############################################################################


def competing_risk_nll_per_row(
    logits: torch.Tensor,
    time_days: torch.Tensor,
    cause: torch.Tensor,
) -> torch.Tensor:

    log_probability = F.log_softmax(
        logits.float(),
        dim=-1,
    )

    last_month = torch.ceil(
        time_days.float()
        / MONTH_DAYS
    ).long()

    last_month = torch.clamp(
        last_month,
        1,
        MONTHS,
    )

    losses = []

    for row in range(
        logits.shape[
            0
        ]
    ):

        month = int(
            last_month[
                row
            ]
        )

        cause_value = int(
            cause[
                row
            ]
        )

        if cause_value == CAUSE_CENSOR:

            loss = -(
                log_probability[
                    row,
                    :month,
                    CAUSE_CENSOR,
                ]
                .sum()
            )

        else:

            survival_part = (
                log_probability[
                    row,
                    :month - 1,
                    CAUSE_CENSOR,
                ]
                .sum()
            )

            event_part = (
                log_probability[
                    row,
                    month - 1,
                    cause_value,
                ]
            )

            loss = -(
                survival_part
                + event_part
            )

        losses.append(
            loss
        )

    return torch.stack(
        losses
    )


def weighted_masked_mean(
    loss: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:

    if not bool(
        mask.any()
    ):

        return loss.sum() * 0.0

    selected_loss = loss[
        mask
    ]

    selected_weight = weight[
        mask
    ]

    return (
        (
            selected_loss
            * selected_weight
        ).sum()
        / selected_weight.sum().clamp_min(
            1e-8
        )
    )


def multitask_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> tuple[
    torch.Tensor,
    dict[str, float],
]:

    survival_row_loss = (
        competing_risk_nll_per_row(
            output[
                "survival_logits"
            ],
            batch[
                "survival_time"
            ],
            batch[
                "survival_cause"
            ],
        )
    )

    survival = weighted_masked_mean(
        survival_row_loss,
        batch[
            "survival_mask"
        ],
        batch[
            "survival_weight"
        ],
    )

    next_row_loss = F.cross_entropy(
        output[
            "next_scan_logits"
        ].float(),
        batch[
            "next_scan_label"
        ],
        reduction="none",
    )

    next_scan = weighted_masked_mean(
        next_row_loss,
        batch[
            "next_scan_mask"
        ],
        batch[
            "next_scan_weight"
        ],
    )

    post_row_loss = F.cross_entropy(
        output[
            "postprog_logits"
        ].float(),
        batch[
            "postprog_label"
        ],
        reduction="none",
    )

    postprog = weighted_masked_mean(
        post_row_loss,
        batch[
            "postprog_mask"
        ],
        batch[
            "postprog_weight"
        ],
    )

    post_time_row_loss = F.cross_entropy(
        output[
            "postprog_time_logits"
        ].float(),
        batch[
            "postprog_time_bucket"
        ],
        reduction="none",
    )

    postprog_time = weighted_masked_mean(
        post_time_row_loss,
        batch[
            "postprog_time_mask"
        ],
        batch[
            "postprog_weight"
        ],
    )

    total = (
        survival
        + 0.25
        * next_scan
        + 0.15
        * postprog
        + 0.05
        * postprog_time
    )

    metrics = {
        "total":
            float(
                total.detach()
            ),

        "survival":
            float(
                survival.detach()
            ),

        "next_scan":
            float(
                next_scan.detach()
            ),

        "postprog":
            float(
                postprog.detach()
            ),

        "postprog_time":
            float(
                postprog_time.detach()
            ),
    }

    return (
        total,
        metrics,
    )


###############################################################################
# Training helpers
###############################################################################


def move_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:

    return {
        key:
            value.to(
                device,
                non_blocking=True,
            )
        for key, value
        in batch.items()
    }


@torch.no_grad()
def validation_post_survival_nll(
    model: DynamicFusionModel,
    dataset: DynamicScanDataset,
    device: torch.device,
) -> float:

    ###########################################################################
    # Patient-balanced validation criterion.
    #
    # Repeated scan landmarks are correlated within patient. Model selection
    # therefore averages landmark NLL within patient first, then gives each
    # validation patient equal weight.
    ###########################################################################

    model.eval()

    base_index = dataset.index

    selected = base_index[
        base_index[
            "survival_mask"
        ]
        .fillna(
            False
        )
        .astype(
            bool
        )
    ].copy()

    if selected.empty:

        raise RuntimeError(
            "Validation contains no survival landmarks."
        )

    rows = selected.index.to_numpy(
        dtype=int
    )

    global_rows = dataset.base_rows[
        rows
    ]

    batch_size = 1024

    row_losses = []

    for start_row in range(
        0,
        len(
            rows
        ),
        batch_size,
    ):

        local = np.arange(
            start_row,
            min(
                start_row
                + batch_size,
                len(
                    rows
                ),
            ),
        )

        batch_global = global_rows[
            local
        ]

        temporal = torch.from_numpy(
            np.asarray(
                dataset.temporal[
                    batch_global,
                    1,
                ],
                dtype=np.float32,
            ).copy()
        ).to(
            device
        )

        tumor = torch.from_numpy(
            np.asarray(
                dataset.tumor[
                    batch_global
                ],
                dtype=np.float32,
            ).copy()
        ).to(
            device
        )

        scan = torch.from_numpy(
            np.asarray(
                dataset.scan[
                    batch_global
                ],
                dtype=np.float32,
            ).copy()
        ).to(
            device
        )

        context = torch.from_numpy(
            np.asarray(
                dataset.context[
                    batch_global
                ],
                dtype=np.float32,
            ).copy()
        ).to(
            device
        )

        output = model(
            temporal,
            tumor,
            scan,
            context,
        )

        frame = selected.iloc[
            local
        ]

        loss = (
            competing_risk_nll_per_row(
                output[
                    "survival_logits"
                ],
                torch.tensor(
                    frame[
                        "survival_time_days"
                    ]
                    .to_numpy(
                        dtype=np.float32
                    ),
                    device=device,
                ),
                torch.tensor(
                    frame[
                        "survival_cause"
                    ]
                    .astype(
                        int
                    )
                    .to_numpy(),
                    device=device,
                ),
            )
            .float()
            .cpu()
            .numpy()
        )

        row_losses.append(
            loss
        )

    selected[
        "_validation_nll"
    ] = np.concatenate(
        row_losses
    )

    patient_nll = (
        selected.groupby(
            "patient_id",
            observed=True,
        )[
            "_validation_nll"
        ]
        .mean()
    )

    if len(
        patient_nll
    ) == 0:

        raise RuntimeError(
            "Validation patient-balanced NLL has zero patients."
        )

    value = float(
        patient_nll.mean()
    )

    if not np.isfinite(
        value
    ):

        raise RuntimeError(
            "Non-finite patient-balanced validation NLL."
        )

    return value


def train_model(
    repo: Path,
    out: Path,
    epochs: int,
    batch_size: int,
) -> None:

    if not torch.cuda.is_available():

        raise RuntimeError(
            "Full CKPT5 training requires CUDA."
        )

    random.seed(
        SEED
    )

    np.random.seed(
        SEED
    )

    torch.manual_seed(
        SEED
    )

    torch.cuda.manual_seed_all(
        SEED
    )

    device = torch.device(
        "cuda:0"
    )

    prepared = (
        out
        / "prepared"
    )

    schema = json.loads(
        (
            prepared
            / "feature_schema.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    train_dataset = DynamicScanDataset(
        prepared,
        "train",
        paired_views=True,
    )

    val_dataset = DynamicScanDataset(
        prepared,
        "val",
        paired_views=False,
    )

    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    model = DynamicFusionModel(
        scan_dim=len(
            schema[
                "current_scan"
            ]
        ),
        context_dim=len(
            schema[
                "context"
            ]
        ),
    ).to(
        device
    )

    parameter_count = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.02,
        betas=(
            0.9,
            0.95,
        ),
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(
                epochs
                * len(
                    loader
                ),
                1,
            ),
            eta_min=3e-5,
        )
    )

    bf16_supported = bool(
        torch.cuda.is_bf16_supported()
    )

    amp_dtype = (
        torch.bfloat16
        if bf16_supported
        else torch.float16
    )

    scaler_enabled = (
        not bf16_supported
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=scaler_enabled,
    )

    print(
        "[CKPT5_TRAIN_PRECISION]",
        {
            "gpu":
                torch.cuda.get_device_name(
                    device
                ),

            "bf16_supported":
                bf16_supported,

            "autocast_dtype":
                (
                    "bfloat16"
                    if bf16_supported
                    else "float16"
                ),

            "grad_scaler_enabled":
                scaler_enabled,
        },
        flush=True,
    )

    best_nll = float(
        "inf"
    )

    best_state = None

    patience = 4
    stale_epochs = 0

    history = []

    for epoch in range(
        1,
        epochs + 1,
    ):

        model.train()

        totals = Counter()

        for batch in loader:

            batch = move_batch(
                batch,
                device,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.amp.autocast(
                "cuda",
                dtype=amp_dtype,
                enabled=True,
            ):

                output = model(
                    batch[
                        "temporal"
                    ],
                    batch[
                        "tumor"
                    ],
                    batch[
                        "scan"
                    ],
                    batch[
                        "context"
                    ],
                )

                (
                    loss,
                    components,
                ) = multitask_loss(
                    output,
                    batch,
                )

            if not torch.isfinite(
                loss
            ):

                raise RuntimeError(
                    "Non-finite training loss."
                )

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            scheduler.step()

            totals[
                "batches"
            ] += 1

            for key, value in components.items():

                totals[
                    key
                ] += value

        val_nll = (
            validation_post_survival_nll(
                model,
                val_dataset,
                device,
            )
        )

        denominator = max(
            totals[
                "batches"
            ],
            1,
        )

        row = {
            "epoch":
                epoch,

            "train_total":
                totals[
                    "total"
                ]
                / denominator,

            "train_survival":
                totals[
                    "survival"
                ]
                / denominator,

            "train_next_scan":
                totals[
                    "next_scan"
                ]
                / denominator,

            "train_postprog":
                totals[
                    "postprog"
                ]
                / denominator,

            "train_postprog_time":
                totals[
                    "postprog_time"
                ]
                / denominator,

            "validation_post_survival_nll":
                val_nll,

            "lr":
                float(
                    optimizer.param_groups[
                        0
                    ][
                        "lr"
                    ]
                ),
        }

        history.append(
            row
        )

        print(
            "[CKPT5_TRAIN]",
            row,
            flush=True,
        )

        if (
            val_nll
            < best_nll
            - 1e-4
        ):

            best_nll = val_nll

            best_state = copy.deepcopy(
                model.state_dict()
            )

            stale_epochs = 0

        else:

            stale_epochs += 1

            if (
                epoch >= 8
                and stale_epochs
                >= patience
            ):

                print(
                    "[CKPT5_EARLY_STOP]",
                    {
                        "epoch":
                            epoch,

                        "best_validation_post_survival_nll":
                            best_nll,
                    },
                    flush=True,
                )

                break

    if best_state is None:
        raise RuntimeError(
            "No best model checkpoint was captured."
        )

    checkpoint = {
        "model_state":
            best_state,

        "scan_dim":
            len(
                schema[
                    "current_scan"
                ]
            ),

        "context_dim":
            len(
                schema[
                    "context"
                ]
            ),

        "temporal_dim":
            TEMPORAL_DIM,

        "genomic_dim":
            GENOMIC_DIM,

        "months":
            MONTHS,

        "parameter_count":
            parameter_count,

        "best_validation_post_survival_nll":
            best_nll,

        "seed":
            SEED,

        "loss_weights": {
            "survival":
                1.0,

            "next_scan":
                0.25,

            "post_progression":
                0.15,

            "post_progression_time":
                0.05,
        },

        "pre_post_contract": {
            "pre":
                (
                    "CKPT4 PRE state; current-scan feature vector zeroed"
                ),

            "post":
                (
                    "CKPT4 POST state; current-scan feature vector supplied"
                ),
        },
    }

    tmp_path = (
        out
        / "dynamic_scan_model.pt.tmp"
    )

    torch.save(
        checkpoint,
        tmp_path,
    )

    validation = torch.load(
        tmp_path,
        map_location="cpu",
        weights_only=False,
    )

    if (
        "model_state"
        not in validation
    ):

        raise RuntimeError(
            "Dynamic model checkpoint verification failed."
        )

    tmp_path.replace(
        out
        / "dynamic_scan_model.pt"
    )

    atomic_json(
        out
        / "training_history.json",
        history,
    )

    print(
        "[CKPT5_TRAIN_COMPLETE]",
        {
            "parameters":
                parameter_count,

            "best_validation_post_survival_nll":
                best_nll,

            "epochs_completed":
                len(
                    history
                ),
        },
    )


###############################################################################
# Evaluation
###############################################################################


def probability_curves(
    logits: torch.Tensor,
) -> dict[str, torch.Tensor]:

    conditional = torch.softmax(
        logits.float(),
        dim=-1,
    )

    batch = conditional.shape[
        0
    ]

    event_free = torch.ones(
        batch,
        device=conditional.device,
    )

    cif_progression = torch.zeros_like(
        event_free
    )

    cif_death = torch.zeros_like(
        event_free
    )

    cif_switch = torch.zeros_like(
        event_free
    )

    pfs = []

    switch = []

    all_event_free = []

    for month in range(
        MONTHS
    ):

        current = conditional[
            :,
            month,
        ]

        cif_progression = (
            cif_progression
            + event_free
            * current[
                :,
                CAUSE_PROGRESSION
            ]
        )

        cif_death = (
            cif_death
            + event_free
            * current[
                :,
                CAUSE_DEATH
            ]
        )

        cif_switch = (
            cif_switch
            + event_free
            * current[
                :,
                CAUSE_SWITCH
            ]
        )

        event_free = (
            event_free
            * current[
                :,
                CAUSE_CENSOR
            ]
        )

        pfs.append(
            1.0
            - cif_progression
            - cif_death
        )

        switch.append(
            cif_switch
        )

        all_event_free.append(
            event_free
        )

    return {
        "pfs":
            torch.stack(
                pfs,
                dim=1,
            ),

        "switch_cif":
            torch.stack(
                switch,
                dim=1,
            ),

        "event_free":
            torch.stack(
                all_event_free,
                dim=1,
            ),

        "conditional":
            conditional,
    }


def macro_auprc(
    logits: np.ndarray,
    labels: np.ndarray,
    classes: int,
) -> float:

    probability = torch.softmax(
        torch.from_numpy(
            logits
        ),
        dim=-1,
    ).numpy()

    values = []

    for class_index in range(
        classes
    ):

        target = (
            labels
            == class_index
        ).astype(
            int
        )

        if (
            target.sum()
            < 2
            or (
                1
                - target
            ).sum()
            < 2
        ):

            continue

        values.append(
            average_precision_score(
                target,
                probability[
                    :,
                    class_index,
                ],
            )
        )

    return (
        float(
            np.mean(
                values
            )
        )
        if values
        else float(
            "nan"
        )
    )


def patient_bootstrap_gain(
    frame: pd.DataFrame,
    repetitions: int = 1000,
) -> dict[str, float | int]:

    patient = (
        frame.groupby(
            "patient_id",
            observed=True,
        )[
            [
                "pre_nll",
                "post_nll",
            ]
        ]
        .mean()
        .dropna()
    )

    delta = (
        patient[
            "pre_nll"
        ]
        - patient[
            "post_nll"
        ]
    ).to_numpy(
        dtype=float
    )

    if len(
        delta
    ) == 0:

        return {
            "mean":
                float(
                    "nan"
                ),

            "ci_low":
                float(
                    "nan"
                ),

            "ci_high":
                float(
                    "nan"
                ),

            "patients":
                0,
        }

    rng = np.random.default_rng(
        SEED
    )

    samples = np.empty(
        repetitions,
        dtype=float,
    )

    for index in range(
        repetitions
    ):

        samples[
            index
        ] = (
            rng.choice(
                delta,
                size=len(
                    delta
                ),
                replace=True,
            )
            .mean()
        )

    return {
        "mean":
            float(
                delta.mean()
            ),

        "ci_low":
            float(
                np.quantile(
                    samples,
                    0.025,
                )
            ),

        "ci_high":
            float(
                np.quantile(
                    samples,
                    0.975,
                )
            ),

        "patients":
            int(
                len(
                    delta
                )
            ),
    }


@torch.no_grad()
def evaluate_view(
    model: DynamicFusionModel,
    prepared: Path,
    split: str,
    view: int,
    device: torch.device,
) -> dict[str, Any]:

    index = pd.read_parquet(
        prepared
        / "scan_index.parquet"
    )

    selected_index = index[
        index[
            "split"
        ]
        == split
    ].copy()

    global_rows = selected_index.index.to_numpy(
        dtype=int
    )

    temporal_array = np.load(
        prepared
        / "temporal_prepost_f16.npy",
        mmap_mode="r",
    )

    tumor_array = np.load(
        prepared
        / "tumor_embeddings_f16.npy",
        mmap_mode="r",
    )

    scan_array = np.load(
        prepared
        / "current_scan_features_f32.npy",
        mmap_mode="r",
    )

    context_array = np.load(
        prepared
        / "context_features_f32.npy",
        mmap_mode="r",
    )

    survival_rows = selected_index[
        "survival_mask"
    ].to_numpy()

    next_rows = selected_index[
        "next_scan_mask"
    ].to_numpy()

    post_rows = selected_index[
        "postprog_mask"
    ].to_numpy()

    batch_size = 1024

    survival_nll = np.full(
        len(
            selected_index
        ),
        np.nan,
        dtype=float,
    )

    next_logits = []
    next_labels = []

    post_logits = []
    post_labels = []

    pfs_horizons = {
        3:
            np.full(
                len(
                    selected_index
                ),
                np.nan,
            ),

        6:
            np.full(
                len(
                    selected_index
                ),
                np.nan,
            ),

        12:
            np.full(
                len(
                    selected_index
                ),
                np.nan,
            ),

        18:
            np.full(
                len(
                    selected_index
                ),
                np.nan,
            ),
    }

    switch_12m = np.full(
        len(
            selected_index
        ),
        np.nan,
    )

    hidden_blocks = []

    for start in range(
        0,
        len(
            selected_index
        ),
        batch_size,
    ):

        local = np.arange(
            start,
            min(
                start
                + batch_size,
                len(
                    selected_index
                ),
            ),
        )

        rows = global_rows[
            local
        ]

        temporal = torch.from_numpy(
            np.asarray(
                temporal_array[
                    rows,
                    view,
                ],
                dtype=np.float32,
            ).copy()
        ).to(
            device
        )

        tumor = torch.from_numpy(
            np.asarray(
                tumor_array[
                    rows
                ],
                dtype=np.float32,
            ).copy()
        ).to(
            device
        )

        context = torch.from_numpy(
            np.asarray(
                context_array[
                    rows
                ],
                dtype=np.float32,
            ).copy()
        ).to(
            device
        )

        if view == 0:

            scan = torch.zeros(
                (
                    len(
                        local
                    ),
                    scan_array.shape[
                        1
                    ],
                ),
                dtype=torch.float32,
                device=device,
            )

        else:

            scan = torch.from_numpy(
                np.asarray(
                    scan_array[
                        rows
                    ],
                    dtype=np.float32,
                ).copy()
            ).to(
                device
            )

        output = model(
            temporal,
            tumor,
            scan,
            context,
        )

        curves = probability_curves(
            output[
                "survival_logits"
            ]
        )

        hidden_blocks.append(
            output[
                "hidden"
            ]
            .float()
            .cpu()
            .numpy()
        )

        for horizon in (
            3,
            6,
            12,
            18,
        ):

            pfs_horizons[
                horizon
            ][
                local
            ] = (
                curves[
                    "pfs"
                ][
                    :,
                    horizon - 1,
                ]
                .cpu()
                .numpy()
            )

        switch_12m[
            local
        ] = (
            curves[
                "switch_cif"
            ][
                :,
                11,
            ]
            .cpu()
            .numpy()
        )

        current_frame = selected_index.iloc[
            local
        ]

        mask = current_frame[
            "survival_mask"
        ].to_numpy()

        if mask.any():

            positions = np.where(
                mask
            )[
                0
            ]

            loss = (
                competing_risk_nll_per_row(
                    output[
                        "survival_logits"
                    ][
                        positions
                    ],
                    torch.tensor(
                        current_frame.iloc[
                            positions
                        ][
                            "survival_time_days"
                        ]
                        .to_numpy(
                            dtype=np.float32
                        ),
                        device=device,
                    ),
                    torch.tensor(
                        current_frame.iloc[
                            positions
                        ][
                            "survival_cause"
                        ]
                        .astype(
                            int
                        )
                        .to_numpy(),
                        device=device,
                    ),
                )
                .cpu()
                .numpy()
            )

            survival_nll[
                local[
                    positions
                ]
            ] = loss

        mask = current_frame[
            "next_scan_mask"
        ].to_numpy()

        if mask.any():

            positions = np.where(
                mask
            )[
                0
            ]

            next_logits.append(
                output[
                    "next_scan_logits"
                ][
                    positions
                ]
                .float()
                .cpu()
                .numpy()
            )

            next_labels.append(
                current_frame.iloc[
                    positions
                ][
                    "next_scan_label"
                ]
                .astype(
                    int
                )
                .to_numpy()
            )

        mask = current_frame[
            "postprog_mask"
        ].to_numpy()

        if mask.any():

            positions = np.where(
                mask
            )[
                0
            ]

            post_logits.append(
                output[
                    "postprog_logits"
                ][
                    positions
                ]
                .float()
                .cpu()
                .numpy()
            )

            post_labels.append(
                current_frame.iloc[
                    positions
                ][
                    "postprog_label"
                ]
                .astype(
                    int
                )
                .to_numpy()
            )

    hidden = np.concatenate(
        hidden_blocks,
        axis=0,
    )

    centered = (
        hidden
        - hidden.mean(
            axis=0,
            keepdims=True,
        )
    )

    singular = np.linalg.svd(
        centered[
            :min(
                len(
                    centered
                ),
                10000,
            )
        ],
        full_matrices=False,
        compute_uv=False,
    )

    variance = singular ** 2

    probability = (
        variance
        / variance.sum()
    )

    effective_rank = float(
        np.exp(
            -np.sum(
                probability
                * np.log(
                    probability
                    + 1e-12
                )
            )
        )
    )

    metrics = {
        "view":
            (
                "PRE"
                if view == 0
                else "POST"
            ),

        "split":
            split,

        "survival_rows":
            int(
                np.isfinite(
                    survival_nll
                ).sum()
            ),

        "mean_competing_risk_nll":
            float(
                np.nanmean(
                    survival_nll
                )
            ),

        "hidden_effective_rank":
            effective_rank,

        "hidden_noncollapsed_dimensions":
            int(
                (
                    hidden.std(
                        axis=0
                    )
                    > 1e-5
                ).sum()
            ),
    }

    if next_logits:

        next_logits_array = np.concatenate(
            next_logits,
            axis=0,
        )

        next_labels_array = np.concatenate(
            next_labels,
            axis=0,
        )

        metrics[
            "next_scan_macro_auprc"
        ] = macro_auprc(
            next_logits_array,
            next_labels_array,
            3,
        )

        metrics[
            "next_scan_accuracy"
        ] = float(
            (
                next_logits_array.argmax(
                    axis=1
                )
                == next_labels_array
            ).mean()
        )

        metrics[
            "next_scan_rows"
        ] = int(
            len(
                next_labels_array
            )
        )

    if post_logits:

        post_logits_array = np.concatenate(
            post_logits,
            axis=0,
        )

        post_labels_array = np.concatenate(
            post_labels,
            axis=0,
        )

        metrics[
            "postprog_macro_auprc"
        ] = macro_auprc(
            post_logits_array,
            post_labels_array,
            3,
        )

        metrics[
            "postprog_accuracy"
        ] = float(
            (
                post_logits_array.argmax(
                    axis=1
                )
                == post_labels_array
            ).mean()
        )

        metrics[
            "postprog_rows"
        ] = int(
            len(
                post_labels_array
            )
        )

    predictions = selected_index[
        [
            "patient_id",
            "scan_episode_id",
            "landmark_day",
            "line",
            "scan_state",
            "split",
            "survival_mask",
            "survival_time_days",
            "survival_cause",
            "genomic_available",
            "genomic_age_days",
        ]
    ].copy()

    predictions[
        (
            "pre_nll"
            if view == 0
            else "post_nll"
        )
    ] = survival_nll

    for horizon in (
        3,
        6,
        12,
        18,
    ):

        predictions[
            (
                f"pre_pfs_{horizon}m"
                if view == 0
                else f"post_pfs_{horizon}m"
            )
        ] = pfs_horizons[
            horizon
        ]

    predictions[
        (
            "pre_switch_cif_12m"
            if view == 0
            else "post_switch_cif_12m"
        )
    ] = switch_12m

    return {
        "metrics":
            metrics,

        "predictions":
            predictions,

        "hidden":
            hidden,
    }


def evaluate(
    repo: Path,
    out: Path,
) -> None:

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint = torch.load(
        out
        / "dynamic_scan_model.pt",
        map_location=device,
        weights_only=False,
    )

    model = DynamicFusionModel(
        scan_dim=checkpoint[
            "scan_dim"
        ],
        context_dim=checkpoint[
            "context_dim"
        ],
    ).to(
        device
    )

    model.load_state_dict(
        checkpoint[
            "model_state"
        ]
    )

    model.eval()

    prepared = (
        out
        / "prepared"
    )

    pre = evaluate_view(
        model,
        prepared,
        "test",
        0,
        device,
    )

    post = evaluate_view(
        model,
        prepared,
        "test",
        1,
        device,
    )

    predictions = pre[
        "predictions"
    ].merge(
        post[
            "predictions"
        ][
            [
                column
                for column
                in post[
                    "predictions"
                ].columns
                if (
                    column
                    in {
                        "patient_id",
                        "scan_episode_id",
                    }
                    or column.startswith(
                        "post_"
                    )
                )
            ]
        ],
        on=[
            "patient_id",
            "scan_episode_id",
        ],
        how="inner",
        validate="one_to_one",
    )

    paired = predictions[
        predictions[
            "survival_mask"
        ]
        & predictions[
            "pre_nll"
        ].notna()
        & predictions[
            "post_nll"
        ].notna()
    ].copy()

    paired[
        "pre_minus_post_nll"
    ] = (
        paired[
            "pre_nll"
        ]
        - paired[
            "post_nll"
        ]
    )

    bootstrap = patient_bootstrap_gain(
        paired
    )

    atomic_parquet(
        out
        / "test_prepost_predictions.parquet",
        predictions,
    )

    atomic_parquet(
        out
        / "paired_survival_nll.parquet",
        paired,
    )

    metrics = {
        "pre":
            pre[
                "metrics"
            ],

        "post":
            post[
                "metrics"
            ],

        "paired_scan_information_gain":
            bootstrap,

        "interpretation":
            (
                "Positive PRE minus POST competing-risk NLL means "
                "adding exactly the current scan improves the held-out "
                "conditional outcome likelihood."
            ),
    }

    atomic_json(
        out
        / "test_metrics.json",
        metrics,
    )

    print("")
    print(
        "========== CKPT5 EVALUATION =========="
    )

    print(
        "pre_metrics="
        f"{metrics['pre']}"
    )

    print(
        "post_metrics="
        f"{metrics['post']}"
    )

    print(
        "paired_scan_information_gain="
        f"{bootstrap}"
    )

    print(
        "========== CKPT5 EVALUATION END =========="
    )


###############################################################################
# Self-tests
###############################################################################


def self_test() -> None:

    torch.manual_seed(
        10
    )

    model = DynamicFusionModel(
        scan_dim=17,
        context_dim=6,
    )

    batch = 8

    output = model(
        torch.randn(
            batch,
            TEMPORAL_DIM,
        ),
        torch.randn(
            batch,
            GENOMIC_DIM,
        ),
        torch.randn(
            batch,
            17,
        ),
        torch.randn(
            batch,
            6,
        ),
    )

    assert output[
        "survival_logits"
    ].shape == (
        batch,
        MONTHS,
        4,
    )

    assert output[
        "next_scan_logits"
    ].shape == (
        batch,
        3,
    )

    assert output[
        "postprog_logits"
    ].shape == (
        batch,
        3,
    )

    ###########################################################################
    # Censor likelihood versus event likelihood smoke test.
    ###########################################################################

    times = torch.tensor(
        [
            40.0,
            80.0,
        ]
    )

    causes = torch.tensor(
        [
            CAUSE_PROGRESSION,
            CAUSE_CENSOR,
        ]
    )

    loss = competing_risk_nll_per_row(
        output[
            "survival_logits"
        ][
            :2
        ],
        times,
        causes,
    )

    assert loss.shape == (
        2,
    )

    assert torch.isfinite(
        loss
    ).all()

    curves = probability_curves(
        output[
            "survival_logits"
        ]
    )

    assert curves[
        "pfs"
    ].shape == (
        batch,
        MONTHS,
    )

    # PFS curve must be non-increasing.
    assert bool(
        (
            curves[
                "pfs"
            ][
                :,
                1:
            ]
            <= curves[
                "pfs"
            ][
                :,
                :-1
            ]
            + 1e-6
        )
        .all()
    )

    print(
        "[CKPT5_SELF_TEST_PASS]"
    )


###############################################################################
# Finalize
###############################################################################


def finalize(
    repo: Path,
    out: Path,
) -> None:

    prepare_report = json.loads(
        (
            out
            / "prepared"
            / "prepare_report.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    metrics = json.loads(
        (
            out
            / "test_metrics.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    history = json.loads(
        (
            out
            / "training_history.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    checkpoint = torch.load(
        out
        / "dynamic_scan_model.pt",
        map_location="cpu",
        weights_only=False,
    )

    gain = metrics[
        "paired_scan_information_gain"
    ]

    finite = all(
        np.isfinite(
            value
        )
        for value in (
            metrics[
                "pre"
            ][
                "mean_competing_risk_nll"
            ],
            metrics[
                "post"
            ][
                "mean_competing_risk_nll"
            ],
            gain[
                "mean"
            ],
        )
    )

    if not finite:

        status = (
            "NEEDS_SUPERVISED_MODEL_RESOLUTION"
        )

    elif (
        gain[
            "ci_low"
        ]
        > 0
    ):

        status = (
            "PASS_SUPERVISED_DYNAMIC_MODEL_SCAN_GAIN"
        )

    else:

        status = (
            "PASS_SUPERVISED_DYNAMIC_MODEL"
        )

    qc = {
        "status":
            status,

        "scan_rows":
            prepare_report[
                "scan_rows"
            ],

        "scan_patients":
            prepare_report[
                "scan_patients"
            ],

        "survival_landmarks":
            prepare_report[
                "survival_landmarks"
            ],

        "next_scan_targets":
            prepare_report[
                "next_scan_targets"
            ],

        "postprog_targets":
            prepare_report[
                "postprog_targets"
            ],

        "genomic_available_fraction":
            prepare_report[
                "genomic_available_fraction"
            ],

        "model_parameters":
            int(
                checkpoint[
                    "parameter_count"
                ]
            ),

        "pre_test_nll":
            metrics[
                "pre"
            ][
                "mean_competing_risk_nll"
            ],

        "post_test_nll":
            metrics[
                "post"
            ][
                "mean_competing_risk_nll"
            ],

        "paired_scan_gain":
            gain,

        "pre_hidden_effective_rank":
            metrics[
                "pre"
            ][
                "hidden_effective_rank"
            ],

        "post_hidden_effective_rank":
            metrics[
                "post"
            ][
                "hidden_effective_rank"
            ],

        "timing_contract":
            prepare_report[
                "timing_contract"
            ],

        "errors":
            [],
    }

    atomic_json(
        out
        / "qc.json",
        qc,
    )

    audit = f"""# Checkpoint 5 — Supervised scan-updated dynamic model

Status: **{status}**

## Frozen upstream interfaces

- CKPT3 tumor state: 128-D coverage-aware genomic embedding
- CKPT4 PRE state: 192-D causal history before the current scan
- CKPT4 POST state: 192-D history after adding exactly the current scan
- Authoritative W3 scan episodes: {prepare_report['scan_rows']}

No CKPT1/2/3/4 construction was changed.

## Supervised model

Inputs:
- 192-D frozen temporal state
- 128-D latest availability-safe tumor embedding
- explicit current-scan state / imaging coverage / observed-site features
- line, elapsed-time, scan-count, genomic-recency context

PRE view:
- CKPT4 PRE temporal state
- current-scan feature vector is zero
- genomics uses only results with availability_day < landmark_day

POST view:
- CKPT4 POST temporal state
- current scan features supplied
- no other same-day event is added

The same model parameters are used for PRE and POST.

## Heads

1. 24-month competing-risk head
   - no event
   - progression
   - death
   - treatment switch

2. next-scan state
   - non-progressive
   - indeterminate
   - progressive

3. post-progression transition
   - next treatment
   - continued therapy
   - death before next treatment

4. auxiliary next-treatment time bucket

Loss weights:
- survival: 1.00
- next scan: 0.25
- post-progression transition: 0.15
- post-progression time: 0.05

## Training population

Survival landmarks: {prepare_report['survival_landmarks']}
Next-scan targets: {prepare_report['next_scan_targets']}
Post-progression targets: {prepare_report['postprog_targets']}

Genomic availability at scans:
- rows: {prepare_report['genomic_available_scan_rows']}
- fraction: {prepare_report['genomic_available_fraction']}
- median genomic age: {prepare_report['genomic_age_days_median']}

## Held-out PRE → POST result

PRE competing-risk NLL:
{metrics['pre']['mean_competing_risk_nll']}

POST competing-risk NLL:
{metrics['post']['mean_competing_risk_nll']}

Patient-bootstrap PRE minus POST NLL:
{gain}

Positive values mean adding exactly the current scan improves held-out likelihood.

## Representation QC

PRE fused-state effective rank:
{metrics['pre']['hidden_effective_rank']}

POST fused-state effective rank:
{metrics['post']['hidden_effective_rank']}

## Artifacts

- artifacts/checkpoint5/dynamic_scan_model.pt
- artifacts/checkpoint5/training_history.json
- artifacts/checkpoint5/test_metrics.json
- artifacts/checkpoint5/test_prepost_predictions.parquet
- artifacts/checkpoint5/paired_survival_nll.parquet
- artifacts/checkpoint5/prepared/scan_index.parquet
- artifacts/checkpoint5/prepared/temporal_prepost_f16.npy
- artifacts/checkpoint5/prepared/tumor_embeddings_f16.npy
- artifacts/checkpoint5/prepared/current_scan_features_f32.npy
- artifacts/checkpoint5/qc.json
- artifacts/handoff/checkpoint_05.json
"""

    atomic_text(
        out
        / "audit.md",
        audit,
    )

    ###########################################################################
    # PROJECT_STATE
    ###########################################################################

    state_path = (
        repo
        / "PROJECT_STATE.md"
    )

    existing = (
        state_path.read_text(
            encoding="utf-8"
        )
        if state_path.exists()
        else "# PROJECT STATE\n"
    )

    marker = (
        "<!-- CKPT5_SUPERVISED_DYNAMIC_MODEL -->"
    )

    if marker in existing:

        existing = (
            existing
            .split(
                marker
            )[
                0
            ]
            .rstrip()
            + "\n"
        )

    section = f"""
{marker}
## Checkpoint 5 — Supervised scan-updated dynamic model

Status: **{status}**

Frozen supervised interface:
- CKPT3 128-D tumor encoder remains frozen.
- CKPT4 192-D temporal encoder remains frozen.
- PRE uses all information available strictly before scan day.
- POST differs from PRE only by addition of the current scan.
- Same-day non-scan events remain excluded.
- Tumor embedding at a scan uses the latest genomic result with availability_day < scan day.
- Patient history remains persistent across treatment lines.
- Progressive scans train auxiliary/post-progression state but are not residual-PFS prediction landmarks.

Heads:
- 24-month progression/death/switch competing-risk survival
- next-scan state
- post-progression transition
- post-progression next-treatment-time bucket

Training:
- patient-disjoint CKPT1 split retained
- repeated landmarks patient-balanced
- paired PRE/POST views use the same model weights

Held-out paired PRE minus POST NLL gain:
- mean: {gain['mean']}
- 95% patient-bootstrap CI: [{gain['ci_low']}, {gain['ci_high']}]

Artifacts:
- artifacts/checkpoint5/dynamic_scan_model.pt
- artifacts/checkpoint5/test_metrics.json
- artifacts/checkpoint5/test_prepost_predictions.parquet
- artifacts/checkpoint5/paired_survival_nll.parquet
- artifacts/checkpoint5/audit.md
- artifacts/handoff/checkpoint_05.json
"""

    atomic_text(
        state_path,
        existing.rstrip()
        + "\n\n"
        + section.strip()
        + "\n",
    )

    ###########################################################################
    # Handoff
    ###########################################################################

    handoff = {
        "checkpoint":
            "5",

        "name":
            "supervised_scan_updated_dynamic_model",

        "status":
            status,

        "frozen_upstream": {
            "CKPT3":
                "PASS_GENOMIC_PRETRAINING",

            "CKPT4":
                "PASS_TEMPORAL_PRETRAINING",
        },

        "architecture": {
            "temporal_input":
                192,

            "tumor_input":
                128,

            "fusion_hidden":
                256,

            "survival_horizon_months":
                24,

            "competing_causes": [
                "progression",
                "death",
                "switch",
            ],

            "parameter_count":
                int(
                    checkpoint[
                        "parameter_count"
                    ]
                ),
        },

        "training_population": {
            "scan_rows":
                prepare_report[
                    "scan_rows"
                ],

            "survival_landmarks":
                prepare_report[
                    "survival_landmarks"
                ],

            "next_scan_targets":
                prepare_report[
                    "next_scan_targets"
                ],

            "postprog_targets":
                prepare_report[
                    "postprog_targets"
                ],
        },

        "genomics": {
            "available_fraction_at_scan":
                prepare_report[
                    "genomic_available_fraction"
                ],

            "timing_rule":
                (
                    "latest sample with availability_day < scan day"
                ),
        },

        "test": {
            "pre_nll":
                metrics[
                    "pre"
                ][
                    "mean_competing_risk_nll"
                ],

            "post_nll":
                metrics[
                    "post"
                ][
                    "mean_competing_risk_nll"
                ],

            "paired_gain":
                gain,

            "pre_next_scan_macro_auprc":
                metrics[
                    "pre"
                ].get(
                    "next_scan_macro_auprc"
                ),

            "post_next_scan_macro_auprc":
                metrics[
                    "post"
                ].get(
                    "next_scan_macro_auprc"
                ),

            "postprog_macro_auprc":
                metrics[
                    "post"
                ].get(
                    "postprog_macro_auprc"
                ),
        },

        "next_action":
            (
                "Proceed to CKPT6: paired scientific proof, "
                "stale-line-start comparison, scan-number analysis, "
                "major ablations, genomic/temporal/current-scan "
                "ablation tests, and final internal candidate freeze."
            ),
    }

    atomic_json(
        repo
        / "artifacts"
        / "handoff"
        / "checkpoint_05.json",
        handoff,
    )

    print("")
    print(
        "========== CKPT5 SUMMARY =========="
    )

    print(
        f"status={status}"
    )

    print(
        "scan_rows="
        f"{prepare_report['scan_rows']}"
    )

    print(
        "survival_landmarks="
        f"{prepare_report['survival_landmarks']}"
    )

    print(
        "next_scan_targets="
        f"{prepare_report['next_scan_targets']}"
    )

    print(
        "postprog_targets="
        f"{prepare_report['postprog_targets']}"
    )

    print(
        "genomic_available_fraction="
        f"{prepare_report['genomic_available_fraction']}"
    )

    print(
        "model_parameters="
        f"{checkpoint['parameter_count']}"
    )

    print(
        "pre_test_competing_risk_nll="
        f"{metrics['pre']['mean_competing_risk_nll']}"
    )

    print(
        "post_test_competing_risk_nll="
        f"{metrics['post']['mean_competing_risk_nll']}"
    )

    print(
        "paired_pre_minus_post_nll_gain="
        f"{gain}"
    )

    print(
        "pre_next_scan_macro_auprc="
        f"{metrics['pre'].get('next_scan_macro_auprc')}"
    )

    print(
        "post_next_scan_macro_auprc="
        f"{metrics['post'].get('next_scan_macro_auprc')}"
    )

    print(
        "post_postprog_macro_auprc="
        f"{metrics['post'].get('postprog_macro_auprc')}"
    )

    print(
        "handoff="
        "artifacts/handoff/checkpoint_05.json"
    )

    print(
        "========== CKPT5 SUMMARY END =========="
    )

    print("")
    print(
        "========== CKPT5 DECISION PACKET =========="
    )

    print(
        "prepare_report="
        f"{prepare_report}"
    )

    print(
        "pre_metrics="
        f"{metrics['pre']}"
    )

    print(
        "post_metrics="
        f"{metrics['post']}"
    )

    print(
        "paired_scan_information_gain="
        f"{gain}"
    )

    print(
        "last_training_epoch="
        f"{history[-1] if history else None}"
    )

    print(
        "========== CKPT5 DECISION PACKET END =========="
    )


###############################################################################
# CLI
###############################################################################


def main() -> int:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "command",
        choices=[
            "self-test",
            "prepare",
            "train",
            "evaluate",
            "finalize",
        ],
    )

    parser.add_argument(
        "--repo-root",
        default=".",
    )

    parser.add_argument(
        "--output-dir",
        default="artifacts/checkpoint5",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=24,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
    )

    args = parser.parse_args()

    repo = Path(
        args.repo_root
    ).resolve()

    output_arg = Path(
        args.output_dir
    )

    out = (
        output_arg.resolve()
        if output_arg.is_absolute()
        else (
            repo
            / output_arg
        ).resolve()
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.command == "self-test":

        self_test()

    elif args.command == "prepare":

        prepare(
            repo,
            out,
        )

    elif args.command == "train":

        train_model(
            repo,
            out,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )

    elif args.command == "evaluate":

        evaluate(
            repo,
            out,
        )

    elif args.command == "finalize":

        finalize(
            repo,
            out,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
