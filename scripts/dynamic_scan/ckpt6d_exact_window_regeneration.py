#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


SEED = 20260927


###############################################################################
# Basic helpers
###############################################################################


def atomic_json(path: Path, obj: Any) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            obj,
            indent=2,
            sort_keys=True,
            default=lambda x:
                int(x)
                if isinstance(x, np.integer)
                else float(x)
                if isinstance(x, np.floating)
                else bool(x)
                if isinstance(x, np.bool_)
                else str(x),
        ),
        encoding="utf-8",
    )

    if not tmp.exists() or tmp.stat().st_size == 0:
        raise RuntimeError(
            f"empty JSON write: {path}"
        )

    json.loads(
        tmp.read_text(
            encoding="utf-8"
        )
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

    if (
        not tmp.exists()
        or tmp.stat().st_size == 0
    ):
        raise RuntimeError(
            f"empty parquet write: {path}"
        )

    check = pd.read_parquet(
        tmp
    )

    if len(check) != len(frame):
        raise RuntimeError(
            f"parquet row verification failed: {path}"
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


def sha256(path: Path) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as handle:

        while True:

            block = handle.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def import_module(
    name: str,
    path: Path,
):

    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            f"cannot import {path}"
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
# source_days_json
###############################################################################


def flatten_numeric(
    obj: Any,
    output: list[int],
) -> None:

    if obj is None:
        return

    if isinstance(
        obj,
        (
            int,
            np.integer,
        ),
    ):

        output.append(
            int(obj)
        )
        return

    if isinstance(
        obj,
        float,
    ):

        if np.isfinite(obj):
            output.append(
                int(round(obj))
            )

        return

    if isinstance(
        obj,
        str,
    ):

        text = obj.strip()

        if not text:
            return

        try:

            value = float(text)

            if np.isfinite(value):
                output.append(
                    int(round(value))
                )
                return

        except Exception:
            return

    if isinstance(
        obj,
        dict,
    ):

        for value in obj.values():
            flatten_numeric(
                value,
                output,
            )

        return

    if isinstance(
        obj,
        (
            list,
            tuple,
            set,
        ),
    ):

        for value in obj:
            flatten_numeric(
                value,
                output,
            )


def parse_source_days(
    value: Any,
) -> list[int]:

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
            dict,
        ),
    ):

        parsed = []

        flatten_numeric(
            value,
            parsed,
        )

        return sorted(
            set(parsed)
        )

    if pd.isna(value):
        return []

    text = str(
        value
    ).strip()

    if not text:
        return []

    try:

        obj = json.loads(
            text
        )

        parsed = []

        flatten_numeric(
            obj,
            parsed,
        )

        if parsed:
            return sorted(
                set(parsed)
            )

    except Exception:
        pass

    numbers = re.findall(
        r"-?\d+(?:\.\d+)?",
        text,
    )

    return sorted(
        {
            int(
                round(
                    float(number)
                )
            )
            for number in numbers
        }
    )


###############################################################################
# Authoritative CKPT1 line semantics
###############################################################################


def normalize_scan_table(
    path: Path,
) -> pd.DataFrame:

    frame = pd.read_parquet(
        path
    ).copy()

    required = {
        "patient_id",
        "scan_episode_id",
        "landmark_day",
        "episode_end_day",
        "source_days_json",
    }

    missing = (
        required
        - set(
            frame.columns
        )
    )

    if missing:

        raise RuntimeError(
            f"{path} missing required fields: {missing}"
        )

    frame[
        "patient_id"
    ] = (
        frame[
            "patient_id"
        ]
        .astype(str)
        .str.strip()
    )

    frame[
        "scan_episode_id"
    ] = (
        frame[
            "scan_episode_id"
        ]
        .astype(str)
        .str.strip()
    )

    for column in (
        "landmark_day",
        "episode_end_day",
    ):

        frame[
            column
        ] = pd.to_numeric(
            frame[
                column
            ],
            errors="raise",
        ).astype(
            int
        )

    if not (
        frame[
            "landmark_day"
        ].to_numpy()
        == frame[
            "episode_end_day"
        ].to_numpy()
    ).all():

        raise RuntimeError(
            f"{path}: landmark is not episode END day"
        )

    return frame


def build_authoritative_source_day_map(
    w3: pd.DataFrame,
) -> tuple[
    dict[tuple[str, int], int],
    dict[str, Any],
]:

    if "treatment_line" not in w3.columns:

        raise RuntimeError(
            "authoritative W3 table lacks treatment_line"
        )

    line_sets: dict[
        tuple[str, int],
        set[int],
    ] = defaultdict(
        set
    )

    episode_missing_landmark = 0
    source_day_occurrences = 0

    for row in w3.itertuples(
        index=False
    ):

        patient = str(
            row.patient_id
        )

        line = int(
            row.treatment_line
        )

        landmark = int(
            row.landmark_day
        )

        days = parse_source_days(
            row.source_days_json
        )

        if landmark not in days:

            episode_missing_landmark += 1
            continue

        for day in days:

            line_sets[
                (
                    patient,
                    int(day),
                )
            ].add(
                line
            )

            source_day_occurrences += 1

    if episode_missing_landmark:

        raise RuntimeError(
            "Authoritative W3 source_days_json does not contain "
            f"its landmark day for {episode_missing_landmark} episodes"
        )

    conflicts = {
        key:
            sorted(values)
        for key, values
        in line_sets.items()
        if len(values) != 1
    }

    if conflicts:

        preview = list(
            conflicts.items()
        )[:20]

        raise RuntimeError(
            "A raw radiology source day belongs to multiple "
            "authoritative CKPT1 treatment lines. "
            f"conflicts={len(conflicts)} preview={preview}"
        )

    mapping = {
        key:
            next(
                iter(values)
            )
        for key, values
        in line_sets.items()
    }

    mismatches = []

    for row in w3.itertuples(
        index=False
    ):

        key = (
            str(
                row.patient_id
            ),
            int(
                row.landmark_day
            ),
        )

        observed = mapping.get(
            key
        )

        expected = int(
            row.treatment_line
        )

        if observed != expected:

            mismatches.append(
                (
                    key,
                    expected,
                    observed,
                )
            )

    if mismatches:

        raise RuntimeError(
            "W3 authoritative source-day replay failed: "
            f"{len(mismatches)} mismatches; "
            f"preview={mismatches[:20]}"
        )

    report = {
        "authoritative_w3_rows":
            int(
                len(
                    w3
                )
            ),

        "authoritative_w3_patients":
            int(
                w3[
                    "patient_id"
                ].nunique()
            ),

        "unique_authoritative_source_days":
            int(
                len(
                    mapping
                )
            ),

        "source_day_occurrences":
            int(
                source_day_occurrences
            ),

        "source_day_line_conflicts":
            0,

        "w3_landmark_line_replay_matches":
            int(
                len(
                    w3
                )
            ),

        "w3_landmark_line_replay_fraction":
            1.0,
    }

    return (
        mapping,
        report,
    )


def assign_exact_lines(
    frame: pd.DataFrame,
    source_day_to_line: dict[
        tuple[str, int],
        int
    ],
    window: int,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
    pd.DataFrame,
]:

    records = []
    excluded = []

    for row in frame.itertuples(
        index=False
    ):

        patient = str(
            row.patient_id
        )

        landmark = int(
            row.landmark_day
        )

        days = parse_source_days(
            row.source_days_json
        )

        mapped = [
            source_day_to_line[
                (
                    patient,
                    day,
                )
            ]
            for day in days
            if (
                patient,
                day,
            )
            in source_day_to_line
        ]

        mapped_unique = sorted(
            set(
                mapped
            )
        )

        landmark_line = (
            source_day_to_line.get(
                (
                    patient,
                    landmark,
                )
            )
        )

        reason = None

        if landmark_line is None:

            reason = (
                "LANDMARK_DAY_NOT_IN_AUTHORITATIVE_MBC_SOURCE_DAY_MAP"
            )

        elif len(
            mapped_unique
        ) > 1:

            reason = (
                "EPISODE_SPANS_MULTIPLE_AUTHORITATIVE_TREATMENT_LINES"
            )

        elif (
            mapped_unique
            and landmark_line
            not in mapped_unique
        ):

            reason = (
                "LANDMARK_LINE_DISAGREES_WITH_COMPONENT_LINES"
            )

        if reason is not None:

            excluded.append(
                {
                    "patient_id":
                        patient,

                    "scan_episode_id":
                        str(
                            row.scan_episode_id
                        ),

                    "landmark_day":
                        landmark,

                    "episode_window_days":
                        window,

                    "mapped_component_lines":
                        json.dumps(
                            mapped_unique
                        ),

                    "reason":
                        reason,
                }
            )

            continue

        record = row._asdict()

        record[
            "treatment_line"
        ] = int(
            landmark_line
        )

        record[
            "line_mapping_source"
        ] = (
            "CKPT1_AUTHORITATIVE_W3_SOURCE_DAY_TRANSFER"
        )

        record[
            "line_mapping_component_lines_json"
        ] = json.dumps(
            mapped_unique
        )

        records.append(
            record
        )

    kept = pd.DataFrame(
        records
    )

    dropped = pd.DataFrame(
        excluded
    )

    if kept.empty:

        raise RuntimeError(
            f"W{window}: exact line mapping retained zero scans"
        )

    if (
        kept[
            [
                "patient_id",
                "scan_episode_id",
            ]
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            f"W{window}: mapped episode key is not unique"
        )

    reason_counts = (
        dropped[
            "reason"
        ]
        .value_counts()
        .to_dict()
        if not dropped.empty
        else {}
    )

    report = {
        "window":
            window,

        "input_rows":
            int(
                len(
                    frame
                )
            ),

        "input_patients":
            int(
                frame[
                    "patient_id"
                ].nunique()
            ),

        "mapped_rows":
            int(
                len(
                    kept
                )
            ),

        "mapped_patients":
            int(
                kept[
                    "patient_id"
                ].nunique()
            ),

        "excluded_rows":
            int(
                len(
                    dropped
                )
            ),

        "mapping_fraction":
            float(
                len(
                    kept
                )
                / max(
                    len(
                        frame
                    ),
                    1,
                )
            ),

        "excluded_reason_counts":
            {
                str(key):
                    int(value)
                for key, value
                in reason_counts.items()
            },

        "line_counts":
            {
                str(
                    int(key)
                ):
                    int(value)
                for key, value
                in kept[
                    "treatment_line"
                ]
                .value_counts()
                .sort_index()
                .to_dict()
                .items()
            },
    }

    return (
        kept,
        report,
        dropped,
    )


###############################################################################
# CKPT4 exact regeneration
###############################################################################


def identify_prepare_breast_outputs(
    result: Any,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    frames: list[
        pd.DataFrame
    ] = []

    def collect(
        value: Any,
    ) -> None:

        if isinstance(
            value,
            pd.DataFrame,
        ):

            frames.append(
                value
            )

        elif isinstance(
            value,
            dict,
        ):

            for child in value.values():
                collect(
                    child
                )

        elif isinstance(
            value,
            (
                list,
                tuple,
            ),
        ):

            for child in value:
                collect(
                    child
                )

    collect(
        result
    )

    token_candidates = []
    scan_candidates = []

    for frame in frames:

        columns = set(
            frame.columns
        )

        if {
            "patient_id",
            "event_name",
        }.issubset(
            columns
        ) and (
            "day"
            in columns
        ):

            token_candidates.append(
                frame
            )

        if {
            "patient_id",
            "scan_episode_id",
            "landmark_day",
        }.issubset(
            columns
        ):

            scan_candidates.append(
                frame
            )

    if len(
        token_candidates
    ) != 1:

        raise RuntimeError(
            "Could not uniquely identify breast token table from "
            "prepare_breast() return. "
            f"frame_columns={[list(x.columns) for x in frames]}"
        )

    scan_candidates = [
        frame
        for frame in scan_candidates
        if len(frame)
        < len(
            token_candidates[
                0
            ]
        )
    ]

    if len(
        scan_candidates
    ) != 1:

        raise RuntimeError(
            "Could not uniquely identify breast scan-index table from "
            "prepare_breast() return. "
            f"candidates={[list(x.columns) for x in scan_candidates]}"
        )

    return (
        token_candidates[
            0
        ],
        scan_candidates[
            0
        ],
    )


def regenerate_window(
    repo: Path,
    out: Path,
    ckpt4: Any,
    scans_path: Path,
    label: str,
    batch_size: int,
) -> dict[str, Any]:

    target = (
        out
        / "regenerated"
        / label
    )

    prepared = (
        target
        / "prepared"
    )

    prepared.mkdir(
        parents=True,
        exist_ok=True,
    )

    ###########################################################################
    # Frozen CKPT4 weights.
    ###########################################################################

    frozen_checkpoint = (
        repo
        / "artifacts"
        / "checkpoint4"
        / "temporal_encoder.pt"
    )

    copied_checkpoint = (
        target
        / "temporal_encoder.pt"
    )

    shutil.copy2(
        frozen_checkpoint,
        copied_checkpoint,
    )

    if (
        sha256(
            frozen_checkpoint
        )
        != sha256(
            copied_checkpoint
        )
    ):

        raise RuntimeError(
            f"{label}: frozen temporal checkpoint copy changed bytes"
        )

    ###########################################################################
    # IMPORTANT:
    # frozen CKPT4 prepare_breast() requires the split DATAFRAME itself,
    # not a dict mapping patient -> split.
    ###########################################################################

    splits_frame = pd.read_parquet(
        repo
        / "artifacts"
        / "checkpoint1"
        / "canonical"
        / "chord_patient_splits.parquet"
    ).copy()

    required_split_columns = {
        "patient_id",
        "split",
    }

    missing_split_columns = (
        required_split_columns
        - set(
            splits_frame.columns
        )
    )

    if missing_split_columns:

        raise RuntimeError(
            f"{label}: split table missing "
            f"{sorted(missing_split_columns)}"
        )

    splits_frame[
        "patient_id"
    ] = (
        splits_frame[
            "patient_id"
        ]
        .astype(str)
        .str.strip()
    )

    splits_frame[
        "split"
    ] = (
        splits_frame[
            "split"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    if (
        splits_frame[
            "patient_id"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            f"{label}: split table has duplicate patients"
        )

    unknown_splits = (
        set(
            splits_frame[
                "split"
            ]
        )
        - {
            "train",
            "val",
            "test",
        }
    )

    if unknown_splits:

        raise RuntimeError(
            f"{label}: unknown patient split values "
            f"{sorted(unknown_splits)}"
        )

    ###########################################################################
    # Frozen site vocabulary.
    ###########################################################################

    site_vocab = [
        line.strip()
        for line
        in (
            repo
            / "artifacts"
            / "checkpoint4"
            / "prepared"
            / "site_vocab_32.txt"
        )
        .read_text(
            encoding="utf-8"
        )
        .splitlines()
        if line.strip()
    ]

    if not site_vocab:

        raise RuntimeError(
            "Frozen CKPT4 site vocabulary is empty"
        )

    canonical_path = (
        repo
        / "artifacts"
        / "checkpoint1"
        / "canonical"
        / "chord_canonical_events.parquet"
    )

    ###########################################################################
    # Exact frozen CKPT4 preparation pathway.
    #
    # prepare_breast() returns BASE EVENTS.
    # prepare_command() subsequently runs add_time_and_sequence_targets()
    # and separately constructs the scan-index table from breast_base.
    ###########################################################################

    print(
        f"[CKPT6D1_REGEN] {label} prepare_breast start",
        flush=True,
    )

    breast_base = ckpt4.prepare_breast(
        canonical_path,
        scans_path,
        splits_frame,
        site_vocab,
    )

    if not isinstance(
        breast_base,
        pd.DataFrame,
    ):

        raise RuntimeError(
            f"{label}: prepare_breast returned "
            f"{type(breast_base).__name__}, expected DataFrame"
        )

    if breast_base.empty:

        raise RuntimeError(
            f"{label}: prepare_breast returned zero rows"
        )

    required_base_columns = {
        "patient_id",
        "day",
        "type_id",
        "line",
        "scan_state",
        "scan_episode_id",
        "site_mask",
        "site_known",
        "region_imaged_mask",
        "region_known_mask",
        "split",
    }

    missing_base = (
        required_base_columns
        - set(
            breast_base.columns
        )
    )

    if missing_base:

        raise RuntimeError(
            f"{label}: breast_base missing "
            f"{sorted(missing_base)}"
        )

    print(
        f"[CKPT6D1_REGEN] {label} "
        f"breast_base_rows={len(breast_base)}",
        flush=True,
    )

    ###########################################################################
    # Exact frozen CKPT4 sequence construction.
    ###########################################################################

    print(
        f"[CKPT6D1_REGEN] {label} "
        "add_time_and_sequence_targets start",
        flush=True,
    )

    breast_tokens = (
        ckpt4.add_time_and_sequence_targets(
            breast_base
        )
    )

    if not isinstance(
        breast_tokens,
        pd.DataFrame,
    ):

        raise RuntimeError(
            f"{label}: add_time_and_sequence_targets "
            "did not return DataFrame"
        )

    if breast_tokens.empty:

        raise RuntimeError(
            f"{label}: breast token table is empty"
        )

    ###########################################################################
    # Exact frozen CKPT4 scan-index construction copied from prepare_command().
    ###########################################################################

    if not hasattr(
        ckpt4,
        "EVENT_TO_ID",
    ):

        raise RuntimeError(
            "Frozen CKPT4 lacks EVENT_TO_ID"
        )

    if (
        "SCAN_EPISODE"
        not in ckpt4.EVENT_TO_ID
    ):

        raise RuntimeError(
            "Frozen CKPT4 EVENT_TO_ID lacks SCAN_EPISODE"
        )

    scan_index = (
        breast_base[
            breast_base[
                "type_id"
            ]
            == ckpt4.EVENT_TO_ID[
                "SCAN_EPISODE"
            ]
        ][
            [
                "patient_id",
                "day",
                "line",
                "scan_state",
                "scan_episode_id",
                "site_mask",
                "site_known",
                "region_imaged_mask",
                "region_known_mask",
                "split",
            ]
        ]
        .copy()
    )

    scan_index = (
        scan_index
        .sort_values(
            [
                "patient_id",
                "day",
                "scan_episode_id",
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    if (
        scan_index[
            [
                "patient_id",
                "scan_episode_id",
            ]
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            f"{label}: duplicate breast scan episode keys"
        )

    ###########################################################################
    # Sanity-check against mapped input scans.
    ###########################################################################

    source_scans = pd.read_parquet(
        scans_path
    ).copy()

    source_scans[
        "patient_id"
    ] = (
        source_scans[
            "patient_id"
        ]
        .astype(str)
        .str.strip()
    )

    source_scans[
        "scan_episode_id"
    ] = (
        source_scans[
            "scan_episode_id"
        ]
        .astype(str)
        .str.strip()
    )

    source_scans = source_scans[
        source_scans[
            "patient_id"
        ].isin(
            set(
                splits_frame[
                    "patient_id"
                ]
            )
        )
    ].copy()

    expected_keys = set(
        zip(
            source_scans[
                "patient_id"
            ],
            source_scans[
                "scan_episode_id"
            ],
        )
    )

    actual_keys = set(
        zip(
            scan_index[
                "patient_id"
            ].astype(str),
            scan_index[
                "scan_episode_id"
            ].astype(str),
        )
    )

    if expected_keys != actual_keys:

        missing = (
            expected_keys
            - actual_keys
        )

        extra = (
            actual_keys
            - expected_keys
        )

        raise RuntimeError(
            f"{label}: prepared scan keys differ from "
            f"mapped input scans: missing={len(missing)} "
            f"extra={len(extra)} "
            f"missing_preview={list(missing)[:10]} "
            f"extra_preview={list(extra)[:10]}"
        )

    ###########################################################################
    # Persist exactly what cache_scan_states() expects.
    ###########################################################################

    atomic_parquet(
        prepared
        / "breast_tokens.parquet",
        breast_tokens,
    )

    atomic_parquet(
        prepared
        / "breast_scan_index.parquet",
        scan_index,
    )

    atomic_text(
        prepared
        / "site_vocab_32.txt",
        "\n".join(
            site_vocab
        )
        + "\n",
    )

    prep_report = {
        "label":
            label,

        "breast_base_rows":
            int(
                len(
                    breast_base
                )
            ),

        "breast_token_rows":
            int(
                len(
                    breast_tokens
                )
            ),

        "scan_rows":
            int(
                len(
                    scan_index
                )
            ),

        "scan_patients":
            int(
                scan_index[
                    "patient_id"
                ].nunique()
            ),

        "scan_state_counts": {
            str(
                int(key)
            ):
                int(
                    value
                )
            for key, value
            in scan_index[
                "scan_state"
            ]
            .value_counts()
            .sort_index()
            .to_dict()
            .items()
        },

        "split_patient_counts": {
            str(key):
                int(value)
            for key, value
            in (
                breast_base[
                    [
                        "patient_id",
                        "split",
                    ]
                ]
                .drop_duplicates()
                [
                    "split"
                ]
                .value_counts()
                .to_dict()
                .items()
            )
        },

        "source_scan_keys":
            int(
                len(
                    expected_keys
                )
            ),

        "prepared_scan_keys":
            int(
                len(
                    actual_keys
                )
            ),

        "exact_source_key_match":
            True,

        "preparation_contract":
            (
                "Frozen CKPT4 prepare_breast -> "
                "add_time_and_sequence_targets -> "
                "prepare_command scan-index construction"
            ),
    }

    atomic_json(
        prepared
        / "regeneration_prepare_report.json",
        prep_report,
    )

    print(
        f"[CKPT6D1_REGEN] {label} "
        f"tokens={len(breast_tokens)} "
        f"scan_index={len(scan_index)} "
        f"patients={scan_index['patient_id'].nunique()}",
        flush=True,
    )

    ###########################################################################
    # Frozen CKPT4 PRE/POST cache.
    ###########################################################################

    print(
        f"[CKPT6D1_REGEN] {label} cache_scan_states start",
        flush=True,
    )

    cache_result = (
        ckpt4.cache_scan_states(
            target,
            batch_size,
        )
    )

    embedding_path = (
        target
        / "breast_scan_prepost_embeddings_f16.npy"
    )

    index_path = (
        target
        / "breast_scan_prepost_index.parquet"
    )

    if (
        not embedding_path.exists()
        or embedding_path.stat().st_size == 0
    ):

        raise RuntimeError(
            f"{label}: scan-state cache missing"
        )

    if (
        not index_path.exists()
        or index_path.stat().st_size == 0
    ):

        raise RuntimeError(
            f"{label}: scan-state index missing"
        )

    embeddings = np.load(
        embedding_path,
        mmap_mode="r",
    )

    index = pd.read_parquet(
        index_path
    )

    if (
        embeddings.shape[
            0
        ]
        != len(
            index
        )
    ):

        raise RuntimeError(
            f"{label}: embedding/index row mismatch"
        )

    if (
        embeddings.shape[
            1:
        ]
        != (
            2,
            192,
        )
    ):

        raise RuntimeError(
            f"{label}: unexpected embedding shape "
            f"{embeddings.shape}"
        )

    print(
        f"[CKPT6D1_REGEN_COMPLETE] {label} "
        f"embedding_shape={embeddings.shape}",
        flush=True,
    )

    return {
        "label":
            label,

        "base_rows":
            int(
                len(
                    breast_base
                )
            ),

        "token_rows":
            int(
                len(
                    breast_tokens
                )
            ),

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

        "embedding_shape":
            list(
                embeddings.shape
            ),

        "checkpoint_sha256":
            sha256(
                copied_checkpoint
            ),

        "exact_source_key_match":
            True,

        "cache_result":
            cache_result,
    }

###############################################################################
# W3 regeneration equivalence
###############################################################################


def compare_w3_cache(
    repo: Path,
    regenerated_root: Path,
) -> dict[str, Any]:

    frozen_index = pd.read_parquet(
        repo
        / "artifacts"
        / "checkpoint4"
        / "breast_scan_prepost_index.parquet"
    )

    frozen_embedding = np.load(
        repo
        / "artifacts"
        / "checkpoint4"
        / "breast_scan_prepost_embeddings_f16.npy",
        mmap_mode="r",
    )

    new_index = pd.read_parquet(
        regenerated_root
        / "breast_scan_prepost_index.parquet"
    )

    new_embedding = np.load(
        regenerated_root
        / "breast_scan_prepost_embeddings_f16.npy",
        mmap_mode="r",
    )

    frozen_keys = list(
        zip(
            frozen_index[
                "patient_id"
            ].astype(str),
            frozen_index[
                "scan_episode_id"
            ].astype(str),
        )
    )

    new_keys = list(
        zip(
            new_index[
                "patient_id"
            ].astype(str),
            new_index[
                "scan_episode_id"
            ].astype(str),
        )
    )

    if set(
        frozen_keys
    ) != set(
        new_keys
    ):

        missing = (
            set(
                frozen_keys
            )
            - set(
                new_keys
            )
        )

        extra = (
            set(
                new_keys
            )
            - set(
                frozen_keys
            )
        )

        raise RuntimeError(
            "W3 regenerated scan keys do not equal frozen CKPT4 keys: "
            f"missing={len(missing)} extra={len(extra)}"
        )

    new_row = {
        key:
            row
        for row, key
        in enumerate(
            new_keys
        )
    }

    reorder = np.asarray(
        [
            new_row[
                key
            ]
            for key in frozen_keys
        ],
        dtype=np.int64,
    )

    a = np.asarray(
        frozen_embedding,
        dtype=np.float32,
    )

    b = np.asarray(
        new_embedding[
            reorder
        ],
        dtype=np.float32,
    )

    difference = np.abs(
        a - b
    )

    max_abs = float(
        difference.max()
    )

    mean_abs = float(
        difference.mean()
    )

    flat_a = a.reshape(
        -1,
        192,
    )

    flat_b = b.reshape(
        -1,
        192,
    )

    cosine = np.sum(
        flat_a
        * flat_b,
        axis=1,
    ) / (
        np.linalg.norm(
            flat_a,
            axis=1,
        )
        * np.linalg.norm(
            flat_b,
            axis=1,
        )
        + 1e-12
    )

    cosine_mean = float(
        cosine.mean()
    )

    cosine_min = float(
        cosine.min()
    )

    if (
        max_abs
        > 0.02
        or cosine_mean
        < 0.9999
    ):

        raise RuntimeError(
            "W3 regenerated temporal states fail frozen-cache replay: "
            f"max_abs={max_abs} "
            f"mean_abs={mean_abs} "
            f"cosine_mean={cosine_mean}"
        )

    common_columns = [
        column
        for column
        in (
            "patient_id",
            "scan_episode_id",
            "line",
            "landmark_day",
            "scan_state",
            "split",
        )
        if (
            column
            in frozen_index.columns
            and column
            in new_index.columns
        )
    ]

    frozen_compare = (
        frozen_index[
            common_columns
        ]
        .copy()
        .sort_values(
            [
                "patient_id",
                "scan_episode_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    new_compare = (
        new_index[
            common_columns
        ]
        .copy()
        .sort_values(
            [
                "patient_id",
                "scan_episode_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    if not frozen_compare.equals(
        new_compare
    ):

        raise RuntimeError(
            "W3 regenerated cache metadata differs from frozen CKPT4 metadata"
        )

    return {
        "frozen_rows":
            int(
                len(
                    frozen_index
                )
            ),

        "regenerated_rows":
            int(
                len(
                    new_index
                )
            ),

        "exact_key_set":
            True,

        "metadata_exact":
            True,

        "embedding_max_abs_difference":
            max_abs,

        "embedding_mean_abs_difference":
            mean_abs,

        "embedding_cosine_mean":
            cosine_mean,

        "embedding_cosine_min":
            cosine_min,
    }


###############################################################################
# Main
###############################################################################


def main() -> int:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo-root",
        default=".",
    )

    parser.add_argument(
        "--output-dir",
        default="artifacts/checkpoint6d",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    args = parser.parse_args()

    repo = Path(
        args.repo_root
    ).resolve()

    out = (
        repo
        / args.output_dir
    ).resolve()

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    canonical = (
        repo
        / "artifacts"
        / "checkpoint1"
        / "canonical"
    )

    w0 = normalize_scan_table(
        canonical
        / "chord_breast_scan_episodes_w0.parquet"
    )

    w3 = normalize_scan_table(
        canonical
        / "chord_mbc_scan_episodes_w3.parquet"
    )

    w7 = normalize_scan_table(
        canonical
        / "chord_breast_scan_episodes_w7.parquet"
    )

    if len(
        w3
    ) != 49189:

        raise RuntimeError(
            f"authoritative W3 count changed: {len(w3)}"
        )

    (
        source_day_map,
        replay_report,
    ) = build_authoritative_source_day_map(
        w3
    )

    print(
        "[CKPT6D1_EXACT_W3_LINE_REPLAY_PASS]",
        replay_report,
        flush=True,
    )

    (
        w0_mapped,
        w0_report,
        w0_excluded,
    ) = assign_exact_lines(
        w0,
        source_day_map,
        0,
    )

    (
        w7_mapped,
        w7_report,
        w7_excluded,
    ) = assign_exact_lines(
        w7,
        source_day_map,
        7,
    )

    w3_mapped = w3.copy()

    w3_mapped[
        "line_mapping_source"
    ] = (
        "CKPT1_AUTHORITATIVE_W3"
    )

    mapped_dir = (
        out
        / "mapped_scans"
    )

    atomic_parquet(
        mapped_dir
        / "w0_mbc_exact_line.parquet",
        w0_mapped,
    )

    atomic_parquet(
        mapped_dir
        / "w3_mbc_authoritative.parquet",
        w3_mapped,
    )

    atomic_parquet(
        mapped_dir
        / "w7_mbc_exact_line.parquet",
        w7_mapped,
    )

    atomic_parquet(
        mapped_dir
        / "w0_excluded.parquet",
        w0_excluded,
    )

    atomic_parquet(
        mapped_dir
        / "w7_excluded.parquet",
        w7_excluded,
    )

    w3_days = set(
        zip(
            w3[
                "patient_id"
            ],
            w3[
                "landmark_day"
            ],
        )
    )

    w0_days = set(
        zip(
            w0_mapped[
                "patient_id"
            ],
            w0_mapped[
                "landmark_day"
            ],
        )
    )

    w7_days = set(
        zip(
            w7_mapped[
                "patient_id"
            ],
            w7_mapped[
                "landmark_day"
            ],
        )
    )

    matching = {
        "w3_rows":
            int(
                len(
                    w3
                )
            ),

        "w3_landmarks_present_w0":
            int(
                len(
                    w3_days
                    & w0_days
                )
            ),

        "w3_fraction_present_w0":
            float(
                len(
                    w3_days
                    & w0_days
                )
                / len(
                    w3_days
                )
            ),

        "w3_landmarks_present_w7":
            int(
                len(
                    w3_days
                    & w7_days
                )
            ),

        "w3_fraction_present_w7":
            float(
                len(
                    w3_days
                    & w7_days
                )
                / len(
                    w3_days
                )
            ),
    }

    ckpt4_path = (
        repo
        / "scripts"
        / "dynamic_scan"
        / "ckpt4_temporal_pretrain.py"
    )

    ckpt4 = import_module(
        "ckpt4_frozen_for_ckpt6d",
        ckpt4_path,
    )

    for function in (
        "prepare_breast",
        "cache_scan_states",
    ):

        if not hasattr(
            ckpt4,
            function,
        ):

            raise RuntimeError(
                f"frozen CKPT4 lacks {function}()"
            )

    print(
        "[CKPT6D1_CKPT4_API_PASS]",
        {
            "prepare_breast":
                str(
                    inspect.signature(
                        ckpt4.prepare_breast
                    )
                ),

            "cache_scan_states":
                str(
                    inspect.signature(
                        ckpt4.cache_scan_states
                    )
                ),
        },
        flush=True,
    )

    w3_run = regenerate_window(
        repo,
        out,
        ckpt4,
        mapped_dir
        / "w3_mbc_authoritative.parquet",
        "w3_replay",
        args.batch_size,
    )

    w3_replay = compare_w3_cache(
        repo,
        out
        / "regenerated"
        / "w3_replay",
    )

    print(
        "[CKPT6D1_W3_TEMPORAL_REPLAY_PASS]",
        w3_replay,
        flush=True,
    )

    w0_run = regenerate_window(
        repo,
        out,
        ckpt4,
        mapped_dir
        / "w0_mbc_exact_line.parquet",
        "w0",
        args.batch_size,
    )

    w7_run = regenerate_window(
        repo,
        out,
        ckpt4,
        mapped_dir
        / "w7_mbc_exact_line.parquet",
        "w7",
        args.batch_size,
    )

    report = {
        "status":
            "PASS_EXACT_WINDOW_TEMPORAL_REGENERATION",

        "method":
            (
                "Propagate authoritative CKPT1 W3 treatment-line labels "
                "through the raw radiology source days in source_days_json. "
                "Alternate grouping episodes are retained only when the "
                "landmark raw day has an authoritative mapping and the "
                "episode does not span multiple authoritative lines."
            ),

        "authoritative_line_replay":
            replay_report,

        "w0_line_mapping":
            w0_report,

        "w7_line_mapping":
            w7_report,

        "exact_landmark_overlap":
            matching,

        "w3_temporal_replay":
            w3_replay,

        "regeneration": {
            "w0":
                w0_run,

            "w3":
                w3_run,

            "w7":
                w7_run,
        },

        "frozen_temporal_checkpoint_sha256":
            sha256(
                repo
                / "artifacts"
                / "checkpoint4"
                / "temporal_encoder.pt"
            ),

        "scientific_contract": {
            "changed":
                "scan episode grouping window only",

            "unchanged": [
                "CKPT1 treatment-line semantics",
                "CKPT1 patient splits",
                "CKPT4 temporal encoder weights",
                "CKPT4 tokenization/availability rules",
                "PRE/POST current-scan timing convention",
                "CKPT6B alpha=0.75",
                "CKPT6C frozen candidate",
            ],

            "ambiguous_cross_line_episode_policy":
                "exclude; never force a treatment-line assignment",
        },
    }

    atomic_json(
        out
        / "exact_window_regeneration.json",
        report,
    )

    print("")
    print(
        "========== CKPT6D1 SUMMARY =========="
    )

    print(
        "status="
        f"{report['status']}"
    )

    print(
        "w3_line_replay_fraction="
        f"{replay_report['w3_landmark_line_replay_fraction']}"
    )

    print(
        "w0_mapping="
        f"{w0_report}"
    )

    print(
        "w7_mapping="
        f"{w7_report}"
    )

    print(
        "exact_landmark_overlap="
        f"{matching}"
    )

    print(
        "w3_temporal_replay="
        f"{w3_replay}"
    )

    print(
        "w0_regeneration="
        f"{w0_run}"
    )

    print(
        "w7_regeneration="
        f"{w7_run}"
    )

    print(
        "report="
        "artifacts/checkpoint6d/exact_window_regeneration.json"
    )

    print(
        "========== CKPT6D1 SUMMARY END =========="
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
