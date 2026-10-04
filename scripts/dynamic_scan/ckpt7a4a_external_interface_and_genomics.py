#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch


EXTERNAL_SITES = (
    "DFCI",
    "VICC",
)

PRIMARY_SCAN_STATE = (
    "NON_PROGRESSIVE"
)

EXPECTED_DIMENSIONS = {
    "genomic":
        128,

    "temporal":
        192,

    "scan":
        18,

    "context":
        6,
}


###############################################################################
# Utilities
###############################################################################


def norm(
    value: Any,
) -> str:

    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value).strip().lower(),
    ).strip("_")


def clean(
    value: Any,
) -> str:

    return str(
        value
    ).strip().upper()


def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:

        for block in iter(
            lambda:
                handle.read(
                    1024 * 1024
                ),
            b"",
        ):

            digest.update(
                block
            )

    return digest.hexdigest()


def atomic_json(
    path: Path,
    payload: Any,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix
        + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=lambda value:
                int(value)
                if isinstance(
                    value,
                    np.integer,
                )
                else float(value)
                if isinstance(
                    value,
                    np.floating,
                )
                else bool(value)
                if isinstance(
                    value,
                    np.bool_,
                )
                else str(value),
        ),
        encoding="utf-8",
    )

    if (
        not tmp.exists()
        or tmp.stat().st_size
        == 0
    ):

        raise RuntimeError(
            f"empty JSON write: {path}"
        )

    json.loads(
        tmp.read_text(
            encoding="utf-8"
        )
    )

    tmp.replace(
        path
    )


def atomic_text(
    path: Path,
    text: str,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix
        + ".tmp"
    )

    tmp.write_text(
        text,
        encoding="utf-8",
    )

    if (
        not tmp.exists()
        or tmp.stat().st_size
        == 0
    ):

        raise RuntimeError(
            f"empty text write: {path}"
        )

    tmp.replace(
        path
    )


def atomic_parquet(
    path: Path,
    frame: pd.DataFrame,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix
        + ".tmp"
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
            f"Parquet verification failed: {path}"
        )

    tmp.replace(
        path
    )


def atomic_npy(
    path: Path,
    array: np.ndarray,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = Path(
        str(path)
        + ".tmp"
    )

    with tmp.open(
        "wb"
    ) as handle:

        np.save(
            handle,
            array,
        )

    check = np.load(
        tmp,
        mmap_mode="r",
    )

    if check.shape != array.shape:

        raise RuntimeError(
            f"NPY verification failed: {path}"
        )

    tmp.replace(
        path
    )


def find_col(
    columns: Iterable[str],
    candidates: Iterable[str],
    *,
    required: bool = False,
) -> str | None:

    lookup = {
        norm(column):
            column
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
            f"Missing {list(candidates)}; "
            f"available={list(columns)}"
        )

    return None


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
            f"Unable to import {path}"
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
# Frozen feature-schema extraction
###############################################################################


def recursive_lists(
    payload: Any,
    prefix: str = "",
) -> list[
    tuple[
        str,
        list[Any],
    ]
]:

    result = []

    if isinstance(
        payload,
        dict,
    ):

        for key, value in payload.items():

            child = (
                f"{prefix}.{key}"
                if prefix
                else str(key)
            )

            result.extend(
                recursive_lists(
                    value,
                    child,
                )
            )

    elif isinstance(
        payload,
        list,
    ):

        result.append(
            (
                prefix,
                payload,
            )
        )

    return result


def resolve_feature_names(
    schema: dict[str, Any],
    *,
    expected_length: int,
    preferred_tokens: tuple[str, ...],
) -> tuple[
    str | None,
    list[str],
]:

    candidates = []

    for path, values in recursive_lists(
        schema
    ):

        if len(values) != expected_length:
            continue

        if not all(
            isinstance(
                value,
                str,
            )
            for value in values
        ):

            continue

        score = sum(
            token
            in norm(
                path
            )
            for token in preferred_tokens
        )

        candidates.append(
            (
                score,
                path,
                [
                    str(value)
                    for value
                    in values
                ],
            )
        )

    if not candidates:

        return (
            None,
            [],
        )

    candidates.sort(
        key=lambda item:
            (
                item[0],
                item[1],
            ),
        reverse=True,
    )

    return (
        candidates[
            0
        ][
            1
        ],
        candidates[
            0
        ][
            2
        ],
    )


###############################################################################
# AST/source-interface inventory
###############################################################################


def function_inventory(
    path: Path,
    keywords: tuple[str, ...],
    *,
    max_functions: int = 20,
) -> list[
    dict[str, Any]
]:

    source = path.read_text(
        encoding="utf-8"
    )

    lines = source.splitlines()

    tree = ast.parse(
        source
    )

    records = []

    for node in ast.walk(
        tree
    ):

        if not isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):

            continue

        if not hasattr(
            node,
            "end_lineno",
        ):

            continue

        start = int(
            node.lineno
        )

        end = int(
            node.end_lineno
        )

        segment = "\n".join(
            lines[
                start - 1:
                end
            ]
        )

        lower = segment.lower()

        hits = sorted(
            {
                keyword
                for keyword in keywords
                if keyword.lower()
                in lower
            }
        )

        if not hits:
            continue

        arguments = []

        for argument in (
            list(
                node.args.posonlyargs
            )
            + list(
                node.args.args
            )
            + list(
                node.args.kwonlyargs
            )
        ):

            arguments.append(
                argument.arg
            )

        relevant = []

        for line_number in range(
            start,
            end + 1,
        ):

            text = lines[
                line_number
                - 1
            ]

            if any(
                keyword.lower()
                in text.lower()
                for keyword in keywords
            ):

                relevant.append(
                    f"{line_number}: {text}"
                )

        records.append(
            {
                "function":
                    node.name,

                "start_line":
                    start,

                "end_line":
                    end,

                "arguments":
                    arguments,

                "hits":
                    hits,

                "relevant_lines":
                    relevant[
                        :100
                    ],
            }
        )

    records.sort(
        key=lambda item:
            (
                len(
                    item[
                        "hits"
                    ]
                ),
                -item[
                    "start_line"
                ],
            ),
        reverse=True,
    )

    return records[
        :max_functions
    ]


###############################################################################
# Torch checkpoint inventory without dumping tensors
###############################################################################


def summarize_object(
    value: Any,
    *,
    depth: int = 0,
) -> Any:

    if depth >= 3:

        return {
            "type":
                type(
                    value
                ).__name__,
        }

    if torch.is_tensor(
        value
    ):

        return {
            "type":
                "tensor",

            "shape":
                list(
                    value.shape
                ),

            "dtype":
                str(
                    value.dtype
                ),
        }

    if isinstance(
        value,
        np.ndarray,
    ):

        return {
            "type":
                "ndarray",

            "shape":
                list(
                    value.shape
                ),

            "dtype":
                str(
                    value.dtype
                ),
        }

    if isinstance(
        value,
        dict,
    ):

        result = {
            "type":
                "dict",

            "keys":
                list(
                    value.keys()
                )[
                    :100
                ],
        }

        children = {}

        for key in list(
            value.keys()
        )[
            :30
        ]:

            children[
                str(
                    key
                )
            ] = summarize_object(
                value[
                    key
                ],
                depth=(
                    depth
                    + 1
                ),
            )

        result[
            "children"
        ] = children

        return result

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):

        return {
            "type":
                type(
                    value
                ).__name__,

            "length":
                len(
                    value
                ),

            "first":
                (
                    summarize_object(
                        value[
                            0
                        ],
                        depth=(
                            depth
                            + 1
                        ),
                    )
                    if len(
                        value
                    )
                    else None
                ),
        }

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
            type(
                None
            ),
        ),
    ):

        return value

    return {
        "type":
            type(
                value
            ).__name__,
    }


def checkpoint_inventory(
    path: Path,
) -> dict[str, Any]:

    try:

        try:

            payload = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )

        except TypeError:

            payload = torch.load(
                path,
                map_location="cpu",
            )

        return {
            "status":
                "LOADED",

            "sha256":
                sha256_file(
                    path
                ),

            "summary":
                summarize_object(
                    payload
                ),
        }

    except Exception as exc:

        return {
            "status":
                "LOAD_FAILED",

            "sha256":
                sha256_file(
                    path
                ),

            "error":
                (
                    type(
                        exc
                    ).__name__
                    + ": "
                    + str(
                        exc
                    )
                ),
        }


###############################################################################
# External predictor access
###############################################################################


def load_external_predictors(
    repo: Path,
    bpc_root: Path,
) -> tuple[
    Any,
    dict[str, pd.DataFrame],
    dict[str, str],
]:

    ckpt7a2 = import_module(
        "ckpt7a2_7a4a",
        repo
        / "scripts"
        / "dynamic_scan"
        / "ckpt7a2_external_semantic_audit.py",
    )

    inventory = pd.read_parquet(
        repo
        / "artifacts"
        / "checkpoint7a1"
        / "schema_inventory.parquet"
    )

    guard = ckpt7a2.AccessGuard(
        bpc_root,
        inventory,
    )

    relative = {
        "clinical_patient":
            "cBioPortal_files/"
            "data_clinical_patient.txt",

        "clinical_sample":
            "cBioPortal_files/"
            "data_clinical_sample.txt",

        "sequencing":
            "cBioPortal_files/"
            "data_timeline_sequencing.txt",
    }

    frames = {
        name:
            guard.read_predictor(
                path
            )
        for name, path
        in relative.items()
    }

    (
        site_map,
        site_audit,
    ) = ckpt7a2.build_site_map(
        frames[
            "clinical_patient"
        ]
    )

    if (
        site_audit[
            "counts"
        ]
        != {
            "MSK": 529,
            "DFCI": 428,
            "VICC": 173,
        }
    ):

        raise RuntimeError(
            "BPC site partition changed."
        )

    ###########################################################################
    # Explicit quarantine assertion.
    ###########################################################################

    for item in guard.access_log:

        relative_path = str(
            item[
                "relative_path"
            ]
        ).lower()

        if any(
            token
            in relative_path
            for token in (
                "survival",
                "pfs",
                "censor",
            )
        ):

            raise RuntimeError(
                "External outcome quarantine violation."
            )

    return (
        guard,
        frames,
        site_map,
    )


###############################################################################
# External W3 landmark freeze
###############################################################################


def build_external_landmarks(
    repo: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
]:

    episodes = pd.read_parquet(
        repo
        / "artifacts"
        / "checkpoint7a3"
        / "bpc_w3_predictor_episodes.parquet"
    )

    external = episodes[
        episodes[
            "site"
        ].isin(
            EXTERNAL_SITES
        )
    ].copy()

    external[
        "landmark_day"
    ] = pd.to_numeric(
        external[
            "episode_end_day"
        ],
        errors="coerce",
    )

    if external[
        "landmark_day"
    ].isna().any():

        raise RuntimeError(
            "External W3 episode has missing landmark day."
        )

    external[
        "primary_candidate"
    ] = (
        external[
            "scan_state"
        ]
        == PRIMARY_SCAN_STATE
    )

    primary = external[
        external[
            "primary_candidate"
        ]
    ].copy()

    primary = (
        primary
        .sort_values(
            [
                "site",
                "patient_id",
                "landmark_day",
                "bpc_episode_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    primary[
        "external_landmark_row"
    ] = np.arange(
        len(
            primary
        ),
        dtype=int,
    )

    counts = {}

    for site in EXTERNAL_SITES:

        all_site = external[
            external[
                "site"
            ]
            == site
        ]

        primary_site = primary[
            primary[
                "site"
            ]
            == site
        ]

        counts[
            site
        ] = {
            "all_w3_episodes":
                int(
                    len(
                        all_site
                    )
                ),

            "all_w3_patients":
                int(
                    all_site[
                        "patient_id"
                    ].nunique()
                ),

            "non_progressive_landmarks":
                int(
                    len(
                        primary_site
                    )
                ),

            "non_progressive_patients":
                int(
                    primary_site[
                        "patient_id"
                    ].nunique()
                ),

            "progressive_episodes":
                int(
                    (
                        all_site[
                            "scan_state"
                        ]
                        == "PROGRESSIVE"
                    ).sum()
                ),

            "indeterminate_episodes":
                int(
                    (
                        all_site[
                            "scan_state"
                        ]
                        == "INDETERMINATE"
                    ).sum()
                ),

            "unobserved_state_episodes":
                int(
                    (
                        all_site[
                            "scan_state"
                        ]
                        == "UNOBSERVED"
                    ).sum()
                ),
        }

    return (
        external,
        primary,
        counts,
    )


###############################################################################
# External genomic availability candidates
###############################################################################


def build_external_sample_table(
    frames: dict[
        str,
        pd.DataFrame,
    ],
    site_map: dict[
        str,
        str,
    ],
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
]:

    sample = frames[
        "clinical_sample"
    ].copy()

    patient_col = find_col(
        sample.columns,
        [
            "PATIENT_ID",
        ],
        required=True,
    )

    sample_col = find_col(
        sample.columns,
        [
            "SAMPLE_ID",
        ],
        required=True,
    )

    panel_col = find_col(
        sample.columns,
        [
            "GENE_PANEL",
            "GENE_PANEL_ID",
            "SEQ_ASSAY_ID",
            "ASSAY_ID",
            "PANEL_ID",
        ],
        required=False,
    )

    clinical_date_col = find_col(
        sample.columns,
        [
            "CPT_SEQ_DATE",
            "SEQ_DATE",
            "SEQUENCING_DATE",
        ],
        required=False,
    )

    result = pd.DataFrame(
        {
            "patient_id":
                sample[
                    patient_col
                ]
                .astype(str)
                .str.strip(),

            "sample_id":
                sample[
                    sample_col
                ]
                .astype(str)
                .str.strip(),
        }
    )

    result[
        "site"
    ] = result[
        "patient_id"
    ].map(
        site_map
    )

    result = result[
        result[
            "site"
        ].isin(
            EXTERNAL_SITES
        )
    ].copy()

    result[
        "panel"
    ] = (
        sample.loc[
            result.index,
            panel_col,
        ].astype(str)
        .str.strip()
        if panel_col
        else ""
    )

    if clinical_date_col:

        result[
            "clinical_sample_availability_day"
        ] = pd.to_numeric(
            sample.loc[
                result.index,
                clinical_date_col,
            ],
            errors="coerce",
        )

    else:

        result[
            "clinical_sample_availability_day"
        ] = np.nan

    sequencing = frames[
        "sequencing"
    ].copy()

    seq_patient_col = find_col(
        sequencing.columns,
        [
            "PATIENT_ID",
        ],
        required=True,
    )

    seq_sample_col = find_col(
        sequencing.columns,
        [
            "SAMPLE_ID",
        ],
        required=False,
    )

    seq_day_col = find_col(
        sequencing.columns,
        [
            "START_DATE",
            "SEQ_DATE",
            "SEQUENCING_DATE",
        ],
        required=False,
    )

    sequencing_audit = {
        "sample_column":
            seq_sample_col,

        "date_column":
            seq_day_col,
    }

    if (
        seq_sample_col
        and seq_day_col
    ):

        seq = pd.DataFrame(
            {
                "patient_id":
                    sequencing[
                        seq_patient_col
                    ]
                    .astype(str)
                    .str.strip(),

                "sample_id":
                    sequencing[
                        seq_sample_col
                    ]
                    .astype(str)
                    .str.strip(),

                "day":
                    pd.to_numeric(
                        sequencing[
                            seq_day_col
                        ],
                        errors="coerce",
                    ),
            }
        )

        seq = seq[
            seq[
                "sample_id"
            ].ne("")
            & seq[
                "day"
            ].notna()
        ].copy()

        group = (
            seq.groupby(
                [
                    "patient_id",
                    "sample_id",
                ],
                observed=True,
            )[
                "day"
            ]
            .agg(
                [
                    "min",
                    "max",
                    "nunique",
                ]
            )
            .reset_index()
        )

        group[
            "timeline_sequencing_availability_day"
        ] = np.where(
            group[
                "nunique"
            ]
            == 1,
            group[
                "min"
            ],
            np.nan,
        )

        result = result.merge(
            group[
                [
                    "patient_id",
                    "sample_id",
                    "timeline_sequencing_availability_day",
                    "nunique",
                ]
            ].rename(
                columns={
                    "nunique":
                        "timeline_day_nunique",
                }
            ),
            on=[
                "patient_id",
                "sample_id",
            ],
            how="left",
            validate="one_to_one",
        )

    else:

        result[
            "timeline_sequencing_availability_day"
        ] = np.nan

        result[
            "timeline_day_nunique"
        ] = np.nan

    both = result[
        result[
            "clinical_sample_availability_day"
        ].notna()
        & result[
            "timeline_sequencing_availability_day"
        ].notna()
    ].copy()

    if len(
        both
    ):

        difference = (
            both[
                "timeline_sequencing_availability_day"
            ]
            - both[
                "clinical_sample_availability_day"
            ]
        )

        date_comparison = {
            "samples_with_both":
                int(
                    len(
                        both
                    )
                ),

            "exact_day_fraction":
                float(
                    (
                        difference
                        == 0
                    ).mean()
                ),

            "median_timeline_minus_clinical":
                float(
                    difference.median()
                ),

            "min_difference":
                float(
                    difference.min()
                ),

            "max_difference":
                float(
                    difference.max()
                ),
        }

    else:

        date_comparison = {
            "samples_with_both":
                0,
        }

    audit = {
        "clinical_sample_date_column":
            clinical_date_col,

        "sequencing":
            sequencing_audit,

        "date_comparison":
            date_comparison,

        "site_counts": {
            site: {
                "samples":
                    int(
                        result.loc[
                            result[
                                "site"
                            ]
                            == site,
                            "sample_id",
                        ].nunique()
                    ),

                "patients":
                    int(
                        result.loc[
                            result[
                                "site"
                            ]
                            == site,
                            "patient_id",
                        ].nunique()
                    ),

                "clinical_date_samples":
                    int(
                        result.loc[
                            result[
                                "site"
                            ]
                            == site,
                            "clinical_sample_availability_day",
                        ].notna().sum()
                    ),

                "timeline_date_samples":
                    int(
                        result.loc[
                            result[
                                "site"
                            ]
                            == site,
                            "timeline_sequencing_availability_day",
                        ].notna().sum()
                    ),
            }
            for site in EXTERNAL_SITES
        },
    }

    return (
        result.reset_index(
            drop=True
        ),
        audit,
    )


###############################################################################
# Frozen CKPT3 embedding bridge
###############################################################################


def load_genie_embedding_bridge(
    repo: Path,
    external_samples: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    dict[str, Any],
]:

    index = pd.read_parquet(
        repo
        / "artifacts"
        / "checkpoint3"
        / "genie_embedding_index.parquet"
    )

    embeddings = np.load(
        repo
        / "artifacts"
        / "checkpoint3"
        / "genie_sample_embeddings_f16.npy",
        mmap_mode="r",
    )

    if (
        embeddings.ndim
        != 2
        or embeddings.shape[
            1
        ]
        != EXPECTED_DIMENSIONS[
            "genomic"
        ]
    ):

        raise RuntimeError(
            "Frozen CKPT3 embedding dimension mismatch."
        )

    sample_col = find_col(
        index.columns,
        [
            "SAMPLE_ID",
            "sample_id",
            "Tumor_Sample_Barcode",
        ],
        required=True,
    )

    row_col = find_col(
        index.columns,
        [
            "embedding_row",
            "row",
            "row_index",
            "embedding_index",
            "index",
        ],
        required=False,
    )

    bridge = index.copy()

    bridge[
        "_sample_key"
    ] = bridge[
        sample_col
    ].map(
        clean
    )

    if bridge[
        "_sample_key"
    ].duplicated().any():

        duplicates = (
            bridge.loc[
                bridge[
                    "_sample_key"
                ].duplicated(
                    keep=False
                ),
                sample_col,
            ]
            .head(
                20
            )
            .tolist()
        )

        raise RuntimeError(
            "GENIE embedding index has duplicate sample IDs: "
            f"{duplicates}"
        )

    if row_col:

        bridge[
            "_embedding_row"
        ] = pd.to_numeric(
            bridge[
                row_col
            ],
            errors="coerce",
        )

        if bridge[
            "_embedding_row"
        ].isna().any():

            raise RuntimeError(
                "Embedding row column contains nonnumeric values."
            )

        bridge[
            "_embedding_row"
        ] = bridge[
            "_embedding_row"
        ].astype(
            int
        )

    else:

        if len(
            bridge
        ) != embeddings.shape[
            0
        ]:

            raise RuntimeError(
                "No embedding-row column and index length "
                "does not equal embedding array length."
            )

        bridge[
            "_embedding_row"
        ] = np.arange(
            len(
                bridge
            ),
            dtype=int,
        )

    if (
        bridge[
            "_embedding_row"
        ].min()
        < 0
        or bridge[
            "_embedding_row"
        ].max()
        >= embeddings.shape[
            0
        ]
    ):

        raise RuntimeError(
            "GENIE embedding rows out of bounds."
        )

    external = external_samples.copy()

    external[
        "_sample_key"
    ] = external[
        "sample_id"
    ].map(
        clean
    )

    mapped = external.merge(
        bridge[
            [
                "_sample_key",
                "_embedding_row",
            ]
        ],
        on="_sample_key",
        how="left",
        validate="one_to_one",
    )

    audit = {
        "genie_index_rows":
            int(
                len(
                    bridge
                )
            ),

        "embedding_shape":
            list(
                embeddings.shape
            ),

        "index_sample_column":
            sample_col,

        "index_row_column":
            (
                row_col
                if row_col
                else "ROW_ORDER"
            ),

        "site_mapping": {},
    }

    for site in EXTERNAL_SITES:

        current = mapped[
            mapped[
                "site"
            ]
            == site
        ]

        matched = current[
            "_embedding_row"
        ].notna()

        audit[
            "site_mapping"
        ][
            site
        ] = {
            "external_samples":
                int(
                    current[
                        "sample_id"
                    ].nunique()
                ),

            "mapped_samples":
                int(
                    matched.sum()
                ),

            "mapping_fraction":
                float(
                    matched.mean()
                )
                if len(
                    current
                )
                else 0.0,

            "external_patients":
                int(
                    current[
                        "patient_id"
                    ].nunique()
                ),

            "patients_with_mapped_embedding":
                int(
                    current.loc[
                        matched,
                        "patient_id",
                    ].nunique()
                ),
        }

    return (
        mapped,
        embeddings,
        audit,
    )


###############################################################################
# Landmark -> latest available genomic sample
###############################################################################


def materialize_landmark_genomics(
    landmarks: pd.DataFrame,
    samples: pd.DataFrame,
    embeddings: np.ndarray,
    availability_column: str,
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    dict[str, Any],
]:

    assignments = landmarks[
        [
            "external_landmark_row",
            "site",
            "patient_id",
            "bpc_episode_id",
            "landmark_day",
            "scan_state",
        ]
    ].copy()

    assignments[
        "availability_source"
    ] = availability_column

    assignments[
        "selected_sample_id"
    ] = ""

    assignments[
        "selected_availability_day"
    ] = np.nan

    assignments[
        "embedding_row"
    ] = -1

    output = np.zeros(
        (
            len(
                landmarks
            ),
            EXPECTED_DIMENSIONS[
                "genomic"
            ],
        ),
        dtype=np.float16,
    )

    by_patient = {}

    candidate = samples[
        samples[
            availability_column
        ].notna()
        & samples[
            "_embedding_row"
        ].notna()
    ].copy()

    candidate[
        availability_column
    ] = pd.to_numeric(
        candidate[
            availability_column
        ],
        errors="coerce",
    )

    candidate[
        "_embedding_row"
    ] = candidate[
        "_embedding_row"
    ].astype(
        int
    )

    for patient, group in candidate.groupby(
        "patient_id",
        observed=True,
    ):

        group = (
            group
            .sort_values(
                [
                    availability_column,
                    "sample_id",
                ],
                kind="mergesort",
            )
            .reset_index(
                drop=True
            )
        )

        by_patient[
            patient
        ] = group

    for row_index, row in assignments.iterrows():

        patient = row[
            "patient_id"
        ]

        landmark_day = float(
            row[
                "landmark_day"
            ]
        )

        group = by_patient.get(
            patient
        )

        if (
            group is None
            or group.empty
        ):

            continue

        days = group[
            availability_column
        ].to_numpy(
            dtype=float
        )

        #######################################################################
        # STRICT availability_day < landmark_day.
        #######################################################################

        selected_index = (
            np.searchsorted(
                days,
                landmark_day,
                side="left",
            )
            - 1
        )

        if selected_index < 0:
            continue

        selected = group.iloc[
            selected_index
        ]

        embedding_row = int(
            selected[
                "_embedding_row"
            ]
        )

        output[
            row_index
        ] = embeddings[
            embedding_row
        ]

        assignments.at[
            row_index,
            "selected_sample_id",
        ] = selected[
            "sample_id"
        ]

        assignments.at[
            row_index,
            "selected_availability_day",
        ] = float(
            selected[
                availability_column
            ]
        )

        assignments.at[
            row_index,
            "embedding_row",
        ] = embedding_row

    assignments[
        "genomic_available"
    ] = (
        assignments[
            "embedding_row"
        ]
        >= 0
    )

    assignments[
        "genomic_age_days"
    ] = np.where(
        assignments[
            "genomic_available"
        ],
        assignments[
            "landmark_day"
        ]
        - assignments[
            "selected_availability_day"
        ],
        np.nan,
    )

    if (
        assignments.loc[
            assignments[
                "genomic_available"
            ],
            "genomic_age_days",
        ]
        <= 0
    ).any():

        raise RuntimeError(
            "Same-day or future genomic availability leaked."
        )

    audit = {
        "availability_source":
            availability_column,

        "rows":
            int(
                len(
                    assignments
                )
            ),

        "available_rows":
            int(
                assignments[
                    "genomic_available"
                ].sum()
            ),

        "availability_fraction":
            float(
                assignments[
                    "genomic_available"
                ].mean()
            ),

        "site": {},
    }

    for site in EXTERNAL_SITES:

        current = assignments[
            assignments[
                "site"
            ]
            == site
        ]

        audit[
            "site"
        ][
            site
        ] = {
            "landmarks":
                int(
                    len(
                        current
                    )
                ),

            "available":
                int(
                    current[
                        "genomic_available"
                    ].sum()
                ),

            "fraction":
                float(
                    current[
                        "genomic_available"
                    ].mean()
                )
                if len(
                    current
                )
                else 0.0,

            "patients":
                int(
                    current[
                        "patient_id"
                    ].nunique()
                ),

            "patients_with_genomics":
                int(
                    current.loc[
                        current[
                            "genomic_available"
                        ],
                        "patient_id",
                    ].nunique()
                ),
        }

    return (
        assignments,
        output,
        audit,
    )


###############################################################################
# Canonical / prepared schema inventory
###############################################################################


def parquet_schema_inventory(
    repo: Path,
) -> list[
    dict[str, Any]
]:

    candidates = []

    roots = [
        repo
        / "artifacts"
        / "checkpoint1"
        / "canonical",

        repo
        / "artifacts"
        / "checkpoint4"
        / "prepared",

        repo
        / "artifacts"
        / "checkpoint5"
        / "prepared",
    ]

    for root in roots:

        if not root.exists():
            continue

        for path in sorted(
            root.glob(
                "*.parquet"
            )
        ):

            try:

                frame = pd.read_parquet(
                    path
                )

            except Exception as exc:

                candidates.append(
                    {
                        "path":
                            str(
                                path.relative_to(
                                    repo
                                )
                            ),

                        "status":
                            "READ_FAILED",

                        "error":
                            (
                                type(
                                    exc
                                ).__name__
                                + ": "
                                + str(
                                    exc
                                )
                            ),
                    }
                )

                continue

            candidates.append(
                {
                    "path":
                        str(
                            path.relative_to(
                                repo
                            )
                        ),

                    "status":
                        "OK",

                    "rows":
                        int(
                            len(
                                frame
                            )
                        ),

                    "columns":
                        list(
                            frame.columns
                        ),
                }
            )

    return candidates


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
        "--bpc-root",
        default=(
            "data/external_sources/"
            "bpc_brca_1_0_public"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "artifacts/checkpoint7a4a"
        ),
    )

    args = parser.parse_args()

    repo = Path(
        args.repo_root
    ).resolve()

    bpc_root = (
        repo
        / args.bpc_root
    ).resolve()

    out = (
        repo
        / args.output_dir
    ).resolve()

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    ###########################################################################
    # Confirm upstream scientific freeze.
    ###########################################################################

    bridge = json.loads(
        (
            repo
            / "artifacts"
            / "checkpoint7a3g"
            / "bridge_confirmation.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    if (
        bridge[
            "status"
        ]
        != "PASS_CLOCK_AND_EXACT_FEATURE_BRIDGE"
    ):

        raise RuntimeError(
            "CKPT7A3G is not PASS."
        )

    if bridge[
        "external_outcomes_opened"
    ]:

        raise RuntimeError(
            "External outcomes already reported as opened."
        )

    ###########################################################################
    # Freeze predictor-only W3 external landmarks.
    ###########################################################################

    (
        all_external_episodes,
        primary_landmarks,
        landmark_counts,
    ) = build_external_landmarks(
        repo
    )

    atomic_parquet(
        out
        / "external_w3_episodes.parquet",
        all_external_episodes,
    )

    atomic_parquet(
        out
        / "external_primary_landmarks.parquet",
        primary_landmarks,
    )

    ###########################################################################
    # Predictor-only BPC sample metadata.
    ###########################################################################

    (
        guard,
        frames,
        site_map,
    ) = load_external_predictors(
        repo,
        bpc_root,
    )

    (
        external_samples,
        sample_audit,
    ) = build_external_sample_table(
        frames,
        site_map,
    )

    ###########################################################################
    # Exact frozen CKPT3 sample-embedding bridge.
    ###########################################################################

    (
        external_samples,
        genie_embeddings,
        genie_audit,
    ) = load_genie_embedding_bridge(
        repo,
        external_samples,
    )

    atomic_parquet(
        out
        / "external_genomic_samples.parquet",
        external_samples,
    )

    ###########################################################################
    # Materialize BOTH candidate availability interpretations.
    #
    # Do not choose between them here.
    ###########################################################################

    genomic_variants = {}

    for availability_column in (
        "clinical_sample_availability_day",
        "timeline_sequencing_availability_day",
    ):

        if (
            availability_column
            not in external_samples.columns
            or external_samples[
                availability_column
            ].notna().sum()
            == 0
        ):

            genomic_variants[
                availability_column
            ] = {
                "status":
                    "NO_AVAILABLE_DATES",
            }

            continue

        (
            assignments,
            tensor,
            audit,
        ) = materialize_landmark_genomics(
            primary_landmarks,
            external_samples,
            genie_embeddings,
            availability_column,
        )

        stem = (
            "clinical"
            if availability_column.startswith(
                "clinical_"
            )
            else "timeline"
        )

        assignment_path = (
            out
            / (
                "genomic_assignments_"
                + stem
                + ".parquet"
            )
        )

        tensor_path = (
            out
            / (
                "tumor_embeddings_"
                + stem
                + "_f16.npy"
            )
        )

        atomic_parquet(
            assignment_path,
            assignments,
        )

        atomic_npy(
            tensor_path,
            tensor,
        )

        genomic_variants[
            availability_column
        ] = {
            "status":
                "MATERIALIZED",

            "assignment_file":
                str(
                    assignment_path.relative_to(
                        repo
                    )
                ),

            "tensor_file":
                str(
                    tensor_path.relative_to(
                        repo
                    )
                ),

            "tensor_sha256":
                sha256_file(
                    tensor_path
                ),

            "audit":
                audit,
        }

    ###########################################################################
    # Compare landmark assignments under the two date sources.
    ###########################################################################

    assignment_comparison = {}

    clinical_file = (
        out
        / "genomic_assignments_clinical.parquet"
    )

    timeline_file = (
        out
        / "genomic_assignments_timeline.parquet"
    )

    if (
        clinical_file.exists()
        and timeline_file.exists()
    ):

        clinical = pd.read_parquet(
            clinical_file
        )

        timeline = pd.read_parquet(
            timeline_file
        )

        comparison = clinical[
            [
                "external_landmark_row",
                "selected_sample_id",
                "genomic_available",
            ]
        ].merge(
            timeline[
                [
                    "external_landmark_row",
                    "selected_sample_id",
                    "genomic_available",
                ]
            ],
            on="external_landmark_row",
            suffixes=(
                "_clinical",
                "_timeline",
            ),
            validate="one_to_one",
        )

        assignment_comparison = {
            "rows":
                int(
                    len(
                        comparison
                    )
                ),

            "availability_equal_fraction":
                float(
                    (
                        comparison[
                            "genomic_available_clinical"
                        ]
                        == comparison[
                            "genomic_available_timeline"
                        ]
                    ).mean()
                ),

            "selected_sample_equal_fraction":
                float(
                    (
                        comparison[
                            "selected_sample_id_clinical"
                        ]
                        == comparison[
                            "selected_sample_id_timeline"
                        ]
                    ).mean()
                ),

            "different_selected_sample_rows":
                int(
                    (
                        comparison[
                            "selected_sample_id_clinical"
                        ]
                        != comparison[
                            "selected_sample_id_timeline"
                        ]
                    ).sum()
                ),
        }

    ###########################################################################
    # Exact CKPT5 feature schema.
    ###########################################################################

    feature_schema = json.loads(
        (
            repo
            / "artifacts"
            / "checkpoint5"
            / "prepared"
            / "feature_schema.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    (
        scan_schema_path,
        scan_feature_names,
    ) = resolve_feature_names(
        feature_schema,
        expected_length=18,
        preferred_tokens=(
            "current",
            "scan",
        ),
    )

    (
        context_schema_path,
        context_feature_names,
    ) = resolve_feature_names(
        feature_schema,
        expected_length=6,
        preferred_tokens=(
            "context",
            "structured",
        ),
    )

    ###########################################################################
    # Exact source interface inventories.
    ###########################################################################

    ckpt3_functions = function_inventory(
        repo
        / "scripts"
        / "dynamic_scan"
        / "ckpt3_genie_pretrain.py",
        (
            "embedding",
            "coverage",
            "encode",
            "panel",
            "teacher",
            "gene_space",
        ),
    )

    ckpt4_functions = function_inventory(
        repo
        / "scripts"
        / "dynamic_scan"
        / "ckpt4_temporal_pretrain.py",
        (
            "token",
            "sequence",
            "event",
            "day_end",
            "scan",
            "embedding",
            "encode",
            "cache",
            "pre",
            "post",
        ),
    )

    ckpt5_functions = function_inventory(
        repo
        / "scripts"
        / "dynamic_scan"
        / "ckpt5_supervised_dynamic_model.py",
        (
            "context",
            "availability_day",
            "genomic",
            "cpt_seq_date",
            "seq_date",
            "scan_features",
            "current_scan",
            "dynamic_scan_model",
            "predict",
            "forward",
        ),
    )

    ###########################################################################
    # Checkpoint structure inventory.
    ###########################################################################

    checkpoint_files = {
        "ckpt3_genomic_encoder":
            repo
            / "artifacts"
            / "checkpoint3"
            / "genomic_encoder.pt",

        "ckpt4_temporal_encoder":
            repo
            / "artifacts"
            / "checkpoint4"
            / "temporal_encoder.pt",

        "ckpt5_dynamic_model":
            repo
            / "artifacts"
            / "checkpoint5"
            / "dynamic_scan_model.pt",

        "ckpt6b_bounded_candidate":
            repo
            / "artifacts"
            / "checkpoint6b"
            / "bounded_dynamic_scan_candidate.pt",
    }

    checkpoint_report = {
        name:
            checkpoint_inventory(
                path
            )
        for name, path
        in checkpoint_files.items()
    }

    ###########################################################################
    # Prepared/canonical schemas.
    ###########################################################################

    parquet_schemas = parquet_schema_inventory(
        repo
    )

    ###########################################################################
    # Interface readiness.
    ###########################################################################

    primary_population_ok = all(
        landmark_counts[
            site
        ][
            "non_progressive_landmarks"
        ]
        > 0
        for site in EXTERNAL_SITES
    )

    feature_schema_ok = (
        len(
            scan_feature_names
        )
        == EXPECTED_DIMENSIONS[
            "scan"
        ]
        and len(
            context_feature_names
        )
        == EXPECTED_DIMENSIONS[
            "context"
        ]
    )

    genomic_mapping_fraction = {
        site:
            genie_audit[
                "site_mapping"
            ][
                site
            ][
                "mapping_fraction"
            ]
        for site in EXTERNAL_SITES
    }

    genomic_identity_ok = all(
        genomic_mapping_fraction[
            site
        ]
        >= 0.90
        for site in EXTERNAL_SITES
    )

    temporal_interface_found = bool(
        ckpt4_functions
    )

    model_interface_found = bool(
        ckpt5_functions
    )

    if not primary_population_ok:

        status = (
            "NEEDS_EXTERNAL_LANDMARK_RESOLUTION"
        )

    elif not genomic_identity_ok:

        status = (
            "NEEDS_EXTERNAL_GENOMIC_IDENTITY_RESOLUTION"
        )

    elif not feature_schema_ok:

        status = (
            "NEEDS_FROZEN_FEATURE_SCHEMA_RESOLUTION"
        )

    elif not (
        temporal_interface_found
        and model_interface_found
    ):

        status = (
            "NEEDS_FROZEN_ENCODER_INTERFACE_RESOLUTION"
        )

    else:

        status = (
            "READY_FOR_CKPT7A4B_EXTERNAL_TENSOR_BUILD"
        )

    ###########################################################################
    # No outcome access.
    ###########################################################################

    for item in guard.access_log:

        relative = str(
            item[
                "relative_path"
            ]
        ).lower()

        if any(
            token in relative
            for token in (
                "survival",
                "pfs",
                "censor",
                "death",
            )
        ):

            raise RuntimeError(
                "External outcome table was accessed."
            )

    ###########################################################################
    # Persistent report.
    ###########################################################################

    report = {
        "status":
            status,

        "external_outcomes_opened":
            False,

        "external_outcome_distributions_inspected":
            False,

        "ckpt7a3g_status":
            bridge[
                "status"
            ],

        "landmarks":
            landmark_counts,

        "primary_landmark_rows":
            int(
                len(
                    primary_landmarks
                )
            ),

        "primary_landmark_patients":
            {
                site:
                    int(
                        primary_landmarks.loc[
                            primary_landmarks[
                                "site"
                            ]
                            == site,
                            "patient_id",
                        ].nunique()
                    )
                for site in EXTERNAL_SITES
            },

        "external_sample_audit":
            sample_audit,

        "genie_embedding_bridge":
            genie_audit,

        "genomic_variants":
            genomic_variants,

        "genomic_assignment_comparison":
            assignment_comparison,

        "frozen_feature_schema": {
            "scan_schema_path":
                scan_schema_path,

            "scan_feature_names":
                scan_feature_names,

            "context_schema_path":
                context_schema_path,

            "context_feature_names":
                context_feature_names,
        },

        "function_inventory": {
            "ckpt3":
                ckpt3_functions,

            "ckpt4":
                ckpt4_functions,

            "ckpt5":
                ckpt5_functions,
        },

        "checkpoint_inventory":
            checkpoint_report,

        "parquet_schema_inventory":
            parquet_schemas,

        "policy": {
            "w3_primary":
                True,

            "primary_current_scan_state":
                "NON_PROGRESSIVE",

            "landmark_time":
                "episode_end_day",

            "genomic_availability_rule":
                "availability_day < landmark_day",

            "unavailable_genomic_embedding":
                "exact_zero_128d",

            "same_day_genomics_excluded":
                True,

            "msk_clock_offsets_applied_to_dfci_vicc":
                False,

            "external_outcomes_accessed":
                False,

            "alpha":
                0.75,
        },

        "next_action":
            (
                "Use the exact CKPT4/CKPT5 interfaces captured here "
                "to build DFCI/VICC temporal PRE/POST states, exact "
                "18-D scan features and 6-D context, run the frozen "
                "candidate, then hash/freeze all predictions before "
                "opening outcomes."
                if status
                == "READY_FOR_CKPT7A4B_EXTERNAL_TENSOR_BUILD"
                else
                "Resolve the reported predictor-interface mismatch "
                "without opening DFCI/VICC outcomes."
            ),
    }

    atomic_json(
        out
        / "interface_report.json",
        report,
    )

    ###########################################################################
    # Compact exact-interface digest.
    ###########################################################################

    digest_lines = [
        "# CKPT7A4A exact frozen-interface digest",
        "",
        f"Status: {status}",
        "",
        "External outcomes opened: NO",
        "",
        "## Frozen CKPT5 scan features",
        "",
        repr(
            scan_feature_names
        ),
        "",
        "## Frozen CKPT5 context features",
        "",
        repr(
            context_feature_names
        ),
        "",
        "## CKPT5 functions relevant to genomic availability/context/model",
        "",
    ]

    for record in ckpt5_functions[
        :12
    ]:

        digest_lines.append(
            json.dumps(
                record,
                indent=2,
            )
        )

    digest_lines.extend(
        [
            "",
            "## CKPT4 temporal functions",
            "",
        ]
    )

    for record in ckpt4_functions[
        :15
    ]:

        digest_lines.append(
            json.dumps(
                record,
                indent=2,
            )
        )

    atomic_text(
        out
        / "interface_digest.txt",
        "\n".join(
            digest_lines
        )
        + "\n",
    )

    ###########################################################################
    # Handoff.
    ###########################################################################

    handoff = {
        "checkpoint":
            "7A4A",

        "name":
            "external_landmark_genomic_materialization_and_interface_freeze",

        "status":
            status,

        "external_outcomes_opened":
            False,

        "primary_landmark_rows":
            int(
                len(
                    primary_landmarks
                )
            ),

        "landmark_counts":
            landmark_counts,

        "genie_embedding_bridge":
            genie_audit[
                "site_mapping"
            ],

        "genomic_assignment_comparison":
            assignment_comparison,

        "scan_feature_names":
            scan_feature_names,

        "context_feature_names":
            context_feature_names,

        "next_action":
            report[
                "next_action"
            ],
    }

    atomic_json(
        repo
        / "artifacts"
        / "handoff"
        / "checkpoint_07A4A.json",
        handoff,
    )

    ###########################################################################
    # Terminal packet.
    ###########################################################################

    print("")
    print(
        "========== CKPT7A4A SUMMARY =========="
    )

    print(
        f"status={status}"
    )

    print(
        "external_outcomes_opened=False"
    )

    print(
        "landmark_counts="
        f"{landmark_counts}"
    )

    print(
        "primary_landmark_rows="
        f"{len(primary_landmarks)}"
    )

    print(
        "genie_embedding_bridge="
        f"{genie_audit['site_mapping']}"
    )

    print(
        "sample_date_audit="
        f"{sample_audit}"
    )

    print(
        "genomic_assignment_comparison="
        f"{assignment_comparison}"
    )

    print(
        "scan_feature_schema="
        f"{scan_feature_names}"
    )

    print(
        "context_feature_schema="
        f"{context_feature_names}"
    )

    print(
        "handoff="
        "artifacts/handoff/checkpoint_07A4A.json"
    )

    print(
        "========== CKPT7A4A SUMMARY END =========="
    )

    print("")
    print(
        "========== CKPT7A4A DECISION PACKET =========="
    )

    print("")
    print(
        "----- GENOMIC VARIANTS -----"
    )

    print(
        json.dumps(
            genomic_variants,
            indent=2,
            default=str,
        )
    )

    print(
        "----- GENOMIC VARIANTS END -----"
    )

    print("")
    print(
        "----- CKPT5 EXACT INTERFACE -----"
    )

    for record in ckpt5_functions[
        :12
    ]:

        print(
            json.dumps(
                record,
                indent=2,
            )
        )

    print(
        "----- CKPT5 EXACT INTERFACE END -----"
    )

    print("")
    print(
        "----- CKPT4 EXACT INTERFACE -----"
    )

    for record in ckpt4_functions[
        :15
    ]:

        print(
            json.dumps(
                record,
                indent=2,
            )
        )

    print(
        "----- CKPT4 EXACT INTERFACE END -----"
    )

    print("")
    print(
        "----- CHECKPOINT STRUCTURES -----"
    )

    print(
        json.dumps(
            checkpoint_report,
            indent=2,
            default=str,
        )
    )

    print(
        "----- CHECKPOINT STRUCTURES END -----"
    )

    print("")
    print(
        "next_action="
        f"{report['next_action']}"
    )

    print(
        "========== CKPT7A4A DECISION PACKET END =========="
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
