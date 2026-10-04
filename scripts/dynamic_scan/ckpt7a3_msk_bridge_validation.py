#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


WINDOW_DAYS = 3

STATE_CROSSWALK = {
    "Progressing/Worsening/Enlarging":
        "PROGRESSIVE",

    "Stable/No change":
        "NON_PROGRESSIVE",

    "Improving/Responding":
        "NON_PROGRESSIVE",

    "Not stated/Indeterminate":
        "INDETERMINATE",

    "Mixed":
        "INDETERMINATE",
}

QUARANTINED = {
    "cBioPortal_files/data_clinical_supp_survival.txt",
    "cBioPortal_files/data_clinical_supp_survival_treatment.txt",
    "clinical_data/cancer_level_dataset_index.csv",
    "clinical_data/cancer_panel_test_level_dataset.csv",
    "clinical_data/patient_level_dataset.csv",
    "clinical_data/regimen_cancer_level_dataset.csv",
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


def find_col(
    columns: Iterable[str],
    candidates: Iterable[str],
    *,
    required: bool = True,
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
            default=lambda x:
                int(x)
                if isinstance(
                    x,
                    np.integer,
                )
                else float(x)
                if isinstance(
                    x,
                    np.floating,
                )
                else bool(x)
                if isinstance(
                    x,
                    np.bool_,
                )
                else str(x),
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
        path.suffix + ".tmp"
    )

    frame.to_parquet(
        tmp,
        index=False,
    )

    check = pd.read_parquet(
        tmp
    )

    if len(
        check
    ) != len(
        frame
    ):

        raise RuntimeError(
            f"Parquet verification failed: {path}"
        )

    tmp.replace(
        path
    )


def import_ckpt7a2(
    repo: Path,
):

    path = (
        repo
        / "scripts"
        / "dynamic_scan"
        / "ckpt7a2_external_semantic_audit.py"
    )

    spec = (
        importlib.util
        .spec_from_file_location(
            "ckpt7a2_frozen",
            path,
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):

        raise RuntimeError(
            "Unable to import CKPT7A2."
        )

    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    sys.modules[
        spec.name
    ] = module

    spec.loader.exec_module(
        module
    )

    return module


###############################################################################
# BPC source loading with outcome guard
###############################################################################


def load_bpc_sources(
    repo: Path,
    bpc_root: Path,
) -> tuple[
    Any,
    dict[str, pd.DataFrame],
    dict[str, str],
]:

    ckpt7a2 = import_ckpt7a2(
        repo
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
            "cBioPortal_files/data_clinical_patient.txt",

        "imaging":
            "cBioPortal_files/data_timeline_imaging.txt",

        "treatment":
            "cBioPortal_files/data_timeline_treatment.txt",

        "medonc":
            "cBioPortal_files/data_timeline_medonc.txt",

        "pathology":
            "cBioPortal_files/data_timeline_pathology.txt",

        "diagnosis":
            "cBioPortal_files/data_timeline_cancer_diagnosis.txt",

        "lab":
            "cBioPortal_files/data_timeline_labtest.txt",

        "sequencing":
            "cBioPortal_files/data_timeline_sequencing.txt",

        "clinical_sample":
            "cBioPortal_files/data_clinical_sample.txt",
    }

    frames = {}

    for name, path in relative.items():

        frames[
            name
        ] = guard.read_predictor(
            path
        )

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

    accessed = {
        item[
            "relative_path"
        ]
        for item
        in guard.access_log
    }

    violation = (
        accessed
        & QUARANTINED
    )

    if violation:

        raise RuntimeError(
            "Outcome quarantine violation: "
            f"{sorted(violation)}"
        )

    return (
        ckpt7a2,
        frames,
        site_map,
    )


###############################################################################
# Imaging episode construction
###############################################################################


def map_scan_state(
    value: Any,
) -> str | None:

    text = str(
        value
    ).strip()

    # Missing curated response is genuinely unobserved.
    # It must not be collapsed into observed INDETERMINATE.
    if not text:

        return None

    if text not in STATE_CROSSWALK:

        raise RuntimeError(
            "Unexpected nonblank CURATED_CANCER_STATUS value: "
            f"{text!r}"
        )

    return STATE_CROSSWALK[
        text
    ]


def aggregate_state(
    values: Iterable[str | None],
) -> str:

    states = {
        str(value)
        for value in values
        if (
            value is not None
            and str(value).strip()
            and str(value) != "<NA>"
            and str(value).lower() != "nan"
        )
    }

    # No curated response anywhere in the episode.
    # Preserve this as explicit missingness.
    if not states:

        return "UNOBSERVED"

    if "PROGRESSIVE" in states:
        return "PROGRESSIVE"

    if "INDETERMINATE" in states:
        return "INDETERMINATE"

    if (
        states
        == {
            "NON_PROGRESSIVE",
        }
    ):
        return "NON_PROGRESSIVE"

    raise RuntimeError(
        f"Unable to aggregate observed states: {states}"
    )


def build_bpc_w3(
    imaging: pd.DataFrame,
    site_map: dict[str, str],
) -> pd.DataFrame:

    patient_col = find_col(
        imaging.columns,
        [
            "PATIENT_ID",
            "patient_id",
        ],
    )

    start_col = find_col(
        imaging.columns,
        [
            "START_DATE",
            "start_date",
        ],
    )

    status_col = find_col(
        imaging.columns,
        [
            "CURATED_CANCER_STATUS",
            "curated_cancer_status",
        ],
    )

    scan_type_col = find_col(
        imaging.columns,
        [
            "IMAGE_SCAN_TYPE",
            "image_scan_type",
        ],
        required=False,
    )

    scan_sites_col = find_col(
        imaging.columns,
        [
            "SCAN_SITES",
            "scan_sites",
        ],
        required=False,
    )

    cancer_presence_col = find_col(
        imaging.columns,
        [
            "CANCER_STATUS",
            "cancer_status",
        ],
        required=False,
    )

    work = pd.DataFrame(
        {
            "patient_id":
                imaging[
                    patient_col
                ]
                .astype(str)
                .str.strip(),

            "scan_day":
                pd.to_numeric(
                    imaging[
                        start_col
                    ],
                    errors="coerce",
                ),

            "curated_status":
                imaging[
                    status_col
                ]
                .astype(str)
                .str.strip(),
        }
    )

    work[
        "site"
    ] = work[
        "patient_id"
    ].map(
        site_map
    )

    work[
        "scan_type"
    ] = (
        imaging[
            scan_type_col
        ]
        .astype(str)
        .fillna("")
        if scan_type_col
        else ""
    )

    work[
        "scan_sites"
    ] = (
        imaging[
            scan_sites_col
        ]
        .astype(str)
        .fillna("")
        if scan_sites_col
        else ""
    )

    work[
        "cancer_presence"
    ] = (
        imaging[
            cancer_presence_col
        ]
        .astype(str)
        .fillna("")
        if cancer_presence_col
        else ""
    )

    work = work[
        work[
            "scan_day"
        ].notna()
        & work[
            "site"
        ].notna()
    ].copy()

    work[
        "scan_day"
    ] = work[
        "scan_day"
    ].astype(
        float
    )

    work[
        "row_state"
    ] = work[
        "curated_status"
    ].map(
        map_scan_state
    )

    episode_rows = []

    for (
        patient_id,
        site,
    ), group in work.groupby(
        [
            "patient_id",
            "site",
        ],
        observed=True,
    ):

        group = (
            group
            .sort_values(
                "scan_day",
                kind="mergesort",
            )
            .reset_index(
                drop=True
            )
        )

        current = []

        previous_day = None

        def emit(
            rows,
        ):

            if not rows:
                return

            subset = group.iloc[
                rows
            ]

            episode_rows.append(
                {
                    "patient_id":
                        patient_id,

                    "site":
                        site,

                    "episode_start_day":
                        float(
                            subset[
                                "scan_day"
                            ].min()
                        ),

                    "episode_end_day":
                        float(
                            subset[
                                "scan_day"
                            ].max()
                        ),

                    "scan_state":
                        aggregate_state(
                            subset[
                                "row_state"
                            ]
                        ),

                    "component_rows":
                        int(
                            len(
                                subset
                            )
                        ),

                    "state_observed_component_rows":
                        int(
                            subset[
                                "row_state"
                            ].notna().sum()
                        ),

                    "state_unobserved_component_rows":
                        int(
                            subset[
                                "row_state"
                            ].isna().sum()
                        ),

                    "scan_type_text":
                        " | ".join(
                            sorted(
                                set(
                                    value
                                    for value
                                    in subset[
                                        "scan_type"
                                    ].astype(str)
                                    if value
                                )
                            )
                        ),

                    "scan_sites_text":
                        " | ".join(
                            sorted(
                                set(
                                    value
                                    for value
                                    in subset[
                                        "scan_sites"
                                    ].astype(str)
                                    if value
                                )
                            )
                        ),

                    "cancer_presence_text":
                        " | ".join(
                            sorted(
                                set(
                                    value
                                    for value
                                    in subset[
                                        "cancer_presence"
                                    ].astype(str)
                                    if value
                                )
                            )
                        ),
                }
            )

        for row_index, row in group.iterrows():

            current_day = float(
                row[
                    "scan_day"
                ]
            )

            if previous_day is None:

                current = [
                    row_index
                ]

            elif (
                current_day
                - previous_day
                <= WINDOW_DAYS
            ):

                current.append(
                    row_index
                )

            else:

                emit(
                    current
                )

                current = [
                    row_index
                ]

            previous_day = current_day

        emit(
            current
        )

    episodes = pd.DataFrame(
        episode_rows
    )

    episodes = (
        episodes
        .sort_values(
            [
                "site",
                "patient_id",
                "episode_end_day",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    episodes[
        "bpc_episode_id"
    ] = [
        f"BPC-{site}-{patient}-{int(day)}-{i}"
        for i, (
            site,
            patient,
            day,
        )
        in enumerate(
            zip(
                episodes[
                    "site"
                ],
                episodes[
                    "patient_id"
                ],
                episodes[
                    "episode_end_day"
                ],
            )
        )
    ]

    return episodes


###############################################################################
# Frozen 18-D current-scan feature recreation
###############################################################################


def bool_pattern(
    text: str,
    pattern: str,
) -> float:

    return float(
        bool(
            re.search(
                pattern,
                str(
                    text
                ).upper(),
            )
        )
    )


def bpc_feature_dict(
    row: pd.Series,
) -> dict[str, float]:

    state = str(
        row[
            "scan_state"
        ]
    )

    scan_type = str(
        row[
            "scan_type_text"
        ]
    )

    scan_sites = str(
        row[
            "scan_sites_text"
        ]
    )

    combined = (
        scan_type
        + " | "
        + scan_sites
    )

    features = {
        "state_non_progressive":
            float(
                state
                == "NON_PROGRESSIVE"
            ),

        "state_indeterminate":
            float(
                state
                == "INDETERMINATE"
            ),

        "state_progressive":
            float(
                state
                == "PROGRESSIVE"
            ),

        "modality_ct":
            bool_pattern(
                scan_type,
                r"\bCT\b|COMPUTED TOMOGRAPH",
            ),

        "modality_pet":
            bool_pattern(
                scan_type,
                r"\bPET\b|PET-CT",
            ),

        "modality_mr":
            bool_pattern(
                scan_type,
                r"\bMRI?\b|MAGNETIC",
            ),

        "modality_bone_scan":
            bool_pattern(
                scan_type,
                r"BONE SCAN",
            ),

        "site_bone":
            bool_pattern(
                combined,
                r"\bBONE\b|SKELET|VERTEBR|RIB",
            ),

        "site_liver":
            bool_pattern(
                combined,
                r"\bLIVER\b",
            ),

        "site_lung":
            bool_pattern(
                combined,
                r"\bLUNG\b|PULMON",
            ),

        "site_brain":
            bool_pattern(
                combined,
                r"\bBRAIN\b|\bCNS\b|CEREBR",
            ),

        "site_lymph":
            bool_pattern(
                combined,
                r"LYMPH",
            ),

        "site_pleura":
            bool_pattern(
                combined,
                r"PLEURA",
            ),

        "raw_coverage_chest":
            bool_pattern(
                scan_sites,
                r"CHEST|THORAX|FULL BODY",
            ),

        "raw_coverage_abdomen":
            bool_pattern(
                scan_sites,
                r"ABDOM|FULL BODY",
            ),

        "raw_coverage_pelvis":
            bool_pattern(
                scan_sites,
                r"PELV|FULL BODY",
            ),

        "raw_coverage_head":
            bool_pattern(
                scan_sites,
                r"HEAD|BRAIN|FULL BODY",
            ),
    }

    specific = (
        features[
            "raw_coverage_chest"
        ]
        or features[
            "raw_coverage_abdomen"
        ]
        or features[
            "raw_coverage_pelvis"
        ]
        or features[
            "raw_coverage_head"
        ]
    )

    features[
        "raw_coverage_other"
    ] = float(
        bool(
            str(
                scan_sites
            ).strip()
        )
        and not specific
    )

    ###########################################################################
    # Aliases used if CKPT5 stored non-raw coverage names.
    ###########################################################################

    for anatomical in (
        "chest",
        "abdomen",
        "pelvis",
        "head",
        "other",
    ):

        features[
            "coverage_"
            + anatomical
        ] = features[
            "raw_coverage_"
            + anatomical
        ]

    return features


def construct_feature_matrix(
    episodes: pd.DataFrame,
    frozen_names: list[str],
) -> tuple[
    np.ndarray,
    list[str],
]:

    matrix = np.zeros(
        (
            len(
                episodes
            ),
            len(
                frozen_names
            ),
        ),
        dtype=np.float32,
    )

    unresolved = set()

    for row_index, (_, row) in enumerate(
        episodes.iterrows()
    ):

        values = bpc_feature_dict(
            row
        )

        for column_index, name in enumerate(
            frozen_names
        ):

            normalized = norm(
                name
            )

            if normalized in values:

                matrix[
                    row_index,
                    column_index,
                ] = values[
                    normalized
                ]

                continue

            ###################################################################
            # CKPT5 names should already be normalized, but permit raw prefix
            # variation.
            ###################################################################

            stripped = re.sub(
                r"^raw_",
                "",
                normalized,
            )

            if stripped in values:

                matrix[
                    row_index,
                    column_index,
                ] = values[
                    stripped
                ]

                continue

            unresolved.add(
                name
            )

    return (
        matrix,
        sorted(
            unresolved
        ),
    )


###############################################################################
# Canonical CHORD W3 normalization
###############################################################################


def load_chord_w3(
    repo: Path,
) -> pd.DataFrame:

    frame = pd.read_parquet(
        repo
        / "artifacts"
        / "checkpoint1"
        / "canonical"
        / "chord_mbc_scan_episodes_w3.parquet"
    )

    patient_col = find_col(
        frame.columns,
        [
            "patient_id",
            "PATIENT_ID",
        ],
    )

    day_col = find_col(
        frame.columns,
        [
            "episode_end_day",
            "landmark_day",
        ],
    )

    state_col = find_col(
        frame.columns,
        [
            "progression_state_3",
            "scan_state",
        ],
    )

    line_col = find_col(
        frame.columns,
        [
            "treatment_line",
            "line",
        ],
        required=False,
    )

    scan_id_col = find_col(
        frame.columns,
        [
            "scan_episode_id",
            "episode_id",
        ],
        required=False,
    )

    out = pd.DataFrame(
        {
            "patient_id":
                frame[
                    patient_col
                ]
                .astype(str)
                .str.strip(),

            "chord_episode_end_day":
                pd.to_numeric(
                    frame[
                        day_col
                    ],
                    errors="coerce",
                ),

            "chord_scan_state":
                frame[
                    state_col
                ]
                .astype(str)
                .str.strip()
                .str.upper(),
        }
    )

    if line_col:

        out[
            "chord_line"
        ] = pd.to_numeric(
            frame[
                line_col
            ],
            errors="coerce",
        )

    if scan_id_col:

        out[
            "scan_episode_id"
        ] = (
            frame[
                scan_id_col
            ]
            .astype(str)
            .str.strip()
        )

    return out


###############################################################################
# Episode matching
###############################################################################


def nearest_match(
    bpc: pd.DataFrame,
    chord: pd.DataFrame,
    tolerance: float,
) -> pd.DataFrame:

    matches = []

    common_patients = sorted(
        set(
            bpc[
                "patient_id"
            ]
        )
        & set(
            chord[
                "patient_id"
            ]
        )
    )

    for patient in common_patients:

        left = (
            bpc[
                bpc[
                    "patient_id"
                ]
                == patient
            ]
            .sort_values(
                "episode_end_day"
            )
            .copy()
        )

        right = (
            chord[
                chord[
                    "patient_id"
                ]
                == patient
            ]
            .sort_values(
                "chord_episode_end_day"
            )
            .copy()
        )

        if (
            left.empty
            or right.empty
        ):
            continue

        candidates = []

        for li, lrow in left.iterrows():

            for ri, rrow in right.iterrows():

                delta = abs(
                    float(
                        lrow[
                            "episode_end_day"
                        ]
                    )
                    - float(
                        rrow[
                            "chord_episode_end_day"
                        ]
                    )
                )

                if delta <= tolerance:

                    candidates.append(
                        (
                            delta,
                            li,
                            ri,
                        )
                    )

        candidates.sort()

        used_left = set()
        used_right = set()

        for delta, li, ri in candidates:

            if (
                li in used_left
                or ri in used_right
            ):
                continue

            used_left.add(
                li
            )

            used_right.add(
                ri
            )

            lrow = left.loc[
                li
            ]

            rrow = right.loc[
                ri
            ]

            row = {
                "patient_id":
                    patient,

                "bpc_index":
                    int(
                        li
                    ),

                "chord_index":
                    int(
                        ri
                    ),

                "bpc_episode_end_day":
                    float(
                        lrow[
                            "episode_end_day"
                        ]
                    ),

                "chord_episode_end_day":
                    float(
                        rrow[
                            "chord_episode_end_day"
                        ]
                    ),

                "abs_day_delta":
                    float(
                        delta
                    ),

                "bpc_scan_state":
                    lrow[
                        "scan_state"
                    ],

                "chord_scan_state":
                    rrow[
                        "chord_scan_state"
                    ],
            }

            if (
                "chord_line"
                in right.columns
            ):

                row[
                    "chord_line"
                ] = rrow[
                    "chord_line"
                ]

            if (
                "scan_episode_id"
                in right.columns
            ):

                row[
                    "scan_episode_id"
                ] = rrow[
                    "scan_episode_id"
                ]

            matches.append(
                row
            )

    return pd.DataFrame(
        matches
    )


###############################################################################
# Treatment-line mappings
###############################################################################


def prepare_treatment(
    treatment: pd.DataFrame,
    site_map: dict[str, str],
) -> pd.DataFrame:

    patient_col = find_col(
        treatment.columns,
        [
            "PATIENT_ID",
            "patient_id",
        ],
    )

    start_col = find_col(
        treatment.columns,
        [
            "START_DATE",
            "start_date",
        ],
    )

    stop_col = find_col(
        treatment.columns,
        [
            "STOP_DATE",
            "stop_date",
        ],
        required=False,
    )

    line_col = find_col(
        treatment.columns,
        [
            "REGIMEN_NUMBER",
            "regimen_number",
        ],
    )

    out = pd.DataFrame(
        {
            "patient_id":
                treatment[
                    patient_col
                ]
                .astype(str)
                .str.strip(),

            "tx_start":
                pd.to_numeric(
                    treatment[
                        start_col
                    ],
                    errors="coerce",
                ),

            "tx_line":
                pd.to_numeric(
                    treatment[
                        line_col
                    ],
                    errors="coerce",
                ),
        }
    )

    out[
        "tx_stop"
    ] = (
        pd.to_numeric(
            treatment[
                stop_col
            ],
            errors="coerce",
        )
        if stop_col
        else np.nan
    )

    out[
        "site"
    ] = out[
        "patient_id"
    ].map(
        site_map
    )

    return out[
        out[
            "tx_start"
        ].notna()
        & out[
            "tx_line"
        ].notna()
    ].copy()


def assign_lines(
    episodes: pd.DataFrame,
    treatment: pd.DataFrame,
) -> pd.DataFrame:

    result = episodes.copy()

    strict_active = []
    carry_forward = []

    for _, episode in result.iterrows():

        patient = episode[
            "patient_id"
        ]

        day = float(
            episode[
                "episode_end_day"
            ]
        )

        tx = treatment[
            treatment[
                "patient_id"
            ]
            == patient
        ]

        prior = tx[
            tx[
                "tx_start"
            ]
            <= day
        ]

        #######################################################################
        # A: latest active interval at scan.
        #######################################################################

        active = prior[
            prior[
                "tx_stop"
            ].isna()
            | (
                prior[
                    "tx_stop"
                ]
                >= day
            )
        ]

        if active.empty:

            strict_active.append(
                np.nan
            )

        else:

            selected = (
                active
                .sort_values(
                    [
                        "tx_start",
                        "tx_line",
                    ]
                )
                .iloc[
                    -1
                ]
            )

            strict_active.append(
                float(
                    selected[
                        "tx_line"
                    ]
                )
            )

        #######################################################################
        # B: latest started regimen carried forward.
        #######################################################################

        if prior.empty:

            carry_forward.append(
                np.nan
            )

        else:

            selected = (
                prior
                .sort_values(
                    [
                        "tx_start",
                        "tx_line",
                    ]
                )
                .iloc[
                    -1
                ]
            )

            carry_forward.append(
                float(
                    selected[
                        "tx_line"
                    ]
                )
            )

    result[
        "line_active_interval"
    ] = strict_active

    result[
        "line_latest_started"
    ] = carry_forward

    return result


def line_agreement(
    matches: pd.DataFrame,
    episodes: pd.DataFrame,
) -> dict[str, Any]:

    if (
        matches.empty
        or "chord_line"
        not in matches.columns
    ):

        return {}

    merged = matches.merge(
        episodes[
            [
                "bpc_episode_id",
                "line_active_interval",
                "line_latest_started",
            ]
        ].reset_index(
            names="bpc_index"
        ),
        on="bpc_index",
        how="left",
        validate="one_to_one",
    )

    report = {}

    for name in (
        "line_active_interval",
        "line_latest_started",
    ):

        subset = merged[
            merged[
                "chord_line"
            ].notna()
            & merged[
                name
            ].notna()
        ]

        if subset.empty:

            agreement = float(
                "nan"
            )

        else:

            agreement = float(
                (
                    subset[
                        "chord_line"
                    ].astype(float)
                    == subset[
                        name
                    ].astype(float)
                ).mean()
            )

        report[
            name
        ] = {
            "rows":
                int(
                    len(
                        subset
                    )
                ),

            "agreement":
                agreement,
        }

    return report


###############################################################################
# CKPT5 frozen feature comparison
###############################################################################


def load_ckpt5_feature_reference(
    repo: Path,
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    list[str],
]:

    index = pd.read_parquet(
        repo
        / "artifacts"
        / "checkpoint5"
        / "prepared"
        / "scan_index.parquet"
    )

    features = np.load(
        repo
        / "artifacts"
        / "checkpoint5"
        / "prepared"
        / "current_scan_features_f32.npy",
        mmap_mode="r",
    )

    schema = json.loads(
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

    names = list(
        schema[
            "current_scan"
        ]
    )

    if features.shape != (
        len(
            index
        ),
        len(
            names
        ),
    ):

        raise RuntimeError(
            "CKPT5 feature-array shape mismatch."
        )

    patient_col = find_col(
        index.columns,
        [
            "patient_id",
        ],
    )

    day_col = find_col(
        index.columns,
        [
            "landmark_day",
            "episode_end_day",
        ],
    )

    reference = pd.DataFrame(
        {
            "patient_id":
                index[
                    patient_col
                ]
                .astype(str)
                .str.strip(),

            "landmark_day":
                pd.to_numeric(
                    index[
                        day_col
                    ],
                    errors="coerce",
                ),

            "feature_row":
                np.arange(
                    len(
                        index
                    ),
                    dtype=int,
                ),
        }
    )

    return (
        reference,
        features,
        names,
    )


def compare_features(
    bpc_episodes: pd.DataFrame,
    bpc_features: np.ndarray,
    ckpt5_index: pd.DataFrame,
    ckpt5_features: np.ndarray,
    feature_names: list[str],
) -> dict[str, Any]:

    left = bpc_episodes[
        [
            "patient_id",
            "episode_end_day",
        ]
    ].copy()

    left[
        "bpc_feature_row"
    ] = np.arange(
        len(
            left
        ),
        dtype=int,
    )

    right = ckpt5_index.copy()

    merged = left.merge(
        right,
        left_on=[
            "patient_id",
            "episode_end_day",
        ],
        right_on=[
            "patient_id",
            "landmark_day",
        ],
        how="inner",
    )

    if merged.empty:

        return {
            "exact_day_matches":
                0,
        }

    bpc_rows = merged[
        "bpc_feature_row"
    ].to_numpy(
        dtype=int
    )

    frozen_rows = merged[
        "feature_row"
    ].to_numpy(
        dtype=int
    )

    a = np.asarray(
        bpc_features[
            bpc_rows
        ],
        dtype=np.float32,
    )

    b = np.asarray(
        ckpt5_features[
            frozen_rows
        ],
        dtype=np.float32,
    )

    absolute = np.abs(
        a
        - b
    )

    per_feature = {}

    for column, name in enumerate(
        feature_names
    ):

        per_feature[
            name
        ] = {
            "mean_absolute_difference":
                float(
                    absolute[
                        :,
                        column
                    ].mean()
                ),

            "exact_fraction":
                float(
                    (
                        absolute[
                            :,
                            column
                        ]
                        < 1e-6
                    ).mean()
                ),

            "bpc_mean":
                float(
                    a[
                        :,
                        column
                    ].mean()
                ),

            "frozen_mean":
                float(
                    b[
                        :,
                        column
                    ].mean()
                ),
        }

    return {
        "exact_day_matches":
            int(
                len(
                    merged
                )
            ),

        "vector_exact_fraction":
            float(
                (
                    absolute.max(
                        axis=1
                    )
                    < 1e-6
                ).mean()
            ),

        "mean_absolute_difference":
            float(
                absolute.mean()
            ),

        "per_feature":
            per_feature,
    }


###############################################################################
# Temporal vocabulary/artifact inventory
###############################################################################


def temporal_artifact_inventory(
    repo: Path,
) -> dict[str, Any]:

    root = (
        repo
        / "artifacts"
        / "checkpoint4"
    )

    files = []

    json_summaries = {}

    for path in sorted(
        root.rglob(
            "*"
        )
    ):

        if (
            not path.is_file()
            or path.stat().st_size
            == 0
        ):
            continue

        relative = str(
            path.relative_to(
                repo
            )
        )

        files.append(
            {
                "path":
                    relative,

                "bytes":
                    int(
                        path.stat().st_size
                    ),
            }
        )

        if (
            path.suffix.lower()
            == ".json"
            and path.stat().st_size
            < 5_000_000
        ):

            try:

                payload = json.loads(
                    path.read_text(
                        encoding="utf-8"
                    )
                )

            except Exception:
                continue

            if isinstance(
                payload,
                dict,
            ):

                interesting = {}

                for key, value in payload.items():

                    normalized = norm(
                        key
                    )

                    if any(
                        token
                        in normalized
                        for token in (
                            "vocab",
                            "event",
                            "token",
                            "type",
                            "feature",
                            "channel",
                        )
                    ):

                        if isinstance(
                            value,
                            (
                                str,
                                int,
                                float,
                                bool,
                                list,
                                dict,
                            ),
                        ):

                            interesting[
                                key
                            ] = value

                if interesting:

                    json_summaries[
                        relative
                    ] = interesting

    return {
        "files":
            files,

        "interesting_json":
            json_summaries,
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
        "--bpc-root",
        default=(
            "data/external_sources/"
            "bpc_brca_1_0_public"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="artifacts/checkpoint7a3",
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
    # Verify outcome-blinded upstream.
    ###########################################################################

    semantic = json.loads(
        (
            repo
            / "artifacts"
            / "checkpoint7a2"
            / "semantic_audit.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    if semantic[
        "external_outcomes_opened"
    ]:

        raise RuntimeError(
            "CKPT7A2 reports external outcomes opened."
        )

    ###########################################################################
    # Predictor sources.
    ###########################################################################

    (
        ckpt7a2,
        frames,
        site_map,
    ) = load_bpc_sources(
        repo,
        bpc_root,
    )

    ###########################################################################
    # Build BPC W3 episodes.
    ###########################################################################

    episodes = build_bpc_w3(
        frames[
            "imaging"
        ],
        site_map,
    )

    msk = episodes[
        episodes[
            "site"
        ]
        == "MSK"
    ].copy()

    dfci = episodes[
        episodes[
            "site"
        ]
        == "DFCI"
    ].copy()

    vicc = episodes[
        episodes[
            "site"
        ]
        == "VICC"
    ].copy()

    ###########################################################################
    # Frozen feature schema.
    ###########################################################################

    (
        ckpt5_index,
        ckpt5_features,
        feature_names,
    ) = load_ckpt5_feature_reference(
        repo
    )

    (
        episode_features,
        unresolved_features,
    ) = construct_feature_matrix(
        episodes,
        feature_names,
    )

    if unresolved_features:

        print(
            "[CKPT7A3_UNRESOLVED_FEATURES]",
            unresolved_features,
            flush=True,
        )

    feature_rows = {
        episode_id:
            index
        for index, episode_id
        in enumerate(
            episodes[
                "bpc_episode_id"
            ]
        )
    }

    msk_feature_rows = np.asarray(
        [
            feature_rows[
                value
            ]
            for value
            in msk[
                "bpc_episode_id"
            ]
        ],
        dtype=int,
    )

    msk_features = (
        episode_features[
            msk_feature_rows
        ]
    )

    ###########################################################################
    # CHORD positive-control episode match.
    ###########################################################################

    chord = load_chord_w3(
        repo
    )

    overlap_patients = (
        set(
            msk[
                "patient_id"
            ]
        )
        & set(
            chord[
                "patient_id"
            ]
        )
    )

    # Only observed-state episodes can participate in the scan-state
    # semantic bridge. UNOBSERVED episodes remain in the imaging history
    # and are reported separately.
    msk_state_observed = msk[
        msk[
            "scan_state"
        ]
        != "UNOBSERVED"
    ].copy()

    match_0 = nearest_match(
        msk_state_observed,
        chord,
        tolerance=0,
    )

    match_3 = nearest_match(
        msk_state_observed,
        chord,
        tolerance=3,
    )

    match_7 = nearest_match(
        msk_state_observed,
        chord,
        tolerance=7,
    )

    def match_summary(
        matched: pd.DataFrame,
    ):

        if matched.empty:

            return {
                "rows":
                    0,

                "patients":
                    0,

                "state_agreement":
                    None,

                "median_abs_day_delta":
                    None,
            }

        return {
            "rows":
                int(
                    len(
                        matched
                    )
                ),

            "patients":
                int(
                    matched[
                        "patient_id"
                    ].nunique()
                ),

            "state_agreement":
                float(
                    (
                        matched[
                            "bpc_scan_state"
                        ]
                        == matched[
                            "chord_scan_state"
                        ]
                    ).mean()
                ),

            "median_abs_day_delta":
                float(
                    matched[
                        "abs_day_delta"
                    ].median()
                ),
        }

    ###########################################################################
    # Treatment-line positive control.
    ###########################################################################

    treatment = prepare_treatment(
        frames[
            "treatment"
        ],
        site_map,
    )

    episodes_with_lines = assign_lines(
        episodes,
        treatment,
    )

    msk_with_lines = episodes_with_lines[
        episodes_with_lines[
            "site"
        ]
        == "MSK"
    ].copy()

    line_report_0 = line_agreement(
        match_0,
        msk_with_lines,
    )

    line_report_3 = line_agreement(
        match_3,
        msk_with_lines,
    )

    ###########################################################################
    # Frozen current-scan-feature positive control.
    ###########################################################################

    observed_msk_mask = (
        msk[
            "scan_state"
        ]
        != "UNOBSERVED"
    ).to_numpy()

    msk_feature_report = compare_features(
        msk.loc[
            observed_msk_mask
        ].copy(),
        msk_features[
            observed_msk_mask
        ],
        ckpt5_index,
        ckpt5_features,
        feature_names,
    )

    ###########################################################################
    # Temporal encoder/vocabulary inventory for CKPT7A4 mapping.
    ###########################################################################

    temporal_inventory = temporal_artifact_inventory(
        repo
    )

    ###########################################################################
    # Decide bridge status.
    #
    # These are not endpoint-performance thresholds. They are semantic adapter
    # fidelity gates on the development-only MSK bridge.
    ###########################################################################

    summary_0 = match_summary(
        match_0
    )

    summary_3 = match_summary(
        match_3
    )

    matched_fraction = (
        summary_3[
            "rows"
        ]
        / max(
            len(
                msk_state_observed
            ),
            1,
        )
    )

    state_ok = (
        summary_3[
            "state_agreement"
        ]
        is not None
        and summary_3[
            "state_agreement"
        ]
        >= 0.90
    )

    matching_ok = (
        len(
            overlap_patients
        )
        >= 450
        and matched_fraction
        >= 0.70
    )

    line_candidates = [
        value[
            "agreement"
        ]
        for value in line_report_3.values()
        if value.get(
            "agreement"
        )
        is not None
        and np.isfinite(
            value[
                "agreement"
            ]
        )
    ]

    best_line_agreement = (
        max(
            line_candidates
        )
        if line_candidates
        else float(
            "nan"
        )
    )

    line_ok = (
        np.isfinite(
            best_line_agreement
        )
        and best_line_agreement
        >= 0.90
    )

    feature_ok = (
        not unresolved_features
        and msk_feature_report.get(
            "exact_day_matches",
            0,
        )
        >= 500
        and msk_feature_report.get(
            "mean_absolute_difference",
            1.0,
        )
        <= 0.10
    )

    if (
        matching_ok
        and state_ok
        and line_ok
        and feature_ok
    ):

        status = (
            "PASS_MSK_BRIDGE_READY_FOR_EXTERNAL_TENSOR_BUILD"
        )

    else:

        status = (
            "NEEDS_MSK_BRIDGE_RESOLUTION"
        )

    ###########################################################################
    # Choose line strategy only if it is empirically supported on development
    # MSK. DFCI/VICC outcomes remain irrelevant to this choice.
    ###########################################################################

    selected_line_strategy = None

    if line_report_3:

        finite = {
            name:
                value[
                    "agreement"
                ]
            for name, value
            in line_report_3.items()
            if value.get(
                "agreement"
            )
            is not None
            and np.isfinite(
                value[
                    "agreement"
                ]
            )
        }

        if finite:

            selected_line_strategy = max(
                finite,
                key=finite.get,
            )

    ###########################################################################
    # Save crosswalk manifest.
    ###########################################################################

    crosswalk = {
        "scan_window_days":
            3,

        "episode_landmark":
            "episode_end_day",

        "row_state_source":
            "CURATED_CANCER_STATUS",

        "row_state_crosswalk":
            STATE_CROSSWALK,

        "episode_state_precedence":
            [
                "PROGRESSIVE",
                "INDETERMINATE",
                "NON_PROGRESSIVE",
            ],

        "cancer_status_policy":
            (
                "CANCER_STATUS is cancer-presence evidence, "
                "not progression state, and is not used to "
                "define the three-state endpoint."
            ),

        "mixed_policy":
            (
                "Mixed maps to INDETERMINATE rather than "
                "PROGRESSIVE or NON_PROGRESSIVE."
            ),

        "missing_curated_status_policy":
            (
                "Blank CURATED_CANCER_STATUS is UNOBSERVED, "
                "not INDETERMINATE. Such imaging rows remain "
                "eligible to contribute modality/coverage and "
                "episode grouping, but an episode with no observed "
                "curated response is not a primary scan-state "
                "prediction landmark."
            ),

        "selected_line_strategy":
            selected_line_strategy,

        "frozen_current_scan_feature_names":
            feature_names,

        "unresolved_current_scan_features":
            unresolved_features,
    }

    atomic_json(
        out
        / "external_semantic_crosswalk.json",
        crosswalk,
    )

    ###########################################################################
    # Save episode tables WITHOUT outcomes.
    ###########################################################################

    atomic_parquet(
        out
        / "bpc_w3_predictor_episodes.parquet",
        episodes_with_lines,
    )

    np.save(
        out
        / "bpc_w3_scan_features_f32.npy",
        episode_features,
    )

    atomic_parquet(
        out
        / "msk_bridge_matches_exact.parquet",
        match_0,
    )

    atomic_parquet(
        out
        / "msk_bridge_matches_within3.parquet",
        match_3,
    )

    ###########################################################################
    # QC/report.
    ###########################################################################

    report = {
        "status":
            status,

        "external_outcomes_opened":
            False,

        "external_outcome_distributions_inspected":
            False,

        "crosswalk":
            crosswalk,

        "bpc_w3_counts": {
            "MSK": {
                "episodes":
                    int(
                        len(
                            msk
                        )
                    ),

                "patients":
                    int(
                        msk[
                            "patient_id"
                        ].nunique()
                    ),

                "observed_state_episodes":
                    int(
                        (
                            msk[
                                "scan_state"
                            ]
                            != "UNOBSERVED"
                        ).sum()
                    ),

                "unobserved_state_episodes":
                    int(
                        (
                            msk[
                                "scan_state"
                            ]
                            == "UNOBSERVED"
                        ).sum()
                    ),
            },

            "DFCI": {
                "episodes":
                    int(
                        len(
                            dfci
                        )
                    ),

                "patients":
                    int(
                        dfci[
                            "patient_id"
                        ].nunique()
                    ),

                "observed_state_episodes":
                    int(
                        (
                            dfci[
                                "scan_state"
                            ]
                            != "UNOBSERVED"
                        ).sum()
                    ),

                "unobserved_state_episodes":
                    int(
                        (
                            dfci[
                                "scan_state"
                            ]
                            == "UNOBSERVED"
                        ).sum()
                    ),
            },

            "VICC": {
                "episodes":
                    int(
                        len(
                            vicc
                        )
                    ),

                "patients":
                    int(
                        vicc[
                            "patient_id"
                        ].nunique()
                    ),

                "observed_state_episodes":
                    int(
                        (
                            vicc[
                                "scan_state"
                            ]
                            != "UNOBSERVED"
                        ).sum()
                    ),

                "unobserved_state_episodes":
                    int(
                        (
                            vicc[
                                "scan_state"
                            ]
                            == "UNOBSERVED"
                        ).sum()
                    ),
            },
        },

        "msk_chord_overlap_patients":
            int(
                len(
                    overlap_patients
                )
            ),

        "episode_matching": {
            "exact_day":
                summary_0,

            "within_3_days":
                summary_3,

            "within_7_days":
                match_summary(
                    match_7
                ),

            "within3_fraction_of_bpc_msk":
                float(
                    matched_fraction
                ),
        },

        "line_mapping": {
            "exact_day_matches":
                line_report_0,

            "within3_matches":
                line_report_3,

            "selected_strategy":
                selected_line_strategy,

            "best_within3_agreement":
                (
                    float(
                        best_line_agreement
                    )
                    if np.isfinite(
                        best_line_agreement
                    )
                    else None
                ),
        },

        "frozen_scan_feature_bridge":
            msk_feature_report,

        "temporal_artifact_inventory":
            temporal_inventory,

        "gates": {
            "episode_matching_ok":
                matching_ok,

            "scan_state_mapping_ok":
                state_ok,

            "treatment_line_mapping_ok":
                line_ok,

            "current_scan_feature_mapping_ok":
                feature_ok,
        },
    }

    atomic_json(
        out
        / "bridge_report.json",
        report,
    )

    ###########################################################################
    # Human-readable audit.
    ###########################################################################

    audit = f"""# CKPT7A3 — BPC-MSK semantic bridge validation

Status: **{status}**

External outcomes opened: **NO**

## Frozen scan-state crosswalk

- Progressing/Worsening/Enlarging -> PROGRESSIVE
- Stable/No change -> NON_PROGRESSIVE
- Improving/Responding -> NON_PROGRESSIVE
- Not stated/Indeterminate -> INDETERMINATE
- Mixed -> INDETERMINATE
- Blank curated status -> UNOBSERVED

`CANCER_STATUS` is not used as progression state.

Blank `CURATED_CANCER_STATUS` remains explicit missingness. It is never
collapsed into observed INDETERMINATE.

## BPC W3 predictor episodes

MSK:
{report['bpc_w3_counts']['MSK']}

DFCI:
{report['bpc_w3_counts']['DFCI']}

VICC:
{report['bpc_w3_counts']['VICC']}

## MSK -> CHORD positive control

MSK/CHORD overlapping patients:
{report['msk_chord_overlap_patients']}

Exact day:
{summary_0}

Within 3 days:
{summary_3}

Within 7 days:
{report['episode_matching']['within_7_days']}

## Treatment-line bridge

{report['line_mapping']}

## Frozen CKPT5 scan-feature bridge

{msk_feature_report}

## Adapter gates

{report['gates']}

## Policy

This checkpoint used BPC-MSK only as a development/schema/label bridge.
No DFCI/VICC survival, death, censoring, PFS, or outcome distributions were opened.
"""

    atomic_text(
        out
        / "audit.md",
        audit,
    )

    ###########################################################################
    # Handoff.
    ###########################################################################

    handoff = {
        "checkpoint":
            "7A3",

        "name":
            "bpc_msk_semantic_bridge_validation",

        "status":
            status,

        "external_outcomes_opened":
            False,

        "crosswalk":
            crosswalk,

        "msk_overlap_patients":
            len(
                overlap_patients
            ),

        "within3_episode_match":
            summary_3,

        "line_mapping":
            report[
                "line_mapping"
            ],

        "scan_feature_bridge":
            {
                key:
                    value
                for key, value
                in msk_feature_report.items()
                if key
                != "per_feature"
            },

        "next_action":
            (
                "Build complete outcome-blinded DFCI/VICC "
                "temporal states, genomic embeddings, frozen "
                "18-D scan vectors, PRE/POST candidate outputs, "
                "and hash prediction tensors before outcome access."
                if status
                == "PASS_MSK_BRIDGE_READY_FOR_EXTERNAL_TENSOR_BUILD"
                else
                "Resolve the specific failed MSK semantic bridge "
                "gate before constructing DFCI/VICC model tensors."
            ),
    }

    atomic_json(
        repo
        / "artifacts"
        / "handoff"
        / "checkpoint_07A3.json",
        handoff,
    )

    ###########################################################################
    # Terminal packet.
    ###########################################################################

    print("")
    print(
        "========== CKPT7A3 SUMMARY =========="
    )

    print(
        f"status={status}"
    )

    print(
        "external_outcomes_opened=False"
    )

    print(
        "bpc_w3_counts="
        f"{report['bpc_w3_counts']}"
    )

    print(
        "msk_chord_overlap_patients="
        f"{len(overlap_patients)}"
    )

    print(
        "exact_day_match="
        f"{summary_0}"
    )

    print(
        "within3_match="
        f"{summary_3}"
    )

    print(
        "selected_line_strategy="
        f"{selected_line_strategy}"
    )

    print(
        "best_line_agreement="
        f"{report['line_mapping']['best_within3_agreement']}"
    )

    print(
        "scan_feature_bridge_summary="
        f"{ {k:v for k,v in msk_feature_report.items() if k != 'per_feature'} }"
    )

    print(
        "unresolved_scan_features="
        f"{unresolved_features}"
    )

    print(
        "gates="
        f"{report['gates']}"
    )

    print(
        "handoff="
        "artifacts/handoff/checkpoint_07A3.json"
    )

    print(
        "========== CKPT7A3 SUMMARY END =========="
    )

    print("")
    print(
        "========== CKPT7A3 DECISION PACKET =========="
    )

    print(
        "scan_state_crosswalk="
        f"{STATE_CROSSWALK}"
    )

    print(
        "episode_matching="
        f"{report['episode_matching']}"
    )

    print(
        "line_mapping="
        f"{report['line_mapping']}"
    )

    print(
        "scan_feature_per_feature="
        f"{msk_feature_report.get('per_feature')}"
    )

    print(
        "temporal_artifact_inventory="
        f"{temporal_inventory}"
    )

    print(
        "========== CKPT7A3 DECISION PACKET END =========="
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
