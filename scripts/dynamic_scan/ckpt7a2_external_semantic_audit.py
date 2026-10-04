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


EXTERNAL_SITES = (
    "DFCI",
    "VICC",
)

ALL_SITES = (
    "MSK",
    "DFCI",
    "VICC",
)

EXPECTED_SITE_COUNTS = {
    "MSK": 529,
    "DFCI": 428,
    "VICC": 173,
}

###############################################################################
# Files that CKPT7A1 explicitly quarantined.
###############################################################################

QUARANTINED = {
    "cBioPortal_files/data_clinical_supp_survival.txt",
    "cBioPortal_files/data_clinical_supp_survival_treatment.txt",
    "clinical_data/cancer_level_dataset_index.csv",
    "clinical_data/cancer_panel_test_level_dataset.csv",
    "clinical_data/patient_level_dataset.csv",
    "clinical_data/regimen_cancer_level_dataset.csv",
}

###############################################################################
# Predictor tables we are deliberately permitting.
###############################################################################

PREDICTOR_TABLES = {
    "clinical_patient":
        "cBioPortal_files/data_clinical_patient.txt",

    "clinical_sample":
        "cBioPortal_files/data_clinical_sample.txt",

    "imaging":
        "cBioPortal_files/data_timeline_imaging.txt",

    "imaging_raw":
        "clinical_data/imaging_level_dataset.csv",

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

    "gene_matrix":
        "cBioPortal_files/data_gene_matrix.txt",

    "mutations":
        "cBioPortal_files/data_mutations_extended.txt",
}


###############################################################################
# Utilities
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
            f"Parquet verification failed: {path}"
        )

    tmp.replace(path)


def find_col(
    columns: Iterable[str],
    candidates: Iterable[str],
    *,
    required: bool = False,
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
            f"Missing {list(candidates)}; "
            f"available={list(columns)}"
        )

    return None


def delimiter_for(
    path: Path,
) -> str:

    return (
        ","
        if path.suffix.lower()
        == ".csv"
        else "\t"
    )


def read_table(
    path: Path,
    *,
    usecols=None,
) -> pd.DataFrame:

    return pd.read_csv(
        path,
        sep=delimiter_for(
            path
        ),
        comment="#",
        dtype=str,
        keep_default_na=False,
        low_memory=False,
        usecols=usecols,
    )


def top_values(
    series: pd.Series,
    n: int = 40,
) -> dict[str, int]:

    values = (
        series
        .astype(str)
        .str.strip()
        .replace(
            "",
            pd.NA,
        )
        .dropna()
    )

    return {
        str(key):
            int(value)
        for key, value
        in values.value_counts(
            dropna=False
        )
        .head(
            n
        )
        .items()
    }


def numeric_summary(
    series: pd.Series,
) -> dict[str, Any]:

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    good = numeric.dropna()

    if good.empty:

        return {
            "n":
                0,

            "fraction":
                0.0,
        }

    return {
        "n":
            int(
                len(
                    good
                )
            ),

        "fraction":
            float(
                len(
                    good
                )
                / max(
                    len(
                        series
                    ),
                    1,
                )
            ),

        "min":
            float(
                good.min()
            ),

        "p25":
            float(
                good.quantile(
                    0.25
                )
            ),

        "median":
            float(
                good.median()
            ),

        "p75":
            float(
                good.quantile(
                    0.75
                )
            ),

        "max":
            float(
                good.max()
            ),
    }


def normalize_site(
    value: Any,
) -> str | None:

    text = str(
        value
    ).strip().upper()

    if not text:
        return None

    if (
        "DFCI"
        in text
        or "DANA"
        in text
    ):
        return "DFCI"

    if (
        "VICC"
        in text
        or "VANDERBILT"
        in text
    ):
        return "VICC"

    if (
        text == "MSK"
        or "MEMORIAL"
        in text
        or "SLOAN"
        in text
    ):
        return "MSK"

    return None


###############################################################################
# Explicit access guard
###############################################################################


class AccessGuard:

    def __init__(
        self,
        root: Path,
        schema_inventory: pd.DataFrame,
    ):

        self.root = root

        self.inventory = (
            schema_inventory
            .copy()
        )

        self.access_log = []

        self.inventory_map = {
            str(
                row[
                    "relative_path"
                ]
            ):
                bool(
                    row[
                        "hard_outcome_sensitive"
                    ]
                )
            for _, row
            in self.inventory.iterrows()
        }

    def path(
        self,
        relative: str,
    ) -> Path:

        return (
            self.root
            / relative
        )

    def read_predictor(
        self,
        relative: str,
        *,
        usecols=None,
    ) -> pd.DataFrame:

        if relative in QUARANTINED:

            raise RuntimeError(
                "Attempted row access to quarantined file: "
                f"{relative}"
            )

        sensitive = (
            self.inventory_map.get(
                relative
            )
        )

        if sensitive is None:

            raise RuntimeError(
                "Predictor table missing from CKPT7A1 inventory: "
                f"{relative}"
            )

        if sensitive:

            raise RuntimeError(
                "CKPT7A1 marked predictor table outcome-sensitive: "
                f"{relative}"
            )

        path = self.path(
            relative
        )

        if not path.exists():

            raise RuntimeError(
                f"Predictor file missing: {relative}"
            )

        frame = read_table(
            path,
            usecols=usecols,
        )

        self.access_log.append(
            {
                "relative_path":
                    relative,

                "mode":
                    "FULL_PREDICTOR_VALUES"
                    if usecols is None
                    else "SELECTED_PREDICTOR_COLUMNS",

                "columns_loaded":
                    list(
                        frame.columns
                    ),

                "rows":
                    int(
                        len(
                            frame
                        )
                    ),
            }
        )

        return frame


###############################################################################
# Site partition
###############################################################################


def build_site_map(
    clinical_patient: pd.DataFrame,
) -> tuple[
    dict[str, str],
    dict[str, Any],
]:

    patient_col = find_col(
        clinical_patient.columns,
        [
            "PATIENT_ID",
            "patient_id",
            "record_id",
        ],
        required=True,
    )

    site_candidates = [
        column
        for column
        in clinical_patient.columns
        if (
            norm(
                column
            )
            in {
                "institution",
                "institution_name",
                "site",
                "center",
                "center_id",
                "bpc_site",
                "source_site",
            }
            or "institution"
            in norm(
                column
            )
        )
    ]

    best = None
    best_count = -1

    for column in site_candidates:

        resolved = (
            clinical_patient[
                column
            ]
            .map(
                normalize_site
            )
            .notna()
            .sum()
        )

        if resolved > best_count:

            best = column
            best_count = int(
                resolved
            )

    ###########################################################################
    # If data_clinical_patient itself does not expose institution, reconstruct
    # from the GENIE-style patient prefix only when it is explicit.
    ###########################################################################

    if best is not None:

        site = clinical_patient[
            best
        ].map(
            normalize_site
        )

        method = (
            f"column:{best}"
        )

    else:

        def prefix_site(
            patient_id: str,
        ):

            text = str(
                patient_id
            ).upper()

            if "DFCI" in text:
                return "DFCI"

            if (
                "VICC"
                in text
                or "VANDERBILT"
                in text
            ):
                return "VICC"

            if "MSK" in text:
                return "MSK"

            return None

        site = (
            clinical_patient[
                patient_col
            ]
            .map(
                prefix_site
            )
        )

        method = (
            "patient_id_prefix"
        )

    patient = (
        clinical_patient[
            patient_col
        ]
        .astype(str)
        .str.strip()
    )

    mapping = {}

    for patient_id, current_site in zip(
        patient,
        site,
    ):

        if (
            patient_id
            and current_site
        ):
            mapping[
                patient_id
            ] = current_site

    observed = Counter(
        mapping.values()
    )

    counts = {
        site_name:
            int(
                observed.get(
                    site_name,
                    0
                )
            )
        for site_name
        in ALL_SITES
    }

    ###########################################################################
    # We require exact reconciliation to CKPT7A1. If the ordinary clinical
    # patient file cannot do this, stop instead of opening patient_level.
    ###########################################################################

    if counts != EXPECTED_SITE_COUNTS:

        raise RuntimeError(
            "Could not reproduce CKPT7A1 site partition from "
            "non-quarantined data_clinical_patient.txt. "
            f"method={method} counts={counts}"
        )

    return (
        mapping,
        {
            "patient_column":
                patient_col,

            "method":
                method,

            "counts":
                counts,
        },
    )


def add_site(
    frame: pd.DataFrame,
    site_map: dict[str, str],
) -> tuple[
    pd.DataFrame,
    str,
]:

    patient_col = find_col(
        frame.columns,
        [
            "PATIENT_ID",
            "patient_id",
        ],
        required=True,
    )

    out = frame.copy()

    out[
        "_patient_id"
    ] = (
        out[
            patient_col
        ]
        .astype(str)
        .str.strip()
    )

    out[
        "_site"
    ] = out[
        "_patient_id"
    ].map(
        site_map
    )

    return (
        out,
        patient_col,
    )


###############################################################################
# Imaging semantics
###############################################################################


def audit_imaging(
    imaging: pd.DataFrame,
    site_map: dict[str, str],
) -> dict[str, Any]:

    frame, patient_col = add_site(
        imaging,
        site_map,
    )

    frame = frame[
        frame[
            "_site"
        ].isin(
            ALL_SITES
        )
    ].copy()

    start_col = find_col(
        frame.columns,
        [
            "START_DATE",
            "start_date",
        ],
        required=True,
    )

    stop_col = find_col(
        frame.columns,
        [
            "STOP_DATE",
            "stop_date",
        ],
        required=False,
    )

    scan_number_col = find_col(
        frame.columns,
        [
            "SCAN_NUMBER",
            "scan_number",
        ],
        required=False,
    )

    scan_type_col = find_col(
        frame.columns,
        [
            "IMAGE_SCAN_TYPE",
            "image_scan_type",
        ],
        required=False,
    )

    scan_sites_col = find_col(
        frame.columns,
        [
            "SCAN_SITES",
            "scan_sites",
        ],
        required=False,
    )

    cancer_status_col = find_col(
        frame.columns,
        [
            "CANCER_STATUS",
            "cancer_status",
        ],
        required=False,
    )

    curated_status_col = find_col(
        frame.columns,
        [
            "CURATED_CANCER_STATUS",
            "curated_cancer_status",
        ],
        required=False,
    )

    frame[
        "_scan_day"
    ] = pd.to_numeric(
        frame[
            start_col
        ],
        errors="coerce",
    )

    report: dict[
        str,
        Any
    ] = {
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

        "date_column":
            start_col,

        "date_numeric":
            numeric_summary(
                frame[
                    start_col
                ]
            ),

        "site_counts": {},
    }

    for site in ALL_SITES:

        current = frame[
            frame[
                "_site"
            ]
            == site
        ]

        valid_dates = current[
            "_scan_day"
        ].notna()

        duplicate_patient_days = int(
            current.loc[
                valid_dates,
                [
                    "_patient_id",
                    "_scan_day",
                ]
            ]
            .duplicated()
            .sum()
        )

        report[
            "site_counts"
        ][
            site
        ] = {
            "rows":
                int(
                    len(
                        current
                    )
                ),

            "patients":
                int(
                    current[
                        "_patient_id"
                    ].nunique()
                ),

            "patients_with_numeric_scan_day":
                int(
                    current.loc[
                        valid_dates,
                        "_patient_id",
                    ].nunique()
                ),

            "numeric_date_fraction":
                float(
                    valid_dates.mean()
                )
                if len(
                    current
                )
                else 0.0,

            "unique_patient_days":
                int(
                    current.loc[
                        valid_dates,
                        [
                            "_patient_id",
                            "_scan_day",
                        ]
                    ]
                    .drop_duplicates()
                    .shape[
                        0
                    ]
                ),

            "duplicate_patient_day_rows":
                duplicate_patient_days,
        }

    if scan_number_col:

        report[
            "scan_number"
        ] = {
            site:
                numeric_summary(
                    frame.loc[
                        frame[
                            "_site"
                        ]
                        == site,
                        scan_number_col,
                    ]
                )
            for site in ALL_SITES
        }

    if scan_type_col:

        report[
            "scan_type_values"
        ] = {
            site:
                top_values(
                    frame.loc[
                        frame[
                            "_site"
                        ]
                        == site,
                        scan_type_col,
                    ],
                    60,
                )
            for site in ALL_SITES
        }

    if scan_sites_col:

        report[
            "scan_site_values"
        ] = {
            site:
                top_values(
                    frame.loc[
                        frame[
                            "_site"
                        ]
                        == site,
                        scan_sites_col,
                    ],
                    80,
                )
            for site in ALL_SITES
        }

    if cancer_status_col:

        report[
            "cancer_status_values"
        ] = {
            site:
                top_values(
                    frame.loc[
                        frame[
                            "_site"
                        ]
                        == site,
                        cancer_status_col,
                    ],
                    80,
                )
            for site in ALL_SITES
        }

    if curated_status_col:

        report[
            "curated_cancer_status_values"
        ] = {
            site:
                top_values(
                    frame.loc[
                        frame[
                            "_site"
                        ]
                        == site,
                        curated_status_col,
                    ],
                    80,
                )
            for site in ALL_SITES
        }

    ###########################################################################
    # Candidate direct mapping surface for frozen CKPT5 18-D scan features.
    # We do NOT yet freeze disease-state semantics. We only quantify whether
    # modality, body coverage, and site indicators can be constructed.
    ###########################################################################

    modality_patterns = {
        "CT":
            r"\bCT\b|COMPUTED",

        "PET":
            r"\bPET\b",

        "MR":
            r"\bMRI?\b|MAGNETIC",

        "BONE_SCAN":
            r"BONE",
    }

    site_patterns = {
        "BONE":
            r"BONE|SKELET|RIB|VERTEBR",

        "LIVER":
            r"LIVER",

        "LUNG":
            r"LUNG|PULMON",

        "BRAIN":
            r"BRAIN|CNS|HEAD",

        "LYMPH":
            r"LYMPH|NODE",

        "PLEURA":
            r"PLEURA",
    }

    coverage_patterns = {
        "CHEST":
            r"CHEST|THORAX",

        "ABDOMEN":
            r"ABDOM",

        "PELVIS":
            r"PELV",

        "HEAD":
            r"HEAD|BRAIN",

        "OTHER":
            r".+",
    }

    scan_type_text = (
        frame[
            scan_type_col
        ].fillna("")
        .astype(str)
        .str.upper()
        if scan_type_col
        else pd.Series(
            "",
            index=frame.index,
        )
    )

    site_text = (
        frame[
            scan_sites_col
        ].fillna("")
        .astype(str)
        .str.upper()
        if scan_sites_col
        else pd.Series(
            "",
            index=frame.index,
        )
    )

    feature_surface = {}

    for feature, pattern in modality_patterns.items():

        feature_surface[
            "modality_"
            + feature.lower()
        ] = (
            scan_type_text
            .str.contains(
                pattern,
                regex=True,
                na=False,
            )
        )

    for feature, pattern in site_patterns.items():

        feature_surface[
            "tumor_site_"
            + feature.lower()
        ] = (
            site_text
            .str.contains(
                pattern,
                regex=True,
                na=False,
            )
        )

    for feature, pattern in coverage_patterns.items():

        if feature == "OTHER":

            matched_specific = (
                site_text.str.contains(
                    coverage_patterns[
                        "CHEST"
                    ],
                    regex=True,
                    na=False,
                )
                | site_text.str.contains(
                    coverage_patterns[
                        "ABDOMEN"
                    ],
                    regex=True,
                    na=False,
                )
                | site_text.str.contains(
                    coverage_patterns[
                        "PELVIS"
                    ],
                    regex=True,
                    na=False,
                )
                | site_text.str.contains(
                    coverage_patterns[
                        "HEAD"
                    ],
                    regex=True,
                    na=False,
                )
            )

            value = (
                (site_text.str.len() > 0)
                & ~matched_specific
            )

        else:

            value = (
                site_text
                .str.contains(
                    pattern,
                    regex=True,
                    na=False,
                )
            )

        feature_surface[
            "coverage_"
            + feature.lower()
        ] = value

    report[
        "candidate_scan_feature_surface"
    ] = {}

    for site in ALL_SITES:

        mask = (
            frame[
                "_site"
            ]
            == site
        )

        report[
            "candidate_scan_feature_surface"
        ][
            site
        ] = {
            feature:
                int(
                    values.loc[
                        mask
                    ].sum()
                )
            for feature, values
            in feature_surface.items()
        }

    ###########################################################################
    # W0/W3/W7 input episode counts from imaging timestamps only.
    ###########################################################################

    episode_counts = {}

    for window in (
        0,
        3,
        7,
    ):

        episode_counts[
            str(
                window
            )
        ] = {}

        for site in ALL_SITES:

            current = (
                frame[
                    (
                        frame[
                            "_site"
                        ]
                        == site
                    )
                    & frame[
                        "_scan_day"
                    ].notna()
                ][
                    [
                        "_patient_id",
                        "_scan_day",
                    ]
                ]
                .drop_duplicates()
                .sort_values(
                    [
                        "_patient_id",
                        "_scan_day",
                    ]
                )
            )

            episodes = 0

            for patient_id, group in current.groupby(
                "_patient_id",
                observed=True,
            ):

                days = np.sort(
                    group[
                        "_scan_day"
                    ].to_numpy(
                        dtype=float
                    )
                )

                if len(
                    days
                ) == 0:
                    continue

                episodes += 1

                previous = days[
                    0
                ]

                for value in days[
                    1:
                ]:

                    if (
                        value
                        - previous
                        > window
                    ):

                        episodes += 1

                    previous = value

            episode_counts[
                str(
                    window
                )
            ][
                site
            ] = int(
                episodes
            )

    report[
        "candidate_episode_counts"
    ] = episode_counts

    return report


###############################################################################
# Generic longitudinal table audit
###############################################################################


def audit_longitudinal_table(
    name: str,
    frame: pd.DataFrame,
    site_map: dict[str, str],
) -> dict[str, Any]:

    current, patient_col = add_site(
        frame,
        site_map,
    )

    current = current[
        current[
            "_site"
        ].isin(
            ALL_SITES
        )
    ].copy()

    date_columns = [
        column
        for column in current.columns
        if (
            "date"
            in norm(
                column
            )
            or norm(
                column
            )
            in {
                "start_date",
                "stop_date",
            }
        )
    ]

    report = {
        "name":
            name,

        "rows":
            int(
                len(
                    current
                )
            ),

        "columns":
            list(
                frame.columns
            ),

        "date_columns":
            {
                column:
                    numeric_summary(
                        current[
                            column
                        ]
                    )
                for column in date_columns
            },

        "site_counts":
            {},
    }

    for site in ALL_SITES:

        site_frame = current[
            current[
                "_site"
            ]
            == site
        ]

        report[
            "site_counts"
        ][
            site
        ] = {
            "rows":
                int(
                    len(
                        site_frame
                    )
                ),

            "patients":
                int(
                    site_frame[
                        "_patient_id"
                    ].nunique()
                ),
        }

    ###########################################################################
    # Report actual semantic vocabularies from likely event descriptors.
    ###########################################################################

    interesting = []

    semantic_hints = (
        "event",
        "subtype",
        "treatment",
        "therapy",
        "drug",
        "regimen",
        "lab",
        "test",
        "result",
        "path",
        "hist",
        "receptor",
        "er_",
        "pr_",
        "her2",
        "status",
        "procedure",
        "specimen",
        "site",
        "type",
        "unit",
    )

    for column in frame.columns:

        normalized = norm(
            column
        )

        if any(
            hint
            in normalized
            for hint in semantic_hints
        ):

            interesting.append(
                column
            )

    report[
        "semantic_value_domains"
    ] = {}

    for column in interesting:

        #######################################################################
        # Avoid dumping massive free-text/high-cardinality fields.
        #######################################################################

        unique = (
            current[
                column
            ]
            .astype(str)
            .str.strip()
            .replace(
                "",
                pd.NA,
            )
            .nunique(
                dropna=True
            )
        )

        if unique <= 200:

            report[
                "semantic_value_domains"
            ][
                column
            ] = {
                site:
                    top_values(
                        current.loc[
                            current[
                                "_site"
                            ]
                            == site,
                            column,
                        ],
                        50,
                    )
                for site in ALL_SITES
            }

        else:

            report[
                "semantic_value_domains"
            ][
                column
            ] = {
                "unique_values":
                    int(
                        unique
                    ),

                "high_cardinality":
                    True,
            }

    return report


###############################################################################
# Genomics
###############################################################################


def import_ckpt3(
    repo: Path,
):

    path = (
        repo
        / "scripts"
        / "dynamic_scan"
        / "ckpt3_genie_pretrain.py"
    )

    spec = (
        importlib.util
        .spec_from_file_location(
            "ckpt3_frozen_for_external",
            path,
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):

        raise RuntimeError(
            "Unable to import frozen CKPT3."
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


def panel_alias_candidates(
    value: str,
) -> list[str]:

    text = str(
        value
    ).strip()

    upper = text.upper()

    candidates = [
        text,
        upper,
    ]

    ###########################################################################
    # Common MSK IMPACT spelling normalization.
    ###########################################################################

    match = re.search(
        r"(?:MSK[-_ ]*)?IMPACT[-_ ]*(\d+)",
        upper,
    )

    if match:

        size = match.group(
            1
        )

        candidates.extend(
            [
                f"MSK-IMPACT{size}",
                f"IMPACT{size}",
                f"IMPACT-{size}",
            ]
        )

    return list(
        dict.fromkeys(
            candidates
        )
    )


def audit_genomics(
    repo: Path,
    clinical_sample: pd.DataFrame,
    sequencing: pd.DataFrame,
    gene_matrix: pd.DataFrame,
    mutations: pd.DataFrame,
    site_map: dict[str, str],
    genie_root: Path,
) -> dict[str, Any]:

    sample, patient_col = add_site(
        clinical_sample,
        site_map,
    )

    sample = sample[
        sample[
            "_site"
        ].isin(
            ALL_SITES
        )
    ].copy()

    sample_id_col = find_col(
        sample.columns,
        [
            "SAMPLE_ID",
            "sample_id",
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

    seq_date_col = find_col(
        sample.columns,
        [
            "CPT_SEQ_DATE",
            "SEQ_DATE",
            "SEQUENCING_DATE",
        ],
        required=False,
    )

    report: dict[
        str,
        Any
    ] = {
        "clinical_sample_rows":
            int(
                len(
                    sample
                )
            ),

        "site_counts":
            {},
    }

    for site in ALL_SITES:

        current = sample[
            sample[
                "_site"
            ]
            == site
        ]

        report[
            "site_counts"
        ][
            site
        ] = {
            "samples":
                int(
                    current[
                        sample_id_col
                    ].nunique()
                ),

            "patients":
                int(
                    current[
                        "_patient_id"
                    ].nunique()
                ),
        }

    if panel_col:

        report[
            "panel_values"
        ] = {
            site:
                top_values(
                    sample.loc[
                        sample[
                            "_site"
                        ]
                        == site,
                        panel_col,
                    ],
                    100,
                )
            for site in ALL_SITES
        }

    if seq_date_col:

        report[
            "sequence_date"
        ] = {
            site:
                numeric_summary(
                    sample.loc[
                        sample[
                            "_site"
                        ]
                        == site,
                        seq_date_col,
                    ]
                )
            for site in ALL_SITES
        }

    ###########################################################################
    # Frozen GENIE panel definitions.
    ###########################################################################

    ckpt3 = import_ckpt3(
        repo
    )

    source = ckpt3.discover_sources(
        genie_root
    )

    panel_defs = (
        ckpt3.build_panel_definitions(
            source[
                "gene_panel_definition"
            ]
        )
    )

    lower_lookup = {
        str(
            key
        ).lower():
            key
        for key in panel_defs
    }

    gene_space = [
        line.strip()
        for line in (
            repo
            / "artifacts"
            / "checkpoint3"
            / "prepared"
            / "gene_space_468.txt"
        ).read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    gene_set = set(
        gene_space
    )

    crosswalk = {}

    if panel_col:

        values = sorted(
            set(
                sample[
                    panel_col
                ]
                .astype(str)
                .str.strip()
            )
            - {
                "",
            }
        )

        for value in values:

            resolved = None

            for candidate in panel_alias_candidates(
                value
            ):

                if candidate in panel_defs:

                    resolved = candidate
                    break

                if (
                    candidate.lower()
                    in lower_lookup
                ):

                    resolved = lower_lookup[
                        candidate.lower()
                    ]
                    break

            if resolved is None:

                crosswalk[
                    value
                ] = {
                    "resolved":
                        False,
                }

            else:

                panel_genes = {
                    str(
                        gene
                    ).upper()
                    for gene in panel_defs[
                        resolved
                    ]
                }

                crosswalk[
                    value
                ] = {
                    "resolved":
                        True,

                    "genie_panel":
                        resolved,

                    "panel_gene_count":
                        int(
                            len(
                                panel_genes
                            )
                        ),

                    "frozen_gene_space_coverage":
                        int(
                            len(
                                panel_genes
                                & gene_set
                            )
                        ),
                }

    report[
        "panel_crosswalk"
    ] = crosswalk

    if panel_col:

        resolved_panel = {
            key
            for key, value
            in crosswalk.items()
            if value.get(
                "resolved"
            )
        }

        report[
            "panel_resolution_by_site"
        ] = {}

        for site in ALL_SITES:

            current = sample[
                sample[
                    "_site"
                ]
                == site
            ]

            resolved = current[
                panel_col
            ].astype(str).isin(
                resolved_panel
            )

            report[
                "panel_resolution_by_site"
            ][
                site
            ] = {
                "samples":
                    int(
                        len(
                            current
                        )
                    ),

                "resolved":
                    int(
                        resolved.sum()
                    ),

                "fraction":
                    float(
                        resolved.mean()
                    )
                    if len(
                        current
                    )
                    else 0.0,
            }

    ###########################################################################
    # Mutation patient/sample mapping.
    ###########################################################################

    mutation_sample_col = find_col(
        mutations.columns,
        [
            "Tumor_Sample_Barcode",
            "SAMPLE_ID",
            "sample_id",
        ],
        required=False,
    )

    mutation_patient_col = find_col(
        mutations.columns,
        [
            "PATIENT_ID",
            "patient_id",
        ],
        required=False,
    )

    sample_to_patient = {
        str(
            sample_id
        ).strip():
            str(
                patient_id
            ).strip()
        for sample_id, patient_id
        in zip(
            sample[
                sample_id_col
            ],
            sample[
                "_patient_id"
            ],
        )
        if (
            str(
                sample_id
            ).strip()
        )
    }

    mutation_site = []

    if mutation_patient_col:

        mutation_patients = (
            mutations[
                mutation_patient_col
            ]
            .astype(str)
            .str.strip()
        )

        mutation_site = mutation_patients.map(
            site_map
        )

    elif mutation_sample_col:

        mutation_patients = (
            mutations[
                mutation_sample_col
            ]
            .astype(str)
            .str.strip()
            .map(
                sample_to_patient
            )
        )

        mutation_site = mutation_patients.map(
            site_map
        )

    else:

        mutation_patients = pd.Series(
            "",
            index=mutations.index,
        )

        mutation_site = pd.Series(
            None,
            index=mutations.index,
        )

    report[
        "mutation_rows_by_site"
    ] = {
        site:
            int(
                (
                    mutation_site
                    == site
                ).sum()
            )
        for site in ALL_SITES
    }

    report[
        "mutation_patients_by_site"
    ] = {
        site:
            int(
                mutation_patients.loc[
                    mutation_site
                    == site
                ]
                .nunique()
            )
        for site in ALL_SITES
    }

    ###########################################################################
    # Gene matrix / sequencing schema only plus external row coverage.
    ###########################################################################

    report[
        "gene_matrix_columns"
    ] = list(
        gene_matrix.columns
    )

    report[
        "gene_matrix_rows"
    ] = int(
        len(
            gene_matrix
        )
    )

    sequencing_audit = (
        audit_longitudinal_table(
            "sequencing",
            sequencing,
            site_map,
        )
    )

    report[
        "sequencing"
    ] = sequencing_audit

    return report


###############################################################################
# Cross-table temporal coverage
###############################################################################


def temporal_coverage_summary(
    audits: dict[str, dict[str, Any]],
) -> dict[str, Any]:

    result = {}

    for site in ALL_SITES:

        result[
            site
        ] = {}

        for name, audit in audits.items():

            site_count = (
                audit
                .get(
                    "site_counts",
                    {}
                )
                .get(
                    site,
                    {}
                )
            )

            result[
                site
            ][
                name
            ] = {
                "patients":
                    int(
                        site_count.get(
                            "patients",
                            0,
                        )
                    ),

                "rows":
                    int(
                        site_count.get(
                            "rows",
                            0,
                        )
                    ),
            }

    return result


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
        "--genie-root",
        default=(
            "data/external_sources/"
            "genie_20_0_public"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="artifacts/checkpoint7a2",
    )

    args = parser.parse_args()

    repo = Path(
        args.repo_root
    ).resolve()

    bpc_root = (
        repo
        / args.bpc_root
    ).resolve()

    genie_root = (
        repo
        / args.genie_root
    ).resolve()

    out = (
        repo
        / args.output_dir
    ).resolve()

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    discovery = json.loads(
        (
            repo
            / "artifacts"
            / "checkpoint7a1"
            / "discovery.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    inventory = pd.read_parquet(
        repo
        / "artifacts"
        / "checkpoint7a1"
        / "schema_inventory.parquet"
    )

    if (
        discovery[
            "status"
        ]
        != "READY_FOR_EXTERNAL_ADAPTER_IMPLEMENTATION"
    ):

        raise RuntimeError(
            "CKPT7A1 is not adapter-ready."
        )

    if (
        discovery[
            "policy"
        ][
            "external_outcomes_opened"
        ]
    ):

        raise RuntimeError(
            "CKPT7A1 indicates external outcomes were opened."
        )

    guard = AccessGuard(
        bpc_root,
        inventory,
    )

    ###########################################################################
    # Load predictor sources only.
    ###########################################################################

    frames = {}

    for name, relative in PREDICTOR_TABLES.items():

        print(
            "[CKPT7A2_READ]",
            name,
            relative,
            flush=True,
        )

        frames[
            name
        ] = guard.read_predictor(
            relative
        )

    ###########################################################################
    # Site mapping.
    ###########################################################################

    (
        site_map,
        site_audit,
    ) = build_site_map(
        frames[
            "clinical_patient"
        ]
    )

    print(
        "[CKPT7A2_SITE_PARTITION_PASS]",
        site_audit[
            "counts"
        ],
        flush=True,
    )

    ###########################################################################
    # Imaging semantics.
    ###########################################################################

    imaging_audit = audit_imaging(
        frames[
            "imaging"
        ],
        site_map,
    )

    ###########################################################################
    # Longitudinal predictor surfaces.
    ###########################################################################

    longitudinal = {}

    for name in (
        "treatment",
        "medonc",
        "pathology",
        "diagnosis",
        "lab",
    ):

        longitudinal[
            name
        ] = (
            audit_longitudinal_table(
                name,
                frames[
                    name
                ],
                site_map,
            )
        )

    ###########################################################################
    # Genomics.
    ###########################################################################

    genomic_audit = audit_genomics(
        repo,
        frames[
            "clinical_sample"
        ],
        frames[
            "sequencing"
        ],
        frames[
            "gene_matrix"
        ],
        frames[
            "mutations"
        ],
        site_map,
        genie_root,
    )

    ###########################################################################
    # Adapter-readiness logic.
    ###########################################################################

    external_imaging_patients = {
        site:
            int(
                imaging_audit[
                    "site_counts"
                ][
                    site
                ][
                    "patients"
                ]
            )
        for site in EXTERNAL_SITES
    }

    expected_external = {
        "DFCI":
            428,

        "VICC":
            173,
    }

    imaging_coverage = {
        site:
            float(
                external_imaging_patients[
                    site
                ]
                / expected_external[
                    site
                ]
            )
        for site in EXTERNAL_SITES
    }

    imaging_dates_ok = all(
        imaging_audit[
            "site_counts"
        ][
            site
        ][
            "numeric_date_fraction"
        ]
        >= 0.98
        for site in EXTERNAL_SITES
    )

    imaging_population_ok = all(
        imaging_coverage[
            site
        ]
        >= 0.95
        for site in EXTERNAL_SITES
    )

    temporal_core_ok = all(
        longitudinal[
            "treatment"
        ][
            "site_counts"
        ][
            site
        ][
            "patients"
        ]
        >= int(
            expected_external[
                site
            ]
            * 0.80
        )
        for site in EXTERNAL_SITES
    )

    panel_resolution = (
        genomic_audit.get(
            "panel_resolution_by_site",
            {}
        )
    )

    genomic_ok = all(
        panel_resolution.get(
            site,
            {}
        ).get(
            "fraction",
            0.0,
        )
        >= 0.80
        for site in EXTERNAL_SITES
    )

    ###########################################################################
    # Scan-state mapping is intentionally NOT automatically frozen here.
    #
    # If disease-state vocabularies exist, we stop for one scientific semantic
    # decision before tensor generation.
    ###########################################################################

    state_domains_present = bool(
        imaging_audit.get(
            "curated_cancer_status_values"
        )
        or imaging_audit.get(
            "cancer_status_values"
        )
    )

    if not (
        imaging_dates_ok
        and imaging_population_ok
        and temporal_core_ok
        and genomic_ok
    ):

        status = (
            "NEEDS_EXTERNAL_INPUT_RESOLUTION"
        )

    elif state_domains_present:

        status = (
            "READY_FOR_SCAN_STATE_CROSSWALK_DECISION"
        )

    else:

        status = (
            "NEEDS_EXTERNAL_SCAN_STATE_SOURCE"
        )

    ###########################################################################
    # Blinding assertion.
    ###########################################################################

    accessed = {
        item[
            "relative_path"
        ]
        for item in guard.access_log
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

    ###########################################################################
    # Save audit.
    ###########################################################################

    temporal_summary = (
        temporal_coverage_summary(
            longitudinal
        )
    )

    report = {
        "status":
            status,

        "external_outcomes_opened":
            False,

        "outcome_distribution_inspected":
            False,

        "quarantined_files":
            sorted(
                QUARANTINED
            ),

        "accessed_predictor_tables":
            guard.access_log,

        "site_partition":
            site_audit,

        "imaging":
            imaging_audit,

        "longitudinal":
            longitudinal,

        "temporal_coverage":
            temporal_summary,

        "genomics":
            genomic_audit,

        "readiness": {
            "imaging_population_fraction":
                imaging_coverage,

            "imaging_dates_ok":
                imaging_dates_ok,

            "temporal_core_ok":
                temporal_core_ok,

            "genomic_panel_resolution_ok":
                genomic_ok,

            "scan_state_semantics_not_yet_frozen":
                True,
        },

        "next_decision":
            (
                "Freeze exact BPC imaging disease-state crosswalk "
                "and event-to-CKPT4 semantic mapping, then construct "
                "DFCI/VICC predictor tensors without outcomes."
            ),
    }

    atomic_json(
        out
        / "semantic_audit.json",
        report,
    )

    ###########################################################################
    # Compact table for later adapter implementation.
    ###########################################################################

    rows = []

    for site in ALL_SITES:

        rows.append(
            {
                "site":
                    site,

                "patients_expected":
                    EXPECTED_SITE_COUNTS[
                        site
                    ],

                "imaging_patients":
                    imaging_audit[
                        "site_counts"
                    ][
                        site
                    ][
                        "patients"
                    ],

                "imaging_rows":
                    imaging_audit[
                        "site_counts"
                    ][
                        site
                    ][
                        "rows"
                    ],

                "imaging_numeric_date_fraction":
                    imaging_audit[
                        "site_counts"
                    ][
                        site
                    ][
                        "numeric_date_fraction"
                    ],

                "treatment_patients":
                    longitudinal[
                        "treatment"
                    ][
                        "site_counts"
                    ][
                        site
                    ][
                        "patients"
                    ],

                "medonc_patients":
                    longitudinal[
                        "medonc"
                    ][
                        "site_counts"
                    ][
                        site
                    ][
                        "patients"
                    ],

                "pathology_patients":
                    longitudinal[
                        "pathology"
                    ][
                        "site_counts"
                    ][
                        site
                    ][
                        "patients"
                    ],

                "lab_patients":
                    longitudinal[
                        "lab"
                    ][
                        "site_counts"
                    ][
                        site
                    ][
                        "patients"
                    ],

                "genomic_samples":
                    genomic_audit[
                        "site_counts"
                    ][
                        site
                    ][
                        "samples"
                    ],

                "genomic_panel_resolved_fraction":
                    genomic_audit.get(
                        "panel_resolution_by_site",
                        {}
                    ).get(
                        site,
                        {}
                    ).get(
                        "fraction",
                        0.0,
                    ),
            }
        )

    coverage_frame = pd.DataFrame(
        rows
    )

    atomic_parquet(
        out
        / "center_input_coverage.parquet",
        coverage_frame,
    )

    ###########################################################################
    # Human-readable audit
    ###########################################################################

    lines = [
        "# CKPT7A2 — Outcome-blinded external semantic audit",
        "",
        f"Status: **{status}**",
        "",
        "External outcomes opened: **NO**",
        "",
        "## Imaging",
        "",
    ]

    for site in EXTERNAL_SITES:

        current = imaging_audit[
            "site_counts"
        ][
            site
        ]

        lines.extend(
            [
                f"### {site}",
                "",
                f"- patients with imaging: {current['patients']}",
                f"- rows: {current['rows']}",
                (
                    "- numeric scan-date fraction: "
                    f"{current['numeric_date_fraction']}"
                ),
                (
                    "- unique patient scan-days: "
                    f"{current['unique_patient_days']}"
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Current imaging-status vocabularies",
            "",
            "These values are predictors at the scan landmark, not future outcomes.",
            "",
            "```json",
            json.dumps(
                {
                    "CANCER_STATUS":
                        imaging_audit.get(
                            "cancer_status_values"
                        ),

                    "CURATED_CANCER_STATUS":
                        imaging_audit.get(
                            "curated_cancer_status_values"
                        ),
                },
                indent=2,
            ),
            "```",
            "",
            "## Genomic panel crosswalk",
            "",
            "```json",
            json.dumps(
                genomic_audit.get(
                    "panel_crosswalk",
                    {}
                ),
                indent=2,
            ),
            "```",
            "",
            "## Outcome quarantine",
            "",
            "No row values were read from:",
        ]
    )

    for relative in sorted(
        QUARANTINED
    ):

        lines.append(
            f"- `{relative}`"
        )

    atomic_text(
        out
        / "audit.md",
        "\n".join(
            lines
        )
        + "\n",
    )

    ###########################################################################
    # Handoff
    ###########################################################################

    handoff = {
        "checkpoint":
            "7A2",

        "name":
            "outcome_blinded_external_semantic_audit",

        "status":
            status,

        "external_outcomes_opened":
            False,

        "dfci_imaging_patients":
            external_imaging_patients[
                "DFCI"
            ],

        "vicc_imaging_patients":
            external_imaging_patients[
                "VICC"
            ],

        "dfci_imaging_fraction":
            imaging_coverage[
                "DFCI"
            ],

        "vicc_imaging_fraction":
            imaging_coverage[
                "VICC"
            ],

        "candidate_episode_counts":
            imaging_audit[
                "candidate_episode_counts"
            ],

        "panel_crosswalk":
            genomic_audit.get(
                "panel_crosswalk",
                {}
            ),

        "next_action":
            (
                "Freeze scan-state and event semantic crosswalks, "
                "then build complete outcome-blinded DFCI/VICC model "
                "inputs and run the frozen candidate before opening outcomes."
            ),
    }

    atomic_json(
        repo
        / "artifacts"
        / "handoff"
        / "checkpoint_07A2.json",
        handoff,
    )

    ###########################################################################
    # Terminal packet
    ###########################################################################

    print("")
    print(
        "========== CKPT7A2 SUMMARY =========="
    )

    print(
        f"status={status}"
    )

    print(
        "external_outcomes_opened=False"
    )

    print(
        "site_partition="
        f"{site_audit['counts']}"
    )

    print(
        "imaging_patients="
        f"{external_imaging_patients}"
    )

    print(
        "imaging_population_fraction="
        f"{imaging_coverage}"
    )

    print(
        "imaging_dates_ok="
        f"{imaging_dates_ok}"
    )

    print(
        "temporal_core_ok="
        f"{temporal_core_ok}"
    )

    print(
        "genomic_panel_resolution_ok="
        f"{genomic_ok}"
    )

    print(
        "candidate_episode_counts="
        f"{imaging_audit['candidate_episode_counts']}"
    )

    print(
        "handoff="
        "artifacts/handoff/checkpoint_07A2.json"
    )

    print(
        "========== CKPT7A2 SUMMARY END =========="
    )

    print("")
    print(
        "========== CKPT7A2 DECISION PACKET =========="
    )

    print(
        "cancer_status_values="
        f"{imaging_audit.get('cancer_status_values')}"
    )

    print(
        "curated_cancer_status_values="
        f"{imaging_audit.get('curated_cancer_status_values')}"
    )

    print(
        "scan_type_values="
        f"{imaging_audit.get('scan_type_values')}"
    )

    print(
        "scan_site_values="
        f"{imaging_audit.get('scan_site_values')}"
    )

    print(
        "candidate_scan_feature_surface="
        f"{imaging_audit.get('candidate_scan_feature_surface')}"
    )

    print(
        "temporal_coverage="
        f"{temporal_summary}"
    )

    print(
        "treatment_semantics="
        f"{longitudinal['treatment']['semantic_value_domains']}"
    )

    print(
        "medonc_semantics="
        f"{longitudinal['medonc']['semantic_value_domains']}"
    )

    print(
        "pathology_semantics="
        f"{longitudinal['pathology']['semantic_value_domains']}"
    )

    print(
        "lab_semantics="
        f"{longitudinal['lab']['semantic_value_domains']}"
    )

    print(
        "genomic_panel_values="
        f"{genomic_audit.get('panel_values')}"
    )

    print(
        "genomic_panel_crosswalk="
        f"{genomic_audit.get('panel_crosswalk')}"
    )

    print(
        "genomic_panel_resolution_by_site="
        f"{genomic_audit.get('panel_resolution_by_site')}"
    )

    print(
        "mutation_rows_by_site="
        f"{genomic_audit.get('mutation_rows_by_site')}"
    )

    print(
        "mutation_patients_by_site="
        f"{genomic_audit.get('mutation_patients_by_site')}"
    )

    print(
        "outcome_quarantine_access_violation=False"
    )

    print(
        "========== CKPT7A2 DECISION PACKET END =========="
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
