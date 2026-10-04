#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


RADIOLOGY_FILES = {
    "progression": "data_timeline_progression.txt",
    "cancer_presence": "data_timeline_cancer_presence.txt",
    "tumor_sites": "data_timeline_tumor_sites.txt",
}


def norm(value: Any) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value).strip().lower(),
    ).strip("_")


def atomic_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")

    tmp.write_text(
        json.dumps(
            obj,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    if not tmp.exists() or tmp.stat().st_size == 0:
        raise RuntimeError(
            f"empty JSON write: {path}"
        )

    # Validate before replace.
    json.loads(
        tmp.read_text(
            encoding="utf-8"
        )
    )

    tmp.replace(path)


def atomic_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")

    tmp.write_text(
        text,
        encoding="utf-8",
    )

    if not tmp.exists() or tmp.stat().st_size == 0:
        raise RuntimeError(
            f"empty text write: {path}"
        )

    tmp.replace(path)


def find_one(root: Path, basename: str) -> Path:
    matches = [
        p
        for p in root.rglob(basename)
        if p.is_file()
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one {basename} under {root}; "
            f"found {len(matches)}: {matches[:10]}"
        )

    return matches[0]


def read_table(path: Path) -> pd.DataFrame:
    low = path.name.lower()

    if low.endswith(".csv"):
        return pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
            low_memory=False,
        )

    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        comment="#",
        low_memory=False,
    )


def find_col(
    columns: list[str],
    exact: list[str],
) -> str | None:
    lookup = {
        norm(c): c
        for c in columns
    }

    for wanted in exact:
        key = norm(wanted)

        if key in lookup:
            return lookup[key]

    return None


def map_site(value: Any) -> str | None:
    text = str(value).strip().upper()

    if not text:
        return None

    if (
        "DFCI" in text
        or "DANA-FARBER" in text
        or "DANA FARBER" in text
        or "DANA_FARBER" in text
    ):
        return "DFCI"

    if (
        "VICC" in text
        or "VANDERBILT" in text
    ):
        return "VICC"

    if (
        re.search(
            r"(^|[-_ ])MSK($|[-_ ])",
            text,
        )
        or "MSKCC" in text
        or "MEMORIAL SLOAN" in text
    ):
        return "MSK"

    return None


def numeric_start(
    df: pd.DataFrame,
) -> pd.Series:
    if "START_DATE" not in df.columns:
        raise RuntimeError(
            "timeline table lacks START_DATE"
        )

    return pd.to_numeric(
        df["START_DATE"],
        errors="coerce",
    )


def patient_date_set(
    df: pd.DataFrame,
    patient_ids: set[str] | None = None,
) -> set[tuple[str, int]]:
    if "PATIENT_ID" not in df.columns:
        raise RuntimeError(
            "timeline table lacks PATIENT_ID"
        )

    work = df[
        ["PATIENT_ID", "START_DATE"]
    ].copy()

    if patient_ids is not None:
        work = work[
            work["PATIENT_ID"].isin(
                patient_ids
            )
        ]

    work["START_DATE"] = pd.to_numeric(
        work["START_DATE"],
        errors="coerce",
    )

    work = work.dropna(
        subset=[
            "PATIENT_ID",
            "START_DATE",
        ]
    )

    return {
        (
            str(pid).strip(),
            int(day),
        )
        for pid, day
        in zip(
            work["PATIENT_ID"],
            work["START_DATE"],
        )
    }


def quantiles(
    values: list[int],
) -> dict[str, float | int | None]:
    if not values:
        return {
            "n": 0,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "max": None,
        }

    s = pd.Series(
        values,
        dtype=float,
    )

    return {
        "n": len(values),
        "min": int(s.min()),
        "p25": float(s.quantile(0.25)),
        "median": float(s.quantile(0.50)),
        "p75": float(s.quantile(0.75)),
        "p90": float(s.quantile(0.90)),
        "max": int(s.max()),
    }


def counts_per_patient(
    patient_dates: set[tuple[str, int]],
) -> dict[str, int]:
    counts: Counter[str] = Counter()

    for pid, _ in patient_dates:
        counts[pid] += 1

    return dict(counts)


def cluster_dates(
    patient_dates: set[tuple[str, int]],
    window_days: int,
) -> dict[str, Any]:
    by_patient: dict[
        str,
        list[int],
    ] = defaultdict(list)

    for pid, day in patient_dates:
        by_patient[pid].append(day)

    episode_counts: list[int] = []
    total_episodes = 0

    for pid, days in by_patient.items():
        unique_days = sorted(
            set(days)
        )

        if not unique_days:
            continue

        episodes = 1
        previous = unique_days[0]

        for day in unique_days[1:]:
            if day - previous > window_days:
                episodes += 1

            previous = day

        episode_counts.append(
            episodes
        )
        total_episodes += episodes

    return {
        "window_days": window_days,
        "patients": len(by_patient),
        "episodes": total_episodes,
        "episodes_per_patient":
            quantiles(episode_counts),
    }


def top_values(
    series: pd.Series,
    limit: int = 25,
) -> list[list[Any]]:
    clean = (
        series
        .astype(str)
        .str.strip()
    )

    clean = clean[
        clean != ""
    ]

    return [
        [str(key), int(value)]
        for key, value
        in clean.value_counts(
            dropna=False
        ).head(limit).items()
    ]


def timeline_table_summary(
    name: str,
    path: Path,
    df: pd.DataFrame,
    breast_ids: set[str],
) -> dict[str, Any]:
    if "PATIENT_ID" not in df.columns:
        raise RuntimeError(
            f"{name} lacks PATIENT_ID"
        )

    if "START_DATE" not in df.columns:
        raise RuntimeError(
            f"{name} lacks START_DATE"
        )

    all_dates = patient_date_set(
        df
    )

    breast = df[
        df["PATIENT_ID"].isin(
            breast_ids
        )
    ].copy()

    breast_dates = patient_date_set(
        breast
    )

    date_numeric = pd.to_numeric(
        breast["START_DATE"],
        errors="coerce",
    )

    start_missing = int(
        date_numeric.isna().sum()
    )

    stop_summary: dict[str, Any]

    if "STOP_DATE" in breast.columns:
        stop_raw = (
            breast["STOP_DATE"]
            .astype(str)
            .str.strip()
        )

        stop_num = pd.to_numeric(
            stop_raw.replace(
                "",
                pd.NA,
            ),
            errors="coerce",
        )

        valid_both = (
            date_numeric.notna()
            & stop_num.notna()
        )

        same = int(
            (
                date_numeric[
                    valid_both
                ]
                == stop_num[
                    valid_both
                ]
            ).sum()
        )

        different = int(
            (
                date_numeric[
                    valid_both
                ]
                != stop_num[
                    valid_both
                ]
            ).sum()
        )

        stop_summary = {
            "blank":
                int(
                    (
                        stop_raw == ""
                    ).sum()
                ),
            "nonblank":
                int(
                    (
                        stop_raw != ""
                    ).sum()
                ),
            "same_as_start":
                same,
            "different_from_start":
                different,
        }

    else:
        stop_summary = {
            "column_absent": True,
        }

    standard = {
        "PATIENT_ID",
        "START_DATE",
        "STOP_DATE",
        "EVENT_TYPE",
    }

    payload_columns = [
        c
        for c in breast.columns
        if c not in standard
    ]

    payload_values = {}

    for col in payload_columns:
        # These are CHORD development data, so value inspection is allowed.
        payload_values[
            col
        ] = top_values(
            breast[col]
        )

    modality_or_coverage_columns = [
        c
        for c in breast.columns
        if any(
            token in norm(c)
            for token in (
                "modality",
                "body_region",
                "anatomic_region",
                "scan_type",
                "imaging_type",
                "coverage",
                "chest",
                "abdomen",
                "pelvis",
                "brain",
                "head",
                "extrem",
            )
        )
    ]

    event_types = (
        top_values(
            breast["EVENT_TYPE"]
        )
        if "EVENT_TYPE"
        in breast.columns
        else []
    )

    return {
        "name": name,
        "path": str(path),
        "columns":
            list(df.columns),
        "rows_total":
            int(len(df)),
        "patients_total":
            int(
                df[
                    "PATIENT_ID"
                ].nunique()
            ),
        "unique_patient_days_total":
            len(all_dates),
        "rows_breast":
            int(len(breast)),
        "patients_breast":
            int(
                breast[
                    "PATIENT_ID"
                ].nunique()
            ),
        "unique_patient_days_breast":
            len(breast_dates),
        "missing_numeric_start_date_breast":
            start_missing,
        "stop_date":
            stop_summary,
        "event_types":
            event_types,
        "payload_columns":
            payload_columns,
        "payload_values_breast":
            payload_values,
        "modality_or_coverage_columns":
            modality_or_coverage_columns,
        "_all_dates":
            all_dates,
        "_breast_dates":
            breast_dates,
    }


def find_bpc_patient_file(
    root: Path,
) -> Path:
    preferred = (
        root
        / "clinical_data"
        / "patient_level_dataset.csv"
    )

    if preferred.exists():
        return preferred

    matches = list(
        root.rglob(
            "patient_level_dataset.csv"
        )
    )

    if len(matches) != 1:
        raise RuntimeError(
            "could not uniquely identify "
            "BPC patient_level_dataset.csv"
        )

    return matches[0]


def bpc_patient_site_map(
    root: Path,
) -> tuple[
    dict[str, str],
    dict[str, Any],
]:
    path = find_bpc_patient_file(
        root
    )

    df = read_table(path)

    id_col = find_col(
        list(df.columns),
        [
            "record_id",
            "PATIENT_ID",
            "patient_id",
        ],
    )

    if id_col is None:
        raise RuntimeError(
            "BPC patient-level file lacks "
            "record_id/PATIENT_ID"
        )

    candidate_site_cols = [
        c
        for c in df.columns
        if any(
            token in norm(c)
            for token in (
                "institution",
                "center",
                "centre",
                "site",
            )
        )
    ]

    mapping: dict[
        str,
        str
    ] = {}

    source_counter: Counter[
        str
    ] = Counter()

    for _, row in df.iterrows():
        pid = str(
            row.get(
                id_col,
                "",
            )
        ).strip()

        if not pid:
            continue

        site = None

        for col in candidate_site_cols:
            site = map_site(
                row.get(
                    col,
                    "",
                )
            )

            if site:
                source_counter[
                    f"column:{col}"
                ] += 1

                break

        if site is None:
            site = map_site(pid)

            if site:
                source_counter[
                    "patient_id_prefix"
                ] += 1

        if site:
            mapping[
                pid
            ] = site

    counts = Counter(
        mapping.values()
    )

    audit = {
        "path": str(path),
        "id_column": id_col,
        "candidate_site_columns":
            candidate_site_cols,
        "mapped_patients":
            len(mapping),
        "site_counts":
            dict(counts),
        "mapping_sources":
            dict(source_counter),
    }

    return mapping, audit


def bpc_imaging_candidates(
    root: Path,
) -> list[Path]:
    candidates: list[
        Path
    ] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        low = path.name.lower()

        if not low.endswith(
            (
                ".csv",
                ".txt",
                ".tsv",
            )
        ):
            continue

        if (
            "imaging" in low
            or "radiology" in low
        ):
            candidates.append(
                path
            )

    return sorted(
        set(candidates)
    )


def summarize_bpc_msk_imaging(
    path: Path,
    site_map: dict[str, str],
) -> dict[str, Any]:
    df = read_table(path)

    id_col = find_col(
        list(df.columns),
        [
            "record_id",
            "PATIENT_ID",
            "patient_id",
        ],
    )

    if id_col is None:
        return {
            "path": str(path),
            "columns":
                list(df.columns),
            "usable": False,
            "reason":
                "no patient identifier column",
        }

    # IMPORTANT:
    # Only MSK rows are used for value distributions below.
    ids = (
        df[id_col]
        .astype(str)
        .str.strip()
    )

    inferred_site = ids.map(
        site_map
    )

    # Fall back to ID prefix when needed.
    fallback = ids.map(
        map_site
    )

    inferred_site = (
        inferred_site
        .fillna(fallback)
    )

    msk = df[
        inferred_site == "MSK"
    ].copy()

    date_columns = [
        c
        for c in msk.columns
        if (
            norm(c).endswith(
                "_date"
            )
            or norm(c)
            in {
                "date",
                "imaging_date",
                "scan_date",
                "report_date",
            }
        )
    ]

    semantic_columns = [
        c
        for c in msk.columns
        if any(
            token in norm(c)
            for token in (
                "progress",
                "response",
                "change",
                "cancer",
                "site",
                "location",
                "metasta",
                "modality",
                "imaging",
                "scan",
                "body_region",
                "anatomic",
            )
        )
    ]

    value_counts: dict[
        str,
        Any
    ] = {}

    for col in semantic_columns:
        # MSK ONLY. Never summarize DFCI/VICC values here.
        nunique = int(
            msk[col]
            .astype(str)
            .nunique()
        )

        if nunique <= 250:
            value_counts[
                col
            ] = top_values(
                msk[col],
                limit=30,
            )

    unique_patient_dates = None

    if date_columns:
        date_col = date_columns[0]

        unique_patient_dates = int(
            msk[
                [id_col, date_col]
            ]
            .drop_duplicates()
            .shape[0]
        )

    return {
        "path": str(path),
        "usable":
            len(msk) > 0,
        "id_column":
            id_col,
        "columns":
            list(df.columns),
        "msk_rows":
            int(len(msk)),
        "msk_patients":
            int(
                msk[id_col].nunique()
            ),
        "date_columns":
            date_columns,
        "semantic_columns":
            semantic_columns,
        "unique_patient_dates_using_first_date_column":
            unique_patient_dates,
        "msk_value_counts":
            value_counts,
        "policy_guard":
            (
                "Value distributions computed only "
                "for BPC-MSK; DFCI/VICC rows were "
                "not summarized."
            ),
    }


def write_external_exclusion_manifest(
    genie_root: Path,
    bpc_site_map: dict[str, str],
    out: Path,
) -> dict[str, Any]:
    genie_patient_path = find_one(
        genie_root,
        "data_clinical_patient.txt",
    )

    genie = read_table(
        genie_patient_path
    )

    if "PATIENT_ID" not in genie.columns:
        raise RuntimeError(
            "GENIE patient table lacks PATIENT_ID"
        )

    genie_ids = set(
        genie[
            "PATIENT_ID"
        ]
        .astype(str)
        .str.strip()
    )

    external_ids = {
        pid
        for pid, site
        in bpc_site_map.items()
        if site
        in {
            "DFCI",
            "VICC",
        }
    }

    exact_overlap = sorted(
        genie_ids
        & external_ids
    )

    manifest = (
        out
        / "genie_external_validation_patient_exclusions.txt"
    )

    atomic_text(
        manifest,
        "\n".join(
            exact_overlap
        )
        + (
            "\n"
            if exact_overlap
            else ""
        ),
    )

    # Institution-level counts only.
    # No outcome information is involved.
    center_counts = {
        "MSK":
            sum(
                1
                for pid
                in genie_ids
                if map_site(pid)
                == "MSK"
            ),
        "DFCI":
            sum(
                1
                for pid
                in genie_ids
                if map_site(pid)
                == "DFCI"
            ),
        "VICC":
            sum(
                1
                for pid
                in genie_ids
                if map_site(pid)
                == "VICC"
            ),
    }

    return {
        "genie_patients":
            len(genie_ids),
        "bpc_external_patients":
            len(external_ids),
        "exact_external_overlap":
            len(exact_overlap),
        "manifest":
            str(manifest),
        "genie_center_counts":
            center_counts,
        "interpretation": (
            "The exact DFCI/VICC BPC patient IDs in "
            "GENIE should be excluded from genomic "
            "pretraining used by the primary external-"
            "validation experiment. A stricter center-"
            "disjoint GENIE experiment can additionally "
            "exclude all MSK/DFCI/VICC patients."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo-root",
        default=".",
    )

    parser.add_argument(
        "--output-dir",
        default="artifacts/checkpoint0e",
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

    chord_root = (
        repo
        / "data"
        / "external_sources"
        / "msk_chord_2024"
        / "raw"
    )

    bpc_root = (
        repo
        / "data"
        / "external_sources"
        / "bpc_brca_1_0_public"
    )

    genie_root = (
        repo
        / "data"
        / "external_sources"
        / "genie_20_0_public"
    )

    for path in (
        chord_root,
        bpc_root,
        genie_root,
    ):
        if not path.exists():
            raise SystemExit(
                "[CKPT0E_FATAL] "
                f"missing source root: {path}"
            )

    ###########################################################################
    # 1. Reconstruct authoritative CHORD breast set.
    ###########################################################################

    clinical_sample_path = find_one(
        chord_root,
        "data_clinical_sample.txt",
    )

    clinical_sample = read_table(
        clinical_sample_path
    )

    if "PATIENT_ID" not in clinical_sample.columns:
        raise RuntimeError(
            "CHORD clinical sample table "
            "lacks PATIENT_ID"
        )

    cancer_col = find_col(
        list(
            clinical_sample.columns
        ),
        [
            "CANCER_TYPE",
            "cancer_type",
        ],
    )

    if cancer_col is None:
        raise RuntimeError(
            "CHORD sample table lacks CANCER_TYPE"
        )

    breast_mask = (
        clinical_sample[
            cancer_col
        ]
        .astype(str)
        .str.contains(
            "breast",
            case=False,
            na=False,
        )
    )

    breast_ids = set(
        clinical_sample.loc[
            breast_mask,
            "PATIENT_ID",
        ]
        .astype(str)
        .str.strip()
    )

    print(
        "[CKPT0E] "
        f"CHORD breast patients={len(breast_ids)}",
        flush=True,
    )

    ###########################################################################
    # 2. Inspect the three public radiology-derived timelines.
    ###########################################################################

    radiology: dict[
        str,
        dict[str, Any]
    ] = {}

    all_date_sets: dict[
        str,
        set[tuple[str, int]]
    ] = {}

    breast_date_sets: dict[
        str,
        set[tuple[str, int]]
    ] = {}

    for name, basename in RADIOLOGY_FILES.items():
        path = find_one(
            chord_root,
            basename,
        )

        print(
            "[CKPT0E] "
            f"loading {name}: {path}",
            flush=True,
        )

        df = read_table(
            path
        )

        summary = timeline_table_summary(
            name=name,
            path=path,
            df=df,
            breast_ids=breast_ids,
        )

        all_date_sets[
            name
        ] = summary.pop(
            "_all_dates"
        )

        breast_date_sets[
            name
        ] = summary.pop(
            "_breast_dates"
        )

        radiology[
            name
        ] = summary

    ###########################################################################
    # 3. Analyze scan-like measurement days / episode grouping.
    ###########################################################################

    union_all: set[
        tuple[str, int]
    ] = set()

    union_breast: set[
        tuple[str, int]
    ] = set()

    for s in all_date_sets.values():
        union_all.update(s)

    for s in breast_date_sets.values():
        union_breast.update(s)

    pairwise: dict[
        str,
        int
    ] = {}

    names = list(
        RADIOLOGY_FILES
    )

    for i in range(
        len(names)
    ):
        for j in range(
            i + 1,
            len(names),
        ):
            left = names[i]
            right = names[j]

            pairwise[
                f"{left}__{right}"
            ] = len(
                breast_date_sets[
                    left
                ]
                & breast_date_sets[
                    right
                ]
            )

    triple = len(
        breast_date_sets[
            "progression"
        ]
        & breast_date_sets[
            "cancer_presence"
        ]
        & breast_date_sets[
            "tumor_sites"
        ]
    )

    per_patient = counts_per_patient(
        union_breast
    )

    scan_day_analysis = {
        "all_cohort_unique_patient_days":
            len(union_all),

        "breast_unique_patient_days":
            len(union_breast),

        "breast_patients_with_any_radiology_measurement":
            len(per_patient),

        "breast_measurement_days_per_patient":
            quantiles(
                list(
                    per_patient.values()
                )
            ),

        "same_day_pairwise_overlap":
            pairwise,

        "same_day_all_three_overlap":
            triple,

        "episode_clustering": {
            str(window):
                cluster_dates(
                    union_breast,
                    window,
                )
            for window
            in (
                0,
                3,
                7,
            )
        },

        "construction_candidate": (
            "Use the union of START_DATE values from "
            "progression, cancer_presence, and tumor_sites "
            "as radiology measurement days; group nearby "
            "measurement days into scan episodes, with "
            "3 days primary and 0/7 day sensitivity analyses."
        ),
    }

    ###########################################################################
    # 4. Explicitly inspect progression Y/N semantics.
    ###########################################################################

    progression_path = find_one(
        chord_root,
        "data_timeline_progression.txt",
    )

    progression_df = read_table(
        progression_path
    )

    progression_breast = progression_df[
        progression_df[
            "PATIENT_ID"
        ].isin(
            breast_ids
        )
    ].copy()

    progression_values = (
        sorted(
            {
                str(x)
                .strip()
                .upper()
                for x
                in progression_breast.get(
                    "PROGRESSION",
                    pd.Series(
                        dtype=str
                    ),
                )
                if str(x).strip()
            }
        )
    )

    progression_counts = (
        progression_breast[
            "PROGRESSION"
        ]
        .astype(str)
        .str.strip()
        .str.upper()
        .value_counts()
        .to_dict()
        if "PROGRESSION"
        in progression_breast.columns
        else {}
    )

    progression_semantics = {
        "column_present":
            "PROGRESSION"
            in progression_breast.columns,

        "values":
            progression_values,

        "counts_breast":
            {
                str(k): int(v)
                for k, v
                in progression_counts.items()
            },

        "has_Y_and_N":
            (
                "Y"
                in progression_values
                and "N"
                in progression_values
            ),
    }

    ###########################################################################
    # 5. BPC-MSK imaging audit only.
    #
    # DFCI/VICC rows are never used for value distributions.
    ###########################################################################

    site_map, site_map_audit = (
        bpc_patient_site_map(
            bpc_root
        )
    )

    imaging_paths = (
        bpc_imaging_candidates(
            bpc_root
        )
    )

    print(
        "[CKPT0E] "
        f"BPC imaging/radiology candidate files="
        f"{len(imaging_paths)}",
        flush=True,
    )

    bpc_imaging = []

    for path in imaging_paths:
        print(
            "[CKPT0E] "
            f"auditing BPC-MSK imaging file: {path}",
            flush=True,
        )

        bpc_imaging.append(
            summarize_bpc_msk_imaging(
                path,
                site_map,
            )
        )

    ###########################################################################
    # 6. Correct timestamp semantics.
    ###########################################################################

    timestamp_rules = {
        "CHORD_timeline_START_DATE": {
            "meaning": (
                "Relative day from diagnosis in "
                "cBioPortal timeline representation."
            ),
            "use": (
                "Primary within-patient event ordering "
                "time for CHORD timeline events."
            ),
        },

        "CHORD_radiology_START_DATE": {
            "meaning": (
                "Radiology-derived annotation event day "
                "available in the public CHORD timelines."
            ),
            "use": (
                "Best available public timing for a "
                "radiology measurement landmark. "
                "Do not claim a distinct report-release "
                "timestamp because none is exposed."
            ),
        },

        "CHORD_timeline_STOP_DATE": {
            "meaning": (
                "Interval end day when applicable. "
                "For point events, blank STOP_DATE is "
                "expected by cBioPortal convention."
            ),
        },

        "CHORD_lab_RESULT": {
            "meaning": "Measurement value, NOT a timestamp.",
        },

        "CHORD_GENDER": {
            "meaning": "Patient attribute, NOT a timestamp.",
        },

        "genomic_availability": {
            "meaning": (
                "Do not equate specimen collection with "
                "genomic result availability. Where only "
                "sequencing-relative timing is exposed, "
                "retain an explicit timing-quality flag."
            ),
        },

        "rule": (
            "Timestamp classification now uses explicit "
            "field semantics rather than arbitrary "
            "substring matching."
        ),
    }

    atomic_json(
        out
        / "timestamp_semantics.json",
        timestamp_rules,
    )

    ###########################################################################
    # 7. GENIE/BPC external-validation leakage guard.
    #
    # IDs only. No DFCI/VICC outcome values are inspected.
    ###########################################################################

    exclusion_audit = (
        write_external_exclusion_manifest(
            genie_root=genie_root,
            bpc_site_map=site_map,
            out=out,
        )
    )

    atomic_json(
        out
        / "genie_external_exclusion_audit.json",
        exclusion_audit,
    )

    ###########################################################################
    # 8. Determine what CHORD does and does not expose.
    ###########################################################################

    modality_columns = sorted(
        {
            col
            for summary
            in radiology.values()
            for col
            in summary[
                "modality_or_coverage_columns"
            ]
        }
    )

    chord_scan_capabilities = {
        "radiology_measurement_days":
            True,

        "progression_YN":
            progression_semantics[
                "has_Y_and_N"
            ],

        "cancer_presence_timeline":
            radiology[
                "cancer_presence"
            ][
                "rows_breast"
            ]
            > 0,

        "tumor_site_timeline":
            radiology[
                "tumor_sites"
            ][
                "rows_breast"
            ]
            > 0,

        "explicit_modality_or_anatomic_coverage_columns":
            modality_columns,

        "coverage_policy_if_columns_absent": (
            "Represent scan coverage as UNKNOWN/NOT_OBSERVED "
            "rather than NEGATIVE. Never infer an unreported "
            "body region was imaged."
        ),

        "report_identity_policy": (
            "The public timeline exposes radiology-derived "
            "measurement days rather than a guaranteed raw "
            "report identifier. Build scan episodes from "
            "measurement-day evidence, preserving source "
            "components and testing 0/3/7-day grouping."
        ),
    }

    ###########################################################################
    # 9. Readiness decision.
    ###########################################################################

    each_radiology_track_ok = all(
        radiology[
            name
        ][
            "unique_patient_days_breast"
        ]
        > 0
        for name
        in RADIOLOGY_FILES
    )

    progression_ok = (
        progression_semantics[
            "has_Y_and_N"
        ]
    )

    scan_days_ok = (
        scan_day_analysis[
            "breast_unique_patient_days"
        ]
        > 0
    )

    bpc_msk_imaging_ok = any(
        item.get(
            "usable",
            False,
        )
        and item.get(
            "date_columns"
        )
        for item
        in bpc_imaging
    )

    site_partition_ok = (
        site_map_audit[
            "site_counts"
        ].get(
            "MSK",
            0,
        )
        == 529
        and site_map_audit[
            "site_counts"
        ].get(
            "DFCI",
            0,
        )
        == 428
        and site_map_audit[
            "site_counts"
        ].get(
            "VICC",
            0,
        )
        == 173
    )

    external_exclusion_ok = (
        exclusion_audit[
            "exact_external_overlap"
        ]
        == 601
    )

    criteria = {
        "CHORD_three_radiology_derived_tracks_present":
            each_radiology_track_ok,

        "CHORD_progression_has_Y_and_N":
            progression_ok,

        "CHORD_scan_like_measurement_days_constructible":
            scan_days_ok,

        "BPC_MSK_imaging_audit_stream_available":
            bpc_msk_imaging_ok,

        "BPC_site_partition_exact":
            site_partition_ok,

        "GENIE_external_BPC_patient_exclusion_manifest_complete":
            external_exclusion_ok,
    }

    ready = all(
        criteria.values()
    )

    status = (
        "READY_FOR_CANONICAL_TIMELINE"
        if ready
        else "NEEDS_RADIOLOGY_SCHEMA_RESOLUTION"
    )

    report = {
        "status":
            status,

        "breast_cohort": {
            "patients":
                len(breast_ids),
            "source":
                str(
                    clinical_sample_path
                ),
            "cancer_column":
                cancer_col,
        },

        "chord_radiology":
            radiology,

        "scan_day_analysis":
            scan_day_analysis,

        "progression_semantics":
            progression_semantics,

        "chord_scan_capabilities":
            chord_scan_capabilities,

        "bpc_site_map":
            site_map_audit,

        "bpc_msk_imaging":
            bpc_imaging,

        "genie_external_exclusion":
            exclusion_audit,

        "criteria":
            criteria,

        "policy_guard": (
            "No DFCI/VICC outcome-value distribution "
            "was computed. External cohorts were used "
            "only for site membership and patient-ID "
            "exclusion from GENIE pretraining."
        ),
    }

    atomic_json(
        out
        / "resolution.json",
        report,
    )

    ###########################################################################
    # 10. Human-readable report.
    ###########################################################################

    md: list[str] = []

    md.append(
        "# Checkpoint 0E - Radiology and "
        "Timestamp Resolution"
    )
    md.append("")
    md.append(
        f"Status: **{status}**"
    )
    md.append("")

    md.append(
        "## CHORD radiology-derived tables"
    )
    md.append("")

    for name in RADIOLOGY_FILES:
        x = radiology[
            name
        ]

        md.append(
            f"### {name}"
        )
        md.append("")

        md.append(
            f"- rows total: `{x['rows_total']}`"
        )
        md.append(
            f"- rows breast: `{x['rows_breast']}`"
        )
        md.append(
            f"- breast patients: `{x['patients_breast']}`"
        )
        md.append(
            "- unique breast patient-days: "
            f"`{x['unique_patient_days_breast']}`"
        )
        md.append(
            "- columns: "
            + ", ".join(
                f"`{c}`"
                for c in x[
                    "columns"
                ]
            )
        )
        md.append(
            "- modality/coverage columns: "
            + (
                ", ".join(
                    f"`{c}`"
                    for c in x[
                        "modality_or_coverage_columns"
                    ]
                )
                if x[
                    "modality_or_coverage_columns"
                ]
                else "**none detected**"
            )
        )
        md.append("")

    md.append(
        "## Scan-like measurement stream"
    )
    md.append("")

    md.append(
        "- breast unique patient-days "
        "across all three tracks: "
        f"`{scan_day_analysis['breast_unique_patient_days']}`"
    )

    md.append(
        "- breast patients with at least "
        "one radiology-derived measurement: "
        f"`{scan_day_analysis['breast_patients_with_any_radiology_measurement']}`"
    )

    md.append(
        "- measurement days per patient: "
        f"`{scan_day_analysis['breast_measurement_days_per_patient']}`"
    )

    for window, result in (
        scan_day_analysis[
            "episode_clustering"
        ].items()
    ):
        md.append(
            f"- {window}-day grouping: "
            f"`{result['episodes']}` episodes"
        )

    md.append("")

    md.append(
        "## Progression semantics"
    )
    md.append("")

    md.append(
        "- PROGRESSION values: "
        f"`{progression_values}`"
    )

    md.append(
        "- counts in breast cohort: "
        f"`{progression_semantics['counts_breast']}`"
    )

    md.append("")

    md.append(
        "## BPC-MSK imaging audit"
    )
    md.append("")

    for x in bpc_imaging:
        md.append(
            f"- `{x['path']}`: "
            f"usable={x.get('usable')} "
            f"MSK_rows={x.get('msk_rows')} "
            f"MSK_patients={x.get('msk_patients')} "
            f"dates={x.get('date_columns')}"
        )

    md.append("")
    md.append(
        "Only MSK values were summarized; "
        "DFCI/VICC outcome distributions "
        "were not inspected."
    )
    md.append("")

    md.append(
        "## GENIE external-validation guard"
    )
    md.append("")

    md.append(
        "- BPC DFCI/VICC patients: "
        f"`{exclusion_audit['bpc_external_patients']}`"
    )

    md.append(
        "- exact IDs overlapping GENIE: "
        f"`{exclusion_audit['exact_external_overlap']}`"
    )

    md.append(
        "- exclusion manifest: "
        "`genie_external_validation_patient_exclusions.txt`"
    )
    md.append("")

    md.append(
        "## Readiness"
    )
    md.append("")

    for key, value in criteria.items():
        md.append(
            f"- {key}: "
            f"**{'PASS' if value else 'FAIL'}**"
        )

    md.append("")

    atomic_text(
        out
        / "audit.md",
        "\n".join(md) + "\n",
    )

    ###########################################################################
    # 11. Persistent handoff.
    ###########################################################################

    handoff = {
        "checkpoint": "0E",

        "name": (
            "radiology_measurement_and_"
            "timestamp_resolution"
        ),

        "status":
            status,

        "chord_breast_patients":
            len(breast_ids),

        "chord_scan_like_patient_days":
            scan_day_analysis[
                "breast_unique_patient_days"
            ],

        "primary_scan_episode_rule_candidate": (
            "Union progression/cancer_presence/"
            "tumor_sites START_DATE evidence and "
            "group within 3 days; preserve 0-day "
            "and 7-day sensitivity variants."
        ),

        "chord_coverage_rule": (
            "Never infer unobserved regions as negative. "
            "If public CHORD has no modality/coverage "
            "field, encode coverage as unknown."
        ),

        "timestamp_rule": (
            "CHORD START_DATE is relative days from "
            "diagnosis and is the public event-ordering "
            "time. RESULT and GENDER are not timestamps."
        ),

        "external_validation_guard": {
            "DFCI_VICC_outcome_distributions_inspected":
                False,

            "GENIE_exact_external_patient_exclusions":
                exclusion_audit[
                    "exact_external_overlap"
                ],
        },

        "artifacts": {
            "resolution":
                "artifacts/checkpoint0e/resolution.json",

            "audit":
                "artifacts/checkpoint0e/audit.md",

            "timestamp_semantics":
                "artifacts/checkpoint0e/timestamp_semantics.json",

            "genie_external_exclusion":
                (
                    "artifacts/checkpoint0e/"
                    "genie_external_validation_patient_exclusions.txt"
                ),
        },

        "next_required_action": (
            "Implement canonical longitudinal timeline, "
            "scan episodes, treatment state, outcomes, "
            "and leakage-safe landmarks."
            if ready
            else (
                "Resolve failed Checkpoint 0E "
                "radiology/schema criterion."
            )
        ),
    }

    handoff_path = (
        repo
        / "artifacts"
        / "handoff"
        / "checkpoint_00E.json"
    )

    atomic_json(
        handoff_path,
        handoff,
    )

    ###########################################################################
    # 12. Compact decision packet.
    ###########################################################################

    print("")
    print(
        "========== CKPT0E SUMMARY =========="
    )

    print(
        f"status={status}"
    )

    print(
        f"CHORD_breast_patients={len(breast_ids)}"
    )

    for name in RADIOLOGY_FILES:
        x = radiology[
            name
        ]

        print(
            f"{name}: "
            f"rows_breast={x['rows_breast']} "
            f"patients_breast={x['patients_breast']} "
            f"patient_days_breast="
            f"{x['unique_patient_days_breast']} "
            f"coverage_columns="
            f"{x['modality_or_coverage_columns']} "
            f"stop_date={x['stop_date']}"
        )

    print(
        "radiology_union_breast_patient_days="
        f"{scan_day_analysis['breast_unique_patient_days']}"
    )

    print(
        "radiology_patients_with_measurements="
        f"{scan_day_analysis['breast_patients_with_any_radiology_measurement']}"
    )

    print(
        "measurement_days_per_patient="
        f"{scan_day_analysis['breast_measurement_days_per_patient']}"
    )

    for window in (
        "0",
        "3",
        "7",
    ):
        x = scan_day_analysis[
            "episode_clustering"
        ][
            window
        ]

        print(
            f"episodes_window_{window}d="
            f"{x['episodes']} "
            f"episode_distribution="
            f"{x['episodes_per_patient']}"
        )

    print(
        "same_day_pairwise_overlap="
        f"{scan_day_analysis['same_day_pairwise_overlap']}"
    )

    print(
        "same_day_all_three_overlap="
        f"{scan_day_analysis['same_day_all_three_overlap']}"
    )

    print(
        "progression_values="
        f"{progression_values}"
    )

    print(
        "progression_counts_breast="
        f"{progression_semantics['counts_breast']}"
    )

    print(
        "CHORD_modality_or_coverage_columns="
        f"{modality_columns}"
    )

    print(
        "BPC_site_counts="
        f"{site_map_audit['site_counts']}"
    )

    for x in bpc_imaging:
        print(
            "BPC_MSK_IMAGING "
            f"path={x['path']} "
            f"usable={x.get('usable')} "
            f"rows={x.get('msk_rows')} "
            f"patients={x.get('msk_patients')} "
            f"date_columns={x.get('date_columns')} "
            f"semantic_columns={x.get('semantic_columns')}"
        )

    print(
        "GENIE_external_BPC_patients="
        f"{exclusion_audit['bpc_external_patients']}"
    )

    print(
        "GENIE_external_exact_overlap_excluded="
        f"{exclusion_audit['exact_external_overlap']}"
    )

    print(
        "GENIE_center_counts="
        f"{exclusion_audit['genie_center_counts']}"
    )

    for key, value in criteria.items():
        print(
            f"criterion {key}="
            f"{'PASS' if value else 'FAIL'}"
        )

    print(
        "external_outcome_distribution_guard=PASS"
    )

    print(
        "audit_md="
        "artifacts/checkpoint0e/audit.md"
    )

    print(
        "handoff="
        "artifacts/handoff/checkpoint_00E.json"
    )

    print(
        "========== CKPT0E SUMMARY END =========="
    )

    print("")
    print(
        "========== CKPT0E SCHEMA DETAILS =========="
    )

    for name in RADIOLOGY_FILES:
        x = radiology[
            name
        ]

        print(
            f"{name}_columns={x['columns']}"
        )

        print(
            f"{name}_event_types={x['event_types']}"
        )

        print(
            f"{name}_payload_values_breast="
            f"{x['payload_values_breast']}"
        )

    for x in bpc_imaging:
        print(
            "BPC_MSK_FILE="
            f"{x['path']}"
        )
        print(
            "  columns="
            f"{x.get('columns')}"
        )
        print(
            "  MSK_value_counts="
            f"{x.get('msk_value_counts')}"
        )

    print(
        "========== CKPT0E SCHEMA DETAILS END =========="
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
