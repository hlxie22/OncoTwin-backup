#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PREFIX = "GENIE-MSK-"

EXPECTED_BPC_MSK_PATIENTS = 529
EXPECTED_BPC_MSK_SAMPLES = 610


###############################################################################
# Utilities
###############################################################################


def norm(value: Any) -> str:

    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value).strip().lower(),
    ).strip("_")


def clean(value: Any) -> str:

    return str(
        value
    ).strip().upper()


def strip_prefix(value: Any) -> str:

    text = clean(
        value
    )

    if not text.startswith(
        PREFIX
    ):

        return ""

    return text[
        len(
            PREFIX
        ):
    ]


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


def find_col(
    columns,
    candidates,
    *,
    required=True,
):

    lookup = {
        norm(
            column
        ):
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
            f"Unable to resolve {candidates}; "
            f"columns={list(columns)}"
        )

    return None


def find_unique_file(
    root: Path,
    basename: str,
) -> Path:

    candidates = [
        path
        for path in root.rglob(
            basename
        )
        if path.is_file()
    ]

    if len(
        candidates
    ) != 1:

        raise RuntimeError(
            f"Expected one {basename} under {root}, "
            f"found={candidates}"
        )

    return candidates[
        0
    ]


def read_tabular(
    path: Path,
) -> pd.DataFrame:

    separator = (
        ","
        if path.suffix.lower()
        == ".csv"
        else "\t"
    )

    return pd.read_csv(
        path,
        sep=separator,
        comment="#",
        dtype=str,
        keep_default_na=False,
        low_memory=False,
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
            "ckpt7a2_identity_validation",
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
# BPC MSK patient/sample extraction without outcomes
###############################################################################


def load_bpc_msk(
    repo: Path,
    bpc_root: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    inventory = pd.read_parquet(
        repo
        / "artifacts"
        / "checkpoint7a1"
        / "schema_inventory.parquet"
    )

    ckpt7a2 = import_ckpt7a2(
        repo
    )

    guard = ckpt7a2.AccessGuard(
        bpc_root,
        inventory,
    )

    patient = guard.read_predictor(
        "cBioPortal_files/data_clinical_patient.txt"
    )

    sample = guard.read_predictor(
        "cBioPortal_files/data_clinical_sample.txt"
    )

    (
        site_map,
        site_audit,
    ) = ckpt7a2.build_site_map(
        patient
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
            "Unexpected BPC site partition."
        )

    patient_col = find_col(
        patient.columns,
        [
            "PATIENT_ID",
        ],
    )

    sample_patient_col = find_col(
        sample.columns,
        [
            "PATIENT_ID",
        ],
    )

    sample_col = find_col(
        sample.columns,
        [
            "SAMPLE_ID",
        ],
    )

    bpc_patients = pd.DataFrame(
        {
            "bpc_patient_id":
                patient[
                    patient_col
                ]
                .map(
                    clean
                ),
        }
    )

    bpc_patients[
        "site"
    ] = bpc_patients[
        "bpc_patient_id"
    ].map(
        {
            clean(
                patient_id
            ):
                site
            for patient_id, site
            in site_map.items()
        }
    )

    bpc_patients = (
        bpc_patients[
            bpc_patients[
                "site"
            ]
            == "MSK"
        ][
            [
                "bpc_patient_id",
            ]
        ]
        .drop_duplicates()
        .reset_index(
            drop=True
        )
    )

    bpc_patients[
        "derived_chord_patient_id"
    ] = bpc_patients[
        "bpc_patient_id"
    ].map(
        strip_prefix
    )

    bpc_samples = pd.DataFrame(
        {
            "bpc_patient_id":
                sample[
                    sample_patient_col
                ]
                .map(
                    clean
                ),

            "bpc_sample_id":
                sample[
                    sample_col
                ]
                .map(
                    clean
                ),
        }
    )

    bpc_samples = bpc_samples[
        bpc_samples[
            "bpc_patient_id"
        ].isin(
            set(
                bpc_patients[
                    "bpc_patient_id"
                ]
            )
        )
    ].copy()

    bpc_samples = (
        bpc_samples
        .drop_duplicates()
        .reset_index(
            drop=True
        )
    )

    bpc_samples[
        "derived_chord_patient_id"
    ] = bpc_samples[
        "bpc_patient_id"
    ].map(
        strip_prefix
    )

    bpc_samples[
        "derived_chord_sample_id"
    ] = bpc_samples[
        "bpc_sample_id"
    ].map(
        strip_prefix
    )

    if len(
        bpc_patients
    ) != EXPECTED_BPC_MSK_PATIENTS:

        raise RuntimeError(
            "Unexpected BPC-MSK patient count: "
            f"{len(bpc_patients)}"
        )

    if (
        bpc_samples[
            "bpc_sample_id"
        ].nunique()
        != EXPECTED_BPC_MSK_SAMPLES
    ):

        raise RuntimeError(
            "Unexpected BPC-MSK sample count: "
            f"{bpc_samples['bpc_sample_id'].nunique()}"
        )

    return (
        bpc_patients,
        bpc_samples,
    )


###############################################################################
# CHORD identity tables
###############################################################################


def load_chord_identity(
    chord_root: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    patient_path = find_unique_file(
        chord_root,
        "data_clinical_patient.txt",
    )

    sample_path = find_unique_file(
        chord_root,
        "data_clinical_sample.txt",
    )

    patient = read_tabular(
        patient_path
    )

    sample = read_tabular(
        sample_path
    )

    patient_col = find_col(
        patient.columns,
        [
            "PATIENT_ID",
        ],
    )

    sample_patient_col = find_col(
        sample.columns,
        [
            "PATIENT_ID",
        ],
    )

    sample_col = find_col(
        sample.columns,
        [
            "SAMPLE_ID",
        ],
    )

    chord_patients = pd.DataFrame(
        {
            "chord_patient_id":
                patient[
                    patient_col
                ]
                .map(
                    clean
                ),
        }
    )

    chord_patients = (
        chord_patients[
            chord_patients[
                "chord_patient_id"
            ]
            != ""
        ]
        .drop_duplicates()
        .reset_index(
            drop=True
        )
    )

    chord_samples = pd.DataFrame(
        {
            "chord_patient_id":
                sample[
                    sample_patient_col
                ]
                .map(
                    clean
                ),

            "chord_sample_id":
                sample[
                    sample_col
                ]
                .map(
                    clean
                ),
        }
    )

    chord_samples = (
        chord_samples[
            (
                chord_samples[
                    "chord_patient_id"
                ]
                != ""
            )
            & (
                chord_samples[
                    "chord_sample_id"
                ]
                != ""
            )
        ]
        .drop_duplicates()
        .reset_index(
            drop=True
        )
    )

    return (
        chord_patients,
        chord_samples,
    )


###############################################################################
# Validate candidate parquet against literal-prefix transform
###############################################################################


def validate_candidate_file(
    repo: Path,
) -> dict[str, Any]:

    candidates = pd.read_parquet(
        repo
        / "artifacts"
        / "checkpoint7a3b"
        / "identifier_overlap_candidates.parquet"
    )

    prefix = candidates[
        candidates[
            "match_type"
        ]
        == "KNOWN_PREFIX_STRIPPED"
    ].copy()

    if len(
        prefix
    ) != 832:

        raise RuntimeError(
            "Expected 832 known-prefix candidates, got "
            f"{len(prefix)}"
        )

    expected = prefix[
        "bpc_identifier"
    ].map(
        strip_prefix
    )

    literal_match = (
        expected
        == prefix[
            "chord_identifier"
        ].map(
            clean
        )
    )

    if not literal_match.all():

        bad = prefix.loc[
            ~literal_match
        ].head(
            20
        )

        raise RuntimeError(
            "Prefix candidate is not literal GENIE-MSK- removal:\n"
            + bad.to_string(
                index=False
            )
        )

    bpc_unique = (
        prefix.groupby(
            "bpc_identifier",
            observed=True,
        )[
            "chord_identifier"
        ]
        .nunique()
    )

    chord_unique = (
        prefix.groupby(
            "chord_identifier",
            observed=True,
        )[
            "bpc_identifier"
        ]
        .nunique()
    )

    return {
        "rows":
            int(
                len(
                    prefix
                )
            ),

        "literal_prefix_transform_fraction":
            float(
                literal_match.mean()
            ),

        "bpc_to_chord_max_multiplicity":
            int(
                bpc_unique.max()
            ),

        "chord_to_bpc_max_multiplicity":
            int(
                chord_unique.max()
            ),

        "one_to_one":
            bool(
                (
                    bpc_unique
                    <= 1
                ).all()
                and (
                    chord_unique
                    <= 1
                ).all()
            ),
    }


###############################################################################
# Main identity validation
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
        "--chord-root",
        default=(
            "data/external_sources/"
            "msk_chord_2024/raw"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "artifacts/checkpoint7a3c"
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

    chord_root = (
        repo
        / args.chord_root
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
    # Candidate artifact validation.
    ###########################################################################

    candidate_validation = (
        validate_candidate_file(
            repo
        )
    )

    ###########################################################################
    # Independently reconstruct both identity systems.
    ###########################################################################

    (
        bpc_patients,
        bpc_samples,
    ) = load_bpc_msk(
        repo,
        bpc_root,
    )

    (
        chord_patients,
        chord_samples,
    ) = load_chord_identity(
        chord_root
    )

    chord_patient_set = set(
        chord_patients[
            "chord_patient_id"
        ]
    )

    chord_sample_set = set(
        chord_samples[
            "chord_sample_id"
        ]
    )

    ###########################################################################
    # Patient-level prefix mapping.
    ###########################################################################

    bpc_patients[
        "chord_patient_exists"
    ] = bpc_patients[
        "derived_chord_patient_id"
    ].isin(
        chord_patient_set
    )

    patient_crosswalk = bpc_patients[
        bpc_patients[
            "chord_patient_exists"
        ]
    ][
        [
            "bpc_patient_id",
            "derived_chord_patient_id",
        ]
    ].copy()

    patient_crosswalk = patient_crosswalk.rename(
        columns={
            "derived_chord_patient_id":
                "chord_patient_id",
        }
    )

    ###########################################################################
    # Sample-level prefix mapping.
    ###########################################################################

    bpc_samples[
        "chord_sample_exists"
    ] = bpc_samples[
        "derived_chord_sample_id"
    ].isin(
        chord_sample_set
    )

    sample_crosswalk = bpc_samples[
        bpc_samples[
            "chord_sample_exists"
        ]
    ][
        [
            "bpc_patient_id",
            "bpc_sample_id",
            "derived_chord_patient_id",
            "derived_chord_sample_id",
        ]
    ].copy()

    sample_crosswalk = sample_crosswalk.rename(
        columns={
            "derived_chord_patient_id":
                "expected_chord_patient_id",

            "derived_chord_sample_id":
                "chord_sample_id",
        }
    )

    ###########################################################################
    # Resolve actual CHORD parent patient for every matched sample.
    ###########################################################################

    chord_sample_parent = (
        chord_samples[
            [
                "chord_sample_id",
                "chord_patient_id",
            ]
        ]
        .drop_duplicates()
    )

    sample_crosswalk = sample_crosswalk.merge(
        chord_sample_parent,
        on="chord_sample_id",
        how="left",
        validate="many_to_one",
    )

    sample_crosswalk[
        "parent_consistent"
    ] = (
        sample_crosswalk[
            "expected_chord_patient_id"
        ]
        == sample_crosswalk[
            "chord_patient_id"
        ]
    )

    ###########################################################################
    # One-to-one checks.
    ###########################################################################

    patient_forward = (
        patient_crosswalk.groupby(
            "bpc_patient_id",
            observed=True,
        )[
            "chord_patient_id"
        ]
        .nunique()
    )

    patient_reverse = (
        patient_crosswalk.groupby(
            "chord_patient_id",
            observed=True,
        )[
            "bpc_patient_id"
        ]
        .nunique()
    )

    sample_forward = (
        sample_crosswalk.groupby(
            "bpc_sample_id",
            observed=True,
        )[
            "chord_sample_id"
        ]
        .nunique()
    )

    sample_reverse = (
        sample_crosswalk.groupby(
            "chord_sample_id",
            observed=True,
        )[
            "bpc_sample_id"
        ]
        .nunique()
    )

    patient_one_to_one = bool(
        (
            patient_forward
            <= 1
        ).all()
        and (
            patient_reverse
            <= 1
        ).all()
    )

    sample_one_to_one = bool(
        (
            sample_forward
            <= 1
        ).all()
        and (
            sample_reverse
            <= 1
        ).all()
    )

    parent_consistency = (
        float(
            sample_crosswalk[
                "parent_consistent"
            ].mean()
        )
        if len(
            sample_crosswalk
        )
        else 0.0
    )

    ###########################################################################
    # Sample support per matched patient.
    ###########################################################################

    sample_support = (
        sample_crosswalk[
            sample_crosswalk[
                "parent_consistent"
            ]
        ]
        .groupby(
            [
                "bpc_patient_id",
                "chord_patient_id",
            ],
            observed=True,
        )
        .size()
        .rename(
            "matched_sample_count"
        )
        .reset_index()
    )

    patient_crosswalk = (
        patient_crosswalk
        .merge(
            sample_support,
            on=[
                "bpc_patient_id",
                "chord_patient_id",
            ],
            how="left",
            validate="one_to_one",
        )
    )

    patient_crosswalk[
        "matched_sample_count"
    ] = (
        patient_crosswalk[
            "matched_sample_count"
        ]
        .fillna(
            0
        )
        .astype(
            int
        )
    )

    patients_with_sample_support = int(
        (
            patient_crosswalk[
                "matched_sample_count"
            ]
            > 0
        ).sum()
    )

    ###########################################################################
    # Cross-check the 832 candidate rows against independently derived maps.
    ###########################################################################

    candidate_expected_rows = (
        len(
            patient_crosswalk
        )
        + len(
            sample_crosswalk
        )
    )

    ###########################################################################
    # Identity bridge decision.
    #
    # This is not a model-performance threshold. The essential requirements are
    # deterministic one-to-one mapping and perfect sample-parent consistency.
    ###########################################################################

    bridge_valid = bool(
        candidate_validation[
            "one_to_one"
        ]
        and candidate_validation[
            "literal_prefix_transform_fraction"
        ]
        == 1.0
        and patient_one_to_one
        and sample_one_to_one
        and parent_consistency
        == 1.0
        and len(
            patient_crosswalk
        )
        >= 100
        and len(
            sample_crosswalk
        )
        >= 50
        and patients_with_sample_support
        >= 50
    )

    status = (
        "PASS_PREFIX_IDENTITY_BRIDGE"
        if bridge_valid
        else "NEEDS_IDENTITY_BRIDGE_RESOLUTION"
    )

    ###########################################################################
    # Persist immutable crosswalks.
    ###########################################################################

    atomic_parquet(
        out
        / "msk_patient_crosswalk.parquet",
        patient_crosswalk,
    )

    atomic_parquet(
        out
        / "msk_sample_crosswalk.parquet",
        sample_crosswalk,
    )

    ###########################################################################
    # Report.
    ###########################################################################

    report = {
        "status":
            status,

        "external_outcomes_opened":
            False,

        "mapping_rule":
            (
                "For BPC-MSK identifiers only, remove the exact "
                "leading literal prefix 'GENIE-MSK-'. "
                "No fuzzy or punctuation-normalized matching is used."
            ),

        "candidate_artifact_validation":
            candidate_validation,

        "bpc": {
            "msk_patients":
                int(
                    len(
                        bpc_patients
                    )
                ),

            "msk_samples":
                int(
                    bpc_samples[
                        "bpc_sample_id"
                    ].nunique()
                ),
        },

        "chord": {
            "patients":
                int(
                    len(
                        chord_patients
                    )
                ),

            "samples":
                int(
                    chord_samples[
                        "chord_sample_id"
                    ].nunique()
                ),
        },

        "bridge": {
            "matched_patients":
                int(
                    len(
                        patient_crosswalk
                    )
                ),

            "patient_fraction_of_bpc_msk":
                float(
                    len(
                        patient_crosswalk
                    )
                    / max(
                        len(
                            bpc_patients
                        ),
                        1,
                    )
                ),

            "matched_samples":
                int(
                    len(
                        sample_crosswalk
                    )
                ),

            "sample_fraction_of_bpc_msk":
                float(
                    len(
                        sample_crosswalk
                    )
                    / max(
                        bpc_samples[
                            "bpc_sample_id"
                        ].nunique(),
                        1,
                    )
                ),

            "patients_with_matched_sample_support":
                patients_with_sample_support,

            "patient_one_to_one":
                patient_one_to_one,

            "sample_one_to_one":
                sample_one_to_one,

            "sample_parent_consistency":
                parent_consistency,

            "derived_patient_plus_sample_rows":
                int(
                    candidate_expected_rows
                ),

            "ckpt7a3b_candidate_rows":
                int(
                    candidate_validation[
                        "rows"
                    ]
                ),
        },

        "policy": {
            "fuzzy_identity_matching_used":
                False,

            "punctuation_normalization_used":
                False,

            "external_outcome_tables_accessed":
                False,

            "bridge_scope":
                (
                    "BPC-MSK <-> CHORD positive-control linkage only. "
                    "This mapping is not applied to DFCI or VICC."
                ),
        },

        "next_action":
            (
                "Use the frozen msk_patient_crosswalk.parquet to rerun "
                "CKPT7A3 episode/state/treatment-line/current-scan-feature "
                "positive-control validation."
                if bridge_valid
                else
                "Inspect the failed identity-consistency gate before "
                "using the prefix bridge."
            ),
    }

    atomic_json(
        out
        / "identity_validation.json",
        report,
    )

    ###########################################################################
    # Human-readable audit.
    ###########################################################################

    audit = f"""# CKPT7A3C — MSK identity bridge validation

Status: **{status}**

External outcomes opened: **NO**

## Frozen identity rule

For BPC-MSK identifiers only:

`GENIE-MSK-<identifier>` -> `<identifier>`

No fuzzy matching is used.

## Candidate-artifact consistency

{candidate_validation}

## Patient bridge

- BPC-MSK patients: {len(bpc_patients)}
- matched CHORD patients: {len(patient_crosswalk)}
- matched fraction: {report['bridge']['patient_fraction_of_bpc_msk']}
- one-to-one: {patient_one_to_one}

## Sample bridge

- BPC-MSK samples: {bpc_samples['bpc_sample_id'].nunique()}
- matched CHORD samples: {len(sample_crosswalk)}
- matched fraction: {report['bridge']['sample_fraction_of_bpc_msk']}
- one-to-one: {sample_one_to_one}
- parent-patient consistency: {parent_consistency}
- matched patients with sample-level support: {patients_with_sample_support}

## Scope

This bridge is used only to validate the BPC-MSK -> CHORD positive control.
It does not alter DFCI/VICC identifiers or use external outcomes.
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
            "7A3C",

        "name":
            "validate_msk_prefix_identity_bridge",

        "status":
            status,

        "external_outcomes_opened":
            False,

        "mapping_rule":
            (
                "strip exact leading GENIE-MSK- prefix for BPC-MSK only"
            ),

        "matched_patients":
            len(
                patient_crosswalk
            ),

        "matched_samples":
            len(
                sample_crosswalk
            ),

        "sample_parent_consistency":
            parent_consistency,

        "next_action":
            report[
                "next_action"
            ],
    }

    atomic_json(
        repo
        / "artifacts"
        / "handoff"
        / "checkpoint_07A3C.json",
        handoff,
    )

    ###########################################################################
    # Terminal packet.
    ###########################################################################

    print("")
    print(
        "========== CKPT7A3C SUMMARY =========="
    )

    print(
        f"status={status}"
    )

    print(
        "external_outcomes_opened=False"
    )

    print(
        "mapping_rule="
        "strip exact leading GENIE-MSK- prefix"
    )

    print(
        "bpc_msk_patients="
        f"{len(bpc_patients)}"
    )

    print(
        "matched_patients="
        f"{len(patient_crosswalk)}"
    )

    print(
        "patient_match_fraction="
        f"{report['bridge']['patient_fraction_of_bpc_msk']}"
    )

    print(
        "bpc_msk_samples="
        f"{bpc_samples['bpc_sample_id'].nunique()}"
    )

    print(
        "matched_samples="
        f"{len(sample_crosswalk)}"
    )

    print(
        "sample_match_fraction="
        f"{report['bridge']['sample_fraction_of_bpc_msk']}"
    )

    print(
        "patients_with_sample_support="
        f"{patients_with_sample_support}"
    )

    print(
        "patient_one_to_one="
        f"{patient_one_to_one}"
    )

    print(
        "sample_one_to_one="
        f"{sample_one_to_one}"
    )

    print(
        "sample_parent_consistency="
        f"{parent_consistency}"
    )

    print(
        "candidate_rows="
        f"{candidate_validation['rows']}"
    )

    print(
        "derived_patient_plus_sample_rows="
        f"{candidate_expected_rows}"
    )

    print(
        "handoff="
        "artifacts/handoff/checkpoint_07A3C.json"
    )

    print(
        "========== CKPT7A3C SUMMARY END =========="
    )

    print("")
    print(
        "========== CKPT7A3C DECISION PACKET =========="
    )

    print(
        "candidate_validation="
        f"{candidate_validation}"
    )

    print(
        "bridge="
        f"{report['bridge']}"
    )

    print(
        "sample_support_distribution="
        f"{patient_crosswalk['matched_sample_count'].value_counts().sort_index().to_dict()}"
    )

    print(
        "patient_crosswalk_examples="
        f"{patient_crosswalk.head(20).to_dict(orient='records')}"
    )

    print(
        "sample_crosswalk_examples="
        f"{sample_crosswalk.head(20).to_dict(orient='records')}"
    )

    print(
        "next_action="
        f"{report['next_action']}"
    )

    print(
        "========== CKPT7A3C DECISION PACKET END =========="
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
