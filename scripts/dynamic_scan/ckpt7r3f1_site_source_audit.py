#!/usr/bin/env python3

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

BPC = (
    ROOT
    / "data/external_sources/bpc_brca_1_0_public"
)

R3E = (
    ROOT
    / "artifacts/checkpoint7r3e_transport_safe_external_protocol"
)

OUT = (
    ROOT
    / "artifacts/checkpoint7r3f1_site_source_audit"
)


###############################################################################
# Frozen quarantine from R3E.
#
# These files are NEVER opened in this checkpoint, not even for headers.
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
# Column semantics
###############################################################################

HIGH_CONFIDENCE_EXACT = {
    "METASTATIC_SITE",
    "METASTASIS_SITE",
    "METASTATIC_SITES",
    "METASTASIS_SITES",
    "MET_SITE",
    "MET_SITES",
    "DISEASE_SITE",
    "DISEASE_SITES",
    "TUMOR_SITE",
    "TUMOR_SITES",
    "LESION_SITE",
    "LESION_SITES",
    "SITE_OF_DISEASE",
    "METASTATIC_LOCATION",
    "METASTASIS_LOCATION",
    "TUMOR_LOCATION",
    "LESION_LOCATION",
}


COVERAGE_EXACT = {
    "SCAN_SITES",
    "SCAN_SITE",
    "SCAN_REGIONS",
    "SCAN_REGION",
    "REGIONS_SCANNED",
    "REGION_SCANNED",
    "IMAGING_SITES",
    "IMAGING_SITE",
}


PRIMARY_SITE_EXACT = {
    "PRIMARY_SITE",
    "PRIMARY_TUMOR_SITE",
    "PRIMARY_TUMOR_LOCATION",
    "SITE_OF_ORIGIN",
    "ORIGIN_SITE",
}


SPECIMEN_SITE_EXACT = {
    "SPECIMEN_SITE",
    "BIOPSY_SITE",
    "SAMPLE_SITE",
    "COLLECTION_SITE",
}


AMBIGUOUS_EXACT = {
    "SITE",
    "SITES",
    "ANATOMIC_SITE",
    "ANATOMIC_SITES",
    "ANATOMIC_LOCATION",
    "ANATOMICAL_SITE",
    "ANATOMICAL_LOCATION",
    "LOCATION",
    "LOCATIONS",
}


SAFE_KEY_EXACT = {
    "PATIENT_ID",
    "PATIENTID",
    "PATIENT",
    "SAMPLE_ID",
    "SAMPLEID",
    "START_DATE",
    "STOP_DATE",
    "DATE",
    "EVENT_DATE",
    "SCAN_DATE",
    "IMAGING_DATE",
}


###############################################################################
# We do not use these columns, but their presence is useful as a risk flag.
###############################################################################

OUTCOME_LIKE_PATTERNS = [
    re.compile(
        r"(^|_)(PFS|PFS_MONTHS|PFS_DAYS|OS|OS_MONTHS|OS_DAYS)(_|$)"
    ),
    re.compile(
        r"(^|_)(SURVIVAL|DEATH|DECEASED|VITAL_STATUS)(_|$)"
    ),
    re.compile(
        r"(^|_)(LAST_FOLLOWUP|LAST_FOLLOW_UP|ENDPOINT)(_|$)"
    ),
]


###############################################################################
# Six frozen CKPT7R2 disease-site channels.
###############################################################################

TARGET_SITES = {
    "BONE": [
        "BONE",
        "OSSEOUS",
        "SKELETAL",
    ],

    "LIVER": [
        "LIVER",
        "HEPATIC",
    ],

    "LUNG": [
        "LUNG",
        "PULMONARY",
    ],

    "BRAIN": [
        "BRAIN",
        "CNS",
        "CENTRAL NERVOUS",
        "CEREBRAL",
        "INTRACRANIAL",
    ],

    "LYMPH": [
        "LYMPH",
        "LYMPH NODE",
        "LYMPH NODES",
        "NODAL",
    ],

    "PLEURA": [
        "PLEURA",
        "PLEURAL",
    ],
}


TEXT_SUFFIXES = (
    ".txt",
    ".tsv",
    ".csv",
    ".txt.gz",
    ".tsv.gz",
    ".csv.gz",
)


def normalize_column(value):

    value = str(value).strip().upper()

    value = re.sub(
        r"[^A-Z0-9]+",
        "_",
        value,
    )

    return value.strip("_")


def rel_string(path):

    return str(
        path.relative_to(BPC)
    ).replace(
        "\\",
        "/",
    )


def is_quarantined(rel):

    return rel in QUARANTINED


def open_text(path):

    if path.name.lower().endswith(
        ".gz"
    ):

        return gzip.open(
            path,
            "rt",
            encoding="utf-8",
            errors="replace",
        )

    return path.open(
        "rt",
        encoding="utf-8",
        errors="replace",
    )


def discover_header(path):

    with open_text(path) as f:

        for line in f:

            stripped = line.strip()

            if not stripped:

                continue

            if stripped.startswith("#"):

                continue

            tab_count = line.count("\t")
            comma_count = line.count(",")

            delimiter = (
                "\t"
                if tab_count >= comma_count
                else ","
            )

            row = next(
                csv.reader(
                    [line],
                    delimiter=delimiter,
                )
            )

            columns = [
                str(x).strip()
                for x in row
            ]

            return {
                "delimiter":
                    delimiter,

                "columns":
                    columns,

                "normalized_columns":
                    [
                        normalize_column(x)
                        for x in columns
                    ],
            }

    return None


def classify_column(norm):

    if norm in COVERAGE_EXACT:

        return "COVERAGE_ONLY"

    if norm in PRIMARY_SITE_EXACT:

        return "PRIMARY_SITE_NOT_METASTATIC_STREAM"

    if norm in SPECIMEN_SITE_EXACT:

        return "SPECIMEN_SITE_NOT_DISEASE_STATE"

    if norm in HIGH_CONFIDENCE_EXACT:

        return "HIGH_CONFIDENCE_DISEASE_SITE_CANDIDATE"

    if norm in AMBIGUOUS_EXACT:

        return "AMBIGUOUS_SITE_COLUMN"

    if (
        "METAST" in norm
        and (
            "SITE" in norm
            or "LOCATION" in norm
            or "ORGAN" in norm
        )
    ):

        return "HIGH_CONFIDENCE_DISEASE_SITE_CANDIDATE"

    if (
        "DISEASE" in norm
        and (
            "SITE" in norm
            or "LOCATION" in norm
        )
    ):

        return "HIGH_CONFIDENCE_DISEASE_SITE_CANDIDATE"

    if (
        "LESION" in norm
        and (
            "SITE" in norm
            or "LOCATION" in norm
        )
    ):

        return "HIGH_CONFIDENCE_DISEASE_SITE_CANDIDATE"

    if (
        "TUMOR" in norm
        and (
            "SITE" in norm
            or "LOCATION" in norm
        )
    ):

        return "HIGH_CONFIDENCE_DISEASE_SITE_CANDIDATE"

    return None


def has_outcome_like_columns(
    normalized_columns,
):

    hits = []

    for column in normalized_columns:

        for pattern in OUTCOME_LIKE_PATTERNS:

            if pattern.search(column):

                hits.append(
                    column
                )

                break

    return sorted(
        set(hits)
    )


def infer_sep(path):

    name = path.name.lower()

    if (
        name.endswith(".tsv")
        or name.endswith(".tsv.gz")
        or name.endswith(".txt")
        or name.endswith(".txt.gz")
    ):

        return "\t"

    return ","


def profile_values(
    path,
    raw_column,
    normalized_column,
    classification,
    safe_keys,
):

    ###########################################################################
    # Exact usecols only.
    #
    # No survival/outcome columns can be pulled accidentally.
    ###########################################################################

    usecols = list(
        dict.fromkeys(
            safe_keys
            + [
                raw_column
            ]
        )
    )

    try:

        df = pd.read_csv(
            path,
            sep=infer_sep(path),
            comment="#",
            usecols=usecols,
            dtype=str,
            compression="infer",
            low_memory=False,
        )

    except Exception as exc:

        return {
            "status":
                "SAFE_COLUMN_READ_FAILED",

            "error":
                repr(exc),

            "column":
                raw_column,

            "normalized_column":
                normalized_column,

            "classification":
                classification,
        }

    if raw_column not in df.columns:

        return {
            "status":
                "SAFE_COLUMN_MISSING_AFTER_READ",

            "column":
                raw_column,

            "normalized_column":
                normalized_column,

            "classification":
                classification,
        }

    values = (
        df[
            raw_column
        ]
        .dropna()
        .astype(str)
        .str.strip()
    )

    values = values[
        values
        != ""
    ]

    counts = Counter(
        values.tolist()
    )

    target_hits = {
        site:
            0
        for site in TARGET_SITES
    }

    matching_examples = {
        site:
            []
        for site in TARGET_SITES
    }

    for value, count in counts.items():

        upper = value.upper()

        for site, tokens in TARGET_SITES.items():

            if any(
                token in upper
                for token in tokens
            ):

                target_hits[
                    site
                ] += int(
                    count
                )

                if len(
                    matching_examples[
                        site
                    ]
                ) < 10:

                    matching_examples[
                        site
                    ].append(
                        value
                    )

    distinct_target_sites = [
        site
        for site, count in target_hits.items()
        if count > 0
    ]

    id_columns_present = [
        key
        for key in safe_keys
        if key in df.columns
        and normalize_column(
            key
        ) in {
            "PATIENT_ID",
            "PATIENTID",
            "PATIENT",
        }
    ]

    date_columns_present = [
        key
        for key in safe_keys
        if key in df.columns
        and normalize_column(
            key
        ) in {
            "START_DATE",
            "STOP_DATE",
            "DATE",
            "EVENT_DATE",
            "SCAN_DATE",
            "IMAGING_DATE",
        }
    ]

    return {
        "status":
            "SAFE_COLUMN_PROFILED",

        "column":
            raw_column,

        "normalized_column":
            normalized_column,

        "classification":
            classification,

        "rows_read":
            int(
                len(df)
            ),

        "nonempty_values":
            int(
                len(values)
            ),

        "distinct_values":
            int(
                len(counts)
            ),

        "top_values":
            [
                {
                    "value":
                        value,

                    "count":
                        int(count),
                }
                for value, count
                in counts.most_common(
                    40
                )
            ],

        "target_site_hits":
            target_hits,

        "target_site_matching_examples":
            matching_examples,

        "distinct_target_sites_present":
            distinct_target_sites,

        "patient_key_available":
            bool(
                id_columns_present
            ),

        "patient_key_columns":
            id_columns_present,

        "date_available":
            bool(
                date_columns_present
            ),

        "date_columns":
            date_columns_present,
    }


def sha256_file(path):

    h = hashlib.sha256()

    with path.open("rb") as f:

        for block in iter(
            lambda:
                f.read(
                    1024 * 1024
                ),
            b"",
        ):

            h.update(block)

    return h.hexdigest()


def write_json(
    path,
    payload,
):

    tmp = Path(
        str(path)
        + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    json.loads(
        tmp.read_text(
            encoding="utf-8"
        )
    )

    tmp.replace(path)


def main():

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    ###########################################################################
    # Confirm frozen protocol.
    ###########################################################################

    r3e_qc = json.loads(
        (
            R3E
            / "freeze_qc.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    adapter = json.loads(
        (
            R3E
            / "external_adapter_contract.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    if (
        r3e_qc[
            "remaining_blocker"
        ]
        != "DISEASE_SITE_FEATURE_TRANSPORT_GATE"
    ):

        raise RuntimeError(
            "Unexpected R3E blocker."
        )

    if (
        adapter[
            "scan_features"
        ][
            "SCAN_SITES_as_disease_site_positivity"
        ]
        != "PROHIBITED"
    ):

        raise RuntimeError(
            "R3E SCAN_SITES prohibition missing."
        )

    if r3e_qc[
        "external_outcomes_opened"
    ]:

        raise RuntimeError(
            "R3E says external outcomes were opened."
        )

    ###########################################################################
    # Inventory all text-like files, skipping quarantine before open().
    ###########################################################################

    inventory = []
    profiles = []

    quarantined_seen = []

    all_files = sorted(
        path
        for path in BPC.rglob("*")
        if path.is_file()
    )

    for path in all_files:

        rel = rel_string(path)

        if is_quarantined(rel):

            quarantined_seen.append(
                {
                    "path":
                        rel,

                    "bytes":
                        int(
                            path.stat().st_size
                        ),

                    "opened":
                        False,
                }
            )

            continue

        if not path.name.lower().endswith(
            TEXT_SUFFIXES
        ):

            continue

        header = discover_header(
            path
        )

        if header is None:

            inventory.append(
                {
                    "path":
                        rel,

                    "bytes":
                        int(
                            path.stat().st_size
                        ),

                    "status":
                        "NO_READABLE_HEADER",

                    "opened_header_only":
                        True,
                }
            )

            continue

        raw_columns = header[
            "columns"
        ]

        normalized = header[
            "normalized_columns"
        ]

        classified = []

        for raw, norm in zip(
            raw_columns,
            normalized,
        ):

            classification = classify_column(
                norm
            )

            if classification is None:

                continue

            classified.append(
                {
                    "column":
                        raw,

                    "normalized_column":
                        norm,

                    "classification":
                        classification,
                }
            )

        outcome_like = has_outcome_like_columns(
            normalized
        )

        inventory.append(
            {
                "path":
                    rel,

                "bytes":
                    int(
                        path.stat().st_size
                    ),

                "status":
                    "HEADER_AUDITED",

                "delimiter":
                    (
                        "TAB"
                        if header[
                            "delimiter"
                        ]
                        == "\t"
                        else "COMMA"
                    ),

                "column_count":
                    len(
                        raw_columns
                    ),

                "site_like_columns":
                    classified,

                "outcome_like_header_columns":
                    outcome_like,

                "opened_header_only":
                    True,
            }
        )

        if not classified:

            continue

        #######################################################################
        # Safe key columns: only identifiers / timing.
        #######################################################################

        safe_keys = [
            raw
            for raw, norm
            in zip(
                raw_columns,
                normalized,
            )
            if norm in SAFE_KEY_EXACT
        ]

        #######################################################################
        # Value profiling is allowed only for site-like columns and safe keys.
        #
        # Even when a non-quarantined file has outcome-looking columns, those
        # columns are never in usecols.
        #######################################################################

        for item in classified:

            profile = profile_values(
                path=path,
                raw_column=item[
                    "column"
                ],
                normalized_column=item[
                    "normalized_column"
                ],
                classification=item[
                    "classification"
                ],
                safe_keys=safe_keys,
            )

            profile[
                "path"
            ] = rel

            profile[
                "outcome_like_columns_not_read"
            ] = outcome_like

            profiles.append(
                profile
            )

    ###########################################################################
    # Verify exact quarantine coverage.
    ###########################################################################

    seen_paths = {
        item[
            "path"
        ]
        for item
        in quarantined_seen
    }

    existing_quarantined = {
        rel
        for rel in QUARANTINED
        if (
            BPC
            / rel
        ).exists()
    }

    if seen_paths != existing_quarantined:

        raise RuntimeError(
            "Quarantine accounting mismatch: "
            f"seen={sorted(seen_paths)} "
            f"existing={sorted(existing_quarantined)}"
        )

    if any(
        item[
            "opened"
        ]
        for item
        in quarantined_seen
    ):

        raise RuntimeError(
            "A quarantined file was opened."
        )

    ###########################################################################
    # Candidate decision.
    #
    # A candidate must:
    # - have disease-site-like naming,
    # - actually contain at least one of the six target organ concepts,
    # - have patient identity,
    # - have temporal information.
    #
    # It is NOT declared equivalent yet. BPC-MSK positive-control validation is
    # required in the next checkpoint.
    ###########################################################################

    high_confidence_candidates = []

    ambiguous_candidates = []

    coverage_profiles = []

    primary_or_specimen_profiles = []

    for profile in profiles:

        if profile[
            "status"
        ] != "SAFE_COLUMN_PROFILED":

            continue

        classification = profile[
            "classification"
        ]

        if classification == "COVERAGE_ONLY":

            coverage_profiles.append(
                profile
            )

            continue

        if classification in {
            "PRIMARY_SITE_NOT_METASTATIC_STREAM",
            "SPECIMEN_SITE_NOT_DISEASE_STATE",
        }:

            primary_or_specimen_profiles.append(
                profile
            )

            continue

        target_sites = profile[
            "distinct_target_sites_present"
        ]

        if (
            classification
            == "HIGH_CONFIDENCE_DISEASE_SITE_CANDIDATE"
            and target_sites
            and profile[
                "patient_key_available"
            ]
            and profile[
                "date_available"
            ]
        ):

            high_confidence_candidates.append(
                profile
            )

        elif (
            target_sites
            and profile[
                "patient_key_available"
            ]
        ):

            ambiguous_candidates.append(
                profile
            )

    ###########################################################################
    # SCAN_SITES can never make the candidate list.
    ###########################################################################

    for profile in high_confidence_candidates:

        if profile[
            "normalized_column"
        ] in COVERAGE_EXACT:

            raise RuntimeError(
                "Coverage column entered disease-site candidate set."
            )

    if high_confidence_candidates:

        status = (
            "CANDIDATE_DISEASE_SITE_STREAM_FOUND_"
            "REQUIRES_MSK_POSITIVE_CONTROL"
        )

        next_action = (
            "Run BPC-MSK positive-control semantic bridge for the candidate "
            "stream using frozen MSK clock offsets. Require replay against "
            "CHORD site_bone/site_liver/site_lung/site_brain/site_lymph/"
            "site_pleura semantics before external prediction generation."
        )

        prediction_ready = False

    else:

        status = (
            "NO_SAFE_EQUIVALENT_DISEASE_SITE_STREAM_FOUND"
        )

        next_action = (
            "Run CHORD-only current-scan site-channel ablation with frozen "
            "CKPT7R1/R2/R3B alpha=0.5. Based on semantic unavailability, "
            "predeclare a missing-site transport transformation before "
            "generating DFCI/VICC predictions. Do not inspect external "
            "outcomes."
        )

        prediction_ready = False

    ###########################################################################
    # Write artifacts.
    ###########################################################################

    schema_report = {
        "status":
            "PASS_OUTCOME_BLIND_BPC_SCHEMA_AUDIT",

        "bpc_root":
            str(
                BPC.relative_to(
                    ROOT
                )
            ),

        "files_total":
            len(
                all_files
            ),

        "text_files_audited":
            sum(
                item.get(
                    "status"
                )
                == "HEADER_AUDITED"
                for item
                in inventory
            ),

        "quarantined_files_existing":
            sorted(
                existing_quarantined
            ),

        "quarantined_files_opened":
            [],

        "inventory":
            inventory,

        "external_outcomes_opened":
            False,
    }

    write_json(
        OUT
        / "schema_inventory.json",
        schema_report,
    )

    profile_report = {
        "status":
            "PASS_SAFE_SITE_COLUMN_PROFILING",

        "profiles":
            profiles,

        "high_confidence_candidates":
            [
                {
                    "path":
                        item[
                            "path"
                        ],

                    "column":
                        item[
                            "column"
                        ],

                    "normalized_column":
                        item[
                            "normalized_column"
                        ],

                    "distinct_target_sites_present":
                        item[
                            "distinct_target_sites_present"
                        ],

                    "patient_key_columns":
                        item[
                            "patient_key_columns"
                        ],

                    "date_columns":
                        item[
                            "date_columns"
                        ],

                    "target_site_hits":
                        item[
                            "target_site_hits"
                        ],

                    "top_values":
                        item[
                            "top_values"
                        ][:20],
                }
                for item
                in high_confidence_candidates
            ],

        "ambiguous_candidates":
            [
                {
                    "path":
                        item[
                            "path"
                        ],

                    "column":
                        item[
                            "column"
                        ],

                    "classification":
                        item[
                            "classification"
                        ],

                    "distinct_target_sites_present":
                        item[
                            "distinct_target_sites_present"
                        ],

                    "patient_key_columns":
                        item[
                            "patient_key_columns"
                        ],

                    "date_columns":
                        item[
                            "date_columns"
                        ],

                    "top_values":
                        item[
                            "top_values"
                        ][:20],
                }
                for item
                in ambiguous_candidates
            ],

        "coverage_columns":
            [
                {
                    "path":
                        item[
                            "path"
                        ],

                    "column":
                        item[
                            "column"
                        ],

                    "classification":
                        item[
                            "classification"
                        ],

                    "top_values":
                        item[
                            "top_values"
                        ][:20],
                }
                for item
                in coverage_profiles
            ],

        "primary_or_specimen_site_columns":
            [
                {
                    "path":
                        item[
                            "path"
                        ],

                    "column":
                        item[
                            "column"
                        ],

                    "classification":
                        item[
                            "classification"
                        ],

                    "top_values":
                        item[
                            "top_values"
                        ][:20],
                }
                for item
                in primary_or_specimen_profiles
            ],

        "external_outcomes_opened":
            False,
    }

    write_json(
        OUT
        / "site_candidate_profiles.json",
        profile_report,
    )

    decision = {
        "status":
            status,

        "high_confidence_candidate_count":
            len(
                high_confidence_candidates
            ),

        "ambiguous_candidate_count":
            len(
                ambiguous_candidates
            ),

        "coverage_column_count":
            len(
                coverage_profiles
            ),

        "SCAN_SITES_as_disease_site_positivity":
            "PROHIBITED",

        "candidate_equivalence_established":
            False,

        "external_prediction_ready":
            prediction_ready,

        "external_outcomes_opened":
            False,

        "next_action":
            next_action,
    }

    write_json(
        OUT
        / "decision.json",
        decision,
    )

    ###########################################################################
    # Manifest
    ###########################################################################

    generated = [
        OUT
        / "schema_inventory.json",

        OUT
        / "site_candidate_profiles.json",

        OUT
        / "decision.json",
    ]

    manifest = {
        "status":
            status,

        "external_prediction_ready":
            False,

        "external_outcomes_opened":
            False,

        "files": {
            str(
                path.relative_to(
                    ROOT
                )
            ):
                {
                    "bytes":
                        int(
                            path.stat().st_size
                        ),

                    "sha256":
                        sha256_file(
                            path
                        ),
                }
            for path
            in generated
        },
    }

    write_json(
        OUT
        / "manifest.json",
        manifest,
    )

    print(
        "[CKPT7R3F1_SITE_SOURCE_AUDIT_PASS]",
        {
            "status":
                status,

            "text_files_audited":
                schema_report[
                    "text_files_audited"
                ],

            "quarantined_files_existing":
                len(
                    existing_quarantined
                ),

            "quarantined_files_opened":
                0,

            "high_confidence_candidates":
                len(
                    high_confidence_candidates
                ),

            "ambiguous_candidates":
                len(
                    ambiguous_candidates
                ),

            "coverage_columns":
                len(
                    coverage_profiles
                ),

            "external_prediction_ready":
                False,

            "external_outcomes_opened":
                False,
        },
    )

    for item in high_confidence_candidates:

        print(
            "[CKPT7R3F1_HIGH_CONFIDENCE_CANDIDATE]",
            {
                "path":
                    item[
                        "path"
                    ],

                "column":
                    item[
                        "column"
                    ],

                "target_sites":
                    item[
                        "distinct_target_sites_present"
                    ],

                "patient_keys":
                    item[
                        "patient_key_columns"
                    ],

                "date_columns":
                    item[
                        "date_columns"
                    ],

                "target_site_hits":
                    item[
                        "target_site_hits"
                    ],
            },
        )

    for item in ambiguous_candidates:

        print(
            "[CKPT7R3F1_AMBIGUOUS_CANDIDATE]",
            {
                "path":
                    item[
                        "path"
                    ],

                "column":
                    item[
                        "column"
                    ],

                "classification":
                    item[
                        "classification"
                    ],

                "target_sites":
                    item[
                        "distinct_target_sites_present"
                    ],

                "patient_keys":
                    item[
                        "patient_key_columns"
                    ],

                "date_columns":
                    item[
                        "date_columns"
                    ],
            },
        )

    for item in coverage_profiles:

        print(
            "[CKPT7R3F1_COVERAGE_ONLY]",
            {
                "path":
                    item[
                        "path"
                    ],

                "column":
                    item[
                        "column"
                    ],

                "note":
                    "Never use as tumor-site positivity.",
            },
        )

    print(
        "[CKPT7R3F1_NEXT_ACTION]",
        next_action,
    )


if __name__ == "__main__":

    main()
