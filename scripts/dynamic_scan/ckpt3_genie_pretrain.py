#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import hashlib
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

from sklearn.metrics import average_precision_score

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import (
    DataLoader,
    Dataset,
    DistributedSampler,
)


SEED = 20260920
GENE_COUNT = 468

# External-validation institutions are excluded from molecular pretraining.
# MSK remains a development/training institution because MSK-CHORD is the
# primary longitudinal training cohort and BPC-MSK is development-only.
EXCLUDED_PRIMARY_CENTERS = {
    "DFCI",
    "VICC",
}

# Development centers should not accidentally become synthetic
# institution-holdout cohorts.
NON_HOLDOUT_DEVELOPMENT_CENTERS = {
    "MSK",
}

PATHWAY_GROUPS = [
    [
        "PIK3CA",
        "PIK3R1",
        "PTEN",
        "AKT1",
        "AKT2",
        "MTOR",
        "TSC1",
        "TSC2",
    ],
    [
        "ERBB2",
        "ERBB3",
        "EGFR",
        "GRB2",
        "SOS1",
        "KRAS",
        "NRAS",
        "BRAF",
        "MAP2K1",
        "MAPK1",
    ],
    [
        "ESR1",
        "GATA3",
        "FOXA1",
        "NCOR1",
        "NCOA3",
    ],
    [
        "BRCA1",
        "BRCA2",
        "PALB2",
        "ATM",
        "ATR",
        "CHEK1",
        "CHEK2",
        "RAD51",
        "RAD51C",
        "RAD51D",
    ],
    [
        "TP53",
        "MDM2",
        "MDM4",
        "ATM",
        "CHEK2",
        "CDKN2A",
    ],
    [
        "CCND1",
        "CCND2",
        "CCND3",
        "CDK4",
        "CDK6",
        "CDKN2A",
        "CDKN2B",
        "RB1",
    ],
    [
        "FGFR1",
        "FGFR2",
        "FGFR3",
        "FGFR4",
        "FRS2",
    ],
    [
        "ARID1A",
        "ARID1B",
        "SMARCA4",
        "SMARCB1",
        "PBRM1",
    ],
]


###############################################################################
# General utilities
###############################################################################


def norm(value: Any) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value).strip().lower(),
    ).strip("_")


def stable_hash(
    value: str,
) -> int:
    return int(
        hashlib.sha256(
            value.encode("utf-8")
        ).hexdigest()[:16],
        16,
    )


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
            f"missing columns {list(candidates)}; "
            f"available={list(columns)}"
        )

    return None


def read_table(
    path: Path,
    **kwargs,
) -> pd.DataFrame:

    return pd.read_csv(
        path,
        sep="\t",
        comment="#",
        dtype=str,
        keep_default_na=False,
        low_memory=False,
        **kwargs,
    )


def infer_center(
    patient_id: str,
    explicit: str = "",
) -> str:

    explicit = str(
        explicit
    ).strip()

    if explicit:

        upper = explicit.upper()

        aliases = {
            "MEMORIAL SLOAN KETTERING":
                "MSK",

            "MEMORIAL SLOAN KETTERING CANCER CENTER":
                "MSK",

            "DANA-FARBER":
                "DFCI",

            "DANA FARBER":
                "DFCI",

            "DANA-FARBER CANCER INSTITUTE":
                "DFCI",

            "VANDERBILT":
                "VICC",

            "VANDERBILT-INGRAM":
                "VICC",
        }

        for key, value in aliases.items():

            if key in upper:
                return value

        if re.fullmatch(
            r"[A-Z0-9]{2,12}",
            upper,
        ):
            return upper

    text = str(
        patient_id
    ).strip()

    match = re.match(
        r"^GENIE-([A-Za-z0-9]+)-",
        text,
    )

    if match:
        return (
            match
            .group(1)
            .upper()
        )

    parts = text.split(
        "-"
    )

    if (
        len(parts) >= 2
        and parts[0].upper()
        == "GENIE"
    ):
        return (
            parts[1]
            .upper()
        )

    return "UNKNOWN"


###############################################################################
# GENIE source discovery
###############################################################################


def discover_sources(
    root: Path,
) -> dict[str, list[str]]:

    categories: dict[
        str,
        list[str],
    ] = {
        "clinical_sample": [],
        "mutation": [],
        "gene_panel_matrix": [],
        "gene_panel_definition": [],
        "cna": [],
        "fusion": [],
    }

    for path in root.rglob(
        "*"
    ):

        if not path.is_file():
            continue

        name = path.name.lower()

        if name == "data_clinical_sample.txt":
            categories[
                "clinical_sample"
            ].append(
                str(path)
            )

        if (
            name.startswith(
                "data_mutations"
            )
            and name.endswith(
                ".txt"
            )
        ):
            categories[
                "mutation"
            ].append(
                str(path)
            )

        if name == "data_gene_panel_matrix.txt":
            categories[
                "gene_panel_matrix"
            ].append(
                str(path)
            )

        if (
            "gene_panel"
            in name
            and "matrix"
            not in name
            and "meta_"
            not in name
            and name.endswith(
                ".txt"
            )
        ):
            categories[
                "gene_panel_definition"
            ].append(
                str(path)
            )

        if (
            (
                "cna"
                in name
                or "copy_number"
                in name
            )
            and name.endswith(
                (
                    ".txt",
                    ".seg",
                )
            )
        ):
            categories[
                "cna"
            ].append(
                str(path)
            )

        if (
            (
                "fusion"
                in name
                or "structural_variant"
                in name
            )
            and name.endswith(
                ".txt"
            )
        ):
            categories[
                "fusion"
            ].append(
                str(path)
            )

    for key in categories:
        categories[
            key
        ] = sorted(
            set(
                categories[
                    key
                ]
            )
        )

    return categories


###############################################################################
# Panel definitions / clinical sample map
###############################################################################


def parse_panel_file(
    path: Path,
) -> tuple[
    str | None,
    set[str],
]:

    panel_id = None
    genes: set[
        str
    ] = set()

    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    for raw in text.splitlines():

        line = raw.strip()

        if not line:
            continue

        lower = line.lower()

        if (
            lower.startswith(
                "stable_id:"
            )
            or lower.startswith(
                "gene_panel_id:"
            )
            or lower.startswith(
                "panel_id:"
            )
        ):

            panel_id = (
                line
                .split(
                    ":",
                    1,
                )[
                    1
                ]
                .strip()
            )

        elif lower.startswith(
            "gene_list:"
        ):

            payload = (
                line
                .split(
                    ":",
                    1,
                )[
                    1
                ]
            )

            for token in re.split(
                r"[\s,;]+",
                payload,
            ):

                token = (
                    token
                    .strip()
                    .upper()
                )

                if token:
                    genes.add(
                        token
                    )

    if (
        not genes
        and "\t"
        in text
    ):

        try:

            frame = read_table(
                path
            )

            gene_col = find_col(
                frame.columns,
                [
                    "Hugo_Symbol",
                    "gene",
                    "gene_symbol",
                ],
                required=False,
            )

            if gene_col:

                genes.update(
                    frame[
                        gene_col
                    ]
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    .replace(
                        "",
                        np.nan,
                    )
                    .dropna()
                    .tolist()
                )

        except Exception:
            pass

    if panel_id is None:

        panel_id = re.sub(
            r"\.(txt|tsv|csv)$",
            "",
            path.name,
            flags=re.I,
        )

        panel_id = re.sub(
            r"^data_gene_panel_",
            "",
            panel_id,
            flags=re.I,
        )

    return (
        panel_id,
        genes,
    )


def build_panel_definitions(
    source_paths: list[str],
) -> dict[
    str,
    set[str],
]:

    panels: dict[
        str,
        set[str],
    ] = {}

    for raw_path in source_paths:

        path = Path(
            raw_path
        )

        panel_id, genes = (
            parse_panel_file(
                path
            )
        )

        if (
            panel_id
            and genes
        ):

            panels[
                str(
                    panel_id
                ).strip()
            ] = genes

    return panels


def load_clinical_samples(
    paths: list[str],
) -> pd.DataFrame:

    frames = []

    for raw_path in paths:

        path = Path(
            raw_path
        )

        frame = read_table(
            path
        )

        sample_col = find_col(
            frame.columns,
            [
                "SAMPLE_ID",
                "sample_id",
            ],
        )

        patient_col = find_col(
            frame.columns,
            [
                "PATIENT_ID",
                "patient_id",
            ],
        )

        cancer_col = find_col(
            frame.columns,
            [
                "CANCER_TYPE",
                "CANCER_TYPE_DETAILED",
                "ONCOTREE_CODE",
                "PRIMARY_SITE",
            ],
            required=False,
        )

        sample_type_col = find_col(
            frame.columns,
            [
                "SAMPLE_TYPE",
                "SAMPLE_CLASS",
                "SPECIMEN_TYPE",
                "PRIMARY_METASTASIS",
            ],
            required=False,
        )

        panel_col = find_col(
            frame.columns,
            [
                "SEQ_ASSAY_ID",
                "GENE_PANEL",
                "GENE_PANEL_ID",
                "ASSAY_ID",
                "PANEL_ID",
            ],
            required=False,
        )

        center_col = find_col(
            frame.columns,
            [
                "CENTER",
                "CENTER_ID",
                "INSTITUTION",
                "INSTITUTION_ID",
                "SITE",
            ],
            required=False,
        )

        current = pd.DataFrame(
            {
                "sample_id":
                    frame[
                        sample_col
                    ]
                    .astype(str)
                    .str.strip(),

                "patient_id":
                    frame[
                        patient_col
                    ]
                    .astype(str)
                    .str.strip(),
            }
        )

        current[
            "cancer_type"
        ] = (
            frame[
                cancer_col
            ]
            .astype(str)
            .str.strip()
            if cancer_col
            else ""
        )

        current[
            "sample_type"
        ] = (
            frame[
                sample_type_col
            ]
            .astype(str)
            .str.strip()
            if sample_type_col
            else ""
        )

        current[
            "panel_from_clinical"
        ] = (
            frame[
                panel_col
            ]
            .astype(str)
            .str.strip()
            if panel_col
            else ""
        )

        explicit_center = (
            frame[
                center_col
            ]
            .astype(str)
            .str.strip()
            if center_col
            else pd.Series(
                "",
                index=frame.index,
            )
        )

        current[
            "center"
        ] = [
            infer_center(
                patient,
                center,
            )
            for patient, center
            in zip(
                current[
                    "patient_id"
                ],
                explicit_center,
            )
        ]

        current[
            "clinical_source"
        ] = str(
            path
        )

        frames.append(
            current
        )

    if not frames:

        raise RuntimeError(
            "No GENIE clinical sample tables found."
        )

    result = pd.concat(
        frames,
        ignore_index=True,
    )

    result = result[
        (
            result[
                "sample_id"
            ]
            != ""
        )
        & (
            result[
                "patient_id"
            ]
            != ""
        )
    ].copy()

    result = (
        result
        .drop_duplicates(
            subset=[
                "sample_id",
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    return result


def load_panel_matrix(
    paths: list[str],
) -> dict[
    str,
    str
]:

    mapping: dict[
        str,
        str
    ] = {}

    for raw_path in paths:

        frame = read_table(
            Path(
                raw_path
            )
        )

        sample_col = find_col(
            frame.columns,
            [
                "SAMPLE_ID",
                "sample_id",
            ],
            required=False,
        )

        if sample_col is None:
            continue

        mutation_panel_col = None

        for column in frame.columns:

            normalized = norm(
                column
            )

            if (
                "mut"
                in normalized
                and (
                    "panel"
                    in normalized
                    or "profile"
                    in normalized
                )
            ):

                mutation_panel_col = column
                break

        if mutation_panel_col is None:

            candidates = [
                column
                for column
                in frame.columns
                if column
                != sample_col
            ]

            if len(
                candidates
            ) == 1:
                mutation_panel_col = (
                    candidates[
                        0
                    ]
                )

        if mutation_panel_col is None:
            continue

        for sample, panel in zip(
            frame[
                sample_col
            ],
            frame[
                mutation_panel_col
            ],
        ):

            sample = str(
                sample
            ).strip()

            panel = str(
                panel
            ).strip()

            if (
                sample
                and panel
            ):

                mapping[
                    sample
                ] = panel

    return mapping


###############################################################################
# Mutation streaming
###############################################################################


def mutation_header(
    path: Path,
) -> dict[
    str,
    str | None
]:

    header = pd.read_csv(
        path,
        sep="\t",
        comment="#",
        dtype=str,
        nrows=0,
    )

    columns = list(
        header.columns
    )

    return {
        "sample":
            find_col(
                columns,
                [
                    "Tumor_Sample_Barcode",
                    "SAMPLE_ID",
                    "sample_id",
                ],
            ),

        "gene":
            find_col(
                columns,
                [
                    "Hugo_Symbol",
                    "gene",
                    "gene_symbol",
                ],
            ),

        "chromosome":
            find_col(
                columns,
                [
                    "Chromosome",
                    "chr",
                ],
                required=False,
            ),

        "start":
            find_col(
                columns,
                [
                    "Start_Position",
                    "start_position",
                    "position",
                ],
                required=False,
            ),
    }


def mutation_pass_frequency(
    paths: list[str],
    valid_samples: set[str],
) -> tuple[
    Counter,
    dict[
        str,
        tuple[
            str,
            int,
        ]
    ],
    int,
]:

    gene_rows: Counter = Counter()

    positions: dict[
        str,
        tuple[
            str,
            int,
        ]
    ] = {}

    retained_rows = 0

    for path_index, raw_path in enumerate(
        paths,
        start=1,
    ):

        path = Path(
            raw_path
        )

        columns = mutation_header(
            path
        )

        usecols = [
            columns[
                "sample"
            ],
            columns[
                "gene"
            ],
        ]

        if columns[
            "chromosome"
        ]:
            usecols.append(
                columns[
                    "chromosome"
                ]
            )

        if columns[
            "start"
        ]:
            usecols.append(
                columns[
                    "start"
                ]
            )

        usecols = list(
            dict.fromkeys(
                usecols
            )
        )

        print(
            "[CKPT3_PREP] mutation frequency pass "
            f"file={path_index}/{len(paths)} "
            f"path={path}",
            flush=True,
        )

        for chunk in pd.read_csv(
            path,
            sep="\t",
            comment="#",
            dtype=str,
            usecols=usecols,
            chunksize=250000,
            low_memory=False,
        ):

            sample = (
                chunk[
                    columns[
                        "sample"
                    ]
                ]
                .astype(str)
                .str.strip()
            )

            keep = sample.isin(
                valid_samples
            )

            if not keep.any():
                continue

            selected = chunk.loc[
                keep
            ]

            genes = (
                selected[
                    columns[
                        "gene"
                    ]
                ]
                .astype(str)
                .str.strip()
                .str.upper()
            )

            genes = genes[
                genes
                != ""
            ]

            gene_rows.update(
                genes.tolist()
            )

            retained_rows += int(
                len(
                    genes
                )
            )

            if (
                columns[
                    "chromosome"
                ]
                and columns[
                    "start"
                ]
            ):

                for (
                    gene,
                    chromosome,
                    start,
                ) in zip(
                    selected[
                        columns[
                            "gene"
                        ]
                    ],
                    selected[
                        columns[
                            "chromosome"
                        ]
                    ],
                    selected[
                        columns[
                            "start"
                        ]
                    ],
                ):

                    gene = (
                        str(
                            gene
                        )
                        .strip()
                        .upper()
                    )

                    chromosome = str(
                        chromosome
                    ).strip()

                    try:
                        start_int = int(
                            float(
                                start
                            )
                        )
                    except Exception:
                        continue

                    if (
                        gene
                        and gene
                        not in positions
                    ):

                        positions[
                            gene
                        ] = (
                            chromosome,
                            start_int,
                        )

    return (
        gene_rows,
        positions,
        retained_rows,
    )


###############################################################################
# Dataset preparation
###############################################################################


def choose_holdouts(
    clinical: pd.DataFrame,
) -> tuple[
    list[str],
    list[str],
]:

    eligible = clinical[
        ~clinical[
            "center"
        ].isin(
            EXCLUDED_PRIMARY_CENTERS
            | NON_HOLDOUT_DEVELOPMENT_CENTERS
        )
        & (
            clinical[
                "panel_id"
            ]
            != ""
        )
    ].copy()

    center_counts = (
        eligible[
            "center"
        ]
        .value_counts()
    )

    center_candidates = [
        str(
            center
        )
        for center, count
        in center_counts.items()
        if (
            center
            != "UNKNOWN"
            and count
            >= 1000
        )
    ]

    heldout_centers = (
        center_candidates[
            :2
        ]
    )

    panel_pool = eligible[
        ~eligible[
            "center"
        ].isin(
            heldout_centers
        )
    ]

    panel_counts = (
        panel_pool[
            "panel_id"
        ]
        .value_counts()
    )

    panel_candidates = [
        str(
            panel
        )
        for panel, count
        in panel_counts.items()
        if count
        >= 1000
    ]

    # Avoid taking the overwhelmingly dominant panel first if alternatives
    # exist: held-out panels should test generalization without deleting most
    # of the training cohort.
    total = max(
        len(
            panel_pool
        ),
        1,
    )

    reasonable = [
        panel
        for panel
        in panel_candidates
        if (
            panel_counts[
                panel
            ]
            / total
        )
        <= 0.20
    ]

    if len(
        reasonable
    ) >= 2:
        heldout_panels = reasonable[
            :2
        ]
    else:
        heldout_panels = panel_candidates[
            -2:
        ]

    return (
        heldout_centers,
        heldout_panels,
    )


def assign_splits(
    clinical: pd.DataFrame,
    exact_external_ids: set[str],
    heldout_centers: list[str],
    heldout_panels: list[str],
) -> pd.DataFrame:

    patient_groups = (
        clinical
        .groupby(
            "patient_id",
            observed=True,
        )
    )

    patient_split: dict[
        str,
        str
    ] = {}

    for patient_id, group in patient_groups:

        centers = set(
            group[
                "center"
            ]
        )

        panels = set(
            group[
                "panel_id"
            ]
        )

        if patient_id in exact_external_ids:

            split = (
                "excluded_exact_external"
            )

        elif centers & EXCLUDED_PRIMARY_CENTERS:

            split = (
                "excluded_center_disjoint"
            )

        elif centers & set(
            heldout_centers
        ):

            split = (
                "institution_holdout"
            )

        elif panels & set(
            heldout_panels
        ):

            split = (
                "panel_holdout"
            )

        else:

            bucket = (
                stable_hash(
                    patient_id
                )
                % 100
            )

            split = (
                "val"
                if bucket
                >= 90
                else "train"
            )

        patient_split[
            patient_id
        ] = split

    result = clinical.copy()

    result[
        "split"
    ] = (
        result[
            "patient_id"
        ]
        .map(
            patient_split
        )
    )

    return result


def select_gene_space(
    clinical: pd.DataFrame,
    panels: dict[
        str,
        set[str],
    ],
    mutation_counts: Counter,
) -> list[str]:

    train = clinical[
        clinical[
            "split"
        ]
        == "train"
    ]

    panel_sample_counts = (
        train[
            "panel_id"
        ]
        .value_counts()
        .to_dict()
    )

    coverage_weight: Counter = Counter()

    coverage_panel_count: Counter = Counter()

    for panel_id, count in panel_sample_counts.items():

        genes = panels.get(
            panel_id,
            set(),
        )

        for gene in genes:

            coverage_weight[
                gene
            ] += int(
                count
            )

            coverage_panel_count[
                gene
            ] += 1

    candidates = set(
        coverage_weight
    )

    if len(
        candidates
    ) < GENE_COUNT:

        raise RuntimeError(
            "Fewer than 468 genes have resolved "
            f"training-panel coverage: {len(candidates)}"
        )

    ranked = sorted(
        candidates,
        key=lambda gene: (
            mutation_counts.get(
                gene,
                0,
            ),
            coverage_weight.get(
                gene,
                0,
            ),
            coverage_panel_count.get(
                gene,
                0,
            ),
            gene,
        ),
        reverse=True,
    )

    return ranked[
        :GENE_COUNT
    ]


def build_graph(
    genes: list[str],
    positions: dict[
        str,
        tuple[
            str,
            int,
        ]
    ],
) -> np.ndarray:

    index = {
        gene: idx
        for idx, gene
        in enumerate(
            genes
        )
    }

    edges: set[
        tuple[
            int,
            int,
        ]
    ] = set()

    # Self-loops.
    for idx in range(
        len(
            genes
        )
    ):
        edges.add(
            (
                idx,
                idx,
            )
        )

    # Chromosomal neighborhood:
    # connect each selected gene to the two nearest selected genes
    # in either direction on the same chromosome.
    by_chromosome: dict[
        str,
        list[
            tuple[
                int,
                str,
            ]
        ],
    ] = defaultdict(
        list
    )

    for gene in genes:

        if gene not in positions:
            continue

        chromosome, position = (
            positions[
                gene
            ]
        )

        by_chromosome[
            str(
                chromosome
            )
        ].append(
            (
                int(
                    position
                ),
                gene,
            )
        )

    for chromosome, members in by_chromosome.items():

        members.sort()

        for i, (_, gene) in enumerate(
            members
        ):

            left = max(
                0,
                i - 2,
            )

            right = min(
                len(
                    members
                ),
                i + 3,
            )

            for j in range(
                left,
                right,
            ):

                if i == j:
                    continue

                other = members[
                    j
                ][
                    1
                ]

                edges.add(
                    (
                        index[
                            gene
                        ],
                        index[
                            other
                        ],
                    )
                )

    # Curated canonical cancer-pathway anchors.
    for group in PATHWAY_GROUPS:

        present = [
            gene
            for gene in group
            if gene
            in index
        ]

        for left in present:

            for right in present:

                if (
                    left
                    != right
                ):
                    edges.add(
                        (
                            index[
                                left
                            ],
                            index[
                                right
                            ],
                        )
                    )

    array = np.asarray(
        sorted(
            edges
        ),
        dtype=np.int64,
    ).T

    if (
        array.ndim
        != 2
        or array.shape[
            0
        ]
        != 2
    ):

        raise RuntimeError(
            "Invalid graph edge array."
        )

    return array


def prepare(
    repo: Path,
    out: Path,
) -> None:

    root = (
        repo
        / "data"
        / "external_sources"
        / "genie_20_0_public"
    )

    prep = (
        out
        / "prepared"
    )

    prep.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "[CKPT3_PREP] discovering GENIE source files",
        flush=True,
    )

    sources = discover_sources(
        root
    )

    atomic_json(
        prep
        / "source_discovery.json",
        sources,
    )

    if not sources[
        "clinical_sample"
    ]:

        raise RuntimeError(
            "No data_clinical_sample.txt files found."
        )

    if not sources[
        "mutation"
    ]:

        raise RuntimeError(
            "No GENIE mutation files found."
        )

    panels = build_panel_definitions(
        sources[
            "gene_panel_definition"
        ]
    )

    if not panels:

        raise RuntimeError(
            "No usable GENIE gene-panel definitions found."
        )

    clinical = load_clinical_samples(
        sources[
            "clinical_sample"
        ]
    )

    panel_matrix = load_panel_matrix(
        sources[
            "gene_panel_matrix"
        ]
    )

    clinical[
        "panel_id"
    ] = (
        clinical[
            "sample_id"
        ]
        .map(
            panel_matrix
        )
        .fillna(
            ""
        )
    )

    no_matrix = (
        clinical[
            "panel_id"
        ]
        == ""
    )

    clinical.loc[
        no_matrix,
        "panel_id",
    ] = clinical.loc[
        no_matrix,
        "panel_from_clinical",
    ]

    # Normalize exact panel IDs through exact and case-insensitive lookup.
    lower_panel_lookup = {
        str(
            key
        ).lower():
            key
        for key in panels
    }

    normalized_panel = []

    for panel_id in clinical[
        "panel_id"
    ]:

        panel_id = str(
            panel_id
        ).strip()

        if panel_id in panels:

            normalized_panel.append(
                panel_id
            )

        elif (
            panel_id.lower()
            in lower_panel_lookup
        ):

            normalized_panel.append(
                lower_panel_lookup[
                    panel_id.lower()
                ]
            )

        else:

            normalized_panel.append(
                ""
            )

    clinical[
        "panel_id"
    ] = normalized_panel

    external_path = (
        repo
        / "artifacts"
        / "checkpoint0e"
        / "genie_external_validation_patient_exclusions.txt"
    )

    exact_external_ids = {
        line.strip()
        for line
        in external_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    }

    heldout_centers, heldout_panels = (
        choose_holdouts(
            clinical
        )
    )

    clinical = assign_splits(
        clinical,
        exact_external_ids,
        heldout_centers,
        heldout_panels,
    )

    # Only samples with a valid panel can participate in coverage-aware
    # training/evaluation.
    modeled = clinical[
        clinical[
            "panel_id"
        ]
        != ""
    ].copy()

    modeled = (
        modeled
        .sort_values(
            [
                "sample_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    if len(
        modeled
    ) < 50000:

        raise RuntimeError(
            "Unexpectedly small panel-resolved GENIE cohort: "
            f"{len(modeled)}"
        )

    # IMPORTANT:
    # The 468-gene representation itself is a learned design decision.
    # Therefore mutation frequency used to rank the gene space must come
    # only from the primary training population. External institutions,
    # panel holdouts, institution holdouts, and validation patients must
    # not influence which genes enter the representation.
    gene_selection_samples = set(
        modeled.loc[
            modeled[
                "split"
            ]
            == "train",
            "sample_id",
        ]
    )

    (
        mutation_counts,
        positions,
        retained_mutation_rows,
    ) = mutation_pass_frequency(
        sources[
            "mutation"
        ],
        gene_selection_samples,
    )

    genes = select_gene_space(
        modeled,
        panels,
        mutation_counts,
    )

    gene_index = {
        gene: idx
        for idx, gene
        in enumerate(
            genes
        )
    }

    atomic_text(
        prep
        / "gene_space_468.txt",
        "\n".join(
            genes
        )
        + "\n",
    )

    graph_edges = build_graph(
        genes,
        positions,
    )

    np.save(
        prep
        / "graph_edges.npy",
        graph_edges,
    )

    ###########################################################################
    # Coverage tensor.
    ###########################################################################

    n = len(
        modeled
    )

    coverage_path = (
        prep
        / "coverage.npy"
    )

    alteration_path = (
        prep
        / "snv_indel.npy"
    )

    coverage = np.lib.format.open_memmap(
        coverage_path,
        mode="w+",
        dtype=np.uint8,
        shape=(
            n,
            GENE_COUNT,
        ),
    )

    alteration = np.lib.format.open_memmap(
        alteration_path,
        mode="w+",
        dtype=np.uint8,
        shape=(
            n,
            GENE_COUNT,
        ),
    )

    sample_to_index = {
        sample:
            idx
        for idx, sample
        in enumerate(
            modeled[
                "sample_id"
            ]
        )
    }

    panel_mask_cache: dict[
        str,
        np.ndarray
    ] = {}

    for panel_id, panel_genes in panels.items():

        mask = np.zeros(
            GENE_COUNT,
            dtype=np.uint8,
        )

        for gene in panel_genes:

            if gene in gene_index:

                mask[
                    gene_index[
                        gene
                    ]
                ] = 1

        panel_mask_cache[
            panel_id
        ] = mask

    for row_index, panel_id in enumerate(
        modeled[
            "panel_id"
        ]
    ):

        coverage[
            row_index
        ] = panel_mask_cache[
            panel_id
        ]

    coverage.flush()

    ###########################################################################
    # Mutation tensor second pass.
    ###########################################################################

    outside_coverage = 0

    mutation_assignments = 0

    for path_index, raw_path in enumerate(
        sources[
            "mutation"
        ],
        start=1,
    ):

        path = Path(
            raw_path
        )

        columns = mutation_header(
            path
        )

        print(
            "[CKPT3_PREP] mutation tensor pass "
            f"file={path_index}/{len(sources['mutation'])} "
            f"path={path}",
            flush=True,
        )

        for chunk in pd.read_csv(
            path,
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
            chunksize=250000,
            low_memory=False,
        ):

            samples = (
                chunk[
                    columns[
                        "sample"
                    ]
                ]
                .astype(str)
                .str.strip()
            )

            genes_chunk = (
                chunk[
                    columns[
                        "gene"
                    ]
                ]
                .astype(str)
                .str.strip()
                .str.upper()
            )

            for sample, gene in zip(
                samples,
                genes_chunk,
            ):

                row_index = (
                    sample_to_index.get(
                        sample
                    )
                )

                gene_idx = (
                    gene_index.get(
                        gene
                    )
                )

                if (
                    row_index is None
                    or gene_idx is None
                ):
                    continue

                alteration[
                    row_index,
                    gene_idx,
                ] = 1

                mutation_assignments += 1

                if (
                    coverage[
                        row_index,
                        gene_idx,
                    ]
                    == 0
                ):

                    outside_coverage += 1

    alteration.flush()

    ###########################################################################
    # Auxiliary labels.
    ###########################################################################

    train = modeled[
        modeled[
            "split"
        ]
        == "train"
    ]

    cancer_counts = (
        train[
            "cancer_type"
        ]
        .replace(
            "",
            "UNKNOWN",
        )
        .value_counts()
    )

    cancer_classes = list(
        cancer_counts.head(
            31
        ).index
    )

    if "OTHER" not in cancer_classes:
        cancer_classes.append(
            "OTHER"
        )

    cancer_to_index = {
        value: idx
        for idx, value
        in enumerate(
            cancer_classes
        )
    }

    modeled[
        "cancer_label"
    ] = [
        cancer_to_index.get(
            value
            if value
            else "UNKNOWN",
            cancer_to_index[
                "OTHER"
            ],
        )
        for value in modeled[
            "cancer_type"
        ]
    ]

    def simplify_sample_type(
        value: str,
    ) -> str:

        text = str(
            value
        ).upper()

        if "META" in text:
            return "METASTATIC"

        if "PRIMARY" in text:
            return "PRIMARY"

        return "OTHER"

    modeled[
        "sample_type_simple"
    ] = modeled[
        "sample_type"
    ].map(
        simplify_sample_type
    )

    sample_type_classes = [
        "PRIMARY",
        "METASTATIC",
        "OTHER",
    ]

    sample_type_to_index = {
        value: idx
        for idx, value
        in enumerate(
            sample_type_classes
        )
    }

    modeled[
        "sample_type_label"
    ] = modeled[
        "sample_type_simple"
    ].map(
        sample_type_to_index
    )

    ###########################################################################
    # Training prevalence / positive weights.
    ###########################################################################

    train_indices = np.where(
        modeled[
            "split"
        ].to_numpy()
        == "train"
    )[
        0
    ]

    if len(
        train_indices
    ) < 10000:

        raise RuntimeError(
            "Too few primary center-disjoint training samples: "
            f"{len(train_indices)}"
        )

    train_alteration = np.asarray(
        alteration[
            train_indices
        ],
        dtype=np.float32,
    )

    train_coverage = np.asarray(
        coverage[
            train_indices
        ],
        dtype=np.float32,
    )

    positive = train_alteration.sum(
        axis=0
    )

    observed = train_coverage.sum(
        axis=0
    )

    negative = np.maximum(
        observed
        - positive,
        1.0,
    )

    pos_weight = np.clip(
        negative
        / np.maximum(
            positive,
            1.0,
        ),
        1.0,
        50.0,
    ).astype(
        np.float32
    )

    np.save(
        prep
        / "positive_weights.npy",
        pos_weight,
    )

    ###########################################################################
    # Save metadata.
    ###########################################################################

    atomic_parquet(
        prep
        / "sample_metadata.parquet",
        modeled,
    )

    panel_rows = []

    for panel_id, mask in panel_mask_cache.items():

        panel_rows.append(
            {
                "panel_id":
                    panel_id,

                "selected_gene_coverage":
                    int(
                        mask.sum()
                    ),

                "full_gene_count":
                    int(
                        len(
                            panels[
                                panel_id
                            ]
                        )
                    ),
            }
        )

    atomic_parquet(
        prep
        / "panel_inventory.parquet",
        pd.DataFrame(
            panel_rows
        ),
    )

    split_counts = {}

    for split, group in modeled.groupby(
        "split"
    ):

        split_counts[
            str(
                split
            )
        ] = {
            "samples":
                int(
                    len(
                        group
                    )
                ),

            "patients":
                int(
                    group[
                        "patient_id"
                    ].nunique()
                ),
        }

    selected_gene_coverage = (
        np.asarray(
            coverage,
            dtype=np.uint8,
        )
        .sum(
            axis=1
        )
    )

    report = {
        "status":
            "PREPARED",

        "gene_selection_population":
            "primary_train_split_only",

        "source_discovery":
            {
                key:
                    len(
                        value
                    )
                for key, value
                in sources.items()
            },

        "panel_definitions":
            len(
                panels
            ),

        "clinical_samples_total":
            int(
                len(
                    clinical
                )
            ),

        "panel_resolved_samples":
            int(
                len(
                    modeled
                )
            ),

        "panel_resolution_fraction":
            float(
                len(
                    modeled
                )
                / max(
                    len(
                        clinical
                    ),
                    1,
                )
            ),

        "gene_space":
            GENE_COUNT,

        "graph_edges":
            int(
                graph_edges.shape[
                    1
                ]
            ),

        "retained_mutation_rows_frequency_pass":
            int(
                retained_mutation_rows
            ),

        "mutation_assignments_selected_space":
            int(
                mutation_assignments
            ),

        "mutation_rows_outside_resolved_coverage":
            int(
                outside_coverage
            ),

        "external_exact_exclusions_loaded":
            int(
                len(
                    exact_external_ids
                )
            ),

        "primary_center_disjoint_exclusions":
            sorted(
                EXCLUDED_PRIMARY_CENTERS
            ),

        "heldout_institutions":
            heldout_centers,

        "heldout_panels":
            heldout_panels,

        "split_counts":
            split_counts,

        "selected_gene_coverage_per_sample": {
            "min":
                int(
                    selected_gene_coverage.min()
                ),

            "median":
                float(
                    np.median(
                        selected_gene_coverage
                    )
                ),

            "max":
                int(
                    selected_gene_coverage.max()
                ),
        },

        "channel_schema": {
            "coverage":
                (
                    "1 only when the sample's resolved mutation "
                    "panel assayed the selected gene."
                ),

            "snv_indel":
                (
                    "1 when a mutation record exists for the "
                    "sample/gene; absence is only interpreted as "
                    "negative where coverage=1."
                ),
        },

        "cna_sv_inventory": {
            "cna_candidate_files":
                sources[
                    "cna"
                ],

            "fusion_candidate_files":
                sources[
                    "fusion"
                ],

            "checkpoint3_primary_objective":
                (
                    "Coverage-aware SNV/indel reconstruction. "
                    "CNA/SV source files are inventoried without "
                    "silently treating missing channel coverage "
                    "as wild type."
                ),
        },
    }

    atomic_json(
        prep
        / "prepare_report.json",
        report,
    )

    print("")
    print(
        "========== CKPT3 PREP SUMMARY =========="
    )

    print(
        "panel_resolved_samples="
        f"{len(modeled)}"
    )

    print(
        "panel_definitions="
        f"{len(panels)}"
    )

    print(
        "heldout_institutions="
        f"{heldout_centers}"
    )

    print(
        "heldout_panels="
        f"{heldout_panels}"
    )

    print(
        "split_counts="
        f"{split_counts}"
    )

    print(
        "graph_edges="
        f"{graph_edges.shape[1]}"
    )

    print(
        "selected_gene_coverage="
        f"{report['selected_gene_coverage_per_sample']}"
    )

    print(
        "mutation_rows_outside_coverage="
        f"{outside_coverage}"
    )

    print(
        "========== CKPT3 PREP SUMMARY END =========="
    )


###############################################################################
# Torch dataset
###############################################################################


class GenieDataset(
    Dataset
):

    def __init__(
        self,
        prep: Path,
        split: str,
    ):

        metadata = pd.read_parquet(
            prep
            / "sample_metadata.parquet"
        )

        self.indices = np.where(
            metadata[
                "split"
            ].to_numpy()
            == split
        )[
            0
        ]

        self.cancer = metadata[
            "cancer_label"
        ].to_numpy(
            dtype=np.int64
        )

        self.sample_type = metadata[
            "sample_type_label"
        ].to_numpy(
            dtype=np.int64
        )

        self.coverage = np.load(
            prep
            / "coverage.npy",
            mmap_mode="r",
        )

        self.alteration = np.load(
            prep
            / "snv_indel.npy",
            mmap_mode="r",
        )

    def __len__(
        self,
    ):

        return len(
            self.indices
        )

    def __getitem__(
        self,
        index: int,
    ):

        row = int(
            self.indices[
                index
            ]
        )

        return (
            torch.from_numpy(
                np.asarray(
                    self.alteration[
                        row
                    ],
                    dtype=np.float32,
                ).copy()
            ),

            torch.from_numpy(
                np.asarray(
                    self.coverage[
                        row
                    ],
                    dtype=np.float32,
                ).copy()
            ),

            torch.tensor(
                int(
                    self.cancer[
                        row
                    ]
                ),
                dtype=torch.long,
            ),

            torch.tensor(
                int(
                    self.sample_type[
                        row
                    ]
                ),
                dtype=torch.long,
            ),

            torch.tensor(
                row,
                dtype=torch.long,
            ),
        )


###############################################################################
# Graph encoder
###############################################################################


class FixedGraphAttention(
    nn.Module
):

    def __init__(
        self,
        hidden: int,
        heads: int,
        dropout: float,
    ):

        super().__init__()

        if (
            hidden
            % heads
            != 0
        ):

            raise ValueError(
                "hidden must divide by heads"
            )

        self.hidden = hidden
        self.heads = heads
        self.head_dim = (
            hidden
            // heads
        )

        self.value = nn.Linear(
            hidden,
            hidden,
            bias=False,
        )

        self.attn_src = nn.Parameter(
            torch.zeros(
                heads,
                self.head_dim,
            )
        )

        self.attn_dst = nn.Parameter(
            torch.zeros(
                heads,
                self.head_dim,
            )
        )

        self.out = nn.Linear(
            hidden,
            hidden,
        )

        self.norm1 = nn.LayerNorm(
            hidden
        )

        self.ff = nn.Sequential(
            nn.Linear(
                hidden,
                hidden
                * 4,
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                hidden
                * 4,
                hidden,
            ),
        )

        self.norm2 = nn.LayerNorm(
            hidden
        )

        self.dropout = nn.Dropout(
            dropout
        )

        nn.init.xavier_uniform_(
            self.attn_src
        )

        nn.init.xavier_uniform_(
            self.attn_dst
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:

        batch, genes, _ = (
            x.shape
        )

        src = edge_index[
            0
        ]

        dst = edge_index[
            1
        ]

        value = (
            self.value(
                x
            )
            .view(
                batch,
                genes,
                self.heads,
                self.head_dim,
            )
        )

        src_value = value[
            :,
            src,
        ]

        dst_value = value[
            :,
            dst,
        ]

        score = (
            (
                src_value
                * self.attn_src[
                    None,
                    None,
                ]
            ).sum(
                dim=-1
            )
            + (
                dst_value
                * self.attn_dst[
                    None,
                    None,
                ]
            ).sum(
                dim=-1
            )
        )

        score = F.leaky_relu(
            score,
            negative_slope=0.2,
        )

        index = (
            dst[
                None,
                :,
                None,
            ]
            .expand(
                batch,
                -1,
                self.heads,
            )
        )

        maxima = torch.full(
            (
                batch,
                genes,
                self.heads,
            ),
            -1e30,
            dtype=score.dtype,
            device=score.device,
        )

        maxima.scatter_reduce_(
            1,
            index,
            score,
            reduce="amax",
            include_self=True,
        )

        stabilized = (
            score
            - maxima.gather(
                1,
                index,
            )
        )

        exp_score = torch.exp(
            stabilized
        )

        denominator = torch.zeros(
            (
                batch,
                genes,
                self.heads,
            ),
            dtype=score.dtype,
            device=score.device,
        )

        denominator.scatter_add_(
            1,
            index,
            exp_score,
        )

        alpha = (
            exp_score
            / denominator.gather(
                1,
                index,
            ).clamp_min(
                1e-8
            )
        )

        messages = (
            src_value
            * alpha[
                ...,
                None
            ]
        )

        aggregate = torch.zeros(
            (
                batch,
                genes,
                self.heads,
                self.head_dim,
            ),
            dtype=x.dtype,
            device=x.device,
        )

        message_index = (
            dst[
                None,
                :,
                None,
                None,
            ]
            .expand(
                batch,
                -1,
                self.heads,
                self.head_dim,
            )
        )

        aggregate.scatter_add_(
            1,
            message_index,
            messages,
        )

        aggregate = aggregate.reshape(
            batch,
            genes,
            self.hidden,
        )

        x = self.norm1(
            x
            + self.dropout(
                self.out(
                    aggregate
                )
            )
        )

        x = self.norm2(
            x
            + self.dropout(
                self.ff(
                    x
                )
            )
        )

        return x


class GenomicEncoder(
    nn.Module
):

    def __init__(
        self,
        gene_count: int,
        cancer_classes: int,
        sample_type_classes: int,
        hidden: int = 128,
        heads: int = 8,
        dropout: float = 0.15,
    ):

        super().__init__()

        self.gene_count = gene_count
        self.hidden = hidden

        self.gene_embedding = (
            nn.Embedding(
                gene_count,
                hidden,
            )
        )

        self.feature_projection = nn.Sequential(
            nn.Linear(
                2,
                hidden,
            ),
            nn.GELU(),
            nn.Linear(
                hidden,
                hidden,
            ),
        )

        self.blocks = nn.ModuleList(
            [
                FixedGraphAttention(
                    hidden=hidden,
                    heads=heads,
                    dropout=dropout,
                ),
                FixedGraphAttention(
                    hidden=hidden,
                    heads=heads,
                    dropout=dropout,
                ),
            ]
        )

        self.pool_score = nn.Sequential(
            nn.Linear(
                hidden,
                hidden,
            ),
            nn.Tanh(),
            nn.Linear(
                hidden,
                1,
            ),
        )

        self.decoder = nn.Sequential(
            nn.Linear(
                hidden
                * 2,
                hidden,
            ),
            nn.GELU(),
            nn.Linear(
                hidden,
                1,
            ),
        )

        self.cancer_head = nn.Linear(
            hidden,
            cancer_classes,
        )

        self.sample_type_head = nn.Linear(
            hidden,
            sample_type_classes,
        )

    def encode(
        self,
        alteration: torch.Tensor,
        coverage: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

        batch = alteration.shape[
            0
        ]

        gene_ids = torch.arange(
            self.gene_count,
            device=alteration.device,
        )

        gene = self.gene_embedding(
            gene_ids
        )[
            None,
        ].expand(
            batch,
            -1,
            -1,
        )

        features = torch.stack(
            [
                alteration,
                coverage,
            ],
            dim=-1,
        )

        x = (
            gene
            + self.feature_projection(
                features
            )
        )

        for block in self.blocks:

            x = block(
                x,
                edge_index,
            )

        score = (
            self.pool_score(
                x
            )
            .squeeze(
                -1
            )
        )

        score = score.masked_fill(
            coverage
            <= 0,
            -1e4,
        )

        attention = torch.softmax(
            score,
            dim=1,
        )

        pooled = torch.sum(
            x
            * attention[
                ...,
                None
            ],
            dim=1,
        )

        return (
            x,
            pooled,
        )

    def forward(
        self,
        alteration: torch.Tensor,
        coverage: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> dict[
        str,
        torch.Tensor,
    ]:

        node, pooled = self.encode(
            alteration,
            coverage,
            edge_index,
        )

        pooled_expanded = (
            pooled[
                :,
                None,
                :,
            ]
            .expand(
                -1,
                self.gene_count,
                -1,
            )
        )

        logits = (
            self.decoder(
                torch.cat(
                    [
                        node,
                        pooled_expanded,
                    ],
                    dim=-1,
                )
            )
            .squeeze(
                -1
            )
        )

        return {
            "node":
                node,

            "embedding":
                pooled,

            "reconstruction_logits":
                logits,

            "cancer_logits":
                self.cancer_head(
                    pooled
                ),

            "sample_type_logits":
                self.sample_type_head(
                    pooled
                ),
        }


###############################################################################
# Teacher/student masking
###############################################################################


def synthetic_student_mask(
    actual_coverage: torch.Tensor,
    panel_library: torch.Tensor,
    rng: torch.Generator,
) -> torch.Tensor:

    batch, genes = (
        actual_coverage.shape
    )

    donor_idx = torch.randint(
        low=0,
        high=panel_library.shape[
            0
        ],
        size=(
            batch,
        ),
        generator=rng,
        device=actual_coverage.device,
    )

    donor = panel_library[
        donor_idx
    ]

    student = (
        actual_coverage
        * donor
    )

    covered = actual_coverage.sum(
        dim=1
    )

    hidden = (
        actual_coverage
        - student
    ).clamp_min(
        0
    ).sum(
        dim=1
    )

    needed = torch.maximum(
        torch.full_like(
            covered,
            8.0,
        ),
        covered
        * 0.05,
    )

    supplement_rows = torch.where(
        hidden
        < needed
    )[
        0
    ]

    if len(
        supplement_rows
    ):

        random_values = torch.rand(
            (
                len(
                    supplement_rows
                ),
                genes,
            ),
            generator=rng,
            device=actual_coverage.device,
        )

        supplement_drop = (
            random_values
            < 0.25
        ).float()

        current = student[
            supplement_rows
        ]

        actual = actual_coverage[
            supplement_rows
        ]

        current = (
            current
            * (
                1.0
                - supplement_drop
            )
        )

        # Never fabricate coverage.
        student[
            supplement_rows
        ] = (
            current
            * actual
        )

    return student


def masked_reconstruction_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    teacher_coverage: torch.Tensor,
    student_coverage: torch.Tensor,
    positive_weights: torch.Tensor,
) -> torch.Tensor:

    masked = (
        (
            teacher_coverage
            > 0
        )
        & (
            student_coverage
            <= 0
        )
    ).float()

    if float(
        masked.sum()
    ) <= 0:

        masked = (
            teacher_coverage
            > 0
        ).float()

    raw = F.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
    )

    weight = (
        1.0
        + target
        * (
            positive_weights[
                None,
            ]
            - 1.0
        )
    )

    return (
        (
            raw
            * weight
            * masked
        ).sum()
        / (
            weight
            * masked
        ).sum().clamp_min(
            1.0
        )
    )


def ema_update(
    teacher: nn.Module,
    student: nn.Module,
    decay: float,
) -> None:

    with torch.no_grad():

        for teacher_parameter, student_parameter in zip(
            teacher.parameters(),
            student.parameters(),
        ):

            teacher_parameter.data.mul_(
                decay
            ).add_(
                student_parameter.data,
                alpha=(
                    1.0
                    - decay
                ),
            )


###############################################################################
# Distributed setup
###############################################################################


def distributed_setup() -> tuple[
    int,
    int,
    int,
]:

    world_size = int(
        os.environ.get(
            "WORLD_SIZE",
            "1",
        )
    )

    rank = int(
        os.environ.get(
            "RANK",
            "0",
        )
    )

    local_rank = int(
        os.environ.get(
            "LOCAL_RANK",
            "0",
        )
    )

    if world_size > 1:

        torch.cuda.set_device(
            local_rank
        )

        dist.init_process_group(
            backend="nccl"
        )

    return (
        world_size,
        rank,
        local_rank,
    )


def distributed_cleanup() -> None:

    if (
        dist.is_available()
        and dist.is_initialized()
    ):

        dist.barrier()
        dist.destroy_process_group()


###############################################################################
# Training
###############################################################################


def load_class_counts(
    prep: Path,
) -> tuple[
    int,
    int,
]:

    metadata = pd.read_parquet(
        prep
        / "sample_metadata.parquet"
    )

    cancer_classes = (
        int(
            metadata[
                "cancer_label"
            ].max()
        )
        + 1
    )

    sample_classes = (
        int(
            metadata[
                "sample_type_label"
            ].max()
        )
        + 1
    )

    return (
        cancer_classes,
        sample_classes,
    )


def train(
    repo: Path,
    out: Path,
    epochs: int,
    batch_size: int,
) -> None:

    prep = (
        out
        / "prepared"
    )

    (
        world_size,
        rank,
        local_rank,
    ) = distributed_setup()

    try:

        if not torch.cuda.is_available():

            raise RuntimeError(
                "CUDA is required for full CKPT3 training."
            )

        random.seed(
            SEED + rank
        )

        np.random.seed(
            SEED + rank
        )

        torch.manual_seed(
            SEED + rank
        )

        device = torch.device(
            "cuda",
            local_rank,
        )

        (
            cancer_classes,
            sample_type_classes,
        ) = load_class_counts(
            prep
        )

        edge_index = (
            torch.from_numpy(
                np.load(
                    prep
                    / "graph_edges.npy"
                )
            )
            .long()
            .to(
                device
            )
        )

        positive_weights = (
            torch.from_numpy(
                np.load(
                    prep
                    / "positive_weights.npy"
                )
            )
            .float()
            .to(
                device
            )
        )

        train_dataset = GenieDataset(
            prep,
            "train",
        )

        val_dataset = GenieDataset(
            prep,
            "val",
        )

        if (
            len(
                train_dataset
            )
            == 0
            or len(
                val_dataset
            )
            == 0
        ):

            raise RuntimeError(
                "train/val split is empty"
            )

        train_sampler = (
            DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=SEED,
            )
            if world_size
            > 1
            else None
        )

        loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(
                train_sampler
                is None
            ),
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

        student = GenomicEncoder(
            gene_count=GENE_COUNT,
            cancer_classes=cancer_classes,
            sample_type_classes=sample_type_classes,
            hidden=128,
            heads=8,
            dropout=0.15,
        ).to(
            device
        )

        teacher = copy.deepcopy(
            student
        ).to(
            device
        )

        for parameter in teacher.parameters():
            parameter.requires_grad_(
                False
            )

        trainable: nn.Module = student

        if world_size > 1:

            trainable = (
                DistributedDataParallel(
                    student,
                    device_ids=[
                        local_rank
                    ],
                    output_device=local_rank,
                )
            )

        # DDP broadcasts rank-0 student parameters during construction.
        # The teacher was cloned before that synchronization, so explicitly
        # resynchronize it here.  The EMA teacher is deterministic and must
        # not apply dropout while producing targets.
        teacher.load_state_dict(
            student.state_dict()
        )

        teacher.eval()

        optimizer = torch.optim.AdamW(
            student.parameters(),
            lr=3e-4,
            weight_decay=0.02,
            betas=(
                0.9,
                0.95,
            ),
        )

        total_steps = max(
            epochs
            * len(
                loader
            ),
            1,
        )

        scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
                eta_min=3e-5,
            )
        )

        panel_library_np = np.load(
            prep
            / "coverage.npy",
            mmap_mode="r",
        )

        metadata = pd.read_parquet(
            prep
            / "sample_metadata.parquet"
        )

        train_rows = np.where(
            metadata[
                "split"
            ].to_numpy()
            == "train"
        )[
            0
        ]

        # A real-panel mask library sampled from up to 4096 training samples.
        rng_np = np.random.default_rng(
            SEED
        )

        if len(
            train_rows
        ) > 4096:

            library_rows = rng_np.choice(
                train_rows,
                size=4096,
                replace=False,
            )

        else:

            library_rows = train_rows

        panel_library = (
            torch.from_numpy(
                np.asarray(
                    panel_library_np[
                        library_rows
                    ],
                    dtype=np.float32,
                ).copy()
            )
            .to(
                device
            )
        )

        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=True,
        )

        history = []

        best_val = None

        generator = torch.Generator(
            device=device
        )

        generator.manual_seed(
            SEED
            + rank
            * 1000
        )

        for epoch in range(
            1,
            epochs + 1,
        ):

            if train_sampler is not None:

                train_sampler.set_epoch(
                    epoch
                )

            trainable.train()

            # The student trains with dropout; the EMA teacher remains a
            # stable target network.
            teacher.eval()

            totals = Counter()

            for (
                alteration,
                coverage,
                cancer,
                sample_type,
                row_index,
            ) in loader:

                alteration = alteration.to(
                    device,
                    non_blocking=True,
                )

                coverage = coverage.to(
                    device,
                    non_blocking=True,
                )

                cancer = cancer.to(
                    device,
                    non_blocking=True,
                )

                sample_type = sample_type.to(
                    device,
                    non_blocking=True,
                )

                student_coverage = (
                    synthetic_student_mask(
                        coverage,
                        panel_library,
                        generator,
                    )
                )

                student_alteration = (
                    alteration
                    * student_coverage
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                with torch.no_grad():

                    teacher_output = (
                        teacher(
                            alteration,
                            coverage,
                            edge_index,
                        )
                    )

                with torch.amp.autocast(
                    "cuda",
                    dtype=torch.bfloat16,
                    enabled=True,
                ):

                    student_output = (
                        trainable(
                            student_alteration,
                            student_coverage,
                            edge_index,
                        )
                    )

                    reconstruction = (
                        masked_reconstruction_loss(
                            student_output[
                                "reconstruction_logits"
                            ],
                            alteration,
                            coverage,
                            student_coverage,
                            positive_weights,
                        )
                    )

                    distillation = (
                        1.0
                        - F.cosine_similarity(
                            student_output[
                                "embedding"
                            ].float(),
                            teacher_output[
                                "embedding"
                            ].float(),
                            dim=-1,
                        )
                    ).mean()

                    cancer_loss = (
                        F.cross_entropy(
                            student_output[
                                "cancer_logits"
                            ].float(),
                            cancer,
                        )
                    )

                    sample_loss = (
                        F.cross_entropy(
                            student_output[
                                "sample_type_logits"
                            ].float(),
                            sample_type,
                        )
                    )

                    loss = (
                        reconstruction
                        + 0.10
                        * distillation
                        + 0.05
                        * cancer_loss
                        + 0.02
                        * sample_loss
                    )

                scaler.scale(
                    loss
                ).backward()

                scaler.unscale_(
                    optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    student.parameters(),
                    1.0,
                )

                scaler.step(
                    optimizer
                )

                scaler.update()

                scheduler.step()

                ema_update(
                    teacher,
                    student,
                    decay=0.995,
                )

                totals[
                    "batches"
                ] += 1

                totals[
                    "loss_sum"
                ] += float(
                    loss.detach()
                )

                totals[
                    "reconstruction_sum"
                ] += float(
                    reconstruction.detach()
                )

                totals[
                    "distill_sum"
                ] += float(
                    distillation.detach()
                )

            if world_size > 1:

                tensor = torch.tensor(
                    [
                        totals[
                            "batches"
                        ],
                        totals[
                            "loss_sum"
                        ],
                        totals[
                            "reconstruction_sum"
                        ],
                        totals[
                            "distill_sum"
                        ],
                    ],
                    dtype=torch.float64,
                    device=device,
                )

                dist.all_reduce(
                    tensor,
                    op=dist.ReduceOp.SUM,
                )

                totals[
                    "batches"
                ] = int(
                    tensor[
                        0
                    ].item()
                )

                totals[
                    "loss_sum"
                ] = float(
                    tensor[
                        1
                    ].item()
                )

                totals[
                    "reconstruction_sum"
                ] = float(
                    tensor[
                        2
                    ].item()
                )

                totals[
                    "distill_sum"
                ] = float(
                    tensor[
                        3
                    ].item()
                )

            if rank == 0:

                denominator = max(
                    totals[
                        "batches"
                    ],
                    1,
                )

                epoch_row = {
                    "epoch":
                        epoch,

                    "loss":
                        totals[
                            "loss_sum"
                        ]
                        / denominator,

                    "reconstruction":
                        totals[
                            "reconstruction_sum"
                        ]
                        / denominator,

                    "distillation":
                        totals[
                            "distill_sum"
                        ]
                        / denominator,

                    "lr":
                        float(
                            optimizer
                            .param_groups[
                                0
                            ][
                                "lr"
                            ]
                        ),
                }

                history.append(
                    epoch_row
                )

                print(
                    "[CKPT3_TRAIN]",
                    epoch_row,
                    flush=True,
                )

        if world_size > 1:
            dist.barrier()

        if rank == 0:

            checkpoint = {
                "model_state":
                    teacher.state_dict(),

                "student_state":
                    student.state_dict(),

                "gene_count":
                    GENE_COUNT,

                "hidden":
                    128,

                "heads":
                    8,

                "dropout":
                    0.15,

                "cancer_classes":
                    cancer_classes,

                "sample_type_classes":
                    sample_type_classes,

                "epochs":
                    epochs,

                "world_size":
                    world_size,

                "seed":
                    SEED,
            }

            tmp_checkpoint = (
                out
                / "genomic_encoder.pt.tmp"
            )

            torch.save(
                checkpoint,
                tmp_checkpoint,
            )

            loaded = torch.load(
                tmp_checkpoint,
                map_location="cpu",
                weights_only=False,
            )

            if (
                "model_state"
                not in loaded
            ):

                raise RuntimeError(
                    "Checkpoint validation failed."
                )

            tmp_checkpoint.replace(
                out
                / "genomic_encoder.pt"
            )

            atomic_json(
                out
                / "training_history.json",
                history,
            )

    finally:

        distributed_cleanup()


###############################################################################
# Evaluation
###############################################################################


def load_trained(
    out: Path,
    device: torch.device,
) -> tuple[
    GenomicEncoder,
    torch.Tensor,
]:

    prep = (
        out
        / "prepared"
    )

    checkpoint = torch.load(
        out
        / "genomic_encoder.pt",
        map_location=device,
        weights_only=False,
    )

    model = GenomicEncoder(
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
                prep
                / "graph_edges.npy"
            )
        )
        .long()
        .to(
            device
        )
    )

    return (
        model,
        edge_index,
    )


@torch.no_grad()
def evaluate_split(
    out: Path,
    split: str,
    max_samples: int = 30000,
    random_mask_fraction: float | None = None,
) -> dict[
    str,
    Any
]:

    prep = (
        out
        / "prepared"
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model, edge_index = load_trained(
        out,
        device,
    )

    metadata = pd.read_parquet(
        prep
        / "sample_metadata.parquet"
    )

    rows = np.where(
        metadata[
            "split"
        ].to_numpy()
        == split
    )[
        0
    ]

    if len(
        rows
    ) == 0:

        return {
            "split":
                split,

            "samples":
                0,
        }

    if len(
        rows
    ) > max_samples:

        generator = np.random.default_rng(
            SEED
            + stable_hash(
                split
            )
            % 100000
        )

        rows = np.sort(
            generator.choice(
                rows,
                size=max_samples,
                replace=False,
            )
        )

    alteration = np.load(
        prep
        / "snv_indel.npy",
        mmap_mode="r",
    )

    coverage = np.load(
        prep
        / "coverage.npy",
        mmap_mode="r",
    )

    train_rows = np.where(
        metadata[
            "split"
        ].to_numpy()
        == "train"
    )[
        0
    ]

    evaluation_panel_rng = np.random.default_rng(
        SEED
        + 424242
    )

    if len(
        train_rows
    ) > 4096:

        library_rows = evaluation_panel_rng.choice(
            train_rows,
            size=4096,
            replace=False,
        )

    else:

        library_rows = train_rows

    panel_library = torch.from_numpy(
        np.asarray(
            coverage[
                library_rows
            ],
            dtype=np.float32,
        ).copy()
    ).to(
        device
    )

    generator = torch.Generator(
        device=device
    )

    generator.manual_seed(
        SEED
        + stable_hash(
            split
        )
        % 100000
        + (
            0
            if random_mask_fraction
            is None
            else int(
                random_mask_fraction
                * 1000
            )
        )
    )

    all_target = []

    all_score = []

    all_mask = []

    all_embedding_cosine = []

    batch_size = 512

    for start in range(
        0,
        len(
            rows
        ),
        batch_size,
    ):

        batch_rows = rows[
            start:
            start
            + batch_size
        ]

        target = torch.from_numpy(
            np.asarray(
                alteration[
                    batch_rows
                ],
                dtype=np.float32,
            ).copy()
        ).to(
            device
        )

        teacher_coverage = (
            torch.from_numpy(
                np.asarray(
                    coverage[
                        batch_rows
                    ],
                    dtype=np.float32,
                ).copy()
            )
            .to(
                device
            )
        )

        if random_mask_fraction is None:

            student_coverage = (
                synthetic_student_mask(
                    teacher_coverage,
                    panel_library,
                    generator,
                )
            )

        else:

            random_values = torch.rand(
                teacher_coverage.shape,
                generator=generator,
                device=device,
            )

            student_coverage = (
                teacher_coverage
                * (
                    random_values
                    >= random_mask_fraction
                ).float()
            )

        student_target = (
            target
            * student_coverage
        )

        teacher_output = model(
            target,
            teacher_coverage,
            edge_index,
        )

        student_output = model(
            student_target,
            student_coverage,
            edge_index,
        )

        mask = (
            (
                teacher_coverage
                > 0
            )
            & (
                student_coverage
                <= 0
            )
        )

        score = torch.sigmoid(
            student_output[
                "reconstruction_logits"
            ]
        )

        cosine = F.cosine_similarity(
            student_output[
                "embedding"
            ],
            teacher_output[
                "embedding"
            ],
            dim=-1,
        )

        all_target.append(
            target.cpu().numpy()
        )

        all_score.append(
            score.cpu().numpy()
        )

        all_mask.append(
            mask.cpu().numpy()
        )

        all_embedding_cosine.append(
            cosine.cpu().numpy()
        )

    target = np.concatenate(
        all_target,
        axis=0,
    )

    score = np.concatenate(
        all_score,
        axis=0,
    )

    mask = np.concatenate(
        all_mask,
        axis=0,
    )

    cosine = np.concatenate(
        all_embedding_cosine,
        axis=0,
    )

    flat_target = target[
        mask
    ]

    flat_score = score[
        mask
    ]

    if (
        len(
            np.unique(
                flat_target
            )
        )
        < 2
    ):

        overall_auprc = float(
            "nan"
        )

    else:

        overall_auprc = float(
            average_precision_score(
                flat_target,
                flat_score,
            )
        )

    per_gene = []

    for gene_idx in range(
        GENE_COUNT
    ):

        selected = mask[
            :,
            gene_idx
        ]

        if (
            selected.sum()
            < 100
        ):
            continue

        y = target[
            selected,
            gene_idx,
        ]

        if (
            y.sum()
            < 5
            or (
                1
                - y
            ).sum()
            < 5
        ):
            continue

        per_gene.append(
            float(
                average_precision_score(
                    y,
                    score[
                        selected,
                        gene_idx,
                    ],
                )
            )
        )

    return {
        "split":
            split,

        "samples":
            int(
                len(
                    rows
                )
            ),

        "masked_gene_observations":
            int(
                mask.sum()
            ),

        "masked_positive_observations":
            int(
                target[
                    mask
                ].sum()
            ),

        "overall_auprc":
            overall_auprc,

        "macro_gene_auprc":
            (
                float(
                    np.mean(
                        per_gene
                    )
                )
                if per_gene
                else float(
                    "nan"
                )
            ),

        "macro_gene_count":
            int(
                len(
                    per_gene
                )
            ),

        "teacher_student_embedding_cosine":
            float(
                np.mean(
                    cosine
                )
            ),

        "random_mask_fraction":
            random_mask_fraction,
    }


@torch.no_grad()
def cache_embeddings(
    out: Path,
) -> dict[
    str,
    Any
]:

    prep = (
        out
        / "prepared"
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model, edge_index = load_trained(
        out,
        device,
    )

    metadata = pd.read_parquet(
        prep
        / "sample_metadata.parquet"
    )

    alteration = np.load(
        prep
        / "snv_indel.npy",
        mmap_mode="r",
    )

    coverage = np.load(
        prep
        / "coverage.npy",
        mmap_mode="r",
    )

    path = (
        out
        / "genie_sample_embeddings_f16.npy"
    )

    embeddings = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float16,
        shape=(
            len(
                metadata
            ),
            128,
        ),
    )

    batch_size = 512

    for start in range(
        0,
        len(
            metadata
        ),
        batch_size,
    ):

        end = min(
            start
            + batch_size,
            len(
                metadata
            ),
        )

        alteration_batch = (
            torch.from_numpy(
                np.asarray(
                    alteration[
                        start:end
                    ],
                    dtype=np.float32,
                ).copy()
            )
            .to(
                device
            )
        )

        coverage_batch = (
            torch.from_numpy(
                np.asarray(
                    coverage[
                        start:end
                    ],
                    dtype=np.float32,
                ).copy()
            )
            .to(
                device
            )
        )

        output = model(
            alteration_batch,
            coverage_batch,
            edge_index,
        )

        embeddings[
            start:end
        ] = (
            output[
                "embedding"
            ]
            .float()
            .cpu()
            .numpy()
            .astype(
                np.float16
            )
        )

        if (
            start == 0
            or start
            % (
                batch_size
                * 100
            )
            == 0
        ):

            print(
                "[CKPT3_EMBED] "
                f"{start}/{len(metadata)}",
                flush=True,
            )

    embeddings.flush()

    index = metadata[
        [
            "sample_id",
            "patient_id",
            "center",
            "panel_id",
            "split",
        ]
    ].copy()

    index[
        "embedding_row"
    ] = np.arange(
        len(
            index
        ),
        dtype=np.int64,
    )

    atomic_parquet(
        out
        / "genie_embedding_index.parquet",
        index,
    )

    return {
        "samples":
            int(
                len(
                    index
                )
            ),

        "dimensions":
            128,

        "dtype":
            "float16",
    }


def evaluate(
    repo: Path,
    out: Path,
) -> None:

    print(
        "[CKPT3_EVAL] evaluating held-out cohorts",
        flush=True,
    )

    results = {
        "val_real_panel_mask":
            evaluate_split(
                out,
                "val",
            ),

        "panel_holdout_real_panel_mask":
            evaluate_split(
                out,
                "panel_holdout",
            ),

        "institution_holdout_real_panel_mask":
            evaluate_split(
                out,
                "institution_holdout",
            ),

        "robustness_random_mask": {},
    }

    for fraction in (
        0.25,
        0.50,
        0.75,
    ):

        results[
            "robustness_random_mask"
        ][
            str(
                fraction
            )
        ] = evaluate_split(
            out,
            "val",
            random_mask_fraction=fraction,
        )

    embeddings = cache_embeddings(
        out
    )

    results[
        "embedding_cache"
    ] = embeddings

    atomic_json(
        out
        / "evaluation.json",
        results,
    )

    print("")
    print(
        "========== CKPT3 EVALUATION =========="
    )

    for key in (
        "val_real_panel_mask",
        "panel_holdout_real_panel_mask",
        "institution_holdout_real_panel_mask",
    ):

        print(
            f"{key}="
            f"{results[key]}"
        )

    print(
        "robustness_random_mask="
        f"{results['robustness_random_mask']}"
    )

    print(
        "embedding_cache="
        f"{embeddings}"
    )

    print(
        "========== CKPT3 EVALUATION END =========="
    )


###############################################################################
# Self-test
###############################################################################


def self_test() -> None:

    torch.manual_seed(
        1
    )

    genes = 12

    edges = []

    for idx in range(
        genes
    ):

        edges.append(
            (
                idx,
                idx,
            )
        )

        if idx + 1 < genes:

            edges.append(
                (
                    idx,
                    idx + 1,
                )
            )

            edges.append(
                (
                    idx + 1,
                    idx,
                )
            )

    edge_index = torch.tensor(
        edges,
        dtype=torch.long,
    ).T

    model = GenomicEncoder(
        gene_count=genes,
        cancer_classes=3,
        sample_type_classes=3,
        hidden=32,
        heads=4,
        dropout=0.0,
    )

    alteration = torch.zeros(
        (
            4,
            genes,
        )
    )

    alteration[
        0,
        2
    ] = 1

    alteration[
        1,
        5
    ] = 1

    coverage = torch.ones(
        (
            4,
            genes,
        )
    )

    panel_library = torch.stack(
        [
            torch.cat(
                [
                    torch.ones(
                        6
                    ),
                    torch.zeros(
                        6
                    ),
                ]
            ),
            torch.cat(
                [
                    torch.zeros(
                        3
                    ),
                    torch.ones(
                        9
                    ),
                ]
            ),
        ]
    )

    generator = torch.Generator()
    generator.manual_seed(
        4
    )

    student_coverage = (
        synthetic_student_mask(
            coverage,
            panel_library,
            generator,
        )
    )

    output = model(
        alteration
        * student_coverage,
        student_coverage,
        edge_index,
    )

    assert output[
        "embedding"
    ].shape == (
        4,
        32,
    )

    assert output[
        "reconstruction_logits"
    ].shape == (
        4,
        genes,
    )

    positive_weights = torch.ones(
        genes
    )

    loss = masked_reconstruction_loss(
        output[
            "reconstruction_logits"
        ],
        alteration,
        coverage,
        student_coverage,
        positive_weights,
    )

    assert torch.isfinite(
        loss
    )

    loss.backward()

    print(
        "[CKPT3_SELF_TEST_PASS]",
        {
            "loss":
                float(
                    loss
                ),
            "student_coverage":
                float(
                    student_coverage.sum()
                ),
        },
    )


###############################################################################
# Finalize project state / handoff
###############################################################################


def finalize(
    repo: Path,
    out: Path,
) -> None:

    prep_report = json.loads(
        (
            out
            / "prepared"
            / "prepare_report.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    evaluation = json.loads(
        (
            out
            / "evaluation.json"
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

    panel_eval = evaluation[
        "panel_holdout_real_panel_mask"
    ]

    institution_eval = evaluation[
        "institution_holdout_real_panel_mask"
    ]

    val_eval = evaluation[
        "val_real_panel_mask"
    ]

    required_metrics = [
        val_eval.get(
            "overall_auprc"
        ),
        panel_eval.get(
            "overall_auprc"
        ),
        institution_eval.get(
            "overall_auprc"
        ),
    ]

    if all(
        value is not None
        and np.isfinite(
            value
        )
        for value in required_metrics
    ):

        status = (
            "PASS_GENOMIC_PRETRAINING"
        )

    else:

        status = (
            "NEEDS_GENOMIC_EVALUATION_RESOLUTION"
        )

    qc = {
        "status":
            status,

        "gene_space":
            GENE_COUNT,

        "panel_resolved_samples":
            prep_report[
                "panel_resolved_samples"
            ],

        "train_samples":
            prep_report[
                "split_counts"
            ]
            .get(
                "train",
                {},
            )
            .get(
                "samples",
                0,
            ),

        "external_exact_exclusions_loaded":
            prep_report[
                "external_exact_exclusions_loaded"
            ],

        "primary_center_disjoint_exclusions":
            prep_report[
                "primary_center_disjoint_exclusions"
            ],

        "heldout_institutions":
            prep_report[
                "heldout_institutions"
            ],

        "heldout_panels":
            prep_report[
                "heldout_panels"
            ],

        "validation_overall_auprc":
            val_eval.get(
                "overall_auprc"
            ),

        "panel_holdout_overall_auprc":
            panel_eval.get(
                "overall_auprc"
            ),

        "institution_holdout_overall_auprc":
            institution_eval.get(
                "overall_auprc"
            ),

        "embedding_dimension":
            128,

        "epochs":
            len(
                history
            ),

        "errors":
            [],
    }

    atomic_json(
        out
        / "qc.json",
        qc,
    )

    report = f"""# Checkpoint 3 - GENIE coverage-aware genomic pretraining

Status: **{status}**

## Frozen design

- Gene space: **468 genes**
- Node input: mutation state + explicit assay coverage
- Graph: fixed chromosomal-neighborhood edges plus canonical cancer-pathway edges
- Encoder: 2 graph-attention blocks, hidden size 128, 8 heads
- Tumor embedding: 128 dimensions
- Teacher: EMA model using the fullest genuinely observed panel
- Student: synthetic reduced view derived from real training-panel masks
- Reconstruction is scored only on genes genuinely assayed by the teacher view
- Altered genes receive prevalence-derived positive weighting
- Auxiliary objectives: cancer type and primary/metastatic sample class

## Leakage policy

- Exact DFCI/VICC external-validation GENIE overlaps loaded: `{prep_report['external_exact_exclusions_loaded']}`
- Primary pretraining excludes all DFCI / VICC patients; MSK remains a development/training center
- Held-out institutions: `{prep_report['heldout_institutions']}`
- Held-out panels: `{prep_report['heldout_panels']}`

## Evaluation

Validation real-panel masking:
`{val_eval}`

Held-out panel:
`{panel_eval}`

Held-out institution:
`{institution_eval}`

Mask robustness:
`{evaluation['robustness_random_mask']}`

## Artifacts

- `artifacts/checkpoint3/prepared/gene_space_468.txt`
- `artifacts/checkpoint3/prepared/sample_metadata.parquet`
- `artifacts/checkpoint3/prepared/coverage.npy`
- `artifacts/checkpoint3/prepared/snv_indel.npy`
- `artifacts/checkpoint3/prepared/graph_edges.npy`
- `artifacts/checkpoint3/genomic_encoder.pt`
- `artifacts/checkpoint3/training_history.json`
- `artifacts/checkpoint3/evaluation.json`
- `artifacts/checkpoint3/genie_sample_embeddings_f16.npy`
- `artifacts/checkpoint3/genie_embedding_index.parquet`
- `artifacts/checkpoint3/qc.json`
"""

    atomic_text(
        out
        / "audit.md",
        report,
    )

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
        "<!-- CKPT3_GENOMIC_PRETRAINING -->"
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

    state = f"""
{marker}
## Checkpoint 3 - Coverage-aware GENIE genomic encoder

Status: **{status}**

Frozen genomic interface:
- fixed 468-gene space
- mutation state is separate from assay coverage
- unassayed is never encoded as observed wild type
- two graph-attention blocks, hidden 128, 8 heads
- pooled tumor embedding: 128D
- EMA teacher receives full genuinely observed sample panel
- student receives synthetic reduced coverage derived from real training panels
- masked reconstruction loss is scored only on genuinely assayed genes
- positive alterations are prevalence-weighted

Leakage policy:
- exact DFCI/VICC overlapping GENIE patients excluded
- primary molecular pretraining is center-disjoint from DFCI and VICC; MSK remains in development/training
- held-out institution and held-out panel evaluations completed independently

Evaluation:
- validation overall AUPRC: {val_eval.get('overall_auprc')}
- held-out panel overall AUPRC: {panel_eval.get('overall_auprc')}
- held-out institution overall AUPRC: {institution_eval.get('overall_auprc')}

Artifacts:
- artifacts/checkpoint3/genomic_encoder.pt
- artifacts/checkpoint3/evaluation.json
- artifacts/checkpoint3/genie_sample_embeddings_f16.npy
- artifacts/checkpoint3/genie_embedding_index.parquet
- artifacts/checkpoint3/audit.md
- artifacts/handoff/checkpoint_03.json
"""

    atomic_text(
        state_path,
        existing.rstrip()
        + "\n\n"
        + state.strip()
        + "\n",
    )

    handoff = {
        "checkpoint":
            "3",

        "name":
            "genie_coverage_aware_genomic_pretraining",

        "status":
            status,

        "architecture": {
            "gene_count":
                468,

            "hidden":
                128,

            "heads":
                8,

            "graph_blocks":
                2,

            "embedding_dimension":
                128,
        },

        "leakage_policy": {
            "exact_external_ids_excluded":
                prep_report[
                    "external_exact_exclusions_loaded"
                ],

            "center_disjoint_exclusions":
                prep_report[
                    "primary_center_disjoint_exclusions"
                ],

            "heldout_institutions":
                prep_report[
                    "heldout_institutions"
                ],

            "heldout_panels":
                prep_report[
                    "heldout_panels"
                ],
        },

        "evaluation":
            evaluation,

        "artifacts": {
            "checkpoint":
                (
                    "artifacts/checkpoint3/"
                    "genomic_encoder.pt"
                ),

            "gene_space":
                (
                    "artifacts/checkpoint3/"
                    "prepared/gene_space_468.txt"
                ),

            "evaluation":
                (
                    "artifacts/checkpoint3/"
                    "evaluation.json"
                ),

            "embedding_cache":
                (
                    "artifacts/checkpoint3/"
                    "genie_sample_embeddings_f16.npy"
                ),
        },

        "next_action":
            (
                "Proceed to CHORD temporal pretraining. "
                "Apply the frozen genomic encoder to CHORD "
                "genomic observations using the same 468-gene "
                "and explicit-coverage interface."
            ),
    }

    atomic_json(
        repo
        / "artifacts"
        / "handoff"
        / "checkpoint_03.json",
        handoff,
    )

    print("")
    print(
        "========== CKPT3 SUMMARY =========="
    )

    print(
        f"status={status}"
    )

    print(
        "panel_resolved_samples="
        f"{prep_report['panel_resolved_samples']}"
    )

    print(
        "train_samples="
        f"{qc['train_samples']}"
    )

    print(
        "external_exact_exclusions_loaded="
        f"{prep_report['external_exact_exclusions_loaded']}"
    )

    print(
        "center_disjoint_exclusions="
        f"{prep_report['primary_center_disjoint_exclusions']}"
    )

    print(
        "heldout_institutions="
        f"{prep_report['heldout_institutions']}"
    )

    print(
        "heldout_panels="
        f"{prep_report['heldout_panels']}"
    )

    print(
        "validation_overall_auprc="
        f"{val_eval.get('overall_auprc')}"
    )

    print(
        "validation_macro_gene_auprc="
        f"{val_eval.get('macro_gene_auprc')}"
    )

    print(
        "panel_holdout_overall_auprc="
        f"{panel_eval.get('overall_auprc')}"
    )

    print(
        "panel_holdout_macro_gene_auprc="
        f"{panel_eval.get('macro_gene_auprc')}"
    )

    print(
        "institution_holdout_overall_auprc="
        f"{institution_eval.get('overall_auprc')}"
    )

    print(
        "institution_holdout_macro_gene_auprc="
        f"{institution_eval.get('macro_gene_auprc')}"
    )

    print(
        "embedding_cache_samples="
        f"{evaluation['embedding_cache']['samples']}"
    )

    print(
        "checkpoint="
        "artifacts/checkpoint3/genomic_encoder.pt"
    )

    print(
        "handoff="
        "artifacts/handoff/checkpoint_03.json"
    )

    print(
        "========== CKPT3 SUMMARY END =========="
    )

    print("")
    print(
        "========== CKPT3 DECISION PACKET =========="
    )

    print(
        "prepare_report="
        f"{prep_report}"
    )

    print(
        "validation="
        f"{val_eval}"
    )

    print(
        "panel_holdout="
        f"{panel_eval}"
    )

    print(
        "institution_holdout="
        f"{institution_eval}"
    )

    print(
        "mask_robustness="
        f"{evaluation['robustness_random_mask']}"
    )

    print(
        "last_training_epoch="
        f"{history[-1] if history else None}"
    )

    print(
        "========== CKPT3 DECISION PACKET END =========="
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
        default="artifacts/checkpoint3",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    args = parser.parse_args()

    repo = Path(
        args.repo_root
    ).resolve()

    out_arg = Path(
        args.output_dir
    )

    out = (
        out_arg.resolve()
        if out_arg.is_absolute()
        else (
            repo
            / out_arg
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

        train(
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
