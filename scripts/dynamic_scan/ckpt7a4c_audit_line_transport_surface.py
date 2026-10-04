#!/usr/bin/env python3

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
import re
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(".").resolve()

OUT = (
    ROOT
    / "artifacts/checkpoint7a4c_transport_surface"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

CKPT4_SCRIPT = (
    ROOT
    / "scripts/dynamic_scan/ckpt4_temporal_pretrain.py"
)

CKPT5_SCRIPT = (
    ROOT
    / "scripts/dynamic_scan/ckpt5_supervised_dynamic_model.py"
)


###############################################################################
# Utilities
###############################################################################


def sha256_file(path: Path) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as handle:

        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:

    tmp = Path(str(path) + ".tmp")

    tmp.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )

    json.loads(
        tmp.read_text(
            encoding="utf-8"
        )
    )

    tmp.replace(path)


def atomic_text(path: Path, text: str) -> None:

    tmp = Path(str(path) + ".tmp")

    tmp.write_text(
        text,
        encoding="utf-8",
    )

    if (
        not tmp.exists()
        or tmp.stat().st_size == 0
    ):
        raise RuntimeError(
            f"Empty temporary text artifact: {tmp}"
        )

    tmp.replace(path)


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

    module = importlib.util.module_from_spec(spec)

    sys.modules[spec.name] = module

    spec.loader.exec_module(module)

    return module


def numeric_summary(series) -> dict[str, Any]:

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).to_numpy(dtype=float)

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return {"n": 0}

    return {
        "n": int(len(values)),
        "min": float(values.min()),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def flatten_dict(
    value: Any,
    prefix: str = "",
) -> list[tuple[str, Any]]:

    rows = []

    if isinstance(value, dict):

        for key, child in value.items():

            child_prefix = (
                f"{prefix}.{key}"
                if prefix
                else str(key)
            )

            rows.extend(
                flatten_dict(
                    child,
                    child_prefix,
                )
            )

    elif isinstance(value, (list, tuple)):

        for index, child in enumerate(value):

            rows.extend(
                flatten_dict(
                    child,
                    f"{prefix}[{index}]",
                )
            )

    else:

        rows.append(
            (
                prefix,
                value,
            )
        )

    return rows


###############################################################################
# 2. FIX9 is the frozen reason this audit exists
###############################################################################

fix9 = json.load(
    open(
        ROOT
        / "artifacts/checkpoint7a4b_fix9_exact_metastatic_semantics/"
        "audit.json",
        encoding="utf-8",
    )
)

if (
    fix9.get("status")
    != "LINE_CONTEXT_NOT_TRANSPORTABLE"
):
    raise RuntimeError(
        "FIX9 does not contain the expected transport failure."
    )

if fix9.get(
    "external_outcome_rows_opened"
) is not False:
    raise RuntimeError(
        "Unexpected external outcome access in FIX9."
    )


###############################################################################
# 3. Source dependency inventory
###############################################################################

SEARCH_PATTERNS = {
    "treatment_line":
        re.compile(
            r"\btreatment_line\b",
            re.I,
        ),

    "line_bucket":
        re.compile(
            r"line[_ ]?bucket",
            re.I,
        ),

    "line_embedding":
        re.compile(
            r"line[_ ]?(?:embed|embedding)",
            re.I,
        ),

    "line_id":
        re.compile(
            r"\bline[_ ]?id",
            re.I,
        ),

    "line_generic":
        re.compile(
            r"""["']line["']""",
            re.I,
        ),

    "elapsed_on_line":
        re.compile(
            r"elapsed_on_line",
            re.I,
        ),

    "scan_number_line":
        re.compile(
            r"scan_number_line",
            re.I,
        ),

    "treatment_active":
        re.compile(
            r"treatment[_ ]?active",
            re.I,
        ),

    "time_on_treatment":
        re.compile(
            r"time[_ ]?on[_ ]?(?:current[_ ]?)?treatment",
            re.I,
        ),
}


def scope_inventory(path: Path):

    text = path.read_text(
        encoding="utf-8"
    )

    lines = text.splitlines()

    tree = ast.parse(
        text,
        filename=str(path),
    )

    scopes = []

    for node in ast.walk(tree):

        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
            ),
        ):

            scopes.append(
                {
                    "name":
                        node.name,

                    "kind":
                        (
                            "class"
                            if isinstance(
                                node,
                                ast.ClassDef,
                            )
                            else "function"
                        ),

                    "start":
                        int(
                            node.lineno
                        ),

                    "end":
                        int(
                            getattr(
                                node,
                                "end_lineno",
                                node.lineno,
                            )
                        ),
                }
            )

    hits = []

    for line_number, line in enumerate(
        lines,
        start=1,
    ):

        matched = [
            name
            for name, pattern
            in SEARCH_PATTERNS.items()
            if pattern.search(line)
        ]

        if not matched:
            continue

        enclosing = [
            scope
            for scope in scopes
            if (
                scope["start"]
                <= line_number
                <= scope["end"]
            )
        ]

        enclosing.sort(
            key=lambda row:
                (
                    row["end"]
                    - row["start"]
                )
        )

        hits.append(
            {
                "line":
                    line_number,

                "text":
                    line.strip(),

                "patterns":
                    matched,

                "scope":
                    (
                        enclosing[0]["name"]
                        if enclosing
                        else None
                    ),

                "scope_kind":
                    (
                        enclosing[0]["kind"]
                        if enclosing
                        else None
                    ),
            }
        )

    important_scopes = {}

    for hit in hits:

        name = hit[
            "scope"
        ]

        if name is None:
            continue

        important_scopes.setdefault(
            name,
            {
                "patterns":
                    set(),

                "lines":
                    [],
            },
        )

        important_scopes[
            name
        ][
            "patterns"
        ].update(
            hit[
                "patterns"
            ]
        )

        important_scopes[
            name
        ][
            "lines"
        ].append(
            hit[
                "line"
            ]
        )

    serial_scopes = {}

    for name, info in important_scopes.items():

        serial_scopes[
            name
        ] = {
            "patterns":
                sorted(
                    info[
                        "patterns"
                    ]
                ),

            "hit_lines":
                sorted(
                    set(
                        info[
                            "lines"
                        ]
                    )
                ),
        }

    return (
        text,
        lines,
        hits,
        serial_scopes,
    )


(
    ckpt4_source,
    ckpt4_lines,
    ckpt4_hits,
    ckpt4_scopes,
) = scope_inventory(
    CKPT4_SCRIPT
)

(
    ckpt5_source,
    ckpt5_lines,
    ckpt5_hits,
    ckpt5_scopes,
) = scope_inventory(
    CKPT5_SCRIPT
)


###############################################################################
# 4. Save compact exact-source windows
###############################################################################


def source_windows(
    lines,
    hits,
    radius=3,
):

    selected = set()

    for hit in hits:

        number = int(
            hit[
                "line"
            ]
        )

        for line_number in range(
            max(
                1,
                number
                - radius,
            ),
            min(
                len(lines),
                number
                + radius,
            )
            + 1,
        ):
            selected.add(
                line_number
            )

    output = []

    previous = None

    for number in sorted(
        selected
    ):

        if (
            previous is not None
            and number
            > previous
            + 1
        ):
            output.append(
                "..."
            )

        output.append(
            f"{number:06d}: "
            + lines[
                number
                - 1
            ]
        )

        previous = number

    return "\n".join(
        output
    )


atomic_text(
    OUT
    / "ckpt4_line_dependency_source.txt",
    source_windows(
        ckpt4_lines,
        ckpt4_hits,
    ),
)

atomic_text(
    OUT
    / "ckpt5_line_dependency_source.txt",
    source_windows(
        ckpt5_lines,
        ckpt5_hits,
    ),
)


###############################################################################
# 5. CKPT4 checkpoint: trainable line parameters/config
###############################################################################

ckpt4_checkpoint = torch.load(
    ROOT
    / "artifacts/checkpoint4/"
    "temporal_encoder.pt",
    map_location="cpu",
    weights_only=False,
)

config = ckpt4_checkpoint.get(
    "config",
    {},
)

config_matches = {}

for path, value in flatten_dict(
    config
):

    lower = path.lower()

    if (
        "line"
        in lower
        or "treat"
        in lower
    ):
        config_matches[
            path
        ] = value


model_state = ckpt4_checkpoint.get(
    "model_state",
    {},
)

state_matches = []

for name, tensor in model_state.items():

    lower = name.lower()

    if (
        "line"
        not in lower
        and "treat"
        not in lower
    ):
        continue

    record = {
        "name":
            name,

        "type":
            type(
                tensor
            ).__name__,
    }

    if torch.is_tensor(
        tensor
    ):

        value = (
            tensor
            .detach()
            .float()
            .cpu()
        )

        record.update(
            {
                "shape":
                    list(
                        value.shape
                    ),

                "norm":
                    float(
                        torch.linalg.vector_norm(
                            value
                        ).item()
                    ),

                "std":
                    float(
                        value.std().item()
                    )
                    if value.numel()
                    > 1
                    else 0.0,

                "nonconstant":
                    bool(
                        value.numel()
                        > 1
                        and float(
                            value.std().item()
                        )
                        > 0.0
                    ),
            }
        )

    state_matches.append(
        record
    )


###############################################################################
# 6. CKPT4 prepared-token line/treatment fields
###############################################################################

breast_tokens = pd.read_parquet(
    ROOT
    / "artifacts/checkpoint4/prepared/"
    "breast_tokens.parquet"
)

token_dependency_columns = [
    column
    for column
    in breast_tokens.columns
    if (
        "line"
        in column.lower()
        or "treat"
        in column.lower()
    )
]

token_column_summary = {}

for column in token_dependency_columns:

    series = breast_tokens[
        column
    ]

    row = {
        "dtype":
            str(
                series.dtype
            ),

        "nonmissing":
            int(
                series.notna().sum()
            ),

        "unique":
            int(
                series.nunique(
                    dropna=True
                )
            ),
    }

    if pd.api.types.is_numeric_dtype(
        series
    ):

        row[
            "numeric"
        ] = numeric_summary(
            series
        )

    else:

        row[
            "top_values"
        ] = {
            str(key):
                int(value)
            for key, value
            in series.astype(str)
            .value_counts()
            .head(
                20
            )
            .to_dict()
            .items()
        }

    token_column_summary[
        column
    ] = row


scan_cache_index = pd.read_parquet(
    ROOT
    / "artifacts/checkpoint4/"
    "breast_scan_prepost_index.parquet"
)

cache_dependency_columns = [
    column
    for column
    in scan_cache_index.columns
    if (
        "line"
        in column.lower()
        or "treat"
        in column.lower()
    )
]

cache_column_summary = {}

for column in cache_dependency_columns:

    cache_column_summary[
        column
    ] = {
        "dtype":
            str(
                scan_cache_index[
                    column
                ].dtype
            ),

        "unique":
            int(
                scan_cache_index[
                    column
                ].nunique(
                    dropna=True
                )
            ),

        "numeric":
            (
                numeric_summary(
                    scan_cache_index[
                        column
                    ]
                )
                if pd.api.types.is_numeric_dtype(
                    scan_cache_index[
                        column
                    ]
                )
                else None
            ),
    }


###############################################################################
# 7. Exact CKPT5 structured context dependency
###############################################################################

feature_schema = json.load(
    open(
        ROOT
        / "artifacts/checkpoint5/prepared/"
        "feature_schema.json",
        encoding="utf-8",
    )
)

context_names = list(
    feature_schema[
        "context"
    ]
)

expected_context_names = [
    "line_number_scaled",
    "elapsed_on_line_scaled",
    "scan_number_patient_log",
    "scan_number_line_log",
    "genomic_available",
    "genomic_age_scaled",
]

if context_names != expected_context_names:

    raise RuntimeError(
        "Frozen CKPT5 context feature order changed: "
        f"{context_names}"
    )

line_dependent_context = [
    "line_number_scaled",
    "elapsed_on_line_scaled",
    "scan_number_line_log",
]

transportable_context_candidate = [
    "scan_number_patient_log",
    "genomic_available",
    "genomic_age_scaled",
]


###############################################################################
# 8. Reproduce the frozen CKPT5 context formulas exactly
###############################################################################

scan_index = pd.read_parquet(
    ROOT
    / "artifacts/checkpoint5/prepared/"
    "scan_index.parquet"
)

context = np.load(
    ROOT
    / "artifacts/checkpoint5/prepared/"
    "context_features_f32.npy",
    mmap_mode="r",
)

if context.shape != (
    len(
        scan_index
    ),
    6,
):

    raise RuntimeError(
        "CKPT5 context tensor/index shape mismatch: "
        f"{context.shape} vs {len(scan_index)}"
    )

formula_arrays = {
    "line_number_scaled":
        (
            scan_index[
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

    "elapsed_on_line_scaled":
        (
            scan_index[
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

    "scan_number_patient_log":
        (
            np.log1p(
                scan_index[
                    "scan_number_patient"
                ]
                .to_numpy(
                    dtype=np.float32
                )
            )
            / 5.0
        ),

    "scan_number_line_log":
        (
            np.log1p(
                scan_index[
                    "scan_number_line"
                ]
                .to_numpy(
                    dtype=np.float32
                )
            )
            / 5.0
        ),

    "genomic_available":
        scan_index[
            "genomic_available"
        ]
        .to_numpy(
            dtype=np.float32
        ),

    "genomic_age_scaled":
        (
            scan_index[
                "genomic_age_days"
            ]
            .fillna(
                0.0
            )
            .clip(
                0.0,
                3650.0,
            )
            .to_numpy(
                dtype=np.float32
            )
            / 365.0
        ),
}

formula_replay = {}

for column_index, name in enumerate(
    context_names
):

    frozen = np.asarray(
        context[
            :,
            column_index,
        ],
        dtype=np.float32,
    )

    replay = np.asarray(
        formula_arrays[
            name
        ],
        dtype=np.float32,
    )

    difference = np.abs(
        frozen
        - replay
    )

    formula_replay[
        name
    ] = {
        "max_abs_error":
            float(
                difference.max()
            ),

        "mean_abs_error":
            float(
                difference.mean()
            ),

        "exact_fraction":
            float(
                (
                    difference
                    == 0
                ).mean()
            ),
    }

    if float(
        difference.max()
    ) != 0.0:

        raise RuntimeError(
            f"Context formula replay not exact for {name}"
        )


###############################################################################
# 9. Explicit source evidence for scan_number_line construction
###############################################################################

scan_number_line_hits = [
    hit
    for hit in ckpt5_hits
    if (
        "scan_number_line"
        in hit[
            "patterns"
        ]
    )
]

elapsed_line_hits = [
    hit
    for hit in ckpt5_hits
    if (
        "elapsed_on_line"
        in hit[
            "patterns"
        ]
    )
]


###############################################################################
# 10. Controlled CKPT4 frozen-forward line sensitivity
#
# This is diagnostic only. No parameters are changed or saved.
###############################################################################

forward_sensitivity = {
    "status":
        "NOT_RUN",

    "trials":
        [],
}

try:

    ckpt4 = import_module(
        "ckpt7a4c_ckpt4",
        CKPT4_SCRIPT,
    )

    required_api = [
        "TokenWindowDataset",
        "TemporalTransformer",
        "collate_windows",
    ]

    missing_api = [
        name
        for name in required_api
        if not hasattr(
            ckpt4,
            name
        )
    ]

    if missing_api:

        raise RuntimeError(
            f"Missing CKPT4 APIs: {missing_api}"
        )

    if "line" not in breast_tokens.columns:

        raise RuntimeError(
            "breast_tokens has no literal line column."
        )

    candidate = breast_tokens.copy()

    candidate[
        "_line_numeric"
    ] = pd.to_numeric(
        candidate[
            "line"
        ],
        errors="coerce",
    ).fillna(
        0
    )

    patient_max = (
        candidate.groupby(
            "patient_id",
            observed=True,
        )[
            "_line_numeric"
        ]
        .max()
        .sort_values(
            ascending=False
        )
    )

    candidate_patients = [
        str(patient)
        for patient, maximum
        in patient_max.items()
        if maximum
        >= 1
    ][
        :8
    ]

    model = ckpt4.TemporalTransformer()

    missing, unexpected = model.load_state_dict(
        ckpt4_checkpoint[
            "model_state"
        ],
        strict=False,
    )

    if missing or unexpected:

        raise RuntimeError(
            "CKPT4 checkpoint state mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )

    model.eval()
    model.to(
        torch.device(
            "cpu"
        )
    )


    def tensor_map(
        value,
        prefix="output",
    ):

        result = {}

        if torch.is_tensor(
            value
        ):

            result[
                prefix
            ] = (
                value
                .detach()
                .float()
                .cpu()
            )

        elif isinstance(
            value,
            dict,
        ):

            for key, child in value.items():

                result.update(
                    tensor_map(
                        child,
                        (
                            prefix
                            + "."
                            + str(
                                key
                            )
                        ),
                    )
                )

        elif isinstance(
            value,
            (
                tuple,
                list,
            ),
        ):

            for index, child in enumerate(
                value
            ):

                result.update(
                    tensor_map(
                        child,
                        (
                            prefix
                            + "["
                            + str(
                                index
                            )
                            + "]"
                        ),
                    )
                )

        return result


    for patient in candidate_patients:

        original_frame = breast_tokens[
            breast_tokens[
                "patient_id"
            ]
            .astype(str)
            .eq(
                patient
            )
        ].copy()

        if original_frame.empty:
            continue

        split_values = (
            original_frame[
                "split"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        if len(
            split_values
        ) != 1:
            continue

        split = split_values[
            0
        ]

        modified_frame = original_frame.copy()

        original_line = pd.to_numeric(
            modified_frame[
                "line"
            ],
            errors="coerce",
        ).fillna(
            0
        )

        positive = (
            original_line
            > 0
        )

        if not positive.any():
            continue

        modified_line = original_line.copy()

        modified_line.loc[
            positive
        ] = np.clip(
            modified_line.loc[
                positive
            ]
            + 1,
            0,
            15,
        )

        if (
            modified_line.to_numpy()
            == original_line.to_numpy()
        ).all():
            continue

        modified_frame[
            "line"
        ] = modified_line

        original_dataset = (
            ckpt4.TokenWindowDataset(
                original_frame,
                split,
                train_mode=False,
            )
        )

        modified_dataset = (
            ckpt4.TokenWindowDataset(
                modified_frame,
                split,
                train_mode=False,
            )
        )

        length = min(
            len(
                original_dataset
            ),
            len(
                modified_dataset
            ),
        )

        if length < 1:
            continue

        #######################################################################
        # Use the final deterministic evaluation window to maximize exposure
        # to the patient's accumulated treatment history.
        #######################################################################

        item_index = (
            length
            - 1
        )

        original_batch = (
            ckpt4.collate_windows(
                [
                    original_dataset[
                        item_index
                    ]
                ]
            )
        )

        modified_batch = (
            ckpt4.collate_windows(
                [
                    modified_dataset[
                        item_index
                    ]
                ]
            )
        )

        original_batch_tensors = (
            tensor_map(
                original_batch,
                "batch",
            )
        )

        modified_batch_tensors = (
            tensor_map(
                modified_batch,
                "batch",
            )
        )

        changed_batch = []

        for name in sorted(
            set(
                original_batch_tensors
            )
            & set(
                modified_batch_tensors
            )
        ):

            left = original_batch_tensors[
                name
            ]

            right = modified_batch_tensors[
                name
            ]

            if left.shape != right.shape:
                continue

            if not torch.equal(
                left,
                right,
            ):

                if (
                    left.dtype.is_floating_point
                    or right.dtype.is_floating_point
                ):
                    max_difference = float(
                        (
                            left
                            - right
                        )
                        .abs()
                        .max()
                        .item()
                    )

                else:
                    max_difference = None

                changed_batch.append(
                    {
                        "name":
                            name,

                        "shape":
                            list(
                                left.shape
                            ),

                        "max_abs_difference":
                            max_difference,
                    }
                )

        with torch.no_grad():

            original_output = model(
                original_batch
            )

            modified_output = model(
                modified_batch
            )

        original_tensors = tensor_map(
            original_output
        )

        modified_tensors = tensor_map(
            modified_output
        )

        output_differences = []

        max_output_difference = 0.0

        for name in sorted(
            set(
                original_tensors
            )
            & set(
                modified_tensors
            )
        ):

            left = original_tensors[
                name
            ]

            right = modified_tensors[
                name
            ]

            if (
                left.shape
                != right.shape
            ):
                continue

            difference = (
                left
                - right
            ).abs()

            maximum = float(
                difference.max().item()
            )

            mean = float(
                difference.mean().item()
            )

            max_output_difference = max(
                max_output_difference,
                maximum,
            )

            if maximum > 0.0:

                output_differences.append(
                    {
                        "name":
                            name,

                        "shape":
                            list(
                                left.shape
                            ),

                        "max_abs_difference":
                            maximum,

                        "mean_abs_difference":
                            mean,
                    }
                )

        forward_sensitivity[
            "trials"
        ].append(
            {
                "patient_id":
                    patient,

                "split":
                    split,

                "dataset_item":
                    int(
                        item_index
                    ),

                "original_positive_line_min":
                    float(
                        original_line.loc[
                            positive
                        ].min()
                    ),

                "original_positive_line_max":
                    float(
                        original_line.loc[
                            positive
                        ].max()
                    ),

                "changed_batch_tensors":
                    changed_batch,

                "changed_output_tensors":
                    output_differences,

                "max_output_abs_difference":
                    max_output_difference,
            }
        )

        if len(
            forward_sensitivity[
                "trials"
            ]
        ) >= 3:
            break

    if not forward_sensitivity[
        "trials"
    ]:

        forward_sensitivity[
            "status"
        ] = "NO_VALID_TRIAL"

    else:

        maximum = max(
            row[
                "max_output_abs_difference"
            ]
            for row in forward_sensitivity[
                "trials"
            ]
        )

        forward_sensitivity[
            "max_output_abs_difference"
        ] = float(
            maximum
        )

        forward_sensitivity[
            "status"
        ] = (
            "FORWARD_DEPENDENCE_CONFIRMED"
            if maximum
            > 1e-8
            else "NO_FORWARD_CHANGE_DETECTED"
        )

except Exception as exc:

    forward_sensitivity[
        "status"
    ] = "FORWARD_TEST_FAILED"

    forward_sensitivity[
        "error"
    ] = (
        type(
            exc
        ).__name__
        + ": "
        + str(
            exc
        )
    )

    forward_sensitivity[
        "traceback"
    ] = traceback.format_exc()


###############################################################################
# 11. Dependency classification
###############################################################################

ckpt4_line_source_confirmed = any(
    (
        "line_bucket"
        in hit[
            "patterns"
        ]
        or "line_embedding"
        in hit[
            "patterns"
        ]
        or "line_id"
        in hit[
            "patterns"
        ]
        or "treatment_line"
        in hit[
            "patterns"
        ]
        or "line_generic"
        in hit[
            "patterns"
        ]
    )
    for hit in ckpt4_hits
)

ckpt4_trainable_line_parameter = any(
    record.get(
        "nonconstant",
        False,
    )
    for record in state_matches
)

ckpt4_forward_confirmed = (
    forward_sensitivity.get(
        "status"
    )
    == "FORWARD_DEPENDENCE_CONFIRMED"
)

ckpt5_context_confirmed = (
    set(
        line_dependent_context
    )
    .issubset(
        set(
            context_names
        )
    )
)

if (
    ckpt4_line_source_confirmed
    and ckpt5_context_confirmed
):

    status = (
        "TEMPORAL_AND_CONTEXT_LINE_DEPENDENCE_CONFIRMED"
    )

elif ckpt5_context_confirmed:

    status = (
        "CKPT5_CONTEXT_LINE_DEPENDENCE_CONFIRMED_"
        "TEMPORAL_DEPENDENCE_NEEDS_RESOLUTION"
    )

else:

    status = (
        "LINE_DEPENDENCY_SURFACE_NEEDS_RESOLUTION"
    )


###############################################################################
# 12. Repair-surface classification
###############################################################################

repair_surface = {
    "frozen_nontransportable_inputs": {
        "ckpt5_context":
            line_dependent_context,

        "ckpt4_temporal_line_conditioning":
            bool(
                ckpt4_line_source_confirmed
            ),
    },

    "candidate_transportable_ckpt5_context":
        transportable_context_candidate,

    "requires_additional_semantic_review": [
        "CKPT4 treatment-active indicator",
        "CKPT4 time-on-current-treatment feature",
    ],

    "not_justified_after_fix9": [
        (
            "Continue tuning BPC treatment-line reconstruction "
            "against MSK positive control"
        ),
        (
            "Use REGIMEN_NUMBER as a replacement for frozen CKPT1 line"
        ),
        (
            "Drop only line_number_scaled and elapsed_on_line_scaled "
            "while leaving CKPT4 line-conditioned temporal states unchanged"
        ),
    ],

    "minimal_repair_if_temporal_dependency_confirmed": [
        (
            "Keep CKPT1 endpoints, cohorts, patient splits, W3 scan semantics, "
            "CKPT3 genomic encoder and genomic timing rules frozen."
        ),
        (
            "Create a line-agnostic CKPT4 temporal variant by removing or "
            "neutralizing nontransportable treatment-line identity during "
            "training, rather than only at external inference."
        ),
        (
            "Remove all nontransportable line-derived CKPT5 structured "
            "context: line_number_scaled, elapsed_on_line_scaled, and "
            "scan_number_line_log."
        ),
        (
            "Retrain only the downstream temporal/fusion path required by "
            "the interface change on the original CHORD train split."
        ),
        (
            "Repeat internal CKPT6A/6B/6C/6D candidate validation and freeze "
            "a new CKPT6E protocol before any DFCI/VICC outcome is opened."
        ),
    ],
}


###############################################################################
# 13. Persist machine-readable report
###############################################################################

report = {
    "status":
        status,

    "external_outcomes_opened":
        False,

    "external_outcome_distributions_inspected":
        False,

    "external_predictor_rows_opened":
        False,

    "audit_scope":
        (
            "INTERNAL_SOURCE_AND_FROZEN_ARTIFACTS_ONLY"
        ),

    "fix9_trigger": {
        "status":
            fix9[
                "status"
            ],

        "overall_line_context":
            fix9[
                "line_context"
            ][
                "overall"
            ],
    },

    "source_sha256": {
        "ckpt4":
            sha256_file(
                CKPT4_SCRIPT
            ),

        "ckpt5":
            sha256_file(
                CKPT5_SCRIPT
            ),
    },

    "ckpt4": {
        "source_line_dependency_confirmed":
            ckpt4_line_source_confirmed,

        "line_related_source_hits":
            ckpt4_hits,

        "line_related_scopes":
            ckpt4_scopes,

        "config_line_treatment_entries":
            config_matches,

        "trainable_line_treatment_state":
            state_matches,

        "trainable_line_parameter_detected":
            ckpt4_trainable_line_parameter,

        "token_dependency_columns":
            token_column_summary,

        "scan_cache_dependency_columns":
            cache_column_summary,

        "forward_sensitivity":
            forward_sensitivity,
    },

    "ckpt5": {
        "context_feature_order":
            context_names,

        "line_dependent_context":
            line_dependent_context,

        "candidate_transportable_context":
            transportable_context_candidate,

        "context_formula_replay":
            formula_replay,

        "scan_number_line_source_hits":
            scan_number_line_hits,

        "elapsed_on_line_source_hits":
            elapsed_line_hits,

        "line_related_source_hits":
            ckpt5_hits,

        "line_related_scopes":
            ckpt5_scopes,
    },

    "repair_surface":
        repair_surface,

    "decision_flags": {
        "ckpt4_line_source_confirmed":
            ckpt4_line_source_confirmed,

        "ckpt4_trainable_line_parameter":
            ckpt4_trainable_line_parameter,

        "ckpt4_forward_confirmed":
            ckpt4_forward_confirmed,

        "ckpt5_three_line_context_features_confirmed":
            ckpt5_context_confirmed,
    },

    "next_action":
        (
            "Do not patch A4B and do not open DFCI/VICC outcomes. "
            "If temporal line dependence is confirmed, design the minimal "
            "line-agnostic CKPT4 + CKPT5 retraining branch and refreeze "
            "CKPT6 before external validation."
        ),
}

atomic_json(
    OUT
    / "audit.json",
    report,
)


###############################################################################
# 14. Compact decision packet
###############################################################################

print("")
print(
    "========== CKPT7A4C SUMMARY =========="
)

print(
    "status="
    + status
)

print(
    "external_outcomes_opened=False"
)

print(
    "external_outcome_distributions_inspected=False"
)

print(
    "external_predictor_rows_opened=False"
)

print(
    "ckpt4_line_source_confirmed="
    + str(
        ckpt4_line_source_confirmed
    )
)

print(
    "ckpt4_trainable_line_parameter="
    + str(
        ckpt4_trainable_line_parameter
    )
)

print(
    "ckpt4_forward_status="
    + str(
        forward_sensitivity.get(
            "status"
        )
    )
)

print(
    "ckpt4_forward_max_abs="
    + str(
        forward_sensitivity.get(
            "max_output_abs_difference"
        )
    )
)

print(
    "ckpt5_line_dependent_context="
    + str(
        line_dependent_context
    )
)

print(
    "========== CKPT7A4C SUMMARY END =========="
)

print("")
print(
    "========== CKPT7A4C DECISION PACKET =========="
)

print(
    "ckpt4_config_line_treatment_entries="
    + json.dumps(
        config_matches,
        sort_keys=True,
        default=str,
    )
)

print(
    "ckpt4_trainable_line_treatment_state="
    + json.dumps(
        state_matches,
        sort_keys=True,
        default=str,
    )
)

print(
    "ckpt4_token_dependency_columns="
    + json.dumps(
        token_column_summary,
        sort_keys=True,
        default=str,
    )
)

print(
    "ckpt4_forward_sensitivity="
    + json.dumps(
        forward_sensitivity,
        sort_keys=True,
        default=str,
    )
)

print(
    "ckpt5_context_feature_order="
    + json.dumps(
        context_names
    )
)

print(
    "ckpt5_line_dependent_context="
    + json.dumps(
        line_dependent_context
    )
)

print(
    "ckpt5_candidate_transportable_context="
    + json.dumps(
        transportable_context_candidate
    )
)

print(
    "ckpt5_context_formula_replay="
    + json.dumps(
        formula_replay,
        sort_keys=True,
    )
)

print(
    "repair_surface="
    + json.dumps(
        repair_surface,
        sort_keys=True,
        default=str,
    )
)

print(
    "decision_flags="
    + json.dumps(
        report[
            "decision_flags"
        ],
        sort_keys=True,
    )
)

print(
    "status="
    + status
)

print(
    "next_action="
    + report[
        "next_action"
    ]
)

print(
    "========== CKPT7A4C DECISION PACKET END =========="
)

