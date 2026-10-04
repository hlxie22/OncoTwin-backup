#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]

OUT = (
    ROOT
    / "artifacts/checkpoint7r3g1_external_prediction_freeze"
)

CKPT5 = (
    ROOT
    / "scripts/dynamic_scan/"
    "ckpt5_supervised_dynamic_model.py"
)


VARIANTS = (
    "timeline_sequencing",
    "clinical_sample",
)

CENTERS = {
    "DFCI":
        1728,

    "VICC":
        804,
}

PREDICTOR_FILES = (
    "temporal_prepost_f16.npy",
    "tumor_embeddings_f16.npy",
    "current_scan_features_f32.npy",
    "context_features_f32.npy",
)

LOGIT_FILES = {
    "pre":
        "pre_survival_logits_f32.npy",

    "full_post":
        "full_post_survival_logits_f32.npy",

    "bounded_post":
        "bounded_post_survival_logits_f32.npy",
}

PFS_FILES = {
    "pre":
        "pre_pfs_survival_f32.npy",

    "full_post":
        "full_post_pfs_survival_f32.npy",

    "bounded_post":
        "bounded_post_pfs_survival_f32.npy",
}


def sha256_file(
    path: Path,
) -> str:

    h = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:

        for block in iter(
            lambda:
                f.read(
                    1024 * 1024
                ),
            b"",
        ):

            h.update(
                block
            )

    return h.hexdigest()


def write_json_atomic(
    path: Path,
    payload,
):

    tmp = Path(
        str(
            path
        )
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

    tmp.replace(
        path
    )


def save_npy_atomic(
    path: Path,
    array: np.ndarray,
):

    tmp = Path(
        str(
            path
        )
        + ".tmp"
    )

    with tmp.open(
        "wb"
    ) as f:

        np.save(
            f,
            array,
            allow_pickle=False,
        )

    check = np.load(
        tmp,
        mmap_mode="r",
        allow_pickle=False,
    )

    if check.shape != array.shape:

        raise RuntimeError(
            f"Atomic NPY validation failed: {path}"
        )

    if check.dtype != array.dtype:

        raise RuntimeError(
            f"Atomic NPY dtype validation failed: {path}"
        )

    tmp.replace(
        path
    )


def save_parquet_atomic(
    path: Path,
    frame: pd.DataFrame,
):

    tmp = Path(
        str(
            path
        )
        + ".tmp"
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
            f"Atomic parquet validation failed: {path}"
        )

    tmp.replace(
        path
    )


def import_probability_helper():

    spec = importlib.util.spec_from_file_location(
        "ckpt7r3g1_probability_helper",
        CKPT5,
    )

    if (
        spec is None
        or spec.loader is None
    ):

        raise RuntimeError(
            "Cannot import frozen CKPT5."
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

    if not hasattr(
        module,
        "probability_curves",
    ):

        raise RuntimeError(
            "Frozen CKPT5 lacks probability_curves()."
        )

    return module


def as_numpy(
    value,
):

    if torch.is_tensor(
        value
    ):

        return (
            value
            .detach()
            .cpu()
            .numpy()
        )

    return np.asarray(
        value
    )


def canonicalize_index(
    frame: pd.DataFrame,
) -> pd.DataFrame:

    frame = frame.copy()

    ###########################################################################
    # Retired line fields remain metadata only; canonical transport index
    # records the frozen no-line contract explicitly.
    ###########################################################################

    for column, value in (
        (
            "treatment_line",
            0,
        ),
        (
            "scan_number_line",
            0,
        ),
    ):

        if column in frame.columns:

            frame[
                column
            ] = value

    for column in (
        "line_start_day",
        "line_stop_day",
    ):

        if column in frame.columns:

            frame[
                column
            ] = np.nan

    if "active_line" in frame.columns:

        frame[
            "active_line"
        ] = False

    frame[
        "ckpt7r3g1_transport_member"
    ] = True

    return frame


def main():

    ###########################################################################
    # Core prediction QC is the input contract.
    ###########################################################################

    runtime_qc = json.loads(
        (
            OUT
            / "runtime_prediction_qc.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    if (
        runtime_qc[
            "status"
        ]
        != "PASS_CKPT7R3G1_CORE_EXTERNAL_PREDICTIONS"
    ):

        raise RuntimeError(
            "Core predictions are not locked."
        )

    if runtime_qc[
        "predictor_access_violations"
    ]:

        raise RuntimeError(
            "Predictor access violations are nonempty."
        )

    if runtime_qc[
        "external_outcomes_opened"
    ]:

        raise RuntimeError(
            "External outcomes already marked opened."
        )

    ###########################################################################
    # Exact frozen CKPT5 PFS conversion.
    ###########################################################################

    module = import_probability_helper()

    variant_qc = {}

    canonical_indices = {}

    for variant in VARIANTS:

        base = (
            OUT
            / variant
        )

        raw_index = pd.read_parquet(
            base
            / "prediction_index.parquet"
        )

        if len(
            raw_index
        ) != 2532:

            raise RuntimeError(
                f"{variant}: index rows != 2532"
            )

        counts = (
            raw_index[
                "site"
            ]
            .value_counts()
            .to_dict()
        )

        if counts != CENTERS:

            raise RuntimeError(
                f"{variant}: center rows changed: {counts}"
            )

        canonical = canonicalize_index(
            raw_index
        )

        canonical_path = (
            base
            / "canonical_prediction_index.parquet"
        )

        save_parquet_atomic(
            canonical_path,
            canonical,
        )

        canonical_indices[
            variant
        ] = canonical

        #######################################################################
        # Re-verify exact predictor transport before deriving anything.
        #######################################################################

        temporal = np.load(
            base
            / "temporal_prepost_f16.npy",
            mmap_mode="r",
            allow_pickle=False,
        )

        tumor = np.load(
            base
            / "tumor_embeddings_f16.npy",
            mmap_mode="r",
            allow_pickle=False,
        )

        scan = np.load(
            base
            / "current_scan_features_f32.npy",
            mmap_mode="r",
            allow_pickle=False,
        )

        context = np.load(
            base
            / "context_features_f32.npy",
            mmap_mode="r",
            allow_pickle=False,
        )

        if temporal.shape != (
            2532,
            2,
            192,
        ):

            raise RuntimeError(
                f"{variant}: temporal shape {temporal.shape}"
            )

        if tumor.shape != (
            2532,
            128,
        ):

            raise RuntimeError(
                f"{variant}: tumor shape {tumor.shape}"
            )

        if scan.shape != (
            2532,
            18,
        ):

            raise RuntimeError(
                f"{variant}: scan shape {scan.shape}"
            )

        if context.shape != (
            2532,
            6,
        ):

            raise RuntimeError(
                f"{variant}: context shape {context.shape}"
            )

        if not np.all(
            scan[
                :,
                8:18,
            ]
            == 0.0
        ):

            raise RuntimeError(
                f"{variant}: scan 8:18 transport violation"
            )

        if not np.all(
            context[
                :,
                [
                    0,
                    1,
                    3,
                ],
            ]
            == 0.0
        ):

            raise RuntimeError(
                f"{variant}: context line-neutralization violation"
            )

        #######################################################################
        # Logits + exact bounded replay + PFS curves.
        #######################################################################

        logits = {}

        pfs = {}

        mode_qc = {}

        for mode, filename in LOGIT_FILES.items():

            x = np.load(
                base
                / filename,
                mmap_mode="r",
                allow_pickle=False,
            )

            if x.shape != (
                2532,
                24,
                4,
            ):

                raise RuntimeError(
                    f"{variant}/{mode}: logits shape {x.shape}"
                )

            if not np.isfinite(
                x
            ).all():

                raise RuntimeError(
                    f"{variant}/{mode}: nonfinite logits"
                )

            logits[
                mode
            ] = x

        expected_bounded = (
            logits[
                "pre"
            ].astype(
                np.float64
            )
            + 0.5
            * (
                logits[
                    "full_post"
                ].astype(
                    np.float64
                )
                - logits[
                    "pre"
                ].astype(
                    np.float64
                )
            )
        )

        bounded_error = float(
            np.max(
                np.abs(
                    logits[
                        "bounded_post"
                    ].astype(
                        np.float64
                    )
                    - expected_bounded
                )
            )
        )

        if bounded_error > 2e-6:

            raise RuntimeError(
                f"{variant}: bounded replay error {bounded_error}"
            )

        for mode in (
            "pre",
            "full_post",
            "bounded_post",
        ):

            tensor = torch.from_numpy(
                np.asarray(
                    logits[
                        mode
                    ],
                    dtype=np.float32,
                ).copy()
            )

            with torch.no_grad():

                curves = module.probability_curves(
                    tensor
                )

            if (
                not isinstance(
                    curves,
                    dict,
                )
                or "pfs"
                not in curves
            ):

                raise RuntimeError(
                    "Unexpected probability_curves() return contract."
                )

            curve = as_numpy(
                curves[
                    "pfs"
                ]
            ).astype(
                np.float32
            )

            if curve.shape != (
                2532,
                24,
            ):

                raise RuntimeError(
                    f"{variant}/{mode}: PFS shape {curve.shape}"
                )

            if not np.isfinite(
                curve
            ).all():

                raise RuntimeError(
                    f"{variant}/{mode}: nonfinite PFS curve"
                )

            minimum = float(
                curve.min()
            )

            maximum = float(
                curve.max()
            )

            if (
                minimum < -1e-6
                or maximum > 1.0 + 1e-6
            ):

                raise RuntimeError(
                    f"{variant}/{mode}: PFS outside [0,1]"
                )

            max_increase = float(
                np.max(
                    np.diff(
                        curve.astype(
                            np.float64
                        ),
                        axis=1,
                    )
                )
            )

            if max_increase > 1e-6:

                raise RuntimeError(
                    f"{variant}/{mode}: PFS curve not monotone; "
                    f"max increase={max_increase}"
                )

            save_npy_atomic(
                base
                / PFS_FILES[
                    mode
                ],
                curve,
            )

            pfs[
                mode
            ] = curve

            mode_qc[
                mode
            ] = {
                "shape":
                    list(
                        curve.shape
                    ),

                "min":
                    minimum,

                "max":
                    maximum,

                "max_month_to_month_increase":
                    max_increase,
            }

        #######################################################################
        # Complete center-specific bundles.
        #######################################################################

        center_qc = {}

        for center, expected_rows in CENTERS.items():

            mask = (
                canonical[
                    "site"
                ]
                .astype(str)
                .eq(
                    center
                )
                .to_numpy()
            )

            if int(
                mask.sum()
            ) != expected_rows:

                raise RuntimeError(
                    f"{variant}/{center}: row count mismatch"
                )

            center_dir = (
                base
                / center.lower()
            )

            center_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            center_index = (
                canonical.loc[
                    mask
                ]
                .copy()
                .reset_index(
                    drop=True
                )
            )

            center_index[
                "site_row"
            ] = np.arange(
                expected_rows,
                dtype=int,
            )

            save_parquet_atomic(
                center_dir
                / "prediction_index.parquet",
                center_index,
            )

            ###################################################################
            # Predictor tensors.
            ###################################################################

            predictor_arrays = {
                "temporal_prepost_f16.npy":
                    temporal,

                "tumor_embeddings_f16.npy":
                    tumor,

                "current_scan_features_f32.npy":
                    scan,

                "context_features_f32.npy":
                    context,
            }

            for filename, array in predictor_arrays.items():

                sliced = np.asarray(
                    array[
                        mask
                    ]
                )

                save_npy_atomic(
                    center_dir
                    / filename,
                    sliced,
                )

                reread = np.load(
                    center_dir
                    / filename,
                    mmap_mode="r",
                    allow_pickle=False,
                )

                if not np.array_equal(
                    reread,
                    sliced,
                ):

                    raise RuntimeError(
                        f"{variant}/{center}: predictor copy mismatch "
                        f"{filename}"
                    )

            ###################################################################
            # Logits.
            ###################################################################

            for mode, filename in LOGIT_FILES.items():

                sliced = np.asarray(
                    logits[
                        mode
                    ][
                        mask
                    ]
                )

                save_npy_atomic(
                    center_dir
                    / filename,
                    sliced,
                )

                reread = np.load(
                    center_dir
                    / filename,
                    mmap_mode="r",
                    allow_pickle=False,
                )

                if not np.array_equal(
                    reread,
                    sliced,
                ):

                    raise RuntimeError(
                        f"{variant}/{center}: logit copy mismatch {filename}"
                    )

            ###################################################################
            # PFS curves.
            ###################################################################

            for mode, filename in PFS_FILES.items():

                sliced = np.asarray(
                    pfs[
                        mode
                    ][
                        mask
                    ]
                )

                save_npy_atomic(
                    center_dir
                    / filename,
                    sliced,
                )

                reread = np.load(
                    center_dir
                    / filename,
                    mmap_mode="r",
                    allow_pickle=False,
                )

                if not np.array_equal(
                    reread,
                    sliced,
                ):

                    raise RuntimeError(
                        f"{variant}/{center}: PFS copy mismatch {filename}"
                    )

            center_qc[
                center
            ] = {
                "rows":
                    expected_rows,

                "patients":
                    int(
                        center_index[
                            "patient_id"
                        ].nunique()
                    ),

                "complete_bundle":
                    True,
            }

        variant_qc[
            variant
        ] = {
            "rows":
                2532,

            "site_counts":
                counts,

            "bounded_alpha_0_5_max_abs_error":
                bounded_error,

            "PFS":
                mode_qc,

            "centers":
                center_qc,

            "canonical_index":
                str(
                    canonical_path.relative_to(
                        ROOT
                    )
                ),

            "scan_8_17_exact_zero":
                True,

            "context_0_1_3_exact_zero":
                True,
        }

    ###########################################################################
    # Primary and timing sensitivity must remain identical in cohort/order.
    ###########################################################################

    keys = [
        "external_landmark_row",
        "patient_id",
        "site",
        "bpc_episode_id",
        "landmark_day",
    ]

    a = canonical_indices[
        "timeline_sequencing"
    ]

    b = canonical_indices[
        "clinical_sample"
    ]

    for key in keys:

        if (
            key not in a.columns
            or key not in b.columns
        ):

            raise RuntimeError(
                f"Cross-variant identity key missing: {key}"
            )

    if not a[
        keys
    ].equals(
        b[
            keys
        ]
    ):

        raise RuntimeError(
            "Primary/sensitivity landmark identity or order changed."
        )

    ###########################################################################
    # Predictor-access quarantine remains clean.
    ###########################################################################

    adapter = json.loads(
        (
            OUT
            / "frozen_adapter_spec.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    access = adapter.get(
        "predictor_access"
    )

    if not isinstance(
        access,
        dict,
    ):

        raise RuntimeError(
            "Predictor access audit missing."
        )

    if access.get(
        "violations"
    ):

        raise RuntimeError(
            f"External outcome quarantine violation: "
            f"{access['violations']}"
        )

    ###########################################################################
    # Pre-manifest QC. This itself will be included in the immutable manifest.
    ###########################################################################

    freeze_qc = {
        "status":
            "PASS_CKPT7R3G1_PRE_MANIFEST_FREEZE_QC",

        "population": {
            "rows":
                2532,

            "DFCI":
                1728,

            "VICC":
                804,
        },

        "candidate":
            "CKPT7R1_R2_R3B_ALPHA_0_5",

        "alpha":
            0.5,

        "primary_variant":
            "timeline_sequencing",

        "sensitivity_variant":
            "clinical_sample",

        "cross_variant_landmark_identity":
            True,

        "variants":
            variant_qc,

        "predictor_access_violations":
            [],

        "external_outcomes_opened":
            False,

        "external_outcome_distributions_inspected":
            False,

        "ready_for_immutable_manifest":
            True,
    }

    write_json_atomic(
        OUT
        / "freeze_qc_pre_manifest.json",
        freeze_qc,
    )

    print(
        "[CKPT7R3G1_FIX4_PFS_AND_CENTER_FREEZE_PASS]",
        freeze_qc,
    )


if __name__ == "__main__":

    main()
