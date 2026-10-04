#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]

G1 = (
    ROOT
    / "artifacts/checkpoint7r3g1_external_prediction_freeze"
)

B1 = (
    ROOT
    / "artifacts/checkpoint7b1_external_endpoint_freeze"
)

OUT = (
    ROOT
    / "artifacts/checkpoint7b2_external_validation"
)

CKPT5_SOURCE = (
    ROOT
    / "scripts/dynamic_scan/ckpt5_supervised_dynamic_model.py"
)

CKPT6A_SOURCE = (
    ROOT
    / "scripts/dynamic_scan/ckpt6a_diagnose_scan_update.py"
)

TRAIN_INDEX = (
    ROOT
    / "artifacts/checkpoint5/prepared/scan_index.parquet"
)

EXPECTED_G1_MANIFEST_SHA = (
    "88fe02b0d9011828ed62b1e723cb9ae1649d4aa9af91abe5e27c19caef452f2b"
)

EXPECTED_B1_ENDPOINT_SHA = (
    "874577fde594d148f3e30a4d4f87027728960699669df0af35ceae380eb9d2ec"
)

VARIANTS = (
    "timeline_sequencing",
    "clinical_sample",
)

CENTERS = (
    "DFCI",
    "VICC",
)

HORIZONS = (
    3,
    6,
    12,
    18,
)

BOOTSTRAP_REPETITIONS = 2000


def sha256_file(
    path: Path,
) -> str:

    h = hashlib.sha256()

    with path.open("rb") as f:

        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def atomic_json(
    path: Path,
    payload,
):

    tmp = Path(
        str(path)
        + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            json_safe(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
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
):

    tmp = Path(
        str(path)
        + ".tmp"
    )

    frame.to_parquet(
        tmp,
        index=False,
    )

    check = pd.read_parquet(tmp)

    if len(check) != len(frame):
        raise RuntimeError(
            f"Parquet verification failed: {path}"
        )

    tmp.replace(path)


def json_safe(
    value,
):

    if isinstance(
        value,
        dict,
    ):
        return {
            str(k):
                json_safe(v)
            for k, v
            in value.items()
        }

    if isinstance(
        value,
        (list, tuple),
    ):
        return [
            json_safe(v)
            for v
            in value
        ]

    if isinstance(
        value,
        np.ndarray,
    ):
        return json_safe(
            value.tolist()
        )

    if isinstance(
        value,
        (
            np.integer,
            int,
        ),
    ):
        return int(value)

    if isinstance(
        value,
        (
            np.bool_,
            bool,
        ),
    ):
        return bool(value)

    if isinstance(
        value,
        (
            np.floating,
            float,
        ),
    ):
        x = float(value)

        if not math.isfinite(x):
            return None

        return x

    return value


def import_module_from_path(
    path: Path,
    name: str,
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
            f"Could not import {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


def find_ckpt6b_source() -> Path:

    candidates = []

    for path in sorted(
        (
            ROOT
            / "scripts/dynamic_scan"
        ).glob(
            "ckpt6b*.py"
        )
    ):

        text = path.read_text(
            encoding="utf-8"
        )

        if (
            "def bootstrap_integrated_brier_gain"
            in text
        ):
            candidates.append(path)

    if len(candidates) != 1:

        raise RuntimeError(
            "Expected exactly one canonical CKPT6B source containing "
            "bootstrap_integrated_brier_gain; found "
            f"{[str(x) for x in candidates]}"
        )

    return candidates[0]


def tensor_to_numpy(
    value,
) -> np.ndarray:

    if torch.is_tensor(value):

        return (
            value
            .detach()
            .float()
            .cpu()
            .numpy()
        )

    return np.asarray(
        value
    )


def curves_from_logits(
    ckpt5,
    logits: np.ndarray,
) -> dict[str, np.ndarray]:

    tensor = torch.from_numpy(
        np.asarray(
            logits,
            dtype=np.float32,
        ).copy()
    )

    with torch.no_grad():

        raw = ckpt5.probability_curves(
            tensor
        )

    if not isinstance(
        raw,
        dict,
    ):

        raise RuntimeError(
            "CKPT5 probability_curves did not return dict."
        )

    curves = {
        str(k):
            tensor_to_numpy(v)
        for k, v
        in raw.items()
    }

    if "pfs" not in curves:

        raise RuntimeError(
            f"CKPT5 probability_curves missing pfs: {list(curves)}"
        )

    if curves[
        "pfs"
    ].shape != (
        len(logits),
        24,
    ):

        raise RuntimeError(
            f"Invalid PFS curve shape {curves['pfs'].shape}"
        )

    if "entropy" not in curves:

        #######################################################################
        # metric_bundle expects entropy. This field is not a primary metric.
        #######################################################################

        curves[
            "entropy"
        ] = np.zeros(
            len(logits),
            dtype=np.float32,
        )

    return curves


def nll_from_logits(
    ckpt5,
    logits: np.ndarray,
    frame: pd.DataFrame,
) -> np.ndarray:

    tensor = torch.from_numpy(
        np.asarray(
            logits,
            dtype=np.float32,
        ).copy()
    )

    time = torch.tensor(
        frame[
            "survival_time_days"
        ].to_numpy(
            dtype=np.float32
        ),
        dtype=torch.float32,
    )

    cause = torch.tensor(
        frame[
            "survival_cause"
        ].astype(
            int
        ).to_numpy(),
        dtype=torch.long,
    )

    with torch.no_grad():

        loss = (
            ckpt5.competing_risk_nll_per_row(
                tensor,
                time,
                cause,
            )
            .float()
            .cpu()
            .numpy()
        )

    if loss.shape != (
        len(frame),
    ):

        raise RuntimeError(
            f"Unexpected per-row NLL shape: {loss.shape}"
        )

    if not np.isfinite(
        loss
    ).all():

        raise RuntimeError(
            "Nonfinite external NLL."
        )

    return loss


def verify_variant_hashes(
    variant: str,
):

    root = (
        G1
        / variant
    )

    manifest = json.loads(
        (
            root
            / "variant_manifest.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    hashes = manifest[
        "hashes"
    ]

    verified = 0

    for relative, expected in hashes.items():

        path = (
            ROOT
            / relative
        )

        if (
            not path.exists()
            or path.stat().st_size
            == 0
        ):
            raise RuntimeError(
                f"Missing frozen variant artifact: {relative}"
            )

        observed = sha256_file(
            path
        )

        if observed != expected:
            raise RuntimeError(
                f"Frozen variant hash mismatch: {relative}"
            )

        verified += 1

    return {
        "variant":
            variant,

        "files_verified":
            verified,
    }


def verify_index_identity(
    endpoints: pd.DataFrame,
    variant_index: pd.DataFrame,
    variant: str,
):

    if len(
        endpoints
    ) != len(
        variant_index
    ):

        raise RuntimeError(
            f"{variant}: endpoint/index row mismatch."
        )

    integer_key = (
        endpoints[
            "external_landmark_row"
        ].to_numpy(
            dtype=int
        )
    )

    variant_key = (
        variant_index[
            "external_landmark_row"
        ].to_numpy(
            dtype=int
        )
    )

    if not np.array_equal(
        integer_key,
        variant_key,
    ):

        raise RuntimeError(
            f"{variant}: external_landmark_row mismatch."
        )

    for column in (
        "patient_id",
        "site",
        "bpc_episode_id",
    ):

        left = (
            endpoints[
                column
            ]
            .astype(str)
            .to_numpy()
        )

        right = (
            variant_index[
                column
            ]
            .astype(str)
            .to_numpy()
        )

        if not np.array_equal(
            left,
            right,
        ):

            raise RuntimeError(
                f"{variant}: identity mismatch for {column}"
            )

    if not np.allclose(
        endpoints[
            "landmark_day"
        ].to_numpy(
            dtype=float
        ),
        variant_index[
            "landmark_day"
        ].to_numpy(
            dtype=float
        ),
        rtol=0.0,
        atol=0.0,
    ):

        raise RuntimeError(
            f"{variant}: landmark_day mismatch."
        )


def load_variant(
    ckpt5,
    endpoints: pd.DataFrame,
    variant: str,
):

    root = (
        G1
        / variant
    )

    index = pd.read_parquet(
        root
        / "canonical_prediction_index.parquet"
    )

    raw_index = pd.read_parquet(
        root
        / "prediction_index.parquet"
    )

    if len(
        index
    ) != 2532:

        raise RuntimeError(
            f"{variant}: canonical index row count changed."
        )

    if not np.array_equal(
        index[
            "external_landmark_row"
        ].to_numpy(
            dtype=int
        ),
        raw_index[
            "external_landmark_row"
        ].to_numpy(
            dtype=int
        ),
    ):

        raise RuntimeError(
            f"{variant}: canonical/raw prediction index order mismatch."
        )

    verify_index_identity(
        endpoints,
        index,
        variant,
    )

    pre_logits = np.load(
        root
        / "pre_survival_logits_f32.npy"
    )

    full_logits = np.load(
        root
        / "full_post_survival_logits_f32.npy"
    )

    bounded_logits = np.load(
        root
        / "bounded_post_survival_logits_f32.npy"
    )

    for name, array in (
        (
            "pre",
            pre_logits,
        ),
        (
            "full_post",
            full_logits,
        ),
        (
            "bounded_post",
            bounded_logits,
        ),
    ):

        if array.shape != (
            2532,
            24,
            4,
        ):

            raise RuntimeError(
                f"{variant}/{name}: invalid logit shape {array.shape}"
            )

        if not np.isfinite(
            array
        ).all():

            raise RuntimeError(
                f"{variant}/{name}: nonfinite logits."
            )

    replay_error = float(
        np.max(
            np.abs(
                bounded_logits
                - (
                    pre_logits
                    + 0.5
                    * (
                        full_logits
                        - pre_logits
                    )
                )
            )
        )
    )

    if replay_error > 5e-7:

        raise RuntimeError(
            f"{variant}: alpha=0.5 bounded-logit replay failed: "
            f"{replay_error}"
        )

    curves = {
        "pre":
            curves_from_logits(
                ckpt5,
                pre_logits,
            ),

        "full_post":
            curves_from_logits(
                ckpt5,
                full_logits,
            ),

        "bounded_post":
            curves_from_logits(
                ckpt5,
                bounded_logits,
            ),
    }

    ###########################################################################
    # Optional replay against G1 materialized PFS files if they exist.
    ###########################################################################

    pfs_replay = {}

    possible = {
        "pre":
            root
            / "pre_pfs_survival_f32.npy",

        "full_post":
            root
            / "full_post_pfs_survival_f32.npy",

        "bounded_post":
            root
            / "bounded_post_pfs_survival_f32.npy",
    }

    for mode, path in possible.items():

        if path.exists():

            frozen = np.load(
                path
            )

            observed = curves[
                mode
            ][
                "pfs"
            ]

            if frozen.shape != observed.shape:

                raise RuntimeError(
                    f"{variant}/{mode}: frozen PFS shape mismatch."
                )

            error = float(
                np.max(
                    np.abs(
                        frozen.astype(float)
                        - observed.astype(float)
                    )
                )
            )

            if error > 2e-6:

                raise RuntimeError(
                    f"{variant}/{mode}: frozen PFS replay failed: {error}"
                )

            pfs_replay[
                mode
            ] = error

    return {
        "index":
            index,

        "pre_logits":
            pre_logits,

        "full_post_logits":
            full_logits,

        "bounded_post_logits":
            bounded_logits,

        "curves":
            curves,

        "bounded_alpha_replay_max_abs":
            replay_error,

        "pfs_replay":
            pfs_replay,
    }


def build_prediction(
    ckpt5,
    frame: pd.DataFrame,
    logits: np.ndarray,
    curves: dict[str, np.ndarray],
) -> dict[str, Any]:

    return {
        "frame":
            frame,

        "nll":
            nll_from_logits(
                ckpt5,
                logits,
                frame,
            ),

        "curves":
            {
                key:
                    value
                for key, value
                in curves.items()
            },
    }


def subset_curves(
    curves: dict[str, np.ndarray],
    rows: np.ndarray,
) -> dict[str, np.ndarray]:

    return {
        key:
            np.asarray(
                value
            )[
                rows
            ]
        for key, value
        in curves.items()
    }


def evaluate_group(
    ckpt5,
    ckpt6a,
    ckpt6b,
    endpoints: pd.DataFrame,
    loaded,
    censoring_km,
    group_mask: np.ndarray,
) -> dict[str, Any]:

    rows = np.flatnonzero(
        group_mask
    )

    frame = (
        endpoints.iloc[
            rows
        ]
        [
            [
                "patient_id",
                "site",
                "external_landmark_row",
                "bpc_episode_id",
                "landmark_day",
                "survival_time_days",
                "survival_cause",
                "survival_cause_name",
            ]
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    if len(frame) < 20:

        raise RuntimeError(
            "Too few evaluable rows for external metric computation."
        )

    if frame[
        "patient_id"
    ].nunique() < 20:

        raise RuntimeError(
            "Too few evaluable patients for external bootstrap."
        )

    predictions = {}

    for mode, logits_key in (
        (
            "pre",
            "pre_logits",
        ),
        (
            "full_post",
            "full_post_logits",
        ),
        (
            "bounded_post",
            "bounded_post_logits",
        ),
    ):

        predictions[
            mode
        ] = build_prediction(
            ckpt5,
            frame,
            loaded[
                logits_key
            ][
                rows
            ],
            subset_curves(
                loaded[
                    "curves"
                ][
                    mode
                ],
                rows,
            ),
        )

    metrics = {
        mode:
            ckpt6a.metric_bundle(
                prediction,
                censoring_km,
            )
        for mode, prediction
        in predictions.items()
    }

    ###########################################################################
    # Frozen paired patient-level bootstraps:
    # positive gain = bounded/current-scan candidate improves over PRE.
    ###########################################################################

    nll_bootstrap = (
        ckpt6a.bootstrap_nll_gain(
            frame,
            predictions[
                "pre"
            ][
                "nll"
            ],
            predictions[
                "bounded_post"
            ][
                "nll"
            ],
            repetitions=
                BOOTSTRAP_REPETITIONS,
        )
    )

    ibs_bootstrap = (
        ckpt6b.bootstrap_integrated_brier_gain(
            ckpt6a,
            frame,
            predictions[
                "pre"
            ][
                "curves"
            ][
                "pfs"
            ],
            predictions[
                "bounded_post"
            ][
                "curves"
            ][
                "pfs"
            ],
            censoring_km,
            repetitions=
                BOOTSTRAP_REPETITIONS,
        )
    )

    pre_nll = float(
        metrics[
            "pre"
        ][
            "patient_mean_nll"
        ]
    )

    bounded_nll = float(
        metrics[
            "bounded_post"
        ][
            "patient_mean_nll"
        ]
    )

    pre_ibs = float(
        metrics[
            "pre"
        ][
            "patient_integrated_brier_4h"
        ]
    )

    bounded_ibs = float(
        metrics[
            "bounded_post"
        ][
            "patient_integrated_brier_4h"
        ]
    )

    return {
        "rows":
            int(
                len(frame)
            ),

        "patients":
            int(
                frame[
                    "patient_id"
                ].nunique()
            ),

        "cause_counts":
            {
                str(k):
                    int(v)
                for k, v
                in frame[
                    "survival_cause_name"
                ]
                .value_counts()
                .to_dict()
                .items()
            },

        "pre":
            metrics[
                "pre"
            ],

        "full_post_descriptive":
            metrics[
                "full_post"
            ],

        "bounded_post":
            metrics[
                "bounded_post"
            ],

        "pre_minus_bounded": {
            "patient_nll":
                (
                    pre_nll
                    - bounded_nll
                ),

            "patient_ibs":
                (
                    pre_ibs
                    - bounded_ibs
                ),
        },

        "bootstrap_2000": {
            "pre_minus_bounded_patient_nll":
                nll_bootstrap,

            "pre_minus_bounded_patient_ibs":
                ibs_bootstrap,
        },
    }


def compact_result(
    result,
):

    return {
        "rows":
            result[
                "rows"
            ],

        "patients":
            result[
                "patients"
            ],

        "pre_nll":
            result[
                "pre"
            ][
                "patient_mean_nll"
            ],

        "bounded_nll":
            result[
                "bounded_post"
            ][
                "patient_mean_nll"
            ],

        "nll_gain":
            result[
                "pre_minus_bounded"
            ][
                "patient_nll"
            ],

        "nll_bootstrap":
            result[
                "bootstrap_2000"
            ][
                "pre_minus_bounded_patient_nll"
            ],

        "pre_ibs":
            result[
                "pre"
            ][
                "patient_integrated_brier_4h"
            ],

        "bounded_ibs":
            result[
                "bounded_post"
            ][
                "patient_integrated_brier_4h"
            ],

        "ibs_gain":
            result[
                "pre_minus_bounded"
            ][
                "patient_ibs"
            ],

        "ibs_bootstrap":
            result[
                "bootstrap_2000"
            ][
                "pre_minus_bounded_patient_ibs"
            ],

        "bounded_horizon_brier":
            {
                horizon:
                    result[
                        "bounded_post"
                    ][
                        "patient_balanced_pfs"
                    ][
                        f"{horizon}m"
                    ].get(
                        "brier"
                    )
                for horizon
                in HORIZONS
            },
    }


def main():

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    ###########################################################################
    # Immutable B1/G1 gates.
    ###########################################################################

    if (
        sha256_file(
            G1
            / "prediction_freeze_manifest.json"
        )
        != EXPECTED_G1_MANIFEST_SHA
    ):

        raise RuntimeError(
            "G1 prediction manifest changed."
        )

    endpoint_path = (
        B1
        / "external_landmark_endpoints.parquet"
    )

    if (
        sha256_file(
            endpoint_path
        )
        != EXPECTED_B1_ENDPOINT_SHA
    ):

        raise RuntimeError(
            "B1 endpoint file changed."
        )

    b1_decision = json.loads(
        (
            B1
            / "decision.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    if (
        b1_decision[
            "status"
        ]
        != "PASS_EXTERNAL_ENDPOINT_FREEZE"
    ):

        raise RuntimeError(
            "B1 endpoint freeze is not PASS."
        )

    ###########################################################################
    # Import exact frozen metric/model semantics.
    ###########################################################################

    ckpt5 = import_module_from_path(
        CKPT5_SOURCE,
        "ckpt7b2_frozen_ckpt5",
    )

    ckpt6a = import_module_from_path(
        CKPT6A_SOURCE,
        "ckpt7b2_frozen_ckpt6a",
    )

    ckpt6b_source = find_ckpt6b_source()

    ckpt6b = import_module_from_path(
        ckpt6b_source,
        "ckpt7b2_frozen_ckpt6b",
    )

    required_functions = {
        "ckpt5":
            [
                "probability_curves",
                "competing_risk_nll_per_row",
            ],

        "ckpt6a":
            [
                "fit_censoring_km",
                "metric_bundle",
                "bootstrap_nll_gain",
                "horizon_status_and_weight",
            ],

        "ckpt6b":
            [
                "bootstrap_integrated_brier_gain",
            ],
    }

    modules = {
        "ckpt5":
            ckpt5,

        "ckpt6a":
            ckpt6a,

        "ckpt6b":
            ckpt6b,
    }

    for module_name, names in required_functions.items():

        module = modules[
            module_name
        ]

        for name in names:

            if not callable(
                getattr(
                    module,
                    name,
                    None,
                )
            ):

                raise RuntimeError(
                    f"Missing metric function {module_name}.{name}"
                )

    print(
        "[CKPT7B2_METRIC_INTERFACE_PASS]",
        {
            "ckpt6b_source":
                str(
                    ckpt6b_source.relative_to(
                        ROOT
                    )
                ),

            "probability_curves":
                str(
                    inspect.signature(
                        ckpt5.probability_curves
                    )
                ),

            "competing_risk_nll_per_row":
                str(
                    inspect.signature(
                        ckpt5.competing_risk_nll_per_row
                    )
                ),

            "metric_bundle":
                str(
                    inspect.signature(
                        ckpt6a.metric_bundle
                    )
                ),

            "bootstrap_nll_gain":
                str(
                    inspect.signature(
                        ckpt6a.bootstrap_nll_gain
                    )
                ),

            "bootstrap_integrated_brier_gain":
                str(
                    inspect.signature(
                        ckpt6b.bootstrap_integrated_brier_gain
                    )
                ),
        },
        flush=True,
    )

    ###########################################################################
    # Exact frozen CHORD censoring KM.
    ###########################################################################

    train_index = pd.read_parquet(
        TRAIN_INDEX
    )

    train = train_index[
        (
            train_index[
                "split"
            ]
            == "train"
        )
        & train_index[
            "survival_mask"
        ].astype(
            bool
        )
    ].copy()

    if len(
        train
    ) != 14000:

        raise RuntimeError(
            f"Frozen censor-KM training rows changed: {len(train)}"
        )

    if train[
        "patient_id"
    ].nunique() != 2014:

        raise RuntimeError(
            "Frozen censor-KM training patient count changed."
        )

    observed_cause_counts = {
        int(k):
            int(v)
        for k, v
        in train[
            "survival_cause"
        ]
        .astype(int)
        .value_counts()
        .to_dict()
        .items()
    }

    expected_cause_counts = {
        int(
            ckpt6a.CAUSE_CENSOR
        ):
            3591,

        int(
            ckpt6a.CAUSE_PROGRESSION
        ):
            7588,

        int(
            ckpt6a.CAUSE_DEATH
        ):
            552,

        int(
            ckpt6a.CAUSE_SWITCH
        ):
            2269,
    }

    if observed_cause_counts != expected_cause_counts:

        raise RuntimeError(
            "Frozen CHORD censor-KM cause counts changed: "
            f"{observed_cause_counts}"
        )

    censoring_km = ckpt6a.fit_censoring_km(
        train[
            "survival_time_days"
        ].to_numpy(
            dtype=float
        ),
        train[
            "survival_cause"
        ].astype(
            int
        ).to_numpy(),
    )

    print(
        "[CKPT7B2_FROZEN_CENSOR_KM_PASS]",
        {
            "rows":
                len(train),

            "patients":
                train[
                    "patient_id"
                ].nunique(),

            "cause_counts":
                observed_cause_counts,
        },
        flush=True,
    )

    ###########################################################################
    # Frozen external endpoints. NO original BPC outcome source is accessed.
    ###########################################################################

    endpoints = pd.read_parquet(
        endpoint_path
    )

    if len(
        endpoints
    ) != 2532:

        raise RuntimeError(
            "External endpoint row count changed."
        )

    if not endpoints[
        "external_landmark_row"
    ].is_unique:

        raise RuntimeError(
            "External landmark key is no longer unique."
        )

    evaluable = endpoints[
        "evaluable"
    ].astype(
        bool
    ).to_numpy()

    if int(
        evaluable.sum()
    ) != 1558:

        raise RuntimeError(
            f"External evaluable row count changed: {evaluable.sum()}"
        )

    expected_evaluable = {
        "DFCI":
            {
                "rows":
                    1099,

                "patients":
                    220,
            },

        "VICC":
            {
                "rows":
                    459,

                "patients":
                    105,
            },
    }

    for center in CENTERS:

        sub = endpoints[
            (
                endpoints[
                    "site"
                ]
                == center
            )
            & endpoints[
                "evaluable"
            ].astype(
                bool
            )
        ]

        observed = {
            "rows":
                len(sub),

            "patients":
                sub[
                    "patient_id"
                ].nunique(),
        }

        if observed != expected_evaluable[
            center
        ]:

            raise RuntimeError(
                f"{center} frozen endpoint population changed: {observed}"
            )

    valid = endpoints.loc[
        evaluable
    ]

    if not (
        valid[
            "survival_time_days"
        ].to_numpy(
            dtype=float
        )
        > 0
    ).all():

        raise RuntimeError(
            "Nonpositive residual external endpoint time."
        )

    if not (
        valid[
            "survival_time_days"
        ].to_numpy(
            dtype=float
        )
        <= 730.0
    ).all():

        raise RuntimeError(
            "External endpoint beyond frozen 730-day horizon."
        )

    if not set(
        valid[
            "survival_cause"
        ].astype(
            int
        )
    ).issubset(
        {
            int(
                ckpt6a.CAUSE_CENSOR
            ),
            int(
                ckpt6a.CAUSE_PROGRESSION
            ),
            int(
                ckpt6a.CAUSE_DEATH
            ),
            int(
                ckpt6a.CAUSE_SWITCH
            ),
        }
    ):

        raise RuntimeError(
            "Unexpected external survival cause."
        )

    print(
        "[CKPT7B2_ENDPOINT_LOCK_PASS]",
        {
            "rows":
                len(endpoints),

            "evaluable_rows":
                int(
                    evaluable.sum()
                ),

            "DFCI":
                expected_evaluable[
                    "DFCI"
                ],

            "VICC":
                expected_evaluable[
                    "VICC"
                ],
        },
        flush=True,
    )

    ###########################################################################
    # Frozen prediction validation.
    ###########################################################################

    hash_qc = {}

    loaded = {}

    for variant in VARIANTS:

        hash_qc[
            variant
        ] = verify_variant_hashes(
            variant
        )

        loaded[
            variant
        ] = load_variant(
            ckpt5,
            endpoints,
            variant,
        )

        print(
            "[CKPT7B2_VARIANT_LOCK_PASS]",
            {
                "variant":
                    variant,

                "files_verified":
                    hash_qc[
                        variant
                    ][
                        "files_verified"
                    ],

                "bounded_alpha_replay_max_abs":
                    loaded[
                        variant
                    ][
                        "bounded_alpha_replay_max_abs"
                    ],

                "pfs_replay":
                    loaded[
                        variant
                    ][
                        "pfs_replay"
                    ],
            },
            flush=True,
        )

    ###########################################################################
    # Evaluate primary variant and predeclared genomic-timing sensitivity.
    ###########################################################################

    report = {
        "status":
            "COMPLETE_ONE_SHOT_EXTERNAL_VALIDATION",

        "candidate":
            "CKPT7R1_R2_R3B_ALPHA_0_5",

        "alpha":
            0.5,

        "prediction_manifest_sha256":
            EXPECTED_G1_MANIFEST_SHA,

        "endpoint_sha256":
            EXPECTED_B1_ENDPOINT_SHA,

        "primary_variant":
            "timeline_sequencing",

        "sensitivity_variant":
            "clinical_sample",

        "primary_centers":
            [
                "DFCI",
                "VICC",
            ],

        "pooled":
            "SECONDARY",

        "horizons_months":
            list(
                HORIZONS
            ),

        "bootstrap_repetitions":
            BOOTSTRAP_REPETITIONS,

        "metric_semantics": {
            "PFS":
                "progression + death",

            "switch":
                "separate competing cause",

            "censoring_KM":
                "frozen CHORD training",

            "primary":
                "patient-balanced four-horizon PFS IBS",

            "complementary":
                "patient-balanced multicause competing-risk NLL",

            "positive_gain":
                "PRE metric minus bounded POST metric > 0 means current-scan update improves",
        },

        "variants":
            {},
    }

    row_output = (
        endpoints.loc[
            evaluable
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    evaluable_rows = np.flatnonzero(
        evaluable
    )

    for variant in VARIANTS:

        variant_report = {
            "role":
                (
                    "PRIMARY"
                    if variant
                    == "timeline_sequencing"
                    else "PREDECLARED_SENSITIVITY"
                ),

            "centers":
                {},

            "pooled_secondary":
                None,
        }

        #######################################################################
        # Row-level frozen prediction columns for audit.
        #######################################################################

        for mode, logits_key in (
            (
                "pre",
                "pre_logits",
            ),
            (
                "full_post",
                "full_post_logits",
            ),
            (
                "bounded_post",
                "bounded_post_logits",
            ),
        ):

            local_frame = (
                endpoints.iloc[
                    evaluable_rows
                ][
                    [
                        "patient_id",
                        "survival_time_days",
                        "survival_cause",
                    ]
                ]
                .copy()
                .reset_index(
                    drop=True
                )
            )

            nll = nll_from_logits(
                ckpt5,
                loaded[
                    variant
                ][
                    logits_key
                ][
                    evaluable_rows
                ],
                local_frame,
            )

            prefix = (
                f"{variant}__{mode}"
            )

            row_output[
                prefix
                + "__nll"
            ] = nll

            pfs = (
                loaded[
                    variant
                ][
                    "curves"
                ][
                    mode
                ][
                    "pfs"
                ][
                    evaluable_rows
                ]
            )

            for horizon in HORIZONS:

                row_output[
                    prefix
                    + f"__pfs_{horizon}m"
                ] = (
                    pfs[
                        :,
                        horizon
                        - 1
                    ]
                )

        #######################################################################
        # DFCI and VICC are the separate primary analyses.
        #######################################################################

        for center in CENTERS:

            mask = (
                evaluable
                & (
                    endpoints[
                        "site"
                    ].astype(str)
                    .to_numpy()
                    == center
                )
            )

            print(
                "[CKPT7B2_EVALUATE]",
                {
                    "variant":
                        variant,

                    "population":
                        center,

                    "rows":
                        int(
                            mask.sum()
                        ),
                },
                flush=True,
            )

            variant_report[
                "centers"
            ][
                center
            ] = evaluate_group(
                ckpt5,
                ckpt6a,
                ckpt6b,
                endpoints,
                loaded[
                    variant
                ],
                censoring_km,
                mask,
            )

        #######################################################################
        # Pooled DFCI+VICC is secondary only.
        #######################################################################

        print(
            "[CKPT7B2_EVALUATE]",
            {
                "variant":
                    variant,

                "population":
                    "POOLED_SECONDARY",

                "rows":
                    int(
                        evaluable.sum()
                    ),
            },
            flush=True,
        )

        variant_report[
            "pooled_secondary"
        ] = evaluate_group(
            ckpt5,
            ckpt6a,
            ckpt6b,
            endpoints,
            loaded[
                variant
            ],
            censoring_km,
            evaluable,
        )

        report[
            "variants"
        ][
            variant
        ] = variant_report

    ###########################################################################
    # No post-outcome model-selection verdict.
    ###########################################################################

    report[
        "post_outcome_actions"
    ] = {
        "retrained":
            False,

        "recalibrated":
            False,

        "alpha_changed":
            False,

        "endpoint_changed":
            False,

        "prediction_variant_selected_from_outcomes":
            False,

        "original_BPC_outcome_source_reopened":
            False,
    }

    atomic_json(
        OUT
        / "external_validation_results.json",
        report,
    )

    atomic_parquet(
        OUT
        / "external_validation_row_predictions.parquet",
        row_output,
    )

    ###########################################################################
    # Compact primary result.
    ###########################################################################

    compact = {
        "status":
            report[
                "status"
            ],

        "primary_variant":
            {},

        "sensitivity_variant":
            {},

        "pooled_secondary":
            {},
    }

    primary = report[
        "variants"
    ][
        "timeline_sequencing"
    ]

    sensitivity = report[
        "variants"
    ][
        "clinical_sample"
    ]

    for center in CENTERS:

        compact[
            "primary_variant"
        ][
            center
        ] = compact_result(
            primary[
                "centers"
            ][
                center
            ]
        )

        compact[
            "sensitivity_variant"
        ][
            center
        ] = compact_result(
            sensitivity[
                "centers"
            ][
                center
            ]
        )

    compact[
        "pooled_secondary"
    ][
        "timeline_sequencing"
    ] = compact_result(
        primary[
            "pooled_secondary"
        ]
    )

    compact[
        "pooled_secondary"
    ][
        "clinical_sample"
    ] = compact_result(
        sensitivity[
            "pooled_secondary"
        ]
    )

    atomic_json(
        OUT
        / "compact_results.json",
        compact,
    )

    ###########################################################################
    # Human-readable audit.
    ###########################################################################

    lines = [
        "# CKPT7B2 one-shot external validation",
        "",
        "Status: COMPLETE_ONE_SHOT_EXTERNAL_VALIDATION",
        "",
        "No retraining, recalibration, alpha change, endpoint change, "
        "or original BPC outcome-source reopening occurred.",
        "",
        "## Primary: timeline-sequencing genomic timing",
        "",
    ]

    for center in CENTERS:

        r = compact[
            "primary_variant"
        ][
            center
        ]

        lines.extend(
            [
                f"### {center}",
                "",
                f"- evaluable rows: {r['rows']}",
                f"- evaluable patients: {r['patients']}",
                f"- PRE patient NLL: {r['pre_nll']}",
                f"- bounded patient NLL: {r['bounded_nll']}",
                f"- PRE - bounded NLL: {r['nll_gain']}",
                f"- NLL bootstrap: {r['nll_bootstrap']}",
                f"- PRE patient IBS: {r['pre_ibs']}",
                f"- bounded patient IBS: {r['bounded_ibs']}",
                f"- PRE - bounded IBS: {r['ibs_gain']}",
                f"- IBS bootstrap: {r['ibs_bootstrap']}",
                f"- bounded horizon Brier: {r['bounded_horizon_brier']}",
                "",
            ]
        )

    lines.extend(
        [
            "## Predeclared sensitivity: clinical-sample genomic timing",
            "",
        ]
    )

    for center in CENTERS:

        r = compact[
            "sensitivity_variant"
        ][
            center
        ]

        lines.extend(
            [
                f"### {center}",
                "",
                f"- PRE patient NLL: {r['pre_nll']}",
                f"- bounded patient NLL: {r['bounded_nll']}",
                f"- PRE - bounded NLL: {r['nll_gain']}",
                f"- PRE patient IBS: {r['pre_ibs']}",
                f"- bounded patient IBS: {r['bounded_ibs']}",
                f"- PRE - bounded IBS: {r['ibs_gain']}",
                "",
            ]
        )

    (
        OUT
        / "audit.md"
    ).write_text(
        "\n".join(
            lines
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "[CKPT7B2_EXTERNAL_VALIDATION_COMPLETE]",
        compact,
        flush=True,
    )


if __name__ == "__main__":
    main()
