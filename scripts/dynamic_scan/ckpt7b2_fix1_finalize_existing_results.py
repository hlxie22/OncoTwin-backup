#!/usr/bin/env python3

from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


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

EVALUATOR = (
    ROOT
    / "scripts/dynamic_scan/ckpt7b2_external_validation.py"
)

EXPECTED_G1_SHA = (
    "88fe02b0d9011828ed62b1e723cb9ae1649d4aa9af91abe5e27c19caef452f2b"
)

EXPECTED_B1_SHA = (
    "874577fde594d148f3e30a4d4f87027728960699669df0af35ceae380eb9d2ec"
)

HORIZONS = (
    3,
    6,
    12,
    18,
)

VARIANTS = (
    "timeline_sequencing",
    "clinical_sample",
)

CENTERS = (
    "DFCI",
    "VICC",
)


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
            payload,
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


def finite(
    value,
    label,
):

    x = float(value)

    if not math.isfinite(x):
        raise RuntimeError(
            f"Nonfinite value: {label}={value}"
        )

    return x


###############################################################################
# 1. Immutable input hashes.
###############################################################################

g1_sha = sha256_file(
    G1
    / "prediction_freeze_manifest.json"
)

b1_sha = sha256_file(
    B1
    / "external_landmark_endpoints.parquet"
)

if g1_sha != EXPECTED_G1_SHA:
    raise RuntimeError(
        "G1 prediction manifest changed."
    )

if b1_sha != EXPECTED_B1_SHA:
    raise RuntimeError(
        "B1 endpoint artifact changed."
    )


###############################################################################
# 2. Prove completed evaluator had frozen 2000-bootstrap contract.
#
# CKPT6A bootstrap_nll_gain does not itself echo repetitions in its returned
# dictionary, so the previous QC should not require that output key.
###############################################################################

source = EVALUATOR.read_text(
    encoding="utf-8"
)

tree = ast.parse(
    source,
    filename=str(EVALUATOR),
)

bootstrap_constant = None

bootstrap_calls = {
    "bootstrap_nll_gain":
        False,

    "bootstrap_integrated_brier_gain":
        False,
}


for node in ast.walk(tree):

    if isinstance(
        node,
        ast.Assign,
    ):

        for target in node.targets:

            if (
                isinstance(
                    target,
                    ast.Name,
                )
                and target.id
                == "BOOTSTRAP_REPETITIONS"
                and isinstance(
                    node.value,
                    ast.Constant,
                )
            ):
                bootstrap_constant = (
                    node.value.value
                )

    if isinstance(
        node,
        ast.Call,
    ):

        func_name = None

        if isinstance(
            node.func,
            ast.Attribute,
        ):
            func_name = node.func.attr

        elif isinstance(
            node.func,
            ast.Name,
        ):
            func_name = node.func.id

        if func_name in bootstrap_calls:

            repetitions_keyword = None

            for keyword in node.keywords:

                if keyword.arg == "repetitions":
                    repetitions_keyword = (
                        keyword.value
                    )

            if (
                isinstance(
                    repetitions_keyword,
                    ast.Name,
                )
                and repetitions_keyword.id
                == "BOOTSTRAP_REPETITIONS"
            ):
                bootstrap_calls[
                    func_name
                ] = True


if bootstrap_constant != 2000:

    raise RuntimeError(
        "Completed evaluator did not freeze BOOTSTRAP_REPETITIONS=2000."
    )

if not all(
    bootstrap_calls.values()
):

    raise RuntimeError(
        f"Completed evaluator bootstrap wiring changed: {bootstrap_calls}"
    )


###############################################################################
# 3. Load only already-computed frozen outputs.
###############################################################################

report = json.loads(
    (
        OUT
        / "external_validation_results.json"
    ).read_text(
        encoding="utf-8"
    )
)

compact = json.loads(
    (
        OUT
        / "compact_results.json"
    ).read_text(
        encoding="utf-8"
    )
)

row_output = pd.read_parquet(
    OUT
    / "external_validation_row_predictions.parquet"
)

endpoints = pd.read_parquet(
    B1
    / "external_landmark_endpoints.parquet"
)


###############################################################################
# 4. Global protocol/result gates.
###############################################################################

if (
    report.get("status")
    != "COMPLETE_ONE_SHOT_EXTERNAL_VALIDATION"
):

    raise RuntimeError(
        f"Unexpected evaluator status: {report.get('status')}"
    )

if (
    report.get("candidate")
    != "CKPT7R1_R2_R3B_ALPHA_0_5"
):

    raise RuntimeError(
        "Candidate changed."
    )

if float(
    report.get(
        "alpha"
    )
) != 0.5:

    raise RuntimeError(
        "Alpha changed."
    )

if (
    report.get(
        "prediction_manifest_sha256"
    )
    != EXPECTED_G1_SHA
):

    raise RuntimeError(
        "Report prediction-manifest SHA mismatch."
    )

if (
    report.get(
        "endpoint_sha256"
    )
    != EXPECTED_B1_SHA
):

    raise RuntimeError(
        "Report endpoint SHA mismatch."
    )

if (
    report.get(
        "primary_variant"
    )
    != "timeline_sequencing"
):

    raise RuntimeError(
        "Primary variant changed."
    )

if (
    report.get(
        "sensitivity_variant"
    )
    != "clinical_sample"
):

    raise RuntimeError(
        "Sensitivity variant changed."
    )

if int(
    report.get(
        "bootstrap_repetitions"
    )
) != 2000:

    raise RuntimeError(
        "Report bootstrap count changed."
    )


post = report.get(
    "post_outcome_actions",
    {}
)

for key in (
    "retrained",
    "recalibrated",
    "alpha_changed",
    "endpoint_changed",
    "prediction_variant_selected_from_outcomes",
    "original_BPC_outcome_source_reopened",
):

    ###########################################################################
    # The completed CKPT7B2 evaluator's json_safe() encoded bool False as
    # numeric 0 because its integer branch precedes its bool branch.
    #
    # Accept only semantically false frozen encodings. True/1, None, strings,
    # or any other value still fail.
    ###########################################################################

    if post.get(key) not in (
        False,
        0,
        0.0,
    ):

        raise RuntimeError(
            f"Forbidden post-outcome action flag: {key}={post.get(key)}"
        )


###############################################################################
# 5. Endpoint/evaluation-row exact identity.
###############################################################################

evaluable = (
    endpoints[
        endpoints[
            "evaluable"
        ].astype(bool)
    ]
    .copy()
    .reset_index(drop=True)
)

if len(evaluable) != 1558:

    raise RuntimeError(
        f"Evaluable endpoint rows changed: {len(evaluable)}"
    )

if len(row_output) != 1558:

    raise RuntimeError(
        f"Evaluator row-output rows changed: {len(row_output)}"
    )

if not evaluable[
    "external_landmark_row"
].is_unique:

    raise RuntimeError(
        "Frozen evaluable endpoint keys are nonunique."
    )

if not row_output[
    "external_landmark_row"
].is_unique:

    raise RuntimeError(
        "Evaluator row-output keys are nonunique."
    )

if not np.array_equal(
    evaluable[
        "external_landmark_row"
    ].to_numpy(
        dtype=int
    ),
    row_output[
        "external_landmark_row"
    ].to_numpy(
        dtype=int
    ),
):

    raise RuntimeError(
        "Evaluator row output no longer exactly follows frozen endpoint order."
    )

for column in (
    "patient_id",
    "site",
    "bpc_episode_id",
):

    if not np.array_equal(
        evaluable[
            column
        ].astype(str).to_numpy(),
        row_output[
            column
        ].astype(str).to_numpy(),
    ):

        raise RuntimeError(
            f"Evaluator row identity mismatch: {column}"
        )


expected_site_rows = {
    "DFCI":
        1099,

    "VICC":
        459,
}

expected_site_patients = {
    "DFCI":
        220,

    "VICC":
        105,
}

site_rows = (
    row_output[
        "site"
    ]
    .value_counts()
    .to_dict()
)

if site_rows != expected_site_rows:

    raise RuntimeError(
        f"Evaluation site rows changed: {site_rows}"
    )


###############################################################################
# 6. Exact row-level PFS replay against immutable G1 predictions.
###############################################################################

pfs_file_names = {
    "pre":
        "pre_pfs_survival_f32.npy",

    "full_post":
        "full_post_pfs_survival_f32.npy",

    "bounded_post":
        "bounded_post_pfs_survival_f32.npy",
}


pfs_replay = {}

for variant in VARIANTS:

    variant_index = pd.read_parquet(
        G1
        / variant
        / "canonical_prediction_index.parquet"
    )

    if not np.array_equal(
        endpoints[
            "external_landmark_row"
        ].to_numpy(
            dtype=int
        ),
        variant_index[
            "external_landmark_row"
        ].to_numpy(
            dtype=int
        ),
    ):

        raise RuntimeError(
            f"{variant}: G1 index no longer matches endpoint order."
        )

    eval_mask = (
        endpoints[
            "evaluable"
        ].astype(bool)
        .to_numpy()
    )

    pfs_replay[
        variant
    ] = {}

    for mode, filename in pfs_file_names.items():

        frozen = np.load(
            G1
            / variant
            / filename,
            mmap_mode="r",
        )

        if frozen.shape != (
            2532,
            24,
        ):

            raise RuntimeError(
                f"{variant}/{mode}: bad frozen PFS shape {frozen.shape}"
            )

        eval_frozen = np.asarray(
            frozen[
                eval_mask
            ],
            dtype=float,
        )

        max_error = 0.0

        for horizon in HORIZONS:

            column = (
                f"{variant}__{mode}"
                f"__pfs_{horizon}m"
            )

            if column not in row_output.columns:

                raise RuntimeError(
                    f"Missing row-output column: {column}"
                )

            expected = (
                eval_frozen[
                    :,
                    horizon
                    - 1
                ]
            )

            observed = row_output[
                column
            ].to_numpy(
                dtype=float
            )

            error = float(
                np.max(
                    np.abs(
                        observed
                        - expected
                    )
                )
            )

            max_error = max(
                max_error,
                error,
            )

        if max_error > 2e-7:

            raise RuntimeError(
                f"{variant}/{mode}: frozen PFS replay error={max_error}"
            )

        pfs_replay[
            variant
        ][
            mode
        ] = max_error


###############################################################################
# 7. Result-structure / finite-metric QC.
###############################################################################

metric_qc = {}

for variant in VARIANTS:

    if variant not in report[
        "variants"
    ]:

        raise RuntimeError(
            f"Missing report variant: {variant}"
        )

    variant_report = report[
        "variants"
    ][
        variant
    ]

    metric_qc[
        variant
    ] = {}

    for center in CENTERS:

        result = variant_report[
            "centers"
        ][
            center
        ]

        if int(
            result[
                "rows"
            ]
        ) != expected_site_rows[
            center
        ]:

            raise RuntimeError(
                f"{variant}/{center}: row count mismatch."
            )

        if int(
            result[
                "patients"
            ]
        ) != expected_site_patients[
            center
        ]:

            raise RuntimeError(
                f"{variant}/{center}: patient count mismatch."
            )

        pre_nll = finite(
            result[
                "pre"
            ][
                "patient_mean_nll"
            ],
            f"{variant}/{center}/pre_nll",
        )

        bounded_nll = finite(
            result[
                "bounded_post"
            ][
                "patient_mean_nll"
            ],
            f"{variant}/{center}/bounded_nll",
        )

        pre_ibs = finite(
            result[
                "pre"
            ][
                "patient_integrated_brier_4h"
            ],
            f"{variant}/{center}/pre_ibs",
        )

        bounded_ibs = finite(
            result[
                "bounded_post"
            ][
                "patient_integrated_brier_4h"
            ],
            f"{variant}/{center}/bounded_ibs",
        )

        nll_gain = finite(
            result[
                "pre_minus_bounded"
            ][
                "patient_nll"
            ],
            f"{variant}/{center}/nll_gain",
        )

        ibs_gain = finite(
            result[
                "pre_minus_bounded"
            ][
                "patient_ibs"
            ],
            f"{variant}/{center}/ibs_gain",
        )

        if abs(
            nll_gain
            - (
                pre_nll
                - bounded_nll
            )
        ) > 1e-12:

            raise RuntimeError(
                f"{variant}/{center}: NLL gain arithmetic mismatch."
            )

        if abs(
            ibs_gain
            - (
                pre_ibs
                - bounded_ibs
            )
        ) > 1e-12:

            raise RuntimeError(
                f"{variant}/{center}: IBS gain arithmetic mismatch."
            )

        for mode in (
            "pre",
            "bounded_post",
        ):

            horizon_briers = []

            for horizon in HORIZONS:

                brier = finite(
                    result[
                        mode
                    ][
                        "patient_balanced_pfs"
                    ][
                        f"{horizon}m"
                    ][
                        "brier"
                    ],
                    (
                        f"{variant}/{center}/"
                        f"{mode}/{horizon}m_brier"
                    ),
                )

                horizon_briers.append(
                    brier
                )

            integrated = finite(
                result[
                    mode
                ][
                    "patient_integrated_brier_4h"
                ],
                f"{variant}/{center}/{mode}/IBS",
            )

            replay = float(
                np.mean(
                    horizon_briers
                )
            )

            if abs(
                replay
                - integrated
            ) > 1e-12:

                raise RuntimeError(
                    f"{variant}/{center}/{mode}: "
                    "four-horizon IBS arithmetic mismatch."
                )

        nll_boot = result[
            "bootstrap_2000"
        ][
            "pre_minus_bounded_patient_nll"
        ]

        ibs_boot = result[
            "bootstrap_2000"
        ][
            "pre_minus_bounded_patient_ibs"
        ]

        for name, boot in (
            (
                "nll",
                nll_boot,
            ),
            (
                "ibs",
                ibs_boot,
            ),
        ):

            mean = finite(
                boot[
                    "mean"
                ],
                f"{variant}/{center}/{name}/bootstrap_mean",
            )

            low = finite(
                boot[
                    "ci_low"
                ],
                f"{variant}/{center}/{name}/bootstrap_low",
            )

            high = finite(
                boot[
                    "ci_high"
                ],
                f"{variant}/{center}/{name}/bootstrap_high",
            )

            if low > mean or mean > high:

                raise RuntimeError(
                    f"{variant}/{center}/{name}: invalid bootstrap interval."
                )

            if int(
                boot[
                    "patients"
                ]
            ) < 20:

                raise RuntimeError(
                    f"{variant}/{center}/{name}: too few bootstrap patients."
                )

        #######################################################################
        # CKPT6B's IBS helper records repetitions explicitly.
        #######################################################################

        if int(
            ibs_boot[
                "repetitions"
            ]
        ) != 2000:

            raise RuntimeError(
                f"{variant}/{center}: IBS bootstrap repetitions != 2000"
            )

        #######################################################################
        # CKPT6A's NLL helper does NOT include repetitions in its returned
        # dictionary. The AST gate above proves the completed evaluator passed
        # BOOTSTRAP_REPETITIONS=2000 into that call.
        #######################################################################

        metric_qc[
            variant
        ][
            center
        ] = {
            "rows":
                int(
                    result[
                        "rows"
                    ]
                ),

            "patients":
                int(
                    result[
                        "patients"
                    ]
                ),

            "pre_nll":
                pre_nll,

            "bounded_nll":
                bounded_nll,

            "nll_gain":
                nll_gain,

            "nll_ci":
                [
                    float(
                        nll_boot[
                            "ci_low"
                        ]
                    ),
                    float(
                        nll_boot[
                            "ci_high"
                        ]
                    ),
                ],

            "pre_ibs":
                pre_ibs,

            "bounded_ibs":
                bounded_ibs,

            "ibs_gain":
                ibs_gain,

            "ibs_ci":
                [
                    float(
                        ibs_boot[
                            "ci_low"
                        ]
                    ),
                    float(
                        ibs_boot[
                            "ci_high"
                        ]
                    ),
                ],

            "nll_bootstrap_repetitions":
                2000,

            "ibs_bootstrap_repetitions":
                2000,
        }


###############################################################################
# 8. Pooled-secondary structure.
###############################################################################

pooled_qc = {}

for variant in VARIANTS:

    pooled = report[
        "variants"
    ][
        variant
    ][
        "pooled_secondary"
    ]

    if int(
        pooled[
            "rows"
        ]
    ) != 1558:

        raise RuntimeError(
            f"{variant}: pooled rows != 1558"
        )

    if int(
        pooled[
            "patients"
        ]
    ) != 325:

        raise RuntimeError(
            f"{variant}: pooled patients != 325"
        )

    pooled_qc[
        variant
    ] = {
        "rows":
            1558,

        "patients":
            325,

        "nll_gain":
            finite(
                pooled[
                    "pre_minus_bounded"
                ][
                    "patient_nll"
                ],
                f"{variant}/pooled/nll_gain",
            ),

        "nll_ci":
            [
                finite(
                    pooled[
                        "bootstrap_2000"
                    ][
                        "pre_minus_bounded_patient_nll"
                    ][
                        "ci_low"
                    ],
                    f"{variant}/pooled/nll_low",
                ),
                finite(
                    pooled[
                        "bootstrap_2000"
                    ][
                        "pre_minus_bounded_patient_nll"
                    ][
                        "ci_high"
                    ],
                    f"{variant}/pooled/nll_high",
                ),
            ],

        "ibs_gain":
            finite(
                pooled[
                    "pre_minus_bounded"
                ][
                    "patient_ibs"
                ],
                f"{variant}/pooled/ibs_gain",
            ),

        "ibs_ci":
            [
                finite(
                    pooled[
                        "bootstrap_2000"
                    ][
                        "pre_minus_bounded_patient_ibs"
                    ][
                        "ci_low"
                    ],
                    f"{variant}/pooled/ibs_low",
                ),
                finite(
                    pooled[
                        "bootstrap_2000"
                    ][
                        "pre_minus_bounded_patient_ibs"
                    ][
                        "ci_high"
                    ],
                    f"{variant}/pooled/ibs_high",
                ),
            ],
    }


###############################################################################
# 9. Compact-results replay against full report.
###############################################################################

for center in CENTERS:

    c = compact[
        "primary_variant"
    ][
        center
    ]

    r = metric_qc[
        "timeline_sequencing"
    ][
        center
    ]

    comparisons = (
        (
            c[
                "pre_nll"
            ],
            r[
                "pre_nll"
            ],
            "pre_nll",
        ),
        (
            c[
                "bounded_nll"
            ],
            r[
                "bounded_nll"
            ],
            "bounded_nll",
        ),
        (
            c[
                "nll_gain"
            ],
            r[
                "nll_gain"
            ],
            "nll_gain",
        ),
        (
            c[
                "pre_ibs"
            ],
            r[
                "pre_ibs"
            ],
            "pre_ibs",
        ),
        (
            c[
                "bounded_ibs"
            ],
            r[
                "bounded_ibs"
            ],
            "bounded_ibs",
        ),
        (
            c[
                "ibs_gain"
            ],
            r[
                "ibs_gain"
            ],
            "ibs_gain",
        ),
    )

    for observed, expected, label in comparisons:

        if abs(
            float(observed)
            - float(expected)
        ) > 1e-12:

            raise RuntimeError(
                f"Compact/full mismatch {center}/{label}"
            )


###############################################################################
# 10. Explicit result interpretation WITHOUT changing protocol.
###############################################################################

primary_dfci = metric_qc[
    "timeline_sequencing"
][
    "DFCI"
]

primary_vicc = metric_qc[
    "timeline_sequencing"
][
    "VICC"
]

sensitivity_dfci = metric_qc[
    "clinical_sample"
][
    "DFCI"
]

sensitivity_vicc = metric_qc[
    "clinical_sample"
][
    "VICC"
]


interpretation = {
    "primary_metric":
        "patient-balanced PFS IBS",

    "DFCI_primary_IBS": {
        "gain":
            primary_dfci[
                "ibs_gain"
            ],

        "ci":
            primary_dfci[
                "ibs_ci"
            ],

        "interval_excludes_zero":
            (
                primary_dfci[
                    "ibs_ci"
                ][
                    0
                ]
                > 0
                or primary_dfci[
                    "ibs_ci"
                ][
                    1
                ]
                < 0
            ),
    },

    "VICC_primary_IBS": {
        "gain":
            primary_vicc[
                "ibs_gain"
            ],

        "ci":
            primary_vicc[
                "ibs_ci"
            ],

        "interval_excludes_zero":
            (
                primary_vicc[
                    "ibs_ci"
                ][
                    0
                ]
                > 0
                or primary_vicc[
                    "ibs_ci"
                ][
                    1
                ]
                < 0
            ),
    },

    "DFCI_complementary_NLL": {
        "gain":
            primary_dfci[
                "nll_gain"
            ],

        "ci":
            primary_dfci[
                "nll_ci"
            ],

        "positive_interval":
            (
                primary_dfci[
                    "nll_ci"
                ][
                    0
                ]
                > 0
            ),
    },

    "VICC_complementary_NLL": {
        "gain":
            primary_vicc[
                "nll_gain"
            ],

        "ci":
            primary_vicc[
                "nll_ci"
            ],

        "positive_interval":
            (
                primary_vicc[
                    "nll_ci"
                ][
                    0
                ]
                > 0
            ),
    },

    "timing_sensitivity": {
        "DFCI_NLL_gain":
            sensitivity_dfci[
                "nll_gain"
            ],

        "DFCI_IBS_gain":
            sensitivity_dfci[
                "ibs_gain"
            ],

        "VICC_NLL_gain":
            sensitivity_vicc[
                "nll_gain"
            ],

        "VICC_IBS_gain":
            sensitivity_vicc[
                "ibs_gain"
            ],
    },

    "protocol_changes_after_results":
        False,

    "retuning_permitted":
        False,
}


###############################################################################
# 11. Write repaired posthoc QC.
###############################################################################

qc = {
    "status":
        "PASS_CKPT7B2_EXISTING_RESULTS_POSTHOC_QC",

    "science_status":
        "COMPLETE_ONE_SHOT_EXTERNAL_VALIDATION",

    "science_recomputed":
        False,

    "bootstrap_recomputed":
        False,

    "model_inference_recomputed":
        False,

    "endpoint_recomputed":
        False,

    "original_BPC_outcome_source_reopened":
        False,

    "prediction_manifest_sha256":
        g1_sha,

    "endpoint_sha256":
        b1_sha,

    "evaluator_sha256":
        sha256_file(
            EVALUATOR
        ),

    "bootstrap_contract": {
        "repetitions":
            2000,

        "nll_repetition_proof":
            (
                "AST: completed evaluator calls bootstrap_nll_gain "
                "with BOOTSTRAP_REPETITIONS=2000"
            ),

        "ibs_repetition_proof":
            (
                "AST plus stored IBS result repetitions=2000"
            ),
    },

    "evaluation_population": {
        "rows":
            1558,

        "patients":
            325,

        "DFCI_rows":
            1099,

        "DFCI_patients":
            220,

        "VICC_rows":
            459,

        "VICC_patients":
            105,
    },

    "pfs_exact_replay":
        pfs_replay,

    "center_metrics":
        metric_qc,

    "pooled_secondary":
        pooled_qc,

    "interpretation":
        interpretation,

    "protocol_changed_after_results":
        False,

    "frozen_report_boolean_serialization": {
        "issue":
            (
                "Completed CKPT7B2 json_safe encoded Python bool values "
                "through its integer branch, so False is stored as JSON 0."
            ),

        "interpretation":
            "0 is accepted only as semantic False for post-outcome action flags",

        "science_affected":
            False,

        "science_recomputed":
            False,
    },
}

atomic_json(
    OUT
    / "posthoc_qc.json",
    qc,
)


###############################################################################
# 12. Final decision artifact.
###############################################################################

decision = {
    "status":
        "COMPLETE_EXTERNAL_VALIDATION_FROZEN_RESULTS",

    "candidate":
        "CKPT7R1_R2_R3B_ALPHA_0_5",

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
        "SECONDARY_ONLY",

    "primary_metric":
        "patient-balanced PFS integrated Brier at 3/6/12/18 months",

    "primary_results": {
        "DFCI":
            primary_dfci,

        "VICC":
            primary_vicc,
    },

    "predeclared_timing_sensitivity": {
        "DFCI":
            sensitivity_dfci,

        "VICC":
            sensitivity_vicc,
    },

    "pooled_secondary":
        pooled_qc,

    "scientific_summary": {
        "primary_IBS":
            (
                "Bounded POST has a favorable point estimate in both "
                "DFCI and VICC under primary timeline sequencing, but "
                "both 95% patient-bootstrap intervals include zero."
            ),

        "complementary_NLL":
            (
                "VICC shows a positive bounded-update likelihood gain "
                "with its 95% patient-bootstrap interval above zero. "
                "DFCI has a small positive point estimate with an "
                "interval including zero."
            ),

        "timing_sensitivity":
            (
                "VICC likelihood improvement persists under the "
                "clinical-sample genomic timing sensitivity. DFCI "
                "shows small unfavorable sensitivity point estimates, "
                "with uncertainty intervals including zero."
            ),

        "replication_statement":
            (
                "A primary PFS-IBS scan-update gain is not established "
                "independently in both external centers."
            ),
    },

    "post_outcome_retuning":
        "PROHIBITED",

    "next_action":
        (
            "Report the frozen external results and move to final synthesis/"
            "manuscript analysis. Do not retrain, recalibrate, alter alpha, "
            "change endpoint mapping, or select genomic timing based on these "
            "external outcomes."
        ),
}

atomic_json(
    OUT
    / "final_decision.json",
    decision,
)


print(
    "[CKPT7B2_FIX1_POSTHOC_QC_PASS]",
    {
        "science_status":
            qc[
                "science_status"
            ],

        "science_recomputed":
            False,

        "bootstrap_recomputed":
            False,

        "original_BPC_outcome_source_reopened":
            False,

        "DFCI_primary":
            primary_dfci,

        "VICC_primary":
            primary_vicc,
    },
)


if not (
    primary_vicc[
        "nll_ci"
    ][
        0
    ]
    > 0
):

    raise RuntimeError(
        "Frozen VICC NLL interval no longer matches completed-run result."
    )


if (
    primary_dfci[
        "ibs_ci"
    ][
        0
    ]
    > 0
    or primary_dfci[
        "ibs_ci"
    ][
        1
    ]
    < 0
):

    raise RuntimeError(
        "Frozen DFCI IBS interval no longer matches completed-run result."
    )


if (
    primary_vicc[
        "ibs_ci"
    ][
        0
    ]
    > 0
    or primary_vicc[
        "ibs_ci"
    ][
        1
    ]
    < 0
):

    raise RuntimeError(
        "Frozen VICC IBS interval no longer matches completed-run result."
    )


print(
    "[CKPT7B2_FIX1_SCIENCE_IDENTITY_PASS]",
    {
        "primary_IBS_established_DFCI":
            False,

        "primary_IBS_established_VICC":
            False,

        "VICC_complementary_NLL_positive_CI":
            True,

        "retuning":
            False,
    },
)
