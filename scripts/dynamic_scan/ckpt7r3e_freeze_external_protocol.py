#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]

OUT = (
    ROOT
    / "artifacts/checkpoint7r3e_transport_safe_external_protocol"
)

R1 = (
    ROOT
    / "artifacts/checkpoint7r1_line_agnostic_temporal"
)

R2 = (
    ROOT
    / "artifacts/checkpoint7r2_transport_safe_supervised"
)

R3B = (
    ROOT
    / "artifacts/checkpoint7r3b_bounded_candidate"
)

R3C = (
    ROOT
    / "artifacts/checkpoint7r3c_update_specificity"
)

R3D2 = (
    ROOT
    / "artifacts/checkpoint7r3d2_window_robustness"
)

CKPT3 = (
    ROOT
    / "artifacts/checkpoint3"
)


def read_json(path: Path):

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def write_json_atomic(
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
        ),
        encoding="utf-8",
    )

    json.loads(
        tmp.read_text(
            encoding="utf-8"
        )
    )

    tmp.replace(path)


def sha256_file(path: Path):

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


def file_record(path: Path):

    if (
        not path.exists()
        or not path.is_file()
        or path.stat().st_size == 0
    ):

        raise RuntimeError(
            f"Missing immutable candidate file: {path}"
        )

    return {
        "path":
            str(
                path.relative_to(
                    ROOT
                )
            ),

        "bytes":
            int(
                path.stat().st_size
            ),

        "sha256":
            sha256_file(
                path
            ),
    }


def main():

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    ###########################################################################
    # Reconfirm repaired candidate
    ###########################################################################

    r1_qc = read_json(
        R1
        / "repair_qc.json"
    )

    r3b_qc = read_json(
        R3B
        / "freeze_qc.json"
    )

    r3c = read_json(
        R3C
        / "decision.json"
    )

    r3d2 = read_json(
        R3D2
        / "decision.json"
    )

    candidate = torch.load(
        R3B
        / "bounded_dynamic_scan_candidate.pt",
        map_location="cpu",
        weights_only=False,
    )

    context = np.load(
        R2
        / "prepared/context_features_f32.npy",
        mmap_mode="r",
    )

    scan = np.load(
        R2
        / "prepared/current_scan_features_f32.npy",
        mmap_mode="r",
    )

    tumor = np.load(
        R2
        / "prepared/tumor_embeddings_f16.npy",
        mmap_mode="r",
    )

    scan_index = pd.read_parquet(
        R2
        / "prepared/scan_index.parquet"
    )

    if (
        r1_qc[
            "trained_line_invariance"
        ][
            "max_abs_difference"
        ]
        != 0.0
    ):

        raise RuntimeError(
            "Temporal encoder is not exactly line invariant."
        )

    if r3b_qc["frozen_alpha"] != 0.5:

        raise RuntimeError(
            "R3B alpha is not 0.5."
        )

    if float(
        candidate[
            "ckpt6b_bounded_update"
        ][
            "alpha"
        ]
    ) != 0.5:

        raise RuntimeError(
            "Candidate checkpoint alpha is not 0.5."
        )

    if (
        r3c["ckpt6c_status"]
        != "PASS_INTERNAL_DYNAMIC_CANDIDATE_LOCK"
    ):

        raise RuntimeError(
            "R3C candidate specificity lock missing."
        )

    if (
        r3d2[
            "window_robustness_status"
        ]
        != "PASS_WINDOW_GROUPING_ROBUSTNESS"
    ):

        raise RuntimeError(
            "R3D2 window robustness did not pass."
        )

    if not (
        r3d2[
            "candidate_replay_pass"
        ]
        and r3d2[
            "metric_replay_pass"
        ]
    ):

        raise RuntimeError(
            "R3D2 W3 replay lock incomplete."
        )

    if r3d2[
        "external_outcomes_opened"
    ]:

        raise RuntimeError(
            "External outcomes were marked opened."
        )

    if context.shape != (
        len(
            scan_index
        ),
        6,
    ):

        raise RuntimeError(
            f"Unexpected context shape: {context.shape}"
        )

    if scan.shape != (
        len(
            scan_index
        ),
        18,
    ):

        raise RuntimeError(
            f"Unexpected scan feature shape: {scan.shape}"
        )

    if tumor.shape != (
        len(
            scan_index
        ),
        128,
    ):

        raise RuntimeError(
            f"Unexpected tumor embedding shape: {tumor.shape}"
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
            "Line-derived context coordinates are not exact zero."
        )

    ###########################################################################
    # Immutable candidate identity
    ###########################################################################

    candidate_files = [
        CKPT3
        / "genomic_encoder.pt",

        R1
        / "temporal_encoder.pt",

        R1
        / "breast_scan_prepost_embeddings_f16.npy",

        R1
        / "breast_scan_prepost_index.parquet",

        R2
        / "dynamic_scan_model.pt",

        R2
        / "prepared/scan_index.parquet",

        R2
        / "prepared/context_features_f32.npy",

        R2
        / "prepared/current_scan_features_f32.npy",

        R2
        / "prepared/tumor_embeddings_f16.npy",

        R3B
        / "bounded_dynamic_scan_candidate.pt",

        R3B
        / "freeze_qc.json",

        R3C
        / "decision.json",

        R3D2
        / "decision.json",

        R3D2
        / "window_robustness.json",

        R3D2
        / "w3_prediction_replay.json",

        R3D2
        / "w3_metric_replay.json",
    ]

    candidate_manifest = {
        "status":
            "IMMUTABLE_TRANSPORT_SAFE_CANDIDATE_LOCK",

        "candidate_name":
            "CKPT7R1_R2_R3B_ALPHA_0_5",

        "alpha":
            0.5,

        "alpha_domain":
            "LOGIT",

        "bounded_update_formula":
            (
                "bounded_post_logits = "
                "pre_logits + 0.5 * "
                "(full_post_logits - pre_logits)"
            ),

        "genomic_dimension":
            128,

        "temporal_dimension":
            192,

        "scan_feature_dimension":
            18,

        "context_dimension":
            6,

        "months":
            24,

        "line_agnostic_temporal":
            True,

        "line_context_zero_indices": [
            0,
            1,
            3,
        ],

        "line_context_zero_names": [
            "line_number_scaled",
            "elapsed_on_line_scaled",
            "scan_number_line_log",
        ],

        "preserved_context_indices": [
            2,
            4,
            5,
        ],

        "preserved_context_names": [
            "scan_number_patient_log",
            "genomic_available",
            "genomic_age_scaled",
        ],

        "internal_specificity_status":
            r3c[
                "ckpt6c_status"
            ],

        "internal_window_robustness_status":
            r3d2[
                "window_robustness_status"
            ],

        "candidate_replay_locked":
            True,

        "model_selection_complete":
            True,

        "files": [
            file_record(
                path
            )
            for path
            in candidate_files
        ],

        "external_outcomes_opened":
            False,
    }

    write_json_atomic(
        OUT
        / "candidate_manifest.json",
        candidate_manifest,
    )

    ###########################################################################
    # External predictor adapter contract
    ###########################################################################

    scan_feature_names = [
        "state_non_progressive",
        "state_indeterminate",
        "state_progressive",
        "raw_coverage_chest",
        "raw_coverage_abdomen",
        "raw_coverage_pelvis",
        "raw_coverage_head",
        "raw_coverage_other",
        "modality_ct",
        "modality_pet",
        "modality_mr",
        "modality_bone_scan",
        "site_bone",
        "site_liver",
        "site_lung",
        "site_brain",
        "site_lymph",
        "site_pleura",
    ]

    adapter_contract = {
        "status":
            "EXTERNAL_ADAPTER_CONTRACT_FROZEN",

        "primary_scan_window":
            "W3",

        "landmark_day":
            "scan_episode_end_day",

        "temporal_encoder":
            "CKPT7R1_LINE_AGNOSTIC",

        "supervised_model":
            "CKPT7R2_TRANSPORT_SAFE",

        "bounded_alpha":
            0.5,

        "absolute_treatment_line_required":
            False,

        "BPC_line_reconstruction":
            "PROHIBITED",

        "temporal_history":
            {
                "persists_across_treatment_lines":
                    True,

                "pre":
                    (
                        "strictly prior clinically available history "
                        "before current scan"
                    ),

                "post":
                    (
                        "PRE plus exactly current scan episode"
                    ),

                "same_day_unrelated_events":
                    "EXCLUDED_FROM_POST",

                "semantic_line_id_model_input":
                    False,
            },

        "context": {
            "dimension":
                6,

            "force_exact_zero": {
                "indices":
                    [
                        0,
                        1,
                        3,
                    ],

                "names":
                    [
                        "line_number_scaled",
                        "elapsed_on_line_scaled",
                        "scan_number_line_log",
                    ],
            },

            "recompute": {
                "2_scan_number_patient_log":
                    (
                        "log1p(chronological patient scan number) / 5"
                    ),

                "4_genomic_available":
                    (
                        "1 iff latest usable genomic sample has "
                        "availability_day < landmark_day; else 0"
                    ),

                "5_genomic_age_scaled":
                    (
                        "if genomic_available=1: "
                        "clip(landmark_day-availability_day,0,3650)/365; "
                        "else 0"
                    ),
            },
        },

        "genomics": {
            "dimension":
                128,

            "primary_availability_source":
                (
                    "predictor-only timeline sequencing START_DATE"
                ),

            "timing_sensitivity":
                (
                    "clinical CPT_SEQ_DATE"
                ),

            "availability_rule":
                "availability_day < landmark_day",

            "unavailable_embedding":
                "exact_zero_128",

            "unavailable_genomic_available":
                0,

            "unavailable_genomic_age_scaled":
                0.0,

            "no_external_timing_tuning":
                True,
        },

        "current_scan_state_crosswalk": {
            "Progressing":
                "PROGRESSIVE",

            "Stable":
                "NON_PROGRESSIVE",

            "Improving":
                "NON_PROGRESSIVE",

            "Mixed":
                "INDETERMINATE",

            "Not stated":
                "INDETERMINATE",

            "blank":
                "UNOBSERVED",
        },

        "current_landmark_primary_eligibility":
            "NON_PROGRESSIVE_ONLY",

        "history_state_policy": {
            "NON_PROGRESSIVE":
                "retain",

            "PROGRESSIVE":
                "retain_in_history_but_never_primary_residual_PFS_landmark",

            "INDETERMINATE":
                "retain_in_history_but_not_primary_current_landmark",

            "UNOBSERVED":
                "retain_in_history_but_not_primary_current_landmark",
        },

        "scan_features": {
            "dimension":
                18,

            "names":
                scan_feature_names,

            "state_channels_0_2":
                (
                    "from frozen CURATED_CANCER_STATUS crosswalk"
                ),

            "coverage_channels_3_7":
                (
                    "may use external scan-region coverage metadata "
                    "including SCAN_SITES strictly as observation coverage"
                ),

            "coverage_semantics":
                (
                    "coverage/unreported is not tumor negativity"
                ),

            "modality_channels_8_11":
                "FORCE_ZERO_TO_MATCH_CHORD_INTERFACE",

            "disease_site_channels_12_17":
                "PENDING_TRANSPORT_GATE",

            "SCAN_SITES_as_disease_site_positivity":
                "PROHIBITED",
        },

        "external_center_positive_control":
            {
                "BPC_MSK":
                    (
                        "adapter/schema positive control only; "
                        "never external model selection"
                    ),

                "patient_clock_offsets":
                    (
                        "MSK-only positive-control artifact; "
                        "never transfer to DFCI/VICC"
                    ),
            },

        "external_primary_centers": [
            "DFCI",
            "VICC",
        ],

        "external_primary_center_handling":
            "EVALUATE_SEPARATELY",

        "pooled_external_analysis":
            "SECONDARY_ONLY",

        "external_outcomes_opened":
            False,
    }

    write_json_atomic(
        OUT
        / "external_adapter_contract.json",
        adapter_contract,
    )

    ###########################################################################
    # Outcome lock
    ###########################################################################

    outcome_lock = {
        "status":
            "EXTERNAL_OUTCOME_ACCESS_LOCKED",

        "external_outcomes_opened":
            False,

        "hard_outcome_paths_quarantined": [
            (
                "cBioPortal_files/"
                "data_clinical_supp_survival.txt"
            ),
            (
                "cBioPortal_files/"
                "data_clinical_supp_survival_treatment.txt"
            ),
            (
                "clinical_data/"
                "cancer_level_dataset_index.csv"
            ),
            (
                "clinical_data/"
                "cancer_panel_test_level_dataset.csv"
            ),
            (
                "clinical_data/"
                "patient_level_dataset.csv"
            ),
            (
                "clinical_data/"
                "regimen_cancer_level_dataset.csv"
            ),
        ],

        "rule":
            (
                "Do not read, summarize, inspect distributions from, "
                "join, or derive endpoints from any quarantined external "
                "outcome source until external predictor tensors, PRE/FULL_POST/"
                "bounded logits, row index, and prediction manifest are "
                "fully materialized and SHA256 frozen."
            ),

        "allowed_before_prediction_freeze": [
            "predictor-only imaging metadata",
            "predictor-only diagnosis metadata",
            "predictor-only treatment metadata",
            "predictor-only sequencing timing metadata",
            "panel/assay metadata",
            "BPC-MSK positive-control predictor bridge",
        ],

        "forbidden_before_prediction_freeze": [
            "DFCI outcome distribution inspection",
            "VICC outcome distribution inspection",
            "endpoint prevalence inspection",
            "external calibration",
            "external alpha selection",
            "external feature-rule selection from performance",
            "external IPCW refitting",
            "external model selection",
        ],
    }

    write_json_atomic(
        OUT
        / "outcome_access_lock.json",
        outcome_lock,
    )

    ###########################################################################
    # Statistical external-validation protocol
    ###########################################################################

    protocol = {
        "status":
            "PASS_TRANSPORT_SAFE_EXTERNAL_PROTOCOL_FREEZE",

        "candidate_manifest":
            "candidate_manifest.json",

        "adapter_contract":
            "external_adapter_contract.json",

        "outcome_lock":
            "outcome_access_lock.json",

        "candidate": {
            "temporal_encoder":
                "CKPT7R1",

            "supervised_model":
                "CKPT7R2",

            "bounded_update":
                "CKPT7R3B",

            "alpha":
                0.5,

            "primary_scan_window":
                "W3",
        },

        "endpoint": {
            "primary":
                "PFS",

            "PFS_event":
                [
                    "progression",
                    "death",
                ],

            "treatment_switch":
                "SEPARATE_COMPETING_CAUSE",

            "administrative_censoring":
                "CENSOR",

            "competing_risk_NLL":
                "COMPLEMENTARY_ANALYSIS",
        },

        "metrics": {
            "primary":
                "PATIENT_BALANCED_PFS_IBS",

            "supporting":
                [
                    "PATIENT_BALANCED_SURVIVAL_NLL",
                    "PFS_BRIER_AT_FIXED_HORIZONS",
                ],

            "horizons_months": [
                3,
                6,
                12,
                18,
            ],

            "patient_balanced":
                True,
        },

        "IPCW": {
            "source":
                "CHORD_CKPT7R2_TRAIN_SURVIVAL_ROWS",

            "refit_on_external":
                False,

            "training_rows":
                14000,

            "training_patients":
                2014,

            "training_cause_counts": {
                "censor":
                    3591,

                "progression":
                    7588,

                "death":
                    552,

                "switch":
                    2269,
            },

            "fit_contract":
                (
                    "train AND survival_mask; raw multicause "
                    "survival_cause; switch is not collapsed into censor"
                ),
        },

        "uncertainty": {
            "bootstrap_unit":
                "PATIENT",

            "repetitions":
                2000,

            "seed":
                20260928,

            "confidence_interval":
                0.95,
        },

        "center_analysis": {
            "DFCI":
                "PRIMARY_SEPARATE",

            "VICC":
                "PRIMARY_SEPARATE",

            "DFCI_PLUS_VICC":
                "SECONDARY_POOLED",
        },

        "primary_recalibration":
            "NONE",

        "external_IPCW_refit":
            "NONE",

        "post_outcome_model_changes":
            "PROHIBITED",

        "post_outcome_alpha_changes":
            "PROHIBITED",

        "post_outcome_feature_mapping_changes":
            "PROHIBITED_EXCEPT_DETERMINISTIC_PLUMBING_CORRECTION",

        "deterministic_plumbing_correction_rule":
            (
                "Only corrections necessary to implement the already-frozen "
                "protocol are allowed after outcome access; correction must "
                "be documented and applied identically to DFCI and VICC and "
                "must not be selected based on metric performance."
            ),

        "genomic_timing": {
            "primary":
                "timeline_sequencing_START_DATE",

            "sensitivity":
                "clinical_CPT_SEQ_DATE",

            "no_timing_selection_after_outcomes":
                True,
        },

        "prediction_freeze_requirement": {
            "before_outcome_access":
                [
                    "external row index",
                    "external predictor tensors",
                    "PRE logits",
                    "FULL_POST logits",
                    "bounded alpha=0.5 logits",
                    "PFS curves",
                    "all prediction file SHA256 hashes",
                    "adapter configuration SHA256",
                    "candidate manifest SHA256",
                ],

            "model_rerun_after_outcome_access":
                "PROHIBITED",
        },

        "external_outcomes_opened":
            False,
    }

    write_json_atomic(
        OUT
        / "protocol.json",
        protocol,
    )

    ###########################################################################
    # Remaining predictor-only transport gate.
    #
    # This is intentionally NOT solved using external outcome information.
    ###########################################################################

    pending = {
        "status":
            "PREDICTION_FREEZE_BLOCKED_PENDING_SITE_FEATURE_TRANSPORT_GATE",

        "external_outcomes_opened":
            False,

        "blocking_issue":
            (
                "CKPT7R2 scan feature indices 12:18 encode disease-site "
                "positivity, while BPC SCAN_SITES is scan-region coverage "
                "and must not be substituted."
            ),

        "blocked_feature_indices": [
            12,
            13,
            14,
            15,
            16,
            17,
        ],

        "blocked_feature_names": [
            "site_bone",
            "site_liver",
            "site_lung",
            "site_brain",
            "site_lymph",
            "site_pleura",
        ],

        "prohibited_mapping":
            "SCAN_SITES -> disease-site positivity",

        "next_gate_must_be_outcome_blind":
            True,

        "allowed_resolution_paths": [
            (
                "Identify an external predictor-only disease-site stream "
                "with semantics genuinely equivalent to CKPT7R2."
            ),
            (
                "If no equivalent stream exists, predeclare a missing-site "
                "transport transformation only after CHORD internal "
                "sensitivity analysis and before generating DFCI/VICC "
                "predictions."
            ),
        ],

        "selection_from_DFCI_or_VICC_performance":
            "PROHIBITED",

        "external_prediction_ready":
            False,
    }

    write_json_atomic(
        OUT
        / "pending_predictor_gates.json",
        pending,
    )

    ###########################################################################
    # Final freeze QC
    ###########################################################################

    generated = [
        OUT
        / "candidate_manifest.json",

        OUT
        / "external_adapter_contract.json",

        OUT
        / "outcome_access_lock.json",

        OUT
        / "protocol.json",

        OUT
        / "pending_predictor_gates.json",
    ]

    freeze_qc = {
        "status":
            "PASS_TRANSPORT_SAFE_EXTERNAL_PROTOCOL_FREEZE",

        "candidate_locked":
            True,

        "alpha":
            0.5,

        "primary_window":
            "W3",

        "line_agnostic_temporal":
            True,

        "line_context_zero_indices": [
            0,
            1,
            3,
        ],

        "external_protocol_locked":
            True,

        "external_prediction_ready":
            False,

        "remaining_blocker":
            "DISEASE_SITE_FEATURE_TRANSPORT_GATE",

        "external_outcomes_opened":
            False,

        "generated_files": [
            file_record(
                path
            )
            for path
            in generated
        ],
    }

    write_json_atomic(
        OUT
        / "freeze_qc.json",
        freeze_qc,
    )

    ###########################################################################
    # Manifest including itself only after all prior files are stable.
    ###########################################################################

    manifest_files = [
        OUT
        / "candidate_manifest.json",

        OUT
        / "external_adapter_contract.json",

        OUT
        / "outcome_access_lock.json",

        OUT
        / "protocol.json",

        OUT
        / "pending_predictor_gates.json",

        OUT
        / "freeze_qc.json",
    ]

    manifest = {
        "status":
            "PASS_TRANSPORT_SAFE_EXTERNAL_PROTOCOL_FREEZE",

        "candidate":
            "CKPT7R1_R2_R3B_ALPHA_0_5",

        "primary_window":
            "W3",

        "external_prediction_ready":
            False,

        "external_outcomes_opened":
            False,

        "files": [
            file_record(
                path
            )
            for path
            in manifest_files
        ],
    }

    write_json_atomic(
        OUT
        / "manifest.json",
        manifest,
    )

    print(
        "[CKPT7R3E_PROTOCOL_FREEZE_PASS]",
        {
            "status":
                protocol[
                    "status"
                ],

            "candidate":
                candidate_manifest[
                    "candidate_name"
                ],

            "alpha":
                candidate_manifest[
                    "alpha"
                ],

            "primary_window":
                protocol[
                    "candidate"
                ][
                    "primary_scan_window"
                ],

            "bootstrap_repetitions":
                protocol[
                    "uncertainty"
                ][
                    "repetitions"
                ],

            "line_context_zero_indices":
                candidate_manifest[
                    "line_context_zero_indices"
                ],

            "external_prediction_ready":
                False,

            "remaining_blocker":
                freeze_qc[
                    "remaining_blocker"
                ],

            "external_outcomes_opened":
                False,
        },
    )


if __name__ == "__main__":

    main()
