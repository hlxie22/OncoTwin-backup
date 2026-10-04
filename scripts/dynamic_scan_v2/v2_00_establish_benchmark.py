#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from access import AccessDeniedError, AccessPolicy
from common import (
    assert_unique,
    atomic_json,
    atomic_parquet,
    atomic_text,
    import_module_from_path,
    sha256_file,
    stable_fold,
)
from metrics import (
    ADMIN_HORIZON_DAYS,
    CAUSE_DEATH,
    CAUSE_NO_EVENT_OR_CENSOR,
    CAUSE_PROGRESSION,
    CAUSE_SWITCH,
    HORIZONS_MONTHS,
    MONTH_DAYS,
    cluster_bootstrap_mean_delta,
    curves_from_logits,
    fit_censoring_km,
    fractional_censor_nll_per_row,
    v2_metric_bundle,
    resample_patient_clusters,
    validate_curve_invariants,
)

from legacy_metrics import legacy_metric_bundle, legacy_nll_per_row

EXPECTED_CRITICAL_HASHES = {
    "artifacts/checkpoint7r1_line_agnostic_temporal/temporal_encoder.pt":
        "daa47e1d30d61a3c3a6c86a146c683a0a1ac0a635e0d8e6247f3ce47c4ae1411",
    "scripts/dynamic_scan/ckpt7r1_line_agnostic_temporal.py":
        "8e747c64af927ae923714f307c1bb796f9805e8c59b84323a57991a0653ff4ba",
    "scripts/dynamic_scan/ckpt5_supervised_dynamic_model.py":
        "00f6782f82604c04cd25248057634fbbe24a28375505f833681865405a4231ef",
    "artifacts/checkpoint7r2_transport_safe_supervised/dynamic_scan_model.pt":
        "bb540d56d05130081ea3738f5515567e3013ef8013ce414d5dcc82c563a22828",
    "artifacts/checkpoint7r3b_bounded_candidate/bounded_dynamic_scan_candidate.pt":
        "1819af166f004cb0911508d4c74d96b7602cb2548aaf1946d532ad21ab1a0eea",
    "scripts/dynamic_scan/ckpt7r3b_freeze_bounded_update.py":
        "3a2d01385c3b25ee1982a51ab6be03f36536300618f7c3b68b36221acd8760ae",
    "artifacts/checkpoint7r3f2_site_unavailable_ablation/transport_rule.json":
        "4b65b554ee6ced8894b98c3ea2ed622dcbf8283656929d146c73975728230576",
    "artifacts/checkpoint7r3e_transport_safe_external_protocol/outcome_access_lock.json":
        "8bb41f49afa8c9077b567074fe161eb3818a4a243f9e2ca13612bb3883a0b371",
}

LINEAGE_MANIFESTS = (
    "artifacts/checkpoint7r1_line_agnostic_temporal/manifest.json",
    "artifacts/checkpoint7r2_transport_safe_supervised/manifest.json",
    "artifacts/checkpoint7r3b_bounded_candidate/manifest.json",
)

KEY_COLUMNS = ["patient_id", "scan_episode_id", "landmark_day"]
FROZEN_ALPHA = 0.5


def git_value(repo: Path, *args: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
    except Exception:
        return "UNKNOWN"


def check_critical_hashes(repo: Path, policy: AccessPolicy) -> dict[str, Any]:
    rows = []
    for relative, expected in EXPECTED_CRITICAL_HASHES.items():
        path = policy.assert_allowed(repo / relative, "critical lineage hash")
        if not path.exists():
            rows.append({"path": relative, "expected_sha256": expected, "actual_sha256": None, "status": "MISSING"})
            continue
        actual = sha256_file(path)
        rows.append({
            "path": relative,
            "expected_sha256": expected,
            "actual_sha256": actual,
            "status": "PASS" if actual == expected else "FAIL",
        })
    return {
        "status": "PASS" if all(r["status"] == "PASS" for r in rows) else "FAIL",
        "files": rows,
    }


def verify_internal_manifest(repo: Path, manifest_relative: str, policy: AccessPolicy) -> dict[str, Any]:
    manifest_path = policy.assert_allowed(repo / manifest_relative, "lineage manifest read")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = []
    for relative, metadata in payload.get("files", {}).items():
        try:
            path = policy.assert_allowed(repo / relative, "manifest verification")
        except AccessDeniedError as exc:
            checks.append({"path": relative, "status": "BLOCKED_BY_POLICY", "detail": str(exc)})
            continue
        if not path.exists():
            checks.append({"path": relative, "status": "MISSING"})
            continue
        actual = sha256_file(path)
        expected = metadata.get("sha256")
        checks.append({
            "path": relative,
            "expected_sha256": expected,
            "actual_sha256": actual,
            "bytes": int(path.stat().st_size),
            "status": "PASS" if actual == expected else "FAIL",
        })
    failures = [x for x in checks if x["status"] not in {"PASS", "BLOCKED_BY_POLICY"}]
    return {
        "manifest": manifest_relative,
        "manifest_sha256": sha256_file(manifest_path),
        "status": "PASS" if not failures else "FAIL",
        "checks": checks,
    }


def build_legacy_reference_manifest(repo: Path, out: Path, policy: AccessPolicy) -> dict[str, Any]:
    critical = check_critical_hashes(repo, policy)
    manifests = [verify_internal_manifest(repo, rel, policy) for rel in LINEAGE_MANIFESTS]
    r2_design = policy.read_json(repo / "artifacts/checkpoint7r2_transport_safe_supervised/design.json")
    r2_manifest = policy.read_json(repo / "artifacts/checkpoint7r2_transport_safe_supervised/manifest.json")
    r3b_design = policy.read_json(repo / "artifacts/checkpoint7r3b_bounded_candidate/design.json")
    r1_design = policy.read_json(repo / "artifacts/checkpoint7r1_line_agnostic_temporal/design.json")
    feature_schema = policy.read_json(repo / "artifacts/checkpoint7r2_transport_safe_supervised/prepared/feature_schema.json")

    dedicated_r2_source = repo / "scripts/dynamic_scan/ckpt7r2_transport_safe_supervised.py"
    lineage = {
        "status": "PASS" if critical["status"] == "PASS" and all(x["status"] == "PASS" for x in manifests) else "FAIL",
        "repo_git_head_at_replay": git_value(repo, "rev-parse", "HEAD"),
        "legacy_candidate": {
            "temporal_encoder": "CKPT7R1 line-agnostic temporal encoder",
            "temporal_dimension": r1_design.get("temporal_dimensions", 192),
            "fusion_model": "CKPT7R2 transport-safe supervised fusion model",
            "bounded_update": "PRE + 0.5 * (FULL_POST - PRE)",
            "alpha": FROZEN_ALPHA,
            "scan_window": "W3 episode-end",
            "feature_schema": feature_schema,
        },
        "source_provenance": {
            "r2_dedicated_source_script_present": dedicated_r2_source.exists(),
            "r2_base_source": "scripts/dynamic_scan/ckpt5_supervised_dynamic_model.py",
            "r2_base_source_sha256": sha256_file(repo / "scripts/dynamic_scan/ckpt5_supervised_dynamic_model.py"),
            "r2_design_declared_base_source_sha256": r2_design.get("source_hashes", {}).get("ckpt5"),
            "r2_repair_documented_by": [
                "artifacts/checkpoint7r2_transport_safe_supervised/design.json",
                "artifacts/checkpoint7r2_transport_safe_supervised/prepared_interface_audit.json",
                "artifacts/checkpoint7r2_transport_safe_supervised/repair_qc.json",
            ],
            "note": "No standalone CKPT7R2 source script is present. V2 records CKPT5 at the exact source hash declared by CKPT7R2 plus the immutable R2 design/interface artifacts as the repaired-model provenance.",
        },
        "frozen_interface": r2_design.get("repaired_interfaces", {}),
        "r3b_design": r3b_design,
        "critical_hashes": critical,
        "manifest_verification": manifests,
        "external_policy": {
            "external_outcomes_reopened": False,
            "external_predictions_recomputed": False,
            "external_prediction_artifacts_read": False,
            "protected_outcome_artifacts_read": False,
        },
    }
    atomic_json(out / "legacy_reference_manifest.json", lineage)
    return lineage


def _exposure_rows(
    frame: pd.DataFrame,
    *,
    dataset: str,
    module: str,
    exposure_type: str,
    selection_role: str,
    patient_col: str,
    split_col: str | None = None,
    extra: dict[str, Any] | None = None,
) -> pd.DataFrame:
    use = frame.copy()
    use[patient_col] = use[patient_col].astype(str)
    group_cols = [patient_col]
    if split_col and split_col in use.columns:
        use[split_col] = use[split_col].astype(str)
        group_cols.append(split_col)
    grouped = use.groupby(group_cols, observed=True).size().rename("source_rows").reset_index()
    grouped = grouped.rename(columns={patient_col: "patient_id"})
    if split_col and split_col in grouped.columns:
        grouped = grouped.rename(columns={split_col: "source_split"})
    else:
        grouped["source_split"] = "NA"
    grouped["dataset"] = dataset
    grouped["module"] = module
    grouped["exposure_type"] = exposure_type
    grouped["selection_role"] = selection_role
    if extra:
        for key, value in extra.items():
            grouped[key] = value
    return grouped[
        ["dataset", "module", "exposure_type", "selection_role", "patient_id", "source_split", "source_rows"]
        + ([] if not extra else list(extra))
    ]


def build_patient_exposure_registry(repo: Path, out: Path, policy: AccessPolicy) -> tuple[pd.DataFrame, dict[str, Any]]:
    pieces: list[pd.DataFrame] = []

    splits = policy.read_parquet(repo / "artifacts/checkpoint1/canonical/chord_patient_splits.parquet")
    pieces.append(_exposure_rows(
        splits,
        dataset="CHORD",
        module="CKPT1_CANONICAL_SPLIT",
        exposure_type="patient_partition",
        selection_role="frozen_patient_split",
        patient_col="patient_id",
        split_col="split",
    ))

    genie = policy.read_parquet(
        repo / "artifacts/checkpoint3/prepared/sample_metadata.parquet",
        columns=["patient_id", "split", "cancer_type"],
    )
    pieces.append(_exposure_rows(
        genie,
        dataset="GENIE",
        module="CKPT3_GENOMIC_PRETRAIN",
        exposure_type="genomic_pretraining_input",
        selection_role="representation_pretraining",
        patient_col="patient_id",
        split_col="split",
    ))

    for name, module in [
        ("pan_cancer_tokens.parquet", "CKPT4_TEMPORAL_PRETRAIN_PAN_CANCER"),
        ("breast_tokens.parquet", "CKPT4_TEMPORAL_PRETRAIN_BREAST"),
    ]:
        token_path = repo / "artifacts/checkpoint4/prepared" / name
        tokens = policy.read_parquet(token_path, columns=["patient_id", "split", "cohort"])
        pieces.append(_exposure_rows(
            tokens,
            dataset="CHORD",
            module=module,
            exposure_type="self_supervised_temporal_token",
            selection_role="representation_pretraining",
            patient_col="patient_id",
            split_col="split",
        ))

    r1_index = policy.read_parquet(
        repo / "artifacts/checkpoint7r1_line_agnostic_temporal/breast_scan_prepost_index.parquet",
        columns=["patient_id", "split", "scan_episode_id"],
    )
    pieces.append(_exposure_rows(
        r1_index,
        dataset="CHORD",
        module="CKPT7R1_LINE_AGNOSTIC_TEMPORAL",
        exposure_type="scan_prepost_embedding_cache",
        selection_role="repaired_representation",
        patient_col="patient_id",
        split_col="split",
    ))

    r2_index = policy.read_parquet(repo / "artifacts/checkpoint7r2_transport_safe_supervised/prepared/scan_index.parquet")
    pieces.append(_exposure_rows(
        r2_index,
        dataset="CHORD",
        module="CKPT7R2_SUPERVISED_ALL_TASKS",
        exposure_type="supervised_multitask_input",
        selection_role="supervised_training_or_evaluation_by_frozen_split",
        patient_col="patient_id",
        split_col="split",
    ))
    survival = r2_index.loc[r2_index["survival_mask"].astype(bool)].copy()
    pieces.append(_exposure_rows(
        survival,
        dataset="CHORD",
        module="CKPT7R2_SURVIVAL_HEAD",
        exposure_type="survival_supervision",
        selection_role="supervised_survival",
        patient_col="patient_id",
        split_col="split",
    ))

    for split in ("val", "test"):
        r3b_path = repo / f"artifacts/checkpoint7r3b_bounded_candidate/{split}_bounded_update_predictions.parquet"
        r3b = policy.read_parquet(r3b_path, columns=["patient_id", "split"])
        role = "alpha_frozen_validation_assessment" if split == "val" else "historical_test_descriptive_only"
        pieces.append(_exposure_rows(
            r3b,
            dataset="CHORD",
            module="CKPT7R3B_BOUNDED_CANDIDATE",
            exposure_type="candidate_evaluation",
            selection_role=role,
            patient_col="patient_id",
            split_col="split",
        ))

    msk = policy.read_parquet(
        repo / "artifacts/checkpoint1/canonical/bpc_msk_scan_audit.parquet",
        columns=["patient_id", "source"],
    )
    msk["schema_split"] = "schema_audit"
    pieces.append(_exposure_rows(
        msk,
        dataset="BPC_MSK",
        module="CKPT1_BPC_MSK_SCAN_AUDIT",
        exposure_type="schema_positive_control",
        selection_role="schema_only_not_model_selection",
        patient_col="patient_id",
        split_col="schema_split",
    ))

    registry = pd.concat(pieces, ignore_index=True)
    registry["patient_id"] = registry["patient_id"].astype(str)
    registry = registry.sort_values(["dataset", "module", "patient_id"], kind="stable").reset_index(drop=True)
    atomic_parquet(out / "patient_exposure_registry.parquet", registry)

    summary = {
        "rows": int(len(registry)),
        "modules": {},
        "protected_external_patient_rows_read": 0,
        "status": "PASS",
    }
    for module, group in registry.groupby("module", observed=True):
        summary["modules"][str(module)] = {
            "datasets": sorted(group["dataset"].unique().tolist()),
            "patients": int(group["patient_id"].nunique()),
            "source_rows": int(group["source_rows"].sum()),
            "splits": {str(k): int(v) for k, v in group.groupby("source_split", observed=True)["patient_id"].nunique().items()},
        }
    atomic_json(out / "patient_exposure_summary.json", summary)
    return registry, summary


def build_development_folds(repo: Path, out: Path, policy: AccessPolicy, protocol: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    splits = policy.read_parquet(repo / "artifacts/checkpoint1/canonical/chord_patient_splits.parquet")[["patient_id", "split"]].copy()
    splits["patient_id"] = splits["patient_id"].astype(str)
    splits["split"] = splits["split"].astype(str).str.lower()
    assert_unique(splits, ["patient_id"], "CHORD patient split")
    fold_count = int(protocol["folds"]["count"])
    salt = str(protocol["folds"]["salt"])
    rows = []
    for row in splits.itertuples(index=False):
        original = row.split
        if original in {"train", "val", "validation"}:
            fold = stable_fold(row.patient_id, fold_count, salt)
            role = "v2_development"
        elif original == "test":
            fold = -1
            role = "historical_test_read_only"
        else:
            raise RuntimeError(f"Unexpected frozen split label: {original}")
        rows.append({
            "patient_id": row.patient_id,
            "original_split": original,
            "v2_fold": fold,
            "v2_role": role,
            "assignment_method": "sha256_outcome_blind",
            "assignment_salt": salt,
        })
    frame = pd.DataFrame(rows)
    atomic_parquet(out / "development_folds.parquet", frame)
    dev = frame.loc[frame["v2_role"] == "v2_development"]
    counts = {str(k): int(v) for k, v in dev.groupby("v2_fold")["patient_id"].nunique().items()}
    if set(counts) != {str(i) for i in range(fold_count)}:
        raise RuntimeError(f"Missing development fold: {counts}")
    report = {
        "status": "PASS",
        "development_patients": int(dev["patient_id"].nunique()),
        "historical_test_patients": int((frame["v2_role"] == "historical_test_read_only").sum()),
        "fold_patient_counts": counts,
        "outcome_blind_assignment": True,
        "original_test_assigned_to_development_fold": False,
    }
    atomic_json(out / "development_fold_summary.json", report)
    return frame, report


def build_source_capabilities(repo: Path, out: Path, policy: AccessPolicy) -> dict[str, Any]:
    scan_path = repo / "artifacts/checkpoint1/canonical/chord_breast_scan_episodes_w3.parquet"
    scans = policy.read_parquet(scan_path)
    events = policy.read_parquet(
        repo / "artifacts/checkpoint1/canonical/chord_canonical_events.parquet",
        columns=["event_type", "event_subtype", "source_table", "availability_quality"],
    )
    genomics = policy.read_parquet(
        repo / "artifacts/checkpoint5/prepared/chord_genomic_samples.parquet",
        columns=["patient_id", "availability_day", "panel_id", "panel_resolved", "availability_resolved", "coverage_gene_count"],
    )
    msk = policy.read_parquet(repo / "artifacts/checkpoint1/canonical/bpc_msk_scan_audit.parquet")
    r2_schema = policy.read_json(repo / "artifacts/checkpoint7r2_transport_safe_supervised/prepared/feature_schema.json")
    site_rule = policy.read_json(repo / "artifacts/checkpoint7r3f2_site_unavailable_ablation/transport_rule.json")

    combined_event_text = (
        events.fillna("")[["event_type", "event_subtype", "source_table"]]
        .astype(str)
        .agg(" | ".join, axis=1)
        .str.lower()
    )
    lab_like = combined_event_text.str.contains(r"lab|ca15|ca 15|cea|ecog", regex=True, na=False)

    nonempty_modality = scans["modalities_json"].fillna("").astype(str).str.strip().isin(["", "[]", "{}", "null", "None"])
    nonempty_sites = scans["tumor_sites_json"].fillna("").astype(str).str.strip().isin(["", "[]", "{}", "null", "None"])
    coverage_cols = [c for c in scans.columns if c.startswith("coverage_") and not c.startswith("coverage_state_")]

    capabilities = {
        "status": "PASS",
        "generated_without_dfci_vicc_row_access": True,
        "licensing_note": "V2-00 records repository/access status only. It does not reinterpret upstream data licenses; any new raw-text or lesion source requires separate authorization confirmation.",
        "fields": {
            "current_scan_state_3": {
                "available_locally": True,
                "source": "artifacts/checkpoint1/canonical/chord_breast_scan_episodes_w3.parquet:progression_state_3",
                "semantic_meaning": "CHORD authoritative 3-state assessment: NON_PROGRESSIVE / INDETERMINATE / PROGRESSIVE",
                "event_clock": "episode component source days",
                "availability_clock": "landmark_day = episode_end_day",
                "required_for_portable_core": True,
                "source_available_mask": True,
                "access_status": "AUTHORIZED_DEVELOPMENT_DERIVED_ARTIFACT",
            },
            "scan_anatomical_coverage": {
                "available_locally": True,
                "source": coverage_cols,
                "semantic_meaning": "anatomical observation coverage only; false/unobserved never means disease-negative",
                "event_clock": "scan episode",
                "availability_clock": "episode_end_day",
                "required_for_portable_core": True,
                "access_status": "AUTHORIZED_DEVELOPMENT_DERIVED_ARTIFACT",
            },
            "scan_modality": {
                "available_locally": True,
                "source": "modalities_json and legacy current-scan modality features",
                "semantic_meaning": "observed imaging modality descriptor; optional transport stream",
                "nonempty_episode_count": int((~nonempty_modality).sum()),
                "required_for_portable_core": False,
                "external_transport": "not required; CKPT7R3E zeroed legacy modality indices externally",
                "access_status": "AUTHORIZED_DEVELOPMENT_DERIVED_ARTIFACT",
            },
            "positive_site_mentions": {
                "available_locally": True,
                "source": "tumor_sites_json and legacy site_bone/site_liver/site_lung/site_brain/site_lymph/site_pleura features",
                "semantic_meaning": "positive site mention/support, never a complete disease-negative anatomy vector",
                "nonempty_episode_count": int((~nonempty_sites).sum()),
                "required_for_portable_core": False,
                "external_transport": site_rule.get("semantic_meaning"),
                "access_status": "AUTHORIZED_DEVELOPMENT_DERIVED_ARTIFACT",
            },
            "longitudinal_scan_sequence": {
                "available_locally": True,
                "source": "authoritative W3 scan episode sequence keyed by patient_id, scan_episode_id, landmark_day",
                "semantic_meaning": "prior assessments and coverage can be derived causally using prior landmark_day only",
                "required_for_portable_core": True,
                "access_status": "AUTHORIZED_DEVELOPMENT_DERIVED_ARTIFACT",
            },
            "treatment_chronology": {
                "available_locally": True,
                "source": "chord_canonical_events treatment start/end chronology",
                "semantic_meaning": "observable treatment timing events",
                "model_policy": "chronology may be represented without absolute treatment-line identity",
                "required_for_portable_core": False,
                "access_status": "AUTHORIZED_DEVELOPMENT_DERIVED_ARTIFACT",
            },
            "absolute_treatment_line_identity": {
                "available_locally": True,
                "semantic_meaning": "historical line labels/line-relative coordinates",
                "model_policy": "PROHIBITED_MODEL_FACING_V2",
                "reason": "CKPT7A4C transport audit; CKPT7R1/R2 neutralized line identity",
                "required_for_portable_core": False,
                "access_status": "AUDIT_ONLY",
            },
            "genomics": {
                "available_locally": True,
                "source": "CKPT5 prepared CHORD genomic samples + frozen CKPT3 128-D encoder",
                "semantic_meaning": "coverage-aware tumor genomic representation",
                "availability_clock": "availability_day < landmark_day; same-day excluded",
                "samples": int(len(genomics)),
                "patients": int(genomics["patient_id"].astype(str).nunique()),
                "panel_resolved_fraction": float(pd.to_numeric(genomics["panel_resolved"], errors="coerce").fillna(0).astype(bool).mean()),
                "required_for_portable_core": False,
                "mask_and_recency_required": True,
                "access_status": "AUTHORIZED_DEVELOPMENT_DERIVED_ARTIFACT",
            },
            "lab_or_performance_status_context": {
                "available_locally": bool(lab_like.any()),
                "source": "canonical event stream if validated per-field in a later checkpoint",
                "matching_event_rows_v2_00_screen": int(lab_like.sum()),
                "semantic_meaning": "not yet promoted to a V2 predictor; field-specific provenance/availability audit required",
                "required_for_portable_core": False,
                "access_status": "DISCOVERED_NOT_VALIDATED_FOR_V2_MODEL",
            },
            "raw_radiology_report_text": {
                "available_locally": False,
                "source": None,
                "semantic_meaning": "No authorized raw-report-text predictor source established by V2-00",
                "required_for_portable_core": False,
                "access_status": "NOT_ESTABLISHED_DO_NOT_SYNTHESIZE",
                "v2_t": "disabled unless a separately authorized genuine source is identified",
            },
            "lesion_level_measurements": {
                "available_locally": False,
                "source": None,
                "semantic_meaning": "No lesion-level measurement source established by V2-00",
                "required_for_portable_core": False,
                "access_status": "NOT_ESTABLISHED_DO_NOT_SYNTHESIZE",
                "v2_t": "disabled unless a separately authorized genuine source is identified",
            },
            "bpc_msk_response_state_5": {
                "available_locally": True,
                "source": "artifacts/checkpoint1/canonical/bpc_msk_scan_audit.parquet:response_state_5",
                "semantic_meaning": "schema/semantic positive control only; richer BPC states do not create missing CHORD stable/responding labels",
                "rows": int(len(msk)),
                "patients": int(msk["patient_id"].astype(str).nunique()),
                "selection_role": "NONE",
                "access_status": "DERIVED_SCHEMA_POSITIVE_CONTROL_ONLY",
            },
            "dfci_vicc_rows": {
                "available_locally": "possibly retained from completed study",
                "model_policy": "V2_00_TO_V2_05_HARD_DENY",
                "outcome_policy": "never reopen for V2 selection or recalibration",
                "required_for_portable_core": False,
                "access_status": "QUARANTINED",
            },
        },
        "legacy_r2_current_scan_feature_order": r2_schema.get("current_scan", []),
        "canonical_event_type_counts": {str(k): int(v) for k, v in events["event_type"].astype(str).value_counts().head(30).items()},
        "canonical_source_table_counts": {str(k): int(v) for k, v in events["source_table"].astype(str).value_counts().head(30).items()},
    }
    atomic_json(out / "source_capabilities.json", capabilities)
    return capabilities


def audit_prepared_interface(repo: Path, out: Path, policy: AccessPolicy) -> dict[str, Any]:
    r2 = repo / "artifacts/checkpoint7r2_transport_safe_supervised/prepared"
    ckpt5 = repo / "artifacts/checkpoint5/prepared"
    r1 = repo / "artifacts/checkpoint7r1_line_agnostic_temporal"

    schema = policy.read_json(r2 / "feature_schema.json")
    context = policy.np_load(r2 / "context_features_f32.npy", mmap_mode="r")
    old_context = policy.np_load(ckpt5 / "context_features_f32.npy", mmap_mode="r")
    temporal = policy.np_load(r2 / "temporal_prepost_f16.npy", mmap_mode="r")
    tumor = policy.np_load(r2 / "tumor_embeddings_f16.npy", mmap_mode="r")
    index = policy.read_parquet(r2 / "scan_index.parquet")
    r1_index = policy.read_parquet(r1 / "breast_scan_prepost_index.parquet")

    names = schema["context"]
    neutral_names = schema["context_transport_policy"]["neutralized_exact_zero"]
    preserved_names = schema["context_transport_policy"]["preserved"]
    neutral_idx = [names.index(name) for name in neutral_names]
    preserved_idx = [names.index(name) for name in preserved_names]

    context_zero = bool(np.all(np.asarray(context[:, neutral_idx]) == 0.0))
    context_preserved = bool(np.array_equal(np.asarray(context[:, preserved_idx]), np.asarray(old_context[:, preserved_idx])))

    temporal_hash_match = sha256_file(r2 / "temporal_prepost_f16.npy") == sha256_file(r1 / "breast_scan_prepost_embeddings_f16.npy")
    scan_hash_match = sha256_file(r2 / "current_scan_features_f32.npy") == sha256_file(ckpt5 / "current_scan_features_f32.npy")
    tumor_hash_match = sha256_file(r2 / "tumor_embeddings_f16.npy") == sha256_file(ckpt5 / "tumor_embeddings_f16.npy")
    index_hash_match = sha256_file(r2 / "scan_index.parquet") == sha256_file(ckpt5 / "scan_index.parquet")

    temporal_rows = pd.to_numeric(index["temporal_row"], errors="raise").astype(int).to_numpy()
    if (temporal_rows < 0).any() or (temporal_rows >= len(r1_index)).any():
        temporal_key_alignment = False
    else:
        selected = r1_index.iloc[temporal_rows].reset_index(drop=True)
        compare = index.reset_index(drop=True)
        temporal_key_alignment = bool(
            selected["patient_id"].astype(str).to_numpy().tolist() == compare["patient_id"].astype(str).to_numpy().tolist()
            and selected["scan_episode_id"].astype(str).to_numpy().tolist() == compare["scan_episode_id"].astype(str).to_numpy().tolist()
            and np.array_equal(selected["landmark_day"].to_numpy(), compare["landmark_day"].to_numpy())
        )

    genomic_available = pd.to_numeric(index["genomic_available"], errors="coerce").fillna(0).to_numpy() > 0
    availability = pd.to_numeric(index["availability_day"], errors="coerce").to_numpy(dtype=float)
    landmark = pd.to_numeric(index["landmark_day"], errors="coerce").to_numpy(dtype=float)
    genomic_time_ok = bool(np.all(availability[genomic_available] < landmark[genomic_available]))
    unavailable_tumor_zero = bool(np.all(np.asarray(tumor[~genomic_available], dtype=np.float32) == 0.0))

    survival_mask = index["survival_mask"].astype(bool).to_numpy()
    survival_time = pd.to_numeric(index["survival_time_days"], errors="coerce").to_numpy(dtype=float)
    positive_time = bool(np.all(survival_time[survival_mask] > 0.0))
    progressive_excluded = bool(~(
        survival_mask
        & (index["scan_state"].astype(str).str.upper().to_numpy() == "PROGRESSIVE")
    ).any())

    patient_split_counts = index[["patient_id", "split"]].drop_duplicates().groupby("patient_id")["split"].nunique()
    patient_disjoint = bool((patient_split_counts == 1).all())

    report = {
        "status": "PASS" if all([
            context_zero,
            context_preserved,
            temporal_hash_match,
            scan_hash_match,
            tumor_hash_match,
            index_hash_match,
            temporal_key_alignment,
            genomic_time_ok,
            unavailable_tumor_zero,
            positive_time,
            progressive_excluded,
            patient_disjoint,
        ]) else "FAIL",
        "rows": int(len(index)),
        "shapes": {
            "context": list(context.shape),
            "temporal": list(temporal.shape),
            "tumor": list(tumor.shape),
        },
        "context_neutralized_exact_zero": context_zero,
        "context_preserved_bit_exact": context_preserved,
        "temporal_cache_exact_r1_hash": temporal_hash_match,
        "current_scan_exact_ckpt5_hash": scan_hash_match,
        "tumor_embedding_exact_ckpt5_hash": tumor_hash_match,
        "scan_index_exact_ckpt5_hash": index_hash_match,
        "temporal_row_key_alignment": temporal_key_alignment,
        "genomic_availability_strictly_before_landmark": genomic_time_ok,
        "unavailable_genomic_embedding_exact_zero": unavailable_tumor_zero,
        "zero_or_negative_survival_time_excluded": positive_time,
        "progressive_scan_excluded_from_survival_landmarks": progressive_excluded,
        "patient_split_disjoint": patient_disjoint,
        "legacy_current_scan_names": schema["current_scan"],
        "legacy_context_names": names,
        "line_identity_model_facing_policy": "neutralized context coordinates plus line-agnostic temporal cache",
    }
    atomic_json(out / "prepared_interface_replay.json", report)
    return report


def load_r2_model(repo: Path, policy: AccessPolicy, device: torch.device):
    ckpt5_path = policy.assert_allowed(repo / "scripts/dynamic_scan/ckpt5_supervised_dynamic_model.py", "model source import")
    ckpt5 = import_module_from_path(ckpt5_path, "oncotwin_v2_frozen_ckpt5")
    checkpoint = policy.torch_load(
        repo / "artifacts/checkpoint7r2_transport_safe_supervised/dynamic_scan_model.pt",
        torch_module=torch,
        map_location=device,
        weights_only=False,
    )
    required = {"scan_dim", "context_dim", "model_state"}
    missing = required - set(checkpoint)
    if missing:
        raise RuntimeError(f"R2 checkpoint missing keys {sorted(missing)}")
    model = ckpt5.DynamicFusionModel(
        scan_dim=int(checkpoint["scan_dim"]),
        context_dim=int(checkpoint["context_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return ckpt5, checkpoint, model


def load_r2_prepared(repo: Path, policy: AccessPolicy) -> dict[str, Any]:
    base = repo / "artifacts/checkpoint7r2_transport_safe_supervised/prepared"
    return {
        "index": policy.read_parquet(base / "scan_index.parquet").reset_index(drop=True),
        "temporal": policy.np_load(base / "temporal_prepost_f16.npy", mmap_mode="r"),
        "tumor": policy.np_load(base / "tumor_embeddings_f16.npy", mmap_mode="r"),
        "scan": policy.np_load(base / "current_scan_features_f32.npy", mmap_mode="r"),
        "context": policy.np_load(base / "context_features_f32.npy", mmap_mode="r"),
        "schema": policy.read_json(base / "feature_schema.json"),
    }


def forward_rows(model, data: dict[str, Any], rows: np.ndarray, temporal_view: int, include_scan: bool, device: torch.device) -> np.ndarray:
    out = np.empty((len(rows), 24, 4), dtype=np.float32)
    batch_size = 1024
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            stop = min(start + batch_size, len(rows))
            local = rows[start:stop]
            temporal = torch.from_numpy(np.asarray(data["temporal"][local, temporal_view], dtype=np.float32).copy()).to(device)
            tumor = torch.from_numpy(np.asarray(data["tumor"][local], dtype=np.float32).copy()).to(device)
            context = torch.from_numpy(np.asarray(data["context"][local], dtype=np.float32).copy()).to(device)
            if include_scan:
                scan_np = np.asarray(data["scan"][local], dtype=np.float32).copy()
            else:
                scan_np = np.zeros((len(local), data["scan"].shape[1]), dtype=np.float32)
            scan = torch.from_numpy(scan_np).to(device)
            output = model(temporal, tumor, scan, context)
            logits = output["survival_logits"].float().cpu().numpy()
            if not np.isfinite(logits).all():
                raise RuntimeError("non-finite replay logits")
            out[start:stop] = logits
    return out


def replay_split(
    repo: Path,
    policy: AccessPolicy,
    data: dict[str, Any],
    model,
    split: str,
    device: torch.device,
    km,
    metrics_contract: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray], pd.DataFrame]:
    index = data["index"]
    selected = (index["split"].astype(str) == split) & index["survival_mask"].astype(bool)
    rows = np.flatnonzero(selected.to_numpy())
    frame = index.iloc[rows].copy().reset_index(drop=True)
    assert_unique(frame, KEY_COLUMNS, f"R2 {split} survival")

    pre = forward_rows(model, data, rows, temporal_view=0, include_scan=False, device=device)
    post = forward_rows(model, data, rows, temporal_view=1, include_scan=True, device=device)
    bounded = pre + FROZEN_ALPHA * (post - pre)
    predictions = {"pre": pre, "full_post": post, "bounded_post": bounded}
    bundles = {}
    for name, logits in predictions.items():
        legacy = legacy_metric_bundle(frame, logits, km)
        v2 = v2_metric_bundle(frame, logits, km)
        bundles[name] = {
            **legacy,
            **v2,
            "legacy_minus_fractional_patient_nll": float(
                legacy["legacy_patient_mean_nll"] - v2["fractional_censor_patient_mean_nll_v1"]
            ),
        }

    archived = policy.read_parquet(repo / f"artifacts/checkpoint7r3b_bounded_candidate/{split}_bounded_update_predictions.parquet")
    archived = archived.reset_index(drop=True)
    if len(archived) != len(frame):
        raise RuntimeError(f"Archived {split} row count mismatch {len(archived)} != {len(frame)}")
    for key in KEY_COLUMNS:
        if key not in archived.columns:
            raise RuntimeError(f"Archived R3B {split} lacks key {key}")
        if key == "landmark_day":
            equal = np.array_equal(pd.to_numeric(frame[key]).to_numpy(), pd.to_numeric(archived[key]).to_numpy())
        else:
            equal = frame[key].astype(str).to_numpy().tolist() == archived[key].astype(str).to_numpy().tolist()
        if not equal:
            raise RuntimeError(f"Archived R3B {split} key mismatch at {key}")

    tol = metrics_contract["numerical_tolerances"]
    comparisons: dict[str, Any] = {"row_keys_exact": True, "variants": {}}
    for name, logits in predictions.items():
        prefix = {"pre": "pre", "full_post": "full_post", "bounded_post": "bounded_post"}[name]
        curves = curves_from_logits(logits)
        nll = legacy_nll_per_row(
            logits,
            frame["survival_time_days"].to_numpy(dtype=float),
            frame["survival_cause"].astype(int).to_numpy(),
        )
        nll_diff = np.abs(nll - archived[f"{prefix}_nll"].to_numpy(dtype=float))
        pfs_diffs = {}
        for horizon in HORIZONS_MONTHS:
            new_pfs = curves["pfs"][:, horizon - 1]
            old_pfs = archived[f"{prefix}_pfs_{horizon}m"].to_numpy(dtype=float)
            pfs_diffs[f"{horizon}m"] = float(np.max(np.abs(new_pfs - old_pfs)))
        comparisons["variants"][name] = {
            "row_nll_max_abs": float(nll_diff.max()),
            "row_nll_tolerance": float(tol["archived_row_nll_max_abs"]),
            "pfs_max_abs_by_horizon": pfs_diffs,
            "pfs_tolerance": float(tol["archived_pfs_max_abs"]),
            "row_replay_pass": bool(
                nll_diff.max() <= tol["archived_row_nll_max_abs"]
                and max(pfs_diffs.values()) <= tol["archived_pfs_max_abs"]
            ),
        }

    historical_metrics = policy.read_json(repo / "artifacts/checkpoint7r3b_bounded_candidate/bounded_update_metrics.json")["splits"][split]
    aggregate = {}
    for name in ("pre", "full_post", "bounded_post"):
        hist = historical_metrics[name]
        ours = bundles[name]
        nll_abs = abs(float(ours["legacy_patient_mean_nll"]) - float(hist["patient_mean_nll"]))
        brier_abs = abs(float(ours["legacy_patient_integrated_brier_4h"]) - float(hist["patient_integrated_brier_4h"]))
        aggregate[name] = {
            "legacy_patient_nll_replay": ours["legacy_patient_mean_nll"],
            "archived_patient_nll": hist["patient_mean_nll"],
            "nll_abs_difference": nll_abs,
            "patient_brier_4h_replay": ours["legacy_patient_integrated_brier_4h"],
            "archived_patient_integrated_brier_4h": hist["patient_integrated_brier_4h"],
            "brier_abs_difference": brier_abs,
            "aggregate_replay_pass": bool(
                nll_abs <= tol["archived_aggregate_nll_abs"]
                and brier_abs <= tol["archived_aggregate_brier_abs"]
            ),
        }

    status = "PASS" if (
        all(v["row_replay_pass"] for v in comparisons["variants"].values())
        and all(v["aggregate_replay_pass"] for v in aggregate.values())
        and all(bundles[name]["legacy_curve_invariants"]["status"] == "PASS" and bundles[name]["curve_invariants"]["status"] == "PASS" for name in bundles)
    ) else "FAIL"

    report = {
        "status": status,
        "split": split,
        "rows": int(len(frame)),
        "patients": int(frame["patient_id"].nunique()),
        "device": str(device),
        "archived_comparison": comparisons,
        "aggregate_comparison": aggregate,
        "metrics": bundles,
        "selection_role": "development_validation" if split == "val" else "historical_test_identity_check_only",
    }
    return report, predictions, frame


def write_replay_fixture(out: Path, frame: pd.DataFrame, predictions: dict[str, np.ndarray]) -> dict[str, Any]:
    key_text = (
        frame["patient_id"].astype(str)
        + "|" + frame["scan_episode_id"].astype(str)
        + "|" + frame["landmark_day"].astype(str)
    ).tolist()
    order = sorted(range(len(frame)), key=lambda i: hashlib.sha256(key_text[i].encode("utf-8")).hexdigest())[:32]
    positions = np.asarray(order, dtype=int)
    fixture_index = frame.iloc[positions][
        [c for c in ["patient_id", "scan_episode_id", "landmark_day", "split", "scan_state", "survival_time_days", "survival_cause"] if c in frame.columns]
    ].copy().reset_index(drop=True)
    fixture_index["fixture_row"] = np.arange(len(fixture_index), dtype=int)
    atomic_parquet(out / "replay_fixture_index.parquet", fixture_index)

    npz_path = out / "replay_fixture_logits.npz"
    tmp = npz_path.with_suffix(".npz.tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(
            handle,
            pre=predictions["pre"][positions].astype(np.float32),
            full_post=predictions["full_post"][positions].astype(np.float32),
            bounded_post=predictions["bounded_post"][positions].astype(np.float32),
        )
    tmp.replace(npz_path)

    curve_path = out / "replay_fixture_curves.npz"
    tmp_curve = curve_path.with_suffix(".npz.tmp")
    with tmp_curve.open("wb") as handle:
        payload = {}
        for name, logits in predictions.items():
            curves = curves_from_logits(logits[positions])
            for curve_name in ("pfs", "progression_cif", "death_cif", "switch_cif", "event_free"):
                payload[f"{name}__{curve_name}"] = curves[curve_name].astype(np.float32)
        np.savez_compressed(handle, **payload)
    tmp_curve.replace(curve_path)

    manifest = {
        "status": "FROZEN_DEVELOPMENT_REPLAY_FIXTURE",
        "rows": int(len(fixture_index)),
        "selection": "32 validation rows with lexicographically smallest SHA256(patient_id|scan_episode_id|landmark_day)",
        "files": {
            "replay_fixture_index.parquet": sha256_file(out / "replay_fixture_index.parquet"),
            "replay_fixture_logits.npz": sha256_file(npz_path),
            "replay_fixture_curves.npz": sha256_file(curve_path),
        },
    }
    atomic_json(out / "replay_fixture_manifest.json", manifest)
    return manifest


def methodology_checks() -> dict[str, Any]:
    rng = np.random.default_rng(20261007)
    logits = rng.normal(size=(128, 24, 4))
    curves = curves_from_logits(logits)
    invariants = validate_curve_invariants(curves)

    switch_logits = np.full((1, 24, 4), -8.0, dtype=float)
    switch_logits[:, :, 0] = 4.0
    switch_logits[:, 0, 3] = 8.0
    switch_curves = curves_from_logits(switch_logits)
    switch_distinct = bool(
        switch_curves["switch_cif"][0, 0] > 0.9
        and switch_curves["progression_cif"][0, 0] < 0.01
        and switch_curves["death_cif"][0, 0] < 0.01
    )

    simple = np.zeros((2, 24, 4), dtype=float)
    q0 = 0.8
    qe = (1.0 - q0) / 3.0
    simple[:, :, 0] = np.log(q0)
    simple[:, :, 1:] = np.log(qe)
    times = np.asarray([MONTH_DAYS, 0.5 * MONTH_DAYS])
    causes = np.asarray([0, 0])
    legacy = legacy_nll_per_row(simple, times, causes)
    fractional = fractional_censor_nll_per_row(simple, times, causes)
    exact_boundary_equal = bool(abs(legacy[0] - fractional[0]) < 1e-12)
    interior_fraction_correct = bool(abs(fractional[1] - 0.5 * legacy[1]) < 1e-12)

    zero_time_rejected = False
    try:
        legacy_nll_per_row(simple[:1], np.asarray([0.0]), np.asarray([1]))
    except ValueError:
        zero_time_rejected = True

    bootstrap_source = pd.DataFrame({"patient_id": ["a", "b", "c"], "x": [1, 2, 3]})
    sampled = resample_patient_clusters(bootstrap_source, np.random.default_rng(0))
    bootstrap_draws_preserved = bool(
        sampled["_bootstrap_cluster_draw"].nunique() == 3
        and sampled["_bootstrap_source_patient"].nunique() < 3
        and len(sampled) == 3
    )

    n = 1000
    synthetic = pd.DataFrame({
        "patient_id": [f"p{i}" for i in range(n)],
        "survival_time_days": np.where(np.arange(n) < 300, 60.0, 730.0),
        "survival_cause": np.where(np.arange(n) < 300, CAUSE_PROGRESSION, CAUSE_NO_EVENT_OR_CENSOR),
    })
    km = fit_censoring_km(synthetic["survival_time_days"].to_numpy(), synthetic["survival_cause"].to_numpy())
    from metrics import brier_at_horizon
    good_pfs = np.where(np.arange(n) < 300, 0.1, 0.9)
    bad_pfs = 1.0 - good_pfs
    good = brier_at_horizon(synthetic, good_pfs, 3, km, patient_balanced=True)["brier"]
    bad = brier_at_horizon(synthetic, bad_pfs, 3, km, patient_balanced=True)["brier"]
    synthetic_ordering = bool(good < bad)

    patient_delta = pd.Series([0.1, 0.2, -0.05, 0.3], index=["a", "b", "c", "d"])
    bootstrap_summary = cluster_bootstrap_mean_delta(patient_delta, repetitions=200, seed=20261007)

    checks = {
        "probability_and_curve_invariants": invariants,
        "switch_remains_distinct": switch_distinct,
        "exact_bin_censor_legacy_equals_fractional": exact_boundary_equal,
        "interior_bin_censor_fractional_exposure_correct": interior_fraction_correct,
        "zero_time_rejected": zero_time_rejected,
        "bootstrap_duplicate_cluster_draws_preserved": bootstrap_draws_preserved,
        "synthetic_brier_metric_ordering": synthetic_ordering,
        "synthetic_good_brier": float(good),
        "synthetic_bad_brier": float(bad),
        "bootstrap_smoke": bootstrap_summary,
        "month_days": MONTH_DAYS,
        "administrative_horizon_days": ADMIN_HORIZON_DAYS,
    }
    scalar_pass = [switch_distinct, exact_boundary_equal, interior_fraction_correct, zero_time_rejected, bootstrap_draws_preserved, synthetic_ordering]
    checks["status"] = "PASS" if invariants["status"] == "PASS" and all(scalar_pass) else "FAIL"
    return checks


def assert_access_guard(policy: AccessPolicy, repo: Path) -> dict[str, Any]:
    should_block = [
        repo / "data/external_sources/bpc_dfci/example.csv",
        repo / "data/external_sources/VICC/example.parquet",
        repo / "artifacts/checkpoint7r3g1_external_prediction_freeze/predictions.parquet",
        repo / "artifacts/checkpoint7b1_external_endpoint_freeze/external_landmark_endpoints.parquet",
        repo / "data/external_sources/bpc_brca_1_0_public/clinical_data/regimen_cancer_level_dataset.csv",
    ]
    blocked = []
    for path in should_block:
        try:
            policy.assert_allowed(path, "guard self-test")
            blocked.append(False)
        except AccessDeniedError:
            blocked.append(True)
    allowed = policy.assert_allowed(
        repo / "artifacts/checkpoint1/canonical/bpc_msk_scan_audit.parquet",
        "guard self-test",
    )
    return {
        "status": "PASS" if all(blocked) and allowed is not None else "FAIL",
        "blocked_cases": int(sum(blocked)),
        "expected_blocked_cases": len(should_block),
        "bpc_msk_schema_positive_control_allowed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="OncoTwin V2-00: establish immutable development benchmark")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    repo = args.repo.expanduser().resolve()
    out = (args.out or (repo / "artifacts/dynamic_scan_v2/v2_00")).expanduser()
    if not out.is_absolute():
        out = (repo / out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    config_root = repo / "configs/dynamic_scan_v2"
    policy = AccessPolicy.from_json(repo, config_root / "data_access_policy.json")
    development_protocol = json.loads((config_root / "development_protocol.json").read_text(encoding="utf-8"))
    metrics_contract = json.loads((config_root / "metrics_contract.json").read_text(encoding="utf-8"))

    print("[V2-00] repo", repo, flush=True)
    print("[V2-00] out", out, flush=True)
    print("[V2-00] host", socket.gethostname(), flush=True)

    guard_report = assert_access_guard(policy, repo)
    atomic_json(out / "access_guard_qc.json", guard_report)
    if guard_report["status"] != "PASS":
        raise RuntimeError("Access guard self-test failed")
    print("[V2-00] access guard PASS", flush=True)

    lineage = build_legacy_reference_manifest(repo, out, policy)
    if lineage["status"] != "PASS":
        raise RuntimeError("Legacy lineage verification failed")
    print("[V2-00] legacy lineage PASS", flush=True)

    interface = audit_prepared_interface(repo, out, policy)
    if interface["status"] != "PASS":
        raise RuntimeError("Prepared-interface replay failed")
    print("[V2-00] prepared interface PASS", flush=True)

    _, exposure_summary = build_patient_exposure_registry(repo, out, policy)
    print("[V2-00] exposure registry PASS", flush=True)

    _, fold_summary = build_development_folds(repo, out, policy, development_protocol)
    print("[V2-00] development folds PASS", fold_summary["fold_patient_counts"], flush=True)

    source_capabilities = build_source_capabilities(repo, out, policy)
    if source_capabilities["status"] != "PASS":
        raise RuntimeError("Source capability audit failed")
    print("[V2-00] source capabilities PASS", flush=True)

    methods = methodology_checks()
    atomic_json(out / "methodology_tests.json", methods)
    if methods["status"] != "PASS":
        raise RuntimeError("Methodology checks failed")
    print("[V2-00] methodology tests PASS", flush=True)

    data = load_r2_prepared(repo, policy)
    index = data["index"]
    train = index.loc[(index["split"].astype(str) == "train") & index["survival_mask"].astype(bool)].copy()
    km = fit_censoring_km(
        train["survival_time_days"].to_numpy(dtype=float),
        train["survival_cause"].astype(int).to_numpy(),
    )
    atomic_json(out / "censoring_km_contract.json", {
        "status": "FROZEN_FOR_V2_DEVELOPMENT_METRICS",
        "source": "CKPT7R2 CHORD original-training survival rows",
        "rows": int(len(train)),
        "patients": int(train["patient_id"].nunique()),
        "cause_counts": {str(k): int(v) for k, v in train["survival_cause"].astype(int).value_counts().sort_index().items()},
        "km_points": int(len(km.times)),
        "final_km_survival": float(km.survival[-1]) if len(km.survival) else 1.0,
    })

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("[V2-00] replay device", device, flush=True)
    _, checkpoint, model = load_r2_model(repo, policy, device)
    print("[V2-00] R2 model parameters", sum(p.numel() for p in model.parameters()), flush=True)

    replay_reports = {}
    validation_predictions = None
    validation_frame = None
    for split in ("val", "test"):
        report, predictions, frame = replay_split(
            repo, policy, data, model, split, device, km, metrics_contract
        )
        replay_reports[split] = report
        atomic_json(out / f"legacy_replay_{split}.json", report)
        print(
            f"[V2-00] replay {split} {report['status']} rows={report['rows']} patients={report['patients']}",
            flush=True,
        )
        if report["status"] != "PASS":
            raise RuntimeError(f"Legacy replay failed for {split}")
        if split == "val":
            validation_predictions = predictions
            validation_frame = frame

    assert validation_predictions is not None and validation_frame is not None
    fixture = write_replay_fixture(out, validation_frame, validation_predictions)

    partial_bin = {
        "status": "DOCUMENTED_VERSIONED_CORRECTION",
        "legacy_behavior": metrics_contract["likelihoods"]["legacy_replay"],
        "v2_behavior": metrics_contract["likelihoods"]["v2_supporting"],
        "development_rescoring": {},
        "primary_metric_changed": False,
    }
    for split, report in replay_reports.items():
        partial_bin["development_rescoring"][split] = {}
        for variant in ("pre", "full_post", "bounded_post"):
            m = report["metrics"][variant]
            partial_bin["development_rescoring"][split][variant] = {
                "legacy_patient_nll": m["legacy_patient_mean_nll"],
                "fractional_censor_patient_nll_v1": m["fractional_censor_patient_mean_nll_v1"],
                "legacy_minus_fractional": m["legacy_minus_fractional_patient_nll"],
                "patient_brier_4h_mean_unchanged_primary": m["legacy_patient_integrated_brier_4h"],
            }
    atomic_json(out / "partial_bin_likelihood_audit.json", partial_bin)

    copied_configs = {}
    for name in ("data_access_policy.json", "development_protocol.json", "metrics_contract.json"):
        source = config_root / name
        destination = out / name
        atomic_text(destination, source.read_text(encoding="utf-8"))
        copied_configs[name] = sha256_file(destination)

    acceptance = {
        "legacy_reference_identity_unambiguous": lineage["status"] == "PASS",
        "prepared_feature_replay_pass": interface["status"] == "PASS",
        "development_replay_pass": replay_reports["val"]["status"] == "PASS",
        "historical_test_identity_replay_pass_descriptive_only": replay_reports["test"]["status"] == "PASS",
        "external_access_guard_pass": guard_report["status"] == "PASS",
        "source_availability_explicit": source_capabilities["status"] == "PASS",
        "patient_exposure_registry_built": exposure_summary["status"] == "PASS",
        "three_outcome_blind_development_folds_frozen": fold_summary["status"] == "PASS",
        "methodology_tests_pass": methods["status"] == "PASS",
        "single_locked_primary_estimand_and_metric": True,
        "legacy_partial_bin_approximation_preserved_not_silently_overwritten": True,
    }
    overall = "PASS" if all(acceptance.values()) else "FAIL"

    result_packet = {
        "checkpoint": "V2-00",
        "status": overall,
        "repo_git_head": git_value(repo, "rev-parse", "HEAD"),
        "device": str(device),
        "acceptance": acceptance,
        "development_folds": fold_summary,
        "replay": {
            split: {
                "status": replay_reports[split]["status"],
                "rows": replay_reports[split]["rows"],
                "patients": replay_reports[split]["patients"],
                "pre_patient_brier_4h": replay_reports[split]["metrics"]["pre"]["legacy_patient_integrated_brier_4h"],
                "bounded_patient_brier_4h": replay_reports[split]["metrics"]["bounded_post"]["legacy_patient_integrated_brier_4h"],
                "pre_legacy_patient_nll": replay_reports[split]["metrics"]["pre"]["legacy_patient_mean_nll"],
                "bounded_legacy_patient_nll": replay_reports[split]["metrics"]["bounded_post"]["legacy_patient_mean_nll"],
                "bounded_fractional_patient_nll_v1": replay_reports[split]["metrics"]["bounded_post"]["fractional_censor_patient_mean_nll_v1"],
            }
            for split in ("val", "test")
        },
        "metric_decision": {
            "primary": "PATIENT_BALANCED_PFS_BRIER_4H_MEAN",
            "supporting_v2": "FRACTIONAL_CENSOR_NLL_V1",
            "legacy_nll": "REPLAY_ONLY",
            "partial_bin_issue_confirmed": True,
            "primary_metric_changed": False,
        },
        "v2_t": {
            "raw_report_text": "NOT_ESTABLISHED",
            "lesion_measurements": "NOT_ESTABLISHED",
            "decision": "DO_NOT_BLOCK_V2_01",
        },
        "next_checkpoint": "V2-01" if overall == "PASS" else "REPAIR_V2_00",
        "external_rows_opened": False,
        "external_predictions_regenerated": False,
        "fixture": fixture,
        "config_hashes": copied_configs,
    }
    atomic_json(out / "result_packet.json", result_packet)

    report_text = f"""# OncoTwin V2-00 decision report

Status: **{overall}**

## Acceptance

""" + "\n".join(
        f"- {'PASS' if value else 'FAIL'}: `{key}`" for key, value in acceptance.items()
    ) + f"""

## Locked development comparison

- Population: original CHORD train + validation patients, repartitioned outcome-blind into three patient-disjoint V2 development folds.
- Original CHORD test remains historical-test/read-only and is not a V2 model-selection split.
- Primary metric: patient-balanced PFS Brier mean over 3/6/12/18 months.
- Switch remains a separate competing cause.
- Genomics remains available only when `availability_day < landmark_day`.
- Absolute treatment-line identity remains prohibited model-facing.

## Likelihood audit

The historical CKPT5 NLL requires a censored row to survive the complete containing month (`ceil(time / month_width)`). V2-00 preserves that exact formula as `LEGACY_MONTH_CEILING_NLL` for replay. Future V2 comparisons use the separately versioned `FRACTIONAL_CENSOR_NLL_V1` only as a supporting metric, with fractional exposure for interior-bin censoring. The primary Brier metric is unchanged.

## Replay

- Validation: {replay_reports['val']['status']}
- Historical test identity check only: {replay_reports['test']['status']}
- Replay fixture: {fixture['rows']} validation rows frozen with logits and cause-specific curves.

## V2-T capability decision

No authorized raw radiology report text or lesion-level measurement source was established. V2-T remains disabled and does not block V2-01.

## Next action

Proceed to V2-01 only if this report and `result_packet.json` are PASS. Build paired longitudinal scan-change representations on the frozen V2 development folds; do not reopen DFCI/VICC data.
"""
    atomic_text(out / "decision_report.md", report_text)

    output_manifest = {}
    for path in sorted(out.iterdir()):
        if path.is_file() and not path.name.endswith(".tmp"):
            output_manifest[path.name] = {"sha256": sha256_file(path), "bytes": int(path.stat().st_size)}
    atomic_json(out / "manifest.json", {
        "checkpoint": "V2-00",
        "status": overall,
        "files": output_manifest,
        "external_outcomes_opened": False,
        "external_predictions_regenerated": False,
    })

    print("V2_00_STATUS=" + overall)
    print("V2_00_RESULT_PACKET=" + str(out / "result_packet.json"))
    print("V2_00_DECISION_REPORT=" + str(out / "decision_report.md"))
    print("V2_00_MANIFEST=" + str(out / "manifest.json"))
    print("V2_00_NEXT=" + result_packet["next_checkpoint"])
    return 0 if overall == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
