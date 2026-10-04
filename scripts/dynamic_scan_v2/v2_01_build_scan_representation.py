#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from access import AccessPolicy
from common import atomic_json, atomic_parquet, atomic_text, sha256_file
from paired_scan_encoder import PairedScanEncoder
from scan_representation import (
    MODALITY_PATTERNS,
    REGIONS,
    SITE_PATTERNS,
    STATE_NAMES,
    _bool_or_missing,
    build_paired_scan_representation,
    engineered_control_matrix,
    representation_feature_schema,
)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}.npz")
    np.savez_compressed(tmp, **arrays)
    with np.load(tmp) as check:
        if set(check.files) != set(arrays):
            raise RuntimeError(f"NPZ verification failed for {path}")
        for name, expected in arrays.items():
            if check[name].shape != expected.shape:
                raise RuntimeError(f"NPZ shape verification failed for {path}:{name}")
    tmp.replace(path)


def _allclose_exact(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True))


def _representation_arrays(result) -> dict[str, np.ndarray]:
    return {
        "history_core": result.history_core.values,
        "history_core_mask": result.history_core.masks,
        "utilization_core": result.utilization_core.values,
        "utilization_core_mask": result.utilization_core.masks,
        "current_core": result.current_core.values,
        "current_core_mask": result.current_core.masks,
        "change_core": result.change_core.values,
        "change_core_mask": result.change_core.masks,
        "optional_history": result.optional_history.values,
        "optional_history_mask": result.optional_history.masks,
        "optional_history_recency": result.optional_history_recency.values,
        "optional_history_recency_mask": result.optional_history_recency.masks,
        "optional_current": result.optional_current.values,
        "optional_current_mask": result.optional_current.masks,
        "stream_availability": result.stream_availability.values,
        "stream_availability_mask": result.stream_availability.masks,
    }


def load_inputs(repo: Path, policy: AccessPolicy) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scans = policy.read_parquet(
        repo / "artifacts/checkpoint1/canonical/chord_breast_scan_episodes_w3.parquet"
    )
    anchors = policy.read_parquet(
        repo / "artifacts/checkpoint7r2_transport_safe_supervised/prepared/scan_index.parquet"
    )
    folds = policy.read_parquet(
        repo / "artifacts/dynamic_scan_v2/v2_00/development_folds.parquet"
    )

    for frame, label in [(scans, "authoritative scans"), (anchors, "R2 anchors")]:
        if frame[["patient_id", "scan_episode_id"]].duplicated().any():
            raise RuntimeError(f"Duplicate patient+episode keys in {label}")

    if "development_fold" not in folds.columns:
        candidates = [c for c in folds.columns if c.lower() in {"v2_fold", "fold", "development_fold"}]
        if len(candidates) != 1:
            raise RuntimeError(f"Cannot identify development fold column: {folds.columns.tolist()}")
        folds = folds.rename(columns={candidates[0]: "development_fold"})
    folds["patient_id"] = folds["patient_id"].astype(str).str.strip()
    if "v2_role" in folds.columns:
        historical = folds["v2_role"].astype(str).str.lower().eq("historical_test_read_only")
        folds.loc[historical, "development_fold"] = np.nan
    else:
        folds.loc[pd.to_numeric(folds["development_fold"], errors="coerce") < 0, "development_fold"] = np.nan
    folds = folds[["patient_id", "development_fold"]].copy()
    if folds["patient_id"].duplicated().any():
        raise RuntimeError("Development folds are not one row per patient")

    anchors["patient_id"] = anchors["patient_id"].astype(str).str.strip()
    anchors = anchors.merge(folds, on="patient_id", how="left", validate="many_to_one")

    is_test = anchors["split"].astype(str).str.lower().eq("test")
    if anchors.loc[is_test, "development_fold"].notna().any():
        raise RuntimeError("Historical test patient received a V2 development fold")
    is_dev = anchors["split"].astype(str).str.lower().isin(["train", "val"])
    if anchors.loc[is_dev, "development_fold"].isna().any():
        raise RuntimeError("Development anchor missing frozen V2 fold")

    return scans, anchors, folds


def _legacy_norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


LEGACY_OBSERVED_PATTERNS: tuple[tuple[str, str], ...] = (
    ("modality_ct", r"\bCT\b|COMPUTED TOMOGRAPH"),
    ("modality_pet", r"\bPET\b|PET-CT"),
    ("modality_mr", r"\bMRI?\b|MAGNETIC"),
    ("modality_bone_scan", r"BONE SCAN"),
    ("site_bone", r"\bBONE\b|SKELET|VERTEBR|RIB"),
    ("site_liver", r"\bLIVER\b"),
    ("site_lung", r"\bLUNG\b|PULMON"),
    ("site_brain", r"\bBRAIN\b|\bCNS\b|CEREBR"),
    ("site_lymph", r"LYMPH"),
    ("site_pleura", r"PLEURA"),
)


def _legacy_text_surface_features(
    frame: pd.DataFrame,
    *,
    candidate_columns: list[str] | None = None,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Reproduce CKPT5's observed-text regex surface exactly.

    This helper exists only for the V2-01 audit. It is *not* the V2 feature
    definition. CKPT5 concatenated every column whose normalized name contained
    procedure/modality/source_specific/tumor_site/tumor_sites/site and then
    applied both modality and site regexes to that surface. Importantly, the
    literal token ``modality`` does not match the canonical column name
    ``modalities_json``; the historical selector therefore skipped the actual
    modality column. V2 explicitly corrects that provenance bug.
    """
    if candidate_columns is None:
        candidate_columns = []
        for column in frame.columns:
            normalized = _legacy_norm(column)
            if any(
                token in normalized
                for token in (
                    "procedure",
                    "modality",
                    "source_specific",
                    "tumor_site",
                    "tumor_sites",
                    "site",
                )
            ):
                candidate_columns.append(column)
    else:
        missing = [column for column in candidate_columns if column not in frame.columns]
        if missing:
            raise RuntimeError(f"Legacy crosswalk columns missing: {missing}")

    if candidate_columns:
        combined = (
            frame[candidate_columns]
            .fillna("")
            .astype(str)
            .agg(" | ".join, axis=1)
            .str.upper()
        )
    else:
        combined = pd.Series("", index=frame.index, dtype="object")

    names: list[str] = []
    columns: list[np.ndarray] = []
    for name, pattern in LEGACY_OBSERVED_PATTERNS:
        names.append(name)
        columns.append(
            combined.str.contains(pattern, regex=True, na=False).to_numpy(dtype=np.float32)
        )
    matrix = np.column_stack(columns).astype(np.float32, copy=False)
    return matrix, names, list(candidate_columns)


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise RuntimeError(f"Crosswalk shape mismatch: {a.shape} vs {b.shape}")
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32))))


def _feature_mismatch_counts(a: np.ndarray, b: np.ndarray, names: list[str]) -> dict[str, int]:
    mismatch = np.asarray(a > 0.5) != np.asarray(b > 0.5)
    return {name: int(mismatch[:, j].sum()) for j, name in enumerate(names)}


def legacy_crosswalk(repo: Path, policy: AccessPolicy, result) -> dict[str, Any]:
    legacy = policy.np_load(
        repo / "artifacts/checkpoint7r2_transport_safe_supervised/prepared/current_scan_features_f32.npy",
        mmap_mode="r",
    )
    schema = policy.read_json(
        repo / "artifacts/checkpoint7r2_transport_safe_supervised/prepared/feature_schema.json"
    )
    names = list(schema["current_scan"])
    if legacy.shape[0] != len(result.index):
        raise RuntimeError(f"Legacy current-scan row mismatch: {legacy.shape[0]} vs {len(result.index)}")

    expected = [
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
    if names != expected:
        raise RuntimeError(f"Legacy current-scan feature order changed: {names}")

    legacy = np.asarray(legacy, dtype=np.float32)
    core_diff = _max_abs(legacy[:, :8], result.current_core.values)

    # Rebuild the exact CKPT5 combined-text surface from the authoritative MBC
    # current-scan rows, aligned by explicit patient+episode keys. This audit is
    # independent of V2's source-separated parser and proves that any V2/legacy
    # discrepancy has a concrete legacy-text-surface explanation.
    canonical = policy.read_parquet(
        repo / "artifacts/checkpoint1/canonical/chord_mbc_scan_episodes_w3.parquet"
    ).copy()
    canonical["patient_id"] = canonical["patient_id"].astype(str).str.strip()
    canonical["scan_episode_id"] = canonical["scan_episode_id"].astype(str).str.strip()
    if canonical[["patient_id", "scan_episode_id"]].duplicated().any():
        raise RuntimeError("Authoritative MBC scan keys are not unique")

    key_frame = result.index[["patient_id", "scan_episode_id"]].copy()
    key_frame["patient_id"] = key_frame["patient_id"].astype(str).str.strip()
    key_frame["scan_episode_id"] = key_frame["scan_episode_id"].astype(str).str.strip()
    aligned = key_frame.merge(
        canonical,
        on=["patient_id", "scan_episode_id"],
        how="left",
        validate="one_to_one",
        indicator=True,
        sort=False,
    )
    if not aligned["_merge"].eq("both").all():
        missing = aligned.loc[aligned["_merge"] != "both", ["patient_id", "scan_episode_id"]].head(5)
        raise RuntimeError(f"Crosswalk keys missing from authoritative MBC scans: {missing.to_dict('records')}")
    aligned = aligned.drop(columns=["_merge"])

    rebuilt_legacy_optional, rebuilt_names, legacy_text_candidates = _legacy_text_surface_features(aligned)
    if rebuilt_names != expected[8:]:
        raise RuntimeError(f"Internal legacy regex order drifted: {rebuilt_names}")
    legacy_optional = legacy[:, 8:18]
    stored_legacy_rebuild_diff = _max_abs(legacy_optional, rebuilt_legacy_optional)

    # Source-specific reconstructions use the same legacy regexes but restrict
    # the text surface to the source that V2 declares as provenance.
    modality_source_all, _, _ = _legacy_text_surface_features(
        aligned, candidate_columns=["modalities_json"]
    )
    site_source_all, _, _ = _legacy_text_surface_features(
        aligned, candidate_columns=["tumor_sites_json"]
    )
    modality_source = modality_source_all[:, :4]
    site_source = site_source_all[:, 4:10]

    v2_modality = np.asarray(result.optional_current.values[:, :4], dtype=np.float32)
    v2_site = np.asarray(result.optional_current.values[:, 4:10], dtype=np.float32)
    v2_modality_source_diff = _max_abs(v2_modality, modality_source)
    v2_site_source_diff = _max_abs(v2_site, site_source)

    legacy_modality = legacy[:, 8:12]
    legacy_site = legacy[:, 12:18]
    legacy_vs_v2_modality_diff = _max_abs(legacy_modality, v2_modality)
    legacy_vs_v2_site_diff = _max_abs(legacy_site, v2_site)

    # A source-separation difference is accepted only when it is fully explained:
    # (1) stored CKPT7R2 exactly replays the reconstructed CKPT5 combined surface,
    # (2) V2 exactly replays the corresponding source-specific surface, and
    # (3) the observed V2/legacy mismatch mask equals the mismatch induced solely
    #     by restricting the legacy combined surface to that declared source.
    modality_observed_mismatch = (legacy_modality > 0.5) != (v2_modality > 0.5)
    modality_explained_mismatch = (
        rebuilt_legacy_optional[:, :4] > 0.5
    ) != (modality_source > 0.5)
    site_observed_mismatch = (legacy_site > 0.5) != (v2_site > 0.5)
    site_explained_mismatch = (
        rebuilt_legacy_optional[:, 4:10] > 0.5
    ) != (site_source > 0.5)

    modality_differences_fully_explained = bool(
        np.array_equal(modality_observed_mismatch, modality_explained_mismatch)
    )
    site_differences_fully_explained = bool(
        np.array_equal(site_observed_mismatch, site_explained_mismatch)
    )

    optional_names = expected[8:]
    modality_names = optional_names[:4]
    site_names = optional_names[4:]

    historical_selector_bug_confirmed = (
        "modalities_json" not in legacy_text_candidates
        and "tumor_sites_json" in legacy_text_candidates
    )

    status_checks = {
        "portable_core_exact": core_diff == 0.0,
        "historical_ckpt5_modalities_json_selector_bug_confirmed": historical_selector_bug_confirmed,
        "stored_legacy_optional_surface_reconstructed_exactly": stored_legacy_rebuild_diff == 0.0,
        "v2_modality_matches_declared_modality_source_exactly": v2_modality_source_diff == 0.0,
        "v2_site_positive_matches_declared_site_source_exactly": v2_site_source_diff == 0.0,
        "legacy_vs_v2_modality_differences_fully_explained_by_source_separation": modality_differences_fully_explained,
        "legacy_vs_v2_site_differences_fully_explained_by_source_separation": site_differences_fully_explained,
    }
    status = "PASS" if all(status_checks.values()) else "FAIL"

    return {
        "status": status,
        "crosswalk_version": "v2_01_2_source_separated_audit",
        "legacy_feature_order": names,
        "legacy_text_candidate_columns": legacy_text_candidates,
        "historical_selector_finding": {
            "modalities_json_selected_by_ckpt5": "modalities_json" in legacy_text_candidates,
            "tumor_sites_json_selected_by_ckpt5": "tumor_sites_json" in legacy_text_candidates,
            "explanation": (
                "CKPT5 searched normalized column names for the literal substring 'modality'. "
                "The canonical field is named 'modalities_json', so it was not selected; "
                "legacy modality regexes therefore did not consume the actual modality field."
            ),
        },
        "portable_core_first_8_max_abs": core_diff,
        "stored_legacy_optional_rebuild_max_abs": stored_legacy_rebuild_diff,
        "legacy_vs_v2_modality_max_abs": legacy_vs_v2_modality_diff,
        "legacy_vs_v2_site_positive_max_abs": legacy_vs_v2_site_diff,
        "v2_vs_modality_source_only_max_abs": v2_modality_source_diff,
        "v2_vs_site_source_only_max_abs": v2_site_source_diff,
        "status_checks": status_checks,
        "modality_source_separation": {
            "legacy_vs_v2_mismatch_counts": _feature_mismatch_counts(
                legacy_modality, v2_modality, modality_names
            ),
            "combined_surface_vs_modality_source_only_mismatch_counts": _feature_mismatch_counts(
                rebuilt_legacy_optional[:, :4], modality_source, modality_names
            ),
            "rule": (
                "CKPT5 modality regexes operated on its legacy text-candidate surface, which "
                "excluded canonical modalities_json because the selector looked for singular "
                "'modality'. V2 modality values are explicitly sourced from modalities_json."
            ),
        },
        "site_positive_source_separation": {
            "legacy_vs_v2_mismatch_counts": _feature_mismatch_counts(
                legacy_site, v2_site, site_names
            ),
            "combined_surface_vs_tumor_site_source_only_mismatch_counts": _feature_mismatch_counts(
                rebuilt_legacy_optional[:, 4:10], site_source, site_names
            ),
            "rule": "CKPT5 searched the combined legacy observed-text surface; V2 site-positive values are sourced only from tumor_sites_json.",
            "disease_negative_inference": False,
        },
        "semantic_change": (
            "V2 preserves the exact state/coverage core, explicitly repairs the historical CKPT5 "
            "modalities_json selector bug, and source-separates optional modality/site-positive evidence. "
            "A legacy/V2 optional-bit difference is accepted only when exact reconstruction proves it "
            "is induced solely by the historical text selector versus the declared V2 source."
        ),
    }


def actual_invariance_checks(scans: pd.DataFrame, anchors: pd.DataFrame, result) -> dict[str, Any]:
    # Keep actual-data perturbation checks bounded; persistent unit tests cover the
    # full synthetic edge cases.
    sample = anchors.sort_values(["patient_id", "landmark_day", "scan_episode_id"], kind="mergesort").head(256).copy()
    base = build_paired_scan_representation(scans, sample)

    shuffled = build_paired_scan_representation(scans.sample(frac=1.0, random_state=20261001), sample)
    replay_exact = all(
        _allclose_exact(a, b)
        for a, b in zip(_representation_arrays(base).values(), _representation_arrays(shuffled).values())
    ) and base.index.equals(shuffled.index)

    # Append a future scan for the first sample patient. Earlier anchors must be invariant.
    first_pid = str(sample.iloc[0]["patient_id"])
    patient_rows = scans[scans["patient_id"].astype(str) == first_pid].sort_values("landmark_day")
    template = patient_rows.iloc[-1].copy()
    future_day = int(patient_rows["landmark_day"].max()) + 10000
    template["scan_episode_id"] = f"{first_pid}::V2_01_SYNTH_FUTURE"
    template["episode_start_day"] = future_day
    template["episode_end_day"] = future_day
    template["landmark_day"] = future_day
    template["source_days_json"] = json.dumps([future_day])
    augmented_scans = pd.concat([scans, pd.DataFrame([template])], ignore_index=True)
    future = build_paired_scan_representation(augmented_scans, sample)
    future_append_invariant = all(
        _allclose_exact(a, b)
        for a, b in zip(_representation_arrays(base).values(), _representation_arrays(future).values())
    ) and base.index.equals(future.index)

    # Model-facing V2 scan builder must ignore semantic line labels entirely.
    perturbed_anchor = sample.copy()
    perturbed_anchor["line"] = np.arange(len(perturbed_anchor)) + 99991
    perturbed_anchor["treatment_line"] = np.arange(len(perturbed_anchor))[::-1] + 88881
    line_perturbed = build_paired_scan_representation(scans, perturbed_anchor)
    line_invariant = all(
        _allclose_exact(a, b)
        for a, b in zip(_representation_arrays(base).values(), _representation_arrays(line_perturbed).values())
    ) and base.index.equals(line_perturbed.index)

    # Extra unrelated same-day columns/events cannot enter because the feature builder
    # consumes only the declared authoritative scan fields.
    scans_extra = scans.copy()
    scans_extra["UNRELATED_SAME_DAY_EVENT_PAYLOAD"] = "ADVERSARIAL"
    unrelated = build_paired_scan_representation(scans_extra, sample)
    unrelated_invariant = all(
        _allclose_exact(a, b)
        for a, b in zip(_representation_arrays(base).values(), _representation_arrays(unrelated).values())
    )

    # Duplicate component burden: count magnitudes are not model-facing, only source
    # presence booleans. Multiplying positive counts must be invariant.
    dup_scans = scans.copy()
    target_key = (str(sample.iloc[0]["patient_id"]), str(sample.iloc[0]["scan_episode_id"]))
    mask = (dup_scans["patient_id"].astype(str) == target_key[0]) & (dup_scans["scan_episode_id"].astype(str) == target_key[1])
    raw = json.loads(str(dup_scans.loc[mask, "component_counts_json"].iloc[0]))
    dup_scans.loc[mask, "component_counts_json"] = json.dumps({k: (int(v) * 17 if int(v) > 0 else 0) for k, v in raw.items()})
    dup = build_paired_scan_representation(dup_scans, sample)
    duplicate_count_invariant = all(
        _allclose_exact(a, b)
        for a, b in zip(_representation_arrays(base).values(), _representation_arrays(dup).values())
    )

    # Current scan perturbation: pick a row with previous history so history must remain
    # bit-identical while current/change must respond.
    idx_with_prev = base.index.index[base.index["previous_scan_episode_id"].notna()].tolist()
    if not idx_with_prev:
        raise RuntimeError("Actual-data sample unexpectedly has no row with prior scan")
    local_i = idx_with_prev[0]
    selected = sample.iloc[[local_i]].copy()
    selected_key = (str(selected.iloc[0]["patient_id"]), str(selected.iloc[0]["scan_episode_id"]))
    modified_scans = scans.copy()
    cmask = (modified_scans["patient_id"].astype(str) == selected_key[0]) & (modified_scans["scan_episode_id"].astype(str) == selected_key[1])
    old_state = str(modified_scans.loc[cmask, "progression_state_3"].iloc[0]).upper()
    new_state = "INDETERMINATE" if old_state != "INDETERMINATE" else "NON_PROGRESSIVE"
    modified_scans.loc[cmask, "progression_state_3"] = new_state
    modified_scans.loc[cmask, "coverage_chest"] = not bool(modified_scans.loc[cmask, "coverage_chest"].iloc[0])
    before_one = build_paired_scan_representation(scans, selected)
    after_one = build_paired_scan_representation(modified_scans, selected)
    current_perturb_history_identical = (
        _allclose_exact(before_one.history_core.values, after_one.history_core.values)
        and _allclose_exact(before_one.history_core.masks, after_one.history_core.masks)
        and _allclose_exact(before_one.utilization_core.values, after_one.utilization_core.values)
        and _allclose_exact(before_one.utilization_core.masks, after_one.utilization_core.masks)
        and _allclose_exact(before_one.optional_history.values, after_one.optional_history.values)
        and _allclose_exact(before_one.optional_history.masks, after_one.optional_history.masks)
        and _allclose_exact(before_one.optional_history_recency.values, after_one.optional_history_recency.values)
        and _allclose_exact(before_one.optional_history_recency.masks, after_one.optional_history_recency.masks)
    )
    current_perturb_post_changes = (
        not _allclose_exact(before_one.current_core.values, after_one.current_core.values)
        and not _allclose_exact(before_one.change_core.values, after_one.change_core.values)
    )

    if result.index["previous_landmark_day"].notna().any():
        has_prev = result.index["previous_landmark_day"].notna()
        strict_episode_cutoff = bool((
            result.index.loc[has_prev, "previous_landmark_day"].astype(float).to_numpy()
            < result.index.loc[has_prev, "episode_start_day"].astype(float).to_numpy()
        ).all())
    else:
        strict_episode_cutoff = True

    zero_value, zero_mask = _bool_or_missing(False)
    missing_value, missing_mask = _bool_or_missing(None)
    zero_distinct_from_missing = zero_value == missing_value == 0.0 and zero_mask == 1.0 and missing_mask == 0.0

    checks = {
        "input_order_exact_replay": replay_exact,
        "append_future_scan_invariant": future_append_invariant,
        "line_label_perturbation_invariant": line_invariant,
        "unrelated_same_day_extra_field_invariant": unrelated_invariant,
        "duplicate_component_count_invariant": duplicate_count_invariant,
        "current_scan_perturbation_history_identical": current_perturb_history_identical,
        "current_scan_perturbation_changes_post_blocks": current_perturb_post_changes,
        "strict_pre_cutoff_before_episode_start": strict_episode_cutoff,
        "zero_numeric_distinct_from_missing": zero_distinct_from_missing,
    }
    checks["status"] = "PASS" if all(checks.values()) else "FAIL"
    return checks


def encoder_smoke(result) -> dict[str, Any]:
    torch.manual_seed(20261001)
    encoder = PairedScanEncoder(
        history_dim=result.history_core.values.shape[1],
        utilization_dim=result.utilization_core.values.shape[1],
        optional_history_dim=result.optional_history.values.shape[1],
        optional_history_recency_dim=result.optional_history_recency.values.shape[1],
        current_dim=result.current_core.values.shape[1],
        change_dim=result.change_core.values.shape[1],
        optional_current_dim=result.optional_current.values.shape[1],
        stream_dim=result.stream_availability.values.shape[1],
        hidden_dim=48,
        output_dim=32,
    )
    encoder.eval()

    n = min(64, len(result.index))
    def t(a: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.asarray(a[:n], dtype=np.float32))

    core_only_kwargs = dict(
        history_values=t(result.history_core.values),
        history_masks=t(result.history_core.masks),
        utilization_values=t(result.utilization_core.values),
        utilization_masks=t(result.utilization_core.masks),
        current_values=t(result.current_core.values),
        current_masks=t(result.current_core.masks),
        change_values=t(result.change_core.values),
        change_masks=t(result.change_core.masks),
    )
    with torch.no_grad():
        core_only = encoder(**core_only_kwargs)

        # Full observed-history/current call used to verify the architecture supports
        # optional streams while preserving the same PRE branch.
        full = encoder(
            **core_only_kwargs,
            optional_history_values=t(result.optional_history.values),
            optional_history_masks=t(result.optional_history.masks),
            optional_history_recency_values=t(result.optional_history_recency.values),
            optional_history_recency_masks=t(result.optional_history_recency.masks),
            optional_current_values=t(result.optional_current.values),
            optional_current_masks=t(result.optional_current.masks),
            stream_values=t(result.stream_availability.values),
            stream_masks=t(result.stream_availability.masks),
        )

        # Unavailable optional blocks must ignore arbitrary values.
        zoh = torch.zeros((n, result.optional_history.values.shape[1]))
        zohr = torch.zeros((n, result.optional_history_recency.values.shape[1]))
        zoc = torch.zeros((n, result.optional_current.values.shape[1]))
        zs = torch.zeros((n, result.stream_availability.values.shape[1]))
        a = encoder(
            **core_only_kwargs,
            optional_history_values=torch.randn_like(zoh) * 1000.0,
            optional_history_masks=torch.zeros_like(zoh),
            optional_history_recency_values=torch.randn_like(zohr) * 1000.0,
            optional_history_recency_masks=torch.zeros_like(zohr),
            optional_current_values=torch.randn_like(zoc) * 1000.0,
            optional_current_masks=torch.zeros_like(zoc),
            stream_values=torch.randn_like(zs) * 1000.0,
            stream_masks=torch.zeros_like(zs),
        )
        b = encoder(
            **core_only_kwargs,
            optional_history_values=torch.randn_like(zoh) * 1000.0,
            optional_history_masks=torch.zeros_like(zoh),
            optional_history_recency_values=torch.randn_like(zohr) * 1000.0,
            optional_history_recency_masks=torch.zeros_like(zohr),
            optional_current_values=torch.randn_like(zoc) * 1000.0,
            optional_current_masks=torch.zeros_like(zoc),
            stream_values=torch.randn_like(zs) * 1000.0,
            stream_masks=torch.zeros_like(zs),
        )

    unavailable_invariant = bool(
        torch.equal(a.history_context, b.history_context)
        and torch.equal(a.current_evidence, b.current_evidence)
        and torch.equal(a.change_evidence, b.change_evidence)
        and torch.equal(a.post_evidence, b.post_evidence)
    )
    core_finite = bool(
        torch.isfinite(core_only.history_context).all()
        and torch.isfinite(core_only.current_evidence).all()
        and torch.isfinite(core_only.change_evidence).all()
        and torch.isfinite(core_only.post_evidence).all()
    )
    # Current optional information is not allowed to feed history_context. Compare
    # two calls that share all PRE blocks but differ in all current blocks.
    with torch.no_grad():
        current_a = encoder(
            history_values=t(result.history_core.values),
            history_masks=t(result.history_core.masks),
            utilization_values=t(result.utilization_core.values),
            utilization_masks=t(result.utilization_core.masks),
            optional_history_values=t(result.optional_history.values),
            optional_history_masks=t(result.optional_history.masks),
            optional_history_recency_values=t(result.optional_history_recency.values),
            optional_history_recency_masks=t(result.optional_history_recency.masks),
            current_values=t(result.current_core.values),
            current_masks=t(result.current_core.masks),
            change_values=t(result.change_core.values),
            change_masks=t(result.change_core.masks),
            optional_current_values=t(result.optional_current.values),
            optional_current_masks=t(result.optional_current.masks),
            stream_values=t(result.stream_availability.values),
            stream_masks=t(result.stream_availability.masks),
        )
        current_b = encoder(
            history_values=t(result.history_core.values),
            history_masks=t(result.history_core.masks),
            utilization_values=t(result.utilization_core.values),
            utilization_masks=t(result.utilization_core.masks),
            optional_history_values=t(result.optional_history.values),
            optional_history_masks=t(result.optional_history.masks),
            optional_history_recency_values=t(result.optional_history_recency.values),
            optional_history_recency_masks=t(result.optional_history_recency.masks),
            current_values=torch.randn_like(t(result.current_core.values)),
            current_masks=t(result.current_core.masks),
            change_values=torch.randn_like(t(result.change_core.values)),
            change_masks=t(result.change_core.masks),
            optional_current_values=torch.randn_like(t(result.optional_current.values)),
            optional_current_masks=t(result.optional_current.masks),
            stream_values=torch.randn_like(t(result.stream_availability.values)),
            stream_masks=t(result.stream_availability.masks),
        )
    current_does_not_change_pre = bool(torch.equal(current_a.history_context, current_b.history_context))

    interface = encoder.interface()
    interface.update(
        {
            "core_only_smoke_rows": n,
            "core_only_finite": core_finite,
            "unavailable_optional_adversarial_invariant": unavailable_invariant,
            "current_content_does_not_change_history_context": current_does_not_change_pre,
            "output_shapes": {
                "history_context": list(core_only.history_context.shape),
                "current_evidence": list(core_only.current_evidence.shape),
                "change_evidence": list(core_only.change_evidence.shape),
                "post_evidence": list(core_only.post_evidence.shape),
            },
            "parameter_count": int(sum(p.numel() for p in encoder.parameters())),
            "status": "PASS" if core_finite and unavailable_invariant and current_does_not_change_pre else "FAIL",
        }
    )
    return interface


def capability_coverage(result) -> dict[str, Any]:
    frame = result.index
    dev = frame[frame["split"].astype(str).str.lower().isin(["train", "val"])].copy()
    state_counts = {str(k): int(v) for k, v in dev["clinical_state"].value_counts().items()}

    cov = {
        region: float(result.current_core.values[dev.index, 3 + i].mean())
        for i, region in enumerate(REGIONS)
    }
    return {
        "status": "PASS",
        "all_anchor_rows": int(len(frame)),
        "development_rows": int(len(dev)),
        "development_patients": int(dev["patient_id"].nunique()),
        "development_state_counts": state_counts,
        "development_fraction_with_prior_scan": float(dev["previous_scan_episode_id"].notna().mean()),
        "development_fraction_with_prior_comparable_scan": float(dev["previous_comparable_scan_episode_id"].notna().mean()),
        "development_current_coverage_fraction": cov,
        "development_modality_observed_fraction": float(dev["current_modality_observed"].mean()),
        "development_site_positive_stream_observed_fraction": float(dev["current_site_positive_stream_observed"].mean()),
        "development_median_prior_scan_count": float(dev["prior_scan_count"].median()),
        "development_median_days_since_previous_assessment": float(dev["days_since_previous_assessment"].dropna().median()),
        "unsupported_streams": {
            "raw_report_text": "UNAVAILABLE",
            "lesion_measurements": "UNAVAILABLE",
            "stable_vs_responding_CHORD_substate": "UNAVAILABLE_DO_NOT_SYNTHESIZE",
            "validated_lab_context": "NOT_PROMOTED_IN_V2_01",
        },
    }


def write_source_manifest(out: Path) -> dict[str, Any]:
    payload = {
        "version": "v2_01_1",
        "status": "PASS",
        "sources": {
            "authoritative_scan_episode": {
                "path": "artifacts/checkpoint1/canonical/chord_breast_scan_episodes_w3.parquet",
                "key": ["patient_id", "scan_episode_id"],
                "availability": "landmark_day = episode_end_day",
                "pre_cutoff": "prior landmark_day < current episode_start_day",
                "model_facing": True,
            },
            "r2_anchor_interface": {
                "path": "artifacts/checkpoint7r2_transport_safe_supervised/prepared/scan_index.parquet",
                "use": "exact row alignment, task masks, split, genomic availability metadata",
                "prohibited_fields": ["line", "treatment_line", "line_start_day", "elapsed_on_line_days", "scan_number_line"],
            },
            "development_folds": {
                "path": "artifacts/dynamic_scan_v2/v2_00/development_folds.parquet",
                "outcome_blind": True,
            },
        },
        "semantic_rules": {
            "non_progressive": "single CHORD category; never split into stable/responding without independent source",
            "site_mentions": "positive support only; disappearing mention is not resolution; first mention is first_observed_positive_mention, not new metastasis",
            "coverage": "observation coverage, not disease negativity",
            "numeric_change": "no lesion percentage-change feature because no matched lesion measurement source is established",
            "component_counts": "count magnitudes excluded from model-facing features; only source-presence booleans retained",
            "line_identity": "prohibited model-facing",
        },
        "optional_streams": {
            "modality": {"available": True, "mask_required": True, "current_recency_days": 0},
            "site_positive": {"available": True, "mask_required": True, "current_recency_days": 0},
            "genomics": {"available": True, "role": "kept outside paired-scan encoder for V2-02 fusion", "strict_rule": "availability_day < landmark_day"},
            "raw_report_text": {"available": False, "policy": "DO_NOT_SYNTHESIZE"},
            "lesion_measurements": {"available": False, "policy": "DO_NOT_SYNTHESIZE"},
        },
        "external_access": "No DFCI/VICC rows are read by V2-01.",
    }
    atomic_json(out / "source_evidence_manifest.json", payload)
    return payload


def build_manifest(out: Path, files: list[Path]) -> dict[str, Any]:
    manifest = {
        "checkpoint": "V2-01",
        "status": "PASS",
        "external_rows_opened": False,
        "external_predictions_regenerated": False,
        "files": {},
    }
    for path in files:
        manifest["files"][path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    atomic_json(out / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    out = repo / "artifacts/dynamic_scan_v2/v2_01"
    out.mkdir(parents=True, exist_ok=True)

    policy = AccessPolicy.from_json(repo, repo / "configs/dynamic_scan_v2/data_access_policy.json")
    scans, anchors, _ = load_inputs(repo, policy)

    result = build_paired_scan_representation(scans, anchors)
    # Preserve exact R2 row order; the builder follows anchor order by construction.
    expected_keys = anchors[["patient_id", "scan_episode_id"]].reset_index(drop=True).astype(str)
    actual_keys = result.index[["patient_id", "scan_episode_id"]].reset_index(drop=True).astype(str)
    if not expected_keys.equals(actual_keys):
        raise RuntimeError("V2-01 row keys/order do not exactly replay the R2 anchor interface")

    # Join frozen development fold after representation construction; it is never a feature.
    result.index["development_fold"] = anchors["development_fold"].to_numpy()

    schema = representation_feature_schema(result)
    source_manifest = write_source_manifest(out)
    crosswalk = legacy_crosswalk(repo, policy, result)
    invariance = actual_invariance_checks(scans, anchors, result)
    encoder = encoder_smoke(result)
    coverage = capability_coverage(result)

    # Persist audit diagnostics before acceptance gating. If an actual-data audit
    # fails, the result directory still contains enough evidence to diagnose it
    # without another blind rerun.
    atomic_json(out / "legacy_scan_crosswalk.json", crosswalk)
    atomic_json(out / "invariance_tests.json", invariance)
    atomic_json(out / "encoder_interface.json", encoder)
    atomic_json(out / "capability_coverage.json", coverage)

    if crosswalk["status"] != "PASS":
        raise RuntimeError(f"Legacy scan crosswalk failed: {crosswalk}")
    if invariance["status"] != "PASS":
        raise RuntimeError(f"V2-01 invariance checks failed: {invariance}")
    if encoder["status"] != "PASS":
        raise RuntimeError(f"V2-01 encoder smoke failed: {encoder}")

    atomic_parquet(out / "paired_scan_index.parquet", result.index)
    atomic_json(out / "feature_schema.json", schema)
    arrays = _representation_arrays(result)
    engineered_core, engineered_core_names = engineered_control_matrix(result, include_optional=False)
    engineered_full, engineered_full_names = engineered_control_matrix(result, include_optional=True)
    arrays.update(
        {
            "engineered_core_control": engineered_core,
            "engineered_full_control": engineered_full,
        }
    )
    atomic_npz(out / "paired_scan_tensors.npz", **arrays)
    atomic_json(
        out / "engineered_control_schema.json",
        {
            "version": "v2_01_1",
            "core_names": list(engineered_core_names),
            "full_names": list(engineered_full_names),
            "normalization": "none in V2-01; V2-02 fits preprocessing within each training fold only",
        },
    )

    result_packet = {
        "checkpoint": "V2-01",
        "status": "PASS",
        "next_checkpoint": "V2-02",
        "acceptance": {
            "exact_r2_anchor_key_and_order_replay": True,
            "legacy_scan_value_crosswalk_pass": crosswalk["status"] == "PASS",
            "feature_provenance_and_availability_explicit": source_manifest["status"] == "PASS",
            "strict_pre_post_episode_contract_pass": invariance["strict_pre_cutoff_before_episode_start"],
            "future_append_invariance_pass": invariance["append_future_scan_invariant"],
            "line_label_invariance_pass": invariance["line_label_perturbation_invariant"],
            "current_scan_perturbation_isolated_to_post_pass": invariance["current_scan_perturbation_history_identical"] and invariance["current_scan_perturbation_changes_post_blocks"],
            "unavailable_optional_stream_masking_pass": encoder["unavailable_optional_adversarial_invariant"],
            "duplicate_component_burden_invariance_pass": invariance["duplicate_component_count_invariant"],
            "unrelated_same_day_isolation_pass": invariance["unrelated_same_day_extra_field_invariant"],
            "zero_distinct_from_missing_pass": invariance["zero_numeric_distinct_from_missing"],
            "core_encoder_without_optional_streams_pass": encoder["core_only_finite"],
            "encoder_pre_branch_current_content_isolation_pass": encoder["current_content_does_not_change_history_context"],
            "unsupported_streams_explicit_not_fabricated": True,
            "no_predictive_superiority_claimed": True,
        },
        "rows": int(len(result.index)),
        "patients": int(result.index["patient_id"].nunique()),
        "development_rows": int(result.index["split"].astype(str).str.lower().isin(["train", "val"]).sum()),
        "historical_test_rows": int(result.index["split"].astype(str).str.lower().eq("test").sum()),
        "feature_dimensions": {k: int(v.shape[1]) for k, v in arrays.items() if v.ndim == 2},
        "coverage": coverage,
        "crosswalk": crosswalk,
        "encoder": {
            "parameter_count": encoder["parameter_count"],
            "output_shapes": encoder["output_shapes"],
            "training_status": encoder["training_status"],
        },
        "config_hashes": {
            "v2_01_representation.json": sha256_file(repo / "configs/dynamic_scan_v2/v2_01_representation.json"),
            "development_protocol.json": sha256_file(repo / "configs/dynamic_scan_v2/development_protocol.json"),
            "data_access_policy.json": sha256_file(repo / "configs/dynamic_scan_v2/data_access_policy.json"),
        },
        "external_rows_opened": False,
        "external_predictions_regenerated": False,
    }
    if not all(result_packet["acceptance"].values()):
        raise RuntimeError(f"V2-01 acceptance failed: {result_packet['acceptance']}")
    atomic_json(out / "result_packet.json", result_packet)

    report = f"""# OncoTwin V2-01 paired scan/change representation\n\nStatus: **PASS**\n\n## Scope\n\nV2-01 constructs causal paired-scan features on the exact {len(result.index):,} repaired R2 anchor episodes. It does not fit or select a survival model and makes no predictive-superiority claim.\n\n## PRE/POST contract\n\n- PRE uses only prior authoritative scan episodes with `prior landmark_day < current episode_start_day`.\n- POST may additionally consume the current clinical state, current anatomical coverage, change features, and explicitly masked optional modality/site-positive streams.\n- Current episode source days are therefore excluded from PRE even for multi-day W3 episodes.\n- Absolute treatment-line identity is not a model-facing feature.\n\n## Semantics\n\n- CHORD `NON_PROGRESSIVE` remains a single state; no stable/responding labels are fabricated.\n- Coverage is observation coverage, not disease negativity.\n- Site features are positive mentions/support only. `first_observed_positive_mention` does not mean new metastasis, and a disappearing mention is not resolution.\n- No lesion percentage-change feature is created because no matched lesion measurement source is established.\n- Utilization/history-intensity descriptors are kept in a separate block.\n\n## Controls\n\n- Transparent engineered core/full matrices are frozen as controls.\n- `PairedScanEncoder` is a compact mask-aware architecture only at V2-01; fold-specific supervised fitting begins at V2-02.\n\n## Validation\n\n- legacy 18-D observed scan surface crosswalk: {crosswalk['status']}\n- perturbation/invariance battery: {invariance['status']}\n- core-only encoder smoke: {encoder['status']}\n\n## Next action\n\nProceed to V2-02 transport-native baseline and missingness-robust survival training on the frozen development folds. Do not use historical CHORD test or DFCI/VICC feedback for selection.\n"""
    atomic_text(out / "decision_report.md", report)

    manifest_files = [
        out / "paired_scan_index.parquet",
        out / "paired_scan_tensors.npz",
        out / "feature_schema.json",
        out / "engineered_control_schema.json",
        out / "source_evidence_manifest.json",
        out / "legacy_scan_crosswalk.json",
        out / "invariance_tests.json",
        out / "encoder_interface.json",
        out / "capability_coverage.json",
        out / "result_packet.json",
        out / "decision_report.md",
    ]
    build_manifest(out, manifest_files)

    print(json.dumps(result_packet, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
