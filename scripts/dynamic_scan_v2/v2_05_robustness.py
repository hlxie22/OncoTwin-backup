from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from access import AccessPolicy
from common import atomic_json, atomic_parquet, atomic_text, import_module_from_path, sha256_file
from metrics import (
    HORIZONS_MONTHS,
    MONTH_DAYS,
    censoring_survival_at,
    fit_censoring_km,
    fractional_censor_nll_per_row,
    horizon_status_and_weight,
    v2_metric_bundle,
)
from scan_representation import build_paired_scan_representation
from v2_02_train import (
    _calibration_table,
    _compact_prediction_rows,
    _gbt_logits,
    attach_and_validate_development_folds,
    build_person_period_rows,
    patient_balance_weights,
)
from v2_03_gate import blend_logits
from v2_04_ablate import CandidateSpec, _batch, _predict, _target_weight, _train_candidate, _train_one_fold
from v2_04_models import HybridTemporalScanModel
from v2_05_controls import (
    calibration_ece,
    decide_candidate_lock,
    deterministic_stratified_permutation,
    mask_explicit_scan_arrays,
    patient_bootstrap_brier_delta,
    seed_robustness,
    window_spread,
)


SELECTED_NAME = "post_temporal_plus_explicit_scan__genomic_none"
SELECTED_REPRESENTATION = "post_temporal_plus_explicit_scan"
SELECTED_GENOMIC = "none"


def _normalize_key_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["patient_id"] = out["patient_id"].astype(str).str.strip()
    out["scan_episode_id"] = out["scan_episode_id"].astype(str).str.strip()
    return out


def _load_core(repo: Path, policy: AccessPolicy):
    cols = [
        "patient_id", "scan_episode_id", "landmark_day", "split", "survival_mask",
        "survival_time_days", "survival_cause", "next_scan_mask", "next_scan_label",
        "postprog_mask", "postprog_label", "postprog_time_mask", "postprog_time_bucket",
        "sample_id", "coverage_gene_count", "genomic_available", "genomic_age_days",
    ]
    scan = policy.read_parquet(
        repo / "artifacts/checkpoint7r2_transport_safe_supervised/prepared/scan_index.parquet",
        columns=cols,
    )
    scan = _normalize_key_frame(scan)
    scan["global_row"] = np.arange(len(scan), dtype=np.int64)
    folds = policy.read_parquet(repo / "artifacts/dynamic_scan_v2/v2_00/development_folds.parquet")
    frame = attach_and_validate_development_folds(scan, folds, fold_count=3)
    is_dev = frame["split"].isin(["train", "val"])
    eligible = is_dev & frame["survival_mask"].fillna(False).astype(bool)
    dev = frame.loc[eligible].copy().reset_index(drop=True)
    all_dev = frame.loc[is_dev].copy().reset_index(drop=True)
    if (dev["development_fold"] < 0).any() or (all_dev["development_fold"] < 0).any():
        raise RuntimeError("historical test sentinel entered V2-05")
    for d in (dev, all_dev):
        d["next_scan_mask"] = d["next_scan_mask"].fillna(False).astype(bool)
        d["next_scan_label_filled"] = pd.to_numeric(d["next_scan_label"], errors="coerce").fillna(0).astype(int)
        d["next_target_weight"] = _target_weight(d, "next_scan_mask")
        d["postprog_mask"] = d["postprog_mask"].fillna(False).astype(bool)
        d["postprog_label_filled"] = pd.to_numeric(d["postprog_label"], errors="coerce").fillna(0).astype(int)
        d["postprog_time_mask"] = d["postprog_time_mask"].fillna(False).astype(bool)
        d["postprog_time_bucket_filled"] = pd.to_numeric(d["postprog_time_bucket"], errors="coerce").fillna(0).astype(int)
        d["postprog_target_weight"] = _target_weight(d, "postprog_mask")
    kmf = frame.loc[frame["split"].eq("train") & frame["survival_mask"].fillna(False).astype(bool)]
    km = fit_censoring_km(kmf["survival_time_days"].to_numpy(float), kmf["survival_cause"].astype(int).to_numpy())
    return frame, dev, all_dev, km


def _load_arrays(repo: Path, policy: AccessPolicy):
    with policy.np_load(repo / "artifacts/dynamic_scan_v2/v2_01/paired_scan_tensors.npz") as z:
        arrays = {k: np.asarray(z[k], np.float32) for k in z.files}
    arrays["tumor"] = np.asarray(policy.np_load(
        repo / "artifacts/checkpoint7r2_transport_safe_supervised/prepared/tumor_embeddings_f16.npy", mmap_mode="r"
    ), np.float32)
    temporal = np.asarray(policy.np_load(
        repo / "artifacts/checkpoint7r2_transport_safe_supervised/prepared/temporal_prepost_f16.npy", mmap_mode="r"
    ), np.float32)
    arrays["temporal_pre"] = temporal[:, 0]
    arrays["temporal_post"] = temporal[:, 1]
    context = np.asarray(policy.np_load(
        repo / "artifacts/checkpoint7r2_transport_safe_supervised/prepared/context_features_f32.npy", mmap_mode="r"
    ), np.float32)
    arrays["portable_context"] = context[:, [2, 4, 5]].astype(np.float32)
    arrays["genomic_available"] = context[:, 4:5].astype(np.float32)
    arrays["genomic_age_scaled"] = context[:, 5:6].astype(np.float32)
    arrays["genomic_simple"] = np.zeros((len(context), 3), np.float32)  # selected candidate ignores it
    schema = json.loads((repo / "artifacts/dynamic_scan_v2/v2_01/feature_schema.json").read_text())
    dims = {k: len(v) for k, v in schema["blocks"].items()}
    return arrays, dims


def _load_model(checkpoint: Path, dims: dict[str, int], v204_cfg: dict[str, Any], device: torch.device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    spec = ckpt["spec"]
    model = HybridTemporalScanModel(
        representation_mode=spec["representation_mode"],
        genomic_mode=spec["genomic_mode"],
        dims=dims,
        adapter_bottleneck=int(v204_cfg["temporal_adapter"]["bottleneck_dim"]),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


def _predict_seed_oof(repo: Path, dev: pd.DataFrame, arrays: dict[str, np.ndarray], dims, v204_cfg, seed: int, device, checkpoint_root: Path | None = None):
    pre = np.full((len(dev), 24, 4), np.nan, np.float32)
    post = np.full_like(pre, np.nan)
    root = checkpoint_root or (repo / "artifacts/dynamic_scan_v2/v2_04/checkpoints" / SELECTED_NAME / f"seed_{seed}")
    for fold in (0, 1, 2):
        va = dev.loc[dev["development_fold"].eq(fold)].copy().reset_index(drop=True)
        ckpt = root / f"fold_{fold}.pt"
        if not ckpt.is_file():
            raise RuntimeError(f"Missing frozen V2-04 checkpoint {ckpt}")
        model = _load_model(ckpt, dims, v204_cfg, device)
        pred = _predict(model, va, arrays, va["global_row"].to_numpy(np.int64), device)
        loc = np.where(dev["development_fold"].to_numpy() == fold)[0]
        pre[loc] = pred["pre"]
        post[loc] = pred["post"]
    if not np.isfinite(pre).all() or not np.isfinite(post).all():
        raise RuntimeError(f"Incomplete OOF predictions for seed {seed}")
    return {"pre": pre, "post": post}


def _subgroup_metrics(dev, logits, km, paired_index, min_patients: int):
    meta = dev[["patient_id", "scan_episode_id", "genomic_available", "genomic_age_days"]].merge(
        paired_index[[
            "patient_id", "scan_episode_id", "clinical_state", "prior_scan_count",
            "days_since_previous_assessment", "coverage_comparable_to_previous",
            "current_modality_observed", "current_site_positive_stream_observed",
        ]], on=["patient_id", "scan_episode_id"], how="left", validate="one_to_one"
    )
    definitions: dict[str, pd.Series] = {
        "genomics_available=false": ~meta["genomic_available"].fillna(False).astype(bool),
        "genomics_available=true": meta["genomic_available"].fillna(False).astype(bool),
        "modality_observed=false": ~meta["current_modality_observed"].fillna(False).astype(bool),
        "modality_observed=true": meta["current_modality_observed"].fillna(False).astype(bool),
        "site_stream_observed=false": ~meta["current_site_positive_stream_observed"].fillna(False).astype(bool),
        "site_stream_observed=true": meta["current_site_positive_stream_observed"].fillna(False).astype(bool),
        "coverage_comparable=false": ~meta["coverage_comparable_to_previous"].fillna(False).astype(bool),
        "coverage_comparable=true": meta["coverage_comparable_to_previous"].fillna(False).astype(bool),
    }
    for state in sorted(meta["clinical_state"].dropna().astype(str).unique()):
        definitions[f"state={state}"] = meta["clinical_state"].astype(str).eq(state)
    definitions["prior_scans=0-2"] = meta["prior_scan_count"].fillna(-1).between(0, 2)
    definitions["prior_scans=3-5"] = meta["prior_scan_count"].fillna(-1).between(3, 5)
    definitions["prior_scans=6-10"] = meta["prior_scan_count"].fillna(-1).between(6, 10)
    definitions["prior_scans=11+"] = meta["prior_scan_count"].fillna(-1).ge(11)
    days = pd.to_numeric(meta["days_since_previous_assessment"], errors="coerce")
    definitions["prev_scan_recency<=60d"] = days.le(60)
    definitions["prev_scan_recency=61-120d"] = days.gt(60) & days.le(120)
    definitions["prev_scan_recency>120d"] = days.gt(120)
    ga = pd.to_numeric(meta["genomic_age_days"], errors="coerce")
    available = meta["genomic_available"].fillna(False).astype(bool) & ga.notna()
    if available.sum() > 0:
        qs = ga.loc[available].quantile([0.33, 0.67]).to_numpy()
        definitions["genomic_age_recent"] = available & ga.le(qs[0])
        definitions["genomic_age_mid"] = available & ga.gt(qs[0]) & ga.le(qs[1])
        definitions["genomic_age_old"] = available & ga.gt(qs[1])
    rows = []
    for name, mask in definitions.items():
        idx = np.where(mask.to_numpy(dtype=bool))[0]
        if len(idx) == 0:
            continue
        sub = dev.iloc[idx].reset_index(drop=True)
        patients = int(sub["patient_id"].nunique())
        if patients < int(min_patients):
            continue
        m = v2_metric_bundle(sub, logits[idx], km)
        rows.append({"stratum": name, "rows": int(len(idx)), "patients": patients, "brier": m["patient_brier_4h_mean"], "nll": m["fractional_censor_patient_mean_nll_v1"]})
    return pd.DataFrame(rows)


def _specificity_strata(paired_dev: pd.DataFrame) -> list[tuple[Any, ...]]:
    out = []
    for _, r in paired_dev.iterrows():
        count = int(r.get("current_coverage_count", 0) or 0)
        prior = int(r.get("prior_scan_count", 0) or 0)
        out.append((
            str(r.get("clinical_state", "UNKNOWN")),
            min(count, 5),
            bool(r.get("current_modality_observed", False)),
            bool(r.get("current_site_positive_stream_observed", False)),
            min(prior // 5, 4),
        ))
    return out


def _permute_explicit_blocks(arrays: dict[str, np.ndarray], dev: pd.DataFrame, perm_local: np.ndarray):
    out = dict(arrays)
    blocks = [
        "current_core", "current_core_mask", "change_core", "change_core_mask",
        "optional_current", "optional_current_mask", "stream_availability", "stream_availability_mask",
    ]
    g = dev["global_row"].to_numpy(np.int64)
    source_g = g[np.asarray(perm_local, dtype=np.int64)]
    for name in blocks:
        out[name] = np.array(arrays[name], copy=True)
        out[name][g] = arrays[name][source_g]
    return out


def _last_scan_only_gbt(dev: pd.DataFrame, arrays: dict[str, np.ndarray], km, seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier
    feats = np.concatenate([
        arrays["current_core"], arrays["current_core_mask"], arrays["optional_current"],
        arrays["optional_current_mask"], arrays["stream_availability"], arrays["stream_availability_mask"],
    ], axis=1).astype(np.float32)
    oof = np.full((len(dev), 24, 4), np.nan, np.float32)
    fold_rows = []
    for fold in (0, 1, 2):
        tr = dev.loc[~dev["development_fold"].eq(fold)].copy().reset_index(drop=True)
        va = dev.loc[dev["development_fold"].eq(fold)].copy().reset_index(drop=True)
        xtr = feats[tr["global_row"].to_numpy(np.int64)]
        w = patient_balance_weights(tr)
        xp, yp, wp = build_person_period_rows(
            xtr, tr["survival_time_days"].to_numpy(float), tr["survival_cause"].astype(int).to_numpy(), w
        )
        clf = HistGradientBoostingClassifier(
            loss="log_loss", learning_rate=0.06, max_iter=160, max_leaf_nodes=15,
            l2_regularization=1.0, min_samples_leaf=40, random_state=int(seed + fold),
        )
        clf.fit(xp, yp, sample_weight=wp)
        logits = _gbt_logits(clf, feats[va["global_row"].to_numpy(np.int64)])
        loc = np.where(dev["development_fold"].to_numpy() == fold)[0]
        oof[loc] = logits
        m = v2_metric_bundle(va, logits, km)
        fold_rows.append({"fold": fold, "brier": m["patient_brier_4h_mean"], "nll": m["fractional_censor_patient_mean_nll_v1"]})
    m = v2_metric_bundle(dev, oof, km)
    return {"brier": m["patient_brier_4h_mean"], "nll": m["fractional_censor_patient_mean_nll_v1"], "folds": fold_rows}, oof


def _prepare_window_cache(repo: Path, out: Path, label: str, batch_size: int = 512):
    source = repo / f"artifacts/checkpoint6d/regenerated/{label}/prepared"
    needed = [source / "breast_tokens.parquet", source / "breast_scan_index.parquet"]
    if not all(p.is_file() for p in needed):
        raise RuntimeError(f"Exact checkpoint6d {label} prepared inputs unavailable: {needed}")
    cache = out / "window_cache" / f"{label}_r1"
    prepared = cache / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    for p in needed:
        target = prepared / p.name
        if not target.exists():
            os.symlink(p, target)
    model_src = repo / "artifacts/checkpoint7r1_line_agnostic_temporal/temporal_encoder.pt"
    model_target = cache / "temporal_encoder.pt"
    if not model_target.exists():
        os.symlink(model_src, model_target)
    emb = cache / "breast_scan_prepost_embeddings_f16.npy"
    idx = cache / "breast_scan_prepost_index.parquet"
    if not emb.is_file() or not idx.is_file():
        legacy = import_module_from_path(
            repo / "scripts/dynamic_scan/ckpt7r1_line_agnostic_temporal.py",
            f"v205_ckpt7r1_{label}",
        )
        legacy.cache_scan_states(cache, int(batch_size))
    return cache


def _rep_to_local_arrays(rep, temporal: np.ndarray):
    arrays = {
        "history_core": rep.history_core.values, "history_core_mask": rep.history_core.masks,
        "utilization_core": rep.utilization_core.values, "utilization_core_mask": rep.utilization_core.masks,
        "current_core": rep.current_core.values, "current_core_mask": rep.current_core.masks,
        "change_core": rep.change_core.values, "change_core_mask": rep.change_core.masks,
        "optional_history": rep.optional_history.values, "optional_history_mask": rep.optional_history.masks,
        "optional_history_recency": rep.optional_history_recency.values, "optional_history_recency_mask": rep.optional_history_recency.masks,
        "optional_current": rep.optional_current.values, "optional_current_mask": rep.optional_current.masks,
        "stream_availability": rep.stream_availability.values, "stream_availability_mask": rep.stream_availability.masks,
        "tumor": np.zeros((len(rep.index), 128), np.float32),
        "genomic_available": np.zeros((len(rep.index), 1), np.float32),
        "genomic_age_scaled": np.zeros((len(rep.index), 1), np.float32),
        "genomic_simple": np.zeros((len(rep.index), 3), np.float32),
        "temporal_pre": np.asarray(temporal[:, 0], np.float32),
        "temporal_post": np.asarray(temporal[:, 1], np.float32),
        "portable_context": np.zeros((len(rep.index), 3), np.float32),
    }
    arrays["portable_context"][:, 0] = rep.utilization_core.values[:, 0]
    return arrays


def _predict_local_by_fold(repo, frame, arrays, dims, v204_cfg, seed, device):
    pre = np.full((len(frame), 24, 4), np.nan, np.float32)
    post = np.full_like(pre, np.nan)
    local_frame = frame.copy().reset_index(drop=True)
    local_frame["global_row"] = np.arange(len(local_frame), dtype=np.int64)
    for fold in (0, 1, 2):
        subset = local_frame.loc[local_frame["development_fold"].eq(fold)].copy().reset_index(drop=True)
        if subset.empty:
            continue
        ckpt = repo / "artifacts/dynamic_scan_v2/v2_04/checkpoints" / SELECTED_NAME / f"seed_{seed}" / f"fold_{fold}.pt"
        model = _load_model(ckpt, dims, v204_cfg, device)
        pred = _predict(model, subset, arrays, subset["global_row"].to_numpy(np.int64), device)
        loc = np.where(local_frame["development_fold"].to_numpy() == fold)[0]
        pre[loc] = pred["pre"]
        post[loc] = pred["post"]
    if not np.isfinite(pre).all() or not np.isfinite(post).all():
        raise RuntimeError("Incomplete window sensitivity prediction")
    return pre, post


def _patient_landmark_index(frame: pd.DataFrame, *, label: str) -> tuple[pd.DataFrame, dict[tuple[str, int], int]]:
    """Return a line-agnostic exact-landmark index.

    Alternate W0/W3/W7 episode IDs are window-specific by construction, so they
    must not be used as cross-window identity keys. V2 alignment is based on the
    invariant observable landmark identity: patient_id + landmark_day.
    """
    out = _normalize_key_frame(frame)
    if "landmark_day" not in out.columns:
        raise RuntimeError(f"{label}: landmark_day is required for window alignment")
    days = pd.to_numeric(out["landmark_day"], errors="raise").to_numpy(float)
    rounded = np.rint(days)
    if not np.allclose(days, rounded, atol=1e-8, rtol=0.0):
        raise RuntimeError(f"{label}: non-integer landmark_day encountered")
    out = out.copy()
    out["landmark_day"] = rounded.astype(np.int64)
    dup = out.duplicated(["patient_id", "landmark_day"], keep=False)
    if dup.any():
        preview = out.loc[dup, ["patient_id", "landmark_day", "scan_episode_id"]].head(20)
        raise RuntimeError(
            f"{label}: patient+landmark_day is not unique; preview={preview.to_dict(orient='records')}"
        )
    mapping = {
        (str(pid), int(day)): int(i)
        for i, (pid, day) in enumerate(zip(out["patient_id"], out["landmark_day"]))
    }
    return out, mapping


def _window_sensitivity(repo, out, dev, paired_index, dims, v204_cfg, alpha, km, seed, device, policy, limit, min_rows):
    caches = {label: _prepare_window_cache(repo, out, label) for label in ("w0", "w7")}

    # Window-specific episode IDs intentionally differ (e.g. ::W0, ::W3, ::W7).
    # Align cross-window sensitivity by the line-agnostic invariant landmark key:
    # patient_id + landmark_day. The current three-state assessment must also be
    # preserved before a row enters the common cohort.
    dev_day, dev_day_map = _patient_landmark_index(dev, label="w3_development")
    dev_landmark_keys = list(zip(dev_day["patient_id"], dev_day["landmark_day"].astype(int)))

    cache_idx: dict[str, pd.DataFrame] = {}
    cache_day_maps: dict[str, dict[tuple[str, int], int]] = {}
    alt_scans: dict[str, pd.DataFrame] = {}
    scan_day_maps: dict[str, dict[tuple[str, int], int]] = {}

    for label in ("w0", "w7"):
        ci_raw = policy.read_parquet(caches[label] / "breast_scan_prepost_index.parquet")
        ci, ci_map = _patient_landmark_index(ci_raw, label=f"{label}_temporal_cache")
        cache_idx[label] = ci
        cache_day_maps[label] = ci_map

        canonical_path = repo / f"artifacts/checkpoint1/canonical/chord_breast_scan_episodes_{label}.parquet"
        scans_raw = policy.read_parquet(canonical_path)
        scans, scans_map = _patient_landmark_index(scans_raw, label=f"{label}_canonical_scans")
        alt_scans[label] = scans
        scan_day_maps[label] = scans_map

        # Within a window, the repaired temporal cache must still refer to the
        # exact canonical episode at that patient/day.
        shared = set(ci_map) & set(scans_map)
        if len(shared) != len(ci_map):
            raise RuntimeError(
                f"{label}: temporal cache has {len(ci_map)-len(shared)} patient/day rows absent from canonical scans"
            )
        mismatched_ids = []
        for key in shared:
            c_id = str(ci.iloc[ci_map[key]]["scan_episode_id"])
            s_id = str(scans.iloc[scans_map[key]]["scan_episode_id"])
            if c_id != s_id:
                mismatched_ids.append((key, c_id, s_id))
                if len(mismatched_ids) >= 20:
                    break
        if mismatched_ids:
            raise RuntimeError(f"{label}: cache/canonical episode-ID mismatch at same landmark: {mismatched_ids}")

    common_landmark_keys = [
        key for key in dev_landmark_keys
        if key in cache_day_maps["w0"]
        and key in cache_day_maps["w7"]
        and key in scan_day_maps["w0"]
        and key in scan_day_maps["w7"]
    ]
    if not common_landmark_keys:
        raise RuntimeError("No exact W0/W3/W7 patient+landmark-day development keys")

    dev_pos = {key: i for i, key in enumerate(dev_landmark_keys)}
    common_positions = np.asarray([dev_pos[k] for k in common_landmark_keys], dtype=np.int64)
    common = dev.iloc[common_positions].copy().reset_index(drop=True)

    # W3 state is indexed by its native W3 episode ID; W0/W7 use their native
    # episode IDs. Only the patient/day identity is shared across windows.
    pidx = _normalize_key_frame(paired_index)
    pmap = pidx.set_index(["patient_id", "scan_episode_id"])
    w3_episode_keys = list(zip(common["patient_id"], common["scan_episode_id"]))
    if not all(k in pmap.index for k in w3_episode_keys):
        missing = sum(k not in pmap.index for k in w3_episode_keys)
        raise RuntimeError(f"W3 paired-scan metadata missing for {missing} common landmark rows")
    w3_state = np.asarray([str(pmap.loc[k, "clinical_state"]) for k in w3_episode_keys], dtype=object)

    keep = np.ones(len(common), dtype=bool)
    alt_scan_rows: dict[str, np.ndarray] = {}
    alt_cache_rows: dict[str, np.ndarray] = {}
    alt_episode_ids: dict[str, np.ndarray] = {}

    for label in ("w0", "w7"):
        scans = alt_scans[label]
        ci = cache_idx[label]
        srows = np.asarray([scan_day_maps[label][k] for k in common_landmark_keys], dtype=np.int64)
        crows = np.asarray([cache_day_maps[label][k] for k in common_landmark_keys], dtype=np.int64)
        selected_scans = scans.iloc[srows]
        selected_cache = ci.iloc[crows]

        states = selected_scans["progression_state_3"].astype(str).str.strip().str.upper().to_numpy(object)
        keep &= states == w3_state
        keep &= selected_scans["landmark_day"].to_numpy(np.int64) == common["landmark_day"].to_numpy(np.int64)
        keep &= selected_cache["landmark_day"].to_numpy(np.int64) == common["landmark_day"].to_numpy(np.int64)

        alt_scan_rows[label] = srows
        alt_cache_rows[label] = crows
        alt_episode_ids[label] = selected_scans["scan_episode_id"].astype(str).to_numpy()

    common = common.loc[keep].reset_index(drop=True)
    common_landmark_keys = [k for k, ok in zip(common_landmark_keys, keep) if ok]
    w3_episode_keys = [k for k, ok in zip(w3_episode_keys, keep) if ok]
    for label in ("w0", "w7"):
        alt_scan_rows[label] = alt_scan_rows[label][keep]
        alt_cache_rows[label] = alt_cache_rows[label][keep]
        alt_episode_ids[label] = alt_episode_ids[label][keep]

    if len(common) < int(min_rows):
        raise RuntimeError(f"W0/W3/W7 common sensitivity cohort too small: {len(common)}")

    # W3 uses immutable selected V2-04 OOF logits with native W3 IDs.
    v204_idx = _normalize_key_frame(
        policy.read_parquet(repo / "artifacts/dynamic_scan_v2/v2_04/selected_v2_04_oof_index.parquet")
    )
    with policy.np_load(repo / "artifacts/dynamic_scan_v2/v2_04/selected_v2_04_oof_logits.npz") as z:
        w3_pre_all = np.asarray(z["pre_logits"], np.float32)
        w3_post_all = np.asarray(z["post_logits"], np.float32)
    vm = {(p, s): i for i, (p, s) in enumerate(zip(v204_idx["patient_id"], v204_idx["scan_episode_id"]))}
    if not all(k in vm for k in w3_episode_keys):
        missing = sum(k not in vm for k in w3_episode_keys)
        raise RuntimeError(f"V2-04 W3 OOF logits missing for {missing} common rows")
    w3_rows = np.asarray([vm[k] for k in w3_episode_keys], dtype=np.int64)

    scores: dict[str, float] = {}
    details: dict[str, dict[str, Any]] = {}
    w3_locked = blend_logits(w3_pre_all[w3_rows], w3_post_all[w3_rows], alpha)
    m3 = v2_metric_bundle(common, w3_locked, km)
    scores["w3"] = m3["patient_brier_4h_mean"]
    details["w3"] = {
        "rows": len(common),
        "patients": int(common["patient_id"].nunique()),
        "brier": scores["w3"],
        "nll": m3["fractional_censor_patient_mean_nll_v1"],
        "episode_id_semantics": "native W3 episode IDs",
    }

    for label in ("w0", "w7"):
        scans = alt_scans[label]
        anchors = scans.iloc[alt_scan_rows[label]].copy().reset_index(drop=True)
        expected_landmarks = list(zip(anchors["patient_id"], anchors["landmark_day"].astype(int)))
        if expected_landmarks != common_landmark_keys:
            raise RuntimeError(f"{label}: canonical anchor order drift after patient/day alignment")

        rep = build_paired_scan_representation(scans, anchors)
        rep_landmarks = list(zip(rep.index["patient_id"].astype(str), rep.index["landmark_day"].astype(int)))
        if rep_landmarks != common_landmark_keys:
            raise RuntimeError(f"{label}: paired-scan representation order drift after patient/day alignment")

        emb_all = np.asarray(
            policy.np_load(caches[label] / "breast_scan_prepost_embeddings_f16.npy", mmap_mode="r"),
            np.float32,
        )
        temporal = emb_all[alt_cache_rows[label]]
        local_arrays = _rep_to_local_arrays(rep, temporal)
        pre, post = _predict_local_by_fold(repo, common, local_arrays, dims, v204_cfg, seed, device)
        locked = blend_logits(pre, post, alpha)
        m = v2_metric_bundle(common, locked, km)
        scores[label] = m["patient_brier_4h_mean"]
        details[label] = {
            "rows": len(common),
            "patients": int(common["patient_id"].nunique()),
            "brier": scores[label],
            "nll": m["fractional_censor_patient_mean_nll_v1"],
            "episode_id_semantics": f"native {label.upper()} episode IDs",
            "episode_ids_differ_from_w3_fraction": float(
                np.mean(alt_episode_ids[label] != common["scan_episode_id"].astype(str).to_numpy())
            ),
        }

    spread = window_spread(scores, limit)
    return {
        "status": "PASS" if spread["pass"] else "FAIL",
        "common_rows": len(common),
        "common_patients": int(common["patient_id"].nunique()),
        "alignment_key": ["patient_id", "landmark_day"],
        "line_identity_used_for_alignment": False,
        "window_specific_episode_ids_expected": True,
        "state_and_landmark_preserved": True,
        "details": details,
        "spread": spread,
        "temporal_encoder": "checkpoint7r1_line_agnostic_temporal",
    }

def _censor_support(frame, km):
    rows = []
    for h in HORIZONS_MONTHS:
        _, w = horizon_status_and_weight(frame, h * MONTH_DAYS, km, patient_balanced=True)
        use = np.isfinite(w) & (w > 0)
        ww = w[use]
        ess = float((ww.sum() ** 2) / max(np.sum(ww ** 2), 1e-12)) if len(ww) else 0.0
        rows.append({
            "horizon_month": h,
            "censoring_survival": censoring_survival_at(km, h * MONTH_DAYS, left_limit=False),
            "known_rows": int(use.sum()),
            "weight_sum": float(ww.sum()),
            "kish_effective_rows": ess,
            "patients": int(frame.loc[use, "patient_id"].nunique()),
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--repo", required=True); args = ap.parse_args()
    repo = Path(args.repo).resolve(); out = repo / "artifacts/dynamic_scan_v2/v2_05"; out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((repo / "configs/dynamic_scan_v2/v2_05_robustness.json").read_text())
    v204_cfg = json.loads((repo / "configs/dynamic_scan_v2/v2_04_ablation.json").read_text())
    policy = AccessPolicy.from_json(repo, repo / "configs/dynamic_scan_v2/data_access_policy.json")
    v204 = json.loads((repo / "artifacts/dynamic_scan_v2/v2_04/result_packet.json").read_text())
    v203 = json.loads((repo / "artifacts/dynamic_scan_v2/v2_03/result_packet.json").read_text())
    if v204.get("status") != "PASS" or v203.get("status") != "PASS": raise RuntimeError("V2-03/V2-04 must PASS")
    if v204["selection"]["selected_candidate"] != SELECTED_NAME: raise RuntimeError("V2-05 patch expects frozen V2-04 no-genomics explicit-scan finalist")
    alpha = float(v204["locked_alpha"])
    plan = {
        "status": "FROZEN_BEFORE_ANALYSIS", "candidate": SELECTED_NAME, "locked_alpha": alpha,
        "primary_metric": "patient-balanced mean PFS Brier at 3/6/12/18m", "selection_rules": cfg["robustness"],
        "specificity_rules": cfg["specificity"], "external_outcomes": "PROHIBITED",
    }
    atomic_json(out / "frozen_analysis_plan.json", plan)

    # Immutable artifact verification.
    manifest = json.loads((repo / "artifacts/dynamic_scan_v2/v2_04/artifact_manifest.json").read_text())
    verified = {}
    for name, expected in manifest["files"].items():
        p = repo / "artifacts/dynamic_scan_v2/v2_04" / name
        actual = sha256_file(p); verified[name] = actual == expected
        if actual != expected: raise RuntimeError(f"V2-04 artifact hash mismatch {name}")
    atomic_json(out / "input_integrity.json", {"v2_04_manifest_verified": all(verified.values()), "files": verified})

    frame, dev, all_dev, km = _load_core(repo, policy)
    arrays, dims = _load_arrays(repo, policy)
    paired_index = _normalize_key_frame(policy.read_parquet(repo / "artifacts/dynamic_scan_v2/v2_01/paired_scan_index.parquet"))
    keys = ["patient_id", "scan_episode_id"]
    paired_dev = dev[keys].merge(paired_index, on=keys, how="left", validate="one_to_one")
    if paired_dev["clinical_state"].isna().any(): raise RuntimeError("V2-01 metadata missing for V2-05 rows")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("[V2_05_DEVICE]", device, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU", flush=True)

    # Frozen baseline and candidate logits.
    idx2 = _normalize_key_frame(policy.read_parquet(repo / "artifacts/dynamic_scan_v2/v2_02/selected_oof_index.parquet"))
    with policy.np_load(repo / "artifacts/dynamic_scan_v2/v2_02/selected_oof_logits.npz") as z:
        bpre = np.asarray(z["pre_logits"], np.float32); bpost = np.asarray(z["post_logits"], np.float32)
    if not np.array_equal(idx2[keys].to_numpy(), dev[keys].to_numpy()): raise RuntimeError("V2-02 OOF key drift")
    baseline_locked = blend_logits(bpre, bpost, alpha)
    idx4 = _normalize_key_frame(policy.read_parquet(repo / "artifacts/dynamic_scan_v2/v2_04/selected_v2_04_oof_index.parquet"))
    with policy.np_load(repo / "artifacts/dynamic_scan_v2/v2_04/selected_v2_04_oof_logits.npz") as z:
        cpre = np.asarray(z["pre_logits"], np.float32); cpost = np.asarray(z["post_logits"], np.float32)
    if not np.array_equal(idx4[keys].to_numpy(), dev[keys].to_numpy()): raise RuntimeError("V2-04 OOF key drift")
    candidate_locked = blend_logits(cpre, cpost, alpha)
    baseline_metric = v2_metric_bundle(dev, baseline_locked, km)
    candidate_metric = v2_metric_bundle(dev, candidate_locked, km)
    pre_metric = v2_metric_bundle(dev, cpre, km); post_metric = v2_metric_bundle(dev, cpost, km)
    controls = {
        "candidate_PRE_history_only": {"brier": pre_metric["patient_brier_4h_mean"], "nll": pre_metric["fractional_censor_patient_mean_nll_v1"]},
        "candidate_POST": {"brier": post_metric["patient_brier_4h_mean"], "nll": post_metric["fractional_censor_patient_mean_nll_v1"]},
        "alpha_0": {"brier": pre_metric["patient_brier_4h_mean"]},
        "alpha_0_5": {"brier": v2_metric_bundle(dev, blend_logits(cpre, cpost, 0.5), km)["patient_brier_4h_mean"]},
        "alpha_1": {"brier": post_metric["patient_brier_4h_mean"]},
        "alpha_0_52_locked": {"brier": candidate_metric["patient_brier_4h_mean"], "nll": candidate_metric["fractional_censor_patient_mean_nll_v1"]},
        "frozen_v2_03_baseline": {"brier": baseline_metric["patient_brier_4h_mean"], "nll": baseline_metric["fractional_censor_patient_mean_nll_v1"]},
    }

    # Confirmation-seed predictions are regenerated from already-trained fold checkpoints, not retrained.
    seeds = [int(v204_cfg["screening_seed"])] + [int(s) for s in v204_cfg["confirmation_seeds"]]
    seed_rows = []; seed_logits = {}
    for seed in seeds:
        pred = {"pre": cpre, "post": cpost} if seed == seeds[0] else _predict_seed_oof(repo, dev, arrays, dims, v204_cfg, seed, device)
        seed_logits[seed] = pred
        locked = blend_logits(pred["pre"], pred["post"], alpha)
        sm = v2_metric_bundle(dev, locked, km)
        row = {"seed": seed, "brier": sm["patient_brier_4h_mean"], "nll": sm["fractional_censor_patient_mean_nll_v1"], "folds": []}
        for fold in (0, 1, 2):
            loc = np.where(dev["development_fold"].to_numpy() == fold)[0]; sub = dev.iloc[loc].reset_index(drop=True)
            fm = v2_metric_bundle(sub, locked[loc], km)
            row["folds"].append({"fold": fold, "rows": len(sub), "patients": int(sub["patient_id"].nunique()), "brier": fm["patient_brier_4h_mean"], "nll": fm["fractional_censor_patient_mean_nll_v1"]})
        seed_rows.append(row)
    seed_report = seed_robustness([x["brier"] for x in seed_rows], baseline_metric["patient_brier_4h_mean"], cfg["robustness"])
    seed_report["details"] = seed_rows
    atomic_json(out / "seed_fold_stability.json", seed_report)

    # Paired patient uncertainty for the screening/frozen selected OOF comparison.
    boot = patient_bootstrap_brier_delta(dev, candidate_locked, baseline_locked, km, repetitions=int(cfg["bootstrap"]["repetitions"]), seed=int(cfg["bootstrap"]["seed"]))
    atomic_json(out / "paired_patient_bootstrap.json", boot)

    # Correct assignment versus outcome-blind within-observable-stratum reassignment of explicit scan evidence.
    strata = _specificity_strata(paired_dev)
    perm = deterministic_stratified_permutation(strata, int(cfg["specificity"]["reassignment_seed"]))
    perm_arrays = _permute_explicit_blocks(arrays, dev, perm)
    perm_pred = _predict_seed_oof(repo, dev, perm_arrays, dims, v204_cfg, seeds[0], device)
    if not np.allclose(perm_pred["pre"], cpre, atol=2e-5, rtol=2e-5): raise RuntimeError("Explicit-evidence reassignment changed PRE branch")
    perm_locked = blend_logits(perm_pred["pre"], perm_pred["post"], alpha)
    perm_metric = v2_metric_bundle(dev, perm_locked, km)
    reassignment_gain = float(perm_metric["patient_brier_4h_mean"] - candidate_metric["patient_brier_4h_mean"])
    specificity = {
        "interpretation": "specificity diagnostic for the explicit scan residual only; not randomized causal evidence and does not reassign the frozen temporal POST state",
        "correct_brier": candidate_metric["patient_brier_4h_mean"], "reassigned_brier": perm_metric["patient_brier_4h_mean"],
        "correct_minus_reassigned_improvement": reassignment_gain, "pre_branch_invariant": True,
        "strata_definition": "clinical state x coverage count x modality/site observed x prior-scan-count bucket",
        "permutation_changed_rows": int(np.sum(perm != np.arange(len(perm)))),
    }

    # Matched clinical-content vs coverage/timing-only controls. Current scan is injected into PRE temporal state only through the explicit branch.
    control_metrics = {}
    for mode, name in [("clinical_content_only", "v205_explicit_clinical_content_only"), ("coverage_timing_only", "v205_explicit_coverage_timing_only")]:
        masked = mask_explicit_scan_arrays(arrays, mode)
        spec = CandidateSpec(name, "pre_temporal_plus_explicit_scan", "none", "none")
        metric, _, history = _train_candidate(spec, dev, all_dev, masked, dims, v204_cfg, device, seeds[0], out, km, alpha)
        control_metrics[mode] = {"brier": metric["locked_alpha_brier"], "nll": metric["locked_alpha_nll"], "folds": metric["folds"], "semantics": "matched PRE-temporal control; current-scan information enters only via the declared explicit subset"}
    content_margin = float(control_metrics["coverage_timing_only"]["brier"] - control_metrics["clinical_content_only"]["brier"])
    specificity["matched_controls"] = control_metrics
    specificity["clinical_content_gain_over_coverage_timing_control"] = content_margin
    specificity["candidate_gain_over_coverage_timing_control"] = float(control_metrics["coverage_timing_only"]["brier"] - candidate_metric["patient_brier_4h_mean"])
    specificity["candidate_gain_over_clinical_content_control"] = float(control_metrics["clinical_content_only"]["brier"] - candidate_metric["patient_brier_4h_mean"])
    atomic_json(out / "scan_specificity.json", specificity)

    # Last-scan-only non-neural baseline.
    last_scan_metric, _ = _last_scan_only_gbt(dev, arrays, km, seed=int(cfg["final_fit"]["seed"]))
    controls["last_scan_only_gbt"] = last_scan_metric
    controls["stale_line_start_control"] = {
        "status": cfg["controls"]["stale_line_start_primary_status"],
        "reason": "absolute treatment-line identity is permanently prohibited model-facing in V2 and archived line-start endpoints are not the same scan-landmark estimand; no incompatible metric is inserted into primary selection",
    }
    atomic_json(out / "reference_controls.json", controls)

    # Natural availability/history strata.
    strata_table = _subgroup_metrics(dev, candidate_locked, km, paired_index, int(cfg["robustness"]["minimum_stratum_patients"]))
    atomic_parquet(out / "subgroup_sensitivity.parquet", strata_table)

    # Explicit branch channel-removal diagnostics (not mislabeled as end-to-end missingness counterfactuals).
    removed = mask_explicit_scan_arrays(arrays, "coverage_timing_only")
    removed_pred = _predict_seed_oof(repo, dev, removed, dims, v204_cfg, seeds[0], device)
    removed_locked = blend_logits(removed_pred["pre"], removed_pred["post"], alpha)
    removed_metric = v2_metric_bundle(dev, removed_locked, km)
    channel_report = {
        "status": "DIAGNOSTIC_ONLY",
        "full_end_to_end_stream_removal": False,
        "reason": "R1 temporal POST is an already-cached state; this diagnostic removes channels only before the explicit V2 paired-scan encoder. V2-02 paired_full_robust remains the clean before-encoder deployment-missingness reference.",
        "coverage_timing_explicit_only_brier": removed_metric["patient_brier_4h_mean"],
    }
    atomic_json(out / "channel_removal_diagnostics.json", channel_report)

    # Calibration, probability, censor support.
    cal_pred = pd.concat([
        _compact_prediction_rows(dev, baseline_locked, family="frozen_v2_03_baseline", profile="full_supported", view="LOCKED_ALPHA", fold=-1),
        _compact_prediction_rows(dev, candidate_locked, family="v2_04_candidate", profile="full_supported", view="LOCKED_ALPHA", fold=-1),
    ], ignore_index=True)
    calibration = _calibration_table(cal_pred, km, bins=int(cfg["calibration"]["bins"]))
    atomic_parquet(out / "calibration.parquet", calibration)
    ece = calibration_ece(calibration)
    def family_ece(prefix):
        vals = [v for k, v in ece.items() if k.startswith(prefix + "|")]
        return float(np.mean(vals)) if vals else float("nan")
    baseline_ece = family_ece("frozen_v2_03_baseline"); candidate_ece = family_ece("v2_04_candidate")
    calibration_regression = float(candidate_ece - baseline_ece)
    censor_support = _censor_support(dev, km)
    atomic_json(out / "calibration_and_support.json", {"ece": ece, "baseline_mean_ece": baseline_ece, "candidate_mean_ece": candidate_ece, "candidate_minus_baseline_ece": calibration_regression, "censor_support": censor_support, "candidate_curve_invariants": candidate_metric["curve_invariants"]})

    # Outcome-defined decomposition is exploratory only.
    nll_c = fractional_censor_nll_per_row(candidate_locked, dev["survival_time_days"].to_numpy(float), dev["survival_cause"].astype(int).to_numpy())
    nll_b = fractional_censor_nll_per_row(baseline_locked, dev["survival_time_days"].to_numpy(float), dev["survival_cause"].astype(int).to_numpy())
    exploratory = []
    for cause in sorted(dev["survival_cause"].astype(int).unique()):
        m = dev["survival_cause"].astype(int).to_numpy() == cause
        exploratory.append({"cause": int(cause), "rows": int(m.sum()), "candidate_mean_nll": float(nll_c[m].mean()), "baseline_mean_nll": float(nll_b[m].mean()), "delta": float((nll_c[m] - nll_b[m]).mean())})
    atomic_json(out / "exploratory_outcome_decomposition.json", {"prospective_rule_use": False, "rows": exploratory})

    # Deterministic invariance inheritance and batch-order replay.
    inv = json.loads((repo / "artifacts/dynamic_scan_v2/v2_01/invariance_tests.json").read_text())
    batch_pred = _predict_seed_oof(repo, dev.iloc[::-1].reset_index(drop=True), arrays, dims, v204_cfg, seeds[0], device)
    rev_keys = list(zip(dev.iloc[::-1]["patient_id"], dev.iloc[::-1]["scan_episode_id"]))
    orig_map = {k: i for i, k in enumerate(zip(dev["patient_id"], dev["scan_episode_id"]))}
    back = np.asarray([orig_map[k] for k in rev_keys], dtype=np.int64)
    batch_order_post_max_abs = float(np.max(np.abs(batch_pred["post"] - cpost[back])))
    deterministic = {
        "future_append_invariance": inv.get("append_future_scan_invariant", False),
        "same_day_exclusion": inv.get("unrelated_same_day_extra_field_invariant", False),
        "line_label_invariance": inv.get("line_label_perturbation_invariant", False),
        "key_alignment_exact": True,
        "batch_order_post_max_abs": batch_order_post_max_abs,
        "batch_order_invariant": batch_order_post_max_abs <= 2e-5,
    }
    atomic_json(out / "deterministic_robustness.json", deterministic)

    # Exact W0/W3/W7 sensitivity under repaired R1 temporal state + V2 paired-scan representation.
    window = _window_sensitivity(
        repo, out, dev, paired_index, dims, v204_cfg, alpha, km, seeds[0], device, policy,
        float(cfg["robustness"]["window_brier_spread_limit"]), int(cfg["robustness"]["minimum_window_common_rows"]),
    )
    atomic_json(out / "window_sensitivity.json", window)

    # Final candidate lock.
    nll_delta = float(candidate_metric["fractional_censor_patient_mean_nll_v1"] - baseline_metric["fractional_censor_patient_mean_nll_v1"])
    decision = decide_candidate_lock(
        seed_report=seed_report,
        nll_delta=nll_delta,
        nll_guardrail=float(cfg["robustness"]["nll_guardrail"]),
        reassignment_gain=reassignment_gain,
        minimum_reassignment_gain=float(cfg["specificity"]["minimum_correct_vs_reassigned_brier_gain"]),
        content_margin=content_margin,
        minimum_content_margin=float(cfg["specificity"]["content_control_margin"]),
        calibration_regression=calibration_regression,
        maximum_calibration_regression=float(cfg["calibration"]["maximum_ece_regression"]),
        window_pass=window["status"] == "PASS",
    )
    lock = {
        "selected": decision.selected, "v2_04_candidate_retained": decision.candidate_passes, "checks": decision.checks,
        "candidate": {"name": SELECTED_NAME, "representation_mode": SELECTED_REPRESENTATION, "genomic_mode": SELECTED_GENOMIC, "auxiliary_mode": "none", "locked_alpha": alpha},
        "candidate_metric": {"brier": candidate_metric["patient_brier_4h_mean"], "nll": candidate_metric["fractional_censor_patient_mean_nll_v1"]},
        "baseline_metric": {"brier": baseline_metric["patient_brier_4h_mean"], "nll": baseline_metric["fractional_censor_patient_mean_nll_v1"]},
        "bootstrap": boot, "seed_robustness": seed_report,
    }

    # Fit one all-development internal candidate only after the robustness choice is frozen.
    final_model_path = None
    if decision.candidate_passes:
        full = dev.copy().reset_index(drop=True); full["patient_weight"] = patient_balance_weights(full)
        spec = CandidateSpec(SELECTED_NAME, SELECTED_REPRESENTATION, SELECTED_GENOMIC, "none")
        final_model_path = out / "internal_candidate" / "final_model.pt"
        _train_one_fold(spec, full, full.iloc[:0].copy(), all_dev.iloc[:0].copy(), arrays, dims, v204_cfg, device, int(cfg["final_fit"]["seed"]), final_model_path)
        lock["final_model"] = {"path": str(final_model_path.relative_to(repo)), "sha256": sha256_file(final_model_path), "training_rows": len(full), "training_patients": int(full["patient_id"].nunique()), "seed": int(cfg["final_fit"]["seed"]), "note": "fit after development candidate lock; not used to estimate development performance"}
    else:
        lock["final_model"] = {"status": "FALLBACK_REUSES_FROZEN_V2_03_SPEC; final deploy fit deferred to V2-07"}
    atomic_json(out / "candidate_lock.json", lock)

    # New-external protocol is frozen without opening old DFCI/VICC outcomes.
    external_protocol = {
        "status": "EXTERNAL_CONFIRMATION_PENDING",
        "candidate_lock_sha256": None,
        "eligible_evidence": "genuinely new patients not present in upstream train/pretrain/development/legacy external evaluation",
        "prohibited": ["DFCI/VICC outcome reuse for V2 selection", "candidate-specific recalibration before primary evaluation", "random subset of old external patients as new validation"],
        "prediction_contract": {"PRE": True, "POST": True, "locked_alpha": alpha, "causes": ["no_event", "progression", "death", "switch"], "horizons_months": list(HORIZONS_MONTHS)},
        "primary_metric": "patient-balanced PFS Brier mean at 3/6/12/18m with prespecified censoring nuisance policy",
        "center_reporting": "separate center results plus pooled descriptive summary only if protocol permits",
        "freeze_before_outcomes": ["identity/overlap checks", "time origin", "predictor crosswalk", "endpoint definitions", "inclusion rules", "missingness profile", "prediction hashes", "metrics"],
    }
    atomic_json(out / "prospective_external_protocol.json", external_protocol)
    external_protocol["candidate_lock_sha256"] = sha256_file(out / "candidate_lock.json")
    atomic_json(out / "prospective_external_protocol.json", external_protocol)

    # Model/data cards and benchmark report.
    model_card = f"""# OncoTwin V2-05 internal candidate model card\n\nStatus: **{'LOCKED V2-04 CANDIDATE' if decision.candidate_passes else 'FALLBACK TO V2-03'}**\n\nPrimary development population: {len(dev)} survival-eligible scan landmarks from {dev['patient_id'].nunique()} patients.\n\nSelected rule: `{decision.selected}`.\n\nLocked scan update alpha: {alpha:.4f}.\n\nV2-04 candidate Brier: {candidate_metric['patient_brier_4h_mean']:.9f}; frozen V2-03 baseline: {baseline_metric['patient_brier_4h_mean']:.9f}.\n\nThis is a research prognostic model, not a clinical decision rule. External confirmation remains pending.\n"""
    data_card = """# OncoTwin V2-05 data card\n\nDevelopment selection uses CHORD train/validation patients only. Historical CHORD test remains read-only and DFCI/VICC outcomes are not opened. Patient identity defines folds. Current scan episodes use exact source-component semantics; absolute treatment-line identity is prohibited model-facing.\n"""
    atomic_text(out / "model_card.md", model_card); atomic_text(out / "data_card.md", data_card)
    benchmark = {
        "candidate_vs_baseline_brier_delta": float(candidate_metric["patient_brier_4h_mean"] - baseline_metric["patient_brier_4h_mean"]),
        "candidate_vs_baseline_nll_delta": nll_delta,
        "reference_controls": controls,
        "specificity": specificity,
        "window": window,
        "calibration": {"baseline_mean_ece": baseline_ece, "candidate_mean_ece": candidate_ece, "regression": calibration_regression},
    }
    atomic_json(out / "benchmark_report.json", benchmark)

    acceptance = {
        "v2_04_pass_required": True,
        "analysis_plan_frozen_before_v205_results": True,
        "exact_17194_primary_rows": len(dev) == 17194,
        "paired_pre_post_same_rows": len(cpre) == len(cpost) == len(dev),
        "seed_fold_variation_reported": len(seed_rows) == 3,
        "paired_patient_bootstrap_completed": boot["repetitions"] == int(cfg["bootstrap"]["repetitions"]),
        "scan_reassignment_specificity_completed": specificity["permutation_changed_rows"] > 0,
        "clinical_content_vs_coverage_timing_control_completed": set(control_metrics) == {"clinical_content_only", "coverage_timing_only"},
        "last_scan_only_control_completed": "last_scan_only_gbt" in controls,
        "stale_line_start_incompatibility_explicit": controls["stale_line_start_control"]["status"] == "NOT_PRIMARY_COMPARABLE_BY_DESIGN",
        "w0_w3_w7_repaired_temporal_sensitivity_completed": window["common_rows"] >= int(cfg["robustness"]["minimum_window_common_rows"]),
        "natural_missingness_history_strata_reported": len(strata_table) > 0,
        "deterministic_invariances_pass": all([deterministic["future_append_invariance"], deterministic["same_day_exclusion"], deterministic["line_label_invariance"], deterministic["batch_order_invariant"]]),
        "probability_invariants_pass": candidate_metric["curve_invariants"]["status"] == "PASS",
        "censor_support_reported": len(censor_support) == 4,
        "candidate_lock_written": True,
        "external_protocol_frozen_without_old_external_outcomes": True,
        "protected_external_rows_not_opened": True,
        "external_predictions_not_regenerated": True,
    }
    if not all(acceptance.values()): raise RuntimeError(f"V2-05 engineering acceptance failed: {acceptance}")
    result = {
        "checkpoint": "V2-05", "status": "PASS", "next_checkpoint": "V2-06",
        "development_rows": len(dev), "development_patients": int(dev["patient_id"].nunique()), "locked_alpha": alpha,
        "candidate_lock": lock, "window_sensitivity": window, "specificity": specificity,
        "calibration": {"baseline_mean_ece": baseline_ece, "candidate_mean_ece": candidate_ece, "regression": calibration_regression},
        "acceptance": acceptance, "external_confirmation": "PENDING_NEW_INDEPENDENT_COHORT",
        "external_rows_opened": False, "external_predictions_regenerated": False,
    }
    atomic_json(out / "result_packet.json", result)
    memo = f"""# OncoTwin V2-05 robustness and internal candidate lock\n\nStatus: **PASS**\n\nInternal lock: **{decision.selected}**.\n\nV2-04 candidate retained: **{decision.candidate_passes}**.\n\nCandidate Brier: {candidate_metric['patient_brier_4h_mean']:.9f}; baseline: {baseline_metric['patient_brier_4h_mean']:.9f}; delta: {candidate_metric['patient_brier_4h_mean']-baseline_metric['patient_brier_4h_mean']:+.9f}.\n\nBootstrap 95% CI for candidate-minus-baseline Brier: [{boot['ci_low']:+.9f}, {boot['ci_high']:+.9f}] (conditional scope; does not include model-selection uncertainty).\n\nW0/W3/W7 common-cohort spread: {window['spread']['spread']:.9f}.\n\nExternal confirmation: **PENDING NEW INDEPENDENT COHORT**. No DFCI/VICC outcomes were opened.\n"""
    atomic_text(out / "decision_report.md", memo)
    manifest_names = [
        "frozen_analysis_plan.json", "input_integrity.json", "seed_fold_stability.json", "paired_patient_bootstrap.json",
        "scan_specificity.json", "reference_controls.json", "subgroup_sensitivity.parquet", "channel_removal_diagnostics.json",
        "calibration.parquet", "calibration_and_support.json", "exploratory_outcome_decomposition.json", "deterministic_robustness.json",
        "window_sensitivity.json", "candidate_lock.json", "prospective_external_protocol.json", "model_card.md", "data_card.md",
        "benchmark_report.json", "result_packet.json", "decision_report.md",
    ]
    if final_model_path is not None: manifest_names.append("internal_candidate/final_model.pt")
    atomic_json(out / "artifact_manifest.json", {"status": "PASS", "files": {name: sha256_file(out / name) for name in manifest_names}})
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
