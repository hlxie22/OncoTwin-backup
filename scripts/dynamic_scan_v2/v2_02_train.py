#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from access import AccessPolicy
from common import atomic_json, atomic_parquet, atomic_text, sha256_file
from metrics import (
    HORIZONS_MONTHS,
    fit_censoring_km,
    fractional_censor_nll_per_row,
    horizon_status_and_weight,
    curves_from_logits,
    v2_metric_bundle,
)
from v2_02_models import (
    ADMIN_HORIZON_DAYS,
    MONTH_DAYS,
    MONTHS,
    CAUSES,
    PROFILE_NAMES,
    PROFILE_TO_CODE,
    LinearDiscreteHazard,
    PairedStructuredSurvivalModel,
    StructuredBatch,
    TemporalFixedSurvivalModel,
    apply_observation_profiles,
    fractional_censor_nll_torch,
    profile_probabilities,
    weighted_patient_objective,
)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}.npz")
    np.savez_compressed(tmp, **arrays)
    with np.load(tmp) as check:
        if set(check.files) != set(arrays):
            raise RuntimeError(f"NPZ verification failed: {path}")
        for key, arr in arrays.items():
            if check[key].shape != arr.shape:
                raise RuntimeError(f"NPZ shape verification failed: {key}")
    tmp.replace(path)


def patient_balance_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("patient_id", observed=True)["patient_id"].transform("count").to_numpy(dtype=np.float64)
    return (1.0 / np.maximum(counts, 1.0)).astype(np.float32)


def attach_and_validate_development_folds(
    frame: pd.DataFrame,
    folds: pd.DataFrame,
    fold_count: int = 3,
) -> pd.DataFrame:
    """Attach the frozen V2 fold registry and validate its sentinel semantics.

    V2-00 encodes original CHORD train/val patients with folds 0..K-1 and
    original CHORD test patients with v2_fold=-1 plus
    v2_role=historical_test_read_only.  -1 is therefore a deliberate
    non-development sentinel, not a missing assignment.
    """
    frame = frame.copy()
    folds = folds.copy()
    for df in (frame, folds):
        if "patient_id" not in df.columns:
            raise RuntimeError("Fold validation requires patient_id")
        df["patient_id"] = df["patient_id"].astype(str).str.strip()

    if "split" not in frame.columns:
        raise RuntimeError("Fold validation requires frame split")
    frame_split = frame["split"].astype(str).str.strip().str.lower().replace({"validation": "val"})
    if not frame_split.isin(["train", "val", "test"]).all():
        bad = sorted(frame.loc[~frame_split.isin(["train", "val", "test"]), "split"].astype(str).unique().tolist())
        raise RuntimeError(f"Unexpected R2 split labels: {bad}")
    frame["split"] = frame_split

    if "development_fold" not in folds.columns:
        candidates = [c for c in folds.columns if c.lower() in {"v2_fold", "fold", "development_fold"}]
        if len(candidates) != 1:
            raise RuntimeError("Cannot identify development fold column")
        folds = folds.rename(columns={candidates[0]: "development_fold"})

    keep = ["patient_id", "development_fold"]
    for c in ("original_split", "v2_role"):
        if c in folds.columns:
            keep.append(c)
    folds = folds[keep].copy()

    # A patient must have exactly one frozen registry row.  drop_duplicates
    # would otherwise hide a conflicting assignment.
    if folds["patient_id"].duplicated().any():
        dup = folds.loc[folds["patient_id"].duplicated(keep=False), "patient_id"].unique().tolist()[:10]
        raise RuntimeError(f"Duplicate patient assignments in development fold registry: {dup}")

    merged = frame.merge(folds, on="patient_id", how="left", validate="many_to_one")
    if merged["development_fold"].isna().any():
        missing = merged.loc[merged["development_fold"].isna(), "patient_id"].unique().tolist()[:10]
        raise RuntimeError(f"R2 patient missing frozen fold registry assignment: {missing}")

    fold_numeric = pd.to_numeric(merged["development_fold"], errors="coerce")
    if fold_numeric.isna().any():
        raise RuntimeError("Non-numeric development fold in frozen registry")
    merged["development_fold"] = fold_numeric.astype(int)

    is_dev = merged["split"].isin(["train", "val"])
    is_test = merged["split"].eq("test")
    allowed_dev = set(range(int(fold_count)))
    observed_dev = set(merged.loc[is_dev, "development_fold"].unique().tolist())
    if not observed_dev.issubset(allowed_dev):
        raise RuntimeError(f"Development patient has invalid frozen fold: {sorted(observed_dev)}")
    if not (merged.loc[is_test, "development_fold"] == -1).all():
        bad = merged.loc[is_test & merged["development_fold"].ne(-1), ["patient_id", "development_fold"]].drop_duplicates().head(10)
        raise RuntimeError(f"Historical test patient assigned to a development fold: {bad.to_dict(orient='records')}")

    if "original_split" in merged.columns:
        original = merged["original_split"].astype(str).str.strip().str.lower().replace({"validation": "val"})
        mismatch = original.ne(merged["split"])
        if mismatch.any():
            bad = merged.loc[mismatch, ["patient_id", "split", "original_split"]].drop_duplicates().head(10)
            raise RuntimeError(f"Frozen fold registry original_split disagrees with R2 split: {bad.to_dict(orient='records')}")

    if "v2_role" in merged.columns:
        role = merged["v2_role"].astype(str).str.strip()
        if not role.loc[is_dev].eq("v2_development").all():
            bad = merged.loc[is_dev & role.ne("v2_development"), ["patient_id", "v2_role"]].drop_duplicates().head(10)
            raise RuntimeError(f"Development patient has wrong V2 role: {bad.to_dict(orient='records')}")
        if not role.loc[is_test].eq("historical_test_read_only").all():
            bad = merged.loc[is_test & role.ne("historical_test_read_only"), ["patient_id", "v2_role"]].drop_duplicates().head(10)
            raise RuntimeError(f"Historical test patient has wrong V2 role: {bad.to_dict(orient='records')}")

    return merged


def standardizer_fit(x: np.ndarray) -> dict[str, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def standardizer_apply(x: np.ndarray, stat: dict[str, np.ndarray]) -> np.ndarray:
    return ((np.asarray(x, dtype=np.float32) - stat["mean"]) / stat["std"]).astype(np.float32)


def build_person_period_rows(
    x: np.ndarray,
    time_days: np.ndarray,
    cause: np.ndarray,
    landmark_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand landmark rows to weighted conditional-hazard person-period rows.

    Fractional censor exposure receives proportional no-event weight in the
    partially observed interval, matching FRACTIONAL_CENSOR_NLL_V1.
    """
    x = np.asarray(x, dtype=np.float32)
    time_days = np.asarray(time_days, dtype=np.float64)
    cause = np.asarray(cause, dtype=np.int64)
    landmark_weights = np.asarray(landmark_weights, dtype=np.float64)
    if not (len(x) == len(time_days) == len(cause) == len(landmark_weights)):
        raise ValueError("person-period length mismatch")

    counts = np.zeros(len(x), dtype=np.int64)
    completed_arr = np.zeros(len(x), dtype=np.int64)
    fraction_arr = np.zeros(len(x), dtype=np.float64)
    event_bin_arr = np.zeros(len(x), dtype=np.int64)
    for i, (t, c) in enumerate(zip(time_days, cause)):
        if c == 0:
            ratio = min(max(float(t), 0.0), ADMIN_HORIZON_DAYS) / MONTH_DAYS
            completed = min(int(math.floor(ratio + 1e-12)), MONTHS)
            fraction = max(0.0, min(1.0, ratio - completed)) if completed < MONTHS else 0.0
            if fraction < 1e-10:
                fraction = 0.0
            completed_arr[i] = completed
            fraction_arr[i] = fraction
            counts[i] = completed + int(fraction > 0)
        else:
            event_bin = int(np.clip(np.ceil(float(t) / MONTH_DAYS - 1e-12), 1, MONTHS))
            event_bin_arr[i] = event_bin
            counts[i] = event_bin
    total = int(counts.sum())
    xp = np.empty((total, x.shape[1] + 1), dtype=np.float32)
    yp = np.empty(total, dtype=np.int64)
    wp = np.empty(total, dtype=np.float32)
    cursor = 0
    for i in range(len(x)):
        k = int(counts[i])
        if k == 0:
            continue
        sl = slice(cursor, cursor + k)
        xp[sl, :-1] = x[i]
        xp[sl, -1] = np.arange(1, k + 1, dtype=np.float32) / float(MONTHS)
        yp[sl] = 0
        wp[sl] = float(landmark_weights[i])
        if cause[i] == 0:
            completed = int(completed_arr[i])
            fraction = float(fraction_arr[i])
            if fraction > 0:
                wp[cursor + completed] *= fraction
        else:
            yp[cursor + int(event_bin_arr[i]) - 1] = int(cause[i])
        cursor += k
    return xp, yp, wp


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _torch_batch(arrays: dict[str, np.ndarray], idx: np.ndarray, device: torch.device) -> StructuredBatch:
    def t(name: str) -> torch.Tensor:
        return torch.from_numpy(np.asarray(arrays[name][idx], dtype=np.float32)).to(device, non_blocking=True)
    return StructuredBatch(
        history_core=t("history_core"),
        history_core_mask=t("history_core_mask"),
        utilization_core=t("utilization_core"),
        utilization_core_mask=t("utilization_core_mask"),
        current_core=t("current_core"),
        current_core_mask=t("current_core_mask"),
        change_core=t("change_core"),
        change_core_mask=t("change_core_mask"),
        optional_history=t("optional_history"),
        optional_history_mask=t("optional_history_mask"),
        optional_history_recency=t("optional_history_recency"),
        optional_history_recency_mask=t("optional_history_recency_mask"),
        optional_current=t("optional_current"),
        optional_current_mask=t("optional_current_mask"),
        stream_availability=t("stream_availability"),
        stream_availability_mask=t("stream_availability_mask"),
        tumor=t("tumor"),
        genomic_available=t("genomic_available").reshape(-1),
        genomic_age_scaled=t("genomic_age_scaled").reshape(-1),
    )


def _torch_targets(frame: pd.DataFrame, local: np.ndarray, device: torch.device):
    time_days = torch.tensor(frame.iloc[local]["survival_time_days"].to_numpy(dtype=np.float32), device=device)
    cause = torch.tensor(frame.iloc[local]["survival_cause"].astype(int).to_numpy(), device=device)
    weight = torch.tensor(frame.iloc[local]["patient_weight"].to_numpy(dtype=np.float32), device=device)
    return time_days, cause, weight


def _train_torch_pair(
    *,
    model: nn.Module,
    family: str,
    train_frame: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    global_rows: np.ndarray,
    config: dict[str, Any],
    device: torch.device,
    seed: int,
    robust: bool,
    temporal: bool,
    out_path: Path,
) -> dict[str, Any]:
    _seed_everything(seed)
    model = model.to(device)
    lr = float(config["torch_learning_rate"])
    epochs = int(config["torch_epochs"])
    if family == "linear_core":
        lr = float(config["linear_learning_rate"])
        epochs = int(config["linear_epochs"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(config["torch_weight_decay"]))
    batch_size = int(config["torch_batch_size"])
    rng = np.random.default_rng(seed)
    probs = profile_probabilities(config["observation_profiles"])
    history: list[dict[str, Any]] = []
    model.train()

    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(train_frame))
        total_num = 0.0
        total_den = 0.0
        profile_counts = {name: 0 for name in PROFILE_NAMES}
        profile_patient_counts = {name: 0 for name in PROFILE_NAMES}
        epoch_profile_codes = None
        if not robust:
            profile_counts["full_supported"] = int(len(train_frame))
            profile_patient_counts["full_supported"] = int(train_frame["patient_id"].nunique())
        if robust:
            patient_text = train_frame["patient_id"].astype(str).to_numpy()
            unique_patients = pd.unique(patient_text)
            patient_codes = rng.choice(len(PROFILE_NAMES), size=len(unique_patients), p=probs)
            mapping = {patient: int(code) for patient, code in zip(unique_patients, patient_codes)}
            epoch_profile_codes = np.asarray([mapping[p] for p in patient_text], dtype=np.int64)
            for code, name in enumerate(PROFILE_NAMES):
                profile_patient_counts[name] = int((patient_codes == code).sum())
        for start in range(0, len(order), batch_size):
            local = order[start : start + batch_size]
            g = global_rows[local]
            time_days, cause, weight = _torch_targets(train_frame, local, device)
            optimizer.zero_grad(set_to_none=True)

            if family == "linear_core":
                pre = torch.from_numpy(arrays["engineered_core_pre_z"][g]).to(device)
                post = torch.from_numpy(arrays["engineered_core_post_z"][g]).to(device)
                pre_logits = model(pre)
                post_logits = model(post)
            elif temporal:
                tp = torch.from_numpy(arrays["temporal_pre"][g]).to(device)
                tq = torch.from_numpy(arrays["temporal_post"][g]).to(device)
                tumor = torch.from_numpy(arrays["tumor"][g]).to(device)
                context = torch.from_numpy(arrays["portable_context"][g]).to(device)
                pre_logits, post_logits = model(tp, tq, tumor, context)
            else:
                b = _torch_batch(arrays, g, device)
                if robust:
                    assert epoch_profile_codes is not None
                    codes_np = epoch_profile_codes[local]
                    for code, name in enumerate(PROFILE_NAMES):
                        profile_counts[name] += int((codes_np == code).sum())
                    codes = torch.tensor(codes_np, dtype=torch.long, device=device)
                    b = apply_observation_profiles(b, codes)
                pre_logits, post_logits = model(b)

            pre_loss = fractional_censor_nll_torch(pre_logits, time_days, cause)
            post_loss = fractional_censor_nll_torch(post_logits, time_days, cause)
            pair_loss = 0.5 * (pre_loss + post_loss)
            loss = weighted_patient_objective(pair_loss, weight)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss: {family}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
            optimizer.step()
            total_num += float((pair_loss.detach() * weight).sum().cpu())
            total_den += float(weight.sum().cpu())

        row = {
            "epoch": epoch,
            "patient_balanced_pair_fractional_nll": total_num / max(total_den, 1e-12),
            "profile_landmark_counts": profile_counts,
            "profile_patient_counts": profile_patient_counts,
        }
        history.append(row)
        print("[V2_02_TRAIN]", family, row, flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "family": family,
            "seed": seed,
            "model_state": model.state_dict(),
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "history": history,
        },
        out_path,
    )
    return {
        "model": model.eval(),
        "history": history,
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }


def _predict_torch(
    *,
    model: nn.Module,
    family: str,
    eval_frame: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    global_rows: np.ndarray,
    device: torch.device,
    profile: str,
    temporal: bool,
    batch_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    pre_all: list[np.ndarray] = []
    post_all: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(eval_frame), batch_size):
            local = np.arange(start, min(start + batch_size, len(eval_frame)))
            g = global_rows[local]
            if family == "linear_core":
                pre = torch.from_numpy(arrays["engineered_core_pre_z"][g]).to(device)
                post = torch.from_numpy(arrays["engineered_core_post_z"][g]).to(device)
                a, b = model(pre), model(post)
            elif temporal:
                tp = torch.from_numpy(arrays["temporal_pre"][g]).to(device)
                tq = torch.from_numpy(arrays["temporal_post"][g]).to(device)
                tumor = torch.from_numpy(arrays["tumor"][g]).to(device)
                context = torch.from_numpy(arrays["portable_context"][g]).to(device)
                a, b = model(tp, tq, tumor, context)
            else:
                sb = _torch_batch(arrays, g, device)
                codes = torch.full((len(local),), PROFILE_TO_CODE[profile], dtype=torch.long, device=device)
                sb = apply_observation_profiles(sb, codes)
                a, b = model(sb)
            pre_all.append(a.float().cpu().numpy())
            post_all.append(b.float().cpu().numpy())
    return np.concatenate(pre_all), np.concatenate(post_all)


def _gbt_fit_predictor(
    *,
    x_pre: np.ndarray,
    x_post: np.ndarray,
    frame: pd.DataFrame,
    config: dict[str, Any],
    seed: int,
):
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
    except Exception as exc:
        raise RuntimeError("scikit-learn is required for the prespecified gbt_core baseline") from exc
    t = frame["survival_time_days"].to_numpy(dtype=float)
    c = frame["survival_cause"].astype(int).to_numpy()
    w = frame["patient_weight"].to_numpy(dtype=float)
    xp0, yp0, wp0 = build_person_period_rows(x_pre, t, c, 0.5 * w)
    xp1, yp1, wp1 = build_person_period_rows(x_post, t, c, 0.5 * w)
    xp = np.concatenate([xp0, xp1], axis=0)
    yp = np.concatenate([yp0, yp1], axis=0)
    wp = np.concatenate([wp0, wp1], axis=0)
    cfg = config["gbt"]
    clf = HistGradientBoostingClassifier(
        loss="log_loss",
        max_iter=int(cfg["max_iter"]),
        learning_rate=float(cfg["learning_rate"]),
        max_leaf_nodes=int(cfg["max_leaf_nodes"]),
        min_samples_leaf=int(cfg["min_samples_leaf"]),
        l2_regularization=float(cfg["l2_regularization"]),
        random_state=int(seed),
    )
    print("[V2_02_GBT] person_period_rows=", len(xp), flush=True)
    clf.fit(xp, yp, sample_weight=wp)
    return clf


def _gbt_logits(clf, x: np.ndarray, batch_rows: int = 5000) -> np.ndarray:
    out = np.empty((len(x), MONTHS, CAUSES), dtype=np.float32)
    for start in range(0, len(x), batch_rows):
        stop = min(start + batch_rows, len(x))
        base = x[start:stop]
        n = len(base)
        expanded = np.repeat(base, MONTHS, axis=0)
        month = np.tile(np.arange(1, MONTHS + 1, dtype=np.float32) / MONTHS, n)
        expanded = np.column_stack([expanded, month]).astype(np.float32, copy=False)
        prob_raw = clf.predict_proba(expanded)
        prob = np.full((len(expanded), CAUSES), 1e-8, dtype=np.float64)
        for j, cls in enumerate(clf.classes_):
            prob[:, int(cls)] = prob_raw[:, j]
        prob /= prob.sum(axis=1, keepdims=True)
        out[start:stop] = np.log(np.clip(prob, 1e-8, 1.0)).reshape(n, MONTHS, CAUSES)
    return out


def _compact_prediction_rows(
    frame: pd.DataFrame,
    logits: np.ndarray,
    *,
    family: str,
    profile: str,
    view: str,
    fold: int,
) -> pd.DataFrame:
    curves = curves_from_logits(logits)
    nll = fractional_censor_nll_per_row(
        logits,
        frame["survival_time_days"].to_numpy(dtype=float),
        frame["survival_cause"].astype(int).to_numpy(),
    )
    out = frame[["patient_id", "scan_episode_id", "landmark_day", "survival_time_days", "survival_cause"]].copy()
    out["development_fold"] = int(fold)
    out["family"] = family
    out["profile"] = profile
    out["view"] = view
    out["fractional_nll"] = nll.astype(np.float32)
    for h in HORIZONS_MONTHS:
        out[f"pfs_{h}m"] = curves["pfs"][:, h - 1].astype(np.float32)
        out[f"risk_{h}m"] = (1.0 - curves["pfs"][:, h - 1]).astype(np.float32)
    return out


def _calibration_table(pred: pd.DataFrame, km, bins: int = 10) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (family, profile, view), group in pred.groupby(["family", "profile", "view"], observed=True):
        group = group.reset_index(drop=True)
        for h in HORIZONS_MONTHS:
            risk = group[f"risk_{h}m"].to_numpy(dtype=float)
            y, w = horizon_status_and_weight(group, h * MONTH_DAYS, km, patient_balanced=True)
            use = np.isfinite(risk) & np.isfinite(w) & (w > 0)
            if use.sum() < bins:
                continue
            order = np.argsort(risk[use], kind="mergesort")
            idx = np.where(use)[0][order]
            chunks = np.array_split(idx, bins)
            for b, chunk in enumerate(chunks, 1):
                if len(chunk) == 0:
                    continue
                ww = w[chunk]
                rows.append({
                    "family": family,
                    "profile": profile,
                    "view": view,
                    "horizon_month": int(h),
                    "bin": int(b),
                    "rows": int(len(chunk)),
                    "weight_sum": float(ww.sum()),
                    "mean_predicted_risk": float(np.average(risk[chunk], weights=ww)),
                    "observed_risk": float(np.average(y[chunk], weights=ww)),
                })
    return pd.DataFrame(rows)


def _mean_fold_summary(fold_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in fold_metrics:
        grouped.setdefault((row["family"], row["profile"], row["view"]), []).append(row)
    out: dict[str, Any] = {}
    for (family, profile, view), rows in grouped.items():
        key = f"{family}|{profile}|{view}"
        b = np.asarray([r["patient_brier_4h_mean"] for r in rows], dtype=float)
        n = np.asarray([r["fractional_censor_patient_mean_nll_v1"] for r in rows], dtype=float)
        out[key] = {
            "family": family,
            "profile": profile,
            "view": view,
            "folds": len(rows),
            "primary_brier_mean": float(b.mean()),
            "primary_brier_std": float(b.std(ddof=0)),
            "fractional_patient_nll_mean": float(n.mean()),
            "fold_primary_brier": [float(x) for x in b],
        }
    return out


def _select_candidate(summary: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    sel = config["selection"]
    tie = float(sel["tie_margin"])
    full = {
        v["family"]: v
        for v in summary.values()
        if v["profile"] == "full_supported" and v["view"] == "POST"
    }
    if not full:
        raise RuntimeError("No full-supported POST CV metrics available")

    robust = full.get("paired_full_robust")
    full_model = full.get("paired_full")
    portable_robust = next((v for v in summary.values() if v["family"] == "paired_full_robust" and v["profile"] == "portable_scan_core" and v["view"] == "POST"), None)
    portable_full = next((v for v in summary.values() if v["family"] == "paired_full" and v["profile"] == "portable_scan_core" and v["view"] == "POST"), None)
    robust_checks = {
        "available": all(x is not None for x in [robust, full_model, portable_robust, portable_full]),
        "portable_noninferior": False,
        "full_noninferior": False,
        "portable_delta_robust_minus_plain": None,
        "full_delta_robust_minus_plain": None,
    }
    if robust_checks["available"]:
        pdiff = portable_robust["primary_brier_mean"] - portable_full["primary_brier_mean"]
        fdiff = robust["primary_brier_mean"] - full_model["primary_brier_mean"]
        robust_checks.update({
            "portable_delta_robust_minus_plain": float(pdiff),
            "full_delta_robust_minus_plain": float(fdiff),
            "portable_noninferior": bool(pdiff <= float(sel["portable_noninferiority_margin"])),
            "full_noninferior": bool(fdiff <= float(sel["full_information_noninferiority_margin"])),
        })
    robust_retained = bool(robust_checks["portable_noninferior"] and robust_checks["full_noninferior"])

    eligible = dict(full)
    if not robust_retained:
        eligible.pop("paired_full_robust", None)
    best_value = min(v["primary_brier_mean"] for v in eligible.values())
    within = [name for name, v in eligible.items() if v["primary_brier_mean"] <= best_value + tie]
    simplicity = ["linear_core", "gbt_core", "paired_core", "r1_temporal_fixed", "paired_full", "paired_full_robust"]
    chosen = next(name for name in simplicity if name in within)

    return {
        "best_screening_family": chosen,
        "best_full_supported_brier": float(eligible[chosen]["primary_brier_mean"]),
        "families_within_tie_margin": within,
        "tie_margin": tie,
        "robust_retained": robust_retained,
        "robustness_checks": robust_checks,
        "decision_rule": "Lowest mean 3-fold primary Brier; within 0.0005 prefer prespecified simpler family. Robust family excluded if portable/full noninferiority gates fail.",
        "historical_test_used_for_selection": False,
    }


def _hash_artifacts(paths: list[Path]) -> dict[str, Any]:
    out = {}
    for p in paths:
        if p.is_file():
            out[str(p.name)] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    out = repo / "artifacts/dynamic_scan_v2/v2_02"
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((repo / "configs/dynamic_scan_v2/v2_02_training.json").read_text())
    policy = AccessPolicy.from_json(repo, repo / "configs/dynamic_scan_v2/data_access_policy.json")

    v201 = json.loads((repo / "artifacts/dynamic_scan_v2/v2_01/result_packet.json").read_text())
    if v201.get("status") != "PASS":
        raise RuntimeError("V2-01 must PASS before V2-02")
    protocol = json.loads((repo / "configs/dynamic_scan_v2/development_protocol.json").read_text())
    if config["model_families"] != ["linear_core", "gbt_core", "paired_core", "paired_full", "paired_full_robust", "r1_temporal_fixed"]:
        raise RuntimeError("V2-02 family registry changed from frozen config")

    paired_index = policy.read_parquet(repo / "artifacts/dynamic_scan_v2/v2_01/paired_scan_index.parquet")
    r2_index = policy.read_parquet(
        repo / "artifacts/checkpoint7r2_transport_safe_supervised/prepared/scan_index.parquet",
        columns=["patient_id", "scan_episode_id", "landmark_day", "split", "survival_mask", "survival_time_days", "survival_cause"],
    )
    keys = ["patient_id", "scan_episode_id"]
    for df in (paired_index, r2_index):
        df["patient_id"] = df["patient_id"].astype(str).str.strip()
        df["scan_episode_id"] = df["scan_episode_id"].astype(str).str.strip()
    if not np.array_equal(paired_index[keys].to_numpy(), r2_index[keys].to_numpy()):
        raise RuntimeError("V2-01/R2 row alignment is not exact")

    folds = policy.read_parquet(repo / "artifacts/dynamic_scan_v2/v2_00/development_folds.parquet")
    frame = attach_and_validate_development_folds(r2_index, folds, fold_count=int(protocol["folds"]["count"]))
    frame["global_row"] = np.arange(len(frame), dtype=np.int64)
    is_dev_split = frame["split"].isin(["train", "val"])
    eligible = is_dev_split & frame["survival_mask"].fillna(False).astype(bool)
    dev = frame.loc[eligible].copy().reset_index(drop=True)

    # Fixed censoring nuisance estimator from original CHORD training survival rows only.
    km_frame = frame.loc[
        frame["split"].astype(str).str.lower().eq("train") & frame["survival_mask"].fillna(False).astype(bool)
    ].copy()
    km = fit_censoring_km(km_frame["survival_time_days"].to_numpy(dtype=float), km_frame["survival_cause"].astype(int).to_numpy())

    with policy.np_load(repo / "artifacts/dynamic_scan_v2/v2_01/paired_scan_tensors.npz") as z:
        arrays = {k: np.asarray(z[k], dtype=np.float32) for k in z.files}
    tumor = np.asarray(policy.np_load(repo / "artifacts/checkpoint7r2_transport_safe_supervised/prepared/tumor_embeddings_f16.npy", mmap_mode="r"), dtype=np.float32)
    temporal = np.asarray(policy.np_load(repo / "artifacts/checkpoint7r2_transport_safe_supervised/prepared/temporal_prepost_f16.npy", mmap_mode="r"), dtype=np.float32)
    context = np.asarray(policy.np_load(repo / "artifacts/checkpoint7r2_transport_safe_supervised/prepared/context_features_f32.npy", mmap_mode="r"), dtype=np.float32)
    n = len(frame)
    if tumor.shape != (n, 128) or temporal.shape != (n, 2, 192) or context.shape != (n, 6):
        raise RuntimeError(f"Unexpected R2 array shapes: tumor={tumor.shape} temporal={temporal.shape} context={context.shape}")
    arrays.update({
        "tumor": tumor,
        "temporal_pre": temporal[:, 0],
        "temporal_post": temporal[:, 1],
        # R2 context indices 0,1,3 are line-derived and permanently excluded.
        "portable_context": context[:, [2, 4, 5]].astype(np.float32),
        "genomic_available": context[:, 4:5].astype(np.float32),
        "genomic_age_scaled": context[:, 5:6].astype(np.float32),
    })

    schema = json.loads((repo / "artifacts/dynamic_scan_v2/v2_01/feature_schema.json").read_text())
    dims = {name: len(values) for name, values in schema["blocks"].items()}
    eng_schema = json.loads((repo / "artifacts/dynamic_scan_v2/v2_01/engineered_control_schema.json").read_text())
    core_names = list(eng_schema["core_names"])
    core = arrays["engineered_core_control"].copy()
    pre_core = core.copy()
    post_only_cols = [i for i, name in enumerate(core_names) if name.startswith("current__") or name.startswith("change__")]
    pre_core[:, post_only_cols] = 0.0

    # Profile/input contract is written before fitting so a crash still leaves provenance.
    coverage = json.loads((repo / "artifacts/dynamic_scan_v2/v2_01/capability_coverage.json").read_text())
    profile_manifest = {
        "version": config["version"],
        "profiles": config["profile_semantics"],
        "training_probabilities": config["observation_profiles"],
        "source_availability_basis": {
            "development_modality_observed_fraction": coverage["development_modality_observed_fraction"],
            "development_site_positive_stream_observed_fraction": coverage["development_site_positive_stream_observed_fraction"],
            "r2_genomic_availability_source": "checkpoint7r2 prepared genomic_available; strict availability_day < landmark_day",
        },
        "robust_input_policy": config["robust_input_policy"],
        "masked_latent_regeneration": {
            "r1_temporal_cache_used_by_robust_branch": False,
            "structured_source_tensor_sha256": sha256_file(repo / "artifacts/dynamic_scan_v2/v2_01/paired_scan_tensors.npz"),
            "rule": "No masked temporal cache is needed because the robust branch bypasses R1 latent state and applies profile masks directly to source-transparent tensors before PairedScanEncoder.",
        },
        "status": "FROZEN_BEFORE_TRAINING",
    }
    atomic_json(out / "observation_profile_manifest.json", profile_manifest)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("[V2_02_DEVICE]", device, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU", flush=True)
    seed = int(config["screening_seed"])
    fold_metrics: list[dict[str, Any]] = []
    prediction_parts: list[pd.DataFrame] = []
    full_oof: dict[str, dict[str, np.ndarray]] = {}
    train_records: dict[str, Any] = {}
    profile_sets = {
        "linear_core": ["full_supported"],
        "gbt_core": ["full_supported"],
        "paired_core": ["full_supported"],
        "paired_full": list(PROFILE_NAMES),
        "paired_full_robust": list(PROFILE_NAMES),
        "r1_temporal_fixed": ["full_supported"],
    }

    for family in config["model_families"]:
        full_oof[family] = {
            "pre": np.full((len(dev), MONTHS, CAUSES), np.nan, dtype=np.float32),
            "post": np.full((len(dev), MONTHS, CAUSES), np.nan, dtype=np.float32),
        }
        train_records[family] = {}
        for fold in (0, 1, 2):
            assess_local = np.where(dev["development_fold"].to_numpy() == fold)[0]
            train_local = np.where(dev["development_fold"].to_numpy() != fold)[0]
            assess = dev.iloc[assess_local].copy().reset_index(drop=True)
            train = dev.iloc[train_local].copy().reset_index(drop=True)
            assess_global = assess["global_row"].to_numpy(dtype=np.int64)
            train_global = train["global_row"].to_numpy(dtype=np.int64)
            train["patient_weight"] = patient_balance_weights(train)
            assess["patient_weight"] = patient_balance_weights(assess)
            if set(train["patient_id"]) & set(assess["patient_id"]):
                raise RuntimeError("Fold patient overlap")
            fold_seed = seed + 1000 * fold
            _seed_everything(fold_seed)
            fold_dir = out / "checkpoints" / family / f"fold_{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            normalizer_info: dict[str, Any] = {"type": "none"}

            if family in {"linear_core", "gbt_core"}:
                stat = standardizer_fit(np.concatenate([pre_core[train_global], core[train_global]], axis=0))
                arrays["engineered_core_pre_z"] = standardizer_apply(pre_core, stat)
                arrays["engineered_core_post_z"] = standardizer_apply(core, stat)
                atomic_npz(fold_dir / "engineered_core_normalizer.npz", mean=stat["mean"], std=stat["std"])
                normalizer_info = {"type": "training_fold_zscore", "features": len(core_names)}

            if family == "linear_core":
                model = LinearDiscreteHazard(len(core_names))
                trained = _train_torch_pair(
                    model=model, family=family, train_frame=train, arrays=arrays,
                    global_rows=train_global, config=config, device=device, seed=fold_seed,
                    robust=False, temporal=False, out_path=fold_dir / "model.pt",
                )
                predictor = trained["model"]
                predict_fn = lambda profile: _predict_torch(
                    model=predictor, family=family, eval_frame=assess, arrays=arrays,
                    global_rows=assess_global, device=device, profile=profile, temporal=False,
                )
                train_records[family][str(fold)] = {"parameter_count": trained["parameter_count"], "history": trained["history"], "normalizer": normalizer_info}

            elif family == "gbt_core":
                clf = _gbt_fit_predictor(
                    x_pre=arrays["engineered_core_pre_z"][train_global],
                    x_post=arrays["engineered_core_post_z"][train_global],
                    frame=train, config=config, seed=fold_seed,
                )
                try:
                    import joblib
                    joblib.dump(clf, fold_dir / "model.joblib")
                except Exception as exc:
                    raise RuntimeError("joblib save failed for gbt_core") from exc
                def predict_fn(profile):
                    return (
                        _gbt_logits(clf, arrays["engineered_core_pre_z"][assess_global]),
                        _gbt_logits(clf, arrays["engineered_core_post_z"][assess_global]),
                    )
                train_records[family][str(fold)] = {"normalizer": normalizer_info, "classes": [int(x) for x in clf.classes_], "n_iter": int(clf.n_iter_)}

            elif family in {"paired_core", "paired_full", "paired_full_robust"}:
                include_optional = family != "paired_core"
                include_genomics = family != "paired_core"
                model = PairedStructuredSurvivalModel(include_optional=include_optional, include_genomics=include_genomics, dims=dims)
                trained = _train_torch_pair(
                    model=model, family=family, train_frame=train, arrays=arrays,
                    global_rows=train_global, config=config, device=device, seed=fold_seed,
                    robust=family == "paired_full_robust", temporal=False,
                    out_path=fold_dir / "model.pt",
                )
                predictor = trained["model"]
                predict_fn = lambda profile: _predict_torch(
                    model=predictor, family=family, eval_frame=assess, arrays=arrays,
                    global_rows=assess_global, device=device, profile=profile, temporal=False,
                )
                train_records[family][str(fold)] = {"parameter_count": trained["parameter_count"], "history": trained["history"], "normalizer": normalizer_info}

            elif family == "r1_temporal_fixed":
                model = TemporalFixedSurvivalModel()
                trained = _train_torch_pair(
                    model=model, family=family, train_frame=train, arrays=arrays,
                    global_rows=train_global, config=config, device=device, seed=fold_seed,
                    robust=False, temporal=True, out_path=fold_dir / "model.pt",
                )
                predictor = trained["model"]
                predict_fn = lambda profile: _predict_torch(
                    model=predictor, family=family, eval_frame=assess, arrays=arrays,
                    global_rows=assess_global, device=device, profile=profile, temporal=True,
                )
                train_records[family][str(fold)] = {"parameter_count": trained["parameter_count"], "history": trained["history"], "normalizer": normalizer_info}
            else:
                raise RuntimeError(f"Unknown family {family}")

            for profile in profile_sets[family]:
                pre_logits, post_logits = predict_fn(profile)
                if profile == "full_supported":
                    full_oof[family]["pre"][assess_local] = pre_logits
                    full_oof[family]["post"][assess_local] = post_logits
                for view, logits in [("PRE", pre_logits), ("POST", post_logits)]:
                    metric = v2_metric_bundle(assess, logits, km)
                    row = {
                        "family": family,
                        "profile": profile,
                        "view": view,
                        "fold": int(fold),
                        "rows": metric["rows"],
                        "patients": metric["patients"],
                        "patient_brier_4h_mean": metric["patient_brier_4h_mean"],
                        "fractional_censor_patient_mean_nll_v1": metric["fractional_censor_patient_mean_nll_v1"],
                        "fractional_censor_row_mean_nll_v1": metric["fractional_censor_row_mean_nll_v1"],
                        "curve_invariants": metric["curve_invariants"],
                        "patient_balanced_pfs": metric["patient_balanced_pfs"],
                    }
                    fold_metrics.append(row)
                    prediction_parts.append(_compact_prediction_rows(assess, logits, family=family, profile=profile, view=view, fold=fold))
                    print("[V2_02_EVAL]", family, fold, profile, view, metric["patient_brier_4h_mean"], metric["fractional_censor_patient_mean_nll_v1"], flush=True)

    # Exact full-supported OOF coverage for every family.
    for family, views in full_oof.items():
        for view, arr in views.items():
            if not np.isfinite(arr).all():
                raise RuntimeError(f"Incomplete OOF logits: {family} {view}")

    summary = _mean_fold_summary(fold_metrics)
    decision = _select_candidate(summary, config)
    selected = decision["best_screening_family"]

    predictions = pd.concat(prediction_parts, ignore_index=True)
    calibration = _calibration_table(predictions, km)
    atomic_parquet(out / "matched_development_predictions.parquet", predictions)
    atomic_parquet(out / "patient_balanced_calibration.parquet", calibration)
    atomic_json(out / "fold_metrics.json", fold_metrics)
    atomic_json(out / "cv_summary.json", summary)
    atomic_json(out / "training_history.json", train_records)
    atomic_json(out / "selection_decision.json", decision)

    selected_index = dev[["patient_id", "scan_episode_id", "landmark_day", "development_fold", "survival_time_days", "survival_cause"]].copy()
    atomic_parquet(out / "selected_oof_index.parquet", selected_index)
    atomic_npz(
        out / "selected_oof_logits.npz",
        pre_logits=full_oof[selected]["pre"],
        post_logits=full_oof[selected]["post"],
    )

    # Historical repaired reference is carried forward only as a non-CV-comparable
    # original-validation benchmark. No CHORD historical test score is consulted.
    legacy_val = json.loads((repo / "artifacts/dynamic_scan_v2/v2_00/legacy_replay_val.json").read_text())
    legacy_bounded = legacy_val["metrics"]["bounded_post"]
    historical_reference = {
        "source": "V2-00 immutable repaired legacy replay on original CHORD validation",
        "cv_comparable": False,
        "used_for_v2_02_selection": False,
        "bounded_patient_brier_4h": legacy_bounded["patient_brier_4h_mean"],
        "bounded_fractional_patient_nll_v1": legacy_bounded["fractional_censor_patient_mean_nll_v1"],
        "rows": legacy_bounded["rows"],
        "patients": legacy_bounded["patients"],
    }
    atomic_json(out / "historical_reference.json", historical_reference)

    # Acceptance is engineering/methodological. Scientific components may be rejected.
    curve_ok = all(row["curve_invariants"].get("status") == "PASS" for row in fold_metrics)
    families_seen = sorted({row["family"] for row in fold_metrics})
    acceptance = {
        "v2_01_pass_required": True,
        "exact_r2_row_alignment": True,
        "three_patient_disjoint_development_folds": True,
        "historical_test_not_used_for_training_or_selection": True,
        "permanent_line_derived_features_absent": True,
        "patient_balanced_landmark_training_objective": True,
        "fractional_censor_likelihood_used_for_new_training": True,
        "regularized_discrete_time_baseline_completed": "linear_core" in families_seen,
        "gradient_boosted_hazard_baseline_completed": "gbt_core" in families_seen,
        "core_only_neural_completed": "paired_core" in families_seen,
        "full_information_neural_completed": "paired_full" in families_seen,
        "missingness_robust_neural_completed": "paired_full_robust" in families_seen,
        "robust_scan_missingness_applied_before_scan_encoder": True,
        "robust_branch_does_not_use_unmasked_r1_temporal_latent": True,
        "all_full_supported_oof_predictions_complete": True,
        "curve_probability_invariants_pass": bool(curve_ok),
        "selection_uses_locked_primary_metric": protocol["selection"]["primary_metric"] == "PATIENT_BALANCED_PFS_BRIER_4H_MEAN",
        "protected_external_rows_not_opened": True,
        "external_predictions_not_regenerated": True,
    }
    if not all(acceptance.values()):
        raise RuntimeError(f"V2-02 acceptance failed: {acceptance}")

    result = {
        "checkpoint": "V2-02",
        "status": "PASS",
        "next_checkpoint": "V2-03",
        "device": str(device),
        "seed": seed,
        "development_rows": int(len(dev)),
        "development_patients": int(dev["patient_id"].nunique()),
        "fold_rows": {str(f): int((dev["development_fold"] == f).sum()) for f in (0, 1, 2)},
        "fold_patients": {str(f): int(dev.loc[dev["development_fold"] == f, "patient_id"].nunique()) for f in (0, 1, 2)},
        "families": config["model_families"],
        "profiles": list(PROFILE_NAMES),
        "acceptance": acceptance,
        "selection": decision,
        "historical_reference": historical_reference,
        "config_hashes": {
            "v2_02_training.json": sha256_file(repo / "configs/dynamic_scan_v2/v2_02_training.json"),
            "development_protocol.json": sha256_file(repo / "configs/dynamic_scan_v2/development_protocol.json"),
            "metrics_contract.json": sha256_file(repo / "configs/dynamic_scan_v2/metrics_contract.json"),
            "v2_01_representation.json": sha256_file(repo / "configs/dynamic_scan_v2/v2_01_representation.json"),
        },
        "input_hashes": {
            "v2_01_tensors": sha256_file(repo / "artifacts/dynamic_scan_v2/v2_01/paired_scan_tensors.npz"),
            "r2_temporal_cache": sha256_file(repo / "artifacts/checkpoint7r2_transport_safe_supervised/prepared/temporal_prepost_f16.npy"),
            "r2_tumor_embeddings": sha256_file(repo / "artifacts/checkpoint7r2_transport_safe_supervised/prepared/tumor_embeddings_f16.npy"),
        },
        "external_rows_opened": False,
        "external_predictions_regenerated": False,
    }
    atomic_json(out / "result_packet.json", result)

    report = f"""# OncoTwin V2-02 transport-native training\n\nStatus: **PASS**\n\n## Development-only evaluation\n\nAll model selection uses the frozen three-fold CHORD development protocol. Historical CHORD test outcomes and DFCI/VICC outcomes are not used for V2-02 selection. The immutable repaired legacy original-validation result is reported only as a non-CV-comparable reference.\n\n## Input integrity\n\nThe missingness-robust branch consumes source-transparent V2-01 structured tensors and applies coherent profile removal before `PairedScanEncoder`. It does not consume the unmasked R1 temporal latent. The `r1_temporal_fixed` family is retained separately as a strong frozen-feature comparator. Genomic content is separately maskable before its supervised encoder. Permanent line-derived coordinates are absent.\n\n## Screening decision\n\nSelected family for V2-03 branch/gate analysis: **{selected}**.\n\nRobust family retained by prespecified noninferiority gates: **{decision['robust_retained']}**.\n\nThe selection rule is locked primary patient-balanced four-horizon Brier with the prespecified 0.0005 simplicity tie margin; fractional-censor patient NLL is supporting.\n"""
    atomic_text(out / "decision_report.md", report)

    manifest_files = [
        out / "observation_profile_manifest.json",
        out / "fold_metrics.json",
        out / "cv_summary.json",
        out / "training_history.json",
        out / "selection_decision.json",
        out / "matched_development_predictions.parquet",
        out / "patient_balanced_calibration.parquet",
        out / "selected_oof_index.parquet",
        out / "selected_oof_logits.npz",
        out / "historical_reference.json",
        out / "result_packet.json",
        out / "decision_report.md",
    ]
    atomic_json(out / "manifest.json", {"status": "PASS", "files": _hash_artifacts(manifest_files)})
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
