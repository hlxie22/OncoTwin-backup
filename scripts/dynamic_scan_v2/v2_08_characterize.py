#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from access import AccessPolicy
from common import atomic_json, atomic_parquet, atomic_text, sha256_file
from metrics import MONTH_DAYS, curves_from_logits, fit_censoring_km, horizon_status_and_weight
from v2_03_gate import blend_logits
from v2_08_metrics import (
    bootstrap_percentile,
    calibration_ece,
    effective_sample_size,
    summarize_binary_metric,
    weighted_auc,
    weighted_brier,
    weighted_mean,
    antolini_style_concordance,
)

LOCKED_HORIZONS = (3, 6, 12, 18)
MONTHLY_HORIZONS = tuple(range(1, 25))


def require(cond: bool, message: str) -> None:
    if not cond:
        raise RuntimeError(message)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _integral_average(values: list[float], months: list[int]) -> float:
    x = np.asarray(months, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    use = np.isfinite(x) & np.isfinite(y)
    x, y = x[use], y[use]
    if len(x) == 0:
        return float("nan")
    if len(x) == 1:
        return float(y[0])
    area = np.trapz(y, x=x) if not hasattr(np, "trapezoid") else np.trapezoid(y, x=x)
    return float(area / (x[-1] - x[0]))


def _risk_curves(logits: np.ndarray) -> np.ndarray:
    curves = curves_from_logits(np.asarray(logits, dtype=np.float64))
    pfs = np.asarray(curves["pfs"], dtype=np.float64)
    require(pfs.shape[1] >= 24, f"Expected >=24 PFS months, got {pfs.shape}")
    risk = 1.0 - pfs[:, :24]
    require(np.isfinite(risk).all(), "Non-finite PFS risk")
    require(((risk >= -1e-8) & (risk <= 1.0 + 1e-8)).all(), "PFS risk outside [0,1]")
    return np.clip(risk, 0.0, 1.0)


def _status_cache(frame: pd.DataFrame, km, months: tuple[int, ...]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for h in months:
        y, w = horizon_status_and_weight(frame, h * MONTH_DAYS, km, patient_balanced=True)
        y = np.asarray(y, dtype=np.float64)
        w = np.asarray(w, dtype=np.float64)
        require(len(y) == len(frame) and len(w) == len(frame), f"Bad horizon arrays at {h}m")
        out[h] = (y, w)
    return out


def _crossfit_null(
    frame: pd.DataFrame,
    status: dict[int, tuple[np.ndarray, np.ndarray]],
    months: tuple[int, ...],
) -> np.ndarray:
    folds = frame["development_fold"].astype(int).to_numpy()
    out = np.zeros((len(frame), 24), dtype=np.float64)
    for h in months:
        y, w = status[h]
        pred = np.full(len(frame), np.nan, dtype=np.float64)
        for fold in sorted(np.unique(folds)):
            fit = folds != fold
            assess = folds == fold
            p0 = weighted_mean(y[fit], w[fit])
            require(np.isfinite(p0), f"Could not fit null risk at {h}m fold {fold}")
            pred[assess] = p0
        require(np.isfinite(pred).all(), f"Crossfit null incomplete at {h}m")
        out[:, h - 1] = pred
    return out


def _model_panel(
    name: str,
    risk: np.ndarray,
    status: dict[int, tuple[np.ndarray, np.ndarray]],
    null_risk: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    locked_rows: list[dict[str, Any]] = []
    cal_rows: list[dict[str, Any]] = []
    monthly_brier: list[float] = []
    monthly_auc: list[float] = []
    monthly_rows: list[dict[str, Any]] = []

    for h in MONTHLY_HORIZONS:
        y, w = status[h]
        p = risk[:, h - 1]
        b = weighted_brier(y, p, w)
        a = weighted_auc(y, p, w)
        nb = weighted_brier(y, null_risk[:, h - 1], w)
        skill = float(1.0 - b / nb) if np.isfinite(nb) and nb > 0 else float("nan")
        monthly_brier.append(b)
        monthly_auc.append(a)
        monthly_rows.append({
            "model": name,
            "horizon_month": h,
            "brier": b,
            "auc": a,
            "null_brier": nb,
            "brier_skill_score": skill,
            "effective_sample_size": effective_sample_size(w),
        })
        if h in LOCKED_HORIZONS:
            summary = summarize_binary_metric(y, p, w)
            summary.update({
                "model": name,
                "horizon_month": h,
                "null_brier": nb,
                "brier_skill_score": skill,
            })
            locked_rows.append(summary)
            _, bins = calibration_ece(y, p, w, bins=10)
            for row in bins:
                cal_rows.append({"model": name, "horizon_month": h, **row})

    by_h = {int(r["horizon_month"]): r for r in locked_rows}
    global_summary = {
        "model": name,
        "patient_brier_4h_mean": float(np.mean([by_h[h]["brier"] for h in LOCKED_HORIZONS])),
        "auc_4h_mean": float(np.mean([by_h[h]["auc"] for h in LOCKED_HORIZONS])),
        "brier_skill_4h_mean": float(np.mean([by_h[h]["brier_skill_score"] for h in LOCKED_HORIZONS])),
        "ibs_1_24_months": _integral_average(monthly_brier, list(MONTHLY_HORIZONS)),
        "mean_auc_1_24_months": _integral_average(monthly_auc, list(MONTHLY_HORIZONS)),
        "ibs_3_18_months": _integral_average(monthly_brier[2:18], list(range(3, 19))),
        "mean_auc_3_18_months": _integral_average(monthly_auc[2:18], list(range(3, 19))),
        "locked_horizons": {str(h): by_h[h] for h in LOCKED_HORIZONS},
    }
    return global_summary, monthly_rows, cal_rows


def _bootstrap_comparison(
    frame: pd.DataFrame,
    status: dict[int, tuple[np.ndarray, np.ndarray]],
    selected: np.ndarray,
    pre: np.ndarray,
    *,
    brier_reps: int,
    auc_reps: int,
    seed: int,
) -> dict[str, Any]:
    patients, patient_code = np.unique(frame["patient_id"].astype(str).to_numpy(), return_inverse=True)
    pcount = len(patients)
    rng = np.random.default_rng(int(seed))

    # Patient-level sufficient statistics make the Brier cluster bootstrap cheap.
    patient_stats: dict[int, dict[str, np.ndarray]] = {}
    point_delta_brier: dict[str, float] = {}
    point_delta_auc: dict[str, float] = {}
    for h in LOCKED_HORIZONS:
        y, w = status[h]
        ps, pp = selected[:, h - 1], pre[:, h - 1]
        num_s = np.bincount(patient_code, weights=w * (y - ps) ** 2, minlength=pcount)
        num_p = np.bincount(patient_code, weights=w * (y - pp) ** 2, minlength=pcount)
        den = np.bincount(patient_code, weights=w, minlength=pcount)
        patient_stats[h] = {"num_selected": num_s, "num_pre": num_p, "den": den}
        point_delta_brier[str(h)] = float(weighted_brier(y, ps, w) - weighted_brier(y, pp, w))
        point_delta_auc[str(h)] = float(weighted_auc(y, ps, w) - weighted_auc(y, pp, w))

    brier_deltas = {h: np.empty(brier_reps, dtype=np.float64) for h in LOCKED_HORIZONS}
    brier_mean_delta = np.empty(brier_reps, dtype=np.float64)
    for r in range(brier_reps):
        counts = rng.multinomial(pcount, np.full(pcount, 1.0 / pcount))
        ds = []
        for h in LOCKED_HORIZONS:
            st = patient_stats[h]
            den = float(np.dot(counts, st["den"]))
            bs = float(np.dot(counts, st["num_selected"]) / den)
            bp = float(np.dot(counts, st["num_pre"]) / den)
            d = bs - bp
            brier_deltas[h][r] = d
            ds.append(d)
        brier_mean_delta[r] = float(np.mean(ds))

    auc_deltas = {h: np.empty(auc_reps, dtype=np.float64) for h in LOCKED_HORIZONS}
    # AUC needs pairwise ranking; cluster bootstrap is recomputed with patient multiplicity.
    for r in range(auc_reps):
        counts = rng.multinomial(pcount, np.full(pcount, 1.0 / pcount))
        mult = counts[patient_code].astype(np.float64)
        for h in LOCKED_HORIZONS:
            y, w = status[h]
            ww = w * mult
            auc_deltas[h][r] = weighted_auc(y, selected[:, h - 1], ww) - weighted_auc(y, pre[:, h - 1], ww)

    out: dict[str, Any] = {
        "scope": "paired patient-cluster bootstrap conditional on frozen OOF branch predictions, frozen alpha=0.52, frozen folds, and frozen censoring nuisance policy",
        "patients": pcount,
        "brier_repetitions": int(brier_reps),
        "auc_repetitions": int(auc_reps),
        "seed": int(seed),
        "mean_4h_brier_delta_selected_minus_pre": {
            "point": float(np.mean(list(point_delta_brier.values()))),
            **bootstrap_percentile(brier_mean_delta),
            "probability_below_zero": float(np.mean(brier_mean_delta < 0)),
        },
        "per_horizon": {},
    }
    for h in LOCKED_HORIZONS:
        out["per_horizon"][str(h)] = {
            "brier_delta_selected_minus_pre": {
                "point": point_delta_brier[str(h)],
                **bootstrap_percentile(brier_deltas[h]),
                "probability_below_zero": float(np.mean(brier_deltas[h] < 0)),
            },
            "auc_delta_selected_minus_pre": {
                "point": point_delta_auc[str(h)],
                **bootstrap_percentile(auc_deltas[h]),
                "probability_above_zero": float(np.mean(auc_deltas[h] > 0)),
            },
        }
    return out


def _update_value(
    frame: pd.DataFrame,
    status: dict[int, tuple[np.ndarray, np.ndarray]],
    selected: np.ndarray,
    pre: np.ndarray,
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    patient_ids = frame["patient_id"].astype(str).to_numpy()
    unique_patients, patient_code = np.unique(patient_ids, return_inverse=True)
    p_num = np.zeros(len(unique_patients), dtype=np.float64)
    p_den = np.zeros(len(unique_patients), dtype=np.float64)

    for h in LOCKED_HORIZONS:
        y, w = status[h]
        delta = selected[:, h - 1] - pre[:, h - 1]
        use = np.isfinite(w) & (w > 0)
        case = use & (y == 1)
        control = use & (y == 0)
        sel_b = weighted_brier(y, selected[:, h - 1], w)
        pre_b = weighted_brier(y, pre[:, h - 1], w)
        sel_a = weighted_auc(y, selected[:, h - 1], w)
        pre_a = weighted_auc(y, pre[:, h - 1], w)
        rows[str(h)] = {
            "mean_signed_risk_update": weighted_mean(delta, w),
            "mean_absolute_risk_update": weighted_mean(np.abs(delta), w),
            "median_absolute_risk_update_unweighted": float(np.median(np.abs(delta[use]))),
            "p90_absolute_risk_update_unweighted": float(np.quantile(np.abs(delta[use]), 0.90)),
            "mean_update_cases": weighted_mean(delta[case], w[case]),
            "mean_update_controls": weighted_mean(delta[control], w[control]),
            "case_minus_control_mean_update": float(weighted_mean(delta[case], w[case]) - weighted_mean(delta[control], w[control])),
            "delta_risk_auc": weighted_auc(y, delta, w),
            "brier_improvement_pre_minus_selected": float(pre_b - sel_b),
            "auc_improvement_selected_minus_pre": float(sel_a - pre_a),
        }
        diff = w * ((y - selected[:, h - 1]) ** 2 - (y - pre[:, h - 1]) ** 2)
        p_num += np.bincount(patient_code, weights=diff, minlength=len(unique_patients))
        p_den += np.bincount(patient_code, weights=w, minlength=len(unique_patients))

    valid = p_den > 0
    patient_delta = p_num[valid] / p_den[valid]
    return {
        "interpretation": "descriptive OOF update-value analysis; delta-risk AUC asks whether the scan-induced risk change itself ranks future PFS events, while PRE-vs-selected metric deltas quantify total incremental value",
        "per_horizon": rows,
        "patient_level": {
            "patients_with_evaluable_weight": int(valid.sum()),
            "fraction_with_lower_weighted_brier_selected_vs_pre": float(np.mean(patient_delta < 0)),
            "median_patient_brier_delta_selected_minus_pre": float(np.median(patient_delta)),
            "q25_patient_brier_delta": float(np.quantile(patient_delta, 0.25)),
            "q75_patient_brier_delta": float(np.quantile(patient_delta, 0.75)),
        },
    }


def _risk_stratification(
    frame: pd.DataFrame,
    status: dict[int, tuple[np.ndarray, np.ndarray]],
    selected: np.ndarray,
    bins: int = 5,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for h in LOCKED_HORIZONS:
        y, w = status[h]
        p = selected[:, h - 1]
        use = np.isfinite(p) & np.isfinite(w) & (w > 0)
        idx = np.where(use)[0]
        idx = idx[np.argsort(p[idx], kind="mergesort")]
        chunks = np.array_split(idx, bins)
        for q, chunk in enumerate(chunks, 1):
            ww = w[chunk]
            rows.append({
                "horizon_month": int(h),
                "risk_group": int(q),
                "rows": int(len(chunk)),
                "patients": int(frame.iloc[chunk]["patient_id"].astype(str).nunique()),
                "mean_predicted_risk": weighted_mean(p[chunk], ww),
                "observed_risk": weighted_mean(y[chunk], ww),
                "effective_sample_size": effective_sample_size(ww),
            })
    return pd.DataFrame(rows)


def _internal_benchmark(
    pred: pd.DataFrame,
    status_lookup: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    use = pred.loc[
        pred["profile"].astype(str).eq("full_supported")
        & pred["view"].astype(str).eq("POST")
    ].copy()
    for family, group in use.groupby("family", observed=True):
        group = group.sort_values(["patient_id", "scan_episode_id"], kind="mergesort").reset_index(drop=True)
        briers, aucs = [], []
        horizon = {}
        for h in LOCKED_HORIZONS:
            key = (str(family), str(h))
            if key not in status_lookup:
                # Status is reconstructed directly from this family's rows because all
                # families are expected to contain the same OOF population.
                raise RuntimeError(f"Missing internal benchmark status lookup for {key}")
            y, w = status_lookup[key]
            p = group[f"risk_{h}m"].to_numpy(dtype=np.float64)
            b = weighted_brier(y, p, w)
            a = weighted_auc(y, p, w)
            briers.append(b); aucs.append(a)
            horizon[str(h)] = {"brier": b, "auc": a}
        rows.append({
            "family": str(family),
            "view": "POST",
            "profile": "full_supported",
            "patient_brier_4h_mean": float(np.mean(briers)),
            "auc_4h_mean": float(np.mean(aucs)),
            "horizons": horizon,
        })
    return sorted(rows, key=lambda x: x["patient_brier_4h_mean"])



def _concordance_panel(frame: pd.DataFrame, risks: dict[str, np.ndarray]) -> dict[str, Any]:
    folds = sorted(frame["development_fold"].astype(int).unique())
    out: dict[str, Any] = {}
    for name, risk in risks.items():
        fold_rows = []
        for fold in folds:
            use = frame["development_fold"].astype(int).to_numpy() == fold
            sub = frame.loc[use].reset_index(drop=True)
            x = antolini_style_concordance(
                sub["survival_time_days"].to_numpy(dtype=float),
                sub["survival_cause"].astype(int).to_numpy(),
                sub["patient_id"].astype(str).to_numpy(),
                risk[use],
                MONTH_DAYS,
            )
            fold_rows.append({"fold": int(fold), **x})
        vals = np.asarray([x["concordance"] for x in fold_rows], dtype=np.float64)
        out[name] = {
            "folds": fold_rows,
            "fold_mean": float(np.nanmean(vals)),
            "fold_sd": float(np.nanstd(vals, ddof=0)),
            "definition": "patient-balanced Antolini-style PFS concordance; progression/death events; switch/censor non-events; 24-month prediction support; same-patient landmark pairs excluded",
            "comparability_warning": "supplementary only; not numerically interchangeable with published line-level Antolini C because V2 has repeated scan landmarks and switch is a competing event in the primary estimand",
        }
    return out

def _publication_table(
    path: Path,
    model_summaries: dict[str, dict[str, Any]],
    internal: list[dict[str, Any]],
) -> None:
    fields = ["model", "source", "brier_4h_mean", "auc_4h_mean", "ibs_3_18", "ibs_1_24", "brier_skill_4h_mean"]
    rows = []
    for name in ["selected_alpha_0_52", "pre_alpha_0", "post_alpha_1", "alpha_0_5", "crossfit_null"]:
        if name not in model_summaries:
            continue
        x = model_summaries[name]
        rows.append({
            "model": name,
            "source": "V2-08 frozen OOF characterization",
            "brier_4h_mean": x["patient_brier_4h_mean"],
            "auc_4h_mean": x["auc_4h_mean"],
            "ibs_3_18": x["ibs_3_18_months"],
            "ibs_1_24": x["ibs_1_24_months"],
            "brier_skill_4h_mean": x["brier_skill_4h_mean"],
        })
    for x in internal:
        rows.append({
            "model": x["family"] + "_POST",
            "source": "V2-02 full-supported OOF benchmark",
            "brier_4h_mean": x["patient_brier_4h_mean"],
            "auc_4h_mean": x["auc_4h_mean"],
            "ibs_3_18": "",
            "ibs_1_24": "",
            "brier_skill_4h_mean": "",
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    repo = args.repo.resolve()
    out = repo / "artifacts/dynamic_scan_v2/v2_08"
    out.mkdir(parents=True, exist_ok=True)

    config = _json(repo / "configs/dynamic_scan_v2/v2_08_research_characterization.json")
    policy = AccessPolicy.from_json(repo, repo / "configs/dynamic_scan_v2/data_access_policy.json")

    v202 = policy.read_json(repo / "artifacts/dynamic_scan_v2/v2_02/result_packet.json")
    v203 = policy.read_json(repo / "artifacts/dynamic_scan_v2/v2_03/result_packet.json")
    v205 = policy.read_json(repo / "artifacts/dynamic_scan_v2/v2_05/result_packet.json")
    v207 = policy.read_json(repo / "artifacts/dynamic_scan_v2/v2_07/result_packet.json")
    for label, packet in [("V2-02", v202), ("V2-03", v203), ("V2-05", v205), ("V2-07", v207)]:
        require(packet.get("status") == "PASS", f"{label} must PASS before V2-08")
        require(all(packet.get("acceptance", {}).values()), f"{label} acceptance incomplete")

    require(v205.get("candidate_lock", {}).get("selected") == "v2_03_frozen_temporal_constant_alpha", "V2-05 final lock drift")
    require(v207.get("selected_rule") == "v2_03_frozen_temporal_constant_alpha", "V2-07 selected rule drift")

    rule_path = repo / "artifacts/dynamic_scan_v2/v2_03/selected_update_rule.json"
    expected_rule_hash = v203.get("artifact_hashes", {}).get("selected_update_rule.json")
    require(expected_rule_hash is not None and sha256_file(rule_path) == expected_rule_hash, "V2-03 selected rule hash mismatch")
    rule = policy.read_json(rule_path)
    alpha = float(rule["alpha"])
    require(abs(alpha - float(config["locked_alpha"])) < 1e-12, "Locked alpha mismatch")

    index_path = repo / "artifacts/dynamic_scan_v2/v2_02/selected_oof_index.parquet"
    logits_path = repo / "artifacts/dynamic_scan_v2/v2_02/selected_oof_logits.npz"
    for path in (index_path, logits_path):
        expected = v202.get("artifact_hashes", {}).get(path.name)
        require(expected is not None and sha256_file(path) == expected, f"Frozen OOF hash mismatch: {path.name}")

    frame = policy.read_parquet(index_path).reset_index(drop=True)
    with policy.np_load(logits_path) as z:
        pre_logits = np.asarray(z["pre_logits"], dtype=np.float64)
        post_logits = np.asarray(z["post_logits"], dtype=np.float64)
    require(pre_logits.shape == post_logits.shape == (len(frame), 24, 4), "Unexpected frozen OOF shape")
    require(len(frame) == 17194 and frame["patient_id"].astype(str).nunique() == 2443, "Frozen OOF population drift")

    # Frozen censoring nuisance policy inherited from V2-02/V2-03.
    r2 = policy.read_parquet(
        repo / "artifacts/checkpoint7r2_transport_safe_supervised/prepared/scan_index.parquet",
        columns=["patient_id", "split", "survival_mask", "survival_time_days", "survival_cause"],
    )
    km_frame = r2.loc[
        r2["split"].astype(str).str.lower().eq("train") & r2["survival_mask"].fillna(False).astype(bool)
    ].copy()
    km = fit_censoring_km(
        km_frame["survival_time_days"].to_numpy(dtype=float),
        km_frame["survival_cause"].astype(int).to_numpy(),
    )

    status = _status_cache(frame, km, MONTHLY_HORIZONS)
    pre = _risk_curves(pre_logits)
    post = _risk_curves(post_logits)
    selected = _risk_curves(blend_logits(pre_logits, post_logits, alpha))
    alpha05 = _risk_curves(blend_logits(pre_logits, post_logits, 0.5))
    null_risk = _crossfit_null(frame, status, MONTHLY_HORIZONS)

    models = {
        "pre_alpha_0": pre,
        "alpha_0_5": alpha05,
        "selected_alpha_0_52": selected,
        "post_alpha_1": post,
        "crossfit_null": null_risk,
    }
    summaries: dict[str, dict[str, Any]] = {}
    monthly_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    for name, risk in models.items():
        summary, monthly, cal = _model_panel(name, risk, status, null_risk)
        summaries[name] = summary
        monthly_rows.extend(monthly)
        calibration_rows.extend(cal)

    # Regression check against the already frozen V2-05 baseline score.
    frozen_baseline = float(v205["candidate_lock"]["baseline_metric"]["brier"])
    observed_baseline = float(summaries["selected_alpha_0_52"]["patient_brier_4h_mean"])
    require(abs(frozen_baseline - observed_baseline) <= float(config["metric_reproduction_tolerance"]),
            f"V2-08 selected Brier does not reproduce frozen V2-05 baseline: {observed_baseline} vs {frozen_baseline}")

    bootstrap = _bootstrap_comparison(
        frame, status, selected, pre,
        brier_reps=int(config["bootstrap"]["brier_repetitions"]),
        auc_reps=int(config["bootstrap"]["auc_repetitions"]),
        seed=int(config["bootstrap"]["seed"]),
    )
    update = _update_value(frame, status, selected, pre)
    strat = _risk_stratification(frame, status, selected, bins=int(config["risk_stratification_bins"]))
    concordance = _concordance_panel(frame, {
        "pre_alpha_0": pre,
        "selected_alpha_0_52": selected,
        "post_alpha_1": post,
    })

    # Internal family benchmark uses the exact V2-02 matched OOF rows. Reconstruct
    # weights independently per family to avoid assuming file ordering.
    matched = policy.read_parquet(repo / "artifacts/dynamic_scan_v2/v2_02/matched_development_predictions.parquet")
    full_post = matched.loc[
        matched["profile"].astype(str).eq("full_supported") & matched["view"].astype(str).eq("POST")
    ].copy()
    status_lookup: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for family, group in full_post.groupby("family", observed=True):
        group = group.sort_values(["patient_id", "scan_episode_id"], kind="mergesort").reset_index(drop=True)
        require(len(group) == len(frame), f"Internal benchmark family {family} has {len(group)} rows, expected {len(frame)}")
        for h in LOCKED_HORIZONS:
            y, w = horizon_status_and_weight(group, h * MONTH_DAYS, km, patient_balanced=True)
            status_lookup[(str(family), str(h))] = (np.asarray(y, dtype=float), np.asarray(w, dtype=float))
    internal = _internal_benchmark(full_post, status_lookup)

    # Publication-oriented outputs.
    atomic_json(out / "metric_panel.json", summaries)
    atomic_parquet(out / "monthly_discrimination_accuracy.parquet", pd.DataFrame(monthly_rows))
    atomic_parquet(out / "calibration_deciles.parquet", pd.DataFrame(calibration_rows))
    atomic_json(out / "paired_patient_bootstrap.json", bootstrap)
    atomic_json(out / "scan_update_value.json", update)
    atomic_parquet(out / "risk_stratification.parquet", strat)
    atomic_json(out / "supplementary_concordance.json", concordance)
    atomic_json(out / "internal_model_benchmark.json", internal)
    _publication_table(out / "publication_metric_table.csv", summaries, internal)
    literature_src = repo / "configs/dynamic_scan_v2/v2_08_literature_benchmark_targets.json"
    if literature_src.exists():
        atomic_text(out / "literature_benchmark_targets.json", literature_src.read_text())

    # Concise report that deliberately separates empirical evidence from global novelty claims.
    s = summaries["selected_alpha_0_52"]
    p = summaries["pre_alpha_0"]
    q = summaries["post_alpha_1"]
    null = summaries["crossfit_null"]
    boot4 = bootstrap["mean_4h_brier_delta_selected_minus_pre"]
    lines = [
        "# V2-08 Research Characterization",
        "",
        "## Scope",
        "Frozen post-selection characterization only. No model fitting, no hyperparameter selection, no historical-test outcomes, and no external outcomes are used.",
        "",
        "## Locked OOF result",
        f"- Selected alpha: {alpha:.2f}",
        f"- Selected 4-horizon patient-balanced Brier: {s['patient_brier_4h_mean']:.6f}",
        f"- PRE Brier: {p['patient_brier_4h_mean']:.6f}",
        f"- full POST Brier: {q['patient_brier_4h_mean']:.6f}",
        f"- cross-fitted null Brier: {null['patient_brier_4h_mean']:.6f}",
        f"- mean Brier skill score vs cross-fitted null: {s['brier_skill_4h_mean']:.4f}",
        f"- mean 3/6/12/18m AUC: {s['auc_4h_mean']:.4f}",
        f"- supplementary Antolini-style concordance (fold mean +/- SD): {concordance['selected_alpha_0_52']['fold_mean']:.4f} +/- {concordance['selected_alpha_0_52']['fold_sd']:.4f}",
        f"- IBS-like 3-18m patient-balanced IPCW integral: {s['ibs_3_18_months']:.6f}",
        f"- IBS-like 1-24m patient-balanced IPCW integral: {s['ibs_1_24_months']:.6f}",
        "",
        "## Incremental value of the current scan",
        f"- Selected minus PRE mean Brier delta: {boot4['point']:+.6f}",
        f"- patient-bootstrap 95% interval: [{boot4['ci_low']:+.6f}, {boot4['ci_high']:+.6f}]",
        f"- bootstrap P(delta < 0): {boot4['probability_below_zero']:.4f}",
        f"- fraction of evaluable patients with lower weighted Brier under selected update: {update['patient_level']['fraction_with_lower_weighted_brier_selected_vs_pre']:.4f}",
        "",
        "## Novelty interpretation boundary",
        "These results can support claims about the internal predictive value and behavior of scan-by-scan updating. They do not by themselves establish that the method is the first of its kind or state of the art across publications; those are literature-comparison claims and must be made only after matching endpoint, landmark, censoring, and metric definitions.",
        "",
        "## Recommended publication metrics",
        "Use horizon-specific Brier/AUC/calibration, Brier skill versus the cross-fitted null, the paired patient-bootstrap PRE-vs-selected deltas, and the internal V2-02 family benchmark. Treat the 1-24m integrated score as a literature-facing supplementary metric because its patient balancing and competing-risk semantics are not automatically identical to published IBS values.",
    ]
    atomic_text(out / "research_characterization.md", "\n".join(lines) + "\n")

    acceptance = {
        "v2_02_pass_required": True,
        "v2_03_pass_required": True,
        "v2_05_final_lock_required": True,
        "v2_07_release_required": True,
        "frozen_oof_artifact_hashes_verified": True,
        "selected_rule_hash_verified": True,
        "exact_17194_oof_rows": True,
        "exact_2443_oof_patients": True,
        "selected_metric_reproduces_v2_05": True,
        "per_horizon_brier_completed": True,
        "per_horizon_auc_completed": True,
        "brier_skill_score_completed": True,
        "calibration_intercept_slope_completed": True,
        "calibration_deciles_completed": True,
        "monthly_integrated_metrics_completed": True,
        "paired_patient_bootstrap_completed": True,
        "scan_update_value_completed": True,
        "internal_family_benchmark_completed": True,
        "risk_stratification_completed": True,
        "supplementary_concordance_completed": True,
        "historical_test_not_used": True,
        "protected_external_rows_not_opened": True,
        "external_predictions_not_regenerated": True,
        "no_model_retraining_or_selection": True,
    }
    result = {
        "checkpoint": "V2-08",
        "status": "PASS" if all(acceptance.values()) else "FAIL",
        "purpose": "post-selection research characterization and literature-facing metric panel",
        "development_rows": int(len(frame)),
        "development_patients": int(frame["patient_id"].astype(str).nunique()),
        "locked_alpha": alpha,
        "selected_summary": s,
        "pre_summary": p,
        "post_summary": q,
        "crossfit_null_summary": null,
        "bootstrap_key_result": boot4,
        "external_confirmation": v207.get("external_confirmation"),
        "historical_test_rows_opened": False,
        "external_rows_opened": False,
        "external_predictions_regenerated": False,
        "model_retraining_performed": False,
        "model_selection_performed": False,
        "acceptance": acceptance,
    }
    files = [
        "metric_panel.json", "monthly_discrimination_accuracy.parquet", "calibration_deciles.parquet",
        "paired_patient_bootstrap.json", "scan_update_value.json", "risk_stratification.parquet",
        "internal_model_benchmark.json", "supplementary_concordance.json", "publication_metric_table.csv", "research_characterization.md",
    ]
    if (out / "literature_benchmark_targets.json").exists():
        files.append("literature_benchmark_targets.json")
    result["artifact_hashes"] = {f: sha256_file(out / f) for f in files}
    atomic_json(out / "result_packet.json", result)
    print("[V2_08_PASS]", json.dumps({
        "brier": s["patient_brier_4h_mean"],
        "auc_mean": s["auc_4h_mean"],
        "brier_skill": s["brier_skill_4h_mean"],
        "delta_brier_vs_pre": boot4["point"],
        "delta_brier_ci": [boot4["ci_low"], boot4["ci_high"]],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
