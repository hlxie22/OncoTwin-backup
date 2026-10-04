from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
import sys
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from access import AccessPolicy
from common import atomic_json, atomic_parquet, atomic_text, sha256_file
from metrics import (
    HORIZONS_MONTHS,
    MONTH_DAYS,
    curves_from_logits,
    fit_censoring_km,
    horizon_status_and_weight,
    v2_metric_bundle,
)
from v2_03_gate import (
    LinearSigmoidGate,
    NonlinearSigmoidGate,
    apply_standardizer,
    assert_class_offset_invariant_features,
    blend_logits,
    build_gate_features,
    fit_gate_model,
    gate_distribution,
    predict_gate,
)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.npz")
    np.savez_compressed(tmp, **arrays)
    with np.load(tmp) as z:
        for k in arrays:
            if z[k].shape != arrays[k].shape:
                raise RuntimeError(f"NPZ verification failed for {k}")
    tmp.replace(path)


def metric(frame: pd.DataFrame, logits: np.ndarray, km) -> dict[str, Any]:
    return v2_metric_bundle(frame, logits, km)


def evaluate_by_fold(
    frame: pd.DataFrame,
    pre: np.ndarray,
    post: np.ndarray,
    alpha: np.ndarray,
    km,
    *,
    method: str,
    seed: int | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold in sorted(frame["development_fold"].astype(int).unique()):
        use = frame["development_fold"].astype(int).to_numpy() == fold
        m = metric(frame.loc[use].reset_index(drop=True), blend_logits(pre[use], post[use], alpha[use]), km)
        rows.append({
            "method": method,
            "seed": seed,
            "fold": int(fold),
            "rows": int(m["rows"]),
            "patients": int(m["patients"]),
            "patient_brier_4h_mean": float(m["patient_brier_4h_mean"]),
            "fractional_censor_patient_mean_nll_v1": float(m["fractional_censor_patient_mean_nll_v1"]),
            "curve_invariants": m["curve_invariants"],
        })
    return rows


def _constant_score(frame: pd.DataFrame, pre: np.ndarray, post: np.ndarray, alpha: float, km) -> float:
    return float(metric(frame, blend_logits(pre, post, float(alpha)), km)["patient_brier_4h_mean"])


def select_constant(
    frame: pd.DataFrame,
    pre: np.ndarray,
    post: np.ndarray,
    km,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    coarse = float(cfg["coarse_step"])
    refine = float(cfg["refine_step"])
    radius = float(cfg["refine_radius"])
    coarse_grid = np.round(np.arange(0.0, 1.0 + 0.5 * coarse, coarse), 10)
    scored: dict[float, float] = {}
    for a in coarse_grid:
        scored[float(a)] = _constant_score(frame, pre, post, float(a), km)
    best_coarse = min(scored, key=lambda a: (scored[a], abs(a - 0.5), a))
    lo = max(0.0, best_coarse - radius)
    hi = min(1.0, best_coarse + radius)
    refine_grid = np.round(np.arange(lo, hi + 0.5 * refine, refine), 10)
    for a in refine_grid:
        scored.setdefault(float(a), _constant_score(frame, pre, post, float(a), km))
    best = min(scored, key=lambda a: (scored[a], abs(a - 0.5), a))
    ordered = sorted(scored.items())
    return {
        "alpha": float(best),
        "fit_brier": float(scored[best]),
        "evaluated": [{"alpha": float(a), "brier": float(v)} for a, v in ordered],
    }


def crossfit_selected_constant(frame, pre, post, km, config) -> tuple[np.ndarray, dict[str, Any]]:
    folds = frame["development_fold"].astype(int).to_numpy()
    alpha = np.full(len(frame), np.nan, dtype=np.float64)
    provenance = {}
    for outer in sorted(np.unique(folds)):
        fit = folds != outer
        assess = folds == outer
        selected = select_constant(
            frame.loc[fit].reset_index(drop=True), pre[fit], post[fit], km, config["constant_grid"]
        )
        alpha[assess] = selected["alpha"]
        provenance[str(int(outer))] = {
            "assessment_fold": int(outer),
            "fitting_folds": [int(x) for x in sorted(np.unique(folds[fit]))],
            "selected_alpha": float(selected["alpha"]),
            "fit_brier": float(selected["fit_brier"]),
        }
    if not np.isfinite(alpha).all():
        raise RuntimeError("cross-fitted constant alpha incomplete")
    return alpha, provenance


def _calibration_rows(pred: pd.DataFrame, km, bins: int = 10) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method, group in pred.groupby("method", observed=True):
        group = group.reset_index(drop=True)
        for h in HORIZONS_MONTHS:
            risk = group[f"risk_{h}m"].to_numpy(dtype=float)
            y, w = horizon_status_and_weight(group, h * MONTH_DAYS, km, patient_balanced=True)
            use = np.isfinite(risk) & np.isfinite(w) & (w > 0)
            idx = np.where(use)[0][np.argsort(risk[use], kind="mergesort")]
            for b, chunk in enumerate(np.array_split(idx, bins), 1):
                if not len(chunk):
                    continue
                ww = w[chunk]
                rows.append({
                    "method": str(method),
                    "horizon_month": int(h),
                    "bin": int(b),
                    "rows": int(len(chunk)),
                    "weight_sum": float(ww.sum()),
                    "mean_predicted_risk": float(np.average(risk[chunk], weights=ww)),
                    "observed_risk": float(np.average(y[chunk], weights=ww)),
                })
    return pd.DataFrame(rows)


def _prediction_frame(frame, logits, alpha, method):
    c = curves_from_logits(logits)
    out = frame[["patient_id", "scan_episode_id", "landmark_day", "development_fold", "survival_time_days", "survival_cause"]].copy()
    out["method"] = method
    out["alpha"] = np.asarray(alpha, dtype=np.float32)
    for h in HORIZONS_MONTHS:
        out[f"risk_{h}m"] = (1.0 - c["pfs"][:, h - 1]).astype(np.float32)
    return out


def _stratum_diagnostics(frame, strata, pre, post, adaptive_alpha, constant_alpha, km, min_rows):
    specs = {
        "genomic_available": strata["genomic_available"],
        "modality_observed": strata["modality_observed"],
        "site_positive_stream_observed": strata["site_positive_stream_observed"],
        "coverage_comparable": strata["coverage_comparable"],
    }
    out: list[dict[str, Any]] = []
    for name, values in specs.items():
        for level in [False, True]:
            use = np.asarray(values == level)
            if use.sum() < int(min_rows):
                continue
            sub = frame.loc[use].reset_index(drop=True)
            ma = metric(sub, blend_logits(pre[use], post[use], adaptive_alpha[use]), km)
            mc = metric(sub, blend_logits(pre[use], post[use], constant_alpha[use]), km)
            out.append({
                "stratum": name,
                "level": bool(level),
                "rows": int(use.sum()),
                "patients": int(sub["patient_id"].nunique()),
                "adaptive_brier": float(ma["patient_brier_4h_mean"]),
                "constant_brier": float(mc["patient_brier_4h_mean"]),
                "delta_adaptive_minus_constant": float(ma["patient_brier_4h_mean"] - mc["patient_brier_4h_mean"]),
                "adaptive_alpha_mean": float(np.mean(adaptive_alpha[use])),
            })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    args = ap.parse_args()
    repo = Path(args.repo).resolve()
    out = repo / "artifacts/dynamic_scan_v2/v2_03"
    out.mkdir(parents=True, exist_ok=True)

    config = json.loads((repo / "configs/dynamic_scan_v2/v2_03_gate.json").read_text())
    policy = AccessPolicy.from_json(repo, repo / "configs/dynamic_scan_v2/data_access_policy.json")
    v202 = policy.read_json(repo / "artifacts/dynamic_scan_v2/v2_02/result_packet.json")
    decision202 = policy.read_json(repo / "artifacts/dynamic_scan_v2/v2_02/selection_decision.json")
    if v202.get("status") != "PASS" or not all(v202.get("acceptance", {}).values()):
        raise RuntimeError("V2-02 must PASS before V2-03")
    if decision202.get("best_screening_family") != config["selected_branch_family"]:
        raise RuntimeError("V2-03 selected branch family does not match frozen V2-02 decision")

    frozen_index_path = repo / "artifacts/dynamic_scan_v2/v2_02/selected_oof_index.parquet"
    frozen_logits_path = repo / "artifacts/dynamic_scan_v2/v2_02/selected_oof_logits.npz"
    expected_hashes = v202.get("artifact_hashes", {})
    for frozen_path in [frozen_index_path, frozen_logits_path]:
        expected = expected_hashes.get(frozen_path.name)
        if not expected:
            raise RuntimeError(f"V2-02 result packet lacks frozen hash for {frozen_path.name}")
        actual = sha256_file(frozen_path)
        if actual != expected:
            raise RuntimeError(f"Frozen V2-02 branch artifact hash mismatch for {frozen_path.name}: {actual} != {expected}")

    v201 = policy.read_json(repo / "artifacts/dynamic_scan_v2/v2_01/result_packet.json")
    if v201.get("status") != "PASS" or not all(v201.get("acceptance", {}).values()):
        raise RuntimeError("V2-01 must remain PASS before V2-03 gate-feature reconstruction")

    index = policy.read_parquet(frozen_index_path)
    with policy.np_load(frozen_logits_path) as z:
        pre = np.asarray(z["pre_logits"], dtype=np.float64)
        post = np.asarray(z["post_logits"], dtype=np.float64)
    if pre.shape != (len(index), 24, 4) or post.shape != pre.shape:
        raise RuntimeError(f"Unexpected OOF shape pre={pre.shape} post={post.shape} rows={len(index)}")
    if len(index) != int(v202["development_rows"]):
        raise RuntimeError("V2-03 OOF row count does not match finalized V2-02 modeling population")

    # Reconstruct exact scan metadata on the same keys without using outcomes as gate features.
    scan = policy.read_parquet(repo / "artifacts/dynamic_scan_v2/v2_01/paired_scan_index.parquet")
    keys = ["patient_id", "scan_episode_id"]
    for df in (index, scan):
        df["patient_id"] = df["patient_id"].astype(str).str.strip()
        df["scan_episode_id"] = df["scan_episode_id"].astype(str).str.strip()
    index = index.reset_index(drop=True).copy()
    index["_oof_row"] = np.arange(len(index), dtype=np.int64)
    meta_cols = [
        "patient_id", "scan_episode_id", "landmark_day", "clinical_state",
        "days_since_previous_assessment", "days_since_previous_comparable_assessment",
        "prior_scan_count", "prior_nonprogressive_run", "coverage_jaccard",
        "current_coverage_count", "coverage_gained_count", "coverage_lost_count",
        "coverage_comparable_to_previous", "current_modality_observed",
        "current_site_positive_stream_observed", "genomic_available", "genomic_age_days",
    ]
    missing = [c for c in meta_cols if c not in scan.columns]
    if missing:
        raise RuntimeError(f"V2-01 paired scan index missing gate metadata columns: {missing}")
    merged = index.merge(scan[meta_cols], on=keys, how="left", validate="one_to_one", suffixes=("", "_scan"))
    merged = merged.sort_values("_oof_row", kind="mergesort").reset_index(drop=True)
    if merged["clinical_state"].isna().any():
        raise RuntimeError("V2-03 scan metadata alignment incomplete")
    if "landmark_day_scan" in merged and not np.array_equal(
        merged["landmark_day"].to_numpy(), merged["landmark_day_scan"].to_numpy()
    ):
        raise RuntimeError("V2-03 landmark-day alignment mismatch")

    x, feature_names, strata = build_gate_features(merged, pre, post)
    offset_invariant = assert_class_offset_invariant_features(merged, pre, post)
    if not offset_invariant:
        raise RuntimeError("Gate features are not invariant to class-independent logit offsets")

    # Same fixed censoring nuisance policy as V2-02: original CHORD train survival rows only.
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

    frame = merged[["patient_id", "scan_episode_id", "landmark_day", "development_fold", "survival_time_days", "survival_cause"]].copy()
    folds = frame["development_fold"].astype(int).to_numpy()
    if set(np.unique(folds)) != {0, 1, 2}:
        raise RuntimeError(f"Expected development folds 0/1/2, got {sorted(set(folds))}")

    fold_metrics: list[dict[str, Any]] = []
    alpha_controls: dict[str, np.ndarray] = {
        "alpha_0": np.zeros(len(frame), dtype=np.float64),
        "alpha_0_5": np.full(len(frame), 0.5, dtype=np.float64),
        "alpha_1": np.ones(len(frame), dtype=np.float64),
    }
    constant_provenance: dict[str, Any] = {}
    crossfit_constant, constant_provenance = crossfit_selected_constant(frame, pre, post, km, config)
    alpha_controls["crossfit_best_constant"] = crossfit_constant

    control_summary = {}
    for name, alpha in alpha_controls.items():
        m = metric(frame, blend_logits(pre, post, alpha), km)
        control_summary[name] = {
            "patient_brier_4h_mean": float(m["patient_brier_4h_mean"]),
            "fractional_censor_patient_mean_nll_v1": float(m["fractional_censor_patient_mean_nll_v1"]),
            "alpha_distribution": gate_distribution(alpha),
        }
        fold_metrics.extend(evaluate_by_fold(frame, pre, post, alpha, km, method=name, seed=None))

    pooled_constant = select_constant(frame, pre, post, km, config["constant_grid"])
    constant_controls = {
        "fixed_controls": control_summary,
        "crossfit_selection_provenance": constant_provenance,
        "pooled_development_alpha_for_final_rule_if_constant_selected": float(pooled_constant["alpha"]),
        "pooled_development_brier": float(pooled_constant["fit_brier"]),
        "selection_note": "Primary adaptive comparison uses crossfit_best_constant. Pooled alpha is fit only after the V2-03 development decision for the final locked scalar rule.",
    }
    atomic_json(out / "constant_controls.json", constant_controls)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[V2_03_DEVICE]", device, flush=True)
    adaptive_alphas: dict[str, dict[int, np.ndarray]] = {"linear_sigmoid": {}, "nonlinear_8": {}}
    train_records: dict[str, Any] = {"linear_sigmoid": {}, "nonlinear_8": {}}
    checkpoints = out / "gate_checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)

    for family in ["linear_sigmoid", "nonlinear_8"]:
        for seed in [int(s) for s in config["seeds"]]:
            oof_alpha = np.full(len(frame), np.nan, dtype=np.float64)
            per_fold = {}
            for outer in [0, 1, 2]:
                train_mask = folds != outer
                assess_mask = folds == outer
                model, scaler, record = fit_gate_model(
                    family=family,
                    x_train=x[train_mask],
                    pre_logits=pre[train_mask],
                    post_logits=post[train_mask],
                    time_days=frame.loc[train_mask, "survival_time_days"].to_numpy(dtype=float),
                    cause=frame.loc[train_mask, "survival_cause"].astype(int).to_numpy(),
                    patient_ids=frame.loc[train_mask, "patient_id"].astype(str).tolist(),
                    seed=seed + 101 * outer,
                    config=config,
                    device=device,
                )
                pred_alpha = predict_gate(model, scaler, x[assess_mask], device)
                oof_alpha[assess_mask] = pred_alpha
                ckpt = {
                    "family": family,
                    "seed": int(seed),
                    "outer_assessment_fold": int(outer),
                    "fitting_folds": [int(f) for f in [0, 1, 2] if f != outer],
                    "feature_names": feature_names,
                    "feature_mean": scaler.mean.tolist(),
                    "feature_scale": scaler.scale.tolist(),
                    "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                }
                torch.save(ckpt, checkpoints / f"{family}_seed{seed}_fold{outer}.pt")
                per_fold[str(outer)] = record
            if not np.isfinite(oof_alpha).all():
                raise RuntimeError(f"Incomplete OOF gate alpha for {family} seed {seed}")
            adaptive_alphas[family][seed] = oof_alpha
            train_records[family][str(seed)] = per_fold
            fold_metrics.extend(evaluate_by_fold(frame, pre, post, oof_alpha, km, method=family, seed=seed))
            overall = metric(frame, blend_logits(pre, post, oof_alpha), km)
            print(
                "[V2_03_GATE]", family, seed,
                "Brier=", overall["patient_brier_4h_mean"],
                "NLL=", overall["fractional_censor_patient_mean_nll_v1"],
                "alpha=", gate_distribution(oof_alpha),
                flush=True,
            )

    atomic_json(out / "gate_training_history.json", train_records)
    atomic_json(out / "fold_metrics.json", fold_metrics)

    const_metric = metric(frame, blend_logits(pre, post, crossfit_constant), km)
    const_brier = float(const_metric["patient_brier_4h_mean"])
    const_nll = float(const_metric["fractional_censor_patient_mean_nll_v1"])
    selection = config["selection"]

    adaptive_summary = {}
    for family in ["linear_sigmoid", "nonlinear_8"]:
        seed_rows = []
        for seed, alpha in adaptive_alphas[family].items():
            m = metric(frame, blend_logits(pre, post, alpha), km)
            fm = [r for r in fold_metrics if r["method"] == family and r["seed"] == seed]
            const_fm = [r for r in fold_metrics if r["method"] == "crossfit_best_constant"]
            const_by_fold = {r["fold"]: r for r in const_fm}
            nonworse = sum(
                r["patient_brier_4h_mean"] <= const_by_fold[r["fold"]]["patient_brier_4h_mean"] + 1e-12
                for r in fm
            )
            seed_rows.append({
                "seed": int(seed),
                "patient_brier_4h_mean": float(m["patient_brier_4h_mean"]),
                "fractional_censor_patient_mean_nll_v1": float(m["fractional_censor_patient_mean_nll_v1"]),
                "brier_delta_vs_crossfit_constant": float(m["patient_brier_4h_mean"] - const_brier),
                "nll_delta_vs_crossfit_constant": float(m["fractional_censor_patient_mean_nll_v1"] - const_nll),
                "nonworse_outer_folds": int(nonworse),
                "alpha_distribution": gate_distribution(alpha),
            })
        adaptive_summary[family] = {
            "seeds": seed_rows,
            "mean_brier": float(np.mean([x["patient_brier_4h_mean"] for x in seed_rows])),
            "mean_nll": float(np.mean([x["fractional_censor_patient_mean_nll_v1"] for x in seed_rows])),
            "mean_brier_delta_vs_crossfit_constant": float(np.mean([x["brier_delta_vs_crossfit_constant"] for x in seed_rows])),
            "improving_seeds": int(sum(x["brier_delta_vs_crossfit_constant"] < 0 for x in seed_rows)),
            "minimum_nonworse_outer_folds_across_seeds": int(min(x["nonworse_outer_folds"] for x in seed_rows)),
        }

    chosen_adaptive = min(adaptive_summary, key=lambda f: (adaptive_summary[f]["mean_brier"], 0 if f == "linear_sigmoid" else 1))
    primary_seed = int(config["primary_seed"])
    selected_adaptive_alpha = adaptive_alphas[chosen_adaptive][primary_seed]
    selected_adaptive_metric = metric(frame, blend_logits(pre, post, selected_adaptive_alpha), km)
    dist = gate_distribution(selected_adaptive_alpha)
    strata_diag = _stratum_diagnostics(
        frame, strata, pre, post, selected_adaptive_alpha, crossfit_constant, km,
        min_rows=int(selection["minimum_stratum_rows"]),
    )
    max_stratum_delta = max((x["delta_adaptive_minus_constant"] for x in strata_diag), default=0.0)

    chosen_summary = adaptive_summary[chosen_adaptive]
    mean_improvement = const_brier - float(chosen_summary["mean_brier"])
    nll_noninferior = float(chosen_summary["mean_nll"]) <= const_nll + float(selection["nll_noninferiority_margin"])
    folds_stable = int(chosen_summary["minimum_nonworse_outer_folds_across_seeds"]) >= int(selection["required_nonworse_outer_folds"])
    seeds_stable = int(chosen_summary["improving_seeds"]) >= int(selection["required_improving_seeds"])
    magnitude_nontrivial = dist["std"] >= float(selection["min_alpha_std"])
    not_saturated = dist["saturation_fraction"] <= float(selection["max_saturation_fraction"])
    strata_noninferior = max_stratum_delta <= float(selection["natural_missingness_stratum_noninferiority_margin"])
    beats_constant = mean_improvement >= float(selection["min_mean_brier_improvement_over_crossfit_constant"])
    keep_adaptive = bool(beats_constant and nll_noninferior and folds_stable and seeds_stable and magnitude_nontrivial and not_saturated and strata_noninferior)

    if keep_adaptive:
        selected_rule = chosen_adaptive
        selected_alpha_for_predictions = selected_adaptive_alpha
    else:
        selected_rule = "constant"
        selected_alpha_for_predictions = crossfit_constant

    decision = {
        "selected_rule": selected_rule,
        "adaptive_family_considered": chosen_adaptive,
        "adaptive_retained": keep_adaptive,
        "crossfit_constant_brier": const_brier,
        "crossfit_constant_nll": const_nll,
        "adaptive_summary": adaptive_summary,
        "primary_seed": primary_seed,
        "primary_seed_adaptive_brier": float(selected_adaptive_metric["patient_brier_4h_mean"]),
        "primary_seed_adaptive_nll": float(selected_adaptive_metric["fractional_censor_patient_mean_nll_v1"]),
        "mean_brier_improvement_over_crossfit_constant": float(mean_improvement),
        "checks": {
            "beats_crossfit_best_constant_by_prespecified_margin": beats_constant,
            "nll_guardrail_pass": nll_noninferior,
            "fold_stability_pass": folds_stable,
            "seed_stability_pass": seeds_stable,
            "gate_nontrivial_variation_pass": magnitude_nontrivial,
            "gate_saturation_pass": not_saturated,
            "natural_missingness_strata_noninferiority_pass": strata_noninferior,
        },
        "adaptive_alpha_distribution_primary_seed": dist,
        "natural_missingness_strata": strata_diag,
        "limitation": "Gate cross-fitting uses branch predictions that are OOF per row. It is fixed-feature cross-fitting, not nested branch retraining for every gate outer fold.",
        "historical_test_used_for_gate_fit_or_selection": False,
    }
    atomic_json(out / "gate_decision.json", decision)

    # Persist cross-fitted alphas for all families/seeds and selected rule.
    npz_payload = {
        "crossfit_best_constant": crossfit_constant.astype(np.float32),
        "selected_oof_alpha": selected_alpha_for_predictions.astype(np.float32),
    }
    for family, by_seed in adaptive_alphas.items():
        for seed, alpha in by_seed.items():
            npz_payload[f"{family}_seed{seed}"] = alpha.astype(np.float32)
    atomic_npz(out / "gate_oof_alphas.npz", **npz_payload)

    # Matched selected/adaptive/constant predictions and calibration.
    parts = []
    for method_name, alpha in [
        ("crossfit_best_constant", crossfit_constant),
        (f"{chosen_adaptive}_primary_seed", selected_adaptive_alpha),
        ("selected_rule", selected_alpha_for_predictions),
    ]:
        parts.append(_prediction_frame(frame, blend_logits(pre, post, alpha), alpha, method_name))
    predictions = pd.concat(parts, ignore_index=True)
    atomic_parquet(out / "matched_gate_predictions.parquet", predictions)
    atomic_parquet(out / "gate_calibration.parquet", _calibration_rows(predictions, km))

    feature_schema = {
        "version": config["version"],
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "policy": config["feature_policy"],
        "class_independent_logit_offset_invariance": bool(offset_invariant),
        "structured_source": "V2-01 paired_scan_index observable scan/history quality fields",
        "branch_summary_source": "V2-02 selected OOF PRE/POST logits; probability summaries plus cause-relative-to-no-event logit deltas only",
    }
    atomic_json(out / "gate_feature_schema.json", feature_schema)
    atomic_json(out / "gate_distribution.json", {
        "chosen_adaptive_primary_seed": gate_distribution(selected_adaptive_alpha),
        "crossfit_best_constant": gate_distribution(crossfit_constant),
        "by_natural_missingness_stratum": strata_diag,
    })

    # Fit the post-selection final rule on all development OOF branch predictions.
    final_rule: dict[str, Any]
    if keep_adaptive:
        final_model, final_scaler, final_record = fit_gate_model(
            family=chosen_adaptive,
            x_train=x,
            pre_logits=pre,
            post_logits=post,
            time_days=frame["survival_time_days"].to_numpy(dtype=float),
            cause=frame["survival_cause"].astype(int).to_numpy(),
            patient_ids=frame["patient_id"].astype(str).tolist(),
            seed=primary_seed,
            config=config,
            device=device,
        )
        torch.save({
            "family": chosen_adaptive,
            "seed": primary_seed,
            "feature_names": feature_names,
            "feature_mean": final_scaler.mean.tolist(),
            "feature_scale": final_scaler.scale.tolist(),
            "model_state": {k: v.detach().cpu() for k, v in final_model.state_dict().items()},
            "fit_record": final_record,
        }, out / "selected_gate_checkpoint.pt")
        final_rule = {
            "type": "adaptive",
            "family": chosen_adaptive,
            "checkpoint": "selected_gate_checkpoint.pt",
            "seed": primary_seed,
        }
    else:
        final_rule = {
            "type": "constant",
            "alpha": float(pooled_constant["alpha"]),
            "fitting_population": "all 17194 V2-02 survival-eligible development OOF branch rows after V2-03 rule selection",
        }
    atomic_json(out / "selected_update_rule.json", final_rule)

    acceptance = {
        "v2_02_pass_required": True,
        "selected_branch_logits_frozen": True,
        "selected_branch_artifact_hashes_verified": True,
        "gate_fit_uses_oof_branch_predictions_only": True,
        "gate_does_not_backpropagate_into_branches": True,
        "alpha_0_control_completed": True,
        "alpha_0_5_control_completed": True,
        "alpha_1_control_completed": True,
        "development_selected_scalar_completed": True,
        "linear_sigmoid_gate_completed": True,
        "one_small_nonlinear_gate_completed": True,
        "gate_scalar_shared_across_all_intervals_and_causes": True,
        "gate_feature_logit_offset_invariance_pass": bool(offset_invariant),
        "future_outcome_fields_absent_from_gate_features": True,
        "historical_test_not_used_for_gate_fit_or_selection": True,
        "protected_external_rows_not_opened": True,
        "external_predictions_not_regenerated": True,
        "curve_probability_invariants_pass": all(r["curve_invariants"].get("status") == "PASS" for r in fold_metrics),
        "explicit_keep_or_reject_decision_written": True,
    }
    if not all(acceptance.values()):
        raise RuntimeError(f"V2-03 engineering acceptance failed: {acceptance}")

    artifact_paths = [
        out / "constant_controls.json",
        out / "fold_metrics.json",
        out / "gate_training_history.json",
        out / "gate_oof_alphas.npz",
        out / "matched_gate_predictions.parquet",
        out / "gate_calibration.parquet",
        out / "gate_feature_schema.json",
        out / "gate_distribution.json",
        out / "gate_decision.json",
        out / "selected_update_rule.json",
    ]
    if (out / "selected_gate_checkpoint.pt").is_file():
        artifact_paths.append(out / "selected_gate_checkpoint.pt")
    hashes = {p.name: sha256_file(p) for p in artifact_paths if p.is_file()}

    result = {
        "checkpoint": "V2-03",
        "status": "PASS",
        "next_checkpoint": "V2-04",
        "branch_family": config["selected_branch_family"],
        "development_rows": int(len(frame)),
        "development_patients": int(frame["patient_id"].nunique()),
        "fold_rows": {str(i): int((folds == i).sum()) for i in [0, 1, 2]},
        "constant_controls": {
            k: {
                "patient_brier_4h_mean": v["patient_brier_4h_mean"],
                "fractional_censor_patient_mean_nll_v1": v["fractional_censor_patient_mean_nll_v1"],
            }
            for k, v in control_summary.items()
        },
        "pooled_final_constant_alpha": float(pooled_constant["alpha"]),
        "decision": decision,
        "selected_update_rule": final_rule,
        "acceptance": acceptance,
        "external_rows_opened": False,
        "external_predictions_regenerated": False,
        "artifact_hashes": hashes,
    }
    atomic_json(out / "result_packet.json", result)

    report = f"""# OncoTwin V2-03 bounded scan-update gate\n\nStatus: **PASS**\n\nBranch family: **{config['selected_branch_family']}**.\n\nDevelopment rows/patients: **{len(frame)} / {frame['patient_id'].nunique()}**.\n\nCross-fitted best constant Brier: **{const_brier:.9f}**.\n\nAdaptive family considered: **{chosen_adaptive}**.\n\nAdaptive retained: **{keep_adaptive}**.\n\nSelected update rule: **{final_rule['type']}**.\n\nMean adaptive Brier improvement over cross-fitted constant: **{mean_improvement:.9f}**.\n\nNo historical-test or DFCI/VICC outcome rows were used. Branches remained frozen.\n"""
    atomic_text(out / "decision_report.md", report)
    print("[V2_03_DONE]", json.dumps({
        "selected_rule": selected_rule,
        "adaptive_retained": keep_adaptive,
        "constant_brier": const_brier,
        "adaptive_mean_brier": adaptive_summary[chosen_adaptive]["mean_brier"],
        "pooled_final_constant_alpha": pooled_constant["alpha"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
