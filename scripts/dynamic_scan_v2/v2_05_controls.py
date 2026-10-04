from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from metrics import HORIZONS_MONTHS, MONTH_DAYS, curves_from_logits, horizon_status_and_weight


CURRENT_STATE = slice(0, 3)
CURRENT_COVERAGE = slice(3, 8)
CHANGE_TRANSITION = slice(0, 9)
CHANGE_COVERAGE_TIMING = slice(9, 19)
CHANGE_COMPARABLE_STATE = slice(19, 22)


@dataclass(frozen=True)
class CandidateLockDecision:
    selected: str
    candidate_passes: bool
    checks: dict[str, bool]


def deterministic_stratified_permutation(strata: list[tuple[Any, ...]], seed: int) -> np.ndarray:
    """Return a deterministic within-stratum permutation with no outcome inputs."""
    n = len(strata)
    out = np.arange(n, dtype=np.int64)
    groups: dict[tuple[Any, ...], list[int]] = {}
    for i, key in enumerate(strata):
        groups.setdefault(tuple(key), []).append(i)
    rng = np.random.default_rng(int(seed))
    for key in sorted(groups, key=lambda x: repr(x)):
        idx = np.asarray(groups[key], dtype=np.int64)
        if len(idx) < 2:
            continue
        perm = idx[rng.permutation(len(idx))]
        # Avoid an all-identity permutation when possible.
        if np.array_equal(perm, idx):
            perm = np.roll(perm, 1)
        out[idx] = perm
    return out


def mask_explicit_scan_arrays(arrays: dict[str, np.ndarray], mode: str) -> dict[str, np.ndarray]:
    """Copy only current/change/optional/stream blocks and apply a declared control.

    This is intended for matched pre-temporal + explicit-scan controls. It never
    alters temporal PRE states, outcomes, or patient/fold assignments.
    """
    if mode not in {"full", "clinical_content_only", "coverage_timing_only"}:
        raise ValueError(mode)
    out = dict(arrays)
    names = [
        "current_core", "current_core_mask", "change_core", "change_core_mask",
        "optional_current", "optional_current_mask", "stream_availability",
        "stream_availability_mask",
    ]
    for name in names:
        out[name] = np.array(arrays[name], copy=True)
    if mode == "full":
        return out
    if mode == "clinical_content_only":
        # Preserve state and state-transition content; remove anatomy/timing and
        # descriptor/utilization cues from the explicit current-scan residual.
        out["current_core"][:, CURRENT_COVERAGE] = 0.0
        out["current_core_mask"][:, CURRENT_COVERAGE] = 0.0
        out["change_core"][:, CHANGE_COVERAGE_TIMING] = 0.0
        out["change_core_mask"][:, CHANGE_COVERAGE_TIMING] = 0.0
        out["optional_current"][:] = 0.0
        out["optional_current_mask"][:] = 0.0
        # Keep state/coverage source-availability indicators only; remove
        # modality/site observation descriptors.
        out["stream_availability"][:, 2:] = 0.0
        out["stream_availability_mask"][:, 2:] = 0.0
        return out
    # coverage_timing_only: remove state/transition/comparable-state and optional
    # descriptor content, preserving anatomy overlap and timing.
    out["current_core"][:, CURRENT_STATE] = 0.0
    out["current_core_mask"][:, CURRENT_STATE] = 0.0
    out["change_core"][:, CHANGE_TRANSITION] = 0.0
    out["change_core_mask"][:, CHANGE_TRANSITION] = 0.0
    out["change_core"][:, CHANGE_COMPARABLE_STATE] = 0.0
    out["change_core_mask"][:, CHANGE_COMPARABLE_STATE] = 0.0
    out["optional_current"][:] = 0.0
    out["optional_current_mask"][:] = 0.0
    # Keep only source availability, not positive/descriptor observations.
    out["stream_availability"][:, 3:] = 0.0
    out["stream_availability_mask"][:, 3:] = 0.0
    return out


def seed_robustness(seed_scores: list[float], baseline: float, cfg: dict[str, Any]) -> dict[str, Any]:
    values = np.asarray(seed_scores, dtype=float)
    deltas = values - float(baseline)
    mean_gain = float(baseline - values.mean())
    improving = int((values < baseline).sum())
    worst_regression = float(max(0.0, deltas.max()))
    checks = {
        "seed_mean_gain": mean_gain >= float(cfg["minimum_seed_mean_brier_gain"]),
        "improving_seed_count": improving >= int(cfg["minimum_improving_seeds"]),
        "worst_seed_regression": worst_regression <= float(cfg["maximum_single_seed_regression"]),
    }
    return {
        "scores": [float(x) for x in values],
        "baseline": float(baseline),
        "mean_score": float(values.mean()),
        "mean_gain": mean_gain,
        "improving_seeds": improving,
        "worst_seed_regression": worst_regression,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def patient_bootstrap_brier_delta(
    frame: pd.DataFrame,
    candidate_logits: np.ndarray,
    baseline_logits: np.ndarray,
    km,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    """Paired patient-cluster bootstrap with fixed models and fixed censoring KM."""
    cand = curves_from_logits(candidate_logits)
    base = curves_from_logits(baseline_logits)
    pids = frame["patient_id"].astype(str).to_numpy()
    patients, patient_inv = np.unique(pids, return_inverse=True)
    h_num_c = np.zeros((len(patients), len(HORIZONS_MONTHS)), dtype=np.float64)
    h_num_b = np.zeros_like(h_num_c)
    h_den = np.zeros_like(h_num_c)
    for j, h in enumerate(HORIZONS_MONTHS):
        y, w = horizon_status_and_weight(frame, h * MONTH_DAYS, km, patient_balanced=True)
        use = np.isfinite(w) & (w > 0)
        rc = 1.0 - cand["pfs"][:, h - 1]
        rb = 1.0 - base["pfs"][:, h - 1]
        np.add.at(h_num_c[:, j], patient_inv[use], w[use] * (rc[use] - y[use]) ** 2)
        np.add.at(h_num_b[:, j], patient_inv[use], w[use] * (rb[use] - y[use]) ** 2)
        np.add.at(h_den[:, j], patient_inv[use], w[use])
    def score(counts: np.ndarray) -> tuple[float, float]:
        den = counts @ h_den
        c = (counts @ h_num_c) / np.maximum(den, 1e-12)
        b = (counts @ h_num_b) / np.maximum(den, 1e-12)
        return float(c.mean()), float(b.mean())
    counts = np.ones(len(patients), dtype=np.float64)
    point_c, point_b = score(counts)
    rng = np.random.default_rng(int(seed))
    deltas = np.empty(int(repetitions), dtype=np.float64)
    for i in range(int(repetitions)):
        draw = rng.integers(0, len(patients), size=len(patients))
        cts = np.bincount(draw, minlength=len(patients)).astype(np.float64)
        sc, sb = score(cts)
        deltas[i] = sc - sb
    return {
        "scope": "paired patient resampling conditional on fixed selected models, fixed folds, fixed censoring KM, and completed model-selection process",
        "patients": int(len(patients)),
        "repetitions": int(repetitions),
        "seed": int(seed),
        "candidate_brier": point_c,
        "baseline_brier": point_b,
        "delta_candidate_minus_baseline": float(point_c - point_b),
        "ci_low": float(np.quantile(deltas, 0.025)),
        "ci_high": float(np.quantile(deltas, 0.975)),
        "probability_delta_below_zero": float(np.mean(deltas < 0.0)),
    }


def calibration_ece(calibration: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    if calibration.empty:
        return out
    group_cols = [c for c in ["family", "profile", "view", "horizon_month"] if c in calibration.columns]
    for key, g in calibration.groupby(group_cols, observed=True):
        ww = g["weight_sum"].to_numpy(dtype=float)
        pred = g["mean_predicted_risk"].to_numpy(dtype=float)
        obs = g["observed_risk"].to_numpy(dtype=float)
        ece = float(np.sum(ww * np.abs(pred - obs)) / max(np.sum(ww), 1e-12))
        if not isinstance(key, tuple): key = (key,)
        out["|".join(map(str, key))] = ece
    return out


def window_spread(scores: dict[str, float], limit: float) -> dict[str, Any]:
    vals = np.asarray(list(scores.values()), dtype=float)
    spread = float(vals.max() - vals.min()) if len(vals) else float("nan")
    return {"scores": {k: float(v) for k, v in scores.items()}, "spread": spread, "limit": float(limit), "pass": bool(np.isfinite(spread) and spread <= float(limit))}


def decide_candidate_lock(
    *,
    seed_report: dict[str, Any],
    nll_delta: float,
    nll_guardrail: float,
    reassignment_gain: float,
    minimum_reassignment_gain: float,
    content_margin: float,
    minimum_content_margin: float,
    calibration_regression: float,
    maximum_calibration_regression: float,
    window_pass: bool,
) -> CandidateLockDecision:
    checks = {
        "seed_robustness": seed_report.get("status") == "PASS",
        "nll_guardrail": float(nll_delta) <= float(nll_guardrail),
        "correct_scan_specificity": float(reassignment_gain) >= float(minimum_reassignment_gain),
        "clinical_content_specificity": float(content_margin) >= float(minimum_content_margin),
        "calibration_guardrail": float(calibration_regression) <= float(maximum_calibration_regression),
        "window_robustness": bool(window_pass),
    }
    passes = all(checks.values())
    return CandidateLockDecision(
        selected="v2_04_explicit_scan_no_genomics" if passes else "v2_03_frozen_temporal_constant_alpha",
        candidate_passes=passes,
        checks=checks,
    )
