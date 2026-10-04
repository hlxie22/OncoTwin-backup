from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

MONTHS = 24
ADMIN_HORIZON_DAYS = 730.0
MONTH_DAYS = ADMIN_HORIZON_DAYS / MONTHS
HORIZONS_MONTHS = (3, 6, 12, 18)

CAUSE_NO_EVENT_OR_CENSOR = 0
CAUSE_PROGRESSION = 1
CAUSE_DEATH = 2
CAUSE_SWITCH = 3
CAUSE_NAMES = {
    0: "CENSOR_OR_NO_EVENT",
    1: "PROGRESSION",
    2: "DEATH",
    3: "SWITCH",
}
EPS = 1e-12


@dataclass(frozen=True)
class CensoringKM:
    times: np.ndarray
    survival: np.ndarray


def _validate_inputs(logits: np.ndarray, time_days: np.ndarray, cause: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logits = np.asarray(logits, dtype=np.float64)
    time_days = np.asarray(time_days, dtype=np.float64)
    cause = np.asarray(cause, dtype=np.int64)
    if logits.ndim != 3 or logits.shape[1:] != (MONTHS, 4):
        raise ValueError(f"Expected logits [N,{MONTHS},4], got {logits.shape}")
    if len(time_days) != len(logits) or len(cause) != len(logits):
        raise ValueError("logit/time/cause length mismatch")
    if not np.isfinite(logits).all():
        raise ValueError("non-finite logits")
    if not np.isfinite(time_days).all():
        raise ValueError("non-finite time")
    if (time_days <= 0).any():
        raise ValueError("zero/negative survival times are excluded by the frozen supervised rule")
    if (time_days > ADMIN_HORIZON_DAYS + 1e-8).any():
        raise ValueError("survival time exceeds 730-day administrative horizon")
    if not np.isin(cause, [0, 1, 2, 3]).all():
        raise ValueError("invalid cause code")
    return logits, time_days, cause


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


def log_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    maximum = np.max(logits, axis=axis, keepdims=True)
    shifted = logits - maximum
    return shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True))


def event_bin(time_days: float) -> int:
    """One-based discrete event bin using the frozen ceil rule."""
    month = int(np.ceil(float(time_days) / MONTH_DAYS - 1e-12))
    return int(np.clip(month, 1, MONTHS))


def censor_exposure(time_days: float) -> tuple[int, float]:
    """Completed full bins and fraction of the next bin observed before censoring."""
    t = float(np.clip(time_days, 0.0, ADMIN_HORIZON_DAYS))
    ratio = t / MONTH_DAYS
    completed = int(np.floor(ratio + 1e-12))
    completed = min(completed, MONTHS)
    if completed >= MONTHS:
        return MONTHS, 0.0
    remainder = t - completed * MONTH_DAYS
    fraction = remainder / MONTH_DAYS
    if fraction < 1e-10:
        fraction = 0.0
    if fraction > 1.0 - 1e-10:
        completed += 1
        fraction = 0.0
    return int(min(completed, MONTHS)), float(np.clip(fraction, 0.0, 1.0))


def fractional_censor_nll_per_row(logits: np.ndarray, time_days: np.ndarray, cause: np.ndarray) -> np.ndarray:
    """
    V2 supporting likelihood v1.

    Events retain the frozen interval-event likelihood. For right censoring inside a
    monthly interval, survival through the observed fraction is q0**fraction, where
    q0 is the model's full-interval no-event probability. This removes the legacy
    requirement to survive unobserved time after an interior-bin censoring time.
    """
    logits, time_days, cause = _validate_inputs(logits, time_days, cause)
    lp = log_softmax(logits, axis=-1)
    out = np.empty(len(logits), dtype=np.float64)
    for i in range(len(logits)):
        c = int(cause[i])
        if c != CAUSE_NO_EVENT_OR_CENSOR:
            month = event_bin(time_days[i])
            out[i] = -(
                lp[i, : month - 1, CAUSE_NO_EVENT_OR_CENSOR].sum()
                + lp[i, month - 1, c]
            )
            continue
        completed, fraction = censor_exposure(time_days[i])
        value = lp[i, :completed, CAUSE_NO_EVENT_OR_CENSOR].sum()
        if completed < MONTHS and fraction > 0:
            value += fraction * lp[i, completed, CAUSE_NO_EVENT_OR_CENSOR]
        out[i] = -value
    return out


def curves_from_logits(logits: np.ndarray) -> dict[str, np.ndarray]:
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 3 or logits.shape[1:] != (MONTHS, 4):
        raise ValueError(f"Expected [N,{MONTHS},4], got {logits.shape}")
    conditional = softmax(logits, axis=-1)
    n = len(logits)
    event_free = np.ones(n, dtype=np.float64)
    cif_p = np.zeros(n, dtype=np.float64)
    cif_d = np.zeros(n, dtype=np.float64)
    cif_s = np.zeros(n, dtype=np.float64)
    pfs = np.zeros((n, MONTHS), dtype=np.float64)
    no_event = np.zeros_like(pfs)
    progression = np.zeros_like(pfs)
    death = np.zeros_like(pfs)
    switch = np.zeros_like(pfs)
    total_mass = np.zeros_like(pfs)
    for m in range(MONTHS):
        current = conditional[:, m]
        cif_p += event_free * current[:, CAUSE_PROGRESSION]
        cif_d += event_free * current[:, CAUSE_DEATH]
        cif_s += event_free * current[:, CAUSE_SWITCH]
        event_free *= current[:, CAUSE_NO_EVENT_OR_CENSOR]
        progression[:, m] = cif_p
        death[:, m] = cif_d
        switch[:, m] = cif_s
        no_event[:, m] = event_free
        pfs[:, m] = 1.0 - cif_p - cif_d
        total_mass[:, m] = event_free + cif_p + cif_d + cif_s
    entropy = -(conditional * np.log(np.clip(conditional, EPS, 1.0))).sum(axis=-1).mean(axis=1)
    return {
        "conditional": conditional,
        "pfs": pfs,
        "progression_cif": progression,
        "death_cif": death,
        "switch_cif": switch,
        "event_free": no_event,
        "total_mass": total_mass,
        "entropy": entropy,
    }


def validate_curve_invariants(curves: dict[str, np.ndarray], atol: float = 1e-10) -> dict[str, Any]:
    conditional = curves["conditional"]
    mass = curves["total_mass"]
    progression = curves["progression_cif"]
    death = curves["death_cif"]
    switch = curves["switch_cif"]
    event_free = curves["event_free"]
    checks = {
        "conditional_mass": bool(np.allclose(conditional.sum(axis=-1), 1.0, atol=atol)),
        "cumulative_mass": bool(np.allclose(mass, 1.0, atol=atol)),
        "progression_nondecreasing": bool((np.diff(progression, axis=1) >= -atol).all()),
        "death_nondecreasing": bool((np.diff(death, axis=1) >= -atol).all()),
        "switch_nondecreasing": bool((np.diff(switch, axis=1) >= -atol).all()),
        "event_free_nonincreasing": bool((np.diff(event_free, axis=1) <= atol).all()),
        "finite": bool(all(np.isfinite(v).all() for v in curves.values())),
    }
    checks["status"] = "PASS" if all(checks.values()) else "FAIL"
    return checks


def fit_censoring_km(time_days: np.ndarray, cause: np.ndarray) -> CensoringKM:
    time_days = np.asarray(time_days, dtype=np.float64)
    cause = np.asarray(cause, dtype=np.int64)
    if len(time_days) != len(cause):
        raise ValueError("time/cause length mismatch")
    order = np.argsort(time_days, kind="mergesort")
    t = time_days[order]
    c = cause[order]
    unique = np.unique(t)
    at_risk = len(t)
    survival = 1.0
    out_t: list[float] = []
    out_s: list[float] = []
    for current in unique:
        mask = t == current
        n_at_time = int(mask.sum())
        censorings = int((c[mask] == CAUSE_NO_EVENT_OR_CENSOR).sum())
        if at_risk > 0 and censorings > 0:
            survival *= 1.0 - censorings / at_risk
        out_t.append(float(current))
        out_s.append(float(max(survival, EPS)))
        at_risk -= n_at_time
    return CensoringKM(np.asarray(out_t), np.asarray(out_s))


def censoring_survival_at(km: CensoringKM, time_days: float, *, left_limit: bool) -> float:
    side = "left" if left_limit else "right"
    index = np.searchsorted(km.times, float(time_days), side=side) - 1
    if index < 0:
        return 1.0
    return float(max(km.survival[index], EPS))


def patient_balance_factor(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("patient_id", observed=True)["patient_id"].transform("count").to_numpy(dtype=float)
    return 1.0 / np.maximum(counts, 1.0)


def horizon_status_and_weight(
    frame: pd.DataFrame,
    horizon_days: float,
    km: CensoringKM,
    *,
    patient_balanced: bool,
) -> tuple[np.ndarray, np.ndarray]:
    time_days = frame["survival_time_days"].to_numpy(dtype=float)
    cause = frame["survival_cause"].astype(int).to_numpy()
    y = np.zeros(len(frame), dtype=np.float64)
    weight = np.zeros(len(frame), dtype=np.float64)
    for i, (t, c) in enumerate(zip(time_days, cause)):
        if t > horizon_days:
            y[i] = 0.0
            weight[i] = 1.0 / censoring_survival_at(km, horizon_days, left_limit=False)
        elif c == CAUSE_NO_EVENT_OR_CENSOR:
            continue
        else:
            y[i] = float(c in {CAUSE_PROGRESSION, CAUSE_DEATH})
            weight[i] = 1.0 / censoring_survival_at(km, t, left_limit=True)
    if patient_balanced:
        weight *= patient_balance_factor(frame)
    return y, weight


def brier_at_horizon(
    frame: pd.DataFrame,
    predicted_pfs: np.ndarray,
    horizon_month: int,
    km: CensoringKM,
    *,
    patient_balanced: bool,
) -> dict[str, Any]:
    if horizon_month not in HORIZONS_MONTHS:
        raise ValueError(f"unsupported horizon {horizon_month}")
    predicted_pfs = np.asarray(predicted_pfs, dtype=np.float64)
    risk = 1.0 - predicted_pfs
    y, weight = horizon_status_and_weight(
        frame,
        horizon_month * MONTH_DAYS,
        km,
        patient_balanced=patient_balanced,
    )
    use = np.isfinite(risk) & np.isfinite(weight) & (weight > 0)
    if not use.any():
        return {"brier": float("nan"), "n_known": 0, "weight_sum": 0.0}
    brier = np.sum(weight[use] * (risk[use] - y[use]) ** 2) / np.sum(weight[use])
    return {
        "brier": float(brier),
        "n_known": int(use.sum()),
        "weight_sum": float(weight[use].sum()),
        "mean_predicted_risk": float(np.average(risk[use], weights=weight[use])),
        "observed_risk": float(np.average(y[use], weights=weight[use])),
    }


def patient_mean(values: np.ndarray, frame: pd.DataFrame) -> float:
    temp = pd.DataFrame({"patient_id": frame["patient_id"].astype(str).to_numpy(), "value": np.asarray(values, dtype=float)})
    return float(temp.groupby("patient_id", observed=True)["value"].mean().mean())


def v2_metric_bundle(
    frame: pd.DataFrame,
    logits: np.ndarray,
    km: CensoringKM,
) -> dict[str, Any]:
    curves = curves_from_logits(logits)
    fractional_nll = fractional_censor_nll_per_row(
        logits,
        frame["survival_time_days"].to_numpy(dtype=float),
        frame["survival_cause"].astype(int).to_numpy(),
    )
    patient_briers = {}
    row_briers = {}
    for horizon in HORIZONS_MONTHS:
        pfs = curves["pfs"][:, horizon - 1]
        patient_briers[f"{horizon}m"] = brier_at_horizon(frame, pfs, horizon, km, patient_balanced=True)
        row_briers[f"{horizon}m"] = brier_at_horizon(frame, pfs, horizon, km, patient_balanced=False)
    patient_values = [patient_briers[f"{h}m"]["brier"] for h in HORIZONS_MONTHS]
    row_values = [row_briers[f"{h}m"]["brier"] for h in HORIZONS_MONTHS]
    return {
        "rows": int(len(frame)),
        "patients": int(frame["patient_id"].nunique()),
        "fractional_censor_row_mean_nll_v1": float(np.mean(fractional_nll)),
        "fractional_censor_patient_mean_nll_v1": patient_mean(fractional_nll, frame),
        "patient_balanced_pfs": patient_briers,
        "row_weighted_pfs": row_briers,
        "patient_brier_4h_mean": float(np.mean(patient_values)),
        "row_brier_4h_mean": float(np.mean(row_values)),
        "curve_invariants": validate_curve_invariants(curves),
    }

def resample_patient_clusters(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    patients = frame["patient_id"].astype(str).drop_duplicates().to_numpy()
    draws = rng.integers(0, len(patients), size=len(patients))
    pieces = []
    patient_text = frame["patient_id"].astype(str)
    for draw_number, patient_index in enumerate(draws):
        patient = patients[patient_index]
        piece = frame.loc[patient_text == patient].copy()
        piece["_bootstrap_cluster_draw"] = draw_number
        piece["_bootstrap_source_patient"] = patient
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def cluster_bootstrap_mean_delta(
    patient_delta: pd.Series,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    values = patient_delta.dropna().to_numpy(dtype=np.float64)
    if len(values) < 2:
        raise ValueError("need at least two patient deltas")
    rng = np.random.default_rng(seed)
    boot = np.empty(repetitions, dtype=np.float64)
    for i in range(repetitions):
        sampled_indices = rng.integers(0, len(values), size=len(values))
        boot[i] = values[sampled_indices].mean()
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "patients": int(len(values)),
        "repetitions": int(repetitions),
        "seed": int(seed),
    }
