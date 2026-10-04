from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

EPS = 1e-8


def weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    use = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not np.any(use):
        return float("nan")
    den = float(w[use].sum())
    if den <= 0:
        return float("nan")
    return float(np.dot(x[use], w[use]) / den)


def effective_sample_size(w: np.ndarray) -> float:
    w = np.asarray(w, dtype=np.float64)
    w = w[np.isfinite(w) & (w > 0)]
    if len(w) == 0:
        return 0.0
    s1 = float(w.sum())
    s2 = float(np.dot(w, w))
    return 0.0 if s2 <= 0 else float((s1 * s1) / s2)


def weighted_brier(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    use = np.isfinite(y) & np.isfinite(p) & np.isfinite(w) & (w > 0)
    if not np.any(use):
        return float("nan")
    err = (y[use] - p[use]) ** 2
    return weighted_mean(err, w[use])


def weighted_logloss(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    use = np.isfinite(y) & np.isfinite(p) & np.isfinite(w) & (w > 0)
    if not np.any(use):
        return float("nan")
    pp = np.clip(p[use], EPS, 1.0 - EPS)
    loss = -(y[use] * np.log(pp) + (1.0 - y[use]) * np.log1p(-pp))
    return weighted_mean(loss, w[use])


def weighted_auc(y: np.ndarray, score: np.ndarray, w: np.ndarray) -> float:
    """Weighted ROC AUC with exact 0.5 handling for score ties.

    This is the weighted Mann-Whitney estimand. Rows with nonpositive/invalid
    weights are excluded. y must be binary on retained rows.
    """
    y = np.asarray(y, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    use = np.isfinite(y) & np.isfinite(score) & np.isfinite(w) & (w > 0)
    y, score, w = y[use], score[use], w[use]
    if len(y) == 0:
        return float("nan")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("weighted_auc expects binary y")
    pos_total = float(w[y == 1].sum())
    neg_total = float(w[y == 0].sum())
    if pos_total <= 0 or neg_total <= 0:
        return float("nan")

    order = np.argsort(score, kind="mergesort")
    y, score, w = y[order], score[order], w[order]
    numer = 0.0
    cum_neg = 0.0
    start = 0
    n = len(y)
    while start < n:
        stop = start + 1
        while stop < n and score[stop] == score[start]:
            stop += 1
        yy = y[start:stop]
        ww = w[start:stop]
        wp = float(ww[yy == 1].sum())
        wn = float(ww[yy == 0].sum())
        numer += wp * (cum_neg + 0.5 * wn)
        cum_neg += wn
        start = stop
    return float(numer / (pos_total * neg_total))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(p) - np.log1p(-p)


def expit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def weighted_logistic_fit(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    *,
    max_iter: int = 100,
    ridge: float = 1e-8,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    use = np.isfinite(y) & np.isfinite(w) & (w > 0) & np.all(np.isfinite(x), axis=1)
    x, y, w = x[use], y[use], w[use]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return np.full(x.shape[1] + 1, np.nan, dtype=np.float64)
    design = np.column_stack([np.ones(len(x)), x])
    beta = np.zeros(design.shape[1], dtype=np.float64)
    penalty = np.eye(len(beta), dtype=np.float64) * ridge
    penalty[0, 0] = 0.0
    for _ in range(max_iter):
        eta = design @ beta
        p = expit(eta)
        grad = design.T @ (w * (y - p)) - penalty @ beta
        h_w = w * p * (1.0 - p)
        hess = -(design.T * h_w) @ design - penalty
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hess) @ grad
        beta_new = beta - step
        if np.max(np.abs(beta_new - beta)) < 1e-9:
            beta = beta_new
            break
        beta = beta_new
    return beta


def calibration_intercept_slope(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> dict[str, float]:
    beta = weighted_logistic_fit(logit(p), y, w)
    return {
        "intercept": float(beta[0]),
        "slope": float(beta[1]),
    }


def calibration_ece(y: np.ndarray, p: np.ndarray, w: np.ndarray, bins: int = 10) -> tuple[float, list[dict[str, Any]]]:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    use = np.isfinite(y) & np.isfinite(p) & np.isfinite(w) & (w > 0)
    idx = np.where(use)[0]
    if len(idx) == 0:
        return float("nan"), []
    idx = idx[np.argsort(p[idx], kind="mergesort")]
    chunks = np.array_split(idx, min(int(bins), len(idx)))
    total_w = float(w[idx].sum())
    ece = 0.0
    rows: list[dict[str, Any]] = []
    for b, chunk in enumerate(chunks, 1):
        if len(chunk) == 0:
            continue
        ww = w[chunk]
        ws = float(ww.sum())
        pred = weighted_mean(p[chunk], ww)
        obs = weighted_mean(y[chunk], ww)
        ece += (ws / total_w) * abs(pred - obs)
        rows.append({
            "bin": int(b),
            "rows": int(len(chunk)),
            "weight_sum": ws,
            "mean_predicted_risk": pred,
            "observed_risk": obs,
        })
    return float(ece), rows


def summarize_binary_metric(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> dict[str, float]:
    cal = calibration_intercept_slope(y, p, w)
    ece, _ = calibration_ece(y, p, w, bins=10)
    return {
        "brier": weighted_brier(y, p, w),
        "auc": weighted_auc(y, p, w),
        "logloss": weighted_logloss(y, p, w),
        "mean_predicted_risk": weighted_mean(p, w),
        "observed_risk": weighted_mean(y, w),
        "calibration_intercept": cal["intercept"],
        "calibration_slope": cal["slope"],
        "ece_10": ece,
        "effective_sample_size": effective_sample_size(w),
        "weight_sum": float(np.asarray(w, dtype=np.float64)[np.isfinite(w) & (w > 0)].sum()),
    }


def bootstrap_percentile(values: np.ndarray, *, alpha: float = 0.05) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"mean": float("nan"), "median": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    return {
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "ci_low": float(np.quantile(v, alpha / 2.0)),
        "ci_high": float(np.quantile(v, 1.0 - alpha / 2.0)),
    }


def antolini_style_concordance(
    time_days: np.ndarray,
    cause: np.ndarray,
    patient_id: np.ndarray,
    risk: np.ndarray,
    month_days: float,
    *,
    max_months: int = 24,
) -> dict[str, float]:
    """Patient-balanced Antolini-style concordance for PFS composite.

    Cause codes 1/2 are progression/death PFS events; other causes are treated
    as non-events for pair formation and can be comparators while observed past
    the index event time. Same-patient landmark pairs are excluded.
    """
    t = np.asarray(time_days, dtype=np.float64)
    c = np.asarray(cause, dtype=int)
    pid = np.asarray(patient_id).astype(str)
    risk = np.asarray(risk, dtype=np.float64)
    if risk.ndim != 2 or risk.shape[0] != len(t) or risk.shape[1] < max_months:
        raise ValueError("risk must be [N, >=max_months]")
    unique, inv, counts = np.unique(pid, return_inverse=True, return_counts=True)
    row_w = 1.0 / counts[inv].astype(np.float64)
    event_idx = np.where(np.isin(c, [1, 2]) & np.isfinite(t) & (t > 0) & (t <= max_months * month_days))[0]
    numer = 0.0
    denom = 0.0
    comparable_pairs = 0
    for i in event_idx:
        month = int(np.clip(np.ceil(t[i] / month_days), 1, max_months))
        ri = float(risk[i, month - 1])
        comp = (t > t[i]) & (pid != pid[i]) & np.isfinite(t)
        if not np.any(comp):
            continue
        wj = row_w[comp]
        rj = risk[comp, month - 1]
        wi = float(row_w[i])
        denom += wi * float(wj.sum())
        numer += wi * float(wj[rj < ri].sum() + 0.5 * wj[rj == ri].sum())
        comparable_pairs += int(comp.sum())
    return {
        "concordance": float(numer / denom) if denom > 0 else float("nan"),
        "weighted_pair_mass": float(denom),
        "raw_comparable_pairs": int(comparable_pairs),
        "pfs_event_rows_within_24m": int(len(event_idx)),
    }
