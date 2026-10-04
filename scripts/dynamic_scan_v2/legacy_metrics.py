"""Exact historical metric reproduction only. Do not use this module to define new V2 likelihoods."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from metrics import (
    CAUSE_NO_EVENT_OR_CENSOR,
    HORIZONS_MONTHS,
    MONTHS,
    MONTH_DAYS,
    brier_at_horizon,
    curves_from_logits,
    patient_mean,
    validate_curve_invariants,
)


def _validate(logits: np.ndarray, time_days: np.ndarray, cause: np.ndarray):
    logits = np.asarray(logits, dtype=np.float64)
    time_days = np.asarray(time_days, dtype=np.float64)
    cause = np.asarray(cause, dtype=np.int64)
    if logits.ndim != 3 or logits.shape[1:] != (MONTHS, 4):
        raise ValueError(f"Expected logits [N,{MONTHS},4], got {logits.shape}")
    if len(time_days) != len(logits) or len(cause) != len(logits):
        raise ValueError("logit/time/cause length mismatch")
    if not np.isfinite(logits).all() or not np.isfinite(time_days).all():
        raise ValueError("non-finite legacy metric input")
    if (time_days <= 0).any():
        raise ValueError("zero/negative survival times are excluded by the frozen supervised rule")
    if not np.isin(cause, [0, 1, 2, 3]).all():
        raise ValueError("invalid cause code")
    return logits, time_days, cause


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    maximum = np.max(logits, axis=-1, keepdims=True)
    shifted = logits - maximum
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def legacy_event_bin(time_days: float) -> int:
    # Mathematically equivalent to frozen torch.ceil(time_days / MONTH_DAYS)
    # for the positive-time supervised rows, with a tiny numerical boundary guard.
    month = int(np.ceil(float(time_days) / MONTH_DAYS - 1e-12))
    return int(np.clip(month, 1, MONTHS))


def legacy_nll_per_row(logits: np.ndarray, time_days: np.ndarray, cause: np.ndarray) -> np.ndarray:
    """Exact historical CKPT5 month-ceiling likelihood, isolated for replay."""
    logits, time_days, cause = _validate(logits, time_days, cause)
    lp = _log_softmax(logits)
    out = np.empty(len(logits), dtype=np.float64)
    for i in range(len(logits)):
        month = legacy_event_bin(time_days[i])
        c = int(cause[i])
        if c == CAUSE_NO_EVENT_OR_CENSOR:
            out[i] = -lp[i, :month, CAUSE_NO_EVENT_OR_CENSOR].sum()
        else:
            out[i] = -(
                lp[i, : month - 1, CAUSE_NO_EVENT_OR_CENSOR].sum()
                + lp[i, month - 1, c]
            )
    return out


def legacy_metric_bundle(frame: pd.DataFrame, logits: np.ndarray, km) -> dict[str, Any]:
    curves = curves_from_logits(logits)
    nll = legacy_nll_per_row(
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
    return {
        "legacy_row_mean_nll": float(np.mean(nll)),
        "legacy_patient_mean_nll": patient_mean(nll, frame),
        "legacy_patient_integrated_brier_4h": float(np.mean([patient_briers[f"{h}m"]["brier"] for h in HORIZONS_MONTHS])),
        "legacy_row_integrated_brier_4h": float(np.mean([row_briers[f"{h}m"]["brier"] for h in HORIZONS_MONTHS])),
        "legacy_patient_balanced_pfs": patient_briers,
        "legacy_row_weighted_pfs": row_briers,
        "legacy_curve_invariants": validate_curve_invariants(curves),
    }
