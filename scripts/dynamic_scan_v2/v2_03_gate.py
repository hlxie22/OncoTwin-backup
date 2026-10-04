from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn

from metrics import (
    MONTHS,
    MONTH_DAYS,
    curves_from_logits,
    softmax,
)


FORBIDDEN_FEATURE_TOKENS = (
    "survival",
    "survival_cause",
    "censor",
    "outcome",
    "event_time",
    "patient_id",
    "scan_episode_id",
    "site_id",
    "development_fold",
    "split",
)


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray


def fit_standardizer(x: np.ndarray) -> Standardizer:
    x = np.asarray(x, dtype=np.float64)
    mean = np.nanmean(x, axis=0)
    scale = np.nanstd(x, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
    return Standardizer(mean.astype(np.float32), scale.astype(np.float32))


def apply_standardizer(x: np.ndarray, scaler: Standardizer) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.where(np.isfinite(x), x, scaler.mean)
    return ((x - scaler.mean) / scaler.scale).astype(np.float32)


def blend_logits(pre_logits: np.ndarray, post_logits: np.ndarray, alpha: np.ndarray | float) -> np.ndarray:
    pre = np.asarray(pre_logits, dtype=np.float64)
    post = np.asarray(post_logits, dtype=np.float64)
    if pre.shape != post.shape:
        raise ValueError(f"PRE/POST shape mismatch: {pre.shape} vs {post.shape}")
    if pre.ndim != 3 or pre.shape[1:] != (MONTHS, 4):
        raise ValueError(f"Expected [N,{MONTHS},4], got {pre.shape}")
    a = np.asarray(alpha, dtype=np.float64)
    if a.ndim == 0:
        a = np.full(len(pre), float(a), dtype=np.float64)
    if a.shape != (len(pre),):
        raise ValueError(f"alpha must be scalar or [N], got {a.shape}")
    if ((a < -1e-9) | (a > 1.0 + 1e-9)).any():
        raise ValueError("alpha must be within [0,1]")
    return pre + a[:, None, None] * (post - pre)


def _safe_log1p(value: pd.Series | np.ndarray, cap: float | None = None) -> np.ndarray:
    x = np.asarray(value, dtype=np.float64)
    x = np.where(np.isfinite(x) & (x >= 0), x, 0.0)
    if cap is not None:
        x = np.minimum(x, cap)
    return np.log1p(x)


def _bool_col(frame: pd.DataFrame, name: str) -> np.ndarray:
    if name not in frame:
        return np.zeros(len(frame), dtype=np.float64)
    s = frame[name]
    if s.dtype == bool:
        return s.to_numpy(dtype=np.float64)
    return s.fillna(False).astype(bool).to_numpy(dtype=np.float64)


def _num_col(frame: pd.DataFrame, name: str, default: float = 0.0) -> np.ndarray:
    if name not in frame:
        return np.full(len(frame), default, dtype=np.float64)
    return pd.to_numeric(frame[name], errors="coerce").fillna(default).to_numpy(dtype=np.float64)


def branch_summary_features(pre_logits: np.ndarray, post_logits: np.ndarray) -> tuple[np.ndarray, list[str]]:
    pre = np.asarray(pre_logits, dtype=np.float64)
    post = np.asarray(post_logits, dtype=np.float64)
    pre_curves = curves_from_logits(pre)
    post_curves = curves_from_logits(post)

    cols: list[np.ndarray] = []
    names: list[str] = []
    for h in (3, 6, 12, 18):
        cols.append(1.0 - pre_curves["pfs"][:, h - 1])
        names.append(f"pre_risk_{h}m")
    for h in (3, 6, 12, 18):
        cols.append((1.0 - post_curves["pfs"][:, h - 1]) - (1.0 - pre_curves["pfs"][:, h - 1]))
        names.append(f"post_minus_pre_risk_{h}m")

    # Algebraically valid update summaries: each cause is relative to the no-event
    # class, so arbitrary class-independent softmax logit offsets cancel exactly.
    pre_rel = pre[:, :, 1:] - pre[:, :, [0]]
    post_rel = post[:, :, 1:] - post[:, :, [0]]
    delta_rel = post_rel - pre_rel
    cause_names = ("progression", "death", "switch")
    for j, name in enumerate(cause_names):
        cols.append(delta_rel[:, :, j].mean(axis=1))
        names.append(f"cause_relative_delta_mean_{name}")
    for j, name in enumerate(cause_names):
        cols.append(np.abs(delta_rel[:, :, j]).mean(axis=1))
        names.append(f"cause_relative_delta_mean_abs_{name}")
    cols.append(np.max(np.abs(delta_rel), axis=(1, 2)))
    names.append("cause_relative_delta_max_abs")

    cols.append(pre_curves["entropy"])
    names.append("pre_conditional_entropy")
    cols.append(post_curves["entropy"] - pre_curves["entropy"])
    names.append("post_minus_pre_conditional_entropy")

    return np.column_stack(cols).astype(np.float32), names


def build_gate_features(
    scan_meta: pd.DataFrame,
    pre_logits: np.ndarray,
    post_logits: np.ndarray,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    if len(scan_meta) != len(pre_logits):
        raise ValueError("metadata/logit length mismatch")

    state = scan_meta.get("clinical_state", pd.Series([""] * len(scan_meta))).astype(str).str.upper()
    days_prev = _num_col(scan_meta, "days_since_previous_assessment", 0.0)
    days_comp = _num_col(scan_meta, "days_since_previous_comparable_assessment", 0.0)
    prior_scans = _num_col(scan_meta, "prior_scan_count", 0.0)
    nonprog_run = _num_col(scan_meta, "prior_nonprogressive_run", 0.0)
    genomic_available = _bool_col(scan_meta, "genomic_available")
    genomic_age = _num_col(scan_meta, "genomic_age_days", 0.0)
    genomic_age = np.where(genomic_available > 0.5, genomic_age, 0.0)

    structured = [
        _safe_log1p(days_prev, cap=7300.0),
        (days_prev > 0).astype(np.float64),
        _safe_log1p(days_comp, cap=7300.0),
        (days_comp > 0).astype(np.float64),
        _safe_log1p(prior_scans, cap=100.0),
        _safe_log1p(nonprog_run, cap=100.0),
        np.clip(_num_col(scan_meta, "coverage_jaccard", 0.0), 0.0, 1.0),
        np.clip(_num_col(scan_meta, "current_coverage_count", 0.0) / 5.0, 0.0, 1.0),
        np.clip(_num_col(scan_meta, "coverage_gained_count", 0.0) / 5.0, 0.0, 1.0),
        np.clip(_num_col(scan_meta, "coverage_lost_count", 0.0) / 5.0, 0.0, 1.0),
        _bool_col(scan_meta, "coverage_comparable_to_previous"),
        (state == "NON_PROGRESSIVE").to_numpy(dtype=np.float64),
        (state == "INDETERMINATE").to_numpy(dtype=np.float64),
        _bool_col(scan_meta, "current_modality_observed"),
        _bool_col(scan_meta, "current_site_positive_stream_observed"),
        genomic_available,
        _safe_log1p(genomic_age, cap=10000.0),
    ]
    structured_names = [
        "log_days_since_previous_assessment",
        "has_previous_assessment",
        "log_days_since_previous_comparable_assessment",
        "has_previous_comparable_assessment",
        "log_prior_scan_count",
        "log_prior_nonprogressive_run",
        "coverage_jaccard",
        "current_coverage_fraction",
        "coverage_gained_fraction",
        "coverage_lost_fraction",
        "coverage_comparable_to_previous",
        "current_state_non_progressive",
        "current_state_indeterminate",
        "current_modality_observed",
        "current_site_positive_stream_observed",
        "genomic_available",
        "log_genomic_age_days",
    ]

    branch, branch_names = branch_summary_features(pre_logits, post_logits)
    x = np.column_stack(structured + [branch]).astype(np.float32)
    names = structured_names + branch_names

    for name in names:
        lowered = name.lower()
        if any(token in lowered for token in FORBIDDEN_FEATURE_TOKENS):
            raise RuntimeError(f"Forbidden gate feature name: {name}")
    if not np.isfinite(x).all():
        raise RuntimeError("Gate features contain non-finite values")

    strata = pd.DataFrame({
        "genomic_available": genomic_available > 0.5,
        "modality_observed": _bool_col(scan_meta, "current_modality_observed") > 0.5,
        "site_positive_stream_observed": _bool_col(scan_meta, "current_site_positive_stream_observed") > 0.5,
        "coverage_comparable": _bool_col(scan_meta, "coverage_comparable_to_previous") > 0.5,
        "current_state": state.to_numpy(),
        "prior_scan_count": prior_scans,
    })
    return x, names, strata


class LinearSigmoidGate(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x)).squeeze(-1)


class NonlinearSigmoidGate(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        # Initialize near the alpha=0.5 baseline.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).squeeze(-1)


def patient_balance_weights(patient_ids: Iterable[str]) -> np.ndarray:
    s = pd.Series([str(x) for x in patient_ids])
    counts = s.map(s.value_counts()).to_numpy(dtype=np.float64)
    w = 1.0 / np.maximum(counts, 1.0)
    return w.astype(np.float32)


def torch_fractional_nll_per_row(
    logits: torch.Tensor,
    time_days: torch.Tensor,
    cause: torch.Tensor,
) -> torch.Tensor:
    if logits.ndim != 3 or logits.shape[1:] != (MONTHS, 4):
        raise ValueError(f"Expected logits [N,{MONTHS},4], got {tuple(logits.shape)}")
    lp = torch.log_softmax(logits, dim=-1)
    n = logits.shape[0]
    month_idx = torch.arange(MONTHS, device=logits.device)[None, :]

    t = time_days
    c = cause.long()
    event = c != 0

    ratio = t / float(MONTH_DAYS)
    event_bin = torch.ceil(ratio - 1e-12).long().clamp(1, MONTHS)
    previous_mask = month_idx < (event_bin[:, None] - 1)
    event_surv = (lp[:, :, 0] * previous_mask).sum(dim=1)
    row_index = torch.arange(n, device=logits.device)
    event_lp = lp[row_index, event_bin - 1, c.clamp(0, 3)]
    event_nll = -(event_surv + event_lp)

    completed = torch.floor(ratio + 1e-12).long().clamp(0, MONTHS)
    full_mask = month_idx < completed[:, None]
    censor_lp = (lp[:, :, 0] * full_mask).sum(dim=1)
    remainder = t - completed.to(t.dtype) * float(MONTH_DAYS)
    fraction = torch.clamp(remainder / float(MONTH_DAYS), 0.0, 1.0)
    has_partial = completed < MONTHS
    partial_index = completed.clamp(max=MONTHS - 1)
    partial_lp = lp[row_index, partial_index, 0]
    censor_lp = censor_lp + torch.where(has_partial, fraction * partial_lp, torch.zeros_like(censor_lp))
    censor_nll = -censor_lp

    return torch.where(event, event_nll, censor_nll)


def torch_blend(pre: torch.Tensor, post: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    return pre + alpha[:, None, None] * (post - pre)


def fit_gate_model(
    *,
    family: str,
    x_train: np.ndarray,
    pre_logits: np.ndarray,
    post_logits: np.ndarray,
    time_days: np.ndarray,
    cause: np.ndarray,
    patient_ids: Iterable[str],
    seed: int,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[nn.Module, Standardizer, dict[str, Any]]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    scaler = fit_standardizer(x_train)
    xz = apply_standardizer(x_train, scaler)

    if family == "linear_sigmoid":
        cfg = config["linear_gate"]
        model: nn.Module = LinearSigmoidGate(xz.shape[1])
    elif family == "nonlinear_8":
        cfg = config["nonlinear_gate"]
        model = NonlinearSigmoidGate(xz.shape[1], hidden_dim=int(cfg["hidden_dim"]))
    else:
        raise ValueError(f"Unknown gate family: {family}")

    model = model.to(device)
    x_t = torch.as_tensor(xz, dtype=torch.float32, device=device)
    pre_t = torch.as_tensor(pre_logits, dtype=torch.float32, device=device)
    post_t = torch.as_tensor(post_logits, dtype=torch.float32, device=device)
    time_t = torch.as_tensor(time_days, dtype=torch.float32, device=device)
    cause_t = torch.as_tensor(cause, dtype=torch.long, device=device)
    weight_t = torch.as_tensor(patient_balance_weights(patient_ids), dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=0.0,
    )
    epochs = int(cfg["epochs"])
    wd = float(cfg["weight_decay"])
    shrink = float(cfg["alpha_shrinkage"])
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        alpha = model(x_t)
        logits = torch_blend(pre_t, post_t, alpha)
        nll = torch_fractional_nll_per_row(logits, time_t, cause_t)
        data_loss = (nll * weight_t).sum() / weight_t.sum().clamp_min(1e-8)
        l2 = torch.zeros((), device=device)
        for p in model.parameters():
            if p.ndim > 1:
                l2 = l2 + p.pow(2).mean()
        alpha_penalty = (alpha - 0.5).pow(2).mean()
        loss = data_loss + wd * l2 + shrink * alpha_penalty
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if epoch in {1, epochs} or epoch % 25 == 0:
            history.append({
                "epoch": float(epoch),
                "loss": float(loss.detach().cpu()),
                "data_nll": float(data_loss.detach().cpu()),
                "alpha_mean": float(alpha.mean().detach().cpu()),
                "alpha_std": float(alpha.std(unbiased=False).detach().cpu()),
            })

    return model, scaler, {
        "family": family,
        "seed": int(seed),
        "epochs": epochs,
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "history": history,
    }


def predict_gate(model: nn.Module, scaler: Standardizer, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    xz = apply_standardizer(x, scaler)
    with torch.no_grad():
        alpha = model(torch.as_tensor(xz, dtype=torch.float32, device=device))
    out = alpha.detach().cpu().numpy().astype(np.float64)
    if not np.isfinite(out).all() or ((out < 0) | (out > 1)).any():
        raise RuntimeError("Gate produced invalid alpha")
    return out


def gate_distribution(alpha: np.ndarray) -> dict[str, float]:
    a = np.asarray(alpha, dtype=np.float64)
    return {
        "mean": float(a.mean()),
        "std": float(a.std(ddof=0)),
        "q01": float(np.quantile(a, 0.01)),
        "q05": float(np.quantile(a, 0.05)),
        "q25": float(np.quantile(a, 0.25)),
        "q50": float(np.quantile(a, 0.50)),
        "q75": float(np.quantile(a, 0.75)),
        "q95": float(np.quantile(a, 0.95)),
        "q99": float(np.quantile(a, 0.99)),
        "fraction_below_0_05": float((a < 0.05).mean()),
        "fraction_above_0_95": float((a > 0.95).mean()),
        "saturation_fraction": float(((a < 0.05) | (a > 0.95)).mean()),
    }


def assert_class_offset_invariant_features(
    scan_meta: pd.DataFrame,
    pre_logits: np.ndarray,
    post_logits: np.ndarray,
    seed: int = 7,
) -> bool:
    x0, names0, _ = build_gate_features(scan_meta, pre_logits, post_logits)
    rng = np.random.default_rng(seed)
    pre_shift = rng.normal(size=(len(pre_logits), MONTHS, 1))
    post_shift = rng.normal(size=(len(post_logits), MONTHS, 1))
    x1, names1, _ = build_gate_features(scan_meta, pre_logits + pre_shift, post_logits + post_shift)
    return names0 == names1 and bool(np.allclose(x0, x1, atol=2e-6, rtol=2e-6))
