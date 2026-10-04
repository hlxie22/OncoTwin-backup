from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import hashlib
import json

import numpy as np
import torch

from v2_02_models import TemporalFixedSurvivalModel

MONTHS = 24
CAUSES = 4
CAUSE_ORDER = ("no_event", "progression", "death", "switch")
HORIZONS_MONTHS = (3, 6, 12, 18)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def blend_logits(pre: np.ndarray, post: np.ndarray, alpha: float) -> np.ndarray:
    pre = np.asarray(pre, dtype=np.float32)
    post = np.asarray(post, dtype=np.float32)
    if pre.shape != post.shape or pre.ndim != 3 or pre.shape[1:] != (MONTHS, CAUSES):
        raise ValueError(f"Expected matched [N,{MONTHS},{CAUSES}] logits")
    a = float(alpha)
    if not (0.0 <= a <= 1.0):
        raise ValueError("alpha must be in [0,1]")
    if a == 0.0:
        return pre.copy()
    if a == 1.0:
        return post.copy()
    return pre + np.float32(a) * (post - pre)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    p = e / np.sum(e, axis=-1, keepdims=True)
    return p.astype(np.float32)


def conditional_probs_to_curves(probs: np.ndarray) -> dict[str, np.ndarray]:
    """Convert monthly conditional competing-risk probabilities to cumulative curves.

    probs[...,0] is conditional no-event probability for the interval. Causes 1..3
    are mutually exclusive progression/death/switch probabilities conditional on
    being event-free at the interval start.
    """
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 3 or p.shape[1:] != (MONTHS, CAUSES):
        raise ValueError(f"Expected probabilities [N,{MONTHS},{CAUSES}]")
    if not np.isfinite(p).all() or (p < -1e-7).any():
        raise ValueError("Invalid conditional probabilities")
    sums = p.sum(axis=-1)
    if np.max(np.abs(sums - 1.0)) > 2e-5:
        raise ValueError("Conditional probabilities do not sum to one")

    n = p.shape[0]
    survival = np.empty((n, MONTHS), dtype=np.float64)
    cif = np.zeros((n, MONTHS, 3), dtype=np.float64)
    s = np.ones(n, dtype=np.float64)
    running = np.zeros((n, 3), dtype=np.float64)
    for m in range(MONTHS):
        start = s.copy()
        running += start[:, None] * p[:, m, 1:4]
        s *= p[:, m, 0]
        survival[:, m] = s
        cif[:, m, :] = running
    total = survival + cif.sum(axis=-1)
    if np.max(np.abs(total - 1.0)) > 2e-5:
        raise ValueError("Competing-risk mass conservation failed")
    if np.min(np.diff(cif, axis=1)) < -2e-6:
        raise ValueError("CIF is not monotone")
    return {
        "event_free_survival": survival.astype(np.float32),
        "progression_cif": cif[:, :, 0].astype(np.float32),
        "death_cif": cif[:, :, 1].astype(np.float32),
        "switch_cif": cif[:, :, 2].astype(np.float32),
        "progression_or_death_cif": (cif[:, :, 0] + cif[:, :, 1]).astype(np.float32),
        "pfs_survival": (1.0 - cif[:, :, 0] - cif[:, :, 1]).astype(np.float32),
    }


@dataclass(frozen=True)
class PreparedLandmarkBatch:
    temporal_pre: np.ndarray
    temporal_post: np.ndarray
    tumor_embedding: np.ndarray
    scan_number_patient_log: np.ndarray
    genomic_available: np.ndarray
    genomic_age_scaled: np.ndarray
    current_scan_state_known: np.ndarray | None = None
    required_clock_known: np.ndarray | None = None

    def validate(self) -> int:
        tp = np.asarray(self.temporal_pre)
        tq = np.asarray(self.temporal_post)
        tumor = np.asarray(self.tumor_embedding)
        if tp.ndim != 2 or tp.shape[1] != 192:
            raise ValueError("temporal_pre must be [N,192]")
        n = tp.shape[0]
        if tq.shape != (n, 192):
            raise ValueError("temporal_post must be [N,192]")
        if tumor.shape != (n, 128):
            raise ValueError("tumor_embedding must be [N,128]")
        for name in ("scan_number_patient_log", "genomic_available", "genomic_age_scaled"):
            a = np.asarray(getattr(self, name)).reshape(-1)
            if a.shape != (n,):
                raise ValueError(f"{name} must contain N values")
            if not np.isfinite(a).all():
                raise ValueError(f"{name} contains non-finite values")
        for name, a in (("temporal_pre", tp), ("temporal_post", tq), ("tumor_embedding", tumor)):
            if not np.isfinite(a).all():
                raise ValueError(f"{name} contains non-finite values")
        ga = np.asarray(self.genomic_available).reshape(-1)
        if not np.all((ga == 0) | (ga == 1)):
            raise ValueError("genomic_available must be exactly 0/1")
        if (np.asarray(self.genomic_age_scaled).reshape(-1) < 0).any():
            raise ValueError("genomic_age_scaled must be nonnegative")
        for name in ("current_scan_state_known", "required_clock_known"):
            value = getattr(self, name)
            if value is not None and np.asarray(value).reshape(-1).shape != (n,):
                raise ValueError(f"{name} must contain N values")
        return n

    def portable_context(self) -> np.ndarray:
        self.validate()
        return np.stack(
            [
                np.asarray(self.scan_number_patient_log, dtype=np.float32).reshape(-1),
                np.asarray(self.genomic_available, dtype=np.float32).reshape(-1),
                np.asarray(self.genomic_age_scaled, dtype=np.float32).reshape(-1),
            ],
            axis=1,
        )

    def applicability(self) -> tuple[np.ndarray, list[str]]:
        n = self.validate()
        scan = np.ones(n, dtype=bool) if self.current_scan_state_known is None else np.asarray(self.current_scan_state_known, dtype=bool).reshape(-1)
        clock = np.ones(n, dtype=bool) if self.required_clock_known is None else np.asarray(self.required_clock_known, dtype=bool).reshape(-1)
        ok = scan & clock
        reasons = []
        for i in range(n):
            if not scan[i]:
                reasons.append("ABSTAIN_UNKNOWN_CURRENT_SCAN_STATE")
            elif not clock[i]:
                reasons.append("ABSTAIN_UNKNOWN_REQUIRED_CLOCK")
            else:
                reasons.append("SUPPORTED")
        return ok, reasons


class OncoTwinV2Predictor:
    """Typed deployment predictor at the frozen prepared-landmark boundary."""

    def __init__(self, checkpoint: str | Path, *, device: str | torch.device = "cpu"):
        self.checkpoint_path = Path(checkpoint)
        payload = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        required = {
            "checkpoint", "family", "selected_rule", "locked_alpha", "training_scope",
            "is_crossfit_fold_head", "model_state", "model_dimensions", "source_hashes",
        }
        missing = required - set(payload)
        if missing:
            raise RuntimeError(f"Deploy checkpoint missing fields: {sorted(missing)}")
        if payload["checkpoint"] != "V2-07" or payload["family"] != "r1_temporal_fixed":
            raise RuntimeError("Not a V2-07 r1_temporal_fixed deploy checkpoint")
        if payload["training_scope"] != "all_v2_development_survival_eligible":
            raise RuntimeError("Deploy checkpoint training scope is invalid")
        if bool(payload["is_crossfit_fold_head"]):
            raise RuntimeError("Cross-fit fold checkpoint cannot be loaded as deploy model")
        dims = payload["model_dimensions"]
        if dims != {"temporal": 192, "tumor": 128, "portable_context": 3, "months": 24, "causes": 4}:
            raise RuntimeError(f"Unexpected deploy dimensions: {dims}")
        alpha = float(payload["locked_alpha"])
        if abs(alpha - 0.52) > 1e-12:
            raise RuntimeError("Locked alpha changed")
        self.alpha = alpha
        self.payload = payload
        self.device = torch.device(device)
        self.model = TemporalFixedSurvivalModel().to(self.device)
        self.model.load_state_dict(payload["model_state"], strict=True)
        self.model.eval()

    def verify_upstream_files(self, repo: str | Path) -> dict[str, bool]:
        repo = Path(repo)
        checks = {}
        for rel, expected in self.payload["source_hashes"].items():
            path = repo / rel
            checks[rel] = path.is_file() and sha256_file(path) == expected
        if not all(checks.values()):
            bad = [k for k, v in checks.items() if not v]
            raise RuntimeError(f"Frozen upstream hash mismatch: {bad}")
        return checks

    def _predict_logits(self, batch: PreparedLandmarkBatch, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        n = batch.validate()
        ctx = batch.portable_context()
        pre_parts, post_parts = [], []
        with torch.no_grad():
            for start in range(0, n, int(batch_size)):
                sl = slice(start, min(start + int(batch_size), n))
                tp = torch.from_numpy(np.asarray(batch.temporal_pre[sl], dtype=np.float32)).to(self.device)
                tq = torch.from_numpy(np.asarray(batch.temporal_post[sl], dtype=np.float32)).to(self.device)
                tumor = torch.from_numpy(np.asarray(batch.tumor_embedding[sl], dtype=np.float32)).to(self.device)
                c = torch.from_numpy(ctx[sl]).to(self.device)
                a, b = self.model(tp, tq, tumor, c)
                pre_parts.append(a.float().cpu().numpy())
                post_parts.append(b.float().cpu().numpy())
        return np.concatenate(pre_parts, axis=0), np.concatenate(post_parts, axis=0)

    def predict(self, batch: PreparedLandmarkBatch, *, batch_size: int = 1024) -> dict[str, Any]:
        ok, reasons = batch.applicability()
        pre_logits, post_logits = self._predict_logits(batch, batch_size)
        selected_logits = blend_logits(pre_logits, post_logits, self.alpha)
        pre_prob = softmax_np(pre_logits)
        post_prob = softmax_np(post_logits)
        selected_prob = softmax_np(selected_logits)
        pre_curves = conditional_probs_to_curves(pre_prob)
        post_curves = conditional_probs_to_curves(post_prob)
        selected_curves = conditional_probs_to_curves(selected_prob)

        # Upstream V2-06 abstention semantics: unsupported rows are surfaced and
        # their clinical probability outputs are intentionally NaN.
        if (~ok).any():
            for a in (pre_logits, post_logits, selected_logits, pre_prob, post_prob, selected_prob):
                a[~ok] = np.nan
            for curves in (pre_curves, post_curves, selected_curves):
                for key in curves:
                    curves[key][~ok] = np.nan

        horizon_idx = np.asarray(HORIZONS_MONTHS, dtype=int) - 1
        return {
            "status": np.asarray(reasons, dtype=object),
            "supported": ok,
            "locked_alpha": self.alpha,
            "cause_order": CAUSE_ORDER,
            "horizons_months": HORIZONS_MONTHS,
            "pre_logits": pre_logits,
            "post_logits": post_logits,
            "selected_logits": selected_logits,
            "pre_conditional_probabilities": pre_prob,
            "post_conditional_probabilities": post_prob,
            "selected_conditional_probabilities": selected_prob,
            "pre_curves": pre_curves,
            "post_curves": post_curves,
            "selected_curves": selected_curves,
            "selected_horizon_progression_or_death_cif": selected_curves["progression_or_death_cif"][:, horizon_idx],
            "selected_horizon_switch_cif": selected_curves["switch_cif"][:, horizon_idx],
            "selected_horizon_pfs_survival": selected_curves["pfs_survival"][:, horizon_idx],
            "selected_horizon_event_free_survival": selected_curves["event_free_survival"][:, horizon_idx],
        }


def batch_from_npz(path: str | Path) -> PreparedLandmarkBatch:
    with np.load(path, allow_pickle=False) as z:
        required = {
            "temporal_pre", "temporal_post", "tumor_embedding",
            "scan_number_patient_log", "genomic_available", "genomic_age_scaled",
        }
        missing = required - set(z.files)
        if missing:
            raise ValueError(f"Prepared input missing arrays: {sorted(missing)}")
        kwargs = {k: np.asarray(z[k]) for k in required}
        for optional in ("current_scan_state_known", "required_clock_known"):
            if optional in z.files:
                kwargs[optional] = np.asarray(z[optional])
    return PreparedLandmarkBatch(**kwargs)


def jsonable_summary(prediction: dict[str, Any]) -> dict[str, Any]:
    return {
        "rows": int(len(prediction["supported"])),
        "supported_rows": int(np.asarray(prediction["supported"]).sum()),
        "locked_alpha": float(prediction["locked_alpha"]),
        "cause_order": list(prediction["cause_order"]),
        "horizons_months": list(prediction["horizons_months"]),
        "status_counts": {str(k): int(v) for k, v in zip(*np.unique(prediction["status"], return_counts=True))},
    }
