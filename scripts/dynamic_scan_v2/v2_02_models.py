from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from paired_scan_encoder import PairedScanEncoder

MONTHS = 24
CAUSES = 4
ADMIN_HORIZON_DAYS = 730.0
MONTH_DAYS = ADMIN_HORIZON_DAYS / MONTHS
EPS = 1e-12

PROFILE_NAMES = (
    "full_supported",
    "portable_scan_core",
    "no_genomics",
    "current_optional_missing",
    "scan_core_no_genomics",
    "mixed_sparse",
)
PROFILE_TO_CODE = {name: i for i, name in enumerate(PROFILE_NAMES)}


@dataclass
class StructuredBatch:
    history_core: torch.Tensor
    history_core_mask: torch.Tensor
    utilization_core: torch.Tensor
    utilization_core_mask: torch.Tensor
    current_core: torch.Tensor
    current_core_mask: torch.Tensor
    change_core: torch.Tensor
    change_core_mask: torch.Tensor
    optional_history: torch.Tensor
    optional_history_mask: torch.Tensor
    optional_history_recency: torch.Tensor
    optional_history_recency_mask: torch.Tensor
    optional_current: torch.Tensor
    optional_current_mask: torch.Tensor
    stream_availability: torch.Tensor
    stream_availability_mask: torch.Tensor
    tumor: torch.Tensor
    genomic_available: torch.Tensor
    genomic_age_scaled: torch.Tensor

    def clone(self) -> "StructuredBatch":
        return StructuredBatch(**{k: v.clone() for k, v in self.__dict__.items()})


def _zero_columns(values: torch.Tensor, masks: torch.Tensor, rows: torch.Tensor, columns: slice | list[int]) -> None:
    if rows.numel() == 0:
        return
    values[rows, columns] = 0.0
    masks[rows, columns] = 0.0


def apply_observation_profiles(batch: StructuredBatch, profile_codes: torch.Tensor) -> StructuredBatch:
    """Apply coherent source-level missingness before the paired-scan encoder.

    The same profile code is attached to the paired PRE/POST landmark. PRE does
    not consume current blocks, while POST consumes the transformed current
    blocks. Stream-availability indicators remain observed (mask=1) and are set
    to zero when a source is synthetically unavailable.
    """
    if profile_codes.ndim != 1 or profile_codes.shape[0] != batch.history_core.shape[0]:
        raise ValueError("profile_codes must be [batch]")
    out = batch.clone()

    portable = torch.where(
        (profile_codes == PROFILE_TO_CODE["portable_scan_core"])
        | (profile_codes == PROFILE_TO_CODE["scan_core_no_genomics"])
    )[0]
    current_only = torch.where(profile_codes == PROFILE_TO_CODE["current_optional_missing"])[0]
    mixed = torch.where(profile_codes == PROFILE_TO_CODE["mixed_sparse"])[0]
    no_genome = torch.where(
        (profile_codes == PROFILE_TO_CODE["no_genomics"])
        | (profile_codes == PROFILE_TO_CODE["scan_core_no_genomics"])
        | (profile_codes == PROFILE_TO_CODE["mixed_sparse"])
    )[0]

    # optional_history: modality[0:4], site[4:10]
    # optional_history_recency follows same partition.
    # optional_current: modality[0:4], site[4:10], first-positive[10:16].
    if portable.numel():
        _zero_columns(out.optional_history, out.optional_history_mask, portable, slice(0, 10))
        _zero_columns(out.optional_history_recency, out.optional_history_recency_mask, portable, slice(0, 10))
        _zero_columns(out.optional_current, out.optional_current_mask, portable, slice(0, 16))
        # stream indices 2:8 describe current/prior modality/site availability.
        out.stream_availability[portable, 2:8] = 0.0
        out.stream_availability_mask[portable, 2:8] = 1.0

    if current_only.numel():
        _zero_columns(out.optional_current, out.optional_current_mask, current_only, slice(0, 16))
        out.stream_availability[current_only, 2:6] = 0.0
        out.stream_availability_mask[current_only, 2:6] = 1.0

    # mixed sparse: preserve modality, remove site-positive history/current and genomics.
    if mixed.numel():
        _zero_columns(out.optional_history, out.optional_history_mask, mixed, slice(4, 10))
        _zero_columns(out.optional_history_recency, out.optional_history_recency_mask, mixed, slice(4, 10))
        _zero_columns(out.optional_current, out.optional_current_mask, mixed, slice(4, 16))
        for col in (4, 5, 7):
            out.stream_availability[mixed, col] = 0.0
            out.stream_availability_mask[mixed, col] = 1.0

    if no_genome.numel():
        out.tumor[no_genome] = 0.0
        out.genomic_available[no_genome] = 0.0
        out.genomic_age_scaled[no_genome] = 0.0

    return out


def torch_event_bin(time_days: torch.Tensor) -> torch.Tensor:
    month = torch.ceil(time_days / MONTH_DAYS - 1e-12).long()
    return torch.clamp(month, 1, MONTHS)


def fractional_censor_nll_torch(
    logits: torch.Tensor,
    time_days: torch.Tensor,
    cause: torch.Tensor,
) -> torch.Tensor:
    """Vectorized torch version of FRACTIONAL_CENSOR_NLL_V1."""
    if logits.ndim != 3 or tuple(logits.shape[1:]) != (MONTHS, CAUSES):
        raise ValueError(f"Expected logits [N,{MONTHS},{CAUSES}], got {tuple(logits.shape)}")
    lp = torch.log_softmax(logits.float(), dim=-1)
    n = logits.shape[0]
    out = torch.zeros(n, dtype=lp.dtype, device=lp.device)
    bins = torch_event_bin(time_days)

    event = cause != 0
    if event.any():
        rows = torch.where(event)[0]
        event_bins = bins[rows]
        for m in range(1, MONTHS + 1):
            prior = event_bins > m
            if prior.any():
                rr = rows[prior]
                out[rr] = out[rr] - lp[rr, m - 1, 0]
        month0 = event_bins - 1
        out[rows] = out[rows] - lp[rows, month0, cause[rows].long()]

    censor = ~event
    if censor.any():
        rows = torch.where(censor)[0]
        ratio = torch.clamp(time_days[rows], 0.0, ADMIN_HORIZON_DAYS) / MONTH_DAYS
        completed = torch.floor(ratio + 1e-12).long().clamp(0, MONTHS)
        fraction = ratio - completed.float()
        fraction = torch.where(fraction < 1e-10, torch.zeros_like(fraction), fraction)
        for m in range(1, MONTHS + 1):
            full = completed >= m
            if full.any():
                rr = rows[full]
                out[rr] = out[rr] - lp[rr, m - 1, 0]
        interior = (completed < MONTHS) & (fraction > 0)
        if interior.any():
            rr = rows[interior]
            cc = completed[interior]
            out[rr] = out[rr] - fraction[interior] * lp[rr, cc, 0]
    return out


def weighted_patient_objective(row_loss: torch.Tensor, patient_weights: torch.Tensor) -> torch.Tensor:
    w = patient_weights.float()
    denom = torch.clamp(w.sum(), min=1e-12)
    return (row_loss.float() * w).sum() / denom


class LinearDiscreteHazard(nn.Module):
    """Regularized linear 24-interval multinomial hazard baseline."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.head = nn.Linear(self.feature_dim, MONTHS * CAUSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x).reshape(-1, MONTHS, CAUSES)


class PairedStructuredSurvivalModel(nn.Module):
    """Shared PRE/POST survival head over causal paired structured evidence."""

    def __init__(self, *, include_optional: bool, include_genomics: bool, dims: dict[str, int]):
        super().__init__()
        self.include_optional = bool(include_optional)
        self.include_genomics = bool(include_genomics)
        self.scan_encoder = PairedScanEncoder(
            history_dim=dims["history_core"],
            utilization_dim=dims["utilization_core"],
            optional_history_dim=dims["optional_history"],
            optional_history_recency_dim=dims["optional_history_recency"],
            current_dim=dims["current_core"],
            change_dim=dims["change_core"],
            optional_current_dim=dims["optional_current"],
            stream_dim=dims["stream_availability"],
            hidden_dim=48,
            output_dim=32,
        )
        extra = 0
        if self.include_genomics:
            self.tumor = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 64), nn.GELU())
            self.genomic_meta = nn.Sequential(nn.Linear(2, 16), nn.GELU())
            extra = 80
        else:
            self.tumor = None
            self.genomic_meta = None
        self.shared_fusion = nn.Sequential(
            nn.Linear(32 + extra, 128),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.LayerNorm(128),
        )
        self.survival_head = nn.Linear(128, MONTHS * CAUSES)

    def _scan(self, b: StructuredBatch):
        kwargs: dict[str, Any] = dict(
            history_values=b.history_core,
            history_masks=b.history_core_mask,
            utilization_values=b.utilization_core,
            utilization_masks=b.utilization_core_mask,
            current_values=b.current_core,
            current_masks=b.current_core_mask,
            change_values=b.change_core,
            change_masks=b.change_core_mask,
        )
        if self.include_optional:
            kwargs.update(
                optional_history_values=b.optional_history,
                optional_history_masks=b.optional_history_mask,
                optional_history_recency_values=b.optional_history_recency,
                optional_history_recency_masks=b.optional_history_recency_mask,
                optional_current_values=b.optional_current,
                optional_current_masks=b.optional_current_mask,
                stream_values=b.stream_availability,
                stream_masks=b.stream_availability_mask,
            )
        return self.scan_encoder(**kwargs)

    def _extras(self, b: StructuredBatch) -> torch.Tensor | None:
        if not self.include_genomics:
            return None
        masked_tumor = b.tumor * b.genomic_available[:, None]
        t = self.tumor(masked_tumor)
        meta = self.genomic_meta(torch.stack([b.genomic_available, b.genomic_age_scaled], dim=-1))
        return torch.cat([t, meta], dim=-1)

    def _logits(self, state: torch.Tensor, extras: torch.Tensor | None) -> torch.Tensor:
        x = state if extras is None else torch.cat([state, extras], dim=-1)
        h = self.shared_fusion(x)
        return self.survival_head(h).reshape(-1, MONTHS, CAUSES)

    def forward(self, b: StructuredBatch) -> tuple[torch.Tensor, torch.Tensor]:
        scan = self._scan(b)
        extras = self._extras(b)
        return self._logits(scan.history_context, extras), self._logits(scan.post_evidence, extras)


class TemporalFixedSurvivalModel(nn.Module):
    """Small supervised head over frozen repaired R1 PRE/POST temporal states."""

    def __init__(self):
        super().__init__()
        self.temporal = nn.Sequential(nn.LayerNorm(192), nn.Linear(192, 128), nn.GELU())
        self.tumor = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 64), nn.GELU())
        self.context = nn.Sequential(nn.Linear(3, 24), nn.GELU())
        self.fusion = nn.Sequential(
            nn.Linear(128 + 64 + 24, 160), nn.GELU(), nn.Dropout(0.10), nn.LayerNorm(160)
        )
        self.head = nn.Linear(160, MONTHS * CAUSES)

    def one(self, temporal: torch.Tensor, tumor: torch.Tensor, portable_context: torch.Tensor) -> torch.Tensor:
        genomic_available = portable_context[:, 1:2]
        tumor = tumor * genomic_available
        h = torch.cat([self.temporal(temporal), self.tumor(tumor), self.context(portable_context)], dim=-1)
        return self.head(self.fusion(h)).reshape(-1, MONTHS, CAUSES)

    def forward(
        self,
        temporal_pre: torch.Tensor,
        temporal_post: torch.Tensor,
        tumor: torch.Tensor,
        portable_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.one(temporal_pre, tumor, portable_context), self.one(temporal_post, tumor, portable_context)


def profile_probabilities(config: dict[str, Any]) -> np.ndarray:
    values = np.asarray([float(config[name]) for name in PROFILE_NAMES], dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any() or values.sum() <= 0:
        raise ValueError("invalid observation profile probabilities")
    return values / values.sum()
