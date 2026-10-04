from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from paired_scan_encoder import PairedScanEncoder
from v2_02_models import StructuredBatch, MONTHS, CAUSES


REPRESENTATION_MODES = (
    "temporal_post_only",
    "pre_temporal_plus_explicit_scan",
    "post_temporal_plus_explicit_scan",
    "temporal_latent_adapter",
)
GENOMIC_MODES = ("embedding", "simple_summary", "availability_age", "none")
POSTPROG_CLASSES = {
    "NEXT_TREATMENT": 0,
    "CONTINUED_OR_CENSORED": 1,
    "DEATH": 2,
}


class LowRankResidualAdapter(nn.Module):
    def __init__(self, dim: int = 192, bottleneck: int = 32):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck)
        self.up = nn.Linear(bottleneck, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(F.gelu(self.down(self.norm(x))))


class GenomicProjector(nn.Module):
    """64D content projector with separately controlled availability/age context."""

    def __init__(self, mode: str):
        super().__init__()
        if mode not in GENOMIC_MODES:
            raise ValueError(f"Unknown genomic mode {mode}")
        self.mode = mode
        self.embedding = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 64), nn.GELU())
        self.simple = nn.Sequential(nn.LayerNorm(3), nn.Linear(3, 64), nn.GELU())

    def forward(
        self,
        tumor: torch.Tensor,
        genomic_available: torch.Tensor,
        simple_summary: torch.Tensor,
    ) -> torch.Tensor:
        avail = genomic_available.reshape(-1, 1)
        if self.mode == "embedding":
            return self.embedding(tumor * avail)
        if self.mode == "simple_summary":
            return self.simple(simple_summary * avail)
        return torch.zeros((len(tumor), 64), dtype=tumor.dtype, device=tumor.device)

    def context(self, portable_context: torch.Tensor) -> torch.Tensor:
        """Preserve scan-number context while explicitly controlling genomic availability/age."""
        ctx = portable_context.clone()
        if self.mode == "none":
            ctx[:, 1:] = 0.0
        # availability_age and simple_summary intentionally preserve columns 1:3.
        return ctx


@dataclass
class HybridOutput:
    pre_logits: torch.Tensor
    post_logits: torch.Tensor
    pre_hidden: torch.Tensor
    post_hidden: torch.Tensor
    next_scan_logits: torch.Tensor
    postprog_logits: torch.Tensor
    postprog_time_logits: torch.Tensor


class HybridTemporalScanModel(nn.Module):
    """V2-04 representation model with a V2-02-compatible temporal-only control.

    When representation_mode=temporal_post_only and genomic_mode=embedding, the
    supervised path has the same layer dimensions as V2-02 TemporalFixedSurvivalModel:
    temporal 192->128, tumor 128->64, portable context 3->24, fusion 216->160.
    Explicit current-scan evidence enters as a residual in the 128D temporal state,
    so adding it does not change fusion dimensionality.
    """

    def __init__(self, *, representation_mode: str, genomic_mode: str, dims: dict[str, int], adapter_bottleneck: int = 32):
        super().__init__()
        if representation_mode not in REPRESENTATION_MODES:
            raise ValueError(f"Unknown representation mode {representation_mode}")
        self.representation_mode = representation_mode
        self.genomic_mode = genomic_mode
        self.temporal_adapter = LowRankResidualAdapter(192, adapter_bottleneck) if representation_mode == "temporal_latent_adapter" else nn.Identity()
        self.temporal = nn.Sequential(nn.LayerNorm(192), nn.Linear(192, 128), nn.GELU())
        self.scan_encoder = PairedScanEncoder(
            history_dim=dims["history_core"], utilization_dim=dims["utilization_core"],
            optional_history_dim=dims["optional_history"], optional_history_recency_dim=dims["optional_history_recency"],
            current_dim=dims["current_core"], change_dim=dims["change_core"],
            optional_current_dim=dims["optional_current"], stream_dim=dims["stream_availability"],
            hidden_dim=48, output_dim=32,
        )
        self.scan_delta = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.LayerNorm(128))
        self.genomic = GenomicProjector(genomic_mode)
        self.context = nn.Sequential(nn.Linear(3, 24), nn.GELU())
        self.fusion = nn.Sequential(nn.Linear(128 + 64 + 24, 160), nn.GELU(), nn.Dropout(0.10), nn.LayerNorm(160))
        self.survival_head = nn.Linear(160, MONTHS * CAUSES)
        self.next_scan_head = nn.Linear(160, 3)
        self.postprog_head = nn.Linear(160, 3)
        self.postprog_time_head = nn.Linear(160, 4)

    def _scan(self, b: StructuredBatch):
        return self.scan_encoder(
            history_values=b.history_core, history_masks=b.history_core_mask,
            utilization_values=b.utilization_core, utilization_masks=b.utilization_core_mask,
            current_values=b.current_core, current_masks=b.current_core_mask,
            change_values=b.change_core, change_masks=b.change_core_mask,
            optional_history_values=b.optional_history, optional_history_masks=b.optional_history_mask,
            optional_history_recency_values=b.optional_history_recency, optional_history_recency_masks=b.optional_history_recency_mask,
            optional_current_values=b.optional_current, optional_current_masks=b.optional_current_mask,
            stream_values=b.stream_availability, stream_masks=b.stream_availability_mask,
        )

    def _hidden(self, temporal_state: torch.Tensor, genomic: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.fusion(torch.cat([temporal_state, genomic, self.context(context)], dim=-1))

    def forward(self, b: StructuredBatch, temporal_pre: torch.Tensor, temporal_post: torch.Tensor, simple_summary: torch.Tensor, portable_context: torch.Tensor) -> HybridOutput:
        scan = self._scan(b)
        direct = self.scan_delta(torch.cat([scan.current_evidence, scan.change_evidence], dim=-1))
        pre_raw = self.temporal_adapter(temporal_pre)
        post_raw = self.temporal_adapter(temporal_post)
        pre_t = self.temporal(pre_raw)
        if self.representation_mode == "temporal_post_only":
            post_t = self.temporal(post_raw)
        elif self.representation_mode == "pre_temporal_plus_explicit_scan":
            post_t = pre_t + direct
        elif self.representation_mode == "post_temporal_plus_explicit_scan":
            post_t = self.temporal(post_raw) + direct
        elif self.representation_mode == "temporal_latent_adapter":
            post_t = self.temporal(post_raw)
        else:  # pragma: no cover
            raise AssertionError(self.representation_mode)

        g = self.genomic(b.tumor, b.genomic_available, simple_summary)
        ctx = self.genomic.context(portable_context)
        pre_h = self._hidden(pre_t, g, ctx)
        post_h = self._hidden(post_t, g, ctx)
        pre_logits = self.survival_head(pre_h).reshape(-1, MONTHS, CAUSES)
        post_logits = self.survival_head(post_h).reshape(-1, MONTHS, CAUSES)
        return HybridOutput(
            pre_logits=pre_logits, post_logits=post_logits, pre_hidden=pre_h, post_hidden=post_h,
            next_scan_logits=self.next_scan_head(post_h), postprog_logits=self.postprog_head(post_h), postprog_time_logits=self.postprog_time_head(post_h),
        )


def masked_weighted_ce(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    mask = mask.bool()
    if not bool(mask.any()):
        return logits.sum() * 0.0
    loss = F.cross_entropy(logits.float()[mask], labels.long()[mask], reduction="none")
    w = weights.float()[mask]
    return (loss * w).sum() / w.sum().clamp_min(1e-8)


def choose_with_simplicity_margin(metrics: dict[str, float], simplicity_order: list[str], margin: float) -> str:
    if not metrics:
        raise ValueError("empty metrics")
    best_value = min(float(v) for v in metrics.values())
    eligible = {k for k, v in metrics.items() if float(v) <= best_value + float(margin)}
    for name in simplicity_order:
        if name in eligible:
            return name
    return min(metrics, key=metrics.get)
