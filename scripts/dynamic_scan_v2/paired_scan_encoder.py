from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class EncoderOutput:
    history_context: torch.Tensor
    current_evidence: torch.Tensor
    change_evidence: torch.Tensor
    post_evidence: torch.Tensor


class MaskedBlockEncoder(nn.Module):
    """Small MLP block that receives values and masks explicitly."""

    def __init__(self, value_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.value_dim = int(value_dim)
        self.net = nn.Sequential(
            nn.Linear(self.value_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, values: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        if values.shape != masks.shape:
            raise ValueError(f"values/masks shape mismatch: {values.shape} vs {masks.shape}")
        masked = values * masks
        return self.net(torch.cat([masked, masks], dim=-1))


class PairedScanEncoder(nn.Module):
    """Compact paired-scan encoder with an explicit PRE/POST boundary.

    PRE-visible blocks are history_core, utilization_core, optional_history, and
    optional_history_recency. POST adds current_core, change_core,
    optional_current, and current stream availability. Thus adding or perturbing
    current-episode content cannot alter the history_context branch.
    """

    def __init__(
        self,
        *,
        history_dim: int,
        utilization_dim: int,
        optional_history_dim: int,
        optional_history_recency_dim: int,
        current_dim: int,
        change_dim: int,
        optional_current_dim: int,
        stream_dim: int,
        hidden_dim: int = 48,
        output_dim: int = 32,
    ):
        super().__init__()
        self.config = {
            "history_dim": int(history_dim),
            "utilization_dim": int(utilization_dim),
            "optional_history_dim": int(optional_history_dim),
            "optional_history_recency_dim": int(optional_history_recency_dim),
            "current_dim": int(current_dim),
            "change_dim": int(change_dim),
            "optional_current_dim": int(optional_current_dim),
            "stream_dim": int(stream_dim),
            "hidden_dim": int(hidden_dim),
            "output_dim": int(output_dim),
        }

        self.history = MaskedBlockEncoder(history_dim, hidden_dim, output_dim)
        self.utilization = MaskedBlockEncoder(utilization_dim, hidden_dim, output_dim)
        self.optional_history = MaskedBlockEncoder(optional_history_dim, hidden_dim, output_dim)
        self.optional_history_recency = MaskedBlockEncoder(optional_history_recency_dim, hidden_dim, output_dim)
        self.current = MaskedBlockEncoder(current_dim, hidden_dim, output_dim)
        self.change = MaskedBlockEncoder(change_dim, hidden_dim, output_dim)
        self.optional_current = MaskedBlockEncoder(optional_current_dim, hidden_dim, output_dim)
        self.stream = MaskedBlockEncoder(stream_dim, hidden_dim, output_dim)

        self.history_fusion = nn.Sequential(
            nn.Linear(output_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.current_fusion = nn.Sequential(
            nn.Linear(output_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.change_fusion = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.post_fusion = nn.Sequential(
            nn.Linear(output_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        *,
        history_values: torch.Tensor,
        history_masks: torch.Tensor,
        utilization_values: torch.Tensor,
        utilization_masks: torch.Tensor,
        optional_history_values: torch.Tensor | None = None,
        optional_history_masks: torch.Tensor | None = None,
        optional_history_recency_values: torch.Tensor | None = None,
        optional_history_recency_masks: torch.Tensor | None = None,
        current_values: torch.Tensor | None = None,
        current_masks: torch.Tensor | None = None,
        change_values: torch.Tensor | None = None,
        change_masks: torch.Tensor | None = None,
        optional_current_values: torch.Tensor | None = None,
        optional_current_masks: torch.Tensor | None = None,
        stream_values: torch.Tensor | None = None,
        stream_masks: torch.Tensor | None = None,
    ) -> EncoderOutput:
        batch = history_values.shape[0]
        device = history_values.device
        dtype = history_values.dtype

        def absent(dim: int) -> tuple[torch.Tensor, torch.Tensor]:
            shape = (batch, dim)
            return (
                torch.zeros(shape, device=device, dtype=dtype),
                torch.zeros(shape, device=device, dtype=dtype),
            )

        if optional_history_values is None or optional_history_masks is None:
            optional_history_values, optional_history_masks = absent(self.config["optional_history_dim"])
        if optional_history_recency_values is None or optional_history_recency_masks is None:
            optional_history_recency_values, optional_history_recency_masks = absent(self.config["optional_history_recency_dim"])
        if current_values is None or current_masks is None:
            current_values, current_masks = absent(self.config["current_dim"])
        if change_values is None or change_masks is None:
            change_values, change_masks = absent(self.config["change_dim"])
        if optional_current_values is None or optional_current_masks is None:
            optional_current_values, optional_current_masks = absent(self.config["optional_current_dim"])
        if stream_values is None or stream_masks is None:
            stream_values, stream_masks = absent(self.config["stream_dim"])

        h = self.history(history_values, history_masks)
        u = self.utilization(utilization_values, utilization_masks)
        oh = self.optional_history(optional_history_values, optional_history_masks)
        ohr = self.optional_history_recency(optional_history_recency_values, optional_history_recency_masks)
        history_context = self.history_fusion(torch.cat([h, u, oh, ohr], dim=-1))

        c = self.current(current_values, current_masks)
        oc = self.optional_current(optional_current_values, optional_current_masks)
        s = self.stream(stream_values, stream_masks)
        current_evidence = self.current_fusion(torch.cat([c, oc, s], dim=-1))

        d = self.change(change_values, change_masks)
        change_evidence = self.change_fusion(torch.cat([d, current_evidence], dim=-1))
        post_evidence = self.post_fusion(torch.cat([history_context, current_evidence, change_evidence], dim=-1))

        return EncoderOutput(
            history_context=history_context,
            current_evidence=current_evidence,
            change_evidence=change_evidence,
            post_evidence=post_evidence,
        )

    def interface(self) -> dict[str, Any]:
        return {
            "name": "PairedScanEncoder",
            "version": "v2_01_1",
            "config": dict(self.config),
            "pre_contract": "history_context consumes only prior-episode core/utilization/optional-history blocks",
            "post_contract": "POST adds current_core + change_core + optional_current + current stream availability",
            "training_status": "ARCHITECTURE_ONLY_IN_V2_01; fold-specific supervised training begins in V2_02",
        }
