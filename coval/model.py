"""The active COVAL prediction path: numerical embeddings and a TabM ensemble."""

from typing import Mapping, Sequence

import torch
from torch import Tensor, nn
import rtdl_num_embeddings as rtdl
import tabm


class COVAL(nn.Module):
    """Concept-aware summary/trajectory encoder with an official TabM head.

    The trajectory branch averages embeddings of clock-sampled values; it is
    order-insensitive and does not add a sequence-ordering module. Defaults
    match the study configuration. Smaller settings are useful for smoke tests.
    """

    def __init__(
        self,
        summary_bins: Sequence[Tensor],
        trajectory_bins: Sequence[Tensor],
        *,
        embedding_dim: int = 16,
        k: int = 32,
        n_blocks: int = 2,
        d_block: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if not summary_bins or not trajectory_bins:
            raise ValueError("both branches need at least one numeric feature")
        self.d = embedding_dim
        self.n_fields = len(trajectory_bins)
        self.summary_embedding = rtdl.PiecewiseLinearEmbeddings(
            summary_bins, self.d, activation=False, version="B"
        )
        self.field_embedding = rtdl.PiecewiseLinearEmbeddings(
            trajectory_bins, self.d, activation=False, version="B"
        )
        weight = torch.zeros(self.n_fields, 4 * self.d, self.d)
        weight[:, : self.d, :] = torch.eye(self.d)[None]
        self.temporal_projection = nn.Parameter(weight)
        chunks = [self.d] * (len(summary_bins) + self.n_fields)
        self.ensemble = tabm.EnsembleView(k=k)
        self.backbone = tabm.make_tabm_backbone(
            d_in=sum(chunks),
            start_scaling_init="normal",
            start_scaling_init_chunks=chunks,
            k=k,
            n_blocks=n_blocks,
            d_block=d_block,
            dropout=dropout,
            arch_type="tabm",
        )
        self.head = tabm.LinearEnsemble(d_block, 1, k=k)

    def forward(self, inputs: Mapping[str, Tensor]) -> Tensor:
        summary, clock, observed = (inputs[key] for key in ("summary", "clock", "observed"))
        if summary.ndim != 2 or clock.ndim != 3 or observed.shape != (clock.shape[0], self.n_fields):
            raise ValueError("expected summary [B,L], clock [B,Q,J], observed [B,J]")
        if summary.shape[0] != clock.shape[0] or clock.shape[2] != self.n_fields:
            raise ValueError("inconsistent batch or trajectory dimensions")

        embedded_summary = self.summary_embedding(summary).flatten(1)
        batch_size, points, fields = clock.shape
        embedded_clock = self.field_embedding(clock.reshape(batch_size * points, fields))
        embedded_clock = embedded_clock.reshape(batch_size, points, fields, self.d).mean(dim=1)
        # The study's clock-static arm uses basis [1, 0, 0, 0]. Keeping the
        # four-block parameter shape preserves compatibility with that path.
        moments = torch.cat((embedded_clock, *[torch.zeros_like(embedded_clock) for _ in range(3)]), dim=-1)
        field = torch.einsum("bfi,fij->bfj", moments, self.temporal_projection)
        field = (field * observed[:, :, None]).flatten(1)
        representation = torch.cat((embedded_summary, field), dim=1)
        return self.head(self.backbone(self.ensemble(representation)))  # [B, k, 1]
