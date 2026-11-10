"""Concept-level removal shared by every representation of a measurement."""

from dataclasses import dataclass
from typing import Mapping, Optional

import torch
from torch import Tensor


@dataclass(frozen=True)
class ConceptMap:
    """Map summary and trajectory columns to the same clinical concepts.

    ``missing_summary`` must be the source-fitted, transformed representation
    of a missing summary coordinate (not necessarily zero).
    """

    summary_to_concept: Tensor  # [L]
    trajectory_to_concept: Tensor  # [J]
    missing_summary: Tensor  # [L]
    num_concepts: int

    def __post_init__(self) -> None:
        if self.num_concepts < 1:
            raise ValueError("num_concepts must be positive")
        for name, mapping in (
            ("summary_to_concept", self.summary_to_concept),
            ("trajectory_to_concept", self.trajectory_to_concept),
        ):
            if mapping.ndim != 1 or mapping.dtype != torch.long:
                raise ValueError(f"{name} must be a 1-D LongTensor")
            if mapping.numel() and (mapping.min() < 0 or mapping.max() >= self.num_concepts):
                raise ValueError(f"{name} contains an invalid concept ID")
        if self.missing_summary.ndim != 1 or self.missing_summary.shape != self.summary_to_concept.shape:
            raise ValueError("missing_summary must have one value per summary column")
        if not torch.isfinite(self.missing_summary).all():
            raise ValueError("missing_summary must be finite")


def sample_concept_mask(
    batch_size: int,
    num_concepts: int,
    rho: float,
    *,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Draw independent Bernoulli removal decisions for each patient/concept."""
    if batch_size < 1 or num_concepts < 1 or not 0.0 <= rho <= 1.0:
        raise ValueError("invalid batch size, concept count, or removal rate")
    return torch.rand((batch_size, num_concepts), device=device, generator=generator) < rho


def remove_concepts(inputs: Mapping[str, Tensor], drop: Tensor, concept_map: ConceptMap) -> dict[str, Tensor]:
    """Construct a coherent reduced view without mutating the original view.

    Required keys: ``summary`` [B,L], ``clock`` [B,Q,J], ``observed`` [B,J].
    Optional ``record``, ``clock_time`` and ``record_time`` are zeroed too.
    """
    summary, clock, observed = (inputs[key] for key in ("summary", "clock", "observed"))
    if summary.ndim != 2 or clock.ndim != 3 or observed.ndim != 2:
        raise ValueError("summary, clock, and observed must have ranks 2, 3, and 2")
    batch_size, num_summary = summary.shape
    if clock.shape[0] != batch_size or observed.shape != (batch_size, clock.shape[2]):
        raise ValueError("inconsistent batch or trajectory dimensions")
    if num_summary != len(concept_map.summary_to_concept) or clock.shape[2] != len(concept_map.trajectory_to_concept):
        raise ValueError("input dimensions do not match ConceptMap")
    if drop.dtype != torch.bool or drop.shape != (batch_size, concept_map.num_concepts) or drop.device != summary.device:
        raise ValueError("drop must be a Boolean [B,C] mask on the input device")

    result = {key: value.clone() for key, value in inputs.items()}
    summary_ids = concept_map.summary_to_concept.to(summary.device)
    field_ids = concept_map.trajectory_to_concept.to(summary.device)
    missing = concept_map.missing_summary.to(device=summary.device, dtype=summary.dtype)
    result["summary"] = torch.where(drop[:, summary_ids], missing[None, :], summary)
    field_drop = drop[:, field_ids]
    result["observed"] = observed.masked_fill(field_drop, 0.0)
    for key in ("clock", "record", "clock_time", "record_time"):
        if key in result:
            result[key] = result[key].masked_fill(field_drop[:, None, :], 0.0)
    return result
