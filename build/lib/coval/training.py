"""Two-view outcome supervision with detached online ensemble guidance."""

from typing import Mapping, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .augmentation import ConceptMap, remove_concepts, sample_concept_mask


def _member_bce(logits: Tensor, targets: Tensor) -> Tensor:
    if logits.ndim != 3 or logits.shape[-1] != 1 or targets.shape != (logits.shape[0],):
        raise ValueError("expected logits [B,k,1] and targets [B]")
    return F.binary_cross_entropy_with_logits(
        logits[..., 0].float(), targets[:, None].expand_as(logits[..., 0]).float()
    )


def coval_loss(original: Tensor, reduced: Tensor, targets: Tensor, *, alpha: float = 0.5) -> Tensor:
    """L(original,y) + (1-alpha)L(reduced,y) + alpha L(reduced,sg(q))."""
    if not 0.0 <= alpha <= 1.0 or original.shape != reduced.shape:
        raise ValueError("alpha must be in [0,1], and view shapes must match")
    hard_original = _member_bce(original, targets)
    hard_reduced = _member_bce(reduced, targets)
    guidance = original[..., 0].float().sigmoid().mean(dim=1).detach()
    soft_reduced = _member_bce(reduced, guidance)
    return hard_original + (1.0 - alpha) * hard_reduced + alpha * soft_reduced


def train_step(
    model: nn.Module,
    inputs: Mapping[str, Tensor],
    targets: Tensor,
    optimizer: torch.optim.Optimizer,
    concept_map: ConceptMap,
    *,
    rho: float = 0.25,
    alpha: float = 0.5,
    warmup: bool = False,
    generator: Optional[torch.Generator] = None,
) -> float:
    """One optimizer update. Use warmup=True for the first two epochs."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    if warmup:
        loss = _member_bce(model(inputs), targets)
    else:
        drop = sample_concept_mask(
            len(targets), concept_map.num_concepts, rho,
            device=inputs["summary"].device, generator=generator,
        )
        reduced = remove_concepts(inputs, drop, concept_map)
        paired = {key: torch.cat((inputs[key], reduced[key]), dim=0) for key in inputs}
        original_logits, reduced_logits = model(paired).split(len(targets), dim=0)
        loss = coval_loss(original_logits, reduced_logits, targets, alpha=alpha)
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite training loss")
    loss.backward()
    optimizer.step()
    return float(loss.detach())


@torch.inference_mode()
def predict_proba(model: nn.Module, inputs: Mapping[str, Tensor]) -> Tensor:
    """Average member probabilities, as in the study inference path."""
    model.eval()
    return model(inputs)[..., 0].float().sigmoid().mean(dim=1)
