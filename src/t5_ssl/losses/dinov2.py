"""DINOv2 image- and masked-patch-level self-distillation with KoLeo.

Follows the DINOv2 objective (Oquab et al., 2023):

- **DINO** (image level): every student view's CLS scores predict the teacher
  distribution of each *other* global view.
- **iBOT** (patch level): the student's masked patch tokens predict the
  teacher's distribution for the same, unmasked, patches. Each masked image
  weighs the same whatever its number of masked patches, and the sum is
  divided by the number of global images (unmasked ones contribute zero).
- **KoLeo** on the student *backbone* CLS tokens of each global view,
  L2-normalized, so it spreads the representation the probes read.
- Teacher targets are balanced with Sinkhorn-Knopp over the batch (DINOv2),
  or with DINO's softmax centering (``centering="softmax"``).

It also reports the collapse monitors of the DINO paper: the teacher entropy
and the KL divergence from teacher to student. A teacher entropy stuck at
``ln(K)`` means uniform collapse; a near-zero entropy with few distinct
argmax prototypes means one-prototype collapse.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


@torch.no_grad()
def sinkhorn_knopp(logits: torch.Tensor, temperature: float, iterations: int = 3) -> torch.Tensor:
    """Teacher probabilities ``(N, K)`` with equal prototype mass over the ``N`` rows.

    Each prototype's scores are shifted by their maximum over the batch before
    the exponential (the first row normalization cancels any per-prototype
    factor) and the iterations run in float64, so low temperatures cannot
    underflow a whole column to zero.
    """
    logits = logits.double()
    q = torch.exp((logits - logits.max(dim=0, keepdim=True).values) / temperature).t()  # (K, N)
    n_prototypes, n_samples = q.shape
    q /= q.sum()
    for _ in range(iterations):
        q /= q.sum(dim=1, keepdim=True)
        q /= n_prototypes
        q /= q.sum(dim=0, keepdim=True)
        q /= n_samples
    return (q * n_samples).t().float()


def koleo(features: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Kozachenko-Leonenko spread term of DINOv2 on L2-normalized features."""
    if features.shape[0] < 2:
        return features.sum() * 0
    x = F.normalize(features.float(), dim=-1, eps=eps)
    with torch.no_grad():
        dots = x @ x.t()
        dots.fill_diagonal_(-2.0)
        nearest = dots.argmax(dim=1)
    distances = F.pairwise_distance(x, x[nearest], eps=eps)
    return -torch.log(distances + eps).mean()


class DINOv2Loss(nn.Module):
    """Combine crossed CLS distillation, iBOT patch prediction and KoLeo spread."""

    def __init__(
        self, out_dim: int, student_temp: float = 0.1, teacher_temp: float = 0.04,
        center_momentum: float = 0.9, ibot_weight: float = 1.0, koleo_weight: float = 0.1,
        centering: str = "sinkhorn_knopp", sinkhorn_iterations: int = 3,
        patch_out_dim: int | None = None,
    ) -> None:
        super().__init__()
        if student_temp <= 0 or teacher_temp <= 0:
            raise ValueError("Distillation temperatures must be positive.")
        if centering not in ("sinkhorn_knopp", "softmax"):
            raise ValueError("centering must be 'sinkhorn_knopp' or 'softmax'.")
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.ibot_weight = ibot_weight
        self.koleo_weight = koleo_weight
        self.centering = centering
        self.sinkhorn_iterations = sinkhorn_iterations
        self.out_dim = out_dim
        self.register_buffer("cls_center", torch.zeros(1, out_dim))
        self.register_buffer("patch_center", torch.zeros(1, patch_out_dim or out_dim))

    @torch.no_grad()
    def _teacher_probs(self, logits: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
        if self.centering == "sinkhorn_knopp":
            return sinkhorn_knopp(logits, self.teacher_temp, self.sinkhorn_iterations)
        return F.softmax((logits.float() - center) / self.teacher_temp, dim=-1)

    def _cross_entropy(self, student: torch.Tensor, teacher_probs: torch.Tensor) -> torch.Tensor:
        """Per-row cross-entropy of the student softmax against teacher probabilities."""
        return -(teacher_probs * F.log_softmax(student.float() / self.student_temp, dim=-1)).sum(-1)

    def forward(
        self, student_cls: list[torch.Tensor], teacher_cls: list[torch.Tensor],
        student_patches: list[torch.Tensor], teacher_patches: list[torch.Tensor],
        patch_weights: list[torch.Tensor], backbone_cls: list[torch.Tensor], *,
        update_center: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the DINOv2 objective.

        Parameters
        ----------
        student_cls:
            Head scores ``(B, K)`` of every student view: the two globals first,
            then the local crops.
        teacher_cls:
            Head scores ``(B, K)`` of the teacher for the two global views.
        student_patches, teacher_patches:
            Patch-head scores ``(M_v, K)`` of the masked patches of each global
            view (student from the masked input, teacher from the full one).
        patch_weights:
            ``(M_v,)`` weight of each masked patch: ``1 / masked patches of its image``.
        backbone_cls:
            Student backbone CLS tokens ``(B, D)`` of the two global views (KoLeo).
        update_center:
            Move the softmax-centering EMA centers (training only; unused by
            Sinkhorn-Knopp).

        """
        n_global = len(teacher_cls)
        batch = teacher_cls[0].shape[0]
        teacher_logits = torch.cat(teacher_cls)
        with torch.no_grad():
            teacher_probs = self._teacher_probs(teacher_logits, self.cls_center).split(batch)
        cls_terms = [self._cross_entropy(student_cls[j], teacher_probs[i]).mean()
                     for i in range(n_global) for j in range(len(student_cls)) if j != i]
        dino = torch.stack(cls_terms).mean()

        student_patch = torch.cat(student_patches)
        teacher_patch = torch.cat(teacher_patches)
        if student_patch.shape[0] > 0:
            with torch.no_grad():
                patch_probs = self._teacher_probs(teacher_patch, self.patch_center)
            per_patch = self._cross_entropy(student_patch, patch_probs)
            ibot = (per_patch * torch.cat(patch_weights)).sum() / (n_global * batch)
        else:
            ibot = dino * 0

        spread = torch.stack([koleo(x) for x in backbone_cls]).sum()
        total = dino + self.ibot_weight * ibot + self.koleo_weight * spread

        if update_center and self.centering == "softmax":
            with torch.no_grad():
                self.cls_center.lerp_(teacher_logits.float().mean(0, keepdim=True),
                                      1 - self.center_momentum)
                if teacher_patch.shape[0] > 0:
                    self.patch_center.lerp_(teacher_patch.float().mean(0, keepdim=True),
                                            1 - self.center_momentum)

        with torch.no_grad():
            probs = torch.cat(teacher_probs)
            entropy = -(probs * probs.clamp_min(1e-12).log()).sum(-1).mean()
            unique = probs.argmax(-1).unique().numel() / probs.shape[0]
        metrics = {
            "dino": dino.detach(), "ibot": ibot.detach(), "koleo": spread.detach(),
            # Collapse monitors (DINO paper, Fig. 7): entropy ~ ln(K) is uniform collapse.
            "teacher_entropy": entropy, "dino_kl": dino.detach() - entropy,
            "teacher_entropy_frac": entropy / math.log(self.out_dim),
            "teacher_unique_frac": torch.tensor(unique, device=dino.device),
        }
        return total, metrics
