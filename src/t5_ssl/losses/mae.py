"""SimMIM-style masked patch reconstruction loss.

Reference:
    Xie, Z. et al. (2022). SimMIM: A Simple Framework for Masked Image Modeling.
    CVPR 2022.

This follows the SimMIM contract (full-length input, masked patches replaced by
a learned mask token, single encoder pass) rather than MAE's asymmetric
encoder/decoder, so it reuses the existing :class:`ViTBackbone` masking path
shared with the DINOv2 module instead of adding a second transformer decoder.
"""

from __future__ import annotations

import torch
from torch import nn


class SimMIMLoss(nn.Module):
    """Mean squared pixel-reconstruction error restricted to masked patches."""

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute masked and (diagnostic) visible reconstruction error.

        Parameters
        ----------
        pred, target:
            Predicted and raw-pixel patch tensors, shape ``(B, N, patch_dim)``.
        mask:
            Boolean tensor ``(B, N)``; ``True`` marks a masked (predicted) patch.

        """
        error = (pred - target) ** 2
        per_patch = error.mean(dim=-1)
        mask = mask.to(per_patch.dtype)
        masked_loss = (per_patch * mask).sum() / mask.sum().clamp_min(1.0)
        with torch.no_grad():
            visible = 1.0 - mask
            visible_loss = (per_patch * visible).sum() / visible.sum().clamp_min(1.0)
        return masked_loss, {"masked_mse": masked_loss.detach(), "visible_mse": visible_loss}
