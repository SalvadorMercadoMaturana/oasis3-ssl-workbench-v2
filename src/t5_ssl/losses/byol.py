"""Bootstrap Your Own Latent (BYOL) symmetric prediction objective."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class BYOLLoss(nn.Module):
    """Match normalized online predictions to stop-gradient target projections."""

    def forward(
        self, prediction_1: torch.Tensor, prediction_2: torch.Tensor,
        target_1: torch.Tensor, target_2: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compare opposite views in both directions; targets never receive gradients."""
        # float32 even under bf16 autocast: the cosines are the whole signal.
        p1 = F.normalize(prediction_1.float(), dim=-1)
        p2 = F.normalize(prediction_2.float(), dim=-1)
        t1 = F.normalize(target_1.detach().float(), dim=-1)
        t2 = F.normalize(target_2.detach().float(), dim=-1)
        loss_12 = 2 - 2 * (p1 * t2).sum(dim=-1).mean()
        loss_21 = 2 - 2 * (p2 * t1).sum(dim=-1).mean()
        total = (loss_12 + loss_21) / 2
        return total, {"direction_12": loss_12.detach(), "direction_21": loss_21.detach()}
