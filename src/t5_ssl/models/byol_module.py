"""BYOL online and exponential-moving-average target branches."""

from __future__ import annotations

import copy
import math
from typing import Any

import torch
from torch import nn

from t5_ssl.losses.byol import BYOLLoss
from t5_ssl.models.ssl_module import SSLLightningModule


class BYOLLightningModule(SSLLightningModule):
    """Train the online encoder/projector/predictor against an EMA target branch."""

    def __init__(
        self, *, backbone: nn.Module, projector: nn.Module, predictor: nn.Module,
        loss_fn: BYOLLoss, ema_base: float = 0.996, **kwargs: Any,
    ) -> None:
        super().__init__(backbone=backbone, projector=projector, loss_fn=loss_fn, **kwargs)
        if not 0 <= ema_base < 1:
            raise ValueError("ema_base must be in [0, 1).")
        self.predictor = predictor
        self.target_backbone = copy.deepcopy(backbone)
        self.target_projector = copy.deepcopy(projector)
        self.target_backbone.requires_grad_(False)
        self.target_projector.requires_grad_(False)
        self.ema_base = ema_base

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Include the online predictor and exclude the EMA target."""
        return super().trainable_parameters() + list(self.predictor.parameters())

    def _step(self, batch: dict[str, Any], stage: str) -> torch.Tensor:
        images = batch["image"]
        if not isinstance(images, (list, tuple)) or len(images) < 2:
            raise ValueError("BYOL requires two independently augmented views.")
        x1, x2 = images[:2]
        p1 = self.predictor(self.projector(self.backbone(x1)))
        p2 = self.predictor(self.projector(self.backbone(x2)))
        with torch.no_grad():
            t1 = self.target_projector(self.target_backbone(x1))
            t2 = self.target_projector(self.target_backbone(x2))
        loss, metrics = self.loss_fn(p1, p2, t1, t2)
        self.log(f"{stage}/loss", loss, on_step=stage == "train", on_epoch=True,
                 prog_bar=True, sync_dist=True)
        for name, value in metrics.items():
            self.log(f"{stage}/{name}", value, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Predict the other view using only the online branch for gradients."""
        return self._step(batch, "train")

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Evaluate the BYOL objective without changing the target."""
        return self._step(batch, "val")

    @torch.no_grad()
    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        """Update target weights after Lightning has stepped the online optimizer."""
        total_steps = max(1, self.trainer.estimated_stepping_batches)
        progress = min(1.0, self.global_step / total_steps)
        momentum = 1 - (1 - self.ema_base) * (math.cos(math.pi * progress) + 1) / 2
        for online, target in (
            (self.backbone, self.target_backbone),
            (self.projector, self.target_projector),
        ):
            for source, destination in zip(online.parameters(), target.parameters(), strict=True):
                destination.lerp_(source, 1 - momentum)
            for source, destination in zip(online.buffers(), target.buffers(), strict=True):
                destination.copy_(source)
