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
    """Train the online encoder/projector/predictor against an EMA target branch.

    ``predictor_lr_mult`` scales the predictor's learning rate (its own AdamW
    group, same warmup/cosine schedule). Analyses of BYOL/SimSiam dynamics
    (Tian et al. 2021, DirectPred; SimSiam's fixed predictor LR) find that a
    predictor adapting faster than the encoder is what keeps the online network
    from collapsing; ``1.0`` keeps the single-group optimizer.
    """

    def __init__(
        self, *, backbone: nn.Module, projector: nn.Module, predictor: nn.Module,
        loss_fn: BYOLLoss, ema_base: float = 0.996, predictor_lr_mult: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(backbone=backbone, projector=projector, loss_fn=loss_fn, **kwargs)
        if not 0 <= ema_base < 1:
            raise ValueError("ema_base must be in [0, 1).")
        if predictor_lr_mult <= 0:
            raise ValueError("predictor_lr_mult must be positive.")
        self.predictor_lr_mult = float(predictor_lr_mult)
        self.predictor = predictor
        self.target_backbone = copy.deepcopy(backbone)
        self.target_projector = copy.deepcopy(projector)
        self.target_backbone.requires_grad_(False)
        self.target_projector.requires_grad_(False)
        # Projector dropout regularizes the online branch only: the EMA target
        # gives deterministic regression targets (as DINOv2's teacher drops no paths).
        for module in self.target_projector.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
        self.ema_base = ema_base

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Include the online predictor and exclude the EMA target."""
        return super().trainable_parameters() + list(self.predictor.parameters())

    def optimizer_param_groups(self) -> list[dict[str, Any]]:
        """One group, or encoder+projector and predictor at ``predictor_lr_mult`` x LR."""
        if self.predictor_lr_mult == 1.0:
            return super().optimizer_param_groups()
        online = list(self.backbone.parameters()) + list(self.projector.parameters())
        return [{"params": online, "weight_decay": self.weight_decay},
                {"params": list(self.predictor.parameters()), "weight_decay": self.weight_decay,
                 "lr": self.lr * self.predictor_lr_mult}]

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
        if not self._optimizer_stepped():  # gradient accumulation: no step, no EMA
            return
        total_steps = max(1, self.trainer.estimated_stepping_batches)
        progress = min(1.0, self.global_step / total_steps)
        momentum = 1 - (1 - self.ema_base) * (math.cos(math.pi * progress) + 1) / 2
        self._ema_update([(self.backbone, self.target_backbone),
                          (self.projector, self.target_projector)], momentum)
