"""VICReg training with an optional feature queue for small-batch covariance."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from t5_ssl.losses.vicreg import VICRegLoss
from t5_ssl.models.ssl_module import SSLLightningModule


class VICRegLightningModule(SSLLightningModule):
    """Two-view VICReg, optionally backed by a FIFO bank of past projections.

    VICReg's variance/covariance terms are estimated over the batch dimension,
    so their rank is capped at ``batch_size - 1`` regardless of the projector
    width. ``queue_size > 0`` concatenates a detached bank of recently seen
    projected embeddings before computing those two terms only; the invariance
    (MSE) term always uses the current-batch positive pairs.
    """

    def __init__(
        self, *, backbone: nn.Module, projector: nn.Module, loss_fn: VICRegLoss,
        queue_size: int = 0, **kwargs: Any,
    ) -> None:
        super().__init__(backbone=backbone, projector=projector, loss_fn=loss_fn, **kwargs)
        if queue_size < 0:
            raise ValueError("queue_size must be nonnegative")
        self.queue_size = queue_size
        if queue_size > 0:
            self.register_buffer("queue", torch.zeros(queue_size, projector.out_dim))
            self.register_buffer("queue_filled", torch.zeros(1, dtype=torch.long))
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def _queue_sample(self) -> torch.Tensor | None:
        if self.queue_size == 0 or int(self.queue_filled.item()) == 0:
            return None
        return self.queue[: int(self.queue_filled.item())].clone()

    @torch.no_grad()
    def _enqueue(self, z1: torch.Tensor, z2: torch.Tensor) -> None:
        if self.queue_size == 0:
            return
        features = torch.cat([z1.detach(), z2.detach()]).to(self.queue.dtype)
        n = features.shape[0]
        ptr = int(self.queue_ptr.item())
        if n >= self.queue_size:
            self.queue.copy_(features[-self.queue_size:])
            ptr = 0
        else:
            end = ptr + n
            if end <= self.queue_size:
                self.queue[ptr:end] = features
            else:
                first = self.queue_size - ptr
                self.queue[ptr:] = features[:first]
                self.queue[: end - self.queue_size] = features[first:]
            ptr = end % self.queue_size
        self.queue_ptr[0] = ptr
        self.queue_filled[0] = min(self.queue_size, int(self.queue_filled.item()) + n)

    def _step(self, batch: dict[str, Any], stage: str) -> torch.Tensor:
        images = batch["image"]
        if isinstance(images, (list, tuple)) and len(images) >= 2:
            x1, x2 = images[0], images[1]
        else:
            x1 = x2 = images
        h1 = self.backbone(x1)
        h2 = self.backbone(x2)
        z1 = self.projector(h1)
        z2 = self.projector(h2)
        queue = self._queue_sample() if stage == "train" else None
        total_loss, metrics = self.loss_fn(z1, z2, h1, h2, queue=queue)
        self.log(f"{stage}/loss", total_loss, on_step=stage == "train", on_epoch=True,
                 prog_bar=True, sync_dist=True)
        self.log(f"{stage}/queue_fill", float(self.queue_filled.item()) if self.queue_size else 0.0,
                 on_step=False, on_epoch=True, sync_dist=True)
        for name, value in metrics.items():
            self.log(f"{stage}/{name}", value, on_step=False, on_epoch=True, sync_dist=True)
        if stage == "train":
            self._enqueue(z1, z2)
        return total_loss

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Update online parameters, then push this batch's projections into the queue."""
        return self._step(batch, "train")

    def validation_step(self, batch: dict[str, Any], batch_idx: int,
                        dataloader_idx: int = 0) -> torch.Tensor:
        """Score VICReg on the current batch alone, without touching the queue."""
        return self._step(batch, "val")
