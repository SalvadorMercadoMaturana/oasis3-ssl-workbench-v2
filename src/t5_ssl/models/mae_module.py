"""SimMIM-style 2.5D masked reconstruction, alone or hybrid with VICReg."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from t5_ssl.losses.mae import SimMIMLoss
from t5_ssl.losses.vicreg import VICRegLoss
from t5_ssl.models.backbones.vit import ViTBackbone
from t5_ssl.models.ssl_module import SSLLightningModule


def extract_patches(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Return non-overlapping raw-pixel patches ``(B, N, C*patch*patch)``.

    Uses the same row-major sliding-window order as the strided ``Conv2d`` in
    :class:`ViTBackbone`, so patch ``i`` here matches patch token ``i``.
    """
    patches = F.unfold(image, kernel_size=patch_size, stride=patch_size)
    return patches.transpose(1, 2)


def random_patch_mask(image: torch.Tensor, patch_size: int, mask_ratio: float) -> torch.Tensor:
    """Sample an independent Bernoulli mask per patch per example."""
    patch_count = (image.shape[-2] // patch_size) * (image.shape[-1] // patch_size)
    return torch.rand(image.shape[0], patch_count, device=image.device) < mask_ratio


class MAELightningModule(SSLLightningModule):
    """Single-view SimMIM-style masked patch reconstruction on a ViT encoder.

    Standard MAE/SimMIM training uses one masked view per image rather than
    the two-view contract shared by VICReg/BYOL; the first view supplied by
    the configured augmentation is used and the second, if any, is ignored.
    """

    def __init__(
        self, *, backbone: ViTBackbone, projector: nn.Module, recon_head: nn.Module,
        loss_fn: SimMIMLoss, mask_ratio: float = 0.6, **kwargs: Any,
    ) -> None:
        super().__init__(backbone=backbone, projector=projector, loss_fn=loss_fn, **kwargs)
        if not 0 < mask_ratio < 1:
            raise ValueError("mask_ratio must be in (0, 1)")
        self.recon_head = recon_head
        self.mask_ratio = mask_ratio

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Optimize the encoder and reconstruction head; the projector is unused."""
        return list(self.backbone.parameters()) + list(self.recon_head.parameters())

    def _step(self, batch: dict[str, Any], stage: str) -> torch.Tensor:
        images = batch["image"]
        image = images[0] if isinstance(images, list | tuple) else images
        mask = random_patch_mask(image, self.backbone.patch_size, self.mask_ratio)
        _, patch_tokens = self.backbone.forward_tokens(image, mask=mask)
        pred = self.recon_head(patch_tokens)
        target = extract_patches(image, self.backbone.patch_size)
        loss, metrics = self.loss_fn(pred, target, mask)
        self.log(f"{stage}/loss", loss, on_step=stage == "train", on_epoch=True,
                 prog_bar=True, sync_dist=True)
        for name, value in metrics.items():
            self.log(f"{stage}/{name}", value, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Reconstruct masked patches from one augmented view."""
        return self._step(batch, "train")

    def validation_step(self, batch: dict[str, Any], batch_idx: int,
                        dataloader_idx: int = 0) -> torch.Tensor:
        """Score reconstruction on one augmented view without updating weights."""
        return self._step(batch, "val")


class MAEVICRegLightningModule(SSLLightningModule):
    """Combine per-view SimMIM reconstruction with a VICReg objective on CLS tokens.

    MAE/SimMIM keeps local anatomical detail from being ignored; VICReg pushes
    the pooled (CLS) representation toward a non-redundant, well-spread global
    code. Both share the same ViT encoder and are optimized jointly.
    """

    def __init__(
        self, *, backbone: ViTBackbone, projector: nn.Module, recon_head: nn.Module,
        mae_loss_fn: SimMIMLoss, vicreg_loss_fn: VICRegLoss,
        mask_ratio: float = 0.6, mae_weight: float = 1.0, vicreg_weight: float = 1.0,
        queue_size: int = 0, **kwargs: Any,
    ) -> None:
        super().__init__(backbone=backbone, projector=projector, loss_fn=vicreg_loss_fn, **kwargs)
        if not 0 < mask_ratio < 1:
            raise ValueError("mask_ratio must be in (0, 1)")
        if queue_size < 0:
            raise ValueError("queue_size must be nonnegative")
        self.recon_head = recon_head
        self.mae_loss_fn = mae_loss_fn
        self.mask_ratio = mask_ratio
        self.mae_weight = mae_weight
        self.vicreg_weight = vicreg_weight
        self.queue_size = queue_size
        if queue_size > 0:
            self.register_buffer("queue", torch.zeros(queue_size, projector.out_dim))
            self.register_buffer("queue_filled", torch.zeros(1, dtype=torch.long))
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Optimize the shared encoder, VICReg projector, and reconstruction head."""
        return (list(self.backbone.parameters()) + list(self.projector.parameters())
                + list(self.recon_head.parameters()))

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
        if not (isinstance(images, list | tuple) and len(images) >= 2):
            raise ValueError("MAE+VICReg requires two augmented views")
        mae_losses = []
        mae_metrics_sum: dict[str, torch.Tensor] = {}
        cls_tokens = []
        for image in images[:2]:
            mask = random_patch_mask(image, self.backbone.patch_size, self.mask_ratio)
            cls, patch_tokens = self.backbone.forward_tokens(image, mask=mask)
            pred = self.recon_head(patch_tokens)
            target = extract_patches(image, self.backbone.patch_size)
            mae_loss, mae_metrics = self.mae_loss_fn(pred, target, mask)
            mae_losses.append(mae_loss)
            cls_tokens.append(cls)
            for name, value in mae_metrics.items():
                mae_metrics_sum[name] = mae_metrics_sum.get(name, 0.0) + value / 2
        z1 = self.projector(cls_tokens[0])
        z2 = self.projector(cls_tokens[1])
        queue = self._queue_sample() if (stage == "train" and self.queue_size) else None
        vicreg_loss, vicreg_metrics = self.loss_fn(z1, z2, cls_tokens[0], cls_tokens[1], queue=queue)
        mae_loss_total = torch.stack(mae_losses).mean()
        total = self.mae_weight * mae_loss_total + self.vicreg_weight * vicreg_loss
        self.log(f"{stage}/loss", total, on_step=stage == "train", on_epoch=True,
                 prog_bar=True, sync_dist=True)
        self.log(f"{stage}/mae_loss", mae_loss_total.detach(), on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/vicreg_loss", vicreg_loss.detach(), on_step=False, on_epoch=True, sync_dist=True)
        for name, value in {**mae_metrics_sum, **vicreg_metrics}.items():
            self.log(f"{stage}/{name}", value, on_step=False, on_epoch=True, sync_dist=True)
        if stage == "train":
            self._enqueue(z1, z2)
        return total

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Jointly optimize per-view reconstruction and cross-view VICReg."""
        return self._step(batch, "train")

    def validation_step(self, batch: dict[str, Any], batch_idx: int,
                        dataloader_idx: int = 0) -> torch.Tensor:
        """Score the joint objective without updating weights or the queue."""
        return self._step(batch, "val")
