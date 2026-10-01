"""PyTorch Lightning Module for Self-Supervised Learning training."""

from __future__ import annotations

import math
from typing import Any

import lightning as L
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from t5_ssl.losses.base import BaseSSLLoss
from t5_ssl.models.backbones.base import BaseBackbone
from t5_ssl.utils.logging import get_logger

logger = get_logger(__name__)


class SSLLightningModule(L.LightningModule):
    """Modular PyTorch Lightning Module coordinating SSL training.

    Decouples the encoder backbone, projection head, and SSL loss objective,
    The two-view, shared-encoder path implements VICReg. Teacher/student
    methods provide their own training steps in dedicated subclasses.

    Parameters
    ----------
    backbone:
        Feature extraction backbone inheriting from :class:`BaseBackbone`.
    projector:
        Projection head mapping representations to the loss space.
    loss_fn:
        SSL criterion inheriting from :class:`BaseSSLLoss`.
    lr:
        Base learning rate for the AdamW optimizer.
        Default: ``1e-4``.
    weight_decay:
        L2 weight decay penalty.
        Default: ``1e-6``.
    max_epochs:
        Cosine decay horizon for learning rate scheduling. Independent of
        the trainer's actual epoch ceiling: pass ``optimizer.schedule_epochs``
        in the training config to decouple them (train.py defaults this to
        the trainer's max_epochs when unset, so existing configs behave
        exactly as before).
        Default: ``100``.
    warmup_epochs:
        Number of initial warmup epochs where learning rate increases linearly.
        Default: ``10``.
    min_lr:
        Minimum learning rate floor at the end of cosine annealing.
        Default: ``1e-6``.

    Runtime switches (class attributes, off by default; ``t5_ssl.utils.runtime``
    turns them on for GPU profiles such as ``perf=rtx4090``):
    ``fused_optimizer`` builds AdamW with ``fused=True`` when every parameter is
    on CUDA, and ``input_memory_format`` (e.g. ``torch.channels_last_3d``) is
    applied to the 4D/5D image tensors of every batch after the device transfer.

    """

    fused_optimizer: bool = False
    input_memory_format: torch.memory_format | None = None
    # {"max_displacement": ..., "control_points": ...}: elastic deformation of every
    # training view on the GPU (t5_ssl.data.gpu_augment); None = off.
    gpu_elastic: dict | None = None

    def __init__(
        self,
        backbone: BaseBackbone,
        projector: nn.Module,
        loss_fn: BaseSSLLoss,
        lr: float = 1e-4,
        weight_decay: float = 1e-6,
        max_epochs: int = 100,
        warmup_epochs: int = 10,
        min_lr: float = 1e-6,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=["backbone", "projector", "loss_fn", "predictor", "patch_head",
                   "recon_head", "mae_loss_fn", "vicreg_loss_fn"]
        )

        self.backbone = backbone
        self.projector = projector
        self.loss_fn = loss_fn

        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract backbone representation features.

        Parameters
        ----------
        x:
            Input image tensor of shape ``(B, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Embeddings of shape ``(B, out_dim)``.

        """
        return self.backbone(x)

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Execute a single SSL training optimization step.

        Expects ``batch["image"]`` to contain two views ``(view1, view2)``.

        Parameters
        ----------
        batch:
            Dictionary containing input images and metadata.
        batch_idx:
            Index of the current training batch.

        Returns
        -------
        torch.Tensor
            Scalar loss tensor for backpropagation.

        """
        images = batch["image"]
        if isinstance(images, (list, tuple)) and len(images) >= 2:
            x1, x2 = images[0], images[1]
        else:
            x1 = x2 = images

        # Forward passes through backbone
        h1 = self.backbone(x1)
        h2 = self.backbone(x2)

        # Forward passes through projection head
        z1 = self.projector(h1)
        z2 = self.projector(h2)

        # Compute SSL loss
        total_loss, metrics = self.loss_fn(z1, z2, h1, h2)

        # Log total loss
        self.log(
            "train/loss",
            total_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        for name, value in metrics.items():
            key = name if name.startswith("train/") else f"train/{name}"
            self.log(key, value, on_step=False, on_epoch=True, sync_dist=True)

        return total_loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Execute a validation evaluation step.

        Parameters
        ----------
        batch:
            Dictionary containing input images and metadata.
        batch_idx:
            Index of the current validation batch.

        Returns
        -------
        torch.Tensor
            Validation loss tensor.

        """
        images = batch["image"]
        if isinstance(images, (list, tuple)) and len(images) >= 2:
            x1, x2 = images[0], images[1]
        else:
            x1 = x2 = images

        h1 = self.backbone(x1)
        h2 = self.backbone(x2)
        z1 = self.projector(h1)
        z2 = self.projector(h2)

        val_loss, metrics = self.loss_fn(z1, z2, h1, h2)

        self.log(
            "val/loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        for name, value in metrics.items():
            key = name if name.startswith("val/") else f"val/{name}"
            self.log(key, value, on_step=False, on_epoch=True, sync_dist=True)

        return val_loss

    def optimizer_param_groups(self) -> list[dict[str, Any]]:
        """AdamW parameter groups; a single group with ``weight_decay`` by default."""
        return [{"params": self.trainable_parameters(), "weight_decay": self.weight_decay}]

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        """Apply the GPU elastic deformation (training only), then ``input_memory_format``."""
        if self.gpu_elastic and self.training and isinstance(batch, dict) and "image" in batch:
            from t5_ssl.data.gpu_augment import elastic_deform

            def deform(image: Any) -> Any:
                if isinstance(image, torch.Tensor) and image.ndim == 5:
                    return elastic_deform(image, **self.gpu_elastic)
                if isinstance(image, list | tuple):
                    return type(image)(deform(view) for view in image)
                return image

            batch["image"] = deform(batch["image"])
        memory_format = self.input_memory_format
        if memory_format is None or not isinstance(batch, dict) or "image" not in batch:
            return batch

        def convert(image: Any) -> Any:
            if isinstance(image, torch.Tensor) and image.ndim in (4, 5):
                fmt = memory_format
                if image.ndim == 5 and fmt is torch.channels_last:
                    fmt = torch.channels_last_3d
                elif image.ndim == 4 and fmt is torch.channels_last_3d:
                    fmt = torch.channels_last
                return image.contiguous(memory_format=fmt)
            if isinstance(image, list | tuple):
                return type(image)(convert(view) for view in image)
            return image

        batch["image"] = convert(batch["image"])
        return batch

    def _optimizer_stepped(self) -> bool:
        """Return True once per optimizer step, False on gradient-accumulation-only batches.

        EMA targets/teachers follow the online weights only when they changed.
        """
        step = int(self.global_step)
        if step == getattr(self, "_last_ema_step", 0):
            return False
        self._last_ema_step = step
        return True

    @staticmethod
    @torch.no_grad()
    def _ema_update(pairs: list[tuple[nn.Module, nn.Module]], momentum: float) -> None:
        """``target = momentum * target + (1 - momentum) * online`` over all pairs, fused."""
        sources: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        for online, target in pairs:
            online_params, target_params = list(online.parameters()), list(target.parameters())
            if len(online_params) != len(target_params):
                raise RuntimeError("EMA pair with different parameter lists")
            sources += online_params
            targets += target_params
        torch._foreach_lerp_(targets, sources, 1 - momentum)
        for online, target in pairs:
            for source, destination in zip(online.buffers(), target.buffers(), strict=True):
                destination.copy_(source)

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure AdamW optimizer with linear warmup + cosine decay scheduler."""
        groups = self.optimizer_param_groups()
        # Lightning refuses gradient clipping with fused optimizers (DINOv2 clips at
        # 3.0); those keep the multi-tensor `foreach` kernels instead.
        clips = bool(self._trainer is not None and self._trainer.gradient_clip_val)
        fused = self.fused_optimizer and not clips and all(
            p.is_cuda for group in groups for p in group["params"])
        optimizer = torch.optim.AdamW(
            groups,
            lr=self.lr,
            weight_decay=self.weight_decay,
            **({"fused": True} if fused else {"foreach": True} if self.fused_optimizer else {}),
        )

        max_epochs = max(1, self.max_epochs)
        warmup_epochs = min(self.warmup_epochs, max_epochs)

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs and warmup_epochs > 0:
                # Linear warmup
                return float(epoch + 1) / float(warmup_epochs)
            # Cosine decay, clamped so epochs beyond max_epochs (e.g. a
            # trainer ceiling raised past the schedule horizon) hold at the
            # fully-decayed floor instead of the cosine wrapping back up.
            progress = min(1.0, float(epoch - warmup_epochs) /
                          float(max(1, max_epochs - warmup_epochs)))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            decayed = (1.0 - self.min_lr / self.lr) * cosine_decay + self.min_lr / self.lr
            return max(0.0, decayed)

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Return the online parameters updated by the SSL objective."""
        return list(self.backbone.parameters()) + list(self.projector.parameters())

    def __repr__(self) -> str:
        """Return developer-friendly summary representation."""
        return (
            f"{self.__class__.__name__}(\n"
            f"  backbone={self.backbone},\n"
            f"  projector={self.projector},\n"
            f"  loss_fn={self.loss_fn},\n"
            f"  lr={self.lr}\n"
            f")"
        )
