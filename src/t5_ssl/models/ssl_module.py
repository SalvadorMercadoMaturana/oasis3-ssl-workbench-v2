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


def _map_views(images: Any, fn: Any) -> Any:
    """Apply ``fn`` to every tensor of ``batch["image"]`` (a tensor or nested views)."""
    if isinstance(images, list | tuple):
        return type(images)(_map_views(view, fn) for view in images)
    return fn(images) if isinstance(images, torch.Tensor) else images


def disable_stochastic_depth(module: nn.Module) -> None:
    """Turn off stochastic depth in an EMA copy (ViT ``DropPath``, torchvision ``StochasticDepth``)."""
    for child in module.modules():
        if hasattr(child, "drop_prob"):
            child.drop_prob = 0.0
        elif type(child).__name__ == "StochasticDepth":
            child.p = 0.0


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
    # {"num_control_points", "max_displacement", "locked_borders"}: training views arrive
    # as whole cached cubes; TorchIO's elastic deformation runs here on the GPU, then the
    # central ``crop_size`` cube (t5_ssl.data.roi3d.finish_deferred_views); None = off.
    tio_gpu_elastic: dict | None = None
    crop_size: int = 32
    # Backbone BatchNorm with frozen statistics: recalibrated once on the training
    # views (``bn_calibration_batches``, None = one epoch) when training starts,
    # then held in eval mode for the whole run, so the encoder normalizes the same
    # way in training and evaluation (gamma/beta still learn).
    freeze_backbone_bn: bool = False
    bn_calibration_batches: int | None = None

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

    @staticmethod
    def two_views(batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the first two views of ``batch["image"]`` (one tensor counts as both)."""
        images = batch["image"]
        if isinstance(images, list | tuple) and len(images) >= 2:
            return images[0], images[1]
        return images, images

    def _two_view_loss(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        """Shared encoder and projector on both views, then the SSL loss (VICReg)."""
        x1, x2 = self.two_views(batch)
        h1, h2 = self.backbone(x1), self.backbone(x2)
        z1, z2 = self.projector(h1), self.projector(h2)
        return self.loss_fn(z1, z2, h1, h2)

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
        total_loss, metrics = self._two_view_loss(batch)
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

    def validation_step(self, batch: dict[str, Any], batch_idx: int,
                        dataloader_idx: int = 0) -> torch.Tensor:
        """Execute a validation evaluation step.

        Parameters
        ----------
        batch:
            Dictionary containing input images and metadata.
        batch_idx:
            Index of the current validation batch.
        dataloader_idx:
            0 = validation subjects, 1 = the fixed training subset (logged ``trainval/``).

        Returns
        -------
        torch.Tensor
            Validation loss tensor.

        """
        val_loss, metrics = self._two_view_loss(batch)
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

    def _backbone_norms(self) -> list[nn.Module]:
        encoders = [getattr(self, name, None) for name in
                    ("backbone", "target_backbone", "teacher_backbone")]
        return [m for enc in encoders if enc is not None for m in enc.modules()
                if isinstance(m, nn.modules.batchnorm._BatchNorm)]

    def train(self, mode: bool = True) -> SSLLightningModule:
        """Switch modes, keeping the backbone BatchNorm frozen when requested."""
        super().train(mode)
        if self.freeze_backbone_bn:
            for norm in self._backbone_norms():
                norm.eval()
        return self

    @torch.no_grad()
    def _calibrate_backbone_bn(self) -> None:
        """Recompute the online backbone's BatchNorm statistics on training views."""
        norms = [m for m in self.backbone.modules()
                 if isinstance(m, nn.modules.batchnorm._BatchNorm)]
        if not norms:
            return
        momenta = [m.momentum for m in norms]
        for m in norms:
            m.reset_running_stats()
            m.momentum = None  # cumulative average over the calibration batches
            m.train()
        loader = self.trainer.datamodule.train_dataloader()
        limit = self.bn_calibration_batches or len(loader)
        for i, batch in enumerate(loader):
            if i >= limit:
                break
            batch = self.on_after_batch_transfer(
                self.transfer_batch_to_device(batch, self.device, 0), 0)
            images = batch["image"]
            for view in images if isinstance(images, list | tuple) else [images]:
                self.backbone(view)
        for m, momentum in zip(norms, momenta, strict=True):
            m.momentum = momentum
            m.eval()
        for target_name in ("target_backbone", "teacher_backbone"):
            target = getattr(self, target_name, None)
            if target is not None:
                for source, destination in zip(self.backbone.buffers(), target.buffers(),
                                               strict=True):
                    destination.copy_(source)

    def on_train_start(self) -> None:
        if self.freeze_backbone_bn:
            self._calibrate_backbone_bn()
            self.train(True)

    # ---- validation on augmented views (augmentation.val_views=train)
    # The validation loaders yield the training views, made fixed per session on the CPU
    # (ROI3DDataset view_seed); ``augment_val`` makes the GPU deformations fixed too, from a
    # generator seeded per validation batch (``val_view_seed``), so every epoch sees the
    # same views. A second validation loader (``trainval``) scores a fixed subset of
    # training subjects the same way, in eval mode: train and validation become comparable.
    augment_val: bool = False
    val_view_seed: int = 12345
    _val_stage: str = "val"
    _val_batches: dict[int, int] | None = None

    def log(self, name: str, value: Any, *args: Any, **kwargs: Any) -> None:
        """Name metrics of the second validation loader ``trainval/...``; no loader suffix."""
        if self._val_stage != "val" and name.startswith("val/"):
            name = f"{self._val_stage}/{name[4:]}"
        kwargs.setdefault("add_dataloader_idx", False)
        super().log(name, value, *args, **kwargs)

    def on_validation_epoch_start(self) -> None:
        self._val_batches = {}

    def on_validation_batch_start(self, batch: Any, batch_idx: int,
                                  dataloader_idx: int = 0) -> None:
        self._val_stage = "val" if dataloader_idx == 0 else "trainval"

    def on_validation_epoch_end(self) -> None:
        self._val_stage = "val"

    def _val_generator(self, device: torch.device, dataloader_idx: int) -> torch.Generator:
        counts = self._val_batches if self._val_batches is not None else {}
        count = counts.get(dataloader_idx, 0)
        counts[dataloader_idx] = count + 1
        self._val_batches = counts
        seed = self.val_view_seed + 1_000_003 * dataloader_idx + 7919 * count
        return torch.Generator(device=device).manual_seed(seed)

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        """Apply the GPU elastic deformations, then ``input_memory_format``.

        Training views get fresh random deformations; validation views (``augment_val``)
        get the same deformation every epoch.
        """
        if not (isinstance(batch, dict) and "image" in batch):
            return batch
        generator = None
        if not self.training and self.augment_val and (self.gpu_elastic or self.tio_gpu_elastic):
            first = batch["image"][0] if isinstance(batch["image"], list | tuple) else batch["image"]
            generator = self._val_generator(first.device, dataloader_idx)
        if self.gpu_elastic and (self.training or generator is not None):
            from t5_ssl.data.gpu_augment import elastic_deform
            batch["image"] = _map_views(batch["image"], lambda view: (
                elastic_deform(view, **self.gpu_elastic, generator=generator)
                if view.ndim == 5 else view))
        if self.tio_gpu_elastic:
            from t5_ssl.data.roi3d import finish_deferred_views
            batch["image"] = _map_views(batch["image"], lambda view: (
                finish_deferred_views(view, self.tio_gpu_elastic, self.crop_size,
                                      generator=generator)
                if view.ndim == 5 and view.shape[-1] != self.crop_size else view))
        if self.input_memory_format is not None:
            batch["image"] = _map_views(batch["image"], self._to_input_memory_format)
        return batch

    def _to_input_memory_format(self, view: torch.Tensor) -> torch.Tensor:
        """``input_memory_format`` for a 4-D or 5-D view, in its 2-D or 3-D flavour."""
        if view.ndim not in (4, 5):
            return view
        fmt = self.input_memory_format
        if view.ndim == 5 and fmt is torch.channels_last:
            fmt = torch.channels_last_3d
        elif view.ndim == 4 and fmt is torch.channels_last_3d:
            fmt = torch.channels_last
        return view.contiguous(memory_format=fmt)

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
