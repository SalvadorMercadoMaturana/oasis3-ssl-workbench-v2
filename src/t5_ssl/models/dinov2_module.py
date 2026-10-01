"""DINOv2 student/teacher training on global and local MRI views (2D slices or 3D volumes)."""

from __future__ import annotations

import copy
import math
import random
from typing import Any

import torch
from torch import nn

from t5_ssl.losses.dinov2 import DINOv2Loss
from t5_ssl.models.masking import sample_masks
from t5_ssl.models.ssl_module import SSLLightningModule


class DINOv2LightningModule(SSLLightningModule):
    """Distill CLS and masked patch tokens from an EMA ViT teacher.

    The student sees the two global views with iBOT block masks (DINOv2 rule:
    ``mask_probability`` of the images masked, ratios in ``mask_ratio_min_max``)
    and the local crops unmasked; the teacher sees the global views unmasked.
    ``projector`` and ``patch_head`` are separate DINO heads (untied, as in
    DINOv2). The prototype layers stay frozen for ``freeze_last_layer_epochs``,
    weight decay skips biases/norms/LayerScale and can follow a cosine schedule
    up to ``weight_decay_end``, and the teacher runs without stochastic depth.
    """

    def __init__(
        self, *, backbone: nn.Module, projector: nn.Module, patch_head: nn.Module,
        loss_fn: DINOv2Loss, mask_probability: float = 0.5,
        mask_ratio_min_max: tuple[float, float] = (0.1, 0.5), min_mask_block: int = 4,
        ema_base: float = 0.992, teacher_temp_final: float | None = None,
        teacher_temp_warmup_epochs: int = 30, freeze_last_layer_epochs: int = 1,
        weight_decay_end: float | None = None, **kwargs: Any,
    ) -> None:
        super().__init__(backbone=backbone, projector=projector, loss_fn=loss_fn, **kwargs)
        low, high = (float(v) for v in mask_ratio_min_max)
        if not 0 <= mask_probability <= 1:
            raise ValueError("mask_probability must be in [0, 1].")
        if not 0 < low <= high < 1:
            raise ValueError("mask_ratio_min_max must satisfy 0 < min <= max < 1.")
        if not 0 <= ema_base < 1:
            raise ValueError("ema_base must be in [0, 1).")
        self.patch_head = patch_head
        self.teacher_backbone = copy.deepcopy(backbone).requires_grad_(False)
        self.teacher_projector = copy.deepcopy(projector).requires_grad_(False)
        self.teacher_patch_head = copy.deepcopy(patch_head).requires_grad_(False)
        for module in self.teacher_backbone.modules():  # the teacher never drops paths
            if hasattr(module, "drop_prob"):
                module.drop_prob = 0.0
        self.mask_probability = float(mask_probability)
        self.mask_ratio_min_max = (low, high)
        self.min_mask_block = int(min_mask_block)
        self.ema_base = ema_base
        self.freeze_last_layer_epochs = int(freeze_last_layer_epochs)
        self.weight_decay_end = weight_decay_end
        # Teacher temperature rises linearly from its initial value to
        # `teacher_temp_final` over `teacher_temp_warmup_epochs`, then holds —
        # the DINO/DINOv2 schedule (0.04 -> 0.07).
        self.teacher_temp_start = float(loss_fn.teacher_temp)
        self.teacher_temp_final = (
            float(teacher_temp_final) if teacher_temp_final is not None else self.teacher_temp_start
        )
        self.teacher_temp_warmup_epochs = max(1, int(teacher_temp_warmup_epochs))

    # ------------------------------------------------------------ optimization
    def trainable_parameters(self) -> list[nn.Parameter]:
        """Optimize student encoder and both heads only."""
        return super().trainable_parameters() + list(self.patch_head.parameters())

    def optimizer_param_groups(self) -> list[dict[str, Any]]:
        """As DINO: no weight decay on biases, norms and LayerScale (1-D parameters)."""
        params = self.trainable_parameters()
        return [
            {"params": [p for p in params if p.ndim > 1], "weight_decay": self.weight_decay,
             "regularized": True},
            {"params": [p for p in params if p.ndim <= 1], "weight_decay": 0.0},
        ]

    def on_train_epoch_start(self) -> None:
        """Advance the teacher-temperature warmup before this epoch's batches."""
        progress = min(1.0, self.current_epoch / self.teacher_temp_warmup_epochs)
        self.loss_fn.teacher_temp = (
            self.teacher_temp_start
            + (self.teacher_temp_final - self.teacher_temp_start) * progress
        )
        self.log("train/teacher_temp", self.loss_fn.teacher_temp, on_step=False, on_epoch=True)

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        """Cosine weight-decay schedule from ``weight_decay`` to ``weight_decay_end``."""
        if self.weight_decay_end is None:
            return
        progress = min(1.0, self.global_step / max(1, self.trainer.estimated_stepping_batches))
        decay = self.weight_decay_end + 0.5 * (self.weight_decay - self.weight_decay_end) * (
            1 + math.cos(math.pi * progress))
        optimizers = self.optimizers(use_pl_optimizer=False)
        for optimizer in optimizers if isinstance(optimizers, list) else [optimizers]:
            for group in optimizer.param_groups:
                if group.get("regularized"):
                    group["weight_decay"] = decay
        self.log("train/weight_decay", decay, on_step=False, on_epoch=True)

    def on_after_backward(self) -> None:
        """Keep the prototype layers frozen during the first epoch(s), as in DINO."""
        if self.current_epoch >= self.freeze_last_layer_epochs:
            return
        for head in (self.projector, self.patch_head):
            for parameter in getattr(head, "last_layer_parameters", list)():
                parameter.grad = None

    # ------------------------------------------------------------ objective
    def _masks(self, batch_size: int, grid: tuple[int, ...], stage: str, batch_idx: int,
               view: int, device: torch.device) -> torch.Tensor:
        if stage == "train":
            rng = random.Random(int(torch.randint(0, 2**31 - 1, (1,)).item()))
        else:  # the same validation masks every epoch
            rng = random.Random(10_007 * batch_idx + view)
        masks = sample_masks(batch_size, grid, self.mask_probability, self.mask_ratio_min_max,
                             rng, min_block=self.min_mask_block)
        return masks.to(device)

    def _step(self, batch: dict[str, Any], stage: str, batch_idx: int = 0) -> torch.Tensor:
        images = batch["image"]
        if not isinstance(images, list | tuple) or len(images) < 2:
            raise ValueError("DINOv2 requires two global views.")
        globals_, locals_ = images[:2], images[2:]
        student_cls, backbone_cls, student_patches, patch_weights, masks = [], [], [], [], []
        for view, image in enumerate(globals_):
            # Spatial dims after (B, C): (H, W) for slices, (D, H, W) for volumes.
            grid = tuple(side // self.backbone.patch_size for side in image.shape[2:])
            mask = self._masks(image.shape[0], grid, stage, batch_idx, view, image.device)
            cls, patches = self.backbone.forward_tokens(image, mask=mask)
            backbone_cls.append(cls)
            student_cls.append(self.projector(cls))
            student_patches.append(self.patch_head(patches[mask]))
            per_image = mask.sum(dim=1, keepdim=True).clamp_min(1).float()
            patch_weights.append((1.0 / per_image).expand_as(mask)[mask])
            masks.append(mask)
        for image in locals_:
            cls, _ = self.backbone.forward_tokens(image)
            student_cls.append(self.projector(cls))
        teacher_cls, teacher_patches = [], []
        with torch.no_grad():
            for image, mask in zip(globals_, masks, strict=True):
                cls, patches = self.teacher_backbone.forward_tokens(image)
                teacher_cls.append(self.teacher_projector(cls))
                teacher_patches.append(self.teacher_patch_head(patches[mask]))
        loss, metrics = self.loss_fn(
            student_cls, teacher_cls, student_patches, teacher_patches, patch_weights,
            backbone_cls, update_center=stage == "train",
        )
        self.log(f"{stage}/loss", loss, on_step=stage == "train", on_epoch=True,
                 prog_bar=True, sync_dist=True)
        for name, value in metrics.items():
            self.log(f"{stage}/{name}", value, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Update student from two masked global and the local crops."""
        return self._step(batch, "train", batch_idx)

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Measure the objective with fixed masks and without EMA updates."""
        return self._step(batch, "val", batch_idx)

    @torch.no_grad()
    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        """Track the student after the optimizer step with cosine EMA momentum."""
        total_steps = max(1, self.trainer.estimated_stepping_batches)
        progress = min(1.0, self.global_step / total_steps)
        momentum = 1 - (1 - self.ema_base) * (math.cos(math.pi * progress) + 1) / 2
        for student, teacher in (
            (self.backbone, self.teacher_backbone),
            (self.projector, self.teacher_projector),
            (self.patch_head, self.teacher_patch_head),
        ):
            for source, destination in zip(
                student.parameters(), teacher.parameters(), strict=True
            ):
                destination.lerp_(source, 1 - momentum)
            for source, destination in zip(student.buffers(), teacher.buffers(), strict=True):
                destination.copy_(source)
