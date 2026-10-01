"""Check BYOL and DINOv2 teacher branches and losses on tiny tensors."""

from __future__ import annotations

import torch
from torch import nn

from t5_ssl.data.augmentations import DINOv2ViewTransform
from t5_ssl.losses.byol import BYOLLoss
from t5_ssl.losses.dinov2 import DINOv2Loss
from t5_ssl.models.backbones.vit import ViTBackbone
from t5_ssl.models.byol_module import BYOLLightningModule
from t5_ssl.models.dinov2_module import DINOv2LightningModule
from t5_ssl.models.heads.dino_head import DINOHead
from t5_ssl.models.heads.projector import MLPProjector


class TinyBackbone(nn.Module):
    """Cheap four-channel feature extractor for the BYOL contract."""

    out_dim = 8

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(4, 8, 3, padding=1), nn.AdaptiveAvgPool2d(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return one vector per image."""
        return self.net(x).flatten(1)


def test_byol_online_gradients_exclude_target():
    """Cross-view predictions train the online branch while EMA targets stay frozen."""
    backbone = TinyBackbone()
    projector = MLPProjector(8, 16, 8, 2, use_bn=False)
    predictor = MLPProjector(8, 16, 8, 2, use_bn=False)
    model = BYOLLightningModule(
        backbone=backbone, projector=projector, predictor=predictor,
        loss_fn=BYOLLoss(), max_epochs=1,
    )
    x1, x2 = torch.rand(3, 4, 16, 16), torch.rand(3, 4, 16, 16)
    loss = model.training_step({"image": (x1, x2)}, 0)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None for p in model.predictor.parameters())
    assert all(p.grad is None for p in model.target_backbone.parameters())
    assert all(p.grad is None for p in model.target_projector.parameters())
    optimizer = model.configure_optimizers()["optimizer"]
    optimized_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert all(id(p) not in optimized_ids for p in model.target_backbone.parameters())


def _dinov2_2d(centering: str = "sinkhorn_knopp") -> DINOv2LightningModule:
    backbone = ViTBackbone(image_size=32, patch_size=8, embed_dim=24, depth=1, num_heads=3)
    return DINOv2LightningModule(
        backbone=backbone, projector=DINOHead(24, 16, hidden_dim=32, bottleneck_dim=8),
        patch_head=DINOHead(24, 16, hidden_dim=32, bottleneck_dim=8),
        loss_fn=DINOv2Loss(out_dim=16, centering=centering), max_epochs=1,
    )


def _dinov2_batch(batch_size: int = 4) -> dict:
    image = torch.rand(4, 32, 32)
    transform = DINOv2ViewTransform(image_size=(32, 32), local_size=(16, 16))
    views = transform(image)
    assert len(views) == 4
    assert views[0].shape == (4, 32, 32)
    assert views[2].shape == (4, 16, 16)
    return {"image": tuple(view.unsqueeze(0).repeat(batch_size, 1, 1, 1) for view in views)}


def test_dinov2_masks_patches_and_excludes_teacher_gradients():
    """Masked student patches and local crops participate in the DINOv2 objective."""
    model = _dinov2_2d()
    batch = _dinov2_batch()
    loss = model.training_step(batch, 0)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.patch_head.mlp[0].weight.grad is not None
    assert all(p.grad is None for p in model.teacher_backbone.parameters())
    model.eval()
    with torch.no_grad():
        first_val = model.validation_step(batch, 0)
        second_val = model.validation_step(batch, 0)
    assert torch.allclose(first_val, second_val)


def test_dinov2_softmax_centering_updates_centers():
    """DINO's EMA centering remains available and moves the centers during training."""
    model = _dinov2_2d(centering="softmax")
    model.training_step(_dinov2_batch(), 0).backward()
    assert model.loss_fn.cls_center.abs().sum() > 0
    assert model.loss_fn.patch_center.abs().sum() > 0
