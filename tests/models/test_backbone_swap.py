"""Encoder swaps across methods: recipe overrides, DINOv2 on a CNN, EMA copies without drop paths."""

from __future__ import annotations

import torch

from t5_ssl import recipes
from t5_ssl.losses.byol import BYOLLoss
from t5_ssl.losses.dinov2 import DINOv2Loss
from t5_ssl.models.backbones.resnet3d import ConvNeXt3DBackbone, ResNet3DBackbone
from t5_ssl.models.backbones.vit3d import ViT3DBackbone
from t5_ssl.models.byol_module import BYOLLightningModule
from t5_ssl.models.dinov2_module import DINOv2LightningModule
from t5_ssl.models.heads.dino_head import DINOHead
from t5_ssl.models.heads.projector import MLPProjector


def test_backbone_overrides_come_after_the_method_and_name_the_run():
    out = recipes.overrides("dinov2", backbone="resnet18", roi="multi3")
    assert out.index("model.backbone.variant=resnet18_3d") > out.index("model=dinov2_vit3d")
    assert "model.backbone.in_channels=6" in out
    assert recipes.overrides("vicreg") == recipes.overrides("vicreg", backbone=None)
    assert recipes.run_name("byol", "campaign", 2, "noise", "32", "vit_small") == \
        "byol_campaign__noise__bbvit_small__roi32_s2"
    # every method's own encoder is a swappable backbone; the extra ones are the ViT-S
    # starting from Meta's DINOv2 weights (2026-11-07) and the ingestion settings
    # (2026-11-22: t5_ssl.models.backbones.ingest, MedicalNet, 2D slice encoders)
    extra = set(recipes.BACKBONES) - set(recipes.DEFAULT_BACKBONE.values())
    ingest_prefixes = ("cnx3d", "rn18_3d", "mednet", "cnx2d_", "rn18_2d_", "rn50_2d_")
    assert {b for b in extra if not b.startswith(ingest_prefixes)} == {"vit_small_dv2"}
    assert all(any("model.backbone.ingest=" in o for o in recipes.BACKBONES[b])
               for b in extra - {"vit_small_dv2"})


def test_dinov2_on_a_cnn_distills_the_pooled_embedding_without_ibot():
    backbone = ResNet3DBackbone(in_channels=2)
    model = DINOv2LightningModule(
        backbone=backbone, projector=DINOHead(512, 16, hidden_dim=16, bottleneck_dim=8),
        patch_head=DINOHead(512, 16, hidden_dim=16, bottleneck_dim=8),
        loss_fn=DINOv2Loss(out_dim=16), max_epochs=1)
    assert not model.has_tokens
    views = [torch.rand(4, 2, 16, 16, 16) for _ in range(2)]
    loss = model.training_step({"image": views}, 0)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.backbone.conv1.weight.grad is not None
    assert all(p.grad is None for p in model.patch_head.parameters())


def test_ema_copies_never_drop_paths():
    vit = ViT3DBackbone(in_channels=2, image_size=16, patch_size=4, embed_dim=12, depth=2,
                        num_heads=3, drop_path=0.1)  # drop rate grows with depth: block 2 > 0
    byol = BYOLLightningModule(
        backbone=vit, projector=MLPProjector(12, 16, 8, num_layers=2),
        predictor=MLPProjector(8, 16, 8, num_layers=2), loss_fn=BYOLLoss(), max_epochs=1)
    assert any(getattr(m, "drop_prob", 0) > 0 for m in byol.backbone.modules())
    assert all(m.drop_prob == 0 for m in byol.target_backbone.modules() if hasattr(m, "drop_prob"))
    convnext = ConvNeXt3DBackbone(in_channels=2)
    dino = DINOv2LightningModule(
        backbone=convnext, projector=DINOHead(768, 16, hidden_dim=16, bottleneck_dim=8),
        patch_head=DINOHead(768, 16, hidden_dim=16, bottleneck_dim=8),
        loss_fn=DINOv2Loss(out_dim=16), max_epochs=1)
    depth = [m for m in dino.teacher_backbone.modules() if type(m).__name__ == "StochasticDepth"]
    assert depth and all(m.p == 0 for m in depth)
    assert any(m.p > 0 for m in dino.backbone.modules() if type(m).__name__ == "StochasticDepth")
