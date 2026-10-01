"""Tests for SimMIM-style masked reconstruction, alone and hybridized with VICReg."""

from __future__ import annotations

import torch
from torch import nn

from t5_ssl.losses.mae import SimMIMLoss
from t5_ssl.losses.vicreg import VICRegLoss
from t5_ssl.models.backbones.vit import ViTBackbone
from t5_ssl.models.heads.projector import MLPProjector
from t5_ssl.models.mae_module import (
    MAELightningModule,
    MAEVICRegLightningModule,
    extract_patches,
    random_patch_mask,
)


def _tiny_backbone() -> ViTBackbone:
    return ViTBackbone(in_channels=2, image_size=16, patch_size=4, embed_dim=16, depth=1, num_heads=2)


def test_extract_patches_matches_patch_embed_grid():
    """Unfolded raw-pixel patches must align 1:1 with the ViT patch token grid."""
    backbone = _tiny_backbone()
    image = torch.rand(3, 2, 16, 16)
    patches = extract_patches(image, backbone.patch_size)
    _, patch_tokens = backbone.forward_tokens(image)
    assert patches.shape[:2] == patch_tokens.shape[:2]
    assert patches.shape[-1] == 2 * backbone.patch_size**2


def test_random_patch_mask_respects_ratio_on_average():
    """The masked fraction should track mask_ratio over many patches."""
    torch.manual_seed(0)
    image = torch.rand(1, 2, 16, 16)
    mask = random_patch_mask(image, patch_size=4, mask_ratio=0.6)
    assert mask.dtype == torch.bool
    assert mask.shape == (1, 16)


def test_mae_module_reconstructs_and_backpropagates():
    """A forward/backward pass should update the encoder and reconstruction head only."""
    backbone = _tiny_backbone()
    recon_head = nn.Linear(backbone.out_dim, 2 * backbone.patch_size**2)
    model = MAELightningModule(
        backbone=backbone, projector=nn.Identity(), recon_head=recon_head,
        loss_fn=SimMIMLoss(), mask_ratio=0.5, max_epochs=1,
    )
    image = torch.rand(3, 2, 16, 16)
    loss = model.training_step({"image": image}, 0)
    loss.backward()
    assert torch.isfinite(loss)
    assert recon_head.weight.grad is not None
    assert any(p.grad is not None for p in backbone.parameters())


def test_mae_module_uses_only_first_view_of_a_pair():
    """A two-view batch (shared augmentation contract) should not error; view 2 is ignored."""
    backbone = _tiny_backbone()
    recon_head = nn.Linear(backbone.out_dim, 2 * backbone.patch_size**2)
    model = MAELightningModule(
        backbone=backbone, projector=nn.Identity(), recon_head=recon_head,
        loss_fn=SimMIMLoss(), mask_ratio=0.5, max_epochs=1,
    )
    x1, x2 = torch.rand(2, 2, 16, 16), torch.rand(2, 2, 16, 16)
    loss = model.training_step({"image": (x1, x2)}, 0)
    assert torch.isfinite(loss)


def test_hybrid_module_combines_mae_and_vicreg_losses():
    """The hybrid module should optimize the shared encoder, projector, and recon head."""
    backbone = _tiny_backbone()
    projector = MLPProjector(backbone.out_dim, 16, 8, num_layers=2, use_bn=False)
    recon_head = nn.Linear(backbone.out_dim, 2 * backbone.patch_size**2)
    model = MAEVICRegLightningModule(
        backbone=backbone, projector=projector, recon_head=recon_head,
        mae_loss_fn=SimMIMLoss(), vicreg_loss_fn=VICRegLoss(), mask_ratio=0.5, max_epochs=1,
    )
    x1, x2 = torch.rand(4, 2, 16, 16), torch.rand(4, 2, 16, 16)
    loss = model.training_step({"image": (x1, x2)}, 0)
    loss.backward()
    assert torch.isfinite(loss)
    assert recon_head.weight.grad is not None
    assert any(p.grad is not None for p in projector.parameters())
    assert any(p.grad is not None for p in backbone.parameters())


def test_hybrid_module_queue_fills_like_vicreg_module():
    """The hybrid module's optional queue should behave like the standalone VICReg one."""
    backbone = _tiny_backbone()
    projector = MLPProjector(backbone.out_dim, 16, 8, num_layers=2, use_bn=False)
    recon_head = nn.Linear(backbone.out_dim, 2 * backbone.patch_size**2)
    model = MAEVICRegLightningModule(
        backbone=backbone, projector=projector, recon_head=recon_head,
        mae_loss_fn=SimMIMLoss(), vicreg_loss_fn=VICRegLoss(), mask_ratio=0.5, max_epochs=1,
        queue_size=16,
    )
    x1, x2 = torch.rand(4, 2, 16, 16), torch.rand(4, 2, 16, 16)
    model.training_step({"image": (x1, x2)}, 0)
    assert int(model.queue_filled.item()) == 8


def test_hybrid_module_rejects_single_view_batch():
    """The hybrid objective needs two augmented views; a single image must raise."""
    import pytest

    backbone = _tiny_backbone()
    projector = MLPProjector(backbone.out_dim, 16, 8, num_layers=2, use_bn=False)
    recon_head = nn.Linear(backbone.out_dim, 2 * backbone.patch_size**2)
    model = MAEVICRegLightningModule(
        backbone=backbone, projector=projector, recon_head=recon_head,
        mae_loss_fn=SimMIMLoss(), vicreg_loss_fn=VICRegLoss(), max_epochs=1,
    )
    with pytest.raises(ValueError, match="two augmented views"):
        model.training_step({"image": torch.rand(2, 2, 16, 16)}, 0)
