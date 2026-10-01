"""Tests for the compact ViT backbone's pooling options."""

from __future__ import annotations

import torch

from t5_ssl.models.backbones.vit import ViTBackbone


def _tiny(pool: str = "cls") -> ViTBackbone:
    return ViTBackbone(in_channels=1, image_size=16, patch_size=4, embed_dim=16,
                       depth=1, num_heads=2, pool=pool)


def test_default_pool_matches_cls_token():
    """The default 'cls' pooling must equal forward_tokens' CLS output."""
    backbone = _tiny("cls")
    image = torch.rand(2, 1, 16, 16)
    cls, _ = backbone.forward_tokens(image)
    assert torch.equal(backbone(image), cls)


def test_mean_pool_matches_patch_token_average():
    """'mean' pooling must average the patch tokens, not the CLS token."""
    backbone = _tiny("mean")
    image = torch.rand(2, 1, 16, 16)
    cls, patches = backbone.forward_tokens(image)
    out = backbone(image)
    assert torch.allclose(out, patches.mean(dim=1), atol=1e-6)
    assert not torch.allclose(out, cls, atol=1e-6)


def test_out_dim_unaffected_by_pool_choice():
    """Pooling strategy must not change the reported embedding dimension."""
    assert _tiny("cls").out_dim == _tiny("mean").out_dim == 16


def test_invalid_pool_rejected():
    """Only 'cls' and 'mean' are supported pooling strategies."""
    import pytest

    with pytest.raises(ValueError, match="pool"):
        ViTBackbone(pool="max")
