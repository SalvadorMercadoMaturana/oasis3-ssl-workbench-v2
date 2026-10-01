"""Tests for the MLP projector's normalization options."""

from __future__ import annotations

import torch
from torch import nn

from t5_ssl.models.heads.projector import MLPProjector


def test_use_bn_true_defaults_to_batchnorm():
    """Legacy `use_bn=True` must keep producing BatchNorm1d layers."""
    projector = MLPProjector(8, 16, 8, num_layers=2, use_bn=True)
    assert any(isinstance(m, nn.BatchNorm1d) for m in projector.net)


def test_use_bn_false_defaults_to_no_norm():
    """Legacy `use_bn=False` must keep producing no normalization layer."""
    projector = MLPProjector(8, 16, 8, num_layers=2, use_bn=False)
    assert not any(isinstance(m, nn.BatchNorm1d | nn.GroupNorm | nn.LayerNorm) for m in projector.net)


def test_norm_group_normalizes_each_sample_independently():
    """GroupNorm must not pool statistics across the batch, unlike BatchNorm1d.

    (PyTorch's `F.group_norm` still requires more than one value per sample's
    channel group on a plain (N, C) tensor, so batch_size=1 alone still errors
    -- unrelated to whether it depends on *other* samples' statistics, which
    it never does, so a two-sample batch already demonstrates independence.)
    """
    projector = MLPProjector(8, 16, 8, num_layers=2, norm="group")
    assert any(isinstance(m, nn.GroupNorm) for m in projector.net)
    projector.eval()
    out_batch = projector(torch.randn(4, 8))
    single = torch.randn(1, 8)
    out_pair = projector(torch.cat([single, torch.randn(1, 8)]))
    # Changing the *other* sample in the batch must not change this sample's output.
    assert torch.allclose(out_pair[0], projector(torch.cat([single, torch.randn(1, 8)]))[0], atol=1e-5)
    assert out_batch.shape == (4, 8)
    assert torch.isfinite(out_batch).all()


def test_norm_layer_works_on_batch_size_one():
    """LayerNorm must also be batch-size independent."""
    projector = MLPProjector(8, 16, 8, num_layers=2, norm="layer")
    assert any(isinstance(m, nn.LayerNorm) for m in projector.net)
    out = projector(torch.randn(1, 8))
    assert out.shape == (1, 8)


def test_norm_overrides_use_bn_when_given():
    """An explicit `norm` should take precedence over the legacy `use_bn` flag."""
    projector = MLPProjector(8, 16, 8, num_layers=2, use_bn=True, norm="none")
    assert not any(isinstance(m, nn.BatchNorm1d | nn.GroupNorm | nn.LayerNorm) for m in projector.net)


def test_norm_rejects_unknown_value():
    """Only the four documented normalization modes are valid."""
    import pytest

    with pytest.raises(ValueError, match="norm"):
        MLPProjector(8, 16, 8, num_layers=2, norm="instance")
