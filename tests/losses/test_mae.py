"""Unit tests for the SimMIM masked-patch reconstruction loss."""

from __future__ import annotations

import torch

from t5_ssl.losses.mae import SimMIMLoss


def test_perfect_reconstruction_on_masked_patches_is_zero():
    """Zero error on masked patches must yield zero loss, regardless of visible ones."""
    loss_fn = SimMIMLoss()
    target = torch.rand(2, 6, 4)
    pred = target.clone()
    pred[:, ::2] += 5.0  # corrupt only the visible (unmasked) patches
    mask = torch.zeros(2, 6, dtype=torch.bool)
    mask[:, 1::2] = True
    loss, metrics = loss_fn(pred, target, mask)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)
    assert metrics["visible_mse"].item() > 0.0


def test_loss_ignores_visible_patch_error():
    """Error on unmasked (visible) patches must not affect the reported loss."""
    loss_fn = SimMIMLoss()
    target = torch.zeros(1, 4, 3)
    pred = torch.zeros(1, 4, 3)
    mask = torch.tensor([[True, False, True, False]])
    pred[:, 1] = 10.0  # visible patch, large error
    pred[:, 2] = 1.0   # masked patch, moderate error
    loss, _ = loss_fn(pred, target, mask)
    expected = (1.0**2) / 2  # mean over the 2 masked patches: (0 + 1)/2
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-6)


def test_all_masked_matches_plain_mse():
    """A fully masked batch should reduce to ordinary MSE."""
    torch.manual_seed(0)
    loss_fn = SimMIMLoss()
    pred = torch.randn(2, 5, 3)
    target = torch.randn(2, 5, 3)
    mask = torch.ones(2, 5, dtype=torch.bool)
    loss, _ = loss_fn(pred, target, mask)
    assert torch.isclose(loss, torch.nn.functional.mse_loss(pred, target), atol=1e-5)


def test_gradients_flow_to_predictions():
    """Gradients must reach the prediction tensor for masked positions."""
    loss_fn = SimMIMLoss()
    pred = torch.randn(2, 4, 3, requires_grad=True)
    target = torch.randn(2, 4, 3)
    mask = torch.tensor([[True, False, True, False], [False, True, False, True]])
    loss, _ = loss_fn(pred, target, mask)
    loss.backward()
    assert pred.grad is not None
    assert not torch.isnan(pred.grad).any()
