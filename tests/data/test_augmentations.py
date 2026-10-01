"""Tests for multi-modal MRI augmentations."""

from __future__ import annotations

import random

import torch

from t5_ssl.data import EvalTransform, SSLViewTransform
from t5_ssl.data.augmentations import DeterministicSSLViews


def test_ssl_transform_returns_two_registered_shape_views():
    """SSL augmentation should preserve channels and produce two views."""
    random.seed(7)
    torch.manual_seed(7)
    image = torch.linspace(0, 1, 4 * 24 * 24).reshape(4, 24, 24)
    transform = SSLViewTransform(image_size=(16, 16), blur_p=0.0)

    view1, view2 = transform(image)

    assert view1.shape == (4, 16, 16)
    assert view2.shape == (4, 16, 16)
    assert torch.all((view1 >= 0) & (view1 <= 1))
    assert not torch.equal(view1, view2)


def test_eval_transform_is_deterministic():
    """Evaluation resizing should return the same tensor on repeated calls."""
    image = torch.rand(4, 20, 20)
    transform = EvalTransform(image_size=(12, 12))

    first = transform(image)
    second = transform(image)

    assert first.shape == (4, 12, 12)
    assert torch.equal(first, second)


def test_full_field_views_preserve_nonsquare_image_extent():
    """Full-field policy must not turn a nonsquare MRI into a center crop."""
    image = torch.zeros(1, 16, 24)
    image[:, 0, 0] = 1
    image[:, -1, -1] = 1
    transform = SSLViewTransform(
        image_size=(16, 24), crop_scale=(1.0, 1.0),
        horizontal_flip_p=0, vertical_flip_p=0, affine_degrees=0,
        blur_p=0, noise_std=0, intensity_scale=(1.0, 1.0),
    )
    first, second = transform(image)
    assert torch.equal(first, image)
    assert torch.equal(second, image)


def test_rectangular_crop_preserves_requested_shape():
    """A near-native crop ratio should retain the configured rectangular field."""
    transform = SSLViewTransform(
        image_size=(96, 112), crop_scale=(0.8, 1.0), crop_ratio=(1.12, 1.22),
        horizontal_flip_p=0.0, vertical_flip_p=0.0, affine_degrees=0.0,
        blur_p=0.0, noise_std=0.0,
    )
    first, second = transform(torch.rand(3, 97, 115))

    assert first.shape == (3, 96, 112)
    assert second.shape == (3, 96, 112)


def test_validation_views_follow_no_flip_policy():
    """Anatomy-preserving validation should use a mild angle, not a mirror."""
    image = torch.zeros(1, 16, 16)
    image[:, 2:7, 2:5] = 1
    transform = DeterministicSSLViews(
        image_size=(16, 16), horizontal_flip=False, second_view_angle=5.0,
    )
    first, second = transform(image)
    assert torch.equal(first, image)
    assert not torch.equal(first, second)
    assert second[:, :, 12:].sum() == 0


def test_translate_frac_shifts_the_view():
    """A nonzero translate_frac should move content instead of leaving it fixed."""
    random.seed(0)
    torch.manual_seed(0)
    image = torch.zeros(1, 32, 32)
    image[:, 14:18, 14:18] = 1.0
    transform = SSLViewTransform(
        image_size=(32, 32), crop_scale=(1.0, 1.0), horizontal_flip_p=0, vertical_flip_p=0,
        affine_degrees=0.0, translate_frac=(0.2, 0.2), blur_p=0, noise_std=0,
        intensity_scale=(1.0, 1.0),
    )
    views = [transform._make_view(image) for _ in range(8)]
    assert any(not torch.equal(v, image) for v in views)


def test_gamma_range_default_is_a_no_op():
    """The default gamma_range must not alter pixel values."""
    image = torch.rand(1, 8, 8)
    transform = SSLViewTransform(
        image_size=(8, 8), crop_scale=(1.0, 1.0), horizontal_flip_p=0, vertical_flip_p=0,
        affine_degrees=0.0, blur_p=0, noise_std=0, intensity_scale=(1.0, 1.0),
    )
    assert torch.equal(transform._make_view(image), image)


def test_gamma_range_applies_power_law():
    """A gamma != 1 should reshape intensities monotonically."""
    torch.manual_seed(0)
    image = torch.full((1, 4, 4), 0.25)
    transform = SSLViewTransform(
        image_size=(4, 4), crop_scale=(1.0, 1.0), horizontal_flip_p=0, vertical_flip_p=0,
        affine_degrees=0.0, blur_p=0, noise_std=0, intensity_scale=(1.0, 1.0),
        gamma_range=(2.0, 2.0),
    )
    view = transform._make_view(image)
    assert torch.allclose(view, torch.full((1, 4, 4), 0.0625), atol=1e-5)


def test_bias_field_p_zero_is_a_no_op():
    """bias_field_p=0 (the default) must never perturb the view."""
    image = torch.rand(1, 8, 8)
    transform = SSLViewTransform(
        image_size=(8, 8), crop_scale=(1.0, 1.0), horizontal_flip_p=0, vertical_flip_p=0,
        affine_degrees=0.0, blur_p=0, noise_std=0, intensity_scale=(1.0, 1.0),
        bias_field_p=0.0,
    )
    assert torch.equal(transform._make_view(image), image)


def test_bias_field_p_one_perturbs_the_view():
    """bias_field_p=1 should always apply the smooth multiplicative field."""
    torch.manual_seed(0)
    image = torch.full((1, 16, 16), 0.5)
    transform = SSLViewTransform(
        image_size=(16, 16), crop_scale=(1.0, 1.0), horizontal_flip_p=0, vertical_flip_p=0,
        affine_degrees=0.0, blur_p=0, noise_std=0, intensity_scale=(1.0, 1.0),
        bias_field_p=1.0, bias_field_alpha=0.2,
    )
    view = transform._make_view(image)
    assert not torch.equal(view, image)
    assert view.std() > 0
