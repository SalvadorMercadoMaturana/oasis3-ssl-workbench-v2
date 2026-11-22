"""Ingestion wrapper (t5_ssl.models.backbones.ingest) and the MedicalNet encoder, on CPU."""

from __future__ import annotations

from pathlib import Path

import hydra
import pytest
import torch
from hydra.core.global_hydra import GlobalHydra
from torch import nn

from t5_ssl import recipes
from t5_ssl.models.backbones.ingest import IngestBackbone
from t5_ssl.models.backbones.medicalnet import MedicalNetBackbone

_CONFIG_PATH = str(Path(__file__).resolve().parents[2] / "configs")


class Pool3D(nn.Module):
    """Per-channel mean and the input's spatial size, to see what the encoder received."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.out_dim, self.seen = channels + 1, []
        self.weight = nn.Parameter(torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.seen.append(tuple(x.shape))
        size = torch.full((x.shape[0], 1), float(x.shape[-1]))
        return torch.cat([x.mean(dim=(2, 3, 4)) * self.weight, size], dim=1)


class Pool2D(Pool3D):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.seen.append(tuple(x.shape))
        size = torch.full((x.shape[0], 1), float(x.shape[-1]))
        return torch.cat([x.mean(dim=(2, 3)) * self.weight, size], dim=1)


def test_default_settings_are_the_plain_encoder():
    encoder = Pool3D(2)
    x = torch.rand(3, 2, 8, 8, 8)
    assert torch.equal(IngestBackbone(encoder, 2)(x), encoder(x))


def test_hemispheres_are_separate_passes_of_the_shared_encoder():
    encoder = Pool3D(1)
    x = torch.rand(3, 2, 8, 8, 8)
    out = IngestBackbone(encoder, 2, groups=1)(x)
    assert out.shape == (3, 4)
    expected = torch.cat([encoder(x[:, :1]), encoder(x[:, 1:])], dim=1)
    assert torch.allclose(out, expected)
    swapped = x.flip(1)  # left-right flip of the head = channel swap
    mean = IngestBackbone(encoder, 2, groups=1, reduce="mean")
    assert torch.allclose(mean(x), mean(swapped))


def test_region_pairs_of_a_multi_region_cache():
    encoder = Pool3D(2)
    x = torch.rand(2, 6, 8, 8, 8)
    out = IngestBackbone(encoder, 6, groups=2)(x)
    assert out.shape == (2, 9)
    assert torch.allclose(out[:, 3:6], encoder(x[:, 2:4]))


def test_scale_resamples_the_volume_before_a_3d_encoder():
    encoder = Pool3D(2)
    IngestBackbone(encoder, 2, scale=2)(torch.rand(2, 2, 16, 16, 16))
    assert encoder.seen[-1] == (2, 2, 32, 32, 32)


def test_normalizations():
    encoder = Pool3D(1)
    x = torch.rand(2, 1, 8, 8, 8) * 3 + 5
    z = IngestBackbone(encoder, 1, norm="zscore")
    assert torch.allclose(z(x)[:, 0], torch.zeros(2), atol=1e-5)
    imagenet = IngestBackbone(encoder, 1, norm="imagenet")
    grey_mean, grey_std = sum((0.485, 0.456, 0.406)) / 3, sum((0.229, 0.224, 0.225)) / 3
    assert torch.allclose(imagenet(x)[:, 0], (x.mean(dim=(1, 2, 3, 4)) - grey_mean) / grey_std,
                          atol=1e-5)


def test_slices_feed_neighbours_as_rgb_and_average_over_the_sequence():
    encoder = Pool2D(3)
    volume = torch.arange(4, dtype=torch.float32).view(1, 1, 4, 1, 1).expand(1, 2, 4, 2, 2)
    backbone = IngestBackbone(encoder, 2, two_d=True, groups=1, axes=(0,), context=3)
    out = backbone(volume.clone())
    # 2 hemispheres x 4 slices of 2x2 pixels, RGB = slices k-1, k, k+1 (edges repeated)
    assert encoder.seen[-1] == (8, 3, 2, 2)
    rgb_means = torch.tensor([[0, 0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 3]], dtype=torch.float32)
    assert torch.allclose(out[0, :3], rgb_means.mean(0))
    assert out.shape == (1, 8)
    grey = IngestBackbone(encoder, 2, two_d=True, groups=1, axes=(0,), context=1, stride=2,
                          slice_reduce="meanmax")
    out = grey(volume.clone())
    assert encoder.seen[-1] == (4, 3, 2, 2)  # every second slice
    assert out.shape == (1, 16)
    assert torch.allclose(out[0, :3], torch.full((3,), 1.0))      # mean of slices 0 and 2
    assert torch.allclose(out[0, 4:7], torch.full((3,), 2.0))     # max


def test_tri_planar_bag_and_in_plane_scale():
    encoder = Pool2D(3)
    backbone = IngestBackbone(encoder, 2, two_d=True, groups=1, axes=(0, 1, 2), scale=2)
    assert backbone(torch.rand(2, 2, 6, 6, 6)).shape == (2, 8)
    assert encoder.seen[-1] == (2 * 2 * 18, 3, 12, 12)


def test_checkpointed_chunks_give_the_same_embedding_and_gradients():
    torch.manual_seed(0)
    conv = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d(1),
                         nn.Flatten())
    conv.out_dim = 4
    x = torch.rand(2, 2, 6, 6, 6)
    grads = []
    for chunk in (0, 5):
        backbone = IngestBackbone(conv, 2, two_d=True, groups=1, axes=(1,), chunk=chunk).train()
        conv.zero_grad()
        out = backbone(x)
        out.square().sum().backward()
        grads.append((out.detach(), conv[0].weight.grad.clone()))
    assert torch.allclose(grads[0][0], grads[1][0], atol=1e-6)
    assert torch.allclose(grads[0][1], grads[1][1], atol=1e-6)


def test_two_d_needs_one_channel_per_pass():
    with pytest.raises(ValueError, match="groups=1"):
        IngestBackbone(Pool2D(3), 2, two_d=True)


def test_medicalnet_keeps_resolution_with_dilated_layers():
    for depth, width in ((18, 512), (10, 512), (50, 2048)):
        net = MedicalNetBackbone(depth=depth, in_channels=1)
        assert net.out_dim == width
        features = net.layer4(net.layer3(net.layer2(net.layer1(net.maxpool(
            net.relu(net.bn1(net.conv1(torch.rand(1, 1, 32, 32, 32)))))))))
        assert features.shape[-3:] == (4, 4, 4)


@pytest.mark.parametrize(("name", "out_dim"), [
    ("cnx3d_x2", 768), ("cnx3d_hemi_inorm", 1536), ("cnx3d_hemimean", 768),
    ("cnx2d_cor_x1", 1536), ("cnx2d_tri_x1_s2_mm", 3072), ("rn18_2d_ax_x1_c1", 1024),
])
def test_recipes_build_the_wrapped_backbones(name, out_dim, monkeypatch):
    from torchvision import models

    from t5_ssl.train import build_backbone
    real = {n: getattr(models, n) for n in ("convnext_tiny", "resnet18")}
    for n, fn in real.items():  # no ImageNet download in tests
        monkeypatch.setattr(models, n, lambda weights=None, _fn=fn: _fn(weights=None))
    GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(config_dir=_CONFIG_PATH, version_base=None):
        cfg = hydra.compose("train", overrides=recipes.overrides(
            "vicreg", "campaign_augval", backbone=name, accelerator="cpu"))
    GlobalHydra.instance().clear()
    backbone = build_backbone(cfg)
    assert isinstance(backbone, IngestBackbone)
    assert backbone.out_dim == out_dim
    with torch.no_grad():
        assert backbone.eval()(torch.rand(2, 2, 32, 32, 32)).shape == (2, out_dim)
