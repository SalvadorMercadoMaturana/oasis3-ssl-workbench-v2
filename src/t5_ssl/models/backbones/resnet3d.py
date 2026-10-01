"""3D ResNet and ConvNeXt backbones inflated from torchvision's 2D ImageNet weights.

See ``inflate3d`` for the inflation rule. The ResNet stem max-pool is dropped
because the hippocampal ROI cubes are small (32-48 voxels per side).
"""

from __future__ import annotations

from typing import ClassVar

import torch
from torch import nn
from torchvision import models

from t5_ssl.models.backbones.base import BaseBackbone
from t5_ssl.models.backbones.inflate3d import inflate_conv, inflate_module, inflate_norm


class ResNet3DBackbone(BaseBackbone):
    """ResNet feature extractor for (B, C, D, H, W) volumes."""

    _DIMS: ClassVar[dict[str, int]] = {"resnet18": 512, "resnet34": 512}

    def __init__(self, variant: str = "resnet18", in_channels: int = 1,
                 pretrained: bool = False) -> None:
        super().__init__()
        variant = variant.lower().removesuffix("_3d")
        if variant not in self._DIMS:
            raise ValueError(f"Unknown 3D ResNet variant '{variant}'. "
                             f"Choose from {[v + '_3d' for v in self._DIMS]}.")
        self.variant, self.in_channels, self._out_dim = variant, in_channels, self._DIMS[variant]
        base = getattr(models, variant)(weights="DEFAULT" if pretrained else None)
        self.conv1 = inflate_conv(base.conv1, in_channels)
        self.bn1 = inflate_norm(base.bn1)
        self.relu = base.relu
        for name in ("layer1", "layer2", "layer3", "layer4"):
            layer = getattr(base, name)
            inflate_module(layer)
            setattr(self, name, layer)
        self.avgpool = nn.AdaptiveAvgPool3d(1)

    @property
    def out_dim(self) -> int:
        """Return the output feature embedding dimension."""
        return self._out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, C, D, H, W)`` volumes to ``(B, out_dim)`` embeddings."""
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return torch.flatten(self.avgpool(x), 1)


class ConvNeXt3DBackbone(BaseBackbone):
    """ConvNeXt feature extractor for (B, C, D, H, W) volumes (stem stride 4)."""

    _DIMS: ClassVar[dict[str, int]] = {"convnext_tiny": 768, "convnext_small": 768}

    def __init__(self, variant: str = "convnext_tiny", in_channels: int = 1,
                 pretrained: bool = False) -> None:
        super().__init__()
        variant = variant.lower().removesuffix("_3d")
        if variant not in self._DIMS:
            raise ValueError(f"Unknown 3D ConvNeXt variant '{variant}'. "
                             f"Choose from {[v + '_3d' for v in self._DIMS]}.")
        self.variant, self.in_channels, self._out_dim = variant, in_channels, self._DIMS[variant]
        base = getattr(models, variant)(weights="DEFAULT" if pretrained else None)
        base.features[0][0] = inflate_conv(base.features[0][0], in_channels)
        inflate_module(base.features)
        self.features = base.features
        self.avgpool = nn.AdaptiveAvgPool3d(1)

    @property
    def out_dim(self) -> int:
        """Return the output feature embedding dimension."""
        return self._out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, C, D, H, W)`` volumes to ``(B, out_dim)`` embeddings."""
        return torch.flatten(self.avgpool(self.features(x)), 1)
