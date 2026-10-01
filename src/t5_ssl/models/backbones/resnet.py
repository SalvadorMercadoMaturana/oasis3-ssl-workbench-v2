"""ResNet backbone adapter for 2D medical imaging SSL.

Wraps torchvision ResNet architectures and provides flexible input channel
configuration, matching the BYOLv4 configuration from the EL7006 project report
(ResNet-18 with ImageNet1K pretrained weights, chosen there for more stable
training than a ViT-Tiny16 backbone).
"""

from __future__ import annotations

from typing import ClassVar

import torch
from torch import nn
from torchvision import models

from t5_ssl.models.backbones.base import BaseBackbone


class ResNetBackbone(BaseBackbone):
    """ResNet feature extractor with configurable input channels.

    Parameters
    ----------
    variant:
        ResNet size variant: ``"resnet18"``, ``"resnet34"``, or ``"resnet50"``.
        Default: ``"resnet18"``.
    in_channels:
        Number of input image channels (1 axial slice or 3 neighboring slices).
        Default: ``1``.
    pretrained:
        Whether to load torchvision pre-trained ImageNet1K weights.
        Default: ``False``.

    """

    _DIMS: ClassVar[dict[str, int]] = {
        "resnet18": 512,
        "resnet34": 512,
        "resnet50": 2048,
    }

    def __init__(
        self,
        variant: str = "resnet18",
        in_channels: int = 1,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        variant = variant.lower()
        if variant not in self._DIMS:
            msg = f"Unknown ResNet variant '{variant}'. Choose from {list(self._DIMS.keys())}."
            raise ValueError(msg)

        self.variant = variant
        self.in_channels = in_channels
        self._out_dim = self._DIMS[variant]

        weights = "DEFAULT" if pretrained else None
        model_fn = getattr(models, variant)
        base_model = model_fn(weights=weights)

        # Adapt the stem convolution if in_channels != 3
        if in_channels != 3:
            first_conv: nn.Conv2d = base_model.conv1
            new_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias is not None,
            )

            if pretrained:
                with torch.no_grad():
                    orig_w = first_conv.weight.data  # (out_c, 3, k, k)
                    if in_channels < 3:
                        new_conv.weight.data.copy_(orig_w[:, :in_channels, :, :])
                    else:
                        repeat_times = (in_channels + 2) // 3
                        repeated = orig_w.repeat(1, repeat_times, 1, 1)[:, :in_channels, :, :]
                        new_conv.weight.data.copy_(repeated * (3.0 / in_channels))
                    if first_conv.bias is not None and new_conv.bias is not None:
                        new_conv.bias.data.copy_(first_conv.bias.data)

            base_model.conv1 = new_conv

        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool

    @property
    def out_dim(self) -> int:
        """Return the output feature embedding dimension."""
        return self._out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract global average pooled representations.

        Parameters
        ----------
        x:
            Input tensor of shape ``(B, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Embeddings of shape ``(B, out_dim)``.

        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def __repr__(self) -> str:
        """Return developer-friendly string representation."""
        return (
            f"{self.__class__.__name__}("
            f"variant='{self.variant}', "
            f"in_channels={self.in_channels}, "
            f"out_dim={self.out_dim})"
        )
