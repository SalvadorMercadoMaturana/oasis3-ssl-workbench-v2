"""ConvNeXt backbone adapter for 2D medical imaging SSL.

Wraps torchvision ConvNeXt architectures (tiny, small, base, large) and provides
flexible input channel configuration for multi-modal MRI scans.
"""

from __future__ import annotations

from typing import ClassVar

import torch
from torch import nn
from torchvision import models

from t5_ssl.models.backbones.base import BaseBackbone


class ConvNeXtBackbone(BaseBackbone):
    """ConvNeXt feature extractor with configurable input channels.

    Parameters
    ----------
    variant:
        ConvNeXt size variant: ``"convnext_tiny"``, ``"convnext_small"``,
        ``"convnext_base"``, or ``"convnext_large"``.
        Default: ``"convnext_tiny"``.
    in_channels:
        Number of input image channels (1 axial slice or 3 neighboring slices).
        Default: ``1``.
    pretrained:
        Whether to load torchvision pre-trained ImageNet weights.
        Default: ``False``.

    """

    _DIMS: ClassVar[dict[str, int]] = {
        "convnext_tiny": 768,
        "convnext_small": 768,
        "convnext_base": 1024,
        "convnext_large": 1536,
    }

    def __init__(
        self,
        variant: str = "convnext_tiny",
        in_channels: int = 1,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        variant = variant.lower()
        if variant not in self._DIMS:
            msg = f"Unknown ConvNeXt variant '{variant}'. Choose from {list(self._DIMS.keys())}."
            raise ValueError(msg)

        self.variant = variant
        self.in_channels = in_channels
        self._out_dim = self._DIMS[variant]

        # Instantiate torchvision model
        weights = "DEFAULT" if pretrained else None
        model_fn = getattr(models, variant)
        base_model = model_fn(weights=weights)

        # Adapt first convolution layer if in_channels != 3
        if in_channels != 3:
            first_conv: nn.Conv2d = base_model.features[0][0]
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
                    # Replicate or average pretrained weights across new channels
                    orig_w = first_conv.weight.data  # (out_c, 3, k, k)
                    if in_channels < 3:
                        new_conv.weight.data.copy_(orig_w[:, :in_channels, :, :])
                    else:
                        repeat_times = (in_channels + 2) // 3
                        repeated = orig_w.repeat(1, repeat_times, 1, 1)[:, :in_channels, :, :]
                        # Rescale weights so overall activation magnitude is preserved
                        new_conv.weight.data.copy_(repeated * (3.0 / in_channels))
                    if first_conv.bias is not None and new_conv.bias is not None:
                        new_conv.bias.data.copy_(first_conv.bias.data)

            base_model.features[0][0] = new_conv

        self.features = base_model.features
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
        feat = self.features(x)
        pooled = self.avgpool(feat)
        return torch.flatten(pooled, 1)

    def __repr__(self) -> str:
        """Return developer-friendly string representation."""
        return (
            f"{self.__class__.__name__}("
            f"variant='{self.variant}', "
            f"in_channels={self.in_channels}, "
            f"out_dim={self.out_dim})"
        )
