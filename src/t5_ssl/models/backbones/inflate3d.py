"""Inflate torchvision 2D ImageNet networks to 3D (I3D-style).

Every ``Conv2d`` k x k becomes a ``Conv3d`` k x k x k whose kernel is the 2D
kernel repeated along depth and divided by k, so a volume that is constant
along depth reproduces the 2D activations of its slices. Normalizations and
channel-last permutes are converted to their 5D counterparts.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.convnext import LayerNorm2d
from torchvision.ops.misc import Permute

_PERMUTES = {(0, 2, 3, 1): [0, 2, 3, 4, 1], (0, 3, 1, 2): [0, 4, 1, 2, 3]}


class LayerNorm3d(nn.LayerNorm):
    """LayerNorm over channels of a (B, C, D, H, W) tensor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 4, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.permute(0, 4, 1, 2, 3)


def inflate_conv(conv: nn.Conv2d, in_channels: int | None = None) -> nn.Conv3d:
    """Conv2d -> Conv3d; `in_channels` re-targets a stem from RGB to grey channels."""
    k = conv.kernel_size[0]
    new = nn.Conv3d(in_channels or conv.in_channels, conv.out_channels, k,
                    stride=conv.stride[0], padding=conv.padding[0], dilation=conv.dilation[0],
                    groups=conv.groups, bias=conv.bias is not None)
    with torch.no_grad():
        weight = conv.weight.unsqueeze(2).repeat(1, 1, k, 1, 1) / k
        if in_channels is not None and in_channels != conv.in_channels:
            # Grey inputs: every new channel gets the RGB-summed kernel, shared so
            # the total response to a grey volume matches the RGB network's.
            weight = weight.sum(dim=1, keepdim=True).repeat(1, in_channels, 1, 1, 1) / in_channels
        new.weight.copy_(weight)
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new


def inflate_norm(norm: nn.Module) -> nn.Module:
    if isinstance(norm, nn.BatchNorm2d):
        new = nn.BatchNorm3d(norm.num_features, eps=norm.eps, momentum=norm.momentum)
    else:
        new = LayerNorm3d(norm.normalized_shape, eps=norm.eps)
    new.load_state_dict(norm.state_dict())
    return new


def inflate_module(module: nn.Module) -> None:
    """Convert every 2D layer below `module` in place."""
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            setattr(module, name, inflate_conv(child))
        elif isinstance(child, nn.BatchNorm2d | LayerNorm2d):
            setattr(module, name, inflate_norm(child))
        elif isinstance(child, Permute) and tuple(child.dims) in _PERMUTES:
            setattr(module, name, Permute(_PERMUTES[tuple(child.dims)]))
        elif isinstance(child, nn.AdaptiveAvgPool2d):
            setattr(module, name, nn.AdaptiveAvgPool3d(1))
        else:
            inflate_module(child)
    # ConvNeXt blocks scale their residual branch by a (C, 1, 1) parameter.
    scale = getattr(module, "layer_scale", None)
    if isinstance(scale, nn.Parameter) and scale.ndim == 3:
        module.layer_scale = nn.Parameter(scale.detach().unsqueeze(-1))
