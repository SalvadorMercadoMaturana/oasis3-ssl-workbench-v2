"""MedicalNet 3D ResNets (Med3D, Chen et al. 2019) with their medical pretraining.

Tencent's MedicalNet trained 3D ResNets as segmentation encoders on 23 CT and MRI
datasets (``resnet_<depth>_23dataset.pth``, MIT licence, mirrored on HuggingFace by
``TencentMedicalNet/MedicalNet-Resnet<depth>``). Unlike the ImageNet ConvNeXt / ResNet
inflated to 3D (``resnet3d``), these kernels were learned on volumes. The architecture is
reproduced here so the checkpoints load key for key:

- stem: 7^3 convolution stride 2, BatchNorm, ReLU, 3^3 max-pool stride 2;
- layer2 stride 2; layer3 and layer4 stride 1 with dilation 2 and 4 (segmentation
  encoders keep resolution: a 32^3 cube ends as 4^3 x 512 instead of 1^3);
- shortcut ``A`` in the released ResNet-18/34 (1x1x1 average pooling with the block's
  stride, then zero channels), ``B`` in ResNet-10/50 (1x1x1 convolution + BatchNorm).

MedicalNet volumes were z-scored per scan: pair it with ``ingest.norm=zscore``. The
segmentation head (``conv_seg``) is dropped; the embedding is the global average of
layer4. Weights: ``model.backbone.pretrained=medicalnet`` downloads
``resnet_<depth>_23dataset.pth`` into ``$TORCH_HOME/medicalnet`` (or reads
``$MEDICALNET_DIR``).
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import ClassVar

import torch
import torch.nn.functional as F
from torch import nn

from t5_ssl.models.backbones.base import BaseBackbone

URL = "https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet{depth}/resolve/main/{name}"


def _conv3(in_planes: int, out_planes: int, stride: int = 1, dilation: int = 1) -> nn.Conv3d:
    return nn.Conv3d(in_planes, out_planes, 3, stride=stride, padding=dilation,
                     dilation=dilation, bias=False)


class ShortcutA(nn.Module):
    """Parameter-free shortcut: stride by 1x1x1 average pooling, then zero channels."""

    def __init__(self, planes: int, stride: int) -> None:
        super().__init__()
        self.planes, self.stride = planes, stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.avg_pool3d(x, kernel_size=1, stride=self.stride)
        return F.pad(out, (0, 0, 0, 0, 0, 0, 0, self.planes - out.shape[1]))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, dilation: int = 1,
                 downsample: nn.Module | None = None) -> None:
        super().__init__()
        self.conv1 = _conv3(inplanes, planes, stride, dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        residual = x if self.downsample is None else self.downsample(x)
        return self.relu(out + residual)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, dilation: int = 1,
                 downsample: nn.Module | None = None) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = _conv3(planes, planes, stride, dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        residual = x if self.downsample is None else self.downsample(x)
        return self.relu(out + residual)


class MedicalNetBackbone(BaseBackbone):
    """MedicalNet 3D ResNet encoder for ``(B, C, D, H, W)`` volumes."""

    # depth -> (block, layers, shortcut of the released 23-dataset checkpoint)
    SPECS: ClassVar[dict[int, tuple[type, list[int], str]]] = {
        10: (BasicBlock, [1, 1, 1, 1], "B"),
        18: (BasicBlock, [2, 2, 2, 2], "A"),
        34: (BasicBlock, [3, 4, 6, 3], "A"),
        50: (Bottleneck, [3, 4, 6, 3], "B"),
    }

    def __init__(self, depth: int = 18, in_channels: int = 1, pretrained: bool = False) -> None:
        super().__init__()
        if depth not in self.SPECS:
            raise ValueError(f"MedicalNet depth must be one of {sorted(self.SPECS)}")
        block, layers, self.shortcut = self.SPECS[depth]
        self.depth, self.in_channels = depth, in_channels
        self.inplanes = 64
        self.conv1 = nn.Conv3d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(3, stride=2, padding=1)
        self.layer1 = self._layer(block, 64, layers[0])
        self.layer2 = self._layer(block, 128, layers[1], stride=2)
        self.layer3 = self._layer(block, 256, layers[2], dilation=2)
        self.layer4 = self._layer(block, 512, layers[3], dilation=4)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self._out_dim = 512 * block.expansion
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if pretrained:
            self.load_medicalnet()

    def _layer(self, block: type, planes: int, blocks: int, stride: int = 1,
               dilation: int = 1) -> nn.Sequential:
        downsample: nn.Module | None = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.shortcut == "A":
                downsample = ShortcutA(planes * block.expansion, stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion, 1, stride=stride,
                              bias=False),
                    nn.BatchNorm3d(planes * block.expansion))
        layers = [block(self.inplanes, planes, stride, dilation, downsample)]
        self.inplanes = planes * block.expansion
        layers += [block(self.inplanes, planes, dilation=dilation) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def checkpoint_path(self) -> Path:
        name = f"resnet_{self.depth}_23dataset.pth"
        folder = Path(os.environ.get("MEDICALNET_DIR") or
                      Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch"))
                      / "medicalnet")
        path = folder / name
        if not path.is_file():
            folder.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".part")
            urllib.request.urlretrieve(URL.format(depth=self.depth, name=name), tmp)
            tmp.replace(path)
        return path

    def load_medicalnet(self) -> None:
        # weights_only: a downloaded file is never unpickled as arbitrary objects
        state = torch.load(self.checkpoint_path(), map_location="cpu", weights_only=True)
        state = state.get("state_dict", state)
        state = {k.removeprefix("module."): v for k, v in state.items()
                 if not k.removeprefix("module.").startswith("conv_seg")}
        stem = state["conv1.weight"]
        if stem.shape[1] != self.in_channels:  # grey kernel shared by every input channel
            state["conv1.weight"] = stem.sum(1, keepdim=True).repeat(
                1, self.in_channels, 1, 1, 1) / self.in_channels
        missing, unexpected = self.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"MedicalNet-{self.depth} checkpoint mismatch: missing "
                               f"{missing[:5]}, unexpected {unexpected[:5]}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return torch.flatten(self.avgpool(x), 1)
