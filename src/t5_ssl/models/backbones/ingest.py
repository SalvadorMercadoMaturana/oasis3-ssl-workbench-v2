"""How a cached ROI volume reaches the encoder: the ingestion settings tuned on 2026-11-22.

Every method (VICReg, BYOL, DINOv2, rotation pretext), the frozen start (sin SSL), the
supervised baseline and the probes call ``backbone(x)`` on ``(B, C, D, H, W)`` ROI
volumes (``t5_ssl.data.roi3d``: 2 mm MNI cubes, channels ``[left, right mirrored]`` per
region). :class:`IngestBackbone` wraps one encoder and decides how those volumes are fed
to it, so the views, losses and probes stay unchanged:

- ``norm``: ``none`` (the [0, 1] cache intensities, as every campaign so far),
  ``imagenet`` (the ImageNet statistics the pretrained weights expect; per RGB channel
  for 2D encoders, their grey mean for 3D ones) or ``zscore`` (each channel of each
  sample standardized over its voxels, as MedicalNet was trained).
- ``groups``: input channels per encoder pass. ``0`` = all channels in one pass (the
  2-channel hemisphere pair, as before). ``1`` = every hemisphere (and every region of a
  multi-region cache) is its own 1-channel volume through the shared encoder; ``2`` = one
  pass per bilateral region pair. ``reduce`` joins the passes: ``concat`` (keeps which
  hemisphere / region each feature came from) or ``mean``.
- ``scale``: trilinear resampling of the volume before a 3D encoder (``2`` = 1 mm voxels:
  the ConvNeXt-T stem divides by 4 and the network by 32, so a 32-voxel cube ends as one
  voxel at scale 1 and as 2^3 at scale 2); for a 2D encoder, the in-plane resampling of
  every slice.
- ``slices`` (2D encoders only): the volume becomes a sequence of 2D slices along
  ``axes`` (0 sagittal, 1 coronal, 2 axial; several = a bag of all their slices), every
  ``stride``-th slice. ``context`` 3 feeds slices k-1, k, k+1 as the RGB channels (2.5D),
  1 repeats the slice in the three channels. The encoder embeds every slice and the
  slice embeddings are averaged (``slice_reduce: mean``) or averaged and max-pooled
  (``meanmax``, twice the width). ``chunk`` > 0 runs the 2D encoder in chunks of that
  many slices under activation checkpointing while training (same result, less memory:
  a batch of 64 views x 2 hemispheres x 32 slices is 4096 images).

The defaults (``norm none``, ``groups 0``, ``scale 1``, no slices) are the identity: the
wrapped encoder sees exactly what it saw before.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from t5_ssl.models.backbones.base import BaseBackbone

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NORMS = ("none", "imagenet", "zscore")
REDUCES = ("concat", "mean")
SLICE_REDUCES = ("mean", "meanmax")
# Slices per encoder call without gradients (probe embeddings: 64 sessions x 2 hemispheres x
# 96 slices would be 12288 images in one call).
EVAL_CHUNK = 1024


def encoder_in_channels(spec: dict[str, Any], in_channels: int, two_d: bool) -> int:
    """Input channels of the wrapped encoder for ``in_channels`` ROI channels."""
    if two_d:
        return 3
    groups = int(spec.get("groups", 0) or 0)
    return groups or in_channels


class IngestBackbone(BaseBackbone):
    """``(B, C, D, H, W)`` ROI volumes -> ``(B, out_dim)`` through one wrapped encoder."""

    def __init__(self, encoder: nn.Module, in_channels: int, *, two_d: bool = False,
                 norm: str = "none", groups: int = 0, reduce: str = "concat",
                 scale: float = 1.0, axes: tuple[int, ...] | list[int] = (1,),
                 context: int = 3, stride: int = 1, slice_reduce: str = "mean",
                 chunk: int = 0) -> None:
        super().__init__()
        if norm not in NORMS:
            raise ValueError(f"ingest norm must be one of {NORMS}, got {norm!r}")
        if reduce not in REDUCES:
            raise ValueError(f"ingest reduce must be one of {REDUCES}, got {reduce!r}")
        if slice_reduce not in SLICE_REDUCES:
            raise ValueError(f"ingest slice_reduce must be one of {SLICE_REDUCES}")
        groups = int(groups or 0)
        if two_d and groups != 1:
            raise ValueError("2D slice ingestion embeds one channel at a time: set groups=1")
        if groups and in_channels % groups:
            raise ValueError(f"{in_channels} input channels do not split into groups of {groups}")
        if context not in (1, 3):
            raise ValueError("ingest context must be 1 (grey slice) or 3 (k-1, k, k+1)")
        axes = tuple(int(a) for a in axes)
        if two_d and (not axes or any(a not in (0, 1, 2) for a in axes)):
            raise ValueError(f"ingest axes must be a non-empty subset of (0, 1, 2), got {axes}")
        self.encoder, self.in_channels, self.two_d = encoder, int(in_channels), bool(two_d)
        self.norm, self.groups, self.reduce = norm, groups, reduce
        self.scale, self.axes, self.context = float(scale), axes, int(context)
        self.stride, self.slice_reduce = max(1, int(stride)), slice_reduce
        self.chunk = max(0, int(chunk))
        self.passes = self.in_channels // groups if groups else 1
        width = int(encoder.out_dim) * (2 if two_d and slice_reduce == "meanmax" else 1)
        self._out_dim = width * (self.passes if reduce == "concat" else 1)
        grey = sum(IMAGENET_MEAN) / 3, sum(IMAGENET_STD) / 3
        shape = (1, 3, 1, 1) if two_d else (1, 1, 1, 1, 1)
        mean = torch.tensor(IMAGENET_MEAN if two_d else grey[:1]).view(shape)
        std = torch.tensor(IMAGENET_STD if two_d else grey[1:]).view(shape)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def _standardize(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm == "zscore":
            dims = tuple(range(2, x.ndim))
            mean = x.mean(dim=dims, keepdim=True)
            std = x.std(dim=dims, keepdim=True).clamp_min(1e-4)
            return (x - mean) / std
        if self.norm == "imagenet":
            return (x - self.mean.to(x.dtype)) / self.std.to(x.dtype)
        return x

    def _slice_stack(self, volume: torch.Tensor, axis: int) -> torch.Tensor:
        """``(N, D, H, W)`` -> ``(N, S, 3, h, w)`` slices along ``axis`` (every ``stride``-th)."""
        v = volume.movedim(axis + 1, 1)
        n_slices = v.shape[1]
        keep = torch.arange((n_slices - 1) % self.stride // 2, n_slices, self.stride,
                            device=v.device)
        if self.context == 3:
            padded = torch.cat([v[:, :1], v, v[:, -1:]], dim=1)
            stack = torch.stack([padded[:, keep], padded[:, keep + 1], padded[:, keep + 2]], dim=2)
        else:
            stack = v[:, keep].unsqueeze(2).expand(-1, -1, 3, -1, -1)
        return stack

    def _embed_slices(self, x: torch.Tensor) -> torch.Tensor:
        """``(N, 1, D, H, W)`` -> ``(N, width)``: every slice through the 2D encoder."""
        volume = x[:, 0]
        cubic = len(set(volume.shape[1:])) == 1
        stacks = [self._slice_stack(volume, axis) for axis in self.axes]
        if cubic:  # same in-plane shape on every axis: one encoder call
            stacks = [torch.cat(stacks, dim=1)]
        features = []
        for stack in stacks:
            n, s = stack.shape[:2]
            images = stack.reshape(n * s, *stack.shape[2:])
            if self.scale != 1:
                images = F.interpolate(images, scale_factor=self.scale, mode="bilinear",
                                       align_corners=False)
            images = self._standardize(images).contiguous(memory_format=torch.channels_last)
            if self.chunk and self.training and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                out = torch.cat([checkpoint(self.encoder, part, use_reentrant=False)
                                 for part in images.split(self.chunk)])
            elif not torch.is_grad_enabled():  # embeddings: bounded memory, same result
                out = torch.cat([self.encoder(part) for part in images.split(EVAL_CHUNK)])
            else:
                out = self.encoder(images)
            features.append(out.reshape(n, s, -1))
        tokens = torch.cat(features, dim=1)
        pooled = tokens.mean(dim=1)
        if self.slice_reduce == "meanmax":
            pooled = torch.cat([pooled, tokens.amax(dim=1)], dim=1)
        return pooled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, C, D, H, W)`` volumes to ``(B, out_dim)`` embeddings."""
        if x.ndim != 5 or x.shape[1] != self.in_channels:
            raise ValueError(f"expected (B, {self.in_channels}, D, H, W), got {tuple(x.shape)}")
        batch = x.shape[0]
        if self.groups:
            x = x.reshape(batch * self.passes, self.groups, *x.shape[2:])
        if self.two_d:
            h = self._embed_slices(x)
        else:
            if self.scale != 1:
                x = F.interpolate(x, scale_factor=self.scale, mode="trilinear",
                                  align_corners=False)
            x = self._standardize(x)
            if x.is_cuda:
                x = x.contiguous(memory_format=torch.channels_last_3d)
            h = self.encoder(x)
        h = h.reshape(batch, self.passes, -1)
        return h.flatten(1) if self.reduce == "concat" else h.mean(dim=1)

    def __repr__(self) -> str:
        slices = (f", axes={self.axes}, context={self.context}, stride={self.stride}, "
                  f"slice_reduce={self.slice_reduce}") if self.two_d else ""
        return (f"IngestBackbone(encoder={type(self.encoder).__name__}, norm={self.norm}, "
                f"groups={self.groups}, reduce={self.reduce}, scale={self.scale}{slices}, "
                f"out_dim={self.out_dim})")
