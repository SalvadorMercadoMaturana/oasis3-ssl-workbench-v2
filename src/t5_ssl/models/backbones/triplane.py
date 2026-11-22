"""One 2D encoder on the axial, coronal and sagittal planes of a brain (2026-11-22).

Input ``(B, P, H, W)``: the ``P`` planes of each sample (t5_ssl.data.brain2d: axial,
coronal, sagittal, z-scored over tissue, ``224 x 224``). Each plane is repeated on the
three RGB inputs of the shared 2D encoder (ImageNet ConvNeXt-T in the campaigns) and the
``P`` embeddings are joined:

- ``concat`` (default): keeps which plane each feature came from (``P x`` width);
- ``mean``: one encoder-wide vector.

``norm``: ``none`` (the z-scored intensities as they are) or ``imagenet`` (the ImageNet
per-channel statistics on top). Without gradients the planes go through the encoder in
chunks of ``EVAL_CHUNK`` images (probe embeddings).
"""

from __future__ import annotations

import torch
from torch import nn

from t5_ssl.models.backbones.base import BaseBackbone
from t5_ssl.models.backbones.ingest import IMAGENET_MEAN, IMAGENET_STD

REDUCES = ("concat", "mean")
NORMS = ("none", "imagenet")
EVAL_CHUNK = 768


class TriPlaneBackbone(BaseBackbone):
    """``(B, P, H, W)`` planes -> ``(B, out_dim)`` through one shared 2D encoder."""

    def __init__(self, encoder: nn.Module, planes: int = 3, reduce: str = "concat",
                 norm: str = "none") -> None:
        super().__init__()
        if reduce not in REDUCES:
            raise ValueError(f"triplane reduce must be one of {REDUCES}, got {reduce!r}")
        if norm not in NORMS:
            raise ValueError(f"triplane norm must be one of {NORMS}, got {norm!r}")
        self.encoder, self.planes, self.reduce, self.norm = encoder, int(planes), reduce, norm
        width = int(encoder.out_dim)
        self._out_dim = width * self.planes if reduce == "concat" else width
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, P, H, W)`` -> ``(B, out_dim)``; ``(B, k P, H, W)``: mean over the ``k`` groups.

        ``k`` groups of planes are a multi-slice probe input (t5_ssl.data.brain2d.TriPlaneEval
        with several offsets): each group is embedded as one sample and the ``k``
        embeddings are averaged.
        """
        if x.ndim != 4 or x.shape[1] % self.planes:
            raise ValueError(f"expected (B, {self.planes}, H, W) planes, got {tuple(x.shape)}")
        groups = x.shape[1] // self.planes
        if groups > 1:
            out = self(x.reshape(x.shape[0] * groups, self.planes, *x.shape[2:]))
            return out.reshape(x.shape[0], groups, -1).mean(dim=1)
        b, p, h, w = x.shape
        images = x.reshape(b * p, 1, h, w).expand(-1, 3, -1, -1)
        if self.norm == "imagenet":
            images = (images - self.mean.to(images.dtype)) / self.std.to(images.dtype)
        images = images.contiguous(memory_format=torch.channels_last)
        if torch.is_grad_enabled():
            features = self.encoder(images)
        else:  # probe embeddings: bounded memory, same result
            features = torch.cat([self.encoder(part) for part in images.split(EVAL_CHUNK)])
        features = features.reshape(b, p, -1)
        return features.flatten(1) if self.reduce == "concat" else features.mean(dim=1)

    def __repr__(self) -> str:
        return (f"TriPlaneBackbone(encoder={type(self.encoder).__name__}, planes={self.planes}, "
                f"reduce={self.reduce}, norm={self.norm}, out_dim={self.out_dim})")
