"""Abstract base class for feature extraction backbones."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseBackbone(nn.Module, ABC):
    """Base interface for computer vision backbones used in SSL frameworks."""

    @property
    @abstractmethod
    def out_dim(self) -> int:
        """Return the output embedding feature dimension."""
        ...

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract a 1D embedding vector representation per image.

        Parameters
        ----------
        x:
            Input tensor of shape ``(B, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Feature representation tensor of shape ``(B, out_dim)``.

        """
        ...
