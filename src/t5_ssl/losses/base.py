"""Base interface for Self-Supervised Learning (SSL) losses."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseSSLLoss(nn.Module, ABC):
    """Abstract base class for all SSL loss functions.

    All SSL loss implementations must adhere to this interface to ensure
    modularity across backbones, projection heads, and training pipelines.
    """

    @abstractmethod
    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        h1: torch.Tensor | None = None,
        h2: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the SSL objective given projected and/or backbone representations.

        Parameters
        ----------
        z1:
            Projected embeddings for the first augmented view, shape ``(B, D)``.
        z2:
            Projected embeddings for the second augmented view, shape ``(B, D)``.
        h1:
            Optional raw backbone representations for view 1, shape ``(B, H)``.
        h2:
            Optional raw backbone representations for view 2, shape ``(B, H)``.

        Returns
        -------
        tuple[torch.Tensor, dict[str, torch.Tensor]]
            A tuple of ``(total_loss, metrics_dict)`` where:
              - ``total_loss``: Scalar tensor used for backward propagation.
              - ``metrics_dict``: Dictionary mapping metric names to scalar tensors
                (e.g., individual loss components for logging).

        """
        ...
