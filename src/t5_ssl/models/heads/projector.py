"""Multi-Layer Perceptron (MLP) projection head for Self-Supervised Learning."""

from __future__ import annotations

import torch
from torch import nn


class MLPProjector(nn.Module):
    """Non-linear projection head mapping representations to the SSL loss space.

    Architecture follows the standard VICReg / Barlow Twins design:
    Linear -> BatchNorm1d -> ReLU/GELU -> ... -> Linear (final layer without activation).

    Parameters
    ----------
    in_dim:
        Input dimension matching the backbone embedding size (e.g., 768 for ConvNeXt-Tiny).
    hidden_dim:
        Dimension of intermediate projection layers.
        Default: ``2048``.
    out_dim:
        Dimension of final projection output embeddings.
        Default: ``2048``.
    num_layers:
        Total number of linear layers (minimum 2).
        Default: ``3``.
    use_bn:
        Whether to apply batch normalization between hidden layers.
        Default: ``True``.
    activation:
        Non-linear activation function: ``"relu"`` or ``"gelu"``.
        Default: ``"relu"``.
    dropout:
        Dropout probability applied after each hidden activation. ``0.0`` disables it.
        Default: ``0.0``.

    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 2048,
        out_dim: int = 2048,
        num_layers: int = 3,
        use_bn: bool = True,
        activation: str = "relu",
        norm: str | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            msg = f"num_layers must be >= 2, got {num_layers}"
            raise ValueError(msg)

        # `norm` supersedes the legacy `use_bn` flag when given explicitly.
        # Batch statistics computed on tiny batches are a known collapse driver
        # for BYOL here, so group/layer norm are available as a batch-size
        # independent alternative for the online projector/predictor.
        norm = norm if norm is not None else ("batch" if use_bn else "none")
        if norm not in ("batch", "group", "layer", "none"):
            raise ValueError("norm must be one of 'batch', 'group', 'layer', 'none'")
        has_bias = norm == "none"

        def make_norm(dim: int) -> nn.Module:
            if norm == "batch":
                return nn.BatchNorm1d(dim)
            if norm == "group":
                groups = min(32, dim)
                while dim % groups != 0:
                    groups -= 1
                return nn.GroupNorm(groups, dim)
            return nn.LayerNorm(dim)

        act_cls = nn.GELU if activation.lower() == "gelu" else nn.ReLU

        layers: list[nn.Module] = []
        # Input layer
        layers.append(nn.Linear(in_dim, hidden_dim, bias=has_bias))
        if norm != "none":
            layers.append(make_norm(hidden_dim))
        layers.append(act_cls())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))

        # Intermediate hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=has_bias))
            if norm != "none":
                layers.append(make_norm(hidden_dim))
            layers.append(act_cls())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

        # Output projection layer (no norm or activation per VICReg specification)
        layers.append(nn.Linear(hidden_dim, out_dim, bias=True))

        self.net = nn.Sequential(*layers)
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project backbone embeddings into the SSL criterion space.

        Parameters
        ----------
        x:
            Backbone representations of shape ``(B, in_dim)``.

        Returns
        -------
        torch.Tensor
            Projected embeddings of shape ``(B, out_dim)``.

        """
        return self.net(x)

    def __repr__(self) -> str:
        """Return developer-friendly string representation."""
        return (
            f"{self.__class__.__name__}("
            f"in_dim={self.in_dim}, "
            f"out_dim={self.out_dim}, "
            f"layers={len(self.net)})"
        )
