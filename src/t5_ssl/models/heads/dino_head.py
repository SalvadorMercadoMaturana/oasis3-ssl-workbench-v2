"""DINO/DINOv2 projection head: MLP, L2-normalized bottleneck and cosine prototypes."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DINOHead(nn.Module):
    """Map backbone tokens to prototype scores bounded in [-1, 1].

    As in DINO and DINOv2: an MLP (GELU) ends in a
    low-dimensional bottleneck that is L2-normalized and scored against
    ``out_dim`` unit-norm prototypes (a weight-normalized linear layer whose
    norm is fixed to 1). The scores are cosines, so the teacher and student
    temperatures act on a fixed scale: the network cannot flatten its softmax
    by shrinking its outputs, which is the uniform-collapse route that an
    unnormalized ``Linear -> GELU -> Linear`` head leaves open.

    ``use_bn`` adds batch normalization after each hidden layer (DINO's
    ``use_bn_in_head``, also an option of DINOv2's head). Standardizing each
    hidden unit over the batch removes the component shared by every input, so
    the prototype scores are driven by between-sample differences; without it,
    near-identical inputs (registered anatomy) leave the centered teacher
    targets almost uniform and the student collapses onto them.

    Parameters
    ----------
    in_dim:
        Backbone token dimension.
    out_dim:
        Number of prototypes ``K``.
    hidden_dim:
        Width of the hidden MLP layers.
    bottleneck_dim:
        Dimension of the normalized bottleneck.
    num_layers:
        Linear layers before the prototypes (1 = straight to the bottleneck).
    use_bn:
        Batch normalization after each hidden layer.

    """

    def __init__(self, in_dim: int, out_dim: int = 4096, hidden_dim: int = 2048,
                 bottleneck_dim: int = 256, num_layers: int = 3, use_bn: bool = False) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        def hidden(width_in: int) -> list[nn.Module]:
            block: list[nn.Module] = [nn.Linear(width_in, hidden_dim)]
            if use_bn:
                block.append(nn.BatchNorm1d(hidden_dim))
            return [*block, nn.GELU()]

        if num_layers == 1:
            layers: list[nn.Module] = [nn.Linear(in_dim, bottleneck_dim)]
        else:
            layers = hidden(in_dim)
            for _ in range(num_layers - 2):
                layers += hidden(hidden_dim)
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        self.mlp = nn.Sequential(*layers)
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)
        self.prototypes = nn.Parameter(torch.empty(out_dim, bottleneck_dim))
        nn.init.trunc_normal_(self.prototypes, std=0.02)
        self.in_dim, self.out_dim = in_dim, out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return cosine scores ``(N, out_dim)`` of ``(N, in_dim)`` tokens."""
        x = F.normalize(self.mlp(x), dim=-1)
        return F.linear(x, F.normalize(self.prototypes, dim=-1))

    def last_layer_parameters(self) -> list[nn.Parameter]:
        """Prototype weights, frozen during the first epoch(s) as in DINO."""
        return [self.prototypes]

    def __repr__(self) -> str:
        """Return developer-friendly string representation."""
        return (f"{self.__class__.__name__}(in_dim={self.in_dim}, "
                f"bottleneck={self.prototypes.shape[1]}, prototypes={self.out_dim})")
