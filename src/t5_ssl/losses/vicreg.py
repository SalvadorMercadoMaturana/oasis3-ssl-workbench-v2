"""VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning.

Reference:
    Bardes, A., Ponce, J., & LeCun, Y. (2021).
    VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning.
    arXiv preprint arXiv:2105.04906.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from t5_ssl.losses.base import BaseSSLLoss


class VICRegLoss(BaseSSLLoss):
    """VICReg criterion combining Invariance, Variance, and Covariance regularization.

    Parameters
    ----------
    sim_coeff:
        Weight factor for the invariance (mean squared error) criterion.
        Default: ``25.0``.
    var_coeff:
        Weight factor for the variance (hinge standard deviation) criterion.
        Default: ``25.0``.
    cov_coeff:
        Weight factor for the covariance (decorrelation) criterion.
        Default: ``1.0``.
    gamma:
        Target standard deviation threshold for variance regularization.
        Default: ``1.0``.
    eps:
        Small constant added to variance for numerical stability.
        Default: ``1e-4``.

    """

    def __init__(
        self,
        sim_coeff: float = 25.0,
        var_coeff: float = 25.0,
        cov_coeff: float = 1.0,
        gamma: float = 1.0,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        self.sim_coeff = sim_coeff
        self.var_coeff = var_coeff
        self.cov_coeff = cov_coeff
        self.gamma = gamma
        self.eps = eps

    def _invariance_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Compute mean squared error between projected representations."""
        return F.mse_loss(z1, z2)

    def _variance_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Compute hinge loss on feature standard deviations along the batch dimension."""
        # Standard deviation along batch dim (dim=0)
        std = torch.sqrt(torch.var(z, dim=0, unbiased=False) + self.eps)
        # Hinge loss against target gamma
        return torch.mean(F.relu(self.gamma - std))

    def _covariance_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Compute off-diagonal squared penalty on the feature covariance matrix."""
        batch_size, num_features = z.shape
        if batch_size <= 1:
            return torch.tensor(0.0, device=z.device, dtype=z.dtype)

        # Center features
        z_centered = z - torch.mean(z, dim=0, keepdim=True)

        # Sample covariance matrix (D x D)
        cov = (z_centered.T @ z_centered) / (batch_size - 1)

        # Zero out diagonal elements
        cov_off_diag = cov - torch.diag_embed(torch.diagonal(cov))

        # Sum of squared off-diagonal elements normalized by feature dimension
        return torch.sum(cov_off_diag**2) / num_features

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        h1: torch.Tensor | None = None,
        h2: torch.Tensor | None = None,
        queue: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute total VICReg loss and decomposed metric terms.

        Parameters
        ----------
        z1:
            Projected embeddings for view 1, shape ``(B, D)``.
        z2:
            Projected embeddings for view 2, shape ``(B, D)``.
        h1:
            Optional raw backbone representations (unused in standard VICReg).
        h2:
            Optional raw backbone representations (unused in standard VICReg).
        queue:
            Optional detached bank of past projected embeddings, shape ``(K, D)``.
            Variance/covariance are estimated jointly over the current batch and
            this bank so small batches still yield a well-conditioned covariance
            (rank is otherwise capped at ``batch_size - 1`` regardless of ``D``).
            The invariance term always uses the current-batch pairs only.

        Returns
        -------
        tuple[torch.Tensor, dict[str, torch.Tensor]]
            ``(total_loss, metrics_dict)``

        """
        # 1. Invariance loss
        sim_loss = self._invariance_loss(z1, z2)

        z1_stats = torch.cat([z1, queue]) if queue is not None else z1
        z2_stats = torch.cat([z2, queue]) if queue is not None else z2

        # 2. Variance loss
        var_loss_1 = self._variance_loss(z1_stats)
        var_loss_2 = self._variance_loss(z2_stats)
        var_loss = var_loss_1 + var_loss_2

        # 3. Covariance loss
        cov_loss_1 = self._covariance_loss(z1_stats)
        cov_loss_2 = self._covariance_loss(z2_stats)
        cov_loss = cov_loss_1 + cov_loss_2

        # Total weighted loss
        total_loss = (
            self.sim_coeff * sim_loss
            + self.var_coeff * var_loss
            + self.cov_coeff * cov_loss
        )

        metrics = {
            "invariance": sim_loss.detach(),
            "variance": var_loss.detach(),
            "covariance": cov_loss.detach(),
        }

        return total_loss, metrics

    def __repr__(self) -> str:
        """Return developer-friendly string representation."""
        return (
            f"{self.__class__.__name__}("
            f"sim_coeff={self.sim_coeff}, "
            f"var_coeff={self.var_coeff}, "
            f"cov_coeff={self.cov_coeff}, "
            f"gamma={self.gamma})"
        )
