"""Unit tests for VICReg loss calculation and gradient flow."""

from __future__ import annotations

import torch

from t5_ssl.losses.vicreg import VICRegLoss


def test_vicreg_invariance_identical():
    """Identical projected vectors should yield zero invariance loss."""
    loss_fn = VICRegLoss(sim_coeff=1.0, var_coeff=0.0, cov_coeff=0.0)
    z = torch.randn(8, 64)
    loss, metrics = loss_fn(z, z)

    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(metrics["invariance"], torch.tensor(0.0), atol=1e-6)


def test_vicreg_invariance_distinct():
    """Distinct projected vectors should yield positive invariance loss."""
    loss_fn = VICRegLoss(sim_coeff=1.0, var_coeff=0.0, cov_coeff=0.0)
    z1 = torch.zeros(8, 64)
    z2 = torch.ones(8, 64)
    loss, _ = loss_fn(z1, z2)

    assert loss.item() > 0.0
    assert torch.isclose(loss, torch.tensor(1.0), atol=1e-5)


def test_vicreg_variance_collapsed():
    """Constant/collapsed representations should suffer high variance penalty."""
    loss_fn = VICRegLoss(sim_coeff=0.0, var_coeff=1.0, cov_coeff=0.0, gamma=1.0)
    # Zero variance representations
    z = torch.zeros(16, 64)
    loss, _ = loss_fn(z, z)

    # Expected penalty: 2 * (gamma - sqrt(eps)) ~ 2 * 0.99 = ~1.98
    assert loss.item() > 1.8


def test_vicreg_variance_high_spread():
    """Representations with standard deviation > gamma should have 0 variance penalty."""
    loss_fn = VICRegLoss(sim_coeff=0.0, var_coeff=1.0, cov_coeff=0.0, gamma=1.0)
    # High standard deviation
    z = torch.randn(1000, 64) * 2.0
    loss, _ = loss_fn(z, z)

    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-3)


def test_vicreg_covariance_orthogonal():
    """Uncorrelated standard normal features should yield very low covariance loss."""
    loss_fn = VICRegLoss(sim_coeff=0.0, var_coeff=0.0, cov_coeff=1.0)
    z = torch.randn(500, 16)
    loss, _ = loss_fn(z, z)

    assert loss.item() < 0.5


def test_vicreg_covariance_fully_correlated():
    """Identical/correlated feature dimensions should suffer positive covariance penalty."""
    loss_fn = VICRegLoss(sim_coeff=0.0, var_coeff=0.0, cov_coeff=1.0)
    base = torch.randn(16, 1)
    # Replicate single feature dimension across 10 channels
    z = base.repeat(1, 10)
    loss, _ = loss_fn(z, z)

    assert loss.item() > 0.0


def test_vicreg_gradients():
    """Gradients must flow back to z1 and z2 smoothly."""
    loss_fn = VICRegLoss()
    z1 = torch.randn(8, 32, requires_grad=True)
    z2 = torch.randn(8, 32, requires_grad=True)

    loss, _ = loss_fn(z1, z2)
    loss.backward()

    assert z1.grad is not None
    assert z2.grad is not None
    assert not torch.isnan(z1.grad).any()
    assert not torch.isnan(z2.grad).any()


def test_vicreg_batch_size_one_edge_case():
    """Batch size 1 should return 0 covariance loss without crashing."""
    loss_fn = VICRegLoss()
    z1 = torch.randn(1, 32)
    z2 = torch.randn(1, 32)

    loss, metrics = loss_fn(z1, z2)
    assert not torch.isnan(loss)
    assert "invariance" in metrics


def test_vicreg_queue_raises_variance_covariance_rank():
    """A queue of past embeddings should relax the batch_size-1 rank cap."""
    torch.manual_seed(0)
    loss_fn = VICRegLoss(sim_coeff=0.0, var_coeff=0.0, cov_coeff=1.0)
    # 4 samples, 64 dims: without extra context the covariance has rank <= 3.
    z1 = torch.randn(4, 64)
    z2 = torch.randn(4, 64)
    loss_no_queue, _ = loss_fn(z1, z2)

    queue = torch.randn(256, 64)
    loss_with_queue, _ = loss_fn(z1, z2, queue=queue)

    assert not torch.isnan(loss_with_queue)
    assert loss_with_queue.item() != loss_no_queue.item()


def test_vicreg_queue_does_not_affect_invariance_term():
    """Invariance must stay batch-only regardless of the queue."""
    loss_fn = VICRegLoss(sim_coeff=1.0, var_coeff=0.0, cov_coeff=0.0)
    z1 = torch.randn(4, 16)
    z2 = torch.randn(4, 16)
    queue = torch.randn(100, 16)

    loss_no_queue, metrics_no_queue = loss_fn(z1, z2)
    loss_with_queue, metrics_with_queue = loss_fn(z1, z2, queue=queue)

    assert torch.isclose(loss_no_queue, loss_with_queue)
    assert torch.isclose(metrics_no_queue["invariance"], metrics_with_queue["invariance"])


def test_vicreg_queue_gradients_flow_only_through_batch():
    """The queue is a detached context; gradients must still reach z1/z2."""
    loss_fn = VICRegLoss()
    z1 = torch.randn(4, 16, requires_grad=True)
    z2 = torch.randn(4, 16, requires_grad=True)
    queue = torch.randn(50, 16, requires_grad=True)

    loss, _ = loss_fn(z1, z2, queue=queue.detach())
    loss.backward()

    assert z1.grad is not None and not torch.isnan(z1.grad).any()
    assert z2.grad is not None and not torch.isnan(z2.grad).any()
    assert queue.grad is None
