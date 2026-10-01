"""Tests for the queue-augmented VICReg Lightning module."""

from __future__ import annotations

import torch
from torch import nn

from t5_ssl.losses.vicreg import VICRegLoss
from t5_ssl.models.heads.projector import MLPProjector
from t5_ssl.models.vicreg_module import VICRegLightningModule


class TinyBackbone(nn.Module):
    """Cheap single-channel feature extractor for the VICReg contract."""

    out_dim = 8

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.AdaptiveAvgPool2d(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return one vector per image."""
        return self.net(x).flatten(1)


def _make_model(queue_size: int = 0) -> VICRegLightningModule:
    return VICRegLightningModule(
        backbone=TinyBackbone(), projector=MLPProjector(8, 16, 8, num_layers=2, use_bn=False),
        loss_fn=VICRegLoss(), max_epochs=1, queue_size=queue_size,
    )


def test_queue_disabled_by_default_matches_plain_forward():
    """queue_size=0 must not register any queue buffers."""
    model = _make_model(queue_size=0)
    assert not hasattr(model, "queue")
    x1, x2 = torch.rand(4, 1, 8, 8), torch.rand(4, 1, 8, 8)
    loss = model.training_step({"image": (x1, x2)}, 0)
    assert torch.isfinite(loss)


def test_queue_fills_up_across_steps():
    """The FIFO bank should accumulate projected embeddings batch by batch."""
    model = _make_model(queue_size=20)
    assert model.queue.shape == (20, 8)
    assert int(model.queue_filled.item()) == 0
    x1, x2 = torch.rand(4, 1, 8, 8), torch.rand(4, 1, 8, 8)
    model.training_step({"image": (x1, x2)}, 0)
    assert int(model.queue_filled.item()) == 8
    model.training_step({"image": (x1, x2)}, 1)
    assert int(model.queue_filled.item()) == 16
    model.training_step({"image": (x1, x2)}, 2)
    # Bank caps at queue_size even once more than queue_size embeddings arrive.
    assert int(model.queue_filled.item()) == 20


def test_queue_wraps_around_without_crashing():
    """Enqueue must handle the FIFO wrap-around when a batch straddles the end."""
    model = _make_model(queue_size=10)
    x1, x2 = torch.rand(4, 1, 8, 8), torch.rand(4, 1, 8, 8)
    for step in range(5):
        loss = model.training_step({"image": (x1, x2)}, step)
        assert torch.isfinite(loss)
    assert int(model.queue_filled.item()) == 10


def test_validation_step_does_not_touch_the_queue():
    """Validation must score VICReg without mutating the training-only bank."""
    model = _make_model(queue_size=10)
    x1, x2 = torch.rand(4, 1, 8, 8), torch.rand(4, 1, 8, 8)
    model.eval()
    with torch.no_grad():
        model.validation_step({"image": (x1, x2)}, 0)
    assert int(model.queue_filled.item()) == 0


def test_gradients_flow_through_backbone_and_projector():
    """The online encoder/projector must receive gradients from the queued loss."""
    model = _make_model(queue_size=32)
    x1, x2 = torch.rand(4, 1, 8, 8), torch.rand(4, 1, 8, 8)
    model.training_step({"image": (x1, x2)}, 0)
    loss = model.training_step({"image": (x1, x2)}, 1)
    loss.backward()
    assert any(p.grad is not None for p in model.backbone.parameters())
    assert any(p.grad is not None for p in model.projector.parameters())
