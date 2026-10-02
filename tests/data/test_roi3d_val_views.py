"""Augmented SSL validation views for the ROI datamodule (augmentation.val_views=train)."""

from __future__ import annotations

import random

import numpy as np
import torch

from t5_ssl.data.roi3d import ROI3DDataset, ROIView3DTransform
from t5_ssl.recipes import overrides


def _dataset(view_seed: int | None) -> ROI3DDataset:
    rng = np.random.default_rng(0)
    cubes = {str(i): rng.random((2, 38, 38, 38), dtype=np.float32) for i in range(3)}
    records = [{"idx": str(i), "subject": f"s{i}", "session": f"d{i}", "group": "HC"}
               for i in range(3)]
    transform = ROIView3DTransform(size=32, degrees=10.0, noise_std=0.05)
    return ROI3DDataset(records, cubes, transform, view_seed=view_seed)


def test_seeded_views_repeat_every_epoch_and_differ_between_records() -> None:
    dataset = _dataset(view_seed=7)
    first, again = dataset[0]["image"], dataset[0]["image"]
    for a, b in zip(first, again, strict=True):
        assert torch.equal(a, b)
    assert not torch.equal(first[0], first[1])  # still two different augmented views
    assert not torch.equal(first[0], dataset[1]["image"][0][:, :, :, :])


def test_seeded_views_leave_global_rng_untouched() -> None:
    dataset = _dataset(view_seed=7)
    random.seed(1)
    torch.manual_seed(1)
    expected = random.random(), torch.rand(1)
    random.seed(1)
    torch.manual_seed(1)
    dataset[2]
    assert (random.random(), torch.rand(1)) == expected


def test_unseeded_views_stay_random() -> None:
    dataset = _dataset(view_seed=None)
    assert not torch.equal(dataset[0]["image"][0], dataset[0]["image"][0])


def test_vicreg_augval_variant_sets_val_views() -> None:
    assert "++augmentation.val_views=train" in overrides("vicreg", "campaign_augval")
