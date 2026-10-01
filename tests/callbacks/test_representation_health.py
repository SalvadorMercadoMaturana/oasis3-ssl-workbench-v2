"""Tests for the training-time representation-health callback."""

from __future__ import annotations

import json

import torch
from torch import nn
from torch.utils.data import Dataset

from t5_ssl.callbacks.representation_health import RepresentationHealthCallback


class PairedSubjectDataset(Dataset):
    """Two deterministic views per item, with repeated subjects for retrieval."""

    def __init__(self, n_items: int = 12, n_channels: int = 1, size: int = 8) -> None:
        self.n_items = n_items
        self.n_channels = n_channels
        self.size = size

    def __len__(self) -> int:
        return self.n_items

    def __getitem__(self, index: int) -> dict:
        generator = torch.Generator().manual_seed(index // 2)
        base = torch.rand(self.n_channels, self.size, self.size, generator=generator)
        view1 = base + 0.001 * torch.randn(base.shape, generator=torch.Generator().manual_seed(index))
        view2 = base + 0.001 * torch.randn(base.shape, generator=torch.Generator().manual_seed(index + 100))
        return {"image": (view1, view2), "subject": f"sub-{index // 2:02d}"}


class SingleViewDataset(Dataset):
    """One deterministic view per item, for the neighbor-slice retrieval path."""

    def __init__(self, n_items: int = 8, n_channels: int = 1, size: int = 8) -> None:
        self.n_items = n_items
        self.n_channels = n_channels
        self.size = size

    def __len__(self) -> int:
        return self.n_items

    def __getitem__(self, index: int) -> dict:
        generator = torch.Generator().manual_seed(index // 2)
        base = torch.rand(self.n_channels, self.size, self.size, generator=generator)
        return {"image": base, "subject": f"sub-{index // 2:02d}"}


class TinyBackbone(nn.Module):
    """Cheap embedding extractor used only to exercise the callback's plumbing."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 6, 3, padding=1), nn.AdaptiveAvgPool2d(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return one embedding vector per image."""
        return self.net(x).flatten(1)


class FakeModule:
    """Minimal stand-in for a LightningModule exposing only what the callback needs."""

    def __init__(self, backbone: nn.Module) -> None:
        self.backbone = backbone
        self.logged: dict[str, float] = {}

    def log(self, name: str, value: float, **kwargs: object) -> None:
        """Record a scalar the same way LightningModule.log would."""
        self.logged[name] = float(value)


class FakeTrainer:
    """Minimal stand-in for a Trainer, only the attributes the callback reads."""

    def __init__(self, epoch: int = 0, step: int = 0, sanity: bool = False) -> None:
        self.current_epoch = epoch
        self.global_step = step
        self.sanity_checking = sanity


def test_measure_reports_core_health_metrics():
    """A basic pass must produce rank, spread, and similarity-gap diagnostics."""
    callback = RepresentationHealthCallback(PairedSubjectDataset(), output_dir="unused", batch_size=4)
    result = callback._measure(FakeModule(TinyBackbone()))
    assert "effective_rank" in result
    assert "mean_feature_std" in result
    assert "positive_similarity" in result
    assert result["session_retrieval_n_eligible"] == 12


def test_neighbor_dataset_drives_neighbor_retrieval():
    """When given, the neighbor dataset should populate the neighbor-retrieval fields."""
    neighbor = SingleViewDataset()
    group_ids = [str(i // 2) for i in range(len(neighbor))]
    callback = RepresentationHealthCallback(
        PairedSubjectDataset(), output_dir="unused", batch_size=4,
        neighbor_dataset=neighbor, neighbor_group_ids=group_ids,
    )
    result = callback._measure(FakeModule(TinyBackbone()))
    assert "neighbor_slice_retrieval_top1" in result


def test_age_and_sex_maps_add_association_metrics():
    """Supplying age/sex maps should add nuisance-association diagnostics."""
    ages = {f"sub-{i:02d}": 60.0 + i for i in range(6)}
    sexes = {f"sub-{i:02d}": "f" if i % 2 else "m" for i in range(6)}
    callback = RepresentationHealthCallback(
        PairedSubjectDataset(), output_dir="unused", batch_size=4, ages=ages, sexes=sexes,
    )
    result = callback._measure(FakeModule(TinyBackbone()))
    assert "age_r2" in result
    assert "sex_accuracy" in result
    assert "sex_majority_baseline" in result


def test_on_validation_epoch_end_writes_history_and_skips_sanity_check(tmp_path):
    """The callback must skip Lightning's sanity check and persist a JSON history."""
    callback = RepresentationHealthCallback(
        PairedSubjectDataset(), output_dir=tmp_path, batch_size=4, every_n_epochs=1,
    )
    module = FakeModule(TinyBackbone())
    callback.on_validation_epoch_end(FakeTrainer(epoch=0, sanity=True), module)
    assert callback.history == []
    assert not (tmp_path / "representation_health.json").exists()

    callback.on_validation_epoch_end(FakeTrainer(epoch=0, step=10), module)
    assert len(callback.history) == 1
    assert module.logged  # some health/* scalars were logged
    saved = json.loads((tmp_path / "representation_health.json").read_text())
    assert saved[0]["epoch"] == 0
    assert saved[0]["global_step"] == 10


def test_every_n_epochs_skips_intermediate_epochs(tmp_path):
    """A cadence of 2 should only measure on even epochs."""
    callback = RepresentationHealthCallback(
        PairedSubjectDataset(), output_dir=tmp_path, batch_size=4, every_n_epochs=2,
    )
    module = FakeModule(TinyBackbone())
    callback.on_validation_epoch_end(FakeTrainer(epoch=0), module)
    callback.on_validation_epoch_end(FakeTrainer(epoch=1), module)
    callback.on_validation_epoch_end(FakeTrainer(epoch=2), module)
    assert [record["epoch"] for record in callback.history] == [0, 2]
