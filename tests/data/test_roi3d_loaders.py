"""ROI datamodule loaders: no one-sample training batch; swap_hemispheres on samples only."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from t5_ssl.data.roi3d import ROI3DDataModule, ROI3DDataset, swap_hemispheres


def _datamodule(subjects: int, batch_size: int) -> ROI3DDataModule:
    cubes = {str(i): np.zeros((2, 4, 4, 4), np.float16) for i in range(subjects)}
    records = [{"idx": str(i), "subject": f"s{i}", "session": f"d{i}", "group": "HC"}
               for i in range(subjects)]
    dm = ROI3DDataModule(csv_path="", cache_path="", manifest_path="", batch_size=batch_size)
    dm.train_dataset = ROI3DDataset(records, cubes, one_session_per_subject=True)
    dm.val_dataset = ROI3DDataset(records, cubes)
    return dm


@pytest.mark.parametrize(("subjects", "batch_size", "sizes"), [
    (745, 8, [8] * 93),             # outer5 folds 1-2 at the first OOM retry: 1 left -> dropped
    (743, 16, [16] * 46 + [7]),     # split_deploy at batch 16: unchanged
    (745, 16, [16] * 46 + [9]),
    (17, 16, [16]),
    (16, 16, [16]),
    (10, 4, [4, 4, 2]),
])
def test_train_loader_never_ends_on_one_sample(subjects, batch_size, sizes):
    loader = _datamodule(subjects, batch_size).train_dataloader()
    assert [len(batch["subject"]) for batch in loader] == sizes


def test_validation_loader_keeps_every_subject():
    loader = _datamodule(17, 16).val_dataloader()
    assert [len(batch["subject"]) for batch in loader] == [16, 1]


def test_swap_hemispheres_rejects_batches_and_odd_channels():
    with pytest.raises(ValueError, match="one"):
        swap_hemispheres(torch.rand(4, 2, 3, 3, 3))  # a batch would swap samples
    with pytest.raises(ValueError, match="pairs"):
        swap_hemispheres(torch.rand(3, 3, 3, 3))
