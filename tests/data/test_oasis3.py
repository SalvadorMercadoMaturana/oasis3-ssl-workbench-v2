"""OASIS3 image orientation and subject-level split contracts."""

from __future__ import annotations

import csv

import nibabel as nib
import numpy as np

from t5_ssl.data.oasis3_datamodule import OASIS3DataModule, load_volume


def test_ras_axial_slices_and_subject_disjoint_sessions(tmp_path):
    """An LIA source must become RAS and sessions must stay with their subject."""
    root = tmp_path / "images"
    root.mkdir()
    table = tmp_path / "clinical.csv"
    affine = np.array([[-1, 0, 0, 0], [0, 0, 1, 0],
                       [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float)
    with table.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["study_name", "original_code", "group"])
        writer.writeheader()
        for subject in range(6):
            for session in range(2):
                code = f"sub-OAS3{subject:04d}sesd{session:04d}"
                writer.writerow({"study_name": "OASIS3", "original_code": code,
                                 "group": "AD" if subject < 3 else "HC"})
                array = np.zeros((8, 9, 10), dtype=np.uint8)
                for y in range(9):
                    array[:, y, :] = y + 1
                nib.save(nib.Nifti1Image(array, affine), root / f"{code}_T1w.nii.gz")
        writer.writerow({"study_name": "ADNI", "original_code": "sub-ADNI01sesd0001",
                         "group": "AD"})
    volume = load_volume(str(next(root.glob("*.nii.gz"))))
    assert volume.shape == (8, 10, 9)
    assert volume[:, :, 0].mean() != volume[:, :, -1].mean()
    dm = OASIS3DataModule(csv_path=str(table), image_root=str(root),
                          batch_size=1, num_workers=0, channels=3,
                          slices_per_volume=1)
    dm.setup()
    assert dm.records is not None and len(dm.records) == 12
    splits = {key: set(value) for key, value in dm.split_subject_ids.items()}
    assert all(splits[a].isdisjoint(splits[b]) for a, b in
               (("train", "val"), ("train", "test"), ("val", "test")))
    assert all(len(ids) == 2 for ids in splits.values())
    for split in splits:
        dataset = getattr(dm, f"{split}_dataset")
        assert dataset is not None
        assert all(row["subject"] in splits[split] for row in dataset.records)
        assert dataset[0]["image"].shape[0] == 3


def test_signed_mri_foreground_is_not_discarded(tmp_path):
    """Negative tissue values remain usable while zero background stays zero."""
    array = np.zeros((8, 8, 8), dtype=np.float32)
    array[2:6, 2:6, 2:6] = np.linspace(-50, -1, 64).reshape(4, 4, 4)
    path = tmp_path / "signed.nii.gz"
    nib.save(nib.Nifti1Image(array, np.eye(4)), path)
    volume = load_volume(str(path))
    assert volume[0, 0, 0] == 0
    assert volume.max() > 0
    assert np.isfinite(volume).all()
