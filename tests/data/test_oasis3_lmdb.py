"""Original t4 CSV/LMDB key and subject split integration."""

from __future__ import annotations

import csv
import pickle

import lmdb
import numpy as np
import pytest

from t5_ssl.data.oasis3_lmdb_datamodule import OASIS3LMDBDataModule, OASIS3LMDBSliceDataset


def test_lmdb_manifest_reads_signed_images_without_subject_leakage(tmp_path, monkeypatch):
    """Manifest keys must fetch matching OASIS3 entries and preserve splits."""
    table = tmp_path / "mni-dataset.csv"
    manifest = tmp_path / "cohort.csv"
    store = tmp_path / "mni-dataset.lmdb"
    environment = lmdb.open(str(store), map_size=16_000_000)
    with table.open("w", newline="") as clinical, manifest.open("w", newline="") as cohort:
        cw = csv.writer(clinical)
        mw = csv.writer(cohort)
        cw.writerow(["idx", "study_name", "original_code", "group"])
        mw.writerow(["split", "subject", "idx", "session", "group", "primary"])
        with environment.begin(write=True) as transaction:
            for i in range(6):
                key = f"{i:06d}"
                subject = f"sub-OAS3{i:04d}"
                session = f"{subject}sesd0001"
                group = "AD" if i < 3 else "HC"
                split = ("train", "val", "test")[i % 3]
                cw.writerow([key, "OASIS3", session, group])
                mw.writerow([split, subject, key, session, group, 1])
                image = np.zeros((8, 9, 10), dtype=np.int16)
                image[2:6, 2:7, 2:8] = -100 - i
                image[3:5, 3:6, 3:7] = 200 + i
                transaction.put(key.encode(), pickle.dumps({
                    "study_name": "OASIS3", "original_code": session,
                    "group": group, "image": image}))
        # An unselected later session changes the subject's clinical label.
        cw.writerow(["000006", "OASIS3", "sub-OAS30000sesd0002", "HC"])
    environment.close()
    dm = OASIS3LMDBDataModule(csv_path=str(table), lmdb_path=str(store),
                              manifest_path=str(manifest), primary_only=True,
                              batch_size=2, num_workers=0, channels=3,
                              slice_range=(0.3, 0.7))
    dm.setup()
    assert {name: len(ids) for name, ids in dm.split_subject_ids.items()} == {
        "train": 2, "val": 2, "test": 2}
    assert "sub-OAS30000" not in dm.subject_labels
    assert dm.train_dataset.random_slice
    assert not dm.val_dataset.random_slice
    assert dm.val_dataset.indices[0][1] == 0.5
    assert not dm.eval_dataset("train", None).random_slice
    seen = []
    monkeypatch.setattr("t5_ssl.data.oasis3_lmdb_datamodule.random.uniform",
                        lambda low, high: seen.append((low, high)) or 0.7)
    batch = next(iter(dm.train_dataloader()))
    assert seen == [(0.3, 0.7), (0.3, 0.7)]
    assert batch["image"].shape == (2, 3, 8, 9)
    assert 0 <= float(batch["image"].min()) < float(batch["image"].max()) <= 1
    assert set(batch["subject"]).isdisjoint(dm.split_subject_ids["val"])


def _write_synthetic_lmdb(tmp_path, n_subjects: int = 3, shape=(8, 9, 10)):
    """Build a tiny CSV+LMDB pair with one session per subject, for dataset-only tests."""
    table = tmp_path / "mni-dataset.csv"
    store = tmp_path / "mni-dataset.lmdb"
    environment = lmdb.open(str(store), map_size=16_000_000)
    records = []
    with table.open("w", newline="") as clinical, environment.begin(write=True) as transaction:
        cw = csv.writer(clinical)
        cw.writerow(["idx", "study_name", "original_code", "group"])
        for i in range(n_subjects):
            key = f"{i:06d}"
            subject = f"sub-OAS3{i:04d}"
            session = f"{subject}sesd0001"
            group = "AD" if i % 2 == 0 else "HC"
            cw.writerow([key, "OASIS3", session, group])
            image = (np.random.default_rng(i).integers(-500, 500, size=shape)
                     .astype(np.int16))
            transaction.put(key.encode(), pickle.dumps({
                "study_name": "OASIS3", "original_code": session,
                "group": group, "image": image}))
            records.append({"idx": key, "subject": subject, "session": session, "group": group})
    environment.close()
    return table, store, records


def test_channels_accepts_any_odd_stack_size(tmp_path):
    """A 2.5D stack of 5 neighboring slices should generalize past the old 1-or-3 cap."""
    _, store, records = _write_synthetic_lmdb(tmp_path)
    dataset = OASIS3LMDBSliceDataset(records, store, channels=5, slice_range=(0.45, 0.55))
    item = dataset[0]
    assert item["image"].shape == (5, 8, 9)


def test_channels_rejects_even_stack_size(tmp_path):
    """An even number of stacked slices has no well-defined center slice."""
    _, store, records = _write_synthetic_lmdb(tmp_path)
    with pytest.raises(ValueError, match="odd"):
        OASIS3LMDBSliceDataset(records, store, channels=4)


def test_plane_selects_a_different_slicing_axis(tmp_path):
    """Coronal/sagittal planes must slice along their own volume axis."""
    _, store, records = _write_synthetic_lmdb(tmp_path, shape=(8, 9, 10))
    axial = OASIS3LMDBSliceDataset(records, store, channels=1, plane="axial",
                                   slice_range=(0.45, 0.55))
    coronal = OASIS3LMDBSliceDataset(records, store, channels=1, plane="coronal",
                                     slice_range=(0.45, 0.55))
    sagittal = OASIS3LMDBSliceDataset(records, store, channels=1, plane="sagittal",
                                      slice_range=(0.45, 0.55))
    assert axial[0]["image"].shape == (1, 8, 9)
    assert coronal[0]["image"].shape == (1, 8, 10)
    assert sagittal[0]["image"].shape == (1, 9, 10)


def test_plane_rejects_unknown_value(tmp_path):
    """Only the three named anatomical planes are supported."""
    _, store, records = _write_synthetic_lmdb(tmp_path)
    with pytest.raises(ValueError, match="plane"):
        OASIS3LMDBSliceDataset(records, store, plane="oblique")


def test_views_per_subject_epoch_multiplies_dataset_length(tmp_path):
    """Each subject should contribute several independently sampled views per epoch."""
    _, store, records = _write_synthetic_lmdb(tmp_path, n_subjects=3)
    dataset = OASIS3LMDBSliceDataset(
        records, store, one_session_per_subject=True, views_per_subject_epoch=4,
        random_slice=True, slice_range=(0.3, 0.7),
    )
    assert len(dataset) == 3 * 4
    # Every one of the 12 flat indices must still resolve to one of the 3 subjects.
    subjects = {dataset[i]["subject"] for i in range(len(dataset))}
    assert subjects == {r["subject"] for r in records}


def test_views_per_subject_epoch_defaults_to_one(tmp_path):
    """The default must reproduce the original one-session-per-subject length."""
    _, store, records = _write_synthetic_lmdb(tmp_path, n_subjects=3)
    dataset = OASIS3LMDBSliceDataset(records, store, one_session_per_subject=True)
    assert len(dataset) == 3
