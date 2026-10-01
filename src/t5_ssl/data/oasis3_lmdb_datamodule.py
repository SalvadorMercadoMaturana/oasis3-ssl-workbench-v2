"""Read the t4 OASIS3 LMDB/CSV format without altering the source store."""

from __future__ import annotations

import csv
import os
import pickle
import random
import re
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Any

import lightning as L
import lmdb
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from t5_ssl.data.oasis3_datamodule import normalize_volume
from t5_ssl.data.paths import DEFAULT_CSV, DEFAULT_LMDB

SUBJECT_RE = re.compile(r"^(sub-OAS\d+)ses")
_ENVIRONMENTS: dict[str, lmdb.Environment] = {}
_ENV_PID = os.getpid()


def clinical_rows(csv_path: str | Path) -> list[dict[str, str]]:
    """Select only OASIS3 rows indexed by the original LMDB key."""
    with Path(csv_path).open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream)
                if row["study_name"].strip().upper() == "OASIS3"]
    for row in rows:
        match = SUBJECT_RE.match(row["original_code"])
        if match is None:
            raise ValueError(f"Invalid OASIS3 code: {row['original_code']}")
        row["subject"] = match.group(1)
        row["session"] = row["original_code"]
        row["group"] = row["group"].strip().upper()
    return rows


def select_rows(csv_path: str | Path, manifest_path: str | Path | None,
                primary_only: bool, seed: int, train_ratio: float,
                val_ratio: float) -> tuple[dict[str, list[dict[str, str]]], dict[str, list[str]]]:
    """Validate a fixed subject manifest, or form deterministic full-data splits."""
    rows = clinical_rows(csv_path)
    indexed = {row["idx"]: row for row in rows}
    selected: dict[str, list[dict[str, str]]] = {name: [] for name in ("train", "val", "test")}
    if manifest_path is not None:
        with Path(manifest_path).open(newline="", encoding="utf-8") as stream:
            manifest = list(csv.DictReader(stream))
        for item in manifest:
            split = item["split"]
            if split not in selected:
                raise ValueError(f"Unexpected split {split}")
            row = indexed.get(item["idx"])
            if row is None or row["subject"] != item["subject"] or row["session"] != item["session"]:
                raise ValueError(f"Manifest/clinical mismatch for LMDB key {item['idx']}")
            if not primary_only or item["primary"] == "1":
                selected[split].append(row)
    else:
        labels: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            labels[row["subject"]].add(row["group"])
        strata: dict[str, list[str]] = defaultdict(list)
        for subject, values in labels.items():
            stratum = next(iter(values)) if len(values) == 1 and next(iter(values)) in ("AD", "HC") else "OTHER"
            strata[stratum].append(subject)
        rng = random.Random(seed)
        assignment: dict[str, str] = {}
        for ids in strata.values():
            ids.sort()
            rng.shuffle(ids)
            n = len(ids)
            n_train, n_val = int(n * train_ratio), int(n * val_ratio)
            if n >= 3:
                n_train = max(1, min(n_train, n - 2))
                n_val = max(1, min(n_val, n - n_train - 1))
            for split, cohort in (("train", ids[:n_train]),
                                  ("val", ids[n_train:n_train + n_val]),
                                  ("test", ids[n_train + n_val:])):
                assignment.update({subject: split for subject in cohort})
        for row in rows:
            selected[assignment[row["subject"]]].append(row)
    subjects = {name: sorted({row["subject"] for row in split_rows})
                for name, split_rows in selected.items()}
    sets = {name: set(ids) for name, ids in subjects.items()}
    if any(sets[a] & sets[b] for a, b in
           (("train", "val"), ("train", "test"), ("val", "test"))):
        raise RuntimeError("Subject leakage across LMDB splits")
    if not all(selected.values()):
        raise ValueError("LMDB splits must each contain at least one record")
    return selected, subjects


PLANE_AXES: dict[str, int] = {"axial": 2, "coronal": 1, "sagittal": 0}


class OASIS3LMDBSliceDataset(Dataset):
    """Slice signed MNI volumes fetched by original zero-padded CSV index."""

    def __init__(self, records: list[dict[str, str]], lmdb_path: str | Path, *,
                 slices_per_volume: int = 1, slice_range: tuple[float, float] = (0.45, 0.55),
                 channels: int = 1, plane: str = "axial", transform: Any = None,
                 random_slice: bool = False, one_session_per_subject: bool = False,
                 views_per_subject_epoch: int = 1,
                 cache_capacity: int = 128) -> None:
        if channels < 1 or channels % 2 == 0:
            raise ValueError("channels must be a positive odd number of stacked slices")
        if plane not in PLANE_AXES:
            raise ValueError(f"plane must be one of {sorted(PLANE_AXES)}")
        if slices_per_volume < 1 or not 0 <= slice_range[0] < slice_range[1] <= 1:
            raise ValueError("Invalid slice settings")
        if views_per_subject_epoch < 1:
            raise ValueError("views_per_subject_epoch must be positive")
        self.records = records
        self.lmdb_path = str(lmdb_path)
        self.channels = channels
        self.plane = plane
        self.axis = PLANE_AXES[plane]
        self.transform = transform
        self.slice_range = slice_range
        self.random_slice = random_slice
        self.one_session_per_subject = one_session_per_subject
        self.views_per_subject_epoch = views_per_subject_epoch
        if cache_capacity < 1:
            raise ValueError("cache_capacity must be positive")
        by_subject: dict[str, list[int]] = defaultdict(list)
        for record_index, record in enumerate(records):
            by_subject[record["subject"]].append(record_index)
        self.subject_record_indices = [by_subject[subject] for subject in sorted(by_subject)]
        fractions = ([(slice_range[0] + slice_range[1]) / 2]
                     if slices_per_volume == 1 else
                     np.linspace(slice_range[0], slice_range[1], slices_per_volume))
        self.indices = [(i, fraction) for i in range(len(records)) for fraction in
                        fractions]
        self._env: lmdb.Environment | None = None
        self._pid: int | None = None
        # Reuse percentile-normalized volumes across epochs. Full-cohort runs
        # can retain the training sessions in RAM instead of rereading LMDB.
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_capacity = cache_capacity

    def __len__(self) -> int:
        if self.one_session_per_subject:
            return len(self.subject_record_indices) * self.views_per_subject_epoch
        return len(self.indices)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state.update(_env=None, _pid=None, _cache=OrderedDict())
        return state

    def _environment(self) -> lmdb.Environment:
        global _ENV_PID
        pid = os.getpid()
        if pid != _ENV_PID:
            _ENVIRONMENTS.clear()
            _ENV_PID = pid
        if self._env is None or self._pid != pid:
            if self.lmdb_path not in _ENVIRONMENTS:
                _ENVIRONMENTS[self.lmdb_path] = lmdb.open(
                    self.lmdb_path, readonly=True, lock=False,
                    readahead=False, max_readers=64)
            self._env = _ENVIRONMENTS[self.lmdb_path]
            self._pid = pid
            self._cache.clear()
        return self._env

    def _volume(self, record: dict[str, str]) -> np.ndarray:
        key = record["idx"]
        self._environment()
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        assert self._env is not None
        with self._env.begin() as transaction:
            value = transaction.get(key.encode("ascii"))
        if value is None:
            raise KeyError(f"Missing LMDB key {key}")
        # The trusted local t4 LMDB stores pickle dictionaries with an int16 image.
        payload = pickle.loads(value)
        if (payload["study_name"] != "OASIS3" or
                payload["original_code"] != record["session"]):
            raise ValueError(f"LMDB/CSV metadata mismatch at {key}")
        volume = normalize_volume(np.asarray(payload["image"]))
        self._cache[key] = volume
        if len(self._cache) > self._cache_capacity:
            self._cache.popitem(last=False)
        return volume

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.one_session_per_subject:
            subject_index = index % len(self.subject_record_indices)
            record_index = random.choice(self.subject_record_indices[subject_index])
            fraction = (self.slice_range[0] + self.slice_range[1]) / 2
        else:
            record_index, fraction = self.indices[index]
        if self.random_slice:
            fraction = random.uniform(*self.slice_range)
        record = self.records[record_index]
        volume = self._volume(record)
        size = volume.shape[self.axis]
        z = round(float(fraction) * (size - 1))
        half = self.channels // 2
        image = np.stack([np.take(volume, min(max(z + offset, 0), size - 1), axis=self.axis)
                          for offset in range(-half, half + 1)], axis=0).copy()
        tensor = torch.from_numpy(image)
        if self.transform is not None:
            tensor = self.transform(tensor)
        return {"image": tensor, "subject": record["subject"],
                "session": record["session"], "group": record["group"]}


class OASIS3LMDBDataModule(L.LightningDataModule):
    """Subject-safe Lightning bridge to the original t4 CSV plus LMDB."""

    def __init__(self, *, csv_path: str = DEFAULT_CSV,
                 lmdb_path: str = DEFAULT_LMDB,
                 manifest_path: str | None = None, primary_only: bool = False,
                 batch_size: int = 16, num_workers: int = 0, pin_memory: bool = False,
                 channels: int = 1, plane: str = "axial", slices_per_volume: int = 1,
                 slice_range: tuple[float, float] = (0.45, 0.55),
                 train_ratio: float = 0.70, val_ratio: float = 0.15, seed: int = 42,
                 one_session_per_subject_train: bool = False,
                 views_per_subject_epoch: int = 1,
                 train_cache_capacity: int = 128,
                 eval_cache_capacity: int = 32,
                 balanced_sampler: bool = False,
                 train_transform: Any = None, eval_transform: Any = None) -> None:
        super().__init__()
        self.csv_path, self.lmdb_path, self.manifest_path = csv_path, lmdb_path, manifest_path
        self.primary_only = primary_only
        self.balanced_sampler = balanced_sampler
        self.batch_size, self.num_workers, self.pin_memory = batch_size, num_workers, pin_memory
        self.channels, self.plane = channels, plane
        self.slices_per_volume, self.slice_range = slices_per_volume, slice_range
        self.train_ratio, self.val_ratio, self.seed = train_ratio, val_ratio, seed
        self.one_session_per_subject_train = one_session_per_subject_train
        self.views_per_subject_epoch = views_per_subject_epoch
        self.train_cache_capacity = train_cache_capacity
        self.eval_cache_capacity = eval_cache_capacity
        self.train_transform, self.eval_transform = train_transform, eval_transform
        self.split_subject_ids: dict[str, list[str]] = {}
        self.subject_labels: dict[str, str] = {}
        self.train_dataset: OASIS3LMDBSliceDataset | None = None
        self.val_dataset: OASIS3LMDBSliceDataset | None = None
        self.test_dataset: OASIS3LMDBSliceDataset | None = None

    def prepare_data(self) -> None:
        if not Path(self.lmdb_path, "data.mdb").is_file():
            raise FileNotFoundError(self.lmdb_path)

    def setup(self, stage: str | None = None) -> None:
        del stage
        self.prepare_data()
        labels: dict[str, set[str]] = defaultdict(set)
        for row in clinical_rows(self.csv_path):
            labels[row["subject"]].add(row["group"])
        self.subject_labels = {subject: next(iter(values)) for subject, values in labels.items()
                               if len(values) == 1 and next(iter(values)) in ("AD", "HC")}
        selected, self.split_subject_ids = select_rows(
            self.csv_path, self.manifest_path, self.primary_only,
            self.seed, self.train_ratio, self.val_ratio)
        for split, rows in selected.items():
            dataset = OASIS3LMDBSliceDataset(
                rows, self.lmdb_path, channels=self.channels, plane=self.plane,
                slices_per_volume=self.slices_per_volume, slice_range=self.slice_range,
                transform=self.train_transform if split == "train" else self.eval_transform,
                random_slice=split == "train",
                one_session_per_subject=split == "train" and self.one_session_per_subject_train,
                views_per_subject_epoch=self.views_per_subject_epoch if split == "train" else 1,
                cache_capacity=(self.train_cache_capacity if split == "train"
                                else self.eval_cache_capacity))
            setattr(self, f"{split}_dataset", dataset)

    def _loader(self, dataset: OASIS3LMDBSliceDataset | None, shuffle: bool,
               sampler: WeightedRandomSampler | None = None) -> DataLoader:
        if dataset is None:
            raise RuntimeError("Call setup() before requesting a dataloader")
        common = dict(batch_size=self.batch_size, num_workers=self.num_workers,
                     pin_memory=self.pin_memory, persistent_workers=self.num_workers > 0)
        if sampler is not None:
            return DataLoader(dataset, sampler=sampler, **common)
        return DataLoader(dataset, shuffle=shuffle, **common)

    def _class_balanced_sample_weights(self, dataset: OASIS3LMDBSliceDataset) -> list[float]:
        """Inverse-frequency AD/HC weights (normalized to the larger class); OTHER untouched.

        Reweights only AD vs. HC so each batch has, in expectation, an equal
        number of each; subjects/sessions with any other (or ambiguous) group
        label keep their natural per-item weight of 1.0, exactly as before.
        """
        def label(group: str) -> str:
            return group if group in ("AD", "HC") else "OTHER"

        if dataset.one_session_per_subject:
            subject_groups = [
                label(next(iter({dataset.records[i]["group"] for i in indices})))
                if len({dataset.records[i]["group"] for i in indices}) == 1 else "OTHER"
                for indices in dataset.subject_record_indices
            ]
            counts = Counter(subject_groups)
            if "AD" not in counts or "HC" not in counts:
                raise ValueError("balanced_sampler requires both AD and HC subjects in train")
            majority = max(counts["AD"], counts["HC"])
            class_weight = {"AD": majority / counts["AD"], "HC": majority / counts["HC"], "OTHER": 1.0}
            per_subject_weight = [class_weight[g] for g in subject_groups]
            return per_subject_weight * dataset.views_per_subject_epoch
        record_groups = [label(record["group"]) for record in dataset.records]
        counts = Counter(record_groups)
        if "AD" not in counts or "HC" not in counts:
            raise ValueError("balanced_sampler requires both AD and HC subjects in train")
        majority = max(counts["AD"], counts["HC"])
        class_weight = {"AD": majority / counts["AD"], "HC": majority / counts["HC"], "OTHER": 1.0}
        per_record_weight = [class_weight[g] for g in record_groups]
        return [per_record_weight[record_index] for record_index, _fraction in dataset.indices]

    def train_dataloader(self) -> DataLoader:
        if not self.balanced_sampler:
            return self._loader(self.train_dataset, True)
        assert self.train_dataset is not None
        weights = self._class_balanced_sample_weights(self.train_dataset)
        sampler = WeightedRandomSampler(weights, num_samples=len(self.train_dataset), replacement=True)
        return self._loader(self.train_dataset, False, sampler=sampler)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset, False)

    def eval_dataset(self, split: str, transform: Any) -> OASIS3LMDBSliceDataset:
        """Rebuild a split with deterministic single-view embeddings."""
        dataset = getattr(self, f"{split}_dataset")
        if dataset is None:
            raise RuntimeError("Call setup() first")
        return OASIS3LMDBSliceDataset(
            dataset.records, self.lmdb_path, channels=self.channels, plane=self.plane,
            slices_per_volume=self.slices_per_volume,
            slice_range=self.slice_range, transform=transform,
            cache_capacity=self.eval_cache_capacity)
