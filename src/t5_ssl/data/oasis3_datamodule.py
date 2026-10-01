"""Subject-disjoint OASIS3 T1w NIfTI data for slice-based SSL."""

from __future__ import annotations

import csv
import random
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import lightning as L
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from t5_ssl.data.paths import DEFAULT_CSV, DEFAULT_IMAGE_ROOT

SUBJECT_RE = re.compile(r"^(sub-OAS\d+)ses")


def read_manifest(csv_path: str | Path, image_root: str | Path) -> list[dict[str, str]]:
    """Join OASIS3 clinical sessions to every existing T1w acquisition."""
    root = Path(image_root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    records: list[dict[str, str]] = []
    with Path(csv_path).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["study_name"].strip().upper() != "OASIS3":
                continue
            session = row["original_code"].strip()
            match = SUBJECT_RE.match(session)
            if match is None:
                raise ValueError(f"Invalid OASIS3 session: {session}")
            for path in sorted(root.glob(f"{session}*_T1w.nii.gz")):
                records.append({
                    "subject": match.group(1), "session": session,
                    "path": str(path), "group": row["group"].strip().upper(),
                })
    if not records:
        raise ValueError("No OASIS3 T1w images matched the clinical CSV.")
    return records


@lru_cache(maxsize=4)
def load_volume(path: str) -> np.ndarray:
    """Load one signed or unsigned MRI volume and normalize foreground robustly."""
    # The server T1w files are commonly LIA: their third array axis is A/P.
    # Canonical RAS makes axis 2 consistently inferior/superior (axial).
    volume = np.asarray(nib.as_closest_canonical(nib.load(path)).get_fdata(dtype=np.float32))
    return normalize_volume(volume)


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    """Map nonzero signed MRI intensities to [0, 1] by robust percentiles."""
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D MRI, got {volume.shape}")
    volume = np.asarray(volume, dtype=np.float32).copy()
    if not np.isfinite(volume).all():
        volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)
    foreground_mask = volume != 0
    foreground = volume[foreground_mask]
    if foreground.size:
        low, high = np.percentile(foreground, [1, 99])
        volume = np.clip((volume - low) / max(float(high - low), 1e-6), 0, 1)
        volume[~foreground_mask] = 0
    else:
        volume = np.zeros_like(volume)
    return volume.astype(np.float32, copy=False)


class OASIS3SliceDataset(Dataset):
    """Sample fixed axial slices, keeping all acquisitions of a session together."""

    def __init__(self, records: list[dict[str, str]], *, slices_per_volume: int = 5,
                 slice_range: tuple[float, float] = (0.35, 0.65),
                 channels: int = 1, transform: Any = None) -> None:
        if channels not in (1, 3):
            raise ValueError("channels must be 1 or 3 (neighboring slices).")
        if slices_per_volume < 1 or not 0 <= slice_range[0] < slice_range[1] <= 1:
            raise ValueError("Invalid slice sampling settings.")
        self.records = records
        self.channels = channels
        self.transform = transform
        self.indices = [(i, fraction) for i in range(len(records)) for fraction in
                        np.linspace(slice_range[0], slice_range[1], slices_per_volume)]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record_index, fraction = self.indices[index]
        record = self.records[record_index]
        volume = load_volume(record["path"])
        z = round(float(fraction) * (volume.shape[2] - 1))
        offsets = (0,) if self.channels == 1 else (-1, 0, 1)
        image = np.stack([volume[:, :, min(max(z + offset, 0), volume.shape[2] - 1)]
                          for offset in offsets], axis=0).copy()
        tensor = torch.from_numpy(image)
        if self.transform is not None:
            tensor = self.transform(tensor)
        return {"image": tensor, "subject": record["subject"],
                "session": record["session"], "group": record["group"]}


class OASIS3DataModule(L.LightningDataModule):
    """Use only OASIS3 and partition subjects before constructing slice datasets."""

    def __init__(self, *, csv_path: str = DEFAULT_CSV,
                 image_root: str = DEFAULT_IMAGE_ROOT, batch_size: int = 16,
                 num_workers: int = 4, pin_memory: bool = False, channels: int = 1,
                 slices_per_volume: int = 5, slice_range: tuple[float, float] = (0.35, 0.65),
                 train_ratio: float = 0.70, val_ratio: float = 0.15, seed: int = 42,
                 max_volumes: int | None = None, train_transform: Any = None,
                 eval_transform: Any = None) -> None:
        super().__init__()
        if not 0 < train_ratio < 1 or not 0 < val_ratio < 1 or train_ratio + val_ratio >= 1:
            raise ValueError("Train and validation ratios must leave a test split.")
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        self.csv_path, self.image_root = csv_path, image_root
        self.batch_size, self.num_workers, self.pin_memory = batch_size, num_workers, pin_memory
        self.channels, self.slices_per_volume, self.slice_range = channels, slices_per_volume, slice_range
        self.train_ratio, self.val_ratio, self.seed = train_ratio, val_ratio, seed
        self.max_volumes = max_volumes
        self.train_transform, self.eval_transform = train_transform, eval_transform
        self.records: list[dict[str, str]] | None = None
        self.split_subject_ids: dict[str, list[str]] = {}
        self.subject_labels: dict[str, str] = {}
        self.train_dataset: OASIS3SliceDataset | None = None
        self.val_dataset: OASIS3SliceDataset | None = None
        self.test_dataset: OASIS3SliceDataset | None = None

    def prepare_data(self) -> None:
        if self.records is None:
            self.records = read_manifest(self.csv_path, self.image_root)

    def setup(self, stage: str | None = None) -> None:
        del stage
        self.prepare_data()
        assert self.records is not None
        groups: dict[str, set[str]] = defaultdict(set)
        for row in self.records:
            groups[row["subject"]].add(row["group"])
        self.subject_labels = {subject: next(iter(labels)) for subject, labels in groups.items()
                               if len(labels) == 1 and next(iter(labels)) in ("AD", "HC")}
        strata: dict[str, list[str]] = defaultdict(list)
        for subject, labels in groups.items():
            key = next(iter(labels)) if len(labels) == 1 and next(iter(labels)) in ("AD", "HC") else "OTHER"
            strata[key].append(subject)
        rng = random.Random(self.seed)
        splits = {"train": [], "val": [], "test": []}
        for ids in strata.values():
            ids.sort()
            rng.shuffle(ids)
            n = len(ids)
            n_train, n_val = int(n * self.train_ratio), int(n * self.val_ratio)
            if n >= 3:
                n_train = max(1, min(n_train, n - 2))
                n_val = max(1, min(n_val, n - n_train - 1))
            splits["train"].extend(ids[:n_train])
            splits["val"].extend(ids[n_train:n_train + n_val])
            splits["test"].extend(ids[n_train + n_val:])
        self.split_subject_ids = {k: sorted(v) for k, v in splits.items()}
        if self.max_volumes is not None and self.max_volumes < 1:
            raise ValueError("max_volumes must be positive.")
        for name in splits:
            allowed = set(splits[name])
            rows = [r for r in self.records if r["subject"] in allowed]
            if self.max_volumes is not None:
                rows = rows[:self.max_volumes]
            dataset = OASIS3SliceDataset(rows, slices_per_volume=self.slices_per_volume,
                                         slice_range=self.slice_range, channels=self.channels,
                                         transform=self.train_transform if name == "train" else self.eval_transform)
            setattr(self, f"{name}_dataset", dataset)
        if any(set(splits[a]) & set(splits[b]) for a, b in
               (("train", "val"), ("train", "test"), ("val", "test"))):
            raise RuntimeError("Subject leakage across splits.")

    def _loader(self, dataset: OASIS3SliceDataset | None, shuffle: bool) -> DataLoader:
        if dataset is None:
            raise RuntimeError("Call setup() before requesting a dataloader.")
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.num_workers > 0, drop_last=False)

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset, False)

    def eval_dataset(self, split: str, transform: Any) -> OASIS3SliceDataset:
        """Rebuild a split with deterministic single-view embeddings."""
        dataset = getattr(self, f"{split}_dataset")
        if dataset is None:
            raise RuntimeError("Call setup() first")
        return OASIS3SliceDataset(dataset.records, channels=self.channels,
                                  slices_per_volume=self.slices_per_volume,
                                  slice_range=self.slice_range, transform=transform)
