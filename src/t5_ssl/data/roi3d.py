"""Hippocampal 3D ROI cubes from the OASIS3 LMDB: preprocessing cache, SSL views, datamodule.

The LMDB volumes are 2 mm MNI grids of shape (97, 115, 97) with axes
(left->right, posterior->anterior, inferior->superior), and most carry a
FreeSurfer-style parcellation in ``roi`` (aseg label 17 = left hippocampus,
53 = right hippocampus). ``build_cache`` crops, once, one cube per hemisphere
around the population-mean hippocampal centroid, shifted when needed so that
it stays inside the brain of every session (no slice without brain voxels),
and stores the pair as channels ``[left, right mirrored left-right]``: both
hippocampi then share one orientation, and an anatomical left-right flip of
the head is exactly a channel swap.

A cube of ``size`` voxels per side (32 -> 64 mm, 48 -> 96 mm) is what the
network sees; the cache keeps ``extract = round(1.2 * size)`` voxels so random
crops and rotations have margin. Evaluation uses the central ``size`` cube.
"""

from __future__ import annotations

import contextlib
import json
import math
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import lightning as L
import lmdb
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from t5_ssl.data.augmentations import build_torchio_transform
from t5_ssl.data.oasis3_datamodule import normalize_volume
from t5_ssl.data.oasis3_lmdb_datamodule import clinical_rows, select_rows

HIPPOCAMPUS_LABELS = {"left": 17, "right": 53}
LR_AXIS = 0
BRAIN_THRESHOLD = 0.05  # normalized intensity that counts as brain tissue


def extract_size(size: int) -> int:
    return round(1.2 * size)


def _payload(txn: Any, key: str) -> dict:
    value = txn.get(key.encode("ascii"))
    if value is None:
        raise KeyError(f"Missing LMDB key {key}")
    return pickle.loads(value)


def _cube_start(center: float, extract: int, low: int, high: int) -> int:
    """First index of a cube around `center`, shifted to lie inside [low, high]."""
    start = round(center) - extract // 2
    if high - low + 1 >= extract:
        start = min(max(start, low), high - extract + 1)
    return start


def build_cache(cache_path: str | Path, *, csv_path: str, lmdb_path: str,
                manifest_path: str, size: int) -> dict[str, Any]:
    """Crop and store both hippocampal cubes of every manifest session."""
    selected, _ = select_rows(csv_path, manifest_path, False, 42, 0.7, 0.15)
    rows = [row for split in ("train", "val", "test") for row in selected[split]]
    extract = extract_size(size)
    train_idx = {row["idx"] for row in selected["train"]}
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=False)

    def volume_of(txn: Any, row: dict[str, str]) -> tuple[np.ndarray, dict]:
        payload = _payload(txn, row["idx"])
        if payload["original_code"] != row["session"]:
            raise ValueError(f"LMDB/CSV mismatch at {row['idx']}")
        return normalize_volume(np.asarray(payload["image"])), payload

    # Pass 1: shared brain box and hippocampal centroids (two passes keep RAM low).
    centroids: dict[str, list[np.ndarray]] = defaultdict(list)
    boxes = []
    with env.begin() as txn:
        for row in rows:
            volume, payload = volume_of(txn, row)
            brain = np.argwhere(volume > BRAIN_THRESHOLD)
            boxes.append((brain.min(axis=0), brain.max(axis=0)))
            # Anatomy only (no diagnosis) and only from training sessions.
            if "roi" in payload and row["idx"] in train_idx:
                roi = np.asarray(payload["roi"])
                for side, label in HIPPOCAMPUS_LABELS.items():
                    voxels = np.argwhere(roi == label)
                    if len(voxels):
                        centroids[side].append(voxels.mean(axis=0))
    low = np.max([box[0] for box in boxes], axis=0)   # brain box shared by every session
    high = np.min([box[1] for box in boxes], axis=0)
    centers = {side: np.mean(values, axis=0) for side, values in centroids.items()}
    starts = {side: [_cube_start(centers[side][a], extract, int(low[a]), int(high[a]))
                     for a in range(3)] for side in HIPPOCAMPUS_LABELS}
    # Pass 2: crop.
    cubes = np.empty((len(rows), 2, extract, extract, extract), dtype=np.float16)
    empty_slices: Counter[str] = Counter()
    txn = env.begin()
    for i, row in enumerate(rows):
        volume = volume_of(txn, row)[0]
        for channel, side in enumerate(("left", "right")):
            s = starts[side]
            cube = volume[s[0]:s[0] + extract, s[1]:s[1] + extract, s[2]:s[2] + extract]
            if cube.shape != (extract,) * 3:
                raise RuntimeError(f"{side} cube leaves the volume: start={s}")
            if side == "right":
                cube = np.flip(cube, axis=LR_AXIS)
            cubes[i, channel] = cube
            # Slices of the evaluated central cube that contain no brain voxel.
            margin = (extract - size) // 2
            core = cube[margin:margin + size, margin:margin + size, margin:margin + size]
            for axis in range(3):
                other = tuple(a for a in range(3) if a != axis)
                empty_slices[f"{side}_axis{axis}"] += int(((core > BRAIN_THRESHOLD).sum(axis=other)
                                                           == 0).sum())
    txn.abort()
    meta = {"size": size, "extract": extract, "sessions": len(rows),
            "hippocampus_centers_voxel": {k: v.round(2).tolist() for k, v in centers.items()},
            "centroid_sessions": {k: len(v) for k, v in centroids.items()},
            "shared_brain_box": [low.tolist(), high.tolist()],
            "cube_starts": starts, "empty_slices_in_evaluated_cube": dict(empty_slices),
            "channels": ["left hippocampus", "right hippocampus mirrored left-right"]}
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # Raw .npy so training processes can memory-map it and share the OS page cache.
    meta["idx"] = [row["idx"] for row in rows]
    np.save(cache_path.with_suffix(".npy"), cubes)
    cache_path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


# Bilateral regions for multi-ROI caches: (left labels, right labels) of the
# FreeSurfer aparc+aseg parcellation stored in the LMDB ``roi`` volumes.
REGIONS: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {
    "hippocampus": ((17,), (53,)),
    "amygdala": ((18,), (54,)),
    "ventricle": ((4,), (43,)),                       # lateral ventricle
    "entorhinal": ((1006,), (2006,)),
    # posterior + isthmus cingulate and precuneus (posteromedial cortex)
    "posteromedial": ((1023, 1010, 1025), (2023, 2010, 2025)),
}


def build_region_cache(cache_path: str | Path, *, csv_path: str, lmdb_path: str,
                       manifest_path: str, size: int, regions: list[str]) -> dict[str, Any]:
    """Like :func:`build_cache` for any list of bilateral ``REGIONS``.

    Channels are ``[left, right mirrored]`` per region, in order (so a
    left-right flip of the head swaps the channels of each pair). Each cube is
    centered on the region's population-mean centroid over the training
    sessions, shifted into the brain box shared by every session when it fits
    and always into the volume (a 64-voxel ROI needs a 77-voxel cube, wider
    than that box). Cubes are written to a memory-mapped ``.npy``.
    """
    selected, _ = select_rows(csv_path, manifest_path, False, 42, 0.7, 0.15)
    rows = [row for split in ("train", "val", "test") for row in selected[split]]
    extract = extract_size(size)
    train_idx = {row["idx"] for row in selected["train"]}
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=False)
    sides = [(region, side, labels) for region in regions
             for side, labels in zip(("left", "right"), REGIONS[region], strict=True)]
    sums = {(r, s): np.zeros(3) for r, s, _ in sides}
    counts = {(r, s): 0 for r, s, _ in sides}
    low = high = None
    shape = None
    with env.begin() as txn:
        for row in rows:
            payload = _payload(txn, row["idx"])
            if payload["original_code"] != row["session"]:
                raise ValueError(f"LMDB/CSV mismatch at {row['idx']}")
            volume = normalize_volume(np.asarray(payload["image"]))
            shape = volume.shape
            brain = np.argwhere(volume > BRAIN_THRESHOLD)
            lo, hi = brain.min(axis=0), brain.max(axis=0)
            low = lo if low is None else np.maximum(low, lo)
            high = hi if high is None else np.minimum(high, hi)
            if "roi" in payload and row["idx"] in train_idx:
                roi = np.asarray(payload["roi"])
                for region, side, labels in sides:
                    voxels = np.argwhere(np.isin(roi, labels))
                    if len(voxels):
                        sums[(region, side)] += voxels.mean(axis=0)
                        counts[(region, side)] += 1
    assert shape is not None and low is not None and high is not None
    centers = {key: sums[key] / max(1, counts[key]) for key in sums}

    def start(center: float, axis: int) -> int:
        first = _cube_start(center, extract, int(low[axis]), int(high[axis]))
        return int(min(max(first, 0), shape[axis] - extract))

    starts = {key: [start(centers[key][a], a) for a in range(3)] for key in centers}
    cache_path = Path(cache_path).with_suffix(".npy")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cubes = np.lib.format.open_memmap(cache_path, mode="w+", dtype=np.float16,
                                      shape=(len(rows), len(sides), extract, extract, extract))
    with env.begin() as txn:
        for i, row in enumerate(rows):
            volume = normalize_volume(np.asarray(_payload(txn, row["idx"])["image"]))
            for channel, (region, side, _) in enumerate(sides):
                o = starts[(region, side)]
                cube = volume[o[0]:o[0] + extract, o[1]:o[1] + extract, o[2]:o[2] + extract]
                if side == "right":
                    cube = np.flip(cube, axis=LR_AXIS)
                cubes[i, channel] = cube
    cubes.flush()
    meta = {"size": size, "extract": extract, "sessions": len(rows), "regions": regions,
            "channels": [f"{region} {side}" + (" mirrored left-right" if side == "right" else "")
                         for region, side, _ in sides],
            "labels": {region: [list(v) for v in REGIONS[region]] for region in regions},
            "centers_voxel": {f"{r}_{s}": centers[(r, s)].round(2).tolist() for r, s, _ in sides},
            "centroid_sessions": {f"{r}_{s}": counts[(r, s)] for r, s, _ in sides},
            "cube_starts": {f"{r}_{s}": starts[(r, s)] for r, s, _ in sides},
            "shared_brain_box": [low.tolist(), high.tolist()],
            "idx": [row["idx"] for row in rows]}
    cache_path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def swap_hemispheres(view: torch.Tensor) -> torch.Tensor:
    """Anatomical left-right flip: swap the [left, right mirrored] channels of every pair."""
    return view.reshape(-1, 2, *view.shape[1:]).flip(1).reshape(view.shape)


# ---------------------------------------------------------------- views
def _rotation_matrix(degrees: float) -> torch.Tensor:
    angles = [math.radians(random.uniform(-degrees, degrees)) for _ in range(3)]
    rx, ry, rz = angles
    x = torch.tensor([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    y = torch.tensor([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    z = torch.tensor([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
    return (z @ y @ x).float()


def _gaussian_blur3d(view: torch.Tensor, sigma: float) -> torch.Tensor:
    radius = 2  # kernel 5, as the 2D pipeline
    grid = torch.arange(-radius, radius + 1, dtype=view.dtype)
    kernel = torch.exp(-grid**2 / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    out = view.unsqueeze(1)  # (C, 1, D, H, W): blur every channel independently
    for dim in range(3):
        shape = [1, 1, 1, 1, 1]
        shape[2 + dim] = len(kernel)
        padding = [0, 0, 0]
        padding[dim] = radius
        out = F.conv3d(out, kernel.view(shape), padding=tuple(padding))
    return out.squeeze(1)


def center_crop(cube: torch.Tensor, size: int) -> torch.Tensor:
    margin = (cube.shape[-1] - size) // 2
    return cube[:, margin:margin + size, margin:margin + size, margin:margin + size]


class ROIView3DTransform:
    """Two SSL views of a cached ``(2, E, E, E)`` hippocampal pair.

    Mirrors ``SSLViewTransform``: TorchIO artifacts on the full cached cube,
    then a random sub-cube resized to ``size``, flips, rotation about the three
    axes, blur, intensity scale and Gaussian noise.

    ``rotation_padding`` is ``grid_sample``'s ``padding_mode`` for the voxels
    the rotation brings in from outside the view. ``"zeros"`` (the default and
    every campaign up to 2026-10-01) fills the corners with black wedges whose
    shape depends only on the angles; BYOL's trained encoder collapsed onto
    that subject-independent cue (campaign_ssl3d_v2/byol_sensitivity.json).
    ``"border"`` repeats the edge voxels and ``"reflection"`` mirrors them.
    ``centered_crop`` takes the sub-cube at the center instead of a random offset
    (no translation): with a side of ``size`` it is exactly the evaluation cube.
    """

    def __init__(self, size: int = 32, side_range: tuple[float, float] = (0.84, 1.0),
                 lr_swap_p: float = 0.5, ap_flip_p: float = 0.0, degrees: float = 10.0,
                 blur_p: float = 0.0, blur_sigma: tuple[float, float] = (0.1, 1.5),
                 noise_std: float = 0.0, intensity_scale: tuple[float, float] = (1.0, 1.0),
                 torchio_transforms: list[dict] | None = None, n_local_crops: int = 0,
                 local_size: int = 16, local_side_range: tuple[float, float] = (0.4, 0.6),
                 rotation_padding: str = "zeros", centered_crop: bool = False,
                 defer_crop: bool = False) -> None:
        if rotation_padding not in ("zeros", "border", "reflection"):
            raise ValueError("rotation_padding must be 'zeros', 'border' or 'reflection'")
        self.size, self.side_range = size, side_range
        self.rotation_padding = rotation_padding
        self.centered_crop = centered_crop
        # DINOv2 multi-crop: extra smaller views, like DINOv2ViewTransform in 2D.
        self.n_local_crops, self.local_size = n_local_crops, local_size
        self.local_side_range = local_side_range
        self.lr_swap_p, self.ap_flip_p, self.degrees = lr_swap_p, ap_flip_p, degrees
        self.blur_p, self.blur_sigma = blur_p, blur_sigma
        self.noise_std, self.intensity_scale = noise_std, intensity_scale
        self.torchio_transform = build_torchio_transform(torchio_transforms, spatial_dims=3)
        # defer_crop: return the whole cached cube (flips only); a GPU deformation of the
        # cube and the central crop follow (finish_deferred_views). Valid only for views
        # that are the evaluation cube itself: centered, eval side, no rotation/photometry.
        self.defer_crop = defer_crop
        if defer_crop and not (centered_crop and degrees == 0 and torchio_transforms is None
                               and blur_p == 0 and noise_std == 0 and n_local_crops == 0
                               and tuple(intensity_scale) == (1.0, 1.0)):
            raise ValueError("defer_crop needs centered evaluation-cube views without "
                             "rotation, TorchIO, photometric changes or local crops")

    def _make_view(self, cube: torch.Tensor, size: int | None = None,
                   side_range: tuple[float, float] | None = None) -> torch.Tensor:
        size, side_range = size or self.size, side_range or self.side_range
        if self.defer_crop:
            if random.random() < self.lr_swap_p:
                cube = swap_hemispheres(cube)
            return cube.flip(2) if random.random() < self.ap_flip_p else cube
        if self.torchio_transform is not None:
            cube = self.torchio_transform(cube)
        extract = cube.shape[-1]
        side = random.randint(round(side_range[0] * extract), round(side_range[1] * extract))
        o = ([(extract - side) // 2] * 3 if self.centered_crop
             else [random.randint(0, extract - side) for _ in range(3)])
        view = cube[:, o[0]:o[0] + side, o[1]:o[1] + side, o[2]:o[2] + side]
        view = F.interpolate(view[None], size=(size,) * 3, mode="trilinear",
                             align_corners=False)[0]
        if random.random() < self.lr_swap_p:  # anatomical left-right flip of the head
            view = swap_hemispheres(view)
        if random.random() < self.ap_flip_p:
            view = view.flip(2)
        if self.degrees > 0:
            theta = torch.zeros(1, 3, 4)
            theta[0, :, :3] = _rotation_matrix(self.degrees)
            grid = F.affine_grid(theta, [1, *view.shape], align_corners=False)
            view = F.grid_sample(view[None], grid, mode="bilinear",
                                 padding_mode=self.rotation_padding, align_corners=False)[0]
        return self._photometric(view)

    def photometric_only(self, cube: torch.Tensor) -> torch.Tensor:
        """TorchIO artifacts and intensity changes on the central cube, no geometry."""
        if self.torchio_transform is not None:
            cube = self.torchio_transform(cube)
        return self._photometric(center_crop(cube, self.size))

    def _photometric(self, view: torch.Tensor) -> torch.Tensor:
        if random.random() < self.blur_p:
            view = _gaussian_blur3d(view, random.uniform(*self.blur_sigma))
        view = view * random.uniform(*self.intensity_scale)
        if self.noise_std > 0:
            view = view + torch.randn_like(view) * self.noise_std
        return view.clamp_(0.0, 1.0)

    def __call__(self, cube: torch.Tensor) -> tuple[torch.Tensor, ...]:
        locals_ = tuple(self._make_view(cube, self.local_size, self.local_side_range)
                        for _ in range(self.n_local_crops))
        return self._make_view(cube), self._make_view(cube), *locals_


class ROIEval3D:
    """The central ``size`` cube, unaugmented (probe input)."""

    def __init__(self, size: int = 32) -> None:
        self.size = size

    def __call__(self, cube: torch.Tensor) -> torch.Tensor:
        return center_crop(cube, self.size).clamp(0.0, 1.0)


class ROIEvalViews3D(ROIEval3D):
    """Fixed validation pair for the SSL loss: the cube and its left-right flip."""

    def __call__(self, cube: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        first = super().__call__(cube)
        return first, swap_hemispheres(first)


def finish_deferred_views(views: torch.Tensor, spec: dict, size: int) -> torch.Tensor:
    """TorchIO elastic deformation of whole cubes ``(B, C, E, E, E)``, central crop, clamp."""
    from t5_ssl.data.gpu_augment import tio_elastic_deform
    deformed = tio_elastic_deform(views, **spec)
    margin = (views.shape[-1] - size) // 2
    crop = deformed[:, :, margin:margin + size, margin:margin + size, margin:margin + size]
    return crop.clamp(0.0, 1.0)


class FinishedViews:
    """Deferred views finished per sample (CPU analyses: identity, collapse, supervised)."""

    def __init__(self, transform: ROIView3DTransform, spec: dict, size: int) -> None:
        self.transform, self.spec, self.size = transform, spec, size

    def _finish(self, view: torch.Tensor) -> torch.Tensor:
        return finish_deferred_views(view[None], self.spec, self.size)[0]

    def _make_view(self, cube: torch.Tensor) -> torch.Tensor:
        return self._finish(self.transform._make_view(cube))

    def __call__(self, cube: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(self._finish(view) for view in self.transform(cube))


def analysis_views(cfg: Any) -> Any:
    """Return the run's own training views as finished samples (deferred GPU recipes too)."""
    train = build_roi3d_transforms(cfg)[0]
    spec = cfg.augmentation.get("tio_gpu_elastic")
    if spec:
        from omegaconf import OmegaConf
        return FinishedViews(train, dict(OmegaConf.to_container(spec, resolve=True)),
                             int(cfg.data.roi_size))
    return train


def build_roi3d_transforms(cfg: Any) -> tuple[ROIView3DTransform, ROIEvalViews3D]:
    aug, size = cfg.augmentation, int(cfg.data.roi_size)
    from omegaconf import OmegaConf
    if aug.get("tio_gpu_elastic") and str(aug.get("val_views", "fixed")) == "train":
        raise ValueError("tio_gpu_elastic views are finished on the GPU in training only")
    train = ROIView3DTransform(
        size=size, side_range=tuple(aug.get("side_range", (0.84, 1.0))),
        lr_swap_p=aug.get("lr_swap_p", 0.5), ap_flip_p=aug.get("ap_flip_p", 0.0),
        degrees=aug.get("affine_degrees", 10.0), blur_p=aug.get("blur_p", 0.0),
        blur_sigma=tuple(aug.get("blur_sigma", (0.1, 1.5))), noise_std=aug.get("noise_std", 0.0),
        intensity_scale=tuple(aug.get("intensity_scale", (1.0, 1.0))),
        torchio_transforms=(OmegaConf.to_container(aug.torchio, resolve=True)
                            if aug.get("torchio") else None),
        n_local_crops=aug.get("n_local_crops", 0), local_size=aug.get("local_size", 16),
        local_side_range=tuple(aug.get("local_side_range", (0.4, 0.6))),
        rotation_padding=str(aug.get("rotation_padding", "zeros")),
        centered_crop=bool(aug.get("centered_crop", False)),
        defer_crop=bool(aug.get("tio_gpu_elastic")))
    return train, ROIEvalViews3D(size)


# ---------------------------------------------------------------- data
@contextlib.contextmanager
def _seeded_rng(seed: int):
    """Seed python, numpy and torch for one call, then restore their previous state."""
    states = random.getstate(), np.random.get_state(), torch.get_rng_state()
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.set_rng_state(states[2])


class ROI3DDataset(Dataset):
    """Cached hippocampal pairs; one random session per subject when training.

    ``view_seed`` makes a random transform deterministic per record (seed
    ``view_seed + index``), so augmented validation views are identical every epoch.
    """

    def __init__(self, records: list[dict[str, str]], cubes: dict[str, np.ndarray],
                 transform: Any = None, one_session_per_subject: bool = False,
                 view_seed: int | None = None) -> None:
        self.records, self.cubes, self.transform = records, cubes, transform
        self.one_session_per_subject, self.view_seed = one_session_per_subject, view_seed
        by_subject: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(records):
            by_subject[record["subject"]].append(index)
        self.subject_record_indices = [by_subject[s] for s in sorted(by_subject)]
        # Same (record, position) layout as the slice dataset, for shared probe code.
        self.indices = [(i, 0.5) for i in range(len(records))]

    def __len__(self) -> int:
        return len(self.subject_record_indices) if self.one_session_per_subject else len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.one_session_per_subject:
            index = random.choice(self.subject_record_indices[index])
        record = self.records[index]
        cube = torch.from_numpy(self.cubes[record["idx"]].astype(np.float32))
        if self.transform is None:
            image = cube
        elif self.view_seed is None:
            image = self.transform(cube)
        else:
            with _seeded_rng(self.view_seed + index):
                image = self.transform(cube)
        return {"image": image, "subject": record["subject"], "session": record["session"],
                "group": record["group"]}


class ROI3DDataModule(L.LightningDataModule):
    """Subject-safe splits from the manifest, images from the ROI cube cache."""

    def __init__(self, *, csv_path: str, cache_path: str, manifest_path: str,
                 batch_size: int = 16, num_workers: int = 0, pin_memory: bool = False,
                 one_session_per_subject_train: bool = True, seed: int = 42,
                 train_transform: Any = None, eval_transform: Any = None,
                 prefetch_factor: int | None = None, val_transform: Any = None,
                 val_view_seed: int | None = None) -> None:
        super().__init__()
        self.prefetch_factor = prefetch_factor
        # Validation SSL views: ``eval_transform`` unless ``val_transform`` is given
        # (e.g. the training augmentations, made fixed per record by ``val_view_seed``).
        self.val_transform, self.val_view_seed = val_transform, val_view_seed
        self.csv_path, self.cache_path, self.manifest_path = csv_path, cache_path, manifest_path
        self.batch_size, self.num_workers, self.pin_memory = batch_size, num_workers, pin_memory
        self.one_session_per_subject_train, self.seed = one_session_per_subject_train, seed
        self.train_transform, self.eval_transform = train_transform, eval_transform
        self.split_subject_ids: dict[str, list[str]] = {}
        self.subject_labels: dict[str, str] = {}
        self.cubes: dict[str, np.ndarray] = {}
        self.train_dataset = self.val_dataset = self.test_dataset = None
        self.unlabeled_subjects: list[str] = []
        self.extra_test_splits: list[str] = []

    def prepare_data(self) -> None:
        if not Path(self.cache_path).is_file():
            raise FileNotFoundError(f"{self.cache_path}: run scripts/run_roi3d.py preprocess first")

    def setup(self, stage: str | None = None) -> None:
        del stage
        self.prepare_data()
        labels: dict[str, set[str]] = defaultdict(set)
        for row in clinical_rows(self.csv_path):
            labels[row["subject"]].add(row["group"])
        self.subject_labels = {s: next(iter(v)) for s, v in labels.items()
                               if len(v) == 1 and next(iter(v)) in ("AD", "HC")}
        selected, self.split_subject_ids = select_rows(self.csv_path, self.manifest_path, False,
                                                       self.seed, 0.7, 0.15)
        # SSL-only train subjects (manifest labeled=0): hidden from probes and supervised models.
        self.unlabeled_subjects = sorted({row["subject"] for row in selected["train"]
                                          if row.get("labeled") == "0"})
        for subject in self.unlabeled_subjects:
            self.subject_labels.pop(subject, None)
        # Extra held-out splits (test_<name>): each is scored together with test.
        self.extra_test_splits = [split for split in selected if split.startswith("test_")]
        cubes = np.load(self.cache_path, mmap_mode="r")
        meta_path = Path(self.cache_path).with_suffix(".json")
        order = json.loads(meta_path.read_text(encoding="utf-8"))["idx"]
        self.cubes = {idx: cubes[i] for i, idx in enumerate(order)}
        missing = [row["idx"] for rows in selected.values() for row in rows
                   if row["idx"] not in self.cubes]
        if missing:
            raise ValueError(f"{len(missing)} sessions missing from {self.cache_path}")
        for split, rows in selected.items():
            transform, view_seed = self.eval_transform, None
            if split == "train":
                transform = self.train_transform
            elif split == "val" and self.val_transform is not None:
                transform, view_seed = self.val_transform, self.val_view_seed
            setattr(self, f"{split}_dataset", ROI3DDataset(
                rows, self.cubes, transform, view_seed=view_seed,
                one_session_per_subject=split == "train" and self.one_session_per_subject_train))

    def _loader(self, dataset: ROI3DDataset | None, shuffle: bool) -> DataLoader:
        if dataset is None:
            raise RuntimeError("Call setup() before requesting a dataloader")
        extra = ({"prefetch_factor": self.prefetch_factor}
                 if self.num_workers > 0 and self.prefetch_factor else {})
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.num_workers > 0, **extra)

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset, False)

    def eval_dataset(self, split: str, transform: Any) -> ROI3DDataset:
        """Rebuild a split with every session and a deterministic transform."""
        dataset = getattr(self, f"{split}_dataset")
        if dataset is None:
            raise RuntimeError("Call setup() first")
        return ROI3DDataset(dataset.records, self.cubes, transform)
