"""Whole-brain multiplanar 2D ingestion: axial, coronal and sagittal planes (2026-11-22).

**Cache** (``build_cache``, scripts/1_data/build_brain2d_cache.py). Each session's MNI
2 mm volume (97, 115, 97; axes left->right, posterior->anterior, inferior->superior) is
preprocessed as follows:

- It is cropped to the minimal bounding box of its tissue (voxels > 0).
- It is z-scored with the mean and standard deviation of those tissue voxels only. The
  transform covers the whole box, so the background (0 and the few negative
  interpolation voxels) becomes the darkest value of the session, ``-mean / std``.
- The box is stored at the origin of a canvas shared by all sessions and filled with
  that background value.

One array per session therefore carries everything (``geometry``):

- the box is where the values exceed the minimum;
- the centre of mass is the centroid weighted by ``value - minimum``, which is
  proportional to the raw tissue intensity.

**Planes** (``TriPlaneViews`` for SSL, ``TriPlaneEval`` for the probes). On each axis the
slice through the centre of mass is taken. For the SSL views it is shifted by a random
offset of ``-offset..offset`` slices; only slices whose box section is at least
``min_tissue`` tissue are eligible, so the extreme slices are discarded. Each plane is
oriented as a display (superior / anterior up) and resampled from its box to
``size x size``, stretching each axis to fill the frame. A sample is one
``(3, size, size)`` tensor with channels (axial, coronal, sagittal). The backbone
(t5_ssl.models.backbones.triplane) repeats each plane on the three RGB inputs of the 2D
encoder.

**Augmentations.** The SSL views read the same ``augmentation`` node as the 3D cubes
(t5_ssl.recipes.AUG_RECIPES) and apply its 2D counterpart. Geometry is one sampling
grid per plane, so each pixel is interpolated once from the box:

- ``side_range`` / ``centered_crop``: zoom and shift of the view relative to the
  evaluated plane, as the 3D crops relative to the evaluated cube. The evaluated plane
  is the central ``1 / 1.2`` of a virtual extract.
- ``affine_degrees``: in-plane rotation.
- ``lr_swap_p``: anatomical left-right flip of the axial and coronal planes.
- ``tio_gpu_elastic`` / ``torchio`` ``RandomElasticDeformation``: a 2D elastic field
  from a coarse control grid, displacements in box voxels.

Photometry is applied per view:

- ``intensity_scale`` around the background;
- ``blur_p`` / ``blur_sigma``;
- ``noise_std`` and ``torchio`` ``RandomNoise``. Their stds are stated on the [0, 1]
  intensities of the 3D campaigns and converted to z units with ``REF_TISSUE_STD01``.
"""

from __future__ import annotations

import json
import math
import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

PLANES = ("axial", "coronal", "sagittal")
PLANE_AXES = (2, 1, 0)  # volume axis held fixed by each plane: I->S, P->A, L->R
# Tissue standard deviation of the [0, 1] intensities the 3D recipes were calibrated on
# (normalize_volume of the LMDB volumes: 0.249-0.269 over sessions): their noise stds
# divided by it are the same noise in z units.
REF_TISSUE_STD01 = 0.26
SUPPORTED_TORCHIO = ("RandomNoise", "RandomElasticDeformation")


# ---------------------------------------------------------------- cache
def preprocess_volume(volume: np.ndarray) -> dict[str, Any]:
    """Minimal tissue box of one raw volume, z-scored over its tissue voxels (> 0)."""
    v = np.nan_to_num(np.asarray(volume, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    tissue = v > 0
    if not tissue.any():
        raise ValueError("volume without tissue voxels (> 0)")
    lo, hi = [], []
    for axis in range(3):
        present = np.flatnonzero(tissue.any(axis=tuple(a for a in range(3) if a != axis)))
        lo.append(int(present[0]))
        hi.append(int(present[-1]) + 1)
    box = np.maximum(v[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], 0.0)
    values = box[box > 0]
    mean, std = float(values.mean()), max(float(values.std()), 1e-6)
    return {"box": (box - mean) / std, "lo": lo, "hi": hi, "mean": mean, "std": std,
            "background": -mean / std}


def build_cache(cache_path: str | Path, *, lmdb_path: str | Path, idx: list[str],
                sessions: list[str] | None = None) -> dict[str, Any]:
    """Preprocess the LMDB volumes ``idx`` into one ``(N, D, H, W)`` float16 canvas array.

    ``sessions`` (same order) are checked against the LMDB ``original_code`` when given.
    """
    import lmdb
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=False)

    def volume(txn: Any, i: int, key: str) -> np.ndarray:
        raw = txn.get(key.encode("ascii"))
        if raw is None:
            raise KeyError(f"Missing LMDB key {key}")
        payload = pickle.loads(raw)
        if sessions is not None and payload["original_code"] != sessions[i]:
            raise ValueError(f"LMDB/CSV mismatch at {key}")
        return np.asarray(payload["image"])

    # Pass 1: box extents (the canvas); pass 2: preprocess straight into the memory map.
    extents = []
    with env.begin() as txn:
        for i, key in enumerate(idx):
            tissue = volume(txn, i, key) > 0
            extents.append([int(np.ptp(np.flatnonzero(tissue.any(axis=tuple(b for b in range(3)
                                                                          if b != a))))) + 1
                            for a in range(3)])
    shapes = np.array(extents)
    canvas = shapes.max(0).tolist()
    cache_path = Path(cache_path).with_suffix(".npy")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    array = np.lib.format.open_memmap(cache_path, mode="w+", dtype=np.float16,
                                      shape=(len(idx), *canvas))
    stats = []
    with env.begin() as txn:
        for i, key in enumerate(idx):
            pre = preprocess_volume(volume(txn, i, key))
            box = pre.pop("box")
            array[i] = np.float16(pre["background"])
            array[i, :box.shape[0], :box.shape[1], :box.shape[2]] = box
            stats.append(pre)
    array.flush()
    meta = {"kind": "brain2d", "sessions": len(idx), "canvas": canvas,
            "box_shape_min": shapes.min(0).tolist(), "box_shape_max": shapes.max(0).tolist(),
            "axes": ["left->right", "posterior->anterior", "inferior->superior"],
            "normalization": "z-score over tissue voxels (> 0), applied to the whole box",
            "idx": list(idx),
            "box_lo": [s["lo"] for s in stats], "box_hi": [s["hi"] for s in stats],
            "tissue_mean": [round(s["mean"], 4) for s in stats],
            "tissue_std": [round(s["std"], 4) for s in stats]}
    cache_path.with_suffix(".json").write_text(json.dumps(meta) + "\n", encoding="utf-8")
    return meta


# ---------------------------------------------------------------- geometry
def geometry(volume: torch.Tensor) -> dict[str, Any]:
    """Box shape, background, centre of mass and per-slice tissue fraction of a canvas."""
    v = volume.float()
    background = v.min()
    weight = v - background
    tissue = weight > 0
    shape, com, fractions = [], [], []
    for axis in range(3):
        others = tuple(a for a in range(3) if a != axis)
        counts = tissue.sum(dim=others)
        present = torch.nonzero(counts).flatten()
        if present.numel() == 0:
            raise ValueError("canvas without tissue")
        shape.append(int(present[-1]) + 1)
    box = weight[:shape[0], :shape[1], :shape[2]]
    total = box.sum().clamp_min(1e-12)
    for axis in range(3):
        others = tuple(a for a in range(3) if a != axis)
        profile = box.sum(dim=others)
        com.append(float((profile * torch.arange(shape[axis], dtype=profile.dtype)).sum() / total))
        area = shape[others[0]] * shape[others[1]]
        fractions.append((box > 0).sum(dim=others).float() / area)
    return {"shape": shape, "background": float(background), "com": com,
            "tissue_fraction": fractions}


_GEOMETRY: dict[bytes, dict[str, Any]] = {}
_GEOMETRY_MAX = 8192  # sessions remembered per process (a few KB each)


def geometry_cached(volume: torch.Tensor) -> dict[str, Any]:
    """:func:`geometry`, remembered per process for every canvas already seen.

    The key is a sparse lattice of the canvas values (z-scores of 1/343 of the voxels):
    sessions never share it, and computing it is ~100 times cheaper than the geometry,
    which every SSL view of every epoch needs (DataLoader workers persist across epochs).
    """
    key = volume[::7, ::7, ::7].contiguous().numpy().tobytes()
    geo = _GEOMETRY.get(key)
    if geo is None:
        if len(_GEOMETRY) >= _GEOMETRY_MAX:
            _GEOMETRY.clear()
        geo = _GEOMETRY[key] = geometry(volume)
    return geo


def eligible_slices(fraction: torch.Tensor, center: int, offset: int, min_tissue: float
                    ) -> list[int]:
    """Slices within ``center +- offset`` whose tissue fraction is at least ``min_tissue``."""
    n = fraction.numel()
    low, high = max(0, center - offset), min(n - 1, center + offset)
    chosen = [k for k in range(low, high + 1) if float(fraction[k]) >= min_tissue]
    if chosen:
        return chosen
    # no eligible slice in the window: the eligible slice nearest the centre (or the centre)
    good = torch.nonzero(fraction >= min_tissue).flatten().tolist()
    return [min(good, key=lambda k: abs(k - center))] if good else [min(max(center, 0), n - 1)]


def plane(volume: torch.Tensor, shape: list[int], axis: int, index: int) -> torch.Tensor:
    """One slice of the box as a display image ``(rows, cols)``: superior / anterior up.

    axial (axis 2): rows anterior -> posterior, columns left -> right;
    coronal (axis 1): rows superior -> inferior, columns left -> right;
    sagittal (axis 0): rows superior -> inferior, columns posterior -> anterior.
    """
    box = volume[:shape[0], :shape[1], :shape[2]]
    section = box.select(axis, index)  # the two remaining axes in order
    return section.transpose(0, 1).flip(0)


# ---------------------------------------------------------------- sampling
def _theta(h: int, w: int, *, zoom: float = 1.0, shift: tuple[float, float] = (0.0, 0.0),
           degrees: float = 0.0, flip: bool = False) -> torch.Tensor:
    """Affine map from output to input normalized coordinates (``F.affine_grid``).

    Output point ``(p, q)`` -> input ``D^-1 R D (zoom * F (p, q) + shift)``: ``F`` flips the
    columns, ``zoom > 1`` widens the field of view, ``R`` rotates by ``degrees`` in the
    isotropic voxel space of the box (``D`` = half extents in voxels).
    """
    f = torch.diag(torch.tensor([-1.0 if flip else 1.0, 1.0]))
    a = math.radians(degrees)
    rot = torch.tensor([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    d = torch.diag(torch.tensor([w / 2.0, h / 2.0]))
    m = torch.linalg.inv(d) @ rot @ d
    linear = m @ (zoom * f)
    t = m @ torch.tensor(shift, dtype=torch.float32)
    return torch.cat([linear, t[:, None]], dim=1)


def elastic_field(size: int, h: int, w: int, control_points: int, max_displacement: float,
                  locked_borders: int) -> torch.Tensor:
    """``(size, size, 2)`` displacement in input normalized units from a coarse grid.

    Control-point displacements are uniform in ``+-max_displacement`` box voxels per
    component; the ``locked_borders`` outer rings stay fixed; bicubic interpolation.
    """
    n = int(control_points)
    coarse = (torch.rand(2, n, n) * 2 - 1) * float(max_displacement)
    rings = min(int(locked_borders), n // 2)
    if rings:
        mask = torch.zeros(n, n)
        mask[rings:n - rings, rings:n - rings] = 1
        coarse = coarse * mask
    field = F.interpolate(coarse[None], size=(size, size), mode="bicubic", align_corners=True)[0]
    # voxels -> normalized units (x: columns of width w, y: rows of height h)
    field = field * torch.tensor([2.0 / w, 2.0 / h]).view(2, 1, 1)
    return field.permute(1, 2, 0)


def sample(image: torch.Tensor, background: float, size: int, theta: torch.Tensor,
           field: torch.Tensor | None = None) -> torch.Tensor:
    """``(h, w)`` -> ``(size, size)``; outside the box is background."""
    grid = F.affine_grid(theta[None], [1, 1, size, size], align_corners=False)
    if field is not None:
        grid = grid + field[None]
    out = F.grid_sample((image - background)[None, None], grid, mode="bilinear",
                        padding_mode="zeros", align_corners=False)
    return out[0, 0] + background


def _blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    radius = max(1, math.ceil(2 * sigma))
    grid = torch.arange(-radius, radius + 1, dtype=x.dtype)
    kernel = torch.exp(-grid**2 / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    out = x[:, None]
    out = F.conv2d(F.pad(out, (radius, radius, 0, 0), mode="replicate"), kernel.view(1, 1, 1, -1))
    out = F.conv2d(F.pad(out, (0, 0, radius, radius), mode="replicate"), kernel.view(1, 1, -1, 1))
    return out[:, 0]


# ---------------------------------------------------------------- transforms
class TriPlaneEval:
    """Probe input: the three centre-of-mass planes, unaugmented, ``(3, size, size)``.

    ``offsets`` (multi-slice probe, ``++eval_slice_offsets``): the three planes at each
    offset from the centre of mass, ``(3 k, size, size)`` in offset order; the backbone
    averages the ``k`` embeddings (t5_ssl.models.backbones.triplane).
    """

    def __init__(self, size: int = 224, min_tissue: float = 0.15,
                 offsets: tuple[int, ...] = (0,)) -> None:
        self.size, self.min_tissue = int(size), float(min_tissue)
        self.offsets = tuple(int(o) for o in offsets) or (0,)

    def planes(self, volume: torch.Tensor, offsets: tuple[int, ...] = (0, 0, 0)
               ) -> tuple[list[torch.Tensor], dict]:
        """Return the three box planes at ``com + offset`` (nearest eligible slice)."""
        geo = geometry_cached(volume)
        out = []
        for axis, offset in zip(PLANE_AXES, offsets, strict=True):
            center = round(geo["com"][axis])
            index = eligible_slices(geo["tissue_fraction"][axis], center + offset, 0,
                                    self.min_tissue)[0]
            out.append(plane(volume.float(), geo["shape"], axis, index))
        return out, geo

    def __call__(self, volume: torch.Tensor) -> torch.Tensor:
        identity = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        out = []
        for offset in self.offsets:
            images, geo = self.planes(volume, (offset,) * 3)
            out += [sample(img, geo["background"], self.size, identity) for img in images]
        return torch.stack(out)


class TriPlaneEvalViews(TriPlaneEval):
    """Fixed validation pair for the SSL loss: the planes and their left-right flip."""

    def __call__(self, volume: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        first = super().__call__(volume)
        flipped = first.clone()
        flipped[:2] = flipped[:2].flip(-1)  # axial and coronal columns run left -> right
        return first, flipped


class TriPlaneViews:
    """Two SSL views (plus ``n_local_crops`` smaller ones) of a whole-brain canvas.

    ``slices`` (``data.train_slices``) > 1: each view holds that many plane triples, at
    distinct slices of the ``+-offset`` window when there are enough, ``(3 k, size, size)``;
    the backbone averages their embeddings, as the multi-slice probe does.
    """

    def __init__(self, size: int = 224, *, offset: int = 5, min_tissue: float = 0.15,
                 slices: int = 1,
                 side_range: tuple[float, float] = (1 / 1.2, 1 / 1.2),
                 centered_crop: bool = True, eval_side: float = 1 / 1.2,
                 degrees: float = 0.0, lr_swap_p: float = 0.5,
                 elastic: dict | None = None, blur_p: float = 0.0,
                 blur_sigma: tuple[float, float] = (0.1, 1.5), noise_std: float = 0.0,
                 noise_range: tuple[float, float] | None = None,
                 intensity_scale: tuple[float, float] = (1.0, 1.0),
                 n_local_crops: int = 0, local_size: int = 96,
                 local_side_range: tuple[float, float] = (0.4, 0.6)) -> None:
        self.size, self.offset, self.min_tissue = int(size), int(offset), float(min_tissue)
        self.slices = max(1, int(slices))
        self.side_range, self.centered_crop = tuple(side_range), bool(centered_crop)
        self.eval_side, self.degrees, self.lr_swap_p = float(eval_side), float(degrees), lr_swap_p
        self.elastic = elastic
        self.blur_p, self.blur_sigma = blur_p, tuple(blur_sigma)
        self.noise_std, self.noise_range = float(noise_std), noise_range
        self.intensity_scale = tuple(intensity_scale)
        self.n_local_crops, self.local_size = int(n_local_crops), int(local_size)
        self.local_side_range = tuple(local_side_range)

    def _view(self, volume: torch.Tensor, geo: dict, size: int,
              side_range: tuple[float, float]) -> torch.Tensor:
        flip = random.random() < self.lr_swap_p
        side = random.uniform(*side_range)
        zoom = side / self.eval_side
        reach = 0.0 if self.centered_crop else max(0.0, (1.0 - side) / self.eval_side)
        indices = []
        for axis in PLANE_AXES:
            window = eligible_slices(geo["tissue_fraction"][axis], round(geo["com"][axis]),
                                     self.offset, self.min_tissue)
            indices.append(random.sample(window, self.slices) if len(window) >= self.slices
                           else random.choices(window, k=self.slices))
        out = []
        for k in range(self.slices):
            for p, axis in enumerate(PLANE_AXES):
                image = plane(volume, geo["shape"], axis, indices[p][k])
                h, w = image.shape
                shift = (random.uniform(-reach, reach), random.uniform(-reach, reach))
                theta = _theta(h, w, zoom=zoom, shift=shift,
                               degrees=random.uniform(-self.degrees, self.degrees),
                               flip=flip and p < 2)  # the sagittal plane has no left-right axis
                field = (elastic_field(size, h, w, **self.elastic) if self.elastic else None)
                out.append(sample(image, geo["background"], size, theta, field))
        return self._photometric(torch.stack(out), geo["background"])

    def _photometric(self, view: torch.Tensor, background: float) -> torch.Tensor:
        if self.blur_p and random.random() < self.blur_p:
            view = _blur(view, random.uniform(*self.blur_sigma))
        scale = random.uniform(*self.intensity_scale)
        if scale != 1.0:
            view = background + (view - background) * scale
        std = self.noise_std
        if self.noise_range is not None:
            std = std + random.uniform(*self.noise_range)
        if std > 0:
            view = view + torch.randn_like(view) * std
        return view

    def __call__(self, volume: torch.Tensor) -> tuple[torch.Tensor, ...]:
        volume = volume.float()
        geo = geometry_cached(volume)
        views = [self._view(volume, geo, self.size, self.side_range) for _ in range(2)]
        views += [self._view(volume, geo, self.local_size, self.local_side_range)
                  for _ in range(self.n_local_crops)]
        return tuple(views)


# ---------------------------------------------------------------- from the config
def _elastic_spec(aug: dict) -> dict | None:
    spec = aug.get("tio_gpu_elastic")
    if spec:
        return {"control_points": int(spec["num_control_points"]),
                "max_displacement": float(spec["max_displacement"]),
                "locked_borders": int(spec.get("locked_borders", 2))}
    for t in aug.get("torchio") or []:
        if t.get("name") == "RandomElasticDeformation":
            return {"control_points": int(t.get("num_control_points", 7)),
                    "max_displacement": float(t.get("max_displacement", 7.5)),
                    "locked_borders": int(t.get("locked_borders", 2))}
    if aug.get("gpu_elastic"):
        raise NotImplementedError("brain2d: use tio_gpu_elastic (TorchIO's field) for elastic views")
    return None


def _noise_range(aug: dict) -> tuple[float, float] | None:
    for t in aug.get("torchio") or []:
        if t.get("name") == "RandomNoise":
            std = t.get("std", 0.25)
            low, high = (0.0, float(std)) if isinstance(std, int | float) else map(float, std)
            return low / REF_TISSUE_STD01, high / REF_TISSUE_STD01
    return None


def build_brain2d_transforms(cfg: Any) -> tuple[TriPlaneViews, TriPlaneEvalViews]:
    """SSL views and the fixed validation pair of a ``data.source=brain2d`` run."""
    from omegaconf import OmegaConf
    aug = OmegaConf.to_container(cfg.augmentation, resolve=True)
    unknown = [t.get("name") for t in aug.get("torchio") or []
               if t.get("name") not in SUPPORTED_TORCHIO]
    if unknown:
        raise NotImplementedError(f"brain2d views support torchio {SUPPORTED_TORCHIO}, got {unknown}")
    size = int(cfg.data.roi_size)
    eval_side = size / round(1.2 * size)  # recipes.aug_spec writes "eval" sides this way
    min_tissue = float(cfg.data.get("min_tissue", 0.15))
    train = TriPlaneViews(
        size, offset=int(cfg.data.get("slice_offset", 5)), min_tissue=min_tissue,
        slices=int(cfg.data.get("train_slices", 1)),
        side_range=tuple(aug.get("side_range", (eval_side, eval_side))),
        centered_crop=bool(aug.get("centered_crop", True)), eval_side=eval_side,
        degrees=float(aug.get("affine_degrees", 0.0)), lr_swap_p=float(aug.get("lr_swap_p", 0.5)),
        elastic=_elastic_spec(aug), blur_p=float(aug.get("blur_p", 0.0)),
        blur_sigma=tuple(aug.get("blur_sigma", (0.1, 1.5))),
        noise_std=float(aug.get("noise_std", 0.0)) / REF_TISSUE_STD01,
        noise_range=_noise_range(aug),
        intensity_scale=tuple(aug.get("intensity_scale", (1.0, 1.0))),
        n_local_crops=int(aug.get("n_local_crops", 0)),
        local_size=int(aug.get("local_size", 96)),
        local_side_range=tuple(aug.get("local_side_range", (0.4, 0.6))))
    return train, TriPlaneEvalViews(size, min_tissue)
