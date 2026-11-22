"""Pseudo-label pretext tasks on 3D cubes: rotation and order (jigsaw) prediction.

The label of every training example is made by the transformation itself, so the
encoder learns anatomy without annotations (the "pseudo-label approach" of the T4
brief):

- **Rotation** (Gidaris et al., ICLR 2018; 3D version of Taleb et al., NeurIPS 2020):
  each view is rotated by one of ``K`` exact 90-degree rotations of the cube and the
  network predicts which. ``K=10`` is Taleb's set (identity + 90/180/270 degrees about
  each of the three axes), ``K=24`` the whole rotation group of the cube, ``K=4`` the
  four rotations about the axial axis. 90-degree rotations of a cube need no
  interpolation and no padding, so there are no border artifacts to read the label from.
- **Order / jigsaw** (Noroozi & Favaro, ECCV 2016; 3D puzzles of Taleb et al.): the cube is
  cut into its 2 x 2 x 2 octants, every octant is shrunk to a tile ``gap`` voxels smaller
  (random offset per tile, so edges do not line up), the tiles are put back in one of
  ``P`` permutations (chosen by maximal Hamming distance, fixed seed) on a zero canvas,
  and the network predicts the permutation. The gaps and jitter remove the low-level
  continuity shortcut; the network has to recognize which part of the hippocampal
  region each tile is.

With both tasks on, the first view of each sample is rotated and the second shuffled,
and the loss is ``rotation_weight * CE_rot + jigsaw_weight * CE_jig``. With one task,
both views feed it (twice the examples). Each task has its own 2-layer MLP head on the
pooled backbone feature; the backbone is what the probes evaluate.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from t5_ssl.models.ssl_module import SSLLightningModule

SPATIAL = (2, 3, 4)  # D, H, W of (B, C, D, H, W)


# ---------------------------------------------------------------- rotations
def _rotation_ops(classes: int) -> list[tuple[tuple[int, tuple[int, int]], ...]]:
    """Sequences of ``torch.rot90`` calls ``(k, dims)``, one per rotation class."""
    planes = ((3, 4), (2, 4), (2, 3))  # about D (axial), about H, about W
    if classes == 4:
        return [((k, planes[0]),) for k in range(4)]
    if classes == 10:
        return [()] + [((k, p),) for p in planes for k in (1, 2, 3)]
    if classes == 24:
        # breadth-first closure of the three quarter turns, deduplicated on a probe cube
        probe = torch.arange(27.0).reshape(1, 1, 3, 3, 3)
        seen, ops, frontier = {probe.flatten().tolist().__str__()}, [()], [()]
        while frontier:
            nxt = []
            for seq in frontier:
                for p in planes:
                    cand = (*seq, (1, p))
                    key = str(apply_rotation(probe, cand).flatten().tolist())
                    if key not in seen:
                        seen.add(key)
                        ops.append(cand)
                        nxt.append(cand)
            frontier = nxt
        assert len(ops) == 24
        return ops
    raise ValueError("rotation_classes must be 4, 10 or 24")


def apply_rotation(x: torch.Tensor, ops: tuple[tuple[int, tuple[int, int]], ...]) -> torch.Tensor:
    for k, dims in ops:
        x = torch.rot90(x, k, dims)
    return x


def rotate_batch(x: torch.Tensor, labels: torch.Tensor,
                 ops: list[tuple[tuple[int, tuple[int, int]], ...]]) -> torch.Tensor:
    """Rotate every cube of ``x`` (B, C, S, S, S) by its class in ``labels``."""
    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    for c in torch.unique(labels).tolist():
        idx = (labels == c).nonzero(as_tuple=True)[0]
        out[idx] = apply_rotation(x[idx], ops[c])
    return out


def rotate_planes(x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Turn every plane of 2D views ``(B, P, S, S)`` by ``labels`` quarter turns in its plane.

    The views are t5_ssl.data.brain2d planes; the three planes of a sample get the same
    turn (Gidaris et al. 2018, 4 classes).
    """
    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    for c in torch.unique(labels).tolist():
        idx = (labels == c).nonzero(as_tuple=True)[0]
        out[idx] = torch.rot90(x[idx], int(c), (2, 3))
    return out


# ---------------------------------------------------------------- jigsaw
def hamming_permutations(count: int, n: int = 8, seed: int = 0) -> np.ndarray:
    """``count`` permutations of ``n`` items chosen greedily for maximal min Hamming distance."""
    perms = np.array(list(itertools.permutations(range(n))), dtype=np.int64)
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(len(perms)))]
    dist = (perms != perms[chosen[0]]).sum(1)
    while len(chosen) < count:
        best = int(np.argmax(dist))
        chosen.append(best)
        dist = np.minimum(dist, (perms != perms[best]).sum(1))
    return perms[chosen]


def shuffle_octants(x: torch.Tensor, labels: torch.Tensor, perms: torch.Tensor, gap: int,
                    generator: torch.Generator | None = None) -> torch.Tensor:
    """Tiles of the 8 octants (side ``S/2 - gap``, jittered) placed in permutation ``labels``.

    Output cell ``j`` holds the tile cut from source octant ``perm[j]``, centered in its cell.
    """
    b, c, s = x.shape[0], x.shape[1], x.shape[-1]
    half = s // 2
    tile = half - gap
    jitter = torch.randint(0, gap + 1, (8, 3), generator=generator,
                           device=generator.device if generator is not None else "cpu").tolist()
    corners = [(i * half, j * half, k * half) for i in (0, 1) for j in (0, 1) for k in (0, 1)]
    tiles = torch.stack([x[:, :, d + jd:d + jd + tile, h + jh:h + jh + tile, w + jw:w + jw + tile]
                         for (d, h, w), (jd, jh, jw) in zip(corners, jitter, strict=True)], dim=1)
    order = perms[labels]  # (B, 8): source octant of each output cell
    tiles = tiles[torch.arange(b, device=x.device)[:, None], order]
    out = torch.zeros((b, c, s, s, s), dtype=x.dtype, device=x.device)
    off = gap // 2
    for cell, (d, h, w) in enumerate(corners):
        out[:, :, d + off:d + off + tile, h + off:h + off + tile, w + off:w + off + tile] = tiles[:, cell]
    return out


def _head(in_dim: int, hidden: int, classes: int) -> nn.Module:
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
                         nn.Linear(hidden, classes))


class PretextLightningModule(SSLLightningModule):
    """Rotation and/or jigsaw-order prediction with cross-entropy (pseudo labels)."""

    def __init__(self, *, backbone: nn.Module, rotation_classes: int = 10,
                 jigsaw_permutations: int = 0, jigsaw_gap: int = 2, head_hidden: int = 512,
                 rotation_weight: float = 1.0, jigsaw_weight: float = 1.0,
                 label_smoothing: float = 0.0, **kwargs: Any) -> None:
        kwargs.pop("projector", None)
        kwargs.pop("loss_fn", None)
        super().__init__(backbone=backbone, projector=nn.Identity(), loss_fn=None, **kwargs)
        if rotation_classes <= 1 and jigsaw_permutations <= 1:
            raise ValueError("enable rotation_classes and/or jigsaw_permutations")
        self.rotation_classes = int(rotation_classes) if rotation_classes > 1 else 0
        self.jigsaw_permutations = int(jigsaw_permutations) if jigsaw_permutations > 1 else 0
        self.jigsaw_gap = int(jigsaw_gap)
        self.rotation_weight, self.jigsaw_weight = float(rotation_weight), float(jigsaw_weight)
        self.label_smoothing = float(label_smoothing)
        dim = backbone.out_dim
        self.rotation_head = (_head(dim, head_hidden, self.rotation_classes)
                              if self.rotation_classes else None)
        self.jigsaw_head = (_head(dim, head_hidden, self.jigsaw_permutations)
                            if self.jigsaw_permutations else None)
        self._rot_ops = _rotation_ops(self.rotation_classes) if self.rotation_classes else []
        if self.jigsaw_permutations:
            self.register_buffer("perms", torch.from_numpy(
                hamming_permutations(self.jigsaw_permutations)), persistent=True)

    # ---------------------------------------------------------------- steps
    def _views(self, batch: dict[str, Any]) -> list[torch.Tensor]:
        images = batch["image"]
        views = list(images) if isinstance(images, list | tuple) else [images]
        return [v for v in views[:2] if v.shape[-1] == views[0].shape[-1]]

    def _task(self, name: str, x: torch.Tensor, generator: torch.Generator | None
              ) -> tuple[torch.Tensor, torch.Tensor]:
        n = x.shape[0]
        classes = self.rotation_classes if name == "rot" else self.jigsaw_permutations
        labels = torch.randint(0, classes, (n,), generator=generator,
                               device=generator.device if generator is not None else "cpu")
        labels = labels.to(x.device)
        planar = x.ndim == 4
        if planar and (name != "rot" or classes != 4):
            raise ValueError("2D plane views support the rotation task with rotation_classes=4")
        if name == "rot":
            xt = rotate_planes(x, labels) if planar else rotate_batch(x, labels, self._rot_ops)
        else:
            xt = shuffle_octants(x, labels, self.perms, self.jigsaw_gap, generator)
        if self.input_memory_format is not None:
            xt = xt.contiguous(memory_format=torch.channels_last if planar
                               else torch.channels_last_3d)
        head = self.rotation_head if name == "rot" else self.jigsaw_head
        logits = head(self.backbone(xt)).float()
        return F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing), \
            (logits.argmax(1) == labels).float().mean()

    def _step(self, batch: dict[str, Any], stage: str, generator: torch.Generator | None
              ) -> torch.Tensor:
        views = self._views(batch)
        tasks = [t for t, on in (("rot", self.rotation_classes), ("jig", self.jigsaw_permutations))
                 if on]
        if len(tasks) == 2:
            inputs = {"rot": views[0], "jig": views[-1]}
        else:
            inputs = {tasks[0]: torch.cat(views) if len(views) > 1 else views[0]}
        total = 0.0
        for name, x in inputs.items():
            loss, acc = self._task(name, x, generator)
            weight = self.rotation_weight if name == "rot" else self.jigsaw_weight
            total = total + weight * loss
            self.log(f"{stage}/{name}_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
            self.log(f"{stage}/{name}_acc", acc, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/loss", total, on_step=stage == "train", on_epoch=True, prog_bar=True,
                 sync_dist=True)
        return total

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train", None)

    def validation_step(self, batch: dict[str, Any], batch_idx: int,
                        dataloader_idx: int = 0) -> torch.Tensor:
        """Score fixed pseudo labels: a generator seeded per validation batch."""
        first = self._views(batch)[0]
        seed = 4242 + 1_000_003 * dataloader_idx + 7919 * batch_idx
        gen = torch.Generator(device=first.device).manual_seed(seed)
        return self._step(batch, "val", gen)

    def trainable_parameters(self) -> list[nn.Parameter]:
        heads = [h for h in (self.rotation_head, self.jigsaw_head) if h is not None]
        return list(self.backbone.parameters()) + [p for h in heads for p in h.parameters()]
