"""Test-time shifts of the probe input (``++eval_tta_shift=<s>``, ingestion tuning 2026-11-22).

The probes embed the clean central ``size`` cube of every session (``ROIEval3D``).
:class:`ShiftTTA` embeds that cube and its six one-axis shifts by ``s`` voxels (2 mm per
voxel) and averages the seven embeddings: the encoder reads the same anatomy at seven
nearby positions, as the cubes are placed at the train-set centroid of the region and a
session's hippocampus sits a few millimetres off it. The shifted cubes are real voxels of
the cache (its cubes are 38 voxels wide around the 32 used), not padding: the input is
the central ``size + 2 s`` cube (``tta_transform``) and the shifts are crops of it.

Training is unchanged; ``t5_ssl.evaluation.run_scoring.evaluate_encoders`` scores the
encoder both ways and stores the shifted scores under ``main.tta`` of ``result.json``.
"""

from __future__ import annotations

import torch
from torch import nn

from t5_ssl.data.roi3d import ROIEval3D


def offsets(shift: int) -> list[tuple[int, int, int]]:
    """Crop corners in the ``size + 2 shift`` cube: the centre first, then +-shift per axis."""
    s = int(shift)
    out = [(s, s, s)]
    for axis in range(3):
        for d in (-s, s):
            out.append(tuple(s + d if a == axis else s for a in range(3)))
    return out


def tta_transform(size: int, shift: int) -> ROIEval3D:
    """Crop the probe input of :class:`ShiftTTA`: the central ``size + 2 shift`` cube."""
    return ROIEval3D(int(size) + 2 * int(shift))


class ShiftTTA(nn.Module):
    """``(B, C, size + 2s, ...)`` -> mean embedding of the central and six shifted cubes."""

    def __init__(self, encoder: nn.Module, size: int, shift: int = 1) -> None:
        super().__init__()
        if int(shift) < 1:
            raise ValueError("ShiftTTA needs a shift of at least one voxel")
        self.encoder, self.size, self.shift = encoder, int(size), int(shift)
        self.corners = offsets(self.shift)

    @property
    def out_dim(self) -> int:
        return int(self.encoder.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, wide = self.size, self.size + 2 * self.shift
        if x.ndim != 5 or tuple(x.shape[2:]) != (wide, wide, wide):
            raise ValueError(f"expected (B, C, {wide}, {wide}, {wide}), got {tuple(x.shape)}")
        total = None
        for a, b, c in self.corners:
            h = self.encoder(x[:, :, a:a + n, b:b + n, c:c + n].contiguous()).float()
            total = h if total is None else total + h
        return total / len(self.corners)
