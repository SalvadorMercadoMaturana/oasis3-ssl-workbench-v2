"""Batched augmentations on the GPU (after the host-to-device transfer).

``elastic_deform`` is a GPU counterpart of TorchIO's ``RandomElasticDeformation``
for the ROI views: an independent random displacement per sample on a coarse
grid of ``control_points``³ (uniform in ``[-max_displacement, max_displacement]``
voxels per axis, zero on the outer layer like ``locked_borders=2``), upsampled
smoothly to the view and applied with trilinear resampling and border padding.
TorchIO's B-spline version costs ~0.75 s per 64³ sample on one CPU core (the
whole data pipeline of a DINOv2 run); this one is a few milliseconds per batch.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def elastic_deform(x: torch.Tensor, max_displacement: float = 2.0, control_points: int = 7,
                   generator: torch.Generator | None = None) -> torch.Tensor:
    """Deform every volume of ``x`` ``(B, C, D, H, W)`` with its own smooth random field."""
    if x.ndim != 5:
        raise ValueError("elastic_deform expects (B, C, D, H, W) volumes")
    batch, _, depth, height, width = x.shape
    coarse = (torch.rand((batch, 3, control_points, control_points, control_points),
                         device=x.device, dtype=torch.float32, generator=generator) * 2 - 1)
    coarse = coarse * max_displacement
    coarse[:, :, [0, -1], :, :] = 0
    coarse[:, :, :, [0, -1], :] = 0
    coarse[:, :, :, :, [0, -1]] = 0
    field = F.interpolate(coarse, size=(depth, height, width), mode="trilinear",
                          align_corners=True)  # voxels, channels = (depth, height, width)
    theta = torch.eye(3, 4, device=x.device).expand(batch, 3, 4)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)  # (..., 3) = (x=W, y=H, z=D)
    scale = torch.tensor([2.0 / width, 2.0 / height, 2.0 / depth], device=x.device)
    grid = grid + field.permute(0, 2, 3, 4, 1).flip(-1) * scale
    out = F.grid_sample(x.float(), grid, mode="bilinear", padding_mode="border",
                        align_corners=False)
    return out.to(x.dtype)
