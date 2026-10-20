"""Batched augmentations on the GPU (after the host-to-device transfer).

``elastic_deform`` is a GPU counterpart of TorchIO's ``RandomElasticDeformation``
for the ROI views: an independent random displacement per sample on a coarse
grid of ``control_points``³ (uniform in ``[-max_displacement, max_displacement]``
voxels per axis, zero on the outer layer like ``locked_borders=2``), upsampled
smoothly to the view and applied with trilinear resampling and border padding.
TorchIO's B-spline version costs ~0.75 s per 64³ sample on one CPU core (the
whole data pipeline of a DINOv2 run); this one is a few milliseconds per batch.
It is *not* TorchIO's deformation (trilinear instead of cubic B-spline, applied to
the cropped view, borders locked on every axis): the 3D campaigns measured worse
probes with it (32³: CV AUROC 0.806 vs 0.853).

``tio_elastic_deform`` reproduces TorchIO's ``RandomElasticDeformation`` (as the
``elastic`` recipe applies it, on the cached cube before the crop) on the GPU:
the same control-point draw (uniform in ``[-max_displacement, max_displacement]``
mm, ``locked_borders`` zeroing only the first two grid axes, as TorchIO does), the
cubic B-spline of ``SimpleITK.BSplineTransformInitializer`` (its grid geometry is
read from SimpleITK once per cube size), linear interpolation with ITK's edge
handling and the per-channel minimum outside the image. Given the same control
points it matches TorchIO to float32 precision (tests/data/test_gpu_augment.py).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

_TIO_GRIDS: dict[tuple[int, int], tuple[list[np.ndarray], np.ndarray]] = {}


def _bspline3(t: np.ndarray) -> np.ndarray:
    """Uniform cubic B-spline kernel (ITK BSplineKernelFunction<3>)."""
    a = np.abs(t)
    return np.where(a < 1, 2 / 3 - a**2 + a**3 / 2, np.where(a < 2, (2 - a) ** 3 / 6, 0.0))


def tio_bspline_grid(size: int, control_points: int) -> tuple[list[np.ndarray], np.ndarray]:
    """Per-axis B-spline weights ``(size, control_points)`` and voxel-axis signs.

    The geometry is the one TorchIO gets for a (C, size, size, size) tensor with an
    identity affine: SimpleITK image in LPS (directions -1, -1, 1), B-spline grid of
    ``control_points - 3`` mesh cells from ``BSplineTransformInitializer``.
    """
    key = (size, control_points)
    if key not in _TIO_GRIDS:
        import SimpleITK as sitk  # noqa: N813
        from torchio.data.io import nib_to_sitk
        image = nib_to_sitk(np.zeros((1, size, size, size), np.float32), np.eye(4),
                            force_3d=True)
        coefficients = sitk.BSplineTransformInitializer(
            image, [control_points - 3] * 3).GetCoefficientImages()[0]
        image_dir = np.array(image.GetDirection()).reshape(3, 3)
        grid_dir = np.array(coefficients.GetDirection()).reshape(3, 3)
        if not (np.allclose(image_dir, np.diag(image_dir.diagonal()))
                and np.allclose(grid_dir, np.eye(3))):
            raise RuntimeError("unexpected SimpleITK directions for the B-spline grid")
        origin, spacing = np.array(coefficients.GetOrigin()), np.array(coefficients.GetSpacing())
        image_origin = np.array(image.GetOrigin())
        weights = []
        for axis in range(3):
            physical = image_origin[axis] + image_dir[axis, axis] * np.arange(size)
            u = (physical - origin[axis]) / spacing[axis]
            if u.min() < 1 or u.max() >= control_points - 2:
                raise RuntimeError("voxels outside the B-spline valid region")
            weights.append(_bspline3(u[:, None] - np.arange(control_points)[None, :]))
        _TIO_GRIDS[key] = (weights, image_dir.diagonal().copy())
    return _TIO_GRIDS[key]


def tio_control_points(batch: int, control_points: int, max_displacement: float,
                       locked_borders: int = 2, device: torch.device | str = "cpu",
                       generator: torch.Generator | None = None) -> torch.Tensor:
    """``(batch, cp, cp, cp, 3)`` displacements (mm) drawn as TorchIO's ``get_params``."""
    coarse = (torch.rand((batch, control_points, control_points, control_points, 3),
                         device=device, generator=generator) - 0.5) * 2 * max_displacement
    for i in range(locked_borders):  # TorchIO locks the first two grid axes only
        coarse[:, i] = 0
        coarse[:, -1 - i] = 0
        coarse[:, :, i] = 0
        coarse[:, :, -1 - i] = 0
    return coarse


def tio_elastic_deform(x: torch.Tensor, num_control_points: int = 7,
                       max_displacement: float = 7.5, locked_borders: int = 2,
                       control_points: torch.Tensor | None = None,
                       generator: torch.Generator | None = None) -> torch.Tensor:
    """TorchIO's elastic deformation of every cube of ``x`` ``(B, C, n, n, n)``, one field each."""
    if x.ndim != 5 or not x.shape[2] == x.shape[3] == x.shape[4]:
        raise ValueError("tio_elastic_deform expects (B, C, n, n, n) cubes")
    batch, n = x.shape[0], x.shape[2]
    if control_points is None:
        control_points = tio_control_points(batch, num_control_points, max_displacement,
                                            locked_borders, x.device, generator)
    weights, signs = tio_bspline_grid(n, control_points.shape[1])
    w0, w1, w2 = (torch.as_tensor(w, dtype=torch.float32, device=x.device) for w in weights)
    field = torch.einsum("ai,bj,ck,nijkd->ndabc", w0, w1, w2,
                         control_points.to(x.device, torch.float32))  # physical mm (LPS)
    # Output voxel v samples the input at v + direction^-1 * displacement (1 mm voxels).
    index = torch.arange(n, device=x.device, dtype=torch.float32)
    base = torch.stack(torch.meshgrid(index, index, index, indexing="ij"))
    sign = torch.as_tensor(signs, dtype=torch.float32, device=x.device).view(1, 3, 1, 1, 1)
    q = base[None] + field * sign
    inside = ((q >= -0.5) & (q <= n - 0.5)).all(dim=1, keepdim=True)
    grid = torch.stack((q[:, 2], q[:, 1], q[:, 0]), dim=-1) * (2 / (n - 1)) - 1
    # ITK's linear interpolator clamps neighbours inside the buffer (border padding) and
    # returns the default value (TorchIO: the channel minimum) outside [-0.5, n - 0.5].
    out = F.grid_sample(x.float(), grid, mode="bilinear", padding_mode="border",
                        align_corners=True)
    fill = x.float().amin(dim=(2, 3, 4), keepdim=True)
    return torch.where(inside, out, fill).to(x.dtype)


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
