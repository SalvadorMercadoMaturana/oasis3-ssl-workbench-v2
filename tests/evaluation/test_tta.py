"""Test-time shifts of the probe input (t5_ssl.evaluation.tta)."""

import pytest
import torch
from torch import nn

from t5_ssl.data.roi3d import ROIEval3D
from t5_ssl.evaluation.tta import ShiftTTA, offsets, tta_transform


class Corner(nn.Module):
    """Embeds a cube as its first voxel (which crop it got) and its mean."""

    out_dim = 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([x[:, 0, 0, 0, 0], x.mean(dim=(1, 2, 3, 4))], dim=1)


def test_offsets_centre_first_then_one_axis_shifts() -> None:
    corners = offsets(1)
    assert corners[0] == (1, 1, 1)
    assert len(set(corners)) == 7
    assert all(sum(abs(c - 1) for c in corner) == 1 for corner in corners[1:])


def test_centre_crop_of_the_wide_cube_is_the_usual_probe_input() -> None:
    cube = torch.rand(2, 38, 38, 38)
    wide = tta_transform(32, 1)(cube)
    assert wide.shape == (2, 34, 34, 34)
    torch.testing.assert_close(wide[:, 1:33, 1:33, 1:33], ROIEval3D(32)(cube))


def test_shift_tta_averages_the_seven_crops() -> None:
    cube = torch.arange(2 * 38 ** 3, dtype=torch.float32).reshape(1, 2, 38, 38, 38)
    wide = tta_transform(32, 2)(cube[0]).unsqueeze(0)
    out = ShiftTTA(Corner(), 32, 2)(wide)
    expected = torch.stack([Corner()(wide[:, :, a:a + 32, b:b + 32, c:c + 32])
                            for a, b, c in offsets(2)]).mean(dim=0)
    torch.testing.assert_close(out, expected)
    assert ShiftTTA(Corner(), 32, 2).out_dim == 2


def test_shift_tta_rejects_the_unshifted_cube() -> None:
    with pytest.raises(ValueError, match="34"):
        ShiftTTA(Corner(), 32, 1)(torch.rand(1, 2, 32, 32, 32))
    with pytest.raises(ValueError):
        ShiftTTA(Corner(), 32, 0)
