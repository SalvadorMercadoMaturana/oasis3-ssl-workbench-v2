"""t5_ssl.data.roi3d.crop_cubes: cubes cut at a cache's positions, right ones mirrored."""

from __future__ import annotations

import numpy as np
import pytest

from t5_ssl.data.roi3d import LR_AXIS, crop_cubes


def test_crop_cubes_order_and_mirroring() -> None:
    volume = np.arange(10 ** 3, dtype=np.float32).reshape(10, 10, 10)
    starts = {"hippo_left": [0, 1, 2], "hippo_right": [5, 5, 5]}
    cubes = crop_cubes(volume, starts, extract=3)
    assert cubes.shape == (2, 3, 3, 3)
    assert cubes.dtype == np.float16
    np.testing.assert_array_equal(cubes[0], volume[0:3, 1:4, 2:5])
    np.testing.assert_array_equal(cubes[1], np.flip(volume[5:8, 5:8, 5:8], axis=LR_AXIS))


def test_crop_cubes_rejects_cubes_outside_the_volume() -> None:
    with pytest.raises(RuntimeError, match="leaves the volume"):
        crop_cubes(np.zeros((10, 10, 10)), {"hippo_left": [8, 0, 0]}, extract=3)
