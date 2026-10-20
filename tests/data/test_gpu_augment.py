"""The GPU counterpart of TorchIO's elastic deformation reproduces TorchIO."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from t5_ssl import recipes
from t5_ssl.data.gpu_augment import tio_control_points, tio_elastic_deform
from t5_ssl.data.roi3d import analysis_views, build_roi3d_transforms

tio = pytest.importorskip("torchio")
from torchio.transforms.augmentation.spatial.random_elastic_deformation import (  # noqa: E402
    ElasticDeformation,
    RandomElasticDeformation,
)


@pytest.mark.parametrize("size", [20, 38])
def test_same_control_points_give_torchio_output(size):
    torch.manual_seed(0)
    x = torch.rand(2, size, size, size)
    displacement = 1.95
    points = RandomElasticDeformation.get_params((7, 7, 7), (displacement,) * 3, 2)
    reference = ElasticDeformation(control_points=points, max_displacement=(displacement,) * 3)(
        tio.ScalarImage(tensor=x)).data
    ours = tio_elastic_deform(x[None], control_points=torch.from_numpy(points)[None])[0]
    assert (reference - x).abs().mean() > 1e-2  # a real deformation
    assert torch.allclose(ours, reference, atol=1e-4)


def test_control_point_draw_matches_torchio_locking():
    ours = tio_control_points(64, 7, 2.0, 2)
    theirs = np.stack([RandomElasticDeformation.get_params((7, 7, 7), (2.0,) * 3, 2)
                       for _ in range(64)])
    assert torch.equal(ours != 0, torch.from_numpy(theirs != 0))
    assert ours.abs().max() <= 2.0


def test_deferred_views_are_finished_to_the_roi_size():
    cfg = OmegaConf.create({"data": {"roi_size": 32},
                            "augmentation": recipes.aug_spec("elastic_tio_gpu", 32)})
    cube = torch.rand(6, 38, 38, 38)
    raw = build_roi3d_transforms(cfg)[0](cube)
    assert [v.shape for v in raw] == [(6, 38, 38, 38)] * 2
    finished = analysis_views(cfg)(cube)
    assert [v.shape for v in finished] == [(6, 32, 32, 32)] * 2
    assert all(v.min() >= 0 and v.max() <= 1 for v in finished)
