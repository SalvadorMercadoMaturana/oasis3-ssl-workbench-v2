"""Whole-brain multiplanar 2D ingestion (t5_ssl.data.brain2d, models.backbones.triplane)."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from t5_ssl.data.brain2d import (
    PLANE_AXES,
    REF_TISSUE_STD01,
    TriPlaneEval,
    TriPlaneEvalViews,
    TriPlaneViews,
    eligible_slices,
    geometry,
    plane,
    preprocess_volume,
)
from t5_ssl.models.backbones.triplane import TriPlaneBackbone
from t5_ssl.models.pretext_module import rotate_planes


def brain(shape=(30, 36, 32), box=((4, 26), (5, 31), (3, 29)), seed=0) -> np.ndarray:
    """Raw int-like volume: positive tissue inside ``box``, zeros and a few negatives outside."""
    rng = np.random.default_rng(seed)
    v = np.zeros(shape, dtype=np.float32)
    (a0, a1), (b0, b1), (c0, c1) = box
    v[a0:a1, b0:b1, c0:c1] = rng.uniform(50, 300, (a1 - a0, b1 - b0, c1 - c0))
    v[0, 0, 0] = -20.0  # interpolation ringing outside the brain
    return v


def canvas(raw: np.ndarray, extra=(3, 2, 4)) -> torch.Tensor:
    pre = preprocess_volume(raw)
    box = pre["box"]
    out = np.full(tuple(s + e for s, e in zip(box.shape, extra, strict=True)),
                  pre["background"], dtype=np.float32)
    out[:box.shape[0], :box.shape[1], :box.shape[2]] = box
    return torch.from_numpy(out.astype(np.float16).astype(np.float32))


def test_preprocess_crops_the_tissue_box_and_zscores_over_tissue() -> None:
    raw = brain()
    pre = preprocess_volume(raw)
    assert pre["box"].shape == (22, 26, 26)
    assert (pre["lo"], pre["hi"]) == ([4, 5, 3], [26, 31, 29])
    tissue = raw[4:26, 5:31, 3:29]
    np.testing.assert_allclose(pre["box"].mean(), 0.0, atol=1e-4)  # the box is all tissue here
    np.testing.assert_allclose(pre["box"].std(), 1.0, atol=1e-4)
    np.testing.assert_allclose(pre["background"], -tissue.mean() / tissue.std(), rtol=1e-5)


def test_background_is_the_darkest_value_and_negative_voxels_join_it() -> None:
    raw = brain()
    raw[10, 10, 10] = 0.0
    raw[11, 10, 10] = -5.0  # negative inside the box: background, not tissue
    pre = preprocess_volume(raw)
    box = pre["box"]
    assert box[6, 5, 7] == pytest.approx(pre["background"])
    assert box[7, 5, 7] == pytest.approx(pre["background"])
    assert box.min() == pytest.approx(pre["background"])


def test_geometry_recovers_box_centre_of_mass_and_tissue_fractions() -> None:
    raw = brain()
    vol = canvas(raw)
    geo = geometry(vol)
    assert geo["shape"] == [22, 26, 26]
    box = np.maximum(raw[4:26, 5:31, 3:29], 0)
    for axis in range(3):
        others = tuple(a for a in range(3) if a != axis)
        profile = box.sum(axis=others)
        expected = (profile * np.arange(len(profile))).sum() / profile.sum()
        assert geo["com"][axis] == pytest.approx(expected, abs=0.02)
        assert len(geo["tissue_fraction"][axis]) == geo["shape"][axis]
        assert float(geo["tissue_fraction"][axis].min()) == pytest.approx(1.0)


def test_eligible_slices_keep_the_window_and_drop_low_tissue_slices() -> None:
    fraction = torch.tensor([0.0, 0.05, 0.1, 0.2, 0.5, 0.9, 0.9, 0.5, 0.14, 0.0])
    assert eligible_slices(fraction, 5, 5, 0.15) == [3, 4, 5, 6, 7]
    assert eligible_slices(fraction, 5, 1, 0.15) == [4, 5, 6]
    assert eligible_slices(fraction, 9, 0, 0.15) == [7]  # nearest eligible slice
    assert eligible_slices(torch.zeros(4), 2, 1, 0.15) == [2]


def test_planes_are_display_oriented() -> None:
    vol = torch.zeros(10, 12, 14)
    vol[2, 11, 13] = 1.0  # left, most anterior, most superior voxel
    shape = [10, 12, 14]
    axial = plane(vol, shape, 2, 13)       # rows anterior -> posterior, cols left -> right
    assert axial.shape == (12, 10) and axial[0, 2] == 1
    coronal = plane(vol, shape, 1, 11)     # rows superior -> inferior, cols left -> right
    assert coronal.shape == (14, 10) and coronal[0, 2] == 1
    sagittal = plane(vol, shape, 0, 2)     # rows superior -> inferior, cols posterior -> anterior
    assert sagittal.shape == (14, 12) and sagittal[0, 11] == 1


def test_eval_is_the_bilinear_resize_of_the_centre_of_mass_planes() -> None:
    vol = canvas(brain())
    out = TriPlaneEval(64)(vol)
    assert out.shape == (3, 64, 64)
    geo = geometry(vol)
    for p, axis in enumerate(PLANE_AXES):
        img = plane(vol, geo["shape"], axis, round(geo["com"][axis]))
        ref = F.interpolate(img[None, None], size=(64, 64), mode="bilinear",
                            align_corners=False)[0, 0]
        # interior: at the outer half voxel the resize replicates the box edge, the planes
        # blend it with the background beyond the box
        torch.testing.assert_close(out[p][3:-3, 3:-3], ref[3:-3, 3:-3], atol=1e-4, rtol=1e-4)


def test_unaugmented_views_are_the_eval_planes_and_offsets_stay_in_the_window() -> None:
    vol = canvas(brain())
    plain = TriPlaneViews(32, offset=0, lr_swap_p=0.0)
    for view in plain(vol):
        torch.testing.assert_close(view, TriPlaneEval(32)(vol), atol=1e-5, rtol=1e-5)
    # slices encode their own index: the view's value says which slice it came from
    idx = torch.zeros(20, 24, 22)
    for k in range(22):
        idx[:, :, k] = 1.0 + k
    geo = geometry(idx)
    centre = round(geo["com"][2])
    seen = {round(float(TriPlaneViews(8, offset=5, lr_swap_p=0.0)(idx)[0][0].mean())) - 1
            for _ in range(200)}
    assert seen <= set(range(centre - 5, centre + 6)) and len(seen) > 5


def test_left_right_flip_mirrors_axial_and_coronal_only() -> None:
    vol = canvas(brain())
    clean = TriPlaneEval(32)(vol)
    view = TriPlaneViews(32, offset=0, lr_swap_p=1.0)(vol)[0]
    torch.testing.assert_close(view[:2], clean[:2].flip(-1), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(view[2], clean[2], atol=1e-5, rtol=1e-5)
    first, second = TriPlaneEvalViews(32)(vol)
    torch.testing.assert_close(second[:2], first[:2].flip(-1))


def test_geometric_and_elastic_views_keep_shape_and_change_pixels() -> None:
    vol = canvas(brain())
    clean = TriPlaneEval(48)(vol)
    views = TriPlaneViews(48, offset=0, lr_swap_p=0.0, side_range=(1 / 1.2, 0.92),
                          centered_crop=False, degrees=5.0,
                          elastic={"control_points": 7, "max_displacement": 2.0,
                                   "locked_borders": 2})(vol)
    assert len(views) == 2 and all(v.shape == (3, 48, 48) for v in views)
    assert not torch.allclose(views[0], clean)
    assert torch.isfinite(views[0]).all()


def test_build_transforms_from_the_recipes() -> None:
    from hydra import compose, initialize_config_dir

    from t5_ssl import recipes
    from t5_ssl.data.brain2d import build_brain2d_transforms
    from t5_ssl.data.paths import PROJECT_ROOT
    for recipe in ("geo", "noise", "elastic_tio_gpu", "flip"):
        overrides = recipes.overrides("vicreg", "campaign_augval", roi="brain2d",
                                      backbone="tri2d_cnx", aug_recipe=recipe)
        with initialize_config_dir(config_dir=str(PROJECT_ROOT / "configs"), version_base="1.3"):
            cfg = compose(config_name="train", overrides=overrides)
        train, val = build_brain2d_transforms(cfg)
        assert cfg.data.source == "brain2d" and train.size == 224 and train.offset == 5
        assert train.min_tissue == pytest.approx(0.15)
        if recipe == "geo":
            assert not train.centered_crop and train.degrees == 5.0
        if recipe == "noise":
            assert train.noise_range[1] == pytest.approx(0.2153708703252311 / REF_TISSUE_STD01)
        if recipe == "elastic_tio_gpu":
            assert train.elastic["control_points"] == 7
        assert isinstance(val, TriPlaneEvalViews)


def test_recipes_pair_brain2d_with_triplane_backbones() -> None:
    from t5_ssl import recipes
    with pytest.raises(ValueError, match="tri2d"):
        recipes.overrides("vicreg", roi="brain2d", backbone="convnext_tiny")
    with pytest.raises(ValueError, match="tri2d"):
        recipes.overrides("vicreg", roi="32", backbone="tri2d_cnx")
    out = recipes.overrides("vicreg", roi="brain2d", backbone="tri2d_cnx_mean")
    assert "data=oasis3_brain2d" in out
    assert "data.cache_path=data/brain2d/brain_mni2.npy" in out


class Flat(nn.Module):
    """Embeds an RGB image as its three channel means (checks the plane replication)."""

    out_dim = 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=(2, 3))


def test_triplane_backbone_shares_the_encoder_over_planes() -> None:
    x = torch.rand(2, 3, 16, 16)
    concat = TriPlaneBackbone(Flat(), 3, "concat")
    out = concat(x)
    assert concat.out_dim == 9 and out.shape == (2, 9)
    torch.testing.assert_close(out[:, 3:6], x[:, 1].mean(dim=(1, 2))[:, None].expand(-1, 3))
    mean = TriPlaneBackbone(Flat(), 3, "mean")
    assert mean.out_dim == 3
    torch.testing.assert_close(mean(x)[:, 0], x.mean(dim=(1, 2, 3)))
    with torch.no_grad():
        torch.testing.assert_close(concat(x), out)
    with pytest.raises(ValueError):
        concat(torch.rand(2, 2, 16, 16))


def test_rotate_planes_turns_every_plane_of_a_sample_alike() -> None:
    x = torch.rand(4, 3, 8, 8)
    labels = torch.tensor([0, 1, 2, 3])
    out = rotate_planes(x, labels)
    for i in range(4):
        torch.testing.assert_close(out[i], torch.rot90(x[i], i, (1, 2)))


def test_cached_geometry_is_the_geometry_and_is_reused() -> None:
    from t5_ssl.data.brain2d import geometry_cached
    vol = canvas(brain(seed=3))
    first = geometry_cached(vol)
    assert geometry_cached(vol.clone()) is first
    ref = geometry(vol)
    assert first["shape"] == ref["shape"] and first["com"] == ref["com"]
    other = geometry_cached(canvas(brain(seed=4)))
    assert other is not first


def test_multislice_probe_input_and_backbone_average() -> None:
    vol = canvas(brain(seed=5))
    multi = TriPlaneEval(32, offsets=(-2, 0, 2))(vol)
    assert multi.shape == (9, 32, 32)
    torch.testing.assert_close(multi[3:6], TriPlaneEval(32)(vol))
    backbone = TriPlaneBackbone(Flat(), 3, "concat")
    x = multi[None]
    expected = torch.stack([backbone(x[:, 3 * k:3 * k + 3]) for k in range(3)]).mean(dim=0)
    torch.testing.assert_close(backbone(x), expected)
