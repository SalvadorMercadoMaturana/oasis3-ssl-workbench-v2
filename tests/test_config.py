"""Unit tests for Hydra configuration composition."""

from __future__ import annotations

from pathlib import Path

import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

_CONFIG_PATH = str(Path(__file__).resolve().parent.parent / "configs")


def test_hydra_default_config_composition():
    """Hydra should compose all default configuration groups without error."""
    GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(config_dir=_CONFIG_PATH, version_base="1.3"):
        cfg = hydra.compose(config_name="train")

        assert isinstance(cfg, DictConfig)
        assert cfg.data.image_root == "data/dementia/oasis3"
        assert cfg.data.channels == 1
        assert cfg.model.backbone.variant == "convnext_tiny"
        assert cfg.loss.name == "vicreg"
        assert cfg.loss.sim_coeff == 25.0
        assert cfg.trainer.max_epochs == 50


def test_hydra_cli_overrides():
    """Hydra should accept CLI overrides seamlessly."""
    GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(config_dir=_CONFIG_PATH, version_base="1.3"):
        cfg = hydra.compose(
            config_name="train",
            overrides=[
                "model.backbone.variant=convnext_small",
                "loss.sim_coeff=10.0",
                "data.channels=3",
                "model.backbone.in_channels=3",
                "trainer.max_epochs=5",
            ],
        )

        assert cfg.model.backbone.variant == "convnext_small"
        assert cfg.loss.sim_coeff == 10.0
        assert cfg.data.channels == cfg.model.backbone.in_channels == 3
        assert cfg.trainer.max_epochs == 5


def test_teacher_method_groups_compose():
    """Method switches must select matching loss, backbone, and view configuration."""
    GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(config_dir=_CONFIG_PATH, version_base="1.3"):
        byol = hydra.compose(
            config_name="train", overrides=["loss=byol", "model=byol_convnext"]
        )
        dino = hydra.compose(
            config_name="train",
            overrides=["loss=dinov2", "model=dinov2_vit", "augmentation=dinov2_multicrop"],
        )
    assert byol.loss.name == "byol" and byol.model.predictor.hidden_dim == 256
    assert byol.model.backbone.variant == "convnext_tiny"
    assert dino.loss.name == "dinov2" and dino.model.backbone.variant == "vit_small"
    assert dino.augmentation.n_local_crops == 2


def test_anatomy_preserving_20_epoch_configs_compose():
    """The new campaign should keep full-image views for all three models."""
    GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(config_dir=_CONFIG_PATH, version_base="1.3"):
        common = hydra.compose(config_name="train", overrides=[
            "data=oasis3_full50", "trainer=full20", "augmentation=mri_fullview20",
        ])
        dino = hydra.compose(config_name="train", overrides=[
            "data=oasis3_full50", "trainer=full20", "model=dinov2_vit",
            "loss=dinov2", "augmentation=mri_fullview20_dino",
        ])
    assert common.trainer.max_epochs == dino.trainer.max_epochs == 20
    assert list(common.augmentation.crop_scale) == [1.0, 1.0]
    assert dino.augmentation.n_local_crops == 0


def test_dinov2_3d_configs_compose():
    """The 3D DINOv2 recipe composes with the ROI data and its own optimizer."""
    GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(config_dir=_CONFIG_PATH, version_base="1.3"):
        cfg = hydra.compose(config_name="train", overrides=[
            "data=oasis3_roi3d", "model=dinov2_vit3d", "loss=dinov2",
            "augmentation=roi3d_dinov2", "optimizer=adamw_dinov2"])
        multicrop = hydra.compose(config_name="train", overrides=["augmentation=roi3d_multicrop"])
    assert cfg.model.backbone.patch_size == 8 and cfg.model.projector.bottleneck_dim == 256
    assert cfg.model.projector.use_bn
    assert cfg.loss.centering == "sinkhorn_knopp"
    assert cfg.loss.teacher_temp == cfg.loss.teacher_temp_final == 0.01
    assert cfg.optimizer.weight_decay_end == 0.2
    # DINOv2's own views: no geometry, no local crops (see roi3d_dinov2.yaml).
    assert list(cfg.augmentation.side_range) == [1.0, 1.0] and cfg.augmentation.affine_degrees == 0
    assert cfg.augmentation.n_local_crops == 0 and multicrop.augmentation.n_local_crops == 4
    assert cfg.model.backbone.in_channels == cfg.data.channels == 2
