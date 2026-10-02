"""BYOL-fix experiments and the optional GPU runtime, on CPU with tiny tensors."""

from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace

import hydra
import numpy as np
import pytest
import torch
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from torch import nn

from t5_ssl import recipes
from t5_ssl.data.roi3d import ROIView3DTransform, build_roi3d_transforms
from t5_ssl.losses.byol import BYOLLoss
from t5_ssl.losses.vicreg import VICRegLoss
from t5_ssl.models.byol_module import BYOLLightningModule
from t5_ssl.models.heads.dino_head import DINOHead
from t5_ssl.models.heads.projector import MLPProjector
from t5_ssl.utils import runtime

_CONFIG_PATH = str(Path(__file__).resolve().parent.parent / "configs")


class TinyEncoder(nn.Module):
    out_dim = 8

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Conv3d(2, 8, 3, padding=1), nn.BatchNorm3d(8), nn.ReLU(),
                                 nn.AdaptiveAvgPool3d(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten(1)


def byol(**kwargs) -> BYOLLightningModule:
    return BYOLLightningModule(
        backbone=TinyEncoder(), projector=MLPProjector(8, 16, 8, 2, use_bn=False),
        predictor=MLPProjector(8, 16, 8, 2, use_bn=False), loss_fn=BYOLLoss(), max_epochs=4,
        warmup_epochs=1, lr=1e-3, **kwargs)


# ---------------------------------------------------------------- views
def test_border_rotation_never_brings_in_black_corners():
    cube = torch.ones(2, 38, 38, 38)
    random.seed(0)
    zeros = ROIView3DTransform(size=32, side_range=(1.0, 1.0), degrees=10.0, lr_swap_p=0.0)
    random.seed(0)
    border = ROIView3DTransform(size=32, side_range=(1.0, 1.0), degrees=10.0, lr_swap_p=0.0,
                                rotation_padding="border")
    random.seed(1)
    zero_view = zeros(cube)[0]
    random.seed(1)
    border_view = border(cube)[0]
    assert zero_view.min() < 0.5  # the black wedges of the campaign views
    assert torch.allclose(border_view, torch.ones_like(border_view), atol=1e-5)
    with pytest.raises(ValueError):
        ROIView3DTransform(rotation_padding="wrap")


def test_rotation_padding_is_read_from_the_config():
    cfg = OmegaConf.create({"data": {"roi_size": 32}, "augmentation": {
        "affine_degrees": 10.0, "rotation_padding": "border"}})
    assert build_roi3d_transforms(cfg)[0].rotation_padding == "border"
    cfg.augmentation.pop("rotation_padding")
    assert build_roi3d_transforms(cfg)[0].rotation_padding == "zeros"  # unchanged default


# ---------------------------------------------------------------- BYOL
def test_predictor_lr_multiplier_gets_its_own_group():
    single = byol().configure_optimizers()["optimizer"]
    assert len(single.param_groups) == 1
    sched = byol(predictor_lr_mult=10.0).configure_optimizers()
    groups = sched["optimizer"].param_groups
    assert len(groups) == 2
    assert groups[1]["lr"] / groups[0]["lr"] == pytest.approx(10.0)
    sched["lr_scheduler"]["scheduler"].step()  # the schedule keeps the ratio
    assert groups[1]["lr"] / groups[0]["lr"] == pytest.approx(10.0)
    with pytest.raises(ValueError):
        byol(predictor_lr_mult=0.0)


def test_ema_only_moves_after_an_optimizer_step():
    model = byol(ema_base=0.5)
    trainer = SimpleNamespace(global_step=0, estimated_stepping_batches=10)
    object.__setattr__(model, "_trainer", trainer)
    with torch.no_grad():
        for p in model.backbone.parameters():
            p.add_(1.0)
    before = [t.clone() for t in model.target_backbone.parameters()]
    model.on_train_batch_end(None, None, 0)  # accumulation batch: global_step unchanged
    assert all(torch.equal(a, b) for a, b in zip(before, model.target_backbone.parameters(),
                                                 strict=True))
    trainer.global_step = 1
    model.on_train_batch_end(None, None, 1)
    moved = list(model.target_backbone.parameters())
    assert not all(torch.equal(a, b) for a, b in zip(before, moved, strict=True))
    again = [t.clone() for t in moved]
    model.on_train_batch_end(None, None, 2)  # same step again: no second update
    assert all(torch.equal(a, b) for a, b in zip(again, model.target_backbone.parameters(),
                                                 strict=True))


def test_fused_ema_matches_per_parameter_lerp():
    online, target = TinyEncoder(), TinyEncoder()
    reference = [t.detach().clone().lerp_(s.detach(), 0.3) for s, t in
                 zip(online.parameters(), target.parameters(), strict=True)]
    BYOLLightningModule._ema_update([(online, target)], momentum=0.7)
    for ref, value in zip(reference, target.parameters(), strict=True):
        assert torch.allclose(ref, value)
    for source, destination in zip(online.buffers(), target.buffers(), strict=True):
        assert torch.equal(source, destination)


def test_channels_last_inputs_after_transfer():
    model = byol()
    batch = {"image": (torch.rand(2, 2, 8, 8, 8), torch.rand(2, 2, 8, 8, 8)), "subject": ["a"]}
    assert model.on_after_batch_transfer(batch, 0) is batch  # default: untouched
    model.input_memory_format = torch.channels_last_3d
    out = model.on_after_batch_transfer(batch, 0)
    assert all(v.is_contiguous(memory_format=torch.channels_last_3d) for v in out["image"])


# ---------------------------------------------------------------- precision
def test_losses_and_dino_scores_stay_float32_under_bf16_autocast():
    z1, z2 = torch.randn(8, 16).bfloat16(), torch.randn(8, 16).bfloat16()
    head = DINOHead(16, out_dim=32, hidden_dim=16, bottleneck_dim=8)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        vicreg, _ = VICRegLoss()(z1, z2)
        byol_loss, _ = BYOLLoss()(z1, z2, z2, z1)
        scores = head(torch.randn(4, 16))
    assert vicreg.dtype == byol_loss.dtype == scores.dtype == torch.float32


def test_resolve_precision_degrades_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert runtime.resolve_precision("bf16-mixed") == "32-true"
    assert runtime.resolve_precision("16-mixed") == "32-true"
    assert runtime.resolve_precision("32-true") == "32-true"


# ---------------------------------------------------------------- compile / monitor
def test_compile_falls_back_to_eager_and_uncompiles():
    model = byol()
    assert runtime.compile_encoders(model, ["no-such-mode"], example=None) is None
    assert "forward" not in model.backbone.__dict__
    model.backbone.forward = lambda x: x  # what a compiled entry point looks like
    runtime.uncompile(model)
    assert "forward" not in model.backbone.__dict__


def test_apply_runtime_is_a_noop_without_perf_node():
    model = byol()
    cfg = OmegaConf.create({"trainer": {"accelerator": "auto"}})
    report = runtime.apply_runtime(cfg, model, "32-true")
    assert report["precision"] == "32-true"
    assert model.fused_optimizer is False and model.input_memory_format is None


def test_perf_monitor_summary_skips_the_first_epoch(tmp_path):
    monitor = runtime.PerfMonitor(tmp_path / "perf.json", extra={"compile_mode": None})
    monitor.epochs = [
        {"epoch": 0, "seconds": 30.0, "train_seconds": 28.0, "samples": 16, "images": 32,
         "images_per_second": 32 / 28},
        {"epoch": 1, "seconds": 2.0, "train_seconds": 1.0, "samples": 16, "images": 32,
         "images_per_second": 32.0},
        {"epoch": 2, "seconds": 2.0, "train_seconds": 1.0, "samples": 16, "images": 32,
         "images_per_second": 32.0}]
    summary = monitor.summary()
    assert summary["images_per_second"] == pytest.approx(32.0)
    assert summary["first_epoch_seconds"] == 30.0
    assert summary["seconds_per_epoch"] == 2.0


# ---------------------------------------------------------------- recipes
def compose(overrides: list[str]):
    GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(config_dir=_CONFIG_PATH, version_base="1.3"):
        return hydra.compose(config_name="train", overrides=overrides)


@pytest.mark.parametrize(("model", "variant"),
                         [(m, v) for m, variants in recipes.VARIANTS.items() for v in variants])
def test_every_recipe_composes(model, variant):
    cfg = compose(recipes.overrides(model, variant, epochs=50, seed=2))
    assert cfg.trainer.max_epochs == 50 and cfg.optimizer.warmup_epochs == 10
    assert cfg.perf.enabled and cfg.trainer.precision == "bf16-mixed"
    bias = cfg.augmentation.torchio[0]
    assert bias.name == "RandomBiasField" and bias.coefficients == pytest.approx(0.1839173196)
    if model == "dinov2":
        assert cfg.augmentation.affine_degrees == 0 and cfg.loss.teacher_temp_warmup_epochs == 15
    else:
        assert cfg.augmentation.affine_degrees == 10
        expected = "zeros" if variant == "campaign" else "border"
        assert cfg.augmentation.rotation_padding == expected


def test_byol_h3_recipe_and_campaign_runtime():
    cfg = compose(recipes.overrides("byol", "border_h3", fast=False))
    assert cfg.model.projector.hidden_dim == cfg.model.predictor.hidden_dim == 4096
    assert cfg.model.projector.norm == cfg.model.predictor.norm == "group"
    assert cfg.optimizer.predictor_lr_mult == 10.0
    assert "perf" not in cfg and cfg.trainer.precision == "32-true"
    assert cfg.loss.ema_base == 0.996


def test_dedupe_keeps_the_last_value():
    assert recipes.dedupe(["a=1", "++b=2", "a=3", "b=4"]) == ["a=3", "b=4"]


# ---------------------------------------------------------------- round 2: dropout, recipes
@pytest.mark.parametrize("recipe", sorted(recipes.AUG_RECIPES))
def test_every_augmentation_recipe_builds_two_views(recipe):
    cfg = OmegaConf.create({"data": {"roi_size": 32}, "augmentation": recipes.aug_spec(recipe)})
    transform = build_roi3d_transforms(cfg)[0]
    random.seed(0), torch.manual_seed(0)
    views = transform(torch.rand(2, 38, 38, 38))
    assert len(views) == 2 and all(v.shape == (2, 32, 32, 32) for v in views)
    assert all(torch.isfinite(v).all() for v in views)


def test_centered_crop_of_eval_side_is_the_probe_cube():
    from t5_ssl.data.roi3d import ROIEval3D
    cube = torch.rand(2, 38, 38, 38)
    cfg = OmegaConf.create({"data": {"roi_size": 32}, "augmentation": recipes.aug_spec("flip")})
    view = build_roi3d_transforms(cfg)[0]
    view.lr_swap_p = 0.0
    assert torch.allclose(view(cube)[0], ROIEval3D(32)(cube), atol=1e-6)


def test_byol_target_projector_never_drops_units():
    model = BYOLLightningModule(
        backbone=TinyEncoder(), projector=MLPProjector(8, 16, 8, 2, use_bn=False, dropout=0.3),
        predictor=MLPProjector(8, 16, 8, 2, use_bn=False, dropout=0.3), loss_fn=BYOLLoss(),
        max_epochs=2)
    online = [m.p for m in model.projector.modules() if isinstance(m, nn.Dropout)]
    target = [m.p for m in model.target_projector.modules() if isinstance(m, nn.Dropout)]
    assert online == [0.3] and target == [0.0]


def test_dropout_variants_compose():
    cfg = compose(recipes.overrides("byol", "wide_dboth0.2", aug_recipe="geo_artifacts"))
    assert cfg.model.projector.dropout == cfg.model.predictor.dropout == 0.2
    assert cfg.model.projector.hidden_dim == 4096
    assert cfg.augmentation.torchio[0].name == "OneOf"
    assert recipes.run_name("byol", "wide_dboth0.2", 3, "geo") == "byol_wide_dboth0.2__geo_s3"


# ---------------------------------------------------------------- round 3: regions of interest
def test_swap_hemispheres_swaps_each_pair():
    from t5_ssl.data.roi3d import swap_hemispheres
    two = torch.rand(2, 4, 4, 4)
    assert torch.equal(swap_hemispheres(two), two.flip(0))
    six = torch.arange(6.0).view(6, 1, 1, 1).expand(6, 2, 2, 2)
    assert swap_hemispheres(six)[:, 0, 0, 0].tolist() == [1, 0, 3, 2, 5, 4]


@pytest.mark.parametrize("roi", sorted(recipes.ROIS))
@pytest.mark.parametrize("model", recipes.MODELS)
def test_every_roi_composes(model, roi):
    cfg = compose(recipes.overrides(model, aug_recipe="noise", roi=roi, epochs=50))
    spec = recipes.ROIS[roi]
    assert cfg.data.roi_size == spec["size"] and cfg.data.channels == spec["channels"]
    assert cfg.model.backbone.in_channels == spec["channels"]
    assert cfg.data.cache_path.endswith(f"{spec['cache']}.npy")
    side = cfg.augmentation.side_range[0] * round(1.2 * spec["size"])
    assert round(side) == spec["size"]  # photometric recipes see exactly the probe cube


def test_vit3d_takes_larger_and_multichannel_rois():
    from t5_ssl.models.backbones.vit3d import ViT3DBackbone
    vit = ViT3DBackbone(in_channels=6, image_size=48, patch_size=8, embed_dim=48, depth=1,
                        num_heads=3)
    cls, patches = vit.forward_tokens(torch.rand(2, 6, 48, 48, 48))
    assert cls.shape == (2, 48) and patches.shape == (2, 216, 48)


def test_gpu_elastic_deforms_only_training_batches():
    from t5_ssl.data.gpu_augment import elastic_deform
    x = torch.rand(2, 2, 16, 16, 16)
    assert torch.allclose(elastic_deform(x, max_displacement=0.0), x, atol=1e-5)
    moved = elastic_deform(x, max_displacement=1.5, generator=torch.Generator().manual_seed(0))
    assert moved.shape == x.shape and not torch.allclose(moved, x)
    model = byol()
    model.gpu_elastic = {"max_displacement": 1.5, "control_points": 7}
    batch = {"image": (x.clone(), x.clone())}
    model.eval()
    assert all(torch.equal(v, x) for v in model.on_after_batch_transfer(batch, 0)["image"])
    model.train()
    out = model.on_after_batch_transfer({"image": (x.clone(), x.clone())}, 0)["image"]
    assert not torch.allclose(out[0], out[1])  # independent fields per view


def test_elastic_gpu_recipe_composes():
    cfg = compose(recipes.overrides("dinov2", aug_recipe="elastic_gpu", roi="48"))
    assert cfg.augmentation.gpu_elastic.max_displacement == pytest.approx(1.5042)
    assert list(cfg.augmentation.torchio) == []
    assert compose(recipes.overrides("vicreg", aug_recipe="geo")).augmentation.gpu_elastic is None


def test_collapse_scores_detect_collapse():
    from t5_ssl.callbacks.collapse_monitor import collapse_scores
    rng = np.random.default_rng(0)
    subjects = rng.normal(size=(40, 16))
    healthy = collapse_scores(subjects, subjects + 0.05 * rng.normal(size=(40, 16)),
                              subjects + 0.05 * rng.normal(size=(40, 16)))
    assert healthy["between_over_within"] > 10 and healthy["view_matching_top1"] > 0.9
    line = np.outer(rng.normal(size=40), np.ones(16))
    collapsed = collapse_scores(line, line + 2 * rng.normal(size=(40, 16)),
                                line + 2 * rng.normal(size=(40, 16)))
    assert collapsed["effective_rank"] < 1.5 and collapsed["between_over_within"] < 1


def test_frozen_backbone_bn_stays_in_eval_mode():
    model = byol()
    model.train()
    assert all(m.training for m in model._backbone_norms())
    model.freeze_backbone_bn = True
    model.train()
    norms = model._backbone_norms()
    assert norms and not any(m.training for m in norms)  # online and target backbones
    assert model.projector.training and model.predictor.training
    cfg = compose(recipes.overrides("byol", "gn_dp0.3_dq0_wd0.01_fbn", aug_recipe="noise"))
    assert cfg.model.freeze_backbone_bn is True and cfg.optimizer.weight_decay == 0.01
