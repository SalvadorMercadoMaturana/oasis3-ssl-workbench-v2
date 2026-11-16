"""Hydra override recipes for the 3D hippocampal-ROI SSL runs (no torch import).

One place that turns ``(model, variant, epochs, seed)`` into the overrides of
``configs/train.yaml``, shared by the root ``train.py``, ``run_all.py`` and
``bench.py``. The ``campaign`` variants reproduce ``scripts/run_ssl3d_v2.py``
(campaign_ssl3d / campaign_ssl3d_v2): balanced split, champion bias-field
views, LR warmup = epochs / 5, DINOv2 teacher-temperature warmup = 0.3 x epochs,
EMA 0.996 for BYOL (the ``loss/byol.yaml`` default).

BYOL variants test the 2026-10-01 diagnosis (docs/informe_ssl3d_v2.pdf, sec. 8):

- ``border``: rotation resampled with border instead of zero padding (H1, the
  artifact the trained encoder collapsed onto);
- ``border_wide``: + projector/predictor hidden width 4096 as in BYOL (H3);
- ``border_gn``: + GroupNorm instead of BatchNorm in projector and predictor (H3);
- ``border_predlr``: + predictor learning rate x10 (H3);
- ``border_h3``: border + all three.

The BYOL + VICReg-terms experiment (H2) is deliberately not here.

``wide_d*`` variants add dropout to ``border_wide`` against the overfitting seen
at 50 epochs (validation loss rising from epoch ~12): projector (``dproj``),
predictor (``dpred``) or both; the EMA target never drops units.

``AUG_RECIPES`` are complete augmentation specs (every field of the ROI views is
set, so nothing but the recipe changes between runs), applied with
``aug_recipe=`` on top of any model/variant. TorchIO artifact strengths are the
3D campaign calibration (RMSE 0.080 against the clean cube:
campaign_roi3d/calibration_s32.json; Spike and Elastic via
scripts/calibrate_spike_elastic.py).
"""

from __future__ import annotations

import itertools

MANIFEST = "campaign_full50_pretrained_balanced/train_sessions_eval_primary_balanced.csv"
MODELS = ("byol", "vicreg", "dinov2", "pretext")

BASE = {
    "vicreg": ["model=vicreg_convnext_3d", "loss=vicreg", "augmentation=roi3d_bias"],
    "byol": ["model=byol_resnet18_3d", "loss=byol", "augmentation=roi3d_bias"],
    "dinov2": ["model=dinov2_vit3d", "loss=dinov2", "augmentation=roi3d_dinov2_bias",
               "optimizer=adamw_dinov2"],
    # pseudo-label pretext tasks (rotation / jigsaw order), t5_ssl.models.pretext_module
    "pretext": ["model=pretext_convnext_3d", "loss=pretext", "augmentation=roi3d_bias"],
}

BORDER = ["augmentation.rotation_padding=border"]
WIDE = ["model.projector.hidden_dim=4096", "model.predictor.hidden_dim=4096"]
GROUP_NORM = ["++model.projector.norm=group", "++model.predictor.norm=group"]
PREDICTOR_LR = ["++optimizer.predictor_lr_mult=10.0"]


def _dropout(projector: float = 0.0, predictor: float = 0.0) -> list[str]:
    return [f"++model.projector.dropout={projector}", f"++model.predictor.dropout={predictor}"]

VARIANTS: dict[str, dict[str, list[str]]] = {
    "byol": {
        "campaign": [],
        "border": BORDER,
        "border_wide": BORDER + WIDE,
        "border_gn": BORDER + GROUP_NORM,
        "border_predlr": BORDER + PREDICTOR_LR,
        "border_h3": BORDER + WIDE + GROUP_NORM + PREDICTOR_LR,
        "wide_dproj0.1": BORDER + WIDE + _dropout(projector=0.1),
        "wide_dproj0.3": BORDER + WIDE + _dropout(projector=0.3),
        "wide_dpred0.1": BORDER + WIDE + _dropout(predictor=0.1),
        "wide_dpred0.3": BORDER + WIDE + _dropout(predictor=0.3),
        "wide_dboth0.2": BORDER + WIDE + _dropout(projector=0.2, predictor=0.2),
        # the current BYOL (wide_dproj0.1) with GroupNorm instead of BatchNorm in the heads
        "wide_dproj0.1_gn": BORDER + WIDE + _dropout(projector=0.1) + GROUP_NORM,
    },
    "vicreg": {"campaign": [], "border": BORDER,
               # SSL validation on augmented views (fixed per session) instead of cube vs mirror.
               "campaign_augval": ["++augmentation.val_views=train"]},
    "dinov2": {"campaign": []},
    # Pseudo-label pretext tasks: rotation classes (4 axial, 10 Taleb et al. 2020, 24 cube
    # group) and/or the order of the 2x2x2 octants (jigsaw, P max-Hamming permutations).
    "pretext": {},
}
_ROT = {"rot4": 4, "rot10": 10, "rot24": 24}
_JIG = {"jig24": 24, "jig100": 100}
for _r, _k in _ROT.items():
    VARIANTS["pretext"][_r] = [f"loss.rotation_classes={_k}", "loss.jigsaw_permutations=0"]
    for _j, _p in _JIG.items():
        VARIANTS["pretext"][f"{_r}_{_j}"] = [f"loss.rotation_classes={_k}",
                                             f"loss.jigsaw_permutations={_p}"]
for _j, _p in _JIG.items():
    VARIANTS["pretext"][_j] = ["loss.rotation_classes=0", f"loss.jigsaw_permutations={_p}"]
VARIANTS["pretext"]["campaign"] = VARIANTS["pretext"]["rot10"]
# Tuning grid on top of any pretext task: _lr{learning rate}, _wd{weight decay},
# _ls{label smoothing}, _gap{jigsaw gap}, _h{head width}.
_PRETEXT_TUNING = {
    **{f"lr{v:g}": [f"optimizer.lr={v}"] for v in (3e-5, 3e-4, 1e-3)},
    **{f"wd{v:g}": [f"optimizer.weight_decay={v}"] for v in (0.01, 0.05)},
    **{f"ls{v:g}": [f"loss.label_smoothing={v}"] for v in (0.1, 0.2)},
    **{f"gap{v}": [f"loss.jigsaw_gap={v}"] for v in (0, 4)},
    **{f"h{v}": [f"model.projector.hidden_dim={v}"] for v in (128, 2048)},
}
for _base in list(VARIANTS["pretext"]):
    if _base == "campaign":
        continue
    for _t, _o in _PRETEXT_TUNING.items():
        VARIANTS["pretext"][f"{_base}_{_t}"] = VARIANTS["pretext"][_base] + _o
    for _t1, _t2 in itertools.combinations(_PRETEXT_TUNING, 2):
        if _t1[:2] != _t2[:2]:
            VARIANTS["pretext"][f"{_base}_{_t1}_{_t2}"] = (VARIANTS["pretext"][_base]
                                                         + _PRETEXT_TUNING[_t1]
                                                         + _PRETEXT_TUNING[_t2])

# DINOv2 sized to this dataset (2026-11-07 diagnosis). The campaign recipe kept DINOv2's
# 4096 prototypes for 743 training subjects and balanced the teacher targets with
# Sinkhorn-Knopp over batches of 16 (32 teacher rows): equipartition then forces every row
# onto >= K/32 = 128 prototypes of its own (teacher entropy 0.65 ln K, every row a different
# argmax), i.e. per-batch instance discrimination instead of clusters shared by similar
# anatomy. ``k256``: 256 prototypes in both heads (about one per three subjects, the
# K/dataset ratio of DINO on ImageNet is ~1/20); ``b64``: batches of 64 (pass
# ``--batch-size 64``) with the learning rate scaled by sqrt(4); ``t07``: DINOv2's own
# teacher temperature (0.04 -> 0.07) instead of the 0.01 the batch-16 recipe needed.
_K256 = ["model.projector.out_dim=256"]
_LR64 = ["optimizer.lr=2.0e-4"]
_T07 = ["loss.teacher_temp=0.04", "loss.teacher_temp_final=0.07"]
VARIANTS["dinov2"].update({
    "k256": _K256,
    "b64": _LR64,
    "b64_k256": _LR64 + _K256,
    "b64_k256_t07": _LR64 + _K256 + _T07,
})

# GroupNorm BYOL tuning grid: gn_dp{projector dropout}_dq{predictor dropout}[_wd{weight decay}]
# (wide heads, GroupNorm in projector and predictor; weight decay default 1e-6).
for _p in (0.0, 0.1, 0.2, 0.3):
    for _q in (0.0, 0.1, 0.2, 0.3):
        _name = f"gn_dp{_p:g}_dq{_q:g}"
        VARIANTS["byol"][_name] = BORDER + WIDE + GROUP_NORM + _dropout(_p, _q)
        for _wd in (0.01, 0.05):
            VARIANTS["byol"][f"{_name}_wd{_wd:g}"] = (VARIANTS["byol"][_name]
                                                     + [f"optimizer.weight_decay={_wd}"])
# The tuned GroupNorm BYOL with the backbone BatchNorm statistics frozen (``_fbn``).
VARIANTS["byol"]["gn_dp0.3_dq0_wd0.01_fbn"] = (VARIANTS["byol"]["gn_dp0.3_dq0_wd0.01"]
                                               + ["++model.freeze_backbone_bn=true"])


# ---------------------------------------------------------------- backbones
# Encoder swaps (``backbone=``) on top of any model/variant: the method keeps its
# heads, loss, optimizer and views; only ``model.backbone`` changes. Each is the
# start one of the methods already used (ImageNet torchvision / DeiT-S weights
# inflated to 3D), so ``DEFAULT_BACKBONE`` reproduces the original pairing. With a
# CNN, DINOv2 drops its iBOT branch (no patch tokens to mask): CLS distillation,
# Sinkhorn-Knopp and KoLeo stay (t5_ssl.models.dinov2_module).
BACKBONES: dict[str, list[str]] = {
    "convnext_tiny": ["model.backbone.variant=convnext_tiny_3d", "model.backbone.pretrained=true"],
    "resnet18": ["model.backbone.variant=resnet18_3d", "model.backbone.pretrained=true"],
    "vit_small": ["model.backbone.variant=vit_small_3d", "model.backbone.pretrained=deit_small_in1k",
                  "++model.backbone.patch_size=8", "++model.backbone.embed_dim=384",
                  "++model.backbone.depth=12", "++model.backbone.num_heads=6",
                  "++model.backbone.drop_path=0.1"],
}
# ViT-S starting from Meta's DINOv2 ViT-S/14 (LVD-142M) instead of DeiT-S, same inflation.
BACKBONES["vit_small_dv2"] = [o.replace("deit_small_in1k", "dinov2_vits14")
                              for o in BACKBONES["vit_small"]]
DEFAULT_BACKBONE = {"vicreg": "convnext_tiny", "byol": "resnet18", "dinov2": "vit_small",
                    "pretext": "convnext_tiny"}


# ---------------------------------------------------------------- regions of interest
# ROI configurations: cube side (voxels of 2 mm) and cache. "multi3" stacks three
# bilateral 32-voxel regions as 6 channels: hippocampus, lateral ventricle and
# posteromedial cortex (posterior/isthmus cingulate + precuneus); see
# t5_ssl.data.roi3d.REGIONS and scripts/build_roi_caches.py.
ROIS: dict[str, dict] = {
    "32": {"size": 32, "cache": "hippo_s32", "channels": 2},
    "48": {"size": 48, "cache": "hippo_s48", "channels": 2},
    "64": {"size": 64, "cache": "hippo_s64", "channels": 2},
    "multi3": {"size": 32, "cache": "multi3_s32", "channels": 6},
    # multi3 + lateral temporal (inferior/middle temporal gyri) and inferior parietal
    # (+ supramarginal) cortex: 5 bilateral regions of the usual AD atrophy, 10 channels.
    "atrophy5": {"size": 32, "cache": "atrophy5_s32", "channels": 10},
}


def roi_overrides(roi: str) -> list[str]:
    spec = ROIS[roi]
    return [f"data.roi_size={spec['size']}", f"data.cache_path=data/roi3d/{spec['cache']}.npy",
            f"data.channels={spec['channels']}", f"model.backbone.in_channels={spec['channels']}",
            f"++model.backbone.image_size={spec['size']}"]


# ---------------------------------------------------------------- augmentation recipes
EVAL = "eval"  # side of the probe's central cube in the cache: size / round(1.2 * size)
_NONE = {"side_range": [EVAL, EVAL], "centered_crop": True, "lr_swap_p": 0.5,
         "ap_flip_p": 0.0, "affine_degrees": 0.0, "rotation_padding": "border", "blur_p": 0.0,
         "blur_sigma": [0.1, 1.5], "noise_std": 0.0, "intensity_scale": [1.0, 1.0],
         "torchio": [], "n_local_crops": 0, "gpu_elastic": None, "tio_gpu_elastic": None}
_TIO = {  # calibrated to RMSE 0.080 (3D campaign strength)
    "bias": {"name": "RandomBiasField", "coefficients": 0.18391731957276947, "order": 2},
    "noise": {"name": "RandomNoise", "std": [0, 0.2153708703252311]},
    "blur": {"name": "RandomBlur", "std": [0, 1.3432277367461767]},
    "gamma": {"name": "RandomGamma", "log_gamma": [-0.6447947973601844, 0.6447947973601844]},
    "motion": {"name": "RandomMotion", "degrees": 1.24186359201996,
               "translation": 1.24186359201996, "num_transforms": 2},
    "ghosting": {"name": "RandomGhosting", "num_ghosts": [2, 6], "axes": [0, 1, 2],
                 "intensity": [0, 1.2644584319150896]},
    "spike": {"name": "RandomSpike", "num_spikes": 1, "intensity": 0.7771382791443595},
    "elastic": {"name": "RandomElasticDeformation", "num_control_points": 7,
                "max_displacement": 1.9532824307146963, "locked_borders": 2},
}
_GEO = {"side_range": [EVAL, 0.92], "centered_crop": False, "affine_degrees": 5.0}


def _tio(*names: str) -> list[dict]:
    return [{**_TIO[n], "p": 1.0} for n in names]


def _one_of(*names: str) -> list[dict]:
    return [{"name": "OneOf", "p": 1.0, "transforms": [_TIO[n] for n in names]}]


_ARTIFACTS = ("bias", "noise", "motion", "ghosting", "spike")
AUG_RECIPES: dict[str, dict] = {
    # identity controls / single components (all with the left-right flip)
    "flip": {},
    "crop": {"side_range": [EVAL, 0.92], "centered_crop": False},         # soft rescale + shift
    "translate": {"centered_crop": False},                                # +/-3 voxels, no rescale
    "rotate": {"affine_degrees": 5.0},                                    # +/-5 deg, border padding
    "intensity": {"intensity_scale": [0.9, 1.1], "torchio": _tio("gamma")},
    "blur": {"torchio": _tio("blur")},
    "bias": {"torchio": _tio("bias")},
    "noise": {"torchio": _tio("noise")},
    "motion": {"torchio": _tio("motion")},
    "ghosting": {"torchio": _tio("ghosting")},
    "spike": {"torchio": _tio("spike")},
    "elastic": {"torchio": _tio("elastic")},
    # The same deformation on the GPU (t5_ssl.data.gpu_augment.elastic_deform), calibrated
    # to the same RMSE (0.080): TorchIO's costs ~0.75 s per 64-voxel sample on one core.
    "elastic_gpu": {"gpu_elastic": {"max_displacement": 1.5042, "control_points": 7}},
    # TorchIO's own deformation (``elastic``) computed on the GPU: identical output for
    # the same control points (t5_ssl.data.gpu_augment.tio_elastic_deform); the loader
    # hands over the whole cached cube and the module deforms it and takes the center.
    "elastic_tio_gpu": {"tio_gpu_elastic": {
        "num_control_points": _TIO["elastic"]["num_control_points"],
        "max_displacement": _TIO["elastic"]["max_displacement"],
        "locked_borders": _TIO["elastic"]["locked_borders"]}},
    # combinations
    "geo": dict(_GEO),
    "artifacts": {"torchio": _one_of(*_ARTIFACTS)},
    "geo_artifacts": {**_GEO, "torchio": _one_of(*_ARTIFACTS)},
    "geo_photo": {**_GEO, "intensity_scale": [0.9, 1.1], "blur_p": 0.3, "blur_sigma": [0.1, 1.0],
                  "torchio": _tio("bias")},
    # the recipes each model used so far (references)
    "campaign_zeros": {"side_range": [0.84, 1.0], "centered_crop": False, "affine_degrees": 10.0,
                       "rotation_padding": "zeros", "torchio": _tio("bias")},  # VICReg
    "campaign_border": {"side_range": [0.84, 1.0], "centered_crop": False,
                        "affine_degrees": 10.0, "torchio": _tio("bias")},     # BYOL border_*
    "dinov2_ref": {"side_range": [1.0, 1.0], "torchio": _tio("bias")},        # DINOv2
}
REFERENCE_RECIPE = {"vicreg": "campaign_zeros", "byol": "campaign_border", "dinov2": "dinov2_ref"}


def aug_spec(recipe: str, size: int = 32) -> dict:
    """Return the full augmentation node of a recipe (``_NONE`` defaults + the recipe)."""
    if recipe not in AUG_RECIPES:
        raise ValueError(f"Unknown augmentation recipe '{recipe}'. Choose from {list(AUG_RECIPES)}.")
    spec = {**_NONE, **AUG_RECIPES[recipe]}
    side = size / round(1.2 * size)
    spec["side_range"] = [side if v == EVAL else v for v in spec["side_range"]]
    return spec


def _hydra(value: object) -> str:
    if isinstance(value, dict):
        return "{" + ",".join(f"{k}:{_hydra(v)}" for k, v in value.items()) + "}"
    if isinstance(value, list | tuple):
        return "[" + ",".join(_hydra(v) for v in value) + "]"
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


def aug_overrides(recipe: str, size: int = 32) -> list[str]:
    return [f"++augmentation.{k}={_hydra(v)}" for k, v in aug_spec(recipe, size).items()]


# The `perf=rtx4090` runtime: everything the Trainer needs besides the perf node.
FAST_RUNTIME = ["+perf=rtx4090", "trainer.precision=bf16-mixed", "data.num_workers=6",
                "data.pin_memory=true", "++data.prefetch_factor=4"]
# What campaign_ssl3d(_v2) ran with (fp32, no workers), for exact reproduction.
CAMPAIGN_RUNTIME = ["trainer.precision=32-true"]


# SSL validation on the training views, fixed per session (and per batch for the GPU
# deformations), plus the same measurement on 120 training subjects (``trainval/``): the
# validation loss is then the training objective on unseen subjects, on the same scale.
# Until 2026-11-07 only VICReg ``campaign_augval`` did this; BYOL and DINOv2 validated on
# the clean cube and its mirror, which made their validation loss incomparable with the
# training loss. ``--set ++augmentation.val_views=fixed`` restores the old validation.
AUGMENTED_VALIDATION = ["++augmentation.val_views=train", "++data.train_monitor_subjects=120"]


def warmup_epochs(epochs: int) -> int:
    return max(1, round(epochs / 5))  # 6 of 30, as campaign_ssl3d; 10 of 50


def overrides(model: str, variant: str = "campaign", *, epochs: int = 30, seed: int = 1,
              fast: bool = True, checkpoints: bool = False, accelerator: str = "auto",
              log_root: str | None = None, run_name: str | None = None,
              aug_recipe: str | None = None, roi: str = "32", backbone: str | None = None,
              extra: list[str] | None = None) -> list[str]:
    """Hydra overrides for one run (later entries win, so ``extra`` overrides anything)."""
    if model not in BASE:
        raise ValueError(f"Unknown model '{model}'. Choose from {list(BASE)}.")
    if variant not in VARIANTS[model]:
        raise ValueError(f"Unknown {model} variant '{variant}'. Choose from {list(VARIANTS[model])}.")
    if roi not in ROIS:
        raise ValueError(f"Unknown ROI '{roi}'. Choose from {list(ROIS)}.")
    if backbone is not None and backbone not in BACKBONES:
        raise ValueError(f"Unknown backbone '{backbone}'. Choose from {list(BACKBONES)}.")
    out = ["data=oasis3_roi3d", *roi_overrides(roi), f"data.manifest_path={MANIFEST}",
           *AUGMENTED_VALIDATION,
           "trainer=full50", f"trainer.accelerator={accelerator}", "trainer.deterministic=false",
           "trainer.enable_progress_bar=false", "trainer.early_stopping_patience=null",
           "trainer.enable_health_check=false", f"trainer.enable_checkpointing={str(checkpoints).lower()}",
           f"trainer.max_epochs={epochs}", f"optimizer.warmup_epochs={warmup_epochs(epochs)}",
           *BASE[model], f"seed={seed}"]
    if model == "dinov2":
        out.append(f"loss.teacher_temp_warmup_epochs={max(1, round(0.3 * epochs))}")
    out += VARIANTS[model][variant]
    if backbone:
        out += BACKBONES[backbone]
    if aug_recipe:
        out += aug_overrides(aug_recipe, ROIS[roi]["size"])
    out += FAST_RUNTIME if fast else CAMPAIGN_RUNTIME
    if log_root:
        out.append(f"++log_root={log_root}")
    if run_name:
        out.append(f"++run_name={run_name}")
    return dedupe(out + list(extra or []))


def dedupe(items: list[str]) -> list[str]:
    """Keep only the last override of each key (Hydra rejects some duplicates)."""
    seen: set[str] = set()
    kept = []
    for item in reversed(items):
        key = item.lstrip("+~").split("=", 1)[0]
        if key not in seen:
            seen.add(key)
            kept.append(item)
    return kept[::-1]


def run_name(model: str, variant: str, seed: int, aug_recipe: str | None = None,
             roi: str | None = None, backbone: str | None = None) -> str:
    return (f"{model}_{variant}" + (f"__{aug_recipe}" if aug_recipe else "")
            + (f"__bb{backbone}" if backbone else "")
            + (f"__roi{roi}" if roi else "") + f"_s{seed}")
