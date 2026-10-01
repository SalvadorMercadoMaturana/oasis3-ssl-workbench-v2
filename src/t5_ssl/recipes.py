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

MANIFEST = "campaign_full50_pretrained_balanced/train_sessions_eval_primary_balanced.csv"
MODELS = ("byol", "vicreg", "dinov2")

BASE = {
    "vicreg": ["model=vicreg_convnext_3d", "loss=vicreg", "augmentation=roi3d_bias"],
    "byol": ["model=byol_resnet18_3d", "loss=byol", "augmentation=roi3d_bias"],
    "dinov2": ["model=dinov2_vit3d", "loss=dinov2", "augmentation=roi3d_dinov2_bias",
               "optimizer=adamw_dinov2"],
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
    },
    "vicreg": {"campaign": [], "border": BORDER},
    "dinov2": {"campaign": []},
}

# ---------------------------------------------------------------- augmentation recipes
_SIDE_EVAL = 32 / 38  # the central 32-voxel cube of the 38-voxel cache = the probe input
_NONE = {"side_range": [_SIDE_EVAL, _SIDE_EVAL], "centered_crop": True, "lr_swap_p": 0.5,
         "ap_flip_p": 0.0, "affine_degrees": 0.0, "rotation_padding": "border", "blur_p": 0.0,
         "blur_sigma": [0.1, 1.5], "noise_std": 0.0, "intensity_scale": [1.0, 1.0],
         "torchio": [], "n_local_crops": 0}
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
_GEO = {"side_range": [_SIDE_EVAL, 0.92], "centered_crop": False, "affine_degrees": 5.0}


def _tio(*names: str) -> list[dict]:
    return [{**_TIO[n], "p": 1.0} for n in names]


def _one_of(*names: str) -> list[dict]:
    return [{"name": "OneOf", "p": 1.0, "transforms": [_TIO[n] for n in names]}]


_ARTIFACTS = ("bias", "noise", "motion", "ghosting", "spike")
AUG_RECIPES: dict[str, dict] = {
    # identity controls / single components (all with the left-right flip)
    "flip": {},
    "crop": {"side_range": [_SIDE_EVAL, 0.92], "centered_crop": False},   # soft rescale + shift
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


def aug_spec(recipe: str) -> dict:
    """Return the full augmentation node of a recipe (``_NONE`` defaults + the recipe)."""
    if recipe not in AUG_RECIPES:
        raise ValueError(f"Unknown augmentation recipe '{recipe}'. Choose from {list(AUG_RECIPES)}.")
    return {**_NONE, **AUG_RECIPES[recipe]}


def _hydra(value: object) -> str:
    if isinstance(value, dict):
        return "{" + ",".join(f"{k}:{_hydra(v)}" for k, v in value.items()) + "}"
    if isinstance(value, list | tuple):
        return "[" + ",".join(_hydra(v) for v in value) + "]"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def aug_overrides(recipe: str) -> list[str]:
    return [f"++augmentation.{k}={_hydra(v)}" for k, v in aug_spec(recipe).items()]


# The `perf=rtx4090` runtime: everything the Trainer needs besides the perf node.
FAST_RUNTIME = ["+perf=rtx4090", "trainer.precision=bf16-mixed", "data.num_workers=6",
                "data.pin_memory=true", "++data.prefetch_factor=4"]
# What campaign_ssl3d(_v2) ran with (fp32, no workers), for exact reproduction.
CAMPAIGN_RUNTIME = ["trainer.precision=32-true"]


def warmup_epochs(epochs: int) -> int:
    return max(1, round(epochs / 5))  # 6 of 30, as campaign_ssl3d; 10 of 50


def overrides(model: str, variant: str = "campaign", *, epochs: int = 30, seed: int = 1,
              fast: bool = True, checkpoints: bool = False, accelerator: str = "auto",
              log_root: str | None = None, run_name: str | None = None,
              aug_recipe: str | None = None, extra: list[str] | None = None) -> list[str]:
    """Hydra overrides for one run (later entries win, so ``extra`` overrides anything)."""
    if model not in BASE:
        raise ValueError(f"Unknown model '{model}'. Choose from {list(BASE)}.")
    if variant not in VARIANTS[model]:
        raise ValueError(f"Unknown {model} variant '{variant}'. Choose from {list(VARIANTS[model])}.")
    out = ["data=oasis3_roi3d", "data.roi_size=32", f"data.manifest_path={MANIFEST}",
           "trainer=full50", f"trainer.accelerator={accelerator}", "trainer.deterministic=false",
           "trainer.enable_progress_bar=false", "trainer.early_stopping_patience=null",
           "trainer.enable_health_check=false", f"trainer.enable_checkpointing={str(checkpoints).lower()}",
           f"trainer.max_epochs={epochs}", f"optimizer.warmup_epochs={warmup_epochs(epochs)}",
           *BASE[model], f"seed={seed}"]
    if model == "dinov2":
        out.append(f"loss.teacher_temp_warmup_epochs={max(1, round(0.3 * epochs))}")
    out += VARIANTS[model][variant]
    if aug_recipe:
        out += aug_overrides(aug_recipe)
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


def run_name(model: str, variant: str, seed: int, aug_recipe: str | None = None) -> str:
    return f"{model}_{variant}" + (f"__{aug_recipe}" if aug_recipe else "") + f"_s{seed}"
