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

VARIANTS: dict[str, dict[str, list[str]]] = {
    "byol": {
        "campaign": [],
        "border": BORDER,
        "border_wide": BORDER + WIDE,
        "border_gn": BORDER + GROUP_NORM,
        "border_predlr": BORDER + PREDICTOR_LR,
        "border_h3": BORDER + WIDE + GROUP_NORM + PREDICTOR_LR,
    },
    "vicreg": {"campaign": [], "border": BORDER},
    "dinov2": {"campaign": []},
}

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
              extra: list[str] | None = None) -> list[str]:
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


def run_name(model: str, variant: str, seed: int) -> str:
    return f"{model}_{variant}_s{seed}"
