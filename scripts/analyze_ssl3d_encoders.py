"""What SSL training does to the 3D encoders' representation (diagnostic for BYOL).

For held-out subjects (val + test splits, one session each) this compares encoders
before and after SSL training on:

- identity retention: centered cosine between two augmented views of the same
  subject vs two different subjects, and top-1 view matching (chance = 1/N);
- signal vs augmentation noise: between-subject variance of the clean central
  cube over within-subject variance across augmented views;
- effective rank of the clean embeddings (variance spectrum, as the probe);
- linear CKA between the initial and the trained clean embeddings.

Stage 1 scores the three starting encoders (ConvNeXt-T 3D and ResNet-18 3D
ImageNet-inflated, ViT-S 3D DeiT-inflated) under the campaign views and under
bias-field-only views. Stage 2 trains BYOL (default recipe, EMA 0.996) and
VICReg with the campaign protocol and scores the trained online encoders (and
BYOL's EMA target). No checkpoints are written; results go to
``campaign_ssl3d_v2/encoder_analysis.json`` after every stage.

Usage: ``python scripts/analyze_ssl3d_encoders.py``.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_ssl3d_v2 as v2

OUT = v2.CAMPAIGN / "encoder_analysis.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def compose(run: str, seed: int, extra: list[str] | None = None):
    from hydra import compose as hydra_compose
    from hydra import initialize_config_dir
    overrides = v2.overrides_for(run, seed)
    if extra:
        keys = {o.lstrip("+").split("=")[0] for o in extra}
        overrides = [o for o in overrides if o.lstrip("+").split("=")[0] not in keys] + extra
    with initialize_config_dir(config_dir=str(v2.ROOT / "configs"), version_base="1.3"):
        return hydra_compose(config_name="train", overrides=overrides)


def held_out_cubes(cfg) -> tuple[list[torch.Tensor], np.ndarray]:
    """Return the cached cubes and AD labels (-1 = other) of the val + test sessions."""
    from t5_ssl.data.roi3d import ROIEval3D
    from t5_ssl.train import build_datamodule
    clean = ROIEval3D(32)
    dm = build_datamodule(cfg, clean, clean)
    dm.setup()
    records = dm.val_dataset.records + dm.test_dataset.records
    cubes = [torch.from_numpy(dm.cubes[r["idx"]].astype(np.float32)) for r in records]
    labels = np.array([{"AD": 1, "HC": 0}.get(dm.subject_labels.get(r["subject"], ""), -1)
                       for r in records])
    return cubes, labels


@torch.no_grad()
def embed(encoder: torch.nn.Module, volumes: list[torch.Tensor]) -> np.ndarray:
    encoder = encoder.to(DEVICE).eval()
    out = [encoder(torch.stack(volumes[i:i + 32]).to(DEVICE)).float().cpu()
           for i in range(0, len(volumes), 32)]
    return torch.cat(out).numpy()


def views(transform, cubes: list[torch.Tensor], seed: int = 0) -> tuple[list, list]:
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)
    pairs = [transform(cube)[:2] for cube in cubes]
    return [a for a, _ in pairs], [b for _, b in pairs]


def effective_rank(x: np.ndarray) -> float:
    spectrum = np.linalg.svd(x - x.mean(0), compute_uv=False) ** 2
    p = spectrum[spectrum > 0] / spectrum.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x, y = x - x.mean(0), y - y.mean(0)
    return float(np.linalg.norm(y.T @ x) ** 2
                 / (np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y)))


def scores(clean: np.ndarray, view_a: np.ndarray, view_b: np.ndarray) -> dict[str, float]:
    mu = np.concatenate([view_a, view_b]).mean(0)
    a = F.normalize(torch.from_numpy(view_a - mu), dim=1)
    b = F.normalize(torch.from_numpy(view_b - mu), dim=1)
    sim = (a @ b.T).numpy()
    n = len(sim)
    within = float(((view_a - view_b) ** 2).sum(1).mean() / 2)
    between = float(((clean - clean.mean(0)) ** 2).sum(1).mean())
    return {"same_subject_cos": float(np.diag(sim).mean()),
            "different_subject_cos": float(sim[~np.eye(n, dtype=bool)].mean()),
            "view_matching_top1": float((sim.argmax(1) == np.arange(n)).mean()),
            "chance": 1 / n, "between_over_within": between / within,
            "effective_rank": effective_rank(clean)}


def save(results: dict) -> None:
    OUT.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"saved {OUT}", flush=True)


def main() -> None:
    from t5_ssl.data.roi3d import ROIEval3D, ROIView3DTransform, build_roi3d_transforms
    from t5_ssl.train import build_backbone, train
    results = json.loads(OUT.read_text(encoding="utf-8")) if OUT.is_file() else {}
    base = compose("byol_3d", 1)
    cubes, _ = held_out_cubes(base)
    results["subjects"] = len(cubes)
    clean_inputs = [ROIEval3D(32)(c) for c in cubes]
    campaign_views = views(build_roi3d_transforms(base)[0], cubes)
    from omegaconf import OmegaConf
    bias = [{**spec, "p": 1.0}
            for spec in OmegaConf.to_container(base.augmentation.torchio, resolve=True)]
    still = ROIView3DTransform(size=32, side_range=(1.0, 1.0), degrees=0.0, lr_swap_p=0.5,
                               torchio_transforms=bias)
    still_views = views(still, cubes)

    starts = {"convnext_imagenet": ("vicreg_3d", []), "resnet_imagenet": ("byol_3d", []),
              "vit_deit": ("dinov2_3d", [])}
    initial = {}
    for name, (run, extra) in starts.items():
        torch.manual_seed(0)
        encoder = build_backbone(compose(run, 1, extra))
        clean = embed(encoder, clean_inputs)
        initial[name] = clean
        results.setdefault("initial", {})[name] = {
            "campaign_views": scores(clean, *(embed(encoder, v) for v in campaign_views)),
            "bias_only_views": scores(clean, *(embed(encoder, v) for v in still_views))}
        print(name, results["initial"][name], flush=True)
        save(results)

    trained = {"byol": ("byol_3d", ["loss.ema_base=0.996"], "resnet_imagenet"),
               "vicreg": ("vicreg_3d", [], "convnext_imagenet")}
    for name, (run, extra, start) in trained.items():
        if name in results.get("trained", {}):
            continue
        result = train(compose(run, 1, extra))
        model = result["model"]
        encoders = {"online": model.backbone}
        if hasattr(model, "target_backbone"):
            encoders["target"] = model.target_backbone
        entry = {"experiment_dir": str(result["exp_dir"])}
        for role, encoder in encoders.items():
            clean = embed(encoder, clean_inputs)
            entry[role] = {"campaign_views": scores(clean, *(embed(encoder, v) for v in campaign_views)),
                           "cka_to_initial": linear_cka(initial[start], clean)}
        results.setdefault("trained", {})[name] = entry
        print(name, entry, flush=True)
        save(results)
        del model, result
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
