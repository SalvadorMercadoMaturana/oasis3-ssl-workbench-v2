"""Diagnose which augmentation dominates BYOL's trained representation.

Trains BYOL once more with the default recipe (EMA 0.996, seed 2, campaign
views) and, for the 142 held-out subjects (val + test), measures how much each
augmentation component alone moves the embedding, relative to how much
subjects differ on the clean cube, for the ImageNet-inflated start and for the
trained online and target encoders:

- ``sensitivity`` = within-subject variance between two independent draws of
  the component / between-subject variance of the clean central cube;
- ``top1`` = fraction of first draws whose nearest second draw is the same
  subject (identity retained under that component; chance = 1/142).

Components: translation (32-voxel sub-cube, random offset, no resize), crop
(84-100% sub-cube resized to 32), rotation +/-10 deg with zero padding and with
border padding, bias field, left-right channel swap, and the full campaign
views. Results: ``campaign_ssl3d_v2/byol_sensitivity.json``.

Usage: ``python scripts/analyze_byol_sensitivity.py``.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_ssl3d_encoders as enc
import run_ssl3d_v2 as v2

OUT = v2.CAMPAIGN / "byol_sensitivity.json"


class BorderRotation:
    """Campaign geometry with rotation resampled using border instead of zero padding."""

    def __init__(self, base) -> None:
        self.base = base

    def __call__(self, cube: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._view(cube), self._view(cube)

    def _view(self, cube: torch.Tensor) -> torch.Tensor:
        from t5_ssl.data.roi3d import _rotation_matrix
        view = F.interpolate(cube[None], size=(32,) * 3, mode="trilinear", align_corners=False)[0]
        theta = torch.zeros(1, 3, 4)
        theta[0, :, :3] = _rotation_matrix(10.0)
        grid = F.affine_grid(theta, [1, *view.shape], align_corners=False)
        return F.grid_sample(view[None], grid, mode="bilinear", padding_mode="border",
                             align_corners=False)[0].clamp(0, 1)


def components(bias: list[dict]) -> dict:
    from t5_ssl.data.roi3d import ROIView3DTransform
    plain = {"degrees": 0.0, "lr_swap_p": 0.0, "torchio_transforms": None}
    return {
        "traslación (sub-cubo de 32, sin reescalar)": ROIView3DTransform(
            size=32, side_range=(32 / 38, 32 / 38), **plain),
        "recorte 84-100 % + reescalado": ROIView3DTransform(size=32, side_range=(0.84, 1.0), **plain),
        "rotación ±10° (relleno con ceros)": ROIView3DTransform(
            size=32, side_range=(1.0, 1.0), degrees=10.0, lr_swap_p=0.0, torchio_transforms=None),
        "rotación ±10° (relleno con borde)": BorderRotation(None),
        "campo de sesgo": ROIView3DTransform(size=32, side_range=(1.0, 1.0), degrees=0.0,
                                             lr_swap_p=0.0, torchio_transforms=bias),
        "intercambio izquierda-derecha": ROIView3DTransform(size=32, side_range=(1.0, 1.0),
                                                            degrees=0.0, lr_swap_p=0.5,
                                                            torchio_transforms=None),
        "vistas completas de la campaña": ROIView3DTransform(size=32, side_range=(0.84, 1.0),
                                                             degrees=10.0, lr_swap_p=0.5,
                                                             torchio_transforms=bias),
    }


def measure(encoder: torch.nn.Module, clean_inputs: list, draws: dict) -> dict:
    clean = enc.embed(encoder, clean_inputs)
    between = float(((clean - clean.mean(0)) ** 2).sum(1).mean())
    out = {}
    for name, (first, second) in draws.items():
        a, b = enc.embed(encoder, first), enc.embed(encoder, second)
        within = float(((a - b) ** 2).sum(1).mean() / 2)
        mu = np.concatenate([a, b]).mean(0)
        sim = (F.normalize(torch.from_numpy(a - mu), dim=1)
               @ F.normalize(torch.from_numpy(b - mu), dim=1).T).numpy()
        out[name] = {"sensitivity": within / between,
                     "top1": float((sim.argmax(1) == np.arange(len(sim))).mean())}
    return out


def main() -> None:
    from omegaconf import OmegaConf

    from t5_ssl.data.roi3d import ROIEval3D
    from t5_ssl.train import build_backbone, train
    cfg = enc.compose("byol_3d", 2, ["loss.ema_base=0.996"])
    cubes, _ = enc.held_out_cubes(cfg)
    clean_inputs = [ROIEval3D(32)(c) for c in cubes]
    bias = [{**spec, "p": 1.0} for spec in OmegaConf.to_container(cfg.augmentation.torchio,
                                                                  resolve=True)]
    draws = {name: enc.views(transform, cubes, seed=11) for name, transform in
             components(bias).items()}
    results = {"subjects": len(cubes), "chance": 1 / len(cubes)}
    torch.manual_seed(0)
    results["inicio ImageNet"] = measure(build_backbone(cfg), clean_inputs, draws)
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    model = train(cfg)["model"]
    results["BYOL online"] = measure(model.backbone, clean_inputs, draws)
    results["BYOL target"] = measure(model.target_backbone, clean_inputs, draws)
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for encoder, scores in results.items():
        if isinstance(scores, dict):
            print(encoder, {k: (round(v["sensitivity"], 4), round(v["top1"], 3))
                            for k, v in scores.items()}, flush=True)
    assert math.isfinite(results["BYOL online"]["vistas completas de la campaña"]["sensitivity"])


if __name__ == "__main__":
    main()
