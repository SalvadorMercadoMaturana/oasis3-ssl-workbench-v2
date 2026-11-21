"""Identity and collapse checks: does the encoder keep subjects apart under augmentation.

On held-out subjects (val + test, one session each):

- ``same_subject_cos`` / ``different_subject_cos``: centered cosine between two augmented
  views of the same subject / of two different subjects;
- ``view_matching_top1``: fraction of first views whose nearest second view is the same
  subject (chance = 1 / subjects);
- ``between_over_within``: between-subject variance of the clean cube over the
  within-subject variance of the views (Var_subject / Var_aug);
- ``effective_rank``: exp(entropy) of the *variance* spectrum of the clean embeddings
  (``metrics.effective_rank``, used by the probes, normalizes the singular values instead);
- ``linear_cka``: similarity of two embeddings of the same inputs (e.g. before/after SSL).

``identity_scores`` is what ``train.py --identity`` stores in ``result.json["identity"]``.

Moved from ``scripts/legacy/analyze_ssl3d_encoders.py`` (embed, views, effective_rank,
linear_cka, scores) and the root ``train.py`` (identity_scores).
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from t5_ssl import recipes

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def embed(encoder: torch.nn.Module, volumes: list[torch.Tensor]) -> np.ndarray:
    """Embeddings of ``volumes`` in batches of 32 (eval mode)."""
    encoder = encoder.to(DEVICE).eval()
    out = [encoder(torch.stack(volumes[i:i + 32]).to(DEVICE)).float().cpu()
           for i in range(0, len(volumes), 32)]
    return torch.cat(out).numpy()


def views(transform, cubes: list[torch.Tensor], seed: int = 0) -> tuple[list, list]:
    """First and second augmented view of every cube, from a fixed seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
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


def identity_scores(encoder: Any, dm: Any, cfg: Any) -> dict[str, Any]:
    """Identity retention and collapse on the val + test subjects (one session each).

    Under the run's own training views and under a fixed reference set (the
    campaign geometry with border padding + bias field, the same for every
    recipe): centered cosine same vs different subject, top-1 view matching,
    between-subject variance of the clean cube over within-subject variance of
    the views (``between_over_within`` = Var_subject / Var_aug) and the
    effective rank (variance spectrum) of the clean embeddings.
    """
    from omegaconf import OmegaConf

    from t5_ssl.data.roi3d import ROIEval3D, analysis_views, build_roi3d_transforms
    for attr in ("forward", "forward_tokens"):
        if attr in encoder.__dict__:
            delattr(encoder, attr)
    encoder = encoder.float().to(memory_format=torch.contiguous_format)
    records = dm.val_dataset.records + dm.test_dataset.records
    cubes = [torch.from_numpy(dm.cubes[r["idx"]].astype(np.float32)) for r in records]
    size = int(cfg.data.roi_size)
    clean = embed(encoder, [ROIEval3D(size)(c) for c in cubes])
    reference = OmegaConf.create({"data": {"roi_size": size},
                                  "augmentation": recipes.aug_spec("campaign_border", size)})
    out: dict[str, Any] = {"subjects": len(cubes)}
    own_cfg = cfg
    if cfg.augmentation.get("gpu_elastic"):  # same deformation, CPU (TorchIO) version
        own_cfg = OmegaConf.create({"data": {"roi_size": size},
                                    "augmentation": recipes.aug_spec("elastic", size)})
    own = (analysis_views(cfg) if cfg.augmentation.get("tio_gpu_elastic")
           else build_roi3d_transforms(own_cfg)[0])
    for name, transform in (("own", own),
                            ("reference", build_roi3d_transforms(reference)[0])):
        first, second = views(transform, cubes, seed=0)
        out[name] = scores(clean, embed(encoder, first), embed(encoder, second))
    print(f"[identity] own: Var_subject/Var_aug={out['own']['between_over_within']:.3g} "
          f"same={out['own']['same_subject_cos']:.3f} diff={out['own']['different_subject_cos']:.3f} "
          f"erank={out['own']['effective_rank']:.2f}", flush=True)
    return out
