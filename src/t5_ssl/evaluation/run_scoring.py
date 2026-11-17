"""Everything ``train.py`` measures on a frozen encoder after training (``result.json``).

``evaluate_encoders`` runs, for each encoder of the run (``main`` = after SSL, ``init`` =
before SSL when it is scored too):

1. the campaign probe protocol (``probe_v1.score``): ``cv`` and ``test``;
2. for ``main``, the probe protocol v2 (``score_protocol_v2`` -> ``protocol_v2.evaluate``):
   representative CV with the SSL-only AD/HC subjects, out-of-fold thresholds, auc_age;
3. with ``--spatial-probe``, the same probes on the pre-pooling feature map
   (``spatial_encoder``).

``mean3_cv`` / ``mean3_test`` are the one-number summaries printed in the logs.

Moved from the root ``train.py`` (spatial_encoder; evaluate, renamed
``evaluate_encoders``; protocol_v2, renamed ``score_protocol_v2``; mean3_cv; mean3_test).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any


def spatial_encoder(backbone: Any) -> Any:
    """Pre-GAP feature map average-pooled to 2^3 cells and flattened (probe H4).

    ResNet-18 3D: layer4 (512 x 2^3 = 4096); ConvNeXt-T 3D: stage 3, the last
    map wider than one voxel at 32^3 (384 x 2^3 = 3072); ViT-S 3D: the 4^3 patch
    tokens pooled to 2^3 (384 x 8 = 3072).
    """
    import torch.nn.functional as F
    from torch import nn

    class SpatialEncoder(nn.Module):
        def __init__(self, inner: nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, x: Any) -> Any:
            b = self.inner
            if hasattr(b, "forward_tokens"):
                _, patches = b.forward_tokens(x)
                grid = round(patches.shape[1] ** (1 / 3))
                fmap = patches.transpose(1, 2).reshape(x.shape[0], -1, grid, grid, grid)
            elif hasattr(b, "layer4"):
                h = b.relu(b.bn1(b.conv1(x)))
                fmap = b.layer4(b.layer3(b.layer2(b.layer1(h))))
            else:
                fmap = b.features[:6](x)
            return F.adaptive_avg_pool3d(fmap, 2).flatten(1)

    return SpatialEncoder(backbone)


def evaluate_encoders(model_name: str, encoders: dict[str, Any], dm: Any, cfg: Any,
                      spatial: bool, keep_embeddings: bool = False) -> dict[str, Any]:
    """Score each encoder with probe v1 (CV 5x5 on train+val, test probe); ``main`` also with v2."""
    import warnings

    import torch

    from t5_ssl.data.roi3d import ROIEval3D
    from t5_ssl.evaluation.probe_v1 import score
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    # Probe features in full fp32, as the campaigns computed them.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    clean = ROIEval3D(int(cfg.data.roi_size))
    out: dict[str, Any] = {}
    for role, encoder in encoders.items():
        for attr in ("forward", "forward_tokens"):  # compiled entry points -> eager again
            if attr in encoder.__dict__:
                delattr(encoder, attr)
        encoder = encoder.float().to(memory_format=torch.contiguous_format)
        start = time.time()
        out[role] = score(encoder, dm, clean, role == "main")
        if role == "main":
            out[role]["protocol_v2"] = score_protocol_v2(encoder, dm, cfg, out[role]["_embeddings"])
            if not keep_embeddings:
                out[role].pop("_embeddings")
        if spatial:
            spatial_scores = score(spatial_encoder(encoder), dm, clean)
            out[role]["spatial"] = {k: v for k, v in spatial_scores.items()
                                    if k not in ("subjects", "labels")}
        out[role]["eval_minutes"] = (time.time() - start) / 60
        print(f"[eval] {model_name}/{role}: cv={mean3_cv(out[role]):.3f} "
              f"test={mean3_test(out[role]):.3f} erank={out[role]['effective_rank']:.1f}"
              + (f" | spatial cv={mean3_cv(out[role]['spatial']):.3f} "
                 f"test={mean3_test(out[role]['spatial']):.3f}" if spatial else ""), flush=True)
    return out


def score_protocol_v2(encoder: Any, dm: Any, cfg: Any, embeddings: dict) -> dict[str, Any]:
    """Score the encoder with probe protocol v2 (``t5_ssl.evaluation.protocol_v2``).

    Representative CV with the SSL-only AD/HC subjects of train, out-of-sample F1
    thresholds, AUPR/F1 at the cohort prevalence.

    Adds the pool embeddings to ``embeddings`` (saved as ``pool_*`` with --save-embeddings).
    """
    import numpy as np
    import torch

    from t5_ssl.data.roi3d import ROIEval3D
    from t5_ssl.evaluation import protocol_v2 as pp
    manifest = Path(cfg.data.manifest_path)
    records = pp.pool_records(manifest)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = encoder.to(device).eval()
    clean = ROIEval3D(int(cfg.data.roi_size))
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(records), 64):
            batch = torch.stack([clean(torch.from_numpy(
                np.asarray(dm.cubes[r["idx"]], dtype=np.float32))) for r in records[start:start + 64]])
            chunks.append(encoder(batch.to(device)).float().cpu().numpy())
    dim = embeddings["cv"][0].shape[1]
    x_pool = np.concatenate(chunks) if chunks else np.zeros((0, dim), np.float32)
    embeddings["pool"] = (x_pool, np.array([r["label"] for r in records], dtype=int),
                          [r["subject"] for r in records])
    ages = pp.session_ages(manifest)
    extra = {k: v for k, v in embeddings.items() if k not in ("cv", "train", "val", "test", "pool")}
    doc = pp.evaluate(embeddings["cv"], embeddings["pool"], embeddings["test"], ages, extra)
    r, t = doc["cv_rep"], doc["test"]
    print(f"[eval v2] rep CV AUROC {r['auc']:.3f} AUPR {r['aupr']:.3f} F1 {r['f1_macro']:.3f} "
          f"mean3 {r['mean3']:.3f} | test {t['auc']:.3f}/{t['aupr']:.3f}/{t['f1_macro']:.3f}",
          flush=True)
    return doc


def mean3_cv(doc: dict) -> float:
    m = doc["cv"]["clean"]["metrics"]
    return (m["auc"] + m["f1"] + m["aupr"]) / 3


def mean3_test(doc: dict) -> float:
    t = doc["test"]
    return (t["test_auc"] + t["test_f1_macro"] + t["test_average_precision"]) / 3
