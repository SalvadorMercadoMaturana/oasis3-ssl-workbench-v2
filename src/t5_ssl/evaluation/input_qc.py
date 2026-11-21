"""Input QC of the deployed model: reject scans whose intensity is outside OASIS3's range.

The statistic is the mean intensity of channel 0 over the evaluated (center) cube of
each extracted ROI cube (``qc_intensity``). ``qc_range`` is every OASIS3 session's value
in a cache, widened by ``QC_MARGIN``; ``predict_ad.py`` rejects scans outside it (45 ADNI
scans, all AD, come with another intensity normalization: they are not anatomy).

Moved from ``scripts/6_deploy/predict_ad.py`` (QC_MARGIN, qc_intensity) and
``scripts/6_deploy/build_deploy_model.py`` (cache_intensity, qc_range). numpy only (no torch).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

# The project root (= t5_ssl.data.paths.PROJECT_ROOT, not imported: that package pulls torch).
ROOT = Path(__file__).resolve().parents[3]


QC_MARGIN = 0.05


def qc_intensity(cubes: np.ndarray, roi_size: int) -> np.ndarray:
    """Mean intensity of channel 0 over the evaluated (center) cube of each extracted cube.

    ``cubes`` is (n, channels, e, e, e) as cut by ``roi3d.crop_cubes`` from a
    ``normalize_volume``d MRI; the model accepts only scans inside OASIS3's range.
    """
    c = (cubes.shape[-1] - roi_size) // 2
    return cubes[:, 0, c:c + roi_size, c:c + roi_size, c:c + roi_size].mean(axis=(1, 2, 3))


def cache_intensity(cache: str, keys: list[str]) -> np.ndarray:
    """Return the QC statistic (``qc_intensity``) of cached sessions."""
    meta = json.loads((ROOT / "data/roi3d" / f"{cache}.json").read_text(encoding="utf-8"))
    cubes = np.load(ROOT / "data/roi3d" / f"{cache}.npy", mmap_mode="r")
    row = {k: i for i, k in enumerate(meta["idx"])}
    return np.array([qc_intensity(np.asarray(cubes[row[k]][None], dtype=np.float32), 32)[0]
                     for k in keys])


def qc_range(cache: str) -> dict:
    """Return the accepted input range: every OASIS3 session's QC statistic, widened.

    Defined from training data only. It exists because 45 ADNI scans (all AD) come with
    another intensity normalization, far outside OASIS3; they are not anatomy.
    """
    meta = json.loads((ROOT / "data/roi3d" / f"{cache}.json").read_text(encoding="utf-8"))
    with (ROOT / "data/dementia/mni-dataset.csv").open(encoding="utf-8") as stream:
        study = {r["idx"]: r["study_name"].strip().upper() for r in csv.DictReader(stream)}
    values = cache_intensity(cache, [k for k in meta["idx"] if study.get(k) == "OASIS3"])
    return {"statistic": "mean intensity of channel 0 over the evaluated (center) cube",
            "lo": float(values.min() - QC_MARGIN), "hi": float(values.max() + QC_MARGIN),
            "oasis3_sessions": len(values)}
