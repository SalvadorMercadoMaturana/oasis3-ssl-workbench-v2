"""Subject embeddings of a frozen encoder: one vector per subject, the mean over its sessions.

Every probe of the workbench sees one vector per subject: the encoder embeds the clean
central cube (``t5_ssl.data.roi3d.ROIEval3D``) of each session and the vectors of one
subject are averaged. Only unambiguous AD/HC subjects (``dm.subject_labels``) are kept,
so SSL-only subjects (manifest ``labeled=0``) never reach a probe through these functions.

- ``train_val_embeddings``: train + val subjects together (the CV population of v1).
- ``split_embeddings``: one split (``train``, ``val``, ``test`` or an extra ``test_<name>``).
- ``external_embeddings``: an external cohort (``roi3d.load_external``), e.g. ADNI.
- ``concat_splits``: join ``(x, y, subjects)`` tuples.

Moved from ``scripts/legacy/screen_aug_equal.py`` (``embed``, renamed ``train_val_embeddings``),
``scripts/legacy/run_roi3d.py`` (split_embeddings), ``scripts/legacy/run_ssl3d_v2.py``
(external_embeddings) and ``scripts/legacy/evaluate_frozen_probe.py`` (concat_splits).
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # torch is imported inside the functions
    import torch


def train_val_embeddings(backbone: torch.nn.Module, dm: Any, transform: Any
                         ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Mean central-slice embedding per unambiguous AD/HC subject of train+val."""
    import torch
    from torch.utils.data import DataLoader, Subset
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = backbone.to(device).eval()
    features: dict[str, list[np.ndarray]] = defaultdict(list)
    for split in ("train", "val"):
        dataset = dm.eval_dataset(split, transform)
        eligible = [i for i, (record, _) in enumerate(dataset.indices)
                    if dataset.records[record]["subject"] in dm.subject_labels]
        loader = DataLoader(Subset(dataset, eligible), batch_size=64, shuffle=False)
        with torch.inference_mode():
            for batch in loader:
                out = backbone(batch["image"].to(device)).float().cpu().numpy()
                for subject, vector in zip(batch["subject"], out, strict=True):
                    features[subject].append(vector)
    subjects = sorted(features)
    x = np.stack([np.mean(features[s], axis=0) for s in subjects])
    y = np.array([int(dm.subject_labels[s] == "AD") for s in subjects])
    return x, y, subjects


def split_embeddings(backbone: torch.nn.Module, dm: Any, transform: Any, split: str
                     ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Mean embedding per unambiguous AD/HC subject of one split (every session)."""
    import torch
    from torch.utils.data import DataLoader, Subset
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = backbone.to(device).eval()
    dataset = dm.eval_dataset(split, transform)
    eligible = [i for i, (record, _) in enumerate(dataset.indices)
                if dataset.records[record]["subject"] in dm.subject_labels]
    features: dict[str, list[np.ndarray]] = defaultdict(list)
    with torch.inference_mode():
        for batch in DataLoader(Subset(dataset, eligible), batch_size=32, shuffle=False):
            out = backbone(batch["image"].to(device)).float().cpu().numpy()
            for subject, vector in zip(batch["subject"], out, strict=True):
                features[subject].append(vector)
    subjects = sorted(features)
    return (np.stack([np.mean(features[s], axis=0) for s in subjects]),
            np.array([int(dm.subject_labels[s] == "AD") for s in subjects]), subjects)


def external_embeddings(encoder: torch.nn.Module, external: dict, clean: Any
                        ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Mean embedding per subject of an external set (t5_ssl.data.roi3d.load_external)."""
    from collections import defaultdict

    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from t5_ssl.data.roi3d import ROI3DDataset
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = encoder.to(device).eval()
    features: dict[str, list[np.ndarray]] = defaultdict(list)
    dataset = ROI3DDataset(external["records"], external["cubes"], clean)
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=32, shuffle=False):
            out = encoder(batch["image"].to(device)).float().cpu().numpy()
            for subject, vector in zip(batch["subject"], out, strict=True):
                features[subject].append(vector)
    subjects = sorted(features)
    return (np.stack([np.mean(features[s], axis=0) for s in subjects]),
            np.array([external["labels"][s] for s in subjects]), subjects)


def concat_splits(*splits: tuple) -> tuple:
    """Join (x, y, subjects) splits; subjects stay sorted within each part."""
    return (np.concatenate([s[0] for s in splits]), np.concatenate([s[1] for s in splits]),
            [subject for s in splits for subject in s[2]])
