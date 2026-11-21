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
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # torch is imported inside the functions
    import torch

Embeddings = tuple[np.ndarray, np.ndarray, list[str]]  # (x, y, subjects), subjects sorted


def train_val_embeddings(backbone: torch.nn.Module, dm: Any, transform: Any) -> Embeddings:
    """Mean embedding per unambiguous AD/HC subject of train+val (every session)."""
    features: dict[str, list[np.ndarray]] = defaultdict(list)
    for split in ("train", "val"):
        _embed_into(features, backbone, _labeled(dm.eval_dataset(split, transform), dm), 64)
    return _per_subject(features, lambda s: int(dm.subject_labels[s] == "AD"))


def split_embeddings(backbone: torch.nn.Module, dm: Any, transform: Any, split: str
                     ) -> Embeddings:
    """Mean embedding per unambiguous AD/HC subject of one split (every session)."""
    features: dict[str, list[np.ndarray]] = defaultdict(list)
    _embed_into(features, backbone, _labeled(dm.eval_dataset(split, transform), dm), 32)
    return _per_subject(features, lambda s: int(dm.subject_labels[s] == "AD"))


def external_embeddings(encoder: torch.nn.Module, external: dict, clean: Any) -> Embeddings:
    """Mean embedding per subject of an external set (t5_ssl.data.roi3d.load_external)."""
    from t5_ssl.data.roi3d import ROI3DDataset
    features: dict[str, list[np.ndarray]] = defaultdict(list)
    _embed_into(features, encoder, ROI3DDataset(external["records"], external["cubes"], clean), 32)
    return _per_subject(features, lambda s: external["labels"][s])


def concat_splits(*splits: tuple) -> tuple:
    """Join (x, y, subjects) splits; subjects stay sorted within each part."""
    return (np.concatenate([s[0] for s in splits]), np.concatenate([s[1] for s in splits]),
            [subject for s in splits for subject in s[2]])


# ---------------------------------------------------------------- helpers
def _labeled(dataset: Any, dm: Any) -> Any:
    """Keep the sessions of ``dataset`` whose subject is an unambiguous AD/HC subject of ``dm``."""
    from torch.utils.data import Subset
    eligible = [i for i, (record, _) in enumerate(dataset.indices)
                if dataset.records[record]["subject"] in dm.subject_labels]
    return Subset(dataset, eligible)


def _embed_into(features: dict[str, list[np.ndarray]], encoder: torch.nn.Module, dataset: Any,
                batch_size: int) -> None:
    """Append the embedding of every item of ``dataset`` to its subject's list, in order.

    The batch sizes (64 for train+val, 32 otherwise) are the ones the campaigns used.
    """
    import torch
    from torch.utils.data import DataLoader
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = encoder.to(device).eval()
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            out = encoder(batch["image"].to(device)).float().cpu().numpy()
            for subject, vector in zip(batch["subject"], out, strict=True):
                features[subject].append(vector)


def _per_subject(features: dict[str, list[np.ndarray]], label: Callable[[str], int]
                 ) -> Embeddings:
    subjects = sorted(features)
    return (np.stack([np.mean(features[s], axis=0) for s in subjects]),
            np.array([label(s) for s in subjects]), subjects)
