"""Training-time representation-health diagnostics.

Catches an SSL objective that is minimizing its pretext loss by collapsing
onto a handful of directions, or by recovering a nuisance variable (age, sex,
slice position) instead of disease-relevant anatomy, long before a downstream
frozen AD/HC probe would reveal it. Every check runs on a held-out validation
subset with the *frozen* encoder as-is; it never sees or uses clinical labels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from t5_ssl.utils.representation import (
    ad_hc_centroid_distance,
    binary_association,
    continuous_association,
    effective_rank,
    feature_std,
    mean_pairwise_cosine_similarity,
    positive_vs_random_similarity,
    retrieval_accuracy,
)


class RepresentationHealthCallback(L.Callback):
    """Log collapse and nuisance-recovery diagnostics on a fixed validation subset.

    Parameters
    ----------
    main_dataset:
        Validation dataset yielding ``{"image": view_or_views, "subject": str, ...}``.
        When ``image`` is a pair of views, positive-vs-random similarity is also
        measured; subjects with more than one session in this dataset drive the
        same-subject session-retrieval check.
    output_dir:
        Directory to append ``representation_health.json`` (one record per check).
    neighbor_dataset, neighbor_group_ids:
        Optional dataset of paired nearby-slice records (single deterministic
        view each) plus a group id per item (equal for the two slices of the
        same record) used for the neighbor-slice-retrieval check.
    ages, sexes:
        Optional ``subject -> value`` maps used only to *diagnose* whether the
        embedding is predictable from a nuisance variable; never used to train.
    groups:
        Optional ``subject -> "AD"/"HC"/...`` map used only to log the AD/HC
        centroid distance; never used to train or to weight sampling.

    """

    def __init__(
        self, main_dataset: Dataset, output_dir: str | Path, *,
        every_n_epochs: int = 1, batch_size: int = 16,
        neighbor_dataset: Dataset | None = None, neighbor_group_ids: list[str] | None = None,
        session_dataset: Dataset | None = None,
        ages: dict[str, float] | None = None, sexes: dict[str, str] | None = None,
        groups: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        self.main_dataset = main_dataset
        self.output_dir = Path(output_dir)
        self.every_n_epochs = max(1, every_n_epochs)
        self.batch_size = batch_size
        self.neighbor_dataset = neighbor_dataset
        self.neighbor_group_ids = neighbor_group_ids
        self.session_dataset = session_dataset
        self.ages = ages or {}
        self.sexes = sexes or {}
        self.groups = groups or {}
        self.history: list[dict[str, Any]] = []

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Run every ``every_n_epochs`` and append the result to the health log."""
        if trainer.sanity_checking or trainer.current_epoch % self.every_n_epochs != 0:
            return
        record = self._measure(pl_module)
        record["epoch"] = trainer.current_epoch
        record["global_step"] = trainer.global_step
        self.history.append(record)
        for name, value in record.items():
            if isinstance(value, int | float) and np.isfinite(value):
                pl_module.log(f"health/{name}", float(value), on_step=False, on_epoch=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "representation_health.json").write_text(
            json.dumps(self.history, indent=2), encoding="utf-8")

    @torch.no_grad()
    def _embed(self, pl_module: L.LightningModule, dataset: Dataset) -> tuple[np.ndarray, list[str]]:
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        device = next(pl_module.backbone.parameters()).device
        embeddings, subjects = [], []
        for batch in loader:
            image = batch["image"][0] if isinstance(batch["image"], list | tuple) else batch["image"]
            embeddings.append(pl_module.backbone(image.to(device)).float().cpu().numpy())
            subjects.extend(batch["subject"])
        return np.concatenate(embeddings), subjects

    @torch.no_grad()
    def _measure(self, pl_module: L.LightningModule) -> dict[str, Any]:
        backbone = pl_module.backbone
        was_training = backbone.training
        backbone.eval()
        device = next(backbone.parameters()).device
        loader = DataLoader(self.main_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        embeddings, subjects = [], []
        pair_h1, pair_h2 = [], []
        for batch in loader:
            image = batch["image"]
            if isinstance(image, list | tuple):
                h1 = backbone(image[0].to(device))
                h2 = backbone(image[1].to(device))
                pair_h1.append(h1.float().cpu())
                pair_h2.append(h2.float().cpu())
                embeddings.append(h1.float().cpu().numpy())
            else:
                embeddings.append(backbone(image.to(device)).float().cpu().numpy())
            subjects.extend(batch["subject"])
        if was_training:
            backbone.train()
        embeddings = np.concatenate(embeddings)

        result: dict[str, Any] = {
            "effective_rank": effective_rank(embeddings),
            "mean_feature_std": feature_std(embeddings),
            "mean_cosine_similarity_diff_samples": mean_pairwise_cosine_similarity(embeddings),
        }
        if self.groups:
            result["ad_hc_centroid_distance"] = ad_hc_centroid_distance(
                embeddings, subjects, self.groups)
        if pair_h1:
            result.update(positive_vs_random_similarity(torch.cat(pair_h1), torch.cat(pair_h2)))

        if self.session_dataset is not None:
            session_embeddings, session_subjects = self._embed(pl_module, self.session_dataset)
            session = retrieval_accuracy(session_embeddings, session_subjects)
        else:
            # Falls back to whatever repeats exist in main_dataset; a
            # primary-session-only val split will report 0 eligible subjects.
            session = retrieval_accuracy(embeddings, subjects)
        result["session_retrieval_top1"] = session["top1_accuracy"]
        result["session_retrieval_chance"] = session["chance_accuracy"]
        result["session_retrieval_n_eligible"] = session["n_eligible"]

        if self.neighbor_dataset is not None and self.neighbor_group_ids is not None:
            neighbor_embeddings, _ = self._embed(pl_module, self.neighbor_dataset)
            neighbor = retrieval_accuracy(neighbor_embeddings, self.neighbor_group_ids)
            result["neighbor_slice_retrieval_top1"] = neighbor["top1_accuracy"]
            result["neighbor_slice_retrieval_chance"] = neighbor["chance_accuracy"]

        if self.ages:
            paired = [(s, self.ages[s]) for s in subjects if s in self.ages]
            if len(paired) >= 5:
                idx = [i for i, s in enumerate(subjects) if s in self.ages]
                ages = np.array([self.ages[subjects[i]] for i in idx], dtype=np.float32)
                result["age_r2"] = continuous_association(embeddings[idx], ages)
        if self.sexes:
            idx = [i for i, s in enumerate(subjects) if s in self.sexes]
            if len(idx) >= 5:
                labels = np.array([1 if self.sexes[subjects[i]].lower().startswith("f") else 0
                                   for i in idx])
                sex_assoc = binary_association(embeddings[idx], labels)
                result["sex_accuracy"] = sex_assoc["accuracy"]
                result["sex_majority_baseline"] = sex_assoc["majority_baseline"]

        return result
