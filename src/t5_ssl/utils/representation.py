"""Representation-health diagnostics shared by training callbacks and probes.

These metrics catch objectives that minimize their pretext loss by collapsing
onto a few directions (or onto nuisance variables such as age) well before a
downstream AD/HC probe would reveal it.
"""

from __future__ import annotations

import numpy as np
import torch


def effective_rank(embeddings: np.ndarray) -> float:
    """Return the exponential entropy of the normalized singular-value spectrum.

    A random encoder spreads variance across most feature directions (effective
    rank close to ``min(n_samples, n_features)``); a collapsed encoder concentrates
    it in a handful of directions (effective rank close to 1).
    """
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    spectrum = singular_values**2
    total = float(spectrum.sum())
    if total <= 0:
        return 0.0
    spectrum = spectrum / total
    positive = spectrum[spectrum > 0]
    return float(np.exp(-np.sum(positive * np.log(positive))))


def feature_std(embeddings: np.ndarray) -> float:
    """Return the mean per-dimension standard deviation after centering."""
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    return float(centered.std(axis=0).mean())


def mean_pairwise_cosine_similarity(embeddings: np.ndarray) -> float:
    """Return the mean cosine similarity across all distinct-sample pairs.

    Complements :func:`positive_vs_random_similarity` (which only compares one
    shuffled pairing) with the full off-diagonal average of the batch's
    similarity matrix. A collapsed encoder pushes this toward 1.0 because
    every embedding points in nearly the same direction.
    """
    n = embeddings.shape[0]
    if n < 2:
        return float("nan")
    normed = embeddings / np.clip(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12, None)
    similarity = normed @ normed.T
    off_diag_sum = float(similarity.sum() - np.trace(similarity))
    return off_diag_sum / (n * (n - 1))


def ad_hc_centroid_distance(embeddings: np.ndarray, subjects: list[str],
                            groups: dict[str, str]) -> float:
    """Return the Euclidean distance between the AD and HC mean embedding centroids.

    A sampler that rebalances AD/HC per batch should let both centroids move
    apart together as training progresses, rather than the space expanding
    only around whichever class dominates the raw session counts. This is a
    diagnostic only: it never feeds back into any loss or sampling decision.
    """
    ad_idx = [i for i, s in enumerate(subjects) if groups.get(s) == "AD"]
    hc_idx = [i for i, s in enumerate(subjects) if groups.get(s) == "HC"]
    if len(ad_idx) < 2 or len(hc_idx) < 2:
        return float("nan")
    ad_centroid = embeddings[ad_idx].mean(axis=0)
    hc_centroid = embeddings[hc_idx].mean(axis=0)
    return float(np.linalg.norm(ad_centroid - hc_centroid))


def positive_vs_random_similarity(z1: torch.Tensor, z2: torch.Tensor) -> dict[str, float]:
    """Compare matched-pair cosine similarity to a shuffled (random-pair) control.

    A healthy invariance objective keeps positives well above random pairs;
    collapse drags both toward 1.0 because every embedding looks alike.
    """
    a = torch.nn.functional.normalize(z1.detach().float(), dim=-1)
    b = torch.nn.functional.normalize(z2.detach().float(), dim=-1)
    positive = (a * b).sum(-1)
    if a.shape[0] < 2:
        random_sim = positive.new_zeros(0)
    else:
        shift = torch.randint(1, a.shape[0], (1,)).item()
        random_sim = (a * b.roll(shifts=shift, dims=0)).sum(-1)
    return {
        "positive_similarity": float(positive.mean().item()),
        "random_similarity": float(random_sim.mean().item()) if random_sim.numel() else float("nan"),
        "similarity_gap": float((positive.mean() - random_sim.mean()).item())
        if random_sim.numel() else float("nan"),
    }


def retrieval_accuracy(embeddings: np.ndarray, group_ids: list[str]) -> dict[str, float]:
    """Top-1 nearest-neighbor accuracy for recovering same-group embeddings.

    Used both for "another session of the same subject" and "a neighboring
    slice" recovery: ``group_ids`` carries the subject (or subject+slice-band)
    identity, and a healthy encoder should retrieve a same-group neighbor more
    often than chance.
    """
    groups = np.asarray(group_ids)
    unique, counts = np.unique(groups, return_counts=True)
    eligible_groups = unique[counts >= 2]
    eligible = np.isin(groups, eligible_groups)
    n_eligible = int(eligible.sum())
    if n_eligible < 2:
        return {"n_eligible": 0, "top1_accuracy": float("nan"), "chance_accuracy": float("nan")}
    normed = embeddings / np.clip(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12, None)
    similarity = normed @ normed.T
    np.fill_diagonal(similarity, -np.inf)
    nearest = similarity.argmax(axis=1)
    correct = groups[nearest] == groups
    n_total = len(groups)
    # Expected top-1 accuracy for a uniformly random nearest-neighbor pick: an
    # eligible item in a group of size c has (c-1) same-group partners among the
    # other (n_total-1) candidates, averaged over eligible items (not groups).
    eligible_counts = counts[counts >= 2]
    chance = float((eligible_counts * (eligible_counts - 1)).sum()
                   / (max(n_total - 1, 1) * n_eligible))
    return {
        "n_eligible": n_eligible,
        "top1_accuracy": float(correct[eligible].mean()),
        "chance_accuracy": chance,
    }


def continuous_association(embeddings: np.ndarray, target: np.ndarray) -> float:
    """Cross-validated R^2 of a linear ridge probe predicting a continuous target.

    High R^2 for age (etc.) warns that the encoder may be recovering a nuisance
    variable instead of, or in addition to, disease-relevant anatomy.
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if len(target) < 5:
        return float("nan")
    n_splits = min(5, len(target))
    probe = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    scores = cross_val_score(probe, embeddings, target, cv=KFold(n_splits, shuffle=True, random_state=0),
                             scoring="r2")
    return float(np.mean(scores))


def binary_association(embeddings: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Cross-validated accuracy of a linear probe predicting a binary nuisance target."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    target = np.asarray(target)
    if len(target) < 5 or len(set(target.tolist())) < 2:
        return {"accuracy": float("nan"), "majority_baseline": float("nan")}
    n_splits = min(5, int(np.bincount(target.astype(int)).min()))
    if n_splits < 2:
        return {"accuracy": float("nan"), "majority_baseline": float("nan")}
    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    scores = cross_val_score(probe, embeddings, target, cv=KFold(n_splits, shuffle=True, random_state=0),
                             scoring="accuracy")
    majority = max(np.mean(target == 0), np.mean(target == 1))
    return {"accuracy": float(np.mean(scores)), "majority_baseline": float(majority)}
