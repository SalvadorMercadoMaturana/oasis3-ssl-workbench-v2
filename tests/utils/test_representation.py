"""Unit tests for representation-health diagnostics."""

from __future__ import annotations

import numpy as np
import torch

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


def test_effective_rank_collapsed_embeddings_is_near_one():
    """A rank-1 embedding cloud must report an effective rank close to 1."""
    rng = np.random.default_rng(0)
    direction = rng.normal(size=(1, 32))
    embeddings = rng.normal(size=(200, 1)) @ direction
    assert effective_rank(embeddings) < 1.5


def test_effective_rank_isotropic_embeddings_is_near_full():
    """Isotropic Gaussian embeddings should spread across most directions."""
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(500, 32))
    assert effective_rank(embeddings) > 25.0


def test_feature_std_zero_for_constant_embeddings():
    """A constant embedding (full collapse) has zero per-dimension spread."""
    embeddings = np.ones((10, 8))
    assert feature_std(embeddings) == 0.0


def test_positive_vs_random_similarity_detects_collapse():
    """Identical positive and negative pairs (collapse) should erase the similarity gap."""
    z = torch.ones(16, 8)
    result = positive_vs_random_similarity(z, z)
    assert result["positive_similarity"] > 0.99
    assert abs(result["similarity_gap"]) < 1e-4


def test_positive_vs_random_similarity_healthy_case():
    """Matched pairs should score higher than a shuffled control on average."""
    torch.manual_seed(0)
    base = torch.randn(64, 16)
    z1 = base + 0.01 * torch.randn_like(base)
    z2 = base + 0.01 * torch.randn_like(base)
    result = positive_vs_random_similarity(z1, z2)
    assert result["positive_similarity"] > result["random_similarity"]


def test_mean_pairwise_cosine_similarity_collapsed_is_near_one():
    """Identical embeddings (full collapse) must report similarity close to 1.0."""
    embeddings = np.ones((16, 8))
    assert mean_pairwise_cosine_similarity(embeddings) > 0.999


def test_mean_pairwise_cosine_similarity_orthogonal_is_near_zero():
    """Random high-dimensional isotropic embeddings should be nearly uncorrelated."""
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(200, 64))
    assert abs(mean_pairwise_cosine_similarity(embeddings)) < 0.1


def test_retrieval_accuracy_perfect_pairs():
    """Two embeddings per group, near-identical within group, should retrieve at 100%."""
    rng = np.random.default_rng(0)
    groups = [f"g{i}" for i in range(10) for _ in range(2)]
    base = rng.normal(size=(10, 8))
    embeddings = np.repeat(base, 2, axis=0) + rng.normal(scale=1e-4, size=(20, 8))
    result = retrieval_accuracy(embeddings, groups)
    assert result["n_eligible"] == 20
    assert result["top1_accuracy"] == 1.0
    # 10 pair-groups of size 2: any single specific partner among 19 others.
    assert abs(result["chance_accuracy"] - 1 / 19) < 1e-9


def test_retrieval_accuracy_chance_scales_with_group_size():
    """A single large group should raise the chance level well above pair-chance."""
    rng = np.random.default_rng(0)
    groups = ["big"] * 10
    embeddings = rng.normal(size=(10, 6))
    result = retrieval_accuracy(embeddings, groups)
    # Every item's 9 same-group partners are the only other candidates: chance = 1.0.
    assert abs(result["chance_accuracy"] - 1.0) < 1e-9


def test_retrieval_accuracy_handles_no_eligible_groups():
    """Singleton groups (no repeats) cannot support retrieval and must report NaN cleanly."""
    embeddings = np.random.default_rng(0).normal(size=(5, 4))
    groups = [f"g{i}" for i in range(5)]
    result = retrieval_accuracy(embeddings, groups)
    assert result["n_eligible"] == 0
    assert np.isnan(result["top1_accuracy"])


def test_continuous_association_recovers_a_linear_signal():
    """A target that is linear in the embeddings should score a high R^2."""
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(60, 5))
    weights = rng.normal(size=5)
    target = embeddings @ weights
    r2 = continuous_association(embeddings, target)
    assert r2 > 0.5


def test_continuous_association_near_zero_for_unrelated_target():
    """Independent noise target should score near/at-or-below zero R^2."""
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(60, 5))
    target = rng.normal(size=60)
    r2 = continuous_association(embeddings, target)
    assert r2 < 0.3


def test_ad_hc_centroid_distance_separated_groups():
    """Well-separated AD/HC clusters must report a large positive distance."""
    rng = np.random.default_rng(0)
    subjects = [f"ad{i}" for i in range(10)] + [f"hc{i}" for i in range(10)]
    groups = {s: ("AD" if s.startswith("ad") else "HC") for s in subjects}
    embeddings = np.concatenate([
        rng.normal(loc=5.0, scale=0.1, size=(10, 8)),
        rng.normal(loc=-5.0, scale=0.1, size=(10, 8)),
    ])
    distance = ad_hc_centroid_distance(embeddings, subjects, groups)
    assert distance > 5.0


def test_ad_hc_centroid_distance_nan_when_a_group_is_missing():
    """Fewer than 2 subjects in either group cannot support a centroid estimate."""
    embeddings = np.random.default_rng(0).normal(size=(5, 4))
    subjects = ["ad0", "hc0", "hc1", "other0", "other1"]
    groups = {"ad0": "AD", "hc0": "HC", "hc1": "HC"}
    assert np.isnan(ad_hc_centroid_distance(embeddings, subjects, groups))


def test_binary_association_reports_majority_baseline():
    """The majority-class baseline must reflect the true class balance."""
    embeddings = np.random.default_rng(0).normal(size=(30, 4))
    target = np.array([0] * 20 + [1] * 10)
    result = binary_association(embeddings, target)
    assert abs(result["majority_baseline"] - 2 / 3) < 1e-6
