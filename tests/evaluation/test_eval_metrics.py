"""t5_ssl.evaluation: the numpy fast paths agree with sklearn (ties included)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from t5_ssl.evaluation import metrics, protocol_v2


def _labels_and_scores(seed: int, ties: bool, n: int = 60) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    y[:2] = [0, 1]  # both classes present
    s = rng.random(n)
    return y, (np.round(s, 1) if ties else s)


@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("seed", range(5))
def test_fast_scores_match_sklearn(seed: int, ties: bool) -> None:
    y, s = _labels_and_scores(seed, ties)
    assert metrics.fast_auc(y, s) == pytest.approx(roc_auc_score(y, s))
    assert metrics.fast_ap(y, s) == pytest.approx(average_precision_score(y, s))
    assert protocol_v2.auc(y, s) == pytest.approx(roc_auc_score(y, s))
    assert protocol_v2.aupr(y, s) == pytest.approx(average_precision_score(y, s))
    assert metrics.fast_mean3(y, s) == pytest.approx(metrics.mean3(metrics.scores(y, s)))


def test_scores_and_mean3() -> None:
    y, s = np.array([0, 0, 1, 1]), np.array([0.1, 0.6, 0.4, 0.9])
    out = metrics.scores(y, s)
    assert out["auc"] == pytest.approx(0.75)
    assert out["f1"] == pytest.approx(f1_score(y, s >= 0.5, average="macro"))
    assert metrics.mean3(out) == pytest.approx((out["auc"] + out["f1"] + out["aupr"]) / 3)


def test_threshold_metrics_v1_and_v2_agree() -> None:
    y = np.array([1, 1, 1, 0, 0, 0, 0])
    s = np.array([0.9, 0.7, 0.2, 0.6, 0.1, 0.3, 0.4])
    v1 = metrics.threshold_metrics(y, s, 0.5)
    assert (v1["tp"], v1["fn"], v1["fp"], v1["tn"]) == (2, 1, 1, 3)
    assert v1["sensitivity"] == pytest.approx(2 / 3)
    assert v1["specificity"] == pytest.approx(3 / 4)
    assert v1["ppv"] == pytest.approx(2 / 3)
    v2 = protocol_v2.at_threshold(y, s, 0.5)
    for key in ("f1_macro", "f1_ad", "balanced_accuracy", "sensitivity", "specificity", "mcc"):
        assert v2[key] == pytest.approx(v1[key]), key


def test_best_f1_threshold_separates_separable_scores() -> None:
    y, s = np.array([0, 0, 0, 1, 1]), np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    threshold = protocol_v2.best_f1_threshold(y, s)
    assert protocol_v2.at_threshold(y, s, threshold)["f1_macro"] == 1.0


def test_auc_age_only_counts_close_pairs() -> None:
    y, s = _labels_and_scores(0, ties=False)
    same_age = np.full(len(y), 70.0)
    assert protocol_v2.auc_age(y, s, same_age) == pytest.approx(roc_auc_score(y, s))
    far_apart = np.where(y == 1, 90.0, 60.0)  # no AD-HC pair within the caliper
    assert np.isnan(protocol_v2.auc_age(y, s, far_apart))


def test_held_out_mean3() -> None:
    y, s = _labels_and_scores(1, ties=False)
    out = protocol_v2.held_out(y, s, 0.5)
    assert out["mean3"] == pytest.approx((out["auc"] + out["aupr"] + out["f1_macro"]) / 3)
    assert out["f1_macro_05"] == pytest.approx(f1_score(y, s >= 0.5, average="macro"))


def test_effective_rank_of_equal_spectrum() -> None:
    x = np.vstack([np.eye(4), -np.eye(4)])  # centered, four equal singular values
    assert metrics.effective_rank(x) == pytest.approx(4.0)


def test_pseudo_tests_are_reproducible_and_sized() -> None:
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(30, int), np.zeros(90, int)]
    s, age = rng.random(120) + 0.3 * y, rng.uniform(60, 90, 120)
    draws = protocol_v2.pseudo_tests(y, s, age, n_ad=10, n_hc=30, draws=50, seed=3)
    assert draws.shape == (50,)
    assert np.all((draws >= 0) & (draws <= 1))
    assert np.array_equal(draws, protocol_v2.pseudo_tests(y, s, age, 10, 30, draws=50, seed=3))
