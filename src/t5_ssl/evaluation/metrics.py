"""Scores of probe predictions: AUROC, AUPR, F1-macro, their mean (``mean3``) and more.

The campaigns computed these quantities in several scripts; they are collected here
verbatim, so every report still reproduces its published numbers exactly:

- ``scores`` / ``mean3``: sklearn AUROC, F1-macro at 0.5 and AUPR, and their mean.
- ``fast_mean3``: the same mean3 in plain numpy (bootstrap loops call it thousands of times).
- ``fast_auc`` / ``fast_ap``: numpy AUROC (Mann-Whitney with average ranks) and average
  precision with sklearn's definition (``protocol_v2.auc`` / ``protocol_v2.aupr`` use the
  same formulas).
- ``threshold_metrics`` / ``held_out_metrics``: confusion-matrix metrics at one threshold
  and the scorecard of one held-out set (campaign protocol, ``probe_v1``).
- ``effective_rank``: exp(entropy) of the normalized singular values of centered
  embeddings (``identity.effective_rank`` uses the variance spectrum instead).

Moved from ``scripts/legacy/report_aug_screen.py`` (scores, mean3, fast_mean3),
``scripts/5_reports/report_bb50.py`` (fast_auc, fast_ap),
``scripts/legacy/evaluate_frozen_probe.py`` (threshold_metrics, held_out_metrics) and
``scripts/legacy/run_ssl3d_v2.py`` (effective_rank).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)


def scores(labels: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    """AUROC, F1-macro at 0.5 and AUPR with sklearn (``mean3`` averages them)."""
    return {"auc": float(roc_auc_score(labels, probs)),
            "f1": float(f1_score(labels, probs >= 0.5, average="macro")),
            "aupr": float(average_precision_score(labels, probs))}


def mean3(values: dict[str, float]) -> float:
    return (values["auc"] + values["f1"] + values["aupr"]) / 3


def fast_mean3(labels: np.ndarray, probs: np.ndarray) -> float:
    """Compute ``mean3(scores(...))`` in plain numpy, ties handled as sklearn does.

    sklearn's per-call overhead makes the 2000-replicate bootstrap take minutes.
    """
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    auc = (rankdata(probs)[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    order = np.argsort(-probs, kind="mergesort")
    sorted_probs, sorted_labels = probs[order], labels[order]
    last_of_tie = np.r_[np.flatnonzero(np.diff(sorted_probs)), len(probs) - 1]
    tp = np.cumsum(sorted_labels)[last_of_tie]
    fp = last_of_tie + 1 - tp
    recall = tp / n_pos
    aupr = float(np.sum(np.diff(np.r_[0.0, recall]) * tp / (tp + fp)))
    pred = probs >= 0.5
    tp1 = np.sum(pred & (labels == 1))
    fp1 = np.sum(pred & (labels == 0))
    fn1 = n_pos - tp1
    tn1 = n_neg - fp1
    f1 = ((2 * tp1 / (2 * tp1 + fp1 + fn1) if tp1 + fp1 + fn1 else 0.0)
          + (2 * tn1 / (2 * tn1 + fn1 + fp1) if tn1 + fn1 + fp1 else 0.0)) / 2
    return float((auc + f1 + aupr) / 3)


def fast_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """AUROC from average ranks (Mann-Whitney), as sklearn with ties."""
    ranks = rankdata(scores)
    pos = labels == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def fast_ap(labels: np.ndarray, scores: np.ndarray) -> float:
    """Average precision with sklearn's definition (one step per distinct score)."""
    order = np.argsort(-scores, kind="mergesort")
    s, y = scores[order], labels[order]
    last = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]  # end of each tied block
    tp = np.cumsum(y)[last]
    precision = tp / (last + 1)
    recall = tp / max(1, y.sum())
    return float(np.sum(np.diff(np.r_[0, recall]) * precision))


def threshold_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict:
    """Confusion-based metrics of AD (positive) vs HC at one score threshold."""
    predicted = scores >= threshold
    tp = int(np.sum(predicted & (labels == 1)))
    fp = int(np.sum(predicted & (labels == 0)))
    fn = int(np.sum(~predicted & (labels == 1)))
    tn = int(np.sum(~predicted & (labels == 0)))
    ratio = lambda a, b: float(a / b) if b else float("nan")  # noqa: E731
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": float(accuracy_score(labels, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
            "f1_macro": float(f1_score(labels, predicted, average="macro")),
            "f1_ad": float(f1_score(labels, predicted, zero_division=0)),
            "sensitivity": ratio(tp, tp + fn), "specificity": ratio(tn, tn + fp),
            "ppv": ratio(tp, tp + fp), "npv": ratio(tn, tn + fn),
            "mcc": ratio(tp * tn - fp * fn, denominator)}


def held_out_metrics(labels: np.ndarray, scores: np.ndarray, subjects: list[str]) -> dict:
    """Rank and threshold (0.5) metrics of one held-out set, with its predictions."""
    return {"subjects": len(labels), "ad": int(labels.sum()),
            "prevalence": float(labels.mean()),
            "auc": float(roc_auc_score(labels, scores)),
            "average_precision": float(average_precision_score(labels, scores)),
            "brier_score": float(brier_score_loss(labels, scores)),
            **threshold_metrics(labels, scores),
            "predictions": [{"subject": s, "AD": int(a), "score": float(p)}
                            for s, a, p in zip(subjects, labels, scores, strict=True)]}


def effective_rank(x: np.ndarray) -> float:
    """exp(entropy) of the normalized singular values of the centered embeddings."""
    s = np.linalg.svd(x - x.mean(axis=0), compute_uv=False)
    p = s[s > 0] / s.sum()
    return float(np.exp(-(p * np.log(p)).sum()))
