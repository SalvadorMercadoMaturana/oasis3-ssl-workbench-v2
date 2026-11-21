"""The campaign probe protocol ("v1", 2026-09 .. 2026-11): the ``cv`` and ``test`` of result.json.

1. ``cv_probe``: 5 x repeated stratified 5-fold CV of a logistic probe (standardization +
   class-balanced L2 logistic regression, C in {0.01, 0.1, 1, 10} by inner 3-fold AUROC)
   over the train+val AD/HC subjects; metrics are averaged over the repeats.
2. ``evaluate``: a probe fit on train only, C chosen on the val AUROC, scored once on test
   (and on every extra held-out set with the same probe).
3. ``score``: both of the above for one frozen encoder, plus the effective rank of its
   train+val embeddings; ``keep_embeddings`` also returns the embeddings.

Model selection moved to ``protocol_v2`` on 2026-11-07 (its docstring explains why); v1 is
still computed for every run so that old and new results stay comparable.

Moved from ``scripts/legacy/screen_aug_equal.py`` (cv_probe),
``scripts/legacy/evaluate_frozen_probe.py`` (evaluate) and ``scripts/legacy/run_ssl3d_v2.py``
(score).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from t5_ssl.evaluation.metrics import effective_rank, held_out_metrics, scores, threshold_metrics

if TYPE_CHECKING:
    import torch


def cv_probe(x: np.ndarray, y: np.ndarray, variants: dict[str, np.ndarray],
             repeats: int = 5, folds: int = 5) -> dict[str, dict]:
    """Repeated stratified K-fold probe fit on clean `x`, scored on every variant."""
    oof = {name: np.zeros((repeats, len(y))) for name in variants}
    for repeat in range(repeats):
        for train_idx, test_idx in StratifiedKFold(folds, shuffle=True,
                                                   random_state=repeat).split(x, y):
            probe = make_pipeline(StandardScaler(), LogisticRegressionCV(
                Cs=[0.01, 0.1, 1.0, 10.0], cv=3, scoring="roc_auc",
                class_weight="balanced", max_iter=2000))
            probe.fit(x[train_idx], y[train_idx])
            for name, values in variants.items():
                oof[name][repeat, test_idx] = probe.predict_proba(values[test_idx])[:, 1]
    out = {}
    for name, predictions in oof.items():
        per_repeat = [scores(y, predictions[r]) for r in range(repeats)]
        out[name] = {"metrics": {k: float(np.mean([m[k] for m in per_repeat]))
                                 for k in per_repeat[0]},
                     "oof_mean": predictions.mean(axis=0).tolist()}
    return out


def evaluate(train: tuple, val: tuple, test: tuple,
             extra_tests: dict[str, tuple] | None = None) -> dict:
    """Fit train only, choose L2 on validation, score test once.

    ``extra_tests`` maps a name to another held-out (x, y, subjects) set, scored
    with the same probe (``extra_tests[name]``) and with an unweighted refit at
    the selected C (``extra_tests[name]["unweighted_probe"]``: the training class
    ratio as prior instead of the class-balanced one, which moves the 0.5 threshold).
    """
    x_train, y_train, _ = train
    x_val, y_val, val_subjects = val
    x_test, y_test, subjects = test
    if any(len(set(y)) != 2 for y in (y_train, y_val, y_test)):
        raise ValueError("Each probe split needs both AD and HC subjects.")
    best = None
    for c in (0.01, 0.1, 1.0, 10.0):
        probe = make_pipeline(StandardScaler(), LogisticRegression(
            C=c, class_weight="balanced", max_iter=1000, random_state=42))
        probe.fit(x_train, y_train)
        val_auc = float(roc_auc_score(y_val, probe.predict_proba(x_val)[:, 1]))
        if best is None or val_auc > best[0]:
            best = (val_auc, c, probe)
    assert best is not None
    val_scores = best[2].predict_proba(x_val)[:, 1]
    test_scores = best[2].predict_proba(x_test)[:, 1]
    # effective rank of the variance spectrum (identity.effective_rank), not of the
    # singular values (metrics.effective_rank, the top-level "effective_rank" of score)
    centered = x_train - x_train.mean(axis=0, keepdims=True)
    spectrum = np.linalg.svd(centered, compute_uv=False) ** 2
    spectrum = spectrum / max(float(spectrum.sum()), 1e-12)
    positive = spectrum[spectrum > 0]
    variance_rank = float(np.exp(-np.sum(positive * np.log(positive))))
    extra: dict[str, dict] = {}
    if extra_tests:
        unweighted = make_pipeline(StandardScaler(), LogisticRegression(
            C=best[1], max_iter=1000, random_state=42)).fit(x_train, y_train)
        for name, (x_extra, y_extra, extra_subjects) in extra_tests.items():
            extra[name] = held_out_metrics(y_extra, best[2].predict_proba(x_extra)[:, 1],
                                           extra_subjects)
            plain = unweighted.predict_proba(x_extra)[:, 1]
            extra[name]["unweighted_probe"] = {
                "auc": float(roc_auc_score(y_extra, plain)),
                "average_precision": float(average_precision_score(y_extra, plain)),
                **threshold_metrics(y_extra, plain)}
    return {**({"extra_tests": extra} if extra else {}),
            "train_subjects": len(y_train), "val_subjects": len(y_val),
            "test_subjects": len(y_test), "selected_C": best[1],
            "embedding_dim": int(x_train.shape[1]),
            "embedding_mean_feature_std": float(centered.std(axis=0).mean()),
            "embedding_effective_rank": variance_rank,
            "val_auc": best[0],
            "val_average_precision": float(average_precision_score(y_val, val_scores)),
            "val_f1_macro": float(f1_score(y_val, val_scores >= 0.5, average="macro")),
            "test_auc": float(roc_auc_score(y_test, test_scores)),
            "test_average_precision": float(average_precision_score(y_test, test_scores)),
            "test_brier_score": float(brier_score_loss(y_test, test_scores)),
            "val_log_loss": float(log_loss(y_val, val_scores, labels=[0, 1])),
            "test_log_loss": float(log_loss(y_test, test_scores, labels=[0, 1])),
            "test_accuracy": float(accuracy_score(y_test, test_scores >= 0.5)),
            "test_balanced_accuracy": float(balanced_accuracy_score(y_test, test_scores >= 0.5)),
            "test_f1_macro": float(f1_score(y_test, test_scores >= 0.5, average="macro")),
            "val_predictions": [{"subject": subject, "AD": int(label), "score": float(score)}
                                for subject, label, score in zip(
                                    val_subjects, y_val, val_scores, strict=True
                                )],
            "test_predictions": [{"subject": subject, "AD": int(label), "score": float(score)}
                                 for subject, label, score in zip(
                                     subjects, y_test, test_scores, strict=True
                                 )]}


def score(encoder: torch.nn.Module, dm: Any, clean: Any,
          keep_embeddings: bool = False) -> dict[str, Any]:
    """Campaign scorecard of one frozen encoder: CV probe, test probe, effective rank."""
    from t5_ssl.evaluation.embeddings import (
        concat_splits,
        external_embeddings,
        split_embeddings,
        train_val_embeddings,
    )
    x, y, subjects = train_val_embeddings(encoder, dm, clean)
    cv = cv_probe(x, y, {"clean": x})
    splits = [split_embeddings(encoder, dm, clean, s) for s in ("train", "val", "test")]
    # Extra held-out splits test_<name>: test + those subjects, scored as "<name>".
    extra = {name.removeprefix("test_"): concat_splits(
        splits[2], split_embeddings(encoder, dm, clean, name))
        for name in getattr(dm, "extra_test_splits", [])}
    # External labeled sets (another cohort): scored on their own with the same probe.
    for name, external in getattr(dm, "external_sets", {}).items():
        extra[name] = external_embeddings(encoder, external, clean)
    test = evaluate(*splits, extra_tests=extra or None)
    out = {"subjects": subjects, "labels": y.tolist(), "cv": cv, "test": test,
           "effective_rank": effective_rank(x)}
    if keep_embeddings:
        out["_embeddings"] = {"cv": (x, y, subjects), "train": splits[0], "val": splits[1],
                              "test": splits[2], **extra}
    return out
