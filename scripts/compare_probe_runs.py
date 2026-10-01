"""Paired bootstrap comparison of two frozen-probe runs on the same test subjects.

Point-estimate deltas between two independent SSL training runs (e.g. with vs.
without a balanced sampler) can't tell you whether the difference is real or
just training noise. Since both runs are evaluated on the identical, fixed
test split, this pairs their per-subject predictions and bootstraps over
subjects (stratified by label, same style as evaluate_frozen_probe.py's
SSL-vs-random comparison) to get a delta with a confidence interval and a
P(delta > 0).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def load_predictions(probe_json: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    doc = json.loads(probe_json.read_text(encoding="utf-8"))
    predictions = doc["results"]["trained"]["test_predictions"]
    subjects = [p["subject"] for p in predictions]
    labels = np.array([p["AD"] for p in predictions])
    scores = np.array([p["score"] for p in predictions])
    return labels, scores, subjects


def bootstrap_delta(labels: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray,
                    metric: str, repetitions: int, seed: int) -> dict:
    """Bootstrap metric(b) - metric(a) over subjects, stratified by label."""
    def compute(y: np.ndarray, s: np.ndarray) -> float:
        if metric == "auroc":
            return float(roc_auc_score(y, s))
        if metric == "aupr":
            return float(average_precision_score(y, s))
        if metric == "f1_macro":
            return float(f1_score(y, s >= 0.5, average="macro"))
        raise ValueError(metric)

    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    rng = np.random.default_rng(seed)
    deltas, a_vals, b_vals = [], [], []
    for _ in range(repetitions):
        indices = np.concatenate((rng.choice(positive, len(positive), replace=True),
                                  rng.choice(negative, len(negative), replace=True)))
        a = compute(labels[indices], scores_a[indices])
        b = compute(labels[indices], scores_b[indices])
        a_vals.append(a)
        b_vals.append(b)
        deltas.append(b - a)
    observed = compute(labels, scores_b) - compute(labels, scores_a)
    return {
        "metric": metric,
        "a_point": compute(labels, scores_a), "b_point": compute(labels, scores_b),
        "observed_delta": observed,
        "delta_bootstrap_95ci": [float(v) for v in np.percentile(deltas, [2.5, 97.5])],
        "bootstrap_probability_b_better": float(np.mean(np.array(deltas) > 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe_a", type=Path, help="Baseline probe JSON (e.g. no balanced sampler)")
    parser.add_argument("probe_b", type=Path, help="Candidate probe JSON (e.g. with balanced sampler)")
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    labels_a, scores_a, subjects_a = load_predictions(args.probe_a)
    labels_b, scores_b, subjects_b = load_predictions(args.probe_b)
    if subjects_a != subjects_b or not np.array_equal(labels_a, labels_b):
        # Align by subject id rather than assuming identical row order.
        order_b = {s: i for i, s in enumerate(subjects_b)}
        if set(subjects_a) != set(subjects_b):
            raise ValueError("The two runs were not evaluated on the same test subjects; "
                             "not a fair paired comparison.")
        reindex = [order_b[s] for s in subjects_a]
        scores_b = scores_b[reindex]
        labels_b = labels_b[reindex]
        if not np.array_equal(labels_a, labels_b):
            raise ValueError("Subject AD/HC labels disagree between the two runs.")

    results = {}
    for metric in ("auroc", "aupr", "f1_macro"):
        results[metric] = bootstrap_delta(labels_a, scores_a, scores_b, metric,
                                          args.repetitions, args.seed)

    print(f"Paired bootstrap comparison on {len(labels_a)} test subjects "
         f"({int(labels_a.sum())} AD, {int((labels_a == 0).sum())} HC)\n"
         f"{args.label_a} = {args.probe_a}\n{args.label_b} = {args.probe_b}\n")
    for metric, r in results.items():
        verdict = ("SIGNIFICANT" if r["delta_bootstrap_95ci"][0] > 0 else
                   "SIGNIFICANTLY WORSE" if r["delta_bootstrap_95ci"][1] < 0 else
                   "not significant (CI crosses 0)")
        print(f"[{metric}] {args.label_a}={r['a_point']:.4f} {args.label_b}={r['b_point']:.4f} "
             f"delta={r['observed_delta']:+.4f} 95%CI=[{r['delta_bootstrap_95ci'][0]:+.4f}, "
             f"{r['delta_bootstrap_95ci'][1]:+.4f}] P({args.label_b}>{args.label_a})="
             f"{r['bootstrap_probability_b_better']:.3f} -> {verdict}")
    print()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
