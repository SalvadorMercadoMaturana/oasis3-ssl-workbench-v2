#!/usr/bin/env python
"""Tune the frozen probe of protocol v2 on the representative CV of split_deploy (CPU only).

    python scripts/tune_probe.py [--out campaign_ssl3d_runpod/outer5/probe_tuning.json]

Variants of ``probe_protocol.make_probe`` (StandardScaler + class-balanced logistic
regression, C by inner 3-fold log-loss), each scored as a 3-seed ensemble on the same fold
plan: ``pca{n}`` (PCA to n components after standardization, whitening off), ``l1``
(lasso path, saga), ``enet`` (elastic net 0.5). Selection would be on the CV composite;
the test is reported only to confirm.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import probe_protocol as pp  # noqa: E402
from analyze_cv_vs_test import MANIFEST, MODELS, RUNS  # noqa: E402


def make(variant: str):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    cv = StratifiedKFold(3, shuffle=True, random_state=0)
    if variant == "base":
        return pp.make_probe()
    if variant.startswith("pca"):
        return make_pipeline(StandardScaler(), PCA(int(variant[3:]), random_state=0),
                             StandardScaler(), LogisticRegressionCV(
                                 Cs=list(pp.CS), cv=cv, scoring="neg_log_loss",
                                 class_weight="balanced", max_iter=5000))
    penalty, ratios = ("l1", None) if variant == "l1" else ("elasticnet", [0.5])
    return make_pipeline(StandardScaler(), LogisticRegressionCV(
        Cs=[1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0], cv=cv, scoring="neg_log_loss",
        class_weight="balanced", penalty=penalty, solver="saga", l1_ratios=ratios,
        max_iter=3000))


def score(run: Path, variant: str, plan: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(run / "embeddings.npz", allow_pickle=True)
    x_lab, y_lab, x_pool = z["cv_x"], z["cv_y"].astype(int), z["pool_x"]
    x_all = np.concatenate([x_lab, x_pool])
    oof = np.zeros((plan.shape[0], len(x_all)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for r in range(plan.shape[0]):
            for k in np.unique(plan[r]):
                fit = plan[r, :len(y_lab)] != k
                held = plan[r] == k
                oof[r, held] = make(variant).fit(x_lab[fit], y_lab[fit]).predict_proba(
                    x_all[held])[:, 1]
        p_test = make(variant).fit(x_lab, y_lab).predict_proba(z["test_x"])[:, 1]
    return oof, p_test


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "campaign_ssl3d_runpod/outer5/probe_tuning.json")
    ap.add_argument("--variants", nargs="+", default=["base", "pca16", "pca64", "pca128", "l1", "enet"])
    args = ap.parse_args()
    ages = pp.session_ages(MANIFEST)
    out = {}
    for label, (tag, name) in MODELS.items():
        docs = pp.load_runs(RUNS, tag, name)
        if not docs or "pool_x" not in np.load(Path(docs[0]["dir"]) / "embeddings.npz").files:
            print("skip (no SSL-only pool embeddings)", label, flush=True)
            continue
        first = docs[0]["scores"]
        plan = np.asarray(first["plan"])
        y = np.asarray(first["cv_y"])
        n_lab = first["n_lab"]
        age_all = np.array([ages.get(s, np.nan) for s in first["cv_subjects"]])
        y_test = np.asarray(first["test_y"])
        age_test = np.array([ages[s] for s in first["test_subjects"]])
        out[label] = {}
        for variant in args.variants:
            scored = [score(Path(d["dir"]), variant, plan) for d in docs]
            oof = np.mean([s[0] for s in scored], axis=0)
            p_test = np.mean([s[1] for s in scored], axis=0)
            summ = pp.summarize(y[:n_lab], y[n_lab:], oof, plan, age_all)
            test = pp.held_out(y_test, p_test, summ["threshold_rep"], age_test)
            out[label][variant] = {"cv": {k: summ["cv_rep"][k] for k in ("auc", "aupr", "f1_macro", "mean3")},
                                   "test": {k: test[k] for k in ("auc", "aupr", "f1_macro", "mean3")}}
            print(f"{label:34s} {variant:7s} CV {summ['cv_rep']['mean3']:.3f} "
                  f"(AUROC {summ['cv_rep']['auc']:.3f}) | test {test['mean3']:.3f} "
                  f"(AUROC {test['auc']:.3f})", flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
