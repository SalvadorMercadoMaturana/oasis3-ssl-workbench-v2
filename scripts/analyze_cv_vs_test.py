#!/usr/bin/env python
"""Why the representative test scores above the representative CV on split_deploy.

    python scripts/analyze_cv_vs_test.py [--out campaign_ssl3d_runpod/split_study/cv_vs_test.json]

Uses the saved frozen embeddings and protocol-v2 scores of the main seed ensembles (no GPU).
Separates four candidate causes of ``test > CV``:

1. population: age-only AUROC and age gap in the CV population vs the test;
2. estimator: AUROC of OOF scores pooled over 5 different fold probes vs the mean of the
   per-fold AUROCs (each fold scored by one probe, like the test);
3. probe training size: the test scored by the 5 CV-fold probes (80 % of the labeled
   subjects, as in the CV) instead of the probe fit on all of them;
4. sampling: where the observed test falls among 2000 pseudo-tests of the same size and
   composition (41 AD + 137 HC, age tertile within class) drawn from the CV population.
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

RUNS = ROOT / "campaign_ssl3d_runpod/deploy/runs"
MANIFEST = ROOT / "campaign_ssl3d_runpod/split_deploy/manifest.csv"
MODELS = {
    "BYOL ConvNeXt-T BN (desplegado)": ("v2_ssl", "byol_wide_dproj0.1__noise__bbconvnext_tiny__roi32"),
    "VICReg ConvNeXt-T": ("v2_ssl", "vicreg_campaign_augval__geo__bbconvnext_tiny__roi32"),
    "BYOL ResNet-18": ("v2_ssl", "byol_gn_dp0.3_dq0_wd0.01__noise__bbresnet18__roi32"),
    "DINOv2 ConvNeXt-T": ("dino_fix", "dinov2_b64_k256__elastic_tio_gpu__bbconvnext_tiny__roi32"),
    "DINOv2 ViT-S hipocampo": ("dino_fix", "dinov2_campaign__elastic_tio_gpu__bbvit_small__roi32"),
    "sin SSL ConvNeXt-T": ("dep50_init", "vicreg_init__bbconvnext_tiny__roi32"),
}


def per_fold_auc(y: np.ndarray, oof: np.ndarray, plan: np.ndarray) -> float:
    vals = []
    for r in range(oof.shape[0]):
        for k in np.unique(plan[r]):
            m = plan[r] == k
            vals.append(pp.auc(y[m], oof[r, m]))
    return float(np.mean(vals))


def fold_probe_test(run: Path, plan: np.ndarray) -> tuple[float, float]:
    """Test AUROC of the mean prediction of the CV-fold probes, and of the full probe."""
    z = np.load(run / "embeddings.npz", allow_pickle=True)
    x_lab, y_lab = z["cv_x"], z["cv_y"].astype(int)
    x_test, y_test = z["test_x"], z["test_y"].astype(int)
    n_lab = len(y_lab)
    folds = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for r in range(min(2, plan.shape[0])):
            for k in np.unique(plan[r]):
                fit = plan[r, :n_lab] != k
                p = pp.make_probe().fit(x_lab[fit], y_lab[fit]).predict_proba(x_test)[:, 1]
                folds.append(pp.auc(y_test, p))
        full = pp.make_probe().fit(x_lab, y_lab).predict_proba(x_test)[:, 1]
    return float(np.mean(folds)), pp.auc(y_test, full)


def pseudo_tests(y: np.ndarray, s: np.ndarray, age: np.ndarray, n_ad: int, n_hc: int,
                 draws: int = 2000, seed: int = 0) -> np.ndarray:
    """AUROC of ``draws`` pseudo-tests (age tertile within class), without replacement."""
    rng = np.random.default_rng(seed)
    idx = {}
    for c, n in ((1, n_ad), (0, n_hc)):
        members = np.flatnonzero(y == c)
        cuts = np.quantile(age[members], [1 / 3, 2 / 3])
        tert = np.digitize(age[members], cuts)
        idx[c] = [(members[tert == t], round(n * (tert == t).mean())) for t in range(3)]
    out = np.empty(draws)
    for d in range(draws):
        pick = np.concatenate([rng.choice(m, k, replace=False) for c in (1, 0) for m, k in idx[c]])
        out[d] = pp.auc(y[pick], s[pick])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=ROOT / "campaign_ssl3d_runpod/split_study/cv_vs_test.json")
    args = ap.parse_args()
    ages = pp.session_ages(MANIFEST)
    rows = {}
    for label, (tag, name) in MODELS.items():
        docs = pp.load_runs(RUNS, tag, name)
        if not docs:
            print("missing", label)
            continue
        ens = pp.ensemble(docs, ages)
        y, s_cv, _ = ens["_cv"]
        y_test, _ = ens["_test"]
        first = docs[0]["scores"]
        oof = np.mean([np.asarray(d["scores"]["cv_oof"]) for d in docs], axis=0)
        plan = np.asarray(first["plan"])
        age_cv = np.array([ages[s] for s in first["cv_subjects"]])
        age_test = np.array([ages[s] for s in first["test_subjects"]])
        pseudo = pseudo_tests(y, s_cv, age_cv, int(y_test.sum()), int((y_test == 0).sum()))
        fold_t, full_t = zip(*(fold_probe_test(Path(d["dir"]), plan) for d in docs), strict=True)
        row = {
            "cv_rep_auc_pooled": ens["cv_rep"]["auc"],
            "cv_rep_auc_per_fold": per_fold_auc(y, oof, plan),
            "cv_rep_auc_mean_over_repeats": pp.auc(y, s_cv),
            "cv_auc_age": ens["cv_rep"]["auc_age"],
            "test_auc": ens["test"]["auc"],
            "test_auc_age": ens["test"]["auc_age"],
            "test_auc_fold_probes_single_seed": float(np.mean(fold_t)),
            "test_auc_full_probe_single_seed": float(np.mean(full_t)),
            "pseudo_test_auc_mean": float(pseudo.mean()),
            "pseudo_test_auc_p2.5_p97.5": [float(np.percentile(pseudo, 2.5)),
                                           float(np.percentile(pseudo, 97.5))],
            "test_percentile_among_pseudo": float((pseudo < ens["test"]["auc"]).mean()),
        }
        rows[label] = row
        print(label, json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                                 for k, v in row.items()}))
    first = docs[0]["scores"]
    y = np.asarray(first["cv_y"])
    age_cv = np.array([ages[s] for s in first["cv_subjects"]])
    y_test = np.asarray(first["test_y"])
    age_test = np.array([ages[s] for s in first["test_subjects"]])
    pop = {
        "cv": {"n": len(y), "ad": int(y.sum()), "age_auc": pp.auc(y, age_cv),
               "age_ad": float(age_cv[y == 1].mean()), "age_hc": float(age_cv[y == 0].mean())},
        "test": {"n": len(y_test), "ad": int(y_test.sum()), "age_auc": pp.auc(y_test, age_test),
                 "age_ad": float(age_test[y_test == 1].mean()),
                 "age_hc": float(age_test[y_test == 0].mean())},
        "age_pseudo_test_auc_p2.5_p97.5": [float(np.percentile(a, q)) for a in
                                          [pseudo_tests(y, age_cv, age_cv, int(y_test.sum()),
                                                        int((y_test == 0).sum()))]
                                          for q in (2.5, 97.5)],
    }
    print("population", json.dumps(pop))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"models": rows, "population": pop}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
