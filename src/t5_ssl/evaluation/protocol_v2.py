"""Frozen-probe evaluation protocol v2 for split_deploy (numpy/sklearn only, no torch).

Moved verbatim from ``scripts/probe_protocol.py`` (2026-11-16 refactor); scripts import it
as ``from t5_ssl.evaluation import protocol_v2 as pp``.

Why a second protocol (2026-11-07). The campaign protocol (``probe_v1.cv_probe`` +
``probe_v1.evaluate``) scored the CV on the age-matched 50/50 labeled
subjects and the test on the representative 1:3.3 test (real ages, AD older). The two
numbers measured different populations: the test AUROC carries the age shortcut (age
alone 0.76) and the CV does not, so the CV came out *below* the test (0.72 vs 0.85 for
the deployed model). The probe trained on train only -> val gave the same 0.72, so it
was not leakage from SSL having seen the train subjects. AUPR and F1-macro on a
50/50 CV say nothing about the class imbalance of a real clinic.

This protocol keeps the probe training exactly as deployed (class-balanced logistic
regression on the age-matched labeled subjects) and changes what it is scored on:

- **Representative CV** (``cv_rep``): every repeat splits the labeled age-matched
  subjects into 5 stratified folds and also deals the SSL-only AD/HC subjects of train
  (``labeled=0``, never used by any probe) into 5 folds. Fold k's probe is fit on the
  labeled subjects outside fold k and scores fold k's labeled *and* SSL-only subjects.
  Together they are every non-test OASIS3 AD/HC subject: cohort prevalence (23 % AD)
  and real ages, like the test, with four times its subjects.
- **Age-matched CV** (``cv_matched``): the same predictions restricted to the labeled
  subjects (the old CV population).
- **Thresholds out of sample**: F1-macro, balanced accuracy, sensitivity, specificity,
  PPV and MCC use, for fold k, the threshold that maximizes F1-macro on the other folds'
  out-of-fold predictions of the same population. The test uses the threshold that
  maximizes F1-macro on the representative CV predictions (never the test itself).
  F1-macro at 0.5 (the class-balanced prior) is kept as ``f1_macro_05``.
- **C** from 1e-5 .. 1 by inner 3-fold CV *log-loss* (a proper, smooth score) instead of
  AUROC: the old 0.01 .. 10 grid hit its lower edge in ~95 % of folds, the old test probe
  chose C on 60 validation subjects (over-fitting the ImageNet encoders), and AUROC on
  ~78-subject inner folds is a noisy criterion (with the wider grid it picked worse C;
  log-loss gave a higher representative and age-matched CV for all five encoders tried,
  +0.004 to +0.05; chosen on CV only).
- **Age-conditional AUROC** (``auc_age``): the AUROC over AD-HC pairs whose ages differ
  by at most 3 years, i.e. discrimination beyond the age shortcut, defined on any
  population (age itself scores ~0.5 there).
- **Composite** for model selection: ``mean3 = (AUROC + AUPR + F1-macro) / 3`` on the
  representative CV (prevalence 23 %), so the imbalance-sensitive metrics weigh as
  much as AUROC.
- The test probe is fit on all labeled train+val subjects (as the deployed one), C by
  inner 3-fold CV log-loss, instead of train only with C chosen on 60 val subjects.
"""

from __future__ import annotations

import csv
import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# The project root (= t5_ssl.data.paths.PROJECT_ROOT, not imported: that package pulls torch).
ROOT = Path(__file__).resolve().parents[3]
CSV = ROOT / "data/dementia/mni-dataset.csv"
CS = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 1.0)
REPEATS, FOLDS = 5, 5
AGE_CALIPER = 3.0
THRESHOLD_METRICS = ("f1_macro", "f1_macro_05", "balanced_accuracy", "sensitivity",
                     "specificity", "ppv", "npv", "mcc")


# ---------------------------------------------------------------- subjects
def subject_labels(csv_path: Path = CSV) -> dict[str, int]:
    """Unambiguous OASIS3 AD/HC subjects (one group over all sessions), as the datamodule."""
    from t5_ssl.data.oasis3_lmdb_datamodule import clinical_rows
    groups: dict[str, set[str]] = defaultdict(set)
    for row in clinical_rows(csv_path):
        groups[row["subject"]].add(row["group"])
    return {s: int(next(iter(g)) == "AD") for s, g in groups.items()
            if len(g) == 1 and next(iter(g)) in ("AD", "HC")}


def pool_records(manifest: Path, csv_path: Path = CSV) -> list[dict[str, Any]]:
    """SSL-only AD/HC subjects of train (manifest ``labeled=0``), primary session, sorted."""
    labels = subject_labels(csv_path)
    rows = {}
    with Path(manifest).open(encoding="utf-8") as stream:
        manifest_rows = list(csv.DictReader(stream))
    for row in manifest_rows:
        if (row["split"] == "train" and row.get("labeled") == "0" and row.get("primary") == "1"
                and row["subject"] in labels):
            rows[row["subject"]] = {"subject": row["subject"], "idx": row["idx"],
                                    "session": row["session"], "group": row["group"],
                                    "label": labels[row["subject"]]}
    return [rows[s] for s in sorted(rows)]


def session_ages(manifest: Path, csv_path: Path = CSV, external: Path | None = None
                 ) -> dict[str, float]:
    """Age at the evaluated session: the primary session of every manifest subject."""
    with Path(csv_path).open(encoding="utf-8") as stream:
        age = {r["idx"]: float(r["age"]) for r in csv.DictReader(stream)
               if r["age"] not in ("", None)}
    with Path(manifest).open(encoding="utf-8") as stream:
        manifest_rows = list(csv.DictReader(stream))
    out: dict[str, float] = {}
    for row in manifest_rows:
        if row.get("primary") == "1" and row["idx"] in age:
            out[row["subject"]] = age[row["idx"]]
    if external is not None:
        with Path(external).open(encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                out[row["subject"]] = float(row["age"])
    return out


# ---------------------------------------------------------------- metrics
def auc(y: np.ndarray, s: np.ndarray) -> float:
    """AUROC from average ranks (Mann-Whitney), as sklearn with ties."""
    from scipy.stats import rankdata
    ranks = rankdata(s)
    pos = y == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def aupr(y: np.ndarray, s: np.ndarray) -> float:
    """Average precision with sklearn's definition (one step per distinct score)."""
    order = np.argsort(-s, kind="mergesort")
    ss, yy = s[order], y[order]
    last = np.r_[np.flatnonzero(np.diff(ss)), len(ss) - 1]
    tp = np.cumsum(yy)[last]
    return float(np.sum(np.diff(np.r_[0, tp / max(1, yy.sum())]) * tp / (last + 1)))


def auc_age(y: np.ndarray, s: np.ndarray, age: np.ndarray, caliper: float = AGE_CALIPER
            ) -> float:
    """AUROC over AD-HC pairs at most ``caliper`` years apart (beyond the age shortcut)."""
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    close = np.abs(age[pos][:, None] - age[neg][None, :]) <= caliper
    if not close.any():
        return float("nan")
    diff = s[pos][:, None] - s[neg][None, :]
    win = (diff > 0) + 0.5 * (diff == 0)
    return float(win[close].mean())


def at_threshold(y: np.ndarray, s: np.ndarray, t: float) -> dict[str, float]:
    """Confusion-matrix metrics predicting AD when ``s >= t`` (0 where undefined)."""
    pred = s >= t
    tp, fp = float((pred & (y == 1)).sum()), float((pred & (y == 0)).sum())
    fn, tn = float((~pred & (y == 1)).sum()), float((~pred & (y == 0)).sum())
    div = lambda a, b: a / b if b else 0.0  # noqa: E731
    f1_pos, f1_neg = div(2 * tp, 2 * tp + fp + fn), div(2 * tn, 2 * tn + fn + fp)
    mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {"f1_macro": (f1_pos + f1_neg) / 2, "f1_ad": f1_pos,
            "balanced_accuracy": (div(tp, tp + fn) + div(tn, tn + fp)) / 2,
            "sensitivity": div(tp, tp + fn), "specificity": div(tn, tn + fp),
            "ppv": div(tp, tp + fp), "npv": div(tn, tn + fn),
            "mcc": float((tp * tn - fp * fn) / mcc_den) if mcc_den else 0.0,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def best_f1_threshold(y: np.ndarray, s: np.ndarray) -> float:
    """Threshold maximizing F1-macro (predict AD when score >= t), vectorized over cut points."""
    order = np.argsort(-s, kind="mergesort")
    ss, yy = s[order], y[order].astype(float)
    last = np.r_[np.flatnonzero(np.diff(ss)), len(ss) - 1]  # cut after each tied block
    tp = np.cumsum(yy)[last]
    fp = (last + 1) - tp
    p, n = yy.sum(), len(yy) - yy.sum()
    fn, tn = p - tp, n - fp
    f1 = (2 * tp / np.maximum(2 * tp + fp + fn, 1) + 2 * tn / np.maximum(2 * tn + fn + fp, 1)) / 2
    k = np.argmax(f1)
    return float(ss[last[k]])


def rank_metrics(y: np.ndarray, s: np.ndarray, age: np.ndarray | None = None) -> dict:
    """Threshold-free metrics: AUROC, AUPR and, given ages, ``auc_age``."""
    out = {"n": len(y), "ad": int(y.sum()), "prevalence": float(y.mean()),
           "auc": auc(y, s), "aupr": aupr(y, s)}
    if age is not None:
        out["auc_age"] = auc_age(y, s, age)
    return out


def held_out(y: np.ndarray, s: np.ndarray, t: float, age: np.ndarray | None = None) -> dict:
    """Rank metrics plus threshold metrics at ``t`` and F1-macro at 0.5; ``mean3``."""
    out = rank_metrics(y, s, age)
    out.update(at_threshold(y, s, t))
    out["f1_macro_05"] = at_threshold(y, s, 0.5)["f1_macro"]
    out["threshold"] = float(t)
    out["mean3"] = (out["auc"] + out["aupr"] + out["f1_macro"]) / 3
    return out


# ---------------------------------------------------------------- folds and OOF metrics
def fold_plan(y_lab: np.ndarray, y_pool: np.ndarray, repeats: int = REPEATS,
              folds: int = FOLDS) -> np.ndarray:
    """Fold index of every labeled then pool subject, per repeat: ``(repeats, n_lab + n_pool)``."""
    from sklearn.model_selection import StratifiedKFold
    plan = np.zeros((repeats, len(y_lab) + len(y_pool)), dtype=int)
    for r in range(repeats):
        for k, (_, te) in enumerate(StratifiedKFold(folds, shuffle=True, random_state=r)
                                    .split(y_lab, y_lab)):
            plan[r, te] = k
        if len(y_pool):
            for k, (_, te) in enumerate(StratifiedKFold(folds, shuffle=True, random_state=1000 + r)
                                        .split(y_pool, y_pool)):
                plan[r, len(y_lab) + te] = k
    return plan


def oof_metrics(y: np.ndarray, oof: np.ndarray, plan: np.ndarray, mask: np.ndarray,
                age: np.ndarray | None = None) -> dict:
    """Mean over repeats of the metrics of ``oof[:, mask]``; thresholds from the other folds."""
    per_repeat = []
    for r in range(oof.shape[0]):
        yy, ss, ff = y[mask], oof[r, mask], plan[r, mask]
        m = rank_metrics(yy, ss, None if age is None else age[mask])
        pred_t = np.zeros(len(yy))
        for k in np.unique(ff):
            other = ff != k
            pred_t[ff == k] = best_f1_threshold(yy[other], ss[other])
        # thresholds differ per fold: compare each score with its own fold's threshold
        m.update(at_threshold(yy, ss - pred_t, 0.0))
        m["f1_macro_05"] = at_threshold(yy, ss, 0.5)["f1_macro"]
        m["threshold"] = float(np.mean(pred_t))
        m["mean3"] = (m["auc"] + m["aupr"] + m["f1_macro"]) / 3
        per_repeat.append(m)
    out = {k: float(np.mean([m[k] for m in per_repeat])) for k in per_repeat[0]}
    out["mean3_sd_repeats"] = float(np.std([m["mean3"] for m in per_repeat]))
    return out


# ---------------------------------------------------------------- probes
def make_probe(cs: tuple[float, ...] = CS) -> Any:
    """Standardization + class-balanced logistic regression, C by inner 3-fold log-loss."""
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegressionCV(
        Cs=list(cs), cv=StratifiedKFold(3, shuffle=True, random_state=0), scoring="neg_log_loss",
        class_weight="balanced", max_iter=5000))


def chosen_c(probe: Any) -> float:
    model = probe[-1]
    return float(model.C_[0]) if hasattr(model, "C_") else float(model.C)


def probe_oof(x_lab: np.ndarray, y_lab: np.ndarray, x_pool: np.ndarray, plan: np.ndarray
              ) -> tuple[np.ndarray, list[float]]:
    """Out-of-fold probe scores of labeled + pool subjects ``(repeats, n)``, and the chosen C."""
    x_all = np.concatenate([x_lab, x_pool]) if len(x_pool) else x_lab
    n_lab = len(y_lab)
    oof = np.zeros((plan.shape[0], len(x_all)))
    chosen = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for r in range(plan.shape[0]):
            for k in np.unique(plan[r]):
                fit = plan[r, :n_lab] != k
                probe = make_probe().fit(x_lab[fit], y_lab[fit])
                chosen.append(chosen_c(probe))
                held = plan[r] == k
                oof[r, held] = probe.predict_proba(x_all[held])[:, 1]
    return oof, chosen


def summarize(y_lab: np.ndarray, y_pool: np.ndarray, oof: np.ndarray, plan: np.ndarray,
              age_all: np.ndarray | None) -> dict:
    """``cv_rep`` (labeled + pool) and ``cv_matched`` (labeled only) from OOF scores."""
    y_all = np.concatenate([y_lab, y_pool])
    lab = np.zeros(len(y_all), bool)
    lab[:len(y_lab)] = True
    out = {"cv_rep": oof_metrics(y_all, oof, plan, np.ones(len(y_all), bool), age_all),
           "cv_matched": oof_metrics(y_all, oof, plan, lab, age_all)}
    mean_oof = oof.mean(0)
    out["threshold_rep"] = best_f1_threshold(y_all, mean_oof)
    out["threshold_matched"] = best_f1_threshold(y_lab, mean_oof[:len(y_lab)])
    return out


def evaluate(lab: tuple, pool: tuple, test: tuple, ages: dict[str, float],
             extra: dict[str, tuple] | None = None, keep_scores: bool = True) -> dict:
    """Run the whole protocol on embeddings ``(x, y, subjects)``; return a JSON-able dict.

    ``extra`` maps a name to another held-out set (e.g. ADNI), scored by the test probe at
    the representative threshold.
    """
    x_lab, y_lab, s_lab = lab
    x_pool, y_pool, s_pool = pool
    x_test, y_test, s_test = test
    y_lab, y_pool, y_test = (np.asarray(v).astype(int) for v in (y_lab, y_pool, y_test))
    plan = fold_plan(y_lab, y_pool)
    oof, chosen = probe_oof(x_lab, y_lab, x_pool, plan)
    subjects = list(s_lab) + list(s_pool)
    age_all = np.array([ages.get(s, np.nan) for s in subjects])
    out = summarize(y_lab, y_pool, oof, plan, age_all)
    out["cv_C"] = {str(c): chosen.count(c) for c in sorted(set(chosen))}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        probe = make_probe().fit(x_lab, y_lab)
    out["test_C"] = chosen_c(probe)
    t_rep = out["threshold_rep"]
    p_test = probe.predict_proba(x_test)[:, 1]
    age_test = np.array([ages.get(s, np.nan) for s in s_test])
    out["test"] = held_out(y_test, p_test, t_rep, age_test)
    out["test"]["at_matched_threshold"] = at_threshold(y_test, p_test, out["threshold_matched"])
    matched = age_matched_subset(list(s_test), y_test, ages)
    out["test_agem"] = held_out(y_test[matched], p_test[matched], t_rep, age_test[matched])
    for name, (x_e, y_e, s_e) in (extra or {}).items():
        y_e = np.asarray(y_e).astype(int)
        p_e = probe.predict_proba(x_e)[:, 1]
        age_e = np.array([ages.get(s, np.nan) for s in s_e])
        out[name] = held_out(y_e, p_e, t_rep, age_e if np.isfinite(age_e).all() else None)
        if keep_scores:
            out[name]["scores"] = {"subjects": list(map(str, s_e)), "y": y_e.tolist(),
                                   "p": p_e.tolist()}
    if keep_scores:
        out["scores"] = {"cv_subjects": subjects, "cv_y": np.concatenate([y_lab, y_pool]).tolist(),
                         "n_lab": len(y_lab), "cv_oof": oof.tolist(), "plan": plan.tolist(),
                         "test_subjects": list(map(str, s_test)), "test_y": y_test.tolist(),
                         "test_p": p_test.tolist()}
    return out


def age_matched_subset(subjects: list[str], y: np.ndarray, ages: dict[str, float],
                       per_ad: int = 2, caliper: float = 2.5) -> np.ndarray:
    """Return the indices of an age-matched subset: each AD with up to ``per_ad`` HC near in age."""
    ad = sorted((i for i in range(len(y)) if y[i] == 1), key=lambda i: -ages[subjects[i]])
    free = [i for i in range(len(y)) if y[i] == 0]
    keep = []
    for _ in range(per_ad):
        for i in ad:
            best = min(free, key=lambda j: abs(ages[subjects[j]] - ages[subjects[i]]), default=None)
            if best is not None and abs(ages[subjects[best]] - ages[subjects[i]]) <= caliper:
                keep += [i, best]
                free.remove(best)
    return np.array(sorted(set(keep)), dtype=int)


def age_reference(y: np.ndarray, subjects: list[str], ages: dict[str, float]) -> dict:
    """Age itself as the score (older = AD): the shortcut available on a population."""
    a = np.array([ages[s] for s in subjects])
    return rank_metrics(np.asarray(y).astype(int), a, a)


# ---------------------------------------------------------------- runs and seed ensembles
def load_runs(runs: Path, tag: str, name: str) -> list[dict]:
    """Protocol-v2 documents of ``<runs>/<tag>/<name>_s<seed>``, with training curves.

    Frozen-probe runs: ``protocol_v2.json`` (reeval_deploy.py); supervised runs:
    ``result.json["protocol_v2"]`` (train_supervised.py).
    """
    docs = []
    for run in sorted((Path(runs) / tag).glob(f"{name}_s*")):
        result = run / "result.json"
        full = json.loads(result.read_text(encoding="utf-8")) if result.is_file() else {}
        path = run / "protocol_v2.json"
        doc = (json.loads(path.read_text(encoding="utf-8")) if path.is_file()
               else full.get("protocol_v2"))
        if doc is None:
            continue
        for key in ("training_metrics", "collapse", "identity", "seed"):
            doc[key] = full.get(key)
        doc["run"], doc["dir"] = run.name, run
        docs.append(doc)
    return docs


def ensemble(docs: list[dict], ages: dict[str, float]) -> dict:
    """Score the seed ensemble (mean probability over seeds) like a single run."""
    sc = [d["scores"] for d in docs]
    first = sc[0]
    for s in sc[1:]:
        if s["cv_subjects"] != first["cv_subjects"] or s["test_subjects"] != first["test_subjects"]:
            raise ValueError("seeds scored different subjects")
    reps = min(len(s["cv_oof"]) for s in sc)
    oof = np.mean([np.asarray(s["cv_oof"])[:reps] for s in sc], axis=0)
    plan = np.asarray(first["plan"])[:reps]
    y = np.asarray(first["cv_y"])
    n_lab = first["n_lab"]
    age_all = np.array([ages.get(s, np.nan) for s in first["cv_subjects"]])
    out = summarize(y[:n_lab], y[n_lab:], oof, plan, age_all)
    p_test = np.mean([s["test_p"] for s in sc], axis=0)
    y_test = np.asarray(first["test_y"])
    age_test = np.array([ages[s] for s in first["test_subjects"]])
    out["test"] = held_out(y_test, p_test, out["threshold_rep"], age_test)
    matched = age_matched_subset(first["test_subjects"], y_test, ages)
    out["test_agem"] = held_out(y_test[matched], p_test[matched], out["threshold_rep"],
                                age_test[matched])
    out["_cv"] = (y, oof.mean(0), n_lab)
    out["_test"] = (y_test, p_test)
    return out


# ---------------------------------------------------------------- pseudo-tests (CV vs test)
# Moved from scripts/4_evaluate/analyze_cv_vs_test.py (also used by report_outer5.py).
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
        out[d] = auc(y[pick], s[pick])
    return out
