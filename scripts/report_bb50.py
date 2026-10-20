"""Backbone swap on split 361 (50 epochs), the 1:6 imbalanced test and the age-matched split.

    python scripts/report_bb50.py [--runs campaign_ssl3d_runpod/bb50]

Reads the pod tags downloaded into ``--runs`` (``bb50_*``: split_361_imb, ``am50_*``:
split_agematch; see campaign_ssl3d_runpod/bb50/phase{1,2}.sh):

- SSL ``<model>_<variant>__<recipe>__bb<backbone>__roi<roi>_s<seed>``: each method's
  champion recipe (views, ROI, heads, loss, optimizer) with every encoder;
- sin SSL ``vicreg_init__bb<backbone>__roi<roi>_s1``: the encoder SSL starts from
  (ImageNet / DeiT-inflated), frozen + probe; seed-independent, so one run serves 3;
- supervised ``<model>_supervised__<recipe>__bb<backbone>__roi<roi>_s<seed>``.

Every result carries the balanced test (43 HC + 41 AD) and ``test.extra_tests.imb``:
the same probe on test + 203 unused HC age-matched to the test HC (41 AD + 246 HC).
"Solo edad" is a class-balanced logistic regression on age alone (train -> test),
the share of the AD/HC separation that the age gap gives for free.

Writes ``<runs>/REPORT.md`` and figures ``campaign_ssl3d_runpod/figures/bb50_*.png``
and ``am50_*.png``. Groups that are missing (e.g. phase 2 not run) are skipped.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))
import plot_champions as champ  # noqa: E402
import plot_ssl3d_runpod as base  # noqa: E402

from t5_ssl import recipes  # noqa: E402
from t5_ssl.data.oasis3_lmdb_datamodule import clinical_rows  # noqa: E402

plt = base.plt
MODELS = ("vicreg", "byol", "dinov2")
BACKBONES = ("convnext_tiny", "resnet18", "vit_small")
BB = {"convnext_tiny": "ConvNeXt-T", "resnet18": "ResNet-18", "vit_small": "ViT-S"}
RECIPE = {"vicreg": ("campaign_augval", "geo", "32"), "byol": ("gn_dp0.3_dq0_wd0.01", "noise", "32"),
          "dinov2": ("campaign", "elastic_tio_gpu", "multi3")}
SUP_RECIPE = {"vicreg": "geo", "byol": "noise", "dinov2": "elastic_gpu"}
OWN = recipes.DEFAULT_BACKBONE
TICK = {"vicreg": "VICReg\ngeo · 32³", "byol": "BYOL\nnoise · 32³ · GN",
        "dinov2": "DINOv2\nelastic · 3 regiones"}
SUP_COLOR, AGE_COLOR = "#52514e", "#b3261e"
MANIFESTS = {"bb50": base.CAMPAIGN / "split_361_imb/manifest.csv",
             "am50": base.CAMPAIGN / "split_agematch/manifest.csv"}
# (where, metric) -> (title); where: cv, test (balanced), imb (test + test_imb, 1:6)
WHERE = {"cv": "CV (train+val)", "test": "Test balanceado (41 AD + 43 HC)",
         "imb": "Test 1:6 (41 AD + 246 HC)"}
METRIC = {"auc": "AUROC", "f1": "F1-macro", "aupr": "AUPR"}


def ssl_name(m: str, bb: str) -> str:
    variant, aug, roi = RECIPE[m]
    return f"{m}_{variant}__{aug}__bb{bb}__roi{roi}"


def init_name(m: str, bb: str) -> str:
    return f"vicreg_init__bb{bb}__roi{RECIPE[m][2]}"


def sup_name(m: str, bb: str) -> str:
    return f"{m}_supervised__{SUP_RECIPE[m]}__bb{bb}__roi{RECIPE[m][2]}"


def load(runs: Path, prefix: str) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for tag in sorted(runs.glob(f"{prefix}_*")):
        if tag.is_dir():
            for name, docs in base.load_dir(tag).items():
                groups[name] += docs
    return groups


# Age-matched subset of test + test_imb ("agem"), filled by main(): AD with up to two HC
# of nearly the same age, so age alone cannot separate the classes there.
AGE_MATCHED: set[str] = set()


def value(doc: dict, where: str, key: str) -> float:
    if where == "cv":
        return doc["cv"]["clean"]["metrics"][key]
    if where == "test":
        return doc["test"][{"auc": "test_auc", "f1": "test_f1_macro",
                            "aupr": "test_average_precision"}[key]]
    if where == "agem":
        rows = predictions(doc, "agem")
        labels = np.array([a for a, _ in rows.values()])
        scores = np.array([s for _, s in rows.values()])
        return {"auc": fast_auc, "aupr": fast_ap,
                "f1": lambda y, p: float(f1_score(y, p >= 0.5, average="macro"))}[key](labels, scores)
    extra = doc["test"]["extra_tests"][where]  # imb (test + test_imb), adni, ...
    return extra[{"auc": "auc", "f1": "f1_macro", "aupr": "average_precision"}.get(key, key)]


def predictions(doc: dict, where: str) -> dict[str, tuple[int, float]]:
    extras = doc["test"].get("extra_tests", {})
    if where == "test" or (where == "agem" and "imb" not in extras):
        rows = doc["test"]["test_predictions"]
    else:
        rows = extras["imb" if where == "agem" else where]["predictions"]
    if where == "agem":
        rows = [r for r in rows if r["subject"] in AGE_MATCHED]
    return {r["subject"]: (r["AD"], r["score"]) for r in rows}


def age_matched_test(manifest: Path, per_ad: int = 2, caliper: float = 2.5) -> dict:
    """Greedy nearest-age matching (oldest AD first) of HC to the AD of test + test_imb."""
    clinical = clinical_rows(base.ROOT / "data/dementia/mni-dataset.csv")
    by_idx = {r["idx"]: r for r in clinical}
    groups: dict[str, set[str]] = defaultdict(set)
    for r in clinical:
        groups[r["subject"]].add(r["group"])
    ages = {row["subject"]: float(by_idx[row["idx"]]["age"])
            for row in csv.DictReader(manifest.open(encoding="utf-8"))
            if row["split"] in ("test", "test_imb")}
    ad = sorted((s for s in ages if groups[s] == {"AD"}), key=lambda s: -ages[s])
    free = sorted(s for s in ages if groups[s] == {"HC"})
    matches: dict[str, list[str]] = {s: [] for s in ad}
    for _ in range(per_ad):
        for s in ad:
            best = min(free, key=lambda h: (abs(ages[h] - ages[s]), h), default=None)
            if best is not None and abs(ages[best] - ages[s]) <= caliper:
                matches[s].append(best)
                free.remove(best)
    kept_ad = [s for s, hc in matches.items() if hc]
    kept_hc = [h for s in kept_ad for h in matches[s]]
    return {"subjects": set(kept_ad) | set(kept_hc), "ad": len(kept_ad), "hc": len(kept_hc),
            "age_ad": float(np.mean([ages[s] for s in kept_ad])),
            "age_hc": float(np.mean([ages[h] for h in kept_hc])),
            "ages": ages}


def mean_sd(docs: list[dict], where: str, key: str) -> str:
    v = [value(d, where, key) for d in docs]
    return f"{np.mean(v):.3f} ± {np.std(v):.3f}" if len(v) > 1 else f"{v[0]:.3f}"


def fast_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """AUROC from average ranks (Mann-Whitney), as sklearn with ties."""
    from scipy.stats import rankdata
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


def bootstrap(a: list[dict], b: list[dict], where: str, metric: str = "auc",
              draws: int = 2000) -> tuple[float, float, float]:
    """Paired, class-stratified bootstrap over subjects of the seed-averaged difference."""
    score = fast_auc if metric == "auc" else fast_ap
    if where == "cv":
        labels = np.array(a[0]["labels"])
        sa = [np.asarray(d["cv"]["clean"]["oof_mean"]) for d in a]
        sb = [np.asarray(d["cv"]["clean"]["oof_mean"]) for d in b]
        if a[0]["subjects"] != b[0]["subjects"]:
            raise ValueError("CV subjects differ")
    else:
        subjects = sorted(predictions(a[0], where))
        labels = np.array([predictions(a[0], where)[s][0] for s in subjects])
        sa = [np.array([predictions(d, where)[s][1] for s in subjects]) for d in a]
        sb = [np.array([predictions(d, where)[s][1] for s in subjects]) for d in b]
    rng = np.random.default_rng(42)
    pos, neg = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    point = np.mean([score(labels, s) for s in sa]) - np.mean([score(labels, s) for s in sb])
    samples = []
    for _ in range(draws):
        idx = np.concatenate((rng.choice(pos, len(pos)), rng.choice(neg, len(neg))))
        samples.append(np.mean([score(labels[idx], s[idx]) for s in sa])
                       - np.mean([score(labels[idx], s[idx]) for s in sb]))
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def fmt(delta: tuple[float, float, float]) -> str:
    star = " *" if delta[1] > 0 or delta[2] < 0 else ""
    return f"{delta[0]:+.3f} [{delta[1]:+.3f}, {delta[2]:+.3f}]{star}"


# ---------------------------------------------------------------- age-only reference
def age_baseline(manifest: Path) -> dict[str, dict[str, float]]:
    """Class-balanced logistic regression on mean session age (labeled train -> test sets)."""
    clinical = clinical_rows(base.ROOT / "data/dementia/mni-dataset.csv")
    by_idx = {r["idx"]: r for r in clinical}
    groups: dict[str, set[str]] = defaultdict(set)
    for r in clinical:
        groups[r["subject"]].add(r["group"])
    ages: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in csv.DictReader(manifest.open(encoding="utf-8")):
        g = groups[row["subject"]]
        if row.get("labeled", "1") == "1" and len(g) == 1 and next(iter(g)) in ("AD", "HC"):
            ages[row["split"]][row["subject"]].append(float(by_idx[row["idx"]]["age"]))

    def xy(*splits: str) -> tuple[np.ndarray, np.ndarray]:
        subs = [(s, a) for sp in splits for s, a in sorted(ages[sp].items())]
        return (np.array([[np.mean(a)] for _, a in subs]),
                np.array([int(next(iter(groups[s])) == "AD") for s, _ in subs]))

    x, y = xy("train")
    probe = LogisticRegression(class_weight="balanced").fit(x, y)
    out = {}
    matched = {k: v for k, v in sorted(ages["test"].items()) + sorted(ages["test_imb"].items())
               if k in AGE_MATCHED}
    for where, splits in (("test", ("test",)), ("imb", ("test", "test_imb")), ("agem", None)):
        if splits is None:
            xt = np.array([[np.mean(a)] for a in matched.values()])
            yt = np.array([int(next(iter(groups[s])) == "AD") for s in matched])
        else:
            xt, yt = xy(*splits)
        p = probe.predict_proba(xt)[:, 1]
        out[where] = {"auc": float(roc_auc_score(yt, p)), "aupr": float(average_precision_score(yt, p)),
                      "f1": float(f1_score(yt, p >= 0.5, average="macro"))}
    return out


# ---------------------------------------------------------------- figures
def bars(ax, x: float, docs: list[dict], where: str, key: str, color: str, width: float,
         label: str | None, hatch: str | None, low: float) -> None:
    v = [value(d, where, key) for d in docs]
    ax.bar(x, np.mean(v), width=width, color=color, edgecolor="white", linewidth=1.2,
           label=label, hatch=hatch, alpha=0.9 if hatch else 1.0)
    if len(v) > 1:
        ax.scatter([x] * len(v), v, s=9, color=base.INK, zorder=3, linewidths=0)
    ax.text(x, low + 0.012, f"{np.mean(v):.3f}", ha="center", va="bottom", fontsize=6.3,
            color=base.INK, rotation=90)


def metrics_figure(g: dict, cells: list[tuple[str, str]], title: str, out: str,
                   age: dict, wheres: tuple[str, ...] = ("cv", "test", "imb")) -> None:
    """Rows = evaluation sets, columns = metrics, x = method; bars supervised, SSL, sin SSL."""
    kinds = (("sup", -0.27, None, "///", "Supervisado completo (end-to-end)"),
             ("ssl", 0.0, None, None, "SSL 50 épocas, congelado + probe lineal"),
             ("init", 0.27, base.BASE, None, "Sin SSL: mismo encoder antes del SSL + probe"))
    present = [(m, bb) for m, bb in cells if g["ssl"].get((m, bb))]
    if not present:
        return
    values = [value(d, w, k) for m, bb in present for kind in ("sup", "ssl", "init")
              for d in g[kind].get((m, bb), []) for w in wheres for k in METRIC]
    low = min(0.1, np.floor((min(values) - 0.04) * 20) / 20)
    fig, axes = plt.subplots(len(wheres), 3, figsize=(12.5, 3.2 * len(wheres) + 0.6), sharey=True)
    for r, where in enumerate(wheres):
        for c, key in enumerate(METRIC):
            ax = axes[r, c]
            for i, (m, bb) in enumerate(present):
                for kind, dx, color, hatch, label in kinds:
                    docs = g[kind].get((m, bb))
                    if docs:
                        bars(ax, i + dx, docs, where, key,
                             color or (SUP_COLOR if kind == "sup" else base.COLOR[m]), 0.26,
                             label if i == 0 else None, hatch, low)
            if where != "cv" and key in age.get(where, {}):
                ax.axhline(age[where][key], color=AGE_COLOR, linewidth=1.1, linestyle="--",
                           label="Solo edad (regresión logística)" if c == 0 else None)
            if where != "cv" and key == "aupr":
                labels = [a for a, _ in predictions(g["ssl"][present[0]][0], where).values()]
                chance = float(np.mean(labels))
                ax.axhline(chance, color=base.INK2, linewidth=1, linestyle=":")
                ax.text(-0.45, chance + 0.01, f"azar {chance:.2f}", ha="left",
                        fontsize=7, color=base.INK2)
            ticks = [TICK[m] + f"\n{BB[bb]}" + (" (original)" if OWN[m] == bb else "")
                     for m, bb in present]
            ax.set_xticks(range(len(present)), ticks, fontsize=7.2)
            ax.set_ylim(low, 0.99)
            ax.set_title(f"{METRIC[key]} — {WHERE[where]}", fontsize=9.2)
            base.style(ax)
    handles, labels = [], []
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels(), strict=True):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.suptitle(title + " (3 semillas; puntos = semillas)", fontsize=10.5, color=base.INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    base.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.FIGURES / out, dpi=160)
    plt.close(fig)
    print("wrote", out)


def matrix_figure(g: dict) -> None:
    panels = (("cv", "auc"), ("test", "auc"), ("imb", "aupr"))
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
    for ax, (where, key) in zip(axes, panels, strict=True):
        grid = np.full((len(MODELS) + 1, len(BACKBONES)), np.nan)
        for i, m in enumerate(MODELS):
            for j, bb in enumerate(BACKBONES):
                if docs := g["ssl"].get((m, bb)):
                    grid[i, j] = np.mean([value(d, where, key) for d in docs])
        for j, bb in enumerate(BACKBONES):  # sin SSL at 32³ (VICReg/BYOL input)
            if docs := g["init"].get(("vicreg", bb)):
                grid[-1, j] = value(docs[0], where, key)
        ax.imshow(grid, cmap="Blues", vmin=np.nanmin(grid) - 0.03, vmax=np.nanmax(grid) + 0.01)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if np.isfinite(grid[i, j]):
                    own = i < len(MODELS) and OWN[MODELS[i]] == BACKBONES[j]
                    ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", fontsize=9,
                            color=base.INK, fontweight="bold" if own else "normal")
        ax.set_xticks(range(len(BACKBONES)), [BB[b] for b in BACKBONES], fontsize=8.5)
        ax.set_yticks(range(len(MODELS) + 1), [base.LABEL[m] for m in MODELS]
                      + ["Sin SSL (32³)"], fontsize=8.5)
        ax.set_title(f"{METRIC[key]} — {WHERE[where]}", fontsize=9.2)
        ax.axhline(len(MODELS) - 0.5, color="white", linewidth=3)
    fig.suptitle("Método × backbone, split 361, 50 épocas (media de 3 semillas; negrita = "
                 "pareja original)", fontsize=10.5, color=base.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(base.FIGURES / "bb50_matriz.png", dpi=160)
    plt.close(fig)
    print("wrote bb50_matriz.png")


def backbone_figure(g: dict) -> None:
    """One line per method across the three encoders: CV, test, age-matched test, SSL gain."""
    panels = (("cv", "auc", "AUROC CV (split 361, 329 sujetos)", False),
              ("test", "auc", "AUROC test balanceado (84)", False),
              ("agem", "auc", "AUROC test emparejado por edad", False),
              ("test", "auc", "Ganancia del SSL: SSL − sin SSL (AUROC test)", True))
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9))
    x = np.arange(len(BACKBONES))
    for ax, (where, key, title, gain) in zip(axes, panels, strict=True):
        for k, m in enumerate(MODELS):
            means, sds = [], []
            for bb in BACKBONES:
                ssl = g["ssl"].get((m, bb), [])
                v = [value(d, where, key) for d in ssl]
                if gain and g["init"].get((m, bb)):
                    v = [u - value(g["init"][(m, bb)][0], where, key) for u in v]
                means.append(np.mean(v) if v else np.nan)
                sds.append(np.std(v) if v else np.nan)
            dx = (k - 1) * 0.06
            ax.errorbar(x + dx, means, yerr=sds, color=base.COLOR[m], marker="o", markersize=5,
                        linewidth=2, capsize=3, label=base.LABEL[m])
            own = BACKBONES.index(OWN[m])
            ax.plot(own + dx, means[own], "o", markersize=12, markerfacecolor="none",
                    markeredgecolor=base.COLOR[m], markeredgewidth=1.5)
        if gain:
            ax.axhline(0, color=base.INK2, linewidth=1)
        ax.set_xticks(x, [BB[b] for b in BACKBONES], fontsize=8.5)
        ax.set_title(title, fontsize=9.2)
        base.style(ax)
    axes[3].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("El backbone solo cambia mucho el resultado de DINOv2 (y de BYOL con ViT, que se "
                 "vuelve inestable); VICReg es plano. Media ± DE de 3 semillas; círculo = "
                 "pareja original", fontsize=10, color=base.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(base.FIGURES / "bb50_backbones.png", dpi=160)
    plt.close(fig)
    print("wrote bb50_backbones.png")


def threshold_figure(sets: dict[str, dict], out: str, title: str) -> None:
    """Sensitivity, specificity, PPV and F1(AD) at 0.5 on the 1:6 test, champions."""
    keys = (("sensitivity", "Sensibilidad"), ("specificity", "Especificidad"),
            ("ppv", "PPV (valor predictivo +)"), ("f1_ad", "F1 de la clase AD"))
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), sharey=True)
    labels = list(sets)
    for ax, (key, name) in zip(axes, keys, strict=True):
        for i, m in enumerate(MODELS):
            for k, label in enumerate(labels):
                docs = sets[label].get(m)
                if not docs:
                    continue
                v = [value(d, "imb", key) for d in docs]
                color = SUP_COLOR if "upervis" in label else (
                    base.BASE if "sin SSL" in label else base.COLOR[m])
                x = i + (k - (len(labels) - 1) / 2) * 0.26
                ax.bar(x, np.mean(v), width=0.26, color=color, edgecolor="white",
                       hatch="///" if "upervis" in label else None,
                       label=label if i == 0 else None)
                if len(v) > 1:
                    ax.scatter([x] * len(v), v, s=9, color=base.INK, zorder=3, linewidths=0)
                ax.text(x, 0.015, f"{np.mean(v):.2f}", ha="center", fontsize=6.3, rotation=90)
        ax.set_xticks(range(3), [base.LABEL[m] for m in MODELS], fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(name, fontsize=9.2)
        base.style(ax)
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.suptitle(title, fontsize=10.5, color=base.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(base.FIGURES / out, dpi=160)
    plt.close(fig)
    print("wrote", out)


# ---------------------------------------------------------------- report
def groups_of(runs: dict[str, list[dict]]) -> dict[str, dict]:
    return {"ssl": {(m, bb): runs[ssl_name(m, bb)] for m in MODELS for bb in BACKBONES
                    if ssl_name(m, bb) in runs},
            "init": {(m, bb): runs[init_name(m, bb)] for m in MODELS for bb in BACKBONES
                     if init_name(m, bb) in runs},
            "sup": {(m, bb): runs[sup_name(m, bb)] for m in MODELS for bb in BACKBONES
                    if sup_name(m, bb) in runs}}


def summary_rows(g: dict, cells: list[tuple[str, str]]) -> list[str]:
    lines = ["| Método | Backbone | Modelo | AUROC CV | AUROC test | AUROC 1:6 | AUPR 1:6 | "
             "F1-macro 1:6 | AUROC emp. edad |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for m, bb in cells:
        for kind, name in (("sup", "Supervisado"), ("ssl", "SSL"), ("init", "Sin SSL")):
            docs = g[kind].get((m, bb))
            if docs:
                lines.append(f"| {base.LABEL[m]} | {BB[bb]}{' (orig.)' if OWN[m] == bb else ''} | "
                             f"{name} | {mean_sd(docs, 'cv', 'auc')} | {mean_sd(docs, 'test', 'auc')} | "
                             f"{mean_sd(docs, 'imb', 'auc')} | {mean_sd(docs, 'imb', 'aupr')} | "
                             f"{mean_sd(docs, 'imb', 'f1')} | {mean_sd(docs, 'agem', 'auc')} |")
    return lines


def threshold_rows(sets: dict[str, dict]) -> list[str]:
    keys = ("sensitivity", "specificity", "ppv", "npv", "f1_ad", "balanced_accuracy", "mcc")
    lines = ["| Modelo | " + " | ".join(("Sens.", "Espec.", "PPV", "NPV", "F1 AD", "Acc. bal.",
                                          "MCC")) + " | PPV probe sin pesos |",
             "|---|" + "---:|" * (len(keys) + 1)]
    for label, by_model in sets.items():
        for m in MODELS:
            if docs := by_model.get(m):
                cells = [f"{np.mean([value(d, 'imb', k) for d in docs]):.3f}" for k in keys]
                plain = [d["test"]["extra_tests"]["imb"].get("unweighted_probe", {}).get("ppv")
                         for d in docs]
                plain_txt = f"{np.mean(plain):.3f}" if all(p is not None for p in plain) else "—"
                lines.append(f"| {base.LABEL[m]} — {label} | " + " | ".join(cells)
                             + f" | {plain_txt} |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=Path, default=base.CAMPAIGN / "bb50")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    phase1, phase2 = groups_of(load(args.runs, "bb50")), groups_of(load(args.runs, "am50"))
    matched = age_matched_test(MANIFESTS["bb50"])
    AGE_MATCHED.update(matched["subjects"])
    WHERE["agem"] = (f"Test emparejado por edad ({matched['ad']} AD + {matched['hc']} HC)")
    age = {k: age_baseline(p) for k, p in MANIFESTS.items() if p.is_file()}
    diag = [(m, OWN[m]) for m in MODELS]
    lines = ["# Backbones, test 1:6 y split emparejado por edad (50 épocas)", "",
             "Probe congelado como siempre (CV 5×5 en train+val; test con C elegido en "
             "validación); el mismo probe puntúa además el test 1:6 = test + 203 HC no usados "
             "con edades como los HC de test. Sin SSL = el encoder de partida (ImageNet / DeiT "
             "inflados), congelado. Supervisado = encoder + capa lineal end-to-end, época por "
             "AUROC de validación. Δ = bootstrap pareado y estratificado por clase sobre "
             "sujetos (IC 95 %, * = excluye 0). 3 semillas.", "",
             f"Test emparejado por edad: de test + test_imb, cada AD con hasta 2 HC de edad a "
             f"≤ 2.5 años (vecino más cercano, AD mayores primero): {matched['ad']} AD "
             f"({matched['age_ad']:.1f} años) + {matched['hc']} HC ({matched['age_hc']:.1f} años). "
             "Ahí la edad ya no separa las clases.", ""]
    for key, ref in age.items():
        lines.append(f"Solo edad ({key}): AUROC test {ref['test']['auc']:.3f}, AUROC 1:6 "
                     f"{ref['imb']['auc']:.3f}, AUPR 1:6 {ref['imb']['aupr']:.3f}, F1-macro 1:6 "
                     f"{ref['imb']['f1']:.3f}, AUROC emparejado por edad {ref['agem']['auc']:.3f}.")
    # ---- A: backbone factor
    if phase1["ssl"]:
        lines += ["", "## A. Mismo backbone para los tres métodos (split 361)", ""]
        cells = [(m, bb) for bb in BACKBONES for m in MODELS]
        lines += summary_rows(phase1, cells)
        lines += ["", "| Δ AUROC | CV | Test | Test 1:6 | Δ AUPR test 1:6 | Test emp. edad |",
                  "|---|---:|---:|---:|---:|---:|"]
        for m, bb in cells:
            ssl = phase1["ssl"].get((m, bb))
            if not ssl:
                continue
            refs = [("sin SSL", phase1["init"].get((m, bb))),
                    ("supervisado", phase1["sup"].get((m, bb)))]
            if bb != OWN[m]:
                refs.append((f"SSL con {BB[OWN[m]]}", phase1["ssl"].get((m, OWN[m]))))
            for ref_name, ref in refs:
                if ref:
                    cv = fmt(bootstrap(ssl, ref, "cv")) if ref[0]["subjects"] == ssl[0]["subjects"] else "—"
                    lines.append(f"| {base.LABEL[m]} {BB[bb]} − {ref_name} | {cv} | "
                                 f"{fmt(bootstrap(ssl, ref, 'test'))} | {fmt(bootstrap(ssl, ref, 'imb'))} | "
                                 f"{fmt(bootstrap(ssl, ref, 'imb', 'aupr'))} | "
                                 f"{fmt(bootstrap(ssl, ref, 'agem'))} |")
        for bb in BACKBONES:
            metrics_figure(phase1, [(m, bb) for m in MODELS],
                           f"Backbone {BB[bb]} 3D en los tres métodos, split 361, 50 épocas",
                           f"bb50_metricas_{bb}.png", age.get("bb50", {}))
        matrix_figure(phase1)
        backbone_figure(phase1)
    # ---- B: champions on the 1:6 test
    if all(phase1["ssl"].get(c) for c in diag):
        lines += ["", "## B. Test 1:6 con la configuración del split 361 (pareja original)", ""]
        lines += summary_rows(phase1, diag)
        sets = {"supervisado": {m: phase1["sup"].get((m, OWN[m])) for m in MODELS},
                "SSL": {m: phase1["ssl"][(m, OWN[m])] for m in MODELS},
                "sin SSL": {m: phase1["init"].get((m, OWN[m])) for m in MODELS}}
        lines += ["", "Umbral 0.5 en el test 1:6:", "", *threshold_rows(sets)]
        metrics_figure(phase1, diag, "Pareja original, split 361, 50 épocas: test balanceado "
                       "vs test 1:6 vs test emparejado por edad", "bb50_desbalanceado.png",
                       age.get("bb50", {}), wheres=("test", "imb", "agem"))
        threshold_figure(sets, "bb50_umbral_1a6.png",
                         "Test 1:6 (41 AD + 246 HC), umbral 0.5 del probe balanceado")
        champ.CHAMPION.update({m: ssl_name(m, OWN[m]) for m in MODELS})
        champ.PREFIX, champ.TITLE = "bb50_", "split 361, 50 épocas"
        champ.TICK.update(TICK)
        by_name = {ssl_name(m, OWN[m]): phase1["ssl"][(m, OWN[m])] for m in MODELS}
        champ.curves_figures(by_name)
        champ.collapse_figure(by_name)
    # ---- C: age-matched split
    if phase2["ssl"]:
        lines += ["", "## C. Split emparejado por edad 55/45 con sujetos solo-SSL", ""]
        lines += summary_rows(phase2, diag)
        sets = {"supervisado": {m: phase2["sup"].get((m, OWN[m])) for m in MODELS},
                "SSL": {m: phase2["ssl"].get((m, OWN[m])) for m in MODELS},
                "sin SSL": {m: phase2["init"].get((m, OWN[m])) for m in MODELS}}
        lines += ["", "Umbral 0.5 en el test 1:6:", "", *threshold_rows(sets)]
        lines += ["", "Dentro del split emparejado (mismos sujetos de CV):", "",
                  "| Δ AUROC | CV | Test | Test 1:6 | Test emp. edad |", "|---|---:|---:|---:|---:|"]
        for m in MODELS:
            g2 = {k: phase2[k].get((m, OWN[m])) for k in ("ssl", "sup", "init")}
            for a_kind, b_kind, name in (("ssl", "init", "SSL − sin SSL"),
                                         ("ssl", "sup", "SSL − supervisado"),
                                         ("sup", "init", "supervisado − sin SSL")):
                a, b = g2[a_kind], g2[b_kind]
                if a and b:
                    cv = fmt(bootstrap(a, b, "cv")) if a[0]["subjects"] == b[0]["subjects"] else "—"
                    lines.append(f"| {base.LABEL[m]} {name} | {cv} | {fmt(bootstrap(a, b, 'test'))} | "
                                 f"{fmt(bootstrap(a, b, 'imb'))} | {fmt(bootstrap(a, b, 'agem'))} |")
        lines += ["", "Split emparejado − split 361 (mismos sujetos de test; la CV no es "
                  "comparable: otros sujetos):", "",
                  "| Δ | AUROC test | AUROC 1:6 | AUPR 1:6 | AUROC emp. edad |",
                  "|---|---:|---:|---:|---:|"]
        for m in MODELS:
            for kind, name in (("ssl", "SSL"), ("sup", "supervisado"), ("init", "sin SSL")):
                a, b = phase2[kind].get((m, OWN[m])), phase1[kind].get((m, OWN[m]))
                if a and b:
                    lines.append(f"| {base.LABEL[m]} {name} | {fmt(bootstrap(a, b, 'test'))} | "
                                 f"{fmt(bootstrap(a, b, 'imb'))} | "
                                 f"{fmt(bootstrap(a, b, 'imb', 'aupr'))} | "
                                 f"{fmt(bootstrap(a, b, 'agem'))} |")
        metrics_figure(phase2, diag, "Split emparejado por edad 55/45 + sujetos solo-SSL, "
                       "50 épocas", "am50_metricas.png", age.get("am50", {}),
                       wheres=("cv", "test", "imb", "agem"))
        threshold_figure(sets, "am50_umbral_1a6.png",
                         "Split emparejado: test 1:6, umbral 0.5 del probe balanceado")
        if all(phase2["ssl"].get(c) for c in diag):
            champ.PREFIX, champ.TITLE = "am50_", "split emparejado, 50 épocas"
            champ.curves_figures({ssl_name(m, OWN[m]): phase2["ssl"][(m, OWN[m])] for m in MODELS})
    (args.runs / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
