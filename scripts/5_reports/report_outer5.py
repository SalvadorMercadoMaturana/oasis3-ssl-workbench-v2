#!/usr/bin/env python
"""Report of the 2026-11-16 campaign: pretext tasks, tuning and the outer 5-fold split.

    python scripts/5_reports/report_outer5.py [--pretext pretext_tune:pretext_rot10_jig100_lr0.0003__geo]

Reads the downloaded runs in ``campaign_ssl3d_runpod/outer5/runs`` (tags pretext_screen,
pretext_tune, tune_lrx*, outer_f1..4[_init]) and the fold-0 references in
``campaign_ssl3d_runpod/deploy/runs``; writes ``campaign_ssl3d_runpod/outer5/REPORT.md``,
``summary.json`` and ``figures/outer5_*.png``. Every number is a 3-seed ensemble scored with
protocol v2 (src/t5_ssl/evaluation/protocol_v2.py).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import plot_ssl3d_runpod as base  # noqa: E402
from report_v2fix import boot_delta, fmt_delta  # noqa: E402

from t5_ssl.evaluation import protocol_v2 as pp  # noqa: E402
from t5_ssl.evaluation.protocol_v2 import pseudo_tests  # noqa: E402

plt = base.plt
CAMPAIGN = base.CAMPAIGN
OLD = CAMPAIGN / "deploy/runs"
NEW = CAMPAIGN / "outer5/runs"
OUT = CAMPAIGN / "outer5"
FOLDS = range(5)
SUFFIX = "__bbconvnext_tiny__roi32"
# key: (label, colour, fold-0 (root, tag), run name)
REFERENCE = {
    "byol": ("BYOL ConvNeXt-T BN (desplegado)", "#eb6834", (OLD, "v2_ssl"),
             "byol_wide_dproj0.1__noise" + SUFFIX),
    "vicreg": ("VICReg ConvNeXt-T", "#2a78d6", (OLD, "v2_ssl"),
               "vicreg_campaign_augval__geo" + SUFFIX),
    "dino_cnx": ("DINOv2 ConvNeXt-T", "#1baf7a", (OLD, "dino_fix"),
                 "dinov2_b64_k256__elastic_tio_gpu" + SUFFIX),
    "dino_vit": ("DINOv2 ViT-S", "#eda100", (OLD, "dino_fix"),
                 "dinov2_campaign__elastic_tio_gpu__bbvit_small__roi32"),
    "init_cnx": ("sin SSL ConvNeXt-T", "#9c9a92", (OLD, "dep50_init"),
                 "vicreg_init" + SUFFIX),
    "init_vit": ("sin SSL ViT-S", "#cfcdc6", (OLD, "dino_fix_init"),
                 "dinov2_init__bbvit_small__roi32"),
    # end-to-end AD/HC training of the same ConvNeXt-T start with the geo views
    # (scripts/3_train/train_supervised.py); folds 1-4 under outer_f<k>_sup
    "sup_cnx": ("Supervisado completo ConvNeXt-T", "#5f6b73", (OLD, "v2_sup"),
                "vicreg_supervised__geo" + SUFFIX),
}
PRETEXT_COLORS = ("#6250d6", "#e87ba4")
# figure / table order (adjacent hues validated with the dataviz palette validator)
ORDER = ("vicreg", "byol", "dino_cnx", "dino_vit", "init_cnx", "init_vit", "sup_cnx")
TASK_LABEL = {"rot4": "rotación 4", "rot10": "rotación 10", "rot24": "rotación 24",
              "jig24": "orden 24", "jig100": "orden 100"}


def manifest(k: int) -> Path:
    return (CAMPAIGN / "split_deploy/manifest.csv" if k == 0
            else CAMPAIGN / f"split_outer5/fold{k}/manifest.csv")


AGES = {k: pp.session_ages(manifest(k)) for k in FOLDS}


def fold_source(key: str, k: int, pretext: tuple[str, str] | None = None
                ) -> tuple[Path, str, str]:
    if key.startswith("pretext"):
        tag, name = pretext
        return (NEW, tag, name) if k == 0 else (NEW, f"outer_f{k}", name)
    _, _, (root, tag), name = REFERENCE[key]
    if k == 0:
        return root, tag, name
    if key.startswith("sup"):
        return NEW, f"outer_f{k}_sup", name
    return NEW, f"outer_f{k}_init" if key.startswith("init") else f"outer_f{k}", name


def ensemble(root: Path, tag: str, name: str, k: int) -> dict | None:
    docs = pp.load_runs(root, tag, name)
    if not docs:
        return None
    ens = pp.ensemble(docs, AGES[k])
    ens["n_seeds"] = len(docs)
    ens["docs"] = docs
    adni = [d["adni"]["scores"] for d in docs if "adni" in d and "scores" in d["adni"]]
    if adni:
        y = np.asarray(adni[0]["y"])
        p = np.mean([a["p"] for a in adni], axis=0)
        ens["adni"] = pp.held_out(y, p, ens["threshold_rep"])
    return ens


def last_val(doc: dict, column: str) -> float | None:
    path = Path(doc["dir"]) / "metrics.csv"
    if not path.is_file():
        return None
    values = [r[column] for r in csv.DictReader(path.open()) if r.get(column)]
    return float(values[-1]) if values else None


def label_of(name: str) -> str:
    variant, aug = name[len("pretext_"):].split("__")[:2]
    parts = variant.split("_")
    tasks = " + ".join(TASK_LABEL[p] for p in parts if p in TASK_LABEL)
    extra = " ".join(p for p in parts if p not in TASK_LABEL)
    return f"Pretexto {tasks}" + (f" · {extra}" if extra else "") + f" · {aug}"


# ---------------------------------------------------------------- fold 0: screen and tuning
def fold0_table(byol: dict) -> tuple[list[dict], list[str]]:
    rows = []
    for tag in ("pretext_screen", "pretext_tune"):
        for run in sorted({p.name.rsplit("_s", 1)[0] for p in (NEW / tag).glob("pretext_*_s*")}):
            ens = ensemble(NEW, tag, run, 0)
            if ens is not None:
                rows.append({"label": label_of(run), "tag": tag, "name": run, "ens": ens,
                             "family": "pretext"})
    for factor_dir in sorted(NEW.glob("tune_lrx*")):
        for key in ("vicreg", "byol", "dino_cnx"):
            ens = ensemble(NEW, factor_dir.name, REFERENCE[key][3], 0)
            if ens is not None:
                rows.append({"label": f"{REFERENCE[key][0]} · lr {factor_dir.name[7:]}",
                             "tag": factor_dir.name, "name": REFERENCE[key][3], "ens": ens,
                             "family": key})
    for key in REFERENCE:
        ens = ensemble(*fold_source(key, 0), 0)
        if ens is not None:
            rows.append({"label": REFERENCE[key][0] + " (referencia)", "tag": "ref", "name": key,
                         "ens": ens, "family": key})
    rows.sort(key=lambda r: -r["ens"]["cv_rep"]["mean3"])
    y, s_b, _ = byol["_cv"]
    lines = ["| Modelo | Semillas | CV AUROC | CV AUPR | CV F1 | **CV compuesto** | Δ compuesto vs BYOL "
             "desplegado [IC 95 %] | Test AUROC | Test compuesto | ADNI AUROC | acierto pretexto "
             "(val) |", "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---|"]
    for r in rows:
        e = r["ens"]
        _, s_a, _ = e["_cv"]
        delta = boot_delta(y, s_a, s_b, e["threshold_rep"], byol["threshold_rep"], draws=1000)
        acc = []
        for col, short in (("val/rot_acc", "rot"), ("val/jig_acc", "orden")):
            vals = [v for v in (last_val(d, col) for d in e["docs"]) if v is not None]
            if vals:
                acc.append(f"{short} {np.mean(vals):.2f}")
        adni = e.get("adni", {}).get("auc")
        lines.append(f"| {r['label']} | {e['n_seeds']} | {e['cv_rep']['auc']:.3f} | "
                     f"{e['cv_rep']['aupr']:.3f} | {e['cv_rep']['f1_macro']:.3f} | "
                     f"**{e['cv_rep']['mean3']:.3f}** | "
                     f"{fmt_delta(delta['mean3']) if r['name'] != 'byol' else '—'} | "
                     f"{e['test']['auc']:.3f} | {e['test']['mean3']:.3f} | "
                     f"{'' if adni is None else f'{adni:.3f}'} | {', '.join(acc)} |")
        r["delta_mean3"] = delta["mean3"]
    return rows, lines


# ---------------------------------------------------------------- outer folds
def outer(key: str, pretext: tuple[str, str] | None) -> dict | None:
    folds = {}
    for k in FOLDS:
        ens = ensemble(*fold_source(key, k, pretext), k)
        if ens is None:
            return None
        folds[k] = ens
    out = {"folds": folds}
    for where in ("cv_rep", "test"):
        for m in ("auc", "aupr", "f1_macro", "mean3"):
            v = [folds[k][where][m] for k in FOLDS]
            out[f"{where}_{m}"] = (float(np.mean(v)), float(np.std(v, ddof=1)))
    y = np.concatenate([folds[k]["_test"][0] for k in FOLDS])
    p = np.concatenate([folds[k]["_test"][1] for k in FOLDS])
    shift = np.concatenate([folds[k]["_test"][1] - folds[k]["threshold_rep"] for k in FOLDS])
    out["pooled_test"] = {"n": len(y), "ad": int(y.sum()), "auc": pp.auc(y, p),
                          "aupr": pp.aupr(y, p),
                          "f1_macro": pp.at_threshold(y, shift, 0.0)["f1_macro"]}
    out["gap_auc"] = [folds[k]["test"]["auc"] - folds[k]["cv_rep"]["auc"] for k in FOLDS]
    out["gap_mean3"] = [folds[k]["test"]["mean3"] - folds[k]["cv_rep"]["mean3"] for k in FOLDS]
    out["pseudo_band"] = []
    for k in FOLDS:
        y_cv, s_cv, _ = folds[k]["_cv"]
        subjects = folds[k]["docs"][0]["scores"]["cv_subjects"]
        age = np.array([AGES[k][s] for s in subjects])
        y_t = folds[k]["_test"][0]
        band = pseudo_tests(y_cv, s_cv, age, int(y_t.sum()), int((y_t == 0).sum()), draws=1000)
        out["pseudo_band"].append([float(np.percentile(band, 2.5)), float(np.percentile(band, 97.5))])
    return out


def fold_delta(a: dict, b: dict, draws: int = 1000) -> dict[str, tuple[float, float, float]]:
    """Mean over folds of metric(a) - metric(b) on each test fold, stratified bootstrap."""
    rng = np.random.default_rng(0)
    keys = ("auc", "aupr", "f1_macro", "mean3")

    def metrics(y: np.ndarray, s: np.ndarray, t: float) -> dict[str, float]:
        m = {"auc": pp.auc(y, s), "aupr": pp.aupr(y, s),
             "f1_macro": pp.at_threshold(y, s, t)["f1_macro"]}
        m["mean3"] = (m["auc"] + m["aupr"] + m["f1_macro"]) / 3
        return m

    data = []
    for k in FOLDS:
        y, pa = a["folds"][k]["_test"]
        _, pb = b["folds"][k]["_test"]
        data.append((y, pa, pb, a["folds"][k]["threshold_rep"], b["folds"][k]["threshold_rep"]))
    point = {m: np.mean([metrics(y, pa, ta)[m] - metrics(y, pb, tb)[m]
                         for y, pa, pb, ta, tb in data]) for m in keys}
    samples = {m: [] for m in keys}
    for _ in range(draws):
        diffs = {m: [] for m in keys}
        for y, pa, pb, ta, tb in data:
            pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
            idx = np.concatenate((rng.choice(pos, len(pos)), rng.choice(neg, len(neg))))
            ma, mb = metrics(y[idx], pa[idx], ta), metrics(y[idx], pb[idx], tb)
            for m in keys:
                diffs[m].append(ma[m] - mb[m])
        for m in keys:
            samples[m].append(np.mean(diffs[m]))
    return {m: (float(point[m]), *np.percentile(samples[m], [2.5, 97.5])) for m in keys}


def figure_gap(models: dict[str, dict], labels: dict[str, str], colors: dict[str, str]) -> None:
    keys = list(models)
    fig, ax = plt.subplots(figsize=(12, 5.2))
    top = max(hi - o["folds"][k]["cv_rep"]["auc"] for o in models.values()
              for k, (_, hi) in enumerate(o["pseudo_band"]))
    for i, key in enumerate(keys):
        o = models[key]
        for k in FOLDS:
            lo, hi = o["pseudo_band"][k]
            cv = o["folds"][k]["cv_rep"]["auc"]
            x = i + (k - 2) * 0.13
            ax.plot([x, x], [lo - cv, hi - cv], color=base.INK2, linewidth=1, alpha=0.35)
            ax.scatter([x], [o["gap_auc"][k]], s=34, color=colors[key], edgecolor="white",
                       linewidth=1.5, zorder=3)
        mean = float(np.mean(o["gap_auc"]))
        ax.plot([i - 0.33, i + 0.33], [mean, mean], color=base.INK, linewidth=2)
        ax.text(i, top + 0.012, f"media {mean:+.3f}", ha="center", va="bottom", fontsize=8,
                color=base.INK)
    ax.axhline(0, color=base.INK2, linewidth=1, linestyle=":")
    ax.set_ylim(top=top + 0.03)
    ax.set_xticks(range(len(keys)), [labels[k].replace(" (", "\n(").replace(" · ", "\n")
                                     for k in keys], fontsize=8)
    ax.set_ylabel("AUROC test − AUROC CV representativa")
    ax.set_title("Brecha test − CV en cada pliegue externo\npunto = pliegue 0..4; barra gris = 95 % "
                 "de pseudo-tests del mismo tamaño sacados de la CV; línea = media", fontsize=9.5)
    base.style(ax)
    fig.tight_layout()
    fig.savefig(base.FIGURES / "outer5_brecha.png", dpi=150)
    plt.close(fig)


def figure_metrics(models: dict[str, dict], labels: dict[str, str], colors: dict[str, str]) -> None:
    keys = list(models)
    metrics = (("auc", "AUROC"), ("aupr", "AUPR"), ("f1_macro", "F1-macro"), ("mean3", "Compuesto"))
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.8), sharey=True)
    for ax, (m, title) in zip(axes, metrics, strict=True):
        for i, key in enumerate(keys):
            for j, (where, alpha) in enumerate((("cv_rep", 0.55), ("test", 1.0))):
                mean, sd = models[key][f"{where}_{m}"]
                x = i + (j - 0.5) * 0.4
                ax.bar(x, mean, width=0.38, color=colors[key], alpha=alpha, edgecolor="white")
                ax.errorbar(x, mean, yerr=sd, color=base.INK, linewidth=1, capsize=2)
                ax.text(x, mean + sd + 0.012, f"{mean:.3f}", ha="center", va="bottom",
                        fontsize=6.5, rotation=90, color=base.INK)
        ax.set_xticks(range(len(keys)), [labels[k].split(" (")[0] for k in keys], rotation=35,
                      ha="right", fontsize=7.5)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0.3, 1.08)
        base.style(ax)
    axes[0].set_ylabel("media de 5 pliegues externos (± DE)")
    fig.suptitle("Split externo de 5 pliegues: CV representativa (barra pálida) vs test (barra "
                 "llena); ensamble de 3 semillas por pliegue", fontsize=10.5, color=base.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(base.FIGURES / "outer5_metricas.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretext", action="append", default=[], help="tag:run name of a pretext "
                    "model of the outer folds (fold 0 under that tag, folds 1-4 under outer_f<k>); "
                    "repeatable")
    args = ap.parse_args()
    pretexts = {("pretext" if i == 0 else f"pretext{i + 1}"): tuple(v.split(":"))
                for i, v in enumerate(args.pretext)}
    byol0 = ensemble(*fold_source("byol", 0), 0)
    rows, fold0_lines = fold0_table(byol0)
    labels = {k: v[0] for k, v in REFERENCE.items()}
    colors = {k: v[1] for k, v in REFERENCE.items()}
    keys = list(ORDER)
    for i, (key, (_, name)) in enumerate(pretexts.items()):
        labels[key] = label_of(name)
        colors[key] = PRETEXT_COLORS[i % len(PRETEXT_COLORS)]
        keys.insert(i, key)
    models = {k: o for k in keys if (o := outer(k, pretexts.get(k))) is not None}
    lines = ["# Tareas pretexto, tuning y split externo de 5 pliegues (2026-11-16)", "",
             "Ensambles de 3 semillas, protocolo v2 (CV representativa = sujetos AD/HC fuera del "
             "test, prevalencia 23 %, edades reales; compuesto = (AUROC + AUPR + F1-macro)/3 con "
             "umbral F1 elegido fuera de muestra).", "",
             "## 1. Pliegue 0 (split_deploy): tareas pretexto y tuning", "",
             "Ordenado por compuesto en la CV representativa (criterio de selección; el test solo "
             "confirma). Δ = bootstrap pareado estratificado sobre la CV (702 sujetos); * = IC sin "
             "el 0.", "", *fold0_lines, ""]
    if models:
        ref, ref_init = models.get("byol"), models.get("init_cnx")
        lines += ["## 2. Split externo de 5 pliegues (cada sujeto AD/HC es test una vez)", "",
                  "Media ± DE sobre los 5 pliegues externos; *test agrupado* = las 880 "
                  "predicciones de test de los 5 pliegues juntas (203 AD + 677 HC).", "",
                  "| Modelo | CV AUROC | Test AUROC | CV compuesto | Test compuesto | "
                  "test − CV AUROC (media; por pliegue) | Test agrupado AUROC / AUPR / F1 | "
                  "Δ test compuesto vs BYOL [IC 95 %] | Δ test compuesto vs sin SSL ConvNeXt-T "
                  "[IC 95 %] |", "|---|---|---|---|---|---|---|---|---|"]
        for key, o in models.items():
            pooled = o["pooled_test"]
            delta = "—" if key == "byol" or ref is None else fmt_delta(fold_delta(o, ref)["mean3"])
            d_init = ("—" if key.startswith("init") or ref_init is None
                      else fmt_delta(fold_delta(o, ref_init)["mean3"]))
            lines.append(
                f"| {labels[key]} | {o['cv_rep_auc'][0]:.3f} ± {o['cv_rep_auc'][1]:.3f} | "
                f"{o['test_auc'][0]:.3f} ± {o['test_auc'][1]:.3f} | "
                f"{o['cv_rep_mean3'][0]:.3f} ± {o['cv_rep_mean3'][1]:.3f} | "
                f"{o['test_mean3'][0]:.3f} ± {o['test_mean3'][1]:.3f} | "
                f"{np.mean(o['gap_auc']):+.3f} ({', '.join(f'{g:+.2f}' for g in o['gap_auc'])}) | "
                f"{pooled['auc']:.3f} / {pooled['aupr']:.3f} / {pooled['f1_macro']:.3f} | {delta} "
                f"| {d_init} |")
        ssl = [k for k in models if not k.startswith("init")]
        per_fold = np.mean([models[k]["gap_auc"] for k in ssl], axis=0)
        inside = np.mean([lo - models[k]["folds"][f]["cv_rep"]["auc"] <= models[k]["gap_auc"][f]
                          <= hi - models[k]["folds"][f]["cv_rep"]["auc"]
                          for k in ssl for f, (lo, hi) in enumerate(models[k]["pseudo_band"])])
        lines += ["", f"Brecha test − CV (AUROC) promediada sobre los {len(ssl)} modelos SSL, por "
                  f"pliegue: {', '.join(f'{g:+.3f}' for g in per_fold)}; media de pliegues "
                  f"{per_fold.mean():+.3f} ± {per_fold.std(ddof=1) / np.sqrt(len(per_fold)):.3f} "
                  f"(EE); {inside:.0%} de las brechas por pliegue y modelo caen dentro de la banda "
                  "95 % de pseudo-tests del mismo tamaño."]
        figure_gap(models, labels, colors)
        figure_metrics(models, labels, colors)
        lines += ["", "![brecha](../figures/outer5_brecha.png)", "",
                  "![métricas](../figures/outer5_metricas.png)", ""]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {"fold0": [{"label": r["label"], "tag": r["tag"], "name": r["name"],
                          "cv_mean3": r["ens"]["cv_rep"]["mean3"], "cv_auc": r["ens"]["cv_rep"]["auc"],
                          "test_mean3": r["ens"]["test"]["mean3"], "test_auc": r["ens"]["test"]["auc"],
                          "delta_mean3_vs_byol": r["delta_mean3"]} for r in rows],
               "outer": {k: {m: v for m, v in o.items() if m not in ("folds",)}
                         for k, o in models.items()}}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    print(OUT / "REPORT.md")


if __name__ == "__main__":
    main()
