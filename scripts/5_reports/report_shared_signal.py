#!/usr/bin/env python
"""Why sin SSL and the supervised model follow the SSL encoders (report of 2026-11-21).

    python scripts/5_reports/report_shared_signal.py [--baselines campaign_ssl3d_runpod/outer5/baselines]

Inputs: the outer 5-fold seed ensembles of ``report_outer5.py`` (protocol v2) and the
trivial baselines of ``scripts/4_evaluate/trivial_baselines.py``
(``campaign_ssl3d_runpod/outer5/baselines``: cube intensity statistics, FreeSurfer
volumes, random-weight ConvNeXt-T, recomputed ImageNet ConvNeXt-T).

Outputs: ``docs/figures_senal_compartida/`` (figures and the LaTeX tables and numbers that
``docs/informe_senal_compartida.tex`` reads) and
``campaign_ssl3d_runpod/outer5/baselines/summary.json``.

Analyses:

1. one table of every model and baseline (mean of the 5 outer folds);
2. fold co-movement: test AUROC per fold, correlation of each row with the SSL mean,
   fold/model variance shares, and the CV -> test gap of every fold;
3. per-subject agreement on the pooled test (880 subjects, within-fold percentile ranks):
   Spearman correlations, AUROC among AD-HC pairs of similar hippocampal volume, share of
   each score's ranking explained by the volume;
4. fold 0 under protocol v1 and v2 (the earlier "jump" of sin SSL).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import report_outer5 as ro  # noqa: E402

from t5_ssl.evaluation import protocol_v2 as pp  # noqa: E402

plt = ro.base.plt
INK, INK2, MUTED, GRID = ro.base.INK, ro.base.INK2, ro.base.MUTED, ro.base.GRID
ROOT = HERE.parents[1]
FIG = ROOT / "docs/figures_senal_compartida"
BASE_DIR = ro.CAMPAIGN / "outer5/baselines"
PRETEXT = ("pretext_tune", "pretext_rot24_h2048__geo__bbconvnext_tiny__roi32")
# key, label, colour (the colours of docs/figures_final/metricas.png)
MODELS = [("pretext", "Rotación", "#e87ba4"), ("vicreg", "VICReg", "#2a78d6"),
          ("dino_cnx", "DINOv2", "#1baf7a"), ("byol", "BYOL", "#eb6834"),
          ("sup_cnx", "Supervisado completo", "#5f6b73"),
          ("init_cnx", "Sin SSL (ImageNet congelado)", "#9c9a92")]
SSL = ("pretext", "vicreg", "dino_cnx", "byol")
BASELINES = [("stats", "Estadísticas de intensidad del cubo (38 números)"),
             ("aseg", "Volúmenes FreeSurfer (15 números)"),
             ("hippo_vol", "Volumen hipocampal izq./der. (2 números)"),
             ("random_s1", "ConvNeXt-T con pesos aleatorios, semilla 1"),
             ("random_s2", "ConvNeXt-T con pesos aleatorios, semilla 2")]
MARKERS = [("vol", "Volumen hipocampal, sin probe (menor = AD)"),
           ("age", "Edad, sin probe (mayor = AD)")]
LABEL = {k: v for k, v, *_ in MODELS + BASELINES + MARKERS}
SHORT = LABEL | {"stats": "Estadísticas de intensidad (38)", "aseg": "Volúmenes FreeSurfer (15)",
                 "hippo_vol": "Volumen hipocampal I/D (2)",
                 "random_s1": "ConvNeXt-T aleatorio, semilla 1",
                 "random_s2": "ConvNeXt-T aleatorio, semilla 2",
                 "vol": "Volumen hipocampal (sin probe)", "age": "Edad (sin probe)"}  # tables
COLOR = {k: c for k, _, c in MODELS} | {"stats": "#6250d6", "vol": INK}
AD_C, HC_C = "#e34948", "#6250d6"
METRICS = ("cv_auc", "cv_matched_auc", "test_auc", "test_aupr", "test_mean3", "test_auc_age",
           "agem_auc")
CALIPER_SD = 0.25


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- loading
def from_doc(cv_rep: dict, cv_matched: dict, test: dict, agem: dict) -> dict:
    return {"cv_auc": cv_rep["auc"], "cv_matched_auc": cv_matched["auc"], "test_auc": test["auc"],
            "test_aupr": test["aupr"], "test_mean3": test["mean3"],
            "test_auc_age": test["auc_age"], "agem_auc": agem["auc"]}


def load_models() -> dict:
    out = {}
    for key, *_ in MODELS:
        for k in ro.FOLDS:
            e = ro.ensemble(*ro.fold_source(key, k, PRETEXT), k)
            row = from_doc(e["cv_rep"], e["cv_matched"], e["test"], e["test_agem"])
            row["test_subjects"] = e["docs"][0]["scores"]["test_subjects"]
            row["test_y"], row["test_p"] = (np.asarray(v) for v in e["_test"])
            row["seeds"] = e["n_seeds"]
            out[key, k] = row
    return out


def load_baselines() -> dict:
    docs = json.loads((BASE_DIR / "protocol_v2.json").read_text(encoding="utf-8"))
    out = {}
    for key, _ in BASELINES:
        for k in ro.FOLDS:
            d = docs[f"{key}|{k}"]
            row = from_doc(d["cv_rep"], d["cv_matched"], d["test"], d["test_agem"])
            row["test_subjects"] = d["scores"]["test_subjects"]
            row["test_y"], row["test_p"] = np.asarray(d["scores"]["test_y"]), np.asarray(d["scores"]["test_p"])
            row["missing"] = d["missing"]
            out[key, k] = row
    return out


def volumes() -> dict[str, float]:
    """Hippocampal volume (left + right voxels) of every cached session idx."""
    f = np.load(BASE_DIR / "features.npz")
    names = list(f["aseg_names"])
    v = f["aseg"][:, names.index("hippocampus_L")] + f["aseg"][:, names.index("hippocampus_R")]
    return dict(zip(f["idx"], v.tolist(), strict=True))


def primary_idx(k: int) -> dict[str, str]:
    import csv
    with ro.manifest(k).open(encoding="utf-8") as stream:
        return {r["subject"]: r["idx"] for r in csv.DictReader(stream) if r["primary"] == "1"}


def marker_rows(models: dict, vol_of_idx: dict[str, float]) -> dict:
    """Age and hippocampal volume used directly as the score (no probe, no training)."""
    out = {}
    for k in ro.FOLDS:
        doc = pp.load_runs(*ro.fold_source("init_cnx", k)[:3])[0]["scores"]
        prim, ages = primary_idx(k), ro.AGES[k]
        parts = {"test": (doc["test_subjects"], np.asarray(doc["test_y"])),
                 "cv": (doc["cv_subjects"], np.asarray(doc["cv_y"])),
                 "matched": (doc["cv_subjects"][:doc["n_lab"]], np.asarray(doc["cv_y"])[:doc["n_lab"]])}
        for key in ("vol", "age"):
            row = {}
            for part, (subjects, y) in parts.items():
                a = np.array([ages[s] for s in subjects])
                s = (-np.array([vol_of_idx[prim[x]] for x in subjects]) if key == "vol" else a)
                ok = np.isfinite(s)
                m = pp.rank_metrics(y[ok], s[ok], a[ok])
                if part == "test":
                    row.update(test_auc=m["auc"], test_aupr=m["aupr"], test_auc_age=m["auc_age"],
                               test_mean3=np.nan, test_n=int(ok.sum()))
                    sub = [x for x, keep in zip(subjects, ok, strict=True) if keep]
                    agem = pp.age_matched_subset(sub, y[ok], ages)
                    row["agem_auc"] = pp.auc(y[ok][agem], s[ok][agem])
                    row["test_subjects"], row["test_y"], row["test_p"] = subjects, y, s
                elif part == "cv":
                    row["cv_auc"] = m["auc"]
                else:
                    row["cv_matched_auc"] = m["auc"]
            out[key, k] = row
    return out


# ---------------------------------------------------------------- analyses
def fold_mean(rows: dict, key: str, metric: str) -> tuple[float, float, float, float]:
    v = np.array([rows[key, k][metric] for k in ro.FOLDS], float)
    if not np.isfinite(v).any():  # e.g. the composite of a score without a threshold
        return (float("nan"),) * 4
    return float(np.nanmean(v)), float(np.nanstd(v, ddof=1)), float(np.nanmin(v)), float(np.nanmax(v))


def variance_shares(rows: dict, keys: list[str], metric: str) -> dict[str, float]:
    m = np.array([[rows[key, k][metric] for k in ro.FOLDS] for key in keys])
    grand = m.mean()
    fold, model = m.mean(0) - grand, m.mean(1) - grand
    resid = m - grand - fold[None] - model[:, None]
    ss = {"fold": m.shape[0] * (fold ** 2).sum(), "model": m.shape[1] * (model ** 2).sum(),
          "interaction": (resid ** 2).sum()}
    total = sum(ss.values())
    return {k: float(v / total) for k, v in ss.items()}


def pooled(rows: dict, key: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Pooled test of the 5 folds: subjects, labels, within-fold percentile rank."""
    subjects, y, r = [], [], []
    for k in ro.FOLDS:
        row = rows[key, k]
        p = np.asarray(row["test_p"], float)
        ok = np.isfinite(p)
        rank = np.full(len(p), np.nan)
        rank[ok] = rankdata(p[ok]) / ok.sum()
        subjects += list(row["test_subjects"])
        y += list(row["test_y"])
        r += list(rank)
    order = np.argsort(subjects)
    return list(np.asarray(subjects)[order]), np.asarray(y)[order], np.asarray(r)[order]


def conditional_auc(y: np.ndarray, s: np.ndarray, z: np.ndarray, caliper: float) -> float:
    ad, hc = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    close = np.abs(z[ad][:, None] - z[hc][None]) <= caliper
    win = (s[ad][:, None] > s[hc][None]) + 0.5 * (s[ad][:, None] == s[hc][None])
    return float(win[close].mean())


def r2(y: np.ndarray, *xs: np.ndarray) -> float:
    x = np.column_stack([np.ones(len(y)), *xs])
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    return float(1 - ((y - x @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def subject_analysis(rows: dict, keys: list[str]) -> dict:
    data = {key: pooled(rows, key) for key in keys}
    subjects, y = data["byol"][0], data["byol"][1]
    for key in keys:
        assert data[key][0] == subjects, key
    vol = data["vol"][2]  # percentile of -volume; NaN where FreeSurfer is missing
    age = data["age"][2]
    ok = np.isfinite(vol)  # every per-subject number below uses the subjects with a volume
    z = (vol - np.nanmean(vol)) / np.nanstd(vol)
    ref = data["byol"][2]
    out = {"n": len(subjects), "n_with_volume": int(ok.sum()), "ad": int(y.sum()),
           "ad_with_volume": int(y[ok].sum()), "rows": {}}
    for key in keys:
        s = data[key][2]
        ad, hc = ok & (y == 1), ok & (y == 0)
        out["rows"][key] = {
            "auc_pooled": pp.auc(y[ok], s[ok]),
            "spearman_byol": float(spearmanr(s[ok], ref[ok]).correlation),
            "spearman_byol_ad": float(spearmanr(s[ad], ref[ad]).correlation),
            "spearman_byol_hc": float(spearmanr(s[hc], ref[hc]).correlation),
            "spearman_vol": float(spearmanr(s[ok], vol[ok]).correlation),
            "auc_given_volume": conditional_auc(y[ok], s[ok], z[ok], CALIPER_SD),
            "r2_volume": r2(s[ok], vol[ok]),
            "r2_volume_age": r2(s[ok], vol[ok], age[ok]),
        }
    out["matrix_keys"] = keys
    out["spearman"] = [[float(spearmanr(data[a][2][ok], data[b][2][ok]).correlation) for b in keys]
                       for a in keys]
    out["_data"] = data
    return out


def history() -> list[dict]:
    """Fold 0 (split_deploy) under protocol v1 (result.json) and v2, mean of the seeds."""
    out = []
    for key in ("vicreg", "byol", "dino_cnx", "sup_cnx", "init_cnx"):
        root, tag, name = ro.fold_source(key, 0)
        v1 = [json.loads((run / "result.json").read_text(encoding="utf-8"))
              for run in sorted((root / tag).glob(f"{name}_s*"))]
        v2 = pp.load_runs(root, tag, name)
        out.append({"key": key, "seeds": len(v1),
                    "v1_test": float(np.mean([d["test_auc"] for d in v1])),
                    "v1_cv": float(np.mean([d["cv_auc"] for d in v1])),
                    "v2_test": float(np.mean([d["test"]["auc"] for d in v2])),
                    "v2_cv_matched": float(np.mean([d["cv_matched"]["auc"] for d in v2]))})
    return out


# ---------------------------------------------------------------- figures
def style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def fig_folds(rows: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
    x = np.arange(len(ro.FOLDS))
    series = [("sup_cnx", "-", 2.0), ("init_cnx", "-", 2.0), ("stats", "-", 1.6), ("vol", "--", 1.8)]
    for ax, metric, title in ((axes[0], "cv_auc", "CV representativa (~700 sujetos por pliegue)"),
                              (axes[1], "test_auc", "Test (~176 sujetos por pliegue)")):
        ssl = np.array([[rows[key, k][metric] for k in ro.FOLDS] for key in SSL])
        ax.fill_between(x, ssl.min(0), ssl.max(0), color="#2a78d6", alpha=0.16, lw=0)
        ax.plot(x, ssl.mean(0), color="#2a78d6", lw=2.0, marker="o", ms=4.5)
        for key, ls, lw in series:
            v = [rows[key, k][metric] for k in ro.FOLDS]
            ax.plot(x, v, color=COLOR[key], ls=ls, lw=lw, marker="o", ms=4.5)
        ax.set_xticks(x, [f"pliegue {k}" for k in ro.FOLDS])
        ax.set_title(title, fontsize=10, loc="left", color=INK)
        style(ax)
    axes[0].set_ylabel("AUROC")
    ends = {"SSL (4 métodos: media y rango)": ("#2a78d6", "-"),
            "Supervisado completo": (COLOR["sup_cnx"], "-"),
            "Sin SSL (ImageNet congelado)": (COLOR["init_cnx"], "-"),
            "Estadísticas de intensidad (38 números)": (COLOR["stats"], "-"),
            "Volumen hipocampal, sin modelo": (INK, "--")}
    handles = [plt.Line2D([], [], color=c, ls=ls, lw=2, marker="o", ms=4.5) for c, ls in ends.values()]
    fig.legend(handles, list(ends), loc="lower center", ncol=3, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "pliegues.png")


def fig_ladder(rows: dict, order: list[str]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2), sharey=True)
    y = np.arange(len(order))[::-1]
    for ax, metric, title in (
            (axes[0], "test_auc", "AUROC test (población real, 23 % AD)"),
            (axes[1], "test_auc_age", "AUROC entre pares AD-HC de edad similar (±3 años)")):
        for yy, key in zip(y, order, strict=True):
            mean, _, lo, hi = fold_mean(rows, key, metric)
            colour = COLOR.get(key, INK2)
            filled = key in COLOR
            ax.plot([lo, hi], [yy, yy], color=colour if filled else MUTED, lw=1.4, solid_capstyle="round")
            ax.plot(mean, yy, marker="D" if key in ("vol", "age") else "o", ms=7,
                    mfc=colour if filled else "white", mec=colour if filled else INK2, mew=1.4)
            ax.text(hi + 0.006, yy, f"{mean:.3f}", va="center", fontsize=8, color=INK2)
        ax.set_title(title, fontsize=10, loc="left", color=INK)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel("media de 5 pliegues (barra = mínimo a máximo)")
    axes[0].set_yticks(y, [LABEL[k] for k in order], fontsize=8.5)
    lo = min(fold_mean(rows, k, m)[2] for k in order for m in ("test_auc", "test_auc_age"))
    for ax in axes:
        ax.set_xlim(max(0.45, lo - 0.02), 0.96)
    fig.tight_layout()
    save(fig, "escalera.png")


def fig_subjects(sub: dict) -> None:
    data = sub["_data"]
    _, y, byol = data["byol"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    panels = ((axes[0], "vol", "volumen hipocampal FreeSurfer", "volumen hipocampal (percentil "
               "dentro del pliegue, menor volumen a la derecha)"),
              (axes[1], "init_cnx", "sin SSL", "puntaje sin SSL (percentil dentro del pliegue)"))
    for ax, key, name_x, xlabel in panels:
        x = data[key][2]
        ok = np.isfinite(x)
        for cls, colour, name in ((0, HC_C, "HC"), (1, AD_C, "AD")):
            m = ok & (y == cls)
            ax.scatter(x[m], byol[m], s=9, color=colour, alpha=0.55, lw=0, label=f"{name} ({m.sum()})")
        rho = spearmanr(x[ok], byol[ok]).correlation
        ax.set_title(f"BYOL frente a {name_x}: Spearman {rho:.2f}", fontsize=10, loc="left", color=INK)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("puntaje BYOL (percentil dentro del pliegue)")
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8.5, loc="upper left", markerscale=1.6)
    fig.tight_layout()
    save(fig, "sujetos.png")


def spread(values: list[float], gap: float) -> list[float]:
    """Label positions at least ``gap`` apart: overlapping labels are pushed upwards."""
    order = np.argsort(values)
    pos = np.asarray(values, float)[order].copy()
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + gap)
    out = np.empty(len(pos))
    out[order] = pos
    return out.tolist()


def fig_history(hist: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    colour = {k: c for k, _, c in MODELS}
    left = spread([h["v1_test"] for h in hist], 0.009)
    right = spread([h["v2_test"] for h in hist], 0.009)
    for h, yl, yr in zip(hist, left, right, strict=True):
        ax.plot([0, 1], [h["v1_test"], h["v2_test"]], color=colour[h["key"]], lw=2, marker="o", ms=6)
        ax.text(1.04, yr, f"{LABEL[h['key']].split(' (')[0]}  {h['v2_test']:.3f}",
                va="center", fontsize=8.5, color=INK)
        ax.text(-0.04, yl, f"{h['v1_test']:.3f}", va="center", ha="right", fontsize=8.5, color=INK2)
    ax.set_xticks([0, 1], ["protocolo v1\n(C elegido con 60 sujetos de val.)",
                           "protocolo v2\n(C por log-loss en la CV)"])
    ax.set_xlim(-0.35, 1.75)
    ax.set_ylabel("AUROC test, pliegue 0 (media de semillas)")
    ax.set_title("El mismo encoder, el mismo test; sólo cambia el probe", fontsize=10, loc="left", color=INK)
    style(ax)
    fig.tight_layout()
    save(fig, "historia.png")


# ---------------------------------------------------------------- LaTeX tables
def f3(v: float) -> str:
    return "---" if v is None or not np.isfinite(v) else f"{v:.3f}"


def write_tables(rows: dict, order_models: list[str], order_base: list[str], sub: dict,
                 hist: list[dict], shares: dict, gaps: dict, corr: dict) -> None:
    lines = [r"\begin{tabular}{@{}l rrrrrr@{}}", r"\toprule",
             r" & CV & CV emp. & \multicolumn{4}{c}{Test} \\ \cmidrule(l){4-7}",
             r"Modelo o línea base & AUROC & AUROC & AUROC & AUPR & compuesto & AUROC $\pm$3 años \\",
             r"\midrule", r"\multicolumn{7}{@{}l}{\textit{Encoders entrenados o congelados (ConvNeXt-T 3D)}} \\"]

    def line(key: str) -> str:
        cells = [f3(fold_mean(rows, key, m)[0]) for m in
                 ("cv_auc", "cv_matched_auc", "test_auc", "test_aupr", "test_mean3", "test_auc_age")]
        return SHORT[key] + " & " + " & ".join(cells) + r" \\"

    lines += [line(k) for k in order_models]
    lines += [r"\midrule", r"\multicolumn{7}{@{}l}{\textit{Líneas base sin aprendizaje de representación "
              r"(mismo probe y protocolo)}} \\"]
    lines += [line(k) for k in order_base]
    lines += [r"\midrule", r"\multicolumn{7}{@{}l}{\textit{Un solo número usado como puntaje "
              r"(sin probe)}} \\"]
    lines += [line(k) for k, _ in MARKERS]
    lines += [r"\bottomrule", r"\end{tabular}"]
    (FIG / "tab_resumen.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    keys = [*SSL, "sup_cnx", "init_cnx", "stats", "vol", "age"]
    lines = [r"\begin{tabular}{@{}l rrrrr r@{}}", r"\toprule",
             r" & \multicolumn{5}{c}{AUROC test por pliegue} & $r$ con \\ \cmidrule(lr){2-6}",
             "Modelo o línea base & " + " & ".join(str(k) for k in ro.FOLDS) + r" & media SSL \\",
             r"\midrule"]
    for key in keys:
        v = [rows[key, k]["test_auc"] for k in ro.FOLDS]
        lines.append(SHORT[key] + " & " + " & ".join(f3(x) for x in v) + f" & {corr[key]:+.2f}" + r" \\")
    lines.append(r"\midrule")
    lines.append(r"Brecha test $-$ CV, media SSL & " + " & ".join(f"{g:+.3f}" for g in gaps["ssl"])
                 + r" & \\")
    lines.append(r"Brecha test $-$ CV, volumen sin modelo & "
                 + " & ".join(f"{g:+.3f}" for g in gaps["vol"]) + r" & \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (FIG / "tab_pliegues.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = [r"\begin{tabular}{@{}l rrrrr@{}}", r"\toprule",
             r" & AUROC & Spearman & Spearman & AUROC con & $R^2$ \\",
             r"Puntaje & agrupado & con BYOL & con volumen & volumen similar & volumen (+ edad) \\",
             r"\midrule"]
    for key in [*SSL, "sup_cnx", "init_cnx", "stats", "vol", "age"]:
        r = sub["rows"][key]
        same = key == "vol"
        lines.append(SHORT[key] + f" & {r['auc_pooled']:.3f} & {r['spearman_byol']:.2f} & "
                     + ("---" if same else f"{r['spearman_vol']:.2f}") + " & "
                     + ("---" if same else f"{r['auc_given_volume']:.3f}") + " & "
                     + ("---" if same else f"{r['r2_volume']:.2f}"
                        + (" (---)" if key == "age" else f" ({r['r2_volume_age']:.2f})"))
                     + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (FIG / "tab_sujetos.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = [r"\begin{tabular}{@{}l r rr rr@{}}", r"\toprule",
             r" & & \multicolumn{2}{c}{AUROC test} & \multicolumn{2}{c}{AUROC CV emparejada} \\"
             r" \cmidrule(lr){3-4} \cmidrule(l){5-6}",
             r"Modelo & semillas & v1 & v2 & v1 & v2 \\", r"\midrule"]
    for h in hist:
        lines.append(f"{SHORT[h['key']]} & {h['seeds']} & {h['v1_test']:.3f} & {h['v2_test']:.3f} & "
                     f"{h['v1_cv']:.3f} & {h['v2_cv_matched']:.3f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (FIG / "tab_historia.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    sr = sub["rows"]
    num = {
        "volTest": fold_mean(rows, "vol", "test_auc")[0], "volCV": fold_mean(rows, "vol", "cv_auc")[0],
        "statsTest": fold_mean(rows, "stats", "test_auc")[0], "statsCV": fold_mean(rows, "stats", "cv_auc")[0],
        "asegTest": fold_mean(rows, "aseg", "test_auc")[0], "asegCV": fold_mean(rows, "aseg", "cv_auc")[0],
        "byolTest": fold_mean(rows, "byol", "test_auc")[0], "byolCV": fold_mean(rows, "byol", "cv_auc")[0],
        "initTest": fold_mean(rows, "init_cnx", "test_auc")[0], "initCV": fold_mean(rows, "init_cnx", "cv_auc")[0],
        "supTest": fold_mean(rows, "sup_cnx", "test_auc")[0], "supCV": fold_mean(rows, "sup_cnx", "cv_auc")[0],
        "randTest": np.mean([fold_mean(rows, k, "test_auc")[0] for k in ("random_s1", "random_s2")]),
        "randCV": np.mean([fold_mean(rows, k, "cv_auc")[0] for k in ("random_s1", "random_s2")]),
        "ageTest": fold_mean(rows, "age", "test_auc")[0], "ageCV": fold_mean(rows, "age", "cv_auc")[0],
        "ageMatched": fold_mean(rows, "age", "cv_matched_auc")[0],
        "rVol": corr["vol"], "rInit": corr["init_cnx"], "rSup": corr["sup_cnx"], "rAge": corr["age"],
        "shareFold": shares["fold"], "shareModel": shares["model"], "shareInter": shares["interaction"],
        "rhoByolVol": sr["byol"]["spearman_vol"], "rhoInitVol": sr["init_cnx"]["spearman_vol"],
        "rhoByolInit": sr["init_cnx"]["spearman_byol"], "rhoByolSup": sr["sup_cnx"]["spearman_byol"],
        "rTwoByol": sr["byol"]["r2_volume"], "rTwoInit": sr["init_cnx"]["r2_volume"],
        "condByol": sr["byol"]["auc_given_volume"], "condInit": sr["init_cnx"]["auc_given_volume"],
        "condAge": sr["age"]["auc_given_volume"], "condSup": sr["sup_cnx"]["auc_given_volume"],
        "rhoByolInitAD": sr["init_cnx"]["spearman_byol_ad"],
        "rhoByolInitHC": sr["init_cnx"]["spearman_byol_hc"],
        "rhoByolVolAD": sr["vol"]["spearman_byol_ad"], "rhoByolVolHC": sr["vol"]["spearman_byol_hc"],
        "rTwoByolAge": sr["byol"]["r2_volume_age"], "rTwoInitAge": sr["init_cnx"]["r2_volume_age"],
        "condStats": sr["stats"]["auc_given_volume"],
        "volMatched": fold_mean(rows, "vol", "cv_matched_auc")[0],
        "asegMatched": fold_mean(rows, "aseg", "cv_matched_auc")[0],
        "modelsMatchedMax": max(fold_mean(rows, k, "cv_matched_auc")[0] for k in order_models),
        "modelsMatchedMin": min(fold_mean(rows, k, "cv_matched_auc")[0] for k in order_models),
        "asegComposite": fold_mean(rows, "aseg", "test_mean3")[0],
        "modelsCompositeMax": max(fold_mean(rows, k, "test_mean3")[0] for k in order_models),
        "volAge": fold_mean(rows, "vol", "test_auc_age")[0],
        "byolAge": fold_mean(rows, "byol", "test_auc_age")[0],
        "initAge": fold_mean(rows, "init_cnx", "test_auc_age")[0],
        "nPooled": sub["n"], "nVol": sub["n_with_volume"], "nAD": sub["ad"],
    }
    tex = []
    for name, value in num.items():
        text = str(value) if isinstance(value, int) else (f"{value:.2f}" if name.startswith(("r", "share"))
                                                           and not name.startswith("rand") else f"{value:.3f}")
        tex.append(rf"\newcommand{{\n{name}}}{{{text}}}")
    (FIG / "numeros.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    global BASE_DIR
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--baselines", type=Path, default=BASE_DIR,
                    help="output folder of scripts/4_evaluate/trivial_baselines.py")
    BASE_DIR = ap.parse_args(argv).baselines
    models = load_models()
    rows = models | load_baselines() | marker_rows(models, volumes())
    order_models = sorted((k for k, *_ in MODELS), key=lambda k: -fold_mean(rows, k, "test_auc")[0])
    order_base = sorted((k for k, _ in BASELINES), key=lambda k: -fold_mean(rows, k, "test_auc")[0])
    keys6 = [k for k, *_ in MODELS]
    shares = variance_shares(rows, keys6, "test_auc")
    ssl_mean = np.mean([[rows[key, k]["test_auc"] for k in ro.FOLDS] for key in SSL], axis=0)
    corr = {key: float(np.corrcoef([rows[key, k]["test_auc"] for k in ro.FOLDS], ssl_mean)[0, 1])
            for key in [*keys6, *(k for k, _ in BASELINES), "vol", "age"]}
    gaps = {"ssl": [float(np.mean([rows[key, k]["test_auc"] - rows[key, k]["cv_auc"] for key in SSL]))
                    for k in ro.FOLDS],
            "vol": [rows["vol", k]["test_auc"] - rows["vol", k]["cv_auc"] for k in ro.FOLDS]}
    sub = subject_analysis(rows, [*SSL, "sup_cnx", "init_cnx", "stats", "vol", "age"])
    hist = history()

    fig_folds(rows)
    order = order_models[:]
    fig_ladder(rows, sorted(order + order_base + ["vol", "age"],
                            key=lambda k: -fold_mean(rows, k, "test_auc")[0]))
    fig_subjects(sub)
    fig_history(hist)
    write_tables(rows, order_models, order_base, sub, hist, shares, gaps, corr)

    summary = {
        "mean_of_folds": {key: {m: fold_mean(rows, key, m)[0] for m in METRICS}
                          for key in [*keys6, *(k for k, _ in BASELINES), "vol", "age"]},
        "per_fold_test_auc": {key: [rows[key, k]["test_auc"] for k in ro.FOLDS] for key in corr},
        "corr_with_ssl_mean": corr, "variance_shares_test_auc": shares, "test_minus_cv": gaps,
        "subjects": {k: v for k, v in sub.items() if k != "_data"},
        "history_fold0": hist,
        "baseline_missing_volumes": {f"{k}": rows["aseg", k]["missing"] for k in ro.FOLDS},
    }
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    for key in summary["mean_of_folds"]:
        m = summary["mean_of_folds"][key]
        print(f"{LABEL[key][:44]:44s} CV {m['cv_auc']:.3f} test {m['test_auc']:.3f} "
              f"edad±3 {m['test_auc_age']:.3f} r {corr[key]:+.2f}")
    print("variance shares", {k: round(v, 2) for k, v in shares.items()})
    print("->", FIG, BASE_DIR / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
