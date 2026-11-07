#!/usr/bin/env python
"""Report of the v2 fixes on split_deploy: protocol v2, augmented SSL validation, BYOL views.

    python scripts/reeval_deploy.py --tags dep50_init dep50_ssl v2_ssl --force
    python scripts/report_v2fix.py [--runs campaign_ssl3d_runpod/deploy/runs]

Reads ``protocol_v2.json`` (frozen probes, scripts/reeval_deploy.py) and
``result.json["protocol_v2"]`` (supervised, scripts/train_supervised.py) of the tags
``dep50_ssl``/``dep50_init`` (2026-10-20) and ``v2_ssl``/``v2_sup`` (2026-11-07).

Model selection is fixed before reading any result: the seed ensemble (mean of the seeds'
probabilities, as deployed) with the highest ``mean3 = (AUROC + AUPR + F1-macro) / 3`` on
the representative CV (cohort prevalence 23 % AD and real ages, 702 subjects). The test
only confirms. Writes ``campaign_ssl3d_runpod/v2fix/REPORT.md`` and
``campaign_ssl3d_runpod/figures/v2_*.png``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import plot_ssl3d_runpod as base  # noqa: E402
import probe_protocol as pp  # noqa: E402

plt = base.plt
OUT = base.CAMPAIGN / "v2fix"
SPLIT = base.CAMPAIGN / "split_deploy"
SUP_COLOR = "#6b6b6b"
BYOL_COLORS = {"byol_noise": "#f6c9ad", "byol_geo": "#f2a06f", "byol_bn": "#eb6834",
               "byol_resnet": "#9a3d16"}
# key: (label, method, kind, tag, run name without _s<seed>)
GROUPS: dict[str, tuple[str, str, str, str, str]] = {
    "sup_convnext": ("Supervisado · ConvNeXt-T · geo", "vicreg", "sup", "v2_sup",
                     "vicreg_supervised__geo__bbconvnext_tiny__roi32"),
    "vicreg": ("VICReg · geo", "vicreg", "ssl", "v2_ssl",
               "vicreg_campaign_augval__geo__bbconvnext_tiny__roi32"),
    "byol_geo": ("BYOL · ConvNeXt-T · cabezas GN · geo", "byol", "ssl", "v2_ssl",
                 "byol_gn_dp0.3_dq0_wd0.01__geo__bbconvnext_tiny__roi32"),
    "byol_bn": ("BYOL · ConvNeXt-T · cabezas BN · noise", "byol", "ssl", "v2_ssl",
                "byol_wide_dproj0.1__noise__bbconvnext_tiny__roi32"),
    "byol_noise": ("BYOL · ConvNeXt-T · cabezas GN · noise (modelo v1)", "byol", "ssl",
                   "dep50_ssl", "byol_gn_dp0.3_dq0_wd0.01__noise__bbconvnext_tiny__roi32"),
    "init_convnext": ("Sin SSL · ConvNeXt-T", "vicreg", "init", "dep50_init",
                      "vicreg_init__bbconvnext_tiny__roi32"),
    "sup_resnet": ("Supervisado · ResNet-18 · noise", "byol", "sup", "v2_sup",
                   "byol_supervised__noise__bbresnet18__roi32"),
    "byol_resnet": ("BYOL · ResNet-18 · cabezas GN · noise", "byol", "ssl", "v2_ssl",
                    "byol_gn_dp0.3_dq0_wd0.01__noise__bbresnet18__roi32"),
    "init_resnet": ("Sin SSL · ResNet-18", "byol", "init", "dep50_init",
                    "vicreg_init__bbresnet18__roi32"),
    "sup_vit": ("Supervisado · ViT-S · 3 regiones", "dinov2", "sup", "v2_sup",
                "dinov2_supervised__elastic_tio_gpu__bbvit_small__roimulti3"),
    "dinov2": ("DINOv2 · 3 regiones", "dinov2", "ssl", "v2_ssl",
               "dinov2_campaign__elastic_tio_gpu__bbvit_small__roimulti3"),
    "init_vit": ("Sin SSL · ViT-S · 3 regiones", "dinov2", "init", "dep50_init",
                 "vicreg_init__bbvit_small__roimulti3"),
}
PREVIOUS = {"vicreg": ("dep50_ssl", "vicreg_campaign_augval__geo__bbconvnext_tiny__roi32"),
            "dinov2": ("dep50_ssl", "dinov2_campaign__elastic_tio_gpu__bbvit_small__roimulti3")}
METRICS = (("auc", "AUROC"), ("aupr", "AUPR"), ("f1_macro", "F1-macro"),
           ("mean3", "Compuesto"))


# ---------------------------------------------------------------- loading
load, ensemble = pp.load_runs, pp.ensemble


# ---------------------------------------------------------------- statistics
def boot_delta(y: np.ndarray, a: np.ndarray, b: np.ndarray, ta: float, tb: float,
               draws: int = 2000) -> dict[str, tuple[float, float, float]]:
    """Paired class-stratified bootstrap of metric(a) - metric(b), thresholds held fixed."""
    rng = np.random.default_rng(0)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)

    def metrics(idx: np.ndarray, s: np.ndarray, t: float) -> dict[str, float]:
        yy, ss = y[idx], s[idx]
        m = {"auc": pp.auc(yy, ss), "aupr": pp.aupr(yy, ss),
             "f1_macro": pp.at_threshold(yy, ss, t)["f1_macro"]}
        m["mean3"] = (m["auc"] + m["aupr"] + m["f1_macro"]) / 3
        return m

    full = np.arange(len(y))
    point = {k: metrics(full, a, ta)[k] - metrics(full, b, tb)[k] for k in ("auc", "aupr",
                                                                              "f1_macro", "mean3")}
    samples: dict[str, list[float]] = {k: [] for k in point}
    for _ in range(draws):
        idx = np.concatenate((rng.choice(pos, len(pos)), rng.choice(neg, len(neg))))
        ma, mb = metrics(idx, a, ta), metrics(idx, b, tb)
        for k in samples:
            samples[k].append(ma[k] - mb[k])
    return {k: (point[k], *np.percentile(samples[k], [2.5, 97.5])) for k in point}


def fmt_delta(d: tuple[float, float, float]) -> str:
    star = " *" if d[1] > 0 or d[2] < 0 else ""
    return f"{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]{star}"


def seeds_cell(docs: list[dict], where: str, key: str) -> str:
    v = [d[where][key] for d in docs]
    return f"{np.mean(v):.3f}" + (f" ± {np.std(v):.3f}" if len(v) > 1 else "")


# ---------------------------------------------------------------- figures
def color_of(key: str) -> str:
    _, method, kind, _, _ = GROUPS[key]
    if kind == "sup":
        return SUP_COLOR
    if kind == "init":
        return base.BASE
    return BYOL_COLORS.get(key, base.COLOR[method])


def cv_vs_test_figure(ens: dict[str, dict], age: dict[str, dict], keys: list[str]) -> None:
    """AUROC per population: the old CV (age-matched) vs the representative CV and the test."""
    wheres = (("cv_matched", "CV emparejada por edad\n(294, 50 % AD: la CV de antes)"),
              ("cv_rep", "CV representativa\n(702, 23 % AD, edades reales)"),
              ("test", "Test representativo\n(178, 23 % AD)"),
              ("test_agem", "Test emparejado\npor edad (108)"))
    fig, ax = plt.subplots(figsize=(13, 4.6))
    width = 0.8 / len(keys)
    for i, (where, _) in enumerate(wheres):
        for k, key in enumerate(keys):
            x = i + (k - (len(keys) - 1) / 2) * width
            v = ens[key][where]["auc"]
            ax.bar(x, v, width=width * 0.92, color=color_of(key),
                   hatch="///" if GROUPS[key][2] == "sup" else None, edgecolor="white",
                   label=GROUPS[key][0] if i == 0 else None)
            ax.text(x, v + 0.008, f"{v:.2f}", ha="center", va="bottom", fontsize=6.8,
                    rotation=90)
        ax.plot([i - 0.42, i + 0.42], [age[where]["auc"]] * 2, color="#b3261e", linewidth=1.6,
                linestyle="--", label="Solo edad" if i == 0 else None)
    ax.axhline(0.5, color=base.INK2, linewidth=0.8, linestyle=":")
    ax.set_xticks(range(len(wheres)), [w[1] for w in wheres], fontsize=8.5)
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel("AUROC")
    base.style(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=8)
    fig.suptitle("¿Por qué la CV salía peor que el test? Distinta población: con la misma "
                 "población (CV representativa) coinciden", fontsize=10.5, color=base.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(base.FIGURES / "v2_cv_vs_test.png", dpi=160)
    plt.close(fig)


def metrics_figure(ens: dict[str, dict], seeds: dict[str, list[dict]], keys: list[str],
                   chance: dict[str, float]) -> None:
    rows = (("cv_rep", "CV representativa (702 sujetos, 23 % AD)"),
            ("test", "Test representativo (178 sujetos, 23 % AD)"))
    fig, axes = plt.subplots(len(rows), len(METRICS), figsize=(17, 8.2), sharey=True)
    for r, (where, row_title) in enumerate(rows):
        for c, (key, name) in enumerate(METRICS):
            ax = axes[r, c]
            for i, g in enumerate(keys):
                v = [d[where][key] for d in seeds[g]]
                e = ens[g][where][key]
                ax.bar(i, e, width=0.78, color=color_of(g), edgecolor="white",
                       hatch="///" if GROUPS[g][2] == "sup" else None)
                if len(v) > 1:
                    ax.scatter([i] * len(v), v, s=9, color=base.INK, zorder=3, linewidths=0)
                ax.text(i, max([e, *v]) + 0.012, f"{e:.3f}", ha="center", va="bottom",
                        fontsize=7.5, rotation=90)
            if key == "aupr":
                ax.axhline(chance[where], color=base.INK2, linewidth=1, linestyle=":")
            ax.set_xticks(range(len(keys)), [GROUPS[g][0].replace(" · ", "\n", 1) for g in keys],
                          fontsize=6.5, rotation=90)
            ax.set_ylim(0.3, 1.0)
            ax.set_title(f"{name} — {row_title}", fontsize=9)
            base.style(ax)
    fig.suptitle("Métricas sensibles al desbalance a la prevalencia real (barra = ensamble de "
                 "semillas, puntos = semillas; F1 con umbral elegido fuera de muestra; "
                 "punteada = AUPR al azar)", fontsize=10.5, color=base.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(base.FIGURES / "v2_metricas.png", dpi=150)
    plt.close(fig)


CURVE_TERMS = {
    "vicreg": [("loss", "Pérdida total"), ("invariance", "Invarianza"),
               ("variance", "Varianza"), ("covariance", "Covarianza")],
    "byol": [("loss", "Pérdida total (2 − 2·cos, media de direcciones)")],
    "dinov2": [("loss", "Pérdida total"), ("dino", "DINO (CLS)"), ("ibot", "iBOT"),
               ("koleo", "KoLeo")],
}
STAGES = (("train", "train (promedio de la época, modo entrenamiento)", ":", 1.2),
          ("trainval", "train, 120 sujetos (modo eval, vistas fijas)", "-", 2.0),
          ("val", "validación, 60 sujetos (modo eval, vistas fijas)", "--", 2.0))


def series(doc: dict, stage: str, term: str) -> list[float] | None:
    metrics = doc.get("training_metrics") or {}
    key = f"{stage}/loss_epoch" if (stage, term) == ("train", "loss") else f"{stage}/{term}"
    return metrics.get(key)


def curves_figure(seeds: dict[str, list[dict]], rows: list[str], method: str, out: str,
                  log: bool = False) -> None:
    terms = CURVE_TERMS[method]
    fig, axes = plt.subplots(len(rows), len(terms), figsize=(3.6 * len(terms) + 1.5,
                                                              3.2 * len(rows)), squeeze=False)
    for r, g in enumerate(rows):
        for c, (term, title) in enumerate(terms):
            ax = axes[r, c]
            for stage, label, style_, width in STAGES:
                curves = [s for d in seeds[g] if (s := series(d, stage, term))]
                if not curves:
                    continue
                n = min(len(s) for s in curves)
                mean = np.mean([s[:n] for s in curves], axis=0)
                for s in curves:
                    ax.plot(s, style_, color=color_of(g), alpha=0.2, linewidth=0.8)
                ax.plot(mean, style_, color=color_of(g) if stage != "train" else base.INK2,
                        linewidth=width, label=label)
            if log:
                ax.set_yscale("log")
            ax.set_title(f"{GROUPS[g][0]}: {title}", fontsize=8.5)
            ax.set_xlabel("época")
            base.style(ax)
    axes[0, 0].legend(frameon=False, fontsize=7.2)
    fig.suptitle("Curvas SSL con validación aumentada (las mismas vistas de entrenamiento, "
                 "fijas por sesión); gruesa = media de 3 semillas", fontsize=10, color=base.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(base.FIGURES / out, dpi=150)
    plt.close(fig)


def byol_views_figure(seeds: dict[str, list[dict]]) -> None:
    """BYOL noise vs geo: loss scale and effective rank over training."""
    keys = [k for k in ("byol_noise", "byol_geo", "byol_bn", "byol_resnet") if seeds.get(k)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0))
    for g in keys:
        for d in seeds[g]:
            s = series(d, "train", "loss")
            if s:
                axes[0].plot(s, color=color_of(g), alpha=0.35, linewidth=0.9)
        curves = [s for d in seeds[g] if (s := series(d, "train", "loss"))]
        n = min(len(s) for s in curves)
        axes[0].plot(np.mean([s[:n] for s in curves], 0), color=color_of(g), linewidth=2,
                     label=GROUPS[g][0])
        rows = [[x for x in (d.get("collapse") or []) if x.get("role") == "online"]
                for d in seeds[g]]
        rows = [r for r in rows if r]
        if rows:
            n = min(len(r) for r in rows)
            ep = [x["epoch"] for x in rows[0][:n]]
            for key, ax in (("effective_rank", axes[1]), ("view_matching_top1", axes[2])):
                ax.plot(ep, np.mean([[x[key] for x in r[:n]] for r in rows], 0), "-o",
                        color=color_of(g), linewidth=2, markersize=3, label=GROUPS[g][0])
    axes[0].set_yscale("log")
    axes[0].set_title("Pérdida BYOL de entrenamiento (escala log)", fontsize=9)
    axes[1].set_title("Rango efectivo de los embeddings (sujetos no vistos)", fontsize=9)
    axes[2].set_title("Top-1 de vistas del mismo sujeto (azar ≈ 0.004)", fontsize=9)
    for ax in axes:
        ax.set_xlabel("época")
        base.style(ax)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("BYOL sin ninguna BatchNorm (ConvNeXt-T + cabezas GroupNorm) colapsa con "
                 "cualquier vista: pérdida ~1e-3, rango ~2; con BatchNorm (cabezas o backbone) no",
                 fontsize=10.5, color=base.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(base.FIGURES / "v2_byol_vistas.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=Path, default=base.CAMPAIGN / "deploy/runs")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    ages = pp.session_ages(SPLIT / "manifest.csv")
    seeds = {k: load(args.runs, g[3], g[4]) for k, g in GROUPS.items()}
    seeds = {k: v for k, v in seeds.items() if v}
    keys = [k for k in GROUPS if k in seeds]
    ens = {k: ensemble(v, ages) for k, v in seeds.items()}
    any_doc = seeds[keys[0]][0]["scores"]
    y_cv = np.asarray(any_doc["cv_y"])
    n_lab = any_doc["n_lab"]
    age = {"cv_rep": pp.age_reference(y_cv, any_doc["cv_subjects"], ages),
           "cv_matched": pp.age_reference(y_cv[:n_lab], any_doc["cv_subjects"][:n_lab], ages),
           "test": pp.age_reference(np.asarray(any_doc["test_y"]), any_doc["test_subjects"], ages)}
    y_test = np.asarray(any_doc["test_y"])
    matched = pp.age_matched_subset(any_doc["test_subjects"], y_test, ages)
    age["test_agem"] = pp.age_reference(y_test[matched],
                                        [any_doc["test_subjects"][i] for i in matched], ages)
    chance = {"cv_rep": float(y_cv.mean()), "test": float(y_test.mean())}
    OUT.mkdir(parents=True, exist_ok=True)
    base.FIGURES.mkdir(parents=True, exist_ok=True)
    # ---- selection (fixed rule: highest representative-CV mean3 of the seed ensemble)
    candidates = [k for k in keys if GROUPS[k][2] in ("ssl", "init")]
    chosen = max(candidates, key=lambda k: ens[k]["cv_rep"]["mean3"])
    lines = ["# Correcciones v2 sobre el split de despliegue (2026-11-07)", "",
             "Protocolo v2 (`scripts/probe_protocol.py`): el probe se entrena igual que antes "
             "(AD/HC emparejados por edad, 50/50) pero la **CV representativa** puntúa además, "
             "fuera de fold, a los 408 AD/HC de train que solo vio el SSL: 702 sujetos con la "
             "prevalencia (23 % AD) y las edades reales de la cohorte, como el test. F1-macro "
             "con umbral elegido fuera de muestra; C del probe por log-loss interno (1e-5 .. 1). "
             "**Compuesto** = (AUROC + AUPR + F1-macro) / 3. Selección fijada de antemano: "
             "ensamble de semillas con mayor compuesto en la CV representativa.", "",
             f"Solo edad: CV representativa AUROC {age['cv_rep']['auc']:.3f} / AUPR "
             f"{age['cv_rep']['aupr']:.3f}; CV emparejada {age['cv_matched']['auc']:.3f}; test "
             f"{age['test']['auc']:.3f} / {age['test']['aupr']:.3f}; test emparejado "
             f"{age['test_agem']['auc']:.3f}. AUROC intra-edad (pares AD-HC a ≤ 3 años) de la "
             f"edad: {age['cv_rep']['auc_age']:.3f}.", "",
             "## Ensamble de semillas (lo que se despliega)", "",
             "| Modelo | n | CV rep. AUROC | AUPR | F1-macro | Compuesto | AUROC intra-edad | "
             "CV emp. AUROC | Test AUROC | AUPR | F1-macro | Compuesto | Test emp. AUROC |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k in keys:
        e = ens[k]
        r, t = e["cv_rep"], e["test"]
        mark = " **(elegido)**" if k == chosen else ""
        lines.append(f"| {GROUPS[k][0]}{mark} | {len(seeds[k])} | {r['auc']:.3f} | {r['aupr']:.3f} "
                     f"| {r['f1_macro']:.3f} | **{r['mean3']:.3f}** | {r['auc_age']:.3f} | "
                     f"{e['cv_matched']['auc']:.3f} | {t['auc']:.3f} | {t['aupr']:.3f} | "
                     f"{t['f1_macro']:.3f} | {t['mean3']:.3f} | {e['test_agem']['auc']:.3f} |")
    lines += ["", "## Por semilla (media ± DE)", "",
              "| Modelo | CV rep. AUROC | AUPR | F1-macro | F1 @0.5 | Compuesto | Test AUROC | "
              "AUPR | F1-macro | Compuesto |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k in keys:
        s = seeds[k]
        lines.append(f"| {GROUPS[k][0]} | {seeds_cell(s, 'cv_rep', 'auc')} | "
                     f"{seeds_cell(s, 'cv_rep', 'aupr')} | {seeds_cell(s, 'cv_rep', 'f1_macro')} | "
                     f"{seeds_cell(s, 'cv_rep', 'f1_macro_05')} | {seeds_cell(s, 'cv_rep', 'mean3')} "
                     f"| {seeds_cell(s, 'test', 'auc')} | {seeds_cell(s, 'test', 'aupr')} | "
                     f"{seeds_cell(s, 'test', 'f1_macro')} | {seeds_cell(s, 'test', 'mean3')} |")
    # ---- paired contrasts on the ensembles
    pairs = [("vicreg", "init_convnext"), ("byol_bn", "init_convnext"),
             ("byol_noise", "init_convnext"), ("byol_resnet", "init_resnet"),
             ("dinov2", "init_vit"), ("vicreg", "sup_convnext"), ("byol_bn", "sup_convnext"),
             ("byol_resnet", "sup_resnet"), ("dinov2", "sup_vit"),
             ("byol_bn", "byol_noise"), ("byol_resnet", "byol_noise"), ("byol_geo", "byol_noise"),
             ("vicreg", "byol_noise"), ("byol_bn", "vicreg"), ("byol_resnet", "vicreg")]
    pairs = [(a, b) for a, b in pairs if a in ens and b in ens]
    lines += ["", "## Contrastes pareados (bootstrap estratificado, IC 95 %, * = excluye 0)", "",
              "| Contraste | CV rep. Δ compuesto | Δ AUROC | Δ AUPR | Δ F1-macro | Test Δ compuesto "
              "| Δ AUROC | Δ AUPR |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for a, b in pairs:
        ya, sa, _ = ens[a]["_cv"]
        _, sb, _ = ens[b]["_cv"]
        cv = boot_delta(ya, sa, sb, ens[a]["threshold_rep"], ens[b]["threshold_rep"])
        yt, pa = ens[a]["_test"]
        _, pb = ens[b]["_test"]
        te = boot_delta(yt, pa, pb, ens[a]["threshold_rep"], ens[b]["threshold_rep"])
        lines.append(f"| {GROUPS[a][0]} − {GROUPS[b][0]} | {fmt_delta(cv['mean3'])} | "
                     f"{fmt_delta(cv['auc'])} | {fmt_delta(cv['aupr'])} | "
                     f"{fmt_delta(cv['f1_macro'])} | {fmt_delta(te['mean3'])} | "
                     f"{fmt_delta(te['auc'])} | {fmt_delta(te['aupr'])} |")
    # ---- collapse / identity of the SSL encoders
    lines += ["", "## Colapso al final del entrenamiento (sujetos no vistos)", "",
              "| Modelo | Rango efectivo | Var_sujeto/Var_aug | Top-1 de vistas | Pérdida train "
              "final | Pérdida val final | Pérdida train (modo eval) final |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for k in keys:
        if GROUPS[k][2] != "ssl":
            continue
        last = [[x for x in (d.get("collapse") or []) if x.get("role") in ("online", "teacher")]
                for d in seeds[k]]
        last = [r[-1] for r in last if r]
        final = {st: [s[-1] for d in seeds[k] if (s := series(d, st, "loss"))]
                 for st in ("train", "val", "trainval")}
        cell = lambda v: f"{np.mean(v):.3g}" if v else "—"  # noqa: E731
        lines.append(f"| {GROUPS[k][0]} | {cell([x['effective_rank'] for x in last])} | "
                     f"{cell([x['between_over_within'] for x in last])} | "
                     f"{cell([x['view_matching_top1'] for x in last])} | {cell(final['train'])} | "
                     f"{cell(final['val'])} | {cell(final['trainval'])} |")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    # ---- figures
    main_keys = [k for k in keys if not k.startswith("_")]
    cv_vs_test_figure(ens, age, main_keys)
    metrics_figure(ens, seeds, main_keys, chance)
    for method, rows in (("vicreg", ["vicreg"]),
                         ("byol", ["byol_bn", "byol_resnet", "byol_geo"]),
                         ("dinov2", ["dinov2"])):
        rows = [r for r in rows if r in seeds and any(series(d, "trainval", "loss")
                                                      for d in seeds[r])]
        if rows:
            curves_figure(seeds, rows, method, f"v2_curvas_{method}.png")
    byol_views_figure(seeds)
    (OUT / "selection.json").write_text(json.dumps({
        "rule": "highest representative-CV mean3 of the seed ensemble (SSL and sin-SSL encoders)",
        "chosen": chosen, "label": GROUPS[chosen][0], "tag": GROUPS[chosen][3],
        "run": GROUPS[chosen][4],
        "cv_rep_mean3": {k: ens[k]["cv_rep"]["mean3"] for k in candidates}}, indent=1) + "\n",
        encoding="utf-8")
    print(f"chosen: {GROUPS[chosen][0]}")


if __name__ == "__main__":
    main()
