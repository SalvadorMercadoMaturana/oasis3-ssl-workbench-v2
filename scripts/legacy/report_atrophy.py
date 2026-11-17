"""Atrophy regions vs the deployment ROIs on split_deploy (CV and representative test).

    python scripts/legacy/report_atrophy.py [--runs campaign_ssl3d_runpod/atrophy/runs]

Compares tag ``atr50`` (input ``atrophy5``: hippocampus, lateral ventricle, posteromedial,
lateral temporal and inferior parietal cortex, 10 channels) with ``dep50`` (VICReg and
BYOL: hippocampus; DINOv2: hippocampus, ventricle, posteromedial). Split, recipes,
backbones, epochs and seeds are the same: only the input changes. Detection at high
specificity uses the deployed recipe (3-seed ensemble probe; threshold where the
out-of-fold CV specificity reaches 0.90), scored on the representative test.

Writes ``<runs>/REPORT.md`` and figures ``atr50_metricas.png`` and ``atr50_vs_roi.png``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_deploy_model as bd  # noqa: E402
import report_bb50 as rb  # noqa: E402
import report_deploy as rd  # noqa: E402

base, plt = rb.base, rb.plt
CELLS = rd.CELLS
OLD_ROI = {"vicreg": "32", "byol": "32", "dinov2": "multi3"}
NEW_ROI = {m: "atrophy5" for m in OLD_ROI}
ROI_LABEL = {"32": "hipocampo", "multi3": "3 regiones", "atrophy5": "5 regiones de atrofia"}
KINDS = (("ssl", "SSL (congelado + probe)"), ("sup", "Supervisado"), ("init", "Sin SSL"))


def names(m: str, bb: str, roi: str) -> dict[str, str]:
    variant, aug, _ = rb.RECIPE[m]
    return {"ssl": f"{m}_{variant}__{aug}__bb{bb}__roi{roi}",
            "init": f"vicreg_init__bb{bb}__roi{roi}",
            "sup": f"{m}_supervised__{rb.SUP_RECIPE[m]}__bb{bb}__roi{roi}"}


def groups(runs: dict[str, list[dict]], roi_of: dict[str, str]) -> dict[str, dict]:
    out: dict[str, dict] = {kind: {} for kind, _ in KINDS}
    for m, bb in CELLS:
        for kind, name in names(m, bb, roi_of[m]).items():
            if runs.get(name):
                out[kind][(m, bb)] = runs[name]
    return out


def detection(seeds: list[dict], specificity: float = 0.90) -> dict:
    """Deployed recipe on the representative test at the CV high-specificity threshold."""
    x_fit, y_fit, _ = bd.fit_set(seeds)
    oof = bd.ensemble_oof(seeds)
    probes = [bd.Probe(x, y_fit, s["seed"]) for x, s in zip(x_fit, seeds, strict=True)]
    p = np.mean([pr.proba(s["test_x"]) for pr, s in zip(probes, seeds, strict=True)], axis=0)
    y = seeds[0]["test_y"]
    hc = oof[y_fit == 0]
    threshold = float(min(t for t in np.unique(oof) if (hc < t).mean() >= specificity))
    return {"cv_auc": rb.fast_auc(y_fit, oof), "test_auc": rb.fast_auc(y, p),
            "threshold": threshold, **bd.threshold_metrics(y, p, threshold)}


def comparison_figure(old: dict, new: dict) -> None:
    """AUROC before/after adding the atrophy regions, per method and kind, values on top."""
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Patch

    def pale(color: str) -> tuple[float, float, float]:
        return tuple(0.3 * c + 0.7 for c in to_rgb(color))

    rows = (("cv", "CV"), ("test", "Test representativo"))
    fig, axes = plt.subplots(len(rows), len(KINDS), figsize=(15, 8.4), sharey=True)
    width = 0.36
    for r, (where, row_title) in enumerate(rows):
        for c, (kind, kind_title) in enumerate(KINDS):
            ax = axes[r, c]
            for i, (m, bb) in enumerate(CELLS):
                a, b = old[kind].get((m, bb)), new[kind].get((m, bb))
                tops = []
                for k, (docs, face, edge) in enumerate(((a, pale(base.COLOR[m]), base.COLOR[m]),
                                                        (b, base.COLOR[m], "white"))):
                    if not docs:
                        continue
                    v = [rb.value(d, where, "auc") for d in docs]
                    x, mean = i + (k - 0.5) * (width + 0.03), float(np.mean(v))
                    sd = float(np.std(v)) if len(v) > 1 else 0.0
                    ax.bar(x, mean, width=width, color=face, edgecolor=edge, linewidth=1.0)
                    if sd:
                        ax.errorbar(x, mean, yerr=sd, color=base.INK, capsize=2.5, linewidth=1)
                    ax.text(x, mean + sd + 0.012, f"{mean:.3f}", ha="center", va="bottom",
                            fontsize=9.5, fontweight="bold" if k == 1 else "normal",
                            color=base.INK)
                    tops.append(mean + sd)
                if a and b:
                    d = rb.bootstrap(b, a, where)
                    star = "*" if d[1] > 0 or d[2] < 0 else ""
                    ax.text(i, max(tops) + 0.075, f"Δ {d[0]:+.3f}{star}", ha="center",
                            fontsize=9.5, color=base.INK2)
            ax.axhline(0.5, color=base.INK2, linewidth=1.1, linestyle=":", zorder=0)
            ax.set_ylim(0, 1.05)
            ax.set_xticks(range(len(CELLS)), [rd.TICK[m] for m, _ in CELLS], fontsize=9.5)
            ax.set_title(f"AUROC {row_title} — {kind_title}", fontsize=10)
            base.style(ax)
    handles = [Patch(facecolor=pale(base.COLOR["vicreg"]), edgecolor=base.COLOR["vicreg"],
                     label="Antes: VICReg/BYOL hipocampo, DINOv2 3 regiones"),
               Patch(facecolor=base.COLOR["vicreg"], label="Con 5 regiones de atrofia"),
               plt.Line2D([], [], color=base.INK2, linestyle=":", label="0.5")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9.5)
    fig.suptitle("¿Ayuda agregar las atrofias que ignorábamos? Mismo split, recetas y semillas; "
                 "Δ = después − antes (bootstrap pareado, * = IC 95 % excluye 0)",
                 fontsize=11.5, color=base.INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(base.FIGURES / "atr50_vs_roi.png", dpi=160)
    plt.close(fig)
    print("wrote atr50_vs_roi.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=Path, default=base.CAMPAIGN / "atrophy" / "runs")
    parser.add_argument("--old-runs", type=Path, default=base.CAMPAIGN / "deploy" / "runs")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    old_runs = rb.load(args.old_runs, "dep50")
    for name in list(old_runs):  # dep50a_* also matches the dep50 prefix
        old_runs[name] = [d for d in old_runs[name] if "/dep50a_ssl/" not in d["experiment_dir"]]
    old, new = groups(old_runs, OLD_ROI), groups(rb.load(args.runs, "atr50"), NEW_ROI)
    rd.metrics_figure(new, "5 regiones de atrofia, 50 épocas", "atr50_metricas.png",
                      {m: f"{rd.TICK[m]}\n5 regiones" for m in rd.TICK})
    comparison_figure(old, new)
    lines = ["# Regiones de atrofia (atrophy5) contra las ROI del despliegue", "",
             "Mismo split_deploy, recetas, backbones, 50 épocas y 3 semillas; solo cambia la "
             "entrada: hipocampo, ventrículo lateral, posteromedial, temporal lateral "
             "(temporal inferior + medio) y parietal inferior (+ supramarginal), 10 canales. "
             "Antes: VICReg/BYOL hipocampo (2 canales), DINOv2 3 regiones (6 canales). "
             "Δ = atrophy5 − antes, bootstrap pareado (IC 95 %, * = excluye 0).", "",
             "| Método | Modelo | AUROC CV antes | AUROC CV atrofia | Δ CV | AUROC test antes | "
             "AUROC test atrofia | Δ test | Δ AUPR test |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for m, bb in CELLS:
        for kind, title in KINDS:
            a, b = old[kind].get((m, bb)), new[kind].get((m, bb))
            if not (a and b):
                continue
            lines.append(f"| {base.LABEL[m]} | {title} | {rb.mean_sd(a, 'cv', 'auc')} | "
                         f"{rb.mean_sd(b, 'cv', 'auc')} | {rb.fmt(rb.bootstrap(b, a, 'cv'))} | "
                         f"{rb.mean_sd(a, 'test', 'auc')} | {rb.mean_sd(b, 'test', 'auc')} | "
                         f"{rb.fmt(rb.bootstrap(b, a, 'test'))} | "
                         f"{rb.fmt(rb.bootstrap(b, a, 'test', 'aupr'))} |")
    lines += ["", "Detección con alta especificidad (receta desplegada: ensamble de 3 semillas, "
              "umbral donde la especificidad fuera de fold en CV llega a 0.90), test "
              "representativo 41 AD + 137 HC:", "",
              "| Método | Entrada | AUROC CV (fuera de fold) | AUROC test | Umbral | AD detectados | "
              "Falsos positivos | Sens. | Espec. | PPV |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    sources = ((args.old_runs, ["dep50_ssl", "dep50_init"], OLD_ROI),
               (args.runs, ["atr50_ssl", "atr50_init"], NEW_ROI))
    for m, bb in CELLS:
        for kind in ("ssl", "init"):
            for runs, tags, roi_of in sources:
                seeds = bd.load_candidate(runs, names(m, bb, roi_of[m])[kind], tags)
                if not seeds:
                    continue
                d = detection(seeds)
                label = base.LABEL[m] if kind == "ssl" else f"Sin SSL ({rb.BB[bb]})"
                lines.append(f"| {label} | {ROI_LABEL[roi_of[m]]} | {d['cv_auc']:.3f} | "
                             f"{d['test_auc']:.3f} | {d['threshold']:.3f} | {d['tp']}/41 | "
                             f"{d['fp']}/137 | {d['sensitivity']:.2f} | {d['specificity']:.2f} | "
                             f"{d['ppv']:.2f} |")
    (args.runs / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
