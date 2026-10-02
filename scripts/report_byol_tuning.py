"""GroupNorm BYOL tuning (dropout, weight decay): table, selection and figures.

    python scripts/report_byol_tuning.py campaign_ssl3d_runpod/byol_tune30 [--select 3]

Runs: train.py BYOL ``gn_dp{p}_dq{q}[_wd{w}]`` + ``noise`` on the 32³ pair, 30 epochs,
3 seeds, split 361/63/84. Selection rule, fixed before the runs: combined rank =
rank of CV AUROC (higher is better) + rank of the validation-loss rise (final minus
minimum of the logged validation curve; lower is better); ties go to the higher CV
AUROC. Test AUROC is reported, never used to choose. ``--select N`` prints the
stage-2 specs (weight decay 0.01 and 0.05 for the N best stage-1 configurations).
Writes ``TUNING.md`` and ``figures/byol_tune_{pareto,validacion,metricas}.png``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import plot_ssl3d_runpod as base  # noqa: E402
from report_aug_recipes3d import bootstrap_auc, mean, std  # noqa: E402

plt = base.plt
BASELINE = "gn_dp0.1_dq0"  # the GroupNorm BYOL of section 11


def load(run_dir: Path) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(run_dir.glob("byol_*/result.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        if "cv" in doc:
            groups[doc["variant"]].append(doc)
    return groups


def summarize(groups: dict[str, list[dict]]) -> list[dict]:
    rows = []
    for variant, docs in groups.items():
        val = [d["training_metrics"]["val/loss"] for d in docs]
        train = [d["training_metrics"]["train/loss_epoch"] for d in docs]
        objective = [[r for r in d.get("collapse") or [] if r["role"] == "objective"][-1]
                     for d in docs if d.get("collapse")]
        own = [d["identity"]["own"] for d in docs if d.get("identity")]
        rows.append({
            "variant": variant, "n": len(docs),
            "cv_auc": mean([d["cv_auc"] for d in docs]), "cv_auc_sd": std([d["cv_auc"] for d in docs]),
            "test_auc": mean([d["test_auc"] for d in docs]),
            "test_auc_sd": std([d["test_auc"] for d in docs]),
            "cv_mean3": mean([d["cv_mean3"] for d in docs]),
            "test_mean3": mean([d["test_mean3"] for d in docs]),
            "val_min": mean([min(v) for v in val]), "val_final": mean([v[-1] for v in val]),
            "val_rise": mean([v[-1] - min(v) for v in val]),
            "val_min_epoch": mean([int(np.argmin(v)) for v in val]),
            "gap_final": mean([v[-1] - t[-1] for v, t in zip(val, train, strict=True)]),
            "objective_heldout": mean([o["byol_loss_heldout"] for o in objective]),
            "objective_gap": mean([o["byol_loss_heldout"] - o["byol_loss_train"] for o in objective]),
            "ratio": mean([o["between_over_within"] for o in own]),
            "rank": mean([o["effective_rank"] for o in own])})
    by_auc = sorted(rows, key=lambda r: -r["cv_auc"])
    by_rise = sorted(rows, key=lambda r: r["val_rise"])
    for row in rows:
        row["score"] = by_auc.index(row) + 1 + by_rise.index(row) + 1
    return sorted(rows, key=lambda r: (r["score"], -r["cv_auc"]))


def table(rows: list[dict], groups: dict, init: list[dict] | None, run_dir: Path) -> None:
    best = rows[0]
    lines = ["# Tuning de BYOL con GroupNorm (30 épocas, split 361/63/84, 3 semillas)", "",
             "Criterio fijado antes de correr: puesto en AUROC de CV + puesto en subida de la "
             "validación (final − mínimo); empate → mayor AUROC de CV. Test solo informativo. "
             f"Base: `{BASELINE}` (sección 11). dp/dq = dropout en proyector/predictor, wd = "
             "weight decay (1e-6 si no se indica).", "",
             "| # | Variante | AUROC CV | AUROC test | Media3 CV | Media3 test | Val mín (época) | "
             "Val final | Subida val | Val − train final | Objetivo no vistos (brecha) | "
             "Var_suj/Var_aug | Rango ef. | Puntaje |",
             "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for i, r in enumerate(rows, 1):
        name = f"**`{r['variant']}`**" if r is best else f"`{r['variant']}`"
        if r["variant"] == BASELINE:
            name += " (base)"
        lines.append(
            f"| {i} | {name} | {r['cv_auc']:.3f} ± {r['cv_auc_sd']:.3f} | {r['test_auc']:.3f} ± "
            f"{r['test_auc_sd']:.3f} | {r['cv_mean3']:.3f} | {r['test_mean3']:.3f} | "
            f"{r['val_min']:.3f} ({r['val_min_epoch']:.0f}) | {r['val_final']:.3f} | "
            f"{r['val_rise']:.3f} | {r['gap_final']:.3f} | {r['objective_heldout']:.3f} "
            f"({r['objective_gap']:+.3f}) | {r['ratio']:.3g} | {r['rank']:.1f} | {r['score']} |")
    lines += ["", f"## Elegida: `{best['variant']}`", ""]
    fmt = lambda b, s: f"{b[s][0]:+.3f} [{b[s][1]:+.3f}, {b[s][2]:+.3f}]"  # noqa: E731
    comparisons = []
    if best["variant"] != BASELINE and BASELINE in groups:
        comparisons.append(("vs base GroupNorm", groups[BASELINE]))
    if init:
        comparisons.append(("vs sin SSL", init))
    if comparisons:
        lines += ["| Δ AUROC (bootstrap pareado, IC 95 %) | CV | Test |", "|---|---:|---:|"]
        for name, other in comparisons:
            b = bootstrap_auc(groups[best["variant"]], other)
            lines.append(f"| {best['variant']} {name} | {fmt(b, 'cv')} | {fmt(b, 'test')} |")
    lines += ["", "*Objetivo no vistos*: 2 − 2·cos con vistas fijas de la receta en los 147 "
              "sujetos de val + test, red en modo evaluación; entre paréntesis, menos el mismo "
              "objetivo en 147 sujetos de train (≈ 0 = no memoriza).", ""]
    (run_dir / "TUNING.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def figures(rows: list[dict], groups: dict, init: list[dict] | None) -> None:
    best = rows[0]
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    for r in rows:
        stage2 = "_wd" in r["variant"]
        color = "#1f8a70" if r is best else ("#7e57c2" if stage2 else "#9c9a92")
        ax.scatter(r["val_rise"], r["cv_auc"], s=70 if r is best else 42, color=color, zorder=3,
                   edgecolor="white", linewidth=0.8)
        ax.annotate(r["variant"].replace("gn_", ""), (r["val_rise"], r["cv_auc"]), fontsize=7,
                    color=base.INK2, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("Subida de la curva de validación (final − mínimo; menor es mejor)")
    ax.set_ylabel("AUROC de CV (media de 3 semillas)")
    ax.set_title("Tuning de BYOL + GroupNorm: dropout (gris), + weight decay (violeta), "
                 "elegida (verde)", fontsize=9.5)
    base.style(ax)
    ax.grid(axis="x", color=base.GRID)
    fig.tight_layout()
    fig.savefig(base.FIGURES / "byol_tune_pareto.png", dpi=160)
    plt.close(fig)

    show = [best] + [r for r in rows[1:5]] + [r for r in rows if r["variant"] == BASELINE and r
                                              not in rows[:5]]
    palette = ["#1f8a70", "#2a78d6", "#c9a227", "#7e57c2", "#d98b5f", "#9c9a92"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=False)
    for r, color in zip(show, palette, strict=False):
        for ax, key in ((axes[0], "train/loss_epoch"), (axes[1], "val/loss")):
            curves = [d["training_metrics"][key] for d in groups[r["variant"]]]
            n = min(len(c) for c in curves)
            ax.plot(np.mean([c[:n] for c in curves], axis=0), color=color,
                    linewidth=2.4 if r is best else 1.5,
                    label=r["variant"] + (" (base)" if r["variant"] == BASELINE else ""))
    axes[0].set_title("Pérdida de entrenamiento", fontsize=9.5)
    axes[1].set_title("Pérdida de validación (cubo vs espejo L/R)", fontsize=9.5)
    for ax in axes:
        ax.set_xlabel("época")
        base.style(ax)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Curvas de las 5 mejores configuraciones y la base (media de 3 semillas)",
                 fontsize=10.5, color=base.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(base.FIGURES / "byol_tune_validacion.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.4), sharey=True)
    bars = [(groups[best["variant"]], "#1f8a70", 1.0, f"Elegida: {best['variant']}")]
    if BASELINE in groups and best["variant"] != BASELINE:
        bars.append((groups[BASELINE], "#1f8a70", 0.4, f"Base: {BASELINE}"))
    if init:
        bars.append((init, base.BASE, 1.0, "ResNet-18 3D sin SSL"))
    for r_, split in enumerate(("cv", "test")):
        for c, key in enumerate(base.METRICS):
            ax = axes[r_, c]
            for i, (docs, color, alpha, label) in enumerate(bars):
                base.bar(ax, i, docs, split, key, color, 0.7, alpha=alpha, label=label)
            ax.set_xticks(range(len(bars)), ["elegida", "base GN", "sin SSL"][:len(bars)])
            ax.set_ylim(0.55, 0.97)
            ax.set_title(f"{key[0]} — {'CV' if split == 'cv' else 'test'}", fontsize=9.5)
            base.style(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8.5)
    fig.suptitle("BYOL + GroupNorm elegido por el tuning (30 épocas, 3 semillas)", fontsize=10.5,
                 color=base.INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(base.FIGURES / "byol_tune_metricas.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir")
    parser.add_argument("--select", type=int, default=0)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    groups = load(run_dir)
    rows = summarize(groups)
    if args.select:
        stage1 = [r["variant"] for r in rows if "_wd" not in r["variant"]][:args.select]
        print(" ".join(f"byol:noise:32:{v}_wd{w}" for v in stage1 for w in ("0.01", "0.05")))
        return
    init_path = base.CAMPAIGN / "split361_30/byol_init__roi32_s1/result.json"
    init = [json.loads(init_path.read_text(encoding="utf-8"))] if init_path.is_file() else None
    table(rows, groups, init, run_dir)
    base.FIGURES.mkdir(parents=True, exist_ok=True)
    figures(rows, groups, init)


if __name__ == "__main__":
    main()
