"""Compare two BYOL variants and the encoder without SSL: metrics, curves, collapse.

    python scripts/plot_byol_pair.py campaign_ssl3d_runpod/byol_fbn30
        --a gn_dp0.3_dq0_wd0.01_fbn "BN del backbone congelado"
        --b gn_dp0.3_dq0_wd0.01 "Elegida (BN del backbone normal)" --prefix byol_fbn

Runs: ``<dir>/byol_*/result.json`` (train.py with ``--identity --collapse-every``);
variants are matched by the ``variant`` field. The no-SSL reference is
split361_30/byol_init__roi32_s1. Writes ``<dir>/<PREFIX>.md`` and
``figures/<prefix>_{metricas,curvas,colapso}.png``.
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
from report_aug_recipes3d import bootstrap_auc, mean, std  # noqa: E402

plt = base.plt
A_COLOR, B_COLOR = "#2a78d6", "#1f8a70"


def load(path: Path, variant: str) -> list[dict]:
    docs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(path.glob("byol_*/result.json"))]
    return [d for d in docs if d.get("variant") == variant]


def objective(doc: dict, key: str) -> tuple[list[int], list[float]]:
    rows = [r for r in doc.get("collapse") or [] if r["role"] == "objective"]
    return [r["epoch"] for r in rows], [r[key] for r in rows]


def metrics_figure(a: list, b: list, init: list, la: str, lb: str, prefix: str, title: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.4), sharey=True)
    for r, split in enumerate(("cv", "test")):
        for c, key in enumerate(base.METRICS):
            ax = axes[r, c]
            base.bar(ax, 0, a, split, key, A_COLOR, 0.7, label=la)
            base.bar(ax, 1, b, split, key, B_COLOR, 0.7, alpha=0.5, label=lb)
            base.bar(ax, 2, init, split, key, base.BASE, 0.7, label="ResNet-18 3D sin SSL")
            ax.set_xticks(range(3), ["A", "B", "sin SSL"])
            ax.set_ylim(0.55, 0.97)
            n = len(a[0]["labels"]) if split == "cv" else len(a[0]["test"]["test_predictions"])
            ax.set_title(f"{key[0]} — {'CV' if split == 'cv' else 'test'} ({n} sujetos)", fontsize=9.5)
            base.style(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, [f"A: {labels[0]}", f"B: {labels[1]}", labels[2]], loc="lower center",
               ncol=3, frameon=False, fontsize=8.5)
    fig.suptitle(f"{title} (3 semillas; puntos = semillas)", fontsize=10.5, color=base.INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(base.FIGURES / f"{prefix}_metricas.png", dpi=160)
    plt.close(fig)


def curves_figure(a: list, b: list, la: str, lb: str, prefix: str, title: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0))
    for docs, color, name in ((a, A_COLOR, la), (b, B_COLOR, lb)):
        for ax, key in ((axes[0], "train/loss_epoch"), (axes[1], "val/loss")):
            curves = [d["training_metrics"][key] for d in docs]
            for c in curves:
                ax.plot(c, color=color, alpha=0.2, linewidth=0.9)
            n = min(len(c) for c in curves)
            ax.plot(np.mean([c[:n] for c in curves], axis=0), color=color, linewidth=2.0, label=name)
        for key, style_, who in (("byol_loss_train", "-", "train"), ("byol_loss_heldout", "--", "no vistos")):
            series = [objective(d, key) for d in docs]
            n = min(len(s[1]) for s in series)
            axes[2].plot(series[0][0][:n], np.mean([s[1][:n] for s in series], axis=0), style_,
                         color=color, linewidth=1.9, label=f"{name}, {who}")
    titles = ("Pérdida de entrenamiento (registrada)", "Validación registrada (cubo vs espejo L/R)",
              "Objetivo en modo evaluación, mismas vistas")
    for ax, t in zip(axes, titles, strict=True):
        ax.set_title(t, fontsize=9.5)
        ax.set_xlabel("época")
        base.style(ax)
    axes[1].legend(frameon=False, fontsize=8)
    axes[2].legend(frameon=False, fontsize=7.5)
    fig.suptitle(f"{title}: pérdidas (gruesa = media de 3 semillas, finas = semillas)",
                 fontsize=10.5, color=base.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(base.FIGURES / f"{prefix}_curvas.png", dpi=160)
    plt.close(fig)


COLLAPSE = [("effective_rank", "Rango efectivo", False),
            ("between_over_within", "Var_sujeto / Var_aug", True),
            ("same_subject_cos", "Coseno mismo sujeto", False),
            ("different_subject_cos", "Coseno sujetos distintos", False),
            ("view_matching_top1", "Top-1 de vistas (azar 0.007)", False)]


def collapse_figure(a: list, b: list, la: str, lb: str, prefix: str, title: str) -> None:
    fig, axes = plt.subplots(1, len(COLLAPSE), figsize=(18, 3.8))
    for ax, (key, name, log) in zip(axes, COLLAPSE, strict=True):
        for docs, color, label in ((a, A_COLOR, la), (b, B_COLOR, lb)):
            for role, style_ in (("online", "-"), ("target", "--")):
                series = [[r for r in d["collapse"] if r["role"] == role] for d in docs]
                n = min(len(s) for s in series)
                ax.plot([r["epoch"] for r in series[0][:n]],
                        np.mean([[r[key] for r in s[:n]] for s in series], axis=0), style_,
                        color=color, linewidth=1.8,
                        label=f"{label}, {'online' if role == 'online' else 'target EMA'}")
        if log:
            ax.set_yscale("log")
            ax.axhline(1.0, color="#b3261e", linestyle=":", linewidth=1)
        ax.set_title(name, fontsize=9.5)
        ax.set_xlabel("época")
        base.style(ax)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"{title}: colapso en 147 sujetos no vistos (media de 3 semillas)", fontsize=10.5,
                 color=base.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(base.FIGURES / f"{prefix}_colapso.png", dpi=160)
    plt.close(fig)


def table(a: list, b: list, init: list, la: str, lb: str, out: Path, title: str) -> None:
    def val_stats(docs: list) -> tuple[float, float, float]:
        curves = [d["training_metrics"]["val/loss"] for d in docs]
        return (mean([min(c) for c in curves]), mean([c[-1] for c in curves]),
                mean([c[-1] - min(c) for c in curves]))

    fmt = lambda x, s: f"{x[s][0]:+.3f} [{x[s][1]:+.3f}, {x[s][2]:+.3f}]"  # noqa: E731
    lines = [f"# {title}", "", "| Modelo | AUROC CV | AUROC test | Media3 CV | Media3 test | "
             "Val mín → final (subida) | Objetivo eval. train → no vistos | Var_suj/Var_aug | Rango ef. |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, docs in ((f"A: {la}", a), (f"B: {lb}", b), ("ResNet-18 3D sin SSL", init)):
        auc = (f"{mean([d['cv_auc'] for d in docs]):.3f} ± {std([d['cv_auc'] for d in docs]):.3f} | "
               f"{mean([d['test_auc'] for d in docs]):.3f} ± {std([d['test_auc'] for d in docs]):.3f} | "
               f"{mean([d['cv_mean3'] for d in docs]):.3f} | {mean([d['test_mean3'] for d in docs]):.3f}")
        if docs is init:
            lines.append(f"| {name} | {auc} | — | — | — | — |")
            continue
        vmin, vfin, rise = val_stats(docs)
        objt = mean([objective(d, "byol_loss_train")[1][-1] for d in docs])
        objh = mean([objective(d, "byol_loss_heldout")[1][-1] for d in docs])
        own = [d["identity"]["own"] for d in docs]
        lines.append(f"| {name} | {auc} | {vmin:.3f} → {vfin:.3f} ({rise:.3f}) | {objt:.3f} → {objh:.3f} | "
                     f"{mean([o['between_over_within'] for o in own]):.3g} | "
                     f"{mean([o['effective_rank'] for o in own]):.1f} |")
    ab, ai = bootstrap_auc(a, b), bootstrap_auc(a, init)
    lines += ["", "| Δ AUROC (bootstrap pareado, IC 95 %) | CV | Test |", "|---|---:|---:|",
              f"| A − B | {fmt(ab, 'cv')} | {fmt(ab, 'test')} |",
              f"| A − sin SSL | {fmt(ai, 'cv')} | {fmt(ai, 'test')} |", ""]
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir")
    parser.add_argument("--a", nargs=2, required=True, metavar=("VARIANT", "LABEL"))
    parser.add_argument("--b", nargs=2, required=True, metavar=("VARIANT", "LABEL"))
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--title", default="BYOL")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    a, b = load(run_dir, args.a[0]), load(run_dir, args.b[0])
    init = [json.loads((base.CAMPAIGN / "split361_30/byol_init__roi32_s1/result.json")
                       .read_text(encoding="utf-8"))]
    base.FIGURES.mkdir(parents=True, exist_ok=True)
    table(a, b, init, args.a[1], args.b[1], run_dir / f"{args.prefix.upper()}.md", args.title)
    metrics_figure(a, b, init, args.a[1], args.b[1], args.prefix, args.title)
    curves_figure(a, b, args.a[1], args.b[1], args.prefix, args.title)
    collapse_figure(a, b, args.a[1], args.b[1], args.prefix, args.title)


if __name__ == "__main__":
    main()
