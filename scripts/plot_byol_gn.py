"""BYOL with GroupNorm heads: metrics, loss curves, collapse, SSL vs no SSL.

    python scripts/plot_byol_gn.py

Runs: campaign_ssl3d_runpod/byol_diag30/gn (BYOL ``wide_dproj0.1_gn`` + ``noise`` on the
32³ hippocampal pair, batch 16, 30 epochs, seeds 1-3, split 361/63/84), next to the
same model with BatchNorm heads (byol_diag30/b16) and the ResNet-18 3D without SSL
(split361_30/byol_init__roi32_s1). Writes ``byol_diag30/GN.md`` and
``figures/byol_gn_{metricas,curvas,colapso}.png``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import plot_ssl3d_runpod as base  # noqa: E402
from report_aug_recipes3d import bootstrap_auc, mean, std  # noqa: E402

plt = base.plt
GN_COLOR, BN_COLOR = "#1f8a70", "#eb6834"


def load(path: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(path.glob("byol_*/result.json"))]


def metrics_figure(gn: list[dict], bn: list[dict], init: list[dict]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.4), sharey=True)
    for r, split in enumerate(("cv", "test")):
        for c, key in enumerate(base.METRICS):
            ax = axes[r, c]
            base.bar(ax, 0, gn, split, key, GN_COLOR, 0.7, label="BYOL + GroupNorm (SSL)")
            base.bar(ax, 1, bn, split, key, BN_COLOR, 0.7, alpha=0.45,
                     label="BYOL + BatchNorm (actual, SSL)")
            base.bar(ax, 2, init, split, key, base.BASE, 0.7, label="ResNet-18 3D sin SSL")
            ax.set_xticks(range(3), ["GroupNorm", "BatchNorm", "sin SSL"])
            ax.set_ylim(0.55, 0.97)
            n = len(gn[0]["labels"]) if split == "cv" else len(gn[0]["test"]["test_predictions"])
            ax.set_title(f"{key[0]} — {'CV' if split == 'cv' else 'test'} ({n} sujetos)", fontsize=9.5)
            base.style(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8.5)
    fig.suptitle("BYOL con GroupNorm en las cabezas, 30 épocas, split 361/63/84 "
                 "(3 semillas; puntos = semillas)", fontsize=10.5, color=base.INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(base.FIGURES / "byol_gn_metricas.png", dpi=160)
    plt.close(fig)


def objective(doc: dict, key: str) -> tuple[list[int], list[float]]:
    rows = [r for r in doc.get("collapse") or [] if r["role"] == "objective"]
    return [r["epoch"] for r in rows], [r[key] for r in rows]


def curves_figure(gn: list[dict], bn: list[dict]) -> None:
    terms = [("loss", "Pérdida total (media de las dos direcciones)"),
             ("direction_12", "Dirección 1→2 (2 − 2·cos)"),
             ("direction_21", "Dirección 2→1 (2 − 2·cos)")]
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 3.9))
    for ax, (term, title) in zip(axes[:3], terms, strict=True):
        for stage, style_ in (("train", "-"), ("val", "--")):
            curves = [s for d in gn if (s := base.series(d, stage, term))]
            for s in curves:
                ax.plot(s, style_, color=GN_COLOR, alpha=0.25, linewidth=0.9)
            n = min(len(s) for s in curves)
            ax.plot(np.mean([s[:n] for s in curves], axis=0), style_, color=GN_COLOR,
                    linewidth=2.0, label="train" if stage == "train" else "validación (cubo vs espejo L/R)")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("época")
        base.style(ax)
    axes[0].legend(frameon=False, fontsize=8)
    ax = axes[3]
    for docs, color, name in ((gn, GN_COLOR, "GroupNorm"), (bn, BN_COLOR, "BatchNorm (actual)")):
        for key, style_, who in (("byol_loss_train", "-", "train"),
                                 ("byol_loss_heldout", "--", "no vistos")):
            series = [objective(d, key) for d in docs]
            n = min(len(s[1]) for s in series)
            ax.plot(series[0][0][:n], np.mean([s[1][:n] for s in series], axis=0), style_,
                    color=color, linewidth=1.9, label=f"{name}, {who}")
    ax.set_title("Objetivo en modo evaluación, mismas vistas", fontsize=9)
    ax.set_xlabel("época")
    ax.legend(frameon=False, fontsize=7.5)
    base.style(ax)
    fig.suptitle("BYOL + GroupNorm: pérdidas en 30 épocas (gruesa = media de 3 semillas, finas = "
                 "semillas)", fontsize=10.5, color=base.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(base.FIGURES / "byol_gn_curvas.png", dpi=160)
    plt.close(fig)


COLLAPSE = [("effective_rank", "Rango efectivo", False),
            ("between_over_within", "Var_sujeto / Var_aug", True),
            ("same_subject_cos", "Coseno mismo sujeto", False),
            ("different_subject_cos", "Coseno sujetos distintos", False),
            ("view_matching_top1", "Top-1 de vistas (azar 0.007)", False)]


def collapse_figure(gn: list[dict], bn: list[dict]) -> None:
    fig, axes = plt.subplots(1, len(COLLAPSE), figsize=(18, 3.8))
    for ax, (key, title, log) in zip(axes, COLLAPSE, strict=True):
        for docs, color, name in ((gn, GN_COLOR, "GroupNorm"), (bn, BN_COLOR, "BatchNorm")):
            for role, style_ in (("online", "-"), ("target", "--")):
                series = [[r for r in d["collapse"] if r["role"] == role] for d in docs]
                n = min(len(s) for s in series)
                ax.plot([r["epoch"] for r in series[0][:n]],
                        np.mean([[r[key] for r in s[:n]] for s in series], axis=0), style_,
                        color=color, linewidth=1.8, alpha=1.0 if color == GN_COLOR else 0.55,
                        label=f"{name}, {'online' if role == 'online' else 'target EMA'}")
        if log:
            ax.set_yscale("log")
            ax.axhline(1.0, color="#b3261e", linestyle=":", linewidth=1)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("época")
        base.style(ax)
    axes[0].legend(frameon=False, fontsize=7.5)
    fig.suptitle("Colapso en 147 sujetos no vistos (media de 3 semillas)", fontsize=10.5,
                 color=base.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(base.FIGURES / "byol_gn_colapso.png", dpi=160)
    plt.close(fig)


def table(gn: list[dict], bn: list[dict], init: list[dict], out: Path) -> None:
    def m3(docs: list[dict], split: str) -> float:
        return mean([d["cv_mean3"] if split == "cv" else d["test_mean3"] for d in docs])

    vs_init, vs_bn = bootstrap_auc(gn, init), bootstrap_auc(gn, bn)
    fmt = lambda b, s: f"{b[s][0]:+.3f} [{b[s][1]:+.3f}, {b[s][2]:+.3f}]"  # noqa: E731
    lines = ["# BYOL + GroupNorm, 30 épocas, split 361/63/84 (3 semillas)", "",
             "| Modelo | AUROC CV | AUROC test | Media3 CV | Media3 test |",
             "|---|---:|---:|---:|---:|"]
    for name, docs in (("BYOL + GroupNorm", gn), ("BYOL + BatchNorm (actual)", bn),
                       ("ResNet-18 3D sin SSL", init)):
        lines.append(f"| {name} | {mean([d['cv_auc'] for d in docs]):.3f} ± "
                     f"{std([d['cv_auc'] for d in docs]):.3f} | {mean([d['test_auc'] for d in docs]):.3f} ± "
                     f"{std([d['test_auc'] for d in docs]):.3f} | {m3(docs, 'cv'):.3f} | {m3(docs, 'test'):.3f} |")
    lines += ["", "| Comparación (Δ AUROC, bootstrap pareado, IC 95 %) | CV | Test |",
              "|---|---:|---:|",
              f"| GroupNorm − sin SSL | {fmt(vs_init, 'cv')} | {fmt(vs_init, 'test')} |",
              f"| GroupNorm − BatchNorm | {fmt(vs_bn, 'cv')} | {fmt(vs_bn, 'test')} |", ""]
    text = "\n".join(lines)
    (out / "GN.md").write_text(text, encoding="utf-8")
    print(text)


def main() -> None:
    root = base.CAMPAIGN / "byol_diag30"
    gn, bn = load(root / "gn"), load(root / "b16")
    init = [json.loads((base.CAMPAIGN / "split361_30/byol_init__roi32_s1/result.json")
                       .read_text(encoding="utf-8"))]
    table(gn, bn, init, root)
    metrics_figure(gn, bn, init)
    curves_figure(gn, bn)
    collapse_figure(gn, bn)


if __name__ == "__main__":
    main()
