"""Figures for the champion models (best augmentation + best ROI per model, 50 epochs).

    python scripts/plot_champions.py

Champions (campaign_ssl3d_runpod/champions50): VICReg `geo` on the 32³ hippocampal
pair, BYOL `wide_dproj0.1` + `noise` on the 32³ pair, DINOv2 TorchIO `elastic` on
three bilateral 32³ regions. Compared with the 50-epoch campaign recipes
(campaign_ssl3d_runpod/final50) and with the same encoders without SSL on each
model's input. Writes to campaign_ssl3d_runpod/figures/:

- ``campeones_metricas.png``: AUROC, F1-macro and AUPR, CV and test;
- ``campeones_curvas_{vicreg,byol,dinov2}.png``: total loss and every term;
- ``campeones_colapso.png``: collapse metrics over training (CollapseMonitor).
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import plot_ssl3d_runpod as base  # noqa: E402

plt = base.plt
CAMPAIGN = base.CAMPAIGN
CHAMPION = {"vicreg": "vicreg_campaign__geo__roi32", "byol": "byol_wide_dproj0.1__noise__roi32",
            "dinov2": "dinov2_campaign__elastic__roimulti3"}
PREVIOUS = {"vicreg": "vicreg_campaign", "byol": "byol_border_wide", "dinov2": "dinov2_campaign"}
INIT = {"vicreg": ("final50", "vicreg_init"), "byol": ("final50", "byol_init"),
        "dinov2": ("champions50", "dinov2_init__roimulti3")}
TICK = {"vicreg": "VICReg\ngeo · 32³", "byol": "BYOL\nnoise · 32³ · drop 0.1",
        "dinov2": "DINOv2\nelastic · 3 regiones"}


def metrics_figure(champions: dict, final: dict) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.8), sharey=True)
    models = ["vicreg", "byol", "dinov2"]
    for r, (split, split_name) in enumerate((("cv", "CV 5×5 (train+val, 357 sujetos)"),
                                             ("test", "Test (56 sujetos)"))):
        for c, key in enumerate(base.METRICS):
            ax = axes[r, c]
            for i, m in enumerate(models):
                base.bar(ax, i - 0.27, champions[CHAMPION[m]], split, key, base.COLOR[m], 0.26,
                         label="Campeón: mejor augmentation + mejor entrada (50 ép.)"
                         if i == 0 else None)
                base.bar(ax, i, final[PREVIOUS[m]], split, key, base.COLOR[m], 0.26, alpha=0.4,
                         label="Receta de la campaña, 32³ (50 ép., RunPod)" if i == 0 else None)
                where, group = INIT[m]
                init = (champions if where == "champions50" else final).get(group)
                if init:
                    base.bar(ax, i + 0.27, init, split, key, base.BASE, 0.26,
                             label="Mismo encoder y entrada, sin SSL" if i == 0 else None)
            ax.set_xticks(range(3), [TICK[m] for m in models], fontsize=7.5)
            ax.set_ylim(0.55, 0.97)
            ax.set_title(f"{key[0]} — {split_name}", fontsize=9.5)
            base.style(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8.5)
    fig.suptitle("Modelos campeones tras 50 épocas (3 semillas; puntos = semillas)",
                 fontsize=10.5, color=base.INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(base.FIGURES / "campeones_metricas.png", dpi=160)
    plt.close(fig)


def curves_figures(champions: dict) -> None:
    for model, (_, terms) in base.CURVES.items():
        docs = champions[CHAMPION[model]]
        fig, axes = plt.subplots(1, len(terms), figsize=(3.3 * len(terms), 3.6))
        for ax, (term, title) in zip(axes, terms, strict=True):
            for stage, style_ in (("train", "-"), ("val", "--")):
                curves = [s for d in docs if (s := base.series(d, stage, term))]
                if not curves:
                    continue
                for s in curves:
                    ax.plot(s, style_, color=base.COLOR[model], alpha=0.25, linewidth=0.9)
                n = min(len(s) for s in curves)
                ax.plot(np.mean([s[:n] for s in curves], axis=0), style_, color=base.COLOR[model],
                        linewidth=2.0, label="train" if stage == "train" else "validación")
            if term == "dino":
                ax.axhline(math.log(4096), color="#b3261e", linewidth=1, linestyle=":")
                ax.text(0.98, math.log(4096), "ln K", color="#b3261e", fontsize=7.5,
                        ha="right", va="bottom", transform=ax.get_yaxis_transform())
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("época")
            base.style(ax)
        axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle(f"{TICK[model].replace(chr(10), ' — ')}: pérdidas en 50 épocas (gruesa = "
                     "media de 3 semillas, finas = semillas)", fontsize=10.5, color=base.INK)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(base.FIGURES / f"campeones_curvas_{model}.png", dpi=160)
        plt.close(fig)


COLLAPSE = [("effective_rank", "Rango efectivo (embeddings limpios)", False),
            ("between_over_within", "Var_sujeto / Var_aug", True),
            ("same_subject_cos", "Coseno mismo sujeto (vistas)", False),
            ("different_subject_cos", "Coseno sujetos distintos", False),
            ("view_matching_top1", "Top-1 de vistas (azar 0.007)", False)]


def collapse_figure(champions: dict) -> None:
    models = ["vicreg", "byol", "dinov2"]
    fig, axes = plt.subplots(len(COLLAPSE), 3, figsize=(13, 2.5 * len(COLLAPSE)), sharex=True)
    for c, model in enumerate(models):
        docs = champions[CHAMPION[model]]
        by_role: dict[str, list[list[dict]]] = defaultdict(list)
        for d in docs:
            rows = d.get("collapse") or []
            for role in sorted({r["role"] for r in rows}):
                by_role[role].append([r for r in rows if r["role"] == role])
        for r, (key, title, log) in enumerate(COLLAPSE):
            ax = axes[r, c]
            for role, runs in by_role.items():
                style_ = "-" if role == "online" else "--"
                for rows in runs:
                    ax.plot([x["epoch"] for x in rows], [x[key] for x in rows], style_,
                            color=base.COLOR[model], alpha=0.25, linewidth=0.9)
                n = min(len(rows) for rows in runs)
                ax.plot([x["epoch"] for x in runs[0][:n]],
                        np.mean([[x[key] for x in rows[:n]] for rows in runs], axis=0), style_,
                        color=base.COLOR[model], linewidth=2.0,
                        label={"online": "encoder online", "target": "target EMA",
                               "teacher": "profesor EMA"}[role])
            if key == "between_over_within":
                ax.axhline(1.0, color="#b3261e", linewidth=1, linestyle=":")
            if log:
                ax.set_yscale("log")
            if r == 0:
                ax.set_title(TICK[model].replace("\n", " — "), fontsize=9.5)
                ax.legend(frameon=False, fontsize=7.5)
            if c == 0:
                ax.set_ylabel(title, fontsize=8.5)
            if r == len(COLLAPSE) - 1:
                ax.set_xlabel("época")
            base.style(ax)
    fig.suptitle("Colapso durante el entrenamiento en 142 sujetos no vistos (vistas fijas de la "
                 "receta de cada modelo; época 0 = antes de entrenar)", fontsize=10.5,
                 color=base.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(base.FIGURES / "campeones_colapso.png", dpi=150)
    plt.close(fig)


def load(path: Path) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for result in sorted(path.glob("*/result.json")):
        groups[result.parent.name.rsplit("_s", 1)[0]].append(
            json.loads(result.read_text(encoding="utf-8")))
    return groups


def main() -> None:
    base.FIGURES.mkdir(parents=True, exist_ok=True)
    champions, final = load(CAMPAIGN / "champions50"), load(CAMPAIGN / "final50")
    metrics_figure(champions, final)
    curves_figures(champions)
    collapse_figure(champions)
    for name in ("campeones_metricas", "campeones_curvas_vicreg", "campeones_curvas_byol",
                 "campeones_curvas_dinov2", "campeones_colapso"):
        print(base.FIGURES / f"{name}.png")


if __name__ == "__main__":
    main()
