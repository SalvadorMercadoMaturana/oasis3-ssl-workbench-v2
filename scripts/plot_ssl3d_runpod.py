"""Figures of campaign_ssl3d_runpod: probe metrics, BYOL measures and training curves.

Reads the run results downloaded from the pod (``campaign_ssl3d_runpod/final50``,
``screen30``) and the 30-epoch campaign (``campaign_ssl3d_v2/runs``), and writes
``campaign_ssl3d_runpod/figures/``:

- ``final50_metricas.png``: AUROC, F1-macro and AUPR (CV and test) of the 50-epoch
  runs next to the 30-epoch campaign runs and the encoders without SSL;
- ``byol_medidas_metricas.png``: the same three metrics for every BYOL variant;
- ``byol_medidas_colapso.png``: between/within-subject variance and augmentation
  sensitivity of the trained online encoders (collapse diagnostic);
- ``curvas_final50_{vicreg,byol,dinov2}.png``: total loss and every loss term,
  train and validation, per seed;
- ``curvas_byol_medidas.png``: BYOL training/validation loss per variant.

Usage: ``python scripts/plot_ssl3d_runpod.py``.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CAMPAIGN = ROOT / "campaign_ssl3d_runpod"
FIGURES = CAMPAIGN / "figures"
V2 = ROOT / "campaign_ssl3d_v2" / "runs"
COLOR = {"vicreg": "#2a78d6", "byol": "#eb6834", "dinov2": "#1baf7a"}
LABEL = {"vicreg": "VICReg", "byol": "BYOL", "dinov2": "DINOv2"}
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
METRICS = [("AUROC", "auc", "test_auc"), ("F1-macro", "f1", "test_f1_macro"),
           ("AUPR", "aupr", "test_average_precision")]
VARIANTS = [("campaign", "Campaña\n(ceros)"), ("border", "Borde"),
            ("border_wide", "Borde\n+ 4096"), ("border_gn", "Borde\n+ GN"),
            ("border_predlr", "Borde\n+ LR×10"), ("border_h3", "Borde\n+ las 3")]
VARIANT_COLORS = ["#9c9a92", "#f2b48f", "#eb6834", "#7e57c2", "#c9a227", "#6e3214"]

plt.rcParams.update({"font.size": 9, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                     "savefig.facecolor": "#fcfcfb"})


def load_dir(path: Path) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for result in sorted(path.glob("*/result.json")):
        groups[result.parent.name.rsplit("_s", 1)[0]].append(
            json.loads(result.read_text(encoding="utf-8")))
    for result in sorted(path.glob("*_s*.json")):
        groups[result.stem.rsplit("_s", 1)[0]].append(
            json.loads(result.read_text(encoding="utf-8")))
    return groups


def metric(doc: dict, split: str, key: tuple[str, str, str]) -> float:
    return doc["cv"]["clean"]["metrics"][key[1]] if split == "cv" else doc["test"][key[2]]


def style(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def bar(ax: plt.Axes, x: float, docs: list[dict], split: str, key: tuple, color: str,
        width: float, alpha: float = 1.0, label: str | None = None) -> None:
    values = [metric(d, split, key) for d in docs]
    mean = float(np.mean(values))
    ax.bar(x, mean, width=width, color=color, alpha=alpha, edgecolor="white", linewidth=1.2,
           label=label)
    if len(values) > 1:
        ax.scatter([x] * len(values), values, s=9, color=INK, zorder=3, linewidths=0)
    ax.text(x, 0.565, f"{mean:.3f}", ha="center", va="bottom", fontsize=6.5, color=INK,
            rotation=90)


def final_metrics(final: dict, v2: dict) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.6), sharey=True)
    models = ["vicreg", "byol", "dinov2"]
    ssl = {"vicreg": "vicreg_campaign", "byol": "byol_border_wide", "dinov2": "dinov2_campaign"}
    previous = {"vicreg": "vicreg_3d", "byol": "byol_v1_3d", "dinov2": "dinov2_3d"}
    for r, (split, split_name) in enumerate((("cv", "CV 5×5 (train+val, 357 sujetos)"),
                                             ("test", "Test (56 sujetos)"))):
        for c, key in enumerate(METRICS):
            ax = axes[r, c]
            for i, m in enumerate(models):
                bar(ax, i - 0.27, final[ssl[m]], split, key, COLOR[m], 0.26,
                    label="SSL 50 épocas (RunPod)" if i == 0 else None)
                bar(ax, i, v2[previous[m]], split, key, COLOR[m], 0.26, alpha=0.4,
                    label="SSL 30 épocas (campañas anteriores)" if i == 0 else None)
                bar(ax, i + 0.27, final[f"{m}_init"], split, key, BASE, 0.26,
                    label="Mismo encoder sin SSL" if i == 0 else None)
            ax.set_xticks(range(3), [LABEL[m] + ("\n(border_wide)" if m == "byol" else "")
                                     for m in models])
            ax.set_ylim(0.55, 0.97)
            ax.set_title(f"{key[0]} — {split_name}", fontsize=9.5)
            style(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Probe congelado tras 50 épocas (3 semillas; puntos = semillas). BYOL 30 épocas = "
                 "receta original de la campaña", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(FIGURES / "final50_metricas.png", dpi=160)
    plt.close(fig)


def byol_metrics(screen: dict, final: dict, v2: dict) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.8), sharey=True)
    for r, (split, split_name) in enumerate((("cv", "CV 5×5"), ("test", "Test"))):
        for c, key in enumerate(METRICS):
            ax = axes[r, c]
            for i, (variant, _) in enumerate(VARIANTS):
                docs = screen[f"byol_{variant}"]
                bar(ax, i, docs, split, key, COLOR["byol"], 0.7,
                    alpha=1.0 if variant == "border_wide" else 0.5 if variant != "campaign" else 0.25)
            start = metric(final["byol_init"][0], split, key)
            vic = float(np.mean([metric(d, split, key) for d in screen["vicreg_campaign"]]))
            ax.axhline(start, color=MUTED, linestyle="--", linewidth=1.1,
                       label="ResNet-18 3D sin SSL (ImageNet)")
            ax.axhline(vic, color=COLOR["vicreg"], linestyle=":", linewidth=1.4,
                       label="VICReg, mismas 30 épocas")
            ax.set_xticks(range(len(VARIANTS)), [v[1] for v in VARIANTS], fontsize=7.5)
            ax.set_ylim(0.55, 0.97)
            ax.set_title(f"{key[0]} — {split_name}", fontsize=9.5)
            style(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Medidas sobre BYOL (30 épocas, 3 semillas; puntos = semillas). Opaca: elegida "
                 "por CV (border_wide).\n4096 = proyector y predictor de 4096, GN = GroupNorm, "
                 "LR×10 = LR del predictor ×10", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(FIGURES / "byol_medidas_metricas.png", dpi=160)
    plt.close(fig)


def byol_collapse(analysis: dict) -> None:
    runs = analysis["runs"]
    panels = [("Varianza entre sujetos / varianza de la augmentation\n(alto = separa sujetos)",
               lambda e: e["identity"]["ceros"]["between_over_within"]),
              ("Sensibilidad a las vistas de la campaña\n(varianza de augmentation / entre sujetos; "
               "bajo = sano)",
               lambda e: e["sensitivity"]["vistas completas de la campaña"]["sensitivity"])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, (title, get) in zip(axes, panels, strict=True):
        for i, (variant, _) in enumerate(VARIANTS):
            values = [get(runs[f"byol_{variant}_s{s}"]["online"]) for s in (1, 2, 3)
                      if f"byol_{variant}_s{s}" in runs]
            ax.scatter([i] * len(values), values, s=36, color=COLOR["byol"],
                       alpha=1.0 if variant == "border_wide" else 0.6, zorder=3)
            ax.plot([i - 0.25, i + 0.25], [math.exp(np.mean(np.log(values)))] * 2, color=INK,
                    linewidth=1.5)
        vic = [get(runs[f"vicreg_campaign_s{s}"]["online"]) for s in (1, 2, 3)]
        ax.axhspan(min(vic), max(vic), color=COLOR["vicreg"], alpha=0.18,
                   label="VICReg entrenado (rango de semillas)")
        ax.axhline(get(runs["byol_init"]["encoder"]), color=MUTED, linestyle="--",
                   label="ResNet-18 3D sin SSL")
        ax.set_yscale("log")
        ax.set_xticks(range(len(VARIANTS)), [v[1] for v in VARIANTS], fontsize=7.5)
        ax.set_title(title, fontsize=9.5)
        style(ax)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Colapso del encoder online de BYOL en 142 sujetos no vistos (puntos = semillas, "
                 "raya = media geométrica)", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURES / "byol_medidas_colapso.png", dpi=160)
    plt.close(fig)


CURVES = {
    "vicreg": ("vicreg_campaign", [("loss", "Pérdida total\n(25·inv + 25·var + 1·cov)"),
                                   ("invariance", "Invarianza (MSE entre vistas)"),
                                   ("variance", "Varianza (hinge de la std)"),
                                   ("covariance", "Covarianza (fuera de la diagonal)")]),
    "byol": ("byol_border_wide", [("loss", "Pérdida total (media de las dos direcciones)"),
                                  ("direction_12", "Dirección 1→2 (2 − 2·cos)"),
                                  ("direction_21", "Dirección 2→1 (2 − 2·cos)")]),
    "dinov2": ("dinov2_campaign", [("loss", "Pérdida total (DINO + iBOT + 0.1·KoLeo)"),
                                   ("dino", "DINO (CLS)"), ("ibot", "iBOT (parches enmascarados)"),
                                   ("koleo", "KoLeo"),
                                   ("teacher_entropy_frac", "Entropía del profesor / ln K")]),
}


def series(doc: dict, stage: str, term: str) -> list[float] | None:
    metrics = doc.get("training_metrics") or {}
    key = f"{stage}/loss_epoch" if (stage, term) == ("train", "loss") else f"{stage}/{term}"
    return metrics.get(key)


def final_curves(final: dict) -> None:
    for model, (group, terms) in CURVES.items():
        docs = final[group]
        fig, axes = plt.subplots(1, len(terms), figsize=(3.3 * len(terms), 3.6))
        for ax, (term, title) in zip(axes, terms, strict=True):
            for stage, style_ in (("train", "-"), ("val", "--")):
                curves = [s for d in docs if (s := series(d, stage, term))]
                if not curves:
                    continue
                for s in curves:
                    ax.plot(s, style_, color=COLOR[model], alpha=0.25, linewidth=0.9)
                n = min(len(s) for s in curves)
                ax.plot(np.mean([s[:n] for s in curves], axis=0), style_, color=COLOR[model],
                        linewidth=2.0, label="train" if stage == "train" else "validación")
            if term == "dino":
                ax.axhline(math.log(4096), color="#b3261e", linewidth=1, linestyle=":")
                ax.text(0.98, math.log(4096), "ln K", color="#b3261e", fontsize=7.5,
                        ha="right", va="bottom", transform=ax.get_yaxis_transform())
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("época")
            style(ax)
        axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle(f"{LABEL[model]}"
                     + (" border_wide" if model == "byol" else "")
                     + ": curvas de 50 épocas (línea gruesa = media de 3 semillas, finas = semillas)",
                     fontsize=10.5, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(FIGURES / f"curvas_final50_{model}.png", dpi=160)
        plt.close(fig)


def byol_curves(screen: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    palette = VARIANT_COLORS
    for ax, stage, title in ((axes[0], "train", "Pérdida de entrenamiento"),
                             (axes[1], "val", "Pérdida de validación")):
        for (variant, name), color in zip(VARIANTS, palette, strict=True):
            curves = [s for d in screen[f"byol_{variant}"] if (s := series(d, stage, "loss"))]
            n = min(len(s) for s in curves)
            ax.plot(np.mean([s[:n] for s in curves], axis=0), color=color,
                    linewidth=2.4 if variant == "border_wide" else 1.5,
                    label=name.replace("\n", " "))
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("época")
        style(ax)
    axes[0].set_ylabel("2 − 2·cos (media de direcciones)")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("BYOL por medida: pérdida (media de 3 semillas, 30 épocas). Una pérdida baja no "
                 "es mejor: con GN la predicción se vuelve trivial", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGURES / "curvas_byol_medidas.png", dpi=160)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    final, screen, v2 = (load_dir(CAMPAIGN / "final50"), load_dir(CAMPAIGN / "screen30"),
                         load_dir(V2))
    analysis = json.loads((CAMPAIGN / "screen30" / "analysis.json").read_text(encoding="utf-8"))
    final_metrics(final, v2)
    byol_metrics(screen, final, v2)
    byol_collapse(analysis)
    final_curves(final)
    byol_curves(screen)
    for path in sorted(FIGURES.glob("*.png")):
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
