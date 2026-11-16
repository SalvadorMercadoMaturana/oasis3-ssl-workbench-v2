#!/usr/bin/env python
"""Presentation figures of the final results (2026-11-16): split, metrics, architectures.

    python scripts/present_final.py

- ``split.png``: the outer 5-fold split (each AD/HC subject is test once; fold 0 =
  split_deploy) and what the training part of each round is used for.
- ``metricas.png``: best model of each method (rotation pretext, VICReg, DINOv2, BYOL;
  all ConvNeXt-T 3D on the hippocampal pair 2x32^3) against the fully supervised
  ConvNeXt-T and the frozen ImageNet start (sin SSL); representative CV and test, mean
  of the 5 outer folds (dots = folds), each fold a 3-seed ensemble (protocol v2).
- ``arquitecturas.png`` and ``arq_<method>.png``: one card per method, in the style of
  the split361 presentation (docs/figures_split361/architectures.png).

Writes ``docs/figures_final/``. Numbers come from scripts/report_outer5.py.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import report_outer5 as r5  # noqa: E402

base = r5.base
plt = base.plt
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Polygon  # noqa: E402

INK, INK2, MUTED = base.INK, base.INK2, base.MUTED
OUT = HERE.parent / "docs" / "figures_final"
PRETEXT = ("pretext_tune", "pretext_rot24_h2048__geo__bbconvnext_tiny__roi32")
# key, label, sub-label, colour (order and hues pass the dataviz palette validator)
METHODS = [
    ("pretext", "Rotación", "24 clases", "#e87ba4"),
    ("vicreg", "VICReg", "geo", "#2a78d6"),
    ("dino_cnx", "DINOv2", "elastic", "#1baf7a"),
    ("byol", "BYOL", "noise", "#eb6834"),
]
SUP = ("sup_cnx", "Supervisado", "completo", "#5f6b73")
INIT = ("init_cnx", "Sin SSL", "congelado", "#c3c2b7")
TITLE_INK = {"pretext": "#c4487d"}  # darker magenta for large text on white


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- split
def split_counts() -> list[dict]:
    rows = []
    for k in r5.FOLDS:
        path = r5.CAMPAIGN / (f"split_outer5/fold{k}/SPLIT.md" if k else "split_deploy/SPLIT.md")
        text = path.read_text(encoding="utf-8")
        row = {}
        for name in ("train", "val", "test"):
            cells = [c.strip() for c in re.search(rf"^\| {name} \|(.*)$", text, re.M).group(1).split("|")]
            row[name] = cells
        hc_ssl, ad_ssl, other = (int(v) for v in row["train"][3].split("/"))
        lab = int(row["train"][1])
        val_hc, val_ad = int(row["val"][1]), int(row["val"][2])
        rows.append({"test_hc": int(row["test"][1]), "test_ad": int(row["test"][2]),
                     "lab": lab, "ssl_subjects": int(row["train"][0]),
                     "ssl_sessions": int(row["train"][4]), "other": other,
                     "ssl_only_ad_hc": hc_ssl + ad_ssl, "val_hc": val_hc, "val_ad": val_ad,
                     # SSL trains on the train split (labels unused); CV = all AD/HC outside test
                     "pre_ad": lab + ad_ssl, "pre_hc": lab + hc_ssl,
                     "cv_ad": lab + ad_ssl + val_ad, "cv_hc": lab + hc_ssl + val_hc})
    return rows


AD_C, HC_C, OTHER_C = "#e34948", "#6250d6", "#c3c2b7"  # validated pair; gray = "otros"


def class_panel(ax: plt.Axes, counts: list[dict], top: float) -> None:
    """AD/HC composition of each use of the data (round 0), one stacked bar per category."""
    c = counts[0]
    scale = 50.0 / c["ssl_subjects"]
    x0 = 27.0
    ax.text(2, top, "Composición por clase (vuelta 0)", fontsize=12, fontweight="bold", color=INK,
            va="center")
    legend = [(AD_C, 1.0, "AD (Alzheimer)"), (HC_C, 1.0, "HC (control sano)"),
              (OTHER_C, 1.0, "otros diagnósticos"), (AD_C, 0.45, "tono claro = validación")]
    lx = 38.0
    for color, alpha, text in legend:
        ax.add_patch(FancyBboxPatch((lx, top - 0.9), 1.8, 1.8, boxstyle="round,pad=0,rounding_size=0.3",
                                    fc=color, ec="none", alpha=alpha))
        t = ax.text(lx + 2.5, top, text, fontsize=9, color=INK, va="center")
        ax.figure.canvas.draw()
        box = t.get_window_extent(ax.figure.canvas.get_renderer()).transformed(ax.transData.inverted())
        lx = box.x1 + 2.6
    pct = lambda ad, hc: f"{100 * ad / (ad + hc):.0f} % AD"  # noqa: E731
    rows = [
        ("Preentrenamiento SSL", "sin usar etiquetas · todas las sesiones",
         [("AD", c["pre_ad"], 1.0), ("HC", c["pre_hc"], 1.0), ("otros", c["other"], 1.0)],
         f"{c['ssl_subjects']} sujetos · {c['ssl_sessions']} sesiones · "
         f"{pct(c['pre_ad'], c['pre_hc'] + c['other'])}"),
        ("Probe lineal y supervisado", "emparejados 1:1 por edad (tramos de 2.5 años)",
         [("AD", c["lab"], 1.0), ("HC", c["lab"], 1.0), ("gap", 6, 0.0),
          ("AD", c["val_ad"], 0.45), ("HC", c["val_hc"], 0.45)],
         f"{c['lab']} + {c['lab']} entrenan · {c['val_ad']} + {c['val_hc']} validan · 50 % AD"),
        ("CV representativa", "AD/HC fuera del test, prevalencia y edades reales",
         [("AD", c["cv_ad"], 1.0), ("HC", c["cv_hc"], 1.0)],
         f"{c['cv_ad'] + c['cv_hc']} sujetos · {pct(c['cv_ad'], c['cv_hc'])} · 5 particiones × 5 rep."),
        ("Test", "1 sesión por sujeto; nunca visto al entrenar",
         [("AD", c["test_ad"], 1.0), ("HC", c["test_hc"], 1.0)],
         f"{c['test_ad'] + c['test_hc']} sujetos · {pct(c['test_ad'], c['test_hc'])}"),
    ]
    colors = {"AD": AD_C, "HC": HC_C, "otros": OTHER_C}
    for i, (head, sub, parts, right) in enumerate(rows):
        y = top - 5.6 - i * 5.6
        ax.text(2, y + 0.75, head, fontsize=10, fontweight="bold", color=INK, va="center")
        ax.text(2, y - 1.15, sub, fontsize=8, color=INK2, va="center")
        x = x0
        for part, n, alpha in parts:
            w = n * scale
            if part != "gap":
                ax.add_patch(FancyBboxPatch((x, y - 1.7), w, 3.4, boxstyle="round,pad=0,rounding_size=0.4",
                                            fc=colors[part], ec="white", lw=1.5, alpha=alpha))
                light = alpha < 1 or part == "otros"
                ax.text(x + w / 2, y, str(n), fontsize=8.6, ha="center", va="center",
                        color=INK if light else "white", fontweight="bold")
            x += w
        ax.text(x + 1.0, y, right, fontsize=8.8, color=INK, va="center")
    spread = max(max(r[k] for r in counts) - min(r[k] for r in counts)
                 for k in ("pre_ad", "pre_hc", "lab", "cv_ad", "cv_hc", "test_ad", "test_hc"))
    ax.text(2, top - 5.6 - 4 * 5.6 + 1.6,
            f"Vuelta 0 = split_deploy; en las otras 4 vueltas cada cifra cambia en ≤ {spread} sujetos. "
            "El test de cada vuelta conserva la prevalencia real (23 % AD) y las edades reales.",
            fontsize=8.2, color=INK2, va="center", style="italic")


def fig_split() -> None:
    counts = split_counts()
    n_ad = sum(c["test_ad"] for c in counts)
    n_hc = sum(c["test_hc"] for c in counts)
    dy = 13.0  # the class panel sits below the fold grid
    fig = plt.figure(figsize=(13, 7.7))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 64 + dy)
    ax.axis("off")
    test_c, train_c = "#5f6b73", "#e6e4dc"
    x0, w, gap = 13.0, 10.0, 0.5
    ax.text(2, 60.5 + dy, "Split externo de 5 pliegues (OASIS-3)", fontsize=15, fontweight="bold",
            color=INK, va="center")
    ax.text(2, 56.8 + dy, f"Los {n_ad + n_hc} sujetos AD/HC ({n_ad} AD + {n_hc} HC) se reparten en 5 "
            "pliegues por clase y tercil de edad. En cada vuelta un pliegue es test y el resto "
            "entrena:\ncada sujeto AD/HC es test exactamente una vez; ningún sujeto ni sesión aparece "
            "en dos splits.", fontsize=9.5, color=INK2, va="center", linespacing=1.5)
    for f in range(5):
        ax.text(x0 + f * (w + gap) + w / 2, 52.4 + dy, f"pliegue {f}", fontsize=9, color=INK2, ha="center")
    for v, c in enumerate(counts):
        y = 46.0 + dy - v * 6.4
        ax.text(x0 - 1.2, y + 2.2, f"Vuelta {v}", fontsize=10, fontweight="bold", color=INK, ha="right",
                va="center")
        if v == 0:
            ax.text(x0 - 1.2, y + 0.2, "(split_deploy)", fontsize=7.5, color=INK2, ha="right", va="center")
        for f in range(5):
            x = x0 + f * (w + gap)
            is_test = f == v
            ax.add_patch(FancyBboxPatch((x, y), w, 4.4, boxstyle="round,pad=0,rounding_size=0.6",
                                        fc=test_c if is_test else train_c, ec="white", lw=1.5))
            ax.text(x + w / 2, y + 2.2, f"test\n{c['test_ad']} AD + {c['test_hc']} HC" if is_test
                    else "entrena", fontsize=8.2 if is_test else 8, ha="center", va="center",
                    color="white" if is_test else INK2, fontweight="bold" if is_test else "normal")
        ax.text(x0 + 5 * (w + gap) + 1.0, y + 2.2,
                f"SSL: {c['ssl_subjects']} sujetos · {c['ssl_sessions']} sesiones\n"
                f"probe: {c['lab']} AD + {c['lab']} HC (+ val 30 + 30)", fontsize=8.6, color=INK,
                va="center", linespacing=1.5)
    ax.plot([2, 98], [31.2, 31.2], color="#e1e0d9", lw=1.0)
    class_panel(ax, counts, top=28.4)
    save(fig, "split.png")


# ---------------------------------------------------------------- metrics
def load_models() -> dict[str, dict]:
    models = {}
    for key, *_ in [SUP, *METHODS, INIT]:
        o = r5.outer(key, PRETEXT if key == "pretext" else None)
        if o is not None:
            models[key] = o
        else:
            print("missing outer folds:", key)
    return models


def fig_metrics(models: dict[str, dict]) -> None:
    cols = (("auc", "AUROC"), ("f1_macro", "F1-macro"), ("aupr", "AUPR"))
    rows = (("cv_rep", "CV representativa"), ("test", "Test"))
    bars = [(SUP, "sup"), *[(m, "ssl") for m in METHODS], (INIT, "init")]
    xs = [0.0, 1.3, 2.3, 3.3, 4.3, 5.6]
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.4), sharey="col")
    for c, (m, mname) in enumerate(cols):
        values = []
        for r, (where, wname) in enumerate(rows):
            ax = axes[r, c]
            for x, ((key, _, _, color), kind) in zip(xs, bars, strict=True):
                if key not in models:
                    continue
                per = [models[key]["folds"][k][where][m] for k in r5.FOLDS]
                mean = float(np.mean(per))
                values += per
                if kind == "sup":
                    ax.bar(x, mean, width=0.8, facecolor="white", edgecolor=color, hatch="///",
                           linewidth=1.4, zorder=2)
                else:
                    ax.bar(x, mean, width=0.8, color=color, edgecolor="white", linewidth=1.5, zorder=2)
                ax.scatter(x + np.linspace(-0.24, 0.24, len(per)), per, s=11, color=INK, zorder=4,
                           linewidths=0)
                ax.text(x, max(max(per), mean) + 0.006, f"{mean:.3f}", ha="center", va="bottom",
                        fontsize=10.5, fontweight="bold" if kind == "ssl" else "normal", color=INK)
            ax.set_title(f"{mname} — {wname}", fontsize=11.5, fontweight="bold", color=INK)
            ax.set_xticks(xs, [f"{b[0][1]}\n{b[0][2]}" for b in bars], fontsize=8.6)
            base.style(ax)
        lo = np.floor((min(values) - 0.04) * 20) / 20
        hi = max(values) + 0.035
        axes[0, c].set_ylim(lo, hi)
    handles = [Patch(facecolor="white", edgecolor=SUP[3], hatch="///"),
               tuple(Patch(color=mm[3]) for mm in METHODS), Patch(color=INIT[3])]
    labels = ["Supervisado completo: ConvNeXt-T entrenado de punta a punta con etiquetas",
              "SSL (50 épocas) + probe lineal congelado", "Sin SSL: ConvNeXt-T ImageNet congelado + probe lineal"]
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=10,
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0)}, handlelength=4.5)
    fig.suptitle("Mejor modelo de cada método · split externo de 5 pliegues · ConvNeXt-T 3D, "
                 "hipocampos 2×32³\nbarra = media de los 5 pliegues (cada pliegue: ensamble de 3 "
                 "semillas), puntos = pliegues", fontsize=12.5, color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94), h_pad=2.2)
    save(fig, "metricas.png")


# ---------------------------------------------------------------- architectures
CNX = {"encoder": "ConvNeXt-T 3D  ·  31.3 M parámetros",
       "weights": "pesos ImageNet-1k (torchvision), inflados de 2D a 3D",
       "stages": [("96 × 8³", 8), ("192 × 4³", 4), ("384 × 2³", 2), ("768 × 1³", 1)],
       "vector": ("h", "768")}
ARCH = {
    "pretext": {
        "name": "Rotación (pseudo-etiqueta)",
        "subtitle": "receta «geo» + 1 de las 24 rotaciones de 90° del cubo · hipocampos 2×32³",
        "heads": [["Cabeza de rotación", "768 → 2048 → 24", "BN · ReLU · 1.6 M"]],
        "loss": ["Pérdida", "entropía cruzada", "¿qué rotación se", "aplicó? (24 clases)"],
        "twin": None,
        "chips": ["AdamW · lr 10⁻⁴", "weight decay 10⁻⁶", "warmup 10 ép. → coseno hasta 10⁻⁶",
                  "lote 16 · 50 épocas · bf16", "una sola red: sin copia EMA"],
    },
    "vicreg": {
        "name": "VICReg",
        "subtitle": "receta «geo» (recorte + rotación ±5°) · hipocampos 2×32³",
        "heads": [["Proyector", "768 → 2048 → 2048 → 2048", "BN · ReLU · dropout 0.1 · 10.0 M"]],
        "loss": ["Pérdida VICReg", "25 · invarianza", "+ 25 · varianza", "+ 1 · covarianza"],
        "twin": None,
        "chips": ["AdamW · lr 10⁻⁴", "weight decay 10⁻⁶", "warmup 10 ép. → coseno hasta 10⁻⁶",
                  "lote 16 · 50 épocas · bf16", "una sola red: sin copia EMA"],
    },
    "dino_cnx": {
        "name": "DINOv2",
        "subtitle": "receta «elastic» (deformación elástica TorchIO) · hipocampos 2×32³",
        "heads": [["Cabeza DINO", "768 → 2048 → 2048 → 256", "→ 256 prototipos · ≈ 6.4 M"]],
        # same objective and weights as the ViT-S of split361; a CNN has no patch tokens,
        # so the iBOT term is 0 (train/ibot = 0.0 in all 50 epochs of the run)
        "loss": ["Pérdida DINOv2", "DINO + 1·iBOT", "+ 0.1·KoLeo", "temp. 0.1 / 0.01"],
        "loss_note": ["iBOT = 0 con ConvNeXt", "(no hay tokens de parche)"],
        "twin": "profesor = copia lenta (EMA τ 0.99 → 1), sin gradiente",
        "chips": ["AdamW · lr 2·10⁻⁴", "weight decay 0.04 → 0.2", "warmup 10 ép. → coseno hasta 10⁻⁶",
                  "lote 64 · 50 épocas · bf16", "centrado Sinkhorn-Knopp"],
    },
    "byol": {
        "name": "BYOL",
        "subtitle": "receta «noise» (ruido gaussiano σ ≤ 0.22) · hipocampos 2×32³ · modelo desplegado",
        "heads": [["Proyector", "768 → 4096 → 256", "BN · dropout 0.1 · 4.2 M"],
                  ["Predictor", "256 → 4096 → 256", "BN · 2.1 M"]],
        "loss": ["Pérdida BYOL", "2 − 2·cos(p, z̄)", "simétrica en las", "dos vistas"],
        "twin": "target = copia lenta (EMA τ 0.996 → 1), sin gradiente",
        "chips": ["AdamW · lr 10⁻⁴", "weight decay 10⁻⁶", "warmup 10 ép. → coseno hasta 10⁻⁶",
                  "lote 16 · 50 épocas · bf16", "EMA τ 0.996 → 1"],
    },
}
CARD_H, CARD_GAP = 32.0, 2.0


def draw_card(fig: plt.Figure, ax: plt.Axes, y0: float, key: str, color: str) -> None:
    """One card: input -> ConvNeXt-T -> probe vector h -> training heads -> loss, plus settings."""
    spec = {**CNX, **ARCH[key]}

    def rbox(x, y, w, h, fc="white", ec=INK2, lw=1.0, ls="-", alpha=1.0, r=1.2, z=2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                    fc=fc, ec=ec, lw=lw, ls=ls, alpha=alpha, zorder=z))

    def arrow(x1, y1, x2, y2, c=INK2, z=3):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9,
                                     color=c, lw=1.1, zorder=z))

    def cube(x, y, s, d):
        faces = [((x, y), (x + s, y), (x + s, y + s), (x, y + s)),
                 ((x, y + s), (x + d, y + s + d), (x + s + d, y + s + d), (x + s, y + s)),
                 ((x + s, y), (x + s + d, y + d), (x + s + d, y + s + d), (x + s, y + s))]
        for face, alpha in zip(faces, (0.35, 0.55, 0.2), strict=True):
            ax.add_patch(Polygon(face, closed=True, fc=color, ec=color, alpha=alpha, lw=0.8, zorder=3))
        for k in (1, 2):
            ax.plot([x + k * s / 3] * 2, [y, y + s], color="white", lw=0.6, zorder=4)
            ax.plot([x, x + s], [y + k * s / 3] * 2, color="white", lw=0.6, zorder=4)

    cy = y0 + 17.5
    title_ink = TITLE_INK.get(key, color)
    rbox(0.6, y0 + 0.4, 98.8, CARD_H - 0.8, fc=color, ec=color, alpha=0.07, r=2.0, z=1)
    rbox(0.6, y0 + 0.4, 98.8, CARD_H - 0.8, fc="none", ec=color, lw=1.2, r=2.0, z=1)
    t = ax.text(2.6, y0 + CARD_H - 3.6, spec["name"], fontsize=14, fontweight="bold", color=title_ink,
                va="center")
    fig.canvas.draw()
    box = t.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transData.inverted())
    ax.text(max(13.0, box.x1 + 2.0), y0 + CARD_H - 3.6, spec["subtitle"], fontsize=9.5, color=INK2,
            va="center")
    cube(3.0, cy - 3.2, 5.4, 2.2)
    ax.text(6.4, cy - 6.0, "entrada", fontsize=8, color=INK, ha="center", va="top")
    ax.text(6.4, cy - 8.0, "2×32³", fontsize=8, color=INK2, ha="center", va="top")
    if key == "pretext":
        ax.text(6.4, cy + 6.2, "↻ rotada", fontsize=8, color=title_ink, ha="center", fontweight="bold")
    arrow(11.4, cy, 13.4, cy)
    ex0, ex1, ey0, ey1 = 13.6, 52.0, y0 + 8.6, y0 + 25.2
    if spec["twin"]:
        rbox(ex0 + 1.0, ey0 - 1.2, ex1 - ex0, ey1 - ey0, fc="#f4f4f2", ec=MUTED, lw=1.0, ls="--", z=2)
        ax.text(ex1 + 0.6, ey0 - 2.2, spec["twin"], fontsize=7.6, color=INK2, ha="right", va="top",
                style="italic")
    rbox(ex0, ey0, ex1 - ex0, ey1 - ey0, fc="white", ec=color, lw=1.4, z=3)
    ax.text(ex0 + 1.4, ey1 - 1.9, spec["encoder"], fontsize=9, fontweight="bold", color=INK, va="center",
            zorder=5)
    ax.text(ex0 + 1.4, ey1 - 4.0, spec["weights"], fontsize=7.8, color=INK2, va="center", zorder=5)
    n, w, g = 4, 6.0, 2.6
    mid = ey0 + 7.4
    for k, (text, side) in enumerate(spec["stages"]):
        h = 2.0 + 5.6 * np.log2(side + 1) / np.log2(17)
        bx = ex0 + 2.4 + k * (w + g)
        rbox(bx, mid - h / 2, w, h, fc=color, ec=color, alpha=0.25 + 0.15 * k, r=0.6, z=4)
        ax.text(bx + w / 2, ey0 + 1.6, text, fontsize=7.6, ha="center", va="center", color=INK, zorder=5)
        if k < n - 1:
            arrow(bx + w + 0.2, mid, bx + w + g - 0.2, mid, c=MUTED)
    ax.text(ex0 + 2.4 + n * (w + g) - g + 0.6, mid, "GAP", fontsize=7.6, color=INK2, va="center", zorder=5)
    vx = 54.0
    arrow(ex1 + 0.2, cy, vx - 0.1, cy)
    rbox(vx, cy - 5.4, 1.6, 10.8, fc=color, ec=color, alpha=0.85, r=0.5, z=4)
    ax.text(vx + 0.8, cy + 6.6, spec["vector"][0], fontsize=8.5, fontweight="bold", ha="center", color=INK)
    ax.text(vx + 0.8, cy - 6.8, spec["vector"][1], fontsize=8, ha="center", va="top", color=INK)
    ax.text(vx + 0.8, cy - 9.0, "probe", fontsize=7.4, ha="center", va="top", color=title_ink,
            fontweight="bold")
    hx0, hx1 = 59.0, 79.0
    arrow(vx + 1.7, cy, hx0 - 0.1, cy)
    heads = spec["heads"]
    hh = 7.4 if len(heads) == 2 else 12.0
    tops = [cy + 6.4] if len(heads) == 1 else [cy + 8.0, cy - 0.4]
    for top, (title, line1, line2) in zip(tops, heads, strict=True):
        rbox(hx0, top - hh, hx1 - hx0, hh, fc="white", ec=color, lw=1.1, z=3)
        mid_h = top - hh / 2
        dy = 2.2 if len(heads) == 2 else 3.0
        ax.text((hx0 + hx1) / 2, mid_h + dy, title, fontsize=8.3, fontweight="bold", ha="center",
                va="center", zorder=5)
        ax.text((hx0 + hx1) / 2, mid_h, line1, fontsize=7.6, ha="center", va="center", zorder=5)
        ax.text((hx0 + hx1) / 2, mid_h - dy, line2, fontsize=7.3, ha="center", va="center", color=INK2,
                zorder=5)
    if len(heads) == 2:
        arrow((hx0 + hx1) / 2, tops[0] - hh - 0.05, (hx0 + hx1) / 2, tops[1] + 0.05)
    ax.text((hx0 + hx1) / 2, y0 + 7.2, "solo para entrenar", fontsize=7.2, ha="center", color=MUTED,
            style="italic")
    lx0, lx1 = 82.5, 98.0
    arrow(hx1 + 0.2, cy, lx0 - 0.1, cy)
    rbox(lx0, cy - 6.4, lx1 - lx0, 12.8, fc="#fdf2e6", ec="#e67e22", lw=1.1, z=3)
    for k, text in enumerate(spec["loss"]):
        ax.text((lx0 + lx1) / 2, cy + 4.2 - k * 2.8, text, fontsize=8.3 if k == 0 else 7.8,
                fontweight="bold" if k == 0 else "normal", ha="center", va="center", zorder=5)
    for k, text in enumerate(spec.get("loss_note", [])):
        ax.text((lx0 + lx1) / 2, cy - 7.6 - k * 2.0, text, fontsize=7.2, ha="center", va="top",
                color=INK2, style="italic", zorder=5)
    renderer = fig.canvas.get_renderer()
    x = 2.6
    for chip in spec["chips"]:
        t = ax.text(x, y0 + 3.4, chip, fontsize=7.8, va="center", ha="left", color=INK, zorder=5,
                    bbox={"boxstyle": "round,pad=0.35,rounding_size=0.6", "fc": "white", "ec": color,
                          "lw": 0.9})
        box = t.get_window_extent(renderer).transformed(ax.transData.inverted())
        x = box.x1 + 1.6


def fig_architectures() -> None:
    n = len(METHODS)
    total = n * CARD_H + (n - 1) * CARD_GAP
    fig = plt.figure(figsize=(10.6, 10.2 * total / (3 * CARD_H + 2 * CARD_GAP)))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, total)
    ax.axis("off")
    for row, (key, _, _, color) in enumerate(METHODS):
        draw_card(fig, ax, (n - 1 - row) * (CARD_H + CARD_GAP), key, color)
    save(fig, "arquitecturas.png")
    for key, _, _, color in METHODS:
        fig = plt.figure(figsize=(10.6, 10.2 * CARD_H / (3 * CARD_H + 2 * CARD_GAP)))
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set_xlim(0, 100)
        ax.set_ylim(0, CARD_H)
        ax.axis("off")
        draw_card(fig, ax, 0.0, key, color)
        save(fig, f"arq_{key.replace('_cnx', '')}.png")


def main() -> None:
    fig_split()
    fig_architectures()
    models = load_models()
    fig_metrics(models)
    for key, label, *_ in [SUP, *METHODS, INIT]:
        if key in models:
            o = models[key]
            print(f"{label:12s} CV AUROC {o['cv_rep_auc'][0]:.3f} F1 {o['cv_rep_f1_macro'][0]:.3f} "
                  f"AUPR {o['cv_rep_aupr'][0]:.3f} | test AUROC {o['test_auc'][0]:.3f} "
                  f"F1 {o['test_f1_macro'][0]:.3f} AUPR {o['test_aupr'][0]:.3f} "
                  f"compuesto {o['test_mean3'][0]:.3f}")
    print(OUT)


if __name__ == "__main__":
    main()
