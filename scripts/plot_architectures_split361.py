"""One architecture figure per SSL model, branch by branch, with real augmented views.

    python scripts/plot_architectures_split361.py

The two branches of each model sit in separate bands with their own colour: VICReg's
two branches are the same network (shared weights, both get gradients); BYOL's
online branch against its EMA target and DINOv2's student (masked view) against its
EMA teacher, the copies dashed and without gradient. Each branch shows a view of
the same validation subject drawn with the model's own recipe (``geo``, ``noise``,
TorchIO ``elastic``), as stacked central slices of the 32³ volume. Settings are the
100-epoch champions of split361_100, BYOL with the chosen ``gn_dp0.3_dq0_wd0.01``
heads. Writes ``docs/figures_split361/arch_{vicreg,byol,dinov2}.png``.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_report_split361 as rep  # noqa: E402  (also sets the report's rcParams)

plt = rep.plt
from matplotlib.colors import to_rgba  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

INK, INK2, MUTED = rep.INK, rep.INK2, rep.MUTED
SLATE, VIOLET = "#5f6b73", "#7b52c7"
LOSS_FC, LOSS_EC = "#fdf2e6", "#e67e22"
DASH = (0, (5, 3))

SPEC = {
    "vicreg": {
        "title": "VICReg  ·  ConvNeXt-T 3D  ·  hipocampos 2×32³  ·  receta «geo»",
        "recipe": "geo", "cache": "hippo_s32", "colors": (rep.COLOR["vicreg"], VIOLET),
        "aug": ["recorte de 32–35 vóx. en", "posición al azar, reescalado",
                "a 32³ y rotación ±5° en 3 ejes"],
        "bands": [("Rama 1", "vista 1 · recibe gradiente"),
                  ("Rama 2", "vista 2 · la misma red · recibe gradiente")],
        "encoder": [("ConvNeXt-T 3D · 31.3 M", "ImageNet-1k inflado a 3D")] * 2,
        "stages": ["96·8³", "192·4³", "384·2³", "768·1³"],
        "h": "768",
        "heads": [[("Proyector", "768 → 2048 → 2048 → 2048", "BN · ReLU · dropout 0.1", "10 M")]] * 2,
        "out": ("z₁", "z₂"),
        "link": ("shared", "misma red: pesos compartidos"),
        "probe": 0,
        "loss": ["Pérdida VICReg", "25 · invarianza:", "z₁ y z₂ deben coincidir",
                 "+ 25 · varianza: std de cada", "dimensión ≥ 1 en el lote",
                 "+ 1 · covarianza: dimensiones", "sin correlación entre sí"],
        "grad": "el gradiente vuelve por\nlas dos ramas (misma red)",
        "note": "Ninguna rama es especial: las dos vistas pasan por la misma red y la pérdida es simétrica; "
                "lo que evita el colapso es el término de varianza.",
        "chips": ["AdamW · lr 10⁻⁴", "weight decay 10⁻⁶", "warmup 20 ép. → coseno hasta 10⁻⁶",
                  "lote 16 · 100 épocas · bf16", "validación SSL con vistas aumentadas"],
    },
    "byol": {
        "title": "BYOL  ·  ResNet-18 3D  ·  hipocampos 2×32³  ·  receta «noise»  ·  cabezas GroupNorm (elegida)",
        "recipe": "noise", "cache": "hippo_s32", "colors": (rep.COLOR["byol"], SLATE),
        "aug": ["ruido gaussiano en cada", "vóxel, σ al azar en [0, 0.22];",
                "sin cambios geométricos"],
        "bands": [("Rama online", "vista 1 · recibe gradiente"),
                  ("Rama target", "vista 2 · copia EMA · sin gradiente")],
        "encoder": [("ResNet-18 3D · 33.2 M", "ImageNet-1k inflado a 3D · sin max-pool"),
                    ("ResNet-18 3D (EMA)", "copia lenta del encoder online")],
        "stages": ["64·16³", "128·8³", "256·4³", "512·2³"],
        "h": "512",
        "heads": [[("Proyector", "512 → 4096 → 256", "GroupNorm · ReLU", "dropout 0.3 · 3.2 M"),
                   ("Predictor", "256 → 4096 → 256", "GroupNorm · ReLU", "sin dropout · 2.1 M")],
                  [("Proyector (EMA)", "512 → 4096 → 256", "copia lenta del", "proyector online"), None]],
        "out": ("p", "z̄"),
        "link": ("ema", "EMA: θ′ ← τθ′ + (1 − τ)θ\nτ 0.996 → 1 (coseno)"),
        "probe": 0,
        "loss": ["Pérdida BYOL", "2 − 2 · cos(p, z̄)", "la salida p del predictor",
                 "debe apuntar como z̄,", "la salida del target", "simétrica: se repite con",
                 "las vistas intercambiadas"],
        "grad": "sg = sin gradiente: el gradiente\nsolo vuelve por la rama online",
        "note": "Sin negativos ni término de varianza: lo que evita el colapso es la asimetría "
                "(predictor solo en la rama online + target lento sin gradiente).",
        "chips": ["AdamW · lr 10⁻⁴", "weight decay 0.01", "warmup 20 ép. → coseno hasta 10⁻⁶",
                  "lote 16 · 100 épocas · bf16", "GroupNorm de 32 grupos en proyector y predictor"],
    },
    "dinov2": {
        "title": "DINOv2  ·  ViT-S 3D  ·  3 regiones 6×32³  ·  receta «elastic» + máscara iBOT",
        "recipe": "elastic", "cache": "multi3_s32", "colors": (rep.COLOR["dinov2"], SLATE),
        "aug": ["deformación elástica 3D", "(TorchIO, 7 puntos de control,",
                "≤ 2 vóx.); sin crops locales"],
        "bands": [("Alumno", "vista 1 con parches ocultos · recibe gradiente"),
                  ("Profesor", "vista 2 completa · copia EMA · sin gradiente")],
        "encoder": [("ViT-S 3D · 22.5 M", "DeiT-S ImageNet-1k inflado · parche 8³"),
                    ("ViT-S 3D (EMA)", "copia lenta del alumno")],
        "stages": None,
        "h": "384",
        "heads": [[("Cabeza DINO (CLS)", "384 → 2048 → 2048 → 256 → 4096", "softmax, temperatura 0.1"),
                   ("Cabeza iBOT (parches)", "misma forma, pesos propios", "softmax, temperatura 0.1")],
                  [("Cabeza DINO (EMA)", "Sinkhorn-Knopp", "temperatura 0.01"),
                   ("Cabeza iBOT (EMA)", "Sinkhorn-Knopp", "temperatura 0.01")]],
        "out": None,  # the heads are already named
        "link": ("ema", "EMA: θ′ ← τθ′ + (1 − τ)θ\nτ 0.99 → 1 (coseno)"),
        "probe": 1,
        "loss": ["Pérdida DINOv2", "DINO: el CLS del alumno", "imita al del profesor",
                 "(vistas cruzadas)", "+ iBOT: los parches ocultos", "imitan al profesor (misma vista)",
                 "+ 0.1 · KoLeo: CLS esparcidos"],
        "grad": "sg = sin gradiente: el gradiente\nsolo vuelve por el alumno",
        "note": "Cada vista pasa por las dos ramas (la figura muestra un cruce). El probe usa el CLS "
                "del profesor, no el del alumno.",
        "chips": ["AdamW · lr 10⁻⁴", "weight decay 0.04 → 0.2", "warmup 20 ép. → coseno",
                  "lote 16 · 100 épocas · bf16", "recorte de gradiente 3.0 · drop path 0.1",
                  "máscara: 50 % de las vistas, 10–50 % de los 64 parches"],
    },
}


def views(model: str, idx: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clean cube and two views (channel 0) of one session with the model's recipe."""
    import torch

    from t5_ssl.data.roi3d import ROIEval3D
    data, _, index = rep.cache(SPEC[model]["cache"])
    cube = torch.from_numpy(np.asarray(data[index[idx]], dtype=np.float32))
    random.seed(7), np.random.seed(7), torch.manual_seed(7)
    transform = rep.view_transform(SPEC[model]["recipe"])
    first, second = (transform(cube)[0][0].numpy() for _ in range(2))
    return ROIEval3D(32)(cube)[0].numpy(), first, second


def ibot_mask(volume: np.ndarray) -> np.ndarray:
    """Hide a block of 8³ patches (12 of 64, as iBOT's block masks), NaN where hidden."""
    out = volume.copy()
    out[8:24, 16:32, 0:24] = np.nan  # patches i 1–2, j 2–3, k 0–2: crosses all three central slices
    return out


def stack(ax: plt.Axes, volume: np.ndarray, frame: str, hidden: str | None = None) -> None:
    """The three central planes stacked like cards (as rep.slice_stack), NaN drawn in `hidden`."""
    cmap = plt.get_cmap("gray").copy()
    cmap.set_bad(hidden or "black")
    n, step = len(rep.STACK), rep.STEP
    for k, plane in enumerate(rep.STACK):
        off = (n - 1 - k) * step
        extent = (off, off + 32, off, off + 32)
        image = np.ma.masked_invalid(rep.central_slice(volume, plane))
        ax.imshow(image, cmap=cmap, origin="lower", vmin=0, vmax=1, extent=extent,
                  interpolation="nearest", zorder=3 * k)
        front = k == n - 1
        if not front:
            ax.add_patch(Rectangle((off, off), 32, 32, facecolor="white", alpha=0.45 - 0.15 * k,
                                   edgecolor="none", zorder=3 * k + 1))
        ax.add_patch(Rectangle((off, off), 32, 32, fill=False, edgecolor=frame if front else INK2,
                               linewidth=2.0 if front else 0.6, zorder=3 * k + 2))
    ax.set_xlim(-1, 32 + (n - 1) * step + 1)
    ax.set_ylim(-1, 32 + (n - 1) * step + 1)
    ax.set_aspect("equal")
    rep.bare(ax)


def figure(model: str, idx: str) -> None:
    spec = SPEC[model]
    ema = spec["link"][0] == "ema"
    fig = plt.figure(figsize=(14, 8.2))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 58)
    ax.axis("off")

    def rbox(x, y, w, h, fc="white", ec=INK2, lw=1.0, ls="-", r=0.9, z=2, alpha=1.0):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                    fc=to_rgba(fc, alpha) if fc != "none" else "none", ec=ec,
                                    lw=lw, ls=ls, zorder=z))

    def arrow(p, q, c=INK2, ls="-", lw=1.3, style="-|>", z=6):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=12, color=c, lw=lw,
                                     ls=ls, zorder=z, shrinkA=0, shrinkB=0))

    def text(x, y, s, **kw):
        kw.setdefault("fontsize", 8.2)
        kw.setdefault("color", INK)
        kw.setdefault("zorder", 8)
        return ax.text(x, y, s, **kw)

    main_c = spec["colors"][0]
    ax.text(1.5, 56.0, spec["title"], fontsize=13, fontweight="bold", color=main_c, va="center")

    # ---- input and the split into two views
    clean, first, second = views(model, idx)
    inset = ax.inset_axes((1.0, 23.6, 11.0, 12.8), transform=ax.transData)
    stack(inset, clean, INK)
    text(6.5, 22.8, "cubo limpio", ha="center", va="top", fontweight="bold")
    text(6.5, 20.9, "una sesión, 32³ vóx.\n(se muestra el canal\ndel hipocampo izq.)",
         ha="center", va="top", fontsize=7.4, color=INK2)
    text(6.5, 48.6, "augmentation", ha="center", fontweight="bold", fontsize=8.6)
    text(6.5, 46.9, f"«{spec['recipe']}»: dos vistas", ha="center", fontsize=8.0, color=INK2)
    text(6.5, 45.3, "al azar del mismo cubo", ha="center", fontsize=8.0, color=INK2)
    for k, line in enumerate(spec["aug"]):
        text(6.5, 43.0 - 1.5 * k, line, ha="center", fontsize=7.3, color=INK2)
    mids = (42.0, 18.0)
    arrow((12.4, 33.0), (14.8, mids[0]), c=spec["colors"][0], lw=1.6)
    arrow((12.4, 27.0), (14.8, mids[1]), c=spec["colors"][1], lw=1.6)

    # ---- the two branch bands
    for b, mid in enumerate(mids):
        c = spec["colors"][b]
        copy = ema and b == 1
        ls = DASH if copy else "-"
        fill = "#f6f6f4" if copy else "white"
        rbox(13.6, mid - 10.0, 67.4, 20.0, fc=c, ec="none", r=1.6, z=1, alpha=0.10)
        rbox(13.6, mid - 10.0, 67.4, 20.0, fc="none", ec=c, lw=2.0, ls=ls, r=1.6, z=1)
        name, sub = spec["bands"][b]
        text(14.6, mid + 8.4, name, fontsize=11, fontweight="bold", color=c, va="center")
        text(14.6 + 0.62 * len(name) + 1.4, mid + 8.4, sub, fontsize=8.4, color=INK2, va="center")
        # view
        view, hidden = (first if b == 0 else second), None
        if model == "dinov2" and b == 0:
            view, hidden = ibot_mask(view), to_rgba(c, 0.6)
        vin = ax.inset_axes((15.0, mid - 7.0, 11.0, 12.6), transform=ax.transData)
        stack(vin, view, c, hidden)
        label = f"vista {b + 1}" + (" + parches ocultos" if hidden is not None else "")
        text(20.5, mid - 8.3, label, ha="center", va="center", fontweight="bold", fontsize=8.2,
             color=c)
        arrow((26.4, mid), (28.4, mid), c=c)
        # encoder
        ex0, ex1 = 28.5, 50.5
        rbox(ex0, mid - 7.6, ex1 - ex0, 14.0, fc=fill, ec=c, lw=1.6, ls=ls, z=3)
        enc_name, enc_sub = spec["encoder"][b]
        text(ex0 + 1.0, mid + 4.9, enc_name, fontweight="bold", fontsize=8.6, va="center")
        text(ex0 + 1.0, mid + 3.1, enc_sub, fontsize=7.4, color=INK2, va="center")
        if spec["stages"]:
            for k, stage in enumerate(spec["stages"]):
                bx = ex0 + 1.2 + k * 5.0
                hh = 5.6 - 1.0 * k
                rbox(bx, mid - 2.0 - hh / 2, 3.8, hh, fc=c, ec=c, r=0.4, z=4, alpha=0.22 + 0.16 * k)
                text(bx + 1.9, mid - 6.6, stage, ha="center", va="center", fontsize=6.8)
                if k < 3:
                    arrow((bx + 3.9, mid - 2.0), (bx + 5.0, mid - 2.0), c=MUTED, lw=0.9)
            text(ex1 - 0.6, mid - 2.0, "GAP", ha="right", va="center", fontsize=7.0, color=INK2)
        else:
            rbox(ex0 + 1.2, mid - 5.8, 4.6, 7.4, fc=c, ec=c, r=0.4, z=4, alpha=0.28)
            text(ex0 + 3.5, mid - 0.4, "64", ha="center", va="center", fontsize=7.4,
                 fontweight="bold")
            text(ex0 + 3.5, mid - 2.5, "parches", ha="center", va="center", fontsize=6.6)
            text(ex0 + 3.5, mid - 4.3, "de 8³", ha="center", va="center", fontsize=6.6)
            for k in range(12):
                rbox(ex0 + 6.8 + k * 0.92, mid - 5.8, 0.62, 7.4, fc=c, ec=c, r=0.2, z=4,
                     alpha=0.25 + 0.05 * k)
            text(ex0 + 12.3, mid - 6.7, "12 bloques · dim 384", ha="center", va="center", fontsize=6.8)
            text(ex1 - 0.5, mid - 2.0, "CLS\n+ parches", ha="right", va="center", fontsize=6.8,
                 color=INK2)
        # representation h
        arrow((ex1 + 0.1, mid), (52.0, mid), c=c)
        rbox(52.1, mid - 4.4, 1.5, 8.8, fc=c, ec=c, r=0.4, z=4, alpha=0.9)
        text(52.85, mid + 5.4, "h", ha="center", va="center", fontweight="bold", fontsize=9)
        text(52.85, mid - 5.4, spec["h"], ha="center", va="center", fontsize=7.6)
        if spec["probe"] == b:
            text(52.85, mid - 7.6, "→ probe", ha="center", va="center", fontsize=7.6,
                 fontweight="bold", color=c,
                 bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": c, "lw": 1.1})
        # heads
        heads = spec["heads"][b]
        arrow((53.7, mid), (56.4, mid), c=c)
        hx0, hx1 = 56.5, 78.5
        if model == "dinov2":
            for k, head in enumerate(heads):
                y0 = mid + 0.3 if k == 0 else mid - 6.9
                rbox(hx0, y0, hx1 - hx0, 6.6, fc=fill, ec=c, lw=1.3, ls=ls, z=3)
                for j, line in enumerate(head):
                    text((hx0 + hx1) / 2, y0 + 5.0 - 1.75 * j, line, ha="center", va="center",
                         fontsize=8.0 if j == 0 else 7.1, fontweight="bold" if j == 0 else "normal",
                         color=INK if j == 0 else INK2)
        else:
            width = (hx1 - hx0 - 1.6 * (len(heads) - 1)) / len(heads)
            for k, head in enumerate(heads):
                x0 = hx0 + k * (width + 1.6)
                if head is None:
                    text(x0 + width / 2, mid + 0.6, "sin predictor:", ha="center", va="center",
                         fontsize=7.6, color=MUTED, style="italic")
                    text(x0 + width / 2, mid - 1.2, "la asimetría de BYOL", ha="center",
                         va="center", fontsize=7.6, color=MUTED, style="italic")
                    continue
                rbox(x0, mid - 5.0, width, 10.0, fc=fill, ec=c, lw=1.3, ls=ls, z=3)
                for j, line in enumerate(head):
                    text(x0 + width / 2, mid + 3.0 - 2.0 * j, line, ha="center", va="center",
                         fontsize=8.4 if j == 0 else 7.2, fontweight="bold" if j == 0 else "normal",
                         color=INK if j == 0 else INK2)
                if k < len(heads) - 1 and heads[k + 1] is not None:
                    arrow((x0 + width + 0.05, mid), (x0 + width + 1.55, mid), c=c)
        if spec["out"]:
            text(hx1 + 0.4, mid + 2.2, spec["out"][b], fontsize=9.5, fontweight="bold", color=c,
                 va="center")

    # ---- link between the branches (encoders, and heads for the EMA copies)
    kind, label = spec["link"]
    if kind == "shared":
        arrow((47.5, 34.3), (47.5, 24.5), c=INK2, style="<|-|>", lw=1.6)
        text(48.4, 30.0, label, fontsize=8.2, color=INK, va="center", fontweight="bold")
    else:
        arrow((47.5, 34.3), (47.5, 24.5), c=SLATE, ls=(0, (4, 3)), lw=1.5)
        top_heads = 37.0 if model != "dinov2" else 35.1
        bottom_heads = 23.0 if model != "dinov2" else 24.9
        x_heads = 62.0 if model != "dinov2" else 74.0
        arrow((x_heads, top_heads), (x_heads, bottom_heads), c=SLATE, ls=(0, (4, 3)), lw=1.5)
        text(48.4, 30.0, label, fontsize=7.9, color=INK2, va="center")
        text(x_heads + 0.8, 30.0, "EMA", fontsize=7.9, color=INK2, va="center")

    # ---- loss
    lx0, lx1, ly0, ly1 = 83.2, 99.2, 18.0, 42.0
    rbox(lx0, ly0, lx1 - lx0, ly1 - ly0, fc=LOSS_FC, ec=LOSS_EC, lw=1.4, z=3)
    for k, line in enumerate(spec["loss"]):
        text((lx0 + lx1) / 2, ly1 - 2.5 - 2.8 * k, line, ha="center", va="center",
             fontsize=9.2 if k == 0 else 7.7, fontweight="bold" if k == 0 else "normal")
    arrow((79.2, mids[0]), (lx0 - 0.1, 36.0), c=spec["colors"][0], lw=1.7)
    arrow((79.2, mids[1]), (lx0 - 0.1, 24.0), c=spec["colors"][1], lw=1.7)
    if ema:  # stop-gradient on the target / teacher side
        ax.add_patch(Circle((81.2, 21.0), 0.95, fc="white", ec=LOSS_EC, lw=1.4, zorder=9))
        text(81.2, 21.0, "sg", ha="center", va="center", fontsize=6.8, fontweight="bold",
             color=LOSS_EC, zorder=10)
    text((lx0 + lx1) / 2, 16.8, spec["grad"], ha="center", va="top", fontsize=7.3, color=INK2)

    # ---- note and training chips
    text(1.5, 5.4, spec["note"], fontsize=8.4, color=INK2, va="center", style="italic")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    x = 1.5
    for chip in spec["chips"]:
        t = text(x, 2.2, chip, fontsize=7.8, va="center", ha="left",
                 bbox={"boxstyle": "round,pad=0.35,rounding_size=0.6", "fc": "white", "ec": main_c,
                       "lw": 0.9})
        box = t.get_window_extent(renderer).transformed(ax.transData.inverted())
        x = box.x1 + 1.3
    rep.save(fig, f"arch_{model}.png")


def main() -> None:
    _, _, hindex = rep.cache("hippo_s32")
    _, _, mindex = rep.cache("multi3_s32")
    val = [r["idx"] for r in rep.manifest() if r["split"] == "val"]
    idx = next(i for i in val if i in hindex and i in mindex)  # the subject of views.png
    for model in ("vicreg", "byol", "dinov2"):
        figure(model, idx)


if __name__ == "__main__":
    main()
