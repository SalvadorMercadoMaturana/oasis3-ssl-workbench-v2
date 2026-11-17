"""One architecture figure per SSL model, branch by branch, with real augmented views.

    python scripts/5_reports/plot_architectures_split361.py

The two branches of each model sit in separate bands with their own colour: VICReg's
two branches are the same network (shared weights); BYOL's online branch against its
EMA target and DINOv2's student (masked view) against its EMA teacher, the copies
dashed and without gradient (sg). Each branch shows a view of the same validation
subject drawn with the model's own recipe (``geo``, ``noise``, TorchIO ``elastic``),
as stacked central slices of the 32³ volume on the branch colour. Only the key
settings of the 100-epoch champions (split361_100; BYOL with the chosen
``gn_dp0.3_dq0_wd0.01`` heads) are written. Writes
``docs/figures_split361/arch_{vicreg,byol,dinov2}.png``.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_report_split361 as rep

plt = rep.plt
from matplotlib.colors import to_rgba  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

INK, INK2, MUTED = rep.INK, rep.INK2, rep.MUTED
SLATE, VIOLET = "#5f6b73", "#7b52c7"
LOSS_FC, LOSS_EC = "#fdf2e6", "#e67e22"
DASH = (0, (5, 3))
BAND_ALPHA = 0.12

SPEC = {
    "vicreg": {
        "name": "VICReg", "sub": "ConvNeXt-T 3D  ·  hipocampos 2×32³",
        "recipe": "geo", "aug": "recorte + rotación ±5°", "cache": "hippo_s32",
        "colors": (rep.COLOR["vicreg"], VIOLET),
        "bands": [("Rama 1", ""), ("Rama 2", "")],
        "encoder": ["ConvNeXt-T 3D", "ConvNeXt-T 3D"], "kind": "cnn", "h": "768",
        "heads": [[("Proyector", "768 → 2048 → 2048 → 2048")]] * 2,
        "out": ("z₁", "z₂"), "link": "pesos compartidos", "probe": 0,
        "loss": ["VICReg", "25·inv + 25·var + 1·cov"],
        "footer": "100 épocas  ·  lote 16  ·  AdamW lr 10⁻⁴",
    },
    "byol": {
        "name": "BYOL", "sub": "ResNet-18 3D  ·  hipocampos 2×32³",
        "recipe": "noise", "aug": "ruido gaussiano σ ≤ 0.22", "cache": "hippo_s32",
        "colors": (rep.COLOR["byol"], SLATE),
        "bands": [("Online", "recibe gradiente"), ("Target", "copia EMA · sin gradiente")],
        "encoder": ["ResNet-18 3D", "ResNet-18 3D"], "kind": "cnn", "h": "512",
        "heads": [[("Proyector", "512 → 4096 → 256", "GroupNorm", "dropout 0.3"),
                   ("Predictor", "256 → 4096 → 256", "GroupNorm")],
                  [("Proyector", "512 → 4096 → 256"), None]],
        "out": ("p", "z̄"), "link": "EMA", "probe": 0,
        "loss": ["BYOL", "2 − 2·cos(p, z̄)"],
        "footer": "100 épocas  ·  lote 16  ·  AdamW lr 10⁻⁴, weight decay 0.01  ·  EMA τ 0.996 → 1",
    },
    "dinov2": {
        "name": "DINOv2", "sub": "ViT-S 3D  ·  3 regiones 6×32³",
        "recipe": "elastic", "aug": "deformación elástica", "cache": "multi3_s32",
        "colors": (rep.COLOR["dinov2"], SLATE),
        "bands": [("Alumno", "recibe gradiente · parches ocultos"),
                  ("Profesor", "copia EMA · sin gradiente")],
        "encoder": ["ViT-S 3D", "ViT-S 3D"], "kind": "vit", "h": "384",
        "heads": [[("DINO (CLS)", "4096 prototipos · T 0.1"),
                   ("iBOT (parches)", "4096 prototipos · T 0.1")],
                  [("DINO", "T 0.01"), ("iBOT", "T 0.01")]],
        "out": None, "link": "EMA", "probe": 1,
        "loss": ["DINOv2", "DINO + iBOT", "+ 0.1·KoLeo"],
        "footer": "100 épocas  ·  lote 16  ·  AdamW lr 10⁻⁴  ·  EMA τ 0.99 → 1  ·  "
                  "máscara iBOT 10–50 % de los 64 parches",
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
    out[8:24, 16:32, 0:24] = np.nan  # patches i 1-2, j 2-3, k 0-2: crosses all three central slices
    return out


def stack(ax: plt.Axes, volume: np.ndarray, frame: str, tint: str = "white",
          hidden: str | None = None) -> None:
    """Draw the three central planes stacked like cards on a transparent background.

    The slices behind are faded towards `tint` (the band colour); NaN voxels are
    drawn in `hidden`.
    """
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
            ax.add_patch(Rectangle((off, off), 32, 32, facecolor=tint, alpha=0.5 - 0.15 * k,
                                   edgecolor="none", zorder=3 * k + 1))
        ax.add_patch(Rectangle((off, off), 32, 32, fill=False, edgecolor=frame,
                               linewidth=2.2 if front else 0.8, zorder=3 * k + 2))
    ax.set_xlim(-1, 32 + (n - 1) * step + 1)
    ax.set_ylim(-1, 32 + (n - 1) * step + 1)
    ax.set_aspect("equal")
    ax.patch.set_alpha(0.0)  # the band colour shows through
    rep.bare(ax)


def band_tint(color: str) -> tuple[float, float, float]:
    """Return the opaque colour of a band (its colour at BAND_ALPHA over white)."""
    r, g, b, _ = to_rgba(color)
    return tuple(1 - BAND_ALPHA * (1 - v) for v in (r, g, b))


def figure(model: str, idx: str) -> None:
    spec = SPEC[model]
    ema = spec["link"] == "EMA"
    fig = plt.figure(figsize=(12.5, 6.8))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 54)
    ax.axis("off")

    def rbox(x, y, w, h, fc="white", ec=INK2, lw=1.0, ls="-", r=0.9, z=2, alpha=1.0):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                    fc=to_rgba(fc, alpha) if fc != "none" else "none", ec=ec,
                                    lw=lw, ls=ls, zorder=z))

    def arrow(p, q, c=INK2, ls="-", lw=1.5, style="-|>", z=6):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=13, color=c, lw=lw,
                                     ls=ls, zorder=z, shrinkA=0, shrinkB=0))

    def text(x, y, s, **kw):
        kw.setdefault("fontsize", 9.5)
        kw.setdefault("color", INK)
        kw.setdefault("zorder", 8)
        kw.setdefault("va", "center")
        return ax.text(x, y, s, **kw)

    main_c = spec["colors"][0]
    title = text(1.5, 51.3, spec["name"], fontsize=17, fontweight="bold", color=main_c)
    fig.canvas.draw()
    end = title.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transData.inverted()).x1
    text(end + 1.5, 51.1, spec["sub"], fontsize=11.5, color=INK2)

    # ---- input and the split into two views
    clean, first, second = views(model, idx)
    inset = ax.inset_axes((1.0, 20.0, 11.0, 12.6), transform=ax.transData)
    stack(inset, clean, INK)
    text(6.5, 18.6, "cubo limpio", ha="center", fontsize=9.5, fontweight="bold")
    text(6.5, 37.4, f"«{spec['recipe']}»", ha="center", fontsize=10.5, fontweight="bold")
    text(6.5, 35.5, spec["aug"], ha="center", fontsize=8.6, color=INK2)
    mids = (38.0, 15.0)
    arrow((12.4, 29.5), (14.8, mids[0]), c=spec["colors"][0], lw=1.8)
    arrow((12.4, 23.0), (14.8, mids[1]), c=spec["colors"][1], lw=1.8)

    # ---- the two branch bands
    for b, mid in enumerate(mids):
        c = spec["colors"][b]
        copy = ema and b == 1
        ls = DASH if copy else "-"
        rbox(13.6, mid - 10.0, 66.4, 20.0, fc=c, ec="none", r=1.6, z=1, alpha=BAND_ALPHA)
        rbox(13.6, mid - 10.0, 66.4, 20.0, fc="none", ec=c, lw=2.2, ls=ls, r=1.6, z=1)
        name, tag = spec["bands"][b]
        label = text(14.8, mid + 8.0, name, fontsize=12, fontweight="bold", color=c)
        if tag:
            fig.canvas.draw()
            x1 = label.get_window_extent(fig.canvas.get_renderer()).transformed(
                ax.transData.inverted()).x1
            text(x1 + 1.2, mid + 7.9, tag, fontsize=9.2, color=INK2)
        # view on the band colour
        view, hidden = (first if b == 0 else second), None
        if model == "dinov2" and b == 0:
            view, hidden = ibot_mask(view), to_rgba(c, 0.65)
        vin = ax.inset_axes((15.0, mid - 7.8, 11.0, 12.6), transform=ax.transData)
        stack(vin, view, c, tint=band_tint(c), hidden=hidden)
        text(20.5, mid - 8.6, f"vista {b + 1}", ha="center", fontsize=9.5, fontweight="bold",
             color=c)
        arrow((26.4, mid - 1.5), (28.6, mid - 1.5), c=c)
        # encoder
        ex0, ex1, ey0, ey1 = 28.7, 47.5, mid - 7.5, mid + 4.5
        rbox(ex0, ey0, ex1 - ex0, ey1 - ey0, fc="white", ec=c, lw=1.8, ls=ls, z=3)
        enc = spec["encoder"][b] + (" (EMA)" if copy else "")
        text((ex0 + ex1) / 2, ey1 - 2.0, enc, ha="center", fontweight="bold", fontsize=10.5)
        base = ey0 + 1.2
        if spec["kind"] == "cnn":
            for k in range(4):
                hh = 5.4 - 1.0 * k
                rbox(ex0 + 1.6 + k * 4.4, base + (5.4 - hh) / 2, 3.4, hh, fc=c, ec=c, r=0.4, z=4,
                     alpha=0.25 + 0.18 * k)
                if k < 3:
                    arrow((ex0 + 5.1 + k * 4.4, base + 2.7), (ex0 + 5.9 + k * 4.4, base + 2.7),
                          c=MUTED, lw=1.0)
        else:
            rbox(ex0 + 1.4, base, 3.2, 5.4, fc=c, ec=c, r=0.4, z=4, alpha=0.3)
            text(ex0 + 3.0, base + 2.7, "8³", ha="center", fontsize=8.2)
            for k in range(12):
                rbox(ex0 + 5.6 + k * 1.0, base, 0.65, 5.4, fc=c, ec=c, r=0.2, z=4,
                     alpha=0.25 + 0.05 * k)
        # representation h
        arrow((ex1 + 0.1, mid - 1.5), (49.6, mid - 1.5), c=c)
        rbox(49.7, mid - 5.7, 1.6, 8.4, fc=c, ec=c, r=0.4, z=4, alpha=0.9)
        text(50.5, mid + 3.8, "h", ha="center", fontweight="bold", fontsize=10.5)
        text(50.5, mid - 6.8, spec["h"], ha="center", fontsize=8.8)
        if spec["probe"] == b:  # the vector the frozen probe reads
            text(50.5, mid + 6.4, "probe", ha="center", fontsize=8.8, fontweight="bold", color=c,
                 bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": c, "lw": 1.2})
        # heads
        heads = spec["heads"][b]
        hx0, hx1 = 54.0, 77.6
        arrow((51.4, mid - 1.5), (hx0 - 0.1, mid - 1.5), c=c)
        if spec["kind"] == "vit":
            for k, head in enumerate(heads):
                y0 = mid - 0.9 if k == 0 else mid - 7.6
                rbox(hx0, y0, hx1 - hx0, 5.8, fc="white", ec=c, lw=1.4, ls=ls, z=3)
                text((hx0 + hx1) / 2, y0 + 3.9, head[0], ha="center", fontweight="bold", fontsize=9.6)
                text((hx0 + hx1) / 2, y0 + 1.7, head[1], ha="center", fontsize=8.6, color=INK2)
        else:
            width = (hx1 - hx0 - 1.8 * (len(heads) - 1)) / len(heads)
            for k, head in enumerate(heads):
                x0 = hx0 + k * (width + 1.8)
                if head is None:
                    continue
                rbox(x0, mid - 6.5, width, 10.0, fc="white", ec=c, lw=1.4, ls=ls, z=3)
                for j, line in enumerate(head):  # lines centred in the box
                    text(x0 + width / 2, mid - 1.5 + 1.1 * (len(head) - 1) - 2.2 * j, line,
                         ha="center",
                         fontsize=9.8 if j == 0 else 8.6, fontweight="bold" if j == 0 else "normal",
                         color=INK if j == 0 else INK2)
                if k < len(heads) - 1 and heads[k + 1] is not None:
                    arrow((x0 + width + 0.05, mid - 1.5), (x0 + width + 1.75, mid - 1.5), c=c)
        if spec["out"]:
            text(hx1 + 0.5, mid + 1.2, spec["out"][b], fontsize=11, fontweight="bold", color=c)

    # ---- link between the branches
    if not ema:
        arrow((38.1, 30.4), (38.1, 19.6), c=INK2, style="<|-|>", lw=1.8)
        text(39.2, 26.5, spec["link"], fontsize=9.6, fontweight="bold")
    else:
        heads_x = 61.5 if spec["kind"] == "cnn" else 74.5
        for x in (38.1, heads_x):
            arrow((x, 30.4 if x == 38.1 else (31.5 if spec["kind"] == "cnn" else 30.4)),
                  (x, 19.6 if x == 38.1 else (18.5 if spec["kind"] == "cnn" else 19.9)),
                  c=SLATE, ls=(0, (4, 3)), lw=1.6)
            text(x + 0.9, 26.5, "EMA", fontsize=9.4, color=SLATE, fontweight="bold")

    # ---- loss
    lx0, lx1, ly0, ly1 = 83.0, 98.8, 19.5, 33.5
    rbox(lx0, ly0, lx1 - lx0, ly1 - ly0, fc=LOSS_FC, ec=LOSS_EC, lw=1.5, z=3)
    lines = spec["loss"]
    top = (ly0 + ly1) / 2 + 1.3 * (len(lines) - 1)
    for k, line in enumerate(lines):
        text((lx0 + lx1) / 2, top - 2.6 * k, line, ha="center",
             fontsize=12 if k == 0 else 9.6, fontweight="bold" if k == 0 else "normal")
    arrow((80.0, mids[0] - 1.5), (lx0 - 0.1, 30.5), c=spec["colors"][0], lw=1.9)
    arrow((80.0, mids[1] - 1.5), (lx0 - 0.1, 22.5), c=spec["colors"][1], lw=1.9)
    if ema:  # stop-gradient on the target / teacher side
        ax.add_patch(Circle((81.4, 18.1), 1.05, fc="white", ec=LOSS_EC, lw=1.5, zorder=9))
        text(81.4, 18.1, "sg", ha="center", fontsize=7.6, fontweight="bold", color=LOSS_EC,
             zorder=10)
        text((lx0 + lx1) / 2, 17.4, "sg: sin gradiente", ha="center", fontsize=8.6, color=INK2)

    # ---- key settings
    text(1.5, 2.2, spec["footer"], fontsize=9.6, color=INK2)
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
