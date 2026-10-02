"""Figures for ``docs/informe_split361.tex`` (RunPod runs on the 361/63/84 split).

Reads ``campaign_ssl3d_runpod/split_361_63_84/manifest.csv``, the runs of
``campaign_ssl3d_runpod/split361_30`` (``collapse.json``, ``DELTAS.md``), the ROI
caches (``hippo_s32`` and ``multi3_s32``) and one MNI volume from the LMDB; writes
PNG figures to ``docs/figures_split361/`` and copies the figures that
``scripts/plot_champions.py`` and ``scripts/plot_byol_diagnosis.py`` already made.

Usage: ``python scripts/build_report_split361.py`` then, from ``docs/``,
``pdflatex informe_split361.tex`` twice.
"""

from __future__ import annotations

import csv
import json
import pickle
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "docs" / "figures_split361"
RUNPOD = ROOT / "campaign_ssl3d_runpod"
CAMPAIGN = RUNPOD / "split361_30"
MANIFEST = RUNPOD / "split_361_63_84" / "manifest.csv"

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, OTHER = "#e1e0d9", "#c3c2b7", "#c3c2b7"
COLOR = {"vicreg": "#2a78d6", "byol": "#eb6834", "dinov2": "#1baf7a"}
LABEL = {"vicreg": "VICReg", "byol": "BYOL", "dinov2": "DINOv2"}
RUN = {"vicreg": "vicreg_campaign__geo__roi32", "byol": "byol_wide_dproj0.1__noise__roi32",
       "dinov2": "dinov2_campaign__elastic__roimulti3"}
REGION_COLOR = {"hippocampus": "#ffd166", "ventricle": "#4cc9f0", "posteromedial": "#f72585"}
REGION_LABEL = {"hippocampus": "hipocampo", "ventricle": "ventrículo lateral",
                "posteromedial": "corteza posteromedial"}

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9,
    "axes.facecolor": SURFACE, "figure.facecolor": "white", "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 9.5,
    "axes.titleweight": "semibold", "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "lines.solid_capstyle": "round", "legend.frameon": False, "legend.fontsize": 8.5,
    "savefig.dpi": 200, "savefig.bbox": "tight",
})


def save(fig: plt.Figure, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name)
    plt.close(fig)
    print("wrote", name)


def bare(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def manifest() -> list[dict]:
    with MANIFEST.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def cache(name: str) -> tuple[np.ndarray, dict, dict[str, int]]:
    array = np.load(ROOT / f"data/roi3d/{name}.npy", mmap_mode="r")
    meta = json.loads((ROOT / f"data/roi3d/{name}.json").read_text(encoding="utf-8"))
    return array, meta, {idx: i for i, idx in enumerate(meta["idx"])}


# ---------------------------------------------------------------- split
def split_counts() -> dict[str, dict[str, int]]:
    """Subject counts per split with the pipeline's strict AD/HC labels (SPLIT.md)."""
    text = (MANIFEST.parent / "SPLIT.md").read_text(encoding="utf-8")
    out = {}
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells[0] in ("train", "val", "test"):
            out[cells[0]] = {"HC": int(cells[2]), "AD": int(cells[3]), "otros": int(cells[4]),
                             "sesiones": int(cells[5])}
    return out


def fig_split(counts: dict[str, dict[str, int]]) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 2.3))
    order = ["test", "val", "train"]
    parts = [("HC", COLOR["vicreg"]), ("AD", COLOR["byol"]), ("otros", OTHER)]
    for y, split in enumerate(order):
        left = 0
        for part, color in parts:
            n = counts[split][part]
            ax.barh(y, n, left=left, height=0.55, color=color, edgecolor="white", linewidth=2)
            if n >= 18:
                ax.text(left + n / 2, y, str(n), ha="center", va="center", fontsize=8.5,
                        color="white" if part != "otros" else INK, fontweight="semibold")
            left += n
        ax.text(left + 5, y, f"{left} sujetos · {counts[split]['sesiones']} sesiones", va="center",
                fontsize=8.5, color=INK2)
    ax.set_yticks(range(3), order)
    ax.set_xlim(0, 470)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("sujetos", fontsize=8)
    handles = [Rectangle((0, 0), 1, 1, color=c) for _, c in parts]
    ax.legend(handles, ["HC (control sano)", "AD (Alzheimer)", "otros diagnósticos (solo SSL)"],
              ncol=3, loc="upper center", bbox_to_anchor=(0.45, 1.3))
    save(fig, "split.png")


# ---------------------------------------------------------------- regions of interest
def mni_volume(idx: str) -> np.ndarray:
    import lmdb

    from t5_ssl.data.oasis3_datamodule import normalize_volume
    from t5_ssl.data.paths import DEFAULT_LMDB
    env = lmdb.open(str(DEFAULT_LMDB), readonly=True, lock=False, readahead=False)
    with env.begin() as txn:
        payload = pickle.loads(txn.get(idx.encode("ascii")))
    return normalize_volume(np.asarray(payload["image"]))


def fig_roi() -> None:
    """Where each model's cubes sit: hippocampal pair (CNNs) and three regions (DINOv2)."""
    _, meta, _ = cache("multi3_s32")
    volume = mni_volume(meta["idx"][0])
    extract, size = meta["extract"], meta["size"]
    starts, centers = meta["cube_starts"], meta["centers_voxel"]
    m = (extract - size) // 2

    def cube(ax: plt.Axes, start: list[int], axes_: tuple[int, int], color: str, dashed: bool) -> None:
        h, v = axes_
        ax.add_patch(Rectangle((start[h] + m - 0.5, start[v] + m - 0.5), size, size, fill=False,
                               edgecolor=color, linewidth=1.8))
        if dashed:
            ax.add_patch(Rectangle((start[h] - 0.5, start[v] - 0.5), extract, extract, fill=False,
                                   edgecolor=color, linewidth=1.0, linestyle=(0, (3, 2))))

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.6), gridspec_kw={"width_ratios": [1, 1, 1.18]})
    z = round((centers["hippocampus_left"][2] + centers["hippocampus_right"][2]) / 2)
    y = round((centers["hippocampus_left"][1] + centers["hippocampus_right"][1]) / 2)
    x = round(centers["posteromedial_left"][0])
    # (a) VICReg / BYOL: hippocampal pair, axial and coronal
    for ax, image, axes_, title in ((axes[0], volume[:, :, z].T, (0, 1), f"axial (z = {z})"),
                                    (axes[1], volume[:, y, :].T, (0, 2), f"coronal (y = {y})")):
        ax.imshow(image, cmap="gray", origin="lower", vmin=0, vmax=1)
        for side in ("left", "right"):
            cube(ax, starts[f"hippocampus_{side}"], axes_, REGION_COLOR["hippocampus"], True)
        ax.set_title(f"VICReg y BYOL · {title}", fontsize=9)
        bare(ax)
    # (b) DINOv2: three regions, left hemisphere, sagittal projection
    ax = axes[2]
    ax.imshow(volume[x, :, :].T, cmap="gray", origin="lower", vmin=0, vmax=1)
    for region in ("hippocampus", "ventricle", "posteromedial"):
        cube(ax, starts[f"{region}_left"], (1, 2), REGION_COLOR[region], False)
    ax.set_title(f"DINOv2 · sagital (x = {x}), hemisferio izq.", fontsize=9)
    bare(ax)
    handles = [Rectangle((0, 0), 1, 1, fill=False, edgecolor=REGION_COLOR[r], linewidth=2)
               for r in REGION_COLOR]
    ax.legend(handles, [REGION_LABEL[r] for r in REGION_COLOR], loc="upper center",
              bbox_to_anchor=(0.5, -0.02), ncol=1, fontsize=8)
    fig.suptitle("Cubo evaluado de 32 vóxeles = 64 mm (sólido) y cubo guardado de 38 (punteado), "
                 "volumen MNI de 2 mm", fontsize=9, color=INK2, y=1.0)
    save(fig, "roi_cubes.png")


def fig_inputs() -> None:
    """Central coronal slice of every input channel: 2 for the CNNs, 6 for DINOv2."""
    hippo, hmeta, hindex = cache("hippo_s32")
    multi, mmeta, mindex = cache("multi3_s32")
    idx = next(i for i in mmeta["idx"] if i in hindex)
    m = (mmeta["extract"] - 32) // 2
    core = slice(m, m + 32)
    fig, axes = plt.subplots(2, 6, figsize=(9.4, 3.6))
    two = hippo[hindex[idx]][:, core, core, core].astype(np.float32)
    six = multi[mindex[idx]][:, core, core, core].astype(np.float32)
    names = [("hipocampo", "izq."), ("hipocampo", "der. espejado"), ("ventrículo", "izq."),
             ("ventrículo", "der. espejado"), ("posteromedial", "izq."), ("posteromedial", "der. espejado")]
    regions = ["hippocampus", "hippocampus", "ventricle", "ventricle", "posteromedial", "posteromedial"]
    for c in range(6):
        for row, data in enumerate((two, six)):
            ax = axes[row, c]
            bare(ax)
            if row == 0 and c >= 2:
                ax.set_visible(False)
                continue
            ax.imshow(data[c][:, 16, :].T, cmap="gray", origin="lower", vmin=0, vmax=1)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(REGION_COLOR[regions[c]])
                spine.set_linewidth(2)
            ax.set_title(f"{names[c][0]}\n{names[c][1]}", fontsize=8)
    axes[0, 0].set_ylabel("VICReg / BYOL\n2 canales", fontsize=9, color=INK)
    axes[1, 0].set_ylabel("DINOv2\n6 canales", fontsize=9, color=INK)
    axes[0, 2].set_visible(True)
    axes[0, 2].axis("off")
    axes[0, 2].text(0.0, 0.5, "Entrada: tensor 2×32³ (CNN) o 6×32³ (ViT).\nEl hemisferio derecho se espeja: "
                    "ambos lados\nquedan orientados igual y un volteo de la cabeza\n= intercambiar canales.",
                    fontsize=8.5, color=INK2, va="center", transform=axes[0, 2].transAxes)
    save(fig, "inputs.png")


# ---------------------------------------------------------------- augmentation views
def view_transform(recipe: str):
    from t5_ssl.data.roi3d import ROIView3DTransform
    from t5_ssl.recipes import aug_spec
    s = aug_spec(recipe, 32)
    return ROIView3DTransform(size=32, side_range=tuple(s["side_range"]), lr_swap_p=0.0,
                              ap_flip_p=s["ap_flip_p"], degrees=s["affine_degrees"],
                              blur_p=s["blur_p"], blur_sigma=tuple(s["blur_sigma"]),
                              noise_std=s["noise_std"], intensity_scale=tuple(s["intensity_scale"]),
                              torchio_transforms=s["torchio"], rotation_padding=s["rotation_padding"],
                              centered_crop=s["centered_crop"])


STACK = ("sagital", "axial", "coronal")  # central slices drawn back to front
STEP = 6                                  # offset between stacked slices, in voxels


def central_slice(volume: np.ndarray, plane: str) -> np.ndarray:
    if plane == "sagital":
        return volume[16, :, :].T
    if plane == "axial":
        return volume[:, :, 16].T
    return volume[:, 16, :].T


def slice_stack(ax: plt.Axes, volume: np.ndarray, labels: bool = False) -> None:
    """The three central planes of one 32^3 volume, stacked like cards: the network gets them all."""
    n = len(STACK)
    for k, plane in enumerate(STACK):
        off = (n - 1 - k) * STEP
        extent = (off, off + 32, off, off + 32)
        z = 3 * k  # images draw below patches by default: order every layer explicitly
        ax.imshow(central_slice(volume, plane), cmap="gray", origin="lower", vmin=0, vmax=1,
                  extent=extent, interpolation="nearest", zorder=z)
        front = k == n - 1
        if not front:  # fade the slices behind
            ax.add_patch(Rectangle((off, off), 32, 32, facecolor="white", alpha=0.45 - 0.15 * k,
                                   edgecolor="none", zorder=z + 1))
        ax.add_patch(Rectangle((off, off), 32, 32, fill=False, edgecolor=INK if front else INK2,
                               linewidth=1.2 if front else 0.6, zorder=z + 2))
        if labels:  # name each plane on the strip of it that stays visible
            x, y, va = (off + 1, off + 1, "bottom") if front else (off + 1, off + 32 - STEP / 2, "center")
            ax.text(x, y, plane, fontsize=6.5, color="white", va=va, ha="left", zorder=3 * n + 1,
                    bbox={"boxstyle": "round,pad=0.15", "facecolor": INK, "alpha": 0.7, "edgecolor": "none"})
    ax.set_xlim(-1, 32 + (n - 1) * STEP + 1)
    ax.set_ylim(-1, 32 + (n - 1) * STEP + 1)
    ax.set_aspect("equal")
    bare(ax)


def fig_views() -> None:
    """Clean cube and two views per model recipe, each a stack of slices of the 3D volume."""
    import torch

    from t5_ssl.data.roi3d import ROIEval3D
    hippo, _, hindex = cache("hippo_s32")
    multi, mmeta, mindex = cache("multi3_s32")
    val = [r["idx"] for r in manifest() if r["split"] == "val"]
    idx = next(i for i in val if i in hindex and i in mindex)
    random.seed(7), np.random.seed(7), torch.manual_seed(7)
    rows = [("vicreg", "geo", hippo[hindex[idx]], "recorte 32–35 vóx. + reescalado, rotación ±5° en 3 ejes (borde)"),
            ("byol", "noise", hippo[hindex[idx]], "ruido gaussiano σ ∈ [0, 0.22] en cada vóxel, sin geometría"),
            ("dinov2", "elastic", multi[mindex[idx]], "deformación elástica 3D (7 puntos de control, ≤ 2 vóx.)")]
    fig, axes = plt.subplots(3, 3, figsize=(6.6, 7.0))
    for r, (model, recipe, raw, text) in enumerate(rows):
        cube = torch.from_numpy(np.asarray(raw, dtype=np.float32))
        transform = view_transform(recipe)
        clean = ROIEval3D(32)(cube)[0]
        views = [transform(cube)[0][0] for _ in range(2)]
        tiles = [(clean, "cubo limpio (probe)"), (views[0], "vista 1"), (views[1], "vista 2")]
        for c, (volume, title) in enumerate(tiles):
            ax = axes[r, c]
            slice_stack(ax, volume.numpy(), labels=c == 0)
            if r == 0:
                ax.set_title(title, fontsize=8.5)
        axes[r, 0].set_ylabel(f"{LABEL[model]}\n«{recipe}»", fontsize=9, color=COLOR[model],
                              fontweight="semibold")
        axes[r, 1].text(0.5, -0.06, text, transform=axes[r, 1].transAxes, ha="center", va="top",
                        fontsize=8, color=INK2)
    fig.suptitle("Cada vista es un volumen 3D de 32×32×32 vóxeles: la red recibe todos sus cortes\n"
                 "en los tres planos a la vez (al frente el corte coronal central; atrás, el axial y el sagital)",
                 fontsize=9, color=INK2, y=0.995)
    fig.subplots_adjust(hspace=0.3, wspace=0.06, top=0.89)
    save(fig, "views.png")


# ---------------------------------------------------------------- architectures
ARCH = {
    "vicreg": {
        "subtitle": "receta «geo» · hipocampos 2×32³",
        "encoder": "ConvNeXt-T 3D  ·  31.3 M parámetros",
        "weights": "pesos ImageNet-1k (torchvision), inflados de 2D a 3D",
        "stages": [("96 × 8³", 8), ("192 × 4³", 4), ("384 × 2³", 2), ("768 × 1³", 1)],
        "vector": ("h", "768"),
        "heads": [["Proyector", "768 → 2048 → 2048 → 2048", "BN · ReLU · dropout 0.1 · 10.0 M"]],
        "loss": ["Pérdida VICReg", "25 · invarianza", "+ 25 · varianza", "+ 1 · covarianza"],
        "twin": None,
        "chips": ["AdamW · lr 10⁻⁴", "weight decay 10⁻⁶", "warmup 6 ép. → coseno hasta 10⁻⁶",
                  "lote 16 · 30 épocas · bf16", "una sola red: sin copia EMA"],
    },
    "byol": {
        "subtitle": "receta «noise» · hipocampos 2×32³",
        "encoder": "ResNet-18 3D sin max-pool  ·  33.2 M",
        "weights": "pesos ImageNet-1k (torchvision), inflados de 2D a 3D",
        "stages": [("64 × 16³", 16), ("128 × 8³", 8), ("256 × 4³", 4), ("512 × 2³", 2)],
        "vector": ("h", "512"),
        "heads": [["Proyector", "512 → 4096 → 256", "BN · dropout 0.1 · 3.2 M"],
                  ["Predictor", "256 → 4096 → 256", "BN · 2.1 M"]],
        "loss": ["Pérdida BYOL", "2 − 2·cos(p, z̄)", "simétrica en las", "dos vistas"],
        "twin": "target = copia lenta (EMA τ 0.996 → 1), sin gradiente",
        "chips": ["AdamW · lr 10⁻⁴", "weight decay 10⁻⁶", "warmup 6 ép. → coseno hasta 10⁻⁶",
                  "lote 16 · 30 épocas · bf16", "EMA τ 0.996 → 1"],
    },
    "dinov2": {
        "subtitle": "receta «elastic» + máscara iBOT · 3 regiones 6×32³",
        "encoder": "ViT-S 3D  ·  22.5 M parámetros",
        "weights": "pesos DeiT-S ImageNet-1k, parche 16² → 8³",
        "stages": None,
        "vector": ("CLS", "384"),
        "heads": [["Cabeza DINO (CLS)", "384 → 2048 → 2048 → 256", "→ 4096 prototipos · 6.6 M"],
                  ["Cabeza iBOT (parches)", "misma forma", "→ 4096 prototipos · 6.6 M"]],
        "loss": ["Pérdida DINOv2", "DINO + 1·iBOT", "+ 0.1·KoLeo", "temp. 0.1 / 0.01"],
        "twin": "profesor = copia lenta (EMA τ 0.99 → 1); el probe usa su CLS",
        "chips": ["AdamW · lr 10⁻⁴", "weight decay 0.04 → 0.2", "warmup 6 ép. → coseno hasta 10⁻⁶",
                  "lote 16 · 30 épocas · bf16", "recorte de gradiente 3.0", "drop path 0.1"],
    },
}


def fig_architectures() -> None:
    """One card per model: input → encoder → probe vector → heads → loss, plus key settings."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon
    H, GAP = 32.0, 2.0
    fig = plt.figure(figsize=(10.6, 10.2))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 3 * H + 2 * GAP)
    ax.axis("off")

    def rbox(x, y, w, h, fc="white", ec=INK2, lw=1.0, ls="-", alpha=1.0, r=1.2, z=2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                    fc=fc, ec=ec, lw=lw, ls=ls, alpha=alpha, zorder=z))

    def arrow(x1, y1, x2, y2, color=INK2, z=3):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9,
                                     color=color, lw=1.1, zorder=z))

    def cube(x, y, s, d, color):
        faces = [((x, y), (x + s, y), (x + s, y + s), (x, y + s)),
                 ((x, y + s), (x + d, y + s + d), (x + s + d, y + s + d), (x + s, y + s)),
                 ((x + s, y), (x + s + d, y + d), (x + s + d, y + s + d), (x + s, y + s))]
        for face, alpha in zip(faces, (0.35, 0.55, 0.2), strict=True):
            ax.add_patch(Polygon(face, closed=True, fc=color, ec=color, alpha=alpha, lw=0.8, zorder=3))
        for k in (1, 2):  # voxel grid hint on the front face
            ax.plot([x + k * s / 3] * 2, [y, y + s], color="white", lw=0.6, zorder=4)
            ax.plot([x, x + s], [y + k * s / 3] * 2, color="white", lw=0.6, zorder=4)

    for row, model in enumerate(("vicreg", "byol", "dinov2")):
        spec, color = ARCH[model], COLOR[model]
        y0 = (2 - row) * (H + GAP)
        cy = y0 + 17.5
        rbox(0.6, y0 + 0.4, 98.8, H - 0.8, fc=color, ec=color, alpha=0.07, r=2.0, z=1)
        rbox(0.6, y0 + 0.4, 98.8, H - 0.8, fc="none", ec=color, lw=1.2, r=2.0, z=1)
        ax.text(2.6, y0 + H - 3.6, LABEL[model], fontsize=14, fontweight="bold", color=color, va="center")
        ax.text(13.0, y0 + H - 3.6, spec["subtitle"], fontsize=9.5, color=INK2, va="center")
        # input cube
        cube(3.0, cy - 3.2, 5.4, 2.2, color)
        ax.text(6.4, cy - 6.0, "entrada", fontsize=8, color=INK, ha="center", va="top")
        ax.text(6.4, cy - 8.0, spec["subtitle"].split("· ")[-1].split(" ")[-1], fontsize=8,
                color=INK2, ha="center", va="top")
        arrow(11.4, cy, 13.4, cy)
        # encoder (and its slow EMA twin behind it)
        ex0, ex1, ey0, ey1 = 13.6, 52.0, y0 + 8.6, y0 + 25.2
        if spec["twin"]:
            rbox(ex0 + 1.0, ey0 - 1.2, ex1 - ex0, ey1 - ey0, fc="#f4f4f2", ec=MUTED, lw=1.0, ls="--", z=2)
            ax.text(ex1 + 0.6, ey0 - 2.2, spec["twin"], fontsize=7.6, color=INK2, ha="right", va="top",
                    style="italic")
        rbox(ex0, ey0, ex1 - ex0, ey1 - ey0, fc="white", ec=color, lw=1.4, z=3)
        ax.text(ex0 + 1.4, ey1 - 1.9, spec["encoder"], fontsize=9, fontweight="bold", color=INK, va="center", zorder=5)
        ax.text(ex0 + 1.4, ey1 - 4.0, spec["weights"], fontsize=7.8, color=INK2, va="center", zorder=5)
        if spec["stages"]:
            n, w, g = 4, 6.0, 2.6
            mid = ey0 + 7.4
            for k, (text, side) in enumerate(spec["stages"]):
                h = 2.0 + 5.6 * np.log2(side + 1) / np.log2(17)
                bx = ex0 + 2.4 + k * (w + g)
                by = mid - h / 2
                rbox(bx, by, w, h, fc=color, ec=color, alpha=0.25 + 0.15 * k, r=0.6, z=4)
                ax.text(bx + w / 2, ey0 + 1.6, text, fontsize=7.6, ha="center", va="center", color=INK, zorder=5)
                if k < n - 1:
                    arrow(bx + w + 0.2, mid, bx + w + g - 0.2, mid, color=MUTED)
            ax.text(ex0 + 2.4 + n * (w + g) - g + 0.6, mid, "GAP", fontsize=7.6, color=INK2, va="center", zorder=5)
        else:  # ViT: patch embedding, 12 blocks, CLS token
            rbox(ex0 + 2.0, ey0 + 3.4, 7.6, 7.8, fc=color, ec=color, alpha=0.3, r=0.6, z=4)
            ax.text(ex0 + 5.8, ey0 + 9.2, "parches", fontsize=7.6, ha="center", va="center", zorder=5)
            ax.text(ex0 + 5.8, ey0 + 7.3, "8³ → 64", fontsize=7.6, ha="center", va="center", zorder=5)
            ax.text(ex0 + 5.8, ey0 + 5.4, "tokens", fontsize=7.6, ha="center", va="center", zorder=5)
            arrow(ex0 + 9.8, ey0 + 7.3, ex0 + 11.6, ey0 + 7.3, color=MUTED)
            for k in range(12):
                rbox(ex0 + 11.9 + k * 1.75, ey0 + 3.4, 1.25, 7.8, fc=color, ec=color, alpha=0.3 + 0.04 * k, r=0.4, z=4)
            ax.text(ex0 + 11.9 + 6 * 1.75 - 0.25, ey0 + 1.6, "12 bloques · 6 cabezas · dim 384",
                    fontsize=7.6, ha="center", va="center", color=INK, zorder=5)
            ax.text(ex0 + 33.6, ey0 + 7.3, "→ CLS", fontsize=7.6, color=INK2, va="center", zorder=5)
        # probe vector
        vx = 54.0
        arrow(ex1 + 0.2, cy, vx - 0.1, cy)
        rbox(vx, cy - 5.4, 1.6, 10.8, fc=color, ec=color, alpha=0.85, r=0.5, z=4)
        ax.text(vx + 0.8, cy + 6.6, spec["vector"][0], fontsize=8.5, fontweight="bold", ha="center", color=INK)
        ax.text(vx + 0.8, cy - 6.8, spec["vector"][1], fontsize=8, ha="center", va="top", color=INK)
        ax.text(vx + 0.8, cy - 9.0, "probe", fontsize=7.4, ha="center", va="top", color=color, fontweight="bold")
        # heads
        hx0, hx1 = 59.0, 79.0
        arrow(vx + 1.7, cy, hx0 - 0.1, cy)
        heads = spec["heads"]
        hh = 7.4 if len(heads) == 2 else 12.0
        tops = [cy + 6.4] if len(heads) == 1 else [cy + 8.0, cy - 0.4]
        for top, (title, line1, line2) in zip(tops, heads, strict=True):
            rbox(hx0, top - hh, hx1 - hx0, hh, fc="white", ec=color, lw=1.1, z=3)
            mid = top - hh / 2
            dy = 2.2 if len(heads) == 2 else 3.0
            ax.text((hx0 + hx1) / 2, mid + dy, title, fontsize=8.3, fontweight="bold", ha="center", va="center", zorder=5)
            ax.text((hx0 + hx1) / 2, mid, line1, fontsize=7.6, ha="center", va="center", zorder=5)
            ax.text((hx0 + hx1) / 2, mid - dy, line2, fontsize=7.3, ha="center", va="center", color=INK2, zorder=5)
        if len(heads) == 2 and model == "byol":
            arrow((hx0 + hx1) / 2, tops[0] - hh - 0.05, (hx0 + hx1) / 2, tops[1] + 0.05)
        ax.text((hx0 + hx1) / 2, y0 + 7.2, "solo para entrenar", fontsize=7.2, ha="center", color=MUTED, style="italic")
        # loss
        lx0, lx1 = 82.5, 98.0
        arrow(hx1 + 0.2, cy, lx0 - 0.1, cy)
        rbox(lx0, cy - 6.4, lx1 - lx0, 12.8, fc="#fdf2e6", ec="#e67e22", lw=1.1, z=3)
        for k, text in enumerate(spec["loss"]):
            ax.text((lx0 + lx1) / 2, cy + 4.2 - k * 2.8, text, fontsize=8.3 if k == 0 else 7.8,
                    fontweight="bold" if k == 0 else "normal", ha="center", va="center", zorder=5)
        # key settings as chips
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        x = 2.6
        for chip in spec["chips"]:
            t = ax.text(x, y0 + 3.4, chip, fontsize=7.8, va="center", ha="left", color=INK, zorder=5,
                        bbox={"boxstyle": "round,pad=0.35,rounding_size=0.6", "fc": "white", "ec": color,
                              "lw": 0.9})
            box = t.get_window_extent(renderer).transformed(ax.transData.inverted())
            x = box.x1 + 1.6
    save(fig, "architectures.png")


# ---------------------------------------------------------------- results
def deltas() -> dict[str, dict]:
    text = (CAMPAIGN / "DELTAS.md").read_text(encoding="utf-8")
    out = {}
    num = r"([+-]?\d+\.\d+)"
    for model, key in (("VICReg", "vicreg"), ("BYOL", "byol"), ("DINOv2", "dinov2")):
        line = next(ln for ln in text.splitlines() if ln.startswith(f"| {model} "))
        values = [float(v) for v in re.findall(num, line)]
        out[key] = {"auroc_cv": values[0], "auroc_test": values[1],
                    "d_cv": values[2:5], "d_test": values[5:8]}
    return out


def fig_delta(data: dict[str, dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.4), sharey=True)
    models = ["dinov2", "byol", "vicreg"]
    for ax, key, title in ((axes[0], "d_cv", "CV 5×5 (329 sujetos de train+val)"),
                           (axes[1], "d_test", "Test (84 sujetos nunca vistos)")):
        ax.axvline(0, color=INK2, linewidth=1)
        for y, model in enumerate(models):
            mid, lo, hi = data[model][key]
            ax.plot([lo, hi], [y, y], color=COLOR[model], linewidth=2.5, solid_capstyle="butt")
            ax.plot(mid, y, "o", color=COLOR[model], markersize=7)
            ax.text(mid, y + 0.26, f"{mid:+.3f}", ha="center", fontsize=8, color=INK)
        ax.set_title(title, fontsize=9)
        ax.set_xlim(-0.06, 0.19)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Δ AUROC frente al mismo encoder sin SSL", fontsize=8)
    axes[0].set_yticks(range(3), [LABEL[m] for m in models])
    axes[0].set_ylim(-0.5, 2.6)
    save(fig, "delta.png")


def fig_collapse() -> None:
    """Two collapse monitors per model (online encoder, held-out subjects, mean of seeds)."""
    metrics = [("between_over_within", "Var entre sujetos / Var por augmentation", True),
               ("view_matching_top1", "Top-1: la vista encuentra a su gemela", False)]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.8))
    for ax, (key, title, log) in zip(axes, metrics, strict=True):
        for model in ("vicreg", "byol", "dinov2"):
            series: dict[int, list[float]] = defaultdict(list)
            for path in sorted(CAMPAIGN.glob(f"{RUN[model]}_s*/collapse.json")):
                for rec in json.loads(path.read_text(encoding="utf-8")):
                    if rec["role"] == "online":
                        series[rec["epoch"]].append(rec[key])
            epochs = sorted(series)
            ax.plot(epochs, [np.mean(series[e]) for e in epochs], color=COLOR[model],
                    label=LABEL[model])
        if log:
            ax.set_yscale("log")
            ax.axhline(1, color=INK2, linewidth=1, linestyle=":")
            ax.text(30, 1.1, "1 = la augmentation pesa\nlo mismo que el sujeto", ha="right",
                    va="bottom", fontsize=7.5, color=INK2)
        else:
            ax.axhline(1 / 147, color=INK2, linewidth=1, linestyle=":")
            ax.text(30, 0.03, "azar (1/147)", ha="right", fontsize=7.5, color=INK2)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("época", fontsize=8)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.1))
    save(fig, "collapse.png")


def copy_existing() -> None:
    for name in ("split361_metricas.png", "split361_curvas_vicreg.png", "split361_curvas_byol.png",
                 "split361_curvas_dinov2.png", "byol_diag_objetivo.png"):
        FIG.mkdir(parents=True, exist_ok=True)
        shutil.copy2(RUNPOD / "figures" / name, FIG / name)
        print("copied", name)


def main() -> None:
    fig_split(split_counts())
    fig_roi()
    fig_inputs()
    fig_views()
    fig_delta(deltas())
    fig_collapse()
    copy_existing()


if __name__ == "__main__":
    main()
