"""Figures for ``docs/informe_metodologia_split361.tex`` (data ingestion, volumes, results).

    python scripts/5_reports/build_report_metodologia.py [ingesta|volumenes|arquitecturas|metricas|supervisado|barras|cubos3d|all]

- ``ingesta_3d.png``: where the cubes come from in the 97x115x97 MNI volume (three
  orthogonal planes through the left hippocampus and the six evaluated 32^3 cubes as
  wireframes), the stored 38^3 cube with the evaluated 32^3 inside, the left-right
  mirroring of the right hemisphere, and the 2- and 6-channel input tensors, every
  cube drawn as a volume (its outer faces) in an oblique projection.
- ``volumenes_3d.png``: what each encoder does with the volume: feature maps as 3D
  boxes (side = spatial grid, deck = channels) for the two CNNs, and the 4x4x4 grid
  of 8^3 patches, one patch pulled out, and the 65-token sequence for the ViT.
- ``architectures_100.png``: ``architectures.png`` of build_report_split361 with the
  settings of the 100-epoch run (BYOL GroupNorm heads, 100 epochs, warmup 20).
- ``metricas_100_puntos.png``: the 100-epoch metrics (supervised, SSL, without SSL) as
  a dot plot with every value written next to its point.
- ``supervisado_curvas_100.png``: train and validation BCE of the supervised baseline.
- ``split361_100_metricas_claras.png`` (``barras``): the 100-epoch metric bars for slides, every
  value written above its bar.
- ``roi_cubes_3d.png`` (``cubos3d``): ``roi_cubes.png`` in 3D: the brain planes of the 2D figure
  with each model's cubes, and the six cubes taken apart as channels.

Data: the ROI caches in ``data/roi3d/``, one MNI volume from the LMDB and the runs in
``campaign_ssl3d_runpod/split361_100`` and ``split361_30``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "src")]
import build_report_split361 as rep  # noqa: E402

plt = rep.plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon  # noqa: E402
from matplotlib.transforms import Affine2D  # noqa: E402

INK, INK2, MUTED, BASELINE = rep.INK, rep.INK2, rep.MUTED, rep.BASELINE
COLOR, LABEL = rep.COLOR, rep.LABEL
REGION_COLOR, REGION_LABEL = rep.REGION_COLOR, rep.REGION_LABEL
RUNS = rep.RUNPOD
MODELS = ("vicreg", "byol", "dinov2")

# Oblique (cabinet) projection: x to the right, z up, y (posterior -> anterior) into the page.
DEPTH, ANGLE = 0.5, np.radians(35)
DX, DY = DEPTH * np.cos(ANGLE), DEPTH * np.sin(ANGLE)


def project(x: float, y: float, z: float, s: float = 1.0, o: tuple[float, float] = (0, 0)
            ) -> tuple[float, float]:
    return o[0] + s * (x + y * DX), o[1] + s * (z + y * DY)


def rgba(image: np.ndarray, transparent_below: float | None = None) -> np.ndarray:
    """Grey [0, 1] image -> RGBA, with near-zero voxels transparent if requested."""
    v = np.clip(image, 0, 1)
    out = np.stack([v, v, v, np.ones_like(v)], axis=-1)
    if transparent_below is not None:
        out[..., 3] = (v > transparent_below).astype(float)
    return out


def face(ax: plt.Axes, image: np.ndarray, matrix: list[list[float]], extent: tuple, z: float,
         alpha_below: float | None = None) -> None:
    """Draw ``image`` (rows = v, columns = u) on the plane given by the affine ``matrix``."""
    im = ax.imshow(rgba(image, alpha_below), origin="lower", extent=extent,
                   interpolation="nearest", zorder=z)
    im.set_transform(Affine2D(np.array([*matrix, [0, 0, 1]])) + ax.transData)
    im.set_clip_on(False)


def textured_cube(ax: plt.Axes, vol: np.ndarray, o: tuple[float, float], s: float,
                  edge: str = INK2, lw: float = 1.0, z: float = 2.0, grid: int = 0,
                  grid_color: str = "white") -> None:
    """Draw an (X, Y, Z) volume by its three visible outer faces: front (y = 0), top, right."""
    nx, ny, nz = vol.shape
    # right side (x = nx), top (z = nz), front (y = 0); each face a separate image
    face(ax, vol[nx - 1].T, [[s * DX, 0, o[0] + s * nx], [s * DY, s, o[1]]], (0, ny, 0, nz), z)
    face(ax, vol[:, :, nz - 1].T, [[s, s * DX, o[0]], [0, s * DY, o[1] + s * nz]], (0, nx, 0, ny), z)
    face(ax, vol[:, 0, :].T, [[s, 0, o[0]], [0, s, o[1]]], (0, nx, 0, nz), z + 0.1)
    box_edges(ax, (0, 0, 0), (nx, ny, nz), o, s, edge, lw, z + 0.5)
    if grid:  # patch grid on the three visible faces
        step = nx / grid
        for k in range(1, grid):
            t = k * step
            for a, b in (((t, 0, 0), (t, 0, nz)), ((0, 0, t), (nx, 0, t)),           # front
                         ((t, 0, nz), (t, ny, nz)), ((0, t, nz), (nx, t, nz)),           # top
                         ((nx, t, 0), (nx, t, nz)), ((nx, 0, t), (nx, ny, t))):          # side
                p, q = project(*a, s, o), project(*b, s, o)
                ax.plot([p[0], q[0]], [p[1], q[1]], color=grid_color, lw=0.9, zorder=z + 0.4)


def box_edges(ax: plt.Axes, p0: tuple, size: tuple, o: tuple[float, float], s: float,
              color: str, lw: float, z: float, ls: str | tuple = "-", hidden: bool = False) -> None:
    """Draw the edges of an axis-aligned box (the three hidden ones only if ``hidden``)."""
    x0, y0, z0 = p0
    x1, y1, z1 = x0 + size[0], y0 + size[1], z0 + size[2]
    visible = [((x0, y0, z0), (x1, y0, z0)), ((x1, y0, z0), (x1, y0, z1)), ((x1, y0, z1), (x0, y0, z1)),
               ((x0, y0, z1), (x0, y0, z0)), ((x0, y0, z1), (x0, y1, z1)), ((x1, y0, z1), (x1, y1, z1)),
               ((x0, y1, z1), (x1, y1, z1)), ((x1, y0, z0), (x1, y1, z0)), ((x1, y1, z0), (x1, y1, z1))]
    back = [((x0, y0, z0), (x0, y1, z0)), ((x0, y1, z0), (x1, y1, z0)), ((x0, y1, z0), (x0, y1, z1))]
    for a, b in visible + (back if hidden else []):
        p, q = project(*a, s, o), project(*b, s, o)
        style = (0, (2, 2)) if (a, b) in back else ls
        ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw if (a, b) not in back else lw * 0.7,
                ls=style, zorder=z, solid_capstyle="round")


def solid_box(ax: plt.Axes, p0: tuple, size: tuple, o: tuple[float, float], s: float, color: str,
              z: float, alpha: float = 1.0, edge: str = "white", lw: float = 0.8) -> None:
    """Draw a flat-shaded box (front, top, right side) for feature maps."""
    x0, y0, z0 = p0
    x1, y1, z1 = x0 + size[0], y0 + size[1], z0 + size[2]
    faces = [([(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)], 1.0),
             ([(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)], 0.72),
             ([(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)], 0.5)]
    for corners, shade in faces:
        rgb = np.array(plt.matplotlib.colors.to_rgb(color))
        fc = 1 - (1 - rgb) * (0.35 + 0.65 * shade)
        ax.add_patch(Polygon([project(*c, s, o) for c in corners], closed=True, fc=fc, ec=edge,
                             lw=lw, alpha=alpha, zorder=z))


def arrow(ax: plt.Axes, p: tuple, q: tuple, color: str = INK2, lw: float = 1.3, z: float = 20,
          style: str = "-|>", ls: str = "-", rad: float = 0.0) -> None:
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=10, color=color, lw=lw,
                                 zorder=z, ls=ls, connectionstyle=f"arc3,rad={rad}"))


def label(ax: plt.Axes, x: float, y: float, text: str, size: float = 8.5, color: str = INK,
          weight: str = "normal", ha: str = "center", va: str = "center", z: float = 30, **kw) -> None:
    ax.text(x, y, text, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, zorder=z, **kw)


def panel_title(ax: plt.Axes, x: float, y: float, letter: str, text: str) -> None:
    ax.text(x, y, letter, fontsize=11, fontweight="bold", color=INK, va="baseline", zorder=30)
    ax.text(x + 2.6, y, text, fontsize=9.5, fontweight="semibold", color=INK, va="baseline", zorder=30)


def canvas(width: float, height: float, xlim: float, ylim: float) -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, xlim)
    ax.set_ylim(0, ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


# ---------------------------------------------------------------- data
def cubes_of_one_session() -> tuple[np.ndarray, np.ndarray, dict, np.ndarray]:
    """Return the 38^3 caches of one validation session (2 and 6 channels) and its MNI volume."""
    hippo, _, hindex = rep.cache("hippo_s32")
    multi, mmeta, mindex = rep.cache("multi3_s32")
    val = [r["idx"] for r in rep.manifest() if r["split"] == "val"]
    idx = next(i for i in val if i in hindex and i in mindex)
    return (np.asarray(hippo[hindex[idx]], dtype=np.float32),
            np.asarray(multi[mindex[idx]], dtype=np.float32), mmeta, rep.mni_volume(idx))


def central(cube: np.ndarray, size: int = 32) -> np.ndarray:
    m = (cube.shape[-1] - size) // 2
    return cube[..., m:m + size, m:m + size, m:m + size]


# ---------------------------------------------------------------- figure 1: ingestion
def fig_ingesta() -> None:
    two, six, meta, volume = cubes_of_one_session()
    starts, centers, extract, size = meta["cube_starts"], meta["centers_voxel"], meta["extract"], meta["size"]
    margin = (extract - size) // 2
    fig, ax = canvas(12.0, 9.4, 120.0, 94.0)

    # ---- A: the MNI volume, three planes through the left hippocampus, six cubes
    panel_title(ax, 1.0, 90.5, "A", "Volumen MNI completo (97 × 115 × 97 vóxeles de 2 mm)")
    s, o = 0.285, (6.0, 50.5)
    nx, ny, nz = volume.shape
    cx, cy, cz = (round(v) for v in centers["hippocampus_left"])
    d = np.array([-DX, 1.0, -DY])  # viewing direction: larger dot = farther
    pieces = []
    for xs in ((0, cx), (cx, nx)):          # axial plane z = cz
        for ys in ((0, cy), (cy, ny)):
            img = volume[xs[0]:xs[1], ys[0]:ys[1], cz].T
            mat = [[s, s * DX, o[0]], [0, s * DY, o[1] + s * cz]]
            ctr = np.array([np.mean(xs), np.mean(ys), cz])
            pieces.append((ctr @ d, img, mat, (xs[0], xs[1], ys[0], ys[1])))
    for xs in ((0, cx), (cx, nx)):          # coronal plane y = cy
        for zs in ((0, cz), (cz, nz)):
            img = volume[xs[0]:xs[1], cy, zs[0]:zs[1]].T
            mat = [[s, 0, o[0] + s * cy * DX], [0, s, o[1] + s * cy * DY]]
            ctr = np.array([np.mean(xs), cy, np.mean(zs)])
            pieces.append((ctr @ d, img, mat, (xs[0], xs[1], zs[0], zs[1])))
    for ys in ((0, cy), (cy, ny)):          # sagittal plane x = cx
        for zs in ((0, cz), (cz, nz)):
            img = volume[cx, ys[0]:ys[1], zs[0]:zs[1]].T
            mat = [[s * DX, 0, o[0] + s * cx], [s * DY, s, o[1]]]
            ctr = np.array([cx, np.mean(ys), np.mean(zs)])
            pieces.append((ctr @ d, img, mat, (ys[0], ys[1], zs[0], zs[1])))
    for k, (_, img, mat, ext) in enumerate(sorted(pieces, key=lambda p: -p[0])):
        face(ax, img, mat, ext, 2 + 0.01 * k, alpha_below=0.04)
    box_edges(ax, (0, 0, 0), (nx, ny, nz), o, s, BASELINE, 0.8, 1.5, hidden=True)
    for key, start in starts.items():
        region = key.rsplit("_", 1)[0]
        p0 = tuple(v + margin for v in start)
        box_edges(ax, p0, (size,) * 3, o, s, REGION_COLOR[region], 1.7, 10, hidden=True)
    # axis arrows
    base = project(-8, 0, 0, s, o)
    for vec, text in (((22, 0, 0), "x: izq. → der."), ((0, 34, 0), "y: post. → ant."), ((0, 0, 22), "z: inf. → sup.")):
        tip = project(-8 + vec[0], vec[1], vec[2], s, o)
        arrow(ax, base, tip, color=MUTED, lw=1.0)
        label(ax, tip[0] + 0.6, tip[1] + 0.9, text, size=7.5, color=INK2, ha="left")
    label(ax, 3.0, 47.0, "Planos axial, coronal y sagital que cruzan el centro del hipocampo izquierdo.\n"
          "Cajas: los 6 cubos de 32³ que se evalúan (posición fija = centroide medio de la región\n"
          "en las sesiones de entrenamiento). Mismas posiciones para todos los sujetos.",
          size=7.6, color=INK2, ha="left", va="top")
    lx = 52.0
    for k, region in enumerate(("hippocampus", "ventricle", "posteromedial")):
        yk = 84.0 - 3.2 * k
        ax.plot([lx, lx + 3], [yk, yk], color=REGION_COLOR[region], lw=2.2, zorder=30)
        label(ax, lx + 4, yk, f"{REGION_LABEL[region]} (izq. y der.)", size=7.8, ha="left")

    # ---- B: one stored cube 38^3 and the evaluated 32^3 inside it
    panel_title(ax, 73.0, 90.5, "B", "Un cubo guardado y el cubo evaluado")
    left38 = two[0]
    sb, ob = 0.56, (76.0, 58.5)
    textured_cube(ax, left38, ob, sb, edge=INK2, lw=1.0, z=3)
    box_edges(ax, (margin,) * 3, (size,) * 3, ob, sb, REGION_COLOR["hippocampus"], 2.0, 6, hidden=True)
    top = project(extract, extract, extract, sb, ob)
    label(ax, top[0] + 1.5, top[1] - 2.0, "38³ guardado en disco\n(76 mm, float16)", size=8, ha="left")
    label(ax, top[0] + 1.5, top[1] - 8.0, "32³ que ve la red\n(64 mm; margen de 3\nvóxeles por lado)",
          size=8, color=INK, ha="left")
    ax.plot([top[0] + 1.0, top[0] + 1.0], [top[1] - 11.0, top[1] - 5.6], color=REGION_COLOR["hippocampus"], lw=2.2)
    label(ax, 74.0, 54.5, "Hipocampo izquierdo, caras exteriores del volumen. El margen permite\n"
          "recortes y deformaciones sin salir del cubo; la evaluación usa solo el centro.",
          size=7.6, color=INK2, ha="left", va="top")

    # ---- C: mirrored right hemisphere
    panel_title(ax, 1.0, 39.0, "C", "El hemisferio derecho se espeja (volteo en x)")
    right = central(np.flip(two[1], axis=0))  # back to the original right cube
    so = 0.52
    textured_cube(ax, right, (6.0, 6.0), so, z=3)
    label(ax, 6.0 + so * 16, 3.6, "hipocampo der. tal cual", size=7.8, color=INK2)
    arrow(ax, (32.5, 17.0), (40.5, 17.0))
    label(ax, 36.5, 20.0, "x → −x", size=8, color=INK2)
    textured_cube(ax, central(two[1]), (43.0, 6.0), so, z=3)
    label(ax, 43.0 + so * 16, 3.6, "der. espejado = canal 1", size=7.8, color=INK2)
    label(ax, 1.0, 35.0, "Así el hipocampo derecho queda orientado como el izquierdo: el mismo filtro\n"
          "sirve para los dos, y voltear la cabeza (izq. ↔ der.) es solo intercambiar canales.",
          size=7.6, color=INK2, ha="left", va="top")

    # ---- D: input tensors
    panel_title(ax, 73.0, 39.0, "D", "El tensor que entra a la red, por sujeto")
    sd = 0.30
    names2 = ["hipocampo izq.", "hipocampo der. esp."]
    for c in range(2):
        oc = (75.0 + c * 15.0, 17.0)
        textured_cube(ax, central(two[c]), oc, sd, edge=REGION_COLOR["hippocampus"], lw=1.4, z=3)
        label(ax, oc[0] + sd * 16, oc[1] - 1.6, f"canal {c}\n{names2[c]}", size=6.8, color=INK2, va="top")
    label(ax, 106.0, 23.0, "VICReg y BYOL\n2 × 32 × 32 × 32", size=8.5, weight="semibold",
          color=INK, ha="left")
    regions = ["hippocampus", "hippocampus", "ventricle", "ventricle", "posteromedial", "posteromedial"]
    short = ["hipoc. izq.", "hipoc. der.", "ventr. izq.", "ventr. der.", "postm. izq.", "postm. der."]
    for c in range(6):
        oc = (75.0 + c * 7.2, 2.6)
        textured_cube(ax, central(six[c]), oc, 0.17, edge=REGION_COLOR[regions[c]], lw=1.3, z=3)
        label(ax, oc[0] + 0.17 * 16, oc[1] - 1.2, f"{c}: {short[c]}", size=6.0, color=INK2, va="top")
    label(ax, 106.0, 13.2, "DINOv2\n6 × 32 × 32 × 32", size=8.5, weight="semibold", color=INK, ha="left")
    label(ax, 73.0, 35.0, "Los cubos no se pegan uno al lado del otro: se apilan como canales\n"
          "(como R, G, B en una foto). Un lote de 16 sujetos es 16 × C × 32 × 32 × 32.",
          size=7.6, color=INK2, ha="left", va="top")
    rep.save(fig, "ingesta_3d.png")


# ---------------------------------------------------------------- figure 2: volumes in the encoders
def side(n: int) -> float:
    """Display side of a feature map with an n^3 grid (log scale: 32 -> 9.6, 1 -> 1.6)."""
    return 1.6 + 1.6 * np.log2(n)


def grid_lines(ax: plt.Axes, o: tuple[float, float], length: float, n: int, color: str,
               z: float, lw: float = 0.6) -> None:
    """Draw an n x n x n grid on the three visible faces of a cube of display side ``length``."""
    for k in range(1, n):
        t = k * length / n
        for a, b in (((t, 0, 0), (t, 0, length)), ((0, 0, t), (length, 0, t)),
                     ((t, 0, length), (t, length, length)), ((0, t, length), (length, t, length)),
                     ((length, t, 0), (length, t, length)), ((length, 0, t), (length, length, t))):
            p, q = project(*a, 1, o), project(*b, 1, o)
            ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw, zorder=z)


def feature_deck(ax: plt.Axes, o: tuple[float, float], n: int, channels: int, color: str,
                 z: float, name: str, copies: int = 3) -> tuple[float, float]:
    """Feature map C x n^3: a deck of cubes (side ~ n), the front one with its n^3 grid.

    Returns the screen point of the right edge of the front cube's centre, for arrows.
    """
    length = side(n)
    for k in reversed(range(copies)):
        ok = (o[0] + 0.55 * k, o[1] + 0.55 * k)
        solid_box(ax, (0, 0, 0), (length,) * 3, ok, 1, color, z + (copies - k) * 0.1,
                  edge="white", lw=0.7)
    if 1 < n <= 8:
        grid_lines(ax, o, length, n, "white", z + copies * 0.1 + 0.05, lw=0.55)
    bottom = o[1] - 1.3
    label(ax, o[0] + length / 2 + 0.4, bottom, f"{channels} × {n}×{n}×{n}", size=8, weight="semibold",
          va="top")
    if name:
        label(ax, o[0] + length / 2 + 0.4, bottom - 2.2, name, size=7.2, color=INK2, va="top")
    return project(length, length / 2, length / 2, 1, o)


def input_deck(ax: plt.Axes, cubes: np.ndarray, o: tuple[float, float], length: float,
               edges: list[str], step: float = 0.9, grid: int = 0) -> None:
    """Draw the input channels as a deck of textured cubes, channel 0 in front."""
    s = length / cubes.shape[-1]
    for c in reversed(range(len(cubes))):
        oc = (o[0] + step * c, o[1] + step * c)
        textured_cube(ax, cubes[c], oc, s, edge=edges[c], lw=1.2, z=3 + 0.5 * (len(cubes) - c),
                      grid=grid if c == 0 else 0)


def kernel_window(ax: plt.Axes, o: tuple[float, float], length: float, frac: float, color: str,
                  z: float) -> tuple[float, float]:
    """Highlight a cubic window of ``frac`` x the side at the front-top-left corner."""
    w = length * frac
    solid_box(ax, (0, 0, length - w), (w, w, w), o, 1, color, z, alpha=0.75, edge=color, lw=1.0)
    return project(w / 2, 0, length - w / 2, 1, o)


def cnn_row(ax: plt.Axes, y0: float, model: str, cubes: np.ndarray) -> None:
    color = COLOR[model]
    if model == "vicreg":
        title = "VICReg · ConvNeXt-T 3D · entrada 2 × 32 × 32 × 32"
        stem = ("tallo: conv. 4×4×4,\npaso 4 (parches 3D\nno solapados) + LN", 4 / 32)
        maps = [(8, 96, "etapa 1 · 3 bloques"), (4, 192, "etapa 2 · 3 bloques"),
                (2, 384, "etapa 3 · 9 bloques"), (1, 768, "etapa 4 · 3 bloques")]
        down = "reducción: LN +\nconv. 2×2×2, paso 2"
        out = 768
        block = ("Bloque ConvNeXt 3D: conv. por canal 7×7×7 → LayerNorm → Lineal C→4C → GELU → Lineal 4C→C "
                 "→ ×γ (escala por canal) → + entrada (residual).\nLas lineales actúan vóxel a vóxel sobre los C "
                 "canales. Drop path de 0 a 0.1 a lo largo de los 18 bloques.")
    else:
        title = "BYOL · ResNet-18 3D sin max-pool · entrada 2 × 32 × 32 × 32"
        stem = ("tallo: conv. 7×7×7,\npaso 2 + BN + ReLU\n(ventanas solapadas)", 7 / 32)
        maps = [(16, 64, "capa 1 · 2 bloques"), (8, 128, "capa 2 · 2 bloques"),
                (4, 256, "capa 3 · 2 bloques"), (2, 512, "capa 4 · 2 bloques")]
        down = "la 1.ª conv. de cada\ncapa 2–4 usa paso 2"
        out = 512
        block = ("Bloque básico 3D: conv. 3×3×3 → BN → ReLU → conv. 3×3×3 → BN → + atajo (conv. 1×1×1 con paso 2 "
                 "cuando cambia el tamaño) → ReLU.\nSin max-pool tras el tallo: con cubos de 32 vóxeles se perdería "
                 "demasiada resolución (la ResNet 2D original lo tiene).")
    label(ax, 1.0, y0 + 28.5, title, size=10, weight="bold", color=color, ha="left")
    # input with the stem's receptive window on channel 0
    length_in = side(32)
    o_in = (3.0, y0 + 10.0)
    input_deck(ax, cubes, o_in, length_in, [REGION_COLOR["hippocampus"]] * 2)
    win = kernel_window(ax, o_in, length_in, stem[1], color, 12)
    label(ax, o_in[0] + length_in / 2 + 0.4, o_in[1] - 1.3, "2 × 32×32×32", size=8, weight="semibold", va="top")
    label(ax, o_in[0] + length_in / 2 + 0.4, o_in[1] - 3.5, "entrada (2 canales)", size=7.2, color=INK2, va="top")
    # feature maps
    x = 29.0
    prev = project(length_in + 0.9, length_in / 2, length_in / 2, 1, o_in)
    for k, (n, c, name) in enumerate(maps):
        o = (x, y0 + 10.0 + (side(32) - side(n)) / 2)
        anchor = project(0, side(n) / 2, side(n) / 2, 1, o)
        arrow(ax, (prev[0] + 0.6, anchor[1]), (anchor[0] - 0.4, anchor[1]), color=MUTED)
        if k == 0:
            label(ax, (prev[0] + anchor[0]) / 2 + 0.6, anchor[1] - 1.2, stem[0], size=7.0, color=INK2, va="top")
            # the stem window maps to one position of the first map
            cell = side(n) / n
            target = project(cell / 2, 0, side(n) - cell / 2, 1, o)
            arrow(ax, win, target, color=color, lw=1.0, ls=(0, (2, 2)), style="-", rad=-0.25)
        elif k == 1:
            label(ax, (prev[0] + anchor[0]) / 2, anchor[1] + 1.0, down, size=7.0, color=INK2, va="bottom")
        prev = feature_deck(ax, o, n, c, color, 5, name)
        x += side(n) * (1 + DX) + 12.0
    # global average pooling and the probe vector
    gx = x + 1.0
    arrow(ax, (prev[0] + 0.6, prev[1]), (gx - 0.3, prev[1]), color=MUTED)
    label(ax, (prev[0] + gx) / 2, prev[1] + 3.2, "promedio\nglobal (GAP)", size=7.0, color=INK2)
    ax.add_patch(FancyBboxPatch((gx, prev[1] - 5.0), 1.4, 10.0, boxstyle="round,pad=0,rounding_size=0.4",
                                fc=color, ec=color, zorder=8))
    label(ax, gx + 0.7, prev[1] - 6.2, f"vector de {out}", size=8, weight="semibold", va="top")
    label(ax, gx + 0.7, prev[1] - 8.2, "lo que usa el probe", size=7.2, color=color, va="top")
    label(ax, 1.0, y0 + 1.2, block, size=7.4, color=INK2, ha="left", va="center")


def patch_lattice(ax: plt.Axes, o: tuple[float, float], cell: float, gap: float, color: str,
                  masked: set[tuple[int, int, int]], picked: tuple[int, int, int]) -> dict:
    """Draw the 4x4x4 patch grid, exploded, back to front; return screen anchors per patch."""
    d = np.array([-DX, 1.0, -DY])
    order = sorted(((i, j, k) for i in range(4) for j in range(4) for k in range(4)),
                   key=lambda p: -(np.array(p) @ d))
    anchors = {}
    for z, (i, j, k) in enumerate(order):
        p0 = (i * (cell + gap), j * (cell + gap), k * (cell + gap))
        fc = "#7b52c7" if (i, j, k) in masked else (color if (i, j, k) == picked else "#cfd8d3")
        solid_box(ax, p0, (cell,) * 3, o, 1, fc, 4 + 0.01 * z, edge="white", lw=0.5)
        anchors[(i, j, k)] = project(p0[0] + cell / 2, p0[1], p0[2] + cell / 2, 1, o)
    return anchors


def vit_rows(ax: plt.Axes, y0: float, cubes: np.ndarray) -> None:
    color = COLOR["dinov2"]
    violet = "#7b52c7"
    label(ax, 1.0, y0 + 51.0, "DINOv2 · ViT-S 3D · entrada 6 × 32 × 32 × 32 · parches de 8×8×8", size=10,
          weight="bold", color=color, ha="left")
    regions = ["hippocampus", "hippocampus", "ventricle", "ventricle", "posteromedial", "posteromedial"]
    # (1) input with its 4x4x4 patch grid
    length_in = side(32)
    o_in = (3.0, y0 + 32.0)
    input_deck(ax, cubes, o_in, length_in, [REGION_COLOR[r] for r in regions], step=0.75, grid=4)
    label(ax, o_in[0] + length_in / 2 + 2.0, o_in[1] - 1.3, "6 × 32×32×32", size=8, weight="semibold", va="top")
    label(ax, o_in[0] + length_in / 2 + 2.0, o_in[1] - 3.5, "6 canales; rejilla de\n4×4×4 = 64 parches",
          size=7.2, color=INK2, va="top")
    # (2) exploded lattice with one picked patch and a masked block
    masked = {(i, j, k) for i in (2, 3) for j in (0, 1) for k in (0, 1)}
    picked = (0, 0, 3)
    o_lat = (26.0, y0 + 31.0)
    anchors = patch_lattice(ax, o_lat, 2.4, 0.6, color, masked, picked)
    arrow(ax, project(length_in + 4.6, length_in / 2, length_in / 2, 1, o_in), (o_lat[0] - 0.6, o_lat[1] + 6.0),
          color=MUTED)
    label(ax, 33.0, o_lat[1] - 2.0, "los 64 parches 8×8×8 (16 mm de lado)\nno se solapan: cada uno es un token",
          size=7.2, color=INK2, va="top")
    # (3) one patch: 6 channels x 8^3 voxels
    m = (cubes.shape[-1] - 8) // 2
    patch = cubes[:, 0:8, 0:8, 24:32]  # the picked corner patch (x 0-7, y 0-7, z 24-31)
    o_p = (54.0, y0 + 35.0)
    for c in reversed(range(6)):
        oc = (o_p[0] + 0.55 * c, o_p[1] + 0.55 * c)
        textured_cube(ax, patch[c], oc, 0.62, edge=REGION_COLOR[regions[c]], lw=1.1, z=6 + 0.3 * (6 - c))
    arrow(ax, anchors[picked], (o_p[0] - 0.5, o_p[1] + 4.5), color=color, lw=1.1, rad=-0.2)
    label(ax, o_p[0] + 4.6, o_p[1] - 1.3, "un parche: 6 × 8×8×8\n= 3072 valores", size=8, weight="semibold",
          va="top")
    del m
    # (4) linear projection to a token
    tx = 76.0
    arrow(ax, (o_p[0] + 10.6, o_p[1] + 4.0), (tx - 0.5, o_p[1] + 4.0), color=MUTED)
    label(ax, (o_p[0] + 10.6 + tx) / 2, o_p[1] + 8.4, "proyección lineal\n(conv. 8×8×8, paso 8):\n3072 → 384",
          size=7.0, color=INK2)
    ax.add_patch(FancyBboxPatch((tx, o_p[1] - 1.0), 1.2, 10.0, boxstyle="round,pad=0,rounding_size=0.3",
                                fc=color, ec=color, zorder=8))
    label(ax, tx + 0.6, o_p[1] - 2.2, "token\n(384)", size=7.6, weight="semibold", va="top")
    label(ax, 82.0, y0 + 44.0,
          "Los 6 canales de un parche entran juntos a la\nmisma proyección: el token de la posición (i, j, k)\n"
          "mezcla los 6 cubos en esa misma posición relativa.\n"
          "Al inicio (pesos DeiT inflados) los 6 canales\ncomparten el mismo filtro: el token ve su promedio.",
          size=7.2, color=INK2, ha="left", va="center")
    # (5) the token sequence
    sy = y0 + 10.8
    label(ax, 1.0, sy + 12.3, "Secuencia que procesa el Transformer: 1 CLS + 64 tokens de parche, cada uno de 384 números",
          size=8.5, weight="semibold", ha="left")
    bw, gap = 0.62, 0.24
    sx = 3.0
    order = [(i, j, k) for i in range(4) for j in range(4) for k in range(4)]
    for t in range(65):
        x = sx + t * (bw + gap)
        if t == 0:
            fc, ec = INK, INK
        else:
            fc = violet if order[t - 1] in masked else (color if order[t - 1] == picked else "#cfd8d3")
            ec = fc
        ax.add_patch(FancyBboxPatch((x, sy), bw, 8.0, boxstyle="round,pad=0,rounding_size=0.15",
                                    fc=fc, ec=ec, lw=0.4, zorder=6))
    end = sx + 65 * (bw + gap)
    label(ax, sx + bw / 2, sy - 1.0, "CLS", size=7.2, weight="semibold", va="top")
    label(ax, sx + 33 * (bw + gap), sy - 1.0, "64 tokens de parche, en el orden de la rejilla (i, j, k)",
          size=7.2, color=INK2, va="top")
    label(ax, end + 0.5, sy + 9.6, "+ posición 3D aprendida\n(una por celda de la rejilla 4×4×4)", size=7.0,
          color=INK2, ha="left", va="bottom")
    # (6) transformer blocks and outputs
    bx = end + 3.0
    arrow(ax, (end + 0.3, sy + 4.0), (bx - 0.3, sy + 4.0), color=MUTED)
    for k in range(12):
        ax.add_patch(FancyBboxPatch((bx + k * 0.95, sy - 0.5), 0.7, 9.0,
                                    boxstyle="round,pad=0,rounding_size=0.2", fc=color, ec=color,
                                    alpha=0.3 + 0.05 * k, zorder=6))
    label(ax, bx + 5.6, sy - 1.6, "12 bloques Transformer\n(cada token atiende a los 65)", size=7.2, color=INK2,
          va="top")
    ox = bx + 12 * 0.95 + 2.6
    arrow(ax, (bx + 12 * 0.95, sy + 4.0), (ox - 0.3, sy + 4.0), color=MUTED)
    ax.add_patch(FancyBboxPatch((ox, sy - 0.5), 1.3, 9.0, boxstyle="round,pad=0,rounding_size=0.3",
                                fc=INK, ec=INK, zorder=8))
    label(ax, ox + 2.0, sy + 6.0, "CLS (384)\n→ probe y cabeza DINO", size=7.4, weight="semibold", ha="left")
    label(ax, ox + 2.0, sy + 1.4, "64 tokens (384)\n→ cabeza iBOT", size=7.4, ha="left", color=INK2)
    # legend of colours
    ly = y0 + 1.6
    for k, (fc, text) in enumerate(((violet, "parche enmascarado (solo en la vista del alumno): su token se reemplaza "
                                              "por el «mask token» aprendido, antes de sumar la posición"),
                                    (color, "el parche elegido como ejemplo"),
                                    ("#cfd8d3", "resto de los parches"))):
        ax.add_patch(FancyBboxPatch((3.0, ly + 4.4 - 2.0 * k - 0.6), 1.2, 1.2, boxstyle="round,pad=0,rounding_size=0.2",
                                    fc=fc, ec=fc, zorder=8))
        label(ax, 5.0, ly + 4.4 - 2.0 * k, text, size=7.2, color=INK2, ha="left")
    label(ax, 3.0, ly - 1.8, "Máscara de bloque iBOT: en cada lote, 8 de las 16 imágenes de cada vista del alumno llevan "
          "máscara; cubre entre 10 y 50 % de los 64 parches (6 a 31), en bloques 3D contiguos de ≥ 4 parches.",
          size=7.2, color=INK2, ha="left")


def fig_volumenes() -> None:
    two, six, _, _ = cubes_of_one_session()
    two, six = central(two), central(six)
    fig, ax = canvas(12.0, 12.2, 120.0, 122.0)
    cnn_row(ax, 89.5, "vicreg", two)
    ax.plot([1, 119], [88.5, 88.5], color=BASELINE, lw=0.8)
    cnn_row(ax, 55.5, "byol", two)
    ax.plot([1, 119], [54.5, 54.5], color=BASELINE, lw=0.8)
    vit_rows(ax, 0.0, six)
    rep.save(fig, "volumenes_3d.png")


# ---------------------------------------------------------------- results of the 100-epoch run
SSL_RUN = {"vicreg": "vicreg_campaign_augval__geo__roi32", "byol": "byol_gn_dp0.3_dq0_wd0.01__noise__roi32",
           "dinov2": "dinov2_campaign__elastic__roimulti3"}
SUP_RUN = {"vicreg": "vicreg_supervised__geo__roi32", "byol": "byol_supervised__noise__roi32",
           "dinov2": "dinov2_supervised__elastic_gpu__roimulti3"}
INIT_RUN = {"vicreg": "vicreg_init__roi32", "byol": "byol_init__roi32", "dinov2": "dinov2_init__roimulti3"}
# (condition, runs directory, run names): "init" in split361_30 = pretrained start without SSL,
# "init" in split361_100 = random weights (extra reference, tables only)
CONDITIONS = (("supervisado", "split361_100", SUP_RUN), ("ssl", "split361_100", SSL_RUN),
              ("sin_ssl", "split361_30", INIT_RUN), ("aleatorio", "split361_100", INIT_RUN))
METRICS = (("auc", "test_auc", "AUROC"), ("f1", "test_f1_macro", "F1-macro"),
           ("aupr", "test_average_precision", "AUPR"))
SUP_COLOR = "#52514e"


def run_docs(base: str, name: str) -> list[dict]:
    docs = []
    for seed in (1, 2, 3):
        path = RUNS / base / f"{name}_s{seed}" / "result.json"
        if path.is_file():
            doc = json.loads(path.read_text(encoding="utf-8"))
            docs.append(doc if "cv" in doc else doc["main"])
    return docs


def metric_values(docs: list[dict], split: str, key: tuple[str, str, str]) -> np.ndarray:
    if split == "cv":
        return np.array([d["cv"]["clean"]["metrics"][key[0]] for d in docs])
    return np.array([d["test"][key[1]] for d in docs])


def results_100() -> dict:
    """{model: {condition: [result docs]}} for the four conditions of the 100-epoch report."""
    return {m: {cond: run_docs(base, names[m]) for cond, base, names in CONDITIONS} for m in MODELS}


def fig_metricas() -> None:
    """Dot plot: every metric of supervised, SSL and without-SSL, the value written beside it."""
    data = results_100()
    rows = [("supervisado", "Supervisado completo"), ("ssl", "SSL 100 ép. + probe"),
            ("sin_ssl", "Sin SSL + probe")]
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.8), sharey=True)
    lo, hi = 0.60, 0.95
    for r, (split, split_name) in enumerate((("cv", "CV 5×5 · 329 sujetos de train+val"),
                                             ("test", "Test · 84 sujetos nunca vistos"))):
        for c, key in enumerate(METRICS):
            ax = axes[r, c]
            ticks, names = [], []
            for m, model in enumerate(MODELS):
                for k, (cond, cond_name) in enumerate(rows):
                    y = -(m * 4.2 + k)
                    values = metric_values(data[model][cond], split, key)
                    mean = float(values.mean())
                    sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
                    if cond == "supervisado":
                        style = {"marker": "s", "color": SUP_COLOR, "mfc": SUP_COLOR}
                    elif cond == "ssl":
                        style = {"marker": "o", "color": COLOR[model], "mfc": COLOR[model]}
                    else:
                        style = {"marker": "D", "color": MUTED, "mfc": "white"}
                    if len(values) > 1:
                        ax.plot([mean - sd, mean + sd], [y, y], color=style["color"], lw=1.6, alpha=0.55,
                                solid_capstyle="butt", zorder=2)
                        ax.scatter(values, [y] * len(values), s=10, color=style["color"], alpha=0.45,
                                   linewidths=0, zorder=3)
                    ax.plot([mean], [y], ls="none", ms=7.5, mew=1.4, mec=style["color"], zorder=4,
                            marker=style["marker"], mfc=style["mfc"])
                    right = max(mean + sd, float(values.max()))
                    ax.text(right + 0.011, y, f"{mean:.3f}", va="center", ha="left", fontsize=8.8,
                            color=INK, fontweight="semibold" if cond == "ssl" else "normal")
                    ticks.append(y)
                    names.append(cond_name)
            ax.set_xlim(lo, hi)
            ax.set_ylim(-(2 * 4.2 + 2) - 0.7, 0.8)
            ax.set_yticks(ticks, names, fontsize=8)
            ax.grid(axis="y", visible=False)
            ax.grid(axis="x", color=rep.GRID, lw=0.8)
            ax.set_title(f"{key[2]} — {split_name}", fontsize=9.5)
            ax.tick_params(axis="y", length=0)
            for m, model in enumerate(MODELS):
                if c == 0:
                    ax.text(-0.42, -(m * 4.2 + 1), LABEL[model], transform=ax.get_yaxis_transform(),
                            rotation=90, ha="center", va="center", fontsize=10, fontweight="bold",
                            color=COLOR[model])
                if m:
                    ax.axhline(-(m * 4.2) + 1.6, color=BASELINE, lw=0.8)
    handles = [plt.Line2D([], [], ls="none", marker="s", ms=7, color=SUP_COLOR, label="Supervisado completo"),
               plt.Line2D([], [], ls="none", marker="o", ms=7, color=INK2,
                          label="SSL 100 épocas, encoder congelado + probe lineal (color del modelo)"),
               plt.Line2D([], [], ls="none", marker="D", ms=7, mfc="white", mec=MUTED, mew=1.4,
                          label="Sin SSL: mismo encoder antes del SSL, congelado + probe")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, -0.005))
    fig.text(0.5, -0.04, "Marcador grande y número = media de 3 semillas; barra = ± 1 desviación estándar entre "
             "semillas; puntos pequeños = cada semilla. «Sin SSL» es una sola corrida: sus pesos no dependen de la "
             "semilla.", ha="center", fontsize=8, color=INK2)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.95, bottom=0.1, wspace=0.08, hspace=0.28)
    rep.save(fig, "metricas_100_puntos.png")


def fig_supervisado() -> None:
    """Train vs validation BCE of the end-to-end supervised baseline, one line per seed."""
    data = results_100()
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.4), sharey=True)
    for ax, model in zip(axes, MODELS, strict=True):
        for doc in data[model]["supervisado"]:
            hist = doc["history"]
            ep = [h["epoch"] for h in hist]
            ax.plot(ep, [h["train_loss"] for h in hist], color=COLOR[model], lw=1.3, alpha=0.9)
            ax.plot(ep, [h["val_loss"] for h in hist], color=COLOR[model], lw=1.3, ls=(0, (4, 2)), alpha=0.9)
            best = doc["best_epoch"]
            ax.plot([best], [hist[best - 1]["val_loss"]], "o", ms=6, color=COLOR[model], mec="white", mew=1.2,
                    zorder=5)
        ax.set_title(f"{LABEL[model]} · {('ConvNeXt-T', 'ResNet-18', 'ViT-S')[MODELS.index(model)]} 3D "
                     "supervisado", fontsize=9.5)
        ax.set_xlabel("época", fontsize=8.5)
    axes[0].set_ylabel("BCE", fontsize=8.5)
    handles = [plt.Line2D([], [], color=INK2, lw=1.4, label="entrenamiento (BCE balanceada, vistas aumentadas)"),
               plt.Line2D([], [], color=INK2, lw=1.4, ls=(0, (4, 2)), label="validación (63 sujetos, cubo limpio)"),
               plt.Line2D([], [], ls="none", marker="o", color=INK2, label="época elegida (mejor AUROC de val.)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.3, bbox_to_anchor=(0.5, -0.06))
    fig.subplots_adjust(bottom=0.2, wspace=0.06)
    rep.save(fig, "supervisado_curvas_100.png")


def fig_metricas_barras() -> None:
    """Draw the 100-epoch metrics as bars with every value written above its bar, for slides."""
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.patches import Patch
    data = results_100()
    tick = {"vicreg": "VICReg\ngeo · 32³", "byol": "BYOL\nnoise · 32³ · GN",
            "dinov2": "DINOv2\nelastic · 3 regiones"}
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 9.6), sharey=True)
    width, step = 0.28, 0.30
    for r, (split, split_name) in enumerate((("cv", "CV (train+val, 329 sujetos)"),
                                             ("test", "Test (84 sujetos)"))):
        for c, key in enumerate(METRICS):
            ax = axes[r, c]
            for i, model in enumerate(MODELS):
                for k, cond in enumerate(("supervisado", "ssl", "sin_ssl")):
                    x = i + (k - 1) * step
                    values = metric_values(data[model][cond], split, key)
                    mean = float(values.mean())
                    if cond == "supervisado":
                        ax.bar(x, mean, width=width, color="white", edgecolor=COLOR[model], hatch="///",
                               linewidth=1.6, zorder=2)
                    elif cond == "ssl":
                        ax.bar(x, mean, width=width, color=COLOR[model], edgecolor="white", linewidth=1.0,
                               zorder=2)
                    else:
                        ax.bar(x, mean, width=width, color=rep.BASELINE, edgecolor="white", linewidth=1.0,
                               zorder=2)
                    if len(values) > 1:
                        ax.scatter([x] * len(values), values, s=16, color=INK, zorder=4, linewidths=0)
                    top = max(mean, float(values.max()))
                    ax.text(x, top + 0.006, f"{mean:.3f}", ha="center", va="bottom", fontsize=10.5,
                            color=INK, fontweight="bold" if cond == "ssl" else "semibold", zorder=5,
                            bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.85})
            ax.set_xticks(range(3), [tick[m] for m in MODELS], fontsize=10.5)
            ax.set_ylim(0.55, 0.97)
            ax.tick_params(axis="y", labelsize=10.5)
            ax.set_title(f"{key[2]} — {split_name}", fontsize=12.5)
            ax.grid(axis="x", visible=False)
    sup = tuple(Patch(fc="white", ec=COLOR[m], hatch="///", lw=1.4) for m in MODELS)
    ssl = tuple(Patch(fc=COLOR[m], ec="white") for m in MODELS)
    handles = [sup, ssl, Patch(fc=rep.BASELINE, ec="white")]
    fig.legend(handles, ["Supervisado completo", "SSL (100 épocas) + probe lineal", "Sin SSL + probe lineal"],
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)}, loc="lower center", ncol=3,
               fontsize=12.5, handlelength=4.2, handleheight=1.4, columnspacing=3.0,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Modelos campeones, split 361/63/84, 100 épocas: supervisado vs SSL vs sin SSL "
                 "(3 semillas; puntos = semillas)", fontsize=14, color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97), h_pad=2.2)
    rep.save(fig, "split361_100_metricas_claras.png")


# ---------------------------------------------------------------- figure: the cubes in 3D
def view_rotation(azim: float, elev: float) -> np.ndarray:
    """Rotation to the screen frame (u right, w into the page, v up) for a view angle in degrees."""
    a, e = np.radians(azim), np.radians(elev)
    rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    return rx @ rz


def draw_box3d(ax: plt.Axes, p0: np.ndarray, size: float, rot: np.ndarray, center: np.ndarray,
               color: str, fill_alpha: float, lw: float, dashed: bool = False, z: float = 5) -> None:
    """Draw a cube in the rotated orthographic view: translucent faces, hidden edges dashed."""
    corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], float) * size + p0
    q = (corners - center) @ rot.T  # columns: u, w (depth), v
    faces = {(0, -1): [0, 1, 3, 2], (0, 1): [4, 5, 7, 6], (1, -1): [0, 1, 5, 4],
             (1, 1): [2, 3, 7, 6], (2, -1): [0, 2, 6, 4], (2, 1): [1, 3, 7, 5]}
    front = {}
    for (axis, sign) in faces:
        normal = np.zeros(3)
        normal[axis] = sign
        front[(axis, sign)] = (rot @ normal)[1] < 0  # faces the viewer
    for key, idx in sorted(faces.items(), key=lambda kv: front[kv[0]]):
        if not dashed:
            ax.add_patch(Polygon(q[idx][:, [0, 2]], closed=True, fc=color, ec="none",
                                 alpha=fill_alpha * (1.0 if front[key] else 0.5), zorder=z))
    edges: dict[tuple[int, int], list[bool]] = {}
    for key, idx in faces.items():
        for a, b in zip(idx, idx[1:] + idx[:1], strict=True):
            edges.setdefault(tuple(sorted((a, b))), []).append(front[key])
    for (a, b), flags in edges.items():
        hidden = not any(flags)
        style = (0, (2.5, 2)) if (dashed or hidden) else "-"
        ax.plot(q[[a, b], 0], q[[a, b], 2], color=color, lw=lw * (0.75 if hidden else 1.0), ls=style,
                alpha=0.8 if hidden else 1.0, zorder=z + (0 if hidden else 1), solid_capstyle="round")


def ortho_planes(ax: plt.Axes, volume: np.ndarray, rot: np.ndarray, planes: list[tuple[int, int]]) -> None:
    """Draw orthogonal slices of the volume in 3D, split where they cross and painted back to front."""
    center = (np.array(volume.shape) - 1) / 2
    m = rot[[0, 2], :]  # screen (u, v) from a 3D point
    d = rot[1, :]       # depth (w) from a 3D point
    pieces = []
    for a, t in planes:
        b, c = (axis for axis in range(3) if axis != a)
        cuts = {b: [0, volume.shape[b]], c: [0, volume.shape[c]]}
        for other, s in planes:
            if other in cuts:
                cuts[other] = sorted({*cuts[other], s})
        for b0, b1 in zip(cuts[b], cuts[b][1:], strict=False):
            for c0, c1 in zip(cuts[c], cuts[c][1:], strict=False):
                index = [slice(None)] * 3
                index[a], index[b], index[c] = t, slice(b0, b1), slice(c0, c1)
                image = volume[tuple(index)].T  # rows = c, columns = b
                mid = np.zeros(3)
                mid[a], mid[b], mid[c] = t, (b0 + b1) / 2, (c0 + c1) / 2
                off = m[:, a] * t - m @ center
                matrix = [[m[0, b], m[0, c], off[0]], [m[1, b], m[1, c], off[1]]]
                pieces.append(((mid - center) @ d, image, matrix, (b0 - 0.5, b1 - 0.5, c0 - 0.5, c1 - 0.5)))
    for k, (_, image, matrix, extent) in enumerate(sorted(pieces, key=lambda p: -p[0])):
        face(ax, image, matrix, extent, 2 + 0.01 * k, alpha_below=0.03)


def ortho_panel(ax: plt.Axes, volume: np.ndarray, meta: dict, regions: list[str], rot: np.ndarray,
                planes: list[tuple[int, int]], outer: bool, title: str, fill: float = 0.12) -> None:
    """Brain planes in 3D with the evaluated cubes (and the stored ones if ``outer``) of ``regions``."""
    ax.set_facecolor("#0b0b0b")
    center = (np.array(volume.shape) - 1) / 2
    ortho_planes(ax, volume, rot, planes)
    extract, size = meta["extract"], meta["size"]
    margin = (extract - size) // 2
    boxes = []
    for region in regions:
        for side in ("left", "right"):
            start = np.array(meta["cube_starts"][f"{region}_{side}"], float)
            mid = start + margin + size / 2
            boxes.append(((mid - center) @ rot[1, :], region, side, start))
    for _, region, side, start in sorted(boxes, key=lambda b: -b[0]):  # far boxes first
        color = REGION_COLOR[region]
        if outer:
            draw_box3d(ax, start - 0.5, extract, rot, center, color, 0.0, 1.3, dashed=True, z=8)
        draw_box3d(ax, start + margin - 0.5, size, rot, center, color, fill, 2.2, z=9)
        # label beside the cube's outer edge: hippocampus low, ventricle mid-height, posteromedial high
        x = start[0] + margin + (0 if side == "left" else size)
        height = {"hippocampus": 0.15, "ventricle": 0.5, "posteromedial": 0.85}[region]
        anchor = np.array([x, start[1] + margin + size / 2, start[2] + margin + height * size])
        tip = (anchor - center) @ rot.T
        ax.text(tip[0] + (-2.0 if side == "left" else 2.0), tip[2], "izq." if side == "left" else "der.",
                color="#0b0b0b", fontsize=9, ha="right" if side == "left" else "left", va="center", zorder=14,
                fontweight="bold", bbox={"boxstyle": "round,pad=0.18", "fc": color, "ec": "none", "alpha": 0.95})
    # axes cue in the top-left corner
    base = np.array([-49.0, 44.0])
    for vec, name in (((1, 0, 0), "x: izq. → der."), ((0, 1, 0), "y: post. → ant."), ((0, 0, 1), "z: inf. → sup.")):
        tip = base + (rot @ np.array(vec, float) * 10)[[0, 2]]
        ax.annotate("", xy=tip, xytext=base, arrowprops={"arrowstyle": "-|>", "color": "#bbbbbb", "lw": 1.1},
                    zorder=20)
        ax.text(tip[0] + 1.0, tip[1], name, color="#dddddd", fontsize=7.5, zorder=20, ha="left", va="center")
    ax.set_xlim(-54, 54)
    ax.set_ylim(-57, 54)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=11.5, fontweight="semibold", color=INK, pad=8)


def fig_roi_3d() -> None:
    """Draw the cubes of each model inside the brain (3D planes) and taken apart as channels."""
    _, six, meta, volume = cubes_of_one_session()
    centers = meta["centers_voxel"]
    z_hippo = round((centers["hippocampus_left"][2] + centers["hippocampus_right"][2]) / 2)
    y_hippo = round((centers["hippocampus_left"][1] + centers["hippocampus_right"][1]) / 2)
    x_mid = round(centers["posteromedial_left"][0])
    rot = view_rotation(azim=-38, elev=33)
    fig = plt.figure(figsize=(16.0, 6.6))
    ax1 = fig.add_axes((0.0, 0.15, 0.31, 0.78))
    ax2 = fig.add_axes((0.315, 0.15, 0.31, 0.78))
    ax3 = fig.add_axes((0.635, 0.03, 0.365, 0.93))
    ortho_panel(ax1, volume, meta, ["hippocampus"], rot, [(2, z_hippo), (1, y_hippo)], True, fill=0.0,
                title=f"VICReg y BYOL · 2 cubos (axial z = {z_hippo}, coronal y = {y_hippo})")
    ortho_panel(ax2, volume, meta, ["hippocampus", "ventricle", "posteromedial"], rot,
                [(0, x_mid), (2, z_hippo)], False, fill=0.07,
                title=f"DINOv2 · 6 cubos (sagital x = {x_mid}, axial z = {z_hippo})")
    for ax, note in ((ax1, "Sólido: cubo evaluado de 32³ (64 mm). Punteado: cubo guardado de 38³ (76 mm)."),
                     (ax2, "Cubos de 32³. Los de la línea media (ventrículo, posteromedial) se solapan\n"
                           "entre izquierda y derecha.")):
        ax.text(0.5, -0.02, note, transform=ax.transAxes, color=INK2, fontsize=9.5, ha="center", va="top")
    # exploded view: rows = regions, columns = hemispheres, textured cubes of the evaluated 32^3
    ax3.set_xlim(0, 86)
    ax3.set_ylim(-4, 70)
    ax3.set_aspect("equal")
    ax3.axis("off")
    ax3.text(1, 68.5, "Los mismos cubos, separados: cada uno es un canal", fontsize=11.5, fontweight="semibold",
             color=INK, va="top")
    cubes = central(six)
    names = {"hippocampus": "hipocampo", "ventricle": "ventrículo\nlateral", "posteromedial": "corteza\nposteromedial"}
    s = 0.36
    for r, region in enumerate(("hippocampus", "ventricle", "posteromedial")):
        y0 = 45.0 - 21.0 * r
        ax3.text(1, y0 + 7.0, names[region], fontsize=10.5, fontweight="semibold", color=INK, va="center")
        for k, side in enumerate(("izq.", "der. espejado")):
            c = 2 * r + k
            o = (16.0 + 20.0 * k, y0)
            textured_cube(ax3, cubes[c], o, s, edge=REGION_COLOR[region], lw=2.4, z=3)
            ax3.text(o[0] + s * 16, y0 - 1.0, f"canal {c} · {side}", fontsize=9, color=INK2, ha="center",
                     va="top")
    for x, (y_lo, y_hi), color, text in ((56.5, (44.0, 61.5), COLOR["vicreg"], "VICReg y BYOL:\nsolo esta fila\n(2 canales)"),
                                         (72.0, (2.0, 61.5), COLOR["dinov2"], "DINOv2:\nlas 3 filas\n(6 canales)")):
        ax3.plot([x, x + 1.0, x + 1.0, x], [y_lo, y_lo, y_hi, y_hi], color=color, lw=2.4, solid_capstyle="round")
        ax3.text(x + 1.8, (y_lo + y_hi) / 2, text, fontsize=9.5, color=INK, va="center", ha="left")
    handles = [Polygon([[0, 0]], closed=True, fc=REGION_COLOR[r], ec=REGION_COLOR[r], alpha=0.8)
               for r in ("hippocampus", "ventricle", "posteromedial")]
    fig.legend(handles, ["hipocampo", "ventrículo lateral", "corteza posteromedial"], loc="lower left",
               bbox_to_anchor=(0.16, -0.01), ncol=3, fontsize=10.5, frameon=False)
    rep.save(fig, "roi_cubes_3d.png")


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("barras", "all"):
        fig_metricas_barras()
    if which in ("cubos3d", "all"):
        fig_roi_3d()
    if which in ("ingesta", "all"):
        fig_ingesta()
    if which in ("volumenes", "all"):
        fig_volumenes()
    if which in ("arquitecturas", "all"):
        rep.fig_architectures(rep.ARCH_100, "architectures_100.png")
    if which in ("metricas", "all"):
        fig_metricas()
    if which in ("supervisado", "all"):
        fig_supervisado()


if __name__ == "__main__":
    main()
