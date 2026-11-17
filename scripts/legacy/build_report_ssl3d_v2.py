"""Figures and table snippets for ``docs/informe_ssl3d_v2.tex``.

Reads the runs of ``campaign_ssl3d`` / ``campaign_ssl3d_v2`` (JSON + each run's
``metrics.csv``), the ROI cube cache, one MNI volume from the LMDB and, if
present, ``campaign_ssl3d_v2/encoder_analysis.json``; writes PNG figures to
``docs/figures_ssl3d_v2/`` and LaTeX snippets to ``docs/informe_ssl3d_v2_tablas.tex``.

Usage: ``python scripts/legacy/build_report_ssl3d_v2.py`` then compile the .tex
(``pdflatex informe_ssl3d_v2.tex`` twice, from ``docs/``).
"""

from __future__ import annotations

import csv
import json
import math
import pickle
import random
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.ticker
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "docs" / "figures_ssl3d_v2"
TABLES = ROOT / "docs" / "informe_ssl3d_v2_tablas.tex"
V1, V2 = ROOT / "campaign_ssl3d", ROOT / "campaign_ssl3d_v2"

# Reference palette of the dataviz method (validated all-pairs for these 3 slots).
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, OTHER = "#e1e0d9", "#c3c2b7", "#c3c2b7"
COLOR = {"vicreg": "#2a78d6", "byol": "#eb6834", "dinov2": "#1baf7a"}
LABEL = {"vicreg": "VICReg", "byol": "BYOL", "dinov2": "DINOv2"}
MINUS = "−"

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


# ---------------------------------------------------------------- data access
def runs(campaign: Path, name: str) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((campaign / "runs").glob(f"{name}_s*.json"))]


def epoch_series(exp_dir: str | Path) -> dict[str, np.ndarray]:
    series: dict[str, dict[int, float]] = defaultdict(dict)
    with (ROOT / exp_dir / "metrics.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            for key, value in row.items():
                if value not in ("", None) and key not in ("epoch", "step", "train/loss_step"):
                    series[key][int(row["epoch"])] = float(value)
    return {k: np.array([v[e] for e in sorted(v)]) for k, v in series.items()}


def curves(campaign: Path, name: str) -> list[dict[str, np.ndarray]]:
    out = []
    for doc in runs(campaign, name):
        exp = doc.get("experiment_dir")
        if exp and (ROOT / exp / "metrics.csv").is_file():
            out.append(epoch_series(exp))
    return out


def mean3_cv(doc: dict) -> float:
    m = doc["cv"]["clean"]["metrics"]
    return (m["auc"] + m["f1"] + m["aupr"]) / 3


def mean3_test(doc: dict) -> float:
    t = doc["test"]
    return (t["test_auc"] + t["test_f1_macro"] + t["test_average_precision"]) / 3


def cubes_and_records() -> tuple[np.ndarray, dict[str, int], list[dict]]:
    cache = np.load(ROOT / "data/roi3d/hippo_s32.npy", mmap_mode="r")
    meta = json.loads((ROOT / "data/roi3d/hippo_s32.json").read_text(encoding="utf-8"))
    index = {idx: i for i, idx in enumerate(meta["idx"])}
    doc = runs(V2, "dinov2_3d")[0]
    with (ROOT / doc["experiment_dir"] / "data_records.csv").open(encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    return cache, index, records


# ---------------------------------------------------------------- figures
def fig_brain_and_cubes() -> None:
    """Where the two hippocampal cubes sit in the MNI volume."""
    import lmdb

    from t5_ssl.data.oasis3_datamodule import normalize_volume
    from t5_ssl.data.paths import DEFAULT_LMDB
    meta = json.loads((ROOT / "data/roi3d/hippo_s32.json").read_text(encoding="utf-8"))
    env = lmdb.open(str(DEFAULT_LMDB), readonly=True, lock=False, readahead=False)
    with env.begin() as txn:
        payload = pickle.loads(txn.get(meta["idx"][0].encode("ascii")))
    volume = normalize_volume(np.asarray(payload["image"]))
    extract, size = meta["extract"], meta["size"]
    starts = meta["cube_starts"]
    centers = meta["hippocampus_centers_voxel"]
    z = round((centers["left"][2] + centers["right"][2]) / 2)
    y = round((centers["left"][1] + centers["right"][1]) / 2)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6))
    panels = [(volume[:, :, z].T, (0, 1), f"Axial (z = {z})", "izquierda → derecha", "posterior → anterior"),
              (volume[:, y, :].T, (0, 2), f"Coronal (y = {y})", "izquierda → derecha", "inferior → superior")]
    for ax, (image, (ax_h, ax_v), title, xlabel, ylabel) in zip(axes, panels, strict=True):
        ax.imshow(image, cmap="gray", origin="lower", vmin=0, vmax=1)
        for side, text in (("left", "izq."), ("right", "der.")):
            s = starts[side]
            ax.add_patch(Rectangle((s[ax_h] - 0.5, s[ax_v] - 0.5), extract, extract, fill=False,
                                   edgecolor="#ffd166", linewidth=1.2, linestyle=(0, (3, 2))))
            m = (extract - size) // 2
            ax.add_patch(Rectangle((s[ax_h] + m - 0.5, s[ax_v] + m - 0.5), size, size, fill=False,
                                   edgecolor="#ffd166", linewidth=1.8))
            ax.text(s[ax_h] + extract / 2, s[ax_v] + extract + 2, text, color="white",
                    ha="center", va="bottom", fontsize=8, fontweight="semibold")
        ax.set_title(title)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.suptitle("Volumen MNI de 2 mm (97×115×97): cubo guardado de 38 vóxeles (punteado) y "
                 "cubo evaluado de 32 (sólido)", fontsize=9, color=INK2)
    save(fig, "brain_cubes.png")


def fig_hc_vs_ad(cache: np.ndarray, index: dict[str, int], records: list[dict]) -> None:
    """Central slices of the two-channel cube for an HC and an AD test subject."""
    margin = (cache.shape[-1] - 32) // 2
    core = slice(margin, margin + 32)
    test = [r for r in records if r["split"] == "test" and r["group"] in ("AD", "HC")]
    brightness = {r["idx"]: float(cache[index[r["idx"]]][:, core, core, core].astype(np.float32).mean())
                  for r in test}
    hc = max((r for r in test if r["group"] == "HC"), key=lambda r: brightness[r["idx"]])
    ad = min((r for r in test if r["group"] == "AD"), key=lambda r: brightness[r["idx"]])
    fig, axes = plt.subplots(2, 6, figsize=(9.2, 3.5))
    for row, (record, label) in enumerate(((hc, "HC (control sano)"), (ad, "AD (Alzheimer)"))):
        cube = cache[index[record["idx"]]][:, core, core, core].astype(np.float32)
        for c in range(2):
            vol = cube[c]
            for k, (name, image) in enumerate((("axial", vol[:, :, 16].T), ("coronal", vol[:, 16, :].T),
                                               ("sagital", vol[16, :, :].T))):
                ax = axes[row, 3 * c + k]
                ax.imshow(image, cmap="gray", origin="lower", vmin=0, vmax=1)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)
                if row == 0:
                    ax.set_title(name, fontsize=8)
        axes[row, 0].set_ylabel(label, fontsize=8.5, color=INK)
    fig.text(0.29, 1.0, "Canal 1 · hipocampo izquierdo", ha="center", fontsize=9, color=INK,
             fontweight="semibold")
    fig.text(0.72, 1.0, "Canal 2 · hipocampo derecho (espejado)", ha="center", fontsize=9,
             color=INK, fontweight="semibold")
    save(fig, "hc_vs_ad.png")


def fig_views(cache: np.ndarray, index: dict[str, int], records: list[dict]) -> None:
    """Augmented views: campaign geometry (VICReg/BYOL) vs DINOv2's views and masks."""
    import torch

    from t5_ssl.data.roi3d import ROIEval3D, ROIView3DTransform
    from t5_ssl.models.masking import sample_masks
    calib = json.loads((ROOT / "campaign_roi3d/calibration_s32.json").read_text(encoding="utf-8"))
    bias = [{**calib["artifacts"]["bias"]["spec"], "p": 1.0}]
    record = next(r for r in records if r["split"] == "val")
    cube = torch.from_numpy(cache[index[record["idx"]]].astype(np.float32))
    campaign = ROIView3DTransform(size=32, side_range=(0.84, 1.0), degrees=10.0, lr_swap_p=0.5,
                                  torchio_transforms=bias)
    still = ROIView3DTransform(size=32, side_range=(1.0, 1.0), degrees=0.0, lr_swap_p=0.5,
                               torchio_transforms=bias)
    random.seed(3), np.random.seed(3), torch.manual_seed(3)
    original = ROIEval3D(32)(cube)[0]
    campaign_views = [campaign(cube)[i][0] for i in range(2) for _ in range(2)][:4]
    still_views = [still(cube)[0][0], still(cube)[1][0]]
    mask = sample_masks(2, (4, 4, 4), 1.0, (0.3, 0.3), random.Random(5), min_block=4)[0]
    mask = mask.reshape(4, 4, 4)
    fig, axes = plt.subplots(2, 4, figsize=(8.6, 4.6))
    tiles = [(original, "Cubo limpio (evaluación)"),
             *((v, f"VICReg/BYOL · vista {i + 1}") for i, v in enumerate(campaign_views[:3])),
             (campaign_views[3], "VICReg/BYOL · vista 4"),
             (still_views[0], "DINOv2 · vista global 1"), (still_views[1], "DINOv2 · vista global 2"),
             (still_views[0], "DINOv2 · máscara iBOT")]
    for ax, (volume, title) in zip(axes.flat, tiles, strict=True):
        ax.imshow(volume[:, 16, :].T.numpy(), cmap="gray", origin="lower", vmin=0, vmax=1)
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    ax = axes.flat[-1]
    for i in range(4):
        for k in range(4):
            if mask[i, 2, k]:  # patch row of the coronal slice y=16 (patch index 2)
                ax.add_patch(Rectangle((i * 8 - 0.5, k * 8 - 0.5), 8, 8, facecolor=COLOR["dinov2"],
                                       alpha=0.45, edgecolor="white", linewidth=1))
    save(fig, "views.png")


def fig_split(records: list[dict]) -> dict[str, dict[str, int]]:
    subjects: dict[str, dict[str, str]] = defaultdict(dict)
    sessions: dict[str, int] = defaultdict(int)
    for r in records:
        subjects[r["split"]][r["subject"]] = r["group"]
        sessions[r["split"]] += 1
    counts = {}
    for split, groups in subjects.items():
        values = list(groups.values())
        counts[split] = {"HC": values.count("HC"), "AD": values.count("AD"),
                         "otros": sum(v not in ("AD", "HC") for v in values),
                         "sesiones": sessions[split]}
    fig, ax = plt.subplots(figsize=(7.2, 2.2))
    order = ["test", "val", "train"]
    parts = [("HC", COLOR["vicreg"]), ("AD", COLOR["byol"]), ("otros", OTHER)]
    for y, split in enumerate(order):
        left = 0
        for part, color in parts:
            n = counts[split][part]
            ax.barh(y, n, left=left, height=0.5, color=color, edgecolor="white", linewidth=2)
            if n >= 18:
                ax.text(left + n / 2, y, str(n), ha="center", va="center", fontsize=8,
                        color="white" if part != "otros" else INK)
            left += n
        ax.text(left + 4, y, f"{left} sujetos · {counts[split]['sesiones']} sesiones",
                va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(3), order)
    ax.set_xlim(0, 470)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("sujetos", fontsize=8)
    handles = [Rectangle((0, 0), 1, 1, color=c) for _, c in parts]
    ax.legend(handles, ["HC", "AD", "otros diagnósticos (solo SSL)"], ncol=3, loc="upper center",
              bbox_to_anchor=(0.45, 1.32))
    save(fig, "split.png")
    return counts


def band(ax: plt.Axes, series: list[np.ndarray], color: str, label: str, dashed: bool = False) -> None:
    length = min(len(s) for s in series)
    data = np.stack([s[:length] for s in series])
    x = np.arange(length)
    ax.fill_between(x, data.min(0), data.max(0), color=color, alpha=0.12, linewidth=0)
    ax.plot(x, data.mean(0), color=color, label=label, linestyle=(0, (4, 2)) if dashed else "-")


def loss_grid(name: str, runs_: list[dict[str, np.ndarray]], panels: list[tuple[str, str, float]],
              color: str, title: str, cols: int = 4) -> None:
    rows = math.ceil(len(panels) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.35 * cols, 2.15 * rows), squeeze=False)
    for ax, (key, panel_title, weight) in zip(axes.flat, panels, strict=False):
        for split, dashed, label, c in (("train", False, "train", color), ("val", True, "val", INK2)):
            series = [r[f"{split}/{key}"] * weight for r in runs_ if f"{split}/{key}" in r]
            if series:
                band(ax, series, c, label, dashed)
        ax.set_title(panel_title, fontsize=8.5)
        ax.set_xlabel("época", fontsize=7.5)
        ax.tick_params(labelsize=7.5)
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="upper right", ncol=2,
               fontsize=8, bbox_to_anchor=(1.0, 1.0))
    fig.suptitle(title, fontsize=9.5, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, name)


def fig_losses() -> None:
    vicreg = curves(V1, "vicreg_3d")
    for r in vicreg:
        r["train/total"], r["val/total"] = r["train/loss_epoch"], r["val/loss"]
    loss_grid("loss_vicreg.png", vicreg, [
        ("total", "Total = 25·inv + 25·var + 1·cov", 1.0),
        ("invariance", "25 × invarianza (MSE entre vistas)", 25.0),
        ("variance", "25 × varianza (hinge de la std)", 25.0),
        ("covariance", "1 × covarianza (fuera de diagonal)", 1.0)],
        COLOR["vicreg"], "VICReg 3D (ConvNeXt-T) · 3 semillas: media y rango")
    for campaign, name, tag in ((V1, "byol_3d", "EMA 0.996 (receta por defecto)"),
                                (V2, "byol_3d", "EMA 0.99 (campaña v2)")):
        byol = curves(campaign, name)
        for r in byol:
            r["train/total"], r["val/total"] = r["train/loss_epoch"], r["val/loss"]
        loss_grid(f"loss_byol_{'v1' if campaign == V1 else 'v2'}.png", byol, [
            ("total", r"Total $= (d_{12} + d_{21})\,/\,2$", 1.0),
            ("direction_12", r"$d_{12} = 2 - 2\cos(p_1, \bar{z}_2)$", 1.0),
            ("direction_21", r"$d_{21} = 2 - 2\cos(p_2, \bar{z}_1)$", 1.0)],
            COLOR["byol"], f"BYOL 3D (ResNet-18) · {tag} · 3 semillas", cols=3)
    dinov2 = curves(V2, "dinov2_3d")
    for r in dinov2:
        r["train/total"], r["val/total"] = r["train/loss_epoch"], r["val/loss"]
    loss_grid("loss_dinov2.png", dinov2, [
        ("total", "Total = DINO + iBOT + 0.1·KoLeo", 1.0),
        ("dino", "DINO: CLS del estudiante vs profesor", 1.0),
        ("ibot", "iBOT: parches enmascarados", 1.0),
        ("koleo", "0.1 × KoLeo (dispersión del CLS)", 0.1),
        ("teacher_entropy_frac", "Entropía del profesor / ln K", 1.0),
        ("dino_kl", "KL profesor → estudiante", 1.0),
        ("teacher_unique_frac", "Prototipos argmax distintos / batch", 1.0)],
        COLOR["dinov2"], "DINOv2 3D (ViT-S, receta final) · 3 semillas", cols=4)


def comparisons() -> dict[str, dict]:
    out = {}
    for path in (V1 / "results.json", V2 / "results.json"):
        for row in json.loads(path.read_text(encoding="utf-8"))["comparisons"]:
            out[f"{path.parent.name}:{row['comparison'].replace(MINUS, '-')}"] = row["result"]
    return out


def fig_forest(comp: dict[str, dict]) -> list[tuple[str, dict]]:
    rows = [
        ("VICReg − inicio ImageNet", "campaign_ssl3d_v2:Aporte SSL: VICReg − su inicio (`imagenet_convnext_3d`)", "vicreg"),
        ("BYOL EMA 0.996 − inicio ImageNet", "campaign_ssl3d:Aporte SSL 3D (BYOL − imagenet_resnet_3d)", "byol"),
        ("BYOL EMA 0.99 − inicio ImageNet", "campaign_ssl3d_v2:Aporte SSL: BYOL − su inicio (`imagenet_resnet_3d`)", "byol"),
        ("DINOv2 (receta final) − inicio DeiT", "campaign_ssl3d_v2:Aporte SSL: DINOv2 − su inicio (`imagenet_vit_3d`)", "dinov2"),
        ("DINOv2 anterior − ViT aleatorio", "campaign_ssl3d:Aporte SSL 3D (DINOv2 − random_vit_3d)", "dinov2"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 2.9), sharey=True)
    for ax, split, title in zip(axes, ("cv", "test"), ("CV 5×5 (357 sujetos)", "Test (56 sujetos)"),
                                strict=True):
        for y, (_label, key, method) in enumerate(reversed(rows)):
            point, (lo, hi), _ = comp[key.replace(MINUS, "-")][split]
            ax.plot([lo, hi], [y, y], color=COLOR[method], linewidth=2)
            ax.plot(point, y, "o", color=COLOR[method], markersize=7, markeredgecolor="white",
                    markeredgewidth=1.5)
        ax.axvline(0, color=INK2, linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Δ media(AUROC, F1, AUPR)  [IC 95 %]", fontsize=8)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(rows)), [r[0] for r in reversed(rows)], fontsize=8)
    save(fig, "forest.png")
    return [(label, comp[key.replace(MINUS, "-")]) for label, key, _ in rows]


def fig_ema() -> None:
    steps = 690
    fig, ax = plt.subplots(figsize=(4.4, 2.6))
    for tau, color, dashed in ((0.996, COLOR["byol"], False), (0.99, INK2, True)):
        m = [1 - (1 - tau) * (math.cos(math.pi * t / steps) + 1) / 2 for t in range(1, steps + 1)]
        remaining = np.cumprod(m)
        ax.plot(np.arange(1, steps + 1) / 23, remaining, color=color,
                linestyle=(0, (4, 2)) if dashed else "-", label=f"τ base = {tau}")
        ax.text(steps / 23 + 0.3, remaining[-1], f"{remaining[-1]:.0%}", va="center", fontsize=8,
                color=INK2)
    ax.set_xlabel("época (23 pasos por época)", fontsize=8)
    ax.set_ylabel("fracción de pesos iniciales\nen el target", fontsize=8)
    ax.set_ylim(0, 1.02)
    ax.legend()
    save(fig, "ema.png")


def fig_seeds() -> None:
    groups = [("Inicio ImageNet\n(sin SSL)", runs(V2, "imagenet_resnet_3d"), BASELINE),
              ("Pesos\naleatorios", runs(V2, "random_resnet_3d"), BASELINE),
              ("BYOL\nEMA 0.996", runs(V2, "byol_v1_3d"), COLOR["byol"]),
              ("BYOL\nEMA 0.99", runs(V2, "byol_3d"), COLOR["byol"]),
              ("VICReg\n(referencia)", runs(V2, "vicreg_3d"), COLOR["vicreg"])]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 2.8), sharey=True)
    for ax, (getter, title) in zip(axes, ((mean3_cv, "CV 5×5"), (mean3_test, "Test")), strict=True):
        for x, (_label, docs, color) in enumerate(groups):
            values = [getter(d) for d in docs]
            ax.scatter([x] * len(values), values, s=46, color=color, edgecolor="white",
                       linewidth=1.5, zorder=3)
            ax.plot([x - 0.22, x + 0.22], [np.mean(values)] * 2, color=INK, linewidth=2, zorder=4)
        ax.set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=7.5)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("media(AUROC, F1, AUPR)", fontsize=8)
    save(fig, "byol_seeds.png")


def fig_rank() -> None:
    families = [("ResNet-18 3D", [("aleatoria", "random_resnet_3d"), ("ImageNet", "imagenet_resnet_3d"),
                                  ("BYOL 0.996", "byol_v1_3d"), ("BYOL 0.99", "byol_3d")], "byol"),
                ("ConvNeXt-T 3D", [("aleatoria", "random_convnext_3d"), ("ImageNet", "imagenet_convnext_3d"),
                                   ("VICReg", "vicreg_3d")], "vicreg"),
                ("ViT-S 3D", [("aleatoria", "random_vit_3d"), ("DeiT", "imagenet_vit_3d"),
                              ("DINOv2 ant.", "dinov2_v1_3d"), ("DINOv2", "dinov2_3d")], "dinov2")]
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.5))
    for ax, (title, items, method) in zip(axes, families, strict=True):
        for x, (label, name) in enumerate(items):
            values = [d["test"]["embedding_effective_rank"] for d in runs(V2, name)]
            trained = label not in ("aleatoria", "ImageNet", "DeiT")
            color = COLOR[method] if trained else BASELINE
            ax.scatter([x] * len(values), values, s=40, color=color, edgecolor="white", linewidth=1.5,
                       zorder=3)
            ax.plot([x - 0.2, x + 0.2], [np.mean(values)] * 2, color=INK, linewidth=2, zorder=4)
        ax.set_xticks(range(len(items)), [i[0] for i in items], fontsize=7.5)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("rango efectivo (log)", fontsize=8)
    save(fig, "rank.png")


def fig_identity() -> dict | None:
    path = V2 / "encoder_analysis.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for key, label, _method in (("convnext_imagenet", "ConvNeXt-T ImageNet", "vicreg"),
                               ("resnet_imagenet", "ResNet-18 ImageNet", "byol"),
                               ("vit_deit", "ViT-S DeiT", "dinov2")):
        entry = data.get("initial", {}).get(key)
        if entry:
            rows.append((f"{label}\n(inicio)", entry["campaign_views"], entry["bias_only_views"],
                         BASELINE))
    for key, role, label, method in (("byol", "online", "BYOL entrenado\n(online)", "byol"),
                                     ("byol", "target", "BYOL entrenado\n(target EMA)", "byol"),
                                     ("vicreg", "online", "VICReg\nentrenado", "vicreg")):
        entry = data.get("trained", {}).get(key, {}).get(role)
        if entry:
            rows.append((label, entry["campaign_views"], None, COLOR[method]))
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.1))
    metrics = [("same_subject_cos", "different_subject_cos", "Coseno centrado entre vistas"),
               ("view_matching_top1", None, "Emparejar vista A con su B (top-1)"),
               ("between_over_within", None, "Varianza entre sujetos / por augmentation")]
    for ax, (key, other, title) in zip(axes, metrics, strict=True):
        for y, (_label, campaign_scores, still_scores, color) in enumerate(reversed(rows)):
            ax.plot(campaign_scores[key], y, "o", color=color if color != BASELINE else INK2,
                    markersize=7, markeredgecolor="white", markeredgewidth=1.5, zorder=3)
            if other:
                ax.plot(campaign_scores[other], y, "o", markerfacecolor="white",
                        markeredgecolor=MUTED, markersize=6, zorder=3)
            if still_scores is not None:
                ax.plot(still_scores[key], y, "D", color=COLOR["dinov2"], markersize=5, zorder=3)
        if key == "view_matching_top1":
            ax.axvline(1 / data["subjects"], color=MUTED, linewidth=1)
            ax.text(1 / data["subjects"], 0.45, " azar", fontsize=7, color=MUTED)
        if key == "between_over_within":
            ax.axvline(1, color=MUTED, linewidth=1)
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(title, fontsize=8.5)
        ax.grid(axis="y", visible=False)
        ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)] if ax is axes[0] else [],
                      fontsize=7.5)
    handles = [plt.Line2D([], [], marker="o", color=INK2, linestyle="", markersize=7),
               plt.Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor=MUTED,
                          linestyle="", markersize=6),
               plt.Line2D([], [], marker="D", color=COLOR["dinov2"], linestyle="", markersize=5)]
    fig.legend(handles, ["mismo sujeto, vistas de la campaña", "sujetos distintos",
                         "mismo sujeto, vistas solo con bias field"], ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    save(fig, "identity.png")
    return data


def fig_sensitivity() -> dict | None:
    """Per-component augmentation sensitivity of the start vs trained BYOL encoders."""
    path = V2 / "byol_sensitivity.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    encoders = [(k, c) for k, c in (("inicio ImageNet", BASELINE), ("BYOL online", COLOR["byol"]),
                                     ("BYOL target", INK2)) if k in data]
    names = list(data[encoders[0][0]])
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2), sharey=True)
    for ax, (metric, title) in zip(axes, (("sensitivity", "Varianza por la augmentation / entre sujetos"),
                                          ("top1", "Emparejar vistas del mismo sujeto (top-1)")),
                                   strict=True):
        for k, (encoder, color) in enumerate(encoders):
            offset = (k - (len(encoders) - 1) / 2) * 0.22
            for y, name in enumerate(reversed(names)):
                value = data[encoder][name][metric]
                if metric == "sensitivity" and value < 2e-5:
                    value = 2e-5
                    ax.text(value * 1.8, y + offset, "≈0", fontsize=6.5, va="center", color=INK2)
                ax.plot(value, y + offset, "o", color=color, markersize=6.5,
                        markeredgecolor="white", markeredgewidth=1.2, zorder=3,
                        label=encoder if y == 0 else None)
        if metric == "sensitivity":
            ax.set_xscale("log")
            ax.set_xlim(1e-5, 1e5)
            ax.axvline(1, color=MUTED, linewidth=1)
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        else:
            ax.axvline(data["chance"], color=MUTED, linewidth=1)
        ax.set_title(title, fontsize=8.5)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(names)), list(reversed(names)), fontsize=7.5)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.6, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "sensitivity.png")
    return data


def num(value: float) -> str:
    """LaTeX number: two decimals, or a power of ten for tiny values."""
    if value != 0 and abs(value) < 0.01:
        exponent = math.floor(math.log10(abs(value)))
        return f"${value / 10 ** exponent:.1f}\\times10^{{{exponent}}}$"
    if abs(value) >= 100:
        return f"{value:,.0f}".replace(",", "\\,")
    return f"{value:.2f}"


def identity_table(data: dict | None) -> list[str]:
    lines = ["\\newcommand{\\TablaIdentidad}{%", "\\begin{tabular}{llrrrrr}", "\\toprule",
             "Encoder & Vistas & Coseno mismo & Coseno distintos & Emparejar & Entre/por aug. & CKA \\\\",
             "\\midrule"]
    if data:
        for key, label in (("convnext_imagenet", "ConvNeXt-T ImageNet"),
                           ("resnet_imagenet", "ResNet-18 ImageNet"), ("vit_deit", "ViT-S DeiT")):
            for views, view_label in (("campaign_views", "campaña"), ("bias_only_views", "solo sesgo")):
                s = data["initial"][key][views]
                lines.append(f"{label} & {view_label} & {s['same_subject_cos']:.3f} & "
                             f"{s['different_subject_cos']:.3f} & {100 * s['view_matching_top1']:.1f}\\,\\% & "
                             f"{num(s['between_over_within'])} & --- \\\\")
        lines.append("\\midrule")
        for key, role, label in (("byol", "online", "BYOL entrenado (online)"),
                                 ("byol", "target", "BYOL entrenado (target)"),
                                 ("vicreg", "online", "VICReg entrenado")):
            entry = data.get("trained", {}).get(key, {}).get(role)
            if entry:
                s = entry["campaign_views"]
                lines.append(f"{label} & campaña & {s['same_subject_cos']:.3f} & "
                             f"{s['different_subject_cos']:.3f} & {100 * s['view_matching_top1']:.1f}\\,\\% & "
                             f"{num(s['between_over_within'])} & {entry['cka_to_initial']:.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", ""]
    return lines


def sensitivity_table(data: dict | None) -> list[str]:
    lines = ["\\newcommand{\\TablaSensibilidad}{%"]
    if not data:
        return [*lines, "(pendiente)}", ""]
    encoders = [k for k in ("inicio ImageNet", "BYOL online", "BYOL target") if k in data]
    lines += ["\\begin{tabular}{l" + "rr" * len(encoders) + "}", "\\toprule",
              "Componente & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{e}}}" for e in encoders)
              + " \\\\",
              " & " + " & ".join(["sensib. & emparejar"] * len(encoders)) + " \\\\", "\\midrule"]
    for name in data[encoders[0]]:
        cells = [f"{num(data[e][name]['sensitivity'])} & {100 * data[e][name]['top1']:.1f}\\,\\%"
                 for e in encoders]
        lines.append(name.replace("%", "\\%") + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", ""]
    return lines


# ---------------------------------------------------------------- tables
def fmt_ci(result: tuple) -> str:
    point, (lo, hi), _ = result
    return f"${point:+.3f}$ [${lo:+.3f}$, ${hi:+.3f}$]"


def write_tables(counts: dict, forest_rows: list[tuple[str, dict]], identity: dict | None,
                 sensitivity: dict | None) -> None:
    lines = ["% Generado por scripts/legacy/build_report_ssl3d_v2.py; no editar a mano.",
             *identity_table(identity), *sensitivity_table(sensitivity),
             "\\newcommand{\\TablaSplit}{%", "\\begin{tabular}{lrrrrr}", "\\toprule",
             "Split & Sujetos & HC & AD & Otros & Sesiones \\\\", "\\midrule"]
    for split in ("train", "val", "test"):
        c = counts[split]
        lines.append(f"{split} & {c['HC'] + c['AD'] + c['otros']} & {c['HC']} & {c['AD']} & "
                     f"{c['otros']} & {c['sesiones']} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "",
              "\\newcommand{\\TablaResultados}{%", "\\begin{tabular}{llrrrr}", "\\toprule",
              "Método & Encoder & CV & Test & Semillas & Rango ef. \\\\", "\\midrule"]
    table = [("VICReg", "SSL (ConvNeXt-T)", "vicreg_3d"), ("", "inicio ImageNet", "imagenet_convnext_3d"),
             ("", "aleatorio", "random_convnext_3d"),
             ("BYOL", "SSL, EMA 0.996", "byol_v1_3d"), ("", "SSL, EMA 0.99", "byol_3d"),
             ("", "inicio ImageNet", "imagenet_resnet_3d"), ("", "aleatorio", "random_resnet_3d"),
             ("DINOv2", "SSL, receta final", "dinov2_3d"), ("", "SSL, anterior", "dinov2_v1_3d"),
             ("", "inicio DeiT", "imagenet_vit_3d"), ("", "aleatorio", "random_vit_3d")]
    for method, label, name in table:
        docs = runs(V2, name)
        if method and method != "VICReg":
            lines.append("\\midrule")
        rank = np.mean([d["test"]["embedding_effective_rank"] for d in docs])
        lines.append(f"{method} & {label} & {np.mean([mean3_cv(d) for d in docs]):.3f} & "
                     f"{np.mean([mean3_test(d) for d in docs]):.3f} & {len(docs)} & {rank:.1f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "",
              "\\newcommand{\\TablaDeltas}{%", "\\begin{tabular}{lll}", "\\toprule",
              "Comparación & $\\Delta$ CV [IC 95\\,\\%] & $\\Delta$ test [IC 95\\,\\%] \\\\", "\\midrule"]
    for label, result in forest_rows:
        lines.append(f"{label.replace(MINUS, '$-$')} & {fmt_ci(result['cv'])} & "
                     f"{fmt_ci(result['test'])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", ""]
    TABLES.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", TABLES.name)


def main() -> None:
    cache, index, records = cubes_and_records()
    fig_brain_and_cubes()
    fig_hc_vs_ad(cache, index, records)
    fig_views(cache, index, records)
    counts = fig_split(records)
    fig_losses()
    forest_rows = fig_forest(comparisons())
    fig_ema()
    fig_seeds()
    fig_rank()
    identity = fig_identity()
    sensitivity = fig_sensitivity()
    write_tables(counts, forest_rows, identity, sensitivity)


if __name__ == "__main__":
    main()
