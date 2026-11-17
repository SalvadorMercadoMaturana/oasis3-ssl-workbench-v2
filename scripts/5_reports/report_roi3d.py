"""Tables and figure for the region-of-interest campaign (50 epochs, train.py --roi).

    python scripts/5_reports/report_roi3d.py campaign_ssl3d_runpod/roi50

Groups runs by model and ROI (32, 48, 64 voxels around the hippocampi, or
``multi3``: hippocampus + lateral ventricle + posteromedial cortex), averages
the seeds and compares every ROI with the 32-voxel one of the same model and
recipe (paired bootstrap of the AUROC difference over subjects). Writes
``TABLA_ROI.md``, ``roi.json`` and ``figures/roi_auroc.png`` next to the runs.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from report_aug_recipes3d import bootstrap_auc, identity, mean, std  # noqa: E402

LABEL = {"vicreg": "VICReg", "byol": "BYOL", "dinov2": "DINOv2"}
COLOR = {"vicreg": "#2a78d6", "byol": "#eb6834", "dinov2": "#1baf7a"}
ROI_ORDER = ["32", "48", "64", "multi3"]
ROI_LABEL = {"32": "32³ (64 mm)", "48": "48³ (96 mm)", "64": "64³ (128 mm)",
             "multi3": "3 regiones × 32³"}


def main() -> None:
    run_dir = Path(sys.argv[1])
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for path in sorted(run_dir.glob("*/result.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        if "cv" in doc:
            groups[(doc["model"], doc.get("roi") or "32", doc.get("aug_recipe") or "")].append(doc)
    main_recipe = {"vicreg": "geo", "byol": "noise", "dinov2": "elastic_gpu"}
    lines = ["# Región de interés, 50 épocas", "",
             "Cada modelo con su mejor receta de augmentation (VICReg `geo`, BYOL `noise` con "
             "`wide_dproj0.1`, DINOv2 `elastic`). Media ± desviación entre semillas. Δ AUROC "
             "frente a 32³ del mismo modelo con su receta principal, con bootstrap pareado sobre "
             "sujetos (IC 95 %). DINOv2 `elastic` (TorchIO, CPU) es el control de `elastic_gpu`. "
             "Identidad en 142 sujetos no vistos con las vistas de la receta.", "",
             "| Modelo | Receta | ROI | n | AUROC CV | AUROC test | Media3 CV | Media3 test | "
             "Δ AUROC CV vs 32³ | Δ AUROC test vs 32³ | Var_suj/Var_aug | Rango ef. | "
             "s/época | min/corrida | VRAM pico (GB) |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    payload = []
    keys = sorted({(m, r) for m, _, r in groups},
                  key=lambda k: (("vicreg", "byol", "dinov2").index(k[0]), k[1] != main_recipe[k[0]]))
    for model, recipe in keys:
        base = groups.get((model, "32", main_recipe[model]))
        for roi in ROI_ORDER:
            docs = groups.get((model, roi, recipe))
            if not docs:
                continue
            perf = [d.get("perf") or {} for d in docs]
            if base and docs is not base and len(docs) > 1 and len(base) > 1:
                b = bootstrap_auc(docs, base)
                delta = " | ".join(f"{b[s][0]:+.3f} [{b[s][1]:+.3f}, {b[s][2]:+.3f}]"
                                   for s in ("cv", "test"))
            else:
                b, delta = None, "— | —"
            own = [identity(d) for d in docs]
            row = {"model": model, "recipe": recipe, "roi": roi, "n": len(docs),
                   "cv_auc": mean([d["cv_auc"] for d in docs]),
                   "test_auc": mean([d["test_auc"] for d in docs]),
                   "cv_auc_seeds": [d["cv_auc"] for d in docs],
                   "test_auc_seeds": [d["test_auc"] for d in docs],
                   "delta_vs_32": b}
            payload.append(row)
            lines.append(
                f"| {LABEL[model]} | `{recipe}` | {ROI_LABEL[roi]} | {len(docs)} | "
                f"{row['cv_auc']:.3f} ± {std(row['cv_auc_seeds']):.3f} | "
                f"{row['test_auc']:.3f} ± {std(row['test_auc_seeds']):.3f} | "
                f"{mean([d['cv_mean3'] for d in docs]):.3f} | {mean([d['test_mean3'] for d in docs]):.3f} | "
                f"{delta} | {mean([o.get('between_over_within') for o in own]):.3g} | "
                f"{mean([o.get('effective_rank') for o in own]):.2f} | "
                f"{mean([p.get('seconds_per_epoch') for p in perf]):.1f} | "
                f"{mean([d.get('minutes') for d in docs]):.1f} | "
                f"{mean([p.get('peak_vram_reserved_gb') for p in perf]):.2f} |")
    (run_dir / "TABLA_ROI.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (run_dir / "roi.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True, facecolor="#fcfcfb")
    rois = [r for r in ROI_ORDER if any(g[1] == r for g in groups)]
    groups = {(m, r): d for (m, r, rec), d in groups.items() if rec == main_recipe[m]}
    width = 0.26
    for ax, split, title in ((axes[0], "cv", "AUROC de CV (357 sujetos)"),
                             (axes[1], "test", "AUROC de test (56 sujetos)")):
        ax.set_facecolor("#fcfcfb")
        for k, model in enumerate(("vicreg", "byol", "dinov2")):
            for i, roi in enumerate(rois):
                docs = groups.get((model, roi))
                if not docs:
                    continue
                values = [d[f"{split}_auc"] for d in docs]
                x = i + (k - 1) * width
                ax.bar(x, np.mean(values), width=width * 0.95, color=COLOR[model],
                       label=LABEL[model] if i == 0 else None, edgecolor="white")
                ax.scatter([x] * len(values), values, s=10, color="#0b0b0b", zorder=3)
        ax.set_xticks(range(len(rois)), [ROI_LABEL[r] for r in rois])
        ax.set_ylim(0.6, 0.95)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", color="#e1e0d9")
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Tamaño y número de regiones de interés, 50 épocas (puntos = semillas)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = run_dir.parent / "figures" / "roi_auroc.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
