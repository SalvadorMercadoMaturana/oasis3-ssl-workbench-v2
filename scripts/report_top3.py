#!/usr/bin/env python
"""Top three models of each SSL method on split_deploy (protocol v2), in the v2_metricas layout.

    python scripts/report_top3.py [--runs campaign_ssl3d_runpod/deploy/runs]

Candidates are every VICReg, BYOL and DINOv2 configuration scored with protocol v2 on the
deployment split (OASIS3 only; 3 seeds each). Within each method the three seed ensembles
with the highest representative-CV composite (AUROC + AUPR + F1-macro) / 3 are kept, the
deployment selection rule; the test only confirms. Writes
``campaign_ssl3d_runpod/top3/REPORT.md`` and ``figures/top3_metricas.png``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import plot_ssl3d_runpod as base  # noqa: E402
import probe_protocol as pp  # noqa: E402

plt = base.plt
OUT = base.CAMPAIGN / "top3"
SPLIT = base.CAMPAIGN / "split_deploy"
E, GN = "elastic_tio_gpu", "gn_dp0.3_dq0_wd0.01"
# method -> [(label, tag, run name without _s<seed>)]
CANDIDATES: dict[str, list[tuple[str, str, str]]] = {
    "vicreg": [
        ("VICReg · ConvNeXt-T · hipocampo · geo", "v2_ssl",
         "vicreg_campaign_augval__geo__bbconvnext_tiny__roi32"),
        ("VICReg · ConvNeXt-T · hipocampo · geo (corrida 20-oct, misma receta)", "dep50_ssl",
         "vicreg_campaign_augval__geo__bbconvnext_tiny__roi32"),
        ("VICReg · ViT-S · 3 regiones · elástica", "dino_fix",
         f"vicreg_campaign_augval__{E}__bbvit_small__roimulti3"),
    ],
    "byol": [
        ("BYOL · ConvNeXt-T · cabezas BN · noise (desplegado v2)", "v2_ssl",
         "byol_wide_dproj0.1__noise__bbconvnext_tiny__roi32"),
        ("BYOL · ResNet-18 · cabezas GN · noise", "v2_ssl", f"byol_{GN}__noise__bbresnet18__roi32"),
        ("BYOL · ConvNeXt-T · cabezas GN · geo", "v2_ssl", f"byol_{GN}__geo__bbconvnext_tiny__roi32"),
        ("BYOL · ConvNeXt-T · cabezas GN · noise (v1)", "dep50_ssl",
         f"byol_{GN}__noise__bbconvnext_tiny__roi32"),
    ],
    "dinov2": [
        ("DINOv2 · ViT-S · 3 regiones (actual)", "v2_ssl",
         f"dinov2_campaign__{E}__bbvit_small__roimulti3"),
        ("DINOv2 · ViT-S · 3 regiones (corrida 20-oct)", "dep50_ssl",
         f"dinov2_campaign__{E}__bbvit_small__roimulti3"),
        ("DINOv2 · ViT-S · 3 regiones · K 256", "dino_fix",
         f"dinov2_k256__{E}__bbvit_small__roimulti3"),
        ("DINOv2 · ViT-S · 3 regiones · lote 64", "dino_fix",
         f"dinov2_b64__{E}__bbvit_small__roimulti3"),
        ("DINOv2 · ViT-S · 3 regiones · lote 64 + K 256", "dino_fix",
         f"dinov2_b64_k256__{E}__bbvit_small__roimulti3"),
        ("DINOv2 · ViT-S · 3 regiones · lote 64 + K 256 + T 0.04→0.07", "dino_fix",
         f"dinov2_b64_k256_t07__{E}__bbvit_small__roimulti3"),
        ("DINOv2 · ViT-S (inicio DINOv2) · 3 regiones · lote 64 + K 256", "dino_fix",
         f"dinov2_b64_k256__{E}__bbvit_small_dv2__roimulti3"),
        ("DINOv2 · ConvNeXt-T · 3 regiones · lote 64 + K 256", "dino_fix",
         f"dinov2_b64_k256__{E}__bbconvnext_tiny__roimulti3"),
        ("DINOv2 · ViT-S · hipocampo", "dino_fix", f"dinov2_campaign__{E}__bbvit_small__roi32"),
        ("DINOv2 · ViT-S · hipocampo · lote 64 + K 256", "dino_fix",
         f"dinov2_b64_k256__{E}__bbvit_small__roi32"),
        ("DINOv2 · ConvNeXt-T · hipocampo · lote 64 + K 256", "dino_fix",
         f"dinov2_b64_k256__{E}__bbconvnext_tiny__roi32"),
    ],
}
NAMES = {"vicreg": "VICReg", "byol": "BYOL", "dinov2": "DINOv2"}
SHADES = {"vicreg": ("#1c5aa6", "#2a78d6", "#8fb9ec"),
          "byol": ("#9a3d16", "#eb6834", "#f6c9ad"),
          "dinov2": ("#0b6e4f", "#1baf7a", "#8fdcbc")}
METRICS = (("auc", "AUROC"), ("aupr", "AUPR"), ("f1_macro", "F1-macro"), ("mean3", "Compuesto"))
ROWS = (("cv_rep", "CV representativa (702 sujetos, 23 % AD)"),
        ("test", "Test representativo (178 sujetos, 23 % AD)"))


def figure(chosen: list[dict], chance: dict[str, float]) -> None:
    fig, axes = plt.subplots(len(ROWS), len(METRICS), figsize=(17, 9.2), sharey=True)
    xs, x = [], 0.0
    for i, c in enumerate(chosen):  # a gap between methods
        if i and c["method"] != chosen[i - 1]["method"]:
            x += 0.7
        xs.append(x)
        x += 1
    centers = {m: np.mean([xi for xi, c in zip(xs, chosen, strict=True) if c["method"] == m])
               for m in NAMES}
    for r, (where, row_title) in enumerate(ROWS):
        for col, (key, name) in enumerate(METRICS):
            ax = axes[r, col]
            for xi, c in zip(xs, chosen, strict=True):
                v = [d[where][key] for d in c["seeds"]]
                e = c["ens"][where][key]
                ax.bar(xi, e, width=0.82, color=c["color"], edgecolor="white", label=c["label"])
                ax.scatter([xi] * len(v), v, s=9, color=base.INK, zorder=3, linewidths=0)
                ax.text(xi, max([e, *v]) + 0.015, f"{e:.3f}", ha="center", va="bottom",
                        fontsize=7.5, rotation=90)
            if key == "aupr":
                ax.axhline(chance[where], color=base.INK2, linewidth=1, linestyle=":")
            ax.set_xticks(list(centers.values()), list(NAMES.values()), fontsize=9)
            ax.set_ylim(0.3, 1.18)
            ax.set_title(name, fontsize=10)
            if col == 0:
                ax.set_ylabel(row_title.replace(" (", "\n("), fontsize=9)
            base.style(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    order = [i for m in NAMES for i, c in enumerate(chosen) if c["method"] == m]
    by_method = [[i for i in order if chosen[i]["method"] == m] for m in NAMES]
    columns = [i for col in zip(*by_method, strict=False) for i in col]  # row-major, 3 columns
    fig.legend([handles[i] for i in columns], [labels[i] for i in columns], loc="lower center",
               ncol=3, fontsize=8.5, frameon=False)
    fig.suptitle("Los 3 mejores modelos de cada método (orden por compuesto en la CV "
                 "representativa; barra = ensamble de 3 semillas, puntos = semillas; F1 con umbral "
                 "elegido fuera de muestra; punteada = AUPR al azar)", fontsize=10.5, color=base.INK)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig.savefig(base.FIGURES / "top3_metricas.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=Path, default=base.CAMPAIGN / "deploy/runs")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    ages = pp.session_ages(SPLIT / "manifest.csv")
    chosen, ranking = [], []
    for method, cands in CANDIDATES.items():
        scored = []
        for label, tag, run in cands:
            seeds = pp.load_runs(args.runs, tag, run)
            if seeds:
                scored.append({"method": method, "label": label, "seeds": seeds,
                               "ens": pp.ensemble(seeds, ages)})
        scored.sort(key=lambda c: -c["ens"]["cv_rep"]["mean3"])
        ranking += [(method, c["label"], c["ens"]["cv_rep"]["mean3"], len(c["seeds"]))
                    for c in scored]
        for c, color in zip(scored[:3], SHADES[method], strict=False):
            c["color"] = color
            c["short"] = c["label"].replace(f"{NAMES[method]} · ", f"{NAMES[method]}\n", 1)
            chosen.append(c)
    first = chosen[0]["seeds"][0]["scores"]
    chance = {"cv_rep": float(np.mean(first["cv_y"])), "test": float(np.mean(first["test_y"]))}
    lines = ["# Los 3 mejores modelos de cada método (split de despliegue, protocolo v2)", "",
             "Ensamble de 3 semillas (media de probabilidades, como se despliega). Orden dentro de "
             "cada método: compuesto = (AUROC + AUPR + F1-macro)/3 en la CV representativa (702 "
             "sujetos, 23 % AD, edades reales). F1-macro con umbral elegido fuera de muestra. "
             "El test solo confirma.", "",
             "| Método | Modelo | CV AUROC | CV AUPR | CV F1-macro | **CV compuesto** | Test AUROC "
             "| Test AUPR | Test F1-macro | Test compuesto |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in chosen:
        r, t = c["ens"]["cv_rep"], c["ens"]["test"]
        lines.append(f"| {NAMES[c['method']]} | {c['label']} | {r['auc']:.3f} | {r['aupr']:.3f} | "
                     f"{r['f1_macro']:.3f} | **{r['mean3']:.3f}** | {t['auc']:.3f} | "
                     f"{t['aupr']:.3f} | {t['f1_macro']:.3f} | {t['mean3']:.3f} |")
    lines += ["", f"AUPR al azar: {chance['cv_rep']:.3f} (CV), {chance['test']:.3f} (test).", "",
              "## Todos los candidatos (compuesto CV representativa)", "",
              "| Método | Modelo | Semillas | CV compuesto |", "|---|---|---:|---:|"]
    lines += [f"| {NAMES[m]} | {lab} | {n} | {v:.3f} |" for m, lab, v, n in ranking]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    figure(chosen, chance)


if __name__ == "__main__":
    main()
