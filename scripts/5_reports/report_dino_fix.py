#!/usr/bin/env python
"""Why DINOv2 lags on split_deploy: one-change-at-a-time variants (campaign_ssl3d_runpod/dino_fix).

    python scripts/5_reports/report_dino_fix.py [--runs campaign_ssl3d_runpod/deploy/runs]

Reads the protocol-v2 scores of the ``dino_fix`` runs (prototype count, batch size, teacher
temperature, DINOv2 ViT-S/14 start, ConvNeXt-T backbone, VICReg on the same ViT) next to the
references of ``v2_ssl``/``v2_sup``/``dep50_init``, and writes
``campaign_ssl3d_runpod/dino_fix/REPORT.md`` and ``figures/dino_fix_*.png``. Everything is
compared on the seed ensembles' representative-CV composite (AUROC + AUPR + F1-macro) / 3,
as in the deployment selection; the test only confirms.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import plot_ssl3d_runpod as base  # noqa: E402
import report_v2fix as v2  # noqa: E402

from t5_ssl.evaluation import protocol_v2 as pp  # noqa: E402

plt = base.plt
OUT = base.CAMPAIGN / "dino_fix"
SPLIT = base.CAMPAIGN / "split_deploy"
E, M3 = "elastic_tio_gpu", "roimulti3"
# key: (label, kind, tag, run name without _s<seed>)
GROUPS: dict[str, tuple[str, str, str, str]] = {
    "base": ("DINOv2 actual (K 4096, lote 16, T 0.01)", "dino", "v2_ssl",
             f"dinov2_campaign__{E}__bbvit_small__{M3}"),
    "k256": ("DINOv2 · K 256", "dino", "dino_fix", f"dinov2_k256__{E}__bbvit_small__{M3}"),
    "b64": ("DINOv2 · lote 64", "dino", "dino_fix", f"dinov2_b64__{E}__bbvit_small__{M3}"),
    "b64_k256": ("DINOv2 · lote 64 + K 256", "dino", "dino_fix",
                 f"dinov2_b64_k256__{E}__bbvit_small__{M3}"),
    "t07": ("DINOv2 · lote 64 + K 256 + T 0.04→0.07", "dino", "dino_fix",
            f"dinov2_b64_k256_t07__{E}__bbvit_small__{M3}"),
    "dv2": ("DINOv2 · lote 64 + K 256 · inicio DINOv2 ViT-S/14", "dino", "dino_fix",
            f"dinov2_b64_k256__{E}__bbvit_small_dv2__{M3}"),
    "cnx": ("DINOv2 · lote 64 + K 256 · ConvNeXt-T", "dino", "dino_fix",
            f"dinov2_b64_k256__{E}__bbconvnext_tiny__{M3}"),
    "vicreg_vit": ("VICReg · ViT-S (mismas vistas y entrada)", "other", "dino_fix",
                   f"vicreg_campaign_augval__{E}__bbvit_small__{M3}"),
    "sup_vit": ("Supervisado · ViT-S", "sup", "v2_sup",
                f"dinov2_supervised__{E}__bbvit_small__{M3}"),
    "init_vit": ("Sin SSL · ViT-S (DeiT-S)", "init", "dep50_init",
                 "vicreg_init__bbvit_small__roimulti3"),
    "init_dv2": ("Sin SSL · ViT-S (DINOv2 ViT-S/14)", "init", "dino_fix_init",
                 "dinov2_init__bbvit_small_dv2__roimulti3"),
    "init_cnx": ("Sin SSL · ConvNeXt-T", "init", "dep50_init",
                 "vicreg_init__bbconvnext_tiny__roimulti3"),
    # the same lines on the hippocampal pair (roi32), the input of VICReg and BYOL
    "base32": ("DINOv2 actual · hipocampo", "dino", "dino_fix",
               f"dinov2_campaign__{E}__bbvit_small__roi32"),
    "b64_k256_32": ("DINOv2 · lote 64 + K 256 · hipocampo", "dino", "dino_fix",
                    f"dinov2_b64_k256__{E}__bbvit_small__roi32"),
    "cnx32": ("DINOv2 · lote 64 + K 256 · ConvNeXt-T · hipocampo", "dino", "dino_fix",
              f"dinov2_b64_k256__{E}__bbconvnext_tiny__roi32"),
    "init_vit32": ("Sin SSL · ViT-S (DeiT-S) · hipocampo", "init", "dino_fix_init",
                   "dinov2_init__bbvit_small__roi32"),
    "init_cnx32": ("Sin SSL · ConvNeXt-T · hipocampo", "init", "dep50_init",
                   "vicreg_init__bbconvnext_tiny__roi32"),
    "byol_bn": ("BYOL · ConvNeXt-T · hipocampo (desplegado v2)", "other", "v2_ssl",
                "byol_wide_dproj0.1__noise__bbconvnext_tiny__roi32"),
    "vicreg": ("VICReg · ConvNeXt-T · hipocampo", "other", "v2_ssl",
               "vicreg_campaign_augval__geo__bbconvnext_tiny__roi32"),
    "sup_convnext": ("Supervisado · ConvNeXt-T · hipocampo", "sup", "v2_sup",
                     "vicreg_supervised__geo__bbconvnext_tiny__roi32"),
}
# Labels without a region are on the three-region input (multi3, 6 channels).
COLORS = {"dino": base.COLOR["dinov2"], "other": "#2a78d6", "sup": "#6b6b6b", "init": base.BASE}
DINO_TERMS = (("teacher_entropy_frac", "Entropía del teacher / ln K"),
              ("dino", "Pérdida DINO (CLS)"), ("koleo", "KoLeo"))


def last_collapse(docs: list[dict]) -> list[dict]:
    rows = [[x for x in (d.get("collapse") or []) if x.get("role") in ("online", "teacher")]
            for d in docs]
    return [r[-1] for r in rows if r]


def final_metric(docs: list[dict], stage: str, term: str) -> list[float]:
    return [s[-1] for d in docs if (s := v2.series(d, stage, term))]


def composite_figure(ens: dict, seeds: dict, keys: list[str]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.42 * len(keys) + 1.6), sharey=True)
    for ax, (where, title) in zip(axes, (("cv_rep", "CV representativa (702, 23 % AD)"),
                                         ("test", "Test representativo (178, 23 % AD)")),
                                  strict=True):
        for i, k in enumerate(keys):
            e = ens[k][where]["mean3"]
            ax.barh(i, e, height=0.72, color=COLORS[GROUPS[k][1]],
                    hatch="///" if GROUPS[k][1] == "sup" else None, edgecolor="white")
            v = [d[where]["mean3"] for d in seeds[k]]
            if len(v) > 1:
                ax.scatter(v, [i] * len(v), s=10, color=base.INK, zorder=3, linewidths=0)
            ax.text(max([e, *v]) + 0.004, i, f"{e:.3f}", va="center", fontsize=8)
        ref = ens["base"][where]["mean3"]
        ax.axvline(ref, color=base.COLOR["dinov2"], linestyle=":", linewidth=1)
        ax.set_xlim(0.5, 0.8)
        ax.set_title(title, fontsize=9.5)
        base.style(ax)
    axes[0].set_yticks(range(len(keys)), [GROUPS[k][0] for k in keys], fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle("Compuesto (AUROC + AUPR + F1-macro)/3 de DINOv2 cambiando una cosa a la vez "
                 "(barra = ensamble de 3 semillas, puntos = semillas; punteada = DINOv2 actual)",
                 fontsize=10.5, color=base.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(base.FIGURES / "dino_fix_compuesto.png", dpi=150)
    plt.close(fig)


def monitors_figure(seeds: dict, keys: list[str]) -> None:
    fig, axes = plt.subplots(1, len(DINO_TERMS), figsize=(15, 3.8))
    palette = ["#0b6e4f", "#1baf7a", "#7fd8b4", "#2a78d6", "#9a3d16", "#c9a227", "#6b6b6b"]
    for ax, (term, title) in zip(axes, DINO_TERMS, strict=True):
        for color, k in zip(palette, keys, strict=False):
            runs = [s for d in seeds[k] if (s := v2.series(d, "train", term))]
            if not runs:
                continue
            n = min(map(len, runs))
            ax.plot(np.mean([r[:n] for r in runs], axis=0), color=color, linewidth=1.8,
                    label=GROUPS[k][0])
        if term == "dino":
            for kk, color in (("base", palette[0]), ("k256", palette[1])):
                if kk in seeds:
                    out_dim = 4096 if kk == "base" else 256
                    ax.axhline(math.log(out_dim), color=color, linestyle=":", linewidth=1)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("época")
        base.style(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle("Monitores de DINOv2 en entrenamiento (media de semillas; punteadas = ln K)",
                 fontsize=10.5, color=base.INK)
    fig.tight_layout(rect=(0, 0.13, 1, 0.92))
    fig.savefig(base.FIGURES / "dino_fix_monitores.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=Path, default=base.CAMPAIGN / "deploy/runs")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    ages = pp.session_ages(SPLIT / "manifest.csv")
    seeds = {k: pp.load_runs(args.runs, g[2], g[3]) for k, g in GROUPS.items()}
    seeds = {k: v for k, v in seeds.items() if v}
    keys = [k for k in GROUPS if k in seeds]
    ens = {k: pp.ensemble(v, ages) for k, v in seeds.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["# Por qué DINOv2 rinde menos (split de despliegue, protocolo v2)", "",
             "Cada variante cambia una sola cosa respecto del DINOv2 actual (ViT-S inflado desde "
             "DeiT-S, 3 regiones, elástica TorchIO, 50 épocas). Sin región en el nombre = 3 "
             "regiones (6 canales); «hipocampo» = el par hipocampal de VICReg y BYOL. "
             "Compuesto = (AUROC + AUPR + "
             "F1-macro)/3 del ensamble de 3 semillas en la CV representativa (702 sujetos, 23 % "
             "AD); el test solo confirma.", "",
             "| Modelo | n | CV rep. AUROC | AUPR | F1-macro | **Compuesto** | Test AUROC | AUPR "
             "| F1-macro | Compuesto | Compuesto por semilla (CV) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k in keys:
        r, t = ens[k]["cv_rep"], ens[k]["test"]
        lines.append(f"| {GROUPS[k][0]} | {len(seeds[k])} | {r['auc']:.3f} | {r['aupr']:.3f} | "
                     f"{r['f1_macro']:.3f} | **{r['mean3']:.3f}** | {t['auc']:.3f} | "
                     f"{t['aupr']:.3f} | {t['f1_macro']:.3f} | {t['mean3']:.3f} | "
                     f"{v2.seeds_cell(seeds[k], 'cv_rep', 'mean3')} |")
    dino_keys = [k for k in keys if GROUPS[k][1] == "dino"]
    best = max(dino_keys, key=lambda k: ens[k]["cv_rep"]["mean3"])
    pairs = [(k, "base") for k in dino_keys if k != "base"]
    pairs += [("vicreg_vit", "base"), ("base", "init_vit"), ("base", "sup_vit"),
              ("dv2", "init_dv2"), ("init_dv2", "init_vit"), ("cnx", "init_cnx"),
              ("base32", "init_vit32"), ("b64_k256_32", "init_vit32"), ("cnx32", "init_cnx32"),
              ("init_vit32", "init_vit"), ("init_cnx32", "init_cnx"),
              (best, "byol_bn"), ("cnx32", "byol_bn"), ("cnx32", "vicreg"),
              ("base32", "byol_bn"), ("b64_k256_32", "byol_bn"),
              ("vicreg_vit", "sup_vit")]
    pairs = list(dict.fromkeys((a, b) for a, b in pairs if a in ens and b in ens and a != b))
    lines += ["", f"Mejor DINOv2 por CV: **{GROUPS[best][0]}**.", "",
              "## Contrastes pareados (bootstrap estratificado, IC 95 %, * = excluye 0)", "",
              "| Contraste | CV rep. Δ compuesto | Δ AUROC | Δ AUPR | Δ F1-macro | Test Δ compuesto "
              "| Δ AUROC |", "|---|---:|---:|---:|---:|---:|---:|"]
    for a, b in pairs:
        ya, sa, _ = ens[a]["_cv"]
        _, sb, _ = ens[b]["_cv"]
        cv = v2.boot_delta(ya, sa, sb, ens[a]["threshold_rep"], ens[b]["threshold_rep"])
        yt, pa = ens[a]["_test"]
        _, pb = ens[b]["_test"]
        te = v2.boot_delta(yt, pa, pb, ens[a]["threshold_rep"], ens[b]["threshold_rep"])
        lines.append(f"| {GROUPS[a][0]} − {GROUPS[b][0]} | {v2.fmt_delta(cv['mean3'])} | "
                     f"{v2.fmt_delta(cv['auc'])} | {v2.fmt_delta(cv['aupr'])} | "
                     f"{v2.fmt_delta(cv['f1_macro'])} | {v2.fmt_delta(te['mean3'])} | "
                     f"{v2.fmt_delta(te['auc'])} |")
    lines += ["", "## Monitores al final del entrenamiento", "",
              "| Modelo | Entropía teacher / ln K | Prototipos argmax distintos por fila | Pérdida "
              "DINO train | val | Rango efectivo | Var_sujeto/Var_aug | Top-1 de vistas |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    cell = lambda v: f"{np.mean(v):.3g}" if v else "—"  # noqa: E731
    for k in keys:
        if GROUPS[k][1] not in ("dino", "other") or k == "byol_bn":
            continue
        last = last_collapse(seeds[k])
        lines.append(f"| {GROUPS[k][0]} | {cell(final_metric(seeds[k], 'train', 'teacher_entropy_frac'))}"
                     f" | {cell(final_metric(seeds[k], 'train', 'teacher_unique_frac'))} | "
                     f"{cell(final_metric(seeds[k], 'train', 'dino'))} | "
                     f"{cell(final_metric(seeds[k], 'val', 'dino'))} | "
                     f"{cell([x['effective_rank'] for x in last])} | "
                     f"{cell([x['between_over_within'] for x in last])} | "
                     f"{cell([x['view_matching_top1'] for x in last])} |")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    base.FIGURES.mkdir(parents=True, exist_ok=True)
    composite_figure(ens, seeds, keys)
    monitors_figure(seeds, [k for k in ("base", "k256", "b64", "b64_k256", "t07", "dv2",
                                        "base32") if k in seeds])


if __name__ == "__main__":
    main()
