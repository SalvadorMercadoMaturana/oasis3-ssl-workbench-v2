"""BYOL overfitting diagnosis: current recipe vs batch 32 / 64 vs GroupNorm heads.

    python scripts/plot_byol_diagnosis.py [campaign_ssl3d_runpod/byol_diag30]

Runs (train.py, split 361/63/84, 30 epochs, seeds 1-3) under ``<dir>/{b16,b32,b64,gn}``:
BYOL ``wide_dproj0.1`` + ``noise`` on the 32³ hippocampal pair with batch 16 (the
current model), 32 and 64, and ``wide_dproj0.1_gn`` (GroupNorm instead of
BatchNorm in projector and predictor) with batch 16. Writes ``DIAGNOSTICO.md`` and
``figures/byol_diag_{metricas,objetivo,colapso}.png``.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import plot_ssl3d_runpod as base  # noqa: E402
from report_aug_recipes3d import bootstrap_auc, mean, std  # noqa: E402

plt = base.plt
CONFIGS = [("b16", "Actual (batch 16, BN)", "#eb6834"), ("b32", "Batch 32", "#c9a227"),
           ("b64", "Batch 64", "#7e57c2"), ("gn", "GroupNorm en cabezas (batch 16)", "#1f8a70")]


def load(run_dir: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for tag, _, _ in CONFIGS:
        for path in sorted((run_dir / tag).glob("byol_*/result.json")):
            out[tag].append(json.loads(path.read_text(encoding="utf-8")))
    return out


def rows_of(doc: dict, role: str) -> list[dict]:
    return [r for r in doc.get("collapse") or [] if r["role"] == role]


def metrics_figure(runs: dict, init: list[dict] | None, out: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.6), sharey=True)
    for r, split in enumerate(("cv", "test")):
        for c, key in enumerate(base.METRICS):
            ax = axes[r, c]
            for i, (tag, label, color) in enumerate(CONFIGS):
                if runs.get(tag):
                    base.bar(ax, i, runs[tag], split, key, color, 0.7, label=label)
            if init:
                ax.axhline(base.metric(init[0], split, key), color=base.MUTED, linestyle="--",
                           linewidth=1.1, label="ResNet-18 3D sin SSL")
            ax.set_xticks(range(len(CONFIGS)), ["actual\nb16", "b32", "b64", "GN"])
            ax.set_ylim(0.55, 0.97)
            n = len(runs["b16"][0]["labels"]) if split == "cv" else \
                len(runs["b16"][0]["test"]["test_predictions"])
            ax.set_title(f"{key[0]} — {'CV' if split == 'cv' else 'test'} ({n} sujetos)",
                         fontsize=9.5)
            base.style(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=8.5)
    fig.suptitle("BYOL: actual vs batch 32 / 64 vs GroupNorm (30 épocas, split 361/63/84, "
                 "3 semillas)", fontsize=10.5, color=base.INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(out / "byol_diag_metricas.png", dpi=160)
    plt.close(fig)


def objective_figure(runs: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.0))
    panels = [("Pérdida de entrenamiento (registrada)", None),
              ("Validación registrada (cubo vs espejo L/R)", None),
              ("Objetivo BYOL, mismas vistas: train vs no vistos", None),
              ("Brecha: no vistos − train", None)]
    for tag, label, color in CONFIGS:
        docs = runs.get(tag) or []
        if not docs:
            continue
        for ax, stage in ((axes[0], "train"), (axes[1], "val")):
            curves = [s for d in docs if (s := base.series(d, stage, "loss"))]
            n = min(len(s) for s in curves)
            ax.plot(np.mean([s[:n] for s in curves], axis=0), color=color, linewidth=1.8,
                    label=label)
        obj = [rows_of(d, "objective") for d in docs]
        obj = [o for o in obj if o]
        if obj:
            n = min(len(o) for o in obj)
            epochs = [x["epoch"] for x in obj[0][:n]]
            held = np.mean([[x["byol_loss_heldout"] for x in o[:n]] for o in obj], axis=0)
            train = np.mean([[x["byol_loss_train"] for x in o[:n]] for o in obj], axis=0)
            axes[2].plot(epochs, train, "-", color=color, linewidth=1.8)
            axes[2].plot(epochs, held, "--", color=color, linewidth=1.8)
            axes[3].plot(epochs, held - train, color=color, linewidth=1.8, label=label)
    for ax, (title, _) in zip(axes, panels, strict=True):
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("época")
        base.style(ax)
    axes[3].axhline(0, color=base.MUTED, linewidth=1)
    axes[2].plot([], [], "-", color=base.INK2, label="sujetos de train")
    axes[2].plot([], [], "--", color=base.INK2, label="sujetos no vistos")
    axes[2].legend(frameon=False, fontsize=8)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("¿Sobreajuste del objetivo? 2 − 2·cos simétrico (media de 3 semillas); en los "
                 "paneles 3–4 con vistas fijas de la receta y la red en modo evaluación",
                 fontsize=10.5, color=base.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / "byol_diag_objetivo.png", dpi=160)
    plt.close(fig)


COLLAPSE = [("effective_rank", "Rango efectivo", False),
            ("between_over_within", "Var_sujeto / Var_aug", True),
            ("same_subject_cos", "Coseno mismo sujeto", False),
            ("view_matching_top1", "Top-1 de vistas", False)]


def collapse_figure(runs: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, len(COLLAPSE), figsize=(17, 3.9))
    for ax, (key, title, log) in zip(axes, COLLAPSE, strict=True):
        for tag, label, color in CONFIGS:
            for role, style_ in (("online", "-"), ("target", "--")):
                series = [rows_of(d, role) for d in runs.get(tag) or []]
                series = [s for s in series if s]
                if not series:
                    continue
                n = min(len(s) for s in series)
                ax.plot([x["epoch"] for x in series[0][:n]],
                        np.mean([[x[key] for x in s[:n]] for s in series], axis=0), style_,
                        color=color, linewidth=1.7, label=label if role == "online" else None)
        if log:
            ax.set_yscale("log")
            ax.axhline(1.0, color="#b3261e", linestyle=":", linewidth=1)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("época")
        base.style(ax)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Colapso en los sujetos no vistos (continua = encoder online, discontinua = "
                 "target EMA; media de 3 semillas)", fontsize=10.5, color=base.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out / "byol_diag_colapso.png", dpi=160)
    plt.close(fig)


def table(runs: dict, out_dir: Path) -> None:
    ref = runs["b16"]
    lines = ["# Diagnóstico de BYOL (30 épocas, split 361/63/84, 3 semillas)", "",
             "| Config | AUROC CV | AUROC test | Media3 CV | Media3 test | Δ AUROC CV vs actual | "
             "Δ AUROC test vs actual | Objetivo train → no vistos (final) | Brecha final | "
             "Subida val registrada | Var_suj/Var_aug | Rango ef. | s/época |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for tag, label, _ in CONFIGS:
        docs = runs.get(tag) or []
        if not docs:
            continue
        delta = "— | —"
        if tag != "b16":
            b = bootstrap_auc(docs, ref)
            delta = " | ".join(f"{b[s][0]:+.3f} [{b[s][1]:+.3f}, {b[s][2]:+.3f}]" for s in ("cv", "test"))
        finals = [rows_of(d, "objective")[-1] for d in docs if rows_of(d, "objective")]
        train_o = mean([f["byol_loss_train"] for f in finals])
        held_o = mean([f["byol_loss_heldout"] for f in finals])
        rise = mean([(lambda v: v[-1] - min(v))(d["training_metrics"]["val/loss"]) for d in docs])
        own = [d["identity"]["own"] for d in docs if d.get("identity")]
        lines.append(
            f"| {label} | {mean([d['cv_auc'] for d in docs]):.3f} ± {std([d['cv_auc'] for d in docs]):.3f} | "
            f"{mean([d['test_auc'] for d in docs]):.3f} ± {std([d['test_auc'] for d in docs]):.3f} | "
            f"{mean([d['cv_mean3'] for d in docs]):.3f} | {mean([d['test_mean3'] for d in docs]):.3f} | "
            f"{delta} | {train_o:.3f} → {held_o:.3f} | {held_o - train_o:+.3f} | {rise:.3f} | "
            f"{mean([o['between_over_within'] for o in own]):.3g} | "
            f"{mean([o['effective_rank'] for o in own]):.1f} | "
            f"{mean([(d.get('perf') or {}).get('seconds_per_epoch') for d in docs]):.2f} |")
    lines += ["", "*Objetivo*: 2 − 2·cos simétrico con vistas fijas de la receta (`noise`) en "
              "147 sujetos de train y en los 147 de val + test, red en modo evaluación. *Subida "
              "val registrada*: pérdida de validación del entrenamiento (cubo vs su espejo L/R) "
              "final − mínima.", ""]
    (out_dir / "DIAGNOSTICO.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    run_dir = Path(sys.argv[1] if len(sys.argv) > 1 else base.CAMPAIGN / "byol_diag30")
    runs = load(run_dir)
    init_path = base.CAMPAIGN / "split361_30" / "byol_init__roi32_s1" / "result.json"
    init = [json.loads(init_path.read_text(encoding="utf-8"))] if init_path.is_file() else None
    base.FIGURES.mkdir(parents=True, exist_ok=True)
    table(runs, run_dir)
    metrics_figure(runs, init, base.FIGURES)
    objective_figure(runs, base.FIGURES)
    collapse_figure(runs, base.FIGURES)
    print("figures in", base.FIGURES)


if __name__ == "__main__":
    main()
