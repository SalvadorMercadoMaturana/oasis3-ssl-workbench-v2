"""Split 361/63/84 at 100 epochs: SSL vs the random-init encoder vs full supervision.

    python scripts/report_split361_100.py

Reads ``campaign_ssl3d_runpod/split361_100/`` (the pod tags split361_100_{dino,cnn,
random,sup} merged): the SSL champions (train.py ``--identity --collapse-every 2``),
the same encoders with random weights and no SSL (``<model>_init__roi*``,
``--init-only`` with ``model.backbone.pretrained=false``) and the end-to-end
supervised baselines (``scripts/train_supervised.py``, ``<model>_supervised__*``).
For reference it also reads the 30-epoch runs (split361_30, BYOL from byol_fbn30)
and the ImageNet-initialized encoders without SSL of split361_30.

Writes ``split361_100/REPORT.md``; ``campaign_ssl3d_runpod/figures/split361_100_
{metricas,curvas_vicreg,curvas_byol,curvas_dinov2,colapso}.png`` (same layout as the
split361 figures of ``plot_champions.py``, metrics with three bars per model); and,
for docs/informe_split361.tex, those figures plus ``delta_100.png``,
``collapse_100.png``, ``byol_objetivo_100.png`` and ``supervised_100.png`` in
``docs/figures_split361/``.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import plot_champions as champ  # noqa: E402
import plot_ssl3d_runpod as base  # noqa: E402
from report_aug_recipes3d import bootstrap_auc, mean, std  # noqa: E402

plt = base.plt
RUNS = base.CAMPAIGN / "split361_100"
MODELS = ("vicreg", "byol", "dinov2")
SSL = {"vicreg": "vicreg_campaign_augval__geo__roi32",
       "byol": "byol_gn_dp0.3_dq0_wd0.01__noise__roi32",
       "dinov2": "dinov2_campaign__elastic__roimulti3"}
RANDOM = {"vicreg": "vicreg_init__roi32", "byol": "byol_init__roi32",
          "dinov2": "dinov2_init__roimulti3"}
SUPERVISED = {"vicreg": "vicreg_supervised__geo__roi32", "byol": "byol_supervised__noise__roi32",
              "dinov2": "dinov2_supervised__elastic_gpu__roimulti3"}
SSL30 = {"vicreg": ("split361_30", "vicreg_campaign__geo__roi32"),
         "byol": ("byol_fbn30", "byol_gn_dp0.3_dq0_wd0.01__noise__roi32"),
         "dinov2": ("split361_30", "dinov2_campaign__elastic__roimulti3")}
IMAGENET = {"vicreg": "vicreg_init__roi32", "byol": "byol_init__roi32",
            "dinov2": "dinov2_init__roimulti3"}
TICK = {"vicreg": "VICReg\ngeo · 32³", "byol": "BYOL\nnoise · 32³ · GN",
        "dinov2": "DINOv2\nelastic · 3 regiones"}
ENCODER = {"vicreg": "ConvNeXt-T 3D", "byol": "ResNet-18 3D", "dinov2": "ViT-S 3D"}
SUP_COLOR, TITLE, PREFIX = "#52514e", "split 361/63/84, 100 épocas", "split361_100_"
DOCS = base.ROOT / "docs" / "figures_split361"


def groups() -> dict[str, dict[str, list[dict]]]:
    runs = base.load_dir(RUNS)
    previous = {d: base.load_dir(base.CAMPAIGN / d) for d in ("split361_30", "byol_fbn30")}
    return {"ssl": {m: runs[SSL[m]] for m in MODELS},
            "random": {m: runs[RANDOM[m]] for m in MODELS},
            "supervised": {m: runs[SUPERVISED[m]] for m in MODELS},
            "ssl30": {m: previous[SSL30[m][0]][SSL30[m][1]] for m in MODELS},
            "imagenet": {m: previous["split361_30"][IMAGENET[m]] for m in MODELS}}


# ---------------------------------------------------------------- campaign figures
def metrics_figure(g: dict) -> None:
    values = [base.metric(d, s, k) for kind in ("ssl", "random", "supervised") for m in MODELS
              for d in g[kind][m] for s in ("cv", "test") for k in base.METRICS]
    low = min(0.55, np.floor((min(values) - 0.04) * 20) / 20)
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.8), sharey=True)
    first = g["ssl"]["vicreg"][0]
    n_cv, n_test = len(first["labels"]), len(first["test"]["test_predictions"])
    bars = (("ssl", -0.27, lambda m: base.COLOR[m], None, "SSL (100 épocas) + probe congelado"),
            ("random", 0.0, lambda m: base.BASE, None,
             "Sin SSL: mismo encoder con pesos aleatorios + probe congelado"),
            ("supervised", 0.27, lambda m: SUP_COLOR, "///",
             "Supervisado completo: mismo encoder entrenado end-to-end en AD/HC"))
    for r, (split, split_name) in enumerate((("cv", f"CV (train+val, {n_cv} sujetos)"),
                                             ("test", f"Test ({n_test} sujetos)"))):
        for c, key in enumerate(base.METRICS):
            ax = axes[r, c]
            for i, m in enumerate(MODELS):
                for kind, dx, color, hatch, label in bars:
                    base.bar(ax, i + dx, g[kind][m], split, key, color(m), 0.26,
                             alpha=0.9 if kind == "supervised" else 1.0, hatch=hatch,
                             label=label if i == 0 else None, text_y=low + 0.015)
            ax.set_xticks(range(3), [TICK[m] for m in MODELS], fontsize=7.5)
            ax.set_ylim(low, 0.97)
            ax.set_title(f"{key[0]} — {split_name}", fontsize=9.5)
            base.style(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.03), ncol=3,
               frameon=False, fontsize=8)
    fig.suptitle(f"Modelos campeones, {TITLE}: SSL vs sin SSL (encoder aleatorio) vs supervisado "
                 "(3 semillas; puntos = semillas)", fontsize=10.5, color=base.INK)
    fig.text(0.5, 0.008, "CV del SSL y sin SSL: probe 5×5; CV supervisada: 5 folds entrenados "
             "end-to-end hasta la época elegida en validación, puntajes calibrados (Platt) en "
             "los sujetos de entrenamiento de cada fold", ha="center", fontsize=7.5,
             color=base.INK2)
    fig.tight_layout(rect=(0, 0.09, 1, 0.97))
    fig.savefig(base.FIGURES / f"{PREFIX}metricas.png", dpi=160)
    plt.close(fig)


def campaign_figures(g: dict) -> None:
    champ.CHAMPION.update(SSL)
    champ.PREFIX, champ.TITLE = PREFIX, "100 épocas"
    champ.TICK.update({"vicreg": "VICReg\ngeo · 32³ · validación aumentada",
                       "byol": "BYOL\nnoise · 32³ · GN dp0.3 wd0.01"})
    base.FIGURES.mkdir(parents=True, exist_ok=True)
    metrics_figure(g)
    by_name = {SSL[m]: g["ssl"][m] for m in MODELS}
    champ.curves_figures(by_name)
    champ.collapse_figure(by_name)


# ---------------------------------------------------------------- report figures
def fig_delta(rep, d: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.0), sharey=True)
    rows = [(m, ref) for m in ("dinov2", "byol", "vicreg") for ref in ("random", "supervised")]
    for ax, split, title in ((axes[0], "cv", "CV (329 sujetos de train+val)"),
                             (axes[1], "test", "Test (84 sujetos nunca vistos)")):
        ax.axvline(0, color=rep.INK2, linewidth=1)
        for y, (m, ref) in enumerate(rows):
            mid, lo, hi, _ = d[m][ref][split]
            marker = "o" if ref == "random" else "s"
            face = rep.COLOR[m] if ref == "random" else "white"
            ax.plot([lo, hi], [y, y], color=rep.COLOR[m], linewidth=2.5, solid_capstyle="butt")
            ax.plot(mid, y, marker, color=rep.COLOR[m], markerfacecolor=face, markersize=7,
                    markeredgewidth=1.6)
            ax.text(mid, y + 0.3, f"{mid:+.3f}", ha="center", fontsize=7.5, color=rep.INK)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Δ AUROC del SSL frente a la referencia", fontsize=8)
    axes[0].set_yticks(range(len(rows)), [f"{rep.LABEL[m]} − {'aleatorio' if ref == 'random' else 'supervisado'}"
                                          for m, ref in rows], fontsize=8)
    axes[0].set_ylim(-0.6, len(rows) - 0.2)
    rep.save(fig, "delta_100.png")


def fig_collapse(rep, g: dict) -> None:
    metrics = [("between_over_within", "Var entre sujetos / Var por augmentation", True),
               ("view_matching_top1", "Top-1: la vista encuentra a su gemela", False)]
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 2.9))
    for ax, (key, title, log) in zip(axes, metrics, strict=True):
        for m in MODELS:
            for docs, style_, label in ((g["ssl"][m], "-", rep.LABEL[m] + ", 100 épocas"),
                                        (g["ssl30"][m], ":", None)):
                series: dict[int, list[float]] = {}
                for d in docs:
                    for rec in d.get("collapse") or []:
                        if rec["role"] == "online":
                            series.setdefault(rec["epoch"], []).append(rec[key])
                epochs = sorted(series)
                ax.plot(epochs, [np.mean(series[e]) for e in epochs], style_, color=rep.COLOR[m],
                        linewidth=2 if style_ == "-" else 1.4, label=label)
        if log:
            ax.set_yscale("log")
            ax.axhline(1, color=rep.INK2, linewidth=1, linestyle=":")
            ax.text(100, 1.1, "1 = la augmentation pesa\nlo mismo que el sujeto", ha="right",
                    va="bottom", fontsize=7.5, color=rep.INK2)
        else:
            ax.axhline(1 / 147, color=rep.INK2, linewidth=1, linestyle=":")
            ax.text(100, 0.03, "azar (1/147)", ha="right", fontsize=7.5, color=rep.INK2)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("época", fontsize=8)
    axes[1].plot([], [], ":", color=rep.INK2, linewidth=1.4, label="punteada: corrida de 30 épocas")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.12), fontsize=8)
    rep.save(fig, "collapse_100.png")


def fig_byol_objective(rep, g: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.0))
    for docs, color, label in ((g["ssl"]["byol"], rep.COLOR["byol"], "100 épocas"),
                               (g["ssl30"]["byol"], rep.MUTED, "30 épocas")):
        for ax, stage in ((axes[0], "train"), (axes[1], "val")):
            curves = [s for d in docs if (s := base.series(d, stage, "loss"))]
            n = min(len(s) for s in curves)
            ax.plot(np.arange(1, n + 1), np.mean([s[:n] for s in curves], axis=0), color=color,
                    label=label)
        rows = [[r for r in d.get("collapse") or [] if r["role"] == "objective"] for d in docs]
        n = min(len(r) for r in rows)
        epochs = [r["epoch"] for r in rows[0][:n]]
        train = np.mean([[r["byol_loss_train"] for r in rr[:n]] for rr in rows], axis=0)
        held = np.mean([[r["byol_loss_heldout"] for r in rr[:n]] for rr in rows], axis=0)
        axes[2].plot(epochs, train, "-", color=color, label=f"{label}, sujetos de train")
        axes[2].plot(epochs, held, "--", color=color, label=f"{label}, sujetos no vistos")
    titles = ("Pérdida de entrenamiento (registrada)", "Validación registrada (cubo vs espejo L/R)",
              "Objetivo en modo evaluación, mismas vistas")
    for ax, title in zip(axes, titles, strict=True):
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("época", fontsize=8)
    axes[0].legend(fontsize=8)
    axes[2].legend(fontsize=7.5)
    rep.save(fig, "byol_objetivo_100.png")


def fig_supervised(rep, g: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.0))
    for m in MODELS:
        for i, d in enumerate(g["supervised"][m]):
            h = d["history"]
            epochs = [r["epoch"] for r in h]
            axes[0].plot(epochs, [r["train_loss"] for r in h], color=rep.COLOR[m], linewidth=1.2,
                         alpha=0.8, label=f"{rep.LABEL[m]} ({ENCODER[m]})" if i == 0 else None)
            axes[1].plot(epochs, [r["val_auc"] for r in h], color=rep.COLOR[m], linewidth=1.2,
                         alpha=0.8)
            best = d["best_epoch"]
            axes[1].plot(best, h[best - 1]["val_auc"], "o", color=rep.COLOR[m], markersize=5)
    axes[0].set_title("BCE de entrenamiento (balanceada)", fontsize=9)
    axes[1].set_title("AUROC en validación (63 sujetos); punto = época elegida", fontsize=9)
    for ax in axes:
        ax.set_xlabel("época", fontsize=8)
    axes[0].legend(fontsize=8)
    rep.save(fig, "supervised_100.png")


# ---------------------------------------------------------------- tables
def table(g: dict) -> dict:
    fmt = lambda x, s: f"{x[s][0]:+.3f} [{x[s][1]:+.3f}, {x[s][2]:+.3f}]"  # noqa: E731
    row = lambda name, docs: (  # noqa: E731
        f"| {name} | {mean([d['cv_auc'] for d in docs]):.3f} ± {std([d['cv_auc'] for d in docs]):.3f} "
        f"| {mean([d['test_auc'] for d in docs]):.3f} ± {std([d['test_auc'] for d in docs]):.3f} "
        f"| {mean([d['cv_mean3'] for d in docs]):.3f} | {mean([d['test_mean3'] for d in docs]):.3f} |")
    lines = [f"# Split 361/63/84, 100 épocas: SSL vs sin SSL (aleatorio) vs supervisado", "",
             "Probe congelado (CV 5×5 en 329 sujetos AD/HC de train+val; test en 84) para SSL y "
             "encoders sin SSL. Supervisado: encoder + capa lineal entrenados end-to-end en los "
             "266 sujetos AD/HC de train, época elegida por AUROC de validación (63 sujetos), "
             "CV = 5 folds entrenados hasta esa época. 3 semillas.", ""]
    deltas: dict = {}
    for m in MODELS:
        ssl, rnd, sup = g["ssl"][m], g["random"][m], g["supervised"][m]
        deltas[m] = {"random": bootstrap_auc(ssl, rnd), "supervised": bootstrap_auc(ssl, sup),
                     "ssl30": bootstrap_auc(ssl, g["ssl30"][m]),
                     "sup_random": bootstrap_auc(sup, rnd)}
        own = [d["identity"]["own"] for d in ssl]
        epochs = [d["best_epoch"] for d in sup]
        lines += [f"## {base.LABEL[m]} ({ENCODER[m]})", "",
                  "| Modelo | AUROC CV | AUROC test | Media3 CV | Media3 test |",
                  "|---|---:|---:|---:|---:|",
                  row("SSL 100 épocas", ssl), row("SSL 30 épocas (ref.)", g["ssl30"][m]),
                  row("Sin SSL, pesos aleatorios", rnd),
                  row("Sin SSL, pesos ImageNet (ref., semilla 1)", g["imagenet"][m]),
                  row(f"Supervisado (épocas elegidas {', '.join(map(str, epochs))})", sup), "",
                  "| Δ AUROC (bootstrap pareado, IC 95 %) | CV | Test |", "|---|---:|---:|",
                  f"| SSL 100 − aleatorio | {fmt(deltas[m]['random'], 'cv')} | "
                  f"{fmt(deltas[m]['random'], 'test')} |",
                  f"| SSL 100 − supervisado | {fmt(deltas[m]['supervised'], 'cv')} | "
                  f"{fmt(deltas[m]['supervised'], 'test')} |",
                  f"| SSL 100 − SSL 30 | {fmt(deltas[m]['ssl30'], 'cv')} | "
                  f"{fmt(deltas[m]['ssl30'], 'test')} |",
                  f"| Supervisado − aleatorio | {fmt(deltas[m]['sup_random'], 'cv')} | "
                  f"{fmt(deltas[m]['sup_random'], 'test')} |", "",
                  f"Identidad al final (147 sujetos no vistos, vistas de la receta): "
                  f"Var_suj/Var_aug {mean([o['between_over_within'] for o in own]):.3g}, "
                  f"rango efectivo {mean([o['effective_rank'] for o in own]):.1f}, "
                  f"coseno mismo − distinto {mean([o['same_subject_cos'] - o['different_subject_cos'] for o in own]):.3f}. "
                  f"Minutos por semilla: {mean([d['minutes'] for d in ssl]):.0f} (SSL), "
                  f"{mean([d['minutes'] for d in sup]):.0f} (supervisado).", ""]
    (RUNS / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return deltas


def main() -> None:
    g = groups()
    deltas = table(g)
    campaign_figures(g)
    import build_report_split361 as rep  # its rcParams: the report's style from here on
    fig_delta(rep, deltas)
    fig_collapse(rep, g)
    fig_byol_objective(rep, g)
    fig_supervised(rep, g)
    for name in ("metricas", "curvas_vicreg", "curvas_byol", "curvas_dinov2", "colapso"):
        shutil.copy2(base.FIGURES / f"{PREFIX}{name}.png", DOCS / f"{PREFIX}{name}.png")
        print("copied", f"{PREFIX}{name}.png")


if __name__ == "__main__":
    main()
