"""Summarize Optuna SSL-only studies and 100-epoch refits without probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
from plot_experiments import epoch_series

METHODS = ("vicreg", "byol", "dinov2")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-dir", type=Path, default=Path("tuning_pretext"))
    args = parser.parse_args()
    figures = args.study_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    results = []
    fig_loss, axes_loss = plt.subplots(3, 1, figsize=(10, 11), squeeze=False)
    fig_trials, axes_trials = plt.subplots(3, 1, figsize=(10, 9), squeeze=False)
    for index, method in enumerate(METHODS):
        candidates = list(args.study_dir.glob(f"{method}_ssl_loss_*_best.json"))
        if len(candidates) != 1:
            raise RuntimeError(f"Missing or ambiguous study for {method}: {candidates}")
        study_info = json.loads(candidates[0].read_text(encoding="utf-8"))
        refit = json.loads((args.study_dir / f"{method}_refit.json").read_text(
            encoding="utf-8"))
        if refit["method"] != method or not refit["no_probe_used"]:
            raise RuntimeError(f"Invalid SSL-only refit for {method}")
        score0 = study_info["baseline_score"]
        score1 = study_info["best_score"]
        if score0 is None or score1 is None:
            raise RuntimeError(f"Missing complete 50-epoch scores for {method}")
        checkpoint_dir = Path(refit["tuned_experiment"]) / "checkpoints"
        best_checkpoints = list(checkpoint_dir.glob("best-loss-epoch=*.ckpt"))
        last_checkpoint = checkpoint_dir / "last.ckpt"
        if len(best_checkpoints) != 1 or not last_checkpoint.is_file():
            raise RuntimeError(f"Missing best or last 100-epoch checkpoint for {method}")
        row = {"method": method, "study": study_info["study"],
               "selection_epochs": study_info["epochs"],
               "trials_requested": study_info["trials_requested"],
               "trials_resolved": study_info.get("trials_resolved",
                                                 study_info["trials_requested"]),
               "early_stopped": study_info.get("early_stopped", False),
               "trials_complete": study_info["trials_complete"],
               "trials_pruned": study_info["trials_pruned"],
               "baseline_50ep_final5_val_loss": score0,
               "best_50ep_final5_val_loss": score1,
               "delta_50ep_val_loss": score1 - score0,
               "best_trial": study_info["best_trial"],
               "best_params": study_info["best_params"],
               "baseline_100ep_final5_val_loss": refit["baseline"]["final5_val_loss"],
               "tuned_100ep_final5_val_loss": refit["tuned"]["final5_val_loss"],
               "tuned_best_val_loss": refit["tuned"]["best_val_loss"],
               "tuned_best_val_epoch": refit["tuned"]["best_val_epoch"],
               "delta_100ep_val_loss": refit["delta_final5_val_loss"],
               "tuned_experiment": refit["tuned_experiment"],
               "best_checkpoint": str(best_checkpoints[0]),
               "last_checkpoint": str(last_checkpoint),
               "reused_original_baseline": refit["reused_original_baseline"]}
        results.append(row)
        ax = axes_loss[index, 0]
        for label, key, color in (("original", "baseline_experiment", "#7f8c8d"),
                                  ("ajustado", "tuned_experiment", "#21618c")):
            values = epoch_series(Path(refit[key]) / "metrics.csv", "val/loss")
            ax.plot([epoch + 1 for epoch in sorted(values)],
                    [values[epoch] for epoch in sorted(values)],
                    label=label, color=color, linewidth=1.7)
        ax.set(title=method.upper() if method != "dinov2" else "DINOv2 compacto",
               xlabel="Época", ylabel="Pérdida SSL de validación")
        ax.grid(alpha=0.2)
        ax.legend()
        db = candidates[0].with_name(study_info["study"] + ".sqlite3")
        study = optuna.load_study(study_name=study_info["study"],
                                  storage=f"sqlite:///{db.resolve()}")
        ax = axes_trials[index, 0]
        pruned_labeled = False
        complete_labeled = False
        best_labeled = False
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                is_best = trial.number == study_info["best_trial"]
                ax.scatter(trial.number, trial.value,
                           color="#27ae60" if is_best
                           else "#21618c", s=65 if trial.number == study_info["best_trial"] else 30)
                # Give each point type one legend entry without duplicating labels.
                if is_best and not best_labeled:
                    ax.scatter([], [], color="#27ae60", s=65, label="mejor completo")
                    best_labeled = True
                elif not is_best and not complete_labeled:
                    ax.scatter([], [], color="#21618c", s=30, label="completo")
                    complete_labeled = True
            elif trial.state == optuna.trial.TrialState.PRUNED and trial.intermediate_values:
                step = max(trial.intermediate_values)
                ax.scatter(trial.number, trial.intermediate_values[step],
                           color="#aab7b8", marker="x",
                           label="podado (última media móvil)" if not pruned_labeled else None)
                pruned_labeled = True
        ax.axhline(score0, color="#c0392b", linestyle="--", linewidth=1,
                   label="referencia 50 épocas")
        ax.set(title=method.upper() if method != "dinov2" else "DINOv2 compacto",
               xlabel="Ensayo Optuna", ylabel="Pérdida SSL val (media de 5 épocas)")
        ax.grid(alpha=0.2)
        ax.legend()
    fig_loss.tight_layout()
    fig_loss.savefig(figures / "refit_100ep_val_loss.png", dpi=180)
    plt.close(fig_loss)
    fig_trials.tight_layout()
    fig_trials.savefig(figures / "optuna_trials.png", dpi=180)
    plt.close(fig_trials)
    output = {"protocol": "OASIS3 LMDB; same subject train/val; fixed native SSL loss; "
                          "Optuna TPE plus median pruning; only SSL validation loss; no probe; "
                          "clinical labels used only for pre-existing split stratification",
              "cohort": "representative_oasis3.csv; 68 train, 14 validation subjects",
              "results": results,
              "limitations": "Method losses have different scales; optimize within each method only. "
                             "One split and seed; lower SSL loss does not guarantee better "
                             "downstream AD/HC representations."}
    (args.study_dir / "results.json").write_text(json.dumps(output, indent=2) + "\n",
                                                 encoding="utf-8")
    lines = ["# Tuning de la tarea de pretexto en OASIS3", "",
             "Optuna TPE, hasta 12 ensayos por método y 50 épocas por ensayo; "
             "VICReg y BYOL se detuvieron tras nueve ensayos resueltos "
             "a petición del usuario. Poda mediana después "
             "de 20 épocas. Selección: media de `val/loss` SSL en las últimas cinco "
             "épocas. Las etiquetas AD/HC sólo estratificaron el split previo; "
             "no se usaron en la selección, ni hubo probe o test. Se mantuvo fija la "
             "definición de cada pérdida; se buscaron tasa de aprendizaje, "
             "weight decay, recorte mínimo y ruido de augmentación.", "",
             "| Método | Resueltos/planificados | Completos/podados | SSL val 50 ep: original → mejor | "
             "SSL val 100 ep: original → ajustado | Δ 100 ep |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in results:
        lines.append(f"| {row['method']} | {row['trials_resolved']}/{row['trials_requested']} "
                     f"| {row['trials_complete']}/{row['trials_pruned']} "
                     f"| {row['baseline_50ep_final5_val_loss']:.4f} → "
                     f"{row['best_50ep_final5_val_loss']:.4f} "
                     f"| {row['baseline_100ep_final5_val_loss']:.4f} → "
                     f"{row['tuned_100ep_final5_val_loss']:.4f} "
                     f"| {row['delta_100ep_val_loss']:+.4f} |")
    lines += ["", "Parámetros elegidos (solo para entrenamiento; validación con transformaciones fijas):", "",
              "| Método | Learning rate | Weight decay | Escala mínima del recorte | Desviación del ruido |",
              "|---|---:|---:|---:|---:|"]
    for row in results:
        params = row["best_params"]
        lines.append(f"| {row['method']} | {params['lr']:.3g} | "
                     f"{params['weight_decay']:.3g} | "
                     f"{params['crop_scale_min']:.3f} | {params['noise_std']:.3f} |")
    lines += ["", "Checkpoints: `best-loss` conserva la época con menor pérdida "
              "individual de validación; `last.ckpt` conserva la época 100. "
              "La comparación de la tabla usa la media de las últimas cinco épocas.", ""]
    for row in results:
        lines.append(f"- {row['method']}: mejor época {row['tuned_best_val_epoch'] + 1} "
                     f"(`val/loss` {row['tuned_best_val_loss']:.4f}); "
                     f"`{row['best_checkpoint']}` y `{row['last_checkpoint']}`.")
    lines += ["", "Los números de pérdida de métodos distintos no son comparables "
              "entre sí. Un valor menor mide mejor ajuste del objetivo SSL fijo, "
              "pero no demuestra un mejor probe AD/HC. La validación tiene sólo "
              "14 sujetos y se usó una semilla.", "",
              "Gráficos: `figures/optuna_trials.png` y "
              "`figures/refit_100ep_val_loss.png`. Configuraciones y rutas exactas "
              "de los checkpoints `best` y `last` en `results.json` y los "
              "`*_best.json`.", ""]
    (args.study_dir / "RESULTADOS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
