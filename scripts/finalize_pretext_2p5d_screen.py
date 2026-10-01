"""Evaluate the 2.5D GPU screening campaign and render its report.

Compares VICReg+queue, BYOL-GroupNorm, MAE, and MAE+VICReg -- all trained on
the same 2.5D axial stack, the tuned intermediate augmentation, and the same
number of optimizer steps -- against their random-backbone controls, and
surfaces the representation-health trend (effective rank, positive-vs-random
similarity, session/neighbor retrieval, age association) recorded during
training so a dropping SSL loss that is actually collapsing does not get
mistaken for progress.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path

from plot_experiments import epoch_series

ROOT = Path(__file__).resolve().parent.parent
CAMPAIGN = ROOT / "campaign_2p5d_screen"
METHODS = ("vicreg_queue", "byol_groupnorm", "mae", "mae_vicreg")


def completed_experiments() -> dict[str, Path]:
    """Read status.log for the latest COMPLETE run of each expected method."""
    paths: dict[str, Path] = {}
    for line in (CAMPAIGN / "status.log").read_text().splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[1] == "COMPLETE":
            paths[parts[2]] = ROOT / parts[3]
    missing = set(METHODS) - set(paths)
    if missing:
        print(f"Still missing/incomplete: {sorted(missing)}")
    return {method: paths[method] for method in METHODS if method in paths}


def last_logged_step(metrics_path: Path) -> int:
    """Return the highest global step Lightning's CSVLogger recorded."""
    steps = [int(row["step"]) for row in csv.DictReader(metrics_path.open(newline=""))
             if row.get("step")]
    return max(steps) if steps else 0


def health_summary(path: Path) -> dict:
    """Collapse the recorded per-checkpoint health history into first/last snapshots."""
    health_path = path / "representation_health.json"
    if not health_path.is_file():
        return {}
    history = json.loads(health_path.read_text())
    if not history:
        return {}
    keys = ("effective_rank", "mean_feature_std", "similarity_gap",
            "session_retrieval_top1", "session_retrieval_chance",
            "neighbor_slice_retrieval_top1", "neighbor_slice_retrieval_chance", "age_r2")
    first, last = history[0], history[-1]
    return {
        "n_checkpoints": len(history),
        "first_step": first.get("global_step"), "last_step": last.get("global_step"),
        **{f"first_{key}": first.get(key) for key in keys},
        **{f"last_{key}": last.get(key) for key in keys},
    }


def main() -> None:
    """Run the frozen probe for each completed method and write results/RESULTADOS.md."""
    paths = completed_experiments()
    if not paths:
        raise RuntimeError("No completed experiments found in campaign_2p5d_screen/status.log")
    python = str(ROOT / ".venv/Scripts/python.exe")
    for path in paths.values():
        probe = path / "ad_hc_subject_probe.json"
        if not probe.is_file():
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "4"
            subprocess.run(
                [python, str(ROOT / "scripts/evaluate_frozen_probe.py"), str(path),
                 "--label-fractions", "0.25,0.5,1.0"],
                cwd=ROOT, env=env, check=True,
            )
    figure_dir = CAMPAIGN / "figures"
    subprocess.run(
        [python, str(ROOT / "scripts/plot_experiments.py"),
         *(str(paths[method]) for method in paths), "--output-dir", str(figure_dir)],
        cwd=ROOT, check=True,
    )
    rows = []
    for method, path in paths.items():
        doc = json.loads((path / "ad_hc_subject_probe.json").read_text())
        trained, random = doc["results"]["trained"], doc["results"]["random"]
        train = epoch_series(path / "metrics.csv", "train/loss_epoch")
        val = epoch_series(path / "metrics.csv", "val/loss")
        rows.append({
            "method": method, "experiment_dir": str(path),
            "train_steps": last_logged_step(path / "metrics.csv"),
            "epochs_logged": len(train),
            "ssl_train_final": train[max(train)] if train else None,
            "ssl_val_final": val[max(val)] if val else None,
            "health": health_summary(path),
            "train_subjects": trained["train_subjects"], "val_subjects": trained["val_subjects"],
            "test_subjects": trained["test_subjects"],
            "trained_test_auc": trained["test_auc"], "random_test_auc": random["test_auc"],
            "delta_test_auc": doc["comparison"]["delta_test_auc"],
            "delta_auc_95ci": doc["comparison"]["delta_test_auc_bootstrap_95ci"],
            "bootstrap_probability_delta_positive":
                doc["comparison"]["bootstrap_probability_delta_positive"],
            "trained_test_log_loss": trained["test_log_loss"],
            "random_test_log_loss": random["test_log_loss"],
            "trained_test_balanced_accuracy": trained["test_balanced_accuracy"],
            "random_test_balanced_accuracy": random["test_balanced_accuracy"],
            "trained_embedding_effective_rank": trained["embedding_effective_rank"],
            "random_embedding_effective_rank": random["embedding_effective_rank"],
            "label_efficiency": doc.get("label_efficiency", {}).get("trained", {}),
        })
    report = {
        "protocol": "representative_oasis3.csv pilot cohort (68 train / 14 val / 16 test "
                    "OASIS3 subjects), 2.5D axial 5-slice stacks, mri_2p5d_intermediate "
                    "augmentation, update-matched GPU screening (max_steps in trainer=gpu_screen); "
                    "frozen encoder, mean embedding per subject, train-only fit, validation-chosen "
                    "C, test evaluated once. This is a fast screening pass on a small cohort, not "
                    "the full-population concluding evaluation.",
        "results": rows,
    }
    (CAMPAIGN / "results.json").write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        "# Screening 2.5D: VICReg+queue, BYOL-GroupNorm, MAE, MAE+VICReg", "",
        "Cohorte piloto `representative_oasis3.csv` (68/14/16 sujetos train/val/test), pila 2.5D "
        "de 5 cortes axiales adyacentes, augmentation `mri_2p5d_intermediate` (correlación entre "
        "vistas ~0.85, ver `outputs/diagnostics`), mismo número de pasos de optimizador por método "
        "(`trainer=gpu_screen`). Screening rápido en cohorte pequeña, no la evaluación final de "
        "población completa.", "",
        "## Pérdida SSL y salud de representación", "",
        "| Método | Pasos | Train final | Val final | Rango efectivo (inicio→fin) | "
        "Retención sesión (top1 vs azar, fin) | Asociación edad R² (fin) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        h = row["health"]
        rank = (f"{h.get('first_effective_rank', float('nan')):.1f}→"
                f"{h.get('last_effective_rank', float('nan')):.1f}") if h else "n/a"
        session = (f"{h.get('last_session_retrieval_top1', float('nan')):.2f} vs "
                  f"{h.get('last_session_retrieval_chance', float('nan')):.3f}") if h else "n/a"
        age_r2 = f"{h.get('last_age_r2', float('nan')):.2f}" if h else "n/a"
        lines.append(f"| {row['method']} | {row['train_steps']} | "
                     f"{row['ssl_train_final']:.4f} | {row['ssl_val_final']:.4f} | "
                     f"{rank} | {session} | {age_r2} |")
    lines += ["", "## Probe AD/HC en test (cohorte piloto)", "",
              "| Método | AUROC SSL | AUROC aleatorio | Δ AUROC [IC 95%] | P(Δ>0) bootstrap | "
              "Log-loss SSL | Log-loss aleatorio |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lo, hi = row["delta_auc_95ci"]
        lines.append(f"| {row['method']} | {row['trained_test_auc']:.3f} | "
                     f"{row['random_test_auc']:.3f} | "
                     f"{row['delta_test_auc']:+.3f} [{lo:+.3f}, {hi:+.3f}] | "
                     f"{row['bootstrap_probability_delta_positive']:.2f} | "
                     f"{row['trained_test_log_loss']:.3f} | {row['random_test_log_loss']:.3f} |")
    lines += ["", "IC 95%: bootstrap pareado por sujeto, un split y semilla; cohorte de test muy "
              "pequeña (16 sujetos), así que estos intervalos son anchos por diseño -- esto es "
              "una señal de screening, no una conclusión.", "",
              "Figuras: [curvas de aprendizaje](figures/ssl_learning_curves.png).", ""]
    (CAMPAIGN / "RESULTADOS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
