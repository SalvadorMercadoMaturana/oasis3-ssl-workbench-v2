"""Evaluate the fixed 50-epoch full-OASIS3 campaign and render its report."""

from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from plot_experiments import epoch_series
from sklearn.metrics import log_loss

from t5_ssl.data.oasis3_lmdb_datamodule import clinical_rows
from t5_ssl.data.paths import DEFAULT_CSV

ROOT = Path(__file__).resolve().parent.parent
CAMPAIGN = ROOT / "campaign_full50"
METHODS = ("vicreg", "byol", "dinov2")


def last_significant_improvement(val: dict[int, float], min_delta: float) -> int:
    first = min(val)
    best, best_epoch = val[first], first + 1
    for index in sorted(val):
        if val[index] < best - min_delta:
            best, best_epoch = val[index], index + 1
    return best_epoch


def completed_experiments() -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for line in (CAMPAIGN / "status.log").read_text().splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[1] in ("COMPLETE", "EARLY_STOP"):
            if parts[2] in paths:
                raise RuntimeError(f"Duplicate completed method {parts[2]}")
            paths[parts[2]] = ROOT / parts[3]
    if set(paths) != set(METHODS):
        raise RuntimeError(f"Campaign incomplete: {sorted(paths)}")
    return paths


def train_prior(path: Path) -> float:
    with (path / "subject_splits.csv").open(newline="") as stream:
        train_ids = {row["subject"] for row in csv.DictReader(stream)
                     if row["split"] == "train"}
    groups: dict[str, set[str]] = {}
    for row in clinical_rows(DEFAULT_CSV):
        groups.setdefault(row["subject"], set()).add(row["group"])
    labels = [next(iter(values)) for subject, values in groups.items()
              if subject in train_ids and len(values) == 1 and
              next(iter(values)) in ("AD", "HC")]
    return labels.count("AD") / len(labels)


def main() -> None:
    paths = completed_experiments()
    stopping_path = CAMPAIGN / "early_stopping.json"
    stopping = json.loads(stopping_path.read_text()) if stopping_path.is_file() else {}
    stopped_methods = stopping.get("methods", {})
    for method, path in paths.items():
        train = epoch_series(path / "metrics.csv", "train/loss_epoch")
        val = epoch_series(path / "metrics.csv", "val/loss")
        if method not in stopped_methods and len(train) < 50:
            cfg = OmegaConf.load(path / "config.yaml")
            configured_patience = cfg.trainer.get("early_stopping_patience", None)
            if configured_patience is None:
                raise RuntimeError(f"{method} ended early without a recorded stopping rule")
            best_epoch = min(val, key=val.get) + 1
            stopped_methods[method] = {"stopped_after_epochs": len(train),
                                       "best_ssl_epoch": best_epoch,
                                       "best_val_loss": min(val.values())}
            stopping.setdefault("patience", int(configured_patience))
            stopping.setdefault("min_delta", float(cfg.trainer.get(
                "early_stopping_min_delta", 0.0)))
        expected_epochs = stopped_methods[method]["stopped_after_epochs"] if method in stopped_methods else 50
        if set(train) != set(range(expected_epochs)) or set(val) != set(range(expected_epochs)):
            raise RuntimeError(f"{method} does not contain {expected_epochs} full epochs")
        if method in stopped_methods:
            best_epoch = min(val, key=val.get) + 1
            reference_epoch = last_significant_improvement(
                val, float(stopping.get("min_delta", 0.0)))
            if (best_epoch != stopped_methods[method]["best_ssl_epoch"] or
                    expected_epochs - reference_epoch < stopping["patience"]):
                raise RuntimeError(f"{method} does not satisfy the early stopping rule")
            matches = list((path / "checkpoints").glob("best-loss-epoch=*.ckpt"))
            if len(matches) != 1:
                raise RuntimeError(f"Expected one best-loss checkpoint for {method}")
            checkpoint_path = matches[0]
        else:
            checkpoint_path = path / "checkpoints" / "last.ckpt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if method not in stopped_methods and checkpoint["epoch"] != 50:
            raise RuntimeError(f"{method} last checkpoint has epoch={checkpoint['epoch']}")
        del checkpoint
        probe = path / ("ad_hc_subject_probe_best.json" if method in stopped_methods else
                        "ad_hc_subject_probe.json")
        if not probe.is_file():
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "2"
            env["MKL_NUM_THREADS"] = "2"
            env["OPENBLAS_NUM_THREADS"] = "2"
            command = [str(ROOT / ".venv/bin/python"),
                       str(ROOT / "scripts/evaluate_frozen_probe.py"), str(path)]
            if method in stopped_methods:
                command += ["--checkpoint", "best"]
            subprocess.run(command,
                           cwd=ROOT, env=env, check=True)
    figure_dir = CAMPAIGN / "figures"
    subprocess.run([str(ROOT / ".venv/bin/python"),
                    str(ROOT / "scripts/plot_experiments.py"),
                    *(str(paths[method]) for method in METHODS),
                    "--output-dir", str(figure_dir)], cwd=ROOT, check=True)
    rows = []
    common_test: tuple[list[str], list[int]] | None = None
    for method, path in paths.items():
        probe_name = ("ad_hc_subject_probe_best.json" if method in stopped_methods else
                      "ad_hc_subject_probe.json")
        doc = json.loads((path / probe_name).read_text())
        trained, random = doc["results"]["trained"], doc["results"]["random"]
        prediction = trained["test_predictions"]
        subjects = [item["subject"] for item in prediction]
        labels = [item["AD"] for item in prediction]
        if common_test is not None and common_test != (subjects, labels):
            raise RuntimeError("Methods did not evaluate the same test subjects")
        common_test = (subjects, labels)
        prior = train_prior(path)
        baseline_loss = float(log_loss(labels, np.full(len(labels), prior), labels=[0, 1]))
        train = epoch_series(path / "metrics.csv", "train/loss_epoch")
        val = epoch_series(path / "metrics.csv", "val/loss")
        rows.append({
            "method": method, "experiment_dir": str(path),
            "epochs": len(train), "checkpoint_selection": (
                "mejor val SSL (early stopping)" if method in stopped_methods else
                "última época"),
            "ssl_train_final": train[max(train)],
            "ssl_val_final": val[max(val)], "ssl_val_best": min(val.values()),
            "ssl_val_best_epoch": min(val, key=val.get) + 1,
            "train_subjects": len({r["subject"] for r in csv.DictReader(
                (path / "subject_splits.csv").open()) if r["split"] == "train"}),
            "probe_train_subjects": trained["train_subjects"],
            "probe_val_subjects": trained["val_subjects"],
            "probe_test_subjects": trained["test_subjects"],
            "test_ad": sum(labels), "test_hc": len(labels) - sum(labels),
            "trained_test_auc": trained["test_auc"],
            "random_test_auc": random["test_auc"],
            "delta_test_auc": doc["comparison"]["delta_test_auc"],
            "delta_auc_95ci": doc["comparison"]["delta_test_auc_bootstrap_95ci"],
            "trained_test_log_loss": trained["test_log_loss"],
            "random_test_log_loss": random["test_log_loss"],
            "prior_test_log_loss": baseline_loss,
            "trained_test_balanced_accuracy": trained["test_balanced_accuracy"],
            "random_test_balanced_accuracy": random["test_balanced_accuracy"],
            "trained_embedding_effective_rank": trained["embedding_effective_rank"],
            "random_embedding_effective_rank": random["embedding_effective_rank"],
            "trained_embedding_mean_feature_std": trained["embedding_mean_feature_std"],
            "random_embedding_mean_feature_std": random["embedding_mean_feature_std"],
        })
    report = {"protocol": "OASIS3 original LMDB; 981 subjects; group-stratified, subject-disjoint "
              "685/146/150 split; SSL sees one randomly selected session and axial slice per "
              "train subject per epoch; 1226 available train sessions; fixed primary session "
              "and central slice for validation and frozen linear probe; up to 50 full epochs "
              "with fixed SSL-validation early stopping; native model/optimizer settings; "
              "checkpoint chosen without labels; "
              "probe C chosen on validation AUROC; test evaluated once.",
              "early_stopping": stopping,
              "results": rows}
    (CAMPAIGN / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=180)
    x = np.arange(len(rows))
    width = 0.24
    for offset, key, title, color in [
        (-width, "trained_test_log_loss", "SSL + probe", "#256b8c"),
        (0, "random_test_log_loss", "Aleatorio + probe", "#9caeb8"),
        (width, "prior_test_log_loss", "Prior AD", "#e3ae65"),
    ]:
        ax.bar(x + offset, [row[key] for row in rows], width,
               label=title, color=color)
    ax.set_xticks(x, [method.upper() if method != "dinov2" else "DINOv2"
                      for method in METHODS])
    ax.set_ylabel("Log-loss AD/HC en test (menor es mejor)")
    ax.set_title("Encoder congelado + probe lineal frente a controles")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "ad_hc_probe_log_loss.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(14.5, 3.8), dpi=180)
    fig.patch.set_facecolor("#f5f8fb")
    ax.axis("off")
    ax.set_title("OASIS3 · Probe lineal AD/HC en sujetos no vistos",
                 fontsize=17, weight="bold", color="#16384f", pad=24)
    values = []
    for row in rows:
        low, high = row["delta_auc_95ci"]
        values.append([
            row["method"].upper() if row["method"] != "dinov2" else "DINOv2",
            str(row["epochs"]),
            f"{row['trained_test_auc']:.3f}",
            f"{row['random_test_auc']:.3f}",
            f"{row['delta_test_auc']:+.3f} [{low:+.3f}, {high:+.3f}]",
            f"{row['trained_test_log_loss']:.3f} / {row['random_test_log_loss']:.3f}",
        ])
    table = ax.table(cellText=values,
                     colLabels=["Modelo", "Épocas", "AUROC SSL", "AUROC aleatorio",
                                "Δ AUROC [IC 95 %]", "Log-loss SSL / aleatorio"],
                     colWidths=[0.13, 0.09, 0.13, 0.16, 0.25, 0.24],
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.35)
    for (row_index, col_index), cell in table.get_celld().items():
        cell.set_edgecolor("#d8e3eb")
        if row_index == 0:
            cell.set_facecolor("#215b79")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#eaf2f7" if row_index % 2 else "#ffffff")
            if col_index == 0:
                cell.set_text_props(weight="bold", color="#16384f")
    fig.text(0.5, 0.09, "Test: 134 sujetos AD/HC · mayor AUROC es mejor · menor log-loss es mejor",
             ha="center", fontsize=10, color="#526b7d")
    fig.savefig(figure_dir / "probe_vs_baseline_table.png", bbox_inches="tight")
    plt.close(fig)
    lines = ["# OASIS3: pretraining SSL extenso, máximo 50 épocas", "",
             "981 sujetos OASIS3 del LMDB original; split fijo y disjunto por sujeto: "
             "685 train, 146 validación, 150 test. SSL utiliza las 1.226 sesiones disponibles "
             "en train, sorteando una sesión y un corte por sujeto en cada época. Validación "
             "y test usan una sesión y un corte fijos. Se mantuvieron las configuraciones "
             "nativas de los tres modelos, sin seleccionar hiperparámetros con etiquetas. "
             "Se aplicó early stopping sólo a modelos con 10 épocas sin mejorar "
             "`val/loss` SSL al menos 0,001; para ellos se evalúa el mejor checkpoint SSL.", "",
             "## Pérdida SSL", "",
             "| Método | Épocas | Checkpoint probe | Train final | Validación final | Mejor validación (época) |",
             "|---|---:|---|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['method']} | {row['epochs']} | {row['checkpoint_selection']} | "
                     f"{row['ssl_train_final']:.4f} | "
                     f"{row['ssl_val_final']:.4f} | {row['ssl_val_best']:.4f} "
                     f"({row['ssl_val_best_epoch']}) |")
    lines += ["", "Las escalas de pérdida SSL difieren entre métodos y no se comparan "
              "directamente.", "", "## Probe AD/HC por sujeto en test", "",
              "Encoder congelado; regresión logística y normalización ajustadas en train "
              "(615 sujetos AD/HC). C elegido por AUROC en validación (131 sujetos). "
              "El test contiene 134 sujetos AD/HC (31 AD, 103 HC). El control aleatorio "
              "también ajusta un probe; el prior AD no usa imagen.", "",
              "| Método | AUROC SSL | AUROC aleatorio | Δ AUROC [IC 95 %] | "
              "Log-loss SSL | Log-loss aleatorio | Log-loss prior |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lo, hi = row["delta_auc_95ci"]
        lines.append(f"| {row['method']} | {row['trained_test_auc']:.3f} | "
                     f"{row['random_test_auc']:.3f} | "
                     f"{row['delta_test_auc']:+.3f} [{lo:+.3f}, {hi:+.3f}] | "
                     f"{row['trained_test_log_loss']:.3f} | "
                     f"{row['random_test_log_loss']:.3f} | "
                     f"{row['prior_test_log_loss']:.3f} |")
    lines += ["", "IC 95 %: bootstrap pareado por sujeto, condicionado a un split y una "
              "semilla. AUROC 0,5 representa ranking aleatorio sin entrenar el probe; "
              "un encoder aleatorio con un probe entrenado puede superar 0,5. "
              "Un descenso de pérdida SSL por sí solo no demuestra utilidad clínica.", "",
              "## Dispersión de embeddings del train para el probe", "",
              "Rango efectivo espectral y desviación media por dimensión. Un rango muy "
              "bajo respecto al control puede indicar concentración o colapso parcial "
              "de las representaciones.", "",
              "| Método | Rango efectivo SSL | Rango efectivo aleatorio | "
              "Desv. media SSL | Desv. media aleatorio |",
              "|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['method']} | {row['trained_embedding_effective_rank']:.1f} | "
                     f"{row['random_embedding_effective_rank']:.1f} | "
                     f"{row['trained_embedding_mean_feature_std']:.4f} | "
                     f"{row['random_embedding_mean_feature_std']:.4f} |")
    lines += ["", "BYOL y DINOv2 concentran su variación en muy pocas direcciones "
              "y no superan al control aleatorio en AUROC. VICReg conserva más "
              "direcciones y mejora AUROC en este test, pero su IC de Δ AUROC incluye "
              "cero. El prior AD logra menor log-loss que todos los probes; éstos "
              "usan ponderación de clases y escogen C por AUROC, no por calibración "
              "de probabilidades. Estos resultados son de un único split y semilla, "
              "y no prueban una mejora clínica generalizable.", "",
              "Figuras: [curvas SSL](figures/ssl_learning_curves.png), "
              "[AUROC](figures/ad_hc_subject_auc.png), "
              "[Δ AUROC](figures/ad_hc_delta_auc.png), "
              "[log-loss](figures/ad_hc_probe_log_loss.png) y "
              "[tabla visual](figures/probe_vs_baseline_table.png).", "",
              "Comandos reproducibles en [README](../README.md).", ""]
    (CAMPAIGN / "RESULTADOS.md").write_text("\n".join(lines))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
