"""Evaluate the 100-epoch GPU campaign (updated backbones) and render its report.

Same evaluation/reporting logic as ``finalize_full50.py``, parameterized for
``campaign_full100_gpu``: a 100-epoch ceiling instead of 50, and backbones
updated to the best 2026-09-20 configurations (VICReg = ConvNeXt-tiny +
ImageNet pretraining + dropout 0.1, BYOL = ResNet-18 + ImageNet pretraining).
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
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

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "campaign_full100_gpu"
METHODS = ("vicreg", "byol", "dinov2")
MAX_EPOCHS = 100


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
        if method not in stopped_methods and len(train) < MAX_EPOCHS:
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
        expected_epochs = (stopped_methods[method]["stopped_after_epochs"]
                           if method in stopped_methods else MAX_EPOCHS)
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
        if method not in stopped_methods and checkpoint["epoch"] != MAX_EPOCHS:
            raise RuntimeError(f"{method} last checkpoint has epoch={checkpoint['epoch']}")
        del checkpoint
        probe = path / ("ad_hc_subject_probe_best.json" if method in stopped_methods else
                        "ad_hc_subject_probe.json")
        if not probe.is_file():
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "2"
            env["MKL_NUM_THREADS"] = "2"
            env["OPENBLAS_NUM_THREADS"] = "2"
            command = [sys.executable,
                       str(ROOT / "scripts/legacy/evaluate_frozen_probe.py"), str(path)]
            if method in stopped_methods:
                command += ["--checkpoint", "best"]
            subprocess.run(command,
                           cwd=ROOT, env=env, check=True)
    figure_dir = CAMPAIGN / "figures"
    subprocess.run([sys.executable,
                    str(ROOT / "scripts/legacy/plot_experiments.py"),
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
            "trained_test_aupr": trained["test_average_precision"],
            "random_test_aupr": random["test_average_precision"],
            "trained_test_f1_macro": trained["test_f1_macro"],
            "random_test_f1_macro": random["test_f1_macro"],
            "trained_embedding_effective_rank": trained["embedding_effective_rank"],
            "random_embedding_effective_rank": random["embedding_effective_rank"],
            "trained_embedding_mean_feature_std": trained["embedding_mean_feature_std"],
            "random_embedding_mean_feature_std": random["embedding_mean_feature_std"],
        })
    report = {"protocol": "OASIS3 original LMDB; 981 subjects; group-stratified, subject-disjoint "
              "685/146/150 split (same manifest as campaign_full50); SSL sees one randomly "
              "selected session and axial slice per train subject per epoch; 1226 available "
              "train sessions; fixed primary session and central slice for validation and "
              "frozen linear probe; up to 100 full epochs with fixed SSL-validation early "
              "stopping; VICReg = ConvNeXt-tiny + ImageNet pretraining + projector dropout "
              "0.1; BYOL = ResNet-18 + ImageNet pretraining; DINOv2 = ViT-Small patch 60 "
              "(unchanged); checkpoint chosen without labels; probe C chosen on validation "
              "AUROC; test evaluated once.",
              "early_stopping": stopping,
              "results": rows}
    (CAMPAIGN / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
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
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), dpi=180)
    for ax, key_prefix, ylabel, title in [
        (axes[0], "test_aupr", "AUPR AD/HC en test (mayor es mejor)",
         "AUPR: precisión-cobertura, sensible a la clase minoritaria (AD)"),
        (axes[1], "test_f1_macro", "F1 macro AD/HC en test (mayor es mejor)",
         "F1 macro: promedio simple del F1 de AD y de HC"),
    ]:
        for offset, kind, label, color in [
            (-width / 2, "trained", "SSL + probe", "#256b8c"),
            (width / 2, "random", "Aleatorio + probe", "#9caeb8"),
        ]:
            ax.bar(x + offset, [row[f"{kind}_{key_prefix}"] for row in rows], width,
                   label=label, color=color)
        ax.set_xticks(x, [method.upper() if method != "dinov2" else "DINOv2"
                          for method in METHODS])
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("AUPR y F1 macro: dos vistas del mismo probe, distintas de AUROC")
    fig.tight_layout()
    fig.savefig(figure_dir / "ad_hc_probe_aupr_f1.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(18, 4.2), dpi=180)
    fig.patch.set_facecolor("#f5f8fb")
    ax.axis("off")
    ax.set_title("OASIS3 · Probe lineal AD/HC en sujetos no vistos (100 épocas, backbones actualizados)",
                 fontsize=15, weight="bold", color="#16384f", pad=24)
    values = []
    for row in rows:
        low, high = row["delta_auc_95ci"]
        values.append([
            row["method"].upper() if row["method"] != "dinov2" else "DINOv2",
            str(row["epochs"]),
            f"{row['trained_test_auc']:.3f}",
            f"{row['random_test_auc']:.3f}",
            f"{row['delta_test_auc']:+.3f}\n[{low:+.3f}, {high:+.3f}]",
            f"{row['trained_test_aupr']:.3f} / {row['random_test_aupr']:.3f}",
            f"{row['trained_test_f1_macro']:.3f} / {row['random_test_f1_macro']:.3f}",
            f"{row['trained_test_log_loss']:.3f} / {row['random_test_log_loss']:.3f}",
        ])
    table = ax.table(cellText=values,
                     colLabels=["Modelo", "Épocas", "AUROC\nSSL", "AUROC\naleatorio",
                                "Δ AUROC\n[IC 95 %]", "AUPR\nSSL / aleatorio",
                                "F1 macro\nSSL / aleatorio", "Log-loss\nSSL / aleatorio"],
                     colWidths=[0.09, 0.06, 0.09, 0.11, 0.17, 0.16, 0.16, 0.16],
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 3.1)
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
    lines = ["# OASIS3: pretraining SSL extenso, máximo 100 épocas (backbones actualizados, GPU)", "",
             "981 sujetos OASIS3 del LMDB original; split fijo y disjunto por sujeto (idéntico "
             "al de campaign_full50): 685 train, 146 validación, 150 test. SSL utiliza las "
             "1.226 sesiones disponibles en train, sorteando una sesión y un corte por sujeto "
             "en cada época. Validación y test usan una sesión y un corte fijos. VICReg usa "
             "ConvNeXt-tiny preentrenado en ImageNet con dropout=0.1 en el proyector; BYOL usa "
             "ResNet-18 preentrenado en ImageNet; DINOv2 mantiene ViT-Small (patch 60) sin "
             "cambios. Se aplicó early stopping sólo a modelos con 10 épocas sin mejorar "
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
              "Un descenso de pérdida SSL por sí solo no demuestra utilidad clínica. "
              "**Nota sobre el control aleatorio**: `evaluate_frozen_probe.py` fuerza "
              "`pretrained=false` al construir el encoder de control, aunque el "
              "checkpoint entrenado use un backbone preentrenado en ImageNet (VICReg, "
              "BYOL). Esto corrige el problema detectado en la sesión del 20-09, donde "
              "el control heredaba los pesos de ImageNet y dejaba de ser un encoder "
              "realmente sin entrenar; aquí el control **sí** es aleatorio real en las "
              "tres filas.", "",
              "## AUPR y F1 macro", "",
              "AUROC resume qué tan bien el probe ordena a los sujetos por riesgo, pero "
              "no depende de un umbral. Como AD es la clase minoritaria en test (31 de "
              "134), se añaden dos métricas más sensibles a ese desbalance: **AUPR** "
              "(área bajo la curva precisión-cobertura, sube si el probe encuentra los "
              "AD reales sin muchos falsos positivos) y **F1 macro** (promedio simple "
              "del F1 de AD y de HC con el umbral por defecto 0,5, penalizando un "
              "modelo que solo acierte en la clase mayoritaria).", "",
              "| Método | AUPR SSL | AUPR aleatorio | F1 macro SSL | F1 macro aleatorio |",
              "|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['method']} | {row['trained_test_aupr']:.3f} | "
                     f"{row['random_test_aupr']:.3f} | "
                     f"{row['trained_test_f1_macro']:.3f} | "
                     f"{row['random_test_f1_macro']:.3f} |")
    lines += ["", "Figuras: [AUPR y F1](figures/ad_hc_probe_aupr_f1.png). Estas dos "
              "métricas no tienen intervalo de confianza calculado en esta campaña "
              "(solo el Δ AUROC lo tiene, vía bootstrap pareado); trátalas como punto "
              "de referencia adicional, no como una prueba estadística.", "",
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
    lines += ["", "Estos resultados son de un único split y semilla, y no prueban una "
              "mejora clínica generalizable.", "",
              "Figuras: [curvas SSL](figures/ssl_learning_curves.png), "
              "[AUROC](figures/ad_hc_subject_auc.png), "
              "[Δ AUROC](figures/ad_hc_delta_auc.png), "
              "[log-loss](figures/ad_hc_probe_log_loss.png), "
              "[AUPR y F1 macro](figures/ad_hc_probe_aupr_f1.png) y "
              "[tabla visual](figures/probe_vs_baseline_table.png).", "",
              "Split y manifiesto usados: ver [SPLIT.md](SPLIT.md).", "",
              "Comandos reproducibles: `scripts/legacy/run_full100_gpu.sh` y "
              "`scripts/legacy/finalize_full100_gpu.py`.", ""]
    (CAMPAIGN / "RESULTADOS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
