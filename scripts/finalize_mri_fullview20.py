"""Evaluate and plot the interrupted full-field OASIS3 SSL campaign."""

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
from plot_experiments import epoch_series
from sklearn.metrics import log_loss

from t5_ssl.data.oasis3_lmdb_datamodule import clinical_rows
from t5_ssl.data.paths import DEFAULT_CSV

ROOT = Path(__file__).resolve().parent.parent
CAMPAIGN = ROOT / "campaign_mri_fullview20"
METHODS = ("vicreg", "byol", "dinov2")


def completed_experiments() -> dict[str, Path]:
    """Read the actual, user-directed stop point for each method."""
    manifest = json.loads((CAMPAIGN / "experiment_manifest.json").read_text())
    return {method: ROOT / manifest[method]["original"] for method in METHODS}


def train_prior(path: Path) -> float:
    """Estimate the AD prior from the training subjects alone."""
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


def probe(path: Path) -> dict:
    """Score the validation-selected SSL checkpoint exactly once on test."""
    output = path / "ad_hc_subject_probe_best.json"
    if not output.is_file():
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
        subprocess.run([
            str(ROOT / ".venv/bin/python"),
            str(ROOT / "scripts/evaluate_frozen_probe.py"),
            str(path), "--checkpoint", "best",
        ], cwd=ROOT, env=env, check=True)
    return json.loads(output.read_text())


def make_extra_figures(rows: list[dict], figure_dir: Path) -> None:
    """Render the two figures produced by the historical full50 finalizer."""
    x = np.arange(len(rows))
    width = 0.24
    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=180)
    for offset, key, label, color in (
        (-width, "trained_test_log_loss", "SSL + probe", "#256b8c"),
        (0, "random_test_log_loss", "Aleatorio + probe", "#9caeb8"),
        (width, "prior_test_log_loss", "Prior AD", "#e3ae65"),
    ):
        ax.bar(x + offset, [row[key] for row in rows], width,
               label=label, color=color)
    ax.set_xticks(x, [m.upper() if m != "dinov2" else "DINOv2" for m in METHODS])
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
    ax.set_title("OASIS3 · Probe AD/HC · vistas anatómicas",
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
    table = ax.table(
        cellText=values,
        colLabels=["Modelo", "Épocas", "AUROC SSL", "AUROC aleatorio",
                   "Δ AUROC [IC 95 %]", "Log-loss SSL / aleatorio"],
        colWidths=[0.13, 0.09, 0.13, 0.16, 0.25, 0.24],
        cellLoc="center", loc="center",
    )
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
    fig.text(0.5, 0.09, "Test: 134 sujetos AD/HC · mayor AUROC es mejor",
             ha="center", fontsize=10, color="#526b7d")
    fig.savefig(figure_dir / "probe_vs_baseline_table.png", bbox_inches="tight")
    plt.close(fig)


def report_markdown(rows: list[dict]) -> str:
    """Summarize the experiment at each method's actual stopping epoch."""
    lines = [
        "# OASIS3: vistas anatómicas, entrenamiento detenido por petición", "",
        "Mismos 981 sujetos y manifest que `campaign_full50`; split fijo por sujeto "
        "685/146/150. Entrenamiento SSL con una sesión y un corte axial aleatorios "
        "por sujeto y época. Dos vistas globales de campo completo, rotación ±5°, "
        "intensidad ×[0,95, 1,05] y ruido σ=0,01. Sin crop, flip ni blur; "
        "DINOv2 sin vistas locales. DINOv2 completó 20 épocas antes del cambio "
        "de límite; VICReg se detuvo tras 15 y BYOL tras 12, por petición. "
        "Checkpoint elegido por la menor pérdida SSL de validación, sin etiquetas.", "",
        "Se intentó reanudar BYOL desde su mejor checkpoint para llegar a 15, "
        "pero se detuvo antes de terminar una época adicional cuando se pidió "
        "finalizarlo inmediatamente. Sus resultados corresponden sólo a la "
        "corrida original de 12 épocas.", "",
        "El corte LMDB de referencia mide 97×115 píxeles y la entrada del modelo "
        "sigue redimensionándose a 96×96 (64×64 para DINOv2), igual que la "
        "evaluación histórica. Se conserva todo el campo de visión, pero el "
        "resize cuadrado no conserva la relación de aspecto física.", "",
        "La política conserva las regiones anatómicas, motivada por "
        "[PD-SIM](https://doi.org/10.1016/j.bspc.2024.106572), que advierte "
        "que el crop aleatorio puede omitir regiones relevantes, y por "
        "[Turrisi et al.](https://doi.org/10.3389/fncom.2024.1360095), "
        "que hallan diferencias entre políticas afines en ADNI. Los valores "
        "concretos son una hipótesis para OASIS3, no un óptimo demostrado.", "",
        "## Pérdida SSL", "",
        "| Método | Épocas | Mejor val (época) | Train final | Val final |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['method']} | {row['epochs']} | "
                     f"{row['ssl_val_best']:.4f} ({row['ssl_val_best_epoch']}) | "
                     f"{row['ssl_train_final']:.4f} | {row['ssl_val_final']:.4f} |")
    lines += [
        "", "Las escalas de pérdida SSL difieren entre métodos.", "",
        "## Probe AD/HC por sujeto en test", "",
        "Encoder congelado; regresión logística y normalización ajustadas en train "
        "(615 sujetos AD/HC); regularización elegida en validación (131 sujetos). "
        "Test: 134 sujetos (31 AD, 103 HC). El encoder aleatorio usa el mismo probe.", "",
        "| Método | AUROC SSL | Aleatorio | Δ AUROC [IC 95 %] | "
        "Log-loss SSL | Aleatorio | Prior AD | Rango efectivo SSL |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lo, hi = row["delta_auc_95ci"]
        lines.append(f"| {row['method']} | {row['trained_test_auc']:.3f} | "
                     f"{row['random_test_auc']:.3f} | "
                     f"{row['delta_test_auc']:+.3f} [{lo:+.3f}, {hi:+.3f}] | "
                     f"{row['trained_test_log_loss']:.3f} | "
                     f"{row['random_test_log_loss']:.3f} | "
                     f"{row['prior_test_log_loss']:.3f} | "
                     f"{row['trained_embedding_effective_rank']:.1f} |")
    lines += ["", "## Referencia histórica: `campaign_full50`", "",
              "| Método | AUROC nuevo | AUROC full50 | Diferencia |",
              "|---|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['method']} ({row['epochs']} épocas) | "
                     f"{row['trained_test_auc']:.3f} | "
                     f"{row['historical_full50_test_auc']:.3f} "
                     f"({row['historical_full50_epochs']} épocas) | "
                     f"{row['difference_vs_full50_auc']:+.3f} |")
    lines += [
        "", "IC 95 %: bootstrap pareado por sujeto para este split y semilla. "
        "Un encoder aleatorio con probe entrenado puede superar AUROC 0,5. "
        "La comparación histórica con `campaign_full50` también cambia el número "
        "de épocas y la vista de validación; una diferencia no aísla el efecto "
        "de la augmentation.", "",
        "Figuras: [curvas SSL](figures/ssl_learning_curves.png), "
        "[AUROC](figures/ad_hc_subject_auc.png), "
        "[Δ AUROC](figures/ad_hc_delta_auc.png), "
        "[log-loss](figures/ad_hc_probe_log_loss.png), "
        "[tabla visual](figures/probe_vs_baseline_table.png) y "
        "[ejemplos de vistas](figures/augmentation_examples.png).", "",
    ]
    return "\n".join(lines)


def main() -> None:
    """Finalize each stopped run using the last complete epoch."""
    paths = completed_experiments()
    manifest = json.loads((CAMPAIGN / "experiment_manifest.json").read_text())
    baseline = json.loads((ROOT / "campaign_full50/results.json").read_text())
    reference = Path(next(row["experiment_dir"] for row in baseline["results"]
                          if row["method"] == "vicreg"))
    for method, path in paths.items():
        for name in ("subject_splits.csv", "data_records.csv"):
            if (path / name).read_bytes() != (reference / name).read_bytes():
                raise ValueError(f"{method} differs from the full50 cohort: {name}")
    rows = []
    old_by_method = {row["method"]: row for row in baseline["results"]}
    reference_subjects: list[str] | None = None
    for method in METHODS:
        path = paths[method]
        train = epoch_series(path / "metrics.csv", "train/loss_epoch")
        val = epoch_series(path / "metrics.csv", "val/loss")
        epochs = manifest[method]["epochs"]
        if set(train) != set(range(epochs)) or set(val) != set(range(epochs)):
            raise ValueError(f"{method} does not have {epochs} complete epochs")
        result = probe(path)
        trained, random = result["results"]["trained"], result["results"]["random"]
        predictions = trained["test_predictions"]
        subjects = [item["subject"] for item in predictions]
        if reference_subjects is not None and subjects != reference_subjects:
            raise ValueError("Probe test subjects differ across methods")
        reference_subjects = subjects
        labels = [item["AD"] for item in predictions]
        prior_loss = float(log_loss(labels, np.full(len(labels), train_prior(path)),
                                    labels=[0, 1]))
        best_epoch = min(val, key=val.get)
        rows.append({
            "method": method,
            "experiment_dir": str(path),
            "epochs": epochs,
            "checkpoint_selection": "menor val/loss SSL",
            "ssl_train_final": train[epochs - 1],
            "ssl_val_final": val[epochs - 1],
            "ssl_val_best": val[best_epoch],
            "ssl_val_best_epoch": best_epoch + 1,
            "trained_test_auc": trained["test_auc"],
            "random_test_auc": random["test_auc"],
            "delta_test_auc": result["comparison"]["delta_test_auc"],
            "delta_auc_95ci": result["comparison"]["delta_test_auc_bootstrap_95ci"],
            "trained_test_log_loss": trained["test_log_loss"],
            "random_test_log_loss": random["test_log_loss"],
            "prior_test_log_loss": prior_loss,
            "trained_test_balanced_accuracy": trained["test_balanced_accuracy"],
            "random_test_balanced_accuracy": random["test_balanced_accuracy"],
            "trained_embedding_effective_rank": trained["embedding_effective_rank"],
            "random_embedding_effective_rank": random["embedding_effective_rank"],
            "historical_full50_epochs": old_by_method[method]["epochs"],
            "historical_full50_test_auc": old_by_method[method]["trained_test_auc"],
            "difference_vs_full50_auc": (
                trained["test_auc"] - old_by_method[method]["trained_test_auc"]
            ),
        })
    figure_dir = CAMPAIGN / "figures"
    subprocess.run([
        str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/plot_experiments.py"),
        *(str(paths[m]) for m in METHODS), "--output-dir", str(figure_dir),
    ], cwd=ROOT, check=True)
    make_extra_figures(rows, figure_dir)
    subprocess.run([
        str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/plot_mri_views.py"),
    ], cwd=ROOT, check=True)
    report = {
        "protocol": "OASIS3 full50 manifest and subject split; 20/15/12 completed epochs "
                    "for DINOv2/VICReg/BYOL at user-directed stops; full-field mild MRI "
                    "views; best SSL-validation checkpoint; frozen subject-level "
                    "AD/HC probe; held-out test once per method",
        "experiment_manifest": manifest,
        "results": rows,
    }
    (CAMPAIGN / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    (CAMPAIGN / "RESULTADOS.md").write_text(report_markdown(rows))
    print(json.dumps({"results": str(CAMPAIGN / "results.json"),
                      "figures": str(figure_dir), "methods": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
