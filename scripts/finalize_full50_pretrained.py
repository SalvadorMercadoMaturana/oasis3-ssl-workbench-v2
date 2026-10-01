"""Evaluate the 50-epoch pretrained-backbone campaign and render its report.

Companion to ``finalize_full50.py`` for the follow-up campaign that swapped in
ImageNet-pretrained backbones for VICReg (ConvNeXt-tiny + dropout=0.1, tuned
this session to correct overfitting without collapsing embeddings) and BYOL
(ResNet-18, unchanged from the report-matching config), keeping DINOv2
(ViT-Small/patch60) as-is. No early stopping: all three ran the full 50
epochs, so this skips the early-stopping bookkeeping in the original script.
Adds F1-macro and AUPR (average precision) to the probe table, and reports
the anti-collapse trajectory (effective rank, mean feature std, mean
pairwise cosine similarity) at every logged health-check epoch, not just the
final one.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
from collections.abc import Callable
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
CAMPAIGN = ROOT / "campaign_full50_pretrained"
METHODS = ("vicreg", "byol", "dinov2")
METHOD_LABEL = {"vicreg": "VICReg", "byol": "BYOL", "dinov2": "DINOv2"}
EXPECTED_EPOCHS = 50

TITLE = "# OASIS3: SSL con backbones pre-entrenados en ImageNet, 50 épocas completas"
PROTOCOL = ("OASIS3 original LMDB; 981 subjects; same fixed subject-disjoint "
            "split as campaign_full50; SSL sees one randomly selected session and axial "
            "slice per train subject per epoch; fixed primary session and central slice "
            "for validation and frozen linear probe; fixed 50 epochs, no early stopping "
            "(disabled to inspect the full trajectory); VICReg and BYOL use ImageNet1K-"
            "pretrained backbones (ConvNeXt-tiny + projector dropout=0.1, and ResNet-18 "
            "respectively); DINOv2 uses the from-scratch ViT-Small/patch60 config from "
            "the EL7006-report replication; checkpoint chosen without labels "
            "(best val/loss SSL checkpoint); probe C chosen on validation AUROC; test "
            "evaluated once. Trained sequentially (one at a time, not concurrently) on "
            "GPU (NVIDIA GTX 1650, 4GB VRAM) at full 32-bit precision after concurrent "
            "CPU runs repeatedly crashed on this machine's limited RAM; "
            "trainer.deterministic=false on GPU (ViTBackbone's bicubic position-embedding "
            "interpolation has no deterministic CUDA backward), unlike the bit-exact CPU "
            "campaigns.")
INTRO = ("981 sujetos OASIS3 del LMDB original; mismo split fijo y disjunto por sujeto "
         "que `campaign_full50`. SSL utiliza las sesiones disponibles en train, "
         "sorteando una sesión y un corte por sujeto en cada época. Validación y test "
         "usan una sesión y un corte fijos. Las 50 épocas se completaron sin early "
         "stopping (desactivado deliberadamente para poder inspeccionar la trayectoria "
         "completa). VICReg usa ConvNeXt-tiny con pesos ImageNet1K y dropout=0.1 en el "
         "proyector (ajustado en esta sesión para corregir sobreajuste sin colapsar los "
         "embeddings); BYOL usa ResNet-18 con pesos ImageNet1K (config del informe "
         "EL7006); DINOv2 usa el ViT-Small/patch60 entrenado desde cero replicando ese "
         "mismo informe. Los tres se entrenaron secuencialmente (uno a la vez, no "
         "concurrentes) en GPU (NVIDIA GTX 1650, 4GB VRAM) a precisión completa "
         "(32-true), tras que las corridas concurrentes en CPU crashearan repetidamente "
         "por memoria en este equipo. En GPU se usó `trainer.deterministic=false` (la "
         "interpolación bicúbica de embeddings posicionales del ViT no tiene backward "
         "determinista en CUDA), a diferencia de las campañas en CPU que sí son "
         "reproducibles bit a bit.")


def experiment_paths(campaign: Path = CAMPAIGN) -> dict[str, Path]:
    paths = {}
    for line in (campaign / "exp_dirs.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        name, rel = line.split("=", 1)
        paths[name] = ROOT / rel
    if set(paths) != set(METHODS):
        raise RuntimeError(f"Campaign incomplete or mislabeled: {sorted(paths)}")
    return paths


def train_prior(path: Path) -> float:
    with (path / "subject_splits.csv").open(newline="", encoding="utf-8") as stream:
        train_ids = {row["subject"] for row in csv.DictReader(stream)
                     if row["split"] == "train"}
    groups: dict[str, set[str]] = {}
    for row in clinical_rows(DEFAULT_CSV):
        groups.setdefault(row["subject"], set()).add(row["group"])
    labels = [next(iter(values)) for subject, values in groups.items()
              if subject in train_ids and len(values) == 1 and
              next(iter(values)) in ("AD", "HC")]
    return labels.count("AD") / len(labels)


def health_trajectory(path: Path) -> list[dict]:
    health_path = path / "representation_health.json"
    if not health_path.is_file():
        return []
    return sorted(json.loads(health_path.read_text(encoding="utf-8")),
                  key=lambda r: r["epoch"])


def ensure_probe(path: Path) -> dict:
    probe_path = path / "ad_hc_subject_probe_best.json"
    if not probe_path.is_file():
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "2"
        env["MKL_NUM_THREADS"] = "2"
        env["OPENBLAS_NUM_THREADS"] = "2"
        command = [str(ROOT / ".venv/Scripts/python.exe"),
                   str(ROOT / "scripts/evaluate_frozen_probe.py"),
                   str(path.relative_to(ROOT)), "--checkpoint", "best"]
        subprocess.run(command, cwd=ROOT, env=env, check=True)
    return json.loads(probe_path.read_text(encoding="utf-8"))


def main(campaign: Path = CAMPAIGN, *, report_title: str = TITLE, intro: str = INTRO,
         protocol: str = PROTOCOL, script_name: str = "finalize_full50_pretrained.py",
         extra_lines: Callable[[list[dict]], list[str]] | None = None) -> list[dict]:
    """Probe every method of `campaign` and write its results.json and RESULTADOS.md.

    `extra_lines(rows)` may add report sections just before the figure links.
    """
    paths = experiment_paths(campaign)
    figure_dir = campaign / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    trajectories: dict[str, list[dict]] = {}
    common_test: tuple[list[str], list[int]] | None = None
    for method, path in paths.items():
        train = epoch_series(path / "metrics.csv", "train/loss_epoch")
        val = epoch_series(path / "metrics.csv", "val/loss")
        if set(train) != set(range(EXPECTED_EPOCHS)) or set(val) != set(range(EXPECTED_EPOCHS)):
            raise RuntimeError(f"{method} does not contain {EXPECTED_EPOCHS} full epochs "
                               f"(train={len(train)}, val={len(val)})")
        trajectories[method] = health_trajectory(path)
        doc = ensure_probe(path)
        trained, random = doc["results"]["trained"], doc["results"]["random"]
        prediction = trained["test_predictions"]
        subjects = [item["subject"] for item in prediction]
        labels = [item["AD"] for item in prediction]
        if common_test is not None and common_test != (subjects, labels):
            raise RuntimeError("Methods did not evaluate the same test subjects")
        common_test = (subjects, labels)
        prior = train_prior(path)
        baseline_loss = float(log_loss(labels, np.full(len(labels), prior), labels=[0, 1]))
        rows.append({
            "method": method, "experiment_dir": str(path.relative_to(ROOT)),
            "epochs": EXPECTED_EPOCHS,
            "ssl_train_final": train[max(train)],
            "ssl_val_final": val[max(val)], "ssl_val_best": min(val.values()),
            "ssl_val_best_epoch": min(val, key=val.get) + 1,
            "probe_train_subjects": trained["train_subjects"],
            "probe_val_subjects": trained["val_subjects"],
            "probe_test_subjects": trained["test_subjects"],
            "test_ad": sum(labels), "test_hc": len(labels) - sum(labels),
            "trained_test_auc": trained["test_auc"],
            "random_test_auc": random["test_auc"],
            "delta_test_auc": doc["comparison"]["delta_test_auc"],
            "delta_auc_95ci": doc["comparison"]["delta_test_auc_bootstrap_95ci"],
            "bootstrap_probability_delta_positive": doc["comparison"][
                "bootstrap_probability_delta_positive"],
            "trained_test_f1_macro": trained["test_f1_macro"],
            "random_test_f1_macro": random["test_f1_macro"],
            "trained_test_aupr": trained["test_average_precision"],
            "random_test_aupr": random["test_average_precision"],
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

    subprocess.run([str(ROOT / ".venv/Scripts/python.exe"),
                    str(ROOT / "scripts/plot_experiments.py"),
                    *(str(paths[method]) for method in METHODS),
                    "--output-dir", str(figure_dir)], cwd=ROOT, check=True)

    # Anti-collapse trajectory figure: one panel per metric, one line per method.
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    colors = {"vicreg": "#256b8c", "byol": "#e3ae65", "dinov2": "#8a4fae"}
    metric_specs = [
        ("effective_rank", "Rango efectivo espectral"),
        ("mean_feature_std", "Desviación media por dimensión"),
        ("mean_cosine_similarity_diff_samples", "Cos-sim media (muestras distintas)"),
    ]
    for ax, (key, title) in zip(axes, metric_specs, strict=True):
        for method in METHODS:
            records = trajectories[method]
            if not records:
                continue
            xs = [r["epoch"] for r in records]
            ys = [r.get(key) for r in records]
            ax.plot(xs, ys, marker="o", markersize=3.5, color=colors[method],
                     label=METHOD_LABEL[method])
        ax.set_ylabel(title, fontsize=9)
        ax.grid(alpha=0.25)
    axes[0].legend(loc="best")
    axes[-1].set_xlabel("Época")
    fig.suptitle("Medidas anti-colapso por época (encoder SSL)")
    fig.tight_layout()
    fig.savefig(figure_dir / "anticollapse_trajectory.png", dpi=180)
    plt.close(fig)

    # Headline metrics figure: AUROC / F1-macro / AUPR, SSL vs random, per method.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), dpi=180)
    metric_cols = [
        ("trained_test_auc", "random_test_auc", "AUROC (test)"),
        ("trained_test_f1_macro", "random_test_f1_macro", "F1-macro (test)"),
        ("trained_test_aupr", "random_test_aupr", "AUPR (test)"),
    ]
    method_order = list(METHODS)
    x = np.arange(len(method_order))
    width = 0.32
    for ax, (trained_key, random_key, title) in zip(axes, metric_cols, strict=True):
        trained_vals = [next(r[trained_key] for r in rows if r["method"] == m) for m in method_order]
        random_vals = [next(r[random_key] for r in rows if r["method"] == m) for m in method_order]
        bars_ssl = ax.bar(x - width / 2, trained_vals, width, label="SSL", color="#256b8c")
        bars_rand = ax.bar(x + width / 2, random_vals, width, label="Aleatorio", color="#9caeb8")
        for bars in (bars_ssl, bars_rand):
            ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
        ax.set_xticks(x, [METHOD_LABEL[m] for m in method_order])
        ax.set_ylim(0, 1.0)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(loc="lower left", fontsize=9)
    axes[0].set_ylabel("Score")
    fig.suptitle("Probe AD/HC en test: SSL vs. control aleatorio, por modelo", fontsize=13)
    fig.tight_layout()
    fig.savefig(figure_dir / "headline_metrics.png", dpi=180)
    plt.close(fig)

    report = {"protocol": protocol,
              "results": rows,
              "anticollapse_trajectory": trajectories}
    (campaign / "results.json").write_text(json.dumps(report, indent=2) + "\n")

    lines = [report_title, "", intro, "",
             "## Pérdida SSL", "",
             "| Método | Épocas | Train final | Validación final | Mejor validación (época) |",
             "|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {METHOD_LABEL[row['method']]} | {row['epochs']} | "
                     f"{row['ssl_train_final']:.4f} | {row['ssl_val_final']:.4f} | "
                     f"{row['ssl_val_best']:.4f} ({row['ssl_val_best_epoch']}) |")
    lines += ["", "Las escalas de pérdida SSL difieren entre métodos y no se comparan "
              "directamente.", "",
              "## Probe AD/HC por sujeto en test", "",
              f"Encoder congelado; regresión logística y normalización ajustadas en train "
              f"({rows[0]['probe_train_subjects']} sujetos AD/HC). C elegido por AUROC en "
              f"validación ({rows[0]['probe_val_subjects']} sujetos). El test contiene "
              f"{rows[0]['probe_test_subjects']} sujetos AD/HC ({rows[0]['test_ad']} AD, "
              f"{rows[0]['test_hc']} HC). El control aleatorio también ajusta un probe; el "
              "prior AD no usa imagen.", "",
              "| Método | AUROC SSL | AUROC aleatorio | Δ AUROC [IC 95 %] | P(Δ>0) | "
              "F1-macro SSL | F1-macro aleatorio | AUPR SSL | AUPR aleatorio | "
              "Log-loss SSL | Log-loss aleatorio | Log-loss prior |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lo, hi = row["delta_auc_95ci"]
        lines.append(
            f"| {METHOD_LABEL[row['method']]} | {row['trained_test_auc']:.3f} | "
            f"{row['random_test_auc']:.3f} | "
            f"{row['delta_test_auc']:+.3f} [{lo:+.3f}, {hi:+.3f}] | "
            f"{row['bootstrap_probability_delta_positive']:.3f} | "
            f"{row['trained_test_f1_macro']:.3f} | {row['random_test_f1_macro']:.3f} | "
            f"{row['trained_test_aupr']:.3f} | {row['random_test_aupr']:.3f} | "
            f"{row['trained_test_log_loss']:.3f} | {row['random_test_log_loss']:.3f} | "
            f"{row['prior_test_log_loss']:.3f} |")
    lines += ["", "IC 95 %: bootstrap pareado por sujeto, condicionado a un split y una "
              "semilla. P(Δ>0) es la probabilidad bootstrap de que el encoder SSL supere al "
              "aleatorio en AUROC test — valores cercanos a 1 son evidencia fuerte, cercanos "
              "a 0.5 son un empate estadístico. AUROC 0,5 representa ranking aleatorio sin "
              "entrenar el probe; un encoder aleatorio con un probe entrenado puede superar "
              "0,5. Un descenso de pérdida SSL por sí solo no demuestra utilidad clínica.",
              "",
              "![Resultado principal: AUROC, F1-macro y AUPR por modelo, SSL vs. aleatorio]"
              "(figures/headline_metrics.png)",
              "",
              "## Medidas anti-colapso (encoder SSL, cada 5 épocas)", "",
              "Rango efectivo espectral (dimensiones efectivamente usadas), desviación media "
              "por dimensión, y similitud coseno media entre muestras distintas del batch de "
              "validación. Rango bajo + cos-sim alta y creciente hacia 1 es la firma de "
              "colapso (dimensional o total); estas son las medidas de salud que se fueron "
              "reportando durante el entrenamiento.", ""]
    for row in rows:
        method = row["method"]
        lines.append(f"### {METHOD_LABEL[method]}")
        lines.append("")
        lines.append("| Época | Rango efectivo | Desv. media | Cos-sim media (dist.) |")
        lines.append("|---:|---:|---:|---:|")
        for record in trajectories[method]:
            lines.append(
                f"| {record['epoch']} | {record.get('effective_rank'):.2f} | "
                f"{record.get('mean_feature_std'):.5f} | "
                f"{record.get('mean_cosine_similarity_diff_samples'):.4f} |")
        lines.append("")
    if extra_lines is not None:
        lines += extra_lines(rows)
    lines += ["Figuras: [resultado principal](figures/headline_metrics.png), "
              "[curvas SSL](figures/ssl_learning_curves.png), "
              "[trayectoria anti-colapso](figures/anticollapse_trajectory.png), "
              "[AUROC](figures/ad_hc_subject_auc.png), "
              "[Δ AUROC](figures/ad_hc_delta_auc.png).", "",
              "Comandos reproducibles en [README](../README.md) y en "
              f"`scripts/{script_name}`.", ""]
    (campaign / "RESULTADOS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return rows


if __name__ == "__main__":
    main()
