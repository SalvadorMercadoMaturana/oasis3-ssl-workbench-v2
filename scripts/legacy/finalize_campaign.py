"""Evaluate completed 100-epoch runs once and create figures/results."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import torch
from plot_experiments import epoch_series

from t5_ssl.data.paths import DEFAULT_CSV

ROOT = Path(__file__).resolve().parents[2]
METHODS = ("vicreg", "byol", "dinov2")


def completed_experiments(status_path: Path) -> dict[str, Path]:
    """Read only successful campaign entries and reject duplicates."""
    experiments = {}
    for line in status_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[1] == "COMPLETE":
            method = parts[2]
            if method in experiments:
                raise ValueError(f"Duplicate completed run for {method}")
            experiments[method] = ROOT / parts[3]
    if set(experiments) != set(METHODS):
        raise RuntimeError(f"Campaign incomplete: {sorted(experiments)}")
    return experiments


def main() -> None:
    campaign = ROOT / "campaign_100ep"
    expanded_test = ROOT / "expanded_test_oasis3.csv"
    if not expanded_test.is_file():
        raise FileNotFoundError(expanded_test)
    test_manifest_info = json.loads((ROOT / "expanded_test_oasis3.json").read_text(
        encoding="utf-8"))
    if hashlib.sha256(expanded_test.read_bytes()).hexdigest() != test_manifest_info[
            "extended_manifest_sha256"]:
        raise RuntimeError("Predeclared held-out test manifest has changed")
    source_csv = Path(DEFAULT_CSV)
    if hashlib.sha256(source_csv.read_bytes()).hexdigest() != test_manifest_info[
            "source_csv_sha256"]:
        raise RuntimeError("OASIS3 clinical source CSV has changed")
    experiments = completed_experiments(campaign / "status.log")
    for method, path in experiments.items():
        train = epoch_series(path / "metrics.csv", "train/loss_epoch")
        val = epoch_series(path / "metrics.csv", "val/loss")
        if set(train) != set(range(100)) or set(val) != set(range(100)):
            raise RuntimeError(f"{method} did not log exactly 100 train/val epochs")
        if not (path / "checkpoints" / "last.ckpt").is_file():
            raise FileNotFoundError(path / "checkpoints" / "last.ckpt")
        checkpoint = torch.load(path / "checkpoints" / "last.ckpt",
                                map_location="cpu", weights_only=False)
        if checkpoint["epoch"] != 100:
            raise RuntimeError(f"{method} last.ckpt is from epoch {checkpoint['epoch']}, not 100")
        del checkpoint
        probe = path / "ad_hc_subject_probe.json"
        if not probe.is_file():
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "2"
            subprocess.run([str(ROOT / ".venv/bin/python"),
                            str(ROOT / "scripts/legacy/evaluate_frozen_probe.py"), str(path),
                            "--test-manifest", str(expanded_test)],
                           cwd=ROOT, env=env, check=True)
    figures = campaign / "figures"
    subprocess.run([str(ROOT / ".venv/bin/python"),
                    str(ROOT / "scripts/legacy/plot_experiments.py"),
                    *(str(experiments[name]) for name in METHODS),
                    "--output-dir", str(figures)], cwd=ROOT, check=True)
    rows = []
    for method in METHODS:
        path = experiments[method]
        probe = json.loads((path / "ad_hc_subject_probe.json").read_text(encoding="utf-8"))
        trained, random = probe["results"]["trained"], probe["results"]["random"]
        comparison = probe["comparison"]
        low, high = comparison["delta_test_auc_bootstrap_95ci"]
        interpretation = ("mejora en este test" if low > 0 else
                          "empeora en este test" if high < 0 else "incierto")
        rows.append({"method": method, "experiment_dir": str(path),
                     "train_subjects": trained["train_subjects"],
                     "val_subjects": trained["val_subjects"],
                     "test_subjects": trained["test_subjects"],
                     "test_AD": sum(p["AD"] for p in trained["test_predictions"]),
                     "test_HC": trained["test_subjects"] - sum(
                         p["AD"] for p in trained["test_predictions"]),
                     "trained_val_auc": trained["val_auc"],
                     "random_val_auc": random["val_auc"],
                     "trained_test_auc": trained["test_auc"],
                     "random_test_auc": random["test_auc"],
                     "trained_test_balanced_accuracy": trained["test_balanced_accuracy"],
                     "random_test_balanced_accuracy": random["test_balanced_accuracy"],
                     "delta_test_auc": comparison["delta_test_auc"],
                     "delta_auc_95ci": [low, high],
                     "trained_auc_95ci": comparison["trained_test_auc_bootstrap_95ci"],
                     "random_auc_95ci": comparison["random_test_auc_bootstrap_95ci"],
                     "interpretation": interpretation})
    output = {"cohort": "representative_oasis3.csv SSL train/val; expanded_test_oasis3.csv "
                        "adds all other OASIS3 subjects to held-out test. One primary "
                        "MNI session per subject; random axial train slice in [0.30, 0.70]; "
                        "central deterministic validation/probe slice",
              "expanded_test_manifest": test_manifest_info,
              "epochs_per_method": 100,
              "final_protocol_test_evaluations_per_method": 1,
              "discarded_preliminary_evaluation": "An earlier probe sampled random "
                  "training slices and was discarded. The frozen encoders and "
                  "hyperparameters were unchanged; this report uses only the "
                  "corrected deterministic probe outputs.",
              "cohort_representativeness": json.loads(
                  (ROOT / "cohort_representativeness.json").read_text(encoding="utf-8")),
              "results": rows,
              "limitations": "Pilot SSL training cohort and small AD validation count; "
                             "subject-bootstrap intervals condition on one fitted model "
                             "and do not capture split/seed variation; different SSL "
                             "losses are not directly comparable."}
    (campaign / "results.json").write_text(json.dumps(output, indent=2) + "\n",
                                           encoding="utf-8")
    lines = ["# OASIS3 LMDB: 100 épocas por método", "",
             "Cohorte piloto SSL: 98 sujetos OASIS3; train 68, val 14. "
             "Test ampliado predefinido: 899 sujetos OASIS3 nunca vistos en train/val. "
             "Una sesión primaria por sujeto, "
             "corte aleatorio en train y corte central fijo en validación/probe. "
             "Split fijo por sujeto. El probe usa encoder congelado y regresión logística; "
             "el control usa el mismo backbone con pesos aleatorios.", "",
             "| Método | AUROC val (T5/control) | AUROC test (T5/control) | Δ AUROC test [IC 95 %] | Lectura |",
             "|---|---:|---:|---:|---|"]
    for row in rows:
        ci = row["delta_auc_95ci"]
        lines.append(f"| {row['method']} | {row['trained_val_auc']:.3f}/{row['random_val_auc']:.3f} "
                     f"| {row['trained_test_auc']:.3f}/{row['random_test_auc']:.3f} "
                     f"| {row['delta_test_auc']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] "
                     f"| {row['interpretation']} |")
    lines += ["", "La validación contiene pocos sujetos AD. "
              "No se ajustaron hiperparámetros con test. Las pérdidas SSL de métodos "
              "distintos tienen escalas diferentes.", "",
              "Una evaluación preliminar se descartó al detectar que el probe "
              "muestreaba cortes aleatorios en train; se repitió con cortes fijos "
              "sin modificar checkpoints ni hiperparámetros. Estas cifras son las "
              "de la evaluación corregida.", "",
              "Figuras: `figures/ssl_learning_curves.png`, "
              "`figures/ad_hc_subject_auc.png` y `figures/ad_hc_delta_auc.png`.", ""]
    (campaign / "RESULTADOS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
