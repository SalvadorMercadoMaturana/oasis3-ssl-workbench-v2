"""Plot per-method SSL learning curves and summarize held-out probe metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


def epoch_series(path: Path, column: str) -> dict[int, float]:
    """Collect the last logged value of a metric for each epoch."""
    values = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("epoch") and row.get(column):
                values[int(row["epoch"])] = float(row[column])
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(args.experiment_dirs), 1,
                             figsize=(9, 3.5 * len(args.experiment_dirs)), squeeze=False)
    summary = {}
    probe_rows = []
    delta_rows = []
    for index, exp_dir in enumerate(args.experiment_dirs):
        metrics = exp_dir / "metrics.csv"
        if not metrics.is_file():
            raise FileNotFoundError(metrics)
        train = epoch_series(metrics, "train/loss_epoch")
        val = epoch_series(metrics, "val/loss")
        if not train or not val:
            raise ValueError(f"Missing train/val losses in {metrics}")
        ax = axes[index, 0]
        ax.plot(sorted(train), [train[e] for e in sorted(train)],
                marker="o", markersize=3, label="train")
        ax.plot(sorted(val), [val[e] for e in sorted(val)],
                marker="o", markersize=3, label="validation")
        config = yaml.safe_load((exp_dir / "config.yaml").read_text(encoding="utf-8"))
        method = config["loss"]["name"]
        ax.set(title=method.upper() if method != "dinov2" else "DINOv2 compacto",
               xlabel="Epoch", ylabel="SSL loss")
        ax.grid(alpha=0.25)
        ax.legend()
        best_epoch = min(val, key=val.get)
        record = {"epochs_logged": len(train), "best_val_epoch": best_epoch,
                  "best_val_loss": val[best_epoch], "last_train_loss": train[max(train)],
                  "last_val_loss": val[max(val)]}
        probe_path = exp_dir / "ad_hc_subject_probe.json"
        if not probe_path.is_file():
            probe_path = exp_dir / "ad_hc_subject_probe_best.json"
        if probe_path.is_file():
            probe = json.loads(probe_path.read_text(encoding="utf-8"))
            delta_rows.append((method, probe["comparison"]["delta_test_auc"],
                               probe["comparison"]["delta_test_auc_bootstrap_95ci"]))
            for kind in ("trained", "random"):
                result = probe["results"][kind]
                bounds = probe["comparison"][f"{kind}_test_auc_bootstrap_95ci"]
                probe_rows.append((method, kind, result["test_auc"], bounds))
                record[f"{kind}_test_auc"] = result["test_auc"]
                record[f"{kind}_val_auc"] = result["val_auc"]
                record[f"{kind}_test_subjects"] = result["test_subjects"]
        summary[exp_dir.name] = record
    fig.tight_layout()
    curves = args.output_dir / "ssl_learning_curves.png"
    fig.savefig(curves, dpi=180)
    plt.close(fig)
    if probe_rows:
        fig, ax = plt.subplots(figsize=(9, max(3, len(probe_rows) * 0.45)))
        labels = [f"{method} ({'T5' if kind == 'trained' else 'aleatorio'})"
                  for method, kind, _, _ in probe_rows]
        values = [value for _, _, value, _ in probe_rows]
        errors = [[max(0, value - bounds[0]) for _, _, value, bounds in probe_rows],
                  [max(0, bounds[1] - value) for _, _, value, bounds in probe_rows]]
        colors = ["#21618c" if kind == "trained" else "#aab7b8"
                  for _, kind, _, _ in probe_rows]
        ax.barh(labels, values, color=colors, xerr=errors, capsize=3)
        ax.set(xlim=(0, 1), xlabel="AUROC AD/HC por sujeto en test (IC 95 % bootstrap)")
        ax.axvline(0.5, color="gray", linestyle="--", linewidth=1)
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        fig.savefig(args.output_dir / "ad_hc_subject_auc.png", dpi=180)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, max(2.8, len(delta_rows) * 0.8)))
        for method, delta, bounds in delta_rows:
            ax.errorbar(delta, method,
                        xerr=[[max(0, delta - bounds[0])],
                              [max(0, bounds[1] - delta)]],
                        fmt="o", color="#21618c", capsize=4, markersize=7)
        limit = max(0.06, max(abs(bound) for _, _, bounds in delta_rows
                              for bound in bounds)) + 0.02
        ax.set(xlim=(-limit, limit), xlabel="Δ AUROC test: T5 − aleatorio (IC 95 % pareado)")
        ax.axvline(0, color="gray", linestyle="--", linewidth=1)
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        fig.savefig(args.output_dir / "ad_hc_delta_auc.png", dpi=180)
        plt.close(fig)
    output = args.output_dir / "results_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"curves": str(curves), "summary": str(output),
                      "experiments": summary}, indent=2))


if __name__ == "__main__":
    main()
