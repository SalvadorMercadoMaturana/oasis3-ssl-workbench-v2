"""Refit one SSL-loss-selected configuration for 100 epochs, without probes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import hydra
import numpy as np
from plot_experiments import epoch_series
from tune_pretext import METHOD_GROUPS, apply_params

from t5_ssl.train import train

ROOT = Path(__file__).resolve().parents[2]


def original_experiment(method: str) -> Path:
    for line in (ROOT / "campaign_100ep/status.log").read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[1:3] == ["COMPLETE", method]:
            return ROOT / parts[3]
    raise RuntimeError(f"No completed original 100-epoch run for {method}")


def loss_summary(path: Path, epochs: int) -> dict[str, float]:
    train_values = epoch_series(path / "metrics.csv", "train/loss_epoch")
    val_values = epoch_series(path / "metrics.csv", "val/loss")
    if set(train_values) != set(range(epochs)) or set(val_values) != set(range(epochs)):
        raise RuntimeError(f"Expected exactly {epochs} complete train/validation epochs in {path}")
    return {"final5_train_loss": float(np.mean([train_values[e] for e in range(epochs - 5, epochs)])),
            "final5_val_loss": float(np.mean([val_values[e] for e in range(epochs - 5, epochs)])),
            "best_val_loss": float(min(val_values.values())),
            "best_val_epoch": int(min(val_values, key=val_values.get))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHOD_GROUPS, required=True)
    parser.add_argument("--study-dir", type=Path, default=Path("tuning_pretext"))
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    if args.epochs < 5:
        parser.error("Need at least five refit epochs")
    candidates = list(args.study_dir.glob(f"{args.method}_ssl_loss_*_best.json"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one completed study for {args.method}: {candidates}")
    selected = json.loads(candidates[0].read_text(encoding="utf-8"))
    resolved = selected["trials_complete"] + selected["trials_pruned"]
    if (selected["trials_complete"] < 3 or selected["trials_requested"] < 4 or
            resolved != selected.get("trials_resolved", selected["trials_requested"]) or
            (not selected.get("early_stopped", False) and
             resolved != selected["trials_requested"])):
        raise RuntimeError("Tuning study is incomplete")
    for name in ("csv_path", "manifest_path"):
        entry = selected["data_identity"][name]
        if hashlib.sha256(Path(entry["path"]).read_bytes()).hexdigest() != entry["sha256"]:
            raise RuntimeError(f"The {name} changed since tuning")
    baseline_path = original_experiment(args.method)
    baseline_metrics = loss_summary(baseline_path, args.epochs)
    params = selected["best_params"]
    baseline = selected["baseline"]
    reused = all(abs(float(params[name]) - float(baseline[name])) < 1e-12
                 for name in baseline)
    if reused:
        tuned_path = baseline_path
    else:
        config_dir = str(ROOT / "configs")
        overrides = METHOD_GROUPS[args.method] + ["data=oasis3_lmdb", "trainer.accelerator=cpu"]
        with hydra.initialize_config_dir(version_base="1.3", config_dir=config_dir):
            cfg = hydra.compose(config_name="train", overrides=overrides)
        cfg.trainer.max_epochs = args.epochs
        cfg.trainer.enable_progress_bar = False
        apply_params(cfg, params)
        result = train(cfg)
        tuned_path = ROOT / result["exp_dir"]
    tuned_metrics = loss_summary(tuned_path, args.epochs)
    output = {"method": args.method, "selection_objective": selected["objective"],
              "selected_trial": selected["best_trial"], "selected_params": params,
              "selection_epochs": selected["epochs"], "refit_epochs": args.epochs,
              "baseline_experiment": str(baseline_path), "tuned_experiment": str(tuned_path),
              "reused_original_baseline": reused, "baseline": baseline_metrics,
              "tuned": tuned_metrics,
              "delta_final5_val_loss": tuned_metrics["final5_val_loss"] -
              baseline_metrics["final5_val_loss"],
              "no_probe_used": True}
    outpath = args.study_dir / f"{args.method}_refit.json"
    outpath.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
