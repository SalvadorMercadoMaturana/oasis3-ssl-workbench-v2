"""Resumable Optuna search; select on validation subjects, never on test."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import hydra
import lightning as L
import numpy as np
import optuna
from evaluate_frozen_probe import extract_subjects
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from t5_ssl.data.augmentations import EvalTransform
from t5_ssl.train import train

METHOD_GROUPS = {
    "vicreg": [],
    "byol": ["model=byol_convnext", "loss=byol"],
    "dinov2": ["model=dinov2_vit", "loss=dinov2",
               "augmentation=dinov2_multicrop"],
}


class ValidationLossReporter(L.Callback):
    """Report intermediate SSL loss for val-loss studies only."""

    def __init__(self, trial: optuna.Trial) -> None:
        self.trial = trial

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.sanity_checking or "val/loss" not in trainer.callback_metrics:
            return
        value = float(trainer.callback_metrics["val/loss"])
        if not np.isfinite(value):
            raise optuna.TrialPruned("Nonfinite validation loss")
        self.trial.report(value, step=trainer.current_epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned()


def validation_auc(result: dict, batch_size: int) -> float:
    """Fit a fixed logistic probe on train embeddings and score validation."""
    dm, model = result["datamodule"], result["model"]
    model.backbone.cpu()
    transform = EvalTransform(tuple(result["config"].augmentation.image_size))
    arrays = []
    for name in ("train", "val"):
        dataset = dm.eval_dataset(name, transform)
        arrays.append(extract_subjects(model.backbone, dataset, batch_size,
                                       dm.subject_labels))
    x_train, y_train, _ = arrays[0]
    x_val, y_val, _ = arrays[1]
    if len(set(y_train)) < 2 or len(set(y_val)) < 2:
        raise ValueError("The representative train/val cohort needs AD and HC subjects.")
    probe = make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=1000, random_state=42))
    probe.fit(x_train, y_train)
    return float(roc_auc_score(y_val, probe.predict_proba(x_val)[:, 1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHOD_GROUPS, required=True)
    parser.add_argument("--objective", choices=("val_auc", "val_loss"), default="val_auc")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--study-dir", type=Path, default=Path("tuning_runs"))
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    if args.trials < 1 or args.epochs < 1:
        parser.error("trials and epochs must be positive")
    args.study_dir.mkdir(parents=True, exist_ok=True)
    config_dir = str(Path(__file__).resolve().parent.parent / "configs")
    direction = "maximize" if args.objective == "val_auc" else "minimize"
    overrides = METHOD_GROUPS[args.method] + args.overrides
    with hydra.initialize_config_dir(version_base="1.3", config_dir=config_dir):
        base_cfg = hydra.compose(config_name="train", overrides=overrides)
    data_files = {}
    for name in ("csv_path", "manifest_path"):
        value = base_cfg.data.get(name)
        if value is not None:
            path = Path(value)
            data_files[name] = {"path": str(path),
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    cohort_key = json.dumps({"epochs": args.epochs, "overrides": overrides,
                             "data": OmegaConf.to_container(base_cfg.data, resolve=True),
                             "files": data_files}, sort_keys=True)
    study_name = (f"{args.method}_{args.objective}_"
                  f"{hashlib.sha256(cohort_key.encode()).hexdigest()[:8]}")
    storage = f"sqlite:///{(args.study_dir / (study_name + '.sqlite3')).resolve()}"
    study = optuna.create_study(study_name=study_name, storage=storage,
                                load_if_exists=True, direction=direction,
                                sampler=optuna.samplers.TPESampler(seed=42),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
                                if args.objective == "val_loss" else optuna.pruners.NopPruner())

    def objective(trial: optuna.Trial) -> float:
        with hydra.initialize_config_dir(version_base="1.3", config_dir=config_dir):
            cfg = hydra.compose(config_name="train", overrides=overrides)
        cfg.trainer.max_epochs = args.epochs
        cfg.trainer.enable_checkpointing = False
        cfg.trainer.enable_progress_bar = False
        cfg.optimizer.lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
        cfg.optimizer.weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)
        cfg.augmentation.crop_scale[0] = trial.suggest_float("crop_scale_min", 0.2, 0.6)
        if args.method == "vicreg":
            cfg.loss.var_coeff = trial.suggest_float("var_coeff", 10, 40)
            cfg.loss.cov_coeff = trial.suggest_float("cov_coeff", 0.1, 3, log=True)
        elif args.method == "byol":
            cfg.loss.ema_base = trial.suggest_float("ema_base", 0.95, 0.999)
        else:
            cfg.loss.mask_ratio = trial.suggest_float("mask_ratio", 0.15, 0.5)
            cfg.loss.teacher_temp = trial.suggest_float("teacher_temp", 0.03, 0.08)
        callbacks = [ValidationLossReporter(trial)] if args.objective == "val_loss" else []
        result = train(cfg, extra_callbacks=callbacks)
        result["config"] = cfg
        trial.set_user_attr("experiment_dir", str(result["exp_dir"]))
        if args.objective == "val_loss":
            return float(result["trainer"].callback_metrics["val/loss"])
        return validation_auc(result, int(cfg.data.batch_size))

    study.optimize(objective, n_trials=args.trials, gc_after_trial=True)
    best = {"study": study_name, "objective": args.objective,
            "best_value": study.best_value, "best_params": study.best_params,
            "trials_finished": len(study.trials), "base_overrides": args.overrides,
            "data_files": data_files,
            "note": "Search uses train/val only. Refit the selected configuration for 100 epochs; "
                    "evaluate test once with the subject-level frozen probe."}
    output = args.study_dir / f"{study_name}_best.json"
    output.write_text(json.dumps(best, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
