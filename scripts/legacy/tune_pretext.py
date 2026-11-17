"""Tune the three native T5 pretext tasks using only fixed-definition SSL loss."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import hydra
import lightning as L
import numpy as np
import optuna
from omegaconf import OmegaConf

from t5_ssl.train import train

METHOD_GROUPS = {
    "vicreg": [],
    "byol": ["model=byol_convnext", "loss=byol"],
    "dinov2": ["model=dinov2_vit", "loss=dinov2",
               "augmentation=dinov2_multicrop"],
}
SEARCH_VERSION = 2


class LossReporter(L.Callback):
    """Keep a rolling validation-loss score and prune after a fair warmup."""

    def __init__(self, trial: optuna.Trial, min_epochs: int) -> None:
        self.trial = trial
        self.min_epochs = min_epochs
        self.values: list[float] = []

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        metric = trainer.callback_metrics.get("val/loss")
        if metric is None:
            raise RuntimeError("The SSL model did not log val/loss")
        value = float(metric)
        if not np.isfinite(value):
            raise optuna.TrialPruned("Nonfinite SSL validation loss")
        self.values.append(value)
        if len(self.values) >= self.min_epochs:
            rolling = float(np.mean(self.values[-5:]))
            self.trial.report(rolling, step=trainer.current_epoch + 1)
            if self.trial.should_prune():
                raise optuna.TrialPruned(f"Pruned at epoch {trainer.current_epoch + 1}")


def data_identity(cfg: object) -> dict:
    """Bind a resumable study to the clinical CSV and subject manifest."""
    result = {}
    for name in ("csv_path", "manifest_path"):
        value = cfg.data.get(name)
        if value is not None:
            path = Path(value)
            result[name] = {"path": str(path.resolve()),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    result["lmdb_path"] = str(Path(cfg.data.lmdb_path).resolve())
    return result


def search_space(trial: optuna.Trial) -> dict[str, float]:
    """Change training conditions while keeping validation loss semantics fixed."""
    return {
        "lr": trial.suggest_float("lr", 1e-5, 5e-4, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True),
        "crop_scale_min": trial.suggest_float("crop_scale_min", 0.2, 0.6),
        "noise_std": trial.suggest_float("noise_std", 0.0, 0.06),
    }


def apply_params(cfg: object, params: dict[str, float]) -> None:
    cfg.optimizer.lr = params["lr"]
    cfg.optimizer.weight_decay = params["weight_decay"]
    cfg.augmentation.crop_scale[0] = params["crop_scale_min"]
    cfg.augmentation.noise_std = params["noise_std"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHOD_GROUPS, required=True)
    parser.add_argument("--trials", type=int, default=12,
                        help="Total trials per resumable study, including the baseline")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--min-epochs", type=int, default=20)
    parser.add_argument("--study-dir", type=Path, default=Path("tuning_pretext"))
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    if args.trials < 4 or args.epochs < 10 or not 5 <= args.min_epochs < args.epochs:
        parser.error("Need at least four trials, ten epochs, and 5 <= min-epochs < epochs")
    args.study_dir.mkdir(parents=True, exist_ok=True)
    overrides = METHOD_GROUPS[args.method] + args.overrides
    config_dir = str(Path(__file__).resolve().parents[2] / "configs")
    with hydra.initialize_config_dir(version_base="1.3", config_dir=config_dir):
        base_cfg = hydra.compose(config_name="train", overrides=overrides)
    if base_cfg.data.source != "lmdb" or base_cfg.data.get("manifest_path") is None:
        parser.error("Tuning requires the fixed OASIS3 LMDB subject manifest")
    if (base_cfg.trainer.get("limit_train_batches") != 1.0 or
            base_cfg.trainer.get("limit_val_batches") != 1.0):
        parser.error("Tuning requires every train and validation batch")
    identity = data_identity(base_cfg)
    baseline = {"lr": float(base_cfg.optimizer.lr),
                "weight_decay": float(base_cfg.optimizer.weight_decay),
                "crop_scale_min": float(base_cfg.augmentation.crop_scale[0]),
                "noise_std": float(base_cfg.augmentation.noise_std)}
    study_settings = {"version": SEARCH_VERSION, "method": args.method,
                      "epochs": args.epochs, "min_epochs": args.min_epochs,
                      "overrides": overrides, "data": OmegaConf.to_container(
                          base_cfg.data, resolve=True), "identity": identity,
                      "baseline": baseline}
    fingerprint = hashlib.sha256(json.dumps(study_settings, sort_keys=True).encode()).hexdigest()[:12]
    study_name = f"{args.method}_ssl_loss_{fingerprint}"
    db = (args.study_dir / f"{study_name}.sqlite3").resolve()
    study = optuna.create_study(
        study_name=study_name, storage=f"sqlite:///{db}", load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=3),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=3, n_warmup_steps=args.min_epochs,
            interval_steps=5, n_min_trials=3),
    )
    if not study.trials:
        study.enqueue_trial(baseline, user_attrs={"role": "original_baseline"})
        study.set_user_attr("settings", study_settings)

    def objective(trial: optuna.Trial) -> float:
        with hydra.initialize_config_dir(version_base="1.3", config_dir=config_dir):
            cfg = hydra.compose(config_name="train", overrides=overrides)
        cfg.trainer.max_epochs = args.epochs
        cfg.trainer.enable_checkpointing = False
        cfg.trainer.enable_progress_bar = False
        params = search_space(trial)
        apply_params(cfg, params)
        reporter = LossReporter(trial, min_epochs=args.min_epochs)
        result = train(cfg, extra_callbacks=[reporter])
        if len(reporter.values) != args.epochs:
            raise RuntimeError(f"Expected {args.epochs} validation epochs")
        score = float(np.mean(reporter.values[-5:]))
        trial.set_user_attr("experiment_dir", str(result["exp_dir"]))
        trial.set_user_attr("last_val_loss", reporter.values[-1])
        trial.set_user_attr("best_val_loss", min(reporter.values))
        return score

    finished = sum(t.state in (optuna.trial.TrialState.COMPLETE,
                               optuna.trial.TrialState.PRUNED)
                   for t in study.trials)
    if finished < args.trials:
        study.optimize(objective, n_trials=args.trials - finished,
                       gc_after_trial=True)
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    if not complete:
        raise RuntimeError("No complete SSL tuning trial")
    best = study.best_trial
    output = {"study": study_name, "method": args.method, "objective":
              "mean val/loss over final five epochs; no probe or label-based selection",
              "epochs": args.epochs, "min_epochs_before_pruning": args.min_epochs,
              "trials_requested": args.trials, "trials_complete": len(complete),
              "trials_pruned": len(pruned), "baseline": baseline,
              "baseline_score": next((t.value for t in complete
                                      if t.user_attrs.get("role") == "original_baseline"), None),
              "best_trial": best.number, "best_score": best.value,
              "best_params": best.params, "best_experiment_dir":
              best.user_attrs.get("experiment_dir"), "data_identity": identity,
              "loss_definition_fixed": True}
    path = args.study_dir / f"{study_name}_best.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
