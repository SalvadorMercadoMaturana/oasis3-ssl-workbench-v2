"""Close interrupted Optuna trials and preserve the best completed SSL run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import optuna


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-dir", type=Path, default=Path("tuning_pretext"))
    parser.add_argument("--method", choices=("vicreg", "byol", "dinov2"), required=True)
    args = parser.parse_args()
    candidates = list(args.study_dir.glob(f"{args.method}_ssl_loss_*.sqlite3"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one study for {args.method}: {candidates}")
    db = candidates[0].resolve()
    name = db.stem
    study = optuna.load_study(study_name=name, storage=f"sqlite:///{db}")
    settings = study.user_attrs["settings"]
    interrupted = [t.number for t in study.trials
                   if t.state == optuna.trial.TrialState.RUNNING]
    for number in interrupted:
        study.tell(number, state=optuna.trial.TrialState.FAIL)
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    if len(complete) < 3:
        raise RuntimeError(f"Need at least three complete trials for {args.method}")
    baseline = next(t for t in complete
                    if t.user_attrs.get("role") == "original_baseline")
    best = study.best_trial
    output = {
        "study": name, "method": args.method,
        "objective": "mean val/loss over final five epochs; no probe or label-based selection",
        "epochs": settings["epochs"],
        "min_epochs_before_pruning": settings["min_epochs"],
        "trials_requested": 12,
        "trials_resolved": len(complete) + len(pruned),
        "trials_complete": len(complete), "trials_pruned": len(pruned),
        "trials_interrupted": len(interrupted), "early_stopped": True,
        "baseline": settings["baseline"], "baseline_score": baseline.value,
        "best_trial": best.number, "best_score": best.value,
        "best_params": best.params,
        "best_experiment_dir": best.user_attrs.get("experiment_dir"),
        "data_identity": settings["identity"], "loss_definition_fixed": True,
    }
    target = args.study_dir / f"{name}_best.json"
    target.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
