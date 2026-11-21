#!/usr/bin/env python
"""Fully supervised AD/HC baseline: the SSL encoders trained end to end.

    python scripts/3_train/train_supervised.py --model byol --roi 32 --aug-recipe noise --seed 1 \
        --set data.manifest_path=campaign_ssl3d_runpod/split_361_63_84/manifest.csv

Same encoder, input, initialization and augmentation recipe as the SSL run of
``--model`` (the first view of its training transform; GPU elastic recipes are
applied on the GPU as in training), plus a linear head, trained with
class-balanced BCE on the AD/HC subjects of the train split, one random session
per subject per epoch (AdamW, warmup + cosine over ``--epochs``, bf16 on CUDA).
The epoch with the best val AUROC (subject score = mean logit over its
sessions) is kept, with early stopping after ``--patience`` epochs without
improvement, and scored on test. The CV row (protocol v2, 2026-11-07) trains
``--folds`` folds (the folds of t5_ssl.evaluation.protocol_v2.fold_plan, repeat 0); each fold
holds out 20 % of its training subjects as an inner validation that picks its epoch
with the same early stopping, so no fold inherits an epoch chosen on its own held-out
subjects. Every fold model also scores its share of the SSL-only AD/HC subjects of
train: ``protocol_v2.cv_rep`` is the representative CV (cohort prevalence and ages).

Scores are calibrated with a balanced, monotone Platt scaling fit on held-out
standardized logits: the inner validation of each fold model, the validation subjects
for the main model (until 2026-11-07 the model's own training subjects, where an
overfit model's logits are extreme, with default L2 that flattened near-constant
logits to one probability). Predictions run in fp32: in bf16 the logits of models still near their
bias tied (13 distinct test scores of 178 in one run; CV folds at AUROC 0.500).
``cv.clean.per_fold`` keeps the within-fold metrics and ``cv.clean.pooled_uncalibrated``
the raw pooled ones.

``<out-root>/<tag>/<model>_supervised[__<recipe>]__roi<roi>_s<seed>/result.json``
has the ``subjects``/``labels``/``cv``/``test`` fields of the train.py probe
results, so the plotting and bootstrap code reads both alike.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from t5_ssl import recipes  # noqa: E402


class FirstView:
    """One augmented view of the SSL training transform (no second view, no local crops)."""

    def __init__(self, transform: Any) -> None:
        self.transform = transform

    def __call__(self, cube: Any) -> Any:
        return self.transform._make_view(cube)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, choices=sorted(recipes.BASE))
    parser.add_argument("--roi", default="32", choices=sorted(recipes.ROIS))
    parser.add_argument("--aug-recipe", default=None, choices=sorted(recipes.AUG_RECIPES))
    parser.add_argument("--external", action="append", default=[], metavar="NAME=CSV",
                        help="external labeled test set (see train.py --external)")
    parser.add_argument("--backbone", default=None, choices=sorted(recipes.BACKBONES),
                        help="encoder instead of the model's own (t5_ssl.recipes.BACKBONES)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100, help="schedule length and max epochs")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4, help="encoder learning rate")
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out-root", default="runs")
    parser.add_argument("--tag", default="supervised")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="extra Hydra override (repeatable)")
    return parser.parse_args(argv)


def compose(args: argparse.Namespace) -> tuple[Any, list[str]]:
    from hydra import compose as hydra_compose
    from hydra import initialize_config_dir
    overrides = recipes.overrides(args.model, "campaign", epochs=args.epochs, seed=args.seed,
                                  accelerator="cpu" if args.cpu else "auto",
                                  aug_recipe=args.aug_recipe, roi=args.roi,
                                  backbone=args.backbone, extra=list(args.set))
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        return hydra_compose(config_name="train", overrides=overrides), overrides


class Trainer:
    """End-to-end training and subject-level scoring of encoder + linear head."""

    def __init__(self, cfg: Any, dm: Any, args: argparse.Namespace) -> None:
        import torch
        from omegaconf import OmegaConf

        from t5_ssl.data.roi3d import ROIEval3D, analysis_views
        self.cfg, self.dm, self.args = cfg, dm, args
        self.device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
        self.train_transform = FirstView(analysis_views(cfg))
        self.clean = ROIEval3D(int(cfg.data.roi_size))
        gpu = cfg.augmentation.get("gpu_elastic")
        self.gpu_elastic = dict(OmegaConf.to_container(gpu, resolve=True)) if gpu else None

    def label(self, subject: str) -> int:
        return int(self.dm.subject_labels[subject] == "AD")

    def build(self, seed: int) -> Any:
        import torch
        from torch import nn

        from t5_ssl.train import build_backbone
        torch.manual_seed(seed)
        encoder = build_backbone(self.cfg)
        return nn.ModuleDict({"encoder": encoder,
                              "head": nn.Linear(encoder.out_dim, 1)}).to(self.device)

    def forward(self, model: Any, images: Any) -> Any:
        return model["head"](model["encoder"](images)).squeeze(1)

    def autocast(self) -> Any:
        import torch
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda")

    def predict(self, model: Any, records: list[dict], cubes: dict | None = None,
                labels: dict | None = None) -> tuple[list[str], Any, Any]:
        """Mean logit over each subject's sessions, subjects sorted."""
        import numpy as np
        import torch
        from torch.utils.data import DataLoader

        from t5_ssl.data.roi3d import ROI3DDataset
        model.eval()
        logits: dict[str, list[float]] = defaultdict(list)
        loader = DataLoader(ROI3DDataset(records, cubes or self.dm.cubes, self.clean),
                            batch_size=32)
        # fp32, not bf16: bf16 keeps ~3 significant digits, so the logits of a model still
        # near its bias tied (13 distinct test scores of 178; CV folds at AUROC 0.500).
        with torch.inference_mode():
            for batch in loader:
                out = self.forward(model, batch["image"].to(self.device)).float().cpu().numpy()
                for subject, value in zip(batch["subject"], out, strict=True):
                    logits[subject].append(float(value))
        subjects = sorted(logits)
        return (subjects, np.array([np.mean(logits[s]) for s in subjects]),
                np.array([labels[s] if labels else self.label(s) for s in subjects]))

    def fit(self, records: list[dict], seed: int, stop_epoch: int | None = None,
            val_records: list[dict] | None = None) -> dict[str, Any]:
        """Train for ``stop_epoch`` epochs, or select the best val-AUROC epoch."""
        import numpy as np
        import torch
        from sklearn.metrics import roc_auc_score
        from torch.utils.data import DataLoader

        from t5_ssl.data.roi3d import ROI3DDataset
        args = self.args
        random.seed(seed)
        np.random.seed(seed)
        model = self.build(seed)
        dataset = ROI3DDataset(records, self.dm.cubes, self.train_transform,
                               one_session_per_subject=True)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                            num_workers=args.workers, persistent_workers=args.workers > 0,
                            pin_memory=self.device.type == "cuda",
                            generator=torch.Generator().manual_seed(seed))
        labels = [self.label(r["subject"]) for r in {r["subject"]: r for r in records}.values()]
        pos_weight = torch.tensor((len(labels) - sum(labels)) / max(1, sum(labels)),
                                  device=self.device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(
            [{"params": model["encoder"].parameters(), "lr": args.lr},
             {"params": model["head"].parameters(), "lr": args.head_lr}],
            weight_decay=args.weight_decay)
        total, warmup = args.epochs * len(loader), args.warmup_epochs * len(loader)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: (
            (step + 1) / max(1, warmup) if step < warmup
            else 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))))
        history, best = [], {"auc": -1.0, "epoch": 0, "state": None}
        for epoch in range(1, (stop_epoch or args.epochs) + 1):
            model.train()
            losses = []
            for batch in loader:
                images = batch["image"].to(self.device, non_blocking=True)
                if self.gpu_elastic:
                    from t5_ssl.data.gpu_augment import elastic_deform
                    images = elastic_deform(images, **self.gpu_elastic)
                target = torch.tensor([self.label(s) for s in batch["subject"]],
                                      dtype=torch.float32, device=self.device)
                with self.autocast():
                    loss = criterion(self.forward(model, images).float(), target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()
                losses.append(loss.item())
            row = {"epoch": epoch, "train_loss": float(np.mean(losses))}
            if val_records is not None:
                _, logit, y = self.predict(model, val_records)
                row["val_auc"] = float(roc_auc_score(y, logit))
                row["val_loss"] = float(torch.nn.functional.binary_cross_entropy_with_logits(
                    torch.tensor(logit), torch.tensor(y, dtype=torch.float64)))
                if row["val_auc"] > best["auc"]:
                    best = {"auc": row["val_auc"], "epoch": epoch,
                            "state": copy.deepcopy({k: v.detach().cpu()
                                                    for k, v in model.state_dict().items()})}
                print(f"[sup] epoch {epoch} loss={row['train_loss']:.4f} "
                      f"val_auc={row['val_auc']:.3f} best={best['auc']:.3f}@{best['epoch']}",
                      flush=True)
            history.append(row)
            if val_records is not None and epoch - best["epoch"] >= args.patience:
                break
        if best["state"] is not None:
            model.load_state_dict(best["state"])
        return {"model": model, "history": history, "best_epoch": best["epoch"] or len(history),
                "best_val_auc": best["auc"]}


def run(args: argparse.Namespace) -> int:
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        f1_score,
        roc_auc_score,
    )

    from t5_ssl.evaluation.metrics import scores
    from t5_ssl.train import build_datamodule
    started = time.time()
    cfg, overrides = compose(args)
    from t5_ssl.data.roi3d import ROIEval3D
    clean = ROIEval3D(int(cfg.data.roi_size))
    dm = build_datamodule(cfg, clean, clean)
    dm.setup()
    trainer = Trainer(cfg, dm, args)

    def records(*splits: str) -> list[dict]:
        return [r for s in splits for r in getattr(dm, f"{s}_dataset").records
                if r["subject"] in dm.subject_labels]

    sigmoid = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731

    def calibrator(model: Any, fit_records: list[dict]) -> Any:
        """Balanced Platt scaling on held-out logits, monotone increasing.

        The logits are standardized first and the fit is barely regularized: a model still
        near its bias has logits that vary by ~1e-3, and the default L2 fit stopped at slope 0
        (constant probabilities: test AUROC 0.500 although the val AUROC was 0.756). A slope
        <= 0 keeps the model's own ranking (standardized logit through the sigmoid).
        """
        from sklearn.linear_model import LogisticRegression
        _, logit, labels = trainer.predict(model, fit_records)
        mu, sd = float(np.mean(logit)), float(np.std(logit)) or 1.0
        platt = LogisticRegression(C=1e4, class_weight="balanced", max_iter=1000).fit(
            ((logit - mu) / sd)[:, None], labels)
        a, b = float(platt.coef_[0, 0]), float(platt.intercept_[0])
        if a <= 0:
            a, b = 1.0, 0.0
        return lambda z: sigmoid(a * (np.asarray(z, dtype=np.float64) - mu) / sd + b)

    # 1. Train split, best val-AUROC epoch, scored on test.
    main = trainer.fit(records("train"), args.seed, val_records=records("val"))
    stop = main["best_epoch"]
    val_subjects, val_logit, y_val = trainer.predict(main["model"], records("val"))
    test_subjects, test_logit, y_test = trainer.predict(main["model"], records("test"))
    calibrate = calibrator(main["model"], records("val"))
    p_val, p_test = calibrate(val_logit), calibrate(test_logit)
    test = {"train_subjects": len({r["subject"] for r in records("train")}),
            "val_subjects": len(val_subjects), "test_subjects": len(test_subjects),
            "val_auc": float(roc_auc_score(y_val, p_val)),
            "test_auc": float(roc_auc_score(y_test, p_test)),
            "test_average_precision": float(average_precision_score(y_test, p_test)),
            "test_accuracy": float(accuracy_score(y_test, p_test >= 0.5)),
            "test_balanced_accuracy": float(balanced_accuracy_score(y_test, p_test >= 0.5)),
            "test_f1_macro": float(f1_score(y_test, p_test >= 0.5, average="macro")),
            "test_f1_macro_uncalibrated": float(f1_score(y_test, sigmoid(test_logit) >= 0.5,
                                                         average="macro")),
            "val_predictions": [{"subject": s, "AD": int(a), "score": float(p)}
                                for s, a, p in zip(val_subjects, y_val, p_val, strict=True)],
            "test_predictions": [{"subject": s, "AD": int(a), "score": float(p), "logit": float(z)}
                                 for s, a, p, z in zip(test_subjects, y_test, p_test, test_logit,
                                                       strict=True)]}
    # Extra held-out sets (test_<name> splits): test + those subjects, same calibration.
    from t5_ssl.evaluation.metrics import held_out_metrics
    extra = {}
    for split in dm.extra_test_splits:
        names, logit, labels = trainer.predict(main["model"], records("test", split))
        extra[split.removeprefix("test_")] = held_out_metrics(labels, calibrate(logit), names)
    from t5_ssl.data.roi3d import load_external
    for spec in args.external:
        name, csv_path = spec.split("=", 1)
        external = load_external(name, csv_path, cfg.data.cache_path)
        names, logit, labels = trainer.predict(main["model"], external["records"],
                                               external["cubes"], external["labels"])
        extra[name] = held_out_metrics(labels, calibrate(logit), names)
    if extra:
        test["extra_tests"] = extra
    print(f"[sup] best epoch {stop}: val_auc={test['val_auc']:.3f} "
          f"test_auc={test['test_auc']:.3f}", flush=True)
    del main["model"]
    # 2. CV over train+val AD/HC subjects (protocol v2, src/t5_ssl/evaluation/protocol_v2.py). Each
    # fold holds out an inner validation (20 % of its training subjects, stratified) that
    # picks the epoch (best AUROC, early stopping as the main run) and fits the balanced
    # Platt calibration; the fold model scores its held-out labeled subjects and its share
    # of the SSL-only AD/HC subjects (never trained on): representative CV.
    from sklearn.model_selection import train_test_split

    from t5_ssl.evaluation import protocol_v2 as pp
    pool = records("train", "val")
    subjects = sorted({r["subject"] for r in pool})
    y = np.array([trainer.label(s) for s in subjects])
    manifest = Path(cfg.data.manifest_path)
    pool_rows = pp.pool_records(manifest)
    pool_records = [{k: r[k] for k in ("subject", "idx", "session", "group")} for r in pool_rows]
    y_pool = np.array([r["label"] for r in pool_rows], dtype=int)
    pool_labels = {r["subject"]: r["label"] for r in pool_rows}
    plan = pp.fold_plan(y, y_pool, repeats=1, folds=args.folds)
    oof, raw = np.zeros(len(subjects) + len(pool_rows)), np.zeros(len(subjects) + len(pool_rows))
    position = {s: i for i, s in enumerate(subjects)}
    position.update({r["subject"]: len(subjects) + i for i, r in enumerate(pool_rows)})
    fold_epochs = []
    for fold in range(args.folds):
        fit_idx = np.flatnonzero(plan[0, :len(subjects)] != fold)
        inner_train, inner_val = train_test_split(
            fit_idx, test_size=0.2, stratify=y[fit_idx], random_state=fold)
        inner_train_set = {subjects[i] for i in inner_train}
        inner_val_set = {subjects[i] for i in inner_val}
        fitted = trainer.fit([r for r in pool if r["subject"] in inner_train_set],
                             args.seed + 1000 * (fold + 1),
                             val_records=[r for r in pool if r["subject"] in inner_val_set])
        fold_epochs.append(fitted["best_epoch"])
        calibrate_fold = calibrator(fitted["model"],
                                    [r for r in pool if r["subject"] in inner_val_set])
        held = {subjects[i] for i in np.flatnonzero(plan[0, :len(subjects)] == fold)}
        names, logit, _ = trainer.predict(fitted["model"], [r for r in pool if r["subject"] in held])
        held_pool = [pool_records[i] for i in np.flatnonzero(plan[0, len(subjects):] == fold)]
        if held_pool:
            pool_names, pool_logit, _ = trainer.predict(fitted["model"], held_pool,
                                                        labels=pool_labels)
            names, logit = names + pool_names, np.concatenate([logit, pool_logit])
        where = [position[s] for s in names]
        oof[where] = calibrate_fold(logit)
        raw[where] = sigmoid(logit)
        print(f"[sup] fold {fold + 1}/{args.folds} done, epoch {fitted['best_epoch']} "
              f"({(time.time() - started) / 60:.1f} min)", flush=True)
        del fitted
    n_lab = len(subjects)
    folds = [(np.flatnonzero(plan[0, :n_lab] != k), np.flatnonzero(plan[0, :n_lab] == k))
             for k in range(args.folds)]
    cv = {"clean": {"metrics": scores(y, oof[:n_lab]), "oof_mean": oof[:n_lab].tolist(),
                    "per_fold": [scores(y[t], oof[:n_lab][t]) for _, t in folds],
                    "pooled_uncalibrated": scores(y, raw[:n_lab]), "fold_epochs": fold_epochs,
                    "raw_sigmoid_all": raw.tolist()}}
    ages = pp.session_ages(manifest)
    cv_subjects = subjects + [r["subject"] for r in pool_rows]
    age_all = np.array([ages.get(s, np.nan) for s in cv_subjects])
    v2 = pp.summarize(y, y_pool, oof[None], plan, age_all)
    age_test = np.array([ages.get(s, np.nan) for s in test_subjects])
    v2["test"] = pp.held_out(y_test, p_test, v2["threshold_rep"], age_test)
    matched = pp.age_matched_subset(test_subjects, y_test, ages)
    v2["test_agem"] = pp.held_out(y_test[matched], p_test[matched], v2["threshold_rep"],
                                  age_test[matched])
    v2["scores"] = {"cv_subjects": cv_subjects, "cv_y": np.concatenate([y, y_pool]).tolist(),
                    "n_lab": n_lab, "cv_oof": oof[None].tolist(), "plan": plan.tolist(),
                    "test_subjects": test_subjects, "test_y": y_test.tolist(),
                    "test_p": p_test.tolist()}
    print(f"[sup v2] rep CV AUROC {v2['cv_rep']['auc']:.3f} AUPR {v2['cv_rep']['aupr']:.3f} "
          f"F1 {v2['cv_rep']['f1_macro']:.3f} | matched {v2['cv_matched']['auc']:.3f} | "
          f"test {v2['test']['auc']:.3f}", flush=True)
    name = (f"{args.model}_supervised" + (f"__{args.aug_recipe}" if args.aug_recipe else "")
            + (f"__bb{args.backbone}" if args.backbone else "") + f"__roi{args.roi}_s{args.seed}")
    exp_dir = Path(args.out_root) / args.tag / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    cv_m = cv["clean"]["metrics"]
    doc = {"model": args.model, "variant": "supervised", "seed": args.seed, "epochs": stop,
           "max_epochs": args.epochs, "aug_recipe": args.aug_recipe, "roi": args.roi,
           "backbone": args.backbone or recipes.DEFAULT_BACKBONE[args.model],
           "overrides": overrides, "hyperparameters": {k: getattr(args, k) for k in (
               "batch_size", "lr", "head_lr", "weight_decay", "warmup_epochs", "patience",
               "folds")},
           "subjects": subjects, "labels": y.tolist(), "cv": cv, "test": test,
           "protocol_v2": v2,
           "best_epoch": stop, "best_val_auc": main["best_val_auc"], "history": main["history"],
           "cv_auc": cv_m["auc"], "test_auc": test["test_auc"],
           "cv_mean3": (cv_m["auc"] + cv_m["f1"] + cv_m["aupr"]) / 3,
           "test_mean3": (test["test_auc"] + test["test_f1_macro"]
                          + test["test_average_precision"]) / 3,
           "minutes": (time.time() - started) / 60, "experiment_dir": exp_dir.as_posix()}
    (exp_dir / "result.json").write_text(json.dumps(doc) + "\n", encoding="utf-8")
    print(f"[done] {exp_dir.as_posix()} in {doc['minutes']:.1f} min cv_auc={doc['cv_auc']:.3f} "
          f"test_auc={doc['test_auc']:.3f}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("OASIS3_CSV", str(ROOT / "data/dementia/mni-dataset.csv"))
    os.chdir(ROOT)  # relative data/manifest paths in the configs
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
