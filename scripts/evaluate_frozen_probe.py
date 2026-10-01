"""Evaluate frozen OASIS3 encoders for AD versus HC at subject level."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset

from t5_ssl.data.augmentations import EvalTransform
from t5_ssl.data.oasis3_lmdb_datamodule import clinical_rows
from t5_ssl.data.paths import rebase_missing_lmdb_paths
from t5_ssl.train import build_backbone, build_datamodule


def _rebase_missing_local_paths(cfg: object) -> None:
    """Make copied LMDB experiment configs evaluable without rewriting them."""
    if str(cfg.data.get("source", "nifti")) != "lmdb":
        return
    csv_path, lmdb_path, manifest_path = rebase_missing_lmdb_paths(
        cfg.data.csv_path, cfg.data.lmdb_path, cfg.data.get("manifest_path", None)
    )
    cfg.data.csv_path = csv_path
    cfg.data.lmdb_path = lmdb_path
    cfg.data.manifest_path = manifest_path


def extract_subjects(backbone: torch.nn.Module, dataset: object,
                     batch_size: int, subject_labels: dict[str, str] | None = None
                     ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Average deterministic slice and session embeddings once per subject."""
    labels: dict[str, set[str]] = defaultdict(set)
    for row in dataset.records:
        labels[row["subject"]].add(row["group"])
    valid = (subject_labels if subject_labels is not None else
             {subject: next(iter(values)) for subject, values in labels.items()
              if len(values) == 1 and next(iter(values)) in ("AD", "HC")})
    features: dict[str, list[np.ndarray]] = defaultdict(list)
    backbone.eval().requires_grad_(False)
    eligible = [index for index, (record_index, _) in enumerate(dataset.indices)
                if dataset.records[record_index]["subject"] in valid]
    loader = DataLoader(Subset(dataset, eligible), batch_size=batch_size,
                        shuffle=False, num_workers=0)
    with torch.inference_mode():
        for batch in loader:
            embeddings = backbone(batch["image"]).cpu().numpy()
            for subject, embedding in zip(batch["subject"], embeddings, strict=True):
                if subject in valid:
                    features[subject].append(embedding)
    subjects = sorted(features)
    if not subjects:
        raise ValueError("No unambiguous AD/HC subjects in this split.")
    x = np.stack([np.mean(features[s], axis=0) for s in subjects])
    y = np.array([int(valid[s] == "AD") for s in subjects])
    return x, y, subjects


def evaluate(train: tuple, val: tuple, test: tuple) -> dict:
    """Fit train only, choose L2 on validation, score test once."""
    x_train, y_train, _ = train
    x_val, y_val, val_subjects = val
    x_test, y_test, subjects = test
    if any(len(set(y)) != 2 for y in (y_train, y_val, y_test)):
        raise ValueError("Each probe split needs both AD and HC subjects.")
    best = None
    for c in (0.01, 0.1, 1.0, 10.0):
        probe = make_pipeline(StandardScaler(), LogisticRegression(
            C=c, class_weight="balanced", max_iter=1000, random_state=42))
        probe.fit(x_train, y_train)
        val_auc = float(roc_auc_score(y_val, probe.predict_proba(x_val)[:, 1]))
        if best is None or val_auc > best[0]:
            best = (val_auc, c, probe)
    assert best is not None
    val_scores = best[2].predict_proba(x_val)[:, 1]
    scores = best[2].predict_proba(x_test)[:, 1]
    centered = x_train - x_train.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    spectrum = singular_values ** 2
    spectrum = spectrum / max(float(spectrum.sum()), 1e-12)
    positive = spectrum[spectrum > 0]
    effective_rank = float(np.exp(-np.sum(positive * np.log(positive))))
    return {"train_subjects": len(y_train), "val_subjects": len(y_val),
            "test_subjects": len(y_test), "selected_C": best[1],
            "embedding_dim": int(x_train.shape[1]),
            "embedding_mean_feature_std": float(centered.std(axis=0).mean()),
            "embedding_effective_rank": effective_rank,
            "val_auc": best[0],
            "val_average_precision": float(average_precision_score(y_val, val_scores)),
            "val_f1_macro": float(f1_score(y_val, val_scores >= 0.5, average="macro")),
            "test_auc": float(roc_auc_score(y_test, scores)),
            "test_average_precision": float(average_precision_score(y_test, scores)),
            "test_brier_score": float(brier_score_loss(y_test, scores)),
            "val_log_loss": float(log_loss(y_val, val_scores, labels=[0, 1])),
            "test_log_loss": float(log_loss(y_test, scores, labels=[0, 1])),
            "test_accuracy": float(accuracy_score(y_test, scores >= 0.5)),
            "test_balanced_accuracy": float(balanced_accuracy_score(y_test, scores >= 0.5)),
            "test_f1_macro": float(f1_score(y_test, scores >= 0.5, average="macro")),
            "val_predictions": [{"subject": subject, "AD": int(label), "score": float(score)}
                                for subject, label, score in zip(
                                    val_subjects, y_val, val_scores, strict=True
                                )],
            "test_predictions": [{"subject": subject, "AD": int(label), "score": float(score)}
                                 for subject, label, score in zip(
                                     subjects, y_test, scores, strict=True
                                 )]}


def with_age(split: tuple, dataset: object, ages: dict[str, float],
             *, age_only: bool = False) -> tuple:
    """Join mean age of the same sessions used for subject embeddings."""
    features, labels, subjects = split
    by_subject: dict[str, list[float]] = defaultdict(list)
    for record in dataset.records:
        by_subject[record["subject"]].append(ages[record["idx"]])
    age_column = np.array([np.mean(by_subject[subject]) for subject in subjects],
                          dtype=np.float32).reshape(-1, 1)
    if not np.isfinite(age_column).all():
        raise ValueError("Age is missing for a probe subject")
    return (age_column if age_only else np.column_stack((features, age_column)),
            labels, subjects)


def evaluate_validation_only(train: tuple, val: tuple) -> dict:
    """Check the probe path without loading or scoring test subjects."""
    x_train, y_train, _ = train
    x_val, y_val, _ = val
    if len(set(y_train)) != 2 or len(set(y_val)) != 2:
        raise ValueError("Train and validation need AD and HC subjects.")
    values = {}
    for c in (0.01, 0.1, 1.0, 10.0):
        probe = make_pipeline(StandardScaler(), LogisticRegression(
            C=c, class_weight="balanced", max_iter=1000, random_state=42))
        probe.fit(x_train, y_train)
        values[c] = float(roc_auc_score(y_val, probe.predict_proba(x_val)[:, 1]))
    selected = max(values, key=values.get)
    return {"train_subjects": len(y_train), "val_subjects": len(y_val),
            "selected_C": selected, "val_auc": values[selected]}


def evaluate_label_fractions(
    train: tuple, val: tuple, test: tuple, fractions: list[float], seed: int = 42,
) -> dict[str, dict]:
    """Refit the probe on class-balanced subsamples of train to trace a label-efficiency curve.

    Val and test stay fixed; only how much of train's *labels* the probe gets
    to see changes. This reuses the fixed test split across every fraction, so
    it is a diagnostic label-efficiency curve, not the once-only headline
    test score computed by :func:`evaluate`.
    """
    x_train, y_train, _ = train
    rng = np.random.default_rng(seed)
    positive, negative = np.flatnonzero(y_train == 1), np.flatnonzero(y_train == 0)
    curve = {}
    for fraction in fractions:
        if not 0 < fraction <= 1:
            raise ValueError("label fractions must be in (0, 1]")
        n_pos = max(1, min(len(positive), round(len(positive) * fraction)))
        n_neg = max(1, min(len(negative), round(len(negative) * fraction)))
        chosen = np.concatenate([
            rng.choice(positive, n_pos, replace=False),
            rng.choice(negative, n_neg, replace=False),
        ])
        sub_train = (x_train[chosen], y_train[chosen], None)
        result = evaluate(sub_train, val, test)
        del result["test_predictions"], result["val_predictions"]
        result["n_labeled_subjects"] = len(chosen)
        curve[f"{fraction:g}"] = result
    return curve


def paired_auc_difference(trained: dict, random: dict, repetitions: int = 2000
                          ) -> dict[str, float | list[float]]:
    """Bootstrap held-out subjects within class for an AUC improvement interval."""
    first = trained["test_predictions"]
    second = random["test_predictions"]
    if [r["subject"] for r in first] != [r["subject"] for r in second]:
        raise RuntimeError("Backbones produced different test subject orders")
    labels = np.array([r["AD"] for r in first])
    trained_scores = np.array([r["score"] for r in first])
    random_scores = np.array([r["score"] for r in second])
    positive, negative = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    rng = np.random.default_rng(42)
    trained_aucs, random_aucs, deltas = [], [], []
    for _ in range(repetitions):
        indices = np.concatenate((rng.choice(positive, len(positive), replace=True),
                                  rng.choice(negative, len(negative), replace=True)))
        first_auc = roc_auc_score(labels[indices], trained_scores[indices])
        second_auc = roc_auc_score(labels[indices], random_scores[indices])
        trained_aucs.append(first_auc)
        random_aucs.append(second_auc)
        deltas.append(first_auc - second_auc)
    return {"delta_test_auc": trained["test_auc"] - random["test_auc"],
            "trained_test_auc_bootstrap_95ci": [float(v) for v in
                                                np.percentile(trained_aucs, [2.5, 97.5])],
            "random_test_auc_bootstrap_95ci": [float(v) for v in
                                               np.percentile(random_aucs, [2.5, 97.5])],
            "delta_test_auc_bootstrap_95ci": [float(v) for v in
                                               np.percentile(deltas, [2.5, 97.5])],
            "bootstrap_probability_delta_positive": float(np.mean(np.array(deltas) > 0))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--checkpoint", choices=("last", "best"), default="last",
                        help="SSL checkpoint for the trained backbone; random control is unchanged")
    parser.add_argument("--checkpoint-epoch", type=int, default=None,
                        help="Evaluate a specific periodic checkpoint (checkpoints/ckpt-epoch=NN.ckpt) "
                             "instead of --checkpoint; for cherry-picking across epochs.")
    parser.add_argument("--slices-per-volume", type=int, default=1,
                        help="Deterministic axial slices to average per session")
    parser.add_argument("--with-age", action="store_true",
                        help="Also evaluate age-only and age + frozen image features")
    parser.add_argument("--test-manifest", type=Path,
                        help="Predeclared OASIS3 manifest with identical train/val and extra held-out test subjects")
    parser.add_argument("--label-fractions", type=str, default=None,
                        help="Comma-separated label-efficiency curve, e.g. '0.01,0.05,0.1,0.25,1.0'. "
                             "Reuses the fixed test split per fraction: a diagnostic curve, not a "
                             "second headline number.")
    args = parser.parse_args()
    if args.slices_per_volume < 1:
        parser.error("--slices-per-volume must be positive")
    if args.validation_only and args.test_manifest is not None:
        parser.error("--test-manifest applies only to the final test evaluation")
    if args.validation_only and args.label_fractions is not None:
        parser.error("--label-fractions requires the final test evaluation")
    label_fractions = ([float(item) for item in args.label_fractions.split(",")]
                       if args.label_fractions else None)
    cfg = OmegaConf.load(args.experiment_dir / "config.yaml")
    _rebase_missing_local_paths(cfg)
    if (cfg.data.get("max_volumes", None) is not None or
            not (isinstance(cfg.trainer.limit_train_batches, float) and
                 cfg.trainer.limit_train_batches == 1.0) or
            not (isinstance(cfg.trainer.limit_val_batches, float) and
                 cfg.trainer.limit_val_batches == 1.0)):
        raise ValueError("Probe requires a full-data checkpoint; smoke runs are not baselines.")
    cfg.data.batch_size = args.batch_size
    cfg.data.num_workers = 0
    if args.test_manifest is not None:
        if cfg.data.source != "lmdb":
            raise ValueError("Expanded test manifest requires the OASIS3 LMDB source")
        cfg.data.manifest_path = str(args.test_manifest.resolve())
    transform = EvalTransform(tuple(cfg.augmentation.image_size))
    dm = build_datamodule(cfg, transform, transform)
    dm.setup()
    saved: dict[str, set[str]] = defaultdict(set)
    with (args.experiment_dir / "subject_splits.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            saved[row["split"]].add(row["subject"])
    if any(saved[name] != set(dm.split_subject_ids[name]) for name in ("train", "val")):
        raise RuntimeError("Current subject split differs from the training split.")
    test_ids = set(dm.split_subject_ids["test"])
    if (saved["test"] != test_ids if args.test_manifest is None else
            not saved["test"] < test_ids):
        raise RuntimeError("Test manifest does not preserve and extend the original test split")
    records_path = args.experiment_dir / "data_records.csv"
    if records_path.is_file():
        with records_path.open(newline="", encoding="utf-8") as stream:
            saved_records = list(csv.DictReader(stream))
        current_records = [
            {"split": split, "subject": row["subject"], "session": row["session"],
             "idx": row.get("idx", ""), "path": row.get("path", ""), "group": row["group"]}
            for split in ("train", "val", "test")
            for row in getattr(dm, f"{split}_dataset").records
        ]
        if args.test_manifest is None and saved_records != current_records:
            raise RuntimeError("Current data records differ from the training cohort.")
        if args.test_manifest is not None:
            for split in ("train", "val"):
                if ([row for row in saved_records if row["split"] == split] !=
                        [row for row in current_records if row["split"] == split]):
                    raise RuntimeError(f"{split} records differ from the training cohort")
            old_test = [row for row in saved_records if row["split"] == "test"]
            new_test = [row for row in current_records if row["split"] == "test"]
            if new_test[:len(old_test)] != old_test:
                raise RuntimeError("Original test records were changed")
    split_names = ("train", "val") if args.validation_only else ("train", "val", "test")
    # The training DataLoader samples a new slice every visit. Rebuild all
    # probe splits with the fixed central slice so trained and random encoders
    # receive identical images and repeated evaluations are reproducible.
    dm.slices_per_volume = args.slices_per_volume
    datasets = tuple(dm.eval_dataset(name, transform) for name in split_names)
    ages = ({row["idx"]: float(row["age"]) for row in clinical_rows(cfg.data.csv_path)}
            if args.with_age else {})
    results = {}
    extracted = {}
    label_efficiency: dict[str, dict] = {}
    checkpoint_path = args.experiment_dir / "checkpoints" / "last.ckpt"
    if args.checkpoint_epoch is not None:
        checkpoint_path = (args.experiment_dir / "checkpoints" /
                           f"ckpt-epoch={args.checkpoint_epoch:02d}.ckpt")
        if not checkpoint_path.is_file():
            raise RuntimeError(f"No periodic checkpoint at epoch {args.checkpoint_epoch}: "
                               f"{checkpoint_path}")
    elif args.checkpoint == "best":
        matches = list((args.experiment_dir / "checkpoints").glob("best-loss-epoch=*.ckpt"))
        if len(matches) != 1:
            raise RuntimeError(f"Expected one best-loss checkpoint: {matches}")
        checkpoint_path = matches[0]
    random_used_pretrained_weights = bool(cfg.model.backbone.get("pretrained", False))
    for kind in ("trained", "random"):
        torch.manual_seed(int(cfg.seed))
        if kind == "random":
            # The "random" control must be a genuinely untrained encoder. If the
            # checkpoint's own config set model.backbone.pretrained=true (e.g. an
            # ImageNet-pretrained ConvNeXt/ResNet), reusing that same cfg here would
            # silently give the control ImageNet weights too, making it not random
            # at all and invalidating any SSL-vs-control comparison.
            backbone = build_backbone(OmegaConf.merge(
                cfg, {"model": {"backbone": {"pretrained": False}}}))
        else:
            backbone = build_backbone(cfg)
        if kind == "trained":
            checkpoint = torch.load(checkpoint_path,
                                    map_location="cpu", weights_only=False)
            state = {key.removeprefix("backbone."): value for key, value in
                     checkpoint["state_dict"].items() if key.startswith("backbone.")}
            backbone.load_state_dict(state, strict=True)
        splits = [extract_subjects(backbone, dataset, args.batch_size, dm.subject_labels)
                  for dataset in datasets]
        extracted[kind] = splits
        results[kind] = (evaluate_validation_only(*splits) if args.validation_only else
                         evaluate(*splits))
        if label_fractions is not None:
            label_efficiency[kind] = evaluate_label_fractions(*splits, label_fractions, seed=int(cfg.seed))
        if args.with_age:
            age_splits = [with_age(split_, dataset, ages) for split_, dataset in
                          zip(splits, datasets, strict=True)]
            results[f"{kind}_plus_age"] = (
                evaluate_validation_only(*age_splits) if args.validation_only else
                evaluate(*age_splits))
    if args.with_age:
        age_splits = [with_age(split_, dataset, ages, age_only=True)
                      for split_, dataset in zip(extracted["trained"], datasets, strict=True)]
        results["age_only"] = (evaluate_validation_only(*age_splits)
                               if args.validation_only else evaluate(*age_splits))
    result = {"method": str(cfg.loss.name), "seed": int(cfg.seed),
              "trained_checkpoint": str(checkpoint_path.resolve()),
              "random_control_forced_from_pretrained": random_used_pretrained_weights,
              "test_manifest": (str(args.test_manifest.resolve())
                                if args.test_manifest is not None else
                                cfg.data.get("manifest_path", None)),
              "test_manifest_sha256": hashlib.sha256(
                  Path(cfg.data.get("manifest_path")).read_bytes()).hexdigest()
              if cfg.data.get("manifest_path") is not None else None,
              "slices_per_volume": args.slices_per_volume,
              "age_source": "mean age of the evaluated sessions, original OASIS3 CSV"
              if args.with_age else None,
              "protocol": "OASIS3 only; subject-disjoint; unambiguous AD/HC; frozen encoder; "
                          "mean embedding per subject; train-only fit/scaling; validation C; "
                          + ("test untouched" if args.validation_only else "test once"),
              "results": results}
    if label_fractions is not None:
        result["label_efficiency"] = label_efficiency
    if not args.validation_only:
        result["comparison"] = paired_auc_difference(results["trained"], results["random"])
        if args.with_age:
            result["age_comparison"] = paired_auc_difference(
                results["trained_plus_age"], results["age_only"])
    stem = "ad_hc_validation_probe" if args.validation_only else "ad_hc_subject_probe"
    if args.checkpoint_epoch is not None:
        checkpoint_suffix = f"_epoch{args.checkpoint_epoch:02d}"
    elif args.checkpoint == "best":
        checkpoint_suffix = "_best"
    else:
        checkpoint_suffix = ""
    output = args.experiment_dir / (stem + checkpoint_suffix
                                    + (f"_{args.slices_per_volume}slices"
                                       if args.slices_per_volume != 1 else "") + ".json")
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "comparison": result.get("comparison"), "metrics": {
        key: {k: v for k, v in value.items() if k not in ("test_predictions", "val_predictions")}
        for key, value in results.items()}}, indent=2))


if __name__ == "__main__":
    main()
