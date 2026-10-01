r"""Measure between-view agreement for a given augmentation config on real MRI data.

Generalizes outputs/diagnostics/diagnose_ssl_inputs.py (which only checked the
two fixed ssl_multiview/full-mild policies) to any augmentation YAML and any
2.5D stack size/plane, so a new "intermediate" augmentation can be tuned to a
target pair-correlation band (the plan targets ~0.75-0.90) before spending any
GPU time training with it.

Usage:
    .venv/Scripts/python.exe scripts/measure_view_agreement.py \\
        --augmentation configs/augmentation/mri_2p5d_intermediate.yaml \\
        --channels 5 --plane axial --n-subjects 16 --n-pairs 8
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torchvision.transforms import functional as F

from t5_ssl.data.augmentations import SSLViewTransform
from t5_ssl.data.oasis3_lmdb_datamodule import OASIS3LMDBDataModule


def correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    """Pearson correlation between two flattened tensors."""
    left = left.flatten().float() - left.float().mean()
    right = right.flatten().float() - right.float().mean()
    denominator = left.norm() * right.norm()
    return float((left @ right / denominator.clamp_min(1e-8)).item())


def main() -> None:
    """Report pair-correlation, MAE, and foreground-retention for one augmentation config."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--augmentation", type=Path, required=True)
    parser.add_argument("--channels", type=int, default=5)
    parser.add_argument("--plane", default="axial")
    parser.add_argument("--slice-range", type=float, nargs=2, default=(0.32, 0.65))
    parser.add_argument("--n-subjects", type=int, default=16)
    parser.add_argument("--n-pairs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("campaign_full50/train_sessions_eval_primary.csv"),
    )
    parser.add_argument("--csv-path", type=Path, default=Path("data/dementia/mni-dataset.csv"))
    parser.add_argument("--lmdb-path", type=Path, default=Path("data/dementia/mni-dataset.lmdb"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    aug_cfg = yaml.safe_load(args.augmentation.read_text(encoding="utf-8"))
    transform = SSLViewTransform(
        image_size=tuple(aug_cfg.get("image_size", (96, 96))),
        crop_scale=tuple(aug_cfg.get("crop_scale", (0.3, 1.0))),
        crop_ratio=tuple(aug_cfg.get("crop_ratio", (0.9, 1.1))),
        horizontal_flip_p=aug_cfg.get("horizontal_flip_p", 0.5),
        vertical_flip_p=aug_cfg.get("vertical_flip_p", 0.5),
        affine_degrees=aug_cfg.get("affine_degrees", 15.0),
        translate_frac=tuple(aug_cfg.get("translate_frac", (0.0, 0.0))),
        blur_p=aug_cfg.get("blur_p", 0.5),
        noise_std=aug_cfg.get("noise_std", 0.03),
        intensity_scale=tuple(aug_cfg.get("intensity_scale", (0.9, 1.1))),
        gamma_range=tuple(aug_cfg.get("gamma_range", (1.0, 1.0))),
        bias_field_p=aug_cfg.get("bias_field_p", 0.0),
        bias_field_alpha=aug_cfg.get("bias_field_alpha", 0.15),
    )

    dm = OASIS3LMDBDataModule(
        csv_path=str(args.csv_path), lmdb_path=str(args.lmdb_path),
        manifest_path=str(args.manifest), channels=args.channels, plane=args.plane,
        slice_range=tuple(args.slice_range), one_session_per_subject_train=True,
        train_cache_capacity=args.n_subjects + 2,
    )
    dm.setup()
    dataset = dm.train_dataset
    assert dataset is not None

    selected, seen = [], set()
    for record in dataset.records:
        if record["subject"] not in seen:
            selected.append(record)
            seen.add(record["subject"])
        if len(selected) == args.n_subjects:
            break

    center_channel = args.channels // 2
    metrics: dict[str, list[float]] = defaultdict(list)
    for record in selected:
        volume = dataset._volume(record)
        size = volume.shape[dataset.axis]
        z = round(sum(args.slice_range) / 2 * (size - 1))
        image = np.stack(
            [np.take(volume, min(max(z + off, 0), size - 1), axis=dataset.axis)
             for off in range(-(args.channels // 2), args.channels // 2 + 1)], axis=0,
        ).copy()
        tensor = torch.from_numpy(image)
        base = F.resize(tensor, transform.image_size, antialias=True).clamp(0, 1)
        base_foreground = float((base[center_channel] > 0.05).float().mean().item())
        for _ in range(args.n_pairs):
            first, second = transform(tensor)
            metrics["pair_correlation"].append(
                correlation(first[center_channel], second[center_channel]))
            metrics["pair_mae"].append(float((first - second).abs().mean()))
            for view in (first, second):
                foreground = float((view[center_channel] > 0.05).float().mean().item())
                metrics["foreground_ratio_to_base"].append(foreground / max(base_foreground, 1e-8))

    report = {
        "augmentation_config": str(args.augmentation),
        "channels": args.channels, "plane": args.plane,
        "n_subjects": len(selected), "n_pairs_per_subject": args.n_pairs,
        "metrics": {
            name: {"mean": float(np.mean(values)), "p10": float(np.percentile(values, 10)),
                  "p90": float(np.percentile(values, 90))}
            for name, values in metrics.items()
        },
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output is not None:
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
