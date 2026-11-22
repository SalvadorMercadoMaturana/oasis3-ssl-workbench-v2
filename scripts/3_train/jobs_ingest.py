#!/usr/bin/env python
"""Job lines for cloud/queue_runner.py: the ingestion / backbone tuning of 2026-11-22.

    python scripts/3_train/jobs_ingest.py init --backbones convnext_tiny cnx3d_x2 cnx2d_cor_x2
    python scripts/3_train/jobs_ingest.py ssl --backbones cnx2d_cor_x2 --seeds 1
    python scripts/3_train/jobs_ingest.py ssl --backbones cnx2d_cor_x2 --seeds 1 2 3 --folds 1 2 3 4
    python scripts/3_train/jobs_ingest.py sup --backbones cnx2d_cor_x2 --folds 0 1 2 3 4

How the ROI reaches the encoder and the encoder itself (``--backbone`` names of
t5_ssl.recipes.BACKBONES: t5_ssl.models.backbones.ingest settings, MedicalNet, 2D slice
encoders) are tuned for the four methods together, since all four share one backbone:

- ``init``: the frozen start of each backbone ("sin SSL"), protocol v2 probe. Cheap
  (no training): it ranks what the pretrained weights already read from each ingestion.
- ``ssl``: the four methods with their tuned recipes (the final models of the outer-fold
  campaign: VICReg campaign_augval geo, BYOL wide_dproj0.1 noise, DINOv2 b64_k256
  elastic_tio_gpu at batch 64, rotation pretext rot24_h2048 geo), 50 epochs.
- ``sup``: the fully supervised baseline on the same backbone (train_supervised.py).

Fold 0 is split_deploy; folds 1-4 are split_outer5/fold<k> (every AD/HC subject is test
once over the five folds). Tags: ``ing_init_f<k>``, ``ing_f<k>``, ``ing_sup_f<k>``.
"""

from __future__ import annotations

import argparse

OUT = "/root/runs"
COMMON = "--epochs 50 --out-root {out} --no-checkpoints --workers {workers}"
# key: (model, variant, aug recipe, batch size)
METHODS = {
    "vicreg": ("vicreg", "campaign_augval", "geo", 16),
    "byol": ("byol", "wide_dproj0.1", "noise", 16),
    "dino": ("dinov2", "b64_k256", "elastic_tio_gpu", 64),
    "pretext": ("pretext", "rot24_h2048", "geo", 16),
}


def manifest(k: int) -> str:
    return ("campaign_ssl3d_runpod/split_deploy/manifest.csv" if k == 0
            else f"campaign_ssl3d_runpod/split_outer5/fold{k}/manifest.csv")


def common(args: argparse.Namespace) -> str:
    return COMMON.format(out=args.out, workers=args.workers)


def init_lines(args: argparse.Namespace) -> list[str]:
    return [f"init_{bb}_f{k} --model vicreg --backbone {bb} --roi {args.roi} --init-only "
            f"--seed 1 --tag ing_init_f{k} {common(args)} {args.extra} "
            f"--set data.manifest_path={manifest(k)}".replace("  ", " ")
            for k in args.folds for bb in args.backbones]


def ssl_lines(args: argparse.Namespace) -> list[str]:
    out = []
    for s in args.seeds:
        for k in args.folds:
            for bb in args.backbones:
                for key in args.methods:
                    model, variant, aug, batch = METHODS[key]
                    out.append(
                        f"{key}_{bb}_f{k}_s{s} --model {model} --variant {variant} "
                        f"--aug-recipe {aug} --backbone {bb} --batch-size {batch} --roi {args.roi} "
                        f"--seed {s} --tag {args.tag or 'ing'}_f{k} {common(args)} {args.extra} "
                        f"--set data.manifest_path={manifest(k)}".replace("  ", " "))
    return out


def sup_lines(args: argparse.Namespace) -> list[str]:
    # as the outer-fold supervised baseline (outer_f<k>_sup): geo views, 50 epochs at most
    return [f"sup_{bb}_f{k}_s{s} script:scripts/3_train/train_supervised.py --model vicreg "
            f"--aug-recipe geo --backbone {bb} --roi {args.roi} --epochs 50 --seed {s} "
            f"--workers {args.workers} --out-root {args.out} --tag ing_sup_f{k} {args.extra} "
            f"--set data.manifest_path={manifest(k)}".replace("  ", " ")
            for s in args.seeds for k in args.folds for bb in args.backbones]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("init", "ssl", "sup"))
    ap.add_argument("--backbones", nargs="+", required=True)
    ap.add_argument("--folds", nargs="+", type=int, default=[0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1])
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    ap.add_argument("--roi", default="32")
    ap.add_argument("--tag", default=None, help="ssl tag prefix (default ing)")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--extra", default="", help="more train.py arguments for every line")
    args = ap.parse_args()
    from t5_ssl import recipes
    unknown = [b for b in args.backbones if b not in recipes.BACKBONES]
    if unknown:
        ap.error(f"unknown backbones {unknown}")
    print("\n".join({"init": init_lines, "ssl": ssl_lines, "sup": sup_lines}[args.phase](args)))


if __name__ == "__main__":
    main()
