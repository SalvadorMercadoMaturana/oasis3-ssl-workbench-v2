#!/usr/bin/env python
"""Job lines for cloud/queue_runner.py: pretext screen, tuning and outer folds (2026-11-16).

    python scripts/3_train/jobs_outer5.py screen tune outer > jobs.txt
    python scripts/3_train/jobs_outer5.py pretext_tune --task rot10_jig100 --aug geo >> jobs.txt
    python scripts/3_train/jobs_outer5.py pretext_outer --variant rot10_jig100_lr0.0003 --aug geo >> jobs.txt

Phases (all 50 epochs, 3 seeds, ConvNeXt-T unless stated, hippocampal pair 32^3):

- ``screen``: pseudo-label pretext tasks on split_deploy (= outer fold 0): rotation 10 /
  24 classes, jigsaw order of the octants (100 permutations), rotation + jigsaw, with the
  ``geo`` views (VICReg's) and rotation + jigsaw with the ``noise`` views (BYOL's).
- ``tune``: learning rate x0.5 / x3 of the three self-distillation / VICReg champions on
  split_deploy (their other hyperparameters were tuned in earlier rounds).
- ``outer``: the four reference models on outer folds 1-4 (fold 0 runs exist) + the
  frozen starts (sin SSL).
- ``pretext_tune``: the tuning grid of t5_ssl.recipes on one pretext task.
- ``pretext_outer``: one pretext variant on outer folds 1-4.
"""

from __future__ import annotations

import argparse

OUT = "/root/runs"
DEPLOY = "campaign_ssl3d_runpod/split_deploy/manifest.csv"
ADNI = "adni=campaign_ssl3d_runpod/split_deploy/external_adni.csv"
COMMON = f"--epochs 50 --out-root {OUT} --no-checkpoints --export-encoder --save-embeddings --workers 1"
SEEDS = (1, 2, 3)
# (model, variant, aug recipe, backbone, batch size)
REFERENCE = {
    "vicreg": ("vicreg", "campaign_augval", "geo", "convnext_tiny", 16),
    "byol": ("byol", "wide_dproj0.1", "noise", "convnext_tiny", 16),
    "dino_cnx": ("dinov2", "b64_k256", "elastic_tio_gpu", "convnext_tiny", 64),
    "dino_vit": ("dinov2", "campaign", "elastic_tio_gpu", "vit_small", 16),
}
BASE_LR = {"vicreg": 1e-4, "byol": 1e-4, "dino_cnx": 2e-4}


def fold_manifest(k: int) -> str:
    return DEPLOY if k == 0 else f"campaign_ssl3d_runpod/split_outer5/fold{k}/manifest.csv"


def line(name: str, model: str, variant: str, aug: str, backbone: str, batch: int, seed: int,
         tag: str, manifest: str, extra: str = "") -> str:
    return (f"{name} --model {model} --variant {variant} --aug-recipe {aug} --backbone {backbone} "
            f"--batch-size {batch} --roi 32 --seed {seed} --tag {tag} {COMMON} {extra} "
            f"--set data.manifest_path={manifest}").replace("  ", " ")


def screen() -> list[str]:
    specs = [("rot10", "geo"), ("rot24", "geo"), ("jig100", "geo"), ("rot10_jig100", "geo"),
             ("rot10_jig100", "noise")]
    return [line(f"pa_{v}_{a}_s{s}", "pretext", v, a, "convnext_tiny", 16, s, "pretext_screen",
                 DEPLOY, f"--identity --collapse-every 10 --external {ADNI}")
            for s in SEEDS for v, a in specs]


def tune() -> list[str]:
    out = []
    for s in SEEDS:
        for key, factor in (("vicreg", 0.5), ("vicreg", 3), ("byol", 0.5), ("byol", 3),
                            ("dino_cnx", 0.5), ("dino_cnx", 3)):
            model, variant, aug, bb, batch = REFERENCE[key]
            lr = BASE_LR[key] * factor
            out.append(line(f"tl_{key}_x{factor:g}_s{s}", model, variant, aug, bb, batch, s,
                            f"tune_lrx{factor:g}", DEPLOY, "", ) + f" --set optimizer.lr={lr:g}")
    return out


def outer() -> list[str]:
    out = []
    for k in (1, 2, 3, 4):
        m = fold_manifest(k)
        out.append(f"init_cnx_f{k} --model vicreg --backbone convnext_tiny --roi 32 --init-only "
                   f"--seed 1 --tag outer_f{k}_init {COMMON} --set data.manifest_path={m}")
        out.append(f"init_vit_f{k} --model dinov2 --backbone vit_small --roi 32 --init-only "
                   f"--seed 1 --tag outer_f{k}_init {COMMON} --set data.manifest_path={m}")
    for s in SEEDS:
        for k in (1, 2, 3, 4):
            for key, (model, variant, aug, bb, batch) in REFERENCE.items():
                out.append(line(f"oc_{key}_f{k}_s{s}", model, variant, aug, bb, batch, s,
                                f"outer_f{k}", fold_manifest(k)))
    return out


def pretext_tune(task: str, aug: str, grid: list[str] | None = None) -> list[str]:
    """``grid`` entries are tuning suffixes of t5_ssl.recipes (``base`` = the task alone)."""
    if not grid:
        grid = ["lr3e-05", "lr0.0003", "lr0.001", "wd0.05", "ls0.1", "h2048"]
        if "jig" in task:
            grid += ["gap0", "gap4"]
    return [line(f"pt_{task}_{t}_{aug}_s{s}", "pretext", task if t == "base" else f"{task}_{t}",
                 aug, "convnext_tiny", 16, s, "pretext_tune", DEPLOY, f"--external {ADNI}")
            for s in SEEDS for t in grid]


def pretext_outer(variant: str, aug: str) -> list[str]:
    return [line(f"op_{variant}_{aug}_f{k}_s{s}", "pretext", variant, aug, "convnext_tiny", 16, s,
                 f"outer_f{k}", fold_manifest(k))
            for s in SEEDS for k in (1, 2, 3, 4)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phases", nargs="+",
                    choices=("screen", "tune", "outer", "pretext_tune", "pretext_outer"))
    ap.add_argument("--task", default="rot10_jig100")
    ap.add_argument("--variant", default="rot10_jig100")
    ap.add_argument("--aug", default="geo")
    ap.add_argument("--grid", nargs="*", default=None)
    args = ap.parse_args()
    lines: list[str] = []
    for phase in args.phases:
        if phase == "pretext_tune":
            lines += pretext_tune(args.task, args.aug, args.grid)
        elif phase == "pretext_outer":
            lines += pretext_outer(args.variant, args.aug)
        else:
            lines += globals()[phase]()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
