"""CPU cost of the training views per recipe and region of interest.

    python scripts/3_train/bench_augment_cost.py --recipes geo noise elastic --rois 32 48 64 multi3

Times the two-view transform of 16 cached sessions in one thread (as one
DataLoader worker runs it) and estimates the CPU-bound seconds per epoch
(366 subjects) for a given number of workers: the floor under which a run
cannot go however fast the GPU is.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from t5_ssl import recipes  # noqa: E402
from t5_ssl.data.roi3d import build_roi3d_transforms  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--recipes", nargs="+", default=["geo", "noise", "elastic"])
    parser.add_argument("--rois", nargs="+", default=list(recipes.ROIS))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--samples", type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(f"{'ROI':>7} {'receta':>9} {'ms/muestra':>11} {'s/época CPU':>12}  (workers={args.workers})")
    for roi in args.rois:
        spec = recipes.ROIS[roi]
        cubes = np.load(ROOT / f"data/roi3d/{spec['cache']}.npy", mmap_mode="r")
        batch = [torch.from_numpy(np.asarray(cubes[i], dtype=np.float32))
                 for i in range(args.samples)]
        for recipe in args.recipes:
            cfg = OmegaConf.create({"data": {"roi_size": spec["size"]},
                                    "augmentation": recipes.aug_spec(recipe, spec["size"])})
            transform = build_roi3d_transforms(cfg)[0]
            transform(batch[0])
            start = time.perf_counter()
            for cube in batch:
                transform(cube)
            ms = 1000 * (time.perf_counter() - start) / len(batch)
            print(f"{roi:>7} {recipe:>9} {ms:11.1f} {366 * ms / 1000 / args.workers:12.1f}",
                  flush=True)


if __name__ == "__main__":
    main()
