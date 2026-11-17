"""Calibrate TorchIO Spike and ElasticDeformation to the 3D campaign artifact strength.

Same procedure as ``run_roi3d.py calibrate`` (16 cached sessions, 2 draws each,
RMSE of the central 32-voxel cube against the clean one, log-bisection on the
knob) for the two artifacts that calibration did not cover. The target is
``campaign_roi3d/calibration_s32.json``'s ``target_rmse`` (0.080). Prints the
specs to paste into ``t5_ssl.recipes``.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from t5_ssl.data.roi3d import ROIEval3D, ROIView3DTransform  # noqa: E402

SPECS = {
    "spike": (lambda k: {"name": "RandomSpike", "num_spikes": 1, "intensity": k}, 0.01, 3.0),
    "elastic": (lambda k: {"name": "RandomElasticDeformation", "num_control_points": 7,
                           "max_displacement": k, "locked_borders": 2}, 0.1, 4.5),
}


def main() -> None:
    target = json.loads((ROOT / "campaign_roi3d/calibration_s32.json").read_text())["target_rmse"]
    cache = np.load(ROOT / "data/roi3d/hippo_s32.npy", mmap_mode="r")
    picks = random.Random(0).sample(range(len(cache)), 16)
    cubes = [torch.from_numpy(cache[i].astype(np.float32)) for i in sorted(picks)]
    clean = ROIEval3D(32)

    def rmse(spec: dict) -> float:
        view = ROIView3DTransform(size=32, torchio_transforms=[{**spec, "p": 1.0}])
        random.seed(0), np.random.seed(0), torch.manual_seed(0)
        return float(np.mean([float((view.photometric_only(c) - clean(c)).pow(2).mean().sqrt())
                              for c in cubes for _ in range(2)]))

    for name, (make, low, high) in SPECS.items():
        lo, hi = np.log(low), np.log(high)
        if rmse(make(high)) < target:
            knob = high
        else:
            for _ in range(10):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if rmse(make(float(np.exp(mid)))) < target else (lo, mid)
            knob = float(np.exp((lo + hi) / 2))
        print(f"{name}: knob={knob:.4f} rmse={rmse(make(knob)):.4f} (target {target:.4f}) "
              f"spec={json.dumps(make(knob))}", flush=True)


if __name__ == "__main__":
    main()
