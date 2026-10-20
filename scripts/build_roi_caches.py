"""Build ROI cube caches beyond the 32/48-voxel hippocampal ones.

    python scripts/build_roi_caches.py hippo_s64                 # 64-voxel hippocampal pair
    python scripts/build_roi_caches.py multi3_s32                # hippocampus, ventricle, posteromedial
    python scripts/build_roi_caches.py atrophy5_s32 --manifest campaign_ssl3d_runpod/split_deploy/manifest.csv

Names: ``hippo_s<size>`` (hippocampus only), ``multi3_s<size>`` (three bilateral
regions, 6 channels) and ``atrophy5_s<size>`` (multi3 + lateral temporal and inferior
parietal cortex, 10 channels: the usual AD atrophy pattern). Written to
``data/roi3d/<name>.npy`` + ``.json`` from the local OASIS3 LMDB
(``t5_ssl.data.roi3d.build_region_cache``); ``--manifest`` picks the split whose
sessions are cached and whose train sessions give the region centroids (default: the
original balanced manifest).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from t5_ssl.data.roi3d import build_region_cache  # noqa: E402
from t5_ssl.recipes import MANIFEST  # noqa: E402

SETS = {"hippo": ["hippocampus"], "multi3": ["hippocampus", "ventricle", "posteromedial"],
        "atrophy5": ["hippocampus", "ventricle", "posteromedial", "lateral_temporal", "parietal"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("names", nargs="+", help="caches to build, e.g. multi3_s32")
    parser.add_argument("--manifest", default=MANIFEST)
    args = parser.parse_args()
    for name in args.names:
        prefix, size = name.rsplit("_s", 1)
        start = time.time()
        meta = build_region_cache(ROOT / "data/roi3d" / name, csv_path=str(
            ROOT / "data/dementia/mni-dataset.csv"), lmdb_path=str(ROOT / "data/dementia/mni-dataset.lmdb"),
            manifest_path=str(ROOT / args.manifest), size=int(size), regions=SETS[prefix])
        print(f"{name}: {meta['sessions']} sessions, channels {meta['channels']}, "
              f"starts {meta['cube_starts']}, {time.time() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
